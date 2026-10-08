from __future__ import annotations

import hashlib
import itertools
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from puppetmaster.budget import (
    BudgetAdmissionError,
    BudgetBusy,
    BudgetUnsettled,
    BudgetConflictError,
    BudgetLiability,
    BudgetPolicy,
    budget_totals,
    check_admission,
)
from puppetmaster.models import (
    AgentRun,
    Artifact,
    ArtifactType,
    GraphEdge,
    GraphEdgeType,
    GraphNodeKind,
    Job,
    JobRef,
    JobStatus,
    MemoryRecord,
    Task,
    TaskStatus,
    apply_running_duration,
    artifact_from_dict,
    assert_legal_task_transition,
    is_cost_final_job_status,
    task_is_satisfied,
    task_is_terminal,
    graph_edge_from_dict,
    job_from_dict,
    make_graph_edge,
    new_id,
    now_iso,
    parse_iso,
    seconds_from_now,
    task_from_dict,
    to_jsonable,
)
from puppetmaster.attempts import (
    ExecutionAttempt, UsageObservation, LedgerConflictError, canonical_record,
)
from puppetmaster.cost import maybe_stamp_terminal_cost_receipt
from puppetmaster.selected_economics import SelectedEconomics
from puppetmaster.redaction import redact_payload_for_storage
from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.state import resolve_state_dir
from puppetmaster.readonly import ReadUnavailable, selection
from puppetmaster.identity import make_ref, read_identity, StoreIdentityError
from puppetmaster.projections import connection as projection_connection

_WINDOWS_LOCK_RETRIES = 10
_WINDOWS_LOCK_BACKOFF_SECONDS = 0.02
# A lock file modified this recently cannot be older than any TTL it was taken
# with, so a waiter can call it fresh from ``stat`` alone (see _lock_is_stale).
_LOCK_STAT_SLACK_SECONDS = 5.0
# Routing provenance a pilot never acts on; full status still carries it.
_STATUS_PAYLOAD_BOOKKEEPING = frozenset((
    "billing_evidence", "billing_source", "registry_billing", "registry_digest",
    "registry_path", "pinned_adapter_model_name", "router_model_id", "extra_args",
))
_MEMORY_CAP = 200
_SCOPE_WEIGHTS = {
    "swarm.findings": 1.0,
    "swarm.decisions": 1.0,
    "swarm.general": 0.7,
    "swarm.verification": 0.4,
}
_DEFAULT_SCOPE_WEIGHT = 0.7
_GRAPH_EDGES_MARKER = ".graph_edges_materialized"
_CONSUMES_JOURNAL_PREFIX = ".consumes_journal_"


class ActiveTaskLeaseError(RuntimeError):
    """Raised when ``reset_subgraph`` would clear a task with a live lease."""

    def __init__(self, task_ids: Iterable[str]) -> None:
        self.task_ids = sorted({task_id for task_id in task_ids if task_id})
        joined = ", ".join(self.task_ids) or "(none)"
        super().__init__(
            f"reset_subgraph refused: active lease on task(s) {joined}"
        )


class LaunchConflictError(ValueError):
    """A launch key was reused for a different normalized request."""


class ProjectionWriteAdmissionError(sqlite3.OperationalError):
    """The file projection writer was unavailable before a source mutation."""


class ResetSubgraphResult(list):
    """Task list from ``reset_subgraph`` plus superseded artifact ids.

    Subclasses ``list`` so existing callers that iterate / index the return
    value keep working; ``superseded_artifact_ids`` carries the canonical
    ids stamped in the same reset (also on the ``subgraph.reset`` event).
    """

    def __init__(
        self,
        tasks: Iterable[Task],
        superseded_artifact_ids: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(list(tasks))
        self.superseded_artifact_ids = [
            str(artifact_id)
            for artifact_id in list(superseded_artifact_ids or [])
            if artifact_id
        ]
_RECENCY_FULL_DAYS = 7
_RECENCY_FLOOR = 0.5
_RECENCY_FLOOR_DAYS = 56


def _normalize_memory_statement(statement: str) -> str:
    return " ".join(str(statement).split())


def _memory_created_at_sort_key(memory: dict[str, Any]) -> str:
    created_at = memory.get("created_at")
    return str(created_at) if created_at else ""


def _memory_is_older_than_days(memory: dict[str, Any], older_than_days: int) -> bool:
    """True when ``memory`` is older than ``older_than_days``; malformed dates are fresh."""
    if older_than_days is None:
        return False
    created_at = memory.get("created_at")
    if not created_at:
        return False
    try:
        created = parse_iso(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - created
        return age.days >= older_than_days
    except (ValueError, TypeError, OSError):
        return False


def _memory_within_max_age(memory: dict[str, Any], max_age_days: Optional[int]) -> bool:
    if max_age_days is None:
        return True
    return not _memory_is_older_than_days(memory, max_age_days)


def _memory_scope_weight(memory: dict[str, Any]) -> float:
    scope = memory.get("scope")
    if isinstance(scope, str):
        return _SCOPE_WEIGHTS.get(scope, _DEFAULT_SCOPE_WEIGHT)
    return _DEFAULT_SCOPE_WEIGHT


def _memory_recency_factor(memory: dict[str, Any]) -> float:
    """Freshness multiplier for retrieval ranking; malformed dates count as fresh."""
    created_at = memory.get("created_at")
    if not created_at:
        return 1.0
    try:
        created = parse_iso(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).days
    except (ValueError, TypeError, OSError):
        return 1.0
    if age_days <= _RECENCY_FULL_DAYS:
        return 1.0
    if age_days >= _RECENCY_FLOOR_DAYS:
        return _RECENCY_FLOOR
    span = _RECENCY_FLOOR_DAYS - _RECENCY_FULL_DAYS
    progress = (age_days - _RECENCY_FULL_DAYS) / span
    return 1.0 - progress * (1.0 - _RECENCY_FLOOR)


def _memory_haystack(memory: dict[str, Any]) -> str:
    return " ".join(
        str(memory.get(key, ""))
        for key in ["scope", "statement", "evidence", "adapter", "role", "topic"]
    ).lower()


def _memory_term_overlap(terms: set[str], haystack: str) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(1, len(terms))


def _memory_retrieval_score(
    memory: dict[str, Any],
    terms: set[str],
) -> tuple[float, float, str, float]:
    haystack = _memory_haystack(memory)
    overlap = _memory_term_overlap(terms, haystack)
    scope_weight = _memory_scope_weight(memory)
    recency = _memory_recency_factor(memory)
    if terms:
        score = overlap * scope_weight * recency
    else:
        score = scope_weight * recency
    confidence = _coerce_confidence(memory.get("confidence"))
    created_at_key = _memory_created_at_sort_key(memory)
    return score, confidence, created_at_key, overlap


_WINDOWS = os.name == "nt"


def _retry_on_windows_lock(operation):
    """Run a filesystem op, retrying briefly on a Windows sharing-violation.

    Windows raises ``PermissionError`` (errno 13) when one process holds a file
    open while another tries to ``os.replace``/read it. The JSON store is touched
    by the orchestrator and worker subprocesses concurrently, so a task file can
    be read mid-rewrite. On POSIX these ops are atomic and the loop succeeds on
    the first try, so this is a Windows-only safety net with no POSIX cost.
    """
    last_error: Optional[PermissionError] = None
    for attempt in range(_WINDOWS_LOCK_RETRIES):
        try:
            return operation()
        except PermissionError as error:
            last_error = error
            time.sleep(_WINDOWS_LOCK_BACKOFF_SECONDS * (attempt + 1))
    raise last_error  # type: ignore[misc]


def _prepare_for_persistence(value: Any) -> Any:
    if isinstance(value, Task):
        return replace(
            value,
            payload=redact_payload_for_storage(value.payload),
        )
    return value


from puppetmaster.store_contracts import StoreContracts


def _capability_values(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def task_matches_capabilities(
    task: Task,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when a worker's advertised adapters/labels can execute ``task``.

    Omitted ``capabilities`` matches everything (legacy workers). An explicit
    empty ``adapters`` list matches nothing. Placement is optional: a task
    without ``payload.placement`` / ``payload.required_labels`` is unscoped.
    """
    if not capabilities:
        return True
    adapters = _capability_values(capabilities.get("adapters"))
    if adapters is not None and str(task.adapter or "local") not in adapters:
        return False
    labels = _capability_values(capabilities.get("labels"))
    if labels is not None:
        required = _capability_values(
            (task.payload or {}).get("placement")
            or (task.payload or {}).get("required_labels")
        )
        if required and not set(required).intersection(labels):
            return False
    return True


class SwarmStore(StoreContracts):
    """File-backed coordination store with Redis-like key spaces."""

    backend_name = "file"
    max_task_attempts = 3
    # Monotonic, process-wide source of unique temp-file suffixes for atomic
    # writes. itertools.count() is thread-safe for next() under CPython's GIL.
    _temp_counter = itertools.count()

    def __init__(self, root: Optional[Union[Path, str]] = None) -> None:
        self.root = resolve_state_dir(root)
        self.jobs_dir = self.root / "jobs"
        self.memory_dir = self.root / "memory"
        self.stream_dir = self.root / "streams"
        self.locks_dir = self.root / "locks"
        # job_id -> (last seen file size, line count). Streams are append-only
        # JSONL, so an unchanged size means an unchanged line count; this lets
        # event_cursor skip re-counting the whole file on every poll.
        self._event_cursor_cache: dict[str, tuple[int, int]] = {}
        self._incarnation = None
        self._metadata_initialized = False
        self._budget_locks = threading.local()
        self._read_selection = selection(self)

    def init(self) -> None:
        for directory in [
            self.root,
            self.jobs_dir,
            self.memory_dir,
            self.stream_dir,
            self.locks_dir,
        ]:
            mkdir_private(directory)
        if self.backend_name == "file" and not self._metadata_initialized:
            from puppetmaster.projections import initialize_file
            initialize_file(self)
            self._metadata_initialized = True
            self._read_selection = selection(self)
        from puppetmaster.host_lifecycle import record_host_start

        record_host_start(self)

    @staticmethod
    def launch_fingerprint(
        goal: str,
        label: Optional[str] = None,
        request: Optional[dict[str, Any]] = None,
    ) -> str:
        value = json.dumps(
            {
                "goal": " ".join(str(goal).split()),
                "label": label or "",
                "request": to_jsonable(request or {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def create_job(
        self,
        goal: str,
        *,
        label: Optional[str] = None,
        budget_policy: Optional[BudgetPolicy] = None,
        origin: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        launch_key: Optional[str] = None,
        launch_fingerprint: Optional[str] = None,
    ) -> Job:
        return self.create_or_get_job(
            goal,
            label=label,
            origin=origin,
            project_id=project_id,
            session_id=session_id,
            budget_policy=budget_policy,
            launch_key=launch_key,
            launch_fingerprint=launch_fingerprint,
        )[0]

    def create_or_get_job(
        self,
        goal: str,
        *,
        label: Optional[str] = None,
        budget_policy: Optional[BudgetPolicy] = None,
        origin: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        launch_key: Optional[str] = None,
        launch_fingerprint: Optional[str] = None,
    ) -> tuple[Job, bool]:
        self.init()
        fingerprint = launch_fingerprint or self.launch_fingerprint(goal, label)
        job = Job(
            goal=goal,
            label=label,
            origin=origin,
            project_id=project_id,
            session_id=session_id,
            budget_policy=budget_policy,
            launch_key=launch_key,
            launch_fingerprint=fingerprint if launch_key else None,
        )
        if launch_key:
            lock_name = f"launch:{launch_key}"
            owner = f"{os.getpid()}:{threading.get_ident()}"
            acquired = self.acquire_lock(lock_name, owner, ttl_seconds=300)
            if not acquired:
                # A concurrent creator may be between lock acquisition and its
                # first durable write. Give that creator a bounded hand-off
                # window so identical retries converge on one job.
                for _ in range(100):
                    if any(item.launch_key == launch_key for item in self.list_jobs()):
                        break
                    time.sleep(0.02)
                    acquired = self.acquire_lock(lock_name, owner, ttl_seconds=300)
                    if acquired:
                        break
            # Search after acquiring too: a completed prior launch leaves no
            # lock, so idempotent retries must inspect durable jobs on both
            # sides of the lock race.
            for existing in self.list_jobs():
                if existing.launch_key != launch_key:
                    continue
                if acquired:
                    self.release_lock(lock_name, owner)
                if (existing.launch_fingerprint != fingerprint or
                        existing.budget_policy != budget_policy or
                        existing.origin != origin or
                        existing.project_id != project_id or
                        existing.session_id != session_id):
                    raise LaunchConflictError(
                        "launch_key already belongs to a different request"
                    )
                return existing, False
            if not acquired:
                raise RuntimeError("launch_key is currently being created")
        else:
            owner = None
        try:
            job_dir = self.job_dir(job.id)
            for directory in [
                job_dir,
                job_dir / "tasks",
                job_dir / "runs",
                job_dir / "artifacts",
                job_dir / "edges",
                job_dir / "summaries",
            ]:
                mkdir_private(directory)
            self.write_json(job_dir / "job.json", job)
            payload: dict[str, Any] = {"goal": goal}
            if label is not None:
                payload["label"] = label
            if launch_key:
                payload["launch_key"] = launch_key
            self.emit(job.id, "job.created", payload)
            return job, True
        finally:
            if owner is not None:
                self.release_lock(f"launch:{launch_key}", owner)

    def save_job(self, job: Job) -> None:
        """Persist a job record. Additive fields round-trip through JSON."""
        self.write_json(self.job_dir(job.id) / "job.json", job)

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        actor: Optional[str] = None,
    ) -> Job:
        job = self.get_job(job_id)
        refused = self._refuse_worker_job_completion(job, status, actor)
        if refused is not None:
            return refused
        updated = self._job_with_status(job, status)
        if is_cost_final_job_status(status):
            updated = maybe_stamp_terminal_cost_receipt(self, updated)
        self.save_job(updated)
        self.emit(job_id, "job.status", {"status": str(status), "actor": actor or "coordinator"})
        return updated

    def _refuse_worker_job_completion(
        self,
        job: Job,
        status: JobStatus,
        actor: Optional[str],
    ) -> Optional[Job]:
        from puppetmaster.metr_seams import (
            REASON_WORKER_JOB_COMPLETE,
            is_worker_actor,
        )

        if not is_worker_actor(actor) if actor is not None else False:
            return None
        if actor is None:
            return None
        if status not in {
            JobStatus.COMPLETE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.STITCHING,
        }:
            return None
        self.emit(
            job.id,
            "job.complete_refused",
            {
                "reason": REASON_WORKER_JOB_COMPLETE,
                "status": str(status),
                "actor": actor,
            },
        )
        return job

    @staticmethod
    def _job_with_status(job: Job, status: JobStatus) -> Job:
        terminal = status in {
            JobStatus.COMPLETE,
            JobStatus.FAILED,
            JobStatus.STALLED,
            JobStatus.CANCELLED,
        }
        # Recoverable / in-flight statuses must not keep a prior receipt so
        # later work cannot reuse stale economics.
        cost_receipt = (
            job.cost_receipt if is_cost_final_job_status(status) else None
        )
        return replace(
            job,
            status=status,
            completed_at=now_iso() if terminal else job.completed_at,
            cost_receipt=cost_receipt,
        )

    @staticmethod
    def _task_saved_payload(task: Task) -> dict[str, Any]:
        return {
            "task_id": task.id,
            "role": task.role,
            "status": str(task.status),
            "adapter": task.adapter,
        }

    def save_task(self, task: Task) -> None:
        mkdir_private(self.job_dir(task.job_id) / "edges")
        self._write_json_retrying_admission(
            self.job_dir(task.job_id) / "tasks" / f"{task.id}.json", task
        )
        self.emit(
            task.job_id,
            "task.saved",
            self._task_saved_payload(task),
        )
        self._materialize_depends_on_edges(task)
        self._mark_graph_edges_materialized(task.job_id)

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            self.save_task(task)

    # Defaults for DeLM-inspired adaptive enqueue (Wave 3). Override per call.
    max_enqueue_depth = 3
    max_enqueue_children_per_parent = 8
    max_enqueue_job_tasks = 64

    def _task_ancestry_depth(self, task: Task, task_map: dict[str, Task]) -> int:
        """Count depends_on hops from ``task`` toward roots (cycle-safe)."""
        depth = 0
        seen: set[str] = set()
        current: Optional[Task] = task
        while current is not None and current.depends_on:
            parent_id = current.depends_on[0]
            if not parent_id or parent_id in seen:
                break
            seen.add(parent_id)
            depth += 1
            current = task_map.get(parent_id)
        return depth

    def _emit_enqueue_refused(
        self,
        job_id: str,
        reason: str,
        *,
        parent_task_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {"reason": reason}
        if parent_task_id:
            payload["parent_task_id"] = parent_task_id
        if extra:
            payload.update(extra)
        self.emit(job_id, "task.enqueue_refused", payload)

    def enqueue_subtask(
        self,
        job_id: str,
        *,
        parent_task_id: str,
        role: str,
        instruction: str,
        adapter: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        max_depth: Optional[int] = None,
        max_children_per_parent: Optional[int] = None,
        max_job_tasks: Optional[int] = None,
        created_by: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> Optional[Task]:
        """Durably enqueue a follow-up task linked to ``parent_task_id``.

        Lease-safe adaptive expansion (DeLM-inspired): workers/orchestrator may
        propose bounded follow-ups without removing the coordinator. Returns
        None when depth/count/idempotency gates refuse the enqueue.
        Graph is the only dispatcher: worker proposals stay same-job children
        of the producing parent. Recruit/HOLD/VETO/mailbox from workers are
        refused.
        """
        from puppetmaster.metr_seams import (
            REASON_WORKER_PROTOCOL,
            is_coordination_protocol_payload,
            is_worker_actor,
        )

        parent = self.get_task_by_id(parent_task_id)
        if parent.job_id != job_id:
            raise ValueError(
                f"parent task {parent_task_id} belongs to job {parent.job_id}, not {job_id}"
            )
        role_text = str(role or "").strip()
        instruction_text = str(instruction or "").strip()
        if not role_text or not instruction_text:
            raise ValueError("enqueue_subtask requires non-empty role and instruction")

        origin = actor
        if origin is None and created_by and is_worker_actor(created_by):
            origin = created_by
        proposal = {
            "role": role_text,
            "instruction": instruction_text,
            **(payload or {}),
        }
        if is_coordination_protocol_payload(proposal) or is_coordination_protocol_payload(
            {"role": role_text, "instruction": instruction_text}
        ):
            # Coordinator/host HOLD/VETO live as first-class store methods, not
            # worker-declared tasks.
            if origin is None or is_worker_actor(origin):
                self._emit_enqueue_refused(
                    job_id,
                    REASON_WORKER_PROTOCOL,
                    parent_task_id=parent_task_id,
                    extra={"role": role_text, "actor": origin or "worker"},
                )
                return None

        depth_limit = (
            self.max_enqueue_depth if max_depth is None else max(0, int(max_depth))
        )
        child_limit = (
            self.max_enqueue_children_per_parent
            if max_children_per_parent is None
            else max(0, int(max_children_per_parent))
        )
        if max_children_per_parent is None:
            try:
                from puppetmaster.continuous_plan import follow_up_limit_for, is_planner_task

                if is_planner_task(parent):
                    child_limit = max(child_limit, follow_up_limit_for(parent) + 1)
            except Exception:
                pass
        job_limit = (
            self.max_enqueue_job_tasks
            if max_job_tasks is None
            else max(1, int(max_job_tasks))
        )

        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        if parent.id not in task_map:
            task_map[parent.id] = parent
        existing_children = [
            task
            for task in tasks
            if parent_task_id in (task.depends_on or [])
            and bool((task.payload or {}).get("enqueued_from_parent"))
        ]
        fingerprint_src = f"{parent_task_id}\n{role_text}\n{instruction_text}"
        fingerprint = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()[:16]
        for task in existing_children:
            if (task.payload or {}).get("enqueue_fingerprint") == fingerprint:
                self._materialize_depends_on_edges(task)
                self.emit(
                    job_id,
                    "task.enqueue_deduped",
                    {
                        "parent_task_id": parent_task_id,
                        "task_id": task.id,
                        "enqueue_fingerprint": fingerprint,
                    },
                )
                return task

        parent_depth = self._task_ancestry_depth(parent, task_map)
        child_depth = parent_depth + 1
        if child_depth > depth_limit:
            self.emit(
                job_id,
                "task.enqueue_refused",
                {
                    "reason": "max_depth",
                    "parent_task_id": parent_task_id,
                    "depth": child_depth,
                    "max_depth": depth_limit,
                },
            )
            return None
        if len(tasks) >= job_limit:
            self.emit(
                job_id,
                "task.enqueue_refused",
                {
                    "reason": "max_job_tasks",
                    "parent_task_id": parent_task_id,
                    "task_count": len(tasks),
                    "max_job_tasks": job_limit,
                },
            )
            return None
        if len(existing_children) >= child_limit:
            self.emit(
                job_id,
                "task.enqueue_refused",
                {
                    "reason": "max_children_per_parent",
                    "parent_task_id": parent_task_id,
                    "child_count": len(existing_children),
                    "max_children_per_parent": child_limit,
                },
            )
            return None

        try:
            job = self.get_job(job_id)
        except Exception:
            job = None
        if job is not None and job.budget_policy is not None:
            try:
                check_admission(
                    job.budget_policy,
                    self.budget_snapshot(job_id)["reservations"],
                )
            except BudgetUnsettled:
                pass  # its dispatch waits for the running siblings to settle
            except BudgetAdmissionError as exc:
                self._emit_enqueue_refused(
                    job_id,
                    "budget_exhausted",
                    parent_task_id=parent_task_id,
                    extra={"detail": str(exc)},
                )
                return None

        child_payload = dict(payload or {})
        child_payload["enqueued_from_parent"] = True
        child_payload["parent_task_id"] = parent_task_id
        child_payload["enqueue_fingerprint"] = fingerprint
        child_payload["enqueue_depth"] = child_depth
        if created_by:
            child_payload["enqueued_by"] = created_by
        child = Task(
            job_id=job_id,
            role=role_text,
            instruction=instruction_text,
            adapter=adapter or parent.adapter or "local",
            payload=child_payload,
            depends_on=[parent_task_id],
            status=TaskStatus.QUEUED,
        )
        self.save_task(child)
        self.emit(
            job_id,
            "task.enqueued",
            {
                "task_id": child.id,
                "parent_task_id": parent_task_id,
                "role": child.role,
                "adapter": child.adapter,
                "enqueue_depth": child_depth,
                "enqueue_fingerprint": fingerprint,
                "created_by": created_by,
            },
        )
        return child

    def maybe_enqueue_follow_ups_from_artifact(
        self,
        artifact: Artifact,
        *,
        parent_task_id: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 4,
        cwd: Optional[Union[str, Path]] = None,
        retry_failures: bool = False,
    ) -> list[Task]:
        """Enqueue follow-ups declared on ``artifact.payload['enqueue_subtasks']``.

        Each entry is ``{"role": "...", "instruction": "..."}`` (optional
        ``adapter``). Best-effort: refusals/dedupes are skipped; never raises
        into the worker hot path unless retry_failures is requested. Worker
        protocol (recruit/HOLD/VETO/mailbox),
        foreign job_id, a new job, a parent that is not the producing task,
        and merge/ship after a failed GATE are refused.
        """
        from puppetmaster.metr_seams import (
            REASON_CROSS_JOB,
            REASON_GATE_FAILED,
            REASON_NEW_JOB,
            REASON_PARENT_MISMATCH,
            REASON_WORKER_PROTOCOL,
            artifact_is_failed_gate,
            gate_failed_for_task,
            is_coordination_protocol_payload,
            is_ship_or_merge_proposal,
            proposal_foreign_job_id,
            proposal_foreign_parent_id,
            proposal_requests_new_job,
        )
        from puppetmaster.negative_claims import (
            REASON_NEGATIVE_CLAIM,
            enqueue_negative_scope,
            resolve_negative_cwd,
            should_skip_negative,
        )

        producing_id = artifact.task_id
        parent_id = parent_task_id or producing_id
        payload = artifact.payload or {}
        if parent_id != producing_id:
            self._emit_enqueue_refused(
                artifact.job_id,
                REASON_PARENT_MISMATCH,
                parent_task_id=producing_id,
                extra={"requested_parent_task_id": parent_id},
            )
            return []
        if is_coordination_protocol_payload(payload) or is_coordination_protocol_payload(
            artifact
        ):
            # Protocol on the artifact itself (not just enqueue_subtasks).
            if not payload.get("enqueue_subtasks"):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_WORKER_PROTOCOL,
                    parent_task_id=parent_id,
                    extra={"artifact_id": artifact.id},
                )
                return []
        proposals = payload.get("enqueue_subtasks")
        if not isinstance(proposals, list) or not proposals:
            if is_coordination_protocol_payload(payload):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_WORKER_PROTOCOL,
                    parent_task_id=parent_id,
                    extra={"artifact_id": artifact.id},
                )
            return []
        try:
            from puppetmaster.jev.edges import apply_stop_spawn

            if apply_stop_spawn(
                self, artifact.job_id, task_id=str(producing_id or "")
            ):
                return []
        except Exception:
            pass
        from puppetmaster.continuous_plan import (
            REASON_INTENT_SPEC,
            follow_up_limit_for,
            intent_spec_for_job,
            is_planner_task,
        )

        producing = None
        try:
            producing = self.get_task_by_id(producing_id)
        except Exception:
            producing = None
        if producing is not None:
            limit = follow_up_limit_for(producing, limit)
            if is_planner_task(producing) and intent_spec_for_job(
                self.list_artifacts(artifact.job_id)
            ) is None:
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_INTENT_SPEC,
                    parent_task_id=parent_id,
                    extra={"artifact_id": artifact.id},
                )
                return []
        gate_failed = artifact_is_failed_gate(artifact) or gate_failed_for_task(
            self, artifact.job_id, producing_id
        )
        resolved_cwd = resolve_negative_cwd(self, artifact, cwd)
        created: list[Task] = []
        for proposal in proposals[: max(0, int(limit))]:
            if not isinstance(proposal, dict):
                continue
            if proposal_requests_new_job(proposal):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_NEW_JOB,
                    parent_task_id=parent_id,
                    extra={"artifact_id": artifact.id},
                )
                continue
            foreign_job = proposal_foreign_job_id(proposal, artifact.job_id)
            if foreign_job:
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_CROSS_JOB,
                    parent_task_id=parent_id,
                    extra={"requested_job_id": foreign_job},
                )
                continue
            foreign_parent = proposal_foreign_parent_id(proposal, producing_id)
            if foreign_parent:
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_PARENT_MISMATCH,
                    parent_task_id=producing_id,
                    extra={"requested_parent_task_id": foreign_parent},
                )
                continue
            if is_coordination_protocol_payload(proposal):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_WORKER_PROTOCOL,
                    parent_task_id=parent_id,
                    extra={"artifact_id": artifact.id},
                )
                continue
            if gate_failed and (
                artifact_is_failed_gate(artifact) or is_ship_or_merge_proposal(proposal)
            ):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_GATE_FAILED,
                    parent_task_id=parent_id,
                    extra={"role": proposal.get("role")},
                )
                continue
            role = str(proposal.get("role") or "").strip()
            instruction = str(
                proposal.get("instruction") or proposal.get("goal") or ""
            ).strip()
            if not role or not instruction:
                continue
            negative_scope = enqueue_negative_scope(artifact, proposal, instruction)
            if should_skip_negative(
                self,
                artifact.job_id,
                instruction,
                negative_scope,
                cwd=resolved_cwd,
            ):
                self._emit_enqueue_refused(
                    artifact.job_id,
                    REASON_NEGATIVE_CLAIM,
                    parent_task_id=parent_id,
                    extra={"role": role, "instruction": instruction},
                )
                continue
            try:
                child = self.enqueue_subtask(
                    artifact.job_id,
                    parent_task_id=parent_id,
                    role=role,
                    instruction=instruction,
                    adapter=proposal.get("adapter"),
                    created_by=created_by or artifact.created_by,
                    actor="worker",
                )
            except Exception:
                if retry_failures:
                    raise
                continue
            if child is not None:
                created.append(child)
        return created

    def update_task_status(
        self,
        task: Task,
        status: TaskStatus,
        worker_id: Optional[str] = None,
        lease_id: Optional[str] = None,
    ) -> Task:
        stored = self.get_task_by_id(task.id)
        assert_legal_task_transition(stored.status, status)
        # The caller carries the lease token granted at claim time; default to
        # the claimed task's own ``lease_id`` so existing call sites fence
        # correctly without having to thread the token through explicitly.
        expected_lease = lease_id if lease_id is not None else task.lease_id
        updated = self._build_status_update(stored, status)
        terminal = task_is_terminal(status)
        if terminal and worker_id is not None and not self._lease_matches(
            stored, worker_id, expected_lease
        ):
            return stored
        return self._atomic_status_update(
            task.id,
            updated,
            terminal=terminal,
            worker_id=worker_id,
            expected_lease=expected_lease,
        )

    @staticmethod
    def _build_status_update(stored: Task, status: TaskStatus) -> Task:
        timed = apply_running_duration(stored, status)
        terminal = task_is_terminal(status)
        return replace(
            timed,
            status=status,
            lease_owner=None if terminal else stored.lease_owner,
            lease_expires_at=None if terminal else stored.lease_expires_at,
            lease_id=None if terminal else stored.lease_id,
            updated_at=now_iso(),
            completed_at=now_iso() if task_is_satisfied(status) else stored.completed_at,
        )

    def _atomic_status_update(
        self,
        task_id: str,
        updated: Task,
        *,
        terminal: bool,
        worker_id: Optional[str],
        expected_lease: Optional[str],
    ) -> Task:
        if terminal and worker_id is not None:
            current = self.get_task_by_id(task_id)
            if not self._lease_matches(current, worker_id, expected_lease):
                return current
        self.save_task(updated)
        return updated

    @staticmethod
    def _lease_matches(
        task: Task, worker_id: str, expected_lease: Optional[str]
    ) -> bool:
        """True when ``worker_id`` (and, when known, the per-claim ``lease_id``)
        still owns ``task``.

        Owner identity is the baseline fence; the lease token is the stronger
        one that survives a worker_id reuse across a stale-lease reclaim. We
        only require the token when both sides actually have one, so pre-claim
        callers and older persisted tasks keep working.
        """
        if task.lease_owner != worker_id:
            return False
        if expected_lease is not None and task.lease_id is not None:
            return task.lease_id == expected_lease
        return True

    def claim_task(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        task_map: Optional[dict[str, Task]] = None,
    ) -> Optional[Task]:
        lock_name = f"task:{task_id}"
        # Cover the read retry budget even for workers with very short leases.
        lock_ttl = max(lease_seconds * 3, lease_seconds + 1, 30)
        if not self.acquire_lock(lock_name, worker_id, ttl_seconds=lock_ttl):
            return None
        try:
            return self._claim_task_locked(
                task_id, worker_id, lease_seconds=lease_seconds, task_map=task_map
            )
        finally:
            self.release_lock(lock_name, owner=worker_id)

    def _claim_task_locked(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        task_map: Optional[dict[str, Task]] = None,
    ) -> Optional[Task]:
        return self._perform_claim(
            task_id, worker_id, lease_seconds=lease_seconds, task_map=task_map
        )

    def _perform_claim(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        task_map: Optional[dict[str, Task]] = None,
    ) -> Optional[Task]:
        task = self.get_task_by_id(task_id)
        if self._claim_precheck(task, task_map=task_map, worker_id=worker_id):
            return None
        claimed = self._build_claimed_task(task, worker_id, lease_seconds)
        if not self._atomic_claim(task_id, task, claimed, worker_id=worker_id):
            return None
        self.emit(
            task.job_id,
            "task.claimed",
            self._task_claim_payload(task.id, worker_id, claimed),
        )
        return claimed

    def _claim_precheck(
        self,
        task: Task,
        *,
        task_map: Optional[dict[str, Task]] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Shared claim decision logic. Returns True when the claim attempt must abort."""
        from puppetmaster.metr_seams import (
            REASON_SUBGRAPH_WRITER,
            foreign_active_writer,
        )

        from puppetmaster.store_contracts import task_binding
        # Retry only the read fence, while the caller still owns the claim lock.
        # Never replay precheck mutations or the claim itself.
        deadline = time.monotonic() + 5.0
        delay = 0.01
        while True:
            try:
                cancelled = self.cancellation_pending(self._claim_job_ref(task.job_id), task_binding(task))
                break
            except sqlite3.OperationalError as exc:
                code = getattr(exc, 'sqlite_errorcode', None)
                if isinstance(exc, ReadUnavailable):
                    transient = str(exc) in (
                        'unable to open database: active reader; sidecars may be missing',
                        'unable to open database: live sidecars; retry after checkpoint',
                    ) and (code is None or isinstance(code, int) and (code & 0xff) in (5, 6))
                else:
                    transient = (isinstance(code, int) and (code & 0xff) in (5, 6)) if code is not None else str(exc) in (
                        'database is locked', 'database table is locked', 'database schema is locked',
                    )
                remaining = deadline - time.monotonic()
                if not transient or remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                if time.monotonic() >= deadline:
                    raise
                delay = min(delay * 2, 0.1)
        if cancelled:
            return True
        if task.status == TaskStatus.RUNNING and self._has_pending_completion(task):
            return True
        if not self.dependencies_complete(task, task_map=task_map):
            blocked = apply_running_duration(task, TaskStatus.BLOCKED)
            blocked = replace(blocked, status=TaskStatus.BLOCKED, updated_at=now_iso())
            self.save_task(blocked)
            return True
        if task.status == TaskStatus.COMPLETE:
            return True
        if task.attempts >= self.max_task_attempts:
            failed = apply_running_duration(task, TaskStatus.FAILED)
            failed = replace(
                failed,
                status=TaskStatus.FAILED,
                lease_owner=None,
                lease_expires_at=None,
                updated_at=now_iso(),
            )
            self.save_task(failed)
            self.emit(
                task.job_id,
                "task.max_attempts_exceeded",
                {"task_id": task.id, "attempts": task.attempts},
            )
            return True
        if task.status == TaskStatus.RUNNING and not self.is_task_stale(task):
            return True
        if worker_id:
            # Lease state must be live. claim_next_task's in-memory task_map is
            # a dependency snapshot and can still show siblings as QUEUED.
            foreign = foreign_active_writer(self, task, worker_id)
            if foreign:
                self.emit(
                    task.job_id,
                    "task.claim_refused",
                    {
                        "reason": REASON_SUBGRAPH_WRITER,
                        "task_id": task.id,
                        "owner": foreign,
                        "worker_id": worker_id,
                    },
                )
                return True
        return False

    @staticmethod
    def _build_claimed_task(task: Task, worker_id: str, lease_seconds: int) -> Task:
        timed = apply_running_duration(task, TaskStatus.RUNNING)
        return replace(
            timed,
            status=TaskStatus.RUNNING,
            attempts=task.attempts + 1,
            generation=(task.generation or 0) + 1,
            lease_owner=worker_id,
            lease_expires_at=seconds_from_now(lease_seconds),
            lease_id=new_id("lease"),
            updated_at=now_iso(),
        )

    @staticmethod
    def _task_claim_payload(
        task_id: str, worker_id: str, claimed: Task
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "worker_id": worker_id,
            "lease_expires_at": claimed.lease_expires_at,
            "attempts": claimed.attempts,
        }

    def _atomic_claim(
        self,
        task_id: str,
        task: Task,
        claimed: Task,
        worker_id: Optional[str] = None,
    ) -> bool:
        claim_snapshot = self._task_claim_snapshot(task)
        try:
            if not self._save_task_if_matches(task_id, claim_snapshot, claimed):
                return False
        except ProjectionWriteAdmissionError:
            # The durable task file has not changed. Treat unavailable
            # projection admission like a lost claim so run_until_idle can
            # retry from fresh state instead of killing the worker.
            return False
        return True

    def renew_task_lease(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        lease_id: Optional[str] = None,
    ) -> Optional[Task]:
        task = self.get_task_by_id(task_id)
        if task.status != TaskStatus.RUNNING or not self._lease_matches(
            task, worker_id, lease_id
        ):
            return None
        renewed = self._build_renewed_task(task, lease_seconds)
        return self._atomic_renew_lease(task_id, task, renewed, worker_id, lease_id)

    @staticmethod
    def _build_renewed_task(task: Task, lease_seconds: int) -> Task:
        return replace(
            task,
            lease_expires_at=seconds_from_now(lease_seconds),
            updated_at=now_iso(),
        )

    def _atomic_renew_lease(
        self,
        task_id: str,
        task: Task,
        renewed: Task,
        worker_id: str,
        lease_id: Optional[str],
    ) -> Optional[Task]:
        self.save_task(renewed)
        self.emit(
            task.job_id,
            "task.lease_renewed",
            {
                "task_id": task.id,
                "worker_id": worker_id,
                "lease_expires_at": renewed.lease_expires_at,
            },
        )
        return renewed

    def claim_next_task(
        self,
        job_id: str,
        worker_id: str,
        role: Optional[str] = None,
        lease_seconds: int = 60,
        capabilities: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Task]:
        for task, task_map in self._iter_claim_candidates(
            job_id,
            worker_id,
            role=role,
            capabilities=capabilities,
            persist_blocked=True,
        ):
            # Peers can drain the queue while this sweep holds an old snapshot.
            # Avoid a known-losing claim; claim_task still fences races after
            # this advisory read under the backend's normal claim protection.
            if self.get_task_by_id(task.id).status != TaskStatus.QUEUED:
                continue
            claimed = self.claim_task(
                task.id, worker_id, lease_seconds=lease_seconds, task_map=task_map
            )
            if claimed is not None:
                return claimed
        return None

    def peek_next_task(
        self,
        job_id: str,
        worker_id: str,
        role: Optional[str] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Task]:
        """Return the next claimable task without taking a lease or flipping status.

        Blocked dependents stay queued in the peek path so a liveness probe
        cannot mutate the job. Candidate order matches ``claim_next_task``.
        """
        for task, _task_map in self._iter_claim_candidates(
            job_id,
            worker_id,
            role=role,
            capabilities=capabilities,
            persist_blocked=False,
        ):
            return task
        return None

    def _iter_claim_candidates(
        self,
        job_id: str,
        worker_id: str,
        role: Optional[str] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
        *,
        persist_blocked: bool,
    ) -> Iterable[tuple[Task, dict[str, Task]]]:
        # Unblock is a supervisor tick (recover/refresh loops), not a claim side
        # effect. Workers must not refresh_blocked_tasks on every claim.
        # Load the job's tasks once and resolve dependency status from the
        # in-memory map instead of re-fetching each dependency by id (which is
        # a per-edge file glob / SQLite SELECT on every claim sweep).
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        if len(tasks) > 1:
            # Workers otherwise stampede the first queued task, then repeat the
            # same losing writer reservation for every task already claimed by
            # a peer. A stable per-worker rotation spreads those first CAS
            # attempts across the queue without changing eligibility or
            # allowing a claim outside the store's atomic fence.
            digest = hashlib.sha256(worker_id.encode("utf-8")).digest()
            offset = int.from_bytes(digest[:8], "big") % len(tasks)
            tasks = tasks[offset:] + tasks[:offset]
        for task in tasks:
            if task.status != TaskStatus.QUEUED:
                continue
            if not self.dependencies_complete(task, task_map=task_map):
                if persist_blocked:
                    self.save_task(replace(task, status=TaskStatus.BLOCKED, updated_at=now_iso()))
                continue
            if role is not None and task.role != role:
                continue
            if not task_matches_capabilities(task, capabilities):
                continue
            yield task, task_map

    def recover_stale_tasks(self, job_id: str) -> list[Task]:
        self.reconcile_completions(job_id)
        recovered: list[Task] = []
        for task in self.list_tasks(job_id):
            if not self.is_task_stale(task) or self._has_pending_completion(task):
                continue
            queued = self._build_recovered_task(task)
            if not self._atomic_recover_stale(task, queued):
                continue
            self.release_lock(f"task:{task.id}")
            self.emit(
                job_id,
                "task.recovered",
                {"task_id": task.id, "previous_owner": task.lease_owner},
            )
            recovered.append(queued)
        return recovered

    @staticmethod
    def _build_recovered_task(task: Task) -> Task:
        timed = apply_running_duration(task, TaskStatus.QUEUED)
        return replace(
            timed,
            status=TaskStatus.QUEUED,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=now_iso(),
        )

    def _atomic_recover_stale(self, task: Task, queued: Task) -> bool:
        try:
            self.save_task(queued)
        except ProjectionWriteAdmissionError:
            # Same lost-tick contract as claim: the task file is unchanged.
            return False
        return True

    def refresh_blocked_tasks(self, job_id: str) -> list[Task]:
        ready: list[Task] = []
        # Hard-failed upstreams cascade BLOCKED descendants to FAILED first so
        # COMPLETE-only unblocking never promotes a permanently doomed child.
        self.propagate_hard_dependency_failures(job_id)
        # Build the dependency lookup once and thread it through
        # dependencies_complete, mirroring claim_next_task — otherwise each
        # blocked task triggers one get_task_by_id() per dependency on every
        # claim sweep (an N+1 file glob / SQLite SELECT).
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        for task in tasks:
            if task.status != TaskStatus.BLOCKED:
                continue
            if not self.dependencies_complete(task, task_map=task_map):
                continue
            queued = replace(task, status=TaskStatus.QUEUED, updated_at=now_iso())
            self.save_task(queued)
            self.emit(job_id, "task.unblocked", {"task_id": task.id, "role": task.role})
            ready.append(queued)
        return ready

    def dependencies_complete(
        self,
        task: Task,
        task_map: Optional[dict[str, Task]] = None,
    ) -> bool:
        for dependency_id in task.depends_on:
            dependency: Optional[Task]
            if task_map is not None:
                dependency = task_map.get(dependency_id)
                if dependency is None:
                    return False
            else:
                try:
                    dependency = self.get_task_by_id(dependency_id)
                except FileNotFoundError:
                    return False
            if not task_is_satisfied(dependency.status):
                return False
        return True

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        """Persist a typed graph edge idempotently (identity is endpoint tuple)."""
        if not edge.id:
            edge = make_graph_edge(
                job_id=edge.job_id,
                type=edge.type,
                from_kind=edge.from_kind,
                from_id=edge.from_id,
                to_kind=edge.to_kind,
                to_id=edge.to_id,
                created_at=edge.created_at,
                meta=edge.meta,
            )
        edges_dir = self.job_dir(edge.job_id) / "edges"
        mkdir_private(edges_dir)
        path = edges_dir / f"{edge.id}.json"
        existing: Optional[GraphEdge] = None
        if path.exists():
            try:
                existing = graph_edge_from_dict(self.read_json(path))
            except Exception:
                existing = None
        # Preserve the original created_at on idempotent re-upsert.
        if existing is not None:
            merged_meta = {**existing.meta, **(edge.meta or {})}
            if merged_meta == existing.meta:
                return existing
            edge = GraphEdge(
                id=existing.id,
                job_id=existing.job_id,
                type=existing.type,
                from_kind=existing.from_kind,
                from_id=existing.from_id,
                to_kind=existing.to_kind,
                to_id=existing.to_id,
                created_at=existing.created_at,
                meta=merged_meta,
            )
        self.write_json(path, edge)
        self.emit(
            edge.job_id,
            "edge.upserted",
            {
                "edge_id": edge.id,
                "type": str(edge.type),
                "from_id": edge.from_id,
                "to_id": edge.to_id,
            },
        )
        return edge

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> list[GraphEdge]:
        return [self.upsert_edge(edge) for edge in edges]

    def delete_edge(self, job_id: str, edge_id: str) -> bool:
        """Remove one persisted edge. Returns True when a file was deleted."""
        path = self.job_dir(job_id) / "edges" / f"{edge_id}.json"
        if not path.exists():
            return False
        # Stale depends_on reconcile unlinks edge files while workers may still
        # hold them open on Windows — retry the sharing violation like write_json.
        _retry_on_windows_lock(path.unlink)
        self.emit(job_id, "edge.deleted", {"edge_id": edge_id})
        return True

    def get_edge(self, job_id: str, edge_id: str) -> Optional[GraphEdge]:
        self.ensure_graph_edges(job_id)
        path = self.job_dir(job_id) / "edges" / f"{edge_id}.json"
        if not path.exists():
            return None
        return graph_edge_from_dict(self.read_json(path))

    def list_edges(
        self,
        job_id: str,
        *,
        edge_type: Optional[Union[GraphEdgeType, str]] = None,
        from_id: Optional[str] = None,
        to_id: Optional[str] = None,
        from_kind: Optional[Union[GraphNodeKind, str]] = None,
        to_kind: Optional[Union[GraphNodeKind, str]] = None,
    ) -> list[GraphEdge]:
        self.ensure_graph_edges(job_id)
        return self._list_edges_from_disk(
            job_id,
            edge_type=edge_type,
            from_id=from_id,
            to_id=to_id,
            from_kind=from_kind,
            to_kind=to_kind,
        )

    def _list_edges_from_disk(
        self,
        job_id: str,
        *,
        edge_type: Optional[Union[GraphEdgeType, str]] = None,
        from_id: Optional[str] = None,
        to_id: Optional[str] = None,
        from_kind: Optional[Union[GraphNodeKind, str]] = None,
        to_kind: Optional[Union[GraphNodeKind, str]] = None,
    ) -> list[GraphEdge]:
        edges_dir = self.job_dir(job_id) / "edges"
        if not edges_dir.exists():
            return []
        wanted_type = str(edge_type) if edge_type is not None else None
        wanted_from_kind = str(from_kind) if from_kind is not None else None
        wanted_to_kind = str(to_kind) if to_kind is not None else None
        edges: list[GraphEdge] = []
        for path in sorted(edges_dir.glob("*.json")):
            edge = graph_edge_from_dict(self.read_json(path))
            if wanted_type is not None and str(edge.type) != wanted_type:
                continue
            if from_id is not None and edge.from_id != from_id:
                continue
            if to_id is not None and edge.to_id != to_id:
                continue
            if wanted_from_kind is not None and str(edge.from_kind) != wanted_from_kind:
                continue
            if wanted_to_kind is not None and str(edge.to_kind) != wanted_to_kind:
                continue
            edges.append(edge)
        return edges

    def _graph_edges_marker(self, job_id: str) -> Path:
        return self.job_dir(job_id) / _GRAPH_EDGES_MARKER

    def _mark_graph_edges_materialized(self, job_id: str) -> None:
        marker = self._graph_edges_marker(job_id)
        if marker.exists():
            return
        mkdir_private(self.job_dir(job_id))
        marker.write_text("2\n", encoding="utf-8")
        chmod_private_file(marker)

    def ensure_graph_edges(self, job_id: str) -> None:
        """Lazy-backfill depends_on/produces edges for pre-graph file jobs."""
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            return
        # Always replay interrupted consumes journals (idempotent upserts).
        self._replay_consumes_journals(job_id)
        marker = self._graph_edges_marker(job_id)
        if marker.exists():
            return
        mkdir_private(job_dir / "edges")
        for task in self.list_tasks(job_id):
            self._reconcile_depends_on_edges(task, list_edges=self._list_edges_from_disk)
        for artifact in self.list_artifacts(job_id):
            self._materialize_produces_edge(artifact)
        self._mark_graph_edges_materialized(job_id)

    def job_graph(self, job_id: str) -> dict[str, Any]:
        """Read-only nodes+edges snapshot for CLI/MCP graph queries."""
        self.ensure_graph_edges(job_id)
        tasks = self.list_tasks(job_id)
        artifacts = self.list_artifacts(job_id)
        edges = self.list_edges(job_id)
        nodes: list[dict[str, Any]] = []
        for task in tasks:
            nodes.append(
                {
                    "id": task.id,
                    "kind": str(GraphNodeKind.TASK),
                    "role": task.role,
                    "status": str(task.status),
                }
            )
        for artifact in artifacts:
            nodes.append(
                {
                    "id": artifact.id,
                    "kind": str(GraphNodeKind.ARTIFACT),
                    "type": str(artifact.type),
                    "task_id": artifact.task_id,
                }
            )
        return {
            "job_id": job_id,
            "nodes": nodes,
            "edges": [to_jsonable(edge) for edge in edges],
        }

    def _reconcile_depends_on_edges(
        self,
        task: Task,
        *,
        list_edges=None,
    ) -> list[GraphEdge]:
        """Upsert current depends_on edges and drop stale ones for ``task``."""
        list_fn = list_edges or self.list_edges
        desired_ids = {dependency_id for dependency_id in task.depends_on if dependency_id}
        existing = list_fn(
            task.job_id,
            edge_type=GraphEdgeType.DEPENDS_ON,
            from_id=task.id,
            from_kind=GraphNodeKind.TASK,
            to_kind=GraphNodeKind.TASK,
        )
        for edge in existing:
            if edge.to_id not in desired_ids:
                self.delete_edge(task.job_id, edge.id)
        edges = [
            make_graph_edge(
                job_id=task.job_id,
                type=GraphEdgeType.DEPENDS_ON,
                from_kind=GraphNodeKind.TASK,
                from_id=task.id,
                to_kind=GraphNodeKind.TASK,
                to_id=dependency_id,
            )
            for dependency_id in task.depends_on
            if dependency_id
        ]
        return self.upsert_edges(edges)

    def _materialize_depends_on_edges(self, task: Task) -> list[GraphEdge]:
        return self._reconcile_depends_on_edges(
            task, list_edges=self._list_edges_from_disk
        )

    def _materialize_produces_edge(self, artifact: Artifact) -> GraphEdge:
        return self.upsert_edge(
            make_graph_edge(
                job_id=artifact.job_id,
                type=GraphEdgeType.PRODUCES,
                from_kind=GraphNodeKind.TASK,
                from_id=artifact.task_id,
                to_kind=GraphNodeKind.ARTIFACT,
                to_id=artifact.id,
            )
        )

    def _consumes_journal_path(self, job_id: str, task_id: str) -> Path:
        safe_task = self._safe_key(task_id)
        return self.job_dir(job_id) / f"{_CONSUMES_JOURNAL_PREFIX}{safe_task}.json"

    def _replay_consumes_journals(self, job_id: str) -> None:
        """Replay any crash-left consumes journals (idempotent upserts)."""
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            return
        for path in sorted(job_dir.glob(f"{_CONSUMES_JOURNAL_PREFIX}*.json")):
            try:
                payload = self.read_json(path)
            except Exception:
                continue
            raw_edges = payload.get("edges") if isinstance(payload, dict) else None
            if not isinstance(raw_edges, list):
                _retry_on_windows_lock(path.unlink)
                continue
            edges = [graph_edge_from_dict(item) for item in raw_edges if isinstance(item, dict)]
            if edges:
                self.upsert_edges(edges)
            if path.exists():
                _retry_on_windows_lock(path.unlink)

    def record_consumes(
        self,
        job_id: str,
        task_id: str,
        artifact_ids: Iterable[str],
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> list[GraphEdge]:
        """Record task→artifact consumes edges (idempotent, crash-recoverable).

        File backend: durable journal of intended edges is written first, then
        each edge is upserted, then the journal is cleared. A crash mid-batch
        leaves the journal for :meth:`_replay_consumes_journals` (called from
        :meth:`ensure_graph_edges` / the next ``record_consumes``). Upserts are
        identity-keyed, so replay is safe.
        """
        self._replay_consumes_journals(job_id)
        edges = [
            make_graph_edge(
                job_id=job_id,
                type=GraphEdgeType.CONSUMES,
                from_kind=GraphNodeKind.TASK,
                from_id=task_id,
                to_kind=GraphNodeKind.ARTIFACT,
                to_id=artifact_id,
                meta=meta,
            )
            for artifact_id in dict.fromkeys(artifact_ids)
            if artifact_id
        ]
        if not edges:
            return []
        journal_path = self._consumes_journal_path(job_id, task_id)
        mkdir_private(self.job_dir(job_id))
        self.write_json(
            journal_path,
            {
                "job_id": job_id,
                "task_id": task_id,
                "edges": [to_jsonable(edge) for edge in edges],
            },
        )
        try:
            result = self.upsert_edges(edges)
        except Exception:
            # Leave the journal so the next open/replay can finish the batch.
            raise
        if journal_path.exists():
            _retry_on_windows_lock(journal_path.unlink)
        return result

    def record_derived_from(
        self,
        job_id: str,
        source_artifact_id: str,
        derived_artifact_id: str,
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> GraphEdge:
        """Idempotent artifact→artifact ``DERIVED_FROM`` edge (same-job only).

        ``derived`` is modeled as *from* and ``source`` as *to*, matching
        ``make_graph_edge`` identity tests: the derived artifact is derived from
        the source. Both artifacts must already belong to ``job_id``.
        """
        if not source_artifact_id or not derived_artifact_id:
            raise ValueError("source_artifact_id and derived_artifact_id are required")
        if source_artifact_id == derived_artifact_id:
            raise ValueError("derived artifact cannot be derived from itself")
        by_id = self.get_artifacts_by_ids(
            job_id, [source_artifact_id, derived_artifact_id]
        )
        if source_artifact_id not in by_id:
            raise ValueError(
                f"source artifact {source_artifact_id!r} not found in job {job_id!r}"
            )
        if derived_artifact_id not in by_id:
            raise ValueError(
                f"derived artifact {derived_artifact_id!r} not found in job {job_id!r}"
            )
        return self.upsert_edge(
            make_graph_edge(
                job_id=job_id,
                type=GraphEdgeType.DERIVED_FROM,
                from_kind=GraphNodeKind.ARTIFACT,
                from_id=derived_artifact_id,
                to_kind=GraphNodeKind.ARTIFACT,
                to_id=source_artifact_id,
                meta=meta,
            )
        )

    def prepare_superseded_artifacts(
        self, job_id: str, task_ids: Iterable[str]
    ) -> list[Artifact]:
        """Return artifact copies stamped ``superseded`` (not persisted).

        Preserves audit history (no artifact deletion). Additive when a
        validation block was absent. Idempotent for already-stale/superseded
        artifacts (those are omitted from the returned list).
        """
        from puppetmaster.validation import (
            validation_status_of,
            with_validation_status,
        )

        selected = {task_id for task_id in task_ids if task_id}
        if not selected:
            return []
        artifact_ids: list[str] = []
        for task_id in selected:
            for edge in self.list_edges(
                job_id,
                edge_type=GraphEdgeType.PRODUCES,
                from_id=task_id,
                from_kind=GraphNodeKind.TASK,
                to_kind=GraphNodeKind.ARTIFACT,
            ):
                artifact_ids.append(edge.to_id)
        if not artifact_ids:
            for artifact in self.list_artifacts(job_id):
                if artifact.task_id in selected:
                    artifact_ids.append(artifact.id)
        by_id = self.get_artifacts_by_ids(job_id, artifact_ids)
        prepared: list[Artifact] = []
        for artifact_id in dict.fromkeys(artifact_ids):
            artifact = by_id.get(artifact_id)
            if artifact is None:
                continue
            status = validation_status_of(artifact)
            if status in {"stale", "superseded"}:
                continue
            validation = (getattr(artifact, "payload", None) or {}).get("validation")
            if isinstance(validation, dict) and validation.get("generation") is not None:
                try:
                    generation = int(validation["generation"]) + 1
                except (TypeError, ValueError):
                    generation = 1
            else:
                generation = 1
            prepared.append(
                with_validation_status(
                    artifact, "superseded", generation=generation
                )
            )
        return prepared

    def mark_produced_artifacts_superseded(
        self, job_id: str, task_ids: Iterable[str]
    ) -> list[str]:
        """Persist ``payload.validation.status=superseded`` on produced artifacts.

        Returns the artifact ids that were updated.
        """
        prepared = self.prepare_superseded_artifacts(job_id, task_ids)
        for artifact in prepared:
            self.save_artifact(artifact)
        return [artifact.id for artifact in prepared]

    def lookup_artifacts_by_validation_fingerprint(
        self,
        fingerprint: str,
        *,
        types: Optional[Iterable[Union[ArtifactType, str]]] = None,
        job_ids: Optional[Iterable[str]] = None,
        include_statuses: Optional[Iterable[str]] = None,
        limit: int = 256,
    ) -> list[Artifact]:
        """Fingerprint-aware lookup of reusable substantive artifacts.

        Scans completed FINDING/VERIFICATION/DECISION artifacts (by default)
        for a matching ``payload.validation.fingerprint`` with status
        ``fresh`` or ``reused``. Excludes ``stale`` / ``superseded`` and
        unlabeled legacy artifacts. Bounded by ``limit`` (default 256); no
        schema migration — walks typed indexes / job artifacts.
        """
        from puppetmaster.validation import (
            DEFAULT_LOOKUP_LIMIT,
            SUBSTANTIVE_VALIDATION_TYPES,
            filter_artifacts_by_validation_fingerprint,
        )

        wanted = str(fingerprint or "").strip()
        if not wanted:
            return []
        bound = DEFAULT_LOOKUP_LIMIT if limit is None else max(0, int(limit))
        if bound == 0:
            return []
        type_list = (
            [str(item) for item in types]
            if types is not None
            else [str(item) for item in SUBSTANTIVE_VALIDATION_TYPES]
        )
        if job_ids is not None:
            jobs_to_scan = [job_id for job_id in dict.fromkeys(job_ids) if job_id]
        else:
            jobs = self.list_jobs()
            jobs_to_scan = [
                job.id
                for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)
            ]
        collected: list[Artifact] = []
        for job_id in jobs_to_scan:
            if len(collected) >= bound:
                break
            batch: list[Artifact] = []
            for artifact_type in type_list:
                batch.extend(
                    self.list_artifacts_by_type(artifact_type, job_ids=[job_id])
                )
            matched = filter_artifacts_by_validation_fingerprint(
                batch,
                wanted,
                types=type_list,
                include_statuses=include_statuses,
                limit=bound - len(collected),
            )
            collected.extend(matched)
        return collected[:bound]

    def resolve_artifacts_via_edges(
        self,
        task: Task,
        *,
        record_consumes: bool = True,
    ) -> list[Artifact]:
        """Resolve artifacts produced by upstream ``depends_on`` tasks via edges.

        Returns an empty list when no produces edges exist so callers can fall
        back to the legacy whole-job artifact load.
        """
        if not task.depends_on:
            return []
        artifact_ids: list[str] = []
        for dependency_id in task.depends_on:
            for edge in self.list_edges(
                task.job_id,
                edge_type=GraphEdgeType.PRODUCES,
                from_id=dependency_id,
                from_kind=GraphNodeKind.TASK,
                to_kind=GraphNodeKind.ARTIFACT,
            ):
                artifact_ids.append(edge.to_id)
        if not artifact_ids:
            return []
        by_id = self.get_artifacts_by_ids(task.job_id, artifact_ids)
        artifacts = [by_id[artifact_id] for artifact_id in artifact_ids if artifact_id in by_id]
        from puppetmaster.gist_admission import filter_shared_context_artifacts

        # Admission filter at the edge-resolution boundary so pending/rejected
        # gists never enter peer prompts. Raw list_artifacts remains unfiltered
        # for tooling/MCP.
        task_payload = getattr(task, "payload", None) or {}
        if not isinstance(task_payload, dict):
            task_payload = {}
        cwd = task_payload.get("cwd") or task_payload.get("workspace")
        if cwd == "":
            cwd = None
        artifacts = filter_shared_context_artifacts(
            artifacts,
            for_job_id=task.job_id,
            cwd=cwd,
            store=self,
        )
        if record_consumes and artifacts:
            self.record_consumes(
                task.job_id,
                task.id,
                [artifact.id for artifact in artifacts],
            )
        return artifacts

    def _recoverable_failed_task_ids(
        self,
        job_id: str,
        *,
        artifacts: Optional[list[Artifact]] = None,
    ) -> set[str]:
        from puppetmaster.workers import RECOVERABLE_FAILURES

        if artifacts is None:
            artifacts = self.list_artifacts(job_id)
        latest_at: dict[str, str] = {}
        recoverable: set[str] = set()
        for artifact in artifacts:
            failure = (artifact.payload or {}).get("failure")
            if failure not in RECOVERABLE_FAILURES:
                continue
            task_id = artifact.task_id
            if task_id not in latest_at or artifact.created_at >= latest_at[task_id]:
                latest_at[task_id] = artifact.created_at
                recoverable.add(task_id)
        # An explicit edge owns the failure until its atomic policy transition
        # has run. Do not cascade descendants or let legacy provider fallback
        # race that transition.
        from puppetmaster.failure_policy import task_failure_policy

        for task in self.list_tasks(job_id):
            if task.status != TaskStatus.FAILED or task_failure_policy(task.payload) is None:
                continue
            # Pending explicit edges keep the coordinator alive long enough to
            # apply their transition. Once handled, they must not re-enter
            # provider fallback/review recovery.
            if (task.payload or {}).get("failure_policy_handled_generation") != task.generation:
                recoverable.add(task.id)
            else:
                recoverable.discard(task.id)
        # A later non-recoverable failure artifact for the same task should win.
        for artifact in artifacts:
            failure = (artifact.payload or {}).get("failure")
            if failure in RECOVERABLE_FAILURES or failure is None:
                continue
            task_id = artifact.task_id
            if task_id in latest_at and artifact.created_at >= latest_at[task_id]:
                recoverable.discard(task_id)
                latest_at[task_id] = artifact.created_at
        return recoverable

    def _latest_failure_reason(self, task: Task) -> str:
        latest: Optional[Artifact] = None
        for artifact in self.list_artifacts(task.job_id):
            if artifact.task_id != task.id:
                continue
            payload = artifact.payload or {}
            if payload.get("failure") is None and payload.get("result") not in {
                "failed", "blocked"
            }:
                continue
            if latest is None or artifact.created_at >= latest.created_at:
                latest = artifact
        if latest is None:
            return "task_failed"
        payload = latest.payload or {}
        return str(payload.get("failure") or payload.get("result") or "task_failed")

    @staticmethod
    def _failure_decision_artifact(task: Task, decision: dict[str, Any]) -> Artifact:
        digest = hashlib.sha256(
            json.dumps(decision, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        return Artifact(
            id=f"artifact_failure_reroute_{digest}",
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.DECISION,
            created_by="failure-policy",
            payload={
                "decision": decision["action"],
                "why": decision["reason"],
                **decision,
            },
            confidence=1.0,
            evidence=[
                f"task:{task.id}",
                f"generation:{decision['generation']}",
                f"source_status:{decision['source_status']}",
            ],
        )

    def _ensure_failure_decision(self, task: Task) -> None:
        decision = (task.payload or {}).get("failure_policy_last_decision")
        if not isinstance(decision, dict):
            return
        artifact = self._failure_decision_artifact(task, decision)
        if artifact.id not in self.get_artifacts_by_ids(task.job_id, [artifact.id]):
            self.save_artifact(artifact)
            self.emit(task.job_id, "task.failure_reroute", decision)

    def apply_failure_policy(self, task_id: str) -> Optional[Task]:
        """Apply one explicit FAILED edge under a per-task serialization lock."""
        owner = new_id("failure-policy")
        lock_name = f"task:{task_id}"
        if not self.acquire_lock(lock_name, owner, ttl_seconds=30):
            return None
        try:
            task = self.get_task_by_id(task_id)
            self._ensure_failure_decision(task)
            from puppetmaster.failure_policy import task_failure_policy

            policy = task_failure_policy(task.payload)
            if policy is None:
                return None
            payload = dict(task.payload or {})
            if payload.get("failure_policy_handled_generation") == task.generation:
                return task
            if task.status != TaskStatus.FAILED:
                return None
            reason = self._latest_failure_reason(task)
            cancelled = reason in {"cancelled", "canceled", "cancellation"}
            count = int(payload.get("failure_policy_retry_count", 0))
            limit = int(policy.get("retries", 0))
            requested = str(policy["action"])
            action = requested
            if requested == "retry":
                action = "retry" if count < limit and not cancelled else "abort"
            target = {
                "retry": TaskStatus.QUEUED,
                "continue": TaskStatus.SKIPPED,
                "abort": TaskStatus.FAILED,
            }[action]
            if action == "retry":
                count += 1
            next_generation = task.generation + 1 if action == "retry" else task.generation
            decision = {
                "kind": "failure_reroute",
                "task_id": task.id,
                "action": action,
                "configured_action": requested,
                "source_status": str(task.status),
                "target_status": str(target),
                "counter": count,
                "limit": limit,
                "generation": task.generation,
                "reason": "explicit_cancellation" if cancelled else reason,
                "allow_routing_change": bool(policy.get("allow_routing_change", False)),
            }
            payload.update(
                failure_policy_retry_count=count,
                failure_policy_handled_generation=task.generation,
                failure_policy_last_decision=decision,
            )
            timed = apply_running_duration(task, target)
            updated = replace(
                timed,
                status=target,
                payload=payload,
                attempts=0 if action == "retry" else task.attempts,
                generation=next_generation,
                lease_owner=None,
                lease_expires_at=None,
                lease_id=None,
                completed_at=now_iso() if target == TaskStatus.SKIPPED else None,
                updated_at=now_iso(),
            )
            self.save_task(updated)
            self._ensure_failure_decision(updated)
            return updated
        finally:
            self.release_lock(lock_name, owner=owner)

    def apply_pending_failure_policies(self, job_id: str) -> list[Task]:
        changed: list[Task] = []
        self.finalize_pending_cuts(job_id)
        for task in self.list_tasks(job_id):
            if task.status != TaskStatus.FAILED:
                continue
            updated = self.apply_failure_policy(task.id)
            if updated is not None:
                changed.append(updated)
        return changed

    def cut_task(
        self,
        job_id: str,
        task_id: str,
        *,
        expected_binding: Optional["TaskBinding"] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Persist a lease-bound cut request; RUNNING work stops cooperatively."""
        from dataclasses import asdict
        from puppetmaster.contracts import TaskBinding
        from puppetmaster.store_contracts import task_binding

        owner = new_id("cut")
        lock_name = f"task:{task_id}"
        if not self.acquire_lock(lock_name, owner, ttl_seconds=30):
            return {"outcome": "busy", "job_id": job_id, "task_id": task_id}
        try:
            task = self.get_task_by_id(task_id)
            if task.job_id != job_id:
                raise ValueError("cut target belongs to another job")
            binding = task_binding(task)
            if expected_binding is not None and expected_binding != binding:
                return {"outcome": "stale_binding", "binding": asdict(binding)}
            existing = (task.payload or {}).get("failure_cut")
            if isinstance(existing, dict):
                # The marker is saved before the cancellation request. If that
                # request failed (a busy store), a retry must finish it: an
                # early return here left the worker uncancelled for good.
                rid = str(existing.get("request_id") or "")
                if not rid or self.get_cancellation_receipt(self.job_ref(job_id), rid) is not None:
                    return {"outcome": existing.get("outcome", "pending"), **existing}
                marker = existing
                binding = TaskBinding(**marker["binding"])
            elif task_is_terminal(task.status):
                return {"outcome": "already_terminal", "binding": asdict(binding)}
            else:
                rid = request_id or f"failure-cut-{task.id}-{task.generation}"
                marker = {
                    "request_id": rid,
                    "binding": asdict(binding),
                    "outcome": "pending" if task.status == TaskStatus.RUNNING else "observed",
                }
                self.save_task(replace(task, payload={**task.payload, "failure_cut": marker}))
            receipt = self.request_cancellation(self.job_ref(job_id), rid, [binding])
        finally:
            self.release_lock(lock_name, owner=owner)
        if task.status != TaskStatus.RUNNING and receipt.outcome == "requested":
            self.observe_cancellation(self.job_ref(job_id), binding)
            self.finalize_pending_cuts(job_id)
            marker["outcome"] = "observed"
        return {
            "job_id": job_id,
            "task_id": task_id,
            "outcome": marker["outcome"],
            "binding": marker["binding"],
            "request_id": marker["request_id"],
            "cancellation_outcome": receipt.outcome,
        }

    def finalize_pending_cuts(self, job_id: str) -> list[Task]:
        """Turn observed/non-running cut generations into terminal SKIPPED nodes."""
        from puppetmaster.contracts import TaskBinding
        from puppetmaster.store_contracts import task_binding

        # A dead worker may leave RUNNING plus an expired lease. Reap only
        # that generation; an active lease is never cleared by a cut.
        self.recover_stale_tasks(job_id)
        changed: list[Task] = []
        for task in self.list_tasks(job_id):
            marker = (task.payload or {}).get("failure_cut")
            if not isinstance(marker, dict):
                continue
            raw_binding = marker.get("binding")
            if not isinstance(raw_binding, dict):
                continue
            binding = TaskBinding(**raw_binding)
            if binding.generation != task.generation:
                continue
            if task.status == TaskStatus.RUNNING:
                continue
            if task.status in {TaskStatus.COMPLETE, TaskStatus.SKIPPED}:
                continue
            payload = dict(task.payload)
            marker = {**marker, "outcome": "observed"}
            decision = {
                "kind": "failure_reroute",
                "task_id": task.id,
                "action": "cut",
                "configured_action": "cut",
                "source_status": str(task.status),
                "target_status": str(TaskStatus.SKIPPED),
                "counter": 0,
                "limit": 0,
                "generation": task.generation,
                "reason": "operator_cut",
                "allow_routing_change": False,
            }
            payload.update(
                failure_cut=marker,
                failure_policy_handled_generation=task.generation,
                failure_policy_last_decision=decision,
            )
            updated = replace(
                apply_running_duration(task, TaskStatus.SKIPPED),
                status=TaskStatus.SKIPPED,
                payload=payload,
                lease_owner=None,
                lease_expires_at=None,
                lease_id=None,
                completed_at=now_iso(),
                updated_at=now_iso(),
            )
            self.save_task(updated)
            self._ensure_failure_decision(updated)
            changed.append(updated)
        return changed

    def restore_task(
        self,
        job_id: str,
        task_id: str,
        *,
        expected_binding: Optional["TaskBinding"] = None,
    ) -> ResetSubgraphResult:
        from puppetmaster.store_contracts import task_binding

        task = self.get_task_by_id(task_id)
        if task.job_id != job_id:
            raise ValueError("restore target belongs to another job")
        if expected_binding is not None and expected_binding != task_binding(task):
            raise ValueError("stale task binding")
        return self.reset_subgraph(
            job_id, [task_id], include_descendants=True, clear_failure_state=True
        )

    def propagate_hard_dependency_failures(self, job_id: str) -> list[Task]:
        """Cascade hard-FAILED deps onto BLOCKED descendants as terminal FAILED.

        Recoverable adapter/billing failures leave dependents BLOCKED so the
        existing auto-fallback path can requeue the upstream. COMPLETE-only
        unblocking is unchanged.
        """
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        recoverable = self._recoverable_failed_task_ids(job_id)
        changed: list[Task] = []
        progress = True
        while progress:
            progress = False
            for task in list(task_map.values()):
                if task.status != TaskStatus.BLOCKED:
                    continue
                hard_failed = [
                    dep_id
                    for dep_id in task.depends_on
                    if (dep := task_map.get(dep_id)) is not None
                    and dep.status == TaskStatus.FAILED
                    and dep.id not in recoverable
                ]
                if not hard_failed:
                    continue
                failed = replace(
                    task,
                    status=TaskStatus.FAILED,
                    lease_owner=None,
                    lease_expires_at=None,
                    lease_id=None,
                    updated_at=now_iso(),
                )
                self.save_task(failed)
                self.emit(
                    job_id,
                    "task.dependency_failed",
                    {
                        "task_id": task.id,
                        "role": task.role,
                        "failed_dependencies": hard_failed,
                    },
                )
                task_map[task.id] = failed
                changed.append(failed)
                progress = True
        return changed

    def consumer_closure(
        self, job_id: str, task_ids: Iterable[str]
    ) -> set[str]:
        """Selected tasks plus transitive dependents (consumers via depends_on)."""
        seeds = {task_id for task_id in task_ids if task_id}
        if not seeds:
            return set()
        tasks = self.list_tasks(job_id)
        dependents: dict[str, list[str]] = {}
        for task in tasks:
            for dependency_id in task.depends_on:
                dependents.setdefault(dependency_id, []).append(task.id)
        selected: set[str] = set()
        stack = list(seeds)
        while stack:
            task_id = stack.pop()
            if task_id in selected:
                continue
            selected.add(task_id)
            for child_id in dependents.get(task_id, []):
                if child_id not in selected:
                    stack.append(child_id)
        return selected

    @staticmethod
    def has_active_lease(task: Task) -> bool:
        """True when a RUNNING task still holds a non-expired worker lease."""
        if task.status != TaskStatus.RUNNING:
            return False
        if not task.lease_owner or not task.lease_expires_at:
            return False
        return not SwarmStore.is_task_stale(task)

    @staticmethod
    def _reopened_job_after_reset(job: Job) -> Job:
        """Coordinator job after an accepted non-empty reset.

        Production finalize only stamps COMPLETE from a non-complete status.
        Reopening to RUNNING and clearing ``completed_at`` / ``cost_receipt``
        lets the next finalize freeze the new generation exactly once.
        """
        return replace(
            job,
            status=JobStatus.RUNNING,
            completed_at=None,
            cost_receipt=None,
        )

    def reset_subgraph(
        self,
        job_id: str,
        task_ids: Iterable[str],
        *,
        include_descendants: bool = True,
        clear_failure_state: bool = False,
    ) -> ResetSubgraphResult:
        """Idempotent targeted rerun reset for selected (downstream) tasks.

        Clears lease/completion state and ``attempts`` for the selected set
        (and optionally their consumer closure), then re-derives QUEUED/BLOCKED
        from ``depends_on``. Completed upstream tasks, artifacts, and edges are
        retained. Produced outputs are labeled ``superseded`` (audit retained).

        Emits one canonical ``subgraph.reset`` event whose payload includes
        ``superseded_artifact_ids``. A non-empty accepted reset reopens the
        coordinator job to RUNNING and clears ``completed_at`` and
        ``cost_receipt``.

        Refuses the whole reset when any selected task still holds an active
        (non-expired RUNNING) lease, so a live worker cannot be fenced into
        emitting stale produces artifacts against a reset generation.
        """
        selected = (
            self.consumer_closure(job_id, task_ids)
            if include_descendants
            else {task_id for task_id in task_ids if task_id}
        )
        if not selected:
            return ResetSubgraphResult([], [])
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        active = [
            task.id
            for task in tasks
            if task.id in selected and self.has_active_lease(task)
        ]
        if active:
            raise ActiveTaskLeaseError(active)
        # Reopen and clear the job before mutating tasks/artifacts so a
        # crash cannot leave a cost-final job with a valid receipt over
        # superseded outputs. SQLite applies the same fields in one txn.
        self.save_job(self._reopened_job_after_reset(self.get_job(job_id)))
        reset: list[Task] = []
        for task in tasks:
            if task.id not in selected:
                continue
            payload = task.payload
            if clear_failure_state:
                from puppetmaster.failure_policy import clear_failure_edge_state

                payload = clear_failure_edge_state(payload)
            cleared = replace(
                task,
                status=TaskStatus.BLOCKED,
                attempts=0,
                generation=(task.generation or 0) + 1,
                lease_owner=None,
                lease_expires_at=None,
                lease_id=None,
                completed_at=None,
                payload=payload,
                updated_at=now_iso(),
            )
            self.save_task(cleared)
            task_map[task.id] = cleared
            reset.append(cleared)
        # Re-derive runnable status from dependencies (COMPLETE-only).
        finalized: list[Task] = []
        for task in reset:
            current = task_map[task.id]
            if self.dependencies_complete(current, task_map=task_map):
                queued = replace(
                    current, status=TaskStatus.QUEUED, updated_at=now_iso()
                )
                self.save_task(queued)
                task_map[task.id] = queued
                finalized.append(queued)
            else:
                finalized.append(current)
        superseded = self.mark_produced_artifacts_superseded(job_id, selected)
        self.emit(
            job_id,
            "subgraph.reset",
            {
                "task_ids": sorted(selected),
                "reset_count": len(finalized),
                "superseded_artifact_ids": superseded,
            },
        )
        return ResetSubgraphResult(finalized, superseded)

    def heartbeat_run(self, run: AgentRun) -> AgentRun:
        updated = replace(run, heartbeat_at=now_iso())
        self.save_run(updated)
        self.emit(
            run.job_id,
            "run.heartbeat",
            {"run_id": run.id, "worker_id": run.worker_id, "task_id": run.task_id},
        )
        return updated

    @property
    def incarnation(self):
        return self._read_incarnation()

    def attach(self) -> None:
        """Worker-safe open: bind identity with the worker attach budget.

        Same binding as SQLite attach: a supervisor mid-transaction delays a
        worker's first read, it does not fail worker startup.
        """
        from puppetmaster.sqlite_store import SQLiteSwarmStore, _SQLITE_LOCK_RETRY_ATTEMPTS
        budget = SQLiteSwarmStore.busy_timeout_ms / 1000 * _SQLITE_LOCK_RETRY_ATTEMPTS
        self._read_incarnation(attach_deadline=time.monotonic() + budget)

    def _read_incarnation(self, attach_deadline=None):
        from puppetmaster.identity import read_identity
        from puppetmaster.projections import connection
        from puppetmaster.readonly import retry_transient

        def read():
            with connection(self, metadata_only=True, attach_deadline=attach_deadline) as c:
                return read_identity(c, self.backend_name)
        return retry_transient(read, time.monotonic() + 5 if attach_deadline is None else attach_deadline)

    def _claim_job_ref(self, job_id):
        """Bind a local claim through its store session, including live WAL."""
        if selection(self) != self._read_selection:
            raise StoreIdentityError('store removed or replaced before claim')
        with projection_connection(self) as c:
            if not c.in_transaction:
                c.execute('BEGIN')
            ref = make_ref(self.root, job_id, read_identity(c, self.backend_name))
            self.validate_job_ref(ref, connection=c, strict=True)
            if selection(self) != self._read_selection:
                raise StoreIdentityError('store removed or replaced during claim')
            return ref

    def job_ref(self, job_id, *, _launch_binding=False):
        """Explicitly bind a job in the currently selected store (read-only)."""
        from puppetmaster.identity import make_ref
        from puppetmaster.projections import connection
        from puppetmaster.identity import read_identity
        with connection(self, metadata_only=True, launch_binding=_launch_binding) as c:
            if not c.in_transaction:
                c.execute("BEGIN")
            ref = make_ref(self.root, job_id, read_identity(c, self.backend_name))
            ref = self.validate_job_ref(ref, connection=c)
            if _launch_binding:
                from puppetmaster.readonly import selection
                from puppetmaster.identity import StoreIdentityError
                if selection(self) != self._read_selection:
                    raise StoreIdentityError('store removed or replaced during launch binding')
            return ref

    def bind_job_ref(self, job_ref, *, legacy_read=False):
        """Pin a continuation before opening any operation connection. Never migrate."""
        from puppetmaster.models import JobRef
        from puppetmaster.projections import connection
        if isinstance(job_ref, dict):
            job_ref = JobRef(**job_ref)
        with connection(self, metadata_only=True) as c:
            if not c.in_transaction:
                c.execute("BEGIN")
            self.validate_job_ref(job_ref, connection=c, strict=not legacy_read)
            if job_ref.version == 2:
                self._incarnation = job_ref.incarnation
            self._legacy_read_ref = job_ref if job_ref.version == 1 else None
        if self.backend_name == "sqlite":
            self._open_mode = "attach"
        return self

    def validate_job_ref(self, job_ref, *, connection=None, strict=False):
        from puppetmaster.identity import validate
        from puppetmaster.projections import connection as open_connection
        if connection is None:
            from puppetmaster.readonly import retry_transient

            def read():
                with open_connection(self, metadata_only=True) as c:
                    return self.validate_job_ref(job_ref, connection=c, strict=strict)
            return retry_transient(read, time.monotonic() + 5)
        validate(self, job_ref, connection, strict=strict)
        table = "jobs" if self.backend_name == "sqlite" else "projection_current"
        where = "id=?" if self.backend_name == "sqlite" else "kind='job' AND id=?"
        if not connection.execute(f"SELECT 1 FROM {table} WHERE {where}", (job_ref.job_id,)).fetchone():
            raise KeyError(job_ref.job_id)
        return job_ref

    @staticmethod
    def _summary_filters(filters, kwargs):
        from dataclasses import fields
        from puppetmaster.contracts import JobSummaryFilter
        if filters is None:
            return kwargs
        if not isinstance(filters, JobSummaryFilter):
            raise TypeError("filters must be a JobSummaryFilter")
        values = {field.name: getattr(filters, field.name) for field in fields(filters)
                  if getattr(filters, field.name) is not None}
        if values.keys() & kwargs.keys():
            raise ValueError("duplicate job summary filter")
        return {**values, **kwargs}

    def get_selected_economics(self, job_ref: JobRef, *, expected_summary_revision: Optional[int] = None) -> SelectedEconomics:
        from puppetmaster.selected_economics import read
        return read(self, job_ref, expected_summary_revision)

    def historical_evidence_counts(self, job_ref):
        from puppetmaster.history_metadata import counts
        return counts(self, job_ref)

    def list_attempt_refs(self, job_ref, **kwargs):
        from puppetmaster.history_metadata import page
        return page(self, "attempt", job_ref, **kwargs)

    def list_run_refs(self, job_ref, **kwargs):
        from puppetmaster.history_metadata import page
        return page(self, "run", job_ref, **kwargs)

    def list_process_outcome_refs(self, job_ref, **kwargs):
        from puppetmaster.history_metadata import page
        return page(self, "outcome", job_ref, **kwargs)

    def list_usage_observation_refs(self, job_ref, **kwargs):
        from puppetmaster.history_metadata import page
        return page(self, "observation", job_ref, **kwargs)

    def list_job_summaries(self, filters=None, **kwargs):
        from puppetmaster.projections import page
        return page(self, "job", **self._summary_filters(filters, kwargs))

    def read_job_summary_changes(self, filters=None, **kwargs):
        from puppetmaster.projections import page
        return page(self, "job", changes=True, **self._summary_filters(filters, kwargs))

    def list_task_refs(self, job_ref, **kwargs):
        from puppetmaster.projections import page
        return page(self, "task", job_ref, **kwargs)

    def list_artifact_refs(self, job_ref, **kwargs):
        from puppetmaster.projections import page
        return page(self, "artifact", job_ref, **kwargs)

    def repair_metadata_index(self):
        """Explicit file-store repair; source scans never occur during a page read.

        Call with writers stopped. Existing cursors expire when the index epoch advances.
        """
        if self.backend_name != "file":
            raise ValueError("SQLite projections are transactional")
        from puppetmaster.projections import connection, project_file
        self.init()
        with connection(self, write=True) as c:
            c.execute("""INSERT INTO projection_changes(kind,job_id,id,status,sha256,stamp,deleted,
                task_count,artifact_count,binding,task_id,artifact_type,scope)
                SELECT kind,job_id,id,status,sha256,'legacy_unknown',1,
                task_count,artifact_count,binding,task_id,artifact_type,scope FROM projection_current""")
            c.execute("DELETE FROM projection_current")
            c.execute("DELETE FROM historical_refs")
            c.execute("DELETE FROM completion_receipts")
            c.execute("DELETE FROM selected_economics_current")
            c.execute("DELETE FROM projection_pending")
            c.execute("UPDATE projection_meta SET value=CAST(value AS INTEGER)+1 WHERE key='epoch'")
            for path in self.jobs_dir.glob("*/job.json"):
                project_file(c, path, self.read_json(path), legacy=True)
                for directory in ("tasks", "artifacts", "runs", "completions", "consumption/attempts", "consumption/observations"):
                    for child in (path.parent / directory).glob("*.json"):
                        project_file(c, child, self.read_json(child), legacy=True)

    @contextmanager
    def _completion_scope(self, job_id: str):
        # The file backend uses its existing crash-expiring lock and atomic
        # rename journal. SQLite overrides this with a writer transaction.
        owner = f"completion-{os.getpid()}-{threading.get_ident()}"
        name = f"completion:{job_id}"
        if not self.acquire_lock(name, owner, ttl_seconds=300):
            yield False
            return
        try:
            yield True
        finally:
            self.release_lock(name, owner=owner)

    @contextmanager
    def _completion_intent_scope(self, job_id: str):
        owner = new_id("intent")
        name = f"completion-intent:{job_id}"
        deadline = time.monotonic() + 5.0
        while not self.acquire_lock(name, owner, ttl_seconds=300):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("completion intent busy; retry (waited 5 seconds)")
            time.sleep(min(0.01, remaining))
        try:
            yield
        finally:
            self.release_lock(name, owner=owner)

    def _has_pending_completion(self, task: Task) -> bool:
        return any(
            not record["done"] and record["task"]["id"] == task.id
            and record["task"].get("lease_id") == task.lease_id
            and record["run"]["worker_id"] == task.lease_owner
            for record in self._pending_completion_records(task.job_id)
        )

    def _save_completion(self, job_id: str, record: dict[str, Any]) -> None:
        self.write_json(self.job_dir(job_id) / "completions" / f"{record['run']['id']}.json", record)

    def _completion_records(self, job_id: str) -> list[dict[str, Any]]:
        return [
            self.read_json(path)
            for path in sorted((self.job_dir(job_id) / "completions").glob("*.json"))
        ]

    def _pending_completion_records(self, job_id: str) -> list[dict[str, Any]]:
        return [record for record in self._completion_records(job_id) if not record["done"]]

    def _get_completion(self, job_id: str, run_id: str):
        path = self._assert_safe_job_dir(job_id) / "completions" / f"{self._safe_key(run_id)}.json"
        return self.read_json(path) if path.exists() else None

    def get_completion_receipt(self, job_ref, run_id: str):
        from puppetmaster.completion_metadata import read
        return read(self, job_ref, run_id)

    def complete_task(
        self, task: Task, run: AgentRun, artifacts: list[Artifact],
        event_payload: dict[str, Any], *, job_ref=None,
    ) -> Task:
        """Journal accepted output before publishing children or terminal state.

        SQLite commits each replay in one writer transaction. File storage uses
        atomic rename for the intent, existing child fingerprints, and stream
        readback for event deduplication. It is process-crash recoverable, not a
        multi-file transaction or a power-loss/fsync guarantee. Its existing
        expiring lock requires publication to finish within 300 seconds; a
        killed publisher can delay recovery until that lock expires. Contention
        defers replay to the next poll; intent acquisition waits up to five
        seconds on a separate lock, without retrying the publication body.
        Concurrent file-backend claim/reset/lease writes retain weaker isolation.
        """
        from puppetmaster.contracts import ContractConflict, immutable_digest
        submission = {"task_id": task.id, "job_id": task.job_id,
                      "lease_id": task.lease_id, "worker_id": run.worker_id,
                      "run_id": run.id, "started_at": run.started_at,
                      "completed_at": run.completed_at,
                      "artifacts": to_jsonable(artifacts), "event_payload": event_payload}
        if not run.id or self._safe_key(run.id) != run.id:
            raise ValueError("invalid completion run id")
        digest = immutable_digest(submission)
        if run.job_id != task.job_id or run.task_id != task.id:
            raise ValueError("completion run does not belong to task")
        if any(a.job_id != task.job_id or a.task_id != task.id for a in artifacts):
            raise ValueError("completion artifact does not belong to task")
        with self._completion_intent_scope(task.job_id):
            if job_ref is not None:
                if job_ref.job_id != task.job_id:
                    raise ValueError("completion job_ref conflicts with task")
                from puppetmaster.projections import connection
                with connection(self) as c:
                    self.validate_job_ref(job_ref, connection=c, strict=True)
            existing = self._get_completion(task.job_id, run.id)
            if existing is not None:
                if existing.get("intent_digest") != digest:
                    raise ContractConflict("completion intent already has different or legacy content")
            else:
                current = self.get_task_by_id(task.id)
                if current.job_id != task.job_id:
                    raise ValueError("completion task belongs to a different job")
                if (current.status != TaskStatus.RUNNING
                        or current.lease_id != task.lease_id
                        or current.generation != task.generation
                        or not self._lease_matches(current, run.worker_id, task.lease_id)):
                    return current
                record = {
                    "task": to_jsonable(task), "run": to_jsonable(run),
                    "artifacts": to_jsonable(artifacts), "event_payload": event_payload,
                    "event_cursor": self.event_cursor(task.job_id), "done": False,
                    "intent_digest": digest, "publication": "pending_publication",
                }
                self._save_completion(task.job_id, record)
        # The intent must commit independently of the retryable publication.
        self.reconcile_completions(task.job_id)
        return self.get_task_by_id(task.id)

    def submit_completion(self, task, run, artifacts, event_payload, *, job_ref=None):
        from puppetmaster.contracts import CompletionReceipt
        from puppetmaster.identity import StoreIdentityError
        if job_ref is None:
            raise StoreIdentityError("submit_completion requires a v2 JobRef; inspect and bind with store.job_ref(job_id)")
        self.complete_task(task, run, artifacts, event_payload, job_ref=job_ref)
        ref = job_ref
        receipt = self.get_completion_receipt(ref, run.id)
        if receipt.outcome == "legacy_unknown":
            return CompletionReceipt(ref, run.id, None, "stale_lease")
        return receipt

    def reconcile_completions(self, job_id: str) -> None:
        if not self._pending_completion_records(job_id):
            return
        with self._completion_scope(job_id) as acquired:
            if not acquired:
                return
            # Re-read after acquiring publication ownership: a peer may have
            # published the probe's records or accepted another durable intent.
            for record in self._pending_completion_records(job_id):
                task = task_from_dict(record["task"])
                run = AgentRun(**{**record["run"], "status": TaskStatus.COMPLETE})
                current = self.get_task_by_id(task.id)
                already_complete = (current.status == TaskStatus.COMPLETE
                                    and current.completed_at == run.completed_at
                                    and current.generation == task.generation)
                owns_lease = (current.status == TaskStatus.RUNNING
                              and current.generation == task.generation
                              and current.lease_id == task.lease_id
                              and current.lease_owner == run.worker_id)
                if not already_complete and not owns_lease:
                    # Reset/reclaim invalidates the old execution's intent.
                    record["done"] = True
                    record["publication"] = "invalidated"
                    self._save_completion(job_id, record)
                    continue
                for raw in record["artifacts"]:
                    self.maybe_enqueue_follow_ups_from_artifact(
                        artifact_from_dict(raw), parent_task_id=task.id,
                        created_by=run.worker_id, cwd=(task.payload or {}).get("cwd"),
                        retry_failures=True,
                    )
                if not already_complete:
                    self.save_run(run)
                    updated = replace(self._build_status_update(current, TaskStatus.COMPLETE),
                                      completed_at=run.completed_at)
                    published = self._atomic_status_update(
                        task.id, updated, terminal=True, worker_id=run.worker_id,
                        expected_lease=task.lease_id,
                    )
                    if (published.status != TaskStatus.COMPLETE or
                            published.completed_at != run.completed_at):
                        continue
                try:
                    from puppetmaster.continuous_plan import maybe_requeue_planner

                    maybe_requeue_planner(self, task)
                except Exception:
                    pass
                events = self.read_events_since(job_id, record["event_cursor"])
                if not any(e["event"] == "worker.completed_task" and
                           e["payload"] == record["event_payload"] for e in events):
                    self.emit(job_id, "worker.completed_task", record["event_payload"])
                record["done"] = True
                if record.get("intent_digest"):
                    record["publication"] = "published"
                self._save_completion(job_id, record)

    def _ledger_dir(self, job_id: str) -> Path:
        return self._assert_safe_job_dir(job_id) / "consumption"

    @staticmethod
    def _ledger_key(*parts: str) -> str:
        return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()

    def _record_ledger_file(self, record: Union[ExecutionAttempt, UsageObservation]) -> bool:
        """Atomic rename under the existing crash-expiring lock.

        Contention raises (caller may retry). This is not a multi-record
        transaction or fsync durability guarantee; a writer paused beyond the
        300s lock TTL can race a reclaimer, as with other file-store locks.
        """
        with self._budget_scope(record.job_id):
            self._check_ledger_reservation(record)
            directory = self._ledger_dir(record.job_id)
            key = self._ledger_key(record.attempt_id)
            name = f"consumption:{record.job_id}:{key}"
            owner = new_id("ledger")
            if not self.acquire_lock(name, owner, ttl_seconds=300):
                raise RuntimeError("consumption ledger busy; retry the write")
            try:
                if isinstance(record, ExecutionAttempt):
                    path = directory / "attempts" / f"{key}.json"
                else:
                    if not (directory / "attempts" / f"{key}.json").exists():
                        raise ValueError("usage observation requires a recorded attempt")
                    path = directory / "observations" / (
                        self._ledger_key(record.attempt_id, record.observation_id) + ".json")
                if path.exists():
                    existing = type(record)(**self.read_json(path))
                    if canonical_record(existing) != canonical_record(record):
                        raise LedgerConflictError("ledger key already has different content")
                    return False
                self.write_json(path, json.loads(canonical_record(record)))
                return True
            finally:
                self.release_lock(name, owner=owner)

    def budget_dispatch_scope(self, job_id: str):
        """Keep file admission stages together; SQLite stages commit separately."""
        return self._budget_scope(job_id) if self.backend_name == "file" else nullcontext()

    @contextmanager
    def _budget_scope(self, job_id: str):
        """Job-wide admission lock; file backend has the existing 300s TTL limits."""
        self._assert_safe_job_dir(job_id)
        held = getattr(self._budget_locks, "held", None)
        if held is None:
            held = self._budget_locks.held = set()
        if job_id in held:
            yield
            return
        owner = new_id("budget")
        name = f"budget:{job_id}"
        if not self.acquire_lock(name, owner, ttl_seconds=300):
            raise BudgetBusy("budget busy; retry")
        held.add(job_id)
        try:
            yield
        finally:
            held.remove(job_id)
            self.release_lock(name, owner=owner)

    def _budget_records(self, job_id: str) -> list[dict[str, Any]]:
        directory = self._assert_safe_job_dir(job_id) / "budget"
        return sorted((self.read_json(path) for path in directory.glob("*.json")),
                      key=lambda record: record["attempt"]["attempt_id"])

    def _save_budget_record(self, record: dict[str, Any]) -> None:
        path = (self._assert_safe_job_dir(record["attempt"]["job_id"]) / "budget" /
                (self._ledger_key(record["attempt"]["attempt_id"]) + ".json"))
        self.write_json(path, record)

    @staticmethod
    def _check_invocation_identity(expected: ExecutionAttempt,
                                   actual: ExecutionAttempt) -> None:
        if canonical_record(expected) != canonical_record(actual):
            raise BudgetConflictError("invocation identity conflict")

    def _check_reserved_identity(self, record: dict[str, Any],
                                 job_id: str, attempt_id: str) -> None:
        expected = ExecutionAttempt(**record["attempt"])
        if (expected.job_id, expected.attempt_id) != (job_id, attempt_id):
            raise BudgetConflictError("invocation identity conflict")
        for actual in self.list_attempts(job_id):
            if actual.attempt_id == attempt_id:
                self._check_invocation_identity(expected, actual)

    def _check_ledger_reservation(self, record: Union[ExecutionAttempt, UsageObservation]) -> None:
        # Caller holds the same job lock/transaction as reservation transitions.
        try:
            reservation = self._get_budget_record(record.job_id, record.attempt_id)
        except KeyError:
            return
        if isinstance(record, ExecutionAttempt):
            self._check_invocation_identity(ExecutionAttempt(**reservation["attempt"]), record)
        if reservation["state"] == "released":
            raise BudgetConflictError("recorded invocation contradicts non-dispatch")

    def _get_budget_record(self, job_id: str, attempt_id: str) -> dict[str, Any]:
        for record in self._budget_records(job_id):
            if record["attempt"]["attempt_id"] == attempt_id:
                self._check_reserved_identity(record, job_id, attempt_id)
                return record
        raise KeyError(attempt_id)

    def reserve_dispatch(self, attempt: ExecutionAttempt,
                         allowance: BudgetLiability) -> dict[str, Any]:
        """Reserve one immutable invocation identity, without recording a dispatch.

        Allowance is a caller-supplied bound, not the router's marginal estimate.
        Exact replay returns the current record, including terminal states.
        """
        record = {"attempt": asdict(attempt), "allowance": asdict(allowance),
                  "state": "reserved", "adoption_id": None, "liability": None,
                  "reconciliations": {}, "release_proof": None}
        with self._budget_scope(attempt.job_id):
            job = self.get_job(attempt.job_id)
            records = self._budget_records(attempt.job_id)
            for existing in records:
                if existing["attempt"]["attempt_id"] == attempt.attempt_id:
                    self._check_reserved_identity(existing, attempt.job_id, attempt.attempt_id)
                    self._check_invocation_identity(ExecutionAttempt(**existing["attempt"]), attempt)
                    if existing["allowance"] != record["allowance"]:
                        raise BudgetConflictError("reservation identity has different facts")
                    return existing
            for item in self.list_attempts(attempt.job_id):
                if item.attempt_id == attempt.attempt_id:
                    self._check_invocation_identity(item, attempt)
                    raise BudgetConflictError("cannot reserve an already recorded invocation")
            check_admission(job.budget_policy, records + [record])
            self._save_budget_record(record)
            return record

    def adopt_dispatch(self, job_id: str, attempt_id: str,
                       *, adoption_id: str) -> dict[str, Any]:
        """Durably mark dispatch ownership BEFORE crossing an invocation boundary.

        A repeated adoption is a recovery read, never permission to invoke twice.
        """
        if not isinstance(adoption_id, str) or not adoption_id.strip():
            raise ValueError("adoption_id is required")
        with self._budget_scope(job_id):
            record = self._get_budget_record(job_id, attempt_id)
            if record["adoption_id"] is not None:
                if record["adoption_id"] != adoption_id:
                    raise BudgetConflictError("dispatch already adopted by another identity")
                return record
            if record["state"] != "reserved":
                raise BudgetConflictError("only a reserved invocation can be adopted")
            record.update(state="dispatching", adoption_id=adoption_id)
            self._save_budget_record(record)
            return record

    def reconcile_reservation(self, job_id: str, attempt_id: str, *,
                              reconciliation_id: str, liability: BudgetLiability,
                              final: bool, evidence: str) -> dict[str, Any]:
        """Replace cumulative liability once per source identity; never add deltas.

        final asserts complete invocation coverage (not that all prices are known).
        Unknown/partial final cost stays pending. New source IDs can resolve pending
        records; a settled result is immutable. Evidence identifies source authority.
        """
        if (not isinstance(reconciliation_id, str) or not reconciliation_id.strip() or
                not isinstance(evidence, str) or not evidence.strip() or type(final) is not bool):
            raise ValueError("reconciliation identity, evidence and boolean final required")
        event = {"liability": asdict(liability), "final": final, "evidence": evidence}
        with self._budget_scope(job_id):
            record = self._get_budget_record(job_id, attempt_id)
            previous = record["reconciliations"].get(reconciliation_id)
            if previous is not None:
                if previous != event:
                    raise BudgetConflictError("reconciliation identity has different facts")
                return record
            if record["state"] not in ("dispatching", "pending_reconciliation"):
                raise BudgetConflictError("reconciliation requires an unsettled dispatch")
            record["reconciliations"][reconciliation_id] = event
            record["liability"] = asdict(liability)
            record["state"] = ("settled" if final and liability.cost_state == "known"
                               else "pending_reconciliation")
            self._save_budget_record(record)
            return record

    def release_undispatched(self, job_id: str, attempt_id: str, *,
                            non_dispatch_proof: str) -> dict[str, Any]:
        """Release only before adoption, with caller evidence of non-dispatch."""
        if not isinstance(non_dispatch_proof, str) or not non_dispatch_proof.strip():
            raise ValueError("non-dispatch proof is required")
        with self._budget_scope(job_id):
            record = self._get_budget_record(job_id, attempt_id)
            if record["state"] == "released" and record["release_proof"] == non_dispatch_proof:
                return record
            if record["state"] != "reserved":
                raise BudgetConflictError("cannot release an adopted or released dispatch")
            if any(item.attempt_id == attempt_id for item in self.list_attempts(job_id)):
                raise BudgetConflictError("recorded invocation contradicts non-dispatch")
            record.update(state="released", release_proof=non_dispatch_proof)
            self._save_budget_record(record)
            return record

    def budget_snapshot(self, job_id: str) -> dict[str, Any]:
        """Consistent reservation liability only; does not sum telemetry observations."""
        with self._budget_scope(job_id):
            return self.budget_view(job_id)

    def budget_view(self, job_id: str) -> dict[str, Any]:
        """The same payload without the admission lock, for read-only viewers.

        A dashboard poll must not take the writer lock the workers reserve
        budget under; a point-in-time read is enough to display.
        """
        job = self.get_job(job_id)
        records = self._budget_records(job_id)
        return {"policy": asdict(job.budget_policy) if job.budget_policy else None,
                "totals": budget_totals(records), "reservations": records}

    def record_attempt(self, attempt: ExecutionAttempt) -> bool:
        """Insert immutable launch facts; True if new, False on exact replay."""
        return self._record_ledger_file(attempt)

    def record_usage_observation(self, observation: UsageObservation) -> bool:
        """Insert a source snapshot; does not update usage reports or events."""
        return self._record_ledger_file(observation)

    def list_attempts(self, job_id: str, *, task_id: Optional[str] = None) -> list[ExecutionAttempt]:
        """Legacy jobs return []; reset_subgraph never removes these records."""
        records = [ExecutionAttempt(**self.read_json(path)) for path in
                   (self._ledger_dir(job_id) / "attempts").glob("*.json")]
        return sorted((r for r in records if task_id is None or r.task_id == task_id),
                      key=lambda r: (r.started_at, r.attempt_id))

    def list_usage_observations(self, job_id: str, *, attempt_id: Optional[str] = None) -> list[UsageObservation]:
        records = [UsageObservation(**self.read_json(path)) for path in
                   (self._ledger_dir(job_id) / "observations").glob("*.json")]
        return sorted((r for r in records if attempt_id is None or r.attempt_id == attempt_id),
                      key=lambda r: (r.attempt_id, r.observation_id))

    def save_run(self, run: AgentRun) -> None:
        self._write_json_retrying_admission(
            self.job_dir(run.job_id) / "runs" / f"{run.id}.json", run
        )
        self.emit(run.job_id, "run.saved", {"run_id": run.id, "role": run.role})

    def _write_json_retrying_admission(self, path: Path, value: Any) -> None:
        deadline = time.monotonic() + 5
        while True:
            try:
                self.write_json(path, value)
                return
            except ProjectionWriteAdmissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _prepare_artifact_for_save(self, artifact: Artifact) -> Artifact:
        """Bound oversized payloads, validate schema, then stamp content hash."""
        from puppetmaster.artifact_bounds import prepare_artifact_for_persist
        from puppetmaster.execution_provenance import (
            stamp_execution_provenance_for_store,
        )

        # Central save seam: truthful execution provenance on typed peer
        # artifacts (all adapters). Additive / optional; never fabricates zeros.
        stamped = stamp_execution_provenance_for_store(artifact, self)
        try:
            from puppetmaster.negative_claims import stamp_failed_gate

            stamped = stamp_failed_gate(stamped, store=self)
        except Exception:
            pass
        prepared = prepare_artifact_for_persist(stamped, state_dir=self.root)
        prepared.validate()
        if prepared.sha256 is None:
            prepared = replace(prepared, sha256=self.artifact_hash(prepared))
        return prepared

    def save_artifact(self, artifact: Artifact) -> None:
        artifact = self._prepare_artifact_for_save(artifact)
        mkdir_private(self.job_dir(artifact.job_id) / "edges")
        path = self.job_dir(artifact.job_id) / "artifacts" / f"{artifact.id}.json"
        self.write_json(path, artifact)
        self.emit(
            artifact.job_id,
            "artifact.saved",
            {
                "artifact_id": artifact.id,
                "task_id": artifact.task_id,
                "type": str(artifact.type),
                "confidence": artifact.confidence,
                "sha256": artifact.sha256,
            },
        )
        self._materialize_produces_edge(artifact)
        self._mark_graph_edges_materialized(artifact.job_id)

    def save_artifacts(self, artifacts: Iterable[Artifact]) -> None:
        for artifact in artifacts:
            self.save_artifact(artifact)

    def promote_memory(self, memory: MemoryRecord) -> None:
        normalized = _normalize_memory_statement(memory.statement)
        for existing in self.list_memory():
            if existing.get("scope") != memory.scope:
                continue
            if _normalize_memory_statement(str(existing.get("statement") or "")) == normalized:
                return
        path = self.memory_dir / f"{memory.id}.json"
        self.write_json(path, memory)
        self._enforce_memory_cap(_MEMORY_CAP)

    def _delete_memory_record(self, memory_id: str) -> None:
        path = self.memory_dir / f"{memory_id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _enforce_memory_cap(self, cap: int) -> None:
        records = self.list_memory()
        if len(records) <= cap:
            return
        sorted_records = sorted(records, key=_memory_created_at_sort_key)
        for memory in sorted_records[: len(records) - cap]:
            self._delete_memory_record(str(memory.get("id") or ""))

    def prune_memory(
        self,
        *,
        scope: Optional[str] = None,
        older_than_days: Optional[int] = None,
    ) -> int:
        deleted = 0
        for memory in list(self.list_memory()):
            if scope is not None and memory.get("scope") != scope:
                continue
            if older_than_days is not None and not _memory_is_older_than_days(
                memory, older_than_days
            ):
                continue
            memory_id = memory.get("id")
            if not memory_id:
                continue
            self._delete_memory_record(str(memory_id))
            deleted += 1
        return deleted

    def promote_memories(self, records: Iterable[MemoryRecord]) -> None:
        for memory in records:
            self.promote_memory(memory)

    def write_summary(self, job_id: str, name: str, body: str) -> Path:
        path = self.job_dir(job_id) / "summaries" / name
        path.write_text(body, encoding="utf-8")
        self.emit(job_id, "summary.written", {"path": str(path)})
        return path

    def get_job(self, job_id: str) -> Job:
        return job_from_dict(self.read_json(self.job_dir(job_id) / "job.json"))

    def get_task_by_id(self, task_id: str) -> Task:
        for path in self.jobs_dir.glob(f"*/tasks/{task_id}.json"):
            return task_from_dict(self.read_json(path))
        raise FileNotFoundError(f"task not found: {task_id}")

    def list_jobs(self) -> list[Job]:
        self.init()
        jobs = []
        for path in sorted(self.jobs_dir.glob("*/job.json")):
            jobs.append(job_from_dict(self.read_json(path)))
        return jobs

    def recent_jobs(self, limit: int) -> list[Job]:
        """Newest ``limit`` jobs by created_at, newest first."""
        jobs = sorted(self.list_jobs(), key=lambda job: job.created_at or "", reverse=True)
        return jobs[:limit]

    def latest_job(self) -> Optional[Job]:
        jobs = self.list_jobs()
        if not jobs:
            return None
        return max(jobs, key=lambda job: job.created_at)

    def list_tasks(self, job_id: str) -> list[Task]:
        return [
            task_from_dict(self.read_json(path))
            for path in sorted((self.job_dir(job_id) / "tasks").glob("*.json"))
        ]

    def list_tasks_for_jobs(self, job_ids: Iterable[str]) -> list[Task]:
        """job_ids are de-duplicated; returns all matching rows; callers should not rely on per-job ordering."""
        tasks: list[Task] = []
        for job_id in dict.fromkeys(job_ids):
            tasks.extend(self.list_tasks(job_id))
        return tasks

    def list_artifacts(self, job_id: str) -> list[Artifact]:
        return [
            artifact_from_dict(self.read_json(path))
            for path in sorted((self.job_dir(job_id) / "artifacts").glob("*.json"))
        ]

    def list_artifacts_for_jobs(self, job_ids: Iterable[str]) -> list[Artifact]:
        """job_ids are de-duplicated; returns all matching rows; callers should not rely on per-job ordering."""
        artifacts: list[Artifact] = []
        for job_id in dict.fromkeys(job_ids):
            artifacts.extend(self.list_artifacts(job_id))
        return artifacts

    def get_artifact_job_id(self, artifact_id: str) -> Optional[str]:
        for job in self.list_jobs():
            if (self.job_dir(job.id) / "artifacts" / f"{artifact_id}.json").exists():
                return job.id
        return None

    def count_artifacts(self, job_id: str) -> int:
        """Cheap artifact count that avoids deserializing every payload."""
        artifacts_dir = self.job_dir(job_id) / "artifacts"
        if not artifacts_dir.exists():
            return 0
        return sum(1 for _ in artifacts_dir.glob("*.json"))

    def get_artifacts_by_ids(
        self, job_id: str, artifact_ids: Iterable[str]
    ) -> dict[str, Artifact]:
        """Load only the requested artifacts (by id) for a job.

        Lets pollers (e.g. the artifact feed) fetch just the artifacts a new
        batch of events references instead of snapshotting the whole job.
        """
        artifacts_dir = self.job_dir(job_id) / "artifacts"
        out: dict[str, Artifact] = {}
        for artifact_id in artifact_ids:
            if not artifact_id or artifact_id in out:
                continue
            path = artifacts_dir / f"{artifact_id}.json"
            if path.exists():
                out[artifact_id] = artifact_from_dict(self.read_json(path))
        return out

    def list_artifacts_by_type(
        self, artifact_type: str, job_ids: Optional[Iterable[str]] = None
    ) -> list[Artifact]:
        """Return every artifact of ``artifact_type``, optionally scoped to
        ``job_ids``.

        The file backend still walks each job; SQLite overrides this with a
        single indexed query. Used by the savings ledger so it doesn't have to
        deserialize every artifact of every job just to find routing records —
        and, when a time window is set, only scans the in-window jobs.
        """
        job_filter = set(job_ids) if job_ids is not None else None
        out: list[Artifact] = []
        for job in self.list_jobs():
            if job_filter is not None and job.id not in job_filter:
                continue
            out.extend(
                artifact
                for artifact in self.list_artifacts(job.id)
                if str(artifact.type) == artifact_type
            )
        return out

    @staticmethod
    def _compact_text_ref(value: Any) -> dict[str, Any]:
        text = str(value)
        return {
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    @classmethod
    def _compact_status_payload(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        compacted = {k: v for k, v in payload.items() if k not in _STATUS_PAYLOAD_BOOKKEEPING}
        prompt = compacted.pop("prompt", None)
        if prompt is not None:
            compacted["prompt_ref"] = cls._compact_text_ref(prompt)
        return compacted

    @classmethod
    def _compact_status_job(cls, job: dict[str, Any]) -> dict[str, Any]:
        compacted = dict(job)
        goal = compacted.pop("goal", None)
        if goal is not None:
            compacted["goal_ref"] = cls._compact_text_ref(goal)
        receipt = compacted.pop("cost_receipt", None)
        if isinstance(receipt, dict):
            compacted["cost_summary"] = cls._compact_cost_receipt(receipt, compacted.get("id"))
        return compacted

    @staticmethod
    def _compact_cost_receipt(receipt: dict[str, Any], job_id: Any) -> dict[str, Any]:
        actual = receipt.get("actual_cost") if isinstance(receipt.get("actual_cost"), dict) else {}
        tasks = [t for t in actual.get("tasks") or [] if isinstance(t, dict)]
        def total(key: str) -> Any:
            values = [t.get(key) for t in tasks if isinstance(t.get(key), (int, float))]
            return round(sum(values), 6) if values else None
        counterfactual = receipt.get("counterfactual") if isinstance(receipt.get("counterfactual"), dict) else {}
        return {
            "tokens_in": total("tokens_in"),
            "tokens_out": total("tokens_out"),
            "cache_read_tokens": total("cache_read_tokens"),
            "marginal_cost_usd": actual.get("total_marginal_cost_usd"),
            "api_equivalent_cost_usd": total("api_equivalent_cost_usd"),
            "cost_basis": actual.get("cost_basis") or receipt.get("cost_basis"),
            "avoided_usd": counterfactual.get("avoided_usd"),
            "full": f"puppetmaster cost {job_id}",
        }

    @classmethod
    def _compact_status_task(cls, task: dict[str, Any]) -> dict[str, Any]:
        compacted = dict(task)
        instruction = compacted.pop("instruction", None)
        if instruction is not None:
            compacted["instruction_ref"] = cls._compact_text_ref(instruction)
        compacted["payload"] = cls._compact_status_payload(compacted.get("payload"))
        return compacted

    def status_snapshot(self, job_id: str, *, compact: bool = False) -> dict[str, Any]:
        self.refresh_blocked_tasks(job_id)
        tasks = self.list_tasks(job_id)
        status_counts: dict[str, int] = {}
        for task in tasks:
            status_counts[str(task.status)] = status_counts.get(str(task.status), 0) + 1
        artifacts = self.list_artifacts(job_id)
        job = self.get_job(job_id)
        job_payload = to_jsonable(self.get_job(job_id))
        task_payloads = [to_jsonable(task) for task in tasks]
        if compact:
            job_payload = self._compact_status_job(job_payload)
            task_payloads = [
                self._compact_status_task(task_payload) for task_payload in task_payloads
            ]
        return {
            "job": job_payload,
            "tasks": task_payloads,
            "task_counts": status_counts,
            "artifact_count": len(artifacts),
            "stale_task_ids": [task.id for task in tasks if self.is_task_stale(task)],
            # A2+F2: a real terminal-quality signal so a "complete" job that did
            # nothing (no diff/commit, only verification, or refused outright) is
            # legible in status/completion instead of looking like success.
            "outcome": self._outcome_signals(artifacts),
            # Wave 4: DeLM-inspired frontier observability (compact, numbers-only).
            "frontier": self._frontier_signals(tasks, artifacts),
            "delivery": self._delivery_signals(job, tasks, artifacts),
            "progress": self._progress_signals(job_id, artifacts),
        }

    def _delivery_signals(self, job: Job, tasks: list[Task], artifacts: list[Any]) -> dict[str, Any]:
        from puppetmaster.delivery import delivery_verdict
        from puppetmaster.quality import assess_run_quality

        quality = assess_run_quality(artifacts)
        stale = [task.id for task in tasks if self.is_task_stale(task)]
        return delivery_verdict(
            job.status,
            quality=quality.get("quality"),
            stale_tasks=stale,
            incomplete_tasks=any(not task_is_satisfied(task.status) for task in tasks),
            required_artifacts=bool(artifacts),
        )

    def _progress_signals(self, job_id: str, artifacts: list[Any]) -> dict[str, Any]:
        substantive = [
            artifact.created_at
            for artifact in artifacts
            if str(getattr(artifact, "type", "")) in {"finding", "decision", "risk", "patch", "gist"}
        ]
        latest_substantive = max(substantive) if substantive else None
        latest_liveness = self.latest_liveness_at(job_id)
        from datetime import datetime, timezone

        def age(value: Optional[str]) -> Optional[float]:
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
            except (TypeError, ValueError):
                return None

        return {
            "last_substantive_artifact_at": latest_substantive,
            "last_liveness_at": latest_liveness,
            "last_substantive_artifact_age_seconds": age(latest_substantive),
            "last_liveness_age_seconds": age(latest_liveness),
        }

    def latest_liveness_at(self, job_id: str) -> Optional[str]:
        values = [
            event.get("at")
            for event in self.read_events(job_id)
            if event.get("event") in {"run.heartbeat", "task.lease_renewed"}
        ]
        return max((str(value) for value in values if value), default=None)

    @staticmethod
    def _frontier_signals(
        tasks: list[Task], artifacts: list[Any]
    ) -> dict[str, Any]:
        """Queue + admitted-gist frontier counts for status/dashboard."""
        from puppetmaster.metr_seams import is_coordination_protocol_payload
        from puppetmaster.models import ArtifactType

        artifacts = [
            artifact
            for artifact in artifacts
            if not is_coordination_protocol_payload(artifact)
        ]

        queued = 0
        running = 0
        blocked = 0
        enqueued_from_parent = 0
        for task in tasks:
            status = task.status
            if status == TaskStatus.QUEUED:
                queued += 1
            elif status == TaskStatus.RUNNING:
                running += 1
            elif status == TaskStatus.BLOCKED:
                blocked += 1
            if (task.payload or {}).get("enqueued_from_parent"):
                enqueued_from_parent += 1
        gist_total = 0
        gist_admitted = 0
        gist_pending = 0
        gist_rejected = 0
        for artifact in artifacts:
            if getattr(artifact, "type", None) != ArtifactType.GIST:
                continue
            gist_total += 1
            admission = str(
                (getattr(artifact, "payload", None) or {}).get("admission") or ""
            ).strip().lower()
            if admission == "admitted":
                gist_admitted += 1
            elif admission == "pending":
                gist_pending += 1
            elif admission == "rejected":
                gist_rejected += 1
        return {
            "queued": queued,
            "running": running,
            "blocked": blocked,
            "enqueued_from_parent": enqueued_from_parent,
            "gists": {
                "total": gist_total,
                "admitted": gist_admitted,
                "pending": gist_pending,
                "rejected": gist_rejected,
            },
        }

    @staticmethod
    def _outcome_signals(artifacts: list[Any]) -> dict[str, Any]:
        """Artifact-derived outcome signals (no git shell-out): quality verdict
        plus whether the run produced a diff and a verified commit."""
        from puppetmaster.quality import assess_run_quality
        from puppetmaster.models import ArtifactType

        verdict = assess_run_quality(artifacts)
        patch_artifact_emitted = any(
            getattr(a, "type", None) == ArtifactType.PATCH for a in artifacts
        )
        baseline_diff_present = any(
            bool((getattr(a, "payload", None) or {}).get("baseline_diff_present"))
            for a in artifacts
        )
        worker_diff_present = any(
            bool((getattr(a, "payload", None) or {}).get("worker_diff_present"))
            for a in artifacts
        )
        commit_present = any(
            getattr(a, "type", None) == ArtifactType.GATE
            and (getattr(a, "payload", None) or {}).get("kind") == "committed"
            and (getattr(a, "payload", None) or {}).get("passed") is True
            for a in artifacts
        )
        return {
            "quality": verdict["quality"],
            "trustworthy": verdict["trustworthy"],
            "reasons": verdict.get("reasons", []),
            "artifact_count": len(artifacts),
            "baseline_diff_present": baseline_diff_present,
            "worker_diff_present": worker_diff_present,
            "patch_artifact_emitted": patch_artifact_emitted,
            "commit_present": commit_present,
        }

    def has_incomplete_tasks(self, job_id: str) -> bool:
        return any(
            not task_is_satisfied(task.status) for task in self.list_tasks(job_id)
        )

    def list_memory(self) -> list[dict[str, Any]]:
        self.init()
        return [self.read_json(path) for path in sorted(self.memory_dir.glob("*.json"))]

    def retrieve_memory(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
        adapter: Optional[str] = None,
        role: Optional[str] = None,
        topic: Optional[str] = None,
        max_age_days: Optional[int] = None,
        min_overlap: float = 0.0,
    ) -> list[dict[str, Any]]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for memory in self.list_memory():
            if not _memory_within_max_age(memory, max_age_days):
                continue
            if not self._memory_matches_filters(memory, scope, adapter, role, topic):
                continue
            score, confidence, created_at_key, overlap = _memory_retrieval_score(memory, terms)
            if terms and min_overlap > 0 and overlap < min_overlap:
                continue
            scored.append((score, confidence, created_at_key, memory))
        from puppetmaster.mmr import finalize_memory_retrieval

        return finalize_memory_retrieval(scored, terms, limit)

    @staticmethod
    def _memory_matches_filters(
        memory: dict[str, Any],
        scope: Optional[str],
        adapter: Optional[str],
        role: Optional[str],
        topic: Optional[str],
    ) -> bool:
        filters = {
            "scope": scope,
            "adapter": adapter,
            "role": role,
            "topic": topic,
        }
        return all(value is None or memory.get(key) == value for key, value in filters.items())

    def _assert_safe_job_dir(self, job_id: str) -> Path:
        """Resolve and validate ``job_id``'s directory before any destructive
        delete, returning the safe path.

        Refuses to act unless the resolved directory is *strictly inside* this
        store's jobs tree. A blank, relative, or absolute ``job_id`` (``""``,
        ``..``, ``/``) would otherwise make ``delete_job`` rglob-unlink the whole
        jobs tree — or escape the state dir entirely into the user's active
        worktree. This is the guard that stops a ``gc --force`` from ever
        nuking the primary/active worktree (D1, P0 data-loss).
        """
        if not job_id or not isinstance(job_id, str) or job_id.strip() in {"", ".", ".."}:
            raise ValueError(f"refusing to delete job with unsafe id: {job_id!r}")
        jobs_root = self.jobs_dir.resolve()
        try:
            resolved = self.job_dir(job_id).resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"refusing to delete job with unresolvable path: {job_id!r}") from exc
        if resolved == jobs_root or jobs_root not in resolved.parents:
            raise ValueError(
                f"refusing to delete a path outside the jobs tree: {job_id!r} -> {resolved}"
            )
        return resolved

    def delete_job(self, job_id: str) -> None:
        job_dir = self._assert_safe_job_dir(job_id)
        from puppetmaster.projections import connection
        self.init()
        with connection(self, write=True) as c:
            c.execute("INSERT OR IGNORE INTO projection_pending VALUES(?)", (str(job_dir),))
        if job_dir.exists():
            for path in sorted(job_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            job_dir.rmdir()
        with connection(self, write=True) as c:
            c.execute("""INSERT INTO projection_changes(kind,job_id,id,status,sha256,stamp,deleted,task_count,artifact_count)
                SELECT kind,job_id,id,status,sha256,'known',1,task_count,artifact_count
                FROM projection_current WHERE job_id=?""", (job_id,))
            c.execute("DELETE FROM projection_current WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM historical_refs WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM completion_receipts WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM selected_economics_current WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM projection_pending WHERE path=?", (str(job_dir),))

    def acquire_lock(
        self,
        name: str,
        owner: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        self.init()
        path = self.locks_dir / f"{self._safe_key(name)}.lock"
        payload = json.dumps({"owner": owner, "at": time.time()}, sort_keys=True)
        while True:
            try:
                # Windows refuses an exclusive create with a sharing violation
                # while the previous owner's unlink of this file is pending.
                descriptor = _retry_on_windows_lock(
                    lambda: os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                return True
            except PermissionError:
                if not _WINDOWS:
                    raise
                return False  # still contended after the retry window: held
            except FileExistsError:
                if ttl_seconds is None:
                    return False
                stale = self._lock_is_stale(path, ttl_seconds)
                if stale is None:
                    continue  # released since our create failed: compete again
                if not stale:
                    return False
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except PermissionError:
                    return False  # Windows: an owner holds it open, so it is live

    def release_lock(self, name: str, owner: Optional[str] = None) -> None:
        path = self.locks_dir / f"{self._safe_key(name)}.lock"
        if not path.exists():
            return
        # When an owner is supplied, only release a lock we actually hold.
        # This prevents a stale/late caller from unlinking another worker's
        # lock and letting a second worker double-claim the same task.
        if owner is not None:
            held_by = self._lock_owner(path)
            if held_by and held_by != owner:
                return
        try:
            _retry_on_windows_lock(path.unlink)
        except FileNotFoundError:
            pass
        except PermissionError:
            # A competing same-key retry may have reclaimed the lock between
            # the owner read and unlink.  The durable job is already written;
            # leave the lock for that owner rather than raising into success.
            pass

    @staticmethod
    def _task_claim_snapshot(task: Task) -> tuple[Any, ...]:
        return (task.status, task.lease_owner, task.updated_at, task.attempts)

    def _save_task_if_matches(
        self,
        task_id: str,
        expected: tuple[Any, ...],
        updated: Task,
    ) -> bool:
        current = self.get_task_by_id(task_id)
        if self._task_claim_snapshot(current) != expected:
            return False
        self.save_task(updated)
        return True

    @staticmethod
    def _lock_owner(path: Path) -> str:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict):
            return str(payload.get("owner") or "")
        return raw

    @staticmethod
    def _empty_lock_is_stale(path: Path, ttl_seconds: int) -> Optional[bool]:
        """Age-gate a contentless lock file by its own mtime.

        ``acquire_lock`` creates the lock with ``O_EXCL`` and writes the owner
        payload in a second step, so a racing acquirer can momentarily read a
        zero-byte file. Treating that empty window as stale would let the racer
        delete a *live* lock and double-claim the task, so reclaim only when the
        empty file is itself older than the TTL (a genuinely orphaned lock).
        """
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        except OSError:
            return False
        return (time.time() - mtime) >= ttl_seconds

    @staticmethod
    def _lock_is_stale(path: Path, ttl_seconds: int) -> Optional[bool]:
        """True only for a lock proven older than the TTL; None if it is gone.

        A vanished lock was released, not orphaned: unlinking the path then
        would delete whichever owner re-acquired it in between. A lock modified
        well inside the TTL is fresh by its stat alone; not opening it matters
        on Windows, where a waiter's open handle blocks the owner's delete.
        """
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            return None
        except OSError:
            modified = None
        if modified is not None and time.time() - modified < ttl_seconds - _LOCK_STAT_SLACK_SECONDS:
            return False
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError:
            return SwarmStore._empty_lock_is_stale(path, ttl_seconds)
        if not raw:
            return SwarmStore._empty_lock_is_stale(path, ttl_seconds)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        created_at = payload.get("at")
        if not isinstance(created_at, (int, float)):
            return False
        return (time.time() - float(created_at)) >= ttl_seconds

    def emit(self, job_id: str, event: str, payload: dict[str, Any]) -> None:
        self.init()
        stream = self.stream_dir / f"{job_id}.jsonl"
        record = {"at": now_iso(), "event": event, "payload": payload}
        with stream.open("a", encoding="utf-8") as handle:
            # Separate a retry from any torn completion append. Readers already
            # skip blank/malformed lines and retain their line-based cursors.
            prefix = "\n" if event == "worker.completed_task" else ""
            handle.write(prefix + json.dumps(record, sort_keys=True) + "\n")

    def read_events(self, job_id: str) -> list[dict[str, Any]]:
        return self.read_events_since(job_id, since=0)

    def read_events_since(
        self, job_id: str, since: int = 0
    ) -> list[dict[str, Any]]:
        """Return events for `job_id` whose monotonic id exceeds `since`.

        Each event dict gains a synthetic ``id`` (1-indexed) so callers can
        use the same cursor protocol across backends.
        """
        stream = self.stream_dir / f"{job_id}.jsonl"
        if not stream.exists():
            return []
        results: list[dict[str, Any]] = []
        with stream.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if index <= since:
                    continue
                cleaned = line.strip().strip("\x00")
                if not cleaned:
                    continue
                try:
                    record = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Torn/partial line from a concurrent append. POSIX
                    # O_APPEND writes are atomic, but Windows appends are
                    # not, so two workers writing at once can interleave a
                    # malformed line. Skip it rather than crash the reader;
                    # the well-formed events around it are still returned.
                    continue
                record["id"] = index
                results.append(record)
        return results

    def event_cursor(self, job_id: str) -> int:
        """Return the highest event id currently stored for `job_id`.

        Uses a size-keyed cache so a hot poll loop (wait_for_events) doesn't
        re-scan the entire append-only stream every iteration: if the file size
        is unchanged since the last count, the line count is unchanged too."""
        stream = self.stream_dir / f"{job_id}.jsonl"
        try:
            size = stream.stat().st_size
        except (FileNotFoundError, NotADirectoryError):
            self._event_cursor_cache.pop(job_id, None)
            return 0
        cached = self._event_cursor_cache.get(job_id)
        if cached is not None and cached[0] == size:
            return cached[1]
        with stream.open("rb") as handle:
            count = sum(1 for _ in handle)
        self._event_cursor_cache[job_id] = (size, count)
        return count

    def wait_for_events(
        self,
        job_id: str,
        since: int = 0,
        timeout_seconds: float = 10.0,
        poll_interval: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Block up to ``timeout_seconds`` waiting for events newer than ``since``.

        Returns the new events (potentially empty if the deadline is reached).
        Uses a cheap ``event_cursor`` check between polls so the underlying
        storage isn't re-read until something actually changed.
        """
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        cursor = self.event_cursor(job_id)
        if cursor > since:
            return self.read_events_since(job_id, since=since)
        while time.monotonic() < deadline:
            time.sleep(max(0.005, poll_interval))
            cursor = self.event_cursor(job_id)
            if cursor > since:
                return self.read_events_since(job_id, since=since)
        return []

    @staticmethod
    def is_task_stale(task: Task) -> bool:
        if task.status != TaskStatus.RUNNING or not task.lease_expires_at:
            return False
        return parse_iso(task.lease_expires_at) <= datetime.now(timezone.utc)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def write_json(self, path: Path, value: Any) -> None:
        from puppetmaster.projections import connection, file_kind, project_file
        projected = self.backend_name == "file" and file_kind(path) is not None
        if projected:
            self.init()
            marker = str(path) + ":" + new_id("write")
            try:
                with connection(self, write=True) as c:
                    c.execute("INSERT INTO projection_pending VALUES(?)", (marker,))
            except sqlite3.OperationalError as exc:
                from puppetmaster.readonly import _locked

                if not _locked(exc):
                    raise
                raise ProjectionWriteAdmissionError(str(exc)) from exc
        if projected and file_kind(path) == 'job':
            from puppetmaster.selected_economics import check_receipt_replacement
            from puppetmaster.contracts import ContractConflict
            try:
                with connection(self, write=True) as c:
                    if path.exists():
                        check_receipt_replacement(self.read_json(path), to_jsonable(value))
                    self._write_json_file(path, value)
                    project_file(c, path, self.read_json(path))
                    c.execute("DELETE FROM projection_pending WHERE path=?", (marker,))
            except ContractConflict:
                # Rejection preceded rename; there is no pending source write.
                with connection(self, write=True) as c:
                    c.execute("DELETE FROM projection_pending WHERE path=?", (marker,))
                raise
            return
        self._write_json_file(path, value)
        if projected:
            with connection(self, write=True) as c:
                # Read back under the index writer lock: a competing rename may
                # have won, so indexing our caller's value would be stale.
                project_file(c, path, self.read_json(path))
                c.execute("DELETE FROM projection_pending WHERE path=?", (marker,))

    @staticmethod
    def _write_json_file(path: Path, value: Any) -> None:
        mkdir_private(path.parent)
        # The temp name must be unique per concurrent writer, not just per
        # process: two threads writing the same file share a pid, so a
        # pid-only suffix collides and the first os.replace moves the shared
        # temp out from under the second writer -> FileNotFoundError. Add the
        # thread id plus a monotonic counter so every writer gets its own temp.
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{next(SwarmStore._temp_counter)}.tmp"
        )
        temp_path.write_text(
            json.dumps(
                to_jsonable(_prepare_for_persistence(value)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        chmod_private_file(temp_path)
        # On Windows os.replace raises PermissionError when a concurrent reader
        # holds the destination open; on POSIX the rename is atomic and this
        # retry never triggers. Keeps cross-process task writes from flaking.
        _retry_on_windows_lock(lambda: os.replace(temp_path, path))
        chmod_private_file(path)

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        # Mirror of write_json: a read that lands mid-replace can hit a transient
        # PermissionError on Windows. Retry briefly instead of crashing the run.
        text = _retry_on_windows_lock(lambda: path.read_text(encoding="utf-8"))
        return json.loads(text)

    @staticmethod
    def artifact_hash(artifact: Artifact) -> str:
        value = to_jsonable(replace(artifact, sha256=None))
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _safe_key(value: str) -> str:
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _coerce_confidence(value: Any) -> float:
    """Best-effort float for a persisted confidence value.

    Malformed JSON (a string, None, or garbage written by an older/buggy
    producer) must not crash memory retrieval — treat anything uncoercible
    as 0.0 so the record sorts last instead of raising.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def group_by_type(artifacts: Iterable[Artifact]) -> dict[str, list[Artifact]]:
    grouped: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(str(artifact.type), []).append(artifact)
    return grouped
