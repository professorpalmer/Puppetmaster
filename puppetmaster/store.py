from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from puppetmaster.models import (
    AgentRun,
    Artifact,
    GraphEdge,
    GraphEdgeType,
    GraphNodeKind,
    Job,
    JobStatus,
    MemoryRecord,
    Task,
    TaskStatus,
    artifact_from_dict,
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
from puppetmaster.redaction import redact_payload_for_storage
from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.state import ensure_state_dir, resolve_state_dir

_WINDOWS_LOCK_RETRIES = 10
_WINDOWS_LOCK_BACKOFF_SECONDS = 0.02
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


class SwarmStore:
    """File-backed coordination store with Redis-like key spaces."""

    backend_name = "file"
    max_task_attempts = 3
    # Monotonic, process-wide source of unique temp-file suffixes for atomic
    # writes. itertools.count() is thread-safe for next() under CPython's GIL.
    _temp_counter = itertools.count()

    def __init__(self, root: Optional[Union[Path, str]] = None) -> None:
        self.root = resolve_state_dir(root)
        ensure_state_dir(self.root)
        self.jobs_dir = self.root / "jobs"
        self.memory_dir = self.root / "memory"
        self.stream_dir = self.root / "streams"
        self.locks_dir = self.root / "locks"
        # job_id -> (last seen file size, line count). Streams are append-only
        # JSONL, so an unchanged size means an unchanged line count; this lets
        # event_cursor skip re-counting the whole file on every poll.
        self._event_cursor_cache: dict[str, tuple[int, int]] = {}

    def init(self) -> None:
        for directory in [
            self.root,
            self.jobs_dir,
            self.memory_dir,
            self.stream_dir,
            self.locks_dir,
        ]:
            mkdir_private(directory)

    def create_job(self, goal: str, *, label: Optional[str] = None) -> Job:
        self.init()
        job = Job(goal=goal, label=label)
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
        self.emit(job.id, "job.created", payload)
        return job

    def update_job_status(self, job_id: str, status: JobStatus) -> Job:
        job = self.get_job(job_id)
        updated = self._job_with_status(job, status)
        self.write_json(self.job_dir(job_id) / "job.json", updated)
        self.emit(job_id, "job.status", {"status": str(status)})
        return updated

    @staticmethod
    def _job_with_status(job: Job, status: JobStatus) -> Job:
        return Job(
            id=job.id,
            goal=job.goal,
            label=job.label,
            status=status,
            created_at=job.created_at,
            completed_at=now_iso()
            if status
            in {
                JobStatus.COMPLETE,
                JobStatus.FAILED,
                JobStatus.STALLED,
                JobStatus.CANCELLED,
            }
            else job.completed_at,
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
        self.write_json(self.job_dir(task.job_id) / "tasks" / f"{task.id}.json", task)
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

    def update_task_status(
        self,
        task: Task,
        status: TaskStatus,
        worker_id: Optional[str] = None,
        lease_id: Optional[str] = None,
    ) -> Task:
        stored = self.get_task_by_id(task.id)
        # The caller carries the lease token granted at claim time; default to
        # the claimed task's own ``lease_id`` so existing call sites fence
        # correctly without having to thread the token through explicitly.
        expected_lease = lease_id if lease_id is not None else task.lease_id
        updated = self._build_status_update(stored, status)
        terminal = status in {TaskStatus.COMPLETE, TaskStatus.FAILED}
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
        terminal = status in {TaskStatus.COMPLETE, TaskStatus.FAILED}
        return replace(
            stored,
            status=status,
            lease_owner=None if terminal else stored.lease_owner,
            lease_expires_at=None if terminal else stored.lease_expires_at,
            lease_id=None if terminal else stored.lease_id,
            updated_at=now_iso(),
            completed_at=now_iso() if status == TaskStatus.COMPLETE else stored.completed_at,
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
        lock_ttl = max(lease_seconds * 3, lease_seconds + 1)
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
        if self._claim_precheck(task, task_map=task_map):
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
    ) -> bool:
        """Shared claim decision logic. Returns True when the claim attempt must abort."""
        if not self.dependencies_complete(task, task_map=task_map):
            blocked = replace(task, status=TaskStatus.BLOCKED, updated_at=now_iso())
            self.save_task(blocked)
            return True
        if task.status == TaskStatus.COMPLETE:
            return True
        if task.attempts >= self.max_task_attempts:
            failed = replace(
                task,
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
        return False

    @staticmethod
    def _build_claimed_task(task: Task, worker_id: str, lease_seconds: int) -> Task:
        return replace(
            task,
            status=TaskStatus.RUNNING,
            attempts=task.attempts + 1,
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
        if not self._save_task_if_matches(task_id, claim_snapshot, claimed):
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
    ) -> Optional[Task]:
        self.refresh_blocked_tasks(job_id)
        # Load the job's tasks once and resolve dependency status from the
        # in-memory map instead of re-fetching each dependency by id (which is
        # a per-edge file glob / SQLite SELECT on every claim sweep).
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        for task in tasks:
            if task.status != TaskStatus.QUEUED:
                continue
            if not self.dependencies_complete(task, task_map=task_map):
                self.save_task(replace(task, status=TaskStatus.BLOCKED, updated_at=now_iso()))
                continue
            if role is not None and task.role != role:
                continue
            # claim_task acquires the per-task lock internally so direct callers
            # are race-safe without requiring claim_next_task's sweep wrapper.
            return self.claim_task(
                task.id, worker_id, lease_seconds=lease_seconds, task_map=task_map
            )
        return None

    def recover_stale_tasks(self, job_id: str) -> list[Task]:
        recovered: list[Task] = []
        for task in self.list_tasks(job_id):
            if not self.is_task_stale(task):
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
        return replace(
            task,
            status=TaskStatus.QUEUED,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=now_iso(),
        )

    def _atomic_recover_stale(self, task: Task, queued: Task) -> bool:
        self.save_task(queued)
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
            if dependency.status != TaskStatus.COMPLETE:
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

    def reset_subgraph(
        self,
        job_id: str,
        task_ids: Iterable[str],
        *,
        include_descendants: bool = True,
    ) -> list[Task]:
        """Idempotent targeted rerun reset for selected (downstream) tasks.

        Clears lease/completion state and ``attempts`` for the selected set
        (and optionally their consumer closure), then re-derives QUEUED/BLOCKED
        from ``depends_on``. Completed upstream tasks, artifacts, and edges are
        retained.

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
            return []
        tasks = self.list_tasks(job_id)
        task_map = {task.id: task for task in tasks}
        active = [
            task.id
            for task in tasks
            if task.id in selected and self.has_active_lease(task)
        ]
        if active:
            raise ActiveTaskLeaseError(active)
        reset: list[Task] = []
        for task in tasks:
            if task.id not in selected:
                continue
            cleared = replace(
                task,
                status=TaskStatus.BLOCKED,
                attempts=0,
                lease_owner=None,
                lease_expires_at=None,
                lease_id=None,
                completed_at=None,
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
        self.emit(
            job_id,
            "subgraph.reset",
            {"task_ids": sorted(selected), "reset_count": len(finalized)},
        )
        return finalized

    def heartbeat_run(self, run: AgentRun) -> AgentRun:
        updated = replace(run, heartbeat_at=now_iso())
        self.save_run(updated)
        self.emit(
            run.job_id,
            "run.heartbeat",
            {"run_id": run.id, "worker_id": run.worker_id, "task_id": run.task_id},
        )
        return updated

    def save_run(self, run: AgentRun) -> None:
        self.write_json(self.job_dir(run.job_id) / "runs" / f"{run.id}.json", run)
        self.emit(run.job_id, "run.saved", {"run_id": run.id, "role": run.role})

    def _prepare_artifact_for_save(self, artifact: Artifact) -> Artifact:
        """Bound oversized payloads, validate schema, then stamp content hash."""
        from puppetmaster.artifact_bounds import prepare_artifact_for_persist

        prepared = prepare_artifact_for_persist(artifact, state_dir=self.root)
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
        compacted = dict(payload)
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
        return compacted

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
            # Legacy alias for patch_artifact_emitted; older consumers key on it.
            "diff_present": patch_artifact_emitted,
            "baseline_diff_present": baseline_diff_present,
            "worker_diff_present": worker_diff_present,
            "patch_artifact_emitted": patch_artifact_emitted,
            "commit_present": commit_present,
        }

    def has_incomplete_tasks(self, job_id: str) -> bool:
        return any(task.status != TaskStatus.COMPLETE for task in self.list_tasks(job_id))

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
        if job_dir.exists():
            for path in sorted(job_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            job_dir.rmdir()

    def acquire_lock(
        self,
        name: str,
        owner: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        self.init()
        path = self.locks_dir / f"{self._safe_key(name)}.lock"
        payload = json.dumps({"owner": owner, "at": time.time()}, sort_keys=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            return True
        except FileExistsError:
            if ttl_seconds is not None and self._lock_is_stale(path, ttl_seconds):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                return self.acquire_lock(name, owner, ttl_seconds=ttl_seconds)
            return False

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
            path.unlink()
        except FileNotFoundError:
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
    def _empty_lock_is_stale(path: Path, ttl_seconds: int) -> bool:
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
            return True
        except OSError:
            return True
        return (time.time() - mtime) >= ttl_seconds

    @staticmethod
    def _lock_is_stale(path: Path, ttl_seconds: int) -> bool:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return True
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
            handle.write(json.dumps(record, sort_keys=True) + "\n")

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

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
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
