from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from puppetmaster.models import (
    AgentRun,
    Artifact,
    Job,
    JobStatus,
    MemoryRecord,
    Task,
    TaskStatus,
    artifact_from_dict,
    job_from_dict,
    now_iso,
    parse_iso,
    seconds_from_now,
    task_from_dict,
    to_jsonable,
)
from puppetmaster.state import resolve_state_dir


class SwarmStore:
    """File-backed coordination store with Redis-like key spaces."""

    backend_name = "file"
    max_task_attempts = 3

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

    def init(self) -> None:
        for directory in [
            self.root,
            self.jobs_dir,
            self.memory_dir,
            self.stream_dir,
            self.locks_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def create_job(self, goal: str) -> Job:
        self.init()
        job = Job(goal=goal)
        job_dir = self.job_dir(job.id)
        for directory in [
            job_dir,
            job_dir / "tasks",
            job_dir / "runs",
            job_dir / "artifacts",
            job_dir / "summaries",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        self.write_json(job_dir / "job.json", job)
        self.emit(job.id, "job.created", {"goal": goal})
        return job

    def update_job_status(self, job_id: str, status: JobStatus) -> Job:
        job = self.get_job(job_id)
        updated = Job(
            id=job.id,
            goal=job.goal,
            status=status,
            created_at=job.created_at,
            completed_at=now_iso()
            if status in {JobStatus.COMPLETE, JobStatus.FAILED}
            else job.completed_at,
        )
        self.write_json(self.job_dir(job_id) / "job.json", updated)
        self.emit(job_id, "job.status", {"status": str(status)})
        return updated

    def save_task(self, task: Task) -> None:
        self.write_json(self.job_dir(task.job_id) / "tasks" / f"{task.id}.json", task)
        self.emit(
            task.job_id,
            "task.saved",
            {
                "task_id": task.id,
                "role": task.role,
                "status": str(task.status),
                "adapter": task.adapter,
            },
        )

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            self.save_task(task)

    def update_task_status(self, task: Task, status: TaskStatus) -> Task:
        terminal = status in {TaskStatus.COMPLETE, TaskStatus.FAILED}
        updated = replace(
            task,
            status=status,
            lease_owner=None if terminal else task.lease_owner,
            lease_expires_at=None if terminal else task.lease_expires_at,
            updated_at=now_iso(),
            completed_at=now_iso() if status == TaskStatus.COMPLETE else task.completed_at,
        )
        self.save_task(updated)
        return updated

    def claim_task(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[Task]:
        task = self.get_task_by_id(task_id)
        if not self.dependencies_complete(task):
            blocked = replace(task, status=TaskStatus.BLOCKED, updated_at=now_iso())
            self.save_task(blocked)
            return None
        if task.status == TaskStatus.COMPLETE:
            return None
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
            return None
        if task.status == TaskStatus.RUNNING and not self.is_task_stale(task):
            return None

        claimed = replace(
            task,
            status=TaskStatus.RUNNING,
            attempts=task.attempts + 1,
            lease_owner=worker_id,
            lease_expires_at=seconds_from_now(lease_seconds),
            updated_at=now_iso(),
        )
        self.save_task(claimed)
        self.emit(
            task.job_id,
            "task.claimed",
            {
                "task_id": task.id,
                "worker_id": worker_id,
                "lease_expires_at": claimed.lease_expires_at,
                "attempts": claimed.attempts,
            },
        )
        return claimed

    def renew_task_lease(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[Task]:
        task = self.get_task_by_id(task_id)
        if task.status != TaskStatus.RUNNING or task.lease_owner != worker_id:
            return None
        renewed = replace(
            task,
            lease_expires_at=seconds_from_now(lease_seconds),
            updated_at=now_iso(),
        )
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
            lock_name = f"task:{task.id}"
            if not self.acquire_lock(lock_name, worker_id):
                continue
            try:
                return self.claim_task(task.id, worker_id, lease_seconds=lease_seconds)
            finally:
                self.release_lock(lock_name, owner=worker_id)
        return None

    def recover_stale_tasks(self, job_id: str) -> list[Task]:
        recovered: list[Task] = []
        for task in self.list_tasks(job_id):
            if not self.is_task_stale(task):
                continue
            queued = replace(
                task,
                status=TaskStatus.QUEUED,
                lease_owner=None,
                lease_expires_at=None,
                updated_at=now_iso(),
            )
            self.save_task(queued)
            self.emit(
                job_id,
                "task.recovered",
                {"task_id": task.id, "previous_owner": task.lease_owner},
            )
            recovered.append(queued)
        return recovered

    def refresh_blocked_tasks(self, job_id: str) -> list[Task]:
        ready: list[Task] = []
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

    def save_artifact(self, artifact: Artifact) -> None:
        artifact.validate()
        if artifact.sha256 is None:
            artifact = replace(artifact, sha256=self.artifact_hash(artifact))
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

    def save_artifacts(self, artifacts: Iterable[Artifact]) -> None:
        for artifact in artifacts:
            self.save_artifact(artifact)

    def promote_memory(self, memory: MemoryRecord) -> None:
        path = self.memory_dir / f"{memory.id}.json"
        self.write_json(path, memory)

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
        tasks: list[Task] = []
        for job_id in job_ids:
            tasks.extend(self.list_tasks(job_id))
        return tasks

    def list_artifacts(self, job_id: str) -> list[Artifact]:
        return [
            artifact_from_dict(self.read_json(path))
            for path in sorted((self.job_dir(job_id) / "artifacts").glob("*.json"))
        ]

    def list_artifacts_for_jobs(self, job_ids: Iterable[str]) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for job_id in job_ids:
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

    def status_snapshot(self, job_id: str) -> dict[str, Any]:
        self.refresh_blocked_tasks(job_id)
        tasks = self.list_tasks(job_id)
        status_counts: dict[str, int] = {}
        for task in tasks:
            status_counts[str(task.status)] = status_counts.get(str(task.status), 0) + 1
        return {
            "job": to_jsonable(self.get_job(job_id)),
            "tasks": [to_jsonable(task) for task in tasks],
            "task_counts": status_counts,
            "artifact_count": self.count_artifacts(job_id),
            "stale_task_ids": [task.id for task in tasks if self.is_task_stale(task)],
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
    ) -> list[dict[str, Any]]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for memory in self.list_memory():
            if not self._memory_matches_filters(memory, scope, adapter, role, topic):
                continue
            haystack = " ".join(
                str(memory.get(key, ""))
                for key in ["scope", "statement", "evidence", "adapter", "role", "topic"]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            confidence = _coerce_confidence(memory.get("confidence"))
            scored.append((score, confidence, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for score, _, memory in scored[:limit] if score > 0 or not terms]

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

    def delete_job(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        if job_dir.exists():
            for path in sorted(job_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            job_dir.rmdir()

    def acquire_lock(self, name: str, owner: str) -> bool:
        self.init()
        path = self.locks_dir / f"{self._safe_key(name)}.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(owner)
            return True
        except FileExistsError:
            return False

    def release_lock(self, name: str, owner: Optional[str] = None) -> None:
        path = self.locks_dir / f"{self._safe_key(name)}.lock"
        if not path.exists():
            return
        # When an owner is supplied, only release a lock we actually hold.
        # This prevents a stale/late caller from unlinking another worker's
        # lock and letting a second worker double-claim the same task.
        if owner is not None:
            try:
                held_by = path.read_text(encoding="utf-8").strip()
            except OSError:
                held_by = ""
            if held_by and held_by != owner:
                return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

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
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

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

