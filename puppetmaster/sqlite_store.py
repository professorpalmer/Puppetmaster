from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

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
    seconds_from_now,
    task_from_dict,
    to_jsonable,
)
from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.store import SwarmStore, _coerce_confidence

_SQLITE_IN_CHUNK = 900


def _chunked(values: Iterable[str], size: int = _SQLITE_IN_CHUNK) -> list[list[str]]:
    unique = list(dict.fromkeys(values))
    return [unique[index : index + size] for index in range(0, len(unique), size)]


class SQLiteSwarmStore(SwarmStore):
    """SQLite-backed coordination store for multi-process worker coordination."""

    backend_name = "sqlite"
    schema_version = 1

    def __init__(self, root: Optional[Union[Path, str]] = None) -> None:
        super().__init__(root)
        self.db_path = self.root / "state.sqlite3"
        # Schema/PRAGMA/DDL setup is idempotent but not free: re-running it from
        # every public method (each opens a connection and replays the whole
        # script) is pure overhead once the DB exists. Run it once per store
        # instance and short-circuit afterward.
        self._initialized = False

    def init(self) -> None:
        if self._initialized:
            return
        for directory in [
            self.root,
            self.jobs_dir,
            self.memory_dir,
            self.stream_dir,
            self.locks_dir,
        ]:
            mkdir_private(directory)
        with self._session() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA busy_timeout = 5000;
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  status TEXT NOT NULL,
                  data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_job_status
                  ON tasks(job_id, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_job_id
                  ON tasks(job_id, id);
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_job
                  ON artifacts(job_id, id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type
                  ON artifacts(type, job_id);
                CREATE TABLE IF NOT EXISTS memory (
                  id TEXT PRIMARY KEY,
                  data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  at TEXT NOT NULL,
                  event TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_job
                  ON events(job_id, id);
                CREATE TABLE IF NOT EXISTS metadata (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                """
            )
            # Record the schema version only when absent. Blindly overwriting it
            # on every init would mask a database written by a newer (or
            # incompatible) Puppetmaster: the metadata would silently claim our
            # version even though the on-disk schema is something else. Leaving
            # an existing value intact lets schema_status() surface the mismatch.
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (str(self.schema_version),),
            )
        chmod_private_file(self.db_path)
        self._initialized = True

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        chmod_private_file(self.db_path)
        return connection

    @contextmanager
    def _session(self) -> "Iterator[sqlite3.Connection]":
        """Open a connection, run a transaction, and ALWAYS close it.

        ``with sqlite3.connect(...) as conn`` is a *transaction* context manager —
        it commits/rolls back but leaves the connection (and its OS file handle)
        open. On POSIX a lingering handle is harmless, but Windows holds a
        mandatory lock on the open database file, so a later unlink / temp-dir
        cleanup fails with ``WinError 32``. Closing the handle here keeps the
        store correct and leak-free on every platform.
        """
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_job(self, goal: str) -> Job:
        self.init()
        job = Job(goal=goal)
        self._ensure_job_dirs(job.id)
        with self._session() as connection:
            connection.execute(
                "INSERT INTO jobs(id, data) VALUES(?, ?)",
                (job.id, self._dumps(job)),
            )
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
            if status in {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.STALLED}
            else job.completed_at,
        )
        payload = {"status": str(status)}
        with self._session() as connection:
            connection.execute(
                "UPDATE jobs SET data = ? WHERE id = ?",
                (self._dumps(updated), job_id),
            )
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (job_id, now_iso(), "job.status", json.dumps(payload, sort_keys=True)),
            )
        return updated

    def save_task(self, task: Task) -> None:
        self.init()
        # Persist the task row and its task.saved event in a single
        # transaction. Splitting them across two connections left a window
        # where a crash after the task write but before the event write would
        # produce state with no corresponding event (a torn write that breaks
        # event-cursor consumers).
        payload = {
            "task_id": task.id,
            "role": task.role,
            "status": str(task.status),
            "adapter": task.adapter,
        }
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO tasks(id, job_id, role, status, data)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  job_id = excluded.job_id,
                  role = excluded.role,
                  status = excluded.status,
                  data = excluded.data
                """,
                (task.id, task.job_id, task.role, str(task.status), self._dumps(task)),
            )
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (task.job_id, now_iso(), "task.saved", json.dumps(payload, sort_keys=True)),
            )

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        task_list = list(tasks)
        if not task_list:
            return
        self.init()
        rows: list[tuple[Any, ...]] = []
        event_rows: list[tuple[Any, ...]] = []
        for task in task_list:
            payload = {
                "task_id": task.id,
                "role": task.role,
                "status": str(task.status),
                "adapter": task.adapter,
            }
            rows.append(
                (task.id, task.job_id, task.role, str(task.status), self._dumps(task))
            )
            event_rows.append(
                (task.job_id, now_iso(), "task.saved", json.dumps(payload, sort_keys=True))
            )
        with self._session() as connection:
            connection.executemany(
                """
                INSERT INTO tasks(id, job_id, role, status, data)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  job_id = excluded.job_id,
                  role = excluded.role,
                  status = excluded.status,
                  data = excluded.data
                """,
                rows,
            )
            connection.executemany(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                event_rows,
            )

    def update_task_status(
        self,
        task: Task,
        status: TaskStatus,
        worker_id: Optional[str] = None,
    ) -> Task:
        terminal = status in {TaskStatus.COMPLETE, TaskStatus.FAILED}
        stored = self.get_task_by_id(task.id)
        if terminal and worker_id is not None and stored.lease_owner != worker_id:
            return stored
        updated = replace(
            stored,
            status=status,
            lease_owner=None if terminal else stored.lease_owner,
            lease_expires_at=None if terminal else stored.lease_expires_at,
            updated_at=now_iso(),
            completed_at=now_iso() if status == TaskStatus.COMPLETE else stored.completed_at,
        )
        payload = {
            "task_id": updated.id,
            "role": updated.role,
            "status": str(updated.status),
            "adapter": updated.adapter,
        }
        with self._session() as connection:
            if terminal and worker_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE tasks SET status = ?, data = ?
                    WHERE id = ? AND json_extract(data, '$.lease_owner') = ?
                    """,
                    (str(status), self._dumps(updated), task.id, worker_id),
                )
                if cursor.rowcount != 1:
                    return self.get_task_by_id(task.id)
            else:
                connection.execute(
                    """
                    INSERT INTO tasks(id, job_id, role, status, data)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      job_id = excluded.job_id,
                      role = excluded.role,
                      status = excluded.status,
                      data = excluded.data
                    """,
                    (
                        updated.id,
                        updated.job_id,
                        updated.role,
                        str(updated.status),
                        self._dumps(updated),
                    ),
                )
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (updated.job_id, now_iso(), "task.saved", json.dumps(payload, sort_keys=True)),
            )
        return updated

    def claim_task(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[Task]:
        self.init()
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

        now = now_iso()
        claimed = replace(
            task,
            status=TaskStatus.RUNNING,
            attempts=task.attempts + 1,
            lease_owner=worker_id,
            lease_expires_at=seconds_from_now(lease_seconds),
            updated_at=now,
        )
        claim_payload = {
            "task_id": task.id,
            "worker_id": worker_id,
            "lease_expires_at": claimed.lease_expires_at,
            "attempts": claimed.attempts,
        }
        with self._session() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET status = ?, data = ?
                WHERE id = ? AND (
                  status = ?
                  OR (status = ? AND json_extract(data, '$.lease_expires_at') <= ?)
                )
                """,
                (
                    str(TaskStatus.RUNNING),
                    self._dumps(claimed),
                    task_id,
                    str(TaskStatus.QUEUED),
                    str(TaskStatus.RUNNING),
                    now,
                ),
            )
            if cursor.rowcount != 1:
                return None
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (
                    task.job_id,
                    now,
                    "task.claimed",
                    json.dumps(claim_payload, sort_keys=True),
                ),
            )
        return claimed

    def recover_stale_tasks(self, job_id: str) -> list[Task]:
        self.init()
        now = now_iso()
        recovered: list[Task] = []
        for task in self.list_tasks(job_id):
            if not self.is_task_stale(task):
                continue
            queued = replace(
                task,
                status=TaskStatus.QUEUED,
                lease_owner=None,
                lease_expires_at=None,
                updated_at=now,
            )
            recover_payload = {
                "task_id": task.id,
                "previous_owner": task.lease_owner,
            }
            with self._session() as connection:
                cursor = connection.execute(
                    """
                    UPDATE tasks SET status = ?, data = ?
                    WHERE id = ? AND status = ? AND json_extract(data, '$.lease_expires_at') <= ?
                    """,
                    (
                        str(TaskStatus.QUEUED),
                        self._dumps(queued),
                        task.id,
                        str(TaskStatus.RUNNING),
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                connection.execute(
                    "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                    (
                        job_id,
                        now,
                        "task.recovered",
                        json.dumps(recover_payload, sort_keys=True),
                    ),
                )
            self.release_lock(f"task:{task.id}")
            recovered.append(queued)
        return recovered

    def save_run(self, run: AgentRun) -> None:
        self.init()
        payload = {"run_id": run.id, "role": run.role}
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO runs(id, job_id, task_id, data)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  job_id = excluded.job_id,
                  task_id = excluded.task_id,
                  data = excluded.data
                """,
                (run.id, run.job_id, run.task_id, self._dumps(run)),
            )
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (run.job_id, now_iso(), "run.saved", json.dumps(payload, sort_keys=True)),
            )

    def save_artifact(self, artifact: Artifact) -> None:
        artifact.validate()
        if artifact.sha256 is None:
            from dataclasses import replace

            artifact = replace(artifact, sha256=self.artifact_hash(artifact))
        self.init()
        event_payload = {
            "artifact_id": artifact.id,
            "task_id": artifact.task_id,
            "type": str(artifact.type),
            "confidence": artifact.confidence,
            "sha256": artifact.sha256,
        }
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(id, job_id, task_id, type, data)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  job_id = excluded.job_id,
                  task_id = excluded.task_id,
                  type = excluded.type,
                  data = excluded.data
                """,
                (
                    artifact.id,
                    artifact.job_id,
                    artifact.task_id,
                    str(artifact.type),
                    self._dumps(artifact),
                ),
            )
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (
                    artifact.job_id,
                    now_iso(),
                    "artifact.saved",
                    json.dumps(event_payload, sort_keys=True),
                ),
            )

    def save_artifacts(self, artifacts: Iterable[Artifact]) -> None:
        from dataclasses import replace

        artifact_list = list(artifacts)
        if not artifact_list:
            return
        prepared: list[Artifact] = []
        for artifact in artifact_list:
            artifact.validate()
            if artifact.sha256 is None:
                artifact = replace(artifact, sha256=self.artifact_hash(artifact))
            prepared.append(artifact)
        self.init()
        rows: list[tuple[Any, ...]] = []
        event_rows: list[tuple[Any, ...]] = []
        for artifact in prepared:
            rows.append(
                (
                    artifact.id,
                    artifact.job_id,
                    artifact.task_id,
                    str(artifact.type),
                    self._dumps(artifact),
                )
            )
            event_rows.append(
                (
                    artifact.job_id,
                    now_iso(),
                    "artifact.saved",
                    json.dumps(
                        {
                            "artifact_id": artifact.id,
                            "task_id": artifact.task_id,
                            "type": str(artifact.type),
                            "confidence": artifact.confidence,
                            "sha256": artifact.sha256,
                        },
                        sort_keys=True,
                    ),
                )
            )
        with self._session() as connection:
            connection.executemany(
                """
                INSERT INTO artifacts(id, job_id, task_id, type, data)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  job_id = excluded.job_id,
                  task_id = excluded.task_id,
                  type = excluded.type,
                  data = excluded.data
                """,
                rows,
            )
            connection.executemany(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                event_rows,
            )

    def promote_memory(self, memory: MemoryRecord) -> None:
        self.init()
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO memory(id, data)
                VALUES(?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                """,
                (memory.id, self._dumps(memory)),
            )

    def promote_memories(self, records: Iterable[MemoryRecord]) -> None:
        memory_list = list(records)
        if not memory_list:
            return
        self.init()
        rows = [(memory.id, self._dumps(memory)) for memory in memory_list]
        with self._session() as connection:
            connection.executemany(
                """
                INSERT INTO memory(id, data)
                VALUES(?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                """,
                rows,
            )

    def get_job(self, job_id: str) -> Job:
        self.init()
        row = self._one("SELECT data FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise FileNotFoundError(f"job not found: {job_id}")
        return job_from_dict(json.loads(row["data"]))

    def get_task_by_id(self, task_id: str) -> Task:
        self.init()
        row = self._one("SELECT data FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            raise FileNotFoundError(f"task not found: {task_id}")
        return task_from_dict(json.loads(row["data"]))

    def list_jobs(self) -> list[Job]:
        self.init()
        return [
            job_from_dict(json.loads(row["data"]))
            for row in self._all("SELECT data FROM jobs ORDER BY id")
        ]

    def latest_job(self) -> Optional[Job]:
        self.init()
        # Sort by the embedded created_at on the SQLite side and only
        # deserialize the single newest row, instead of loading and decoding
        # every job just to take a max().
        row = self._one(
            "SELECT data FROM jobs "
            "ORDER BY json_extract(data, '$.created_at') DESC, id DESC "
            "LIMIT 1"
        )
        if row is None:
            return None
        return job_from_dict(json.loads(row["data"]))

    def list_tasks(self, job_id: str) -> list[Task]:
        self.init()
        return [
            task_from_dict(json.loads(row["data"]))
            for row in self._all(
                "SELECT data FROM tasks WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
        ]

    def list_tasks_for_jobs(self, job_ids: Iterable[str]) -> list[Task]:
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return []
        self.init()
        tasks: list[Task] = []
        for chunk in _chunked(ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._all(
                f"SELECT data FROM tasks WHERE job_id IN ({placeholders}) ORDER BY job_id, id",
                tuple(chunk),
            )
            tasks.extend(task_from_dict(json.loads(row["data"])) for row in rows)
        return tasks

    def list_artifacts(self, job_id: str) -> list[Artifact]:
        self.init()
        return [
            artifact_from_dict(json.loads(row["data"]))
            for row in self._all(
                "SELECT data FROM artifacts WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
        ]

    def list_artifacts_for_jobs(self, job_ids: Iterable[str]) -> list[Artifact]:
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return []
        self.init()
        artifacts: list[Artifact] = []
        for chunk in _chunked(ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._all(
                f"SELECT data FROM artifacts WHERE job_id IN ({placeholders}) ORDER BY job_id, id",
                tuple(chunk),
            )
            artifacts.extend(artifact_from_dict(json.loads(row["data"])) for row in rows)
        return artifacts

    def get_artifact_job_id(self, artifact_id: str) -> Optional[str]:
        self.init()
        row = self._one("SELECT job_id FROM artifacts WHERE id = ?", (artifact_id,))
        if row is None:
            return None
        return str(row["job_id"])

    def count_artifacts(self, job_id: str) -> int:
        self.init()
        row = self._one(
            "SELECT COUNT(*) AS n FROM artifacts WHERE job_id = ?",
            (job_id,),
        )
        return int(row["n"]) if row is not None else 0

    def get_artifacts_by_ids(
        self, job_id: str, artifact_ids: "Iterable[str]"
    ) -> dict[str, Artifact]:
        unique_ids = [aid for aid in dict.fromkeys(artifact_ids) if aid]
        if not unique_ids:
            return {}
        self.init()
        out: dict[str, Artifact] = {}
        for chunk in _chunked(unique_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._all(
                f"SELECT data FROM artifacts WHERE job_id = ? AND id IN ({placeholders})",
                (job_id, *chunk),
            )
            for row in rows:
                artifact = artifact_from_dict(json.loads(row["data"]))
                out[artifact.id] = artifact
        return out

    def list_artifacts_by_type(
        self, artifact_type: str, job_ids: Optional[Iterable[str]] = None
    ) -> list[Artifact]:
        self.init()
        if job_ids is not None:
            ids = list(dict.fromkeys(job_ids))
            if not ids:
                return []
            artifacts: list[Artifact] = []
            for chunk in _chunked(ids):
                placeholders = ",".join("?" for _ in chunk)
                rows = self._all(
                    f"SELECT data FROM artifacts WHERE type = ? AND job_id IN ({placeholders}) ORDER BY id",
                    (artifact_type, *chunk),
                )
                artifacts.extend(
                    artifact_from_dict(json.loads(row["data"])) for row in rows
                )
            return artifacts
        else:
            rows = self._all(
                "SELECT data FROM artifacts WHERE type = ? ORDER BY id",
                (artifact_type,),
            )
        return [artifact_from_dict(json.loads(row["data"])) for row in rows]

    def list_memory(self) -> list[dict[str, Any]]:
        self.init()
        return [
            json.loads(row["data"])
            for row in self._all("SELECT data FROM memory ORDER BY id")
        ]

    def retrieve_memory(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
        adapter: Optional[str] = None,
        role: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        self.init()
        terms = {term.lower() for term in query.split() if len(term) > 2}
        filter_clauses: list[str] = []
        term_clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("scope", scope),
            ("adapter", adapter),
            ("role", role),
            ("topic", topic),
        ):
            if value is not None:
                filter_clauses.append(f"json_extract(data, '$.{column}') = ?")
                params.append(value)
        if terms:
            for term in terms:
                term_clauses.append("instr(lower(data), ?) > 0")
                params.append(term)
        clauses = list(filter_clauses)
        if term_clauses:
            clauses.append("(" + " OR ".join(term_clauses) + ")")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._all(
            f"SELECT data FROM memory{where} ORDER BY id", tuple(params)
        )
        scored = []
        for row in rows:
            memory = json.loads(row["data"])
            haystack = " ".join(
                str(memory.get(key, ""))
                for key in ["scope", "statement", "evidence", "adapter", "role", "topic"]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            confidence = _coerce_confidence(memory.get("confidence"))
            scored.append((score, confidence, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for score, _, memory in scored[:limit] if score > 0 or not terms]

    def delete_job(self, job_id: str) -> None:
        self.init()
        # Validate the path BEFORE touching SQL so an unsafe id (blank/relative/
        # absolute) can neither wipe rows nor rglob-unlink outside the jobs tree.
        job_dir = self._assert_safe_job_dir(job_id)
        with self._session() as connection:
            for table in ["events", "artifacts", "runs", "tasks"]:
                connection.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        if job_dir.exists():
            for path in sorted(job_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            job_dir.rmdir()

    def schema_status(self) -> dict[str, str]:
        self.init()
        row = self._one("SELECT value FROM metadata WHERE key = 'schema_version'")
        journal = self._one("PRAGMA journal_mode")
        return {
            "schema_version": row["value"] if row else "unknown",
            "expected_schema_version": str(self.schema_version),
            "journal_mode": journal[0] if journal else "unknown",
        }

    def emit(self, job_id: str, event: str, payload: dict[str, Any]) -> None:
        self.init()
        with self._session() as connection:
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (job_id, now_iso(), event, json.dumps(payload, sort_keys=True)),
            )

    def read_events(self, job_id: str) -> list[dict[str, Any]]:
        return self.read_events_since(job_id, since=0)

    def read_events_since(
        self, job_id: str, since: int = 0
    ) -> list[dict[str, Any]]:
        self.init()
        return [
            {
                "id": int(row["id"]),
                "at": row["at"],
                "event": row["event"],
                "payload": json.loads(row["payload"]),
            }
            for row in self._all(
                "SELECT id, at, event, payload FROM events "
                "WHERE job_id = ? AND id > ? ORDER BY id",
                (job_id, int(since)),
            )
        ]

    def event_cursor(self, job_id: str) -> int:
        self.init()
        row = self._one(
            "SELECT COALESCE(MAX(id), 0) AS cursor FROM events WHERE job_id = ?",
            (job_id,),
        )
        return int(row["cursor"]) if row is not None else 0

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        with self._session() as connection:
            return connection.execute(query, params).fetchone()

    def _all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._session() as connection:
            return list(connection.execute(query, params).fetchall())

    def _ensure_job_dirs(self, job_id: str) -> None:
        for directory in [
            self.job_dir(job_id),
            self.job_dir(job_id) / "summaries",
        ]:
            mkdir_private(directory)

    @staticmethod
    def _dumps(value: Any) -> str:
        from puppetmaster.store import _prepare_for_persistence

        return json.dumps(to_jsonable(_prepare_for_persistence(value)), indent=2, sort_keys=True)

