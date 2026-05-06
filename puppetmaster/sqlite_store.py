from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.models import (
    AgentRun,
    Artifact,
    Job,
    JobStatus,
    MemoryRecord,
    Task,
    artifact_from_dict,
    job_from_dict,
    now_iso,
    task_from_dict,
    to_jsonable,
)
from puppetmaster.store import SwarmStore


class SQLiteSwarmStore(SwarmStore):
    """SQLite-backed coordination store for multi-process worker coordination."""

    backend_name = "sqlite"
    schema_version = 1

    def __init__(self, root: Union[Path, str] = ".puppetmaster") -> None:
        super().__init__(root)
        self.db_path = self.root / "state.sqlite3"

    def init(self) -> None:
        for directory in [
            self.root,
            self.jobs_dir,
            self.memory_dir,
            self.stream_dir,
            self.locks_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
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
                  ON artifacts(job_id);
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
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.schema_version),),
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def create_job(self, goal: str) -> Job:
        self.init()
        job = Job(goal=goal)
        self._ensure_job_dirs(job.id)
        with self.connect() as connection:
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
            if status in {JobStatus.COMPLETE, JobStatus.FAILED}
            else job.completed_at,
        )
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET data = ? WHERE id = ?",
                (self._dumps(updated), job_id),
            )
        self.emit(job_id, "job.status", {"status": str(status)})
        return updated

    def save_task(self, task: Task) -> None:
        self.init()
        with self.connect() as connection:
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

    def save_run(self, run: AgentRun) -> None:
        self.init()
        with self.connect() as connection:
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
        self.emit(run.job_id, "run.saved", {"run_id": run.id, "role": run.role})

    def save_artifact(self, artifact: Artifact) -> None:
        artifact.validate()
        if artifact.sha256 is None:
            from dataclasses import replace

            artifact = replace(artifact, sha256=self.artifact_hash(artifact))
        self.init()
        with self.connect() as connection:
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

    def promote_memory(self, memory: MemoryRecord) -> None:
        self.init()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO memory(id, data)
                VALUES(?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                """,
                (memory.id, self._dumps(memory)),
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
        jobs = self.list_jobs()
        if not jobs:
            return None
        return max(jobs, key=lambda job: job.created_at)

    def list_tasks(self, job_id: str) -> list[Task]:
        self.init()
        return [
            task_from_dict(json.loads(row["data"]))
            for row in self._all(
                "SELECT data FROM tasks WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
        ]

    def list_artifacts(self, job_id: str) -> list[Artifact]:
        self.init()
        return [
            artifact_from_dict(json.loads(row["data"]))
            for row in self._all(
                "SELECT data FROM artifacts WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
        ]

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
            confidence = float(memory.get("confidence", 0))
            scored.append((score, confidence, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for score, _, memory in scored[:limit] if score > 0 or not terms]

    def delete_job(self, job_id: str) -> None:
        self.init()
        with self.connect() as connection:
            for table in ["events", "artifacts", "runs", "tasks"]:
                connection.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        job_dir = self.job_dir(job_id)
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
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (job_id, now_iso(), event, json.dumps(payload, sort_keys=True)),
            )

    def read_events(self, job_id: str) -> list[dict[str, Any]]:
        self.init()
        return [
            {
                "at": row["at"],
                "event": row["event"],
                "payload": json.loads(row["payload"]),
            }
            for row in self._all(
                "SELECT at, event, payload FROM events WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
        ]

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(query, params).fetchone()

    def _all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(query, params).fetchall())

    def _ensure_job_dirs(self, job_id: str) -> None:
        for directory in [
            self.job_dir(job_id),
            self.job_dir(job_id) / "summaries",
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(to_jsonable(value), indent=2, sort_keys=True)

