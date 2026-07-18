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
    now_iso,
    task_from_dict,
    to_jsonable,
)
from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.store import (
    ActiveTaskLeaseError,
    SwarmStore,
    _MEMORY_CAP,
    _coerce_confidence,
    _memory_created_at_sort_key,
    _memory_is_older_than_days,
    _memory_retrieval_score,
    _memory_within_max_age,
    _normalize_memory_statement,
)

_SQLITE_IN_CHUNK = 900


def _chunked(values: Iterable[str], size: int = _SQLITE_IN_CHUNK) -> list[list[str]]:
    unique = list(dict.fromkeys(values))
    return [unique[index : index + size] for index in range(0, len(unique), size)]


class SQLiteSwarmStore(SwarmStore):
    """SQLite-backed coordination store for multi-process worker coordination."""

    backend_name = "sqlite"
    schema_version = 2

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
                CREATE TABLE IF NOT EXISTS graph_edges (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  from_kind TEXT NOT NULL,
                  from_id TEXT NOT NULL,
                  to_kind TEXT NOT NULL,
                  to_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_edges_job
                  ON graph_edges(job_id, type);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_from
                  ON graph_edges(job_id, from_id, type);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_to
                  ON graph_edges(job_id, to_id, type);
                """
            )
            # Fresh DBs get the current schema_version. Existing DBs keep their
            # recorded version so _migrate_schema can detect v1 and backfill.
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (str(self.schema_version),),
            )
            self._migrate_schema(connection)
        chmod_private_file(self.db_path)
        self._initialized = True

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        """Upgrade older state.sqlite3 files to the current schema_version."""
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        try:
            current = int(row["value"]) if row is not None else 0
        except (TypeError, ValueError):
            current = 0
        if current < 2:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                  id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  from_kind TEXT NOT NULL,
                  from_id TEXT NOT NULL,
                  to_kind TEXT NOT NULL,
                  to_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_edges_job
                  ON graph_edges(job_id, type);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_from
                  ON graph_edges(job_id, from_id, type);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_to
                  ON graph_edges(job_id, to_id, type);
                """
            )
            self._backfill_graph_edges(connection)
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.schema_version),),
            )

    def _backfill_graph_edges(self, connection: sqlite3.Connection) -> None:
        """Materialize depends_on / produces edges from existing v1 rows."""
        existing = {
            row["id"]
            for row in connection.execute("SELECT id FROM graph_edges").fetchall()
        }
        for row in connection.execute("SELECT data FROM tasks").fetchall():
            task = task_from_dict(json.loads(row["data"]))
            for dependency_id in task.depends_on:
                if not dependency_id:
                    continue
                edge = make_graph_edge(
                    job_id=task.job_id,
                    type=GraphEdgeType.DEPENDS_ON,
                    from_kind=GraphNodeKind.TASK,
                    from_id=task.id,
                    to_kind=GraphNodeKind.TASK,
                    to_id=dependency_id,
                )
                if edge.id in existing:
                    continue
                self._insert_edge_row(connection, edge, emit_event=False)
                existing.add(edge.id)
        for row in connection.execute("SELECT data FROM artifacts").fetchall():
            artifact = artifact_from_dict(json.loads(row["data"]))
            edge = make_graph_edge(
                job_id=artifact.job_id,
                type=GraphEdgeType.PRODUCES,
                from_kind=GraphNodeKind.TASK,
                from_id=artifact.task_id,
                to_kind=GraphNodeKind.ARTIFACT,
                to_id=artifact.id,
            )
            if edge.id in existing:
                continue
            self._insert_edge_row(connection, edge, emit_event=False)
            existing.add(edge.id)

    def _insert_edge_row(
        self,
        connection: sqlite3.Connection,
        edge: GraphEdge,
        *,
        emit_event: bool,
    ) -> GraphEdge:
        return self._upsert_edge_connection(
            connection, edge, emit_event=emit_event
        )

    def _upsert_edge_connection(
        self,
        connection: sqlite3.Connection,
        edge: GraphEdge,
        *,
        emit_event: bool = True,
    ) -> GraphEdge:
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
        row = connection.execute(
            "SELECT data FROM graph_edges WHERE id = ?", (edge.id,)
        ).fetchone()
        if row is not None:
            existing = graph_edge_from_dict(json.loads(row["data"]))
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
        connection.execute(
            """
            INSERT INTO graph_edges(
              id, job_id, type, from_kind, from_id, to_kind, to_id, created_at, data
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              job_id = excluded.job_id,
              type = excluded.type,
              from_kind = excluded.from_kind,
              from_id = excluded.from_id,
              to_kind = excluded.to_kind,
              to_id = excluded.to_id,
              data = excluded.data
            """,
            (
                edge.id,
                edge.job_id,
                str(edge.type),
                str(edge.from_kind),
                edge.from_id,
                str(edge.to_kind),
                edge.to_id,
                edge.created_at,
                self._dumps(edge),
            ),
        )
        if emit_event:
            self._emit(
                connection,
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

    @staticmethod
    def _emit(connection: sqlite3.Connection, job_id: str, event: str, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
            (job_id, now_iso(), event, json.dumps(payload, sort_keys=True)),
        )

    def create_job(self, goal: str, *, label: Optional[str] = None) -> Job:
        self.init()
        job = Job(goal=goal, label=label)
        self._ensure_job_dirs(job.id)
        with self._session() as connection:
            connection.execute(
                "INSERT INTO jobs(id, data) VALUES(?, ?)",
                (job.id, self._dumps(job)),
            )
        payload: dict[str, Any] = {"goal": goal}
        if label is not None:
            payload["label"] = label
        self.emit(job.id, "job.created", payload)
        return job

    def update_job_status(self, job_id: str, status: JobStatus) -> Job:
        job = self.get_job(job_id)
        updated = self._job_with_status(job, status)
        payload = {"status": str(status)}
        with self._session() as connection:
            connection.execute(
                "UPDATE jobs SET data = ? WHERE id = ?",
                (self._dumps(updated), job_id),
            )
            self._emit(connection, job_id, "job.status", payload)
        return updated

    def save_task(self, task: Task) -> None:
        self.init()
        # Persist the task row and its task.saved event in a single
        # transaction. Splitting them across two connections left a window
        # where a crash after the task write but before the event write would
        # produce state with no corresponding event (a torn write that breaks
        # event-cursor consumers).
        payload = self._task_saved_payload(task)
        desired_ids = {
            dependency_id for dependency_id in task.depends_on if dependency_id
        }
        depends_edges = [
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
            self._emit(connection, task.job_id, "task.saved", payload)
            self._reconcile_depends_on_edges_connection(
                connection, task, desired_ids, depends_edges
            )

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        task_list = list(tasks)
        if not task_list:
            return
        self.init()
        rows: list[tuple[Any, ...]] = []
        event_rows: list[tuple[Any, ...]] = []
        reconcile_rows: list[tuple[Task, set[str], list[GraphEdge]]] = []
        for task in task_list:
            payload = self._task_saved_payload(task)
            rows.append(
                (task.id, task.job_id, task.role, str(task.status), self._dumps(task))
            )
            event_rows.append((task.job_id, payload))
            desired_ids = {
                dependency_id for dependency_id in task.depends_on if dependency_id
            }
            depends_edges = [
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
            reconcile_rows.append((task, desired_ids, depends_edges))
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
            for job_id, payload in event_rows:
                self._emit(connection, job_id, "task.saved", payload)
            for task, desired_ids, depends_edges in reconcile_rows:
                self._reconcile_depends_on_edges_connection(
                    connection, task, desired_ids, depends_edges
                )

    def update_task_status(
        self,
        task: Task,
        status: TaskStatus,
        worker_id: Optional[str] = None,
        lease_id: Optional[str] = None,
    ) -> Task:
        stored = self.get_task_by_id(task.id)
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

    def _atomic_status_update(
        self,
        task_id: str,
        updated: Task,
        *,
        terminal: bool,
        worker_id: Optional[str],
        expected_lease: Optional[str],
    ) -> Task:
        payload = self._task_saved_payload(updated)
        with self._session() as connection:
            if terminal and worker_id is not None:
                # The CAS must re-check the lease in SQL (not just the Python
                # read above) so a concurrent reclaim that lands between the
                # read and the write still loses. We match the owner always and
                # the token whenever one is expected.
                if expected_lease is not None:
                    cursor = connection.execute(
                        """
                        UPDATE tasks SET status = ?, data = ?
                        WHERE id = ?
                          AND json_extract(data, '$.lease_owner') = ?
                          AND json_extract(data, '$.lease_id') = ?
                        """,
                        (
                            str(updated.status),
                            self._dumps(updated),
                            task_id,
                            worker_id,
                            expected_lease,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE tasks SET status = ?, data = ?
                        WHERE id = ? AND json_extract(data, '$.lease_owner') = ?
                        """,
                        (str(updated.status), self._dumps(updated), task_id, worker_id),
                    )
                if cursor.rowcount != 1:
                    return self.get_task_by_id(task_id)
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

    def _atomic_renew_lease(
        self,
        task_id: str,
        task: Task,
        renewed: Task,
        worker_id: str,
        lease_id: Optional[str],
    ) -> Optional[Task]:
        # Single CAS UPDATE fenced on (RUNNING, lease_owner, and the lease token
        # when present) so a stale owner can never extend a lease it no longer holds.
        self.init()
        with self._session() as connection:
            if lease_id is not None and task.lease_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE tasks SET data = ?
                    WHERE id = ?
                      AND status = ?
                      AND json_extract(data, '$.lease_owner') = ?
                      AND json_extract(data, '$.lease_id') = ?
                    """,
                    (
                        self._dumps(renewed),
                        task_id,
                        str(TaskStatus.RUNNING),
                        worker_id,
                        task.lease_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE tasks SET data = ?
                    WHERE id = ?
                      AND status = ?
                      AND json_extract(data, '$.lease_owner') = ?
                    """,
                    (
                        self._dumps(renewed),
                        task_id,
                        str(TaskStatus.RUNNING),
                        worker_id,
                    ),
                )
            if cursor.rowcount != 1:
                return None
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (
                    task.job_id,
                    now_iso(),
                    "task.lease_renewed",
                    json.dumps(
                        {
                            "task_id": task.id,
                            "worker_id": worker_id,
                            "lease_expires_at": renewed.lease_expires_at,
                        },
                        sort_keys=True,
                    ),
                ),
            )
        return renewed

    def claim_task(
        self,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        task_map: Optional[dict[str, Task]] = None,
    ) -> Optional[Task]:
        self.init()
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
        return claimed

    def _atomic_claim(
        self,
        task_id: str,
        task: Task,
        claimed: Task,
        worker_id: Optional[str] = None,
    ) -> bool:
        owner = worker_id if worker_id is not None else (claimed.lease_owner or "")
        now = now_iso()
        claim_payload = self._task_claim_payload(task_id, owner, claimed)
        with self._session() as connection:
            # We only reach here once ``dependencies_complete`` has passed, so a
            # task persisted as BLOCKED is now runnable: include it in the CAS
            # WHERE clause alongside QUEUED and stale-RUNNING. Without this a
            # BLOCKED task whose deps later completed could never be claimed on
            # SQLite, diverging from the file-backed SwarmStore contract.
            cursor = connection.execute(
                """
                UPDATE tasks SET status = ?, data = ?
                WHERE id = ? AND (
                  status = ?
                  OR status = ?
                  OR (status = ? AND json_extract(data, '$.lease_expires_at') <= ?)
                )
                """,
                (
                    str(TaskStatus.RUNNING),
                    self._dumps(claimed),
                    task_id,
                    str(TaskStatus.QUEUED),
                    str(TaskStatus.BLOCKED),
                    str(TaskStatus.RUNNING),
                    now,
                ),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                (
                    task.job_id,
                    now,
                    "task.claimed",
                    json.dumps(claim_payload, sort_keys=True),
                ),
            )
        return True

    def recover_stale_tasks(self, job_id: str) -> list[Task]:
        self.init()
        now = now_iso()
        stale = [task for task in self.list_tasks(job_id) if self.is_task_stale(task)]
        if not stale:
            return []
        recovered: list[Task] = []
        with self._session() as connection:
            for task in stale:
                queued = self._build_recovered_task(task)
                if not self._atomic_recover_stale(task, queued, connection=connection, now=now):
                    continue
                connection.execute(
                    "INSERT INTO events(job_id, at, event, payload) VALUES(?, ?, ?, ?)",
                    (
                        job_id,
                        now,
                        "task.recovered",
                        json.dumps(
                            {"task_id": task.id, "previous_owner": task.lease_owner},
                            sort_keys=True,
                        ),
                    ),
                )
                recovered.append(queued)
        for task in recovered:
            self.release_lock(f"task:{task.id}")
        return recovered

    def _atomic_recover_stale(
        self,
        task: Task,
        queued: Task,
        *,
        connection: Optional[sqlite3.Connection] = None,
        now: Optional[str] = None,
    ) -> bool:
        if connection is not None:
            stamp = now or now_iso()
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
                    stamp,
                ),
            )
            return cursor.rowcount == 1
        return super()._atomic_recover_stale(task, queued)

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
            artifact = replace(artifact, sha256=self.artifact_hash(artifact))
        self.init()
        event_payload = {
            "artifact_id": artifact.id,
            "task_id": artifact.task_id,
            "type": str(artifact.type),
            "confidence": artifact.confidence,
            "sha256": artifact.sha256,
        }
        produces = make_graph_edge(
            job_id=artifact.job_id,
            type=GraphEdgeType.PRODUCES,
            from_kind=GraphNodeKind.TASK,
            from_id=artifact.task_id,
            to_kind=GraphNodeKind.ARTIFACT,
            to_id=artifact.id,
        )
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
            self._upsert_edge_connection(connection, produces)

    def save_artifacts(self, artifacts: Iterable[Artifact]) -> None:
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
        produces_edges = [
            make_graph_edge(
                job_id=artifact.job_id,
                type=GraphEdgeType.PRODUCES,
                from_kind=GraphNodeKind.TASK,
                from_id=artifact.task_id,
                to_kind=GraphNodeKind.ARTIFACT,
                to_id=artifact.id,
            )
            for artifact in prepared
        ]
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
            for edge in produces_edges:
                self._upsert_edge_connection(connection, edge)

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        self.init()
        with self._session() as connection:
            return self._upsert_edge_connection(connection, edge)

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> list[GraphEdge]:
        """Batch-upsert edges in a single SQLite transaction."""
        edge_list = list(edges)
        if not edge_list:
            return []
        self.init()
        with self._session() as connection:
            return [
                self._upsert_edge_connection(connection, edge) for edge in edge_list
            ]

    def record_consumes(
        self,
        job_id: str,
        task_id: str,
        artifact_ids: Iterable[str],
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> list[GraphEdge]:
        """Record consumes edges in one SQLite transaction (no file journal)."""
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
        return self.upsert_edges(edges)

    def reset_subgraph(
        self,
        job_id: str,
        task_ids: Iterable[str],
        *,
        include_descendants: bool = True,
    ) -> list[Task]:
        """Idempotent subgraph reset in a single SQLite transaction/batch.

        Unlike the file backend (per-task writes), all selected task clears,
        QUEUED/BLOCKED re-derivation, and the ``subgraph.reset`` event commit
        together so a busy/crash mid-reset cannot leave a partial subgraph.
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

        cleared: list[Task] = []
        for task in tasks:
            if task.id not in selected:
                continue
            cleared_task = replace(
                task,
                status=TaskStatus.BLOCKED,
                attempts=0,
                lease_owner=None,
                lease_expires_at=None,
                lease_id=None,
                completed_at=None,
                updated_at=now_iso(),
            )
            task_map[task.id] = cleared_task
            cleared.append(cleared_task)

        finalized: list[Task] = []
        for task in cleared:
            current = task_map[task.id]
            if self.dependencies_complete(current, task_map=task_map):
                queued = replace(
                    current, status=TaskStatus.QUEUED, updated_at=now_iso()
                )
                task_map[task.id] = queued
                finalized.append(queued)
            else:
                finalized.append(current)

        self.init()
        rows: list[tuple[Any, ...]] = []
        event_rows: list[tuple[Any, ...]] = []
        reconcile_rows: list[tuple[Task, set[str], list[GraphEdge]]] = []
        for task in finalized:
            payload = self._task_saved_payload(task)
            rows.append(
                (task.id, task.job_id, task.role, str(task.status), self._dumps(task))
            )
            event_rows.append((task.job_id, payload))
            desired_ids = {
                dependency_id for dependency_id in task.depends_on if dependency_id
            }
            depends_edges = [
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
            reconcile_rows.append((task, desired_ids, depends_edges))

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
            for event_job_id, payload in event_rows:
                self._emit(connection, event_job_id, "task.saved", payload)
            for task, desired_ids, depends_edges in reconcile_rows:
                self._reconcile_depends_on_edges_connection(
                    connection, task, desired_ids, depends_edges
                )
            self._emit(
                connection,
                job_id,
                "subgraph.reset",
                {"task_ids": sorted(selected), "reset_count": len(finalized)},
            )
        return finalized

    def delete_edge(self, job_id: str, edge_id: str) -> bool:
        self.init()
        with self._session() as connection:
            row = connection.execute(
                "SELECT id FROM graph_edges WHERE id = ? AND job_id = ?",
                (edge_id, job_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "DELETE FROM graph_edges WHERE id = ? AND job_id = ?",
                (edge_id, job_id),
            )
            self._emit(
                connection,
                job_id,
                "edge.deleted",
                {"edge_id": edge_id},
            )
            return True

    def _reconcile_depends_on_edges_connection(
        self,
        connection: sqlite3.Connection,
        task: Task,
        desired_ids: set[str],
        depends_edges: list[GraphEdge],
    ) -> None:
        rows = connection.execute(
            """
            SELECT id, to_id FROM graph_edges
            WHERE job_id = ?
              AND type = ?
              AND from_id = ?
              AND from_kind = ?
              AND to_kind = ?
            """,
            (
                task.job_id,
                str(GraphEdgeType.DEPENDS_ON),
                task.id,
                str(GraphNodeKind.TASK),
                str(GraphNodeKind.TASK),
            ),
        ).fetchall()
        for row in rows:
            if row["to_id"] not in desired_ids:
                connection.execute(
                    "DELETE FROM graph_edges WHERE id = ?", (row["id"],)
                )
                self._emit(
                    connection,
                    task.job_id,
                    "edge.deleted",
                    {"edge_id": row["id"]},
                )
        for edge in depends_edges:
            self._upsert_edge_connection(connection, edge)

    def get_edge(self, job_id: str, edge_id: str) -> Optional[GraphEdge]:
        self.init()
        row = self._one(
            "SELECT data FROM graph_edges WHERE id = ? AND job_id = ?",
            (edge_id, job_id),
        )
        if row is None:
            return None
        return graph_edge_from_dict(json.loads(row["data"]))

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
        self.init()
        clauses = ["job_id = ?"]
        params: list[Any] = [job_id]
        if edge_type is not None:
            clauses.append("type = ?")
            params.append(str(edge_type))
        if from_id is not None:
            clauses.append("from_id = ?")
            params.append(from_id)
        if to_id is not None:
            clauses.append("to_id = ?")
            params.append(to_id)
        if from_kind is not None:
            clauses.append("from_kind = ?")
            params.append(str(from_kind))
        if to_kind is not None:
            clauses.append("to_kind = ?")
            params.append(str(to_kind))
        where = " AND ".join(clauses)
        rows = self._all(
            f"SELECT data FROM graph_edges WHERE {where} ORDER BY id",
            tuple(params),
        )
        return [graph_edge_from_dict(json.loads(row["data"])) for row in rows]

    def ensure_graph_edges(self, job_id: str) -> None:
        """SQLite edges are migrated eagerly; no file-store lazy backfill."""
        return None

    def promote_memory(self, memory: MemoryRecord) -> None:
        normalized = _normalize_memory_statement(memory.statement)
        for existing in self.list_memory():
            if existing.get("scope") != memory.scope:
                continue
            if _normalize_memory_statement(str(existing.get("statement") or "")) == normalized:
                return
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
        self._enforce_memory_cap(_MEMORY_CAP)

    def _delete_memory_record(self, memory_id: str) -> None:
        self.init()
        with self._session() as connection:
            connection.execute("DELETE FROM memory WHERE id = ?", (memory_id,))

    def _enforce_memory_cap(self, cap: int) -> None:
        records = self.list_memory()
        if len(records) <= cap:
            return
        sorted_records = sorted(records, key=_memory_created_at_sort_key)
        for memory in sorted_records[: len(records) - cap]:
            memory_id = memory.get("id")
            if memory_id:
                self._delete_memory_record(str(memory_id))

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
        max_age_days: Optional[int] = None,
        min_overlap: float = 0.0,
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
            if not _memory_within_max_age(memory, max_age_days):
                continue
            score, confidence, created_at_key, overlap = _memory_retrieval_score(memory, terms)
            if terms and min_overlap > 0 and overlap < min_overlap:
                continue
            scored.append((score, confidence, created_at_key, memory))
        from puppetmaster.mmr import finalize_memory_retrieval

        return finalize_memory_retrieval(scored, terms, limit)

    def delete_job(self, job_id: str) -> None:
        self.init()
        # Validate the path BEFORE touching SQL so an unsafe id (blank/relative/
        # absolute) can neither wipe rows nor rglob-unlink outside the jobs tree.
        job_dir = self._assert_safe_job_dir(job_id)
        with self._session() as connection:
            for table in ["events", "artifacts", "runs", "tasks", "graph_edges"]:
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

