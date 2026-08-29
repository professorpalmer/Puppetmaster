"""Slice A+B: supervisor-only SQLite schema and worker attach (no init herd)."""

from __future__ import annotations

import os
import sqlite3
import sys
import threading

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import multiprocessing
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster.models import AgentRun, Job, JobStatus, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError
from puppetmaster.store import SwarmStore
from puppetmaster.store_factory import create_store, create_worker_store
from puppetmaster.worker_runtime import WorkerDaemon, WorkerRuntime


def _attach_claim_complete_worker(
    state_dir: str, job_id: str, worker_id: str, error_path: str
) -> None:
    """Spawn-safe worker body: attach only, then claim/complete local tasks."""
    try:
        store = SQLiteSwarmStore(state_dir)
        store.attach()
        runtime = WorkerRuntime(
            store=store,
            job_id=job_id,
            role="implement",
            worker_id=worker_id,
            lease_seconds=30,
            poll_seconds=0.05,
        )
        runtime.run_until_idle()
    except Exception as exc:  # noqa: BLE001 — surface in the parent assert
        Path(error_path).write_text(
            f"{type(exc).__name__}: {exc}", encoding="utf-8"
        )


class SqliteAttachEnsureTests(unittest.TestCase):
    def test_attach_without_schema_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            with self.assertRaises(SqliteSchemaError) as ctx:
                store.attach()
            self.assertIn("schema", str(ctx.exception).lower())

            store.init()
            store.attach()
            store.ensure_schema()
            store.attach()

    def test_ensure_schema_then_attach_works(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            store.attach()
            job = store.create_job("after attach")
            self.assertEqual(store.get_job(job.id).goal, "after attach")

    def test_concurrent_attach_never_writes_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            supervisor = SQLiteSwarmStore(root)
            supervisor.ensure_schema()

            scripts: list[str] = []
            metadata_inserts: list[str] = []
            original_connect = SQLiteSwarmStore.connect

            class _SpyConnection:
                def __init__(self, inner: sqlite3.Connection) -> None:
                    self._inner = inner

                def executescript(self, sql: str, *args: object, **kwargs: object):
                    scripts.append(sql)
                    return self._inner.executescript(sql, *args, **kwargs)

                def execute(self, sql: str, parameters: object = ()):
                    text = str(sql)
                    if "INSERT" in text.upper() and "metadata" in text.lower():
                        metadata_inserts.append(text)
                    if parameters == ():
                        return self._inner.execute(sql)
                    return self._inner.execute(sql, parameters)

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, *args: object):
                    return self._inner.__exit__(*args)

                def __getattr__(self, name: str):
                    return getattr(self._inner, name)

            def spy_connect(self: SQLiteSwarmStore) -> sqlite3.Connection:
                return _SpyConnection(original_connect(self))  # type: ignore[return-value]

            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    store = SQLiteSwarmStore(root)
                    store.attach()
                    store.list_jobs()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            with mock.patch.object(SQLiteSwarmStore, "connect", spy_connect):
                threads = [threading.Thread(target=worker) for _ in range(16)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(scripts, [])
            self.assertEqual(metadata_inserts, [])


class SqliteSessionRetryTests(unittest.TestCase):
    def test_session_retries_locked_then_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            real_connect = store.connect
            attempts = {"n": 0}

            def flaky_connect() -> sqlite3.Connection:
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise sqlite3.OperationalError("database is locked")
                return real_connect()

            store.connect = flaky_connect  # type: ignore[method-assign]
            with store._session() as connection:
                connection.execute("SELECT 1")
            self.assertEqual(attempts["n"], 3)
            self.assertGreaterEqual(store.lock_error_count, 2)

    def test_session_retries_busy_then_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            real_connect = store.connect
            attempts = {"n": 0}

            def flaky_connect() -> sqlite3.Connection:
                attempts["n"] += 1
                if attempts["n"] < 2:
                    raise sqlite3.OperationalError("database is busy")
                return real_connect()

            store.connect = flaky_connect  # type: ignore[method-assign]
            with store._session() as connection:
                connection.execute("SELECT 1")
            self.assertEqual(attempts["n"], 2)
            self.assertGreaterEqual(store.lock_error_count, 1)


class WorkerRecoveryOwnershipTests(unittest.TestCase):
    def test_run_until_idle_does_not_recover_stale_tasks(self) -> None:
        store = mock.Mock()
        store.claim_next_task.return_value = None
        store.list_tasks.return_value = []
        runtime = WorkerRuntime(
            store=store,
            job_id="job-1",
            role="implement",
            worker_id="w-1",
        )
        self.assertEqual(runtime.run_until_idle(), 0)
        store.recover_stale_tasks.assert_not_called()

    def test_daemon_run_once_does_not_recover_or_refresh(self) -> None:
        store = mock.Mock()
        job = Job(goal="running", status=JobStatus.RUNNING)
        store.list_jobs.return_value = [job]
        store.claim_next_task.return_value = None
        daemon = WorkerDaemon(store, roles=["implement"], job_id=job.id)
        with mock.patch("puppetmaster.cell.interned_poll", return_value=[]):
            self.assertFalse(daemon.run_once())
        store.recover_stale_tasks.assert_not_called()
        store.refresh_blocked_tasks.assert_not_called()

    def test_claim_next_task_does_not_refresh_blocked(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("claim owns no unblock")
            with mock.patch.object(store, "refresh_blocked_tasks") as refresh:
                claimed = store.claim_next_task(job.id, "w-1")
            self.assertIsNone(claimed)
            refresh.assert_not_called()


class SqliteHeartbeatCoalesceTests(unittest.TestCase):
    def test_coalesced_heartbeat_and_lease_is_one_session(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            job = store.create_job("coalesce heartbeat")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="noop",
                adapter="local",
                payload={"skip_preflight": True},
            )
            store.save_task(task)
            claimed = store.claim_task(task.id, "w-1", lease_seconds=60)
            self.assertIsNotNone(claimed)
            run = AgentRun(
                job_id=job.id,
                task_id=claimed.id,
                role=claimed.role,
                worker_id="w-1",
            )
            store.save_run(run)

            sessions = {"n": 0}
            original_session = store._session

            def counting_session(*args: object, **kwargs: object):
                sessions["n"] += 1
                return original_session(*args, **kwargs)

            with mock.patch.object(store, "_session", side_effect=counting_session):
                updated, renewed = store.heartbeat_run_and_renew_lease(
                    run, claimed.id, "w-1", 60, claimed.lease_id
                )

            self.assertEqual(sessions["n"], 1)
            self.assertIsNotNone(renewed)
            self.assertEqual(renewed.lease_owner, "w-1")
            self.assertEqual(updated.worker_id, "w-1")
            persisted = store.get_task_by_id(claimed.id)
            self.assertEqual(persisted.status, TaskStatus.RUNNING)
            self.assertEqual(persisted.lease_expires_at, renewed.lease_expires_at)


class SqliteMultiprocessAttachTests(unittest.TestCase):
    def test_supervisor_ensure_then_n_workers_attach_claim_complete(self) -> None:
        worker_count = 32
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            supervisor = SQLiteSwarmStore(root)
            supervisor.ensure_schema()
            job = supervisor.create_job("local attach stress")
            for index in range(worker_count):
                supervisor.save_task(
                    Task(
                        job_id=job.id,
                        role="implement",
                        instruction=f"noop-{index}",
                        adapter="local",
                        payload={"skip_preflight": True},
                    )
                )

            error_dir = Path(tmp) / "worker-errors"
            error_dir.mkdir()
            ctx = multiprocessing.get_context("spawn")
            processes = []
            for index in range(worker_count):
                error_path = error_dir / f"w-{index}.txt"
                process = ctx.Process(
                    target=_attach_claim_complete_worker,
                    args=(
                        str(root),
                        job.id,
                        f"w-{index}",
                        str(error_path),
                    ),
                )
                processes.append((process, error_path))
                process.start()

            errors: list[str] = []
            for process, error_path in processes:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    errors.append(f"timeout:{error_path.name}")
                if process.exitcode not in (0, None):
                    errors.append(f"exit:{error_path.name}={process.exitcode}")
                if error_path.is_file():
                    errors.append(error_path.read_text(encoding="utf-8"))

            self.assertEqual(errors, [])
            tasks = supervisor.list_tasks(job.id)
            complete = sum(task.status == TaskStatus.COMPLETE for task in tasks)
            failed = sum(task.status == TaskStatus.FAILED for task in tasks)
            self.assertEqual(complete, worker_count)
            self.assertEqual(failed, 0)

    def test_create_worker_store_attaches_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            with self.assertRaises(SqliteSchemaError):
                create_worker_store("sqlite", root)
            create_store("sqlite", root, mode="ensure")
            worker = create_worker_store("sqlite", root)
            self.assertIsInstance(worker, SQLiteSwarmStore)
            worker.list_jobs()

    def test_create_store_default_is_deferred_and_create_job_ensures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            store = create_store("sqlite", root)
            self.assertFalse((root / "state.sqlite3").exists())
            self.assertEqual(store.list_jobs(), [])
            job = store.create_job("deferred then ensure")
            self.assertTrue((root / "state.sqlite3").exists())
            self.assertEqual(store.get_job(job.id).goal, "deferred then ensure")

            corrupt = Path(tmp) / "corrupt"
            corrupt.mkdir()
            (corrupt / "state.sqlite3").write_bytes(b"not a sqlite database")
            opened = create_store("sqlite", corrupt)
            with self.assertRaises(sqlite3.DatabaseError):
                opened.list_jobs()

    def test_supervisor_migrates_v1_workers_fail_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            root.mkdir()
            db_path = root / "state.sqlite3"
            connection = sqlite3.connect(str(db_path))
            try:
                connection.executescript(
                    """
                    CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE tasks (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      role TEXT NOT NULL,
                      status TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE metadata (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    INSERT INTO metadata(key, value) VALUES('schema_version', '1');
                    INSERT INTO jobs(id, data) VALUES(
                      'job_legacy',
                      '{"id":"job_legacy","goal":"legacy","status":"running","created_at":"2026-01-01T00:00:00+00:00"}'
                    );
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaises(SqliteSchemaError):
                create_worker_store("sqlite", root)

            supervisor = create_store("sqlite", root)
            jobs = supervisor.list_jobs()
            self.assertEqual(jobs[0].id, "job_legacy")
            verify = sqlite3.connect(str(db_path))
            try:
                row = verify.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
            finally:
                verify.close()
            self.assertEqual(int(row[0]), 2)


if __name__ == "__main__":
    unittest.main()
