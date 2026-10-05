"""Slice A+B: supervisor-only SQLite schema and worker attach (no init herd)."""

from __future__ import annotations

import faulthandler
import json
import os
import sqlite3
import sys
import threading
import time
import traceback

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import multiprocessing
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster.models import AgentRun, Job, JobStatus, MemoryRecord, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError
from puppetmaster.store import SwarmStore
from puppetmaster.store_factory import create_store, create_worker_store
from puppetmaster.worker_runtime import WorkerDaemon, WorkerRuntime
from puppetmaster.workers import LocalWorker


def _attach_claim_complete_worker(
    state_dir: str, job_id: str, worker_id: str, error_path: str, diagnostic_path: str
) -> None:
    """Spawn-safe worker body: attach only, then claim/complete local tasks."""
    error_file = Path(error_path).open("w", encoding="utf-8")
    diagnostic_file = Path(diagnostic_path).open("w", encoding="utf-8")
    # Per-worker dump files. Do not arm only w-0: dest-push last stalled on w-26.
    faulthandler.dump_traceback_later(45, file=diagnostic_file)
    thread_errors: list[str] = []
    previous_hook = threading.excepthook
    threading.excepthook = lambda args: thread_errors.append(
        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    )
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
    except Exception:  # noqa: BLE001 — surface in the parent assert
        error_file.write(traceback.format_exc())
    finally:
        faulthandler.cancel_dump_traceback_later()
        threading.excepthook = previous_hook
        if thread_errors:
            error_file.write("\n".join(thread_errors))
        error_file.close()
        diagnostic_file.close()
        if Path(error_path).stat().st_size == 0:
            Path(error_path).unlink()
        # Hung workers are TerminateProcess'd and never reach this line, so
        # their dump stays for the parent. Clean exits drop the empty dump.
        Path(diagnostic_path).unlink(missing_ok=True)


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

    def test_attach_retries_proven_windows_source_open_contention(self) -> None:
        from puppetmaster.readonly import ReadUnavailable

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            SQLiteSwarmStore(root).ensure_schema()
            store = SQLiteSwarmStore(root)
            original_connect = store._connect_readonly
            denied = ReadUnavailable(
                "unable to open database: source changed or unavailable: "
                "[WinError 5] Access is denied."
            )
            denied.source_open_contention = True
            calls = 0

            def connect(*args: object, **kwargs: object):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise denied
                return original_connect(*args, **kwargs)

            with mock.patch.object(store, "_connect_readonly", side_effect=connect), \
                    mock.patch.object(store, "_sleep_lock_backoff"):
                store.attach()
            self.assertEqual(calls, 2)

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

            if errors:
                self.fail("\n\n".join(str(error) for error in errors[:2]))
            self.assertEqual(scripts, [])
            self.assertEqual(metadata_inserts, [])


class SqliteSessionRetryTests(unittest.TestCase):
    def test_artifact_write_contention_does_not_fail_task_and_exit_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("contended publication")
            task = Task(job_id=job.id, role="implement", instruction="noop",
                        adapter="local", payload={"skip_preflight": True})
            store.save_task(task)
            store.busy_timeout_ms = 0
            blocker = store.connect()
            execute = LocalWorker.run
            save_artifact = store.save_artifact
            executions = []

            def execute_then_contend(worker, *args):
                result = execute(worker, *args)
                executions.append(task.id)
                blocker.execute("BEGIN IMMEDIATE")
                return result

            def publish(artifact):
                try:
                    return save_artifact(artifact)
                finally:
                    # Let the runtime persist FAILED on the broken path so
                    # the regression observes its otherwise clean idle exit.
                    blocker.rollback()

            runtime = WorkerRuntime(store, job.id, "implement", "w-1",
                                    lease_seconds=30, heartbeat_seconds=10)
            try:
                with mock.patch.object(LocalWorker, "run", execute_then_contend), \
                     mock.patch.object(store, "save_artifact", publish), \
                     mock.patch.object(store, "_sleep_lock_backoff",
                                       side_effect=lambda attempt: blocker.rollback()) as retry:
                    self.assertEqual(runtime.run_until_idle(), 1)
                self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
                retry.assert_called_once_with(0)
                self.assertEqual(executions, [task.id])
                events = store.read_events(job.id)
                self.assertEqual(sum(e["event"] == "worker.completed_task" for e in events), 1)
                self.assertFalse(any(e["event"] == "worker.failed_task" for e in events))
                self.assertEqual(len(store.list_artifacts(job.id)), 1)
                with store._session() as connection:
                    rows = connection.execute("SELECT status, data FROM tasks").fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], TaskStatus.COMPLETE.value)
                self.assertEqual(json.loads(rows[0]["data"])["status"], rows[0]["status"])
            finally:
                blocker.close()

    def test_coalesced_heartbeat_retries_real_writer_contention(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("contended heartbeat")
            task = Task(job_id=job.id, role="implement", instruction="noop")
            store.save_task(task)
            claimed = store.claim_task(task.id, "w-1")
            run = AgentRun(job_id=job.id, task_id=task.id, role=task.role, worker_id="w-1")
            store.save_run(run)
            store.busy_timeout_ms = 0
            blocker = store.connect()
            try:
                blocker.execute("BEGIN IMMEDIATE")
                with mock.patch.object(store, "_sleep_lock_backoff",
                                       side_effect=lambda attempt: blocker.rollback()) as retry:
                    _, renewed = store.heartbeat_run_and_renew_lease(
                        run, task.id, "w-1", lease_id=claimed.lease_id)
                retry.assert_called_once_with(0)
                self.assertIsNotNone(renewed)
                self.assertEqual(renewed.lease_id, claimed.lease_id)
                events = store.read_events(job.id)
                for event in ("task.lease_renewed", "run.heartbeat"):
                    self.assertEqual(sum(e["event"] == event for e in events), 1)
            finally:
                blocker.close()

    def test_opportunistic_heartbeat_does_not_queue_behind_writer(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("opportunistic heartbeat")
            task = Task(job_id=job.id, role="implement", instruction="noop")
            store.save_task(task)
            claimed = store.claim_task(task.id, "w-1")
            run = AgentRun(job_id=job.id, task_id=task.id, role=task.role, worker_id="w-1")
            store.save_run(run)
            blocker = store.connect()
            try:
                blocker.execute("BEGIN IMMEDIATE")
                with mock.patch.object(store, "_sleep_lock_backoff") as retry:
                    with self.assertRaises(sqlite3.OperationalError):
                        store.heartbeat_run_and_renew_lease_opportunistic(
                            run, task.id, "w-1", lease_id=claimed.lease_id)
                retry.assert_not_called()
            finally:
                blocker.rollback()
                blocker.close()

    def test_waiting_heartbeat_cannot_overwrite_completion_or_successor_lease(self) -> None:
        for terminal in (False, True):
            with self.subTest(terminal=terminal), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                job = store.create_job("heartbeat fencing")
                task = Task(job_id=job.id, role="implement", instruction="noop")
                store.save_task(task)
                claimed = store.claim_task(task.id, "same-worker")
                run = AgentRun(job_id=job.id, task_id=task.id,
                               role=task.role, worker_id="same-worker")
                changed = (store._build_status_update(claimed, TaskStatus.COMPLETE)
                           if terminal else replace(claimed, lease_id="successor",
                                                    attempts=claimed.attempts + 1))
                store.busy_timeout_ms = 0
                blocker = store.connect()
                try:
                    blocker.execute("BEGIN IMMEDIATE")
                    blocker.execute("UPDATE tasks SET status = ?, data = ? WHERE id = ?",
                                    (str(changed.status), store._dumps(changed), task.id))
                    with mock.patch.object(store, "_sleep_lock_backoff",
                                           side_effect=lambda attempt: blocker.commit()):
                        _, renewed = store.heartbeat_run_and_renew_lease(
                            run, task.id, "same-worker", lease_id=claimed.lease_id)
                    self.assertIsNone(renewed)
                    self.assertEqual(store.get_task_by_id(task.id), changed)
                    self.assertFalse(any(e["event"] == "task.lease_renewed"
                                         for e in store.read_events(job.id)))
                finally:
                    blocker.close()

    def test_attach_retries_real_open_lock(self) -> None:
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            blocker = supervisor.connect()
            try:
                blocker.execute("PRAGMA locking_mode = EXCLUSIVE")
                blocker.execute("BEGIN EXCLUSIVE")
                worker = SQLiteSwarmStore(tmp)
                worker.busy_timeout_ms = 0
                # Release only after SQLite has actually reported contention.
                with mock.patch.object(worker, "_sleep_lock_backoff",
                                       side_effect=lambda attempt: blocker.close()) as retry:
                    worker.attach()
                retry.assert_called_once()
                self.assertGreaterEqual(worker.lock_error_count, 1)
                self.assertEqual(worker.list_jobs(), [])
            finally:
                blocker.close()

    def test_recovery_cas_retries_writer_contention(self) -> None:
        for publish_intent in (False, True):
            with self.subTest(publish_intent=publish_intent), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                job = store.create_job("recovery contention")
                task = Task(job_id=job.id, role="implement", instruction="noop")
                store.save_task(task)
                task = store.claim_task(task.id, "worker")
                task = replace(task, lease_expires_at="2000-01-01T00:00:00+00:00")
                store.save_task(task)
                publisher = SQLiteSwarmStore(tmp)
                publisher.attach()
                store.busy_timeout_ms = 0
                blocker = store.connect()
                original = store._atomic_recover_stale

                def interleaved(*args, **kwargs):
                    blocker.execute("BEGIN IMMEDIATE")
                    return original(*args, **kwargs)

                def release_writer(attempt):
                    blocker.rollback()
                    if publish_intent:
                        run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                                       worker_id="worker", status=TaskStatus.COMPLETE)
                        with mock.patch.object(publisher, "reconcile_completions"):
                            publisher.complete_task(task, run, [], {"task_id": task.id})

                try:
                    with mock.patch.object(store, "_atomic_recover_stale", side_effect=interleaved), \
                         mock.patch.object(store, "_sleep_lock_backoff", side_effect=release_writer) as retry:
                        recovered = store.recover_stale_tasks(job.id)
                    retry.assert_called_once_with(0)
                    self.assertEqual([item.id for item in recovered],
                                     [] if publish_intent else [task.id])
                    events = [event for event in store.read_events(job.id)
                              if event["event"] == "task.recovered"]
                    self.assertEqual(len(events), 0 if publish_intent else 1)
                    store.reconcile_completions(job.id)
                    self.assertEqual(store.get_task_by_id(task.id).status,
                                     TaskStatus.COMPLETE if publish_intent else TaskStatus.QUEUED)
                finally:
                    blocker.close()

    def test_recovery_event_failure_rolls_back_cas_without_replaying(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("recovery rollback")
            task = Task(job_id=job.id, role="implement", instruction="noop")
            store.save_task(task)
            task = store.claim_task(task.id, "worker")
            task = replace(task, lease_expires_at="2000-01-01T00:00:00+00:00")
            store.save_task(task)
            with mock.patch.object(store, "_emit", side_effect=sqlite3.OperationalError("database is locked")) as emit, \
                 mock.patch.object(store, "_sleep_lock_backoff") as retry:
                with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                    store.recover_stale_tasks(job.id)
            emit.assert_called_once()
            retry.assert_not_called()
            self.assertEqual(store.get_task_by_id(task.id), task)
            self.assertFalse(any(event["event"] == "task.recovered"
                                 for event in store.read_events(job.id)))

    def test_previously_bare_mutations_retry_reservation(self) -> None:
        for operation in ("create", "reset", "edge", "delete", "memory", "forget", "memories", "schema"):
            with self.subTest(operation=operation), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                job = store.create_job("contention")
                task = Task(job_id=job.id, role="implement", instruction="noop")
                store.save_task(task)
                memory = MemoryRecord(scope="test", statement="remember", evidence=[],
                                      source_artifacts=[], confidence=1.0)
                store.promote_memory(memory)
                actions = {
                    "create": lambda: store.create_or_get_job("second"),
                    "reset": lambda: store.reset_subgraph(job.id, [task.id]),
                    "edge": lambda: store.delete_edge(job.id, "missing"),
                    "delete": lambda: store.delete_job(job.id),
                    "memory": lambda: store.promote_memory(replace(memory, statement="new")),
                    "forget": lambda: store._delete_memory_record(memory.id),
                    "memories": lambda: store.promote_memories([memory]),
                    "schema": store.ensure_schema,
                }
                if operation == "schema":
                    store._initialized = False
                store.busy_timeout_ms = 0
                blocker = store.connect()
                try:
                    blocker.execute("BEGIN IMMEDIATE")
                    with mock.patch.object(store, "_sleep_lock_backoff",
                                           side_effect=lambda attempt: blocker.rollback()) as retry:
                        actions[operation]()
                    retry.assert_called_once_with(0)
                finally:
                    blocker.close()

    def test_recovery_cannot_overwrite_expired_successor(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("successor recovery")
            task = Task(job_id=job.id, role="implement", instruction="noop")
            store.save_task(task)
            scanned = store.claim_task(task.id, "same-worker")
            expired = "2000-01-01T00:00:00+00:00"
            scanned = replace(scanned, lease_expires_at=expired)
            store.save_task(scanned)
            original = store._atomic_recover_stale
            successors = []

            def interleaved(old, queued):
                successor = store.claim_task(task.id, "same-worker")
                self.assertIsNotNone(successor)
                successor = replace(successor, lease_expires_at=expired)
                store.save_task(successor)
                successors.append(successor)
                return original(old, queued)

            with mock.patch.object(store, "_atomic_recover_stale", side_effect=interleaved):
                self.assertEqual(store.recover_stale_tasks(job.id), [])
            current = store.get_task_by_id(task.id)
            self.assertEqual(current, successors[0])
            self.assertGreater(current.attempts, scanned.attempts)
            self.assertFalse(any(e["event"] == "task.recovered" for e in store.read_events(job.id)))
            recovered = store.recover_stale_tasks(job.id)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0].attempts, current.attempts)

    def test_recovery_fences_each_scanned_identity_field(self) -> None:
        for change in ({"lease_id": "new-token"}, {"generation": 9},
                       {"lease_owner": "new-owner"}, {"attempts": 9}):
            with self.subTest(change=change), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                job = store.create_job("identity fence")
                scanned = Task(job_id=job.id, role="implement", instruction="noop",
                               status=TaskStatus.RUNNING, attempts=2, generation=None,
                               lease_id=None, lease_owner="worker",
                               lease_expires_at="2000-01-01T00:00:00+00:00")
                store.save_task(scanned)
                successor = replace(scanned, **change)
                store.save_task(successor)
                self.assertFalse(store._atomic_recover_stale(scanned, store._build_recovered_task(scanned)))
                self.assertEqual(store.get_task_by_id(scanned.id), successor)
                self.assertTrue(store._atomic_recover_stale(successor, store._build_recovered_task(successor)))

    def test_reset_and_memory_share_outer_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("nested reset")
            task = Task(job_id=job.id, role="implement", instruction="noop", status=TaskStatus.COMPLETE)
            store.save_task(task)
            memory = MemoryRecord(scope="test", statement="nested", evidence=[],
                                  source_artifacts=[], confidence=1.0)
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with store._writer_scope():
                    store.reset_subgraph(job.id, [task.id])
                    self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.QUEUED)
                    store.promote_memory(memory)
                    self.assertEqual(len(store.list_memory()), 1)
                    raise RuntimeError("abort")
            self.assertEqual(store.get_task_by_id(task.id), task)
            self.assertEqual(store.list_memory(), [])
            self.assertFalse(any(e["event"] == "subgraph.reset" for e in store.read_events(job.id)))

    def test_writer_retries_real_reservation_lock_before_body(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("writer contention")
            blocker = store.connect()
            try:
                blocker.execute("BEGIN IMMEDIATE")
                store.busy_timeout_ms = 0
                with mock.patch.object(store, "_sleep_lock_backoff",
                                       side_effect=lambda attempt: blocker.rollback()) as retry:
                    with store._writer_scope():
                        store.emit(job.id, "test.once", {})
                        with store._writer_scope():
                            store.emit(job.id, "test.nested", {})
                retry.assert_called_once_with(0)
                self.assertEqual(store.lock_error_count, 1)
                events = store.read_events(job.id)
                self.assertEqual(sum(e["event"] == "test.once" for e in events), 1)
                self.assertEqual(sum(e["event"] == "test.nested" for e in events), 1)
            finally:
                blocker.close()

    def test_writer_contention_is_bounded_and_never_enters_body(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            blocker = store.connect()
            try:
                blocker.execute("BEGIN IMMEDIATE")
                store.busy_timeout_ms = 0
                with mock.patch.object(store, "_sleep_lock_backoff") as retry:
                    with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                        with store._writer_scope():
                            self.fail("contended transaction body must not run")
                self.assertEqual(retry.call_count, 4)
                self.assertIsNone(getattr(store._completion_connection, "connection", None))
                blocker.rollback()
                with store._writer_scope():
                    pass
            finally:
                blocker.close()

    def test_writer_does_not_retry_body_and_rolls_back_nested_writes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("rollback")
            with mock.patch.object(store, "_sleep_lock_backoff") as retry:
                with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                    with store._writer_scope():
                        with store._writer_scope():
                            store.emit(job.id, "test.rollback", {})
                        raise sqlite3.OperationalError("database is locked")
            retry.assert_not_called()
            self.assertIsNone(store._completion_connection.connection)
            self.assertFalse(any(e["event"] == "test.rollback" for e in store.read_events(job.id)))

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


class WorkerHeartbeatLifecycleTests(unittest.TestCase):
    def test_background_heartbeat_prefers_opportunistic_writer(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("background heartbeat")
            task = Task(job_id=job.id, role="implement", instruction="noop")
            store.save_task(task)
            claimed = store.claim_task(task.id, "w-1")
            run = AgentRun(job_id=job.id, task_id=task.id, role=task.role, worker_id="w-1")
            runtime = WorkerRuntime(store, job.id, "implement", "w-1",
                                    heartbeat_seconds=0.01)
            stop = threading.Event()

            def heartbeat(*args, **kwargs):
                stop.set()
                return run, claimed

            with mock.patch.object(
                SQLiteSwarmStore,
                "heartbeat_run_and_renew_lease_opportunistic",
                autospec=True,
                side_effect=heartbeat,
            ) as opportunistic, mock.patch.object(
                SQLiteSwarmStore,
                "heartbeat_run_and_renew_lease",
                autospec=True,
            ) as blocking:
                runtime._heartbeat_until_stopped(run, task.id, stop,
                                                 lease_id=claimed.lease_id)
            opportunistic.assert_called_once()
            blocking.assert_not_called()

    def test_completion_waits_for_inflight_sqlite_heartbeat(self) -> None:
        self._assert_heartbeat_drained_before_terminal(exception=False)

    def test_exception_waits_for_inflight_sqlite_heartbeat(self) -> None:
        self._assert_heartbeat_drained_before_terminal(exception=True)

    def test_lease_loss_during_exception_drain_skips_publication(self) -> None:
        self._assert_heartbeat_drained_before_terminal(exception=True, lose_lease=True)

    def test_lease_loss_during_completion_drain_skips_publication(self) -> None:
        self._assert_heartbeat_drained_before_terminal(exception=False, lose_lease=True)

    def _assert_heartbeat_drained_before_terminal(
        self, *, exception: bool, lose_lease: bool = False
    ) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("heartbeat shutdown")
            store.save_task(Task(job_id=job.id, role="implement", instruction="noop",
                                 adapter="local", payload={"skip_preflight": True}))
            runtime = WorkerRuntime(store, job.id, "implement", "w-1",
                                    lease_seconds=30, heartbeat_seconds=0.01)
            entered = threading.Event()
            release = threading.Event()
            request_lock = threading.Event()
            locked = threading.Event()
            heartbeat_finished = threading.Event()
            publishing = threading.Event()
            joining = threading.Event()
            join_timeouts = []
            errors = []
            original_heartbeat = runtime._heartbeat_run_and_lease
            original_complete = store.complete_task
            original_save = store.save_run
            original_join = threading.Thread.join

            def join(target, timeout=None):
                if target is not blocker and target is not thread:
                    join_timeouts.append(timeout)
                    joining.set()
                return original_join(target, timeout)

            def assert_drained():
                publishing.set()
                if not heartbeat_finished.is_set():
                    raise AssertionError("terminal publication preceded heartbeat drain")

            def save_run(run):
                if run.status == TaskStatus.FAILED:
                    assert_drained()
                return original_save(run)

            def block_writer():
                try:
                    if not request_lock.wait(5):
                        raise AssertionError("worker did not request contention")
                    with store._session() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        locked.set()
                        if not release.wait(5):
                            raise AssertionError("writer release was not signaled")
                except BaseException as exc:
                    errors.append(exc)

            def delayed_heartbeat(*args):
                try:
                    if not locked.wait(5):
                        raise AssertionError("writer did not acquire lock")
                    # The heartbeat now reserves the writer before reading the
                    # task, so it cannot reach renewal until this lock releases.
                    entered.set()
                    # Force the blocking path: this lifecycle test proves that
                    # an actually in-flight write drains before publication.
                    updated, renewed = original_heartbeat(*args[:3])
                    return updated, None if lose_lease else renewed
                finally:
                    heartbeat_finished.set()

            def local_work(*args, **kwargs):
                request_lock.set()
                if not entered.wait(5):
                    raise AssertionError("heartbeat did not start")
                if exception:
                    raise RuntimeError("injected worker failure")
                return AgentRun(job_id=job.id, task_id=args[0].id,
                                role="implement", worker_id="w-1", status=TaskStatus.COMPLETE), []

            def complete(*args, **kwargs):
                assert_drained()
                return original_complete(*args, **kwargs)

            def run():
                try:
                    runtime.run_once()
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch.object(runtime, "_heartbeat_run_and_lease", delayed_heartbeat), \
                 mock.patch("puppetmaster.worker_runtime.LocalWorker.run", side_effect=local_work), \
                 mock.patch.object(store, "complete_task", side_effect=complete), \
                 mock.patch.object(store, "save_run", side_effect=save_run), \
                 mock.patch.object(threading.Thread, "join", join):
                blocker = threading.Thread(target=block_writer)
                thread = threading.Thread(target=run)
                blocker.start()
                thread.start()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertTrue(joining.wait(5), "runtime did not reach heartbeat shutdown")
                    # The lock is held until shutdown reaches join. A timed join
                    # cannot establish drain ordering, regardless of scheduling.
                    self.assertEqual(join_timeouts, [None])
                    self.assertFalse(publishing.is_set(),
                                     "terminal publication raced an outstanding heartbeat")
                finally:
                    release.set()
                    blocker.join(10)
                    thread.join(10)
                    self.assertTrue(heartbeat_finished.wait(5))
                self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertFalse(blocker.is_alive())
            self.assertEqual(publishing.is_set(), not lose_lease)
            self.assertEqual(runtime._lease_lost.is_set(), lose_lease)
            expected = (TaskStatus.RUNNING if lose_lease else
                        TaskStatus.FAILED if exception else TaskStatus.COMPLETE)
            self.assertEqual(store.list_tasks(job.id)[0].status, expected)
            with store._session() as connection:
                rows = connection.execute("SELECT data FROM runs WHERE job_id = ?", (job.id,)).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]["data"])["status"], expected.value)

    def test_previous_lease_loss_does_not_abandon_next_claim(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("next lease")
            store.save_task(Task(job_id=job.id, role="implement", instruction="noop",
                                 adapter="local", payload={"skip_preflight": True}))
            runtime = WorkerRuntime(store, job.id, "implement", "w-1")
            runtime._lease_lost.set()
            self.assertTrue(runtime.run_once())
            self.assertEqual(store.list_tasks(job.id)[0].status, TaskStatus.COMPLETE)


class SqliteMultiprocessAttachTests(unittest.TestCase):
    def test_supervisor_ensure_then_n_workers_attach_claim_complete(self) -> None:
        # This proves attach/claim/complete correctness under contention, not
        # capacity. 32 processes on a 4-vCPU Windows runner hit the SQLite WAL
        # teardown cycle that the readonly design depends on (stalls, rare
        # native crashes in SQLite <=3.49, disk I/O errors in 3.50) in a few
        # percent of runs; real swarms run 3-8 workers. 16 still contends hard.
        worker_count = 16 if sys.platform == "win32" else 32
        task_count = worker_count * 3
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            supervisor = SQLiteSwarmStore(root)
            supervisor.ensure_schema()
            job = supervisor.create_job("local attach stress")
            for index in range(task_count):
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
                diagnostic_path = error_dir / f"w-{index}.dump.txt"
                process = ctx.Process(
                    target=_attach_claim_complete_worker,
                    args=(
                        str(root),
                        job.id,
                        f"w-{index}",
                        str(error_path),
                        str(diagnostic_path),
                    ),
                )
                processes.append((process, error_path, diagnostic_path))
                process.start()

            errors: list[str] = []
            deadline = time.monotonic() + (120 if sys.platform == "win32" else 60)
            for process, error_path, _ in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            timed_out = [(process, error_path, diagnostic_path)
                         for process, error_path, diagnostic_path in processes
                         if process.is_alive()]
            for process, _, _ in timed_out:
                process.terminate()
            for process, error_path, diagnostic_path in timed_out:
                process.join(timeout=5)
                errors.append(f"timeout:{error_path.name}")
                if diagnostic_path.is_file():
                    errors.append(diagnostic_path.read_text(encoding="utf-8"))
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for process, error_path, _ in processes:
                if process.exitcode not in (0, None):
                    errors.append(f"exit:{error_path.name}={process.exitcode}")
                if error_path.is_file():
                    errors.append(error_path.read_text(encoding="utf-8"))
                if not process.is_alive():
                    process.close()

            if errors:
                self.fail("\n\n".join(errors))
            tasks = supervisor.list_tasks(job.id)
            status_counts = Counter(str(task.status) for task in tasks)
            failed_events = [e["payload"] for e in supervisor.read_events(job.id)
                             if e["event"] == "worker.failed_task"]
            diagnostics = {"statuses": dict(status_counts), "failed_events": failed_events}
            self.assertEqual(len(tasks), task_count, diagnostics)
            complete = sum(task.status == TaskStatus.COMPLETE for task in tasks)
            failed = sum(task.status == TaskStatus.FAILED for task in tasks)
            self.assertEqual(complete, task_count, diagnostics)
            self.assertEqual(failed, 0, diagnostics)
            self.assertTrue(all(task.attempts == 1 for task in tasks))
            completed_events = Counter(e["payload"]["task_id"]
                                       for e in supervisor.read_events(job.id)
                                       if e["event"] == "worker.completed_task")
            self.assertEqual(completed_events, Counter({task.id: 1 for task in tasks}))
            self.assertEqual(len(supervisor.list_artifacts(job.id)), task_count)
            with supervisor._session() as connection:
                rows = connection.execute("SELECT role, status, data FROM tasks").fetchall()
                pending = connection.execute(
                    "SELECT COUNT(*) FROM completions WHERE json_extract(data, '$.done') = 0"
                ).fetchone()[0]
            self.assertEqual(pending, 0)
            self.assertEqual(len(rows), task_count)
            for row in rows:
                data = json.loads(row["data"])
                self.assertEqual(row["status"], TaskStatus.COMPLETE.value)
                self.assertEqual(row["status"], data["status"])
                self.assertEqual(row["role"], data["role"])

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
            self.assertEqual(int(row[0]), 7)


if __name__ == "__main__":
    unittest.main()
