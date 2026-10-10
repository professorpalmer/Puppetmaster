"""A worker that dies attaching to the store is respawned; siblings are not abandoned.

Before: a 2-worker swarm on macOS lost its explore worker at store attach
(exit 1). The supervisor raised on the first non-zero exit, terminated the
review worker mid-task (its task stayed ``running`` on a dead lease), and
marked the job failed with zero findings and no retry.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
import hermetic_env  # noqa: F401

from puppetmaster.identity import StoreMetadataDrift
from puppetmaster.models import JobStatus, TaskStatus
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.store import SwarmStore
from puppetmaster.worker_runtime import WORKER_ATTACH_FAILED_EXIT, main as worker_main


class WorkerAttachRespawnTests(unittest.TestCase):
    def test_attach_failure_exits_tempfail_and_records_startup_error(self) -> None:
        # worker main exports PUPPETMASTER_STATE_DIR; patch.dict restores it.
        with TemporaryDirectory() as tmp, patch.dict(os.environ), patch(
            "puppetmaster.worker_runtime.create_worker_store",
            side_effect=StoreMetadataDrift("store source metadata changed during binding"),
        ):
            code = worker_main(["--state-dir", tmp, "--backend", "file", "--job-id", "job_x",
                                "--role", "explore", "--worker-id", "w-1"])
            self.assertEqual(code, WORKER_ATTACH_FAILED_EXIT)
            log = Path(tmp) / "jobs" / "job_x" / "tasks" / "startup_error-w-1.log"
            self.assertIn("metadata changed", log.read_text())

    def test_locked_file_metadata_init_is_an_attach_failure(self) -> None:
        # Windows CI: a file-store worker built its metadata index lazily at its
        # first lock, after attach, so "database is locked" there exited 1 and
        # failed the job instead of being respawned.
        import sqlite3

        with TemporaryDirectory() as tmp, patch.dict(os.environ):
            store = SwarmStore(Path(tmp))
            store.init()
            job = store.create_job("audit")
            with patch.object(SwarmStore, "init", side_effect=sqlite3.OperationalError("database is locked")):
                code = worker_main(["--state-dir", tmp, "--backend", "file", "--job-id", job.id,
                                    "--role", "explore", "--worker-id", "w-1"])
        self.assertEqual(code, WORKER_ATTACH_FAILED_EXIT)

    def test_lock_create_survives_a_pending_delete_on_windows(self) -> None:
        # Windows refuses O_EXCL create with a sharing violation while the
        # previous owner's unlink is pending; the worker died with PermissionError.
        import puppetmaster.store as store_module

        with TemporaryDirectory() as tmp, patch.object(store_module, "_WINDOWS", True), \
                patch.object(store_module, "_WINDOWS_LOCK_BACKOFF_SECONDS", 0):
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            real_open, refusals = os.open, [2]

            def pending_delete(path, flags, *args):
                if refusals[0]:
                    refusals[0] -= 1
                    raise PermissionError(13, "Permission denied", path)
                return real_open(path, flags, *args)

            with patch.object(store_module.os, "open", pending_delete):
                self.assertTrue(store.acquire_lock("completion_job_x", "w-1", ttl_seconds=300))
            refusals[0] = 10**6
            with patch.object(store_module.os, "open", pending_delete):
                self.assertFalse(store.acquire_lock("other", "w-2", ttl_seconds=300))

    def _swarm(self, failing_spawns: int):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = SwarmStore(Path(tmp.name) / ".puppetmaster")
        real_spawn = Orchestrator._spawn_worker
        failed = {"n": 0}

        def spawn(self, job_id, role, lease_seconds=5, crash_after_claim=False):
            if role == "explore" and failed["n"] < failing_spawns:
                failed["n"] += 1
                return subprocess.Popen([sys.executable, "-c", f"raise SystemExit({WORKER_ATTACH_FAILED_EXIT})"])
            return real_spawn(self, job_id, role, lease_seconds=lease_seconds,
                              crash_after_claim=crash_after_claim)
        return store, spawn

    def test_attach_failure_is_respawned_and_both_roles_complete(self) -> None:
        store, spawn = self._swarm(failing_spawns=1)
        with patch.object(Orchestrator, "_spawn_worker", spawn):
            result = Orchestrator(store).run("two roles", roles=["explore", "review"])
        tasks = store.list_tasks(result.job.id)
        self.assertEqual({t.role: t.status for t in tasks},
                         {"explore": TaskStatus.COMPLETE, "review": TaskStatus.COMPLETE})
        self.assertEqual(store.latest_job().status, JobStatus.COMPLETE)
        respawns = [e for e in store.read_events(result.job.id) if e["event"] == "worker.attach_respawned"]
        self.assertEqual(len(respawns), 1)
        self.assertTrue(store.list_artifacts(result.job.id))

    def test_persistent_attach_failure_fails_without_stranding_the_sibling(self) -> None:
        store, spawn = self._swarm(failing_spawns=99)
        with patch.object(Orchestrator, "_spawn_worker", spawn):
            with self.assertRaises(RuntimeError):
                Orchestrator(store).run("two roles", roles=["explore", "review"])
        job = store.latest_job()
        tasks = {t.role: t for t in store.list_tasks(job.id)}
        # The sibling finished its own task instead of being killed mid-run.
        self.assertEqual(tasks["review"].status, TaskStatus.COMPLETE)
        self.assertNotEqual(tasks["explore"].status, TaskStatus.RUNNING)
        respawns = [e for e in store.read_events(job.id) if e["event"] == "worker.attach_respawned"]
        self.assertEqual(len(respawns), 2)


class IncompleteTasksErrorTests(unittest.TestCase):
    def test_error_names_each_unfinished_task_and_its_recorded_failure(self) -> None:
        # Windows CI ended this suite with only "swarm exited with incomplete
        # tasks"; the worker's failed_task event held the reason and was lost.
        from dataclasses import replace

        from puppetmaster.models import Task

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("two roles")
            done = Task(job_id=job.id, role="explore", instruction="x")
            failed = Task(job_id=job.id, role="review", instruction="y")
            store.save_tasks([replace(done, status=TaskStatus.COMPLETE),
                              replace(failed, status=TaskStatus.FAILED)])
            store.emit(job.id, "worker.failed_task",
                       {"task_id": failed.id, "role": "review", "error": "PermissionError: lock busy"})
            error = Orchestrator(store)._incomplete_tasks_error(job, {done.id, failed.id})
        self.assertEqual(str(error), f"swarm exited with incomplete tasks: review {failed.id} failed "
                                     "(PermissionError: lock busy)")


if __name__ == "__main__":
    unittest.main()
