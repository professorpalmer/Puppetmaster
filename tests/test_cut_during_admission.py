"""A flow stop must reach a worker that still queues for edit admission.

Codex 1.36.0 benchmark: a repair worker queued behind another job's claim.
The stop saved the cut marker, but the cancellation request failed once and
each retry returned early on the marker. Also, the admission wait ran outside
the cancellation scope, so it could not see a task cut. The worker got
admission after the stop and edited files for five minutes.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.edit_admission import EditAdmissionTimeout, edit_admission
from puppetmaster.models import Task
from puppetmaster.sqlite_store import SQLiteSwarmStore


class CutDuringAdmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = SQLiteSwarmStore(self.root / "state")
        self.job = self.store.create_job("repair")

    def claimed(self, role: str) -> Task:
        task = Task(job_id=self.job.id, role=role, instruction="x", adapter="codex",
                    payload={"cwd": str(self.workspace), "sandbox": "workspace-write"})
        self.store.save_task(task)
        return self.store.claim_task(task.id, f"worker-{role}", lease_seconds=60)

    def test_a_retry_finishes_a_cut_whose_request_failed(self):
        task = self.claimed("repair")
        with patch.object(self.store, "request_cancellation",
                          side_effect=sqlite3.OperationalError("database is locked")):
            with self.assertRaises(sqlite3.OperationalError):
                self.store.cut_task(self.job.id, task.id)
        # The marker is saved, but no cancellation exists yet.
        self.assertIn("failure_cut", self.store.get_task_by_id(task.id).payload)
        ref = self.store.job_ref(self.job.id)
        from puppetmaster.store_contracts import task_binding
        self.assertFalse(self.store.cancellation_pending(ref, task_binding(task)))

        result = self.store.cut_task(self.job.id, task.id)
        self.assertEqual(result["cancellation_outcome"], "requested")
        self.assertTrue(self.store.cancellation_pending(ref, task_binding(task)))
        # A third call changes nothing.
        self.assertEqual(self.store.cut_task(self.job.id, task.id)["request_id"], result["request_id"])

    def test_a_cut_stops_a_worker_that_waits_for_admission(self):
        db = self.root / "claims.sqlite3"
        with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
            holder = edit_admission(self.store, self.claimed("holder"), "holder")
            self.addCleanup(holder.close)
            waiter = self.claimed("repair")
            waiter.payload["edit_admission_wait_seconds"] = 30
            outcome = {}

            def wait():
                started = time.monotonic()
                try:
                    with edit_admission(self.store, waiter, "worker-repair"):
                        outcome["admitted"] = True
                except EditAdmissionTimeout as exc:
                    outcome["error"] = str(exc)
                    outcome["type"] = type(exc).__name__
                outcome["seconds"] = time.monotonic() - started

            thread = threading.Thread(target=wait)
            thread.start()
            time.sleep(0.3)
            self.store.cut_task(self.job.id, waiter.id)
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("admitted", outcome)
        self.assertIn("cancelled while waiting", outcome["error"])
        self.assertEqual(outcome["type"], "EditAdmissionCancelled")
        self.assertLess(outcome["seconds"], 5)

    def test_a_long_wait_reports_the_holder_not_each_poll(self):
        db = self.root / "claims.sqlite3"
        with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
            holder = edit_admission(self.store, self.claimed("holder"), "holder")
            threading.Timer(1.0, holder.close).start()
            waiter = self.claimed("repair")
            waiter.payload["edit_admission_wait_seconds"] = 10
            with edit_admission(self.store, waiter, "worker-repair") as owner:
                self.assertGreaterEqual(owner.waited_seconds, 0.9)
        waiting = [e for e in self.store.read_events(self.job.id) if e.get("event") == "edit_admission.waiting"]
        # About 20 polls in a second; one holder gives one event.
        self.assertEqual(len(waiting), 1)


class StopSettlesTests(unittest.TestCase):
    """A stopped run reports its open work; wait returns only when it settles."""

    def test_wait_recuts_and_settles_what_a_stop_left_open(self):
        from puppetmaster import flow
        from puppetmaster.cli.commands_flow import flow_action

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            store = SQLiteSwarmStore(state)
            job = store.create_job("build")
            queued = Task(job_id=job.id, role="build", instruction="x", adapter="codex", payload={})
            store.save_task(queued)
            graph = {"id": "g", "entry": "build", "defaults": {"adapter": "codex"},
                     "nodes": [{"id": "build", "kind": "agent", "task": "x"}], "edges": []}
            run = flow.new_run(state, graph, cwd=tmp)
            run.inflight = {"node": "build", "visit": 1, "attempt": 0, "job_ids": [job.id]}
            flow.save_run(state, run)
            # The stop's own cut fails on a busy store, as in the benchmark.
            with patch.object(SQLiteSwarmStore, "request_cancellation",
                              side_effect=sqlite3.OperationalError("database is locked")):
                body, _ = flow_action(state, "stop", {"run_id": run.run_id})
            self.assertEqual(body["status"], "stopped")
            self.assertFalse(body["settled"])
            self.assertEqual(body["open_work"][0]["task_id"], queued.id)
            self.assertIn("cooperative", body["note"])

            body, _ = flow_action(state, "wait", {"run_id": run.run_id, "timeout": 10})
            self.assertTrue(body["settled"])
            self.assertNotIn("open_work", body)
            self.assertEqual(store.get_task_by_id(queued.id).status.value, "skipped")


if __name__ == "__main__":
    unittest.main()
