"""A stopped worker keeps a terminal receipt, and its cut settles.

Codex 1.40.0 benchmark (saved run): a flow stop cut two running workers. Both
tasks ended FAILED with ``failure_cut.outcome: pending`` and no artifact. The
stream layer re-raised the cancel and dropped the captured attempt, and no
pass settled the cut after the worker exited.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.models import JobStatus, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.worker_runtime import WorkerRuntime

_FAKE_CLI = """import json, os, sys, time
sys.stdin.read()
print(json.dumps({"type": "system", "session_id": "sess-0a1b2c3d"}), flush=True)
open(os.environ["PM_TEST_READY"], "w").close()
time.sleep(60)
print(json.dumps({"type": "result", "result": "done"}), flush=True)
"""


class StopReceiptTests(unittest.TestCase):
    def test_a_cut_running_worker_records_a_stop_receipt_and_settles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            fake = root / "fake_claude.py"
            fake.write_text(_FAKE_CLI, encoding="utf-8")
            store = SQLiteSwarmStore(root / "state")
            job = store.create_job("stop receipt")
            store.update_job_status(job.id, JobStatus.RUNNING)
            task = Task(job_id=job.id, role="build", instruction="build it", adapter="claude-code",
                        payload={"executable": [sys.executable, str(fake)], "cwd": str(repo),
                                 "timeout_seconds": 120})
            store.save_task(task)

            # worker_runtime exports the state dir; the dispatch receipt lands there.
            ready = root / "ready"
            env = patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": str(root / "state"),
                                          "PM_TEST_READY": str(ready)})
            env.start()
            self.addCleanup(env.stop)
            runtime = WorkerRuntime(store, job.id, "build", "worker-build-1", lease_seconds=30)
            thread = threading.Thread(target=runtime.run_once)
            started = time.monotonic()
            thread.start()
            deadline = time.monotonic() + 20
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            # The reader thread records the line before the child signals ready.
            time.sleep(0.3)
            store.cut_task(job.id, task.id)
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
            self.assertLess(time.monotonic() - started, 30)

            settled = store.get_task_by_id(task.id)
            self.assertEqual(settled.status, TaskStatus.SKIPPED)
            self.assertEqual(settled.payload["failure_cut"]["outcome"], "observed")

            stops = [a for a in store.list_artifacts(job.id)
                     if a.task_id == task.id and (a.payload or {}).get("check") == "cancellation"]
            self.assertEqual(len(stops), 1)
            receipt = stops[0].payload
            self.assertEqual(receipt["result"], "cancelled")
            self.assertFalse(receipt["turn_completed"])
            self.assertFalse(receipt["usage_known"])
            self.assertEqual(receipt["usage_unknown_reason"], "stopped_before_final_usage")
            self.assertIsNone(receipt["tokens_in"])
            self.assertIsNone(receipt["real_cost_usd"])
            self.assertEqual(receipt["session_id"], "sess-0a1b2c3d")
            self.assertTrue(receipt["dispatch_receipt"])
            self.assertTrue(Path(receipt["dispatch_receipt"]).is_file())
            self.assertTrue(receipt["attempt_id"])

    def test_two_session_ids_record_none(self):
        from puppetmaster.adapters._streaming import single_observed_session_id

        self.assertIsNone(single_observed_session_id('{"session_id": "aaaaaa1"} {"thread_id": "bbbbbb2"}'))
        self.assertEqual(single_observed_session_id('{"threadId": "th-123456"}'), "th-123456")
        self.assertIsNone(single_observed_session_id("no ids"))


if __name__ == "__main__":
    unittest.main()
