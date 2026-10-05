"""Completion gates run under a live lease, so a slow review cannot be reclaimed mid-verdict."""
from __future__ import annotations

import os
import sys
import time
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.gates import GateEvaluation  # noqa: E402
from puppetmaster.models import Task, TaskStatus  # noqa: E402
from puppetmaster.sqlite_store import SQLiteSwarmStore  # noqa: E402
from puppetmaster.worker_runtime import WorkerRuntime  # noqa: E402


class GateLeaseHeartbeatTests(unittest.TestCase):
    def test_slow_gate_keeps_the_lease_and_completes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            job = store.create_job("gate lease")
            store.save_task(Task(job_id=job.id, role="implement", instruction="noop",
                                 adapter="local", payload={"skip_preflight": True}))
            # Timestamps have one-second precision, so a 1 s lease can read as
            # expired at a second boundary even when renewed; use 2 s.
            runtime = WorkerRuntime(store, job.id, "implement", "w-1",
                                    lease_seconds=2, heartbeat_seconds=0.05)
            recovered = []

            def slow_review(task, artifacts):
                # A live review judge outlasting the 2 s lease, with the
                # orchestrator's stale-lease recovery running mid-review.
                time.sleep(2.6)
                recovered.extend(store.recover_stale_tasks(job.id))
                time.sleep(0.2)
                return GateEvaluation(passed=True, results=[], artifacts=[])

            with patch.object(runtime, "_evaluate_gates", side_effect=slow_review):
                runtime.run_once()

            self.assertEqual(recovered, [])
            self.assertEqual(store.list_tasks(job.id)[0].status, TaskStatus.COMPLETE)


if __name__ == "__main__":
    unittest.main()
