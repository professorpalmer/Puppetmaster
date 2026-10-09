"""Unrouted fx and local tasks record that no effort reached the worker.

Routed fx/local tasks got ``reasoning_effort_source: adapter_unsupported``
and an enforced operator effort refused them (1.33.6). Unrouted ones skipped
the stamp, so they recorded nothing and enforce let them run.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.orchestrator import Orchestrator
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.swarm_reasoning import WorkerEffortError
from puppetmaster.workers import WorkerSpec


class UnroutedNoEffortTests(unittest.TestCase):
    def _task(self, adapter: str):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / "state")
            job = store.create_job("effort")
            return Orchestrator(store)._create_tasks(
                job, [WorkerSpec(role="explore", instruction="x", adapter=adapter, payload={})])[0]

    def test_fx_and_local_record_adapter_unsupported(self):
        for adapter in ("fx", "local"):
            with self.subTest(adapter=adapter):
                task = self._task(adapter)
                self.assertEqual(task.payload["reasoning_effort_source"], "adapter_unsupported")
                self.assertNotIn("reasoning_effort", task.payload)

    def test_enforced_effort_refuses_them(self):
        env = {"PUPPETMASTER_WORKER_EFFORT": "high", "PUPPETMASTER_WORKER_EFFORT_POLICY": "enforce"}
        with patch.dict(os.environ, env):
            with self.assertRaises(WorkerEffortError):
                self._task("fx")


if __name__ == "__main__":
    unittest.main()
