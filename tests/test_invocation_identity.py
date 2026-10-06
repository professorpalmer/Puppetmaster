"""Each invocation links its own captures and verification to its ledger row."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401

from puppetmaster.adapters import CliInvocation, CliWorkerAdapter
from puppetmaster.adapters._streaming import run_streamed_subprocess
from puppetmaster.invocation import invoke_cli
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.worker_runtime import WorkerRuntime


class EchoAdapter(CliWorkerAdapter):
    """Launches one real process per invocation; ``calls`` scripts their exits."""

    name = "codex"

    def __init__(self, calls):
        self.calls = list(calls)

    def run(self, task, goal, worker_id):
        artifacts = []
        while self.calls:
            words, code = self.calls.pop(0)
            completed = invoke_cli(self._launch, task=task, words=words, code=code,
                                   accounting_adapter=self.name)
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.VERIFICATION,
                created_by="test", confidence=1.0, evidence=["test:echo"],
                payload={"adapter": self.name, "check": "echo",
                         "result": "passed" if code == 0 else "failed",
                         "live_log": completed.live_log_path,
                         "attempt_id": completed.attempt_id}))
        return artifacts

    def _launch(self, *, task, words, code):
        return run_streamed_subprocess(
            command=[sys.executable, "-c", f"import sys; print({words!r}); sys.exit({code})"],
            env=None, task=task, sidecar_name="echo", timeout_seconds=30)

    def _resolve_cli_executable(self, task):
        return "test", "test"

    def _prepare_cli_invocation(self, *args):
        return CliInvocation(command=["test"], sidecar_name="echo")


class InvocationIdentityTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "state"
        self.store = SQLiteSwarmStore(self.root)
        self.store.init()
        self.job = self.store.create_job("invocation identity")
        self.task = Task(job_id=self.job.id, role="explore", instruction="echo",
                         adapter="local", payload={"reuse_artifacts": False})
        self.store.save_task(self.task)
        patcher = mock.patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_task(self, adapter):
        runtime = WorkerRuntime(self.store, self.job.id, self.task.role, "worker",
                                lease_seconds=30)
        with mock.patch("puppetmaster.workers.get_adapter", return_value=adapter), \
                mock.patch("puppetmaster.adapters.git_snapshot", return_value={}):
            runtime.run_once()

    def verifications(self):
        return [a.payload for a in self.store.list_artifacts(self.job.id)
                if a.type == ArtifactType.VERIFICATION and a.payload.get("check") == "echo"]

    def captured(self, payloads):
        """Each capture's first line, in the order the invocations ran."""
        return sorted((Path(p["live_log"]).read_text().splitlines()[0] for p in payloads),
                      key=lambda text: text not in ("first", "broken"))

    def assert_linked(self, payloads):
        attempts = {a.attempt_id: a for a in SQLiteSwarmStore(self.root).list_attempts(self.job.id)}
        logs = [p["live_log"] for p in payloads]
        self.assertEqual(len(set(logs)), len(logs), "a later invocation overwrote a capture")
        for payload in payloads:
            attempt = attempts[payload["attempt_id"]]
            self.assertEqual(attempt.task_id, self.task.id)
            self.assertIn(attempt.attempt_id.rsplit(":", 1)[-1], payload["live_log"])
        return attempts

    def test_two_invocations_in_one_run_keep_their_own_captures(self):
        self.run_task(EchoAdapter([("first", 0), ("second", 0)]))
        payloads = self.verifications()
        self.assertEqual(len(payloads), 2)
        attempts = self.assert_linked(payloads)
        first, second = (attempts[p["attempt_id"]] for p in payloads)
        self.assertEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.attempt_id, second.attempt_id)
        self.assertEqual(self.captured(payloads), ["first", "second"])

    def test_failed_invocation_keeps_its_evidence_beside_the_success(self):
        self.run_task(EchoAdapter([("broken", 3), ("fixed", 0)]))
        payloads = self.verifications()
        self.assert_linked(payloads)
        self.assertEqual(self.captured(payloads), ["broken", "fixed"])
        usage = SQLiteSwarmStore(self.root).list_usage_observations(self.job.id)
        exits = {o.attempt_id: o.returncode for o in usage if o.observation_id == "process:exit"}
        self.assertEqual(sorted(exits.values()), [0, 3])

    def test_reset_generation_starts_a_new_run_and_keeps_the_old_capture(self):
        self.run_task(EchoAdapter([("generation one", 0)]))
        before = self.verifications()
        self.store.reset_subgraph(self.job.id, [self.task.id])
        self.run_task(EchoAdapter([("generation two", 0)]))
        attempts = SQLiteSwarmStore(self.root).list_attempts(self.job.id)
        self.assertEqual(len({a.run_id for a in attempts}), 2)
        self.assertIn("generation one", Path(before[0]["live_log"]).read_text())
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)

    def test_unbound_calls_keep_the_task_directory(self):
        completed = run_streamed_subprocess(
            command=[sys.executable, "-c", "print('direct')"], env=None, task=self.task,
            sidecar_name="echo", timeout_seconds=30)
        self.assertIsNone(completed.attempt_id)
        self.assertEqual(Path(completed.live_log_path).parent.name, self.task.id)


if __name__ == "__main__":
    unittest.main()
