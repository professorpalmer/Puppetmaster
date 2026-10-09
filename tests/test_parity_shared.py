"""Shared adapter parity: quality write contract, shell receipts, OpenAI receipts."""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.models import Artifact, ArtifactType, Task  # noqa: E402
from puppetmaster.quality import assess_run_quality  # noqa: E402


def art(type_, **payload):
    return Artifact(job_id="j", task_id="t", type=type_, created_by="w", payload=payload,
                    confidence=1.0, evidence=[])


class WriteCapableReceiptTests(unittest.TestCase):
    """F42: a receipt that stamps write_capable is write-capable for quality."""

    def test_write_capable_run_that_changed_nothing_is_degraded(self):
        # An agentic edit worker has no permission_mode or sandbox key.
        arts = [art(ArtifactType.VERIFICATION, result="passed", adapter="agentic",
                    write_capable=True, worker_diff_present=False),
                art(ArtifactType.VERIFICATION, kind="worker_verdict", verdict="PASS", reason="done"),
                art(ArtifactType.FINDING, claim="I can't proceed")]
        verdict = assess_run_quality(arts)
        self.assertEqual(verdict["quality"], "degraded")
        self.assertIn("changed nothing", verdict["reasons"][0])

    def test_write_capable_run_without_a_verdict_is_unverified(self):
        arts = [art(ArtifactType.VERIFICATION, adapter="cursor", write_capable=True,
                    worker_diff_present=True),
                art(ArtifactType.FINDING, claim="Committed.")]
        verdict = assess_run_quality(arts)
        self.assertEqual(verdict["quality"], "degraded")
        self.assertIn("did not report whether it finished", verdict["reasons"][0])

    def test_unknown_mode_without_the_key_stays_ok(self):
        for flag in (False, None, "true"):
            with self.subTest(flag=flag):
                arts = [art(ArtifactType.VERIFICATION, adapter="hermes", write_capable=flag,
                            worker_diff_present=False),
                        art(ArtifactType.FINDING, claim="audit finding")]
                self.assertEqual(assess_run_quality(arts)["quality"], "ok")


class ShellAdapterReceiptTests(unittest.TestCase):
    def _task(self, command):
        return Task(job_id="job", role="verify", instruction="run", adapter="shell",
                    payload={"command": command, "timeout_seconds": 5})

    def test_timeout_receipt_names_its_failure(self):
        # F27: failure classification saw an unclassified failure.
        from puppetmaster.adapters.local import ShellAdapter

        expired = subprocess.TimeoutExpired(["slow"], 5, output=b"partial", stderr=None)
        with mock.patch("puppetmaster.adapters.local.facade",
                        return_value=mock.Mock(run=mock.Mock(side_effect=expired))):
            (receipt,) = ShellAdapter().run(self._task(["slow"]), "goal", "worker")
        self.assertEqual(receipt.payload["result"], "failed")
        self.assertEqual(receipt.payload["failure"], "timeout")

    def test_output_that_is_not_utf8_does_not_crash_the_run(self):
        # F6 (UTF-8 part): a child that writes bytes the locale cannot decode.
        from puppetmaster.adapters.local import ShellAdapter

        command = [sys.executable, "-c",
                   "import sys; sys.stdout.buffer.write(b'caf\\xe9 \\xff ok\\n')"]
        (receipt,) = ShellAdapter().run(self._task(command), "goal", "worker")
        self.assertEqual(receipt.payload["result"], "passed")
        self.assertIn("ok", receipt.payload["stdout"])
        self.assertIn("�", receipt.payload["stdout"])


def _response(body: dict):
    raw = io.BytesIO(json.dumps(body).encode("utf-8"))
    response = mock.MagicMock()
    response.getcode.return_value = 200
    response.read.side_effect = raw.read
    response.__enter__.return_value = response
    return response


_CONTENT = json.dumps({"artifacts": [{"type": "finding", "claim": "x", "evidence": ["e"]}]})


class OpenAIUsageReceiptTests(unittest.TestCase):
    """F33: the receipt keeps cache reads and does not invent a measured zero."""

    def _run(self, body: dict):
        from puppetmaster.adapters.openai import OpenAIAdapter

        task = Task(job_id="job", role="explore", instruction="look", adapter="openai",
                    payload={"openai_api_key": "sk-test", "disable_codegraph": True})
        with mock.patch("puppetmaster.adapters.openai.urllib.request.urlopen",
                        return_value=_response(body)):
            artifacts = OpenAIAdapter().run(task, "goal", "worker")
        return next(a.payload for a in artifacts if a.type == ArtifactType.VERIFICATION)

    def test_cached_prompt_tokens_reach_the_receipt(self):
        payload = self._run({
            "choices": [{"message": {"content": _CONTENT}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100000, "completion_tokens": 500,
                      "prompt_tokens_details": {"cached_tokens": 90000}},
        })
        self.assertEqual(payload["tokens_in"], 100000)
        self.assertEqual(payload["tokens_out"], 500)
        self.assertEqual(payload["tokens_cached"], 90000)

        from puppetmaster.usage import select_usage_records

        record = select_usage_records([art(ArtifactType.VERIFICATION, **payload)])["t"]
        self.assertEqual(record["tokens_cached"], 90000)
        self.assertEqual((record["usage_unknown"], record["usage_invalid"]), ([], []))

    def test_missing_usage_is_unknown_not_zero(self):
        payload = self._run({
            "choices": [{"message": {"content": _CONTENT}, "finish_reason": "stop"}],
        })
        self.assertIsNone(payload["tokens_in"])
        self.assertIsNone(payload["tokens_out"])
        self.assertNotIn("tokens_cached", payload)

        from puppetmaster.usage import select_usage_records

        record = select_usage_records([art(ArtifactType.VERIFICATION, **payload)])["t"]
        self.assertEqual(record["usage_unknown"], ["tokens_in", "tokens_out"])

    def test_response_without_cache_details_keeps_pricing_known(self):
        payload = self._run({
            "choices": [{"message": {"content": _CONTENT}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
        self.assertNotIn("tokens_cached", payload)
        self.assertEqual((payload["tokens_in"], payload["tokens_out"]), (10, 2))


class OpenAIAttemptReceiptTests(unittest.TestCase):
    """F28: each OpenAI receipt names the ledger attempt it belongs to."""

    def setUp(self):
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "state"
        self.store = SQLiteSwarmStore(self.root)
        self.store.init()
        self.job = self.store.create_job("openai attempt")
        self.task = Task(job_id=self.job.id, role="explore", instruction="look",
                         adapter="openai", payload={"disable_codegraph": True,
                                                    "reuse_artifacts": False})
        self.store.save_task(self.task)
        patcher = mock.patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": str(self.root),
                                               "OPENAI_API_KEY": "sk-test"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, urlopen):
        from puppetmaster.adapters.openai import OpenAIAdapter
        from puppetmaster.sqlite_store import SQLiteSwarmStore
        from puppetmaster.worker_runtime import WorkerRuntime

        runtime = WorkerRuntime(self.store, self.job.id, self.task.role, "worker", lease_seconds=30)
        with mock.patch("puppetmaster.workers.get_adapter", return_value=OpenAIAdapter()), \
                mock.patch("puppetmaster.adapters.git_snapshot", return_value={}), \
                mock.patch("puppetmaster.adapters.openai.urllib.request.urlopen", urlopen):
            runtime.run_once()
        receipts = [a.payload for a in self.store.list_artifacts(self.job.id)
                    if a.type == ArtifactType.VERIFICATION and a.payload.get("adapter") == "openai"]
        attempts = {a.attempt_id for a in SQLiteSwarmStore(self.root).list_attempts(self.job.id)}
        return receipts, attempts

    def test_timeout_receipt_names_its_attempt(self):
        receipts, attempts = self._run(mock.Mock(side_effect=socket.timeout("slow")))
        self.assertTrue(receipts)
        for payload in receipts:
            self.assertEqual(payload["failure"], "timeout")
            self.assertIn(payload.get("attempt_id"), attempts)

    def test_completed_receipt_names_its_attempt(self):
        body = {"choices": [{"message": {"content": _CONTENT}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        receipts, attempts = self._run(mock.Mock(return_value=_response(body)))
        self.assertTrue(receipts)
        for payload in receipts:
            self.assertIn(payload.get("attempt_id"), attempts)


if __name__ == "__main__":
    unittest.main()
