"""Tests for the Antigravity CLI (agy) worker adapter."""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.adapters import ADAPTERS, get_adapter
from puppetmaster.adapters.antigravity import (
    DEFAULT_ANTIGRAVITY_EFFORT,
    DEFAULT_ANTIGRAVITY_MODEL,
    AntigravityAdapter,
    build_antigravity_command,
    resolve_antigravity_model,
)
from puppetmaster.failure import (
    BILLING_OR_QUOTA,
    MODEL_UNAVAILABLE,
    NOT_AUTHENTICATED,
    classify_antigravity_failure,
)
from puppetmaster.models import ArtifactType, Task
from puppetmaster.platform_billing import detect_antigravity_billing


class AntigravityCommandBuilderTests(unittest.TestCase):
    def test_build_antigravity_command_defaults(self) -> None:
        cmd = build_antigravity_command(prompt="audit the auth flow")
        self.assertEqual(cmd[0], "agy")
        self.assertIn("--output-format", cmd)
        self.assertIn("json", cmd)
        self.assertIn("--mode", cmd)
        self.assertIn("accept-edits", cmd)
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--disable-slash-commands", cmd)
        self.assertEqual(cmd[-1], "-p=audit the auth flow")

    def test_build_antigravity_command_plan_mode(self) -> None:
        cmd = build_antigravity_command(
            prompt="read only plan",
            mode="plan",
            model="gemini-3.7-flash",
            effort="medium",
            cwd=Path("/tmp/work"),
        )
        self.assertIn("--mode", cmd)
        self.assertIn("plan", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("gemini-3.7-flash", cmd)
        self.assertIn("--effort", cmd)
        self.assertIn("medium", cmd)
        self.assertIn("--add-dir", cmd)
        self.assertIn(str(Path("/tmp/work")), cmd)
        self.assertEqual(cmd[-1], "-p=read only plan")

    def test_resolve_antigravity_model(self) -> None:
        model, effort = resolve_antigravity_model({"model": "gemini-3.7-flash"})
        self.assertEqual(model, "gemini-3.7-flash")
        self.assertEqual(effort, DEFAULT_ANTIGRAVITY_EFFORT)

        model2, effort2 = resolve_antigravity_model({"model": "gemini-3.7-flash", "effort": "low"})
        self.assertEqual(model2, "gemini-3.7-flash")
        self.assertEqual(effort2, "low")

        model3, effort3 = resolve_antigravity_model({})
        self.assertEqual(model3, DEFAULT_ANTIGRAVITY_MODEL)
        self.assertEqual(effort3, DEFAULT_ANTIGRAVITY_EFFORT)


class AntigravityFailureClassificationTests(unittest.TestCase):
    def test_classify_failures(self) -> None:
        self.assertEqual(
            classify_antigravity_failure("model gemini-2.5-pro is not recognized as a known model"),
            MODEL_UNAVAILABLE,
        )
        self.assertEqual(
            classify_antigravity_failure("--model gemini-3.7-flash requires --effort"),
            MODEL_UNAVAILABLE,
        )
        self.assertEqual(
            classify_antigravity_failure("Error: not authenticated. Run agy to login to your account"),
            NOT_AUTHENTICATED,
        )
        self.assertEqual(
            classify_antigravity_failure("ResourceExhausted: 429 quota exceeded"),
            "rate_limit",
        )


class AntigravityAdapterLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AntigravityAdapter()

    def test_registry_registration(self) -> None:
        self.assertIn("antigravity", ADAPTERS)
        self.assertIn("agy", ADAPTERS)
        self.assertIsInstance(get_adapter("antigravity"), AntigravityAdapter)
        self.assertIsInstance(get_adapter("agy"), AntigravityAdapter)

    @mock.patch("puppetmaster.adapters.git_snapshot")
    @mock.patch("puppetmaster.adapters.resolve_command")
    @mock.patch("puppetmaster.adapters.run_streamed_subprocess")
    def test_run_success_with_findings(
        self,
        mock_run: mock.MagicMock,
        mock_resolve: mock.MagicMock,
        mock_snapshot: mock.MagicMock,
    ) -> None:
        mock_resolve.return_value = "/usr/local/bin/agy"
        mock_snapshot.return_value = {
            "sha": "abc1234",
            "changed_files": [],
            "untracked_files": [],
            "tree": "tree1",
        }

        raw_output = json.dumps({
            "conversation_id": "test-conv-123",
            "status": "SUCCESS",
            "response": '```json\n[{"type": "finding", "claim": "SQL injection vulnerability in auth.py", "evidence": ["auth.py:42"]}]\n```',
            "duration_seconds": 2.5,
            "num_turns": 1,
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "thinking_tokens": 150,
                "cache_read_tokens": 50,
                "total_tokens": 1200,
            },
        })

        mock_run.return_value = mock.MagicMock(
            returncode=0,
            stdout=raw_output,
            stderr="",
            timed_out=False,
            live_log_path="/tmp/live.log",
        )

        task = Task(
            job_id="job1",
            id="task1",
            role="explore",
            instruction="Audit security",
            payload={"mode": "plan", "read_only": True},
        )
        artifacts = self.adapter.run(task, goal="Audit security", worker_id="w1")

        self.assertTrue(len(artifacts) >= 2)
        verification = artifacts[0]
        self.assertEqual(verification.type, ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["result"], "passed")
        self.assertEqual(verification.payload["tokens_in"], 1000)
        self.assertEqual(verification.payload["tokens_out"], 200)
        self.assertEqual(verification.payload["reasoning_output_tokens"], 150)
        self.assertEqual(verification.payload["cached_input_tokens"], 50)
        self.assertEqual(verification.payload["conversation_id"], "test-conv-123")

        finding = next((a for a in artifacts if a.type == ArtifactType.FINDING), None)
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertIn("SQL injection vulnerability", finding.payload["claim"])

    @mock.patch("puppetmaster.adapters.git_snapshot")
    @mock.patch("puppetmaster.adapters.resolve_command")
    @mock.patch("puppetmaster.adapters.run_streamed_subprocess")
    def test_run_success_with_patch(
        self,
        mock_run: mock.MagicMock,
        mock_resolve: mock.MagicMock,
        mock_snapshot: mock.MagicMock,
    ) -> None:
        mock_resolve.return_value = "/usr/local/bin/agy"
        mock_snapshot.side_effect = [
            {"sha": "abc1234", "changed_files": [], "untracked_files": [], "tree": "tree1", "diff": ""},
            {"sha": "abc1234", "changed_files": ["app.py"], "untracked_files": [], "tree": "tree2", "diff": "diff --git a/app.py b/app.py\n+fixed"},
        ]

        raw_output = json.dumps({
            "conversation_id": "test-conv-456",
            "status": "SUCCESS",
            "response": "Updated app.py to fix the bug.",
            "duration_seconds": 3.0,
            "num_turns": 1,
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "thinking_tokens": 50,
                "cache_read_tokens": 0,
                "total_tokens": 600,
            },
        })

        mock_run.return_value = mock.MagicMock(
            returncode=0,
            stdout=raw_output,
            stderr="",
            timed_out=False,
            live_log_path="/tmp/live.log",
        )

        task = Task(
            job_id="job2",
            id="task2",
            role="implement",
            instruction="Fix app.py bug",
            payload={"mode": "accept-edits"},
        )
        artifacts = self.adapter.run(task, goal="Fix app.py bug", worker_id="w1")

        self.assertTrue(any(a.type == ArtifactType.PATCH for a in artifacts))
        self.assertTrue(any(a.type == ArtifactType.VERIFICATION for a in artifacts))

    @mock.patch("puppetmaster.adapters.git_snapshot")
    @mock.patch("puppetmaster.adapters.resolve_command")
    @mock.patch("puppetmaster.adapters.run_streamed_subprocess")
    def test_run_timeout(
        self,
        mock_run: mock.MagicMock,
        mock_resolve: mock.MagicMock,
        mock_snapshot: mock.MagicMock,
    ) -> None:
        mock_resolve.return_value = "/usr/local/bin/agy"
        mock_snapshot.return_value = {
            "sha": "abc1234",
            "changed_files": [],
            "untracked_files": [],
            "tree": "tree1",
        }

        mock_run.return_value = mock.MagicMock(
            returncode=None,
            stdout="partial progress...",
            stderr="",
            timed_out=True,
            live_log_path="/tmp/live.log",
        )

        task = Task(
            job_id="job3",
            id="task3",
            role="explore",
            instruction="Long running task",
            payload={"mode": "plan", "read_only": True},
        )
        artifacts = self.adapter.run(task, goal="Long running task", worker_id="w1")

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].type, ArtifactType.VERIFICATION)
        self.assertEqual(artifacts[0].payload["result"], "failed")
        self.assertEqual(artifacts[0].payload["failure"], "timeout")


class AntigravityBillingTests(unittest.TestCase):
    def test_detect_antigravity_billing_api_key(self) -> None:
        status = detect_antigravity_billing(env={"GEMINI_API_KEY": "test-key"})
        self.assertTrue(status.healthy)
        self.assertEqual(status.billing, "api")
        self.assertIn("gemini_api_key:set", status.evidence)

    def test_detect_antigravity_billing_session(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".gemini" / "antigravity-cli").mkdir(parents=True)

            status = detect_antigravity_billing(env={}, home=home)

        self.assertTrue(status.healthy)
        self.assertEqual(status.billing, "plan")
        self.assertIn("antigravity_session:present", status.evidence)


@unittest.skipUnless(shutil.which("agy") is not None, "Live test requires agy CLI on PATH")
class AntigravityLiveExecutionTests(unittest.TestCase):
    def test_live_antigravity_gemini_37_flash_run(self) -> None:
        adapter = AntigravityAdapter()
        task = Task(
            job_id="live_job",
            id="live_task_1",
            role="explore",
            instruction="What is 7 * 8? Return a single finding in JSON with claim '56'.",
            payload={
                "model": "gemini-3.7-flash",
                "effort": "low",
                "mode": "plan",
                "read_only": True,
            },
        )
        artifacts = adapter.run(task, goal="Verify math", worker_id="live_worker")
        self.assertTrue(len(artifacts) >= 1)
        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "passed")
        self.assertEqual(verification.payload["model"], "gemini-3.7-flash")
        self.assertGreater(verification.payload["tokens_total"], 0)
        self.assertGreater(verification.payload["tokens_in"], 0)
