"""Tests for the Antigravity CLI (agy) worker adapter."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
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
    antigravity_stdin_data,
    build_antigravity_command,
    resolve_antigravity_mode,
    resolve_antigravity_model,
)
from puppetmaster.failure import (
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
        self.assertEqual(cmd[cmd.index("--input-format") + 1], "stream-json")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")
        self.assertIn("--mode", cmd)
        self.assertIn("accept-edits", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--disable-slash-commands", cmd)
        self.assertNotIn("--add-dir", cmd)
        self.assertTrue(all(not part.startswith("-p=") and part != "-p" for part in cmd))
        self.assertNotIn("--print", cmd)

    def test_build_antigravity_command_plan_mode(self) -> None:
        cmd = build_antigravity_command(
            prompt="read only plan",
            mode="plan",
            model="gemini-3.7-flash",
            effort="medium",
            cwd=Path("/tmp/work"),
            timeout_seconds=120,
            dangerously_skip_permissions=True,
        )
        self.assertIn("--mode", cmd)
        self.assertIn("plan", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("gemini-3.7-flash", cmd)
        self.assertIn("--effort", cmd)
        self.assertIn("medium", cmd)
        self.assertEqual(cmd[cmd.index("--print-timeout") + 1], "120s")
        self.assertNotIn("--add-dir", cmd)
        self.assertTrue(all(not part.startswith("-p=") for part in cmd))

    def test_skip_permissions_only_when_explicitly_true(self) -> None:
        default = build_antigravity_command(prompt="x")
        self.assertNotIn("--dangerously-skip-permissions", default)
        explicit_false = build_antigravity_command(
            prompt="x", dangerously_skip_permissions=False
        )
        self.assertNotIn("--dangerously-skip-permissions", explicit_false)
        explicit_true = build_antigravity_command(
            prompt="x", dangerously_skip_permissions=True
        )
        self.assertIn("--dangerously-skip-permissions", explicit_true)

    def test_effort_not_double_applied_on_high_slug(self) -> None:
        cmd = build_antigravity_command(
            prompt="x",
            model="gemini-3.7-flash-high",
            effort="high",
        )
        self.assertNotIn("--effort", cmd)
        self.assertIn("gemini-3.7-flash-high", cmd)

        model, effort = resolve_antigravity_model(
            {"model": "gemini-3.7-flash-high", "effort": "low"}
        )
        self.assertEqual(model, "gemini-3.7-flash-high")
        self.assertIsNone(effort)

    def test_resolve_antigravity_model(self) -> None:
        model, effort = resolve_antigravity_model({"model": "gemini-3.7-flash"})
        self.assertEqual(model, "gemini-3.7-flash")
        self.assertEqual(effort, DEFAULT_ANTIGRAVITY_EFFORT)

        model2, effort2 = resolve_antigravity_model(
            {"model": "gemini-3.7-flash", "effort": "low"}
        )
        self.assertEqual(model2, "gemini-3.7-flash")
        self.assertEqual(effort2, "low")

        model3, effort3 = resolve_antigravity_model({})
        self.assertEqual(model3, DEFAULT_ANTIGRAVITY_MODEL)
        self.assertEqual(effort3, DEFAULT_ANTIGRAVITY_EFFORT)

    def test_resolve_antigravity_mode_maps_cli_verbs(self) -> None:
        self.assertEqual(resolve_antigravity_mode({"mode": "implement"}), "accept-edits")
        self.assertEqual(resolve_antigravity_mode({"mode": "analyze"}), "plan")
        self.assertEqual(resolve_antigravity_mode({"read_only": True}), "plan")
        implement_cmd = build_antigravity_command(prompt="x", mode="implement")
        self.assertIn("accept-edits", implement_cmd)
        analyze_cmd = build_antigravity_command(prompt="x", mode="analyze")
        self.assertIn("plan", analyze_cmd)

    def test_extra_args_cannot_inject_print_or_mode(self) -> None:
        cmd = build_antigravity_command(
            prompt="x",
            mode="plan",
            extra_args=[
                "-p",
                "injected prompt",
                "--mode",
                "accept-edits",
                "--dangerously-skip-permissions",
                "--add-dir",
                "/tmp/escape",
                "--verbose",
            ],
        )
        self.assertNotIn("-p", cmd)
        self.assertNotIn("injected prompt", cmd)
        self.assertEqual(cmd.count("--mode"), 1)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "plan")
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertNotIn("--add-dir", cmd)
        self.assertNotIn("/tmp/escape", cmd)
        self.assertIn("--verbose", cmd)

        equals_cmd = build_antigravity_command(
            prompt="x", extra_args=["-p=secret", "--mode=plan"]
        )
        self.assertTrue(all(not part.startswith("-p=") for part in equals_cmd))
        self.assertNotIn("--mode=plan", equals_cmd)


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


class AntigravityOutputParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AntigravityAdapter()

    def test_parse_single_json_envelope(self) -> None:
        envelope = {
            "status": "SUCCESS",
            "response": "ok",
            "conversation_id": "c1",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
        parsed = self.adapter._parse_agy_output(json.dumps(envelope))
        self.assertEqual(parsed["response"], "ok")
        self.assertEqual(parsed["conversation_id"], "c1")

    def test_parse_ndjson_result_event(self) -> None:
        lines = [
            json.dumps({"event": "init", "init": {"cwd": "/tmp"}}),
            json.dumps({"event": "step_update", "step_update": {"state": "DONE"}}),
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "status": "SUCCESS",
                        "response": "from stream",
                        "conversation_id": "c2",
                        "usage": {"input_tokens": 9, "output_tokens": 2},
                    },
                }
            ),
        ]
        parsed = self.adapter._parse_agy_output("\n".join(lines))
        self.assertEqual(parsed["response"], "from stream")
        self.assertEqual(parsed["conversation_id"], "c2")
        self.assertEqual(parsed["usage"]["input_tokens"], 9)

    def test_parse_last_object_when_logs_prefix(self) -> None:
        blob = (
            "warning: ignoring noise\n"
            + json.dumps({"status": "SUCCESS", "response": "trailing"})
        )
        parsed = self.adapter._parse_agy_output(blob)
        self.assertEqual(parsed["response"], "trailing")


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

        kwargs = mock_run.call_args.kwargs
        command = kwargs["command"]
        stdin_data = kwargs["stdin_data"]
        self.assertEqual(command[command.index("--input-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--output-format") + 1], "stream-json")
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertNotIn("--add-dir", command)
        self.assertTrue(all(not part.startswith("-p=") for part in command))
        event = json.loads(stdin_data.strip())
        self.assertEqual(event["event"], "user")
        self.assertIn("Audit security", event["message"]["content"])
        self.assertEqual(stdin_data, antigravity_stdin_data(event["message"]["content"]))

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
    def test_implement_skip_permissions_opt_in_via_subprocess_kwargs(
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
            "diff": "",
        }
        mock_run.return_value = mock.MagicMock(
            returncode=0,
            stdout=json.dumps({"status": "SUCCESS", "response": "ok"}),
            stderr="",
            timed_out=False,
            live_log_path="/tmp/live.log",
        )
        task = Task(
            job_id="job-skip",
            id="task-skip",
            role="implement",
            instruction="Edit files",
            payload={
                "mode": "accept-edits",
                "dangerously_skip_permissions": True,
                "allow_dirty": True,
                "allow_non_worktree": True,
            },
        )
        self.adapter.run(task, goal="Edit files", worker_id="w1")
        kwargs = mock_run.call_args.kwargs
        self.assertIn("--dangerously-skip-permissions", kwargs["command"])
        self.assertIn("stdin_data", kwargs)
        self.assertEqual(json.loads(kwargs["stdin_data"].strip())["event"], "user")

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
            payload={"mode": "accept-edits", "allow_dirty": True},
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
        with mock.patch("shutil.which", return_value="/usr/bin/agy"):
            status = detect_antigravity_billing(env={"GEMINI_API_KEY": "test-key"})
        self.assertTrue(status.healthy)
        self.assertEqual(status.billing, "api")
        self.assertIn("gemini_api_key:set", status.evidence)

    def test_detect_antigravity_billing_session(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            settings = home / ".gemini" / "antigravity-cli"
            settings.mkdir(parents=True)
            (settings / "session.json").write_text("{}", encoding="utf-8")

            status = detect_antigravity_billing(env={}, home=home)

        self.assertTrue(status.healthy)
        self.assertEqual(status.billing, "plan")
        self.assertIn("antigravity_session:present", status.evidence)

    def test_empty_settings_dir_is_unhealthy(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
            with mock.patch("shutil.which", return_value=None):
                status = detect_antigravity_billing(env={}, home=home)

        self.assertFalse(status.healthy)
        self.assertEqual(status.billing, "unknown")
        self.assertNotIn("antigravity_session:present", status.evidence)

    def test_gemini_key_without_agy_is_unhealthy(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch("shutil.which", return_value=None):
                status = detect_antigravity_billing(
                    env={"GEMINI_API_KEY": "test-key"}, home=home
                )

        self.assertFalse(status.healthy)
        self.assertEqual(status.billing, "unknown")
        self.assertIn("gemini_api_key:unbound", status.evidence)

    def test_detect_antigravity_billing_path_only_is_unhealthy(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch("shutil.which", return_value="/usr/bin/agy"):
                status = detect_antigravity_billing(env={}, home=home)

        self.assertFalse(status.healthy)
        self.assertEqual(status.billing, "unknown")
        self.assertIn("antigravity_auth:missing", status.evidence)
        self.assertNotIn("antigravity_session:present", status.evidence)


class AntigravityWiringTests(unittest.TestCase):
    def test_antigravity_is_wired_into_adapter_contracts(self) -> None:
        from puppetmaster.cli.commands_models import _DISCOVER_SOURCE_BY_ADAPTER
        from puppetmaster.model_registry import _CURATED_DISCOVERY_SOURCES
        from puppetmaster.orchestrator import _MODEL_BACKED_ADAPTERS
        from puppetmaster.swarm_launch import SWARM_ANALYSIS_ADAPTERS
        from puppetmaster.workers import (
            IMPLEMENT_ADAPTER_PRIORITY,
            _EDIT_CAPABLE_ADAPTERS,
            _PREFLIGHTABLE_ADAPTERS,
            adapter_is_available,
        )

        self.assertIn("antigravity", SWARM_ANALYSIS_ADAPTERS)
        self.assertIn("antigravity", _MODEL_BACKED_ADAPTERS)
        self.assertNotIn("antigravity", _PREFLIGHTABLE_ADAPTERS)
        self.assertNotIn("hermes", _PREFLIGHTABLE_ADAPTERS)
        self.assertIn("antigravity", IMPLEMENT_ADAPTER_PRIORITY)
        self.assertLess(
            IMPLEMENT_ADAPTER_PRIORITY.index("antigravity"),
            IMPLEMENT_ADAPTER_PRIORITY.index("agentic"),
        )
        self.assertNotIn("antigravity", _EDIT_CAPABLE_ADAPTERS)
        self.assertNotIn("hermes", _EDIT_CAPABLE_ADAPTERS)
        self.assertEqual(_DISCOVER_SOURCE_BY_ADAPTER["antigravity"], "antigravity")
        self.assertIn("antigravity", _CURATED_DISCOVERY_SOURCES)
        with mock.patch(
            "puppetmaster.diagnostics._antigravity_installed", return_value=True
        ):
            self.assertTrue(adapter_is_available("antigravity"))
        with mock.patch(
            "puppetmaster.diagnostics._antigravity_installed", return_value=False
        ):
            self.assertFalse(adapter_is_available("antigravity"))

    def test_mcp_antigravity_command_and_schema_enums(self) -> None:
        from puppetmaster.mcp_server import (
            _implement_command,
            antigravity_command,
            browser_swarm_schema,
            edit_schema,
            implement_schema,
        )

        command = antigravity_command(
            {
                "goal": "ship it",
                "cwd": "/repo",
                "model": "gemini-3.7-flash",
                "effort": "low",
            },
            implement=True,
        )
        self.assertEqual(command[0], "antigravity")
        self.assertEqual(command[command.index("--mode") + 1], "implement")
        self.assertEqual(command[command.index("--model") + 1], "gemini-3.7-flash")
        self.assertEqual(command[command.index("--effort") + 1], "low")
        self.assertEqual(
            _implement_command({"goal": "x", "cwd": "/r"}, "antigravity")[0],
            "antigravity",
        )
        self.assertIn("antigravity", implement_schema()["properties"]["adapter"]["enum"])
        self.assertIn("antigravity", edit_schema()["properties"]["adapter"]["enum"])
        self.assertNotIn(
            "antigravity", browser_swarm_schema()["properties"]["adapter"]["enum"]
        )

    def test_cli_verb_and_agy_alias(self) -> None:
        from puppetmaster.cli._parser import build_parser

        parsed = build_parser().parse_args(
            ["antigravity", "audit auth", "--mode", "analyze", "--effort", "low"]
        )
        self.assertEqual(parsed.command, "antigravity")
        self.assertEqual(parsed.mode, "analyze")
        alias = build_parser().parse_args(["agy", "audit auth"])
        self.assertEqual(alias.command, "agy")

    def test_agy_alias_cannot_bypass_platform_lock(self) -> None:
        from puppetmaster import platform_lock
        from puppetmaster.platform_billing import detect_adapter_billing

        self.assertNotIn("agy", platform_lock.KNOWN_ADAPTERS)
        self.assertEqual(platform_lock.canonicalize_adapter("agy"), "antigravity")
        with TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(platform_lock.ONLY_ENV, None)
            registry = Path(tmp) / "models.json"
            platform_lock.disable({"antigravity"}, registry)
            self.assertFalse(platform_lock.is_adapter_enabled("antigravity", registry))
            self.assertFalse(platform_lock.is_adapter_enabled("agy", registry))
            platform_lock.enable({"agy"}, registry)
            self.assertTrue(platform_lock.is_adapter_enabled("antigravity", registry))
            self.assertTrue(platform_lock.is_adapter_enabled("agy", registry))
        self.assertEqual(detect_adapter_billing("agy").adapter, "antigravity")

    def test_platform_cli_accepts_agy_alias(self) -> None:
        env = os.environ.copy()
        env.pop("PUPPETMASTER_ONLY_ADAPTERS", None)
        with TemporaryDirectory() as tmp:
            registry = Path(tmp) / "models.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "puppetmaster",
                    "platform",
                    "disable",
                    "agy",
                    "--registry-path",
                    str(registry),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=env,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertRegex(completed.stdout, r"\[off\]\s+antigravity")
        self.assertNotIn("[off] agy", completed.stdout)

    def test_gemini_keys_are_redacted(self) -> None:
        from puppetmaster.redaction import redact_secrets

        with mock.patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "gemini-secret-value", "GOOGLE_API_KEY": "google-secret-value"},
        ):
            redacted = redact_secrets(
                "GEMINI_API_KEY=gemini-secret-value GOOGLE_API_KEY=google-secret-value"
            ) or ""
        self.assertNotIn("gemini-secret-value", redacted)
        self.assertNotIn("google-secret-value", redacted)
        self.assertIn("GEMINI_API_KEY", redacted)
        self.assertIn("GOOGLE_API_KEY", redacted)


@unittest.skipUnless(
    shutil.which("agy") is not None
    and os.environ.get("PUPPETMASTER_LIVE_ANTIGRAVITY") == "1",
    "Live test requires agy CLI and PUPPETMASTER_LIVE_ANTIGRAVITY=1",
)
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
