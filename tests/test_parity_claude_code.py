"""Claude Code adapter parity with the Codex and fx adapters."""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from puppetmaster import hook_runner
from puppetmaster.adapters import ClaudeCodeAdapter, resolve_claude_code_model
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.claude_code import build_claude_code_command
from puppetmaster.models import ArtifactType, Task

_HOMES = tempfile.TemporaryDirectory()
_ENV = patch.dict(
    "os.environ",
    {
        "CLAUDE_CONFIG_DIR": str(Path(_HOMES.name) / "claude"),
        "PUPPETMASTER_HOME": str(Path(_HOMES.name) / "pm"),
    },
)


def setUpModule() -> None:
    _ENV.start()


def tearDownModule() -> None:
    _ENV.stop()
    _HOMES.cleanup()


CLEAN = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
PRIOR_SESSION = "11111111-2222-3333-4444-555555555555"
NEW_SESSION = "66666666-7777-8888-9999-000000000000"
RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "session_id": NEW_SESSION,
    "result": "Built a.py.\nVERDICT: PASS - checks pass", "total_cost_usd": 0.0412,
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 480, "output_tokens": 5},
}
CLAUDE_RESULT = json.dumps(RESULT)


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def _run(task: Task, stdout: str, *, timed_out: bool = False, returncode: int = 0):
    streamed = StreamedProcess(returncode=None if timed_out else returncode, stdout=stdout,
                               stderr="", timed_out=timed_out)
    with patch("puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"), patch(
        "puppetmaster.adapters.worktree_guard", return_value=None
    ), patch("puppetmaster.adapters.git_snapshot", side_effect=[CLEAN, CLEAN]), patch(
        "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
    ) as run, patch(
        "puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda prompt, **_: (prompt, False)
    ):
        artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")
    return artifacts, (run.call_args.kwargs if run.called else None)


def _task(**payload) -> Task:
    return Task(job_id="job-cc", role="builder", instruction="Build a.py.", adapter="claude-code",
                payload={"cwd": str(Path.cwd()), "allow_dirty": True, **payload})


def _flag(command: list, name: str):
    return command[command.index(name) + 1] if name in command else None


class BedrockPinTests(unittest.TestCase):
    """F16: an exact pin is never replaced by ANTHROPIC_MODEL on Bedrock."""

    SONNET = "us.anthropic.claude-sonnet-4-5-v1:0"
    OPUS = "us.anthropic.claude-opus-4-1-20250805-v1:0"

    def _env(self) -> dict:
        return {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_MODEL": self.SONNET}

    def test_pin_ignores_the_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, _ = resolve_claude_code_model(
                {"model": self.OPUS, "pinned_model": "claude-code/opus"}, env=self._env(), home=Path(tmp))
            self.assertEqual(model, self.OPUS)
            short, note = resolve_claude_code_model(
                {"model": "claude-opus-5-5", "pinned_model": "claude-code/opus-5-5"},
                env=self._env(), home=Path(tmp))
            self.assertIsNone(short)
            self.assertIn("pinned", note)
            # Unpinned work keeps the documented override.
            unpinned, _ = resolve_claude_code_model({"model": "claude-opus-5-5"}, env=self._env(), home=Path(tmp))
            self.assertEqual(unpinned, self.SONNET)

    def test_short_pin_on_bedrock_refuses_the_launch(self) -> None:
        with patch.dict("os.environ", self._env()):
            artifacts, kwargs = _run(
                _task(model="claude-opus-5-5", pinned_model="claude-code/opus-5-5"), CLAUDE_RESULT)
        self.assertIsNone(kwargs)
        self.assertEqual(artifacts[0].payload["failure"], "model_unavailable")
        self.assertIn(artifacts[0].payload["result"], ("failed", "blocked"))


class TimeoutSessionTests(unittest.TestCase):
    """F23: a timed-out run records the session id the adapter chose."""

    def test_cold_run_passes_its_own_session_id(self) -> None:
        artifacts, kwargs = _run(_task(), "", timed_out=True)
        session = _flag(kwargs["command"], "--session-id")
        self.assertEqual(str(uuid.UUID(session)), session)
        self.assertEqual(artifacts[0].payload["failure"], "timeout")
        self.assertEqual(artifacts[0].payload["session_id"], session)

    def test_resumed_run_forks_into_its_own_session_id(self) -> None:
        artifacts, kwargs = _run(_task(resume_session_id=PRIOR_SESSION), "", timed_out=True)
        command = kwargs["command"]
        index = command.index("--resume")
        self.assertEqual(command[index:index + 3], ["--resume", PRIOR_SESSION, "--fork-session"])
        session = _flag(command, "--session-id")
        self.assertNotEqual(session, PRIOR_SESSION)
        self.assertEqual(artifacts[0].payload["session_id"], session)

    def test_command_adds_session_id_only_when_given(self) -> None:
        self.assertNotIn("--session-id", build_claude_code_command(executable="claude"))
        command = build_claude_code_command(executable="claude", session_id=NEW_SESSION)
        self.assertEqual(_flag(command, "--session-id"), NEW_SESSION)


class ProviderCostTests(unittest.TestCase):
    """F36: a cold run stamps the provider cost; a resumed run never does."""

    def test_cold_api_run_stamps_real_cost(self) -> None:
        artifacts, _ = _run(_task(output_format="json", billing="api"), CLAUDE_RESULT)
        self.assertEqual(artifacts[0].payload["real_cost_usd"], 0.0412)

    def test_resumed_run_does_not_stamp_session_cumulative_cost(self) -> None:
        artifacts, _ = _run(
            _task(output_format="json", billing="api", resume_session_id=PRIOR_SESSION), CLAUDE_RESULT)
        self.assertNotIn("real_cost_usd", artifacts[0].payload)

    def test_plan_or_unknown_billing_does_not_stamp_notional_cost(self) -> None:
        for billing in ("plan", None):
            with self.subTest(billing=billing):
                artifacts, _ = _run(_task(output_format="json", billing=billing), CLAUDE_RESULT)
                self.assertNotIn("real_cost_usd", artifacts[0].payload)


class BuildContractTests(unittest.TestCase):
    """F44: a write-capable Claude worker gets the build contract."""

    def test_write_capable_worker_gets_the_build_contract(self) -> None:
        _, kwargs = _run(_task(permission_mode="acceptEdits"), CLAUDE_RESULT)
        prompt = kwargs["stdin_data"]
        self.assertIn("Build mode", prompt)
        self.assertIn("Passing checks are the floor", prompt)
        self.assertEqual(prompt.count("Reporting contract"), 1)
        self.assertIn("Your task:\nBuild a.py.", prompt)

    def test_read_only_worker_keeps_the_report_contract(self) -> None:
        _, kwargs = _run(_task(permission_mode="plan"), CLAUDE_RESULT)
        self.assertNotIn("Build mode", kwargs["stdin_data"])

    def test_receipt_stamps_write_capable(self) -> None:
        built, _ = _run(_task(permission_mode="acceptEdits"), CLAUDE_RESULT)
        self.assertIs(built[0].payload["write_capable"], True)
        timed_out, _ = _run(_task(permission_mode="acceptEdits"), "", timed_out=True)
        self.assertIs(timed_out[0].payload["write_capable"], True)
        read_only, _ = _run(_task(permission_mode="plan"), CLAUDE_RESULT)
        self.assertIs(read_only[0].payload["write_capable"], False)


class StreamAttributionTests(unittest.TestCase):
    """F47: a write-capable run streams tool use so the write_scope gate can attribute."""

    STREAM = _stream(
        {"type": "system", "subtype": "init", "session_id": NEW_SESSION},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "pkg/a.py"}}]}},
        RESULT,
    )

    def test_write_capable_default_is_stream_json(self) -> None:
        artifacts, kwargs = _run(_task(permission_mode="acceptEdits", billing="api"), self.STREAM)
        command = kwargs["command"]
        self.assertEqual(_flag(command, "--output-format"), "stream-json")
        self.assertIn("--verbose", command)
        payload = artifacts[0].payload
        self.assertEqual(payload["worker_referenced_paths"], ["pkg/a.py"])
        self.assertEqual(payload["session_id"], NEW_SESSION)
        self.assertFalse(payload["tokens_estimated"])
        self.assertEqual(payload["real_cost_usd"], 0.0412)
        reports = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertTrue(any("Built a.py." in (a.payload.get("report") or "") for a in reports))

    def test_read_only_and_explicit_formats_are_unchanged(self) -> None:
        _, read_only = _run(_task(permission_mode="plan"), CLAUDE_RESULT)
        self.assertEqual(_flag(read_only["command"], "--output-format"), "json")
        self.assertNotIn("--verbose", read_only["command"])
        _, explicit = _run(_task(permission_mode="acceptEdits", output_format="json"), CLAUDE_RESULT)
        self.assertEqual(_flag(explicit["command"], "--output-format"), "json")


class WorkerHookTests(unittest.TestCase):
    """F50: the pilot's host hooks do nothing inside a Puppetmaster worker."""

    TOOLS_ON = {"PUPPETMASTER_HOOK_ASSUME_TOOLS": "1"}
    BROAD = {"tool_name": "shell", "command": "rg -r TODO ./src"}

    def test_pilot_is_redirected_but_worker_is_not(self) -> None:
        pilot = hook_runner.handle_hook(self.BROAD, host="cursor", event="pre-tool", env=self.TOOLS_ON)
        self.assertEqual(pilot.action, "deny")
        worker_env = {**self.TOOLS_ON, "PUPPETMASTER_WORKER": "1"}
        with patch("puppetmaster.hook_runner.record_decision") as record:
            for payload, event in (
                (self.BROAD, "pre-tool"),
                ({"prompt": "Use Puppetmaster to audit the repo"}, "beforeSubmitPrompt"),
                ({"tool_name": "TodoWrite", "tool_input": {"todos": []}}, "PostToolUse"),
            ):
                response = hook_runner.handle_hook(payload, host="claude", event=event, env=worker_env)
                self.assertEqual(response.action, "allow")
                self.assertFalse(response.context)
        record.assert_not_called()


class ErrorResultTests(unittest.TestCase):
    """F60: an is_error result fails the run even when the CLI exits 0."""

    def test_is_error_result_with_exit_zero_fails(self) -> None:
        error = {"type": "result", "subtype": "error_during_execution", "is_error": True,
                 "session_id": NEW_SESSION, "result": "API Error: 429 rate_limit_error",
                 "usage": {"input_tokens": 1, "output_tokens": 0}}
        artifacts, _ = _run(_task(output_format="json"), json.dumps(error), returncode=0)
        payload = artifacts[0].payload
        self.assertEqual(payload["result"], "failed")
        self.assertEqual(payload["failure"], "rate_limit")


if __name__ == "__main__":
    unittest.main()
