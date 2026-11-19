"""The lean profile of a Claude Code worker: the project, not the person."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import ClaudeCodeAdapter
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.claude_code import (
    claude_lean_args,
    claude_lean_enabled,
    claude_user_settings_file,
)
from puppetmaster.models import ArtifactType, Task

CLEAN = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
RESULT = json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "session_id": "66666666-7777-8888-9999-000000000000",
    "result": "Built a.py.\nVERDICT: PASS - checks pass",
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 480, "output_tokens": 5},
})


def _home(settings: object = None) -> Path:
    """A fake home directory. ``settings`` is the text of its settings.json, or None for no file."""
    tmp = tempfile.TemporaryDirectory()
    home = Path(tmp.name)
    home_refs.append(tmp)
    if settings is not None:
        (home / ".claude").mkdir()
        (home / ".claude" / "settings.json").write_text(str(settings), encoding="utf-8")
    return home


home_refs: list = []


def tearDownModule() -> None:
    for tmp in home_refs:
        tmp.cleanup()


class LeanArgsTests(unittest.TestCase):
    def test_default_keeps_the_project_and_the_user_settings_file(self) -> None:
        home = _home('{"env": {"CLAUDE_CODE_USE_BEDROCK": "1"}}')
        self.assertEqual(
            claude_lean_args({}, env={}, home=home),
            ["--setting-sources", "project,local",
             "--settings", str(home / ".claude" / "settings.json"),
             "--strict-mcp-config"],
        )

    def test_no_settings_argument_without_a_usable_file(self) -> None:
        expected = ["--setting-sources", "project,local", "--strict-mcp-config"]
        self.assertEqual(claude_lean_args({}, env={}, home=_home()), expected)
        # The CLI stops on a settings file that it cannot read, so a damaged file is not passed.
        self.assertEqual(claude_lean_args({}, env={}, home=_home("{not json")), expected)
        self.assertEqual(claude_lean_args({}, env={}, home=_home("[1, 2]")), expected)

    def test_config_dir_names_the_settings_file(self) -> None:
        other = _home()
        (other / "settings.json").write_text("{}", encoding="utf-8")
        env = {"CLAUDE_CONFIG_DIR": str(other)}
        self.assertEqual(claude_user_settings_file(env, _home('{"a": 1}')), other / "settings.json")
        self.assertIn(str(other / "settings.json"), claude_lean_args({}, env=env, home=_home()))

    def test_environment_turns_the_profile_off(self) -> None:
        home = _home("{}")
        for value in ("0", "false", "OFF", " no "):
            self.assertEqual(claude_lean_args({}, env={"PUPPETMASTER_CLAUDE_LEAN": value}, home=home), [])
        for value in ("1", "true", ""):
            self.assertTrue(claude_lean_enabled({}, {"PUPPETMASTER_CLAUDE_LEAN": value}))

    def test_payload_wins_over_the_environment(self) -> None:
        self.assertFalse(claude_lean_enabled({"lean_profile": False}, {}))
        self.assertTrue(claude_lean_enabled({"lean_profile": True}, {"PUPPETMASTER_CLAUDE_LEAN": "0"}))
        # A payload value of None is no decision.
        self.assertFalse(claude_lean_enabled({"lean_profile": None}, {"PUPPETMASTER_CLAUDE_LEAN": "0"}))

    def test_payload_text_reads_like_the_environment(self) -> None:
        for value in ("false", "0", "Off", " no "):
            self.assertFalse(claude_lean_enabled({"lean_profile": value}, {}))
        self.assertTrue(claude_lean_enabled({"lean_profile": "true"}, {"PUPPETMASTER_CLAUDE_LEAN": "0"}))

    def test_relative_config_dir_is_read_from_the_worker_directory(self) -> None:
        worker = _home()
        (worker / "cfg").mkdir()
        (worker / "cfg" / "settings.json").write_text("{}", encoding="utf-8")
        env = {"CLAUDE_CONFIG_DIR": "cfg"}
        expected = Path(os.path.abspath(worker / "cfg" / "settings.json"))
        found = claude_user_settings_file(env, _home(), worker)
        self.assertEqual(found, expected)
        self.assertTrue(found.is_absolute())
        args = claude_lean_args({"cwd": str(worker)}, env=env, home=_home())
        self.assertEqual(args[args.index("--settings") + 1], str(expected))

    def test_caller_setting_sources_win(self) -> None:
        home = _home("{}")
        settings = str(home / ".claude" / "settings.json")
        # No user source: the settings file still goes in, so sign-in and permissions stay.
        self.assertEqual(
            claude_lean_args({"extra_args": ["--setting-sources", ""]}, env={}, home=home),
            ["--settings", settings, "--strict-mcp-config"],
        )
        self.assertEqual(
            claude_lean_args({"extra_args": ["--setting-sources=project"]}, env={}, home=home),
            ["--settings", settings, "--strict-mcp-config"],
        )
        # The user source is loaded: a second copy of its settings would run each hook twice.
        self.assertEqual(
            claude_lean_args({"extra_args": ["--setting-sources", "user, project"]}, env={}, home=home),
            ["--strict-mcp-config"],
        )

    def test_caller_settings_and_strict_mcp_are_not_repeated(self) -> None:
        home = _home("{}")
        self.assertEqual(
            claude_lean_args({"extra_args": ["--settings", "/x.json", "--strict-mcp-config"]}, env={}, home=home),
            ["--setting-sources", "project,local"],
        )
        self.assertEqual(
            claude_lean_args({"extra_args": "--settings=/x.json"}, env={}, home=home),
            ["--setting-sources", "project,local", "--strict-mcp-config"],
        )


class LeanAdapterTests(unittest.TestCase):
    def _run(self, env: dict, **payload):
        task = Task(job_id="job-lean", role="builder", instruction="Build a.py.", adapter="claude-code",
                    payload={"cwd": str(Path.cwd()), "allow_dirty": True, **payload})
        streamed = StreamedProcess(returncode=0, stdout=RESULT, stderr="", timed_out=False)
        home = _home('{"hooks": {}}')
        base = {"CLAUDE_CONFIG_DIR": str(home / ".claude"), "PUPPETMASTER_HOME": str(home / "pm")}
        with patch.dict("os.environ", {**base, **env}), patch(
            "puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"
        ), patch("puppetmaster.adapters.worktree_guard", return_value=None), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[CLEAN, CLEAN]
        ), patch("puppetmaster.adapters.run_streamed_subprocess", return_value=streamed) as run, patch(
            "puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda prompt, **_: (prompt, False)
        ):
            artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")
        command = run.call_args.args[0] if run.call_args.args else run.call_args.kwargs["command"]
        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        return command, verification, home

    def test_worker_command_has_the_profile_before_the_caller_arguments(self) -> None:
        command, verification, home = self._run({}, extra_args=["--effort", "high"])
        lean = ["--setting-sources", "project,local",
                "--settings", str(home / ".claude" / "settings.json"), "--strict-mcp-config"]
        start = command.index("--setting-sources")
        self.assertEqual(command[start:start + len(lean)], lean)
        self.assertEqual(command[-2:], ["--effort", "high"])
        self.assertIn("profile:lean", verification.evidence)

    def test_profile_off_leaves_the_command_as_before(self) -> None:
        command, verification, _ = self._run({"PUPPETMASTER_CLAUDE_LEAN": "0"}, extra_args=["--effort", "high"])
        for flag in ("--setting-sources", "--settings", "--strict-mcp-config"):
            self.assertNotIn(flag, command)
        self.assertNotIn("profile:lean", verification.evidence)

    def test_permission_mode_without_a_value_is_no_decision(self) -> None:
        command, _, _ = self._run({}, permission_mode=None, read_only=True)
        self.assertEqual(command[command.index("--permission-mode") + 1], "dontAsk")

    def test_payload_turns_the_profile_off_for_one_task(self) -> None:
        command, _, _ = self._run({}, lean_profile=False)
        self.assertNotIn("--setting-sources", command)


if __name__ == "__main__":
    unittest.main()
