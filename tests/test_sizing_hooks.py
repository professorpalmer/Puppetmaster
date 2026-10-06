"""Sizing advice through host hooks (Claude TodoWrite, Codex update_plan)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster import hook_installers  # noqa: E402
from puppetmaster.hook_runner import _turn_state_path, handle_hook  # noqa: E402


def todos(total, done, prefix="[parallel] "):
    return {"todos": [{"content": f"{prefix}region {i}", "status": "completed" if i < done else "pending"}
                      for i in range(total)]}


class SizingHookTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.env = {"PUPPETMASTER_HOOK_ASSUME_TOOLS": "1", "PUPPETMASTER_HOME": self._tmp.name}

    def _start(self, session, ago):
        handle_hook({"prompt": "build the world", "session_id": session}, host="claude",
                    event="UserPromptSubmit", env=self.env)
        path = _turn_state_path({"session_id": session}, self.env)
        path.write_text(json.dumps({"started": time.time() - ago}))

    def test_projected_overrun_advises_once_per_turn(self):
        self._start("s1", ago=240)
        payload = {"session_id": "s1", "tool_name": "TodoWrite", "tool_input": todos(12, 2)}
        first = handle_hook(payload, host="claude", event="PostToolUse", env=self.env)
        out = first.to_host_json("claude")["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "PostToolUse")
        self.assertIn("SIZING", out["additionalContext"])
        again = handle_hook(payload, host="claude", event="PostToolUse", env=self.env)
        self.assertEqual(again.context, "")

    def test_codex_update_plan_and_small_work(self):
        self._start("s2", ago=240)
        plan = {"plan": [{"step": f"[parallel] module {i}", "status": "completed" if i < 2 else "pending"}
                         for i in range(12)]}
        r = handle_hook({"session_id": "s2", "tool_name": "update_plan", "tool_input": plan},
                        host="codex", event="PostToolUse", env=self.env)
        self.assertIn("SIZING", r.to_host_json("codex")["hookSpecificOutput"]["additionalContext"])
        self._start("s3", ago=12)
        small = handle_hook({"session_id": "s3", "tool_name": "TodoWrite", "tool_input": todos(16, 3)},
                            host="claude", event="PostToolUse", env=self.env)
        self.assertEqual(small.context, "")

    def test_other_tools_and_missing_turn_are_silent(self):
        r = handle_hook({"session_id": "nope", "tool_name": "TodoWrite", "tool_input": todos(12, 2)},
                        host="claude", event="PostToolUse", env=self.env)
        self.assertEqual(r.context, "")
        self._start("s4", ago=240)
        r = handle_hook({"session_id": "s4", "tool_name": "Bash", "tool_input": {}},
                        host="claude", event="PostToolUse", env=self.env)
        self.assertEqual(r.context, "")


class CodexHookInstallTests(unittest.TestCase):
    def test_global_install_merges_and_uninstall_keeps_user_hooks(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / ".codex" / "hooks.json"
            path.parent.mkdir()
            user = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "mine"}]}]}}
            path.write_text(json.dumps(user))
            hook_installers.install_hooks(targets=["codex"], scope="global", home=Path(home))
            data = json.loads(path.read_text())
            self.assertIn("PostToolUse", data["hooks"])
            self.assertEqual(data["hooks"]["PostToolUse"][0]["matcher"], "update_plan")
            self.assertEqual(data["hooks"]["SessionStart"], user["hooks"]["SessionStart"])
            hook_installers.uninstall_hooks(targets=["codex"], scopes=["global"], home=Path(home))
            self.assertEqual(json.loads(path.read_text()), user)
            project = hook_installers.install_hooks(targets=["codex"], scope="project", cwd=Path(home))
            self.assertEqual(project.outcomes[0].status, "skipped")


if __name__ == "__main__":
    unittest.main()
