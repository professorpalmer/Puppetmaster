"""Lean CODEX_HOME for Codex workers: settings kept, context-heavy tables dropped."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster import codex_home  # noqa: E402

CONFIG = '''notify = [
  "/Applications/Notifier.app",
  "turn-ended"
]
model = "gpt-6.1-sol"
service_tier = "priority"

[projects."/repo"]
trust_level = "trusted"

[mcp_servers.puppetmaster]
command = "python"
args = ["-m", "puppetmaster.mcp_server"]

[mcp_servers.puppetmaster.env]
KEY = "x"

[plugins."slack@openai"]
enabled = true

[memories]
enabled = true

[features]
apps = true
'''


class LeanConfigTests(unittest.TestCase):
    def test_settings_kept_and_context_tables_dropped(self):
        lean = codex_home.lean_config(CONFIG)
        for kept in ('model = "gpt-6.1-sol"', 'service_tier = "priority"', '[projects."/repo"]',
                     'trust_level = "trusted"', "[features]", "apps = true"):
            self.assertIn(kept, lean)
        for dropped in ("notify", "Notifier", "mcp_servers", "puppetmaster.mcp_server",
                        'KEY = "x"', "plugins", "[memories]"):
            self.assertNotIn(dropped, lean)
        try:
            import tomllib
        except ImportError:  # Python < 3.11
            return
        tomllib.loads(lean)


class AuthSyncTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.user, self.worker = root / "user", root / "worker"
        self.user.mkdir()
        (self.user / "auth.json").write_text(json.dumps({"refresh": "r1"}))
        (self.user / "config.toml").write_text(CONFIG)
        (self.user / "AGENTS.md").write_text("no emojis")
        self.env = {"CODEX_HOME": str(self.user)}

    def test_prepare_copies_auth_and_agents_never_symlinks(self):
        from puppetmaster.rules import merge_block_into_text, render_agents_block

        merged = merge_block_into_text("no emojis\n", render_agents_block())[0]
        (self.user / "AGENTS.md").write_text(merged)
        home = codex_home.prepare(self.env, root=self.worker)
        self.assertFalse((home / "auth.json").is_symlink())
        self.assertEqual((home / "AGENTS.md").read_text().strip(), "no emojis")
        self.assertNotIn("mcp_servers", (home / "config.toml").read_text())

    def test_refreshed_worker_login_syncs_back_only_if_user_unchanged(self):
        home = codex_home.prepare(self.env, root=self.worker)
        (home / "auth.json").write_text(json.dumps({"refresh": "r2"}))
        self.assertTrue(codex_home.sync_back(self.env, root=self.worker))
        self.assertIn("r2", (self.user / "auth.json").read_text())
        # The user logs in again meanwhile: a stale worker copy must not overwrite it.
        (self.user / "auth.json").write_text(json.dumps({"refresh": "user-new"}))
        (home / "auth.json").write_text(json.dumps({"refresh": "r3"}))
        self.assertFalse(codex_home.sync_back(self.env, root=self.worker))
        self.assertIn("user-new", (self.user / "auth.json").read_text())

    def test_no_user_login_means_no_lean_home(self):
        (self.user / "auth.json").unlink()
        self.assertIsNone(codex_home.prepare(self.env, root=self.worker))

    def test_kill_switch(self):
        self.assertFalse(codex_home.enabled({"PUPPETMASTER_CODEX_LEAN_HOME": "0"}))


if __name__ == "__main__":
    unittest.main()
