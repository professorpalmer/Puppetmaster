"""Codex reads $CODEX_HOME/AGENTS.md; instructions.md is ignored."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster import rules  # noqa: E402


class CodexRulesTargetTests(unittest.TestCase):
    def test_installs_into_codex_home_agents_and_moves_the_legacy_block(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"CODEX_HOME": tmp}):
            home = Path(tmp)
            (home / "AGENTS.md").write_text("# My own rules\nKeep this.\n", encoding="utf-8")
            legacy = rules.merge_block_into_text("Mine too.\n", rules.render_agents_block())[0]
            (home / "instructions.md").write_text(legacy, encoding="utf-8")

            outcome = rules._install_codex_global(dry_run=False, force=False)

            self.assertEqual(outcome.path, str(home / "AGENTS.md"))
            agents = (home / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("Keep this.", agents)
            self.assertIn(rules.render_agents_block().strip().splitlines()[0], agents)
            self.assertEqual((home / "instructions.md").read_text(encoding="utf-8").strip(), "Mine too.")
            again = rules._install_codex_global(dry_run=False, force=False)
            self.assertEqual(again.status, "unchanged")


if __name__ == "__main__":
    unittest.main()
