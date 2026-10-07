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


class ConfiguredModelTests(unittest.TestCase):
    def test_reads_top_level_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_text(CONFIG)
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp}), "gpt-6.1-sol")
            Path(tmp, "config.toml").write_text('[profiles.x]\nmodel = "other"\n')
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp}), "")
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp + "/missing"}), "")


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
        (self.user / "AGENTS.md").write_text(merged, encoding="utf-8")
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

    def test_non_utf8_agents_is_copied_as_is(self):
        (self.user / "AGENTS.md").write_bytes(b"caf\xe9 rules\n")
        home = codex_home.prepare(self.env, root=self.worker)
        self.assertEqual((home / "AGENTS.md").read_bytes(), b"caf\xe9 rules\n")

    def test_no_user_login_means_no_lean_home(self):
        (self.user / "auth.json").unlink()
        self.assertIsNone(codex_home.prepare(self.env, root=self.worker))

    def test_kill_switch(self):
        self.assertFalse(codex_home.enabled({"PUPPETMASTER_CODEX_LEAN_HOME": "0"}))


class WorkerFlagsTests(unittest.TestCase):
    def test_lean_home_disables_land_before_the_stdin_prompt(self) -> None:
        from unittest.mock import patch

        from puppetmaster.adapters._base import CliInvocation
        from puppetmaster.adapters.codex import CodexAdapter, build_codex_exec_command

        command = build_codex_exec_command(
            executable="codex", model="m", extra_args=["--image", "a.png"])
        prepared = CliInvocation(command=command, sidecar_name="x")
        with patch.object(codex_home, "enabled", return_value=True), \
                patch.object(codex_home, "prepare", return_value=Path("/lean")), \
                patch.object(codex_home, "ensure_system_skills") as ensure:
            self.assertEqual(CodexAdapter._worker_home(prepared), Path("/lean"))
        ensure.assert_called_once_with(Path("/lean"), ["codex"])
        self.assertEqual(
            prepared.command[-8:],
            ["--image", "a.png", "--disable", "memories", "--disable", "multi_agent", "--", "-"])

    def test_a_resumed_thread_gets_the_bundle_check_but_keeps_its_flags(self) -> None:
        from unittest.mock import patch

        from puppetmaster.adapters._base import CliInvocation
        from puppetmaster.adapters.codex import CodexAdapter

        command = ["node", "codex.js", "exec", "resume", "thread-1", "--json", "--", "-"]
        prepared = CliInvocation(command=list(command), sidecar_name="x",
                                 extras={"resumed": True, "resume": {"session_id": "thread-1"}})
        with patch.object(codex_home, "enabled", return_value=True), \
                patch.object(codex_home, "home_for_session", return_value=Path("/lean")), \
                patch.object(codex_home, "worker_home_root", return_value=Path("/lean")), \
                patch.object(codex_home, "prepare") as prepare, \
                patch.object(codex_home, "ensure_system_skills") as ensure:
            self.assertEqual(CodexAdapter._worker_home(prepared), Path("/lean"))
        prepare.assert_not_called()
        ensure.assert_called_once_with(Path("/lean"), ["node", "codex.js"])
        self.assertEqual(prepared.command, command)


# Stand-in for stock ``codex debug prompt-input``: installs the builtin bundle
# into $CODEX_HOME/skills/.system (files first, marker last) and counts runs.
FAKE_CODEX = """import os, sys
from pathlib import Path
home = Path(os.environ["CODEX_HOME"])
assert sys.argv[1:3] == ["debug", "prompt-input"], sys.argv
assert not any(k.endswith("_API_KEY") for k in os.environ), "credentials reached the seed"
with open(os.environ["FAKE_CODEX_RUNS"], "a") as runs:
    runs.write("run\\n")
if os.environ.get("FAKE_CODEX_FAIL"):
    sys.exit(3)
root = home / "skills" / ".system"
for name in ("imagegen", "openai-docs", "review-agent", "skill-creator", "skill-installer"):
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(name + os.environ.get("FAKE_CODEX_VERSION", "1"))
(root / ".codex-system-skills.marker").write_text("v" + os.environ.get("FAKE_CODEX_VERSION", "1"))
"""


class SystemSkillsTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.home = root / "worker"
        self.home.mkdir()
        self.launcher = root / "codex.py"
        self.launcher.write_text(FAKE_CODEX)
        self.runs = root / "runs"
        env = patch.dict(os.environ, {"FAKE_CODEX_RUNS": str(self.runs), "OPENAI_API_KEY": "sk-not-real"})
        env.start()
        self.addCleanup(env.stop)

    def ensure(self) -> bool:
        return codex_home.ensure_system_skills(self.home, [sys.executable, str(self.launcher)])

    def run_count(self) -> int:
        return len(self.runs.read_text().splitlines()) if self.runs.exists() else 0

    def files(self) -> dict:
        root = self.home / "skills" / ".system"
        return {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*") if p.is_file()}

    def test_installs_once_then_trusts_the_record(self):
        self.assertTrue(self.ensure())
        self.assertEqual(len(self.files()), 6)
        self.assertTrue(self.ensure())
        self.assertEqual(self.run_count(), 1)
        self.assertEqual([p.name for p in (self.home / "skills").iterdir()], [".system"])

    def test_a_damaged_bundle_with_a_current_marker_is_reinstalled(self):
        self.ensure()
        installed = self.files()
        for name in ("imagegen", "review-agent"):
            for path in (self.home / "skills" / ".system" / name).iterdir():
                path.unlink()
        self.assertTrue(self.ensure())
        self.assertEqual(self.files(), installed)
        self.assertEqual(self.run_count(), 2)

    def test_an_upgraded_codex_reseeds(self):
        from unittest.mock import patch

        self.ensure()
        stat = self.launcher.stat()
        os.utime(self.launcher, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        with patch.dict(os.environ, {"FAKE_CODEX_VERSION": "2"}):
            self.assertTrue(self.ensure())
        self.assertEqual(self.files()[".codex-system-skills.marker"], "v2")

    def test_a_failed_swap_keeps_the_existing_bundle_and_retries(self):
        from unittest.mock import patch

        self.ensure()
        installed = self.files()
        stat = self.launcher.stat()
        os.utime(self.launcher, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        real_replace = os.replace

        def refuse_install(src, dst):
            if Path(src).name.endswith(".new"):
                raise OSError(5, "Access is denied")
            return real_replace(src, dst)

        with patch.dict(os.environ, {"FAKE_CODEX_VERSION": "2"}), \
                patch("puppetmaster.codex_home.os.replace", refuse_install):
            self.assertFalse(self.ensure())
        self.assertEqual(self.files(), installed)
        self.assertEqual([p.name for p in (self.home / "skills").iterdir()], [".system"])
        with patch.dict(os.environ, {"FAKE_CODEX_VERSION": "2"}):
            self.assertTrue(self.ensure())
        self.assertEqual(self.files()[".codex-system-skills.marker"], "v2")

    def test_a_failing_bootstrap_leaves_the_home_to_stock_codex(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"FAKE_CODEX_FAIL": "1"}):
            self.assertFalse(self.ensure())
        self.assertFalse((self.home / "skills").exists())
        self.assertTrue(self.ensure())

    def test_parallel_starts_install_once(self):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as pool:
            self.assertTrue(all(pool.map(lambda _: self.ensure(), range(8))))
        self.assertEqual(self.run_count(), 1)
        self.assertEqual(len(self.files()), 6)


class UnpinnedCliTests(unittest.TestCase):
    def test_codex_verb_pins_no_model_unless_asked(self) -> None:
        from puppetmaster.cli._parser import build_parser

        parser = build_parser()
        self.assertIsNone(parser.parse_args(["codex", "x"]).model)
        self.assertEqual(parser.parse_args(["codex", "x", "--model", "gpt-6.1-sol"]).model,
                         "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
