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


class LeanConfigScanTests(unittest.TestCase):
    """The filter reads the config as TOML statements, not as lines with a pattern."""

    def test_brackets_in_strings_and_comments_are_not_structure(self):
        cfg = """model = "x"
notify = ["a", "b[c"]
arr = [
  [1, 2],
  ["]", "#"],
]
[mcp_servers.one]
command = "run[this"
args = ["a[b", "c"]  # [not a table]
env = { K = "v]" }
lit = 'it[s'
[profiles.p]
model = "y"
[[plugins]]
x = 1
[model_providers.z]
name = "z"
"""
        self.assertEqual(codex_home.lean_config(cfg), """model = "x"
arr = [
  [1, 2],
  ["]", "#"],
]
[profiles.p]
model = "y"
[model_providers.z]
name = "z"
""")

    def test_a_bracket_in_a_comment_after_a_dropped_key_ends_nothing(self):
        cfg = 'notify = ["a"]  # [\nmodel = "m"\n[profiles.p]\ny = 2\n'
        self.assertEqual(codex_home.lean_config(cfg), 'model = "m"\n[profiles.p]\ny = 2\n')

    def test_a_table_header_inside_a_multi_line_string_is_text(self):
        cfg = ('note = """a [ b\n[mcp_servers.fake]\n"""\nkeep = 1\n[hooks]\nx = """\n[profiles.q]\n"""\n'
               "[profiles.p]\ny = 2\n")
        self.assertEqual(codex_home.lean_config(cfg),
                         'note = """a [ b\n[mcp_servers.fake]\n"""\nkeep = 1\n[profiles.p]\ny = 2\n')

    def test_a_multi_line_value_of_a_dropped_top_key_is_dropped_whole(self):
        cfg = 'notify = [\n  "prog",\n  ["x", "y"],\n]\nmodel = "m"\n[t]\nnotify = ["kept: a key of a table"]\n'
        self.assertEqual(codex_home.lean_config(cfg), 'model = "m"\n[t]\nnotify = ["kept: a key of a table"]\n')

    def test_a_dropped_top_key_after_a_nested_array_is_dropped(self):
        cfg = 'arr = [\n  [1, 2],\n]\nnotify = ["prog"]\nmodel = "m"\n'
        self.assertEqual(codex_home.lean_config(cfg), 'arr = [\n  [1, 2],\n]\nmodel = "m"\n')

    def test_a_multi_line_string_can_end_with_one_or_two_quote_characters(self):
        for q in ('"', "'"):
            for extra in (0, 1, 2):
                line = f"a = [{q * 3}x{q * (3 + extra)}]"
                self.assertEqual(codex_home._toml_scan(line, 0, None), (0, None), line)
                self.assertEqual(codex_home._toml_scan(f"x{q * (3 + extra)}]", 1, q * 3), (0, None), line)
            cfg = f'a = [{q * 3}x{q * 4}]\n[mcp_servers.one]\ncommand = "c"\n[profiles.p]\ny = 2\n'
            self.assertEqual(codex_home.lean_config(cfg), f'a = [{q * 3}x{q * 4}]\n[profiles.p]\ny = 2\n')

    def test_a_dropped_table_goes_in_each_form(self):
        keep = 'model = "m"\n[profiles.p]\ny = 2\n'
        for form in ('mcp_servers = { one = { command = "c" } }\n',
                     'mcp_servers.one.command = "c"\n',
                     '"mcp_servers".one.command = "c"\n',
                     "'hooks' = { a = 1 }\n",
                     'mcp_servers = {\n  one = { command = "c" },\n}\n',
                     '"notify" = ["prog"]\n',
                     "'notify' = [\n  \"prog\",\n]\n",
                     ' \t"notify" = ["prog"]\n'):
            self.assertEqual(codex_home.lean_config('model = "m"\n' + form + "[profiles.p]\ny = 2\n"), keep, form)
        for header in ('["mcp_servers".one]', "['mcp_servers'.one]", "[ mcp_servers . one ]", "[[hooks]]",
                       '[["hooks"]]', "[mcp_servers]", '["mcp\\u005fservers"]'):
            cfg = f'model = "m"\n{header}\ncommand = "c"\n[profiles.p]\ny = 2\n'
            self.assertEqual(codex_home.lean_config(cfg), keep, header)

    def test_another_table_or_key_with_a_name_that_starts_the_same_stays(self):
        for cfg in ('[mcp_servers-old]\ncommand = "c"\n', "[hooks_2]\na = 1\n", '["mcp_servers x"]\na = 1\n',
                    "notify-all = 1\n", "notifyx = 1\n", "[a]\nmcp_servers = 1\nnotify = 2\n",
                    'x = [\n  ["mcp_servers"],\n  ["hooks", "y"],\n]\n[profiles.p]\ny = 2\n'):
            self.assertEqual(codex_home.lean_config(cfg), cfg, cfg)

    def test_each_escape_of_a_quoted_key_is_read(self):
        for text, key in ((r'"\U0000006dcp_servers" = 1', "mcp_servers"), (r'"mcp\u005fservers" = 1', "mcp_servers"),
                          (r'"\x6dcp_servers" = 1', "mcp_servers"), (r'"a\tb" = 1', "a\tb"), (r'"a\"b" = 1', 'a"b'),
                          (r'"a\\u005fb" = 1', r"a\u005fb"), (r'"a\qb" = 1', r"a\qb"), (r'"\e" = 1', "\x1b"),
                          (r'"\UFFFFFFFF" = 1', r"\UFFFFFFFF")):
            self.assertEqual(codex_home._first_key(text), key, text)
        cfg = ('model = "m"\n["\\U0000006dcp_servers".one]\ncommand = "c"\n"\\U0000006eotify" = ["p"]\n'
               "[profiles.p]\ny = 2\n")
        self.assertEqual(codex_home.lean_config(cfg), 'model = "m"\n[profiles.p]\ny = 2\n')
        # A literal string has no escape.
        self.assertEqual(codex_home._first_key(r"'mcp\u005fservers' = 1"), r"mcp\u005fservers")

    def test_the_first_part_of_a_key(self):
        for text, key in (("a = 1", "a"), ("a.b = 1", "a"), ('"a b".c = 1', "a b"), ("'a'.b = 1", "a"),
                          ("  a-b_1 = 1", "a-b_1"), ("# a comment", None), ("", None), ("= 1", None),
                          ('"open', None), ("'open", None)):
            self.assertEqual(codex_home._first_key(text), key, text)

    def test_an_escaped_quote_does_not_close_a_basic_multi_line_string(self):
        self.assertEqual(codex_home._toml_scan('a\\"""b', 0, '"""'), (0, '"""'))
        self.assertEqual(codex_home._toml_scan('a\\\\"""] [', 1, '"""'), (1, None))
        # A literal multi-line string has no escape.
        self.assertEqual(codex_home._toml_scan("a\\'''] [", 1, "'''"), (1, None))

    def test_the_scanner_reads_escapes(self):
        self.assertEqual(codex_home._toml_scan('s = "a\\" [', 0, None), (0, None))
        self.assertEqual(codex_home._toml_scan('s = "a\\\\" [', 0, None), (1, None))
        self.assertEqual(codex_home._toml_scan('s = """a', 0, None), (0, '"""'))
        self.assertEqual(codex_home._toml_scan('b""" [', 0, '"""'), (1, None))


class ConfiguredModelTests(unittest.TestCase):
    def test_reads_top_level_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_text(CONFIG)
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp}), "gpt-6.1-sol")
            Path(tmp, "config.toml").write_text('[profiles.x]\nmodel = "other"\n')
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp}), "")
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp + "/missing"}), "")

    def test_a_config_with_a_byte_order_mark_is_read(self):
        # Codex reads a config.toml that starts with a byte order mark.
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_bytes(b"\xef\xbb\xbf" + b'model = "gpt-6.1-sol"\n')
            self.assertEqual(codex_home.configured_model({"CODEX_HOME": tmp}), "gpt-6.1-sol")


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

    def test_a_config_with_a_byte_order_mark_loses_its_first_dropped_statement(self):
        for first in ('notify = ["prog"]\n', '[mcp_servers.one]\ncommand = "c"\n'):
            (self.user / "config.toml").write_bytes(b"\xef\xbb\xbf" + first.encode() + b'[profiles.p]\ny = 2\n')
            home = codex_home.prepare(self.env, root=self.worker)
            self.assertEqual((home / "config.toml").read_bytes(), b"[profiles.p]\ny = 2\n", first)

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
