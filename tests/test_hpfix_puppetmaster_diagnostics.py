from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster import diagnostics


class RegistryStarterSyncUnreadableTests(unittest.TestCase):
    def test_non_utf8_registry_is_unreadable_warning(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "models.json"
            path.write_bytes(b"\xff\xfe[")
            check = diagnostics._registry_starter_sync_check(path)
        self.assertEqual(check.status, "warn")
        self.assertIn("registry unreadable", check.detail)

    def test_load_registry_reports_non_utf8_as_runtime_error(self) -> None:
        from puppetmaster.model_registry import load_registry

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "models.json"
            path.write_bytes(b"\xff\xfe[")
            with self.assertRaisesRegex(RuntimeError, "not UTF-8"):
                load_registry(path)

    def test_invalid_json_registry_keeps_unreadable_warning(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "models.json"
            path.write_text("[", encoding="utf-8")
            check = diagnostics._registry_starter_sync_check(path)
        self.assertEqual(check.status, "warn")
        self.assertIn("registry unreadable", check.detail)

    def test_missing_registry_stays_optional(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            check = diagnostics._registry_starter_sync_check(Path(d) / "models.json")
        self.assertEqual(check.status, "optional")


class AgentRulesHomePrefixTests(unittest.TestCase):
    def _check(self, home: Path, root: Path):
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("CODEX_HOME", None)
            return diagnostics._agent_rules_check(root)

    def test_sibling_of_home_with_shared_prefix_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            home = Path(t) / "a"
            home.mkdir()
            root = Path(t) / "ab" / "proj"
            root.mkdir(parents=True)
            (root / "AGENTS.md").write_text("puppetmaster:rules:begin\n", encoding="utf-8")
            check = self._check(home, root)
        self.assertEqual(check.status, "ok")
        self.assertIn(str(root / "AGENTS.md"), check.detail)

    def test_rule_under_home_stays_home_relative(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            home = Path(t) / "a"
            root = home / "proj"
            root.mkdir(parents=True)
            (root / "AGENTS.md").write_text("puppetmaster:rules:begin\n", encoding="utf-8")
            check = self._check(home, root)
        self.assertEqual(check.status, "ok")
        self.assertEqual(check.detail, f"agent rule present ({Path('proj') / 'AGENTS.md'})")


class BareCommandInCwdTests(unittest.TestCase):
    CASES = (
        ("CLAUDE_CODE_COMMAND", diagnostics._claude_code_installed),
        ("CODEX_COMMAND", diagnostics._codex_cli_installed),
        ("HERMES_COMMAND", diagnostics._hermes_cli_installed),
        ("FX_COMMAND", diagnostics._fx_cli_installed),
        ("AGY_COMMAND", diagnostics._antigravity_installed),
    )

    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(os.chdir, self._cwd)
        os.chdir(self._tmp.name)
        self.tmp = Path(self._tmp.name)

    def test_bare_name_folder_in_cwd_is_not_installed(self) -> None:
        Path("zzcli").mkdir()
        env = {"PATH": str(self.tmp / "empty")}
        for var, probe in self.CASES:
            with self.subTest(var=var), mock.patch.dict(os.environ, {**env, var: "zzcli"}):
                self.assertFalse(probe())

    def test_claude_check_reports_optional_for_bare_name_in_cwd(self) -> None:
        Path("zzclaude").mkdir()
        env = {"PATH": str(self.tmp / "empty"), "CLAUDE_CODE_COMMAND": "zzclaude"}
        with mock.patch.dict(os.environ, env):
            check = diagnostics._claude_code_check()
        self.assertNotEqual(check.status, "ok")

    def test_explicit_relative_path_still_counts(self) -> None:
        Path("zzcli").write_text("", encoding="utf-8")
        env = {"PATH": str(self.tmp / "empty")}
        command = os.path.join(".", "zzcli")
        for var, probe in self.CASES:
            with self.subTest(var=var), mock.patch.dict(os.environ, {**env, var: command}):
                self.assertTrue(probe())

    def test_bare_name_on_path_still_counts(self) -> None:
        bindir = self.tmp / "bin"
        bindir.mkdir()
        exe = bindir / "zzcli"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
        env = {"PATH": str(bindir), "PATHEXT": ""}
        for var, probe in self.CASES:
            with self.subTest(var=var), mock.patch.dict(os.environ, {**env, var: "zzcli"}):
                self.assertTrue(probe())


if __name__ == "__main__":
    unittest.main()
