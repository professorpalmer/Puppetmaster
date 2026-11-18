"""A partial @cursor/sdk directory is not an installed SDK.

A killed npm install can leave node_modules/@cursor/sdk without package.json.
Node cannot import it, so the probe must not report it and the bootstrap must
run again.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
import hermetic_env  # noqa: F401

from puppetmaster import cursor_sdk_home, diagnostics
from puppetmaster.installers import ensure_cursor_sdk


def _sdk_dir(prefix: Path) -> Path:
    sdk = prefix / "node_modules" / "@cursor" / "sdk"
    sdk.mkdir(parents=True)
    return sdk


class PartialCursorSdkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.tmp = tmp
        site = tmp / "site-packages" / "puppetmaster"
        site.mkdir(parents=True)
        self.site = site
        self.state = tmp / "state"
        env = {k: v for k, v in os.environ.items() if k != "PUPPETMASTER_HOME"}
        env["PUPPETMASTER_APP_STATE_ROOT"] = str(self.state)
        self._patches = [
            patch.object(cursor_sdk_home, "PACKAGED_RUNNER", site / "cursor_sdk_runner.mjs"),
            patch.dict(os.environ, env, clear=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def test_empty_workspace_sdk_dir_is_not_installed(self) -> None:
        workspace = self.tmp / "workspace"
        _sdk_dir(workspace)
        self.assertIsNone(diagnostics._find_cursor_sdk_install(workspace))
        self.assertEqual(diagnostics._cursor_sdk_check(workspace).status, "optional")

    def test_sdk_dir_with_package_json_is_installed(self) -> None:
        workspace = self.tmp / "workspace"
        sdk = _sdk_dir(workspace)
        (sdk / "package.json").write_text("{}", encoding="utf-8")
        self.assertEqual(diagnostics._find_cursor_sdk_install(workspace), sdk)

    def test_partial_packaged_sdk_does_not_hide_home_or_skip_bootstrap(self) -> None:
        _sdk_dir(self.site)
        self.assertIsNone(cursor_sdk_home.packaged_sdk())
        calls = []

        def fake_npm(cmd, **kwargs):
            calls.append(cmd)
            sdk = _sdk_dir(Path(cmd[cmd.index("--prefix") + 1]))
            (sdk / "package.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("puppetmaster.installers.subprocess.run", side_effect=fake_npm):
            result = ensure_cursor_sdk(npm_executable="/usr/bin/npm")
        self.assertEqual(result.status, "installed", result.detail)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cursor_sdk_home.home_sdk().parent.parent.parent, cursor_sdk_home.sdk_home())

    def test_partial_home_sdk_is_not_installed(self) -> None:
        _sdk_dir(cursor_sdk_home.sdk_home())
        self.assertIsNone(cursor_sdk_home.home_sdk())

    def test_complete_packaged_sdk_keeps_bootstrap_unchanged(self) -> None:
        sdk = _sdk_dir(self.site)
        (sdk / "package.json").write_text("{}", encoding="utf-8")
        with patch("puppetmaster.installers.subprocess.run") as run:
            result = ensure_cursor_sdk(npm_executable="/usr/bin/npm")
        self.assertEqual(result.status, "unchanged", result.detail)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
