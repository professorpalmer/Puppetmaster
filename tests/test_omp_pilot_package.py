from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.diagnostics import (
    _omp_cli_name_ok,
    _omp_cli_visible,
    omp_pilot_check,
    run_doctor,
)
from puppetmaster.installers import (
    HandshakeResult,
    build_omp_mcp_entry,
    install_omp_mcp,
    omp_agent_dir,
    uninstall_omp_mcp,
)


def _same_path(left, right):
    left = left.resolve()
    right = right.resolve()
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


class OmpInstallerTests(unittest.TestCase):
    def test_entry_targets_stdio_module(self):
        entry = build_omp_mcp_entry("/opt/py")
        self.assertEqual(entry["command"], "/opt/py")
        self.assertEqual(entry["args"], ["-m", "puppetmaster.mcp_server"])
        self.assertEqual(entry["type"], "stdio")

    def test_install_is_idempotent_under_temp_home(self):
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            env = {"HOME": str(home)}
            first = install_omp_mcp(skip_handshake=True, env=env)
            self.assertEqual(first.status, "installed")
            mcp_path = home / ".omp" / "agent" / "mcp.json"
            self.assertTrue(mcp_path.is_file())
            mcp = json.loads(mcp_path.read_text(encoding="utf-8"))
            entry = mcp["mcpServers"]["puppetmaster"]
            self.assertEqual(entry["args"], ["-m", "puppetmaster.mcp_server"])
            self.assertEqual(entry["type"], "stdio")
            resolved_target = Path(first.target).resolve()
            self.assertTrue(_same_path(resolved_target, mcp_path))
            second = install_omp_mcp(skip_handshake=True, env=env)
            self.assertEqual(second.status, "unchanged")
            removed = uninstall_omp_mcp(env=env)
            self.assertEqual(removed.status, "removed")
            third = uninstall_omp_mcp(env=env)
            self.assertEqual(third.status, "unchanged")

    def test_profile_path_is_honored(self):
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            env = {"HOME": str(home), "OMP_PROFILE": "work"}
            agent = omp_agent_dir(env)
            self.assertTrue(_same_path(agent, home / ".omp" / "profiles" / "work" / "agent"))
            result = install_omp_mcp(skip_handshake=True, env=env)
            self.assertEqual(result.status, "installed")
            mcp_path = home / ".omp" / "profiles" / "work" / "agent" / "mcp.json"
            self.assertTrue(mcp_path.is_file())
            self.assertFalse((home / ".omp" / "agent" / "mcp.json").exists())

    def test_handshake_failure_refuses_write(self):
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            env = {"HOME": str(home)}
            boom = HandshakeResult(ok=False, error="nope")
            with patch("puppetmaster.installers.handshake_mcp_server", return_value=boom):
                result = install_omp_mcp(skip_handshake=False, env=env)
            self.assertEqual(result.status, "error")
            self.assertFalse((home / ".omp" / "agent" / "mcp.json").exists())

    def test_preserves_other_servers(self):
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            env = {"HOME": str(home)}
            agent = home / ".omp" / "agent"
            agent.mkdir(parents=True)
            prior = {"mcpServers": {"other": {"command": "echo", "args": ["hi"]}}}
            (agent / "mcp.json").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            result = install_omp_mcp(skip_handshake=True, env=env)
            self.assertEqual(result.status, "installed")
            mcp = json.loads((agent / "mcp.json").read_text(encoding="utf-8"))
            self.assertIn("other", mcp["mcpServers"])
            self.assertEqual(mcp["mcpServers"]["other"]["command"], "echo")


class OmpDoctorTests(unittest.TestCase):
    def _stub_omp(self, bindir, name="omp"):
        bindir.mkdir(parents=True, exist_ok=True)
        omp = bindir / name
        omp.write_text("exit 0\n", encoding="utf-8")
        return omp

    def test_absent_is_optional_not_error(self):
        with tempfile.TemporaryDirectory() as raw:
            env = {"HOME": raw, "PATH": str(Path(raw) / "empty")}
            check = omp_pilot_check(env=env)
            self.assertEqual(check.status, "optional")
            self.assertNotEqual(check.status, "error")
            self.assertIn("install-omp-mcp", check.detail)

    def test_healthy_when_mcp_written(self):
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            env = {
                "HOME": str(home),
                "PATH": str(home / "empty"),
                "PUPPETMASTER_OMP_SKIP_HANDSHAKE": "1",
            }
            install_omp_mcp(skip_handshake=True, env=env)
            check = omp_pilot_check(env=env)
            self.assertEqual(check.status, "ok", check.detail)
            self.assertIn("healthy", check.detail.lower())

    def test_cli_without_mcp_is_warn_not_error(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._stub_omp(base / "bin")
            env = {
                "HOME": str(base / "home"),
                "PATH": str(base / "bin"),
                "OMP_COMMAND": "omp",
            }
            check = omp_pilot_check(env=env)
            self.assertEqual(check.status, "warn")
            self.assertNotEqual(check.status, "error")
            self.assertIn("install-omp-mcp", check.detail)

    def test_run_doctor_includes_omp_pilot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            env = {"HOME": str(root / "no-home"), "PATH": str(root / "empty"), "PUPPETMASTER_OMP_SKIP_HANDSHAKE": "1"}
            with patch.dict(os.environ, env, clear=False):
                names = [c.name for c in run_doctor(root, state_dir=root / "state")]
            self.assertIn("omp-pilot", names)
class OmpCliParserTests(unittest.TestCase):
    def test_parser_has_install_omp_mcp(self):
        from puppetmaster.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["install-omp-mcp", "--dry-run", "--skip-handshake"])
        self.assertEqual(args.command, "install-omp-mcp")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.skip_handshake)

class OmpSetupHostTests(unittest.TestCase):
    def test_platforms_omp_is_not_a_worker_lock(self):
        from puppetmaster.cli.commands_install import _requested_host_pilots, _setup_platform_step
        args = SimpleNamespace(platforms="omp", skip_platforms=False)
        self.assertEqual(_requested_host_pilots(args), {"omp"})
        with tempfile.TemporaryDirectory() as raw:
            with patch("puppetmaster.platform_lock.default_registry_path", return_value=Path(raw) / "models.json"):
                rc = _setup_platform_step(args)
        self.assertEqual(rc, 0)

    def test_platforms_ohmypi_alias(self):
        from puppetmaster.cli.commands_install import _requested_host_pilots
        args = SimpleNamespace(platforms="ohmypi")
        self.assertEqual(_requested_host_pilots(args), {"ohmypi"})
