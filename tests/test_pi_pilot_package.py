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

from puppetmaster.diagnostics import pi_pilot_check, run_doctor
from puppetmaster.installers import (
    HandshakeResult,
    build_pi_mcp_entry,
    bundled_pi_package_dir,
    install_pi_mcp,
    uninstall_pi_mcp,
)


class PiPackageLayoutTests(unittest.TestCase):
    def test_manifest_is_a_pi_package(self) -> None:
        pkg = bundled_pi_package_dir()
        self.assertIsNotNone(pkg)
        assert pkg is not None
        manifest = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
        self.assertIn("pi-package", manifest.get("keywords") or [])
        self.assertIn("extensions", manifest.get("pi") or {})
        self.assertIn("skills", manifest.get("pi") or {})
        self.assertTrue((pkg / "extensions" / "puppetmaster-mcp.ts").is_file())
        self.assertTrue((pkg / "skills" / "puppetmaster-pilot" / "SKILL.md").is_file())
        skill = (pkg / "skills" / "puppetmaster-pilot" / "SKILL.md").read_text(encoding="utf-8")
        for verb in ("start_implement", "start_agentic", "start_prewalk", "effort", "show"):
            self.assertIn(verb, skill)
        self.assertIn("not a leased worker", skill.lower())
        ext = (pkg / "extensions" / "puppetmaster-mcp.ts").read_text(encoding="utf-8")
        self.assertIn("puppetmaster.mcp_server", ext)
        self.assertIn("registerTool", ext)


class PiInstallerTests(unittest.TestCase):
    def test_entry_targets_stdio_module(self) -> None:
        entry = build_pi_mcp_entry("/opt/py")
        self.assertEqual(entry["command"], "/opt/py")
        self.assertEqual(entry["args"], ["-m", "puppetmaster.mcp_server"])
        self.assertEqual(entry["transport"], "stdio")

    def test_install_is_idempotent(self) -> None:
        pkg = bundled_pi_package_dir()
        self.assertIsNotNone(pkg)
        with tempfile.TemporaryDirectory() as raw:
            agent = Path(raw) / "agent"
            first = install_pi_mcp(agent_dir=agent, package_dir=pkg, skip_handshake=True)
            self.assertEqual(first.status, "installed")
            mcp = json.loads((agent / "mcp.json").read_text(encoding="utf-8"))
            entry = mcp["mcpServers"]["puppetmaster"]
            self.assertEqual(entry["args"], ["-m", "puppetmaster.mcp_server"])
            settings = json.loads((agent / "settings.json").read_text(encoding="utf-8"))
            packed = " ".join(str(item) for item in settings["packages"])
            self.assertTrue(str(pkg.resolve()) in packed or str(pkg) in packed)
            second = install_pi_mcp(agent_dir=agent, package_dir=pkg, skip_handshake=True)
            self.assertEqual(second.status, "unchanged")
            removed = uninstall_pi_mcp(agent_dir=agent, package_dir=pkg)
            self.assertEqual(removed.status, "removed")
            third = uninstall_pi_mcp(agent_dir=agent, package_dir=pkg)
            self.assertEqual(third.status, "unchanged")

    def test_handshake_failure_refuses_write(self) -> None:
        pkg = bundled_pi_package_dir()
        with tempfile.TemporaryDirectory() as raw:
            agent = Path(raw) / "agent"
            boom = HandshakeResult(ok=False, error="nope")
            with patch("puppetmaster.installers.handshake_mcp_server", return_value=boom):
                result = install_pi_mcp(agent_dir=agent, package_dir=pkg, skip_handshake=False)
            self.assertEqual(result.status, "error")
            self.assertFalse((agent / "mcp.json").exists())


class PiDoctorTests(unittest.TestCase):
    def _stub_pi(self, bindir: Path) -> Path:
        bindir.mkdir(parents=True, exist_ok=True)
        pi = bindir / "pi"
        pi.write_bytes(b"exit 0\n")
        pi.chmod(0o755)
        return pi

    def test_absent_is_warn_with_fix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            env = {"PI_CODING_AGENT_DIR": str(Path(raw) / "agent"), "PATH": str(Path(raw) / "empty")}
            check = pi_pilot_check(env=env)
            self.assertEqual(check.status, "warn")
            self.assertIn("Fix:", check.detail)
            self.assertIn("install-pi-mcp", check.detail)

    def test_healthy_when_cli_package_and_mcp(self) -> None:
        pkg = bundled_pi_package_dir()
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            agent = base / "agent"
            self._stub_pi(base / "bin")
            install_pi_mcp(agent_dir=agent, package_dir=pkg, skip_handshake=True)
            env = {
                "PI_CODING_AGENT_DIR": str(agent),
                "PATH": str(base / "bin"),
                "PI_COMMAND": "pi",
                "PUPPETMASTER_PI_SKIP_HANDSHAKE": "1",
            }
            check = pi_pilot_check(env=env)
            self.assertEqual(check.status, "ok", check.detail)
            self.assertIn("healthy", check.detail.lower())

    def test_incomplete_is_error_with_exact_fix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            (base / "agent").mkdir()
            self._stub_pi(base / "bin")
            env = {"PI_CODING_AGENT_DIR": str(base / "agent"), "PATH": str(base / "bin"), "PI_COMMAND": "pi"}
            check = pi_pilot_check(env=env)
            self.assertEqual(check.status, "error")
            self.assertIn("install-pi-mcp", check.detail)

    def test_run_doctor_includes_pi_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            env = {
                "PI_CODING_AGENT_DIR": str(root / "no-agent"),
                "PATH": str(root / "empty"),
                "PUPPETMASTER_PI_SKIP_HANDSHAKE": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                names = [c.name for c in run_doctor(root, state_dir=root / "state")]
            self.assertIn("pi-pilot", names)


class PiSetupHostTests(unittest.TestCase):
    def test_platforms_pi_is_not_a_worker_lock(self) -> None:
        from puppetmaster.cli.commands_install import _requested_host_pilots, _setup_platform_step
        args = SimpleNamespace(platforms="pi", skip_platforms=False)
        self.assertEqual(_requested_host_pilots(args), {"pi"})
        with tempfile.TemporaryDirectory() as raw:
            with patch("puppetmaster.platform_lock.default_registry_path", return_value=Path(raw) / "models.json"):
                rc = _setup_platform_step(args)
        self.assertEqual(rc, 0)


class PiCliParserTests(unittest.TestCase):
    def test_parser_has_install_pi_mcp(self) -> None:
        from puppetmaster.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["install-pi-mcp", "--dry-run", "--skip-handshake"])
        self.assertEqual(args.command, "install-pi-mcp")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.skip_handshake)


class PiLiveE2ETests(unittest.TestCase):
    def test_live_pi_package_load(self) -> None:
        if os.environ.get("PI_LIVE_E2E") != "1":
            self.skipTest("set PI_LIVE_E2E=1 to run a live Pi package load")
        import shutil
        if shutil.which("pi") is None:
            self.skipTest("pi CLI not on PATH")
        from puppetmaster.installers import handshake_mcp_server
        result = handshake_mcp_server()
        self.assertTrue(result.ok, result.error)
        self.assertGreater(result.tool_count, 0)


if __name__ == "__main__":
    unittest.main()
