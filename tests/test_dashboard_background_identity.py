"""Identity-aware background dashboard reuse: CLI ``--background`` and the
MCP ``puppetmaster_dashboard`` tool.

Pre-fix, both paths declared "reusing it" on bare liveness -- *something*
answered ``GET /api/jobs`` on host:port -- with no way to tell whether that
something was this project's dashboard, a different project's, or a
mismatched --all-projects scope. The early return also preceded the runfile
write, so the caller that "reused" the server had no runfile of its own:
its own ``--status``/``--stop`` then had nothing to work with, and it could
neither confirm nor stop the server it was told to use. See
docs/CHANGELOG.md.

CLI-level and MCP-level, with the dashboard identity functions and
``subprocess.Popen``/``_spawn_dashboard_server`` patched -- no real process is
spawned, so these stay fast and CI-safe. A new file for the same reason as
test_dashboard_ports.py: keeps this workstream's tests out of the way of the
concurrent Ctrl-C workstream's edits to test_puppetmaster.py.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import puppetmaster.dashboard as dash


class BackgroundDashboardIdentityTests(unittest.TestCase):
    def _args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            port=None,
            port_search=False,
            host="127.0.0.1",
            no_open=True,
            allow_external=False,
            all_projects=False,
            mobile=False,
            qr=False,
            background=False,
            stop=False,
            status=False,
            write_runfile=False,
            job_id=None,
            backend="sqlite",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_background_reuses_only_same_project(self) -> None:
        """A foreign dashboard answering on the requested port must not be
        reused -- _start_background_dashboard must spawn its own instead.
        Pre-fix this was `_dispatch.py`'s bare `dashboard_alive()` check,
        which said "reusing it" here and handed the caller project A's URL
        while it asked for project B (reproduced verbatim in the research
        harness)."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            spawned = MagicMock()
            spawned.pid = 4242
            spawned.poll.return_value = None
            child_runfile = {
                "pid": 4242, "host": "127.0.0.1", "port": 8787,
                "url": "http://127.0.0.1:8787/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", side_effect=[None, child_runfile]
            ), patch.object(
                dash, "dashboard_serves", side_effect=[False, True]
            ), patch("subprocess.Popen", return_value=spawned) as popen:
                rc = _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=8787, auto_port=True,
                    source="loopback", allow_external=False,
                )
        popen.assert_called_once()
        self.assertEqual(rc, 0)

    def test_background_reuse_writes_runfile(self) -> None:
        """Genuine reuse (this project's own dashboard identifies itself)
        must write a runfile before returning -- pre-fix the early return
        preceded the runfile write, so the reusing caller had no runfile of
        its own (verified: --status/--stop then failed for it)."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            identity = {
                "service": "puppetmaster-dashboard", "pid": 9999,
                "state_dir_id": "irrelevant-mocked", "all_projects": False,
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=None
            ), patch.object(
                dash, "dashboard_serves", return_value=True
            ), patch.object(
                dash, "dashboard_identity", return_value=identity
            ), patch("subprocess.Popen") as popen:
                rc = _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=8787, auto_port=True,
                    source="loopback", allow_external=False,
                )
            popen.assert_not_called()
            self.assertEqual(rc, 0)
            info = dash.read_dashboard_runfile(state_dir)
            self.assertIsNotNone(info)
            self.assertEqual(info["pid"], 9999)
            self.assertEqual(info["port"], 8787)

    def test_status_reports_foreign_dashboard_distinctly(self) -> None:
        """Pre-fix, `--status` printed "No background dashboard is running"
        for a runfile that *was* tracked, just answered by another project
        -- confusing when the caller had just been told a dashboard was
        running. Post-fix it must say so distinctly, not fall back to
        "not running"."""
        from puppetmaster.cli._dispatch import _run_dashboard_command

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 1234, "host": "127.0.0.1", "port": 8787,
                "url": "http://127.0.0.1:8787/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "dashboard_serves", return_value=False
            ), patch.object(
                dash, "dashboard_alive", return_value=True
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_dashboard_command(self._args(status=True), state_dir)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("another project", text)
        self.assertNotIn("No background dashboard is running", text)

    def test_stop_after_reuse_can_stop(self) -> None:
        """Pre-fix, the reusing caller's --stop said "no background
        dashboard is tracked here" and the server survived (it had never
        gotten a runfile). Post-fix, reuse writes one, so --stop works."""
        from puppetmaster.cli._dispatch import (
            _run_dashboard_command,
            _start_background_dashboard,
        )

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            identity = {
                "service": "puppetmaster-dashboard", "pid": 424242,
                "state_dir_id": "x", "all_projects": False,
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=None
            ), patch.object(
                dash, "dashboard_serves", return_value=True
            ), patch.object(
                dash, "dashboard_identity", return_value=identity
            ), patch("subprocess.Popen") as popen:
                reuse_rc = _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=8787, auto_port=True,
                    source="loopback", allow_external=False,
                )
            self.assertEqual(reuse_rc, 0)
            popen.assert_not_called()

            stop_rc = _run_dashboard_command(self._args(stop=True), state_dir)
        self.assertEqual(stop_rc, 0)
        self.assertIsNone(dash.read_dashboard_runfile(state_dir))

    def test_background_parent_reports_child_bumped_port(self) -> None:
        """The child may bump past the requested port; the parent's
        announcement must report the *actual* port, not the one it asked
        for (pre-fix the parent always printed its own --port)."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            spawned = MagicMock()
            spawned.pid = 5150
            spawned.poll.return_value = None
            bumped = {
                "pid": 5150, "host": "127.0.0.1", "port": 8788,
                "url": "http://127.0.0.1:8788/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", side_effect=[None, bumped]
            ), patch.object(
                dash, "dashboard_serves", side_effect=[False, True]
            ), patch("subprocess.Popen", return_value=spawned):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _start_background_dashboard(
                        self._args(), state_dir, "127.0.0.1",
                        port=8787, auto_port=True,
                        source="loopback", allow_external=False,
                    )
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("8788", text)
        self.assertNotIn("http://127.0.0.1:8787/", text)

    def test_mcp_dashboard_does_not_reuse_foreign_project(self) -> None:
        """Same defect as test_background_reuses_only_same_project, on the
        MCP puppetmaster_dashboard tool's identical code shape."""
        import puppetmaster.mcp_server as mcp

        spawned = MagicMock()
        spawned.pid = 6060
        spawned.poll.return_value = None
        child_runfile = {
            "pid": 6060, "host": "127.0.0.1", "port": 8787,
            "url": "http://127.0.0.1:8787/",
        }
        with patch.object(
            dash, "dashboard_serves", side_effect=[False, True]
        ), patch.object(
            dash, "read_dashboard_runfile", return_value=child_runfile
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        popen.assert_called_once()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["started"])


if __name__ == "__main__":
    unittest.main()
