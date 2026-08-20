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
import os
import threading
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
        "not running". A real (non-None) foreign identity, not just a
        legacy 404 -- see test_status_falls_back_to_own_pid_pre_upgrade for
        the pre-/api/meta case (§F5)."""
        from puppetmaster.cli._dispatch import _run_dashboard_command

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            # Deliberately not 8787 (this machine's own live dashboard) --
            # dashboard_identity/pid_alive are mocked below so nothing here
            # ever reaches the network either way, but a non-default port
            # keeps that true even if a future edit drops a mock.
            tracked = {
                "pid": 1234, "host": "127.0.0.1", "port": 19100,
                "url": "http://127.0.0.1:19100/",
            }
            foreign_identity = {
                "service": "puppetmaster-dashboard", "pid": 4321,
                "state_dir_id": "not-this-project", "all_projects": False,
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "dashboard_serves", return_value=False
            ), patch.object(
                dash, "dashboard_alive", return_value=True
            ), patch.object(
                dash, "dashboard_identity", return_value=foreign_identity
            ), patch.object(
                dash, "pid_alive", return_value=False
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_dashboard_command(self._args(status=True), state_dir)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("another project", text)
        self.assertNotIn("No background dashboard is running", text)

    def test_status_falls_back_to_own_pid_pre_upgrade(self) -> None:
        """A pre-upgrade dashboard 404s /api/meta (dashboard_identity ->
        None), which used to be indistinguishable from a genuine foreign
        server -- every existing user with a background board hits this in
        the upgrade window. If the runfile's own tracked pid is still
        alive, --status must report it as this project's (§F5), not claim
        it's "serving another project", which is simply false."""
        from puppetmaster.cli._dispatch import _run_dashboard_command

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 5678, "host": "127.0.0.1", "port": 19101,
                "url": "http://127.0.0.1:19101/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "dashboard_serves", return_value=False
            ), patch.object(
                dash, "dashboard_alive", return_value=True
            ), patch.object(
                dash, "dashboard_identity", return_value=None
            ), patch.object(
                dash, "pid_alive", return_value=True
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_dashboard_command(self._args(status=True), state_dir)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("Background dashboard running", text)
        self.assertIn("predates /api/meta", text)
        self.assertNotIn("another project", text)

    def test_status_still_reports_foreign_when_pid_also_dead(self) -> None:
        """The fallback is specifically "no /api/meta, but our own tracked
        pid is alive" -- if the tracked pid is also dead, this must not
        become a way to call a truly foreign/dead-tracked server "ours"."""
        from puppetmaster.cli._dispatch import _run_dashboard_command

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 9012, "host": "127.0.0.1", "port": 19102,
                "url": "http://127.0.0.1:19102/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "dashboard_serves", return_value=False
            ), patch.object(
                dash, "dashboard_alive", return_value=True
            ), patch.object(
                dash, "dashboard_identity", return_value=None
            ), patch.object(
                dash, "pid_alive", return_value=False
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_dashboard_command(self._args(status=True), state_dir)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("another project", text)

    def test_stop_after_reuse_can_stop(self) -> None:
        """Pre-fix, the reusing caller's --stop said "no background
        dashboard is tracked here" and the server survived (it had never
        gotten a runfile). Post-fix, reuse writes one, so --stop works.

        Asserts on more than rc==0 + runfile-is-None: mutation-tested --
        deleting the runfile write in `_reuse` leaves those two vacuous
        (stop returns 0 and the never-written runfile is already None
        either way). Checking for "Stopped the background dashboard" in
        the output actually distinguishes the two (the no-runfile case
        prints "No background dashboard to stop" instead).

        pid_alive is mocked False for the --stop call specifically so this
        never sends a real SIGTERM: pid 424242 is not alive on this
        Windows box (pids are multiples of 4), but on Linux, with a large
        pid_max, it plausibly could be a live, unrelated process."""
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

            with patch.object(dash, "pid_alive", return_value=False):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    stop_rc = _run_dashboard_command(self._args(stop=True), state_dir)
        self.assertEqual(stop_rc, 0)
        self.assertIn("Stopped the background dashboard", out.getvalue())
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

    def test_background_own_runfile_reuse_is_host_gated(self) -> None:
        """A loopback dashboard tracked from an earlier plain --background
        must not be "reused" for a later --mobile request that needs an
        externally-reachable bind -- it must spawn a new, properly-bound
        server on the requested host instead."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            loopback_runfile = {
                "pid": 1111, "host": "127.0.0.1", "port": 8787,
                "url": "http://127.0.0.1:8787/",
            }
            spawned = MagicMock()
            spawned.pid = 2222
            spawned.poll.return_value = None
            mobile_runfile = {
                "pid": 2222, "host": "100.64.0.7", "port": 8787,
                "url": "http://100.64.0.7:8787/",
            }
            with patch.object(
                dash, "read_dashboard_runfile",
                side_effect=[loopback_runfile, mobile_runfile],
            ), patch.object(
                dash, "pid_alive", return_value=True
            ), patch.object(
                dash, "dashboard_serves", side_effect=[False, True]
            ), patch.object(
                dash, "stop_dashboard_pid"
            ) as stop_old, patch("subprocess.Popen", return_value=spawned) as popen:
                rc = _start_background_dashboard(
                    self._args(mobile=True), state_dir, "100.64.0.7",
                    port=8787, auto_port=True,
                    source="tailscale", allow_external=True,
                )
        popen.assert_called_once()
        stop_old.assert_called_once_with(1111)
        self.assertEqual(rc, 0)

    def test_mcp_dashboard_own_runfile_reuse_is_host_gated(self) -> None:
        """Same host-gating as test_background_own_runfile_reuse_is_host_gated,
        on the MCP tool's mirrored pre-check."""
        import puppetmaster.mcp_server as mcp

        loopback_runfile = {
            "pid": 3333, "host": "127.0.0.1", "port": 8787,
            "url": "http://127.0.0.1:8787/",
        }
        spawned = MagicMock()
        spawned.pid = 4444
        spawned.poll.return_value = None
        mobile_runfile = {
            "pid": 4444, "host": "100.64.0.7", "port": 8787,
            "url": "http://100.64.0.7:8787/",
        }
        with patch.object(
            dash, "resolve_mobile_host", return_value=("100.64.0.7", "tailscale")
        ), patch.object(
            dash, "read_dashboard_runfile",
            side_effect=[loopback_runfile, mobile_runfile],
        ), patch.object(
            dash, "pid_alive", return_value=True
        ), patch.object(
            dash, "dashboard_serves", side_effect=[False, True]
        ), patch.object(
            mcp, "_spawn_dashboard_server", return_value=spawned
        ) as popen, patch.object(
            dash, "stop_dashboard_pid"
        ) as stop_old, patch.object(
            dash, "write_qr_png", return_value=True
        ):
            result = mcp.call_tool(
                "puppetmaster_dashboard", {"cwd": "/tmp", "mobile": True}
            )
        popen.assert_called_once()
        stop_old.assert_called_once_with(3333)
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["started"])
        self.assertEqual(body["host"], "100.64.0.7")

    def test_status_uses_tracked_all_projects_not_invocation_flag(self) -> None:
        """`dashboard --status` (all_projects defaults False) after
        `dashboard --background --all-projects` must still report the board
        it started, not "serving another project" -- --status asks whether
        what its own runfile points at is still there, not whether that
        matches this invocation's flags."""
        from puppetmaster.cli._dispatch import _run_dashboard_command

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 5555, "host": "127.0.0.1", "port": 8787,
                "url": "http://127.0.0.1:8787/", "all_projects": True,
            }

            def _serves(host, port, sd, *, all_projects=False, timeout=1.0):
                return all_projects is True  # "matches" only when asked True

            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(dash, "dashboard_serves", side_effect=_serves):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_dashboard_command(
                        self._args(status=True, all_projects=False), state_dir
                    )
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("Background dashboard running", text)
        self.assertNotIn("another project", text)

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
            # None first: MCP's own-runfile pre-check must find nothing
            # tracked for this project before falling through to the
            # literal-port probe (mocked False -- a foreign dashboard) and
            # spawning its own. The child's runfile only appears once the
            # poll loop looks for it (second call).
            dash, "read_dashboard_runfile", side_effect=[None, child_runfile]
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        popen.assert_called_once()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["started"])

    def test_mcp_literal_port_reuse_writes_runfile(self) -> None:
        """A foreground (or otherwise untracked) dashboard that identifies
        as this project must persist a runfile so later stop=true can
        find it. CLI `_reuse` already wrote one; MCP did not."""
        import puppetmaster.mcp_server as mcp

        identity = {
            "pid": 8080,
            "state_dir_id": "abc",
            "service": "puppetmaster-dashboard",
        }
        with patch.object(
            dash, "read_dashboard_runfile", return_value=None
        ), patch.object(
            dash, "dashboard_serves", return_value=True
        ), patch.object(
            dash, "dashboard_identity", return_value=identity
        ), patch.object(
            dash, "write_dashboard_runfile"
        ) as write_runfile, patch.object(mcp, "_spawn_dashboard_server") as popen:
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        popen.assert_not_called()
        write_runfile.assert_called_once()
        self.assertEqual(write_runfile.call_args[0][1]["pid"], 8080)
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["already_running"])
        self.assertEqual(body["pid"], 8080)

    def test_mcp_literal_port_reuse_without_pid_spawns(self) -> None:
        """dashboard_serves True plus a raced-away identity must not persist
        a null-pid runfile (CLI `_reuse` already refuses that)."""
        import puppetmaster.mcp_server as mcp

        spawned = MagicMock()
        spawned.pid = 8081
        spawned.poll.return_value = None
        child_runfile = {
            "pid": 8081, "host": "127.0.0.1", "port": 8787,
            "url": "http://127.0.0.1:8787/",
        }
        with patch.object(
            dash, "dashboard_serves", side_effect=[True, True]
        ), patch.object(
            dash, "dashboard_identity", return_value=None
        ), patch.object(
            dash, "write_dashboard_runfile"
        ) as write_runfile, patch.object(
            dash, "read_dashboard_runfile", side_effect=[None, child_runfile]
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        write_runfile.assert_not_called()
        popen.assert_called_once()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["started"])


class ExplicitPortReuseGateTests(unittest.TestCase):
    """§F1 -- an explicit --port is a promise the caller may depend on (a
    script, a bookmark, a reverse proxy). Pre-fix, the own-runfile
    pre-check consulted neither `auto_port` nor whether the tracked port
    equalled the requested one, so a tracked dashboard on a *different*
    port than the one just asked for was "reused" anyway -- the explicit
    --port was silently discarded with exit 0 and nothing bound on the
    requested port. Both the CLI (`_start_background_dashboard`) and the
    MCP tool (`run_dashboard`) had the identical shape."""

    def _args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            port=None, port_search=False, host="127.0.0.1", no_open=True,
            allow_external=False, all_projects=False, mobile=False, qr=False,
            background=False, stop=False, status=False, write_runfile=False,
            job_id=None, backend="sqlite",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cli_explicit_port_mismatch_does_not_reuse(self) -> None:
        """The tracked dashboard genuinely IS this project's own -- it
        answers identity-true at its own (18801) port -- but the caller
        explicitly asked for a *different* port (18811). Reusing it anyway
        is exactly F1: the explicit --port gets silently discarded with
        exit 0 instead of being honored or strict-failed. Mocking
        dashboard_serves as a blanket False here would pass even on the
        unfixed code for the wrong reason (it'd never reuse *anything*),
        so this keys the mock on which port is being asked about."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 7001, "host": "127.0.0.1", "port": 18801,
                "url": "http://127.0.0.1:18801/",
            }
            spawned = MagicMock()
            spawned.pid = 7002
            spawned.poll.return_value = 0  # "exits" immediately -- this
            # test only cares whether a spawn was *attempted* (i.e. reuse
            # was correctly refused for the mismatched port), not whether
            # the fake child comes up.

            def _serves(host, port, sd, *, all_projects=False, timeout=1.0):
                return port == 18801  # only the tracked port identifies as ours

            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "pid_alive", return_value=True
            ), patch.object(
                dash, "dashboard_serves", side_effect=_serves
            ), patch.object(
                dash, "stop_dashboard_pid"
            ) as stop_old, patch("subprocess.Popen", return_value=spawned) as popen:
                _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=18811, auto_port=False,
                    source="loopback", allow_external=False,
                )
        popen.assert_called_once()
        stop_old.assert_not_called()

    def test_cli_explicit_port_matching_tracked_port_still_reuses(self) -> None:
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 7003, "host": "127.0.0.1", "port": 18802,
                "url": "http://127.0.0.1:18802/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "pid_alive", return_value=True
            ), patch.object(
                dash, "dashboard_serves", return_value=True
            ), patch("subprocess.Popen") as popen:
                rc = _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=18802, auto_port=False,
                    source="loopback", allow_external=False,
                )
        self.assertEqual(rc, 0)
        popen.assert_not_called()

    def test_cli_port_search_with_explicit_port_allows_reuse_at_different_port(self) -> None:
        """--port N --port-search derives auto_port=True (verified in
        DashboardArgvDerivationTests) -- an explicit port paired with
        --port-search is *not* a strict promise, so this must reuse a
        tracked dashboard even on a different port, same as a bare
        --background with no --port at all."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 7004, "host": "127.0.0.1", "port": 18803,
                "url": "http://127.0.0.1:18803/",
            }
            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "pid_alive", return_value=True
            ), patch.object(
                dash, "dashboard_serves", return_value=True
            ), patch("subprocess.Popen") as popen:
                rc = _start_background_dashboard(
                    self._args(), state_dir, "127.0.0.1",
                    port=18813, auto_port=True,
                    source="loopback", allow_external=False,
                )
        self.assertEqual(rc, 0)
        popen.assert_not_called()

    def test_mcp_explicit_port_mismatch_does_not_reuse(self) -> None:
        """Mirrors test_cli_explicit_port_mismatch_does_not_reuse's setup:
        dashboard_serves is keyed on the port it's asked about (True only
        for the tracked 18804) so this actually exercises the buggy shape
        rather than passing vacuously against a blanket False."""
        import puppetmaster.mcp_server as mcp

        tracked = {
            "pid": 7005, "host": "127.0.0.1", "port": 18804,
            "url": "http://127.0.0.1:18804/",
        }
        spawned = MagicMock()
        spawned.pid = 7006
        spawned.poll.return_value = 0  # "exits" immediately -- only the
        # spawn attempt (i.e. reuse correctly refused) matters here.

        def _serves(host, port, sd, *, all_projects=False, timeout=1.0):
            return port == 18804  # only the tracked port identifies as ours

        with patch.object(
            dash, "read_dashboard_runfile", return_value=tracked
        ), patch.object(
            dash, "pid_alive", return_value=True
        ), patch.object(
            dash, "dashboard_serves", side_effect=_serves
        ), patch.object(
            dash, "stop_dashboard_pid"
        ) as stop_old, patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp", "port": 18814})
        popen.assert_called_once()
        stop_old.assert_not_called()

    def test_mcp_explicit_port_matching_tracked_port_still_reuses(self) -> None:
        import puppetmaster.mcp_server as mcp

        tracked = {
            "pid": 7007, "host": "127.0.0.1", "port": 18805,
            "url": "http://127.0.0.1:18805/",
        }
        with patch.object(
            dash, "read_dashboard_runfile", return_value=tracked
        ), patch.object(
            dash, "pid_alive", return_value=True
        ), patch.object(
            dash, "dashboard_serves", return_value=True
        ), patch.object(mcp, "_spawn_dashboard_server") as popen:
            result = mcp.call_tool(
                "puppetmaster_dashboard", {"cwd": "/tmp", "port": 18805}
            )
        popen.assert_not_called()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["already_running"])
        self.assertEqual(body["port"], 18805)


class HostNormalizationReuseTests(unittest.TestCase):
    """§F2 -- a tracked runfile always records the literal address the
    child bound (e.g. "127.0.0.1"), never a loopback alias. Comparing that
    raw string against a fresh request's host *spelling* -- "localhost" --
    never matched, so a repeated --host localhost span up a brand-new
    server on every call and orphaned the rest (only the last spawn ever
    gets tracked)."""

    def _args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            port=None, port_search=False, host="localhost", no_open=True,
            allow_external=False, all_projects=False, mobile=False, qr=False,
            background=False, stop=False, status=False, write_runfile=False,
            job_id=None, backend="sqlite",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cli_localhost_request_reuses_127_0_0_1_tracked_dashboard(self) -> None:
        """The tracked dashboard is on a *different* (bumped) port than the
        one being requested here -- 18806 vs. 18800, mirroring the
        reviewer's repro where a blocker forces the child to bump past the
        literal requested port. That means the literal-port fallback probe
        (at 18800) can't rescue this: only the own-runfile pre-check, at
        the tracked 18806, can find it -- so this only passes if that
        check's host comparison is actually normalized. Keying the
        dashboard_serves mock on which port it's asked about (rather than
        a blanket True) is what makes that fallback path fail on unfixed
        code instead of masking the bug."""
        from puppetmaster.cli._dispatch import _start_background_dashboard

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            tracked = {
                "pid": 7101, "host": "127.0.0.1", "port": 18806,
                "url": "http://127.0.0.1:18806/",
            }

            def _serves(host, port, sd, *, all_projects=False, timeout=1.0):
                return port == 18806  # only the tracked (bumped) port

            with patch.object(
                dash, "read_dashboard_runfile", return_value=tracked
            ), patch.object(
                dash, "pid_alive", return_value=True
            ), patch.object(
                dash, "dashboard_serves", side_effect=_serves
            ), patch("subprocess.Popen") as popen:
                rc = _start_background_dashboard(
                    self._args(), state_dir, "localhost",
                    port=18800, auto_port=True,
                    source="loopback", allow_external=False,
                )
        self.assertEqual(rc, 0)
        popen.assert_not_called()

    def test_mcp_localhost_request_reuses_127_0_0_1_tracked_dashboard(self) -> None:
        """MCP always requests host="127.0.0.1" for a non-mobile call
        (mcp_server.py never sends "localhost" itself), but an
        older/hand-edited runfile could still carry "localhost" from a
        different tool version -- the comparison must fold that too. As
        above, the tracked (bumped) port 18807 differs from the default
        8787 this call implicitly asks for, so only the own-runfile
        pre-check -- not the literal-port fallback -- can find it."""
        import puppetmaster.mcp_server as mcp

        tracked = {
            "pid": 7102, "host": "localhost", "port": 18807,
            "url": "http://localhost:18807/",
        }

        def _serves(host, port, sd, *, all_projects=False, timeout=1.0):
            return port == 18807  # only the tracked (bumped) port

        spawned = MagicMock()
        spawned.pid = 7103
        spawned.poll.return_value = 0
        with patch.object(
            dash, "read_dashboard_runfile", return_value=tracked
        ), patch.object(
            dash, "pid_alive", return_value=True
        ), patch.object(
            dash, "dashboard_serves", side_effect=_serves
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            # No explicit port -- auto_port=True, requested port defaults to
            # 8787, which differs from the tracked (bumped) 18807.
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        popen.assert_not_called()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["already_running"])


class DashboardArgvDerivationTests(unittest.TestCase):
    """§F7 -- `_run_dashboard_command`'s explicit/auto_port derivation
    (the riskiest edit in this branch) had zero coverage: every identity
    test above pre-derives port/auto_port directly rather than parsing
    argv, and `_run_dashboard_command` is otherwise only ever driven with
    status=True/stop=True, both of which return before touching this
    logic at all. Drives the real argv -> parser -> serve() kwargs path."""

    def _serve_kwargs(self, argv: list[str]) -> dict:
        from puppetmaster.cli._dispatch import _run_dashboard_command
        from puppetmaster.cli._parser import build_parser

        args = build_parser().parse_args(argv)
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            with patch(
                "puppetmaster.dashboard.serve", return_value=MagicMock()
            ) as serve_mock, patch.object(
                dash, "resolve_mobile_host", return_value=("100.64.0.7", "tailscale")
            ):
                _run_dashboard_command(args, state_dir)
        return serve_mock.call_args.kwargs

    def test_no_port_flag_auto_bumps_from_8787(self) -> None:
        kwargs = self._serve_kwargs(["dashboard"])
        self.assertEqual(kwargs["port"], 8787)
        self.assertTrue(kwargs["auto_port"])

    def test_explicit_port_is_strict(self) -> None:
        kwargs = self._serve_kwargs(["dashboard", "--port", "18777"])
        self.assertEqual(kwargs["port"], 18777)
        self.assertFalse(kwargs["auto_port"])

    def test_explicit_port_with_port_search_is_not_strict(self) -> None:
        kwargs = self._serve_kwargs(
            ["dashboard", "--port", "18777", "--port-search"]
        )
        self.assertEqual(kwargs["port"], 18777)
        self.assertTrue(kwargs["auto_port"])

    def test_bare_port_search_auto_bumps_from_8787(self) -> None:
        kwargs = self._serve_kwargs(["dashboard", "--port-search"])
        self.assertEqual(kwargs["port"], 8787)
        self.assertTrue(kwargs["auto_port"])

    def test_mobile_without_port_auto_bumps_from_8787(self) -> None:
        kwargs = self._serve_kwargs(["dashboard", "--mobile"])
        self.assertEqual(kwargs["port"], 8787)
        self.assertTrue(kwargs["auto_port"])


class RealServerIdentityTests(unittest.TestCase):
    """§F7 -- every test above (and every test in this file before this
    class) mocks `dashboard_serves`/`dashboard_identity` out entirely.
    Mutation-tested: replacing `dashboard_serves` with the pre-fix bare
    `dashboard_alive` leaves all of them green. The identity *function*
    itself is guarded separately (test_dashboard_ports.py's
    DashboardServesTests, against real bound servers) -- what's missing is
    coverage of the CLI wiring actually calling it for real. These two
    exercise `_start_background_dashboard`'s own-runfile pre-check against
    a real, live dashboard with nothing mocked but the eventual subprocess
    spawn."""

    def test_reuses_real_own_dashboard(self) -> None:
        from puppetmaster.cli._dispatch import _start_background_dashboard
        from puppetmaster.dashboard import serve, write_dashboard_runfile
        from puppetmaster.store_factory import create_store

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            create_store("sqlite", state_dir).init()
            httpd = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=0,
                open_browser=False, serve_forever=False,
            )
            self.addCleanup(httpd.server_close)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(httpd.shutdown)
            port = httpd.server_address[1]
            write_dashboard_runfile(state_dir, {
                "pid": os.getpid(), "host": "127.0.0.1", "port": port,
                "url": f"http://127.0.0.1:{port}/",
            })

            args = argparse.Namespace(
                port=None, port_search=False, host="127.0.0.1", no_open=True,
                allow_external=False, all_projects=False, mobile=False,
                qr=False, background=True, stop=False, status=False,
                write_runfile=False, job_id=None, backend="sqlite",
            )
            with patch("subprocess.Popen") as popen:
                rc = _start_background_dashboard(
                    args, state_dir, "127.0.0.1",
                    port=port, auto_port=True,
                    source="loopback", allow_external=False,
                )
        popen.assert_not_called()
        self.assertEqual(rc, 0)

    def test_does_not_reuse_real_foreign_dashboard(self) -> None:
        """dir_b's runfile claims dir_a's real, live server (a stale
        runfile, or -- pre-fix -- exactly the identity-blind bug this
        whole branch fixes). Real dashboard_serves/dashboard_identity must
        refuse it and fall through to spawning dir_b's own."""
        from puppetmaster.cli._dispatch import _start_background_dashboard
        from puppetmaster.dashboard import serve, write_dashboard_runfile
        from puppetmaster.store_factory import create_store

        with TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "a"
            dir_b = Path(tmp) / "b"
            create_store("sqlite", dir_a).init()
            create_store("sqlite", dir_b).init()
            httpd = serve(
                dir_a, backend="sqlite", host="127.0.0.1", port=0,
                open_browser=False, serve_forever=False,
            )
            self.addCleanup(httpd.server_close)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(httpd.shutdown)
            port = httpd.server_address[1]
            write_dashboard_runfile(dir_b, {
                "pid": os.getpid(), "host": "127.0.0.1", "port": port,
                "url": f"http://127.0.0.1:{port}/",
            })

            spawned = MagicMock()
            spawned.pid = 8080
            spawned.poll.return_value = 0  # "exits" immediately -- this
            # test only cares whether a spawn was *attempted* (i.e. reuse
            # was correctly refused), not whether the fake child comes up.
            args = argparse.Namespace(
                port=None, port_search=False, host="127.0.0.1", no_open=True,
                allow_external=False, all_projects=False, mobile=False,
                qr=False, background=True, stop=False, status=False,
                write_runfile=False, job_id=None, backend="sqlite",
            )
            with patch("subprocess.Popen", return_value=spawned) as popen:
                _start_background_dashboard(
                    args, dir_b, "127.0.0.1",
                    port=port, auto_port=True,
                    source="loopback", allow_external=False,
                )
        popen.assert_called_once()


class ChildStderrTailTests(unittest.TestCase):
    """`--background`'s failure message used to be generic no matter what
    actually went wrong, because the child's stderr was DEVNULL'd outright.
    It's now captured to a file and quoted on failure -- these guard the
    small helper that reads it back."""

    def test_missing_file_returns_empty_string(self) -> None:
        from puppetmaster.cli._dispatch import _read_child_stderr_tail

        with TemporaryDirectory() as tmp:
            self.assertEqual(
                _read_child_stderr_tail(Path(tmp) / "does-not-exist.log"), ""
            )

    def test_empty_file_returns_empty_string(self) -> None:
        from puppetmaster.cli._dispatch import _read_child_stderr_tail

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.log"
            path.write_text("", encoding="utf-8")
            self.assertEqual(_read_child_stderr_tail(path), "")

    def test_returns_captured_content(self) -> None:
        from puppetmaster.cli._dispatch import _read_child_stderr_tail

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "err.log"
            path.write_text(
                "Traceback (most recent call last):\nOSError: port in use\n",
                encoding="utf-8",
            )
            tail = _read_child_stderr_tail(path)
        self.assertIn("OSError: port in use", tail)

    def test_caps_to_max_bytes_from_the_end(self) -> None:
        """A tail, not a head -- the actual error is usually the last
        thing written, so a giant traceback should still surface it."""
        from puppetmaster.cli._dispatch import _read_child_stderr_tail

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.log"
            path.write_text("x" * 10000 + "THE REAL ERROR", encoding="utf-8")
            tail = _read_child_stderr_tail(path, max_bytes=100)
        self.assertLessEqual(len(tail), 100)
        self.assertIn("THE REAL ERROR", tail)


if __name__ == "__main__":
    unittest.main()
