"""Wave 4: cross-platform process-tree teardown + reaper semantics."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import subprocess
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

from puppetmaster.adapters._streaming import _kill_process_tree
from puppetmaster.win_process import (
    _descendant_pids_from_map,
    _taskkill_process_tree,
    kill_process_tree,
)

class WinProcessTreeUnitTests(unittest.TestCase):
    def test_kill_process_tree_noop_off_windows_or_invalid_pid(self) -> None:
        with patch("puppetmaster.win_process.os.name", "posix"):
            self.assertFalse(kill_process_tree(1234))
        with patch("puppetmaster.win_process.os.name", "nt"):
            self.assertFalse(kill_process_tree(0))
            self.assertFalse(kill_process_tree(-1))

    def test_taskkill_tree_invokes_force_tree_flags(self) -> None:
        completed = SimpleNamespace(returncode=0)
        with patch("puppetmaster.win_process.shutil.which", return_value="taskkill.exe"), patch(
            "puppetmaster.win_process.subprocess.run", return_value=completed
        ) as run, patch(
            "puppetmaster.win_process._taskkill_creationflags", return_value=0x08000000
        ):
            self.assertTrue(_taskkill_process_tree(4242))
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["taskkill.exe", "/F", "/T", "/PID", "4242"])
        self.assertEqual(kwargs["creationflags"], 0x08000000)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

    def test_taskkill_treats_already_gone_as_success(self) -> None:
        completed = SimpleNamespace(returncode=128)
        with patch("puppetmaster.win_process.shutil.which", return_value="taskkill.exe"), patch(
            "puppetmaster.win_process.subprocess.run", return_value=completed
        ), patch(
            "puppetmaster.win_process._taskkill_creationflags", return_value=0
        ):
            self.assertTrue(_taskkill_process_tree(99))

    def test_taskkill_missing_binary_returns_false(self) -> None:
        with patch("puppetmaster.win_process.shutil.which", return_value=None):
            self.assertFalse(_taskkill_process_tree(99))

    def test_kill_process_tree_falls_back_to_toolhelp_when_taskkill_unavailable(
        self,
    ) -> None:
        with patch("puppetmaster.win_process.os.name", "nt"), patch(
            "puppetmaster.win_process._taskkill_process_tree", return_value=False
        ) as taskkill, patch(
            "puppetmaster.win_process._toolhelp_kill_process_tree", return_value=True
        ) as toolhelp:
            self.assertTrue(kill_process_tree(777))
        taskkill.assert_called_once_with(777)
        toolhelp.assert_called_once_with(777)

    def test_kill_process_tree_safe_when_both_methods_unavailable(self) -> None:
        with patch("puppetmaster.win_process.os.name", "nt"), patch(
            "puppetmaster.win_process._taskkill_process_tree", return_value=False
        ), patch(
            "puppetmaster.win_process._toolhelp_kill_process_tree", return_value=False
        ):
            self.assertFalse(kill_process_tree(777))

    def test_descendant_ordering_is_deepest_first_then_root(self) -> None:
        # root 1 -> 2 -> 3, and 1 -> 4
        tree = {1: [2, 4], 2: [3]}
        self.assertEqual(_descendant_pids_from_map(1, tree), [3, 2, 4, 1])

    def test_descendant_walk_is_cycle_safe(self) -> None:
        # PID-reuse style cycle: 2 -> 3 -> 2, plus a self-parent edge.
        tree = {1: [2], 2: [3], 3: [2, 4], 4: [4]}
        self.assertEqual(_descendant_pids_from_map(1, tree), [4, 3, 2, 1])

class KillProcessTreeDispatchTests(unittest.TestCase):
    def _fake_process(self, pid: int = 5555) -> MagicMock:
        process = MagicMock()
        process.pid = pid
        return process

    def test_posix_new_session_uses_killpg(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "posix"), patch(
            "puppetmaster.adapters._streaming.signal.SIGKILL", 9, create=True
        ), patch(
            "puppetmaster.adapters._streaming.os.getpgid",
            create=True,
            return_value=9001,
        ) as getpgid, patch(
            "puppetmaster.adapters._streaming.os.killpg", create=True
        ) as killpg, patch(
            "puppetmaster.win_process.kill_process_tree"
        ) as win_kill:
            _kill_process_tree(process, started_new_session=True)
        getpgid.assert_called_once_with(5555)
        killpg.assert_called_once_with(9001, 9)
        process.kill.assert_not_called()
        win_kill.assert_not_called()

    def test_windows_new_session_uses_win_process_tree(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "nt"), patch(
            "puppetmaster.win_process.kill_process_tree", return_value=True
        ) as win_kill, patch(
            "puppetmaster.adapters._streaming.os.killpg", create=True
        ) as killpg:
            _kill_process_tree(process, started_new_session=True)
        win_kill.assert_called_once_with(5555)
        process.kill.assert_not_called()
        killpg.assert_not_called()

    def test_windows_tree_kill_failure_falls_back_to_direct_kill(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "nt"), patch(
            "puppetmaster.win_process.kill_process_tree", return_value=False
        ):
            _kill_process_tree(process, started_new_session=True)
        process.kill.assert_called_once()

    def test_windows_without_new_session_still_tree_kills(self) -> None:
        # Cursor/Node descendants must be reaped even when the adapter did not
        # launch with start_new_session (POSIX stays conservative).
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "nt"), patch(
            "puppetmaster.win_process.kill_process_tree", return_value=True
        ) as win_kill, patch(
            "puppetmaster.adapters._streaming.os.killpg", create=True
        ) as killpg:
            _kill_process_tree(process, started_new_session=False)
        win_kill.assert_called_once_with(5555)
        process.kill.assert_not_called()
        killpg.assert_not_called()

    def test_windows_without_new_session_falls_back_when_tree_kill_fails(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "nt"), patch(
            "puppetmaster.win_process.kill_process_tree", return_value=False
        ):
            _kill_process_tree(process, started_new_session=False)
        process.kill.assert_called_once()

    def test_posix_without_new_session_skips_killpg(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "posix"), patch(
            "puppetmaster.adapters._streaming.os.killpg", create=True
        ) as killpg:
            _kill_process_tree(process, started_new_session=False)
        killpg.assert_not_called()
        process.kill.assert_called_once()

    def test_killpg_failure_falls_back_to_direct_kill(self) -> None:
        process = self._fake_process()
        with patch("puppetmaster.adapters._streaming.os.name", "posix"), patch(
            "puppetmaster.adapters._streaming.signal.SIGKILL", 9, create=True
        ), patch(
            "puppetmaster.adapters._streaming.os.getpgid",
            create=True,
            return_value=1,
        ), patch(
            "puppetmaster.adapters._streaming.os.killpg",
            create=True,
            side_effect=ProcessLookupError,
        ):
            _kill_process_tree(process, started_new_session=True)
        process.kill.assert_called_once()

class McpAsyncReaperSemanticsTests(unittest.TestCase):
    """Preserve MCP launcher reaper behavior while Wave 4 lands beside it."""

    def test_reaper_drops_exited_launchers_keeps_running(self) -> None:
        from puppetmaster import mcp_server

        class _FakeLauncher:
            def __init__(self, exited: bool) -> None:
                self._exited = exited
                self.poll_calls = 0

            def poll(self) -> Optional[int]:
                self.poll_calls += 1
                return 0 if self._exited else None

        done_a, done_b, running = (
            _FakeLauncher(True),
            _FakeLauncher(True),
            _FakeLauncher(False),
        )
        with mcp_server._ASYNC_PROCESSES_LOCK:
            saved = list(mcp_server.ASYNC_PROCESSES)
            mcp_server.ASYNC_PROCESSES[:] = [done_a, running, done_b]
        try:
            reaped = mcp_server._reap_async_processes()
            self.assertEqual(reaped, 2)
            self.assertEqual(mcp_server.ASYNC_PROCESSES, [running])
            self.assertGreaterEqual(running.poll_calls, 1)
        finally:
            with mcp_server._ASYNC_PROCESSES_LOCK:
                mcp_server.ASYNC_PROCESSES[:] = saved

if __name__ == "__main__":
    unittest.main()
