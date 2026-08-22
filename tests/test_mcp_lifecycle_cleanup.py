"""Focused regression coverage for opt-in MCP lifecycle cleanup."""

from __future__ import annotations

import io
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class McpLifecycleCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.registry_dir = Path(self._temporary_directory.name)
        self._environment = patch.dict(
            os.environ,
            {"PUPPETMASTER_MCP_REGISTRY_DIR": str(self.registry_dir)},
        )
        self._environment.start()

    def tearDown(self) -> None:
        self._environment.stop()
        self._temporary_directory.cleanup()

    def test_activity_is_persisted_and_surfaced(self) -> None:
        """Arrange/Act/Assert: inbound activity survives a registry reread."""
        from puppetmaster import mcp_registry

        path = mcp_registry.register(
            pid=os.getpid(), workspace=str(self.registry_dir / "repo")
        )
        inbound_at = time.time() - 12

        self.assertTrue(
            mcp_registry.record_activity(
                path, inbound_at=inbound_at, active_tool_calls=2
            )
        )

        entry = mcp_registry.list_entries()[0]
        payload = entry.to_payload(now=inbound_at + 15)
        self.assertAlmostEqual(entry.last_inbound_at, inbound_at)
        self.assertEqual(entry.active_tool_calls, 2)
        self.assertAlmostEqual(payload["inbound_age_seconds"], 15, delta=0.01)
        self.assertEqual(payload["active_tool_calls"], 2)

    def test_server_request_activity_updates_the_registry(self) -> None:
        """Arrange/Act/Assert: the MCP request lifecycle writes durable activity."""
        from puppetmaster import mcp_registry, mcp_server

        path = mcp_registry.register(
            pid=os.getpid(), workspace=str(self.registry_dir / "repo")
        )
        with mcp_server._INPUT_STATE_LOCK:
            original_path = mcp_server._REGISTRY_ACTIVITY_PATH
            original_active = mcp_server._ACTIVE_TOOL_CALLS
            mcp_server._REGISTRY_ACTIVITY_PATH = path
            mcp_server._ACTIVE_TOOL_CALLS = 0
        try:
            mcp_server._mark_inbound_message()
            mcp_server._tool_call_started()
            active = mcp_registry.list_entries()[0]
            mcp_server._tool_call_finished()
            finished = mcp_registry.list_entries()[0]
        finally:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._REGISTRY_ACTIVITY_PATH = original_path
                mcp_server._ACTIVE_TOOL_CALLS = original_active

        self.assertIsNotNone(active.last_inbound_at)
        self.assertEqual(active.active_tool_calls, 1)
        self.assertEqual(finished.active_tool_calls, 0)

    def test_selector_intersection_normalizes_workspace_and_requires_every_selector(
        self,
    ) -> None:
        """Arrange/Act/Assert: PID, workspace, and idle selectors intersect."""
        from puppetmaster import mcp_registry

        workspace = self.registry_dir / "project"
        other_workspace = self.registry_dir / "other"
        first = mcp_registry.register(pid=41001, workspace=str(workspace))
        second = mcp_registry.register(pid=41002, workspace=str(other_workspace))
        old = time.time() - 60
        mcp_registry.record_activity(first, inbound_at=old, active_tool_calls=0)
        mcp_registry.record_activity(second, inbound_at=old, active_tool_calls=0)
        signalled = []

        with patch.object(mcp_registry, "_pid_alive", return_value=True), patch.object(
            mcp_registry.os, "kill", side_effect=lambda pid, sig: signalled.append(pid)
        ):
            result = mcp_registry.kill_selected(
                pids=[41001, 41002],
                workspace=str(workspace / "."),
                idle_after_seconds=30,
                self_pid=99999,
                grace_seconds=0,
            )

        self.assertEqual([entry.pid for entry in result.killed], [41001])
        self.assertEqual(signalled, [41001, 41001])
        self.assertEqual(
            mcp_registry.normalize_workspace(str(workspace / ".")),
            mcp_registry.normalize_workspace(str(workspace)),
        )

    def test_selector_cleanup_refuses_self_and_servers_with_active_calls(
        self,
    ) -> None:
        """Arrange/Act/Assert: explicit cleanup never kills self or active work."""
        from puppetmaster import mcp_registry

        self_path = mcp_registry.register(
            pid=os.getpid(), workspace=str(self.registry_dir / "self")
        )
        active_path = mcp_registry.register(
            pid=42002, workspace=str(self.registry_dir / "active")
        )
        mcp_registry.record_activity(self_path, active_tool_calls=1)
        mcp_registry.record_activity(active_path, active_tool_calls=1)

        with patch.object(mcp_registry, "_pid_alive", return_value=True), patch.object(
            mcp_registry.os, "kill"
        ) as kill:
            result = mcp_registry.kill_selected(
                pids=[os.getpid(), 42002], self_pid=os.getpid(), grace_seconds=0
            )

        self.assertEqual(result.killed, [])
        self.assertEqual(
            {(row["pid"], row["reason"]) for row in result.refused},
            {(os.getpid(), "self"), (42002, "active_tool_calls")},
        )
        kill.assert_not_called()

    def test_cli_cleanup_wires_explicit_selectors(self) -> None:
        """Arrange/Act/Assert: CLI forwards all selectors to precise cleanup."""
        from puppetmaster import cli, mcp_registry
        from puppetmaster.cli import commands_mcp

        result = mcp_registry.McpSelectionResult(killed=[], refused=[])
        output = io.StringIO()
        with patch.object(
            commands_mcp, "registry_kill_selected", return_value=result
        ) as selected, patch("sys.stdout", output):
            exit_code = cli.main(
                [
                    "mcp",
                    "cleanup",
                    "--pid",
                    "42101",
                    "--workspace",
                    str(self.registry_dir / "repo"),
                    "--idle-after-seconds",
                    "45",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        selected.assert_called_once_with(
            pids=[42101],
            workspace=str(self.registry_dir / "repo"),
            idle_after_seconds=45.0,
            self_pid=os.getpid(),
        )
        self.assertEqual(json.loads(output.getvalue())["selected_killed"], [])

    def test_mcp_cleanup_wires_explicit_selectors_without_default_kill(self) -> None:
        """Arrange/Act/Assert: MCP selectors opt in; no selector is read-only."""
        from puppetmaster import mcp_registry, mcp_server

        result = mcp_registry.McpSelectionResult(killed=[], refused=[])
        with patch.object(
            mcp_server, "registry_kill_selected", return_value=result
        ) as selected:
            response = mcp_server.run_mcp_cleanup(
                {
                    "pids": [42201],
                    "workspace": str(self.registry_dir / "repo"),
                    "idle_after_seconds": 30,
                }
            )

        payload = json.loads(response["content"][0]["text"])
        selected.assert_called_once_with(
            pids=[42201],
            workspace=str(self.registry_dir / "repo"),
            idle_after_seconds=30.0,
            self_pid=os.getpid(),
        )
        self.assertEqual(payload["selected_killed"], [])


if __name__ == "__main__":
    unittest.main()
