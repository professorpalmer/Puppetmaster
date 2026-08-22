"""Focused regression coverage for opt-in MCP lifecycle cleanup."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest


@pytest.fixture
def registry_dir():
    with TemporaryDirectory() as directory:
        with patch.dict(os.environ, {"PUPPETMASTER_MCP_REGISTRY_DIR": directory}):
            yield Path(directory)


def test_activity_is_persisted_and_surfaced(registry_dir):
    """Arrange/Act/Assert: inbound activity survives a registry reread."""
    from puppetmaster import mcp_registry

    path = mcp_registry.register(pid=os.getpid(), workspace=str(registry_dir / "repo"))
    inbound_at = time.time() - 12

    assert mcp_registry.record_activity(path, inbound_at=inbound_at, active_tool_calls=2)

    entry = mcp_registry.list_entries()[0]
    payload = entry.to_payload(now=inbound_at + 15)
    assert entry.last_inbound_at == pytest.approx(inbound_at)
    assert entry.active_tool_calls == 2
    assert payload["inbound_age_seconds"] == pytest.approx(15, abs=0.01)
    assert payload["active_tool_calls"] == 2


def test_server_request_activity_updates_the_registry(registry_dir):
    """Arrange/Act/Assert: the MCP request lifecycle writes durable activity."""
    from puppetmaster import mcp_registry, mcp_server

    path = mcp_registry.register(pid=os.getpid(), workspace=str(registry_dir / "repo"))
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

    assert active.last_inbound_at is not None
    assert active.active_tool_calls == 1
    assert finished.active_tool_calls == 0


def test_selector_intersection_normalizes_workspace_and_requires_every_selector(registry_dir):
    """Arrange/Act/Assert: PID, workspace, and idle selectors intersect."""
    from puppetmaster import mcp_registry

    workspace = registry_dir / "project"
    other_workspace = registry_dir / "other"
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

    assert [entry.pid for entry in result.killed] == [41001]
    assert signalled == [41001, 41001]
    assert mcp_registry.normalize_workspace(str(workspace / ".")) == mcp_registry.normalize_workspace(str(workspace))


def test_selector_cleanup_refuses_self_and_servers_with_active_calls(registry_dir):
    """Arrange/Act/Assert: explicit cleanup never kills self or active work."""
    from puppetmaster import mcp_registry

    self_path = mcp_registry.register(pid=os.getpid(), workspace=str(registry_dir / "self"))
    active_path = mcp_registry.register(pid=42002, workspace=str(registry_dir / "active"))
    mcp_registry.record_activity(self_path, active_tool_calls=1)
    mcp_registry.record_activity(active_path, active_tool_calls=1)

    with patch.object(mcp_registry, "_pid_alive", return_value=True), patch.object(mcp_registry.os, "kill") as kill:
        result = mcp_registry.kill_selected(
            pids=[os.getpid(), 42002], self_pid=os.getpid(), grace_seconds=0
        )

    assert result.killed == []
    assert {(row["pid"], row["reason"]) for row in result.refused} == {
        (os.getpid(), "self"),
        (42002, "active_tool_calls"),
    }
    kill.assert_not_called()


def test_cli_cleanup_wires_explicit_selectors(registry_dir):
    """Arrange/Act/Assert: CLI forwards all selectors to precise cleanup."""
    from puppetmaster import cli
    from puppetmaster import mcp_registry
    from puppetmaster.cli import commands_mcp

    result = mcp_registry.McpSelectionResult(killed=[], refused=[])
    output = io.StringIO()
    with patch.object(commands_mcp, "registry_kill_selected", return_value=result) as selected, patch(
        "sys.stdout", output
    ):
        assert cli.main(
            [
                "mcp",
                "cleanup",
                "--pid",
                "42101",
                "--workspace",
                str(registry_dir / "repo"),
                "--idle-after-seconds",
                "45",
                "--json",
            ]
        ) == 0

    selected.assert_called_once_with(
        pids=[42101],
        workspace=str(registry_dir / "repo"),
        idle_after_seconds=45.0,
        self_pid=os.getpid(),
    )
    assert json.loads(output.getvalue())["selected_killed"] == []


def test_mcp_cleanup_wires_explicit_selectors_without_default_kill(registry_dir):
    """Arrange/Act/Assert: MCP selectors opt in; no selector is read-only."""
    from puppetmaster import mcp_registry, mcp_server

    result = mcp_registry.McpSelectionResult(killed=[], refused=[])
    with patch.object(mcp_server, "registry_kill_selected", return_value=result) as selected:
        response = mcp_server.run_mcp_cleanup(
            {"pids": [42201], "workspace": str(registry_dir / "repo"), "idle_after_seconds": 30}
        )
    payload = json.loads(response["content"][0]["text"])
    selected.assert_called_once_with(
        pids=[42201],
        workspace=str(registry_dir / "repo"),
        idle_after_seconds=30.0,
        self_pid=os.getpid(),
    )
    assert payload["selected_killed"] == []
