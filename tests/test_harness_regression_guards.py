"""Regression guards: stdio MCP surface + adapter inventory must not shrink.

These fail closed if a future change drops Cursor/stdio tools, removes a
shipped adapter, or accidentally registers a fake ``grok-bot`` worker adapter.
They intentionally do **not** hash source files — import + list contracts are
the durable check (hashes churn on every comment).
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from puppetmaster.diagnostics import adapter_status
from puppetmaster.mcp_remote import IMPLEMENT_TOOL_NAMES, filtered_tools
from puppetmaster.mcp_server import handle_message, tools
from puppetmaster.platform_lock import KNOWN_ADAPTERS


# Floor for the stdio tool surface. Bump when tools are added; never lower
# without an explicit product decision (and a CHANGELOG note).
STDIO_TOOL_FLOOR = 45

REQUIRED_STDIO_TOOLS = frozenset(
    {
        "puppetmaster_doctor",
        "puppetmaster_start_cursor_review",
        "puppetmaster_start_review",
        "puppetmaster_start_cursor_plan",
        "puppetmaster_start_cursor_swarm",
        "puppetmaster_start_swarm",
        "puppetmaster_start_implement",
        "puppetmaster_edit",
        "puppetmaster_status",
        "puppetmaster_logs",
        "puppetmaster_live_artifacts",
        "puppetmaster_partial_summary",
        "puppetmaster_artifacts",
        "puppetmaster_show",
        "puppetmaster_codegraph_search",
        "puppetmaster_codegraph_context",
    }
)

REQUIRED_ADAPTERS = frozenset(
    {
        "cursor",
        "claude-code",
        "openai",
        "codex",
        "hermes",
        "antigravity",
        "agentic",
        "shell",
    }
)

FORBIDDEN_ADAPTERS = frozenset({"grok-bot", "grok_bot", "grokbot"})


class StdioMcpSurfaceGuards(unittest.TestCase):
    def test_stdio_tools_list_includes_doctor_and_implement(self) -> None:
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response is not None
        names = {tool["name"] for tool in response["result"]["tools"]}
        missing = sorted(REQUIRED_STDIO_TOOLS - names)
        self.assertEqual(missing, [], f"stdio MCP missing required tools: {missing}")
        self.assertGreaterEqual(
            len(names),
            STDIO_TOOL_FLOOR,
            f"stdio tool surface shrank below floor {STDIO_TOOL_FLOOR} (got {len(names)})",
        )
        # Implement verbs that remote supervise omits must still exist on stdio.
        for name in sorted(IMPLEMENT_TOOL_NAMES):
            self.assertIn(name, names, f"stdio must still expose {name}")

    def test_stdio_subprocess_tools_list_matches_inprocess(self) -> None:
        """Cursor launch path: ``python -m puppetmaster.mcp_server`` over stdio."""
        request = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        ) + "\n"
        completed = subprocess.run(
            [sys.executable, "-m", "puppetmaster.mcp_server"],
            input=request,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertTrue(
            completed.stdout.strip(),
            f"stdio MCP returned no stdout; stderr={completed.stderr[-400:]!r}",
        )
        line = next(
            (ln.strip() for ln in completed.stdout.splitlines() if ln.strip().startswith("{")),
            "",
        )
        self.assertTrue(line, "stdio MCP returned no JSON frame")
        payload = json.loads(line)
        names = {tool["name"] for tool in payload["result"]["tools"]}
        self.assertIn("puppetmaster_doctor", names)
        self.assertIn("puppetmaster_start_implement", names)
        self.assertEqual(len(names), len(tools()))

    def test_remote_supervise_is_strict_subset_of_stdio(self) -> None:
        stdio = {tool.name for tool in tools()}
        remote = {tool.name for tool in filtered_tools("supervise")}
        self.assertTrue(remote.issubset(stdio))
        self.assertTrue(IMPLEMENT_TOOL_NAMES.issubset(stdio))
        self.assertTrue(IMPLEMENT_TOOL_NAMES.isdisjoint(remote))


class AdapterInventoryGuards(unittest.TestCase):
    def test_platform_lock_known_adapters_exclude_grok_bot(self) -> None:
        known = set(KNOWN_ADAPTERS)
        # Platform lock covers billable host adapters (not local/shell demos).
        for name in (
            "cursor",
            "claude-code",
            "openai",
            "codex",
            "hermes",
            "antigravity",
            "agentic",
        ):
            self.assertIn(name, known)
        self.assertNotIn("agy", known)
        leaked = sorted(FORBIDDEN_ADAPTERS & known)
        self.assertEqual(leaked, [], f"forbidden adapter names in KNOWN_ADAPTERS: {leaked}")

    def test_adapter_status_inventory_includes_shipped_set(self) -> None:
        from pathlib import Path

        names = {row["name"] for row in adapter_status(Path.cwd())}
        missing = sorted(REQUIRED_ADAPTERS - names)
        self.assertEqual(missing, [], f"adapter_status missing: {missing}")
        leaked = sorted(FORBIDDEN_ADAPTERS & names)
        self.assertEqual(leaked, [], f"adapter_status leaked: {leaked}")

    def test_cli_adapters_json_matches_contract(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "puppetmaster", "adapters"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = json.loads(completed.stdout)
        names = {row["name"] for row in rows}
        missing = sorted(REQUIRED_ADAPTERS - names)
        self.assertEqual(missing, [], f"`puppetmaster adapters` missing: {missing}")
        leaked = sorted(FORBIDDEN_ADAPTERS & names)
        self.assertEqual(leaked, [], f"`puppetmaster adapters` leaked: {leaked}")


class McpServerImportContract(unittest.TestCase):
    def test_mcp_server_exports_handlers_remote_reuses(self) -> None:
        import puppetmaster.mcp_server as mcp_server
        import puppetmaster.mcp_remote as mcp_remote

        for attr in ("handle_message", "tools", "tool_to_json", "call_tool", "main"):
            self.assertTrue(hasattr(mcp_server, attr), attr)
        # Remote is additive — stdio entry point must remain the Cursor path.
        self.assertEqual(mcp_server.main.__module__, "puppetmaster.mcp_server")
        self.assertEqual(mcp_remote.main.__module__, "puppetmaster.mcp_remote")
        self.assertNotEqual(mcp_server.main, mcp_remote.main)


if __name__ == "__main__":
    unittest.main()
