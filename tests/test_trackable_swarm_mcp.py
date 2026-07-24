"""Marionette trackable-swarm MCP refuse (env-gated)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from puppetmaster.mcp_server import call_tool, trackable_swarm_mcp_blocked


class TrackableSwarmMcpTests(unittest.TestCase):
    def test_blocked_when_env_set(self) -> None:
        with patch.dict("os.environ", {"MARIONETTE_TRACKABLE_SWARMS": "1"}, clear=False):
            blocked = trackable_swarm_mcp_blocked("puppetmaster_start_cursor_swarm")
        self.assertIsNotNone(blocked)
        assert blocked is not None
        self.assertTrue(blocked.get("isError"))
        body = json.loads(blocked["content"][0]["text"])
        self.assertEqual(body.get("code"), "marionette_trackable_swarm")

    def test_sync_implement_also_blocked(self) -> None:
        with patch.dict("os.environ", {"MARIONETTE_TRACKABLE_SWARMS": "1"}, clear=False):
            blocked = trackable_swarm_mcp_blocked("puppetmaster_cursor_implement")
        self.assertIsNotNone(blocked)

    def test_codegraph_allowed(self) -> None:
        with patch.dict("os.environ", {"MARIONETTE_TRACKABLE_SWARMS": "1"}, clear=False):
            self.assertIsNone(
                trackable_swarm_mcp_blocked("puppetmaster_codegraph_search")
            )

    def test_off_by_default(self) -> None:
        with patch.dict("os.environ", {"MARIONETTE_TRACKABLE_SWARMS": ""}, clear=False):
            self.assertIsNone(
                trackable_swarm_mcp_blocked("puppetmaster_start_swarm")
            )

    def test_call_tool_refuses_without_dispatch(self) -> None:
        with patch.dict("os.environ", {"MARIONETTE_TRACKABLE_SWARMS": "1"}, clear=False):
            result = call_tool(
                "puppetmaster_start_cursor_swarm",
                {"goal": "should refuse"},
            )
        self.assertTrue(result.get("isError"))
        body = json.loads(result["content"][0]["text"])
        self.assertIn("tracker-visible", body.get("error", "").lower())


if __name__ == "__main__":
    unittest.main()
