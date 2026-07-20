"""Focused tests for the shared analysis-swarm launch helpers + CLI twin."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster.swarm_launch import (
    DEFAULT_SWARM_ROLES,
    SWARM_ANALYSIS_ADAPTERS,
    build_analysis_swarm_specs,
    write_analysis_swarm_config,
)


class BuildAnalysisSwarmSpecsTests(unittest.TestCase):
    def test_default_roles_and_cursor_auto_route(self) -> None:
        specs = build_analysis_swarm_specs(
            "peel the MCP surface",
            [],
            adapter="cursor",
            cwd=r"C:\repo",
        )
        self.assertEqual([s.role for s in specs], list(DEFAULT_SWARM_ROLES))
        for spec in specs:
            self.assertEqual(spec.adapter, "cursor")
            self.assertTrue(spec.payload.get("auto_route"))
            self.assertTrue(spec.payload.get("read_only"))
            self.assertEqual(spec.payload.get("sandbox"), "read-only")
            self.assertTrue(spec.payload.get("disable_memory"))
            self.assertEqual(spec.payload.get("cwd"), r"C:\repo")
            self.assertIn("Role:", spec.instruction)
            self.assertIn("peel the MCP surface", spec.instruction)

    def test_explore_cheap_review_quality_policies(self) -> None:
        specs = build_analysis_swarm_specs(
            "goal",
            ["explore", "review"],
            adapter="cursor",
            cwd="/tmp/x",
        )
        by_role = {s.role: s for s in specs}
        self.assertEqual(by_role["explore"].payload.get("routing_policy"), "cheap")
        self.assertEqual(by_role["review"].payload.get("routing_policy"), "quality")

    def test_model_pin_disables_auto_route_unless_forced(self) -> None:
        pinned = build_analysis_swarm_specs(
            "goal",
            ["explore"],
            adapter="cursor",
            cwd="/tmp/x",
            model="grok-4.5",
        )
        self.assertNotIn("auto_route", pinned[0].payload)

        forced = build_analysis_swarm_specs(
            "goal",
            ["explore"],
            adapter="cursor",
            cwd="/tmp/x",
            model="grok-4.5",
            auto_route=True,
        )
        self.assertTrue(forced[0].payload.get("auto_route"))

    def test_non_cursor_adapter_pins_allowed_adapters(self) -> None:
        specs = build_analysis_swarm_specs(
            "goal",
            ["audit"],
            adapter="agentic",
            cwd="/tmp/x",
        )
        self.assertEqual(specs[0].payload.get("allowed_adapters"), ["agentic"])

    def test_rejects_unknown_adapter(self) -> None:
        with self.assertRaises(ValueError):
            build_analysis_swarm_specs("g", ["explore"], adapter="nope", cwd="/tmp")


class WriteAnalysisSwarmConfigTests(unittest.TestCase):
    def test_writes_workers_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_analysis_swarm_config(
                goal="audit",
                roles=["explore", "audit"],
                adapter="cursor",
                state_dir=Path(tmp),
                cwd="/repo",
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["lease_seconds"], 10)
            self.assertEqual(len(data["workers"]), 2)
            self.assertEqual(data["workers"][0]["adapter"], "cursor")
            self.assertTrue(data["workers"][0]["payload"]["auto_route"])


class McpWriteGeneratedDelegatesTests(unittest.TestCase):
    def test_mcp_wrapper_matches_shared_builder(self) -> None:
        from puppetmaster.mcp_server import write_generated_swarm_config

        with tempfile.TemporaryDirectory() as tmp:
            args = {
                "goal": "parity check",
                "cwd": "/repo",
                "timeout_seconds": 600,
                "disable_memory": True,
            }
            with mock.patch(
                "puppetmaster.mcp_server.mcp_state_dir",
                return_value=Path(tmp),
            ):
                path = write_generated_swarm_config(args, ["explore", "review"], "cursor")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                {w["role"] for w in data["workers"]},
                {"explore", "review"},
            )
            self.assertTrue(all(w["payload"]["read_only"] for w in data["workers"]))

    def test_swarm_analysis_adapters_exported(self) -> None:
        from puppetmaster import mcp_server

        self.assertIn("cursor", mcp_server.SWARM_ANALYSIS_ADAPTERS)
        self.assertEqual(
            set(mcp_server.SWARM_ANALYSIS_ADAPTERS),
            set(SWARM_ANALYSIS_ADAPTERS),
        )


if __name__ == "__main__":
    unittest.main()
