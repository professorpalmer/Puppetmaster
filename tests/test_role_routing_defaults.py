"""Per-role routing_policy defaults for analysis swarms (WAVE G)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.workers import (
    ANALYSIS_NO_EDIT_PAYLOAD,
    DEFAULT_ROLE_ROUTING_POLICY,
    DEFAULT_WORKERS,
    analysis_auto_route_payload,
    default_routing_policy_for_role,
)

# Expected role → policy map for built-in analysis roles (OMP modelRoles.task
# analogue). Policies only — never pin frontier model ids.
_EXPECTED_DEFAULT_WORKER_POLICIES = {
    "explore": "cheap",
    "architect": "balanced",
    "implement": "balanced",
    "redteam": "quality",
    "test": "cheap",
}

class DefaultRoleRoutingPolicyTests(unittest.TestCase):
    def test_role_to_policy_map(self) -> None:
        self.assertEqual(default_routing_policy_for_role("explore"), "cheap")
        self.assertEqual(default_routing_policy_for_role("test"), "cheap")
        self.assertEqual(default_routing_policy_for_role("architect"), "balanced")
        self.assertEqual(default_routing_policy_for_role("plan"), "balanced")
        self.assertEqual(default_routing_policy_for_role("redteam"), "quality")
        self.assertEqual(default_routing_policy_for_role("review"), "quality")
        self.assertEqual(default_routing_policy_for_role("audit"), "quality")
        self.assertIsNone(default_routing_policy_for_role("unknown-role"))

    def test_helper_preserves_auto_route_and_no_edit(self) -> None:
        for role, policy in (
            ("explore", "cheap"),
            ("architect", "balanced"),
            ("redteam", "quality"),
        ):
            payload = analysis_auto_route_payload(role)
            self.assertTrue(payload.get("auto_route"), role)
            self.assertEqual(payload.get("routing_policy"), policy, role)
            for key, value in ANALYSIS_NO_EDIT_PAYLOAD.items():
                self.assertEqual(payload.get(key), value, f"{role}.{key}")

    def test_helper_unknown_role_skips_policy_stamp(self) -> None:
        payload = analysis_auto_route_payload("custom-role")
        self.assertTrue(payload.get("auto_route"))
        self.assertNotIn("routing_policy", payload)
        for key, value in ANALYSIS_NO_EDIT_PAYLOAD.items():
            self.assertEqual(payload.get(key), value)

    def test_default_workers_carry_per_role_policies(self) -> None:
        by_role = {spec.role: spec for spec in DEFAULT_WORKERS}
        self.assertEqual(set(by_role), set(_EXPECTED_DEFAULT_WORKER_POLICIES))
        for role, policy in _EXPECTED_DEFAULT_WORKER_POLICIES.items():
            payload = by_role[role].payload
            self.assertTrue(payload.get("auto_route"), role)
            self.assertEqual(payload.get("routing_policy"), policy, role)
            for key, value in ANALYSIS_NO_EDIT_PAYLOAD.items():
                self.assertEqual(payload.get(key), value, f"{role}.{key}")

    def test_map_has_no_frontier_model_pins(self) -> None:
        """Reject hardcoding expensive frontier models — policies only."""
        allowed = {"cheap", "balanced", "quality", "escalating"}
        for role, policy in DEFAULT_ROLE_ROUTING_POLICY.items():
            self.assertIn(policy, allowed, role)
            self.assertNotIn("model", policy)

class GeneratedSwarmRoleRoutingTests(unittest.TestCase):
    def test_writer_stamps_per_role_defaults_when_policy_omitted(self) -> None:
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {
                "goal": "map and audit",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
            }
            config_path = write_generated_swarm_config(
                args, ["explore", "architect", "audit", "review"], "cursor"
            )
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            by_role = {w["role"]: w["payload"] for w in cfg["workers"]}

            self.assertEqual(by_role["explore"].get("routing_policy"), "cheap")
            self.assertEqual(by_role["architect"].get("routing_policy"), "balanced")
            self.assertEqual(by_role["audit"].get("routing_policy"), "quality")
            self.assertEqual(by_role["review"].get("routing_policy"), "quality")
            for role, payload in by_role.items():
                self.assertTrue(payload.get("auto_route"), role)
                for key, value in ANALYSIS_NO_EDIT_PAYLOAD.items():
                    self.assertEqual(payload.get(key), value, f"{role}.{key}")

    def test_writer_explicit_policy_overrides_per_role_defaults(self) -> None:
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {
                "goal": "force quality everywhere",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "routing_policy": "quality",
            }
            config_path = write_generated_swarm_config(
                args, ["explore", "test"], "cursor"
            )
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            for worker in cfg["workers"]:
                self.assertEqual(worker["payload"].get("routing_policy"), "quality")

if __name__ == "__main__":
    unittest.main()
