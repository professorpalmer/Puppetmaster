"""Umbrella swarm reasoning default: one payload stamp, pins win."""
from __future__ import annotations

import os
import sys
import unittest

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.model_registry import ModelSpec
from puppetmaster.orchestrator import merge_routing_payload
from puppetmaster.router import RoutingDecision
from puppetmaster.swarm_reasoning import (
    DEFAULT_SWARM_REASONING_EFFORT,
    apply_swarm_reasoning,
    caller_pinned_effort,
)


def _decision(adapter: str, payload_defaults: dict) -> RoutingDecision:
    spec = ModelSpec(
        id=f"{adapter}/test",
        adapter=adapter,
        adapter_model_name="test-model",
        payload_defaults=payload_defaults,
    )
    return RoutingDecision(
        model=spec,
        policy="balanced",
        capability_needed=75,
        estimated_tokens_in=1000,
        estimated_tokens_out=1000,
        estimated_cost_usd=0.01,
        reason="test",
    )


class CallerPinTests(unittest.TestCase):
    def test_empty_is_not_a_pin(self) -> None:
        self.assertIsNone(caller_pinned_effort({}))
        self.assertIsNone(caller_pinned_effort({"reasoning_effort": ""}))
        self.assertIsNone(caller_pinned_effort(None))

    def test_reasoning_effort_and_effort_keys(self) -> None:
        self.assertEqual(caller_pinned_effort({"reasoning_effort": "low"}), "low")
        self.assertEqual(caller_pinned_effort({"effort": "High"}), "high")

    def test_extra_args_and_params_are_pins(self) -> None:
        self.assertEqual(
            caller_pinned_effort({"extra_args": ["--effort", "xhigh"]}),
            "xhigh",
        )
        self.assertEqual(
            caller_pinned_effort(
                {"extra_args": ["-c", "model_reasoning_effort=low"]}
            ),
            "low",
        )
        self.assertEqual(
            caller_pinned_effort(
                {"params": [{"id": "effort", "value": "high"}, {"id": "fast", "value": "true"}]}
            ),
            "high",
        )


class OverlayTests(unittest.TestCase):
    def test_catalog_high_is_not_a_pin(self) -> None:
        merged = apply_swarm_reasoning(
            {
                "reasoning_effort": "high",
                "params": [
                    {"id": "effort", "value": "high"},
                    {"id": "fast", "value": "true"},
                ],
            },
            {},
            adapter="cursor",
        )
        self.assertEqual(merged["reasoning_effort"], DEFAULT_SWARM_REASONING_EFFORT)
        self.assertEqual(
            merged["params"],
            [
                {"id": "effort", "value": "medium"},
                {"id": "fast", "value": "true"},
            ],
        )

    def test_caller_pin_beats_catalog(self) -> None:
        merged = apply_swarm_reasoning(
            {"reasoning_effort": "high"},
            {"reasoning_effort": "low"},
            adapter="openai",
        )
        self.assertEqual(merged["reasoning_effort"], "low")

    def test_claude_and_codex_dialects(self) -> None:
        claude = apply_swarm_reasoning({}, {}, adapter="claude-code")
        self.assertEqual(claude["extra_args"], ["--effort", "medium"])
        codex = apply_swarm_reasoning({}, {}, adapter="codex")
        self.assertEqual(codex["extra_args"], ["-c", "model_reasoning_effort=medium"])

    def test_antigravity_effort_and_encoded_slug(self) -> None:
        flash = apply_swarm_reasoning(
            {"model": "gemini-3.7-flash"},
            {},
            adapter="antigravity",
        )
        self.assertEqual(flash["effort"], "medium")
        encoded = apply_swarm_reasoning(
            {"model": "gemini-3.7-flash-high"},
            {},
            adapter="antigravity",
        )
        self.assertNotIn("effort", encoded)


class MergeRoutingPayloadTests(unittest.TestCase):
    def test_unset_caller_overwrites_catalog_high(self) -> None:
        merged = merge_routing_payload(
            {"prompt": "keep me"},
            _decision("openai", {"reasoning_effort": "high", "temperature": 0}),
        )
        self.assertEqual(merged["reasoning_effort"], "medium")
        self.assertEqual(merged["temperature"], 0)
        self.assertEqual(merged["prompt"], "keep me")

    def test_caller_pin_still_wins(self) -> None:
        merged = merge_routing_payload(
            {"reasoning_effort": "low"},
            _decision("openai", {"reasoning_effort": "high"}),
        )
        self.assertEqual(merged["reasoning_effort"], "low")

    def test_extra_fields_pin_wins(self) -> None:
        merged = merge_routing_payload(
            {},
            _decision("openai", {"reasoning_effort": "high"}),
            {"reasoning_effort": "low"},
        )
        self.assertEqual(merged["reasoning_effort"], "low")

    def test_cursor_catalog_params_are_not_a_pin(self) -> None:
        merged = merge_routing_payload(
            {},
            _decision(
                "cursor",
                {
                    "params": [
                        {"id": "effort", "value": "high"},
                        {"id": "fast", "value": "true"},
                    ]
                },
            ),
        )
        self.assertEqual(merged["reasoning_effort"], "medium")
        self.assertEqual(
            merged["params"],
            [
                {"id": "effort", "value": "medium"},
                {"id": "fast", "value": "true"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
