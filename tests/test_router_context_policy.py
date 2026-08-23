"""Adversarial TASK-002 tests for context fit and routing policy semantics.

These tests were authored against the TASK-001-completed baseline before the
TASK-002 author implementation was inspected.  They intentionally exercise
public behavior rather than private ranking helpers.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster import router
from puppetmaster.model_registry import ModelSpec


def _model(
    model_id: str,
    *,
    capability: int,
    input_cost: float,
    context_window: int = 32_000,
) -> ModelSpec:
    """Arrange a deterministic plan-billed candidate without provider probes."""
    return ModelSpec(
        id=model_id,
        adapter="cursor",
        adapter_model_name=model_id.rsplit("/", 1)[-1],
        capability_score=capability,
        input_per_mtok_usd=input_cost,
        output_per_mtok_usd=input_cost,
        context_window=context_window,
        billing="plan",
    )


class RouterContextPolicyTests(unittest.TestCase):
    def test_context_filter_reserves_output_before_selecting_cheapest_model(self) -> None:
        bargain = _model(
            "cursor/bargain", capability=60, input_cost=0.10, context_window=4_000
        )
        roomy = _model(
            "cursor/roomy", capability=70, input_cost=1.00, context_window=16_000
        )
        task = router.TaskSignals(
            instruction="verify the result",
            role="verify-runtime",
            explicit_min_capability=50,
            estimated_tokens_in=3_000,
            estimated_tokens_out=1_500,
        )
        decision = router.route_task(task, [bargain, roomy], policy="cheap")
        self.assertEqual(decision.model.id, "cursor/roomy")
        rejected_reason = dict((spec.id, why) for spec, why in decision.rejected)
        self.assertIn("context", rejected_reason["cursor/bargain"].lower())
        self.assertIn("3000", rejected_reason["cursor/bargain"])
        self.assertIn("1500", rejected_reason["cursor/bargain"])
        self.assertIn("4000", rejected_reason["cursor/bargain"])

    def test_context_filter_fails_when_every_known_window_overflows(self) -> None:
        task = router.TaskSignals(
            instruction="verify the result",
            role="verify-runtime",
            explicit_min_capability=20,
            estimated_tokens_in=3_500,
            estimated_tokens_out=1_000,
        )
        models = [
            _model("cursor/a", capability=50, input_cost=0.10, context_window=4_000),
            _model("cursor/b", capability=90, input_cost=1.00, context_window=4_499),
        ]
        with self.assertRaisesRegex(router.NoEligibleModelError, "(?i)context"):
            router.route_task(task, models, policy="balanced")

    def test_explicit_token_estimates_fail_closed_when_invalid(self) -> None:
        cases = (
            ("estimated_tokens_in", -1, "negative-input"),
            ("estimated_tokens_in", True, "boolean-input"),
            ("estimated_tokens_in", 1.5, "non-integer-input"),
            ("estimated_tokens_in", "1000", "malformed-input"),
            ("estimated_tokens_out", -1, "negative-output"),
            ("estimated_tokens_out", False, "boolean-output"),
            ("estimated_tokens_out", 1.5, "non-integer-output"),
            ("estimated_tokens_out", "200", "malformed-output"),
        )
        for field_name, invalid_value, case_id in cases:
            with self.subTest(case_id):
                task_values = {
                    "instruction": "verify the result",
                    "role": "verify-runtime",
                    "explicit_min_capability": 20,
                    "estimated_tokens_in": 900,
                    "estimated_tokens_out": 200,
                }
                task_values[field_name] = invalid_value
                task = router.TaskSignals(**task_values)
                model = _model(
                    "cursor/small-window",
                    capability=60,
                    input_cost=0.10,
                    context_window=1_000,
                )
                with self.assertRaisesRegex(ValueError, field_name):
                    router.route_task(task, [model], policy="balanced")

    def test_worker_signals_count_nested_payload_and_declared_enrichment(self) -> None:
        nested_text = "nested repository context " * 400
        spec = SimpleNamespace(
            instruction="inspect",
            role="explore",
            payload={
                "context": {
                    "documents": [
                        {"body": nested_text},
                        ("secondary context", {"notes": "more evidence"}),
                    ]
                },
                "adapter_enrichment_tokens": 1_200,
            },
        )
        signals = router.signals_from_worker_spec(spec)
        estimate = router.estimate_tokens_in(signals)
        self.assertGreaterEqual(signals.payload_size_chars, len(nested_text))
        self.assertEqual(signals.adapter_enrichment_tokens, 1_200)
        un_enriched_floor = max(
            500,
            (len(spec.instruction) + signals.payload_size_chars) // 4 + 500,
        )
        self.assertGreaterEqual(estimate, un_enriched_floor + 1_200)

    def test_measured_calibration_is_explicit_attributable_and_policy_neutral(self) -> None:
        measurements = [
            {"estimated_tokens_in": 1_000, "actual_tokens_in": 2_000},
            {"estimated_tokens_in": 2_000, "actual_tokens_in": 4_000},
            {"estimated_tokens_in": 3_000, "actual_tokens_in": 6_000},
        ]
        calibration = router.calibration_from_measurements(
            measurements,
            adapter="cursor",
            model_id="cursor/calibrated",
            role="verify-runtime",
            source="measured_usage",
        )
        task = router.TaskSignals(
            instruction="verify",
            role="verify-runtime",
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        model = _model(
            "cursor/calibrated", capability=60, input_cost=0.10, context_window=8_000
        )
        calibrated_estimate = router.estimate_tokens_in(task, calibration=calibration)
        decision = router.route_task(
            task,
            [model],
            policy="cheap",
            calibration=calibration,
        )
        payload = decision.to_artifact_payload()
        self.assertAlmostEqual(calibration.multiplier, 2.0)
        self.assertEqual(calibration.sample_count, 3)
        self.assertEqual(calibrated_estimate, 2_000)
        self.assertEqual(decision.estimated_tokens_in, 2_000)
        self.assertEqual(decision.policy, "cheap")
        self.assertEqual(
            payload["token_estimate_calibration"],
            {
                "adapter": "cursor",
                "model_id": "cursor/calibrated",
                "canonical_role": "verify-runtime",
                "source": "measured_usage",
                "sample_count": 3,
                "multiplier": 2.0,
            },
        )

    def test_model_scoped_calibration_does_not_distort_other_candidates(self) -> None:
        calibration = router.calibration_from_measurements(
            [
                {"estimated_tokens_in": 1_000, "actual_tokens_in": 2_000},
                {"estimated_tokens_in": 2_000, "actual_tokens_in": 4_000},
            ],
            adapter="cursor",
            model_id="cursor/calibrated",
            role="verify-runtime",
            source="measured_usage",
        )
        task = router.TaskSignals(
            instruction="verify",
            role="verify-runtime",
            explicit_min_capability=40,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        calibrated = _model(
            "cursor/calibrated", capability=60, input_cost=0.01, context_window=1_900
        )
        uncalibrated = _model(
            "cursor/uncalibrated", capability=60, input_cost=0.10, context_window=1_500
        )
        decision = router.route_task(
            task,
            [calibrated, uncalibrated],
            policy="cheap",
            calibration=calibration,
        )
        self.assertEqual(decision.model.id, "cursor/uncalibrated")
        self.assertEqual(decision.estimated_tokens_in, 1_000)
        self.assertNotIn("token_estimate_calibration", decision.to_artifact_payload())

    def test_cheap_selects_cheapest_sufficient_model(self) -> None:
        task = router.TaskSignals(
            instruction="perform the bounded task",
            role="verify-runtime",
            explicit_min_capability=70,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        bargain_but_weak = _model(
            "cursor/weak", capability=40, input_cost=0.01
        )
        cheapest_sufficient = _model(
            "cursor/sufficient", capability=75, input_cost=0.10
        )
        costly_frontier = _model(
            "cursor/frontier", capability=95, input_cost=1.00
        )
        decision = router.route_task(
            task,
            [bargain_but_weak, cheapest_sufficient, costly_frontier],
            policy="cheap",
        )
        self.assertEqual(decision.model.id, "cursor/sufficient")
        self.assertIn("cheapest sufficient", decision.reason.lower())
        rejected_reason = dict((spec.id, why) for spec, why in decision.rejected)
        self.assertIn("capability", rejected_reason["cursor/weak"].lower())

    def test_absolute_cheapest_requires_explicit_policy_opt_in(self) -> None:
        task = router.TaskSignals(
            instruction="perform the bounded task",
            role="verify-runtime",
            explicit_min_capability=70,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        weak = _model("cursor/weak", capability=40, input_cost=0.01)
        sufficient = _model("cursor/sufficient", capability=75, input_cost=0.10)
        decision = router.route_task(
            task, [weak, sufficient], policy="absolute-cheapest"
        )
        self.assertEqual(decision.model.id, "cursor/weak")
        self.assertEqual(decision.policy, "absolute-cheapest")
        self.assertIn("absolute", decision.reason.lower())

    def test_absolute_cheapest_is_reachable_through_every_cli_policy_flag(self) -> None:
        from puppetmaster.cli._parser import build_parser

        parser = build_parser()
        policy_actions = []
        pending = [parser]
        seen = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            for action in current._actions:
                if action.dest in {"policy", "routing_policy"} and action.choices:
                    policy_actions.append(action)
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    pending.extend(choices.values())
        self.assertTrue(policy_actions)
        missing = [action.dest for action in policy_actions if "absolute-cheapest" not in action.choices]
        self.assertEqual(missing, [])

    def test_absolute_cheapest_is_reachable_through_every_mcp_policy_enum(self) -> None:
        from puppetmaster import mcp_server

        schemas = {
            "route_task": mcp_server.route_task_schema(),
            "auto_route": {"properties": mcp_server._auto_route_schema_properties()},
            "swarm": mcp_server.swarm_schema(),
            "cursor_swarm": mcp_server.cursor_swarm_schema(),
            "codex": mcp_server.codex_schema(),
            "claude": mcp_server.claude_schema(),
            "cursor_implement": mcp_server.cursor_implement_schema(),
            "implement": mcp_server.implement_schema(),
            "edit": mcp_server.edit_schema(),
            "browser_swarm": mcp_server.browser_swarm_schema(),
            "agentic": mcp_server.agentic_schema(),
            "openai": mcp_server.openai_schema(),
        }
        found = []

        def visit(value, schema_name: str) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict):
                    for property_name in ("policy", "routing_policy"):
                        property_schema = properties.get(property_name)
                        if isinstance(property_schema, dict) and "enum" in property_schema:
                            found.append((schema_name, property_name, property_schema["enum"]))
                for child in value.values():
                    visit(child, schema_name)
            elif isinstance(value, list):
                for child in value:
                    visit(child, schema_name)

        for name, schema in schemas.items():
            visit(schema, name)
        self.assertTrue(found)
        missing = [f"{name}.{property_name}" for name, property_name, enum in found if "absolute-cheapest" not in enum]
        self.assertEqual(missing, [])

    def test_strict_capability_fails_closed_when_catalog_is_insufficient(self) -> None:
        task = router.TaskSignals(
            instruction="perform the critical task",
            role="verify-runtime",
            explicit_min_capability=95,
            strict_capability=True,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        models = [
            _model("cursor/mid", capability=60, input_cost=0.10),
            _model("cursor/best", capability=85, input_cost=1.00),
        ]
        with self.assertRaisesRegex(router.NoEligibleModelError, "(?i)capability.*95"):
            router.route_task(task, models, policy="balanced")

    def test_non_strict_balanced_fallback_remains_explicit_and_compatible(self) -> None:
        task = router.TaskSignals(
            instruction="perform the task",
            role="verify-runtime",
            explicit_min_capability=95,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        models = [
            _model("cursor/mid", capability=60, input_cost=0.10),
            _model("cursor/best", capability=85, input_cost=1.00),
        ]
        decision = router.route_task(task, models, policy="balanced")
        self.assertEqual(decision.model.id, "cursor/best")
        self.assertIn("no model meets capability need (95)", decision.reason.lower())
        self.assertIn("falling back", decision.reason.lower())


if __name__ == "__main__":
    unittest.main()
