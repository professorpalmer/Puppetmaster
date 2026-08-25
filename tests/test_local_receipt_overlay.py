"""v1.22.32: local receipts overlay editorial capability_score.

Benton on #101 (issue stays closed): a stale editorial 90 must lose to a
live faster/higher-confidence sibling receipt. This is #28's later overlay,
not a new evidence OS. unittest only.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


def _spec(**kwargs):
    from puppetmaster.model_registry import ModelSpec

    defaults = dict(
        adapter="codex",
        billing="plan",
        input_per_mtok_usd=0.0,
        output_per_mtok_usd=0.0,
    )
    defaults.update(kwargs)
    if "adapter_model_name" not in defaults:
        defaults["adapter_model_name"] = defaults["id"].split("/", 1)[-1]
    return ModelSpec(**defaults)


def _receipt(**kwargs):
    from puppetmaster.scorecards import LocalReceipt

    defaults = dict(
        adapter="codex",
        role="implement",
        success=True,
        elapsed_seconds=17.0,
        confidence=0.99,
        observed_at="2026-08-24T18:00:00+00:00",
    )
    defaults.update(kwargs)
    return LocalReceipt(**defaults)


def _signal(**kwargs):
    from puppetmaster.router import TaskSignals

    defaults = dict(instruction="implement a feature", role="implement")
    defaults.update(kwargs)
    return TaskSignals(**defaults)


class BentonEditorialNinetyTests(unittest.TestCase):
    def test_benton_editorial_90_loses_to_live_faster_higher_confidence_sibling_receipt(
        self,
    ) -> None:
        from puppetmaster.router import route_task

        gpt5 = _spec(id="codex/gpt-5", adapter_model_name="gpt-5", capability_score=90)
        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
        )
        receipts = [
            _receipt(
                registry_id=luna.id,
                elapsed_seconds=17.0,
                confidence=0.99,
                observed_at="2026-08-24T18:00:17+00:00",
            ),
            _receipt(
                registry_id=gpt5.id,
                elapsed_seconds=107.0,
                confidence=0.69,
                observed_at="2026-08-24T18:01:47+00:00",
            ),
            _receipt(
                registry_id=gpt5.id,
                elapsed_seconds=115.0,
                confidence=0.72,
                observed_at="2026-08-24T18:03:42+00:00",
            ),
            _receipt(
                registry_id=gpt5.id,
                elapsed_seconds=136.0,
                confidence=0.74,
                observed_at="2026-08-24T18:05:58+00:00",
            ),
            _receipt(
                registry_id=gpt5.id,
                elapsed_seconds=156.0,
                confidence=0.78,
                observed_at="2026-08-24T18:08:34+00:00",
            ),
        ]
        decision = route_task(
            _signal(),
            [gpt5, luna],
            policy="quality",
            local_receipts=receipts,
        )
        self.assertEqual(decision.model.id, luna.id)
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["score_source"], "local_receipt")
        self.assertIn("local receipt", decision.reason)
        self.assertIn("17", decision.reason)
        self.assertIn("0.99", decision.reason)
        self.assertEqual(payload["score_provenance"]["source"], "local_receipt")
        self.assertEqual(payload["score_provenance"]["sample_count"], 1)
        self.assertEqual(payload["capability_score"], 85)
        rejected = {item["id"]: item["reason"] for item in payload["rejected"]}
        self.assertIn(gpt5.id, rejected)

    def test_quality_without_receipts_still_picks_editorial_90(self) -> None:
        from puppetmaster.router import route_task

        gpt5 = _spec(id="codex/gpt-5.5", adapter_model_name="gpt-5.5", capability_score=90)
        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
        )
        decision = route_task(_signal(), [gpt5, luna], policy="quality")
        self.assertEqual(decision.model.id, gpt5.id)
        self.assertEqual(decision.to_artifact_payload()["score_source"], "manual")


class OverlayGuardrailTests(unittest.TestCase):
    def test_no_cross_adapter_receipt_transfer(self) -> None:
        from puppetmaster.router import route_task

        gpt5 = _spec(id="codex/gpt-5", adapter="codex", adapter_model_name="gpt-5", capability_score=90)
        luna = _spec(
            id="agentic/gpt-5-6-luna",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
            billing="api",
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=6.0,
        )
        receipts = [
            _receipt(
                registry_id=luna.id,
                adapter="agentic",
                elapsed_seconds=17.0,
                confidence=0.99,
            )
        ]
        decision = route_task(
            _signal(),
            [gpt5, luna],
            policy="quality",
            local_receipts=receipts,
        )
        self.assertEqual(decision.model.id, gpt5.id)
        self.assertEqual(decision.to_artifact_payload()["score_source"], "manual")

    def test_no_cross_effort_receipt_transfer(self) -> None:
        from puppetmaster.router import route_task
        from puppetmaster.scorecards import effective_capability_score

        low = _spec(
            id="agentic/openai-api/gpt-5.6-luna-low",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            capability_score=90,
            payload_defaults={"reasoning_effort": "low"},
        )
        max_effort = _spec(
            id="agentic/openai-api/gpt-5.6-luna-max",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            capability_score=70,
            payload_defaults={"reasoning_effort": "max"},
        )
        receipts = [
            _receipt(
                registry_id=max_effort.id,
                adapter="agentic",
                effort="max",
                elapsed_seconds=17.0,
                confidence=0.99,
            )
        ]
        self.assertEqual(
            effective_capability_score(
                low, "implement", receipts=receipts, candidates=[low, max_effort]
            ),
            90,
        )
        decision = route_task(
            _signal(),
            [low, max_effort],
            policy="quality",
            local_receipts=receipts,
        )
        self.assertEqual(decision.model.id, low.id)

    def test_no_cross_harness_receipt_transfer(self) -> None:
        from puppetmaster.scorecards import effective_capability_score

        bash = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=70,
            payload_defaults={"harness": "swe-bench-bash-only"},
        )
        mini = _spec(
            id="codex/gpt-5",
            adapter_model_name="gpt-5",
            capability_score=90,
            payload_defaults={"harness": "mini-swe-agent"},
        )
        receipts = [
            _receipt(
                registry_id=bash.id,
                harness="swe-bench-bash-only",
                elapsed_seconds=17.0,
                confidence=0.99,
            )
        ]
        self.assertEqual(
            effective_capability_score(
                mini, "implement", receipts=receipts, candidates=[bash, mini]
            ),
            90,
        )

    def test_no_cross_role_receipt_transfer(self) -> None:
        from puppetmaster.router import route_task

        gpt5 = _spec(id="codex/gpt-5.5", adapter_model_name="gpt-5.5", capability_score=90)
        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
        )
        receipts = [
            _receipt(
                registry_id=luna.id,
                role="implement",
                elapsed_seconds=17.0,
                confidence=0.99,
            )
        ]
        decision = route_task(
            _signal(instruction="review this change", role="review"),
            [gpt5, luna],
            policy="quality",
            local_receipts=receipts,
        )
        self.assertEqual(decision.model.id, gpt5.id)
        self.assertEqual(decision.to_artifact_payload()["score_source"], "manual")


    def test_editorial_decays_when_newer_sibling_receipt_is_live(self) -> None:
        from puppetmaster.scorecards import resolve_score_authority

        gpt5 = _spec(id="codex/gpt-5", adapter_model_name="gpt-5", capability_score=90)
        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
        )
        receipts = [
            _receipt(
                registry_id=luna.id,
                elapsed_seconds=17.0,
                confidence=0.99,
                observed_at="2026-08-24T18:00:17+00:00",
            )
        ]
        authority = resolve_score_authority(
            gpt5, "implement", receipts=receipts, candidates=[gpt5, luna]
        )
        self.assertTrue(authority.decayed)
        self.assertLess(authority.effective, 90)
        self.assertIn("decayed", authority.note)
        self.assertIn(luna.id, authority.note)

    def test_failed_or_incomplete_receipt_does_not_overlay(self) -> None:
        from puppetmaster.scorecards import SCORE_SOURCE_MANUAL, resolve_score_authority

        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
        )
        failed = resolve_score_authority(
            luna,
            "implement",
            receipts=[_receipt(registry_id=luna.id, success=False)],
        )
        incomplete = resolve_score_authority(
            luna,
            "implement",
            receipts=[
                _receipt(
                    registry_id=luna.id,
                    success=True,
                    elapsed_seconds=17.0,
                    confidence=None,
                )
            ],
        )
        self.assertEqual(failed.source, SCORE_SOURCE_MANUAL)
        self.assertFalse(failed.decayed)
        self.assertEqual(failed.effective, 85)
        self.assertEqual(incomplete.source, SCORE_SOURCE_MANUAL)
        self.assertEqual(incomplete.effective, 85)


class OverlayDisclosureTests(unittest.TestCase):
    def test_routing_discloses_card_vs_fallback_vs_local_receipt(self) -> None:
        from datetime import date
        from puppetmaster.router import route_task

        carded = _spec(
            id="codex/carded",
            adapter_model_name="carded",
            capability_score=40,
            role_scorecards={
                "implement": {
                    "capability": 95,
                    "sample_count": 5,
                    "last_calibrated": date.today().isoformat(),
                    "scale": "puppetmaster-capability-0-100",
                    "scale_version": "1",
                    "provenance": {"source": "test_objective_evaluator", "version": "1"},
                }
            },
        )
        fallback = _spec(
            id="codex/fallback",
            adapter_model_name="fallback-other",
            capability_score=50,
        )
        live = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=60,
        )
        card_decision = route_task(_signal(), [carded, fallback], policy="quality")
        self.assertEqual(card_decision.to_artifact_payload()["score_source"], "role_card")

        fallback_decision = route_task(_signal(), [fallback], policy="quality")
        self.assertEqual(
            fallback_decision.to_artifact_payload()["score_source"], "manual"
        )

        live_decision = route_task(
            _signal(),
            [live, fallback],
            policy="quality",
            local_receipts=[
                _receipt(registry_id=live.id, elapsed_seconds=17.0, confidence=0.99)
            ],
        )
        payload = live_decision.to_artifact_payload()
        self.assertEqual(payload["score_source"], "local_receipt")
        self.assertEqual(payload["score_provenance"]["source"], "local_receipt")
        self.assertIn("local receipt", live_decision.reason)

    def test_one_receipt_is_one_observation_no_average(self) -> None:
        from puppetmaster.scorecards import latest_receipt_for

        luna = _spec(id="codex/gpt-5-6-luna", adapter_model_name="gpt-5.6-luna")
        receipts = [
            _receipt(
                registry_id=luna.id,
                elapsed_seconds=17.0,
                confidence=0.99,
                observed_at="2026-08-24T18:00:17+00:00",
            ),
            _receipt(
                registry_id=luna.id,
                elapsed_seconds=40.0,
                confidence=0.80,
                observed_at="2026-08-24T17:00:00+00:00",
            ),
        ]
        latest = latest_receipt_for(luna, "implement", receipts)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.elapsed_seconds, 17.0)
        self.assertEqual(latest.confidence, 0.99)

    def test_missing_swebench_stays_unknown_without_receipt(self) -> None:
        from puppetmaster.scorecards import (
            SCORE_SOURCE_MANUAL,
            effective_capability_score,
            resolve_score_authority,
        )

        luna = _spec(
            id="codex/gpt-5-6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
            role_scorecards={"implement": {"capability": 50}},
        )
        authority = resolve_score_authority(luna, "implement")
        self.assertEqual(authority.source, SCORE_SOURCE_MANUAL)
        self.assertEqual(effective_capability_score(luna, "implement"), 85)


if __name__ == "__main__":
    unittest.main()
