"""Cache affinity: sibling workers keep a sufficient previous model.

This is not KV-cache portability. It avoids a model switch so the shared
job-brief prefix can stay provider-cacheable on the same model.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster import router
from puppetmaster.model_registry import ModelSpec
from puppetmaster.workers import WorkerSpec


def _model(model_id, *, capability, input_cost):
    return ModelSpec(
        id=model_id,
        adapter="cursor",
        adapter_model_name=model_id.rsplit("/", 1)[-1],
        capability_score=capability,
        input_per_mtok_usd=input_cost,
        output_per_mtok_usd=input_cost,
        context_window=128_000,
        billing="plan",
        tags=["tools"],
    )


CHEAP = _model("cursor/bargain", capability=70, input_cost=0.10)
PRICEY = _model("cursor/pricey", capability=80, input_cost=2.00)
FRONTIER = _model("cursor/frontier", capability=95, input_cost=5.00)
WEAK = _model("cursor/weak", capability=20, input_cost=0.01)


class CacheAffinityTests(unittest.TestCase):
    def test_cheap_without_prefer_picks_cheapest_sufficient(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
        )
        decision = router.route_task(task, [PRICEY, CHEAP], policy="cheap")
        self.assertEqual(decision.model.id, "cursor/bargain")

    def test_cheap_prefer_keeps_sufficient_sibling(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
            prefer_model_id="cursor/pricey",
        )
        decision = router.route_task(task, [PRICEY, CHEAP], policy="cheap")
        self.assertEqual(decision.model.id, "cursor/pricey")
        self.assertIn("cache affinity", decision.reason)

    def test_insufficient_prefer_is_ignored(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
            prefer_model_id="cursor/weak",
        )
        decision = router.route_task(task, [WEAK, CHEAP, PRICEY], policy="cheap")
        self.assertEqual(decision.model.id, "cursor/bargain")
        self.assertNotIn("cache affinity", decision.reason)

    def test_balanced_prefer_keeps_sufficient_sibling(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
            prefer_model_id="cursor/pricey",
        )
        decision = router.route_task(task, [PRICEY, CHEAP], policy="balanced")
        self.assertEqual(decision.model.id, "cursor/pricey")
        self.assertIn("cache affinity", decision.reason)

    def test_prefer_absent_from_candidates_is_ignored(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
            prefer_model_id="cursor/pricey",
        )
        decision = router.route_task(task, [CHEAP], policy="cheap")
        self.assertEqual(decision.model.id, "cursor/bargain")
        self.assertNotIn("cache affinity", decision.reason)
        task = router.TaskSignals(
            instruction="audit the auth module",
            role="audit",
            explicit_min_capability=50,
            prefer_model_id="cursor/bargain",
        )
        decision = router.route_task(
            task, [CHEAP, PRICEY, FRONTIER], policy="quality"
        )
        self.assertEqual(decision.model.id, "cursor/frontier")
        self.assertNotIn("cache affinity", decision.reason)

    def test_kill_switch_disables_affinity(self) -> None:
        task = router.TaskSignals(
            instruction="map the auth module",
            role="explore",
            explicit_min_capability=50,
            prefer_model_id="cursor/pricey",
        )
        previous = os.environ.get(router.CACHE_AFFINITY_ENV)
        os.environ[router.CACHE_AFFINITY_ENV] = "0"
        try:
            decision = router.route_task(task, [PRICEY, CHEAP], policy="cheap")
        finally:
            if previous is None:
                os.environ.pop(router.CACHE_AFFINITY_ENV, None)
            else:
                os.environ[router.CACHE_AFFINITY_ENV] = previous
        self.assertEqual(decision.model.id, "cursor/bargain")

    def test_signals_from_worker_spec_read_prefer_model_id(self) -> None:
        spec = WorkerSpec(
            role="explore",
            instruction="map the auth module",
            payload={"prefer_model_id": "cursor/pricey", "auto_route": True},
        )
        signals = router.signals_from_worker_spec(spec)
        self.assertEqual(signals.prefer_model_id, "cursor/pricey")


if __name__ == "__main__":
    unittest.main()
