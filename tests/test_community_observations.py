"""Issue #138: gated community observations, separate from capability_score."""
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


def _spec(**kwargs):
    from puppetmaster.model_registry import ModelSpec

    defaults = dict(
        adapter="codex",
        billing="api",
        input_per_mtok_usd=1.0,
        output_per_mtok_usd=2.0,
        capability_score=80,
    )
    defaults.update(kwargs)
    if "adapter_model_name" not in defaults:
        defaults["adapter_model_name"] = defaults["id"].split("/", 1)[-1]
    return ModelSpec(**defaults)


def _obs(**kwargs):
    from puppetmaster.community_observations import CommunityObservation

    defaults = dict(
        registry_id="codex/gpt-5.6-sol-high",
        adapter="codex",
        role="implement",
        effort="high",
        provider="openai",
        track="worker",
        bank="strongorc-ranking-v1-2026-09-01",
        harness="0.6.0",
        pass_rate=0.70,
        ci_low=0.55,
        ci_high=0.82,
        sample_count=36,
        published="2026-09-08",
    )
    defaults.update(kwargs)
    return CommunityObservation(**defaults)


def _signal(**kwargs):
    from puppetmaster.router import TaskSignals

    defaults = dict(
        instruction="implement a feature",
        role="implement",
        explicit_min_capability=50,
        prefer_plan_billed=False,
    )
    defaults.update(kwargs)
    return TaskSignals(**defaults)


def _sol_high(**kwargs):
    defaults = dict(
        id="codex/gpt-5.6-sol-high",
        adapter_model_name="gpt-5.6-sol",
        capability_score=85,
        input_per_mtok_usd=5.0,
        output_per_mtok_usd=15.0,
        payload_defaults={"reasoning_effort": "high"},
    )
    defaults.update(kwargs)
    return _spec(**defaults)


def _luna_low(**kwargs):
    defaults = dict(
        id="codex/gpt-5.6-luna-low",
        adapter_model_name="gpt-5.6-luna",
        capability_score=80,
        input_per_mtok_usd=0.5,
        output_per_mtok_usd=1.0,
        payload_defaults={"reasoning_effort": "low"},
    )
    defaults.update(kwargs)
    return _spec(**defaults)


class ImportObservationTests(unittest.TestCase):
    def test_dry_run_writes_nothing(self) -> None:
        from puppetmaster.community_observations import import_observations

        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle.json"
            store = Path(tmp) / "store.json"
            bundle.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "registry_id": "codex/gpt-5.6-sol-high",
                                "adapter": "codex",
                                "role": "implement",
                                "effort": "high",
                                "provider": "openai",
                                "track": "worker",
                                "bank": "b",
                                "harness": "0.6.0",
                                "pass_rate": 0.5,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            import_observations(bundle, store, dry_run=True)
            self.assertFalse(store.exists())

    def test_refuses_incomplete_identity(self) -> None:
        from puppetmaster.community_observations import parse_observation_bundle

        with self.assertRaises(ValueError):
            parse_observation_bundle(
                {
                    "entries": [
                        {
                            "registry_id": "codex/gpt-5.6-sol-high",
                            "adapter": "codex",
                            "role": "implement",
                            "provider": "openai",
                            "track": "worker",
                            "bank": "b",
                            "harness": "0.6.0",
                            "pass_rate": 0.5,
                        }
                    ]
                }
            )

    def test_import_never_mutates_capability_or_cards(self) -> None:
        from puppetmaster.community_observations import import_observations
        from puppetmaster.model_registry import load_registry, save_registry

        spec = _sol_high(role_scorecards={"implement": {"capability": 90}})
        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            bundle = Path(tmp) / "bundle.json"
            store = Path(tmp) / "obs.json"
            save_registry([spec], registry_path)
            before = registry_path.read_text(encoding="utf-8")
            bundle.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "registry_id": spec.id,
                                "adapter": spec.adapter,
                                "role": "implement",
                                "effort": "high",
                                "provider": "openai",
                                "track": "worker",
                                "bank": "b",
                                "harness": "0.6.0",
                                "pass_rate": 0.9,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            import_observations(bundle, store, dry_run=False)
            self.assertEqual(registry_path.read_text(encoding="utf-8"), before)
            loaded = load_registry(registry_path)
            self.assertEqual(loaded[0].capability_score, spec.capability_score)
            self.assertEqual(loaded[0].role_scorecards, spec.role_scorecards)

    def test_cli_dry_run_writes_nothing(self) -> None:
        from puppetmaster.cli.commands_models import _run_models_import_observations
        from puppetmaster.community_observations import default_example_bundle_path

        with TemporaryDirectory() as tmp:
            store = Path(tmp) / "obs.json"
            args = argparse.Namespace(
                path=str(default_example_bundle_path()),
                store_path=str(store),
                dry_run=True,
            )
            rc = _run_models_import_observations(args)
            self.assertEqual(rc, 0)
            self.assertFalse(store.exists())

    def test_cli_incomplete_bundle_fails(self) -> None:
        from puppetmaster.cli.commands_models import _run_models_import_observations

        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bad.json"
            bundle.write_text(
                json.dumps({"entries": [{"registry_id": "x", "pass_rate": 0.1}]}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                path=str(bundle),
                store_path=str(Path(tmp) / "obs.json"),
                dry_run=True,
            )
            self.assertEqual(_run_models_import_observations(args), 1)


class CommunityGateRoutingTests(unittest.TestCase):
    def test_codex_only_ignores_openrouter_kimi(self) -> None:
        from puppetmaster.router import route_task

        kimi = _obs(
            registry_id="agentic/moonshotai/kimi-k3",
            adapter="agentic",
            effort="high",
            provider="openrouter",
            pass_rate=0.95,
            ci_low=0.80,
            ci_high=0.99,
        )
        sol = _sol_high()
        luna = _luna_low()
        decision = route_task(
            _signal(),
            [sol, luna],
            policy="cheap",
            community_observations=[kimi],
            role_preferences={},
        )
        self.assertEqual(decision.model.id, luna.id)
        self.assertNotEqual(decision.score_source, "community_observation")

    def test_clear_delta_picks_sol_high(self) -> None:
        from puppetmaster.router import route_task

        sol = _sol_high()
        luna = _luna_low()
        observations = [
            _obs(
                registry_id=sol.id,
                adapter="codex",
                effort="high",
                pass_rate=0.70,
                ci_low=0.55,
                ci_high=0.82,
            ),
            _obs(
                registry_id=luna.id,
                adapter="codex",
                effort="high",
                pass_rate=0.40,
                ci_low=0.20,
                ci_high=0.48,
            ),
        ]
        decision = route_task(
            _signal(reasoning_effort="high"),
            [sol, luna],
            policy="cheap",
            community_observations=observations,
            role_preferences={},
        )
        self.assertEqual(decision.model.id, sol.id)
        self.assertEqual(decision.score_source, "community_observation")
        self.assertEqual(decision.effective_capability_score, sol.capability_score)
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["score_source"], "community_observation")

    def test_overlapping_ci_falls_through_to_cheap(self) -> None:
        from puppetmaster.router import route_task

        sol = _sol_high()
        luna = _luna_low()
        observations = [
            _obs(
                registry_id=sol.id,
                adapter="codex",
                effort="high",
                pass_rate=0.51,
                ci_low=0.30,
                ci_high=0.70,
            ),
            _obs(
                registry_id=luna.id,
                adapter="codex",
                effort="high",
                pass_rate=0.49,
                ci_low=0.28,
                ci_high=0.68,
            ),
        ]
        decision = route_task(
            _signal(reasoning_effort="high"),
            [sol, luna],
            policy="cheap",
            community_observations=observations,
            role_preferences={},
        )
        self.assertEqual(decision.model.id, luna.id)
        self.assertNotEqual(decision.score_source, "community_observation")

    def test_effort_mismatch_does_not_apply(self) -> None:
        from puppetmaster.router import route_task

        sol = _spec(
            id="codex/gpt-5.6-sol",
            adapter_model_name="gpt-5.6-sol",
            capability_score=85,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=15.0,
        )
        luna = _luna_low()
        observations = [
            _obs(
                registry_id="codex/gpt-5.6-sol-high",
                adapter="codex",
                effort="high",
                pass_rate=0.90,
                ci_low=0.80,
                ci_high=0.95,
            )
        ]
        decision = route_task(
            _signal(),
            [sol, luna],
            policy="cheap",
            community_observations=observations,
            role_preferences={},
        )
        self.assertEqual(decision.model.id, luna.id)

    def test_community_does_not_grant_sufficiency(self) -> None:
        from puppetmaster.router import NoEligibleModelError, route_task

        weak = _sol_high(capability_score=20)
        observations = [
            _obs(
                registry_id=weak.id,
                adapter="codex",
                effort="high",
                pass_rate=0.99,
                ci_low=0.90,
                ci_high=1.0,
            )
        ]
        with self.assertRaises(NoEligibleModelError):
            route_task(
                _signal(explicit_min_capability=80, strict_capability=True),
                [weak],
                policy="cheap",
                community_observations=observations,
                role_preferences={},
            )

    def test_soft_preference_orders_when_no_community_winner(self) -> None:
        from puppetmaster.role_preferences import RolePreference
        from puppetmaster.router import route_task

        sol = _sol_high()
        luna = _luna_low()
        decision = route_task(
            _signal(),
            [sol, luna],
            policy="cheap",
            community_observations=[],
            role_preferences={
                "implement": RolePreference(preferred=(sol.id,), mode="soft")
            },
        )
        self.assertEqual(decision.model.id, sol.id)
        self.assertEqual(decision.score_source, "preference")

    def test_strict_preference_fail_closed(self) -> None:
        from puppetmaster.role_preferences import RolePreference
        from puppetmaster.router import NoEligibleModelError, route_task

        luna = _luna_low()
        with self.assertRaises(NoEligibleModelError):
            route_task(
                _signal(),
                [luna],
                policy="cheap",
                community_observations=[],
                role_preferences={
                    "implement": RolePreference(
                        preferred=("codex/missing-model",),
                        mode="strict",
                    )
                },
            )

    def test_implement_observation_does_not_change_explore(self) -> None:
        from puppetmaster.router import route_task

        sol = _sol_high()
        luna = _luna_low()
        observations = [
            _obs(
                registry_id=sol.id,
                adapter="codex",
                role="implement",
                effort="high",
                pass_rate=0.70,
                ci_low=0.55,
                ci_high=0.82,
            ),
            _obs(
                registry_id=luna.id,
                adapter="codex",
                role="implement",
                effort="high",
                pass_rate=0.40,
                ci_low=0.20,
                ci_high=0.48,
            ),
        ]
        decision = route_task(
            _signal(role="explore", instruction="explore the repo", reasoning_effort="high"),
            [sol, luna],
            policy="cheap",
            community_observations=observations,
            role_preferences={},
        )
        self.assertEqual(decision.model.id, luna.id)
        self.assertNotEqual(decision.score_source, "community_observation")

    def test_join_uses_the_effort_the_worker_runs(self) -> None:
        """Catalog params are not a pin: the worker runs the effective effort."""
        from puppetmaster.community_observations import match_observation

        spec = _spec(
            id="cursor/grok-4-6",
            adapter="cursor",
            adapter_model_name="grok-4.6",
            payload_defaults={
                "params": [
                    {"id": "effort", "value": "xhigh"},
                    {"id": "fast", "value": "true"},
                ]
            },
        )
        hit = match_observation(
            spec,
            "implement",
            [
                _obs(
                    registry_id="cursor/grok-4-6",
                    adapter="cursor",
                    effort="xhigh",
                    provider="cursor",
                )
            ],
            "xhigh",
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit.effort, "xhigh")
        row = _obs(registry_id="cursor/grok-4-6", adapter="cursor", effort="xhigh", provider="cursor")
        self.assertIsNone(match_observation(spec, "implement", [row], "medium"))
        fx = _spec(id="fx/m", adapter="fx", adapter_model_name="m")
        self.assertIsNone(match_observation(fx, "implement", [_obs(registry_id="fx/m", adapter="fx", effort="medium")], "medium"))

    def test_catalog_effort_does_not_join_an_unpinned_worker(self) -> None:
        """A spec whose catalog says high runs medium when nothing pins effort."""
        from puppetmaster.router import route_task

        sol, luna = _sol_high(), _luna_low()
        observations = [
            _obs(registry_id=sol.id, adapter="codex", effort="high", pass_rate=0.95, ci_low=0.85, ci_high=0.99),
            _obs(registry_id=luna.id, adapter="codex", effort="low", pass_rate=0.10, ci_low=0.02, ci_high=0.20),
        ]
        decision = route_task(_signal(), [sol, luna], policy="cheap",
                              community_observations=observations, role_preferences={})
        self.assertEqual(decision.model.id, luna.id)
        self.assertNotEqual(decision.score_source, "community_observation")

    def test_lone_observation_does_not_override_routing(self) -> None:
        from puppetmaster.router import route_task

        sol, luna = _sol_high(), _luna_low()
        observations = [_obs(registry_id=sol.id, adapter="codex", effort="medium", pass_rate=0.07, ci_low=0.02, ci_high=0.2)]
        decision = route_task(_signal(), [sol, luna], policy="cheap",
                              community_observations=observations, role_preferences={})
        self.assertEqual(decision.model.id, luna.id)
        self.assertNotEqual(decision.score_source, "community_observation")

    def test_task_signals_carry_the_caller_effort_pin(self) -> None:
        from puppetmaster.workers import WorkerSpec
        from puppetmaster.router import signals_from_worker_spec

        signals = signals_from_worker_spec(WorkerSpec(role="implement", instruction="x", adapter="codex",
                                                    payload={"reasoning_effort": "high"}))
        self.assertEqual(signals.reasoning_effort, "high")

    def test_orchestrator_track_rejected_for_worker_role(self) -> None:
        from puppetmaster.community_observations import parse_observation_bundle

        with self.assertRaises(ValueError):
            parse_observation_bundle(
                {
                    "entries": [
                        {
                            "registry_id": "cursor/grok-4-6",
                            "adapter": "cursor",
                            "role": "implement",
                            "effort": "xhigh",
                            "provider": "cursor",
                            "track": "orchestrator",
                            "bank": "b",
                            "harness": "0.6.0",
                            "pass_rate": 0.5,
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
