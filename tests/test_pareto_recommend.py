"""Issue #107: default routing must not pick raw GPT-5 across streams.

Ranking can be global (Agent Arena Pareto). Availability is per lane:
(adapter, provider). Cursor GPT-5.6 Sol must not imply Codex GPT-5.4 mini
or retire an agentic openai-codex GPT-5 row. unittest only.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


def _spec(**kwargs):
    from puppetmaster.model_registry import ModelSpec

    defaults = dict(
        billing="plan",
        input_per_mtok_usd=1.0,
        output_per_mtok_usd=6.0,
        tags=["tools", "cursor"],
        capability_score=90,
    )
    defaults.update(kwargs)
    if "adapter_model_name" not in defaults:
        defaults["adapter_model_name"] = defaults["id"].split("/", 1)[-1]
    if "adapter" not in defaults:
        defaults["adapter"] = defaults["id"].split("/", 1)[0]
    return ModelSpec(**defaults)


class LaneIdentityTests(unittest.TestCase):
    def test_agentic_splits_by_provider(self) -> None:
        from puppetmaster.pareto_recommend import spec_lane

        cursor = _spec(id="cursor/gpt-5.6-sol", adapter_model_name="gpt-5.6-sol")
        codex = _spec(id="codex/gpt-5.4-mini", adapter_model_name="gpt-5.4-mini")
        api = _spec(
            id="agentic/gpt-5",
            adapter="agentic",
            adapter_model_name="gpt-5",
            payload_defaults={"provider": "openai-api"},
            billing="api",
        )
        mari_codex = _spec(
            id="agentic/gpt-5.6-luna",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            payload_defaults={"provider": "openai-codex"},
            billing="plan",
        )
        self.assertEqual(spec_lane(cursor), ("cursor", ""))
        self.assertEqual(spec_lane(codex), ("codex", ""))
        self.assertEqual(spec_lane(api), ("agentic", "openai-api"))
        self.assertEqual(spec_lane(mari_codex), ("agentic", "openai-codex"))
        self.assertNotEqual(spec_lane(api), spec_lane(mari_codex))


class PriorGenerationFilterTests(unittest.TestCase):
    def test_same_lane_cursor_gpt5_dropped_when_luna_present(self) -> None:
        from puppetmaster.pareto_recommend import filter_prior_generation

        gpt5 = _spec(
            id="cursor/gpt-5",
            adapter_model_name="gpt-5",
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
        )
        luna = _spec(id="cursor/gpt-5.6-luna", adapter_model_name="gpt-5.6-luna")
        kept, rejected = filter_prior_generation([gpt5, luna])
        self.assertEqual([spec.id for spec in kept], [luna.id])
        self.assertEqual(rejected[0][0].id, gpt5.id)

    def test_cursor_sol_does_not_drop_agentic_codex_gpt5(self) -> None:
        from puppetmaster.pareto_recommend import filter_prior_generation

        cursor_sol = _spec(
            id="cursor/gpt-5.6-sol", adapter_model_name="gpt-5.6-sol"
        )
        mari_gpt5 = _spec(
            id="agentic/gpt-5",
            adapter="agentic",
            adapter_model_name="gpt-5",
            payload_defaults={"provider": "openai-codex"},
            billing="plan",
        )
        kept, rejected = filter_prior_generation([cursor_sol, mari_gpt5])
        ids = {spec.id for spec in kept}
        self.assertIn(cursor_sol.id, ids)
        self.assertIn(mari_gpt5.id, ids)
        self.assertEqual(rejected, [])

    def test_fail_open_when_lane_would_empty(self) -> None:
        from puppetmaster.pareto_recommend import filter_prior_generation

        gpt5 = _spec(id="cursor/gpt-5", adapter_model_name="gpt-5")
        kept, rejected = filter_prior_generation([gpt5])
        self.assertEqual([spec.id for spec in kept], [gpt5.id])
        self.assertEqual(rejected, [])


class BentonDefaultPickTests(unittest.TestCase):
    def test_balanced_implement_skips_same_lane_gpt5_for_luna(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        gpt5 = _spec(
            id="cursor/gpt-5",
            adapter_model_name="gpt-5",
            capability_score=90,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
        )
        luna = _spec(
            id="cursor/gpt-5.6-luna",
            adapter_model_name="gpt-5.6-luna",
            capability_score=90,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=6.0,
        )
        composer = _spec(
            id="cursor/composer-2-5",
            adapter_model_name="composer-2.5",
            capability_score=55,
            input_per_mtok_usd=0.5,
            output_per_mtok_usd=2.5,
        )
        decision = route_task(
            TaskSignals(instruction="implement a feature", role="implement"),
            [gpt5, luna, composer],
            policy="balanced",
        )
        self.assertEqual(decision.model.id, luna.id)
        rejected = {item["id"]: item["reason"] for item in decision.to_artifact_payload()["rejected"]}
        self.assertIn(gpt5.id, rejected)
        self.assertIn("prior generation", rejected[gpt5.id])

    def test_cursor_sol_does_not_make_codex_mini_eligible(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        sol = _spec(
            id="cursor/gpt-5.6-sol",
            adapter_model_name="gpt-5.6-sol",
            capability_score=99,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
        )
        mini = _spec(
            id="codex/gpt-5.4-mini",
            adapter_model_name="gpt-5.4-mini",
            capability_score=70,
            input_per_mtok_usd=0.75,
            output_per_mtok_usd=4.5,
        )
        decision = route_task(
            TaskSignals(
                instruction="implement a feature",
                role="implement",
                allowed_adapters=frozenset(["cursor"]),
            ),
            [sol, mini],
            policy="balanced",
        )
        self.assertEqual(decision.model.id, sol.id)
        self.assertNotEqual(decision.model.id, mini.id)

    def test_explicit_gpt5_allowlist_still_routes(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        gpt5 = _spec(id="cursor/gpt-5", adapter_model_name="gpt-5")
        luna = _spec(id="cursor/gpt-5.6-luna", adapter_model_name="gpt-5.6-luna")
        decision = route_task(
            TaskSignals(
                instruction="implement a feature",
                role="implement",
                allowed_model_ids=frozenset(["cursor/gpt-5"]),
            ),
            [gpt5, luna],
            policy="balanced",
        )
        self.assertEqual(decision.model.id, gpt5.id)


class ParetoApplyLaneTests(unittest.TestCase):
    def test_cursor_sol_allowlist_does_not_disable_agentic_codex_gpt5(self) -> None:
        from puppetmaster.pareto_recommend import apply_pareto_recommendations

        sol = _spec(id="cursor/gpt-5.6-sol", adapter_model_name="gpt-5.6-sol")
        mari_gpt5 = _spec(
            id="agentic/gpt-5",
            adapter="agentic",
            adapter_model_name="gpt-5",
            payload_defaults={"provider": "openai-codex"},
            billing="plan",
        )
        updated, report = apply_pareto_recommendations(
            [sol, mari_gpt5],
            available_ids=["cursor/gpt-5.6-sol"],
            stamp_effort=False,
        )
        by_id = {spec.id: spec for spec in updated}
        self.assertTrue(by_id[mari_gpt5.id].enabled)
        self.assertNotIn(mari_gpt5.id, report["disabled_prior_generation"])
        self.assertIn("pareto-workhorse", by_id[sol.id].tags)
        self.assertNotIn("pareto-workhorse", by_id[mari_gpt5.id].tags)

    def test_same_lane_luna_is_workhorse_and_disables_gpt5(self) -> None:
        from puppetmaster.pareto_recommend import apply_pareto_recommendations

        gpt5 = _spec(id="cursor/gpt-5", adapter_model_name="gpt-5")
        luna = _spec(id="cursor/gpt-5.6-luna", adapter_model_name="gpt-5.6-luna")
        sol = _spec(
            id="cursor/gpt-5.6-sol",
            adapter_model_name="gpt-5.6-sol",
            capability_score=99,
        )
        updated, report = apply_pareto_recommendations(
            [gpt5, luna, sol],
            stamp_effort=False,
        )
        by_id = {spec.id: spec for spec in updated}
        self.assertFalse(by_id[gpt5.id].enabled)
        self.assertEqual(by_id[gpt5.id].disabled_authority, "arena_pareto")
        self.assertIn(luna.id, report["workhorse_ids"])
        self.assertNotIn(sol.id, report["workhorse_ids"])

    def test_user_toggle_is_not_auto_disabled(self) -> None:
        from puppetmaster.pareto_recommend import apply_pareto_recommendations

        gpt5 = _spec(
            id="cursor/gpt-5",
            adapter_model_name="gpt-5",
            disabled_reason="keep for 2+2",
            disabled_authority="user",
            enabled=False,
        )
        luna = _spec(id="cursor/gpt-5.6-luna", adapter_model_name="gpt-5.6-luna")
        updated, _report = apply_pareto_recommendations(
            [gpt5, luna], stamp_effort=False
        )
        by_id = {spec.id: spec for spec in updated}
        self.assertEqual(by_id[gpt5.id].disabled_authority, "user")
        self.assertEqual(by_id[gpt5.id].disabled_reason, "keep for 2+2")


class CatalogDemotionTests(unittest.TestCase):
    def test_static_gpt5_is_legacy_not_deleted(self) -> None:
        from puppetmaster.static_catalog import curated_to_specs

        hermes = {s.adapter_model_name: s for s in curated_to_specs("hermes", "api", [])}
        agentic = {
            s.adapter_model_name: s for s in curated_to_specs("agentic", "api", [])
        }
        self.assertIn("gpt-5", hermes)
        self.assertIn("gpt-5", agentic)
        self.assertEqual(hermes["gpt-5"].capability_score, 62)
        self.assertEqual(agentic["gpt-5"].capability_score, 62)
        self.assertIn("legacy", hermes["gpt-5"].tags)
        self.assertIn("legacy", agentic["gpt-5"].tags)

    def test_cursor_discover_gpt5_is_not_zero_nominal(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs

        specs = {
            s.adapter_model_name: s
            for s in catalog_to_specs([{"id": "gpt-5", "displayName": "GPT-5"}], [])
        }
        self.assertEqual(specs["gpt-5"].input_per_mtok_usd, 1.25)
        self.assertEqual(specs["gpt-5"].output_per_mtok_usd, 10.0)
        self.assertGreater(specs["gpt-5"].input_per_mtok_usd, 0.0)


class ModelsRecommendCliTests(unittest.TestCase):
    def test_parser_has_recommend(self) -> None:
        from puppetmaster.cli._parser import build_parser

        args = build_parser().parse_args(["models", "recommend", "--json"])
        self.assertEqual(args.models_command, "recommend")
        self.assertTrue(args.json)

    def test_recommend_dry_run_does_not_write(self) -> None:
        from puppetmaster.cli.commands_models import _run_models_recommend
        from puppetmaster.model_registry import save_registry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [
                    _spec(id="cursor/gpt-5", adapter_model_name="gpt-5"),
                    _spec(id="cursor/gpt-5.6-luna", adapter_model_name="gpt-5.6-luna"),
                ],
                path,
            )
            args = SimpleNamespace(
                json=True,
                write=False,
                available=None,
                stamp_scores=False,
                no_stamp_effort=True,
            )
            buf_stdout = __import__("io").StringIO()
            from contextlib import redirect_stdout

            with redirect_stdout(buf_stdout):
                rc = _run_models_recommend(args, path)
            self.assertEqual(rc, 0)
            payload = json.loads(buf_stdout.getvalue())
            self.assertIn("cursor/gpt-5", payload["report"]["disabled_prior_generation"])
            from puppetmaster.model_registry import load_registry

            still = {spec.id: spec for spec in load_registry(path)}
            self.assertTrue(still["cursor/gpt-5"].enabled)


if __name__ == "__main__":
    unittest.main()
