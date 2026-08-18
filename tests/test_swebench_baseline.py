from __future__ import annotations

import os
import sys
import unittest

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


class SweBenchBaselineTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict:
        rows = []
        for index, resolved in enumerate((20.0, 40.0, 60.0, 80.0, 90.0), start=1):
            rows.append(
                {
                    "agent": "mini-SWE-agent",
                    "name": f"Model {index}",
                    "model_display": f"Model {index}",
                    "mini-swe-agent_version": "2.0.0",
                    "resolved": resolved,
                    "instance_cost": index / 10,
                    "instance_calls": index * 2,
                    "date": "2026-08-18",
                    "per_instance_details": {
                        f"task-{index}-a": {"resolved": True},
                        f"task-{index}-b": {"resolved": False},
                    },
                }
            )
        rows.append(
            {
                "agent": "OtherAgent",
                "name": "Model 3",
                "model_display": "Model 3",
                "resolved": 100.0,
                "per_instance_details": {"other": {"resolved": True}},
            }
        )
        return {"leaderboards": [{"name": "bash-only", "results": rows}]}

    def test_builds_adapter_scoped_implement_card_from_exact_model_mapping(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        bundle = build_swebench_bash_only_bundle(
            self._payload(),
            mappings=[
                RegistryModelMapping(
                    registry_id="codex/model-3",
                    adapter="codex",
                    adapter_model_name="model-3",
                    leaderboard_name="Model 3",
                )
            ],
            source_revision="abc123",
            published="2026-08-18",
        )

        self.assertTrue(bundle["adapter_scoped"])
        self.assertEqual(bundle["version"], "abc123")
        entry = bundle["entries"][0]
        self.assertEqual(entry["id"], "codex/model-3")
        self.assertEqual(entry["adapter"], "codex")
        card = entry["role_scorecards"]["implement"]
        self.assertEqual(card["capability"], 50)
        self.assertEqual(card["quality"], 0.6)
        self.assertEqual(card["sample_count"], 2)
        self.assertEqual(card["cost_per_task_usd"], 0.3)
        self.assertEqual(card["calls_per_task"], 6.0)
        self.assertEqual(card["provenance"]["benchmark"], "swe-bench-bash-only")
        self.assertEqual(card["provenance"]["source_revision"], "abc123")
        self.assertEqual(card["provenance"]["raw_model_name"], "Model 3")
        self.assertEqual(card["provenance"]["capability_method"], "leaderboard_percentile")
        self.assertEqual(card["provenance"]["resolved_scale"], "percent")

    def test_bundle_import_changes_only_role_capability(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.scorecards import effective_capability_score, import_community_baseline
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        spec = ModelSpec(
            id="codex/model-3",
            adapter="codex",
            adapter_model_name="model-3",
            capability_score=97,
        )
        bundle = build_swebench_bash_only_bundle(
            self._payload(),
            mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
            source_revision="abc123",
            published="2026-08-18",
        )

        imported, report = import_community_baseline([spec], bundle)

        self.assertEqual(report["matched"], ["codex/model-3"])
        self.assertEqual(imported[0].capability_score, 97)
        self.assertEqual(effective_capability_score(imported[0], "implement"), 50)
        self.assertEqual(
            imported[0].role_scorecards["implement"]["provenance"]["benchmark"],
            "swe-bench-bash-only",
        )
        from puppetmaster.router import TaskSignals, route_task

        decision = route_task(
            TaskSignals(instruction="implement a feature", role="implement"),
            imported,
            policy="quality",
        )
        self.assertEqual(
            decision.to_artifact_payload()["score_provenance"]["benchmark"],
            "swe-bench-bash-only",
        )
        self.assertEqual(
            decision.to_artifact_payload()["score_provenance"]["source_revision"],
            "abc123",
        )

    def test_uses_published_bash_only_sample_count_when_details_are_not_embedded(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        payload = self._payload()
        del payload["leaderboards"][0]["results"][2]["per_instance_details"]
        bundle = build_swebench_bash_only_bundle(
            payload,
            mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
            source_revision="abc123",
            published="2026-08-18",
        )

        card = bundle["entries"][0]["role_scorecards"]["implement"]
        self.assertEqual(card["sample_count"], 500)
        self.assertEqual(card["provenance"]["sample_count_source"], "bash_only_board_contract")

    def test_percentile_compares_only_the_same_harness_version(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        payload = self._payload()
        payload["leaderboards"][0]["results"].append(
            {
                "agent": "mini-SWE-agent",
                "name": "Old harness winner",
                "model_display": "Old harness winner",
                "mini-swe-agent_version": "1.0.0",
                "resolved": 100.0,
                "per_instance_details": {"old": {"resolved": True}},
            }
        )
        bundle = build_swebench_bash_only_bundle(
            payload,
            mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
            source_revision="abc123",
            published="2026-08-18",
        )

        card = bundle["entries"][0]["role_scorecards"]["implement"]
        self.assertEqual(card["capability"], 50)
        self.assertEqual(card["provenance"]["harness_version"], "2.0.0")

    def test_mixed_percent_and_unit_interval_resolved_fails_closed(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        payload = self._payload()
        payload["leaderboards"][0]["results"][0]["resolved"] = 0.728
        with self.assertRaisesRegex(ValueError, "mixed resolved scale"):
            build_swebench_bash_only_bundle(
                payload,
                mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
                source_revision="abc123",
                published="2026-08-18",
            )

    def test_all_rate_board_keeps_quality_equal_to_resolved(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        payload = self._payload()
        rates = (0.20, 0.40, 0.60, 0.80, 0.90)
        for row, resolved in zip(payload["leaderboards"][0]["results"][:5], rates):
            row["resolved"] = resolved

        bundle = build_swebench_bash_only_bundle(
            payload,
            mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
            source_revision="abc123",
            published="2026-08-18",
        )

        card = bundle["entries"][0]["role_scorecards"]["implement"]
        self.assertEqual(card["quality"], 0.60)
        self.assertNotEqual(card["quality"], 0.006)
        self.assertEqual(card["capability"], 50)
        self.assertEqual(card["provenance"]["resolved_scale"], "rate")

    def test_rejects_out_of_range_quality_and_negative_economics(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        mapping = RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")
        payload = self._payload()
        payload["leaderboards"][0]["results"][2]["resolved"] = 101.0
        with self.assertRaisesRegex(ValueError, "resolved must be 0..100"):
            build_swebench_bash_only_bundle(
                payload,
                mappings=[mapping],
                source_revision="abc123",
                published="2026-08-18",
            )

        payload = self._payload()
        payload["leaderboards"][0]["results"][2]["instance_cost"] = -1.0
        with self.assertRaisesRegex(ValueError, "instance_cost must be non-negative"):
            build_swebench_bash_only_bundle(
                payload,
                mappings=[mapping],
                source_revision="abc123",
                published="2026-08-18",
            )

    def test_unknown_or_ambiguous_model_mapping_fails_closed(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        with self.assertRaisesRegex(ValueError, "not found"):
            build_swebench_bash_only_bundle(
                self._payload(),
                mappings=[RegistryModelMapping("codex/missing", "codex", "missing", "Missing")],
                source_revision="abc123",
                published="2026-08-18",
            )

        payload = self._payload()
        payload["leaderboards"][0]["results"].append(
            {
                **payload["leaderboards"][0]["results"][2],
                "resolved": 61.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            build_swebench_bash_only_bundle(
                payload,
                mappings=[RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
                source_revision="abc123",
                published="2026-08-18",
            )

    def test_fetches_latest_pinned_leaderboard_with_stdlib_callback(self) -> None:
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            fetch_swebench_bash_only_bundle,
        )

        requested: list[str] = []

        def get_json(url: str):
            requested.append(url)
            if "/commits?" in url:
                return [
                    {
                        "sha": "deadbeefcafebabe",
                        "commit": {"committer": {"date": "2026-08-17T23:59:58Z"}},
                    }
                ]
            self.assertIn("deadbeefcafebabe", url)
            return self._payload()

        bundle = fetch_swebench_bash_only_bundle(
            [RegistryModelMapping("codex/model-3", "codex", "model-3", "Model 3")],
            get_json=get_json,
        )

        self.assertEqual(bundle["version"], "deadbeefcafe")
        self.assertEqual(bundle["published"], "2026-08-17")
        self.assertEqual(len(requested), 2)
        self.assertIn("deadbeefcafebabe", bundle["source_url"])


if __name__ == "__main__":
    unittest.main()
