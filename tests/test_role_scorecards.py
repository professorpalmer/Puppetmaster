"""Issue #28 first slice: role scorecards + provenance."""
from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


def _spec(**kwargs):
    from puppetmaster.model_registry import ModelSpec

    defaults = dict(
        id="cursor/mini",
        adapter="cursor",
        adapter_model_name="mini",
        capability_score=90,
    )
    defaults.update(kwargs)
    return ModelSpec(**defaults)


def _qualified_card(capability, **extra):
    card = {
        "capability": capability,
        "sample_count": 5,
        "last_calibrated": date.today().isoformat(),
        "scale": "puppetmaster-capability-0-100",
        "scale_version": "1",
        "provenance": {"source": "test_objective_evaluator", "version": "1"},
    }
    card.update(extra)
    return card


def _three_tier():
    from puppetmaster.model_registry import ModelSpec

    return [
        ModelSpec(
            id="cheap-model",
            adapter="claude-code",
            adapter_model_name="cheap-v1",
            capability_score=40,
            input_per_mtok_usd=0.10,
            output_per_mtok_usd=0.50,
            tags=["cheap", "fast"],
        ),
        ModelSpec(
            id="mid-model",
            adapter="claude-code",
            adapter_model_name="mid-v1",
            capability_score=80,
            input_per_mtok_usd=3.0,
            output_per_mtok_usd=15.0,
            tags=["balanced"],
        ),
        ModelSpec(
            id="frontier-model",
            adapter="claude-code",
            adapter_model_name="frontier-v1",
            capability_score=95,
            input_per_mtok_usd=15.0,
            output_per_mtok_usd=75.0,
            tags=["frontier", "reasoning"],
        ),
    ]


class RegistryScorecardRoundTripTests(unittest.TestCase):
    def test_empty_cards_omitted_from_disk_json(self) -> None:
        from puppetmaster.model_registry import load_registry, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([_spec()], path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("role_scorecards", raw["models"][0])
            self.assertNotIn("score_provenance", raw["models"][0])
            loaded = load_registry(path)
            self.assertEqual(loaded[0].role_scorecards, {})
            self.assertEqual(loaded[0].score_provenance, {})

    def test_cards_round_trip(self) -> None:
        from puppetmaster.model_registry import load_registry, save_registry

        spec = _spec(
            role_scorecards={"implement": {"capability": 72, "sample_count": 12}},
            score_provenance={"source": "manual", "notes": "tuned"},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([spec], path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                raw["models"][0]["role_scorecards"]["implement"]["capability"], 72
            )
            self.assertEqual(raw["models"][0]["score_provenance"]["source"], "manual")
            loaded = load_registry(path)
            self.assertEqual(loaded[0].role_scorecards["implement"]["capability"], 72)
            self.assertEqual(loaded[0].capability_score, 90)

    def test_old_models_json_without_new_keys_still_loads(self) -> None:
        from puppetmaster.model_registry import load_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "legacy/one",
                                "adapter": "cursor",
                                "adapter_model_name": "legacy",
                                "capability_score": 55,
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_registry(path)
            self.assertEqual(loaded[0].id, "legacy/one")
            self.assertEqual(loaded[0].capability_score, 55)
            self.assertEqual(loaded[0].role_scorecards, {})
            self.assertEqual(loaded[0].score_provenance, {})


class EffectiveCapabilityTests(unittest.TestCase):
    def test_empty_cards_equal_manual_score(self) -> None:
        from puppetmaster.scorecards import effective_capability_score

        spec = _spec(capability_score=97)
        self.assertEqual(effective_capability_score(spec, "implement"), 97)
        self.assertEqual(effective_capability_score(spec, "explore"), 97)

    def test_only_qualified_role_card_overrides_matching_role(self) -> None:
        from puppetmaster.scorecards import effective_capability_score

        unqualified = _spec(
            capability_score=97,
            role_scorecards={"implement": {"capability": 70}},
        )
        self.assertEqual(effective_capability_score(unqualified, "implement"), 97)

        qualified = _spec(
            capability_score=97,
            role_scorecards={
                "implement": {
                    "capability": 70,
                    "sample_count": 5,
                    "last_calibrated": date.today().isoformat(),
                    "scale": "puppetmaster-capability-0-100",
                    "scale_version": "1",
                    "provenance": {
                        "source": "test_objective_evaluator",
                        "version": "1",
                    },
                }
            },
        )
        self.assertEqual(effective_capability_score(qualified, "implement"), 70)
        self.assertEqual(effective_capability_score(qualified, "explore"), 97)
        self.assertEqual(effective_capability_score(qualified, "review"), 97)


class ImportBaselineTests(unittest.TestCase):
    def _bundle(self, entries):
        return {
            "bundle_id": "puppetmaster-community-scorecards",
            "version": "1.0.0",
            "published": "2026-08-18",
            "adapter_scoped": True,
            "entries": entries,
        }

    def test_matches_adapter_and_id_never_changes_capability_score(self) -> None:
        from puppetmaster.scorecards import import_community_baseline

        specs = [
            _spec(
                id="cursor/grok-4-6",
                adapter="cursor",
                adapter_model_name="grok-4.6",
                capability_score=99,
            )
        ]
        bundle = self._bundle(
            [
                {
                    "id": "cursor/grok-4-6",
                    "adapter": "cursor",
                    "adapter_model_name": "grok-4.6",
                    "role_scorecards": {"implement": {"capability": 72}},
                }
            ]
        )
        new_specs, report = import_community_baseline(specs, bundle)
        self.assertEqual(new_specs[0].capability_score, 99)
        self.assertEqual(new_specs[0].role_scorecards["implement"]["capability"], 72)
        self.assertEqual(new_specs[0].score_provenance["source"], "community_baseline")
        self.assertEqual(report["matched"], ["cursor/grok-4-6"])
        self.assertEqual(report["cards_added"], 1)
        self.assertEqual(report["skipped_adapter_mismatch"], [])

    def test_exact_variant_id_prevents_effort_evidence_cross_contamination(self) -> None:
        from puppetmaster.scorecards import import_community_baseline

        low = _spec(
            id="agentic/openai-api/gpt-5.6-luna-low",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            payload_defaults={"provider": "openai-api", "reasoning_effort": "low"},
        )
        max_effort = _spec(
            id="agentic/openai-api/gpt-5.6-luna-max",
            adapter="agentic",
            adapter_model_name="gpt-5.6-luna",
            payload_defaults={"provider": "openai-api", "reasoning_effort": "max"},
        )
        bundle = self._bundle([
            {
                "id": max_effort.id,
                "adapter": "agentic",
                "adapter_model_name": "gpt-5.6-luna",
                "role_scorecards": {"implement": {"quality": 0.9}},
            }
        ])

        imported, report = import_community_baseline([low, max_effort], bundle)
        by_id = {spec.id: spec for spec in imported}

        self.assertEqual(by_id[low.id].role_scorecards, {})
        self.assertEqual(by_id[max_effort.id].role_scorecards["implement"]["quality"], 0.9)
        self.assertEqual(report["matched"], [max_effort.id])
        self.assertEqual(report["cards_added"], 1)

    def test_skips_adapter_mismatch(self) -> None:
        from puppetmaster.scorecards import import_community_baseline

        specs = [
            _spec(
                id="cursor/grok-4-6",
                adapter="openai",
                adapter_model_name="grok-4.6",
                capability_score=88,
            )
        ]
        bundle = self._bundle(
            [
                {
                    "id": "cursor/grok-4-6",
                    "adapter": "cursor",
                    "adapter_model_name": "grok-4.6",
                    "role_scorecards": {"implement": {"capability": 72}},
                }
            ]
        )
        new_specs, report = import_community_baseline(specs, bundle)
        self.assertEqual(new_specs[0].capability_score, 88)
        self.assertEqual(new_specs[0].role_scorecards, {})
        self.assertEqual(len(report["skipped_adapter_mismatch"]), 1)
        self.assertIn("adapter mismatch", report["skipped_adapter_mismatch"][0]["reason"])
        self.assertEqual(report["cards_added"], 0)

    def test_local_cards_win_unless_replace_cards(self) -> None:
        from puppetmaster.scorecards import import_community_baseline

        specs = [
            _spec(
                id="cursor/mini",
                adapter="cursor",
                adapter_model_name="mini",
                role_scorecards={"implement": {"capability": 80}},
            )
        ]
        bundle = self._bundle(
            [
                {
                    "id": "cursor/mini",
                    "adapter": "cursor",
                    "adapter_model_name": "mini",
                    "role_scorecards": {
                        "implement": {"capability": 60},
                        "explore": {"quality": 0.4},
                    },
                }
            ]
        )
        kept, report = import_community_baseline(specs, bundle, replace_cards=False)
        self.assertEqual(kept[0].role_scorecards["implement"]["capability"], 80)
        self.assertEqual(kept[0].role_scorecards["explore"]["quality"], 0.4)
        self.assertEqual(report["cards_added"], 1)
        replaced, _ = import_community_baseline(specs, bundle, replace_cards=True)
        self.assertEqual(replaced[0].role_scorecards["implement"]["capability"], 60)

    def test_published_bundle_loads_and_does_not_mutate_starter_scores(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.scorecards import (
            default_community_baseline_path,
            import_community_baseline,
            load_community_baseline,
        )

        path = default_community_baseline_path()
        self.assertTrue(path.is_file(), f"missing published baseline at {path}")
        packaged = (
            Path(__file__).resolve().parents[1]
            / "puppetmaster"
            / "baselines"
            / "role-scorecards-v1.json"
        )
        self.assertTrue(packaged.is_file(), f"missing packaged baseline at {packaged}")
        bundle = load_community_baseline(path)
        self.assertTrue(bundle["adapter_scoped"])
        before = {s.id: s.capability_score for s in starter_registry()}
        after_specs, report = import_community_baseline(starter_registry(), bundle)
        after = {s.id: s.capability_score for s in after_specs}
        self.assertEqual(before, after)
        self.assertGreaterEqual(len(report["matched"]), 2)

    def test_import_baseline_cli_dry_run_writes_nothing(self) -> None:
        import argparse

        from puppetmaster.cli.commands_models import _run_models_import_baseline
        from puppetmaster.model_registry import load_registry, save_registry
        from puppetmaster.scorecards import default_community_baseline_path

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(
                [
                    _spec(
                        id="cursor/grok-4-6",
                        adapter="cursor",
                        adapter_model_name="grok-4.6",
                        capability_score=99,
                    )
                ],
                registry_path,
            )
            before = registry_path.read_text(encoding="utf-8")
            args = argparse.Namespace(
                path=str(default_community_baseline_path()),
                replace_cards=False,
                dry_run=True,
            )
            rc = _run_models_import_baseline(args, registry_path)
            self.assertEqual(rc, 0)
            self.assertEqual(registry_path.read_text(encoding="utf-8"), before)
            loaded = load_registry(registry_path)
            self.assertEqual(loaded[0].role_scorecards, {})


class RouteTaskScorecardTests(unittest.TestCase):
    def test_empty_cards_use_manual_scores_across_cost_policies(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="add a feature", role="implement")
        decision = route_task(signal, _three_tier(), policy="balanced")
        self.assertEqual(decision.model.id, "mid-model")
        cheap = route_task(signal, _three_tier(), policy="cheap")
        self.assertEqual(cheap.model.id, "mid-model")
        absolute = route_task(signal, _three_tier(), policy="absolute-cheapest")
        self.assertEqual(absolute.model.id, "cheap-model")

    def test_implement_card_rejects_high_manual_score(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import TaskSignals, route_task

        mini = ModelSpec(
            id="cursor/mini",
            adapter="cursor",
            adapter_model_name="mini",
            capability_score=90,
            input_per_mtok_usd=0.5,
            output_per_mtok_usd=2.0,
            billing="api",
            role_scorecards={"implement": _qualified_card(72)},
        )
        gpt = ModelSpec(
            id="cursor/gpt-5-5",
            adapter="cursor",
            adapter_model_name="gpt-5.5",
            capability_score=97,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            billing="api",
            role_scorecards={"implement": _qualified_card(70)},
            score_provenance={
                "source": "community_baseline",
                "bundle_id": "test-bundle",
            },
        )
        signal = TaskSignals(
            instruction="implement a feature",
            role="implement",
            explicit_min_capability=80,
        )
        decision = route_task(signal, [mini, gpt], policy="balanced")
        self.assertNotEqual(decision.model.id, "cursor/gpt-5-5")
        self.assertEqual(decision.model.id, "cursor/mini")
        self.assertIn("role=implement card capability", decision.reason)
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["score_source"], "role_card")
        self.assertEqual(payload["effective_capability_score"], 72)
        self.assertEqual(payload["capability_score"], 90)

    def test_artifact_payload_includes_provenance_when_card_used(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import TaskSignals, route_task

        spec = ModelSpec(
            id="cursor/workhorse",
            adapter="cursor",
            adapter_model_name="workhorse",
            capability_score=97,
            input_per_mtok_usd=2.0,
            output_per_mtok_usd=6.0,
            billing="plan",
            role_scorecards={
                "implement": _qualified_card(
                    88,
                    quality=0.71,
                    latency_p50_ms=1200,
                    sample_count=40,
                    provenance={
                        "source": "community_baseline",
                        "version": "1.0.0",
                        "bundle_id": "puppetmaster-community-scorecards",
                    },
                )
            },
            score_provenance={
                "source": "community_baseline",
                "bundle_id": "puppetmaster-community-scorecards",
                "bundle_version": "1.0.0",
            },
        )
        cheap = ModelSpec(
            id="cursor/cheap",
            adapter="cursor",
            adapter_model_name="cheap",
            capability_score=40,
            input_per_mtok_usd=0.1,
            output_per_mtok_usd=0.2,
            billing="plan",
        )
        signal = TaskSignals(instruction="implement a feature", role="implement")
        decision = route_task(signal, [cheap, spec], policy="balanced")
        self.assertEqual(decision.model.id, "cursor/workhorse")
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["score_source"], "role_card")
        self.assertEqual(
            payload["score_provenance"]["source"], "community_baseline"
        )
        self.assertEqual(payload["sample_count"], 40)
        self.assertEqual(payload["predicted_quality"], 0.71)


class DiscoveryPreservesCardsTests(unittest.TestCase):
    def test_cursor_overlay_preserves_cards_kin_does_not(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        overlay = ModelSpec(
            id="cursor/gpt-5-5",
            adapter="cursor",
            adapter_model_name="gpt-5.5",
            capability_score=92,
            tags=["cursor", "frontier"],
            role_scorecards={"implement": {"capability": 81}},
            score_provenance={"source": "manual"},
        )
        kin = ModelSpec(
            id="claude-code/opus-4-8",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-8",
            capability_score=99,
            tags=["frontier"],
            role_scorecards={"implement": {"capability": 70}},
            score_provenance={"source": "community_baseline"},
        )
        catalog = [
            {"id": "gpt-5.5", "displayName": "GPT 5.5"},
            {"id": "claude-opus-4-8", "displayName": "Claude Opus 4.8"},
        ]
        specs = {s.adapter_model_name: s for s in catalog_to_specs(catalog, [overlay, kin])}
        self.assertEqual(
            specs["gpt-5.5"].role_scorecards["implement"]["capability"], 81
        )
        self.assertEqual(specs["gpt-5.5"].score_provenance["source"], "manual")
        self.assertEqual(specs["claude-opus-4-8"].capability_score, 99)
        self.assertEqual(specs["claude-opus-4-8"].role_scorecards, {})
        self.assertEqual(specs["claude-opus-4-8"].score_provenance, {})

    def test_api_overlay_preserves_cards(self) -> None:
        from puppetmaster.api_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(
                id="openai/gpt-5-5",
                adapter="openai",
                adapter_model_name="gpt-5.5",
                capability_score=96,
                billing="api",
                role_scorecards={"review": {"capability": 90}},
                score_provenance={"source": "local_audit"},
            )
        ]
        specs = catalog_to_specs("openai", "api", [{"id": "gpt-5.5"}], existing)
        self.assertEqual(specs[0].role_scorecards["review"]["capability"], 90)
        self.assertEqual(specs[0].score_provenance["source"], "local_audit")

    def test_curated_overlay_preserves_cards(self) -> None:
        from puppetmaster.static_catalog import curated_catalog, curated_to_specs
        from puppetmaster.model_registry import ModelSpec

        catalog = curated_catalog("codex")
        self.assertTrue(catalog)
        model = str(catalog[0]["model"])
        existing = [
            ModelSpec(
                id=f"codex/{model}",
                adapter="codex",
                adapter_model_name=model,
                capability_score=91,
                role_scorecards={"implement": {"capability": 85}},
                score_provenance={"source": "manual", "notes": "keep"},
            )
        ]
        specs = {
            s.adapter_model_name: s for s in curated_to_specs("codex", "plan", existing)
        }
        self.assertEqual(specs[model].role_scorecards["implement"]["capability"], 85)
        self.assertEqual(specs[model].score_provenance["source"], "manual")
        self.assertEqual(specs[model].capability_score, 91)


class AuditScorecardTests(unittest.TestCase):
    def test_collect_records_captures_role_elapsed_and_verification(self) -> None:
        from puppetmaster.audit import collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scorecard goal")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="do it",
                adapter="cursor",
                status=TaskStatus.COMPLETE,
                created_at="2026-08-18T12:00:00+00:00",
                completed_at="2026-08-18T12:00:12+00:00",
                attempts=2,
                payload={
                    "router_model_id": "m/60",
                    "router_capability_needed": 60,
                    "router_estimated_cost_usd": 0.001,
                    "fallback_attempts": 1,
                    "escalation_attempts": 0,
                },
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "m/60",
                        "adapter": "cursor",
                        "policy": "balanced",
                        "capability_needed": 60,
                    },
                    confidence=0.9,
                    evidence=["r"],
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="w",
                    payload={"check": "x", "result": "passed"},
                    confidence=0.92,
                    evidence=["e"],
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GATE,
                    created_by="gate",
                    payload={"gate": "tests", "passed": True},
                    confidence=1.0,
                    evidence=["g"],
                )
            )
            records, jobs = collect_records(store)
            self.assertEqual(jobs, 1)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.role, "implement")
            self.assertEqual(record.elapsed_seconds, 12.0)
            self.assertEqual(record.verification_result, "passed")
            self.assertTrue(record.gate_passed)
            self.assertEqual(record.attempts, 2)
            self.assertEqual(record.fallback_attempts, 1)

    def test_apply_ignores_self_signal_and_preserves_authority(self) -> None:
        import argparse

        from puppetmaster.cli import _run_audit_command
        from puppetmaster.model_registry import load_registry, save_registry
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            cards = {"implement": {"capability": 50, "sample_count": 9}}
            save_registry(
                [
                    _spec(
                        id="weak/55",
                        adapter="cursor",
                        adapter_model_name="weak",
                        capability_score=55,
                        role_scorecards=cards,
                        score_provenance={"source": "manual"},
                    ),
                    _spec(
                        id="strong/80",
                        adapter="cursor",
                        adapter_model_name="strong",
                        capability_score=80,
                    ),
                ],
                registry_path,
            )
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("g")
            for i in range(6):
                escalated = i < 4
                final = "strong/80" if escalated else "weak/55"
                task = Task(
                    job_id=job.id,
                    role="implement",
                    instruction="x",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                    payload={
                        "router_model_id": final,
                        "router_capability_needed": 50,
                        "router_estimated_cost_usd": 0.001,
                    },
                )
                store.save_task(task)
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by="router",
                        payload={
                            "model_id": "weak/55",
                            "adapter": "cursor",
                            "policy": "balanced",
                            "capability_needed": 50,
                        },
                        confidence=0.9,
                        evidence=["r"],
                    )
                )
                if escalated:
                    store.save_artifact(
                        Artifact(
                            job_id=job.id,
                            task_id=task.id,
                            type=ArtifactType.ROUTING,
                            created_by="router-escalation",
                            payload={
                                "model_id": "strong/80",
                                "adapter": "cursor",
                                "policy": "escalating",
                                "escalated_from_model": "weak/55",
                            },
                            confidence=0.9,
                            evidence=["e"],
                        )
                    )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.VERIFICATION,
                        created_by="w",
                        payload={"check": "c", "result": "passed"},
                        confidence=0.5 if not escalated else 0.95,
                        evidence=["v"],
                    )
                )

            args = argparse.Namespace(
                registry_path=str(registry_path),
                window=None,
                apply=True,
                json=False,
            )
            rc = _run_audit_command(args, store)
            self.assertEqual(rc, 0)
            after = {s.id: s for s in load_registry(registry_path)}
            self.assertEqual(after["weak/55"].capability_score, 55)
            self.assertEqual(after["strong/80"].capability_score, 80)
            self.assertEqual(
                after["weak/55"].role_scorecards["implement"]["capability"], 50
            )
            self.assertEqual(after["weak/55"].score_provenance["source"], "manual")

    def test_apply_writes_only_qualified_objective_role_card(self) -> None:
        import argparse

        from puppetmaster.cli import _run_audit_command
        from puppetmaster.model_registry import load_registry, save_registry
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            review_card = {"capability": 73, "notes": "preserve unrelated role"}
            save_registry(
                [
                    _spec(
                        id="weak/55",
                        adapter="cursor",
                        adapter_model_name="weak",
                        capability_score=55,
                        role_scorecards={"review": review_card},
                        score_provenance={"source": "manual"},
                    ),
                    _spec(
                        id="strong/80",
                        adapter="cursor",
                        adapter_model_name="strong",
                        capability_score=80,
                    ),
                ],
                registry_path,
            )
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("objective role-card authority")
            epoch = {
                "registry_digest": "registry-a",
                "classifier_version": "classifier-a",
                "taxonomy_version": "taxonomy-a",
                "adapter_version": "cursor-a",
            }
            for index in range(6):
                task = Task(
                    job_id=job.id,
                    role="implement",
                    instruction=f"objective evaluation {index}",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                    payload={
                        "router_model_id": "weak/55",
                        "router_capability_needed": 50,
                        "router_estimated_cost_usd": 0.001,
                    },
                )
                store.save_task(task)
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by="router",
                        payload={
                            "model_id": "weak/55",
                            "adapter": "cursor",
                            "policy": "balanced",
                            "capability_needed": 50,
                            **epoch,
                        },
                        confidence=0.9,
                        evidence=["routing"],
                        created_at=f"2026-08-22T12:00:{index:02d}Z",
                    )
                )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.GATE,
                        created_by="objective-review",
                        payload={
                            "gate": "review",
                            "passed": False,
                            "review_status": "completed",
                            "objective_score": 0.0,
                            "evaluator_revision": "review-v2",
                        },
                        confidence=1.0,
                        evidence=["objective evaluator"],
                        created_at=f"2026-08-22T12:01:{index:02d}Z",
                    )
                )

            args = argparse.Namespace(
                registry_path=str(registry_path),
                window=None,
                apply=True,
                json=False,
            )
            rc = _run_audit_command(args, store)

            self.assertEqual(rc, 0)
            after = {s.id: s for s in load_registry(registry_path)}
            weak = after["weak/55"]
            self.assertEqual(weak.capability_score, 55)
            self.assertEqual(after["strong/80"].capability_score, 80)
            self.assertEqual(weak.role_scorecards["review"], review_card)
            implement = weak.role_scorecards["implement"]
            self.assertEqual(implement["capability"], 50)
            self.assertEqual(implement["sample_count"], 6)
            self.assertEqual(implement["last_calibrated"], "2026-08-22")
            self.assertEqual(implement["scale"], "puppetmaster-capability-0-100")
            self.assertEqual(implement["scale_version"], "1")
            self.assertEqual(implement["provenance"]["source"], "local_audit")
            self.assertEqual(
                implement["provenance"]["epoch"],
                {**epoch, "evaluator_revision": "review-v2"},
            )
            self.assertEqual(weak.score_provenance["source"], "manual")


if __name__ == "__main__":
    unittest.main()
