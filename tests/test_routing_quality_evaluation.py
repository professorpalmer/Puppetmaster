"""Independent TASK-007 RED contract for paired routing-quality evidence.

The reviewer authored these tests before inspecting an author solution.  They
exercise a public, provider-neutral evaluation seam and production routing
behavior.  No test contacts a model provider or treats artifact structure as
semantic correctness.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster.model_registry import ModelSpec


REQUIRED_ROLES = {"implement", "explore", "audit", "plan"}
ARM_NAMES = {"routed_balanced", "strongest_eligible"}


def _model(
    model_id: str,
    *,
    capability: int,
    cost: float,
    context_window: int = 16_000,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        adapter="cursor",
        adapter_model_name=model_id.rsplit("/", 1)[-1],
        capability_score=capability,
        input_per_mtok_usd=cost,
        output_per_mtok_usd=cost,
        context_window=context_window,
        billing="plan",
    )


def _corpus_payload() -> dict:
    cases = []
    for role in sorted(REQUIRED_ROLES):
        case_id = f"{role}-deterministic"
        snapshot_files = {
            f"fixtures/{role}/authority.txt": (
                f"The fixture-specific accepted result is EXPECTED::{case_id}."
            )
        }
        canonical_snapshot = json.dumps(
            snapshot_files,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        snapshot_digest = "sha256:" + hashlib.sha256(canonical_snapshot).hexdigest()
        cases.append(
            {
                "case_id": case_id,
                "role": role,
                "instruction": f"Complete the deterministic {role} fixture.",
                "snapshot_id": f"fixture-{role}-v1",
                "snapshot_digest": snapshot_digest,
                "snapshot_files": snapshot_files,
                "min_capability": 60,
                "estimated_tokens_in": 1_000,
                "estimated_tokens_out": 200,
                "allowed_changed_files": [f"fixtures/{role}/allowed.txt"],
                "criteria": [
                    {
                        "criterion_id": f"{case_id}-marker",
                        "description": "The fixture-specific success marker is present.",
                        "evaluator": {
                            "kind": "contains",
                            "expected": f"EXPECTED::{case_id}",
                        },
                    }
                ],
                "seeded_failures": (
                    [
                        {
                            "failure_id": "missing-required-marker",
                            "description": "Output omits the required deterministic marker.",
                            "output_text": "A plausible answer that misses the fixture fact.",
                            "changed_files": [],
                            "catastrophic": False,
                        }
                    ]
                    if role == "audit"
                    else []
                ),
            }
        )
    return {
        "schema_version": 1,
        "corpus_id": "task-007-test-corpus",
        "corpus_version": "2026-08-22.1",
        "cases": cases,
    }


def _write_corpus(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "routing-quality-corpus.json"
    path.write_text(json.dumps(payload or _corpus_payload(), indent=2), encoding="utf-8")
    return path


def _passing_executor(request):
    """Return deterministic provider-neutral receipts for a runner request."""
    return {
        "output_text": request.case.criteria[0].expected,
        "snapshot_digest": request.snapshot_digest,
        "changed_files": list(request.case.allowed_changed_files[:1]),
        "catastrophic": False,
        "correction_cycles": 0,
        "elapsed_seconds": 1.0,
        "retries": 0,
        "tokens_in": 1_000,
        "tokens_out": 100,
        "nominal_cost_usd": 0.01,
        "marginal_cost_usd": 0.0,
    }


class RoutingQualityEvaluationTests(unittest.TestCase):
    def test_default_corpus_is_versioned_multi_role_and_has_seeded_failures(self) -> None:
        from puppetmaster.routing_evaluation import (
            grade_case,
            load_evaluation_corpus,
            snapshot_digest_for_files,
        )

        corpus = load_evaluation_corpus()
        self.assertGreaterEqual(corpus.schema_version, 1)
        self.assertTrue(corpus.corpus_id)
        self.assertTrue(corpus.corpus_version)
        self.assertGreaterEqual({case.role for case in corpus.cases}, REQUIRED_ROLES)
        self.assertEqual(len({case.case_id for case in corpus.cases}), len(corpus.cases))
        self.assertTrue(any(case.seeded_failures for case in corpus.cases))
        self.assertTrue(all(case.criteria for case in corpus.cases))
        self.assertTrue(all(case.snapshot_digest.startswith("sha256:") for case in corpus.cases))
        self.assertTrue(all(case.snapshot_files for case in corpus.cases))
        self.assertTrue(
            all(
                snapshot_digest_for_files(case.snapshot_files) == case.snapshot_digest
                for case in corpus.cases
            )
        )
        self.assertTrue(
            all(
                criterion.evaluator_kind != "llm_judge"
                for case in corpus.cases
                for criterion in case.criteria
            )
        )
        self.assertTrue(
            all(
                criterion.expected.casefold() not in case.instruction.casefold()
                for case in corpus.cases
                for criterion in case.criteria
            )
        )
        for case in corpus.cases:
            for failure in case.seeded_failures:
                grade = grade_case(
                    case,
                    output_text=failure.output_text,
                    changed_files=failure.changed_files,
                    catastrophic=failure.catastrophic,
                )
                self.assertIs(grade.acceptance_passed, False)

    def test_corpus_loader_fails_closed_without_version_or_required_roles(self) -> None:
        from puppetmaster.routing_evaluation import load_evaluation_corpus

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unversioned = _corpus_payload()
            unversioned.pop("corpus_version")
            unversioned_path = _write_corpus(tmp_path, unversioned)
            with self.assertRaisesRegex(ValueError, "corpus_version"):
                load_evaluation_corpus(unversioned_path)

            incomplete = _corpus_payload()
            incomplete["corpus_version"] = "2026-08-22.2"
            incomplete["cases"] = [
                case for case in incomplete["cases"] if case["role"] != "plan"
            ]
            incomplete_path = tmp_path / "incomplete.json"
            incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plan"):
                load_evaluation_corpus(incomplete_path)

    def test_corpus_loader_rejects_a_seeded_failure_that_passes_acceptance(self) -> None:
        from puppetmaster.routing_evaluation import load_evaluation_corpus

        payload = _corpus_payload()
        audit_case = next(case for case in payload["cases"] if case["role"] == "audit")
        audit_case["seeded_failures"][0]["output_text"] = audit_case["criteria"][0][
            "evaluator"
        ]["expected"]
        with TemporaryDirectory() as tmp:
            path = _write_corpus(Path(tmp), payload)
            with self.assertRaisesRegex(ValueError, "seeded failure.*must fail"):
                load_evaluation_corpus(path)

    def test_deterministic_grader_scores_criteria_and_unintended_files(self) -> None:
        from puppetmaster.routing_evaluation import grade_case, load_evaluation_corpus

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        case = next(item for item in corpus.cases if item.role == "implement")
        allowed = list(case.allowed_changed_files)
        first = grade_case(
            case,
            output_text=f"done\n{case.criteria[0].expected}\n",
            changed_files=allowed,
            catastrophic=False,
        )
        repeat = grade_case(
            case,
            output_text=f"done\n{case.criteria[0].expected}\n",
            changed_files=allowed,
            catastrophic=False,
        )
        failed = grade_case(
            case,
            output_text="plausible prose without the required marker",
            changed_files=[*allowed, "unintended.txt"],
            catastrophic=False,
        )
        self.assertEqual(first, repeat)
        self.assertIs(first.acceptance_passed, True)
        self.assertAlmostEqual(first.criterion_score, 1.0)
        self.assertEqual(first.unintended_files, ())
        self.assertIs(failed.acceptance_passed, False)
        self.assertAlmostEqual(failed.criterion_score, 0.0)
        self.assertEqual(failed.unintended_files, ("unintended.txt",))

    def test_paired_runner_uses_identical_snapshots_and_strongest_eligible_baseline(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            run_paired_evaluation,
        )

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [
            _model("cursor/bargain", capability=70, cost=0.10),
            _model("cursor/frontier", capability=95, cost=1.00),
            _model(
                "cursor/overflowing-strongest",
                capability=100,
                cost=2.00,
                context_window=1_100,
            ),
        ]
        calls = []

        def execute(request):
            calls.append(request)
            return _passing_executor(request)

        report = run_paired_evaluation(
            corpus,
            registry,
            execute=execute,
            repetitions=3,
            seed=41,
        )
        grouped = defaultdict(list)
        for request in calls:
            grouped[(request.case.case_id, request.repetition)].append(request)
        self.assertEqual(len(grouped), len(corpus.cases) * 3)
        for pair in grouped.values():
            self.assertEqual({request.arm for request in pair}, ARM_NAMES)
            self.assertEqual(len({request.snapshot_digest for request in pair}), 1)
            model_by_arm = {request.arm: request.model_id for request in pair}
            self.assertEqual(
                model_by_arm,
                {
                    "routed_balanced": "cursor/bargain",
                    "strongest_eligible": "cursor/frontier",
                },
            )
        arm_orders = {tuple(pair.arm_order) for pair in report.pairs}
        self.assertEqual(
            arm_orders,
            {
                ("routed_balanced", "strongest_eligible"),
                ("strongest_eligible", "routed_balanced"),
            },
        )

    def test_paired_runner_requires_three_repetitions(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            run_paired_evaluation,
        )

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [_model("cursor/model", capability=90, cost=1.0)]
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run_paired_evaluation(
                corpus,
                registry,
                execute=_passing_executor,
                repetitions=2,
                seed=1,
            )

    def test_paired_runner_rejects_unverified_or_mismatched_snapshot_receipts(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            run_paired_evaluation,
        )

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [_model("cursor/model", capability=90, cost=1.0)]

        def mismatched_executor(request):
            receipt = dict(_passing_executor(request))
            receipt["snapshot_digest"] = "sha256:" + "0" * 64
            return receipt

        with self.assertRaisesRegex(ValueError, "snapshot_digest"):
            run_paired_evaluation(
                corpus,
                registry,
                execute=mismatched_executor,
                repetitions=3,
                seed=1,
            )

    def test_report_includes_quality_cost_failure_metrics_and_paired_uncertainty(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            render_evaluation_report,
            run_paired_evaluation,
        )

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [
            _model("cursor/bargain", capability=70, cost=0.10),
            _model("cursor/frontier", capability=95, cost=1.00),
        ]

        def execute(request):
            routed_passes = request.repetition != 1
            baseline_passes = request.repetition != 2
            passed = routed_passes if request.arm == "routed_balanced" else baseline_passes
            return {
                "output_text": request.case.criteria[0].expected if passed else "missing",
                "snapshot_digest": request.snapshot_digest,
                "changed_files": (
                    [*request.case.allowed_changed_files, "unintended.txt"]
                    if request.arm == "routed_balanced" and request.repetition == 1
                    else list(request.case.allowed_changed_files)
                ),
                "catastrophic": request.arm == "strongest_eligible" and request.repetition == 2,
                "correction_cycles": request.repetition,
                "elapsed_seconds": 2.0 + request.repetition,
                "retries": 1 if request.repetition == 1 else 0,
                "tokens_in": 1_000 + request.repetition,
                "tokens_out": 100 + request.repetition,
                "nominal_cost_usd": 0.02 + request.repetition / 100,
                "marginal_cost_usd": 0.01 + request.repetition / 100,
            }

        first = run_paired_evaluation(
            corpus,
            registry,
            execute=execute,
            repetitions=3,
            seed=19,
            noninferiority_margin=0.05,
        )
        repeat = run_paired_evaluation(
            corpus,
            registry,
            execute=execute,
            repetitions=3,
            seed=19,
            noninferiority_margin=0.05,
        )
        payload = first.to_dict()
        self.assertEqual(payload, repeat.to_dict())
        self.assertEqual(payload["claim"]["status"], "inconclusive")
        self.assertEqual(payload["claim"]["basis"], "paired_noninferiority")
        required_metrics = {
            "acceptance_pass_rate",
            "criterion_score_mean",
            "unintended_file_rate",
            "catastrophic_failure_rate",
            "correction_cycles_mean",
            "elapsed_seconds_mean",
            "retries_mean",
            "tokens_in_mean",
            "tokens_out_mean",
            "nominal_cost_usd_mean",
            "marginal_cost_usd_mean",
        }
        self.assertGreaterEqual(set(payload["arms"]["routed_balanced"]), required_metrics)
        self.assertGreaterEqual(set(payload["arms"]["strongest_eligible"]), required_metrics)
        self.assertGreaterEqual(set(payload["paired_deltas"]), required_metrics)
        for metric in required_metrics:
            uncertainty = payload["paired_deltas"][metric]
            self.assertGreaterEqual(set(uncertainty), {"estimate", "lower", "upper", "method"})
            self.assertLessEqual(uncertainty["lower"], uncertainty["estimate"])
            self.assertLessEqual(uncertainty["estimate"], uncertainty["upper"])

        markdown = render_evaluation_report(first).lower()
        self.assertIn("inconclusive", markdown)
        self.assertIn("paired non-inferiority", markdown)
        self.assertIn("does not establish semantic quality", markdown)
        self.assertNotIn("proves that routing improves quality", markdown)

    def test_finite_all_equal_sample_does_not_collapse_quality_uncertainty_to_zero(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            run_paired_evaluation,
        )

        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [
            _model("cursor/bargain", capability=70, cost=0.10),
            _model("cursor/frontier", capability=95, cost=1.00),
        ]
        report = run_paired_evaluation(
            corpus,
            registry,
            execute=_passing_executor,
            repetitions=3,
            seed=7,
            noninferiority_margin=0.05,
        )
        quality = report.to_dict()["paired_deltas"]["acceptance_pass_rate"]
        self.assertLess(quality["lower"], quality["estimate"])
        self.assertLess(quality["estimate"], quality["upper"])
        self.assertEqual(report.claim["status"], "inconclusive")

    def test_noninferiority_margin_fails_closed_outside_finite_unit_interval(self) -> None:
        from puppetmaster.routing_evaluation import (
            load_evaluation_corpus,
            run_paired_evaluation,
        )

        cases = (
            (True, "boolean"),
            (float("nan"), "nan"),
            (float("inf"), "infinite"),
            (1.01, "above-pass-rate-range"),
        )
        with TemporaryDirectory() as tmp:
            corpus = load_evaluation_corpus(_write_corpus(Path(tmp)))
        registry = [_model("cursor/model", capability=90, cost=1.0)]
        for invalid_margin, case_id in cases:
            with self.subTest(case_id):
                with self.assertRaisesRegex(ValueError, "noninferiority_margin"):
                    run_paired_evaluation(
                        corpus,
                        registry,
                        execute=_passing_executor,
                        repetitions=3,
                        seed=1,
                        noninferiority_margin=invalid_margin,
                    )

    def test_shadow_routing_is_opt_in_and_cannot_change_production_selection(self) -> None:
        from puppetmaster import router

        task = router.TaskSignals(
            instruction="perform the bounded task",
            role="explore",
            explicit_min_capability=60,
            estimated_tokens_in=1_000,
            estimated_tokens_out=200,
        )
        registry = [
            _model("cursor/bargain", capability=70, cost=0.10),
            _model("cursor/frontier", capability=95, cost=1.00),
        ]
        production = router.route_task(task, registry, policy="balanced")
        shadowed = router.route_task(
            task,
            registry,
            policy="balanced",
            shadow_policy="quality",
        )
        production_payload = production.to_artifact_payload()
        shadow_payload = shadowed.to_artifact_payload()
        self.assertEqual(production.model.id, "cursor/bargain")
        self.assertEqual(shadowed.model.id, production.model.id)
        self.assertNotIn("shadow_routing", production_payload)
        self.assertEqual(shadow_payload["model_id"], production_payload["model_id"])
        self.assertEqual(shadow_payload["adapter"], production_payload["adapter"])
        self.assertEqual(shadow_payload["policy"], production_payload["policy"])
        self.assertEqual(
            shadow_payload["shadow_routing"],
            {
                "enabled": True,
                "policy": "quality",
                "production_model_id": "cursor/bargain",
                "counterfactual_model_id": "cursor/frontier",
                "production_selection_changed": False,
            },
        )

    def test_orchestrator_shadow_opt_in_persists_evidence_without_changing_dispatch(self) -> None:
        from puppetmaster.model_registry import save_registry
        from puppetmaster.models import ArtifactType
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.store_factory import create_store
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "models.json"
            save_registry(
                [
                    _model("cursor/bargain", capability=70, cost=0.10),
                    _model("cursor/frontier", capability=95, cost=1.00),
                ],
                registry_path,
            )
            store = create_store("file", tmp_path / ".puppetmaster")
            store.init()
            orchestrator = Orchestrator(store)
            job = store.create_job("shadow route production seam")
            spec = WorkerSpec(
                role="explore",
                instruction="perform the bounded task",
                adapter="local",
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                    "routing_policy": "balanced",
                    "min_capability": 60,
                    "estimated_tokens_in": 1_000,
                    "estimated_tokens_out": 200,
                    "shadow_policy": "quality",
                },
            )
            with patch(
                "puppetmaster.preflight.adapter_cli_present",
                lambda _adapter: True,
            ):
                task = orchestrator._create_tasks(job, [spec])[0]
            routing = [
                artifact
                for artifact in store.list_artifacts(job.id)
                if artifact.type == ArtifactType.ROUTING
            ]
            self.assertEqual(task.payload["router_model_id"], "cursor/bargain")
            self.assertEqual(task.payload["model"], "bargain")
            self.assertEqual(len(routing), 1)
            self.assertEqual(routing[0].payload["model_id"], "cursor/bargain")
            self.assertEqual(
                routing[0].payload["shadow_routing"],
                {
                    "enabled": True,
                    "policy": "quality",
                    "production_model_id": "cursor/bargain",
                    "counterfactual_model_id": "cursor/frontier",
                    "production_selection_changed": False,
                },
            )

    def test_documentation_states_quality_claim_boundaries(self) -> None:
        text = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "routing-quality-evaluation.md"
        ).read_text(encoding="utf-8").lower()
        self.assertIn("non-inferiority", text)
        self.assertIn("inconclusive", text)
        self.assertIn("structural artifact presence", text)
        self.assertIn("not semantic quality", text)
        self.assertNotIn("proves that routing improves quality", text)


if __name__ == "__main__":
    unittest.main()
