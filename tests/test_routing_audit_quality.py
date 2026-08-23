"""Focused adversarial contract for objective, epoch-aware routing audit.

These tests intentionally exercise only TASK-006.  They preserve the public
legacy ``TaskAuditRecord`` construction shape while requiring newly collected
evidence to remain attributable and qualified before it affects routing.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation


def _legacy_record(model_id: str, *, role: str = "implement", **overrides):
    from puppetmaster.audit import TaskAuditRecord

    values = {
        "model_id": model_id,
        "adapter": "cursor",
        "capability_needed": 60,
        "est_cost_usd": 0.01,
        "confidence": 0.9,
        "escalated": False,
        "escalated_from": None,
        "fell_back": False,
        "role": role,
    }
    values.update(overrides)
    return TaskAuditRecord(**values)


def _save_routed_task_with_reroute(*, reroute_created_by: str, reroute_payload: dict):
    from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
    from puppetmaster.store import SwarmStore

    root = TemporaryDirectory()
    store = SwarmStore(Path(root.name) / ".puppetmaster")
    store.init()
    job = store.create_job("attribute the rejected model")
    task = Task(
        job_id=job.id,
        role="review",
        instruction="review the patch",
        adapter="cursor",
        status=TaskStatus.COMPLETE,
        payload={
            "router_model_id": "cursor/strong",
            "router_capability_needed": 70,
            "router_estimated_cost_usd": 0.02,
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
                "model_id": "cursor/weak",
                "adapter": "cursor",
                "policy": "balanced",
                "capability_needed": 60,
            },
            confidence=0.9,
            evidence=["role:review"],
            created_at="2026-08-22T12:00:00Z",
        )
    )
    store.save_artifact(
        Artifact(
            job_id=job.id,
            task_id=task.id,
            type=ArtifactType.ROUTING,
            created_by=reroute_created_by,
            payload={
                "model_id": "cursor/strong",
                "adapter": "cursor",
                "policy": "balanced",
                **reroute_payload,
            },
            confidence=0.9,
            evidence=["rerouted"],
            created_at="2026-08-22T12:01:00Z",
        )
    )
    store.save_artifact(
        Artifact(
            job_id=job.id,
            task_id=task.id,
            type=ArtifactType.VERIFICATION,
            created_by="strong-worker",
            payload={"check": "review", "result": "passed"},
            confidence=0.95,
            evidence=["test"],
            created_at="2026-08-22T12:02:00Z",
        )
    )
    return root, store


def _objective_record(*, role: str, passed: bool, **epoch):
    return _legacy_record(
        "cursor/model",
        role=role,
        confidence=0.99,
        verification_result="passed",
        gate_passed=passed,
        predicted_quality=0.95,
        objective_quality=1.0 if passed else 0.0,
        objective_passed=passed,
        **epoch,
    )


def _model_with_card(card: dict):
    from puppetmaster.model_registry import ModelSpec

    return ModelSpec(
        id="cursor/card-model",
        adapter="cursor",
        adapter_model_name="card-model",
        capability_score=50,
        role_scorecards={"audit": card},
    )


def _qualified_card() -> dict:
    return {
        "capability": 85,
        "sample_count": 20,
        "last_calibrated": date.today().isoformat(),
        "scale": "puppetmaster-capability-0-100",
        "scale_version": "1",
        "provenance": {
            "source": "paired_evaluation",
            "version": "eval-v1",
        },
    }


class RoutingAuditQualityTests(unittest.TestCase):
    def test_ordinary_fallback_is_attributed_to_the_rejected_model(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records

        root, store = _save_routed_task_with_reroute(
            reroute_created_by="router-fallback",
            reroute_payload={
                "fallback_from_model": "cursor/weak",
                "fallback_reason": "model_unavailable",
            },
        )
        try:
            records, _ = collect_records(store)
            report = build_audit_report(
                records, {"cursor/weak": 55, "cursor/strong": 85}
            )
            self.assertEqual(records[0].fallback_from, "cursor/weak")
            weak = next(model for model in report.models if model.model_id == "cursor/weak")
            self.assertEqual(weak.fell_back_away, 1)
        finally:
            root.cleanup()

    def test_review_escalation_is_attributed_to_the_review_rejected_model(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records

        root, store = _save_routed_task_with_reroute(
            reroute_created_by="router-review-escalation",
            reroute_payload={"review_escalated_from_model": "cursor/weak"},
        )
        try:
            records, _ = collect_records(store)
            report = build_audit_report(
                records, {"cursor/weak": 55, "cursor/strong": 85}
            )
            self.assertEqual(records[0].review_escalated_from, "cursor/weak")
            weak = next(model for model in report.models if model.model_id == "cursor/weak")
            self.assertEqual(weak.review_escalated_away, 1)
        finally:
            root.cleanup()

    def test_two_step_fallback_attributes_every_rejected_model(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("retain every fallback hop")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement with fallbacks",
                adapter="cursor",
                status=TaskStatus.COMPLETE,
                payload={"router_model_id": "cursor/strong"},
            )
            store.save_task(task)
            for created_by, created_at, payload in (
                (
                    "router",
                    "2026-08-22T12:00:00Z",
                    {"model_id": "cursor/weak", "policy": "balanced"},
                ),
                (
                    "router-fallback",
                    "2026-08-22T12:01:00Z",
                    {
                        "model_id": "cursor/mid",
                        "policy": "balanced",
                        "fallback_from_model": "cursor/weak",
                    },
                ),
                (
                    "router-fallback",
                    "2026-08-22T12:02:00Z",
                    {
                        "model_id": "cursor/strong",
                        "policy": "balanced",
                        "fallback_from_model": "cursor/mid",
                    },
                ),
            ):
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by=created_by,
                        payload={"adapter": "cursor", **payload},
                        confidence=0.9,
                        evidence=[created_by],
                        created_at=created_at,
                    )
                )
            records, _ = collect_records(store)
            report = build_audit_report(
                records,
                {"cursor/weak": 50, "cursor/mid": 65, "cursor/strong": 85},
            )
        by_model = {model.model_id: model for model in report.models}
        self.assertEqual(by_model["cursor/weak"].fell_back_away, 1)
        self.assertEqual(by_model["cursor/mid"].fell_back_away, 1)
        self.assertEqual(by_model["cursor/strong"].fell_back_away, 0)

    def test_two_step_review_escalation_attributes_every_rejected_model(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("retain every review escalation hop")
            task = Task(
                job_id=job.id,
                role="review",
                instruction="review through multiple producers",
                adapter="cursor",
                status=TaskStatus.COMPLETE,
                payload={"router_model_id": "cursor/strong"},
            )
            store.save_task(task)
            epoch = {
                "registry_digest": "registry-a",
                "classifier_version": "classifier-a",
                "taxonomy_version": "taxonomy-a",
                "adapter_version": "cursor-a",
            }
            events = (
                (
                    ArtifactType.ROUTING,
                    "router",
                    "2026-08-22T13:00:00Z",
                    {
                        "model_id": "cursor/weak",
                        "adapter": "cursor",
                        "policy": "balanced",
                        **epoch,
                    },
                ),
                (
                    ArtifactType.GATE,
                    "review-gate",
                    "2026-08-22T13:00:10Z",
                    {
                        "gate": "review",
                        "passed": False,
                        "review_status": "completed",
                        "objective_score": 0.0,
                        "evaluator_revision": "review-v2",
                    },
                ),
                (
                    ArtifactType.ROUTING,
                    "router-review-escalation",
                    "2026-08-22T13:00:20Z",
                    {
                        "model_id": "cursor/mid",
                        "adapter": "cursor",
                        "policy": "quality",
                        "review_escalated_from_model": "cursor/weak",
                        **epoch,
                    },
                ),
                (
                    ArtifactType.GATE,
                    "review-gate",
                    "2026-08-22T13:00:30Z",
                    {
                        "gate": "review",
                        "passed": False,
                        "review_status": "completed",
                        "objective_score": 0.0,
                        "evaluator_revision": "review-v2",
                    },
                ),
                (
                    ArtifactType.ROUTING,
                    "router-review-escalation",
                    "2026-08-22T13:00:40Z",
                    {
                        "model_id": "cursor/strong",
                        "adapter": "cursor",
                        "policy": "quality",
                        "review_escalated_from_model": "cursor/mid",
                        **epoch,
                    },
                ),
                (
                    ArtifactType.GATE,
                    "review-gate",
                    "2026-08-22T13:00:50Z",
                    {
                        "gate": "review",
                        "passed": True,
                        "review_status": "completed",
                        "objective_score": 1.0,
                        "evaluator_revision": "review-v2",
                    },
                ),
            )
            for artifact_type, created_by, created_at, payload in events:
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=artifact_type,
                        created_by=created_by,
                        payload=payload,
                        confidence=1.0,
                        evidence=[created_by],
                        created_at=created_at,
                    )
                )
            records, _ = collect_records(store)
            report = build_audit_report(
                records,
                {"cursor/weak": 50, "cursor/mid": 65, "cursor/strong": 85},
            )
        by_model = {model.model_id: model for model in report.models}
        self.assertEqual(by_model["cursor/weak"].review_escalated_away, 1)
        self.assertEqual(by_model["cursor/mid"].review_escalated_away, 1)
        self.assertEqual(by_model["cursor/strong"].review_escalated_away, 0)
        self.assertEqual(by_model["cursor/weak"].runs_with_objective_outcomes, 1)
        self.assertEqual(by_model["cursor/mid"].runs_with_objective_outcomes, 1)
        self.assertEqual(by_model["cursor/strong"].runs_with_objective_outcomes, 1)

    def test_review_rejection_outcome_survives_later_successful_escalated_review(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("retain rejected review outcomes")
            for index in range(5):
                task = Task(
                    job_id=job.id,
                    role="review",
                    instruction=f"review patch {index}",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                    payload={
                        "router_model_id": "cursor/strong",
                        "router_capability_needed": 70,
                    },
                )
                store.save_task(task)
                common_epoch = {
                    "registry_digest": "registry-a",
                    "taxonomy_version": "taxonomy-a",
                    "classifier_version": "classifier-a",
                    "adapter_version": "cursor-a",
                }
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by="router",
                        payload={
                            "model_id": "cursor/weak",
                            "adapter": "cursor",
                            "policy": "balanced",
                            "predicted_quality": 0.9,
                            **common_epoch,
                        },
                        confidence=0.9,
                        evidence=["initial"],
                        created_at=f"2026-08-22T12:{index:02d}:00Z",
                    )
                )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.GATE,
                        created_by="review-gate",
                        payload={
                            "gate": "review",
                            "passed": False,
                            "objective_score": 0.0,
                            "evaluator_revision": "review-v2",
                            "review_status": "completed",
                        },
                        confidence=1.0,
                        evidence=["weak-review"],
                        created_at=f"2026-08-22T12:{index:02d}:10Z",
                    )
                )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by="router-review-escalation",
                        payload={
                            "model_id": "cursor/strong",
                            "adapter": "cursor",
                            "policy": "quality",
                            "review_escalated_from_model": "cursor/weak",
                            "predicted_quality": 0.98,
                            **common_epoch,
                        },
                        confidence=0.9,
                        evidence=["review-escalation"],
                        created_at=f"2026-08-22T12:{index:02d}:20Z",
                    )
                )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.GATE,
                        created_by="review-gate",
                        payload={
                            "gate": "review",
                            "passed": True,
                            "objective_score": 1.0,
                            "evaluator_revision": "review-v2",
                            "review_status": "completed",
                        },
                        confidence=1.0,
                        evidence=["strong-review"],
                        created_at=f"2026-08-22T12:{index:02d}:30Z",
                    )
                )
            records, _ = collect_records(store)
            report = build_audit_report(
                records, {"cursor/weak": 55, "cursor/strong": 85}, min_sample=5
            )
        weak = next(model for model in report.models if model.model_id == "cursor/weak")
        strong = next(model for model in report.models if model.model_id == "cursor/strong")
        self.assertEqual(weak.runs_with_objective_outcomes, 5)
        self.assertAlmostEqual(weak.objective_pass_rate, 0.0)
        self.assertEqual(strong.runs_with_objective_outcomes, 5)
        self.assertAlmostEqual(strong.objective_pass_rate, 1.0)
        self.assertEqual(
            [
                (item["model_id"], item["role"])
                for item in report.role_scorecard_suggestions
            ],
            [("cursor/weak", "review")],
        )
        self.assertEqual(report.role_scorecard_suggestions[0]["last_calibrated"], "2026-08-22")

    def test_collector_compares_predicted_quality_with_objective_evaluator_evidence(self) -> None:
        from puppetmaster.audit import collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("compare prediction to evaluator")
            task = Task(
                job_id=job.id,
                role="audit",
                instruction="audit the result",
                adapter="cursor",
                status=TaskStatus.COMPLETE,
                payload={
                    "router_model_id": "cursor/model",
                    "router_capability_needed": 70,
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
                        "model_id": "cursor/model",
                        "adapter": "cursor",
                        "policy": "balanced",
                        "predicted_quality": 0.95,
                        "registry_digest": "registry-a",
                        "taxonomy_version": "taxonomy-a",
                        "adapter_version": "cursor-a",
                    },
                    confidence=0.9,
                    evidence=["policy:balanced"],
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GATE,
                    created_by="review-gate",
                    payload={
                        "gate": "review",
                        "passed": False,
                        "objective_score": 0.2,
                        "evaluator_revision": "review-v2",
                        "reviewed_artifact_fingerprint": "sha256:artifact-a",
                    },
                    confidence=1.0,
                    evidence=["evaluator:review-v2"],
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="self-reporting-worker",
                    payload={"check": "self", "result": "passed"},
                    confidence=0.99,
                    evidence=["self-report"],
                )
            )
            records, _ = collect_records(store)
        record = records[0]
        self.assertAlmostEqual(record.predicted_quality, 0.95)
        self.assertAlmostEqual(record.objective_quality, 0.2)
        self.assertIs(record.objective_passed, False)
        self.assertAlmostEqual(record.confidence, 0.99)
        self.assertEqual(record.evaluator_revision, "review-v2")
        self.assertEqual(record.registry_digest, "registry-a")
        self.assertEqual(record.taxonomy_version, "taxonomy-a")
        self.assertEqual(record.adapter_version, "cursor-a")

    def test_unavailable_review_is_not_recorded_as_an_objective_evaluator_outcome(self) -> None:
        from puppetmaster.audit import collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.quality import assess_run_quality
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("review unavailable")
            task = Task(
                job_id=job.id,
                role="review",
                instruction="review",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={"router_model_id": "cursor/model"},
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "cursor/model",
                        "adapter": "cursor",
                        "policy": "balanced",
                    },
                    confidence=0.9,
                    evidence=["route"],
                )
            )
            unavailable = Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.GATE,
                created_by="review-gate",
                payload={
                    "gate": "review",
                    "passed": False,
                    "review_status": "unavailable",
                    "evaluator_revision": "review-v2",
                    "reviewed_artifact_fingerprint": "sha256:artifact-a",
                },
                confidence=1.0,
                evidence=["no-judge"],
            )
            store.save_artifact(unavailable)
            records, _ = collect_records(store)
            quality = assess_run_quality([unavailable])
        self.assertIsNone(records[0].objective_passed)
        self.assertIsNone(records[0].objective_quality)
        self.assertEqual(quality["semantic_quality"], "not_evaluated")
        self.assertEqual(quality["objective_evaluations"], 0)

    def test_failed_objective_evaluation_keeps_objective_trust_basis(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.quality import assess_run_quality

        gate = Artifact(
            job_id="job-quality",
            task_id="task-quality",
            type=ArtifactType.GATE,
            created_by="review-gate",
            payload={
                "gate": "review",
                "passed": False,
                "review_status": "completed",
                "objective_score": 0.0,
                "evaluator_revision": "review-v2",
            },
            confidence=1.0,
            evidence=["objective-review"],
        )
        quality = assess_run_quality([gate])
        self.assertEqual(quality["semantic_quality"], "failed")
        self.assertEqual(quality["objective_evaluations"], 1)
        self.assertEqual(quality["trust_basis"], "objective_evaluator")

    def test_objective_failures_drive_only_the_underperforming_role_recommendation(self) -> None:
        from puppetmaster.audit import build_audit_report

        epoch = {
            "registry_digest": "registry-a",
            "taxonomy_version": "taxonomy-a",
            "adapter_version": "cursor-a",
            "evaluator_revision": "eval-a",
            "evaluated_at": "2026-08-22T12:00:00Z",
        }
        records = [
            *[_objective_record(role="audit", passed=False, **epoch) for _ in range(5)],
            *[_objective_record(role="implement", passed=True, **epoch) for _ in range(5)],
        ]
        report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)
        model = next(item for item in report.models if item.model_id == "cursor/model")
        self.assertEqual(model.runs_with_objective_outcomes, 10)
        self.assertAlmostEqual(model.objective_pass_rate, 0.5)
        self.assertAlmostEqual(model.mean_predicted_quality, 0.95)
        self.assertAlmostEqual(model.mean_objective_quality, 0.5)
        self.assertIsNone(model.suggested_score)
        self.assertEqual(report.suggestions, [])
        self.assertEqual([item["role"] for item in report.role_scorecard_suggestions], ["audit"])
        self.assertEqual(report.role_scorecard_suggestions[0]["model_id"], "cursor/model")

    def test_fully_qualified_role_card_affects_effective_capability(self) -> None:
        from puppetmaster.scorecards import effective_capability_score

        spec = _model_with_card(_qualified_card())
        score = effective_capability_score(spec, "audit")
        self.assertEqual(score, 85)

    def test_swebench_card_producer_emits_qualified_routing_authority(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.scorecards import (
            effective_capability_score,
            import_community_baseline,
        )
        from puppetmaster.swebench_baseline import (
            RegistryModelMapping,
            build_swebench_bash_only_bundle,
        )

        rows = []
        for index, resolved in enumerate((20.0, 40.0, 60.0, 80.0, 90.0), start=1):
            rows.append(
                {
                    "agent": "mini-SWE-agent",
                    "name": f"Model {index}",
                    "model_display": f"Model {index}",
                    "mini-swe-agent_version": "2.0.0",
                    "resolved": resolved,
                    "date": date.today().isoformat(),
                    "per_instance_details": {
                        f"task-{index}-{sample}": {"resolved": sample % 2 == 0}
                        for sample in range(10)
                    },
                }
            )
        bundle = build_swebench_bash_only_bundle(
            {"leaderboards": [{"name": "bash-only", "results": rows}]},
            mappings=[
                RegistryModelMapping(
                    "codex/model-3", "codex", "model-3", "Model 3"
                )
            ],
            source_revision="abc123",
            published=date.today().isoformat(),
        )
        spec = ModelSpec(
            id="codex/model-3",
            adapter="codex",
            adapter_model_name="model-3",
            capability_score=97,
        )
        imported, _ = import_community_baseline([spec], bundle)
        card = imported[0].role_scorecards["implement"]
        effective = effective_capability_score(imported[0], "implement")
        self.assertEqual(card["scale"], "puppetmaster-capability-0-100")
        self.assertTrue(card["scale_version"])
        self.assertEqual(card["provenance"]["source"], "community_benchmark")
        self.assertTrue(card["provenance"]["version"])
        self.assertEqual(effective, card["capability"])

    def test_unqualified_role_card_falls_back_to_manual_capability(self) -> None:
        from puppetmaster.scorecards import effective_capability_score

        mutators = {
            "missing-provenance": lambda card: card.update(provenance={}),
            "too-few-samples": lambda card: card.update(sample_count=4),
            "missing-scale": lambda card: card.pop("scale"),
            "missing-scale-version": lambda card: card.pop("scale_version"),
            "stale-calibration": lambda card: card.update(last_calibrated="2000-01-01"),
        }
        for case_id, mutate in mutators.items():
            with self.subTest(case_id):
                card = _qualified_card()
                mutate(card)
                spec = _model_with_card(card)
                score = effective_capability_score(spec, "audit")
                self.assertEqual(score, 50)

    def test_historical_aggregation_does_not_pool_incompatible_epochs(self) -> None:
        from puppetmaster.audit import build_audit_report

        cases = (
            ("registry_digest", "registry-a", "registry-b"),
            ("taxonomy_version", "taxonomy-a", "taxonomy-b"),
            ("adapter_version", "cursor-a", "cursor-b"),
            ("evaluator_revision", "eval-a", "eval-b"),
        )
        for epoch_field, first, second in cases:
            with self.subTest(epoch_field):
                base_epoch = {
                    "registry_digest": "registry-a",
                    "taxonomy_version": "taxonomy-a",
                    "adapter_version": "cursor-a",
                    "evaluator_revision": "eval-a",
                    "evaluated_at": "2026-08-22T12:00:00Z",
                }
                first_epoch = {**base_epoch, epoch_field: first}
                second_epoch = {**base_epoch, epoch_field: second}
                records = [
                    *[
                        _objective_record(role="audit", passed=False, **first_epoch)
                        for _ in range(3)
                    ],
                    *[
                        _objective_record(role="audit", passed=False, **second_epoch)
                        for _ in range(3)
                    ],
                ]
                report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)
                self.assertEqual(report.epoch_count, 2)
                self.assertEqual(report.role_scorecard_suggestions, [])

    def test_incomplete_epoch_cannot_authorize_a_role_card_recommendation(self) -> None:
        from puppetmaster.audit import build_audit_report

        records = [
            _objective_record(role="audit", passed=False)
            for _ in range(5)
        ]
        report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)
        self.assertEqual(report.epoch_count, 1)
        self.assertEqual(report.role_scorecard_suggestions, [])

    def test_undated_complete_lineage_is_reportable_but_non_authoritative(self) -> None:
        from puppetmaster.audit import build_audit_report

        epoch_without_date = {
            "registry_digest": "registry-a",
            "taxonomy_version": "taxonomy-a",
            "classifier_version": "classifier-a",
            "adapter_version": "cursor-a",
            "evaluator_revision": "eval-a",
        }
        records = [
            _objective_record(role="audit", passed=False, **epoch_without_date)
            for _ in range(5)
        ]
        report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)
        model = next(item for item in report.models if item.model_id == "cursor/model")
        self.assertEqual(model.runs_with_objective_outcomes, 5)
        self.assertAlmostEqual(model.objective_pass_rate, 0.0)
        self.assertEqual(report.epoch_count, 1)
        self.assertEqual(report.role_scorecard_suggestions, [])

    def test_stale_objective_history_cannot_be_re_stamped_as_a_fresh_role_card(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("stale objective history")
            for index in range(5):
                task = Task(
                    job_id=job.id,
                    role="audit",
                    instruction=f"historical audit {index}",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                    payload={"router_model_id": "cursor/model"},
                )
                store.save_task(task)
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.ROUTING,
                        created_by="router",
                        payload={
                            "model_id": "cursor/model",
                            "adapter": "cursor",
                            "policy": "balanced",
                            "predicted_quality": 0.95,
                            "registry_digest": "registry-historical",
                            "taxonomy_version": "taxonomy-historical",
                            "classifier_version": "classifier-historical",
                            "adapter_version": "cursor-historical",
                        },
                        confidence=0.9,
                        evidence=["historical-route"],
                        created_at=f"1999-12-31T23:59:{index:02d}Z",
                    )
                )
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.GATE,
                        created_by="review-gate",
                        payload={
                            "gate": "review",
                            "passed": False,
                            "objective_score": 0.0,
                            "review_status": "completed",
                            "evaluator_revision": "review-historical",
                        },
                        confidence=1.0,
                        evidence=["historical-evaluation"],
                        created_at=f"2000-01-01T00:00:{index:02d}Z",
                    )
                )
            records, _ = collect_records(store)
            report = build_audit_report(
                records, {"cursor/model": 80}, min_sample=5
            )
        self.assertEqual(report.role_scorecard_suggestions, [])

    def test_latest_qualified_epoch_wins_role_card_recommendation(self) -> None:
        from puppetmaster.audit import build_audit_report

        common = {
            "registry_digest": "registry-a",
            "taxonomy_version": "taxonomy-a",
            "classifier_version": "classifier-a",
            "adapter_version": "cursor-a",
        }
        records = [
            *[
                _objective_record(
                    role="audit",
                    passed=False,
                    evaluator_revision="eval-a",
                    evaluated_at="2026-05-20T12:00:00Z",
                    **common,
                )
                for _ in range(5)
            ],
            *[
                _objective_record(
                    role="audit",
                    passed=False,
                    evaluator_revision="eval-b",
                    evaluated_at="2026-08-22T12:00:00Z",
                    **common,
                )
                for _ in range(5)
            ],
        ]
        report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)
        self.assertEqual(len(report.role_scorecard_suggestions), 1)
        suggestion = report.role_scorecard_suggestions[0]
        self.assertEqual(suggestion["epoch"]["evaluator_revision"], "eval-b")
        self.assertEqual(suggestion["last_calibrated"], "2026-08-22")

    def test_legacy_audit_record_and_report_shape_remain_supported(self) -> None:
        from puppetmaster.audit import build_audit_report

        records = [_legacy_record("cursor/legacy") for _ in range(2)]
        report = build_audit_report(records, {"cursor/legacy": 60})
        self.assertEqual(report.tasks_considered, 2)
        self.assertEqual(report.epoch_count, 1)
        self.assertEqual(report.suggestions, [])
        model = report.models[0]
        self.assertEqual(model.model_id, "cursor/legacy")
        self.assertEqual(model.selections, 2)
        self.assertEqual(model.runs_with_objective_outcomes, 0)
        self.assertIsNone(model.objective_pass_rate)


if __name__ == "__main__":
    unittest.main()
