"""Focused adversarial contract for objective, epoch-aware routing audit.

These tests intentionally exercise only TASK-006.  They preserve the public
legacy ``TaskAuditRecord`` construction shape while requiring newly collected
evidence to remain attributable and qualified before it affects routing.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


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


def test_ordinary_fallback_is_attributed_to_the_rejected_model() -> None:
    # Arrange
    from puppetmaster.audit import build_audit_report, collect_records

    root, store = _save_routed_task_with_reroute(
        reroute_created_by="router-fallback",
        reroute_payload={
            "fallback_from_model": "cursor/weak",
            "fallback_reason": "model_unavailable",
        },
    )

    try:
        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records, {"cursor/weak": 55, "cursor/strong": 85}
        )

        # Assert
        assert records[0].fallback_from == "cursor/weak"
        weak = next(model for model in report.models if model.model_id == "cursor/weak")
        assert weak.fell_back_away == 1
    finally:
        root.cleanup()


def test_review_escalation_is_attributed_to_the_review_rejected_model() -> None:
    # Arrange
    from puppetmaster.audit import build_audit_report, collect_records

    root, store = _save_routed_task_with_reroute(
        reroute_created_by="router-review-escalation",
        reroute_payload={"review_escalated_from_model": "cursor/weak"},
    )

    try:
        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records, {"cursor/weak": 55, "cursor/strong": 85}
        )

        # Assert
        assert records[0].review_escalated_from == "cursor/weak"
        weak = next(model for model in report.models if model.model_id == "cursor/weak")
        assert weak.review_escalated_away == 1
    finally:
        root.cleanup()


def test_two_step_fallback_attributes_every_rejected_model() -> None:
    # Arrange: weak fails over to mid, then mid fails over to strong. A single
    # latest fallback_from field must not erase the first rejected producer.
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

        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records,
            {"cursor/weak": 50, "cursor/mid": 65, "cursor/strong": 85},
        )

    # Assert
    by_model = {model.model_id: model for model in report.models}
    assert by_model["cursor/weak"].fell_back_away == 1
    assert by_model["cursor/mid"].fell_back_away == 1
    assert by_model["cursor/strong"].fell_back_away == 0


def test_two_step_review_escalation_attributes_every_rejected_model() -> None:
    # Arrange: weak and mid each fail completed objective review before strong
    # passes. Both rejected producers must retain review-escalation attribution.
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

        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records,
            {"cursor/weak": 50, "cursor/mid": 65, "cursor/strong": 85},
        )

    # Assert
    by_model = {model.model_id: model for model in report.models}
    assert by_model["cursor/weak"].review_escalated_away == 1
    assert by_model["cursor/mid"].review_escalated_away == 1
    assert by_model["cursor/strong"].review_escalated_away == 0
    assert by_model["cursor/weak"].runs_with_objective_outcomes == 1
    assert by_model["cursor/mid"].runs_with_objective_outcomes == 1
    assert by_model["cursor/strong"].runs_with_objective_outcomes == 1


def test_review_rejection_outcome_survives_later_successful_escalated_review() -> None:
    # Arrange: each task first fails objective review on weak, escalates, and
    # later passes objective review on strong. The later success must not erase
    # the rejected producer's outcome from the audit.
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

        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records, {"cursor/weak": 55, "cursor/strong": 85}, min_sample=5
        )

    # Assert: both producer outcomes remain auditable, and only the weak review
    # role is recommended for adjustment.
    weak = next(model for model in report.models if model.model_id == "cursor/weak")
    strong = next(model for model in report.models if model.model_id == "cursor/strong")
    assert weak.runs_with_objective_outcomes == 5
    assert weak.objective_pass_rate == pytest.approx(0.0)
    assert strong.runs_with_objective_outcomes == 5
    assert strong.objective_pass_rate == pytest.approx(1.0)
    assert [
        (item["model_id"], item["role"])
        for item in report.role_scorecard_suggestions
    ] == [("cursor/weak", "review")]
    assert report.role_scorecard_suggestions[0]["last_calibrated"] == "2026-08-22"


def test_collector_compares_predicted_quality_with_objective_evaluator_evidence() -> None:
    # Arrange
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

        # Act
        records, _ = collect_records(store)

    # Assert: objective failure is retained even though the model was highly
    # self-confident and its own verification called the work passed.
    record = records[0]
    assert record.predicted_quality == pytest.approx(0.95)
    assert record.objective_quality == pytest.approx(0.2)
    assert record.objective_passed is False
    assert record.confidence == pytest.approx(0.99)
    assert record.evaluator_revision == "review-v2"
    assert record.registry_digest == "registry-a"
    assert record.taxonomy_version == "taxonomy-a"
    assert record.adapter_version == "cursor-a"


def test_unavailable_review_is_not_recorded_as_an_objective_evaluator_outcome() -> None:
    # Arrange
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

        # Act
        records, _ = collect_records(store)
        quality = assess_run_quality([unavailable])

    # Assert
    assert records[0].objective_passed is None
    assert records[0].objective_quality is None
    assert quality["semantic_quality"] == "not_evaluated"
    assert quality["objective_evaluations"] == 0


def test_failed_objective_evaluation_keeps_objective_trust_basis() -> None:
    # Arrange
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

    # Act
    quality = assess_run_quality([gate])

    # Assert
    assert quality["semantic_quality"] == "failed"
    assert quality["objective_evaluations"] == 1
    assert quality["trust_basis"] == "objective_evaluator"


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


def test_objective_failures_drive_only_the_underperforming_role_recommendation() -> None:
    # Arrange: the same model objectively fails audit work while objectively
    # passing implementation work; self-confidence is 0.99 in both roles.
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

    # Act
    report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)

    # Assert
    model = next(item for item in report.models if item.model_id == "cursor/model")
    assert model.runs_with_objective_outcomes == 10
    assert model.objective_pass_rate == pytest.approx(0.5)
    assert model.mean_predicted_quality == pytest.approx(0.95)
    assert model.mean_objective_quality == pytest.approx(0.5)
    assert model.suggested_score is None
    assert report.suggestions == []
    assert [item["role"] for item in report.role_scorecard_suggestions] == ["audit"]
    assert report.role_scorecard_suggestions[0]["model_id"] == "cursor/model"


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


def test_fully_qualified_role_card_affects_effective_capability() -> None:
    # Arrange
    from puppetmaster.scorecards import effective_capability_score

    spec = _model_with_card(_qualified_card())

    # Act
    score = effective_capability_score(spec, "audit")

    # Assert
    assert score == 85


def test_swebench_card_producer_emits_qualified_routing_authority() -> None:
    # Arrange: the shipped benchmark importer is an existing role-card producer.
    # New qualification rules must stamp its cards rather than silently making
    # a successfully imported benchmark card inert.
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

    # Act
    imported, _ = import_community_baseline([spec], bundle)
    card = imported[0].role_scorecards["implement"]
    effective = effective_capability_score(imported[0], "implement")

    # Assert
    assert card["scale"] == "puppetmaster-capability-0-100"
    assert card["scale_version"]
    assert card["provenance"]["source"] == "community_benchmark"
    assert card["provenance"]["version"]
    assert effective == card["capability"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda card: card.update(provenance={}), id="missing-provenance"),
        pytest.param(lambda card: card.update(sample_count=4), id="too-few-samples"),
        pytest.param(lambda card: card.pop("scale"), id="missing-scale"),
        pytest.param(lambda card: card.pop("scale_version"), id="missing-scale-version"),
        pytest.param(
            lambda card: card.update(last_calibrated="2000-01-01"),
            id="stale-calibration",
        ),
    ],
)
def test_unqualified_role_card_falls_back_to_manual_capability(mutate) -> None:
    # Arrange
    from puppetmaster.scorecards import effective_capability_score

    card = _qualified_card()
    mutate(card)
    spec = _model_with_card(card)

    # Act
    score = effective_capability_score(spec, "audit")

    # Assert: an unqualified card is evidence, not routing authority.
    assert score == 50


@pytest.mark.parametrize(
    ("epoch_field", "first", "second"),
    [
        ("registry_digest", "registry-a", "registry-b"),
        ("taxonomy_version", "taxonomy-a", "taxonomy-b"),
        ("adapter_version", "cursor-a", "cursor-b"),
        ("evaluator_revision", "eval-a", "eval-b"),
    ],
)
def test_historical_aggregation_does_not_pool_incompatible_epochs(
    epoch_field: str, first: str, second: str
) -> None:
    # Arrange: each epoch has only three failures, below min_sample=5. Pooling
    # the six would manufacture a recommendation unsupported by either epoch.
    from puppetmaster.audit import build_audit_report

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

    # Act
    report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)

    # Assert
    assert report.epoch_count == 2
    assert report.role_scorecard_suggestions == []


def test_incomplete_epoch_cannot_authorize_a_role_card_recommendation() -> None:
    # Arrange: legacy records remain reportable, but missing reproducibility
    # lineage cannot become qualified routing authority.
    from puppetmaster.audit import build_audit_report

    records = [
        _objective_record(role="audit", passed=False)
        for _ in range(5)
    ]

    # Act
    report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)

    # Assert
    assert report.epoch_count == 1
    assert report.role_scorecard_suggestions == []


def test_undated_complete_lineage_is_reportable_but_non_authoritative() -> None:
    # Arrange: every lineage dimension is present, but there is no immutable
    # evaluator timestamp. The outcomes belong in diagnostics, not routing
    # authority or a recommendation which could be applied as fresh evidence.
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

    # Act
    report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)

    # Assert: diagnostic aggregation remains available while authority fails
    # closed at the role-card recommendation boundary.
    model = next(item for item in report.models if item.model_id == "cursor/model")
    assert model.runs_with_objective_outcomes == 5
    assert model.objective_pass_rate == pytest.approx(0.0)
    assert report.epoch_count == 1
    assert report.role_scorecard_suggestions == []


def test_stale_objective_history_cannot_be_re_stamped_as_a_fresh_role_card() -> None:
    # Arrange: the run is visible in the current store, but its evaluator
    # evidence is decades old. Applying the audit today must not launder that
    # evidence into a newly fresh last_calibrated date.
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

        # Act
        records, _ = collect_records(store)
        report = build_audit_report(
            records, {"cursor/model": 80}, min_sample=5
        )

    # Assert
    assert report.role_scorecard_suggestions == []


def test_latest_qualified_epoch_wins_role_card_recommendation() -> None:
    # Arrange: both evaluator epochs independently qualify. Deduplication must
    # select the newest calibrated evidence, not whichever epoch key sorts first.
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

    # Act
    report = build_audit_report(records, {"cursor/model": 80}, min_sample=5)

    # Assert
    assert len(report.role_scorecard_suggestions) == 1
    suggestion = report.role_scorecard_suggestions[0]
    assert suggestion["epoch"]["evaluator_revision"] == "eval-b"
    assert suggestion["last_calibrated"] == "2026-08-22"


def test_legacy_audit_record_and_report_shape_remain_supported() -> None:
    # Arrange: this is the pre-TASK-006 construction shape used by existing
    # callers and tests. New lineage and objective fields must have safe defaults.
    from puppetmaster.audit import build_audit_report

    records = [_legacy_record("cursor/legacy") for _ in range(2)]

    # Act
    report = build_audit_report(records, {"cursor/legacy": 60})

    # Assert
    assert report.tasks_considered == 2
    assert report.epoch_count == 1
    assert report.suggestions == []
    model = report.models[0]
    assert model.model_id == "cursor/legacy"
    assert model.selections == 2
    assert model.runs_with_objective_outcomes == 0
    assert model.objective_pass_rate is None
