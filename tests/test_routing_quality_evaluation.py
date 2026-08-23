"""Independent TASK-007 RED contract for paired routing-quality evidence.

The reviewer authored these tests before inspecting an author solution.  They
exercise a public, provider-neutral evaluation seam and production routing
behavior.  No test contacts a model provider or treats artifact structure as
semantic correctness.
"""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import pytest

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


def test_default_corpus_is_versioned_multi_role_and_has_seeded_failures() -> None:
    # Arrange / Act.
    from puppetmaster.routing_evaluation import (
        grade_case,
        load_evaluation_corpus,
        snapshot_digest_for_files,
    )

    corpus = load_evaluation_corpus()

    # Assert: this is a real versioned quality corpus, not one generic prompt.
    assert corpus.schema_version >= 1
    assert corpus.corpus_id
    assert corpus.corpus_version
    assert {case.role for case in corpus.cases} >= REQUIRED_ROLES
    assert len({case.case_id for case in corpus.cases}) == len(corpus.cases)
    assert any(case.seeded_failures for case in corpus.cases)
    assert all(case.criteria for case in corpus.cases)
    assert all(case.snapshot_digest.startswith("sha256:") for case in corpus.cases)
    assert all(case.snapshot_files for case in corpus.cases)
    assert all(
        snapshot_digest_for_files(case.snapshot_files) == case.snapshot_digest
        for case in corpus.cases
    )
    assert all(
        criterion.evaluator_kind != "llm_judge"
        for case in corpus.cases
        for criterion in case.criteria
    )
    assert all(
        criterion.expected.casefold() not in case.instruction.casefold()
        for case in corpus.cases
        for criterion in case.criteria
    )

    # Seeded failures are executable negative vectors, not descriptions which
    # have never been graded against their claimed case.
    for case in corpus.cases:
        for failure in case.seeded_failures:
            grade = grade_case(
                case,
                output_text=failure.output_text,
                changed_files=failure.changed_files,
                catastrophic=failure.catastrophic,
            )
            assert grade.acceptance_passed is False


def test_corpus_loader_fails_closed_without_version_or_required_roles(tmp_path: Path) -> None:
    # Arrange.
    from puppetmaster.routing_evaluation import load_evaluation_corpus

    unversioned = _corpus_payload()
    unversioned.pop("corpus_version")
    unversioned_path = _write_corpus(tmp_path, unversioned)

    # Act / Assert.
    with pytest.raises(ValueError, match="corpus_version"):
        load_evaluation_corpus(unversioned_path)

    incomplete = _corpus_payload()
    incomplete["corpus_version"] = "2026-08-22.2"
    incomplete["cases"] = [
        case for case in incomplete["cases"] if case["role"] != "plan"
    ]
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="plan"):
        load_evaluation_corpus(incomplete_path)


def test_corpus_loader_rejects_a_seeded_failure_that_passes_acceptance(
    tmp_path: Path,
) -> None:
    # Arrange: a corpus must not gain seeded-failure credibility merely by
    # labeling a successful vector as a failure.
    from puppetmaster.routing_evaluation import load_evaluation_corpus

    payload = _corpus_payload()
    audit_case = next(case for case in payload["cases"] if case["role"] == "audit")
    audit_case["seeded_failures"][0]["output_text"] = audit_case["criteria"][0][
        "evaluator"
    ]["expected"]
    path = _write_corpus(tmp_path, payload)

    # Act / Assert.
    with pytest.raises(ValueError, match="seeded failure.*must fail"):
        load_evaluation_corpus(path)


def test_deterministic_grader_scores_criteria_and_unintended_files(tmp_path: Path) -> None:
    # Arrange.
    from puppetmaster.routing_evaluation import grade_case, load_evaluation_corpus

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
    case = next(item for item in corpus.cases if item.role == "implement")
    allowed = list(case.allowed_changed_files)

    # Act.
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

    # Assert: grading is repeatable and semantic acceptance is criterion-bound.
    assert first == repeat
    assert first.acceptance_passed is True
    assert first.criterion_score == pytest.approx(1.0)
    assert first.unintended_files == ()
    assert failed.acceptance_passed is False
    assert failed.criterion_score == pytest.approx(0.0)
    assert failed.unintended_files == ("unintended.txt",)


def test_paired_runner_uses_identical_snapshots_and_strongest_eligible_baseline(
    tmp_path: Path,
) -> None:
    # Arrange: the absolute strongest model cannot fit; the strongest eligible
    # baseline must therefore pin frontier, while balanced should route bargain.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
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

    # Act.
    report = run_paired_evaluation(
        corpus,
        registry,
        execute=execute,
        repetitions=3,
        seed=41,
    )

    # Assert: every case/repetition is a pair over the same immutable baseline.
    grouped = defaultdict(list)
    for request in calls:
        grouped[(request.case.case_id, request.repetition)].append(request)
    assert len(grouped) == len(corpus.cases) * 3
    for pair in grouped.values():
        assert {request.arm for request in pair} == ARM_NAMES
        assert len({request.snapshot_digest for request in pair}) == 1
        model_by_arm = {request.arm: request.model_id for request in pair}
        assert model_by_arm == {
            "routed_balanced": "cursor/bargain",
            "strongest_eligible": "cursor/frontier",
        }

    # Randomized execution order is deterministic under a seed and exercises
    # both orderings instead of always favoring the same first arm.
    arm_orders = {tuple(pair.arm_order) for pair in report.pairs}
    assert arm_orders == {
        ("routed_balanced", "strongest_eligible"),
        ("strongest_eligible", "routed_balanced"),
    }


def test_paired_runner_requires_three_repetitions(tmp_path: Path) -> None:
    # Arrange.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
    registry = [_model("cursor/model", capability=90, cost=1.0)]

    # Act / Assert.
    with pytest.raises(ValueError, match="at least 3"):
        run_paired_evaluation(
            corpus,
            registry,
            execute=_passing_executor,
            repetitions=2,
            seed=1,
        )


def test_paired_runner_rejects_unverified_or_mismatched_snapshot_receipts(
    tmp_path: Path,
) -> None:
    # Arrange: sharing an asserted digest in two requests is not proof that an
    # executor actually reset both arms to that snapshot.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
    registry = [_model("cursor/model", capability=90, cost=1.0)]

    def mismatched_executor(request):
        receipt = dict(_passing_executor(request))
        receipt["snapshot_digest"] = "sha256:" + "0" * 64
        return receipt

    # Act / Assert: the report must not certify same-snapshot pairing from an
    # unchecked request field alone.
    with pytest.raises(ValueError, match="snapshot_digest"):
        run_paired_evaluation(
            corpus,
            registry,
            execute=mismatched_executor,
            repetitions=3,
            seed=1,
        )


def test_report_includes_quality_cost_failure_metrics_and_paired_uncertainty(
    tmp_path: Path,
) -> None:
    # Arrange: opposite wins in different repetitions create a genuine wide,
    # paired quality interval that must remain inconclusive.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        render_evaluation_report,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
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

    # Act.
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

    # Assert: a fixed corpus, executor, and seed produce a stable sample report.
    assert payload == repeat.to_dict()
    assert payload["claim"]["status"] == "inconclusive"
    assert payload["claim"]["basis"] == "paired_noninferiority"
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
    assert set(payload["arms"]["routed_balanced"]) >= required_metrics
    assert set(payload["arms"]["strongest_eligible"]) >= required_metrics
    assert set(payload["paired_deltas"]) >= required_metrics
    for metric in required_metrics:
        uncertainty = payload["paired_deltas"][metric]
        assert set(uncertainty) >= {"estimate", "lower", "upper", "method"}
        assert uncertainty["lower"] <= uncertainty["estimate"] <= uncertainty["upper"]

    markdown = render_evaluation_report(first).lower()
    assert "inconclusive" in markdown
    assert "paired non-inferiority" in markdown
    assert "does not establish semantic quality" in markdown
    assert "proves that routing improves quality" not in markdown


def test_finite_all_equal_sample_does_not_collapse_quality_uncertainty_to_zero(
    tmp_path: Path,
) -> None:
    # Arrange: twelve observed ties (four cases x three repetitions) are
    # encouraging, but a finite binary sample does not establish a population
    # pass-rate difference of exactly zero at a five-point margin.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
    registry = [
        _model("cursor/bargain", capability=70, cost=0.10),
        _model("cursor/frontier", capability=95, cost=1.00),
    ]

    # Act.
    report = run_paired_evaluation(
        corpus,
        registry,
        execute=_passing_executor,
        repetitions=3,
        seed=7,
        noninferiority_margin=0.05,
    )
    quality = report.to_dict()["paired_deltas"]["acceptance_pass_rate"]

    # Assert: a variance-zero normal approximation would incorrectly emit
    # [0, 0] and claim non-inferiority from only twelve pairs.
    assert quality["lower"] < quality["estimate"] < quality["upper"]
    assert report.claim["status"] == "inconclusive"


@pytest.mark.parametrize(
    "invalid_margin",
    [
        pytest.param(True, id="boolean"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(1.01, id="above-pass-rate-range"),
    ],
)
def test_noninferiority_margin_fails_closed_outside_finite_unit_interval(
    tmp_path: Path,
    invalid_margin: object,
) -> None:
    # Arrange.
    from puppetmaster.routing_evaluation import (
        load_evaluation_corpus,
        run_paired_evaluation,
    )

    corpus = load_evaluation_corpus(_write_corpus(tmp_path))
    registry = [_model("cursor/model", capability=90, cost=1.0)]

    # Act / Assert: an invalid margin cannot manufacture a favorable claim.
    with pytest.raises(ValueError, match="noninferiority_margin"):
        run_paired_evaluation(
            corpus,
            registry,
            execute=_passing_executor,
            repetitions=3,
            seed=1,
            noninferiority_margin=invalid_margin,
        )


def test_shadow_routing_is_opt_in_and_cannot_change_production_selection() -> None:
    # Arrange.
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

    # Act.
    production = router.route_task(task, registry, policy="balanced")
    shadowed = router.route_task(
        task,
        registry,
        policy="balanced",
        shadow_policy="quality",
    )
    production_payload = production.to_artifact_payload()
    shadow_payload = shadowed.to_artifact_payload()

    # Assert: opt-in adds counterfactual evidence only; dispatch identity is
    # byte-for-byte the same production choice as the non-shadow route.
    assert production.model.id == "cursor/bargain"
    assert shadowed.model.id == production.model.id
    assert "shadow_routing" not in production_payload
    assert shadow_payload["model_id"] == production_payload["model_id"]
    assert shadow_payload["adapter"] == production_payload["adapter"]
    assert shadow_payload["policy"] == production_payload["policy"]
    assert shadow_payload["shadow_routing"] == {
        "enabled": True,
        "policy": "quality",
        "production_model_id": "cursor/bargain",
        "counterfactual_model_id": "cursor/frontier",
        "production_selection_changed": False,
    }


def test_orchestrator_shadow_opt_in_persists_evidence_without_changing_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Arrange: exercise the actual auto-route-to-task/artifact seam, not only a
    # direct library call which no production caller can enable.
    from puppetmaster.model_registry import save_registry
    from puppetmaster.models import ArtifactType
    from puppetmaster.orchestrator import Orchestrator
    from puppetmaster.store_factory import create_store
    from puppetmaster.workers import WorkerSpec

    registry_path = tmp_path / "models.json"
    save_registry(
        [
            _model("cursor/bargain", capability=70, cost=0.10),
            _model("cursor/frontier", capability=95, cost=1.00),
        ],
        registry_path,
    )
    monkeypatch.setattr(
        "puppetmaster.preflight.adapter_cli_present", lambda _adapter: True
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

    # Act.
    task = orchestrator._create_tasks(job, [spec])[0]
    routing = [
        artifact
        for artifact in store.list_artifacts(job.id)
        if artifact.type == ArtifactType.ROUTING
    ]

    # Assert: dispatch remains the balanced pick while the durable artifact
    # carries the quality-policy counterfactual.
    assert task.payload["router_model_id"] == "cursor/bargain"
    assert task.payload["model"] == "bargain"
    assert len(routing) == 1
    assert routing[0].payload["model_id"] == "cursor/bargain"
    assert routing[0].payload["shadow_routing"] == {
        "enabled": True,
        "policy": "quality",
        "production_model_id": "cursor/bargain",
        "counterfactual_model_id": "cursor/frontier",
        "production_selection_changed": False,
    }


def test_documentation_states_quality_claim_boundaries() -> None:
    # Arrange / Act.
    text = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "routing-quality-evaluation.md"
    ).read_text(encoding="utf-8").lower()

    # Assert: structural presence is useful health evidence, but not semantic
    # correctness, and a finite paired run may support only a bounded claim.
    assert "non-inferiority" in text
    assert "inconclusive" in text
    assert "structural artifact presence" in text
    assert "not semantic quality" in text
    assert "proves that routing improves quality" not in text
