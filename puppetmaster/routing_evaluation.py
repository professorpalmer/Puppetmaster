"""Provider-neutral paired evaluation for model-routing quality.

The harness in this module deliberately stops at an injected ``execute``
callable.  Codex, Claude Code, Hermes, Antigravity, or a deterministic fixture
can all supply that callable without this module importing a provider SDK or
reading credentials.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, log, sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Iterable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec
from puppetmaster.router import TaskSignals, route_task


REQUIRED_CORPUS_ROLES = frozenset({"implement", "explore", "audit", "plan"})
ARM_NAMES = ("routed_balanced", "strongest_eligible")
SUPPORTED_EVALUATORS = frozenset({"contains", "not_contains", "exact", "regex"})
_DEFAULT_CORPUS = (
    Path(__file__).resolve().parent / "baselines" / "routing-quality-corpus-v1.json"
)


@dataclass(frozen=True)
class EvaluationCriterion:
    criterion_id: str
    description: str
    evaluator_kind: str
    expected: str


@dataclass(frozen=True)
class SeededFailure:
    failure_id: str
    description: str
    output_text: str
    changed_files: tuple[str, ...]
    catastrophic: bool


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    role: str
    instruction: str
    snapshot_id: str
    snapshot_digest: str
    snapshot_files: Mapping[str, str]
    min_capability: int
    estimated_tokens_in: int
    estimated_tokens_out: int
    allowed_changed_files: tuple[str, ...]
    criteria: tuple[EvaluationCriterion, ...]
    seeded_failures: tuple[SeededFailure, ...]


@dataclass(frozen=True)
class EvaluationCorpus:
    schema_version: int
    corpus_id: str
    corpus_version: str
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True)
class CriterionResult:
    criterion_id: str
    passed: bool


@dataclass(frozen=True)
class CaseGrade:
    acceptance_passed: bool
    criterion_score: float
    criterion_results: tuple[CriterionResult, ...]
    unintended_files: tuple[str, ...]
    catastrophic: bool


@dataclass(frozen=True)
class EvaluationRequest:
    case: EvaluationCase
    arm: str
    repetition: int
    arm_order: tuple[str, str]
    snapshot_digest: str
    model_id: str
    adapter: str
    adapter_model_name: str
    routing_policy: str


@dataclass(frozen=True)
class ArmObservation:
    request: EvaluationRequest
    grade: CaseGrade
    correction_cycles: float
    elapsed_seconds: float
    retries: float
    tokens_in: float
    tokens_out: float
    nominal_cost_usd: float
    marginal_cost_usd: float


@dataclass(frozen=True)
class PairObservation:
    case_id: str
    repetition: int
    snapshot_digest: str
    arm_order: tuple[str, str]
    observations: tuple[ArmObservation, ArmObservation]


@dataclass(frozen=True)
class EvaluationReport:
    corpus_id: str
    corpus_version: str
    repetitions: int
    seed: int
    noninferiority_margin: float
    pairs: tuple[PairObservation, ...]
    arms: Mapping[str, Mapping[str, float]]
    paired_deltas: Mapping[str, Mapping[str, Any]]
    claim: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "noninferiority_margin": self.noninferiority_margin,
            "pairs": [
                {
                    "case_id": pair.case_id,
                    "repetition": pair.repetition,
                    "snapshot_digest": pair.snapshot_digest,
                    "arm_order": list(pair.arm_order),
                    "observations": {
                        item.request.arm: _observation_dict(item)
                        for item in pair.observations
                    },
                }
                for pair in self.pairs
            ],
            "arms": {name: dict(metrics) for name, metrics in self.arms.items()},
            "paired_deltas": {
                name: dict(interval) for name, interval in self.paired_deltas.items()
            },
            "claim": dict(self.claim),
        }


Executor = Callable[[EvaluationRequest], Mapping[str, Any]]


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _parse_criterion(raw: Any, *, case_id: str) -> EvaluationCriterion:
    if not isinstance(raw, Mapping):
        raise ValueError(f"criteria for {case_id} must be objects")
    evaluator = raw.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError(f"criterion evaluator for {case_id} must be an object")
    kind = _required_text(evaluator.get("kind"), "evaluator.kind")
    if kind not in SUPPORTED_EVALUATORS:
        raise ValueError(
            f"unsupported deterministic evaluator {kind!r}; "
            f"expected one of {sorted(SUPPORTED_EVALUATORS)}"
        )
    expected = _required_text(evaluator.get("expected"), "evaluator.expected")
    if kind == "regex":
        try:
            re.compile(expected)
        except re.error as exc:
            raise ValueError(f"invalid regex evaluator for {case_id}: {exc}") from exc
    return EvaluationCriterion(
        criterion_id=_required_text(raw.get("criterion_id"), "criterion_id"),
        description=_required_text(raw.get("description"), "criterion.description"),
        evaluator_kind=kind,
        expected=expected,
    )


def snapshot_digest_for_files(snapshot_files: Mapping[str, str]) -> str:
    """Return the canonical digest of a snapshot's packaged text files."""
    if not isinstance(snapshot_files, Mapping) or not snapshot_files:
        raise ValueError("snapshot_files must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for path, content in snapshot_files.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError("snapshot file paths must be non-empty strings")
        if not isinstance(content, str):
            raise ValueError(f"snapshot file {path!r} content must be text")
        normalized[path] = content
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + sha256(canonical).hexdigest()


def _parse_case(raw: Any) -> EvaluationCase:
    if not isinstance(raw, Mapping):
        raise ValueError("corpus cases must be objects")
    case_id = _required_text(raw.get("case_id"), "case_id")
    criteria_raw = raw.get("criteria")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise ValueError(f"case {case_id} requires at least one criterion")
    criteria = tuple(_parse_criterion(item, case_id=case_id) for item in criteria_raw)
    criterion_ids = [item.criterion_id for item in criteria]
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError(f"case {case_id} has duplicate criterion IDs")

    failures_raw = raw.get("seeded_failures", [])
    if not isinstance(failures_raw, list):
        raise ValueError(f"seeded_failures for {case_id} must be a list")
    failures: list[SeededFailure] = []
    for item in failures_raw:
        if not isinstance(item, Mapping):
            continue
        failure_files = item.get("changed_files", [])
        if not isinstance(failure_files, list) or any(
            not isinstance(path, str) for path in failure_files
        ):
            raise ValueError(f"seeded failure changed_files for {case_id} must be a string list")
        failure_catastrophic = item.get("catastrophic", False)
        if not isinstance(failure_catastrophic, bool):
            raise ValueError(f"seeded failure catastrophic for {case_id} must be boolean")
        failures.append(
            SeededFailure(
                failure_id=_required_text(item.get("failure_id"), "failure_id"),
                description=_required_text(item.get("description"), "failure.description"),
                output_text=str(item.get("output_text", "")),
                changed_files=tuple(failure_files),
                catastrophic=failure_catastrophic,
            )
        )
    failures_tuple = tuple(failures)
    if len(failures_tuple) != len(failures_raw):
        raise ValueError(f"seeded_failures for {case_id} must be objects")

    allowed = raw.get("allowed_changed_files", [])
    if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
        raise ValueError(f"allowed_changed_files for {case_id} must be a string list")
    digest = _required_text(raw.get("snapshot_digest"), "snapshot_digest")
    if not digest.startswith("sha256:"):
        raise ValueError(f"snapshot_digest for {case_id} must start with sha256:")
    snapshot_files = raw.get("snapshot_files")
    computed_digest = snapshot_digest_for_files(snapshot_files)
    if digest != computed_digest:
        raise ValueError(
            f"snapshot_digest for {case_id} does not match packaged snapshot_files"
        )
    min_capability = _nonnegative_int(raw.get("min_capability"), "min_capability")
    if min_capability > 100:
        raise ValueError(f"min_capability for {case_id} must be at most 100")
    return EvaluationCase(
        case_id=case_id,
        role=_required_text(raw.get("role"), "role"),
        instruction=_required_text(raw.get("instruction"), "instruction"),
        snapshot_id=_required_text(raw.get("snapshot_id"), "snapshot_id"),
        snapshot_digest=digest,
        snapshot_files=dict(snapshot_files),
        min_capability=min_capability,
        estimated_tokens_in=_nonnegative_int(
            raw.get("estimated_tokens_in"), "estimated_tokens_in"
        ),
        estimated_tokens_out=_nonnegative_int(
            raw.get("estimated_tokens_out"), "estimated_tokens_out"
        ),
        allowed_changed_files=tuple(allowed),
        criteria=criteria,
        seeded_failures=failures_tuple,
    )


def load_evaluation_corpus(path: Optional[Path] = None) -> EvaluationCorpus:
    """Load and validate a versioned deterministic routing-quality corpus."""
    source = Path(path) if path is not None else _DEFAULT_CORPUS
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation corpus must be a JSON object")
    schema_version = _nonnegative_int(raw.get("schema_version"), "schema_version")
    if schema_version < 1:
        raise ValueError("schema_version must be at least 1")
    corpus = EvaluationCorpus(
        schema_version=schema_version,
        corpus_id=_required_text(raw.get("corpus_id"), "corpus_id"),
        corpus_version=_required_text(raw.get("corpus_version"), "corpus_version"),
        cases=tuple(_parse_case(item) for item in raw.get("cases", [])),
    )
    if not corpus.cases:
        raise ValueError("evaluation corpus requires cases")
    case_ids = [item.case_id for item in corpus.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("evaluation corpus contains duplicate case IDs")
    missing = REQUIRED_CORPUS_ROLES - {item.role for item in corpus.cases}
    if missing:
        raise ValueError(
            "evaluation corpus is missing required roles: " + ", ".join(sorted(missing))
        )
    if not any(item.seeded_failures for item in corpus.cases):
        raise ValueError("evaluation corpus requires at least one seeded failure")
    for case in corpus.cases:
        for failure in case.seeded_failures:
            grade = grade_case(
                case,
                output_text=failure.output_text,
                changed_files=failure.changed_files,
                catastrophic=failure.catastrophic,
            )
            if grade.acceptance_passed:
                raise ValueError(
                    f"seeded failure {failure.failure_id!r} for {case.case_id} "
                    "must fail deterministic acceptance"
                )
    return corpus


def grade_case(
    case: EvaluationCase,
    *,
    output_text: str,
    changed_files: Iterable[str],
    catastrophic: bool,
) -> CaseGrade:
    """Deterministically grade observable output and file changes."""
    text = str(output_text)
    results: list[CriterionResult] = []
    for criterion in case.criteria:
        if criterion.evaluator_kind == "contains":
            passed = criterion.expected in text
        elif criterion.evaluator_kind == "not_contains":
            passed = criterion.expected not in text
        elif criterion.evaluator_kind == "exact":
            passed = criterion.expected == text
        else:  # regex; validated at corpus load.
            passed = re.search(criterion.expected, text) is not None
        results.append(CriterionResult(criterion.criterion_id, passed))

    allowed = set(case.allowed_changed_files)
    unintended = tuple(sorted({str(path) for path in changed_files if str(path) not in allowed}))
    criterion_score = fmean(float(item.passed) for item in results)
    accepted = criterion_score == 1.0 and not unintended and not bool(catastrophic)
    return CaseGrade(
        acceptance_passed=accepted,
        criterion_score=criterion_score,
        criterion_results=tuple(results),
        unintended_files=unintended,
        catastrophic=bool(catastrophic),
    )


def _number(receipt: Mapping[str, Any], field_name: str) -> float:
    value = receipt.get(field_name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"executor receipt {field_name} must be non-negative numeric")
    return float(value)


def _observe(request: EvaluationRequest, execute: Executor) -> ArmObservation:
    receipt = execute(request)
    if not isinstance(receipt, Mapping):
        raise ValueError("executor must return a mapping receipt")
    observed_snapshot = receipt.get("snapshot_digest")
    if observed_snapshot != request.snapshot_digest:
        raise ValueError(
            "executor receipt snapshot_digest must match the requested immutable snapshot"
        )
    changed = receipt.get("changed_files", [])
    if not isinstance(changed, (list, tuple)) or any(
        not isinstance(path, str) for path in changed
    ):
        raise ValueError("executor receipt changed_files must be a string list")
    catastrophic = receipt.get("catastrophic", False)
    if not isinstance(catastrophic, bool):
        raise ValueError("executor receipt catastrophic must be boolean")
    grade = grade_case(
        request.case,
        output_text=str(receipt.get("output_text", "")),
        changed_files=changed,
        catastrophic=catastrophic,
    )
    return ArmObservation(
        request=request,
        grade=grade,
        correction_cycles=_number(receipt, "correction_cycles"),
        elapsed_seconds=_number(receipt, "elapsed_seconds"),
        retries=_number(receipt, "retries"),
        tokens_in=_number(receipt, "tokens_in"),
        tokens_out=_number(receipt, "tokens_out"),
        nominal_cost_usd=_number(receipt, "nominal_cost_usd"),
        marginal_cost_usd=_number(receipt, "marginal_cost_usd"),
    )


def _observation_metrics(item: ArmObservation) -> dict[str, float]:
    return {
        "acceptance_pass_rate": float(item.grade.acceptance_passed),
        "criterion_score_mean": item.grade.criterion_score,
        "unintended_file_rate": float(bool(item.grade.unintended_files)),
        "catastrophic_failure_rate": float(item.grade.catastrophic),
        "correction_cycles_mean": item.correction_cycles,
        "elapsed_seconds_mean": item.elapsed_seconds,
        "retries_mean": item.retries,
        "tokens_in_mean": item.tokens_in,
        "tokens_out_mean": item.tokens_out,
        "nominal_cost_usd_mean": item.nominal_cost_usd,
        "marginal_cost_usd_mean": item.marginal_cost_usd,
    }


def _observation_dict(item: ArmObservation) -> dict[str, Any]:
    return {
        "model_id": item.request.model_id,
        "adapter": item.request.adapter,
        "routing_policy": item.request.routing_policy,
        "acceptance_passed": item.grade.acceptance_passed,
        "criterion_score": item.grade.criterion_score,
        "unintended_files": list(item.grade.unintended_files),
        "catastrophic": item.grade.catastrophic,
        **_observation_metrics(item),
    }


def _paired_interval(
    values: list[float], *, bounded: Optional[tuple[float, float]] = None
) -> dict[str, Any]:
    estimate = fmean(values)
    if bounded is not None:
        lower_bound, upper_bound = bounded
        width = upper_bound - lower_bound
        half_width = width * sqrt(log(40.0) / (2.0 * len(values)))
        lower = max(lower_bound, estimate - half_width)
        upper = min(upper_bound, estimate + half_width)
        method = "paired_hoeffding_95pct"
    elif len(values) < 2:
        half_width = 0.0
        method = "paired_point_only"
        lower = estimate
        upper = estimate
    else:
        half_width = 1.96 * stdev(values) / sqrt(len(values))
        method = "paired_normal_95pct"
        lower = estimate - half_width
        upper = estimate + half_width
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "method": method,
        "pairs": len(values),
    }


def run_paired_evaluation(
    corpus: EvaluationCorpus,
    registry: Iterable[ModelSpec],
    *,
    execute: Executor,
    repetitions: int = 3,
    seed: int = 0,
    noninferiority_margin: float = 0.05,
) -> EvaluationReport:
    """Run balanced and strongest-eligible arms from identical snapshots."""
    if repetitions < 3:
        raise ValueError("paired evaluation requires at least 3 repetitions")
    if (
        isinstance(noninferiority_margin, bool)
        or not isinstance(noninferiority_margin, (int, float))
        or not isfinite(float(noninferiority_margin))
        or not 0.0 <= float(noninferiority_margin) <= 1.0
    ):
        raise ValueError(
            "noninferiority_margin must be a finite non-boolean number in [0, 1]"
        )
    noninferiority_margin = float(noninferiority_margin)
    models = tuple(registry)
    rng = random.Random(seed)
    pairs: list[PairObservation] = []
    by_arm: dict[str, list[ArmObservation]] = {name: [] for name in ARM_NAMES}

    for case in corpus.cases:
        signals = TaskSignals(
            instruction=case.instruction,
            role=case.role,
            explicit_min_capability=case.min_capability,
            estimated_tokens_in=case.estimated_tokens_in,
            estimated_tokens_out=case.estimated_tokens_out,
        )
        routed = route_task(signals, models, policy="balanced")
        strongest = route_task(signals, models, policy="quality")
        decisions = {
            "routed_balanced": routed,
            "strongest_eligible": strongest,
        }
        for repetition in range(repetitions):
            order = list(ARM_NAMES)
            rng.shuffle(order)
            arm_order = (order[0], order[1])
            observations: list[ArmObservation] = []
            for arm in arm_order:
                decision = decisions[arm]
                request = EvaluationRequest(
                    case=case,
                    arm=arm,
                    repetition=repetition,
                    arm_order=arm_order,
                    snapshot_digest=case.snapshot_digest,
                    model_id=decision.model.id,
                    adapter=decision.model.adapter,
                    adapter_model_name=decision.model.adapter_model_name,
                    routing_policy=decision.policy,
                )
                observation = _observe(request, execute)
                observations.append(observation)
                by_arm[arm].append(observation)
            pairs.append(
                PairObservation(
                    case_id=case.case_id,
                    repetition=repetition,
                    snapshot_digest=case.snapshot_digest,
                    arm_order=arm_order,
                    observations=(observations[0], observations[1]),
                )
            )

    arm_metrics: dict[str, dict[str, float]] = {}
    for arm, observations in by_arm.items():
        rows = [_observation_metrics(item) for item in observations]
        arm_metrics[arm] = {
            metric: fmean(row[metric] for row in rows) for metric in rows[0]
        }

    paired_values: dict[str, list[float]] = {
        metric: [] for metric in next(iter(arm_metrics.values()))
    }
    for pair in pairs:
        by_name = {item.request.arm: item for item in pair.observations}
        routed_metrics = _observation_metrics(by_name["routed_balanced"])
        baseline_metrics = _observation_metrics(by_name["strongest_eligible"])
        for metric in paired_values:
            paired_values[metric].append(routed_metrics[metric] - baseline_metrics[metric])
    intervals = {
        metric: _paired_interval(
            values,
            bounded=(-1.0, 1.0) if metric == "acceptance_pass_rate" else None,
        )
        for metric, values in paired_values.items()
    }

    quality = intervals["acceptance_pass_rate"]
    if quality["lower"] >= -noninferiority_margin:
        status = "noninferior"
    elif quality["upper"] < -noninferiority_margin:
        status = "inferior"
    else:
        status = "inconclusive"
    claim = {
        "status": status,
        "basis": "paired_noninferiority",
        "metric": "acceptance_pass_rate",
        "margin": noninferiority_margin,
        "lower_bound": quality["lower"],
        "upper_bound": quality["upper"],
    }
    return EvaluationReport(
        corpus_id=corpus.corpus_id,
        corpus_version=corpus.corpus_version,
        repetitions=repetitions,
        seed=seed,
        noninferiority_margin=noninferiority_margin,
        pairs=tuple(pairs),
        arms=arm_metrics,
        paired_deltas=intervals,
        claim=claim,
    )


def render_evaluation_report(report: EvaluationReport) -> str:
    """Render a deterministic, claim-bounded Markdown report."""
    payload = report.to_dict()
    lines = [
        "# Routing quality paired evaluation",
        "",
        f"Corpus: `{report.corpus_id}` version `{report.corpus_version}`",
        f"Repetitions per case: {report.repetitions}; seed: {report.seed}",
        "",
        "## Paired non-inferiority result",
        "",
        f"Status: **{payload['claim']['status']}** at margin "
        f"{report.noninferiority_margin:.3f}.",
        "",
        "This bounded run grades only the deterministic corpus criteria. "
        "Structural artifact presence is health evidence; it does not establish "
        "semantic quality outside those criteria.",
        "",
        "## Arm metrics",
        "",
    ]
    for arm in ARM_NAMES:
        lines.append(f"### {arm}")
        lines.append("")
        for metric, value in report.arms[arm].items():
            lines.append(f"- `{metric}`: {value:.6f}")
        lines.append("")
    lines.extend(["## Paired deltas (routed minus strongest eligible)", ""])
    for metric, interval in report.paired_deltas.items():
        lines.append(
            f"- `{metric}`: {interval['estimate']:.6f} "
            f"[{interval['lower']:.6f}, {interval['upper']:.6f}] "
            f"({interval['method']})"
        )
    return "\n".join(lines).rstrip() + "\n"
