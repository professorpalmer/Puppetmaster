"""Objective, role-specific routing audit over durable run artifacts.

Design stance (deliberate): this **recommends**, it does not silently
autopilot. The signals available (model self-reported confidence, escalation
rate) are noisy and gameable, so closing the loop without a human in it risks
feedback ratchets that only ever raise cost — the opposite of the point. So:

* The aggregator (:func:`build_audit_report`) is a pure function over records
  the caller collects from the store. Same input, same output.
* Self-confidence and ordinary escalation remain visible diagnostics, but they
  never authorize a score change.
* Objective evaluator outcomes may recommend a role card change only inside a
  compatible registry/classifier/adapter/evaluator epoch. They never lower the
  model's global manual score.
* "Over-used" (a strong model doing trivial work) is **flagged but never
  auto-adjusted** — proving a cheaper model would have sufficed needs a
  counterfactual (a shadow run), which this audit does not perform.
* Nothing is written unless the CLI is invoked with ``--apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional

from puppetmaster.scorecards import MAX_CALIBRATION_AGE_DAYS

# Confidence at or above this is "fine"; below it counts toward low-confidence.
LOW_CONFIDENCE_BAR = 0.6
# Don't propose a score change for a model we've seen fewer than this many times
# — small samples are noise, not signal.
MIN_SAMPLE = 5
# An under-delivering model gets its score lowered, but never below this floor
# (it should still be reachable for trivial work).
MIN_SCORE_FLOOR = 10
# Escalated-away / low-confidence rates that trip the "under-provisioned" flag.
UNDER_PROVISIONED_RATE = 0.4
SEVERE_RATE = 0.6
# A model whose typical task needed this much less capability than its score is
# "possibly over-used" (informational only).
OVER_USE_GAP = 20


@dataclass(frozen=True)
class ObjectiveAuditOutcome:
    """One completed objective evaluation attributed to its producing model."""

    model_id: str
    predicted_quality: Optional[float]
    objective_quality: Optional[float]
    passed: bool
    source: str = ""
    registry_digest: str = ""
    classifier_version: str = ""
    taxonomy_version: str = ""
    adapter_version: str = ""
    evaluator_revision: str = ""
    evaluated_at: str = ""


@dataclass(frozen=True)
class TaskAuditRecord:
    """One finished task's routing outcome, normalized for aggregation."""

    model_id: str  # the FINAL model that produced the accepted result
    adapter: str
    capability_needed: int
    est_cost_usd: float
    confidence: Optional[float]  # latest VERIFICATION confidence, if any
    escalated: bool  # this task was escalated up from a weaker model
    escalated_from: Optional[str]  # the weaker model it escalated off of
    fell_back: bool  # this task fell back after an adapter failure
    # Estimate-vs-actual reconciliation. The router's pre-flight token/cost
    # estimate drives every routing decision; measuring it against what the run
    # actually consumed is the only way to know the estimate (and therefore the
    # savings ledger) is calibrated rather than asserted.
    est_tokens_in: int = 0
    est_tokens_out: int = 0
    actual_tokens_in: int = 0
    actual_tokens_out: int = 0
    # True when actuals came from a real SDK usage object; False when they are a
    # char/4 approximation. None when the run reported no token usage at all
    # (so drift is simply unknown, never faked as zero).
    actual_tokens_measured: Optional[bool] = None
    role: str = ""
    elapsed_seconds: Optional[float] = None
    verification_result: Optional[str] = None
    gate_passed: Optional[bool] = None
    attempts: int = 0
    fallback_attempts: int = 0
    escalation_attempts: int = 0
    # Reroute attribution is kept per mechanism.  A fallback or independent
    # review rejection belongs to the model that produced the rejected run,
    # not to the stronger model that eventually completed the task.
    fallback_from: Optional[str] = None
    review_escalated_from: Optional[str] = None
    fallback_from_models: tuple[str, ...] = ()
    review_escalated_from_models: tuple[str, ...] = ()
    # Prediction-versus-outcome evidence.  ``confidence`` above remains the
    # worker's weak self-report; these fields are populated only from routing
    # provenance and objective gate/evaluator artifacts.
    predicted_quality: Optional[float] = None
    objective_quality: Optional[float] = None
    objective_passed: Optional[bool] = None
    objective_source: Optional[str] = None
    objective_model_id: Optional[str] = None
    # Historical audit samples are comparable only inside one complete epoch.
    registry_digest: str = ""
    classifier_version: str = ""
    taxonomy_version: str = ""
    adapter_version: str = ""
    evaluator_revision: str = ""
    evaluated_at: str = ""
    objective_outcomes: tuple[ObjectiveAuditOutcome, ...] = ()

    @property
    def has_actuals(self) -> bool:
        return self.actual_tokens_measured is not None

    @property
    def est_tokens_total(self) -> int:
        return self.est_tokens_in + self.est_tokens_out

    @property
    def actual_tokens_total(self) -> int:
        return self.actual_tokens_in + self.actual_tokens_out


@dataclass
class ModelAudit:
    model_id: str
    adapter: str
    score: Optional[int]
    selections: int  # times this model was the INITIAL pick
    runs_with_confidence: int
    mean_confidence: Optional[float]
    min_confidence: Optional[float]
    low_confidence_rate: float
    escalated_away: int
    escalated_away_rate: float
    fell_back_away: int
    est_spend_usd: float
    # Estimate-vs-actual reconciliation, aggregated over this model's retained
    # tasks that reported token usage. ``*_drift_ratio`` is actual/estimated:
    # 1.0 = spot on, >1 = the router under-estimated, <1 = it over-estimated.
    runs_with_actuals: int = 0
    measured_runs: int = 0  # of runs_with_actuals, how many were measured (not char/4)
    est_tokens: int = 0
    actual_tokens: int = 0
    token_drift_ratio: Optional[float] = None
    actual_spend_usd: float = 0.0
    cost_drift_ratio: Optional[float] = None
    mean_elapsed_seconds: Optional[float] = None
    timeout_or_failed_rate: float = 0.0
    degraded_rate: float = 0.0
    roles: dict = field(default_factory=dict)
    review_escalated_away: int = 0
    runs_with_objective_outcomes: int = 0
    objective_pass_rate: Optional[float] = None
    mean_predicted_quality: Optional[float] = None
    mean_objective_quality: Optional[float] = None
    flags: list[str] = field(default_factory=list)
    suggested_score: Optional[int] = None
    rationale: Optional[str] = None


@dataclass
class AuditReport:
    jobs_considered: int
    tasks_considered: int
    window_days: Optional[float]
    total_est_spend_usd: float
    models: list[ModelAudit]
    # Job-wide reconciliation rollup across every task that reported usage.
    tasks_with_actuals: int = 0
    total_est_tokens: int = 0
    total_actual_tokens: int = 0
    total_actual_spend_usd: float = 0.0
    # Estimated spend over *only the reconciled tasks* — the apples-to-apples
    # denominator for cost drift (``total_est_spend_usd`` covers every task,
    # including ones with no actuals, so it must not anchor the ratio).
    total_est_spend_reconciled_usd: float = 0.0
    role_scorecard_suggestions: list[dict] = field(default_factory=list)
    epoch_count: int = 0

    # Headline drift is measured-only (SDK usage), never char/4. The
    # all-reconciled totals above stay available for accounting; these
    # fields are the calibration denominator.
    tasks_with_measured: int = 0
    total_est_tokens_measured: int = 0
    total_actual_tokens_measured: int = 0
    total_actual_spend_measured_usd: float = 0.0
    total_est_spend_measured_usd: float = 0.0

    @property
    def token_drift_ratio(self) -> Optional[float]:
        denom = self.total_est_tokens_measured
        return (self.total_actual_tokens_measured / denom) if denom else None

    @property
    def cost_drift_ratio(self) -> Optional[float]:
        denom = self.total_est_spend_measured_usd
        return (self.total_actual_spend_measured_usd / denom) if denom else None

    @property
    def suggestions(self) -> list[dict]:
        out = []
        for m in self.models:
            if m.suggested_score is not None and m.suggested_score != m.score:
                out.append(
                    {
                        "model_id": m.model_id,
                        "from_score": m.score,
                        "to_score": m.suggested_score,
                        "rationale": m.rationale,
                    }
                )
        return out


_FAILED_RESULTS = frozenset({"failed", "timeout", "error", "fail"})
_DEGRADED_RESULTS = frozenset({"degraded"})


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _verification_label(result: Optional[str]) -> str:
    if not isinstance(result, str):
        return ""
    return result.strip().lower()


def _optional_unit_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def _record_objective_outcomes(
    record: TaskAuditRecord,
) -> tuple[ObjectiveAuditOutcome, ...]:
    if record.objective_outcomes:
        return record.objective_outcomes
    if record.objective_passed is None:
        return ()
    return (
        ObjectiveAuditOutcome(
            model_id=record.objective_model_id or record.model_id,
            predicted_quality=record.predicted_quality,
            objective_quality=record.objective_quality,
            passed=record.objective_passed,
            source=record.objective_source or "",
            registry_digest=record.registry_digest,
            classifier_version=record.classifier_version,
            taxonomy_version=record.taxonomy_version,
            adapter_version=record.adapter_version,
            evaluator_revision=record.evaluator_revision,
            evaluated_at=record.evaluated_at,
        ),
    )


def _complete_epoch(outcome: ObjectiveAuditOutcome) -> bool:
    # Classifier logic is governed by the external taxonomy; either explicit
    # classifier version or taxonomy version is accepted, while every other
    # reproducibility dimension is mandatory.
    complete = all(
        (
            outcome.registry_digest,
            outcome.classifier_version or outcome.taxonomy_version,
            outcome.adapter_version,
            outcome.evaluator_revision,
        )
    )
    if not complete:
        return False
    if not outcome.evaluated_at:
        return False
    evaluated_on = _evaluated_on(outcome.evaluated_at)
    if evaluated_on is None:
        return False
    age_days = (date.today() - evaluated_on).days
    return 0 <= age_days <= MAX_CALIBRATION_AGE_DAYS


def _evaluated_on(value: str) -> Optional[date]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        try:
            return date.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _epoch_keys(records: list[TaskAuditRecord]) -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for record in records:
        outcomes = _record_objective_outcomes(record)
        if outcomes:
            keys.update(
                (
                    outcome.registry_digest,
                    outcome.classifier_version,
                    outcome.taxonomy_version,
                    outcome.adapter_version,
                    outcome.evaluator_revision,
                )
                for outcome in outcomes
            )
        else:
            keys.add(
                (
                    record.registry_digest,
                    record.classifier_version,
                    record.taxonomy_version,
                    record.adapter_version,
                    record.evaluator_revision,
                )
            )
    return keys


def _record_failed(record: TaskAuditRecord) -> bool:
    if _verification_label(record.verification_result) in _FAILED_RESULTS:
        return True
    return record.gate_passed is False


def _fallback_sources(record: TaskAuditRecord) -> tuple[str, ...]:
    if record.fallback_from_models:
        return record.fallback_from_models
    return (record.fallback_from,) if record.fallback_from else ()


def _review_escalation_sources(record: TaskAuditRecord) -> tuple[str, ...]:
    if record.review_escalated_from_models:
        return record.review_escalated_from_models
    return (record.review_escalated_from,) if record.review_escalated_from else ()


def _record_degraded(record: TaskAuditRecord) -> bool:
    return _verification_label(record.verification_result) in _DEGRADED_RESULTS


def _role_counts(
    model_id: str, retained: list[TaskAuditRecord], records: list[TaskAuditRecord]
) -> dict:
    counts: dict[str, int] = {}
    for record in retained:
        role = record.role or ""
        if role:
            counts[role] = counts.get(role, 0) + 1
    for record in records:
        if (
            (record.escalated and record.escalated_from == model_id)
            or model_id in _fallback_sources(record)
            or model_id in _review_escalation_sources(record)
        ):
            role = record.role or ""
            if role:
                counts[role] = counts.get(role, 0) + 1
    return counts


def _role_scorecard_suggestions(
    records: list[TaskAuditRecord],
    registry_scores: dict[str, int],
    min_sample: int,
) -> list[dict]:
    """Objective, epoch-local role-card hints. Never mutate a global score.

    Each epoch is evaluated independently.  Suggestions with the same
    model/adapter/role are deduplicated only after one complete epoch has met
    the sample floor, so incompatible historical samples can never manufacture
    authority by pooling.
    """
    groups: dict[tuple, list[ObjectiveAuditOutcome]] = {}
    for record in records:
        role = record.role or ""
        if not role:
            continue
        for outcome in _record_objective_outcomes(record):
            if not _complete_epoch(outcome):
                continue
            key = (
                outcome.model_id,
                record.adapter,
                role,
                outcome.registry_digest,
                outcome.classifier_version,
                outcome.taxonomy_version,
                outcome.adapter_version,
                outcome.evaluator_revision,
            )
            groups.setdefault(key, []).append(outcome)

    candidates: dict[tuple[str, str, str], tuple[tuple, dict]] = {}
    for group_key, recs in sorted(groups.items()):
        model_id, adapter, role = group_key[:3]
        n = len(recs)
        if n < min_sample:
            continue
        failed_rate = sum(1 for outcome in recs if not outcome.passed) / n
        if failed_rate < UNDER_PROVISIONED_RATE:
            continue
        from_cap = registry_scores.get(model_id)
        if from_cap is None:
            continue
        to_cap = max(MIN_SCORE_FLOOR, from_cap - 5)
        if to_cap >= from_cap:
            continue
        identity = (model_id, adapter, role)
        evidence_dates = [
            evaluated
            for outcome in recs
            if outcome.evaluated_at
            and (evaluated := _evaluated_on(outcome.evaluated_at)) is not None
        ]
        last_calibrated = (
            max(evidence_dates).isoformat()
            if evidence_dates
            else ""
        )
        if not last_calibrated:
            continue
        suggestion = {
            "model_id": model_id,
            "adapter": adapter,
            "role": role,
            "from_capability": from_cap,
            "to_capability": to_cap,
            "rationale": (
                f"objective evaluator failure rate {failed_rate:.0%} over "
                f"{n} {role} runs in one compatible epoch; recommendation-only "
                "role card, not applied to capability_score."
            ),
            "sample_count": n,
            "last_calibrated": last_calibrated,
            "epoch": {
                "registry_digest": group_key[3],
                "classifier_version": group_key[4],
                "taxonomy_version": group_key[5],
                "adapter_version": group_key[6],
                "evaluator_revision": group_key[7],
            },
        }
        rank = (last_calibrated, *group_key[3:])
        previous = candidates.get(identity)
        if previous is None or rank > previous[0]:
            candidates[identity] = (rank, suggestion)
    return [candidates[identity][1] for identity in sorted(candidates)]


def build_audit_report(
    records: list[TaskAuditRecord],
    registry_scores: dict[str, int],
    *,
    window_days: Optional[float] = None,
    jobs_considered: int = 0,
    low_confidence_bar: float = LOW_CONFIDENCE_BAR,
    min_sample: int = MIN_SAMPLE,
    actual_cost_fn: Optional[Callable[[str, int, int], float]] = None,
) -> AuditReport:
    """Aggregate per-model routing behavior and propose conservative score
    adjustments. Pure function — no I/O.

    ``actual_cost_fn(model_id, tokens_in, tokens_out)`` prices measured token
    consumption the same way the router priced its estimate (marginal cost, so
    plan-billed models read $0 on both sides — an honest apples-to-apples
    comparison). When omitted, only token drift is reconciled and actual-cost
    columns stay zero rather than guessing a dollar figure.
    """
    model_ids = set(registry_scores) | {r.model_id for r in records}
    model_ids |= {r.escalated_from for r in records if r.escalated_from}
    model_ids |= {
        source for record in records for source in _fallback_sources(record)
    }
    model_ids |= {
        source
        for record in records
        for source in _review_escalation_sources(record)
    }

    # escalated-away counts keyed by the model the task escalated OFF of.
    escalated_away: dict[str, int] = {}
    for r in records:
        if r.escalated and r.escalated_from:
            escalated_away[r.escalated_from] = escalated_away.get(r.escalated_from, 0) + 1
    fallback_away: dict[str, int] = {}
    review_escalated_away: dict[str, int] = {}
    for r in records:
        for source in _fallback_sources(r):
            fallback_away[source] = fallback_away.get(source, 0) + 1
        for source in _review_escalation_sources(r):
            review_escalated_away[source] = review_escalated_away.get(source, 0) + 1

    audits: list[ModelAudit] = []
    for model_id in sorted(model_ids):
        retained = [r for r in records if r.model_id == model_id]
        away = escalated_away.get(model_id, 0)
        fallback_away_count = fallback_away.get(model_id, 0)
        review_away_count = review_escalated_away.get(model_id, 0)
        # A model was the initial pick if it either retained the task (no
        # escalation) or the task escalated away from it.
        retained_initial = [
            r
            for r in retained
            if not r.escalated
            and not _fallback_sources(r)
            and not _review_escalation_sources(r)
        ]
        selections = (
            len(retained_initial) + away + fallback_away_count + review_away_count
        )

        confidences = [r.confidence for r in retained if r.confidence is not None]
        low = [c for c in confidences if c < low_confidence_bar]
        spend = sum(r.est_cost_usd for r in retained)
        objective_outcomes = [
            outcome
            for r in records
            for outcome in _record_objective_outcomes(r)
            if outcome.model_id == model_id
        ]
        predicted = [
            outcome.predicted_quality
            for outcome in objective_outcomes
            if outcome.predicted_quality is not None
        ]
        objective_quality = [
            outcome.objective_quality
            for outcome in objective_outcomes
            if outcome.objective_quality is not None
        ]

        score = registry_scores.get(model_id)
        escalated_away_rate = (away / selections) if selections else 0.0
        low_conf_rate = (len(low) / len(confidences)) if confidences else 0.0

        reconciled = [r for r in retained if r.has_actuals]
        measured_recs = [r for r in reconciled if r.actual_tokens_measured]
        est_tokens = sum(r.est_tokens_total for r in reconciled)
        actual_tokens = sum(r.actual_tokens_total for r in reconciled)
        measured_runs = len(measured_recs)
        actual_spend = 0.0
        if actual_cost_fn is not None:
            actual_spend = sum(
                actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
                for r in reconciled
            )
        recon_est_spend = sum(r.est_cost_usd for r in reconciled)
        # Headline calibration uses measured (SDK) runs only so char/4
        # approximations cannot skew token/cost drift.
        est_tokens_m = sum(r.est_tokens_total for r in measured_recs)
        actual_tokens_m = sum(r.actual_tokens_total for r in measured_recs)
        actual_spend_m = 0.0
        if actual_cost_fn is not None:
            actual_spend_m = sum(
                actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
                for r in measured_recs
            )
        recon_est_spend_m = sum(r.est_cost_usd for r in measured_recs)
        elapsed_vals = [r.elapsed_seconds for r in retained if r.elapsed_seconds is not None]
        mean_elapsed = _mean(elapsed_vals)

        audit = ModelAudit(
            model_id=model_id,
            adapter=retained[0].adapter if retained else "",
            score=score,
            selections=selections,
            runs_with_confidence=len(confidences),
            mean_confidence=_mean(confidences),
            min_confidence=min(confidences) if confidences else None,
            low_confidence_rate=round(low_conf_rate, 3),
            escalated_away=away,
            escalated_away_rate=round(escalated_away_rate, 3),
            fell_back_away=fallback_away_count,
            est_spend_usd=round(spend, 6),
            runs_with_actuals=len(reconciled),
            measured_runs=measured_runs,
            est_tokens=est_tokens,
            actual_tokens=actual_tokens,
            token_drift_ratio=(
                round(actual_tokens_m / est_tokens_m, 3) if est_tokens_m else None
            ),
            actual_spend_usd=round(actual_spend, 6),
            cost_drift_ratio=(
                round(actual_spend_m / recon_est_spend_m, 3)
                if recon_est_spend_m
                else None
            ),
            mean_elapsed_seconds=(
                round(mean_elapsed, 3) if mean_elapsed is not None else None
            ),
            timeout_or_failed_rate=round(
                (sum(1 for r in retained if _record_failed(r)) / len(retained))
                if retained
                else 0.0,
                3,
            ),
            degraded_rate=round(
                (sum(1 for r in retained if _record_degraded(r)) / len(retained))
                if retained
                else 0.0,
                3,
            ),
            roles=_role_counts(model_id, retained, records),
            review_escalated_away=review_away_count,
            runs_with_objective_outcomes=len(objective_outcomes),
            objective_pass_rate=(
                round(
                    sum(1 for outcome in objective_outcomes if outcome.passed) /
                    len(objective_outcomes),
                    3,
                )
                if objective_outcomes
                else None
            ),
            mean_predicted_quality=(
                round(_mean(predicted), 3) if predicted else None
            ),
            mean_objective_quality=(
                round(_mean(objective_quality), 3) if objective_quality else None
            ),
        )
        _classify(audit, retained, low_confidence_bar, min_sample)
        audits.append(audit)

    audits.sort(key=lambda m: (m.selections, m.est_spend_usd), reverse=True)
    reconciled_all = [r for r in records if r.has_actuals]
    measured_all = [r for r in reconciled_all if r.actual_tokens_measured]
    total_actual_spend = 0.0
    total_actual_spend_m = 0.0
    if actual_cost_fn is not None:
        total_actual_spend = sum(
            actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
            for r in reconciled_all
        )
        total_actual_spend_m = sum(
            actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
            for r in measured_all
        )
    return AuditReport(
        jobs_considered=jobs_considered,
        tasks_considered=len(records),
        window_days=window_days,
        total_est_spend_usd=round(sum(r.est_cost_usd for r in records), 6),
        models=audits,
        tasks_with_actuals=len(reconciled_all),
        total_est_tokens=sum(r.est_tokens_total for r in reconciled_all),
        total_actual_tokens=sum(r.actual_tokens_total for r in reconciled_all),
        total_actual_spend_usd=round(total_actual_spend, 6),
        total_est_spend_reconciled_usd=round(
            sum(r.est_cost_usd for r in reconciled_all), 6
        ),
        tasks_with_measured=len(measured_all),
        total_est_tokens_measured=sum(r.est_tokens_total for r in measured_all),
        total_actual_tokens_measured=sum(r.actual_tokens_total for r in measured_all),
        total_actual_spend_measured_usd=round(total_actual_spend_m, 6),
        total_est_spend_measured_usd=round(
            sum(r.est_cost_usd for r in measured_all), 6
        ),
        role_scorecard_suggestions=_role_scorecard_suggestions(
            records, registry_scores, min_sample
        ),
        epoch_count=len(_epoch_keys(records)),
    )


def _classify(
    audit: ModelAudit,
    retained: list[TaskAuditRecord],
    low_confidence_bar: float,
    min_sample: int,
) -> None:
    """Attach flags + (only when defensible) a suggested score to ``audit``."""
    # Confidence/escalation is retained as a weak diagnostic, never as routing
    # authority.  Objective failures are handled by epoch-local role cards.
    under = (
        audit.selections >= min_sample
        and (
            audit.escalated_away_rate >= UNDER_PROVISIONED_RATE
            or audit.low_confidence_rate >= 0.5
        )
    )
    if under:
        audit.flags.append("weak-self-signal")
        audit.rationale = (
            f"escalated away {audit.escalated_away_rate:.0%} of "
            f"{audit.selections} picks / low-confidence "
            f"{audit.low_confidence_rate:.0%}; diagnostic only, because worker "
            "self-confidence is not objective routing authority."
        )
        return

    # Possibly over-used: a strong model doing work that needed much less.
    # Informational only — proving a cheaper model would have sufficed needs a
    # shadow run, which this audit doesn't do, so no score is proposed.
    if audit.score is not None and retained:
        needs = [r.capability_needed for r in retained if r.capability_needed]
        if needs:
            typical_need = sorted(needs)[len(needs) // 2]  # median
            high_conf = (audit.mean_confidence or 0) >= 0.85
            if (
                audit.selections >= min_sample
                and audit.score - typical_need >= OVER_USE_GAP
                and high_conf
            ):
                audit.flags.append("possibly-over-used")
                audit.rationale = (
                    f"typical task needed ~{typical_need} but ran on a "
                    f"score-{audit.score} model at {audit.mean_confidence:.0%} "
                    f"confidence; a cheaper tier may suffice (verify with a "
                    f"shadow run before lowering anything)."
                )


# --- store collector -------------------------------------------------------


def collect_records(store, *, window_days: Optional[float] = None) -> tuple[list[TaskAuditRecord], int]:
    """Pull per-task routing outcomes from ``store``, optionally limited to jobs
    created within ``window_days``. Returns (records, jobs_considered)."""
    from datetime import datetime, timedelta, timezone

    from puppetmaster.receipt import _elapsed_seconds

    cutoff: Optional[datetime] = None
    if window_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    records: list[TaskAuditRecord] = []
    jobs_considered = 0
    eligible_jobs = []
    for job in store.list_jobs():
        if cutoff is not None and not _within(job.created_at, cutoff):
            continue
        eligible_jobs.append(job)
    job_ids = [job.id for job in eligible_jobs]
    all_artifacts = store.list_artifacts_for_jobs(job_ids)
    all_tasks = store.list_tasks_for_jobs(job_ids)
    artifacts_by_job: dict[str, list] = {}
    for artifact in all_artifacts:
        artifacts_by_job.setdefault(artifact.job_id, []).append(artifact)
    tasks_by_job: dict[str, dict] = {}
    for task in all_tasks:
        tasks_by_job.setdefault(task.job_id, {})[task.id] = task

    for job in eligible_jobs:
        jobs_considered += 1
        artifacts = artifacts_by_job.get(job.id, [])
        tasks = tasks_by_job.get(job.id, {})

        # Initial routing picks and escalation/fallback events, per task.
        initial_by_task: dict[str, dict] = {}
        routing_history_by_task: dict[str, list[tuple[str, dict]]] = {}
        escalated_from: dict[str, str] = {}
        fallback_events: dict[str, list[tuple[str, str]]] = {}
        review_escalation_events: dict[str, list[tuple[str, str]]] = {}
        fell_back: set[str] = set()
        latest_conf: dict[str, tuple[str, float]] = {}  # task_id -> (created_at, confidence)
        # task_id -> (created_at, tokens_in, tokens_out, estimated) for the latest
        # run that actually reported token usage. Only verification artifacts that
        # carry a usage record contribute, so a task with no usage stays unknown.
        latest_usage: dict[str, tuple[str, int, int, bool]] = {}
        latest_verif_result: dict[str, tuple[str, object]] = {}
        objective_events: dict[
            str, list[tuple[str, bool, Optional[float], str, str]]
        ] = {}
        gate_flags_by_task: dict[str, list[bool]] = {}
        for a in artifacts:
            payload = a.payload or {}
            kind = a.type.value
            if kind == "routing":
                routing_history_by_task.setdefault(a.task_id, []).append(
                    (a.created_at, payload)
                )
                if a.created_by == "router":
                    initial_by_task[a.task_id] = payload
                elif a.created_by == "router-escalation":
                    frm = payload.get("escalated_from_model")
                    if frm:
                        escalated_from[a.task_id] = frm
                elif a.created_by == "router-fallback":
                    fell_back.add(a.task_id)
                    frm = payload.get("fallback_from_model")
                    if frm:
                        fallback_events.setdefault(a.task_id, []).append(
                            (a.created_at, str(frm))
                        )
                elif a.created_by == "router-review-escalation":
                    frm = payload.get("review_escalated_from_model")
                    if frm:
                        review_escalation_events.setdefault(a.task_id, []).append(
                            (a.created_at, str(frm))
                        )
            elif kind == "verification":
                prev = latest_conf.get(a.task_id)
                if prev is None or a.created_at >= prev[0]:
                    latest_conf[a.task_id] = (a.created_at, float(a.confidence))
                if "result" in payload:
                    prev_result = latest_verif_result.get(a.task_id)
                    if prev_result is None or a.created_at >= prev_result[0]:
                        latest_verif_result[a.task_id] = (
                            a.created_at,
                            payload.get("result"),
                        )
                if "tokens_in" in payload or "tokens_out" in payload:
                    prev_usage = latest_usage.get(a.task_id)
                    if prev_usage is None or a.created_at >= prev_usage[0]:
                        # Fail-closed: measured only when tokens_estimated is
                        # explicitly False. An omitted key is unknown, never
                        # treated as ground-truth SDK usage.
                        if "tokens_estimated" not in payload:
                            measured_flag = None
                        else:
                            measured_flag = payload.get("tokens_estimated") is False
                        latest_usage[a.task_id] = (
                            a.created_at,
                            int(payload.get("tokens_in") or 0),
                            int(payload.get("tokens_out") or 0),
                            measured_flag,
                        )
            elif kind == "gate" and "passed" in payload:
                passed = bool(payload.get("passed"))
                gate_flags_by_task.setdefault(a.task_id, []).append(passed)
                review_status = str(payload.get("review_status") or "").lower()
                if review_status in {
                    "unavailable",
                    "skipped",
                    "independence_failed",
                }:
                    continue
                raw_score = payload.get("objective_score")
                objective_score: Optional[float] = None
                if (
                    not isinstance(raw_score, bool)
                    and isinstance(raw_score, (int, float))
                    and 0.0 <= float(raw_score) <= 1.0
                ):
                    objective_score = float(raw_score)
                elif raw_score is None:
                    objective_score = 1.0 if passed else 0.0
                source = str(
                    payload.get("evaluator_slot")
                    or payload.get("kind")
                    or payload.get("gate")
                    or "gate"
                )
                evaluator_revision = str(
                    payload.get("evaluator_revision")
                    or payload.get("evaluator_version")
                    or ""
                )
                objective_events.setdefault(a.task_id, []).append(
                    (
                        a.created_at,
                        passed,
                        objective_score,
                        source,
                        evaluator_revision,
                    )
                )

        for task_id, task in tasks.items():
            payload = task.payload or {}
            final_model = payload.get("router_model_id")
            if not final_model:
                continue  # not a router-placed task
            initial = initial_by_task.get(task_id, {})
            conf = latest_conf.get(task_id)
            usage = latest_usage.get(task_id)
            verif = latest_verif_result.get(task_id)
            objectives = sorted(
                objective_events.get(task_id, []), key=lambda item: item[0]
            )
            objective = objectives[-1] if objectives else None
            objective_routing = initial
            fallback_sources = tuple(
                model_id
                for _, model_id in sorted(fallback_events.get(task_id, []))
            )
            review_escalation_sources = tuple(
                model_id
                for _, model_id in sorted(
                    review_escalation_events.get(task_id, [])
                )
            )
            if objective:
                prior_routes = [
                    item
                    for item in routing_history_by_task.get(task_id, [])
                    if item[0] <= objective[0]
                ]
                if prior_routes:
                    objective_routing = max(prior_routes, key=lambda item: item[0])[1]
            attributed_outcomes: list[ObjectiveAuditOutcome] = []
            for event in objectives:
                prior_routes = [
                    item
                    for item in routing_history_by_task.get(task_id, [])
                    if item[0] <= event[0]
                ]
                event_routing = (
                    max(prior_routes, key=lambda item: item[0])[1]
                    if prior_routes
                    else initial
                )
                attributed_outcomes.append(
                    ObjectiveAuditOutcome(
                        model_id=str(event_routing.get("model_id") or final_model),
                        predicted_quality=_optional_unit_float(
                            event_routing.get("predicted_quality")
                        ),
                        objective_quality=event[2],
                        passed=event[1],
                        source=event[3],
                        registry_digest=str(event_routing.get("registry_digest") or ""),
                        classifier_version=str(
                            event_routing.get("classifier_version") or ""
                        ),
                        taxonomy_version=str(
                            event_routing.get("taxonomy_version") or ""
                        ),
                        adapter_version=str(
                            event_routing.get("adapter_version") or ""
                        ),
                        evaluator_revision=event[4],
                        evaluated_at=event[0],
                    )
                )
            verification_result = verif[1] if verif else None
            if verification_result is not None and not isinstance(
                verification_result, str
            ):
                verification_result = str(verification_result)
            gates = gate_flags_by_task.get(task_id)
            records.append(
                TaskAuditRecord(
                    model_id=final_model,
                    adapter=task.adapter,
                    capability_needed=int(
                        payload.get("router_capability_needed")
                        or initial.get("capability_needed")
                        or 0
                    ),
                    est_cost_usd=float(payload.get("router_estimated_cost_usd") or 0.0),
                    confidence=conf[1] if conf else None,
                    escalated=task_id in escalated_from,
                    escalated_from=escalated_from.get(task_id),
                    fell_back=task_id in fell_back,
                    est_tokens_in=int(initial.get("estimated_tokens_in") or 0),
                    est_tokens_out=int(initial.get("estimated_tokens_out") or 0),
                    actual_tokens_in=usage[1] if usage else 0,
                    actual_tokens_out=usage[2] if usage else 0,
                    actual_tokens_measured=usage[3] if usage else None,
                    role=task.role or initial.get("role") or "",
                    elapsed_seconds=_elapsed_seconds(
                        task.created_at, task.completed_at
                    ),
                    verification_result=verification_result,
                    gate_passed=all(gates) if gates else None,
                    attempts=int(getattr(task, "attempts", 0) or 0),
                    fallback_attempts=int(payload.get("fallback_attempts") or 0),
                    escalation_attempts=int(payload.get("escalation_attempts") or 0),
                    fallback_from=(fallback_sources[-1] if fallback_sources else None),
                    review_escalated_from=(
                        review_escalation_sources[-1]
                        if review_escalation_sources
                        else None
                    ),
                    fallback_from_models=fallback_sources,
                    review_escalated_from_models=review_escalation_sources,
                    predicted_quality=_optional_unit_float(
                        objective_routing.get("predicted_quality")
                    ),
                    objective_quality=objective[2] if objective else None,
                    objective_passed=objective[1] if objective else None,
                    objective_source=objective[3] if objective else None,
                    objective_model_id=(
                        str(objective_routing.get("model_id") or final_model)
                        if objective
                        else None
                    ),
                    registry_digest=str(
                        objective_routing.get("registry_digest")
                        or payload.get("router_registry_digest")
                        or ""
                    ),
                    classifier_version=str(
                        objective_routing.get("classifier_version")
                        or payload.get("router_classifier_version")
                        or ""
                    ),
                    taxonomy_version=str(
                        objective_routing.get("taxonomy_version")
                        or payload.get("router_taxonomy_version")
                        or ""
                    ),
                    adapter_version=str(
                        objective_routing.get("adapter_version")
                        or payload.get("router_adapter_version")
                        or ""
                    ),
                    evaluator_revision=(objective[4] if objective else ""),
                    evaluated_at=(objective[0] if objective else ""),
                    objective_outcomes=tuple(attributed_outcomes),
                )
            )
    return records, jobs_considered


def _within(created_at: str, cutoff) -> bool:
    from datetime import datetime, timezone

    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= cutoff
    except (ValueError, AttributeError):
        return True  # undated/odd timestamps are kept rather than silently dropped
