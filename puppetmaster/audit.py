"""Routing self-audit: turn the artifacts you already store into a
"here's how your routing actually behaved, here's where it looks mis-scored"
report — plus a *suggested* models.json diff you apply by hand.

Design stance (deliberate): this **recommends**, it does not silently
autopilot. The signals available (model self-reported confidence, escalation
rate) are noisy and gameable, so closing the loop without a human in it risks
feedback ratchets that only ever raise cost — the opposite of the point. So:

* The aggregator (:func:`build_audit_report`) is a pure function over records
  the caller collects from the store. Same input, same output.
* It only proposes a score change for the one defensible case:
  **an under-delivering model** (it keeps getting picked, then escalated away
  from or finishing with low confidence). Lowering its score reserves the
  harder work for a stronger model and stops the cheap-then-expensive
  double-run.
* "Over-used" (a strong model doing trivial work) is **flagged but never
  auto-adjusted** — proving a cheaper model would have sufficed needs a
  counterfactual (a shadow run), which this audit does not perform.
* Nothing is written unless the CLI is invoked with ``--apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

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

    @property
    def token_drift_ratio(self) -> Optional[float]:
        return (self.total_actual_tokens / self.total_est_tokens) if self.total_est_tokens else None

    @property
    def cost_drift_ratio(self) -> Optional[float]:
        denom = self.total_est_spend_reconciled_usd
        return (self.total_actual_spend_usd / denom) if denom else None

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


def _record_failed(record: TaskAuditRecord) -> bool:
    if _verification_label(record.verification_result) in _FAILED_RESULTS:
        return True
    return record.gate_passed is False


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
        if record.escalated and record.escalated_from == model_id:
            role = record.role or ""
            if role:
                counts[role] = counts.get(role, 0) + 1
    return counts


def _role_scorecard_suggestions(
    records: list[TaskAuditRecord],
    registry_scores: dict[str, int],
    min_sample: int,
) -> list[dict]:
    """Recommendation-only per-role card hints. Never written by --apply."""
    groups: dict[tuple, list[TaskAuditRecord]] = {}
    for record in records:
        role = record.role or ""
        if not role:
            continue
        key = (record.model_id, record.adapter, role)
        groups.setdefault(key, []).append(record)

    suggestions: list[dict] = []
    for (model_id, adapter, role), recs in sorted(groups.items()):
        n = len(recs)
        if n < min_sample:
            continue
        elapsed_vals = [r.elapsed_seconds for r in recs if r.elapsed_seconds is not None]
        mean_elapsed = _mean(elapsed_vals)
        failed_rate = sum(1 for r in recs if _record_failed(r)) / n
        degraded_rate = sum(1 for r in recs if _record_degraded(r)) / n

        peer_means: list[float] = []
        for (other_id, _adapter, other_role), other_recs in groups.items():
            if other_role != role or other_id == model_id:
                continue
            other_elapsed = [
                r.elapsed_seconds for r in other_recs if r.elapsed_seconds is not None
            ]
            other_mean = _mean(other_elapsed)
            if other_mean is not None:
                peer_means.append(other_mean)
        high_elapsed = False
        if mean_elapsed is not None and peer_means:
            peer_median = sorted(peer_means)[len(peer_means) // 2]
            if peer_median > 0 and mean_elapsed >= 2.0 * peer_median:
                high_elapsed = True
        high_fail = (
            failed_rate >= UNDER_PROVISIONED_RATE
            or degraded_rate >= UNDER_PROVISIONED_RATE
        )
        if not high_elapsed and not high_fail:
            continue
        from_cap = registry_scores.get(model_id)
        if from_cap is None:
            continue
        to_cap = max(MIN_SCORE_FLOOR, from_cap - 5)
        if to_cap >= from_cap:
            continue
        reasons: list[str] = []
        if high_elapsed:
            reasons.append(
                f"mean elapsed {mean_elapsed:.1f}s is much higher than peer "
                f"models for role={role}"
            )
        if high_fail:
            reasons.append(
                f"failed/timeout rate {failed_rate:.0%} / degraded rate "
                f"{degraded_rate:.0%} over {n} {role} runs"
            )
        suggestions.append(
            {
                "model_id": model_id,
                "adapter": adapter,
                "role": role,
                "from_capability": from_cap,
                "to_capability": to_cap,
                "rationale": (
                    "; ".join(reasons)
                    + "; recommendation-only role card, not applied to capability_score."
                ),
                "sample_count": n,
            }
        )
    return suggestions


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

    # escalated-away counts keyed by the model the task escalated OFF of.
    escalated_away: dict[str, int] = {}
    for r in records:
        if r.escalated and r.escalated_from:
            escalated_away[r.escalated_from] = escalated_away.get(r.escalated_from, 0) + 1

    audits: list[ModelAudit] = []
    for model_id in sorted(model_ids):
        retained = [r for r in records if r.model_id == model_id]
        away = escalated_away.get(model_id, 0)
        # A model was the initial pick if it either retained the task (no
        # escalation) or the task escalated away from it.
        retained_initial = [r for r in retained if not r.escalated]
        selections = len(retained_initial) + away

        confidences = [r.confidence for r in retained if r.confidence is not None]
        low = [c for c in confidences if c < low_confidence_bar]
        spend = sum(r.est_cost_usd for r in retained)
        fell_back_away = sum(1 for r in records if r.fell_back and r.escalated_from == model_id)

        score = registry_scores.get(model_id)
        escalated_away_rate = (away / selections) if selections else 0.0
        low_conf_rate = (len(low) / len(confidences)) if confidences else 0.0

        reconciled = [r for r in retained if r.has_actuals]
        est_tokens = sum(r.est_tokens_total for r in reconciled)
        actual_tokens = sum(r.actual_tokens_total for r in reconciled)
        measured_runs = sum(1 for r in reconciled if r.actual_tokens_measured)
        actual_spend = 0.0
        if actual_cost_fn is not None:
            actual_spend = sum(
                actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
                for r in reconciled
            )
        recon_est_spend = sum(r.est_cost_usd for r in reconciled)
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
            fell_back_away=fell_back_away,
            est_spend_usd=round(spend, 6),
            runs_with_actuals=len(reconciled),
            measured_runs=measured_runs,
            est_tokens=est_tokens,
            actual_tokens=actual_tokens,
            token_drift_ratio=round(actual_tokens / est_tokens, 3) if est_tokens else None,
            actual_spend_usd=round(actual_spend, 6),
            cost_drift_ratio=(
                round(actual_spend / recon_est_spend, 3) if recon_est_spend else None
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
        )
        _classify(audit, retained, low_confidence_bar, min_sample)
        audits.append(audit)

    audits.sort(key=lambda m: (m.selections, m.est_spend_usd), reverse=True)
    reconciled_all = [r for r in records if r.has_actuals]
    total_actual_spend = 0.0
    if actual_cost_fn is not None:
        total_actual_spend = sum(
            actual_cost_fn(r.model_id, r.actual_tokens_in, r.actual_tokens_out)
            for r in reconciled_all
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
        role_scorecard_suggestions=_role_scorecard_suggestions(
            records, registry_scores, min_sample
        ),
    )


def _classify(
    audit: ModelAudit,
    retained: list[TaskAuditRecord],
    low_confidence_bar: float,
    min_sample: int,
) -> None:
    """Attach flags + (only when defensible) a suggested score to ``audit``."""
    # Under-provisioned: gets picked, then can't finish confidently. Lower the
    # score so harder work routes to a stronger model. Defensible — there's a
    # real failure signal (escalation / low confidence), not a guess.
    under = (
        audit.selections >= min_sample
        and (
            audit.escalated_away_rate >= UNDER_PROVISIONED_RATE
            or audit.low_confidence_rate >= 0.5
        )
    )
    if under and audit.score is not None:
        audit.flags.append("under-provisioned")
        severe = (
            audit.escalated_away_rate >= SEVERE_RATE
            or audit.low_confidence_rate >= 0.7
        )
        step = 10 if severe else 5
        audit.suggested_score = max(MIN_SCORE_FLOOR, audit.score - step)
        audit.rationale = (
            f"escalated away {audit.escalated_away_rate:.0%} of "
            f"{audit.selections} picks / low-confidence "
            f"{audit.low_confidence_rate:.0%}; lower score so harder work "
            f"routes to a stronger model."
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
        escalated_from: dict[str, str] = {}
        fell_back: set[str] = set()
        latest_conf: dict[str, tuple[str, float]] = {}  # task_id -> (created_at, confidence)
        # task_id -> (created_at, tokens_in, tokens_out, estimated) for the latest
        # run that actually reported token usage. Only verification artifacts that
        # carry a usage record contribute, so a task with no usage stays unknown.
        latest_usage: dict[str, tuple[str, int, int, bool]] = {}
        latest_verif_result: dict[str, tuple[str, object]] = {}
        gate_flags_by_task: dict[str, list[bool]] = {}
        for a in artifacts:
            payload = a.payload or {}
            kind = a.type.value
            if kind == "routing":
                if a.created_by == "router":
                    initial_by_task[a.task_id] = payload
                elif a.created_by == "router-escalation":
                    frm = payload.get("escalated_from_model")
                    if frm:
                        escalated_from[a.task_id] = frm
                elif a.created_by == "router-fallback":
                    fell_back.add(a.task_id)
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
                        latest_usage[a.task_id] = (
                            a.created_at,
                            int(payload.get("tokens_in") or 0),
                            int(payload.get("tokens_out") or 0),
                            bool(payload.get("tokens_estimated")),
                        )
            elif kind == "gate" and "passed" in payload:
                gate_flags_by_task.setdefault(a.task_id, []).append(
                    bool(payload.get("passed"))
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
                    actual_tokens_measured=(not usage[3]) if usage else None,
                    role=task.role or initial.get("role") or "",
                    elapsed_seconds=_elapsed_seconds(
                        task.created_at, task.completed_at
                    ),
                    verification_result=verification_result,
                    gate_passed=all(gates) if gates else None,
                    attempts=int(getattr(task, "attempts", 0) or 0),
                    fallback_attempts=int(payload.get("fallback_attempts") or 0),
                    escalation_attempts=int(payload.get("escalation_attempts") or 0),
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
