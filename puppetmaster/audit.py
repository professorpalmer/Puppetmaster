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
from typing import Optional

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


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def build_audit_report(
    records: list[TaskAuditRecord],
    registry_scores: dict[str, int],
    *,
    window_days: Optional[float] = None,
    jobs_considered: int = 0,
    low_confidence_bar: float = LOW_CONFIDENCE_BAR,
    min_sample: int = MIN_SAMPLE,
) -> AuditReport:
    """Aggregate per-model routing behavior and propose conservative score
    adjustments. Pure function — no I/O."""
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
        )
        _classify(audit, retained, low_confidence_bar, min_sample)
        audits.append(audit)

    audits.sort(key=lambda m: (m.selections, m.est_spend_usd), reverse=True)
    return AuditReport(
        jobs_considered=jobs_considered,
        tasks_considered=len(records),
        window_days=window_days,
        total_est_spend_usd=round(sum(r.est_cost_usd for r in records), 6),
        models=audits,
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

    cutoff: Optional[datetime] = None
    if window_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    records: list[TaskAuditRecord] = []
    jobs_considered = 0
    for job in store.list_jobs():
        if cutoff is not None and not _within(job.created_at, cutoff):
            continue
        jobs_considered += 1
        artifacts = store.list_artifacts(job.id)
        tasks = {t.id: t for t in store.list_tasks(job.id)}

        # Initial routing picks and escalation/fallback events, per task.
        initial_by_task: dict[str, dict] = {}
        escalated_from: dict[str, str] = {}
        fell_back: set[str] = set()
        latest_conf: dict[str, tuple[str, float]] = {}  # task_id -> (created_at, confidence)
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

        for task_id, task in tasks.items():
            payload = task.payload or {}
            final_model = payload.get("router_model_id")
            if not final_model:
                continue  # not a router-placed task
            initial = initial_by_task.get(task_id, {})
            conf = latest_conf.get(task_id)
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
