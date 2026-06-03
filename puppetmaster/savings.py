"""Cumulative savings ledger — the per-user (and, later, per-org) receipt for
what Puppetmaster actually saved. Two *measured* pillars plus clearly-labeled
estimates, kept rigorously separate so the headline numbers survive scrutiny.

Honesty rules baked in (learned the hard way from a probe that "lost money"):

* **Rule 1 — symmetric costing.** Routing savings use the ``baseline_cost_usd``
  the router *snapshotted at decision time* (see ``router.RoutingDecision``),
  never a recomputed baseline that could drift against a changed registry.
  Artifacts predating the snapshot simply don't count toward the dollar figure.
* **Rule 2 — policy-aware.** Only cost-optimizing policies (``balanced`` /
  ``cheap``) count as savings. ``quality`` / ``escalating`` are *deliberate
  spend by request* — reported on their own line, never as a loss.
* **Measured vs estimated.** Routing dollars and CodeGraph context-tokens-fed
  are measured. "Avoided exploration" is an estimate with a stated baseline and
  is always presented as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

COST_OPTIMIZING_POLICIES = frozenset({"balanced", "cheap"})


@dataclass(frozen=True)
class RoutingRecord:
    policy: str
    chosen_cost_usd: float
    baseline_cost_usd: float
    has_baseline: bool


@dataclass
class SelfHeal:
    """Reliability lever — real counts, never dollarized. A 'fallback' is a task
    re-routed off a dead/unfunded provider (would otherwise have failed); an
    'escalation' is a task re-run one capability tier up for low confidence."""

    fallbacks: int = 0
    escalations: int = 0


@dataclass
class RoutingSavings:
    tasks_total: int = 0
    tasks_without_baseline: int = 0
    plan_routed_tasks: int = 0
    # cost-optimizing bucket (the headline)
    saved_usd: float = 0.0
    baseline_usd: float = 0.0
    chosen_usd: float = 0.0
    cost_optimizing_tasks: int = 0
    # deliberate spend bucket (quality/escalating, by request)
    deliberate_spend_usd: float = 0.0
    deliberate_tasks: int = 0

    @property
    def pct_cheaper(self) -> float:
        return (self.saved_usd / self.baseline_usd * 100.0) if self.baseline_usd else 0.0


def summarize_routing(records: list[RoutingRecord]) -> RoutingSavings:
    """Aggregate routing records into the policy-aware savings picture."""
    out = RoutingSavings()
    for r in records:
        out.tasks_total += 1
        if r.chosen_cost_usd == 0.0:
            out.plan_routed_tasks += 1
        if r.policy in COST_OPTIMIZING_POLICIES:
            if not r.has_baseline:
                out.tasks_without_baseline += 1
                continue
            out.cost_optimizing_tasks += 1
            out.saved_usd += r.baseline_cost_usd - r.chosen_cost_usd
            out.baseline_usd += r.baseline_cost_usd
            out.chosen_usd += r.chosen_cost_usd
        else:
            out.deliberate_tasks += 1
            out.deliberate_spend_usd += r.chosen_cost_usd
    out.saved_usd = round(out.saved_usd, 6)
    out.baseline_usd = round(out.baseline_usd, 6)
    out.chosen_usd = round(out.chosen_usd, 6)
    out.deliberate_spend_usd = round(out.deliberate_spend_usd, 6)
    return out


def collect_routing_records(
    stores: list, *, since: Optional[datetime] = None
) -> tuple[list[RoutingRecord], int, SelfHeal]:
    """Pull routing artifacts from every store in a single pass. Returns
    (records, jobs_considered, self_heal)."""
    records: list[RoutingRecord] = []
    self_heal = SelfHeal()
    jobs = 0
    for store in stores:
        try:
            job_list = store.list_jobs()
        except Exception:
            continue
        for job in job_list:
            if since is not None and not _job_within(job, since):
                continue
            jobs += 1
            try:
                artifacts = store.list_artifacts(job.id)
            except Exception:
                continue
            for a in artifacts:
                if a.type.value != "routing":
                    continue
                if a.created_by == "router-fallback":
                    self_heal.fallbacks += 1
                    continue
                if a.created_by == "router-escalation":
                    self_heal.escalations += 1
                    continue
                if a.created_by != "router":
                    continue
                p = a.payload or {}
                baseline = float(p.get("baseline_cost_usd") or 0.0)
                records.append(
                    RoutingRecord(
                        policy=p.get("policy") or "?",
                        chosen_cost_usd=float(p.get("estimated_cost_usd") or 0.0),
                        baseline_cost_usd=baseline,
                        has_baseline="baseline_cost_usd" in p and baseline > 0.0,
                    )
                )
    return records, jobs, self_heal


def _job_within(job, since: datetime) -> bool:
    try:
        ts = datetime.fromisoformat(str(job.created_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= since
    except (ValueError, AttributeError, TypeError):
        return True


def build_report(
    stores: list,
    *,
    window_days: Optional[float] = None,
) -> dict:
    """Top-level: routing savings (measured) + CodeGraph usage (measured +
    estimated). Pure aggregation over local data; emits nothing."""
    from puppetmaster import codegraph_usage, reads_log

    since: Optional[datetime] = None
    if window_days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=window_days)

    routing_records, jobs, self_heal = collect_routing_records(stores, since=since)
    routing = summarize_routing(routing_records)
    codegraph = codegraph_usage.aggregate(codegraph_usage.load_usage(since=since))
    reads = reads_log.aggregate(reads_log.load_reads(since=since))

    return {
        "window_days": window_days,
        "jobs_considered": jobs,
        "routing": routing,
        "self_heal": self_heal,
        "codegraph": codegraph,
        "reads": reads,
    }
