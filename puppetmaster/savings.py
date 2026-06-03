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

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

COST_OPTIMIZING_POLICIES = frozenset({"balanced", "cheap"})


@dataclass(frozen=True)
class RoutingRecord:
    policy: str
    chosen_cost_usd: float
    baseline_cost_usd: float
    has_baseline: bool
    picked_model_id: str = ""
    baseline_model_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


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
    # A task contributes exactly one *initial* routing decision. Dedup by task_id
    # guards against a re-dispatched task emitting multiple created_by=="router"
    # artifacts, which would otherwise inflate the savings dollars and counts.
    seen_router_tasks: set = set()
    for store in stores:
        try:
            job_list = store.list_jobs()
        except Exception:
            continue
        # Determine the in-window jobs once, then pull routing artifacts with a
        # single indexed query (SQLite: WHERE type='routing') instead of loading
        # and deserializing every artifact of every job just to find them.
        in_window = [
            job for job in job_list if since is None or _job_within(job, since)
        ]
        jobs += len(in_window)
        window_job_ids = {job.id for job in in_window}
        # When a window is set, scope the indexed query to the in-window jobs so
        # the backend doesn't deserialize the full historical routing ledger.
        scope = window_job_ids if since is not None else None
        routing_artifacts = None
        try:
            routing_artifacts = store.list_artifacts_by_type("routing", job_ids=scope)
        except TypeError:
            # Store predates the job_ids kwarg; fall back to the unscoped query.
            try:
                routing_artifacts = store.list_artifacts_by_type("routing")
            except Exception:
                routing_artifacts = None
        except Exception:
            routing_artifacts = None
        if routing_artifacts is None:
            # Fallback for stores without the indexed helper: per-job scan.
            routing_artifacts = []
            for job in in_window:
                try:
                    routing_artifacts.extend(
                        a for a in store.list_artifacts(job.id)
                        if a.type.value == "routing"
                    )
                except Exception:
                    continue
        for a in routing_artifacts:
            if getattr(a, "job_id", None) not in window_job_ids:
                continue
            if a.created_by == "router-fallback":
                self_heal.fallbacks += 1
                continue
            if a.created_by == "router-escalation":
                self_heal.escalations += 1
                continue
            if a.created_by != "router":
                continue
            tid = getattr(a, "task_id", None)
            if tid:
                if tid in seen_router_tasks:
                    continue
                seen_router_tasks.add(tid)
            p = a.payload or {}
            baseline = float(p.get("baseline_cost_usd") or 0.0)
            records.append(
                RoutingRecord(
                    policy=p.get("policy") or "?",
                    chosen_cost_usd=float(p.get("estimated_cost_usd") or 0.0),
                    baseline_cost_usd=baseline,
                    has_baseline="baseline_cost_usd" in p and baseline > 0.0,
                    picked_model_id=p.get("model_id") or "",
                    baseline_model_id=p.get("baseline_model_id") or "",
                    tokens_in=int(p.get("estimated_tokens_in") or 0),
                    tokens_out=int(p.get("estimated_tokens_out") or 0),
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
        # A job whose created_at can't be parsed has no place in a *windowed*
        # report — including it would silently pad windowed savings with
        # undatable jobs. Exclude it (the caller only invokes this when a
        # window is set; unwindowed reports never reach here).
        return False


def build_metrics(
    routing_records: list[RoutingRecord],
    self_heal: SelfHeal,
    codegraph: dict,
    reads: dict,
    jobs: int,
) -> dict:
    """Derived **rates and ratios** — the $-free org-success view. Every rate is
    over an explicit denominator, surfaced in ``sample`` so small samples are
    visible and nothing reads like a vanity total. A rate is ``None`` (JSON
    null) when its denominator is zero, never a misleading 0.0."""
    total_routed = len(routing_records)
    cost_opt = [r for r in routing_records if r.policy in COST_OPTIMIZING_POLICIES]
    # Capability-match is by model *identity*, not dollars, so it stays meaningful
    # for plan-billed shops where every model costs $0: did we run something other
    # than the strongest eligible model because the task didn't need it?
    judgeable = [r for r in cost_opt if r.baseline_model_id]
    right_sized = [
        r
        for r in judgeable
        if r.picked_model_id and r.picked_model_id != r.baseline_model_id
    ]

    def rate(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 3) if denominator else None

    return {
        # Discipline: of the cost-optimizing tasks we can judge, how often did we
        # run a model other than the strongest available (i.e. right-size down)?
        "capability_match_rate": rate(len(right_sized), len(judgeable)),
        # Calibration: how often did a task need a bump up a tier?
        "escalation_rate": rate(self_heal.escalations, total_routed),
        # Reliability: how often did a task fail over off a dead provider?
        "fallback_rate": rate(self_heal.fallbacks, total_routed),
        # Leverage: result reads served per job (work produced once, reused).
        "reuse_reads_per_job": rate(reads.get("reads", 0), jobs),
        # Efficiency: focused-context tokens fed per job.
        "context_tokens_per_job": (
            round(codegraph.get("context_tokens_fed", 0) / jobs, 1) if jobs else None
        ),
        "sample": {
            "routed_tasks": total_routed,
            "cost_optimizing_judgeable": len(judgeable),
            "jobs": jobs,
        },
    }


COUNTERFACTUAL_MODEL_ENV = "PUPPETMASTER_COUNTERFACTUAL_MODEL"


@dataclass
class Counterfactual:
    """The 'avoided spend vs the naive approach' number — explicitly a
    *counterfactual*, not measured cash. It prices the exact token volume we
    actually routed against a single reference model at its API rates, and
    subtracts what the routed work actually cost. On a plan-billed setup the
    actual cost is ~$0, so ``avoided_usd`` ≈ ``naive_cost_usd``.

    Only as honest as ``reference_model_id``: it answers "what would this have
    cost if every task had run on <reference> at metered rates?" — so the
    reference must be a model leadership agrees you'd otherwise have used.
    ``reference_priced`` is False when the chosen reference has no per-token
    price (then the number is $0 and we say so, rather than implying savings).
    """

    reference_model_id: str
    reference_priced: bool
    naive_cost_usd: float
    actual_cost_usd: float
    avoided_usd: float
    tasks: int


def resolve_counterfactual_model(registry: list):
    """Pick the reference model for the counterfactual.

    1. ``$PUPPETMASTER_COUNTERFACTUAL_MODEL`` (a registry model id) wins.
    2. Otherwise the highest-capability model that has a real per-token price
       (so the counterfactual reflects metered-API spend, not a $0 plan model).
    3. Falls back to the highest-capability model overall if nothing is priced.

    Returns ``None`` only for an empty registry.
    """
    if not registry:
        return None
    env = os.environ.get(COUNTERFACTUAL_MODEL_ENV)
    if env and env.strip():
        for m in registry:
            if getattr(m, "id", None) == env.strip():
                return m
    priced = [
        m
        for m in registry
        if (getattr(m, "input_per_mtok_usd", 0) or 0) > 0
        or (getattr(m, "output_per_mtok_usd", 0) or 0) > 0
    ]
    pool = priced or registry
    return max(pool, key=lambda m: getattr(m, "capability_score", 0))


def compute_counterfactual(
    records: list[RoutingRecord], reference
) -> Optional[Counterfactual]:
    """Price every routed task's token volume against ``reference`` and compare
    to what the work actually cost. ``None`` when there's no reference."""
    if reference is None:
        return None
    naive = 0.0
    actual = 0.0
    for r in records:
        naive += reference.estimate_cost_usd(r.tokens_in, r.tokens_out)
        actual += r.chosen_cost_usd
    in_price = getattr(reference, "input_per_mtok_usd", 0) or 0
    out_price = getattr(reference, "output_per_mtok_usd", 0) or 0
    return Counterfactual(
        reference_model_id=getattr(reference, "id", "?"),
        reference_priced=(in_price > 0 or out_price > 0),
        naive_cost_usd=round(naive, 6),
        actual_cost_usd=round(actual, 6),
        avoided_usd=round(naive - actual, 6),
        tasks=len(records),
    )


def build_report(
    stores: list,
    *,
    window_days: Optional[float] = None,
) -> dict:
    """Top-level: routing savings (measured) + CodeGraph usage (measured +
    estimated) + reliability/reuse counts + derived rates. Pure aggregation
    over local data; emits nothing."""
    from puppetmaster import codegraph_usage, reads_log

    since: Optional[datetime] = None
    if window_days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=window_days)

    routing_records, jobs, self_heal = collect_routing_records(stores, since=since)
    routing = summarize_routing(routing_records)
    codegraph = codegraph_usage.aggregate(codegraph_usage.load_usage(since=since))
    reads = reads_log.aggregate(reads_log.load_reads(since=since))
    metrics = build_metrics(routing_records, self_heal, codegraph, reads, jobs)

    counterfactual: Optional[Counterfactual] = None
    try:
        from puppetmaster.model_registry import load_registry

        reference = resolve_counterfactual_model(load_registry())
        counterfactual = compute_counterfactual(routing_records, reference)
    except Exception:
        counterfactual = None

    return {
        "window_days": window_days,
        "jobs_considered": jobs,
        "routing": routing,
        "self_heal": self_heal,
        "codegraph": codegraph,
        "reads": reads,
        "metrics": metrics,
        "counterfactual": counterfactual,
    }
