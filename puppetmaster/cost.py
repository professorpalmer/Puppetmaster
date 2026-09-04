"""Actual spend, decoupled from the router.

The original ``cost`` command read only ``ROUTING`` artifacts, so a *pinned*
run — where the router never executes and emits no ROUTING artifact — reported
``$0`` with "didn't auto-route". That conflated two unrelated things: how a
model got chosen (the router) and what the work cost (tokens × price). Cost
must be a pure downstream function of *(tokens actually consumed)* × *(registry
price of the model actually used)*, independent of whether routing happened.

``price_job`` does exactly that. It reads the token usage every adapter already
stamps on its artifacts (see :mod:`puppetmaster.usage`), resolves which registry
model each task actually ran on — preferring the router's recorded ``model_id``,
then the model the adapter stamped on its verification artifact — and prices it
against the registry. Pinned, auto-routed, or plan-billed: every run that
produced token usage gets a priced ledger.

``job_counterfactual`` reuses the same per-task token volume against a single
reference model (resolved by :mod:`puppetmaster.savings`) so "what would this
have cost on the flagship at metered rates?" is answerable post-hoc — again,
pinned or not. On a plan-billed setup the actual marginal cost is ~$0, so the
avoided figure ≈ the naive figure.

``build_cost_report`` is the structured payload behind
``puppetmaster cost <job_id> --json``. CLI, MCP ``puppetmaster_job_cost``, and
Marionette all consume that one function. A coordinator-stamped
``Job.cost_receipt`` is returned detached after a cost-final completion.
Pre-upgrade final jobs without a valid receipt get a labeled artifact-only
report that ignores current registry rates. Stalled and active jobs may
reprice against the current registry and say so.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Optional

from puppetmaster.models import (
    Artifact,
    ArtifactType,
    Job,
    is_cost_final_job_status,
)
from puppetmaster.savings import (
    Counterfactual,
    resolve_counterfactual_model,
)
from puppetmaster.model_registry import default_registry_path, load_registry
from puppetmaster.usage import aggregate_token_usage, select_usage_records
from puppetmaster.validation import validation_status_of

_WITHDRAWN_VALIDATION = frozenset({"stale", "superseded"})

PRICING_SOURCE_TERMINAL = "terminal_receipt"
PRICING_SOURCE_CURRENT = "current_registry"
PRICING_SOURCE_LEGACY = "legacy_artifacts"

# Published cache-read ratio; conservative vs provider-specific tiers.
CACHE_READ_MULTIPLIER = 0.1


@dataclass(frozen=True)
class TaskCost:
    """One task's measured/estimated token spend, priced against the model it
    actually ran on. ``priced`` is False when no registry model could be matched
    (the tokens are still counted, but no dollar figure can be attributed)."""

    task_id: str
    model_id: str
    billing: str
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool
    marginal_cost_usd: float
    priced: bool


@dataclass
class JobCost:
    """A job's priced ledger, split measured vs estimated and by model.

    ``total_marginal_cost_usd`` is the out-of-pocket spend (plan-billed models
    contribute $0). ``measured_cost_usd`` / ``estimated_cost_usd`` partition that
    total by whether the underlying token counts were measured from an SDK/usage
    block or approximated char/4 — so a number derived from estimated tokens is
    never silently presented as measured.
    """

    total_marginal_cost_usd: float = 0.0
    measured_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    measured_runs: int = 0
    estimated_runs: int = 0
    priced_tasks: int = 0
    unpriced_tasks: int = 0
    by_model: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)
    route_estimated_tokens: int = 0
    measured_usage_tokens: int = 0
    token_estimate_drift_ratio: Optional[float] = None
    route_nominal_cost_usd: float = 0.0
    nominal_usage_cost_usd: float = 0.0
    nominal_cost_drift_ratio: Optional[float] = None


def _model_index(registry: list) -> tuple[dict, dict]:
    """Index a registry by ``id`` and by ``adapter_model_name`` so a task's
    recorded model string resolves whichever spelling the adapter stamped."""
    by_id: dict = {}
    by_adapter_name: dict = {}
    for spec in registry:
        spec_id = getattr(spec, "id", None)
        if spec_id:
            by_id[spec_id] = spec
        adapter_name = getattr(spec, "adapter_model_name", None)
        # Don't let a generic placeholder ("default") shadow a real id, and
        # never overwrite an id that's already a real match.
        if adapter_name and adapter_name not in by_adapter_name:
            by_adapter_name[adapter_name] = spec
    return by_id, by_adapter_name


# Final routing wins: a task may emit router + router-fallback (+ escalation).
# Pricing must follow the model that actually ran, not the initial pick that
# failed over (e.g. plan-billed cursor -> agentic glm).
_ROUTING_CREATED_BY_RANK = {
    "router-escalation": 3,
    "router-fallback": 2,
    "router": 1,
}


def _routing_created_by_rank(created_by: Optional[str]) -> int:
    return _ROUTING_CREATED_BY_RANK.get(created_by or "", 0)


def _is_current_routing_artifact(artifact: Artifact) -> bool:
    return validation_status_of(artifact) not in _WITHDRAWN_VALIDATION


def final_routing_artifacts(artifacts: Iterable[Artifact]) -> dict[str, Artifact]:
    best: dict[str, tuple[int, Artifact]] = {}
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING:
            continue
        if not _is_current_routing_artifact(artifact):
            continue
        rank = _routing_created_by_rank(getattr(artifact, "created_by", None))
        task_id = getattr(artifact, "task_id", None)
        if rank == 0 or not task_id:
            continue
        previous = best.get(task_id)
        if previous is None or rank > previous[0]:
            best[task_id] = (rank, artifact)
    return {task_id: artifact for task_id, (_rank, artifact) in best.items()}


def _usage_records(artifacts: Iterable[Artifact]) -> dict:
    """Selected usage keyed by task_id, excluding untasked artifacts."""
    return {
        task_id: record
        for task_id, record in select_usage_records(artifacts).items()
        if not str(task_id).startswith("__untasked_")
    }


def _real_cost_usd(raw: Any) -> float:
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _job_cost_from_routes(final_routes: dict) -> JobCost:
    result = JobCost()
    result.route_estimated_tokens = sum(
        int((artifact.payload or {}).get("estimated_tokens_in") or 0)
        + int((artifact.payload or {}).get("estimated_tokens_out") or 0)
        for artifact in final_routes.values()
    )
    result.route_nominal_cost_usd = round(
        sum(
            float((artifact.payload or {}).get("nominal_cost_usd") or 0.0)
            for artifact in final_routes.values()
        ),
        6,
    )
    return result


def _add_priced_task(
    result: JobCost,
    *,
    task_id: str,
    model_id: str,
    billing: str,
    tokens_in: int,
    tokens_out: int,
    estimated: bool,
    cost: float,
    priced: bool,
    nominal_cost: float = 0.0,
) -> None:
    result.nominal_usage_cost_usd += nominal_cost
    result.measured_usage_tokens += tokens_in + tokens_out
    if priced:
        result.priced_tasks += 1
    else:
        result.unpriced_tasks += 1
    result.total_marginal_cost_usd += cost
    if estimated:
        result.estimated_cost_usd += cost
        result.estimated_runs += 1
    else:
        result.measured_cost_usd += cost
        result.measured_runs += 1
    bucket = result.by_model.setdefault(
        model_id,
        {
            "calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "marginal_cost_usd": 0.0,
            "billing": billing,
        },
    )
    bucket["calls"] += 1
    bucket["tokens_in"] += tokens_in
    bucket["tokens_out"] += tokens_out
    bucket["marginal_cost_usd"] += cost
    result.tasks.append(
        TaskCost(
            task_id=task_id,
            model_id=model_id,
            billing=billing,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_estimated=estimated,
            marginal_cost_usd=round(cost, 6),
            priced=priced,
        )
    )


def _finalize_job_cost(result: JobCost) -> JobCost:
    result.total_marginal_cost_usd = round(result.total_marginal_cost_usd, 6)
    result.measured_cost_usd = round(result.measured_cost_usd, 6)
    result.estimated_cost_usd = round(result.estimated_cost_usd, 6)
    for bucket in result.by_model.values():
        bucket["marginal_cost_usd"] = round(bucket["marginal_cost_usd"], 6)
    result.nominal_usage_cost_usd = round(result.nominal_usage_cost_usd, 6)
    if result.route_estimated_tokens > 0:
        result.token_estimate_drift_ratio = round(
            result.measured_usage_tokens / result.route_estimated_tokens, 6
        )
    if result.route_nominal_cost_usd > 0:
        result.nominal_cost_drift_ratio = round(
            result.nominal_usage_cost_usd / result.route_nominal_cost_usd, 6
        )
    return result


def _resolve_spec(
    routing_model_id: Optional[str],
    recorded_model: Optional[str],
    by_id: dict,
    by_adapter_name: dict,
):
    """Pick the registry spec a task ran on: final routing decision first, then
    the model the adapter recorded (matched by id, then by adapter_model_name).

    Fail closed when a final routing ``model_id`` is present but absent from
    the registry — do not fall through to an unrelated recorded-model match.
    """
    if routing_model_id:
        return by_id.get(routing_model_id)
    if recorded_model:
        if recorded_model in by_id:
            return by_id[recorded_model]
        if recorded_model in by_adapter_name:
            return by_adapter_name[recorded_model]
    return None


def _cost_with_cache_discount(spec, tokens_in: int, tokens_out: int, tokens_cached: int) -> float:
    """Price input tokens with cache-read discount on the cached portion."""
    uncached_in = max(0, tokens_in - tokens_cached)
    scaled_tokens_out = tokens_out * getattr(spec, "output_token_multiplier", 1)
    return (
        (uncached_in / 1_000_000.0) * spec.input_per_mtok_usd
        + (tokens_cached / 1_000_000.0) * spec.input_per_mtok_usd * CACHE_READ_MULTIPLIER
        + (scaled_tokens_out / 1_000_000.0) * spec.output_per_mtok_usd
    )


def price_job(artifacts: Iterable[Artifact], registry: list) -> JobCost:
    """Price each task, then sum a selected-model usage cost.

    Per-task precedence (unchanged): matching registry model with
    ``billing="plan"`` → $0 marginal; else positive artifact
    ``real_cost_usd`` → that reported value; else matching registry model →
    tokens × registry prices (cache-read discount + output multiplier);
    else unpriced (aggregate unknown, not $0). Measured vs estimated
    describes the token source, not the billing basis. Independent of
    routing.
    """
    artifacts = list(artifacts)
    by_id, by_adapter_name = _model_index(registry)
    final_routes = final_routing_artifacts(artifacts)
    routing_models = {
        task_id: str((artifact.payload or {}).get("model_id"))
        for task_id, artifact in final_routes.items()
        if (artifact.payload or {}).get("model_id")
    }
    result = _job_cost_from_routes(final_routes)
    for task_id, record in _usage_records(artifacts).items():
        spec = _resolve_spec(
            routing_models.get(task_id), record["model"], by_id, by_adapter_name
        )
        tokens_in = record["tokens_in"]
        tokens_out = record["tokens_out"]
        tokens_cached = record["tokens_cached"]
        estimated = record["tokens_estimated"]
        real_cost_f = _real_cost_usd(record["real_cost_usd"])
        plan_billed = spec is not None and getattr(spec, "billing", None) == "plan"
        nominal_cost = (
            _cost_with_cache_discount(spec, tokens_in, tokens_out, tokens_cached)
            if spec is not None
            else 0.0
        )

        if plan_billed:
            model_id = spec.id
            billing = spec.billing
            cost = 0.0
            priced = True
        elif real_cost_f > 0:
            cost = real_cost_f
            priced = True
            if spec is not None:
                model_id = spec.id
                billing = spec.billing
            else:
                model_id = routing_models.get(task_id) or record["model"] or "<unknown>"
                billing = "reported"
        elif spec is not None:
            model_id = spec.id
            billing = spec.billing
            if tokens_cached > 0:
                cost = _cost_with_cache_discount(spec, tokens_in, tokens_out, tokens_cached)
            else:
                cost = spec.marginal_cost_usd(tokens_in, tokens_out)
            priced = True
        else:
            model_id = routing_models.get(task_id) or record["model"] or "<unknown>"
            billing = "unknown"
            cost = 0.0
            priced = False

        _add_priced_task(
            result,
            task_id=task_id,
            model_id=model_id,
            billing=billing,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated=estimated,
            cost=cost,
            priced=priced,
            nominal_cost=nominal_cost,
        )
    return _finalize_job_cost(result)


def job_counterfactual(job_cost: JobCost, registry: list) -> Optional[Counterfactual]:
    """Price this job's measured/estimated token volume against a single
    reference (flagship-priced) model and subtract what the work actually cost.

    Answers "what would this job have cost if every task had run on
    <reference> at metered rates?" — computable post-hoc, pinned or not. Returns
    ``None`` for an empty registry."""
    reference = resolve_counterfactual_model(registry)
    if reference is None:
        return None
    naive = 0.0
    for task in job_cost.tasks:
        naive += reference.estimate_cost_usd(task.tokens_in, task.tokens_out)
    actual = job_cost.total_marginal_cost_usd
    in_price = getattr(reference, "input_per_mtok_usd", 0) or 0
    out_price = getattr(reference, "output_per_mtok_usd", 0) or 0
    return Counterfactual(
        reference_model_id=getattr(reference, "id", "?"),
        reference_priced=(in_price > 0 or out_price > 0),
        naive_cost_usd=round(naive, 6),
        actual_cost_usd=round(actual, 6),
        avoided_usd=round(naive - actual, 6),
        tasks=len(job_cost.tasks),
    )



def routing_estimate_rows(artifacts: Iterable[Artifact]) -> tuple[list[dict], dict[str, dict], float]:
    """The pre-flight routing estimate: per-task rows + per-model rollup + total.

    Only the router's *initial* decision per task counts. Fallback/escalation
    reroutes (created_by 'router-fallback' / 'router-escalation') emit their own
    ROUTING artifacts; summing all of them double-counts a rerouted task. Dedup
    by task_id mirrors ``savings.collect_routing_records``.
    """
    rows: list[dict] = []
    by_model: dict[str, dict] = {}
    total = 0.0
    seen_router_tasks: set = set()
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        if not _is_current_routing_artifact(artifact):
            continue
        task_id = artifact.task_id
        if task_id:
            if task_id in seen_router_tasks:
                continue
            seen_router_tasks.add(task_id)
        payload = artifact.payload or {}
        model_id = payload.get("model_id", "<unknown>")
        cost = float(payload.get("estimated_cost_usd") or 0.0)
        total += cost
        rows.append(
            {
                "task_id": task_id,
                "role": payload.get("role"),
                "model_id": model_id,
                "adapter": payload.get("adapter"),
                "policy": payload.get("policy"),
                "capability_needed": payload.get("capability_needed"),
                "estimated_cost_usd": cost,
            }
        )
        bucket = by_model.setdefault(model_id, {"calls": 0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["cost"] += cost
    return rows, by_model, round(total, 6)


def valid_terminal_cost_receipt(receipt: Any, job_id: str) -> bool:
    """True for an additive completion receipt: matching job_id, terminal
    pricing_source, and actual_cost / token_usage / tasks objects.
    """
    if not isinstance(receipt, dict) or not receipt:
        return False
    if receipt.get("job_id") != job_id:
        return False
    if receipt.get("pricing_source") != PRICING_SOURCE_TERMINAL:
        return False
    if not isinstance(receipt.get("actual_cost"), dict):
        return False
    if not isinstance(receipt.get("token_usage"), dict):
        return False
    if not isinstance(receipt.get("tasks"), list):
        return False
    return True


def _actual_cost_payload(
    job_cost: JobCost,
    *,
    cost_basis: str = "measured_usage_x_registry_price",
) -> dict:
    """Serialize priced ledgers; unknown selected cost is null, not $0."""
    priced_subtotal = round(
        sum(task.marginal_cost_usd for task in job_cost.tasks if task.priced),
        6,
    )
    selected_unknown = job_cost.unpriced_tasks > 0
    unpriced_models = {task.model_id for task in job_cost.tasks if not task.priced}
    actual_by_model = {}
    for model_id, bucket in job_cost.by_model.items():
        actual_by_model[model_id] = {
            "calls": bucket["calls"],
            "tokens_in": bucket["tokens_in"],
            "tokens_out": bucket["tokens_out"],
            "marginal_cost_usd": (
                None if model_id in unpriced_models else bucket["marginal_cost_usd"]
            ),
            "billing": bucket["billing"],
        }
    tasks = []
    for task in job_cost.tasks:
        row = asdict(task)
        if not task.priced:
            row["marginal_cost_usd"] = None
        tasks.append(row)
    return {
        "cost_basis": cost_basis,
        "total_marginal_cost_usd": (
            None if selected_unknown else job_cost.total_marginal_cost_usd
        ),
        "measured_cost_usd": None if selected_unknown else job_cost.measured_cost_usd,
        "estimated_cost_usd": None if selected_unknown else job_cost.estimated_cost_usd,
        "priced_subtotal_usd": priced_subtotal,
        "measured_runs": job_cost.measured_runs,
        "estimated_runs": job_cost.estimated_runs,
        "priced_tasks": job_cost.priced_tasks,
        "unpriced_tasks": job_cost.unpriced_tasks,
        "by_model": actual_by_model,
        "tasks": tasks,
    }


def _counterfactual_payload(job_cost: JobCost, registry: list) -> Optional[dict]:
    counterfactual = job_counterfactual(job_cost, registry)
    if counterfactual is None:
        return None
    payload = asdict(counterfactual)
    actual_priced = job_cost.unpriced_tasks == 0
    payload["actual_priced"] = actual_priced
    if not actual_priced:
        payload["actual_cost_usd"] = None
        payload["avoided_usd"] = None
    return payload


def _report_envelope(
    job_id: str,
    artifacts: list,
    job_cost: JobCost,
    *,
    pricing_source: str,
    pricing_note: str,
    actual: dict,
    counterfactual: Optional[dict],
) -> dict:
    routing_rows, routing_by_model, routing_total = routing_estimate_rows(artifacts)
    return {
        "job_id": job_id,
        "pricing_source": pricing_source,
        "pricing_note": pricing_note,
        # Backward-compatible: the pre-flight routing estimate fields.
        "cost_basis": "preflight_routing_estimate",
        "total_estimated_cost_usd": routing_total,
        "by_model": {
            mid: {"calls": v["calls"], "estimated_cost_usd": round(v["cost"], 6)}
            for mid, v in routing_by_model.items()
        },
        "token_usage": aggregate_token_usage(artifacts),
        "estimate_drift": {
            "route_estimated_tokens": job_cost.route_estimated_tokens,
            "measured_usage_tokens": job_cost.measured_usage_tokens,
            "token_ratio": job_cost.token_estimate_drift_ratio,
            "route_nominal_cost_usd": job_cost.route_nominal_cost_usd,
            "nominal_usage_cost_usd": job_cost.nominal_usage_cost_usd,
            "nominal_cost_ratio": job_cost.nominal_cost_drift_ratio,
        },
        "actual_cost": actual,
        "counterfactual": counterfactual,
        "tasks": routing_rows if routing_rows else list(actual["tasks"]),
    }


def build_current_registry_cost_report(
    job_id: str,
    artifacts: Iterable[Artifact],
    registry: Optional[list] = None,
) -> dict:
    """Live selected-model report against the registry supplied (or current).

    Does not read or write ``Job.cost_receipt``. Used to stamp a new
    completion receipt and to label stalled/active jobs that have none.
    """
    artifacts = list(artifacts)
    if registry is None:
        try:
            registry = load_registry(default_registry_path()) or []
        except Exception:
            registry = []
    job_cost = price_job(artifacts, registry)
    return _report_envelope(
        job_id,
        artifacts,
        job_cost,
        pricing_source=PRICING_SOURCE_CURRENT,
        pricing_note=(
            "Live reprice against the current model registry; not a completion receipt."
        ),
        actual=_actual_cost_payload(job_cost),
        counterfactual=_counterfactual_payload(job_cost, registry),
    )


def price_job_from_artifacts(artifacts: Iterable[Artifact]) -> JobCost:
    """Price from persisted artifacts only — no current registry rates.

    Provider-reported ``real_cost_usd`` stays known. A final ROUTING
    artifact with ``billing=plan`` is a known plan-billed zero. API usage
    without an artifact price stays honestly unpriced.
    """
    artifacts = list(artifacts)
    final_routes = final_routing_artifacts(artifacts)
    result = _job_cost_from_routes(final_routes)
    for task_id, record in _usage_records(artifacts).items():
        route = final_routes.get(task_id)
        route_payload = route.payload or {} if route is not None else {}
        tokens_in = record["tokens_in"]
        tokens_out = record["tokens_out"]
        estimated = record["tokens_estimated"]
        real_cost_f = _real_cost_usd(record["real_cost_usd"])
        model_id = str(
            route_payload.get("model_id") or record["model"] or "<unknown>"
        )
        if route_payload.get("billing") == "plan":
            billing = "plan"
            cost = 0.0
            priced = True
        elif real_cost_f > 0:
            billing = "reported"
            cost = real_cost_f
            priced = True
        else:
            billing = "unknown"
            cost = 0.0
            priced = False
        _add_priced_task(
            result,
            task_id=task_id,
            model_id=model_id,
            billing=billing,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated=estimated,
            cost=cost,
            priced=priced,
        )
    return _finalize_job_cost(result)


def build_legacy_artifact_cost_report(
    job_id: str,
    artifacts: Iterable[Artifact],
) -> dict:
    """Stable economics for a pre-upgrade final job with no valid receipt.

    Ignores caller/current registry rates so a later registry add / change /
    remove cannot rewrite completed-job dollars.
    """
    artifacts = list(artifacts)
    job_cost = price_job_from_artifacts(artifacts)
    return _report_envelope(
        job_id,
        artifacts,
        job_cost,
        pricing_source=PRICING_SOURCE_LEGACY,
        pricing_note=(
            "Stable artifact-only economics for a completed job with no "
            "completion receipt; ignores current registry rates."
        ),
        actual=_actual_cost_payload(
            job_cost, cost_basis="measured_usage_x_artifact_price"
        ),
        counterfactual=None,
    )


def maybe_stamp_terminal_cost_receipt(store: Any, job: Job) -> Job:
    """Best-effort freeze of current-registry economics onto a cost-final job.

    Does not stamp when there is no selected usage yet, so a status write
    before artifacts flush cannot freeze numeric zero. A valid existing
    receipt is first-writer-wins. Failure must never block the status write.
    """
    if valid_terminal_cost_receipt(job.cost_receipt, job.id):
        return job
    if not is_cost_final_job_status(job.status):
        return job
    try:
        artifacts = store.list_artifacts(job.id)
        if not _usage_records(artifacts):
            return job
        receipt = copy.deepcopy(build_current_registry_cost_report(job.id, artifacts))
        receipt["pricing_source"] = PRICING_SOURCE_TERMINAL
        receipt["pricing_note"] = (
            "Frozen at job completion against the registry then in force."
        )
        return replace(job, cost_receipt=receipt)
    except Exception:
        return job


def build_cost_report(store: Any, job_id: str, registry: Optional[list] = None) -> dict:
    """Structured report behind ``puppetmaster cost <job_id> --json``.

    Shared by the CLI, the MCP ``puppetmaster_job_cost`` tool, and Marionette.
    A valid cost-final ``Job.cost_receipt`` is returned detached (no recompute,
    no write). Pre-upgrade final jobs without one get a labeled artifact-only
    report. Stalled and active jobs get a labeled current-registry reprice.
    """
    job = None
    if store is not None:
        try:
            job = store.get_job(job_id)
        except Exception:
            job = None
    receipt = job.cost_receipt if job is not None else None
    if (
        job is not None
        and is_cost_final_job_status(job.status)
        and valid_terminal_cost_receipt(receipt, job_id)
    ):
        return copy.deepcopy(receipt)

    artifacts = store.list_artifacts(job_id) if store is not None else []
    if job is not None and is_cost_final_job_status(job.status):
        return build_legacy_artifact_cost_report(job_id, artifacts)
    return build_current_registry_cost_report(job_id, artifacts, registry)
