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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.savings import (
    Counterfactual,
    resolve_counterfactual_model,
)


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


def _routing_model_ids(artifacts: Iterable[Artifact]) -> dict:
    """task_id -> registry model id from the router's initial decision."""
    out: dict = {}
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        task_id = getattr(artifact, "task_id", None)
        if not task_id or task_id in out:
            continue
        model_id = (artifact.payload or {}).get("model_id")
        if model_id:
            out[task_id] = str(model_id)
    return out


def _usage_records(artifacts: Iterable[Artifact]) -> dict:
    """task_id -> the first usage-bearing artifact's token record + model.

    Mirrors :func:`puppetmaster.usage.aggregate_token_usage` dedup: a task
    contributes its first artifact carrying ``tokens_in``/``tokens_out``.
    """
    records: dict = {}
    for artifact in artifacts:
        payload = getattr(artifact, "payload", None) or {}
        if "tokens_in" not in payload and "tokens_out" not in payload:
            continue
        task_id = getattr(artifact, "task_id", None)
        if not task_id or task_id in records:
            continue
        records[task_id] = {
            "tokens_in": int(payload.get("tokens_in") or 0),
            "tokens_out": int(payload.get("tokens_out") or 0),
            "tokens_estimated": bool(payload.get("tokens_estimated")),
            "model": payload.get("model"),
        }
    return records


def _resolve_spec(
    routing_model_id: Optional[str],
    recorded_model: Optional[str],
    by_id: dict,
    by_adapter_name: dict,
):
    """Pick the registry spec a task ran on: router decision first, then the
    model the adapter recorded (matched by id, then by adapter_model_name)."""
    if routing_model_id and routing_model_id in by_id:
        return by_id[routing_model_id]
    if recorded_model:
        if recorded_model in by_id:
            return by_id[recorded_model]
        if recorded_model in by_adapter_name:
            return by_adapter_name[recorded_model]
    return None


def price_job(artifacts: Iterable[Artifact], registry: list) -> JobCost:
    """Price a job's measured/estimated token usage against the registry price
    of the model each task actually ran on. Independent of routing."""
    artifacts = list(artifacts)
    by_id, by_adapter_name = _model_index(registry)
    routing_models = _routing_model_ids(artifacts)
    usage = _usage_records(artifacts)

    result = JobCost()
    for task_id, record in usage.items():
        spec = _resolve_spec(
            routing_models.get(task_id), record["model"], by_id, by_adapter_name
        )
        tokens_in = record["tokens_in"]
        tokens_out = record["tokens_out"]
        estimated = record["tokens_estimated"]
        if spec is not None:
            model_id = spec.id
            billing = spec.billing
            cost = spec.marginal_cost_usd(tokens_in, tokens_out)
            result.priced_tasks += 1
        else:
            model_id = routing_models.get(task_id) or record["model"] or "<unknown>"
            billing = "unknown"
            cost = 0.0
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
            {"calls": 0, "tokens_in": 0, "tokens_out": 0, "marginal_cost_usd": 0.0, "billing": billing},
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
                priced=spec is not None,
            )
        )

    result.total_marginal_cost_usd = round(result.total_marginal_cost_usd, 6)
    result.measured_cost_usd = round(result.measured_cost_usd, 6)
    result.estimated_cost_usd = round(result.estimated_cost_usd, 6)
    for bucket in result.by_model.values():
        bucket["marginal_cost_usd"] = round(bucket["marginal_cost_usd"], 6)
    return result


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
