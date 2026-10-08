"""Durable admission contract. No dispatch instrumentation or billing inference.

Reconciliation supplies cumulative invocation totals, never observation deltas.
The caller owns finality/provenance; telemetry is not silently spend authority.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Mapping, Optional


def _number(value, name, *, integer=False):
    if value is None:
        return
    if (type(value) not in ((int,) if integer else (int, float)) or
            not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BudgetPolicy:
    max_usd: Optional[float] = None
    max_tokens_in: Optional[int] = None
    max_tokens_out: Optional[int] = None
    max_attempts: Optional[int] = None
    max_elapsed_seconds: Optional[float] = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            _number(value, name, integer=name in (
                "max_tokens_in", "max_tokens_out", "max_attempts"))


@dataclass(frozen=True)
class BudgetLiability:
    billing: str = "unknown"
    cost_state: str = "unknown"
    api_usd: Optional[float] = None
    plan_marginal_usd: Optional[float] = None
    api_equivalent_usd: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    elapsed_seconds: Optional[float] = None

    def __post_init__(self):
        if self.billing not in ("api", "plan", "unknown"):
            raise ValueError("invalid billing basis")
        if self.cost_state not in ("known", "unknown", "partial"):
            raise ValueError("invalid cost state")
        for name in ("api_usd", "plan_marginal_usd", "api_equivalent_usd",
                     "tokens_in", "tokens_out", "elapsed_seconds"):
            _number(getattr(self, name), name, integer=name.startswith("tokens_"))
        if self.billing != "api" and self.api_usd is not None:
            raise ValueError("API charges require API billing")
        if self.billing != "plan" and self.plan_marginal_usd is not None:
            raise ValueError("plan marginal charges require plan billing")
        if self.billing != "plan" and self.api_equivalent_usd is not None:
            raise ValueError("API equivalent is only a separate plan estimate")
        if (self.cost_state == "unknown") != (self.marginal_usd is None):
            raise ValueError("cost state must agree with marginal liability")

    @property
    def marginal_usd(self):
        return self.api_usd if self.billing == "api" else self.plan_marginal_usd


class BudgetConflictError(ValueError):
    """An identity or terminal transition was replayed with different facts."""


class BudgetAdmissionError(ValueError):
    """A configured cap is exhausted or its liability is indeterminate."""


class BudgetBusy(RuntimeError):
    """Another writer holds the job's budget lock; trying again later can succeed."""


class BudgetUnsettled(BudgetAdmissionError):
    """Indeterminate only because an earlier attempt has not settled yet.

    It can clear without any change to the policy: when the earlier attempt
    reports, the total becomes known. A caller may wait and try again. An
    unknown allowance of the attempt itself, or a settled unknown, is a plain
    ``BudgetAdmissionError``: no wait can make that total known.
    """


def budget_totals(records):
    """Separate complete totals from known subtotals; unknown is never zero."""
    liabilities = []
    for record in records:
        if record["state"] == "released":
            continue
        pending = record["state"] == "pending_reconciliation"
        value = record["liability"] if record["liability"] is not None else record["allowance"]
        liabilities.append((BudgetLiability(**value), pending))
    result = {"attempts": len(liabilities)}
    for name in ("marginal_usd", "api_usd", "plan_marginal_usd", "api_equivalent_usd",
                 "tokens_in", "tokens_out", "elapsed_seconds"):
        values = []
        complete = True
        for liability, pending in liabilities:
            # Non-applicable billing categories contribute explicit zero.
            if ((name == "api_usd" and liability.billing == "plan") or
                    (name in ("plan_marginal_usd", "api_equivalent_usd") and
                     liability.billing == "api")):
                value = 0
            else:
                value = getattr(liability, name)
            values.append(value)
            if value is None or pending or (name in (
                    "marginal_usd", "api_usd", "plan_marginal_usd") and
                    liability.cost_state != "known"):
                complete = False
        known = [value for value in values if value is not None]
        subtotal = (float(sum((Decimal(str(value)) for value in known), Decimal(0)))
                    if name.endswith("usd") or name == "elapsed_seconds" else sum(known))
        result[name] = {"total": subtotal if complete else None,
                        "known_subtotal": subtotal,
                        "state": "known" if complete else (
                            "partial" if any(v is not None for v in values) else "unknown")}
    return result


def check_admission(policy, records):
    if policy is None:
        return
    totals = budget_totals(records)
    for cap, metric in (("max_usd", "marginal_usd"),
                        ("max_tokens_in", "tokens_in"),
                        ("max_tokens_out", "tokens_out"),
                        ("max_elapsed_seconds", "elapsed_seconds"),
                        ("max_attempts", "attempts")):
        limit = getattr(policy, cap)
        if limit is None:
            continue
        value = totals[metric] if metric == "attempts" else totals[metric]["total"]
        if value is None:
            settled = [record for record in records if record["state"] != "pending_reconciliation"]
            unsettled_only = budget_totals(settled)[metric]["total"] is not None
            if unsettled_only:
                raise BudgetUnsettled(
                    f"{cap}: indeterminate: an earlier attempt of this job has no settled {metric} "
                    "(it runs now, or it ended without a final report). Under this cap, the job "
                    "admits a new attempt only after each earlier attempt settles")
            raise BudgetAdmissionError(
                f"{cap}: indeterminate: an attempt of this job has no bounded {metric} (no "
                "allowance, or it settled without a report), so the total can never be known")
        if value > limit:
            raise BudgetAdmissionError(
                f"{cap}: exhausted: this attempt brings the total to {value:g}, above the cap of {limit:g}")


BUDGET_FIELDS = {
    "max_usd": float,
    "max_tokens_in": int,
    "max_tokens_out": int,
    "max_attempts": int,
    "max_elapsed_seconds": float,
}


_PUBLIC_CAP_NAMES = {
    "max_usd": "budget_max_usd",
    "max_tokens_in": "budget_max_tokens_in",
    "max_tokens_out": "budget_max_tokens_out",
    "max_elapsed_seconds": "budget_max_elapsed_seconds",
}

_CAP_REMEDIATION = {
    "budget_max_usd": (
        "set payload.billing=plan or payload.budget_allowance "
        "api_usd/plan_marginal_usd"
    ),
    "budget_max_tokens_in": "set payload.budget_allowance.tokens_in",
    "budget_max_tokens_out": (
        "set payload.max_output_tokens or payload.budget_allowance.tokens_out"
    ),
    "budget_max_elapsed_seconds": (
        "set --timeout-seconds or payload.timeout_seconds"
    ),
}


def stamp_payload_budget_allowance(policy, payload, *, adapter):
    """Derive a conservative per-invocation allowance or refuse the launch.

    Public job caps are fail-closed. Unknown is never treated as zero. Elapsed
    and output-token bounds come only from an enforceable timeout or provider
    output limit. Attempt caps need no allowance.
    """
    payload = dict(payload or {})
    if policy is None:
        return payload
    values = dict(payload.get("budget_allowance") or {})
    values.setdefault("billing", values.get("billing") or payload.get("billing") or "unknown")
    if values["billing"] == "plan":
        values.setdefault("plan_marginal_usd", 0)
        values.setdefault("cost_state", "known")
    if values.get("elapsed_seconds") is None:
        timeout = payload.get("timeout_seconds")
        if timeout is None:
            timeout = payload.get("max_timeout_seconds")
        if timeout is not None:
            bound = float(timeout)
            if policy.max_elapsed_seconds is not None:
                bound = min(bound, float(policy.max_elapsed_seconds))
            values["elapsed_seconds"] = bound
    if values.get("tokens_out") is None:
        for key in ("max_output_tokens", "max_tokens"):
            raw = payload.get(key)
            if raw is not None:
                bound = int(raw)
                if policy.max_tokens_out is not None:
                    bound = min(bound, int(policy.max_tokens_out))
                values["tokens_out"] = bound
                break
    try:
        allowance = BudgetLiability(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "payload.budget_allowance is invalid for adapter %s: %s" % (adapter, exc)
        ) from exc
    missing = []
    if policy.max_usd is not None and allowance.marginal_usd is None:
        missing.append("budget_max_usd")
    if policy.max_tokens_in is not None and allowance.tokens_in is None:
        missing.append("budget_max_tokens_in")
    if policy.max_tokens_out is not None and allowance.tokens_out is None:
        missing.append("budget_max_tokens_out")
    if policy.max_elapsed_seconds is not None and allowance.elapsed_seconds is None:
        missing.append("budget_max_elapsed_seconds")
    if missing:
        named = ", ".join(missing)
        tips = "; ".join(_CAP_REMEDIATION[name] for name in missing)
        raise ValueError(
            "%s unsupported for adapter %s: no bounded per-invocation "
            "allowance. %s. Or use budget_max_attempts." % (named, adapter, tips)
        )
    if any(getattr(policy, field) is not None for field in _PUBLIC_CAP_NAMES):
        payload["budget_allowance"] = {
            key: value for key, value in asdict(allowance).items() if value is not None
        }
        if allowance.billing != "unknown":
            payload.setdefault("billing", allowance.billing)
    return payload


def budget_policy_from_inputs(values: Mapping[str, object]) -> Optional[BudgetPolicy]:
    """Validate public job-total inputs; omission preserves legacy launches.

    Runtime policies may use zero to represent an exhausted budget. Public
    launches require positive limits so an accidental zero cannot start a job.
    """
    supplied = {}
    for field, kind in BUDGET_FIELDS.items():
        name = "budget_" + field
        value = values.get(name)
        if value is None:
            continue
        expected = (int,) if kind is int else (int, float)
        if type(value) not in expected:
            raise ValueError(f"{name} must be a positive finite {kind.__name__}")
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a positive finite {kind.__name__}")
        supplied[field] = value
    return BudgetPolicy(**supplied) if supplied else None


def budget_cli_flags(policy: Optional[BudgetPolicy]) -> list[str]:
    """Serialize a validated policy for detached CLI launches."""
    if policy is None:
        return []
    return [part for field, value in asdict(policy).items() if value is not None
            for part in ("--budget-" + field.replace("_", "-"), str(value))]


# A pilot read the elapsed cap as a wall deadline and set 240 s for four
# parallel 180 s workers. Say what the cap adds up.
_SCHEMA_NOTES = {
    "max_elapsed_seconds": (
        " It adds up worker-seconds over all attempts, not wall time: four "
        "parallel 180 s workers need 720. Each attempt reserves its "
        "timeout_seconds. While an earlier attempt has no settled elapsed time, "
        "a new attempt waits for it (payload.budget_admission_wait_seconds, "
        "default 900), so under this cap the job runs its attempts one at a "
        "time. An exhausted cap fails at once."
    ),
}


def budget_schema_properties() -> dict:
    return {
        "budget_" + field: {
            "type": "integer" if kind is int else "number",
            "exclusiveMinimum": 0,
            "description": ("Cumulative job-total " + field +
                            "; independent of per-call routing max_cost_usd."
                            + _SCHEMA_NOTES.get(field, "")),
        }
        for field, kind in BUDGET_FIELDS.items()
    }
