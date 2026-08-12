"""Truthful per-artifact execution provenance.

Stamped onto FINDING / RISK / DECISION / VERIFICATION / GIST artifacts at
the central store save seam so every adapter path is covered. Fields are
optional and additive: unknown stays null/absent with explicit
``usage_known`` / ``cost_known`` / ``priced`` flags — never a fabricated
numeric zero for missing data.

ROUTING / PATCH / GATE / MEMORY_SUMMARY artifacts are left untouched.
Aggregate rollups continue to read top-level token fields on VERIFICATION
payloads; this nested block is an honest per-artifact audit lane.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from puppetmaster.models import Artifact, ArtifactType

_PROVENANCE_TYPES = frozenset(
    {
        ArtifactType.FINDING,
        ArtifactType.RISK,
        ArtifactType.DECISION,
        ArtifactType.VERIFICATION,
        ArtifactType.GIST,
    }
)

PROVENANCE_KEY = "execution_provenance"

# Higher rank wins on merge — never downgrade a more authoritative stamp.
_COST_SOURCE_RANK = {
    "unknown": 0,
    "unpriced": 1,
    "plan": 2,
    "reported": 3,
    "real_cost_usd": 4,
}


def _nonempty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_nonneg_int(value: Any) -> Optional[int]:
    """Return an int only when the caller supplied a real numeric value.

    Booleans are rejected. Missing / unparseable values stay ``None`` — never
    coerced to ``0``.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _coerce_cost(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def build_execution_provenance(
    *,
    adapter: Optional[str] = None,
    router_model_id: Optional[str] = None,
    adapter_model_name: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    tokens_estimated: Optional[bool] = None,
    usage_known: Optional[bool] = None,
    cost_usd: Optional[float] = None,
    cost_known: Optional[bool] = None,
    priced: Optional[bool] = None,
    cost_source: Optional[str] = None,
) -> dict[str, Any]:
    """Build an honest provenance block. Omit numeric token/cost when unknown."""
    tin = _coerce_nonneg_int(tokens_in)
    tout = _coerce_nonneg_int(tokens_out)
    if usage_known is None:
        usage_known = tin is not None or tout is not None

    cost = _coerce_cost(cost_usd)
    if cost_known is None:
        cost_known = cost is not None
    if priced is None:
        priced = bool(cost_known)
    if not cost_source:
        cost_source = "unknown" if not cost_known else "reported"

    block: dict[str, Any] = {
        "adapter": _nonempty_str(adapter),
        "usage_known": bool(usage_known),
        "cost_known": bool(cost_known),
        "priced": bool(priced),
        "cost_source": str(cost_source),
    }
    # Identity fields: include when known; leave absent when unknown (never "").
    router_id = _nonempty_str(router_model_id)
    if router_id is not None:
        block["router_model_id"] = router_id
    adapter_name = _nonempty_str(adapter_model_name)
    if adapter_name is not None:
        block["adapter_model_name"] = adapter_name
    if tin is not None:
        block["tokens_in"] = tin
    if tout is not None:
        block["tokens_out"] = tout
    # Missing tokens_estimated stays absent (unknown) even when token counts exist.
    # Never invent measured=False.
    if tokens_estimated is not None:
        block["tokens_estimated"] = bool(tokens_estimated)
    if cost is not None and cost_known:
        block["cost_usd"] = cost
    return block


def _adapter_from_evidence(evidence: list[str]) -> Optional[str]:
    for item in evidence or []:
        if isinstance(item, str) and item.startswith("adapter:"):
            return item.split(":", 1)[1].strip() or None
    return None


def provenance_from_task_and_payload(
    task: Any,
    payload: dict[str, Any],
    *,
    evidence: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Derive provenance from a task envelope + existing artifact payload.

    Never invents token/cost zeros. Plan-billed $0 is only recorded when the
    registry billing is known to be ``plan`` (via task payload). Bare
    ``cost_usd=0`` under metered/unknown billing is not a known real cost —
    require ``real_cost_usd`` or an explicit ``cost_known`` / ``cost_source``.
    """
    task_payload = getattr(task, "payload", None) or {}
    if not isinstance(task_payload, dict):
        task_payload = {}
    if not isinstance(payload, dict):
        payload = {}

    adapter = (
        _nonempty_str(payload.get("adapter"))
        or _nonempty_str(getattr(task, "adapter", None))
        or _adapter_from_evidence(evidence or [])
    )
    router_model_id = (
        _nonempty_str(payload.get("router_model_id"))
        or _nonempty_str(task_payload.get("router_model_id"))
        or _nonempty_str(task_payload.get("pinned_model"))
    )
    adapter_model_name = (
        _nonempty_str(payload.get("adapter_model_name"))
        or _nonempty_str(payload.get("model"))
        or _nonempty_str(task_payload.get("adapter_model_name"))
        or _nonempty_str(task_payload.get("pinned_adapter_model_name"))
        or _nonempty_str(task_payload.get("model"))
    )
    # "default" is a Cursor placeholder, not a truthful provider model name.
    if adapter_model_name == "default":
        adapter_model_name = None

    tin = _coerce_nonneg_int(payload.get("tokens_in"))
    tout = _coerce_nonneg_int(payload.get("tokens_out"))
    usage_flag = payload.get("tokens_estimated")
    tokens_estimated: Optional[bool]
    if isinstance(usage_flag, bool):
        tokens_estimated = usage_flag
    else:
        tokens_estimated = None
    usage_known = tin is not None or tout is not None

    real_cost = _coerce_cost(payload.get("real_cost_usd"))
    reported_cost = _coerce_cost(payload.get("cost_usd"))
    billing = _nonempty_str(task_payload.get("billing")) or _nonempty_str(
        payload.get("billing")
    )
    explicit_cost_known = payload.get("cost_known")
    explicit_cost_source = _nonempty_str(payload.get("cost_source"))

    cost_usd: Optional[float] = None
    cost_known = False
    priced = False
    cost_source = "unknown"
    if real_cost is not None:
        cost_usd = real_cost
        cost_known = True
        priced = True
        cost_source = "real_cost_usd"
    elif explicit_cost_known is True and reported_cost is not None:
        cost_usd = reported_cost
        cost_known = True
        priced = True
        cost_source = explicit_cost_source or "reported"
    elif (
        explicit_cost_source
        and explicit_cost_source not in {"unknown", "unpriced"}
        and reported_cost is not None
    ):
        cost_usd = reported_cost
        cost_known = True
        priced = True
        cost_source = explicit_cost_source
    elif billing == "plan" and router_model_id:
        # Plan-billed marginal cost is known $0 only when we know which model ran.
        cost_usd = 0.0
        cost_known = True
        priced = True
        cost_source = "plan"
    elif billing == "metered" and router_model_id and usage_known:
        # Metered but not yet priced — bare cost_usd=0 is not known real cost.
        cost_known = False
        priced = False
        cost_source = "unpriced"
    elif reported_cost is not None and reported_cost != 0.0:
        # Non-zero reported spend without real_cost_usd still counts as reported.
        cost_usd = reported_cost
        cost_known = True
        priced = True
        cost_source = "reported"
    # else: bare zero / missing under unknown billing stays unknown

    return build_execution_provenance(
        adapter=adapter,
        router_model_id=router_model_id,
        adapter_model_name=adapter_model_name,
        tokens_in=tin,
        tokens_out=tout,
        tokens_estimated=tokens_estimated,
        usage_known=usage_known,
        cost_usd=cost_usd,
        cost_known=cost_known,
        priced=priced,
        cost_source=cost_source,
    )


def _merge_provenance(existing: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    """Prefer already-known fields; upgrade unknown/plan when better data arrives."""
    base = dict(existing) if isinstance(existing, dict) else {}
    merged = dict(base)
    incoming_source = str(incoming.get("cost_source") or "unknown")
    existing_source = str(merged.get("cost_source") or "unknown")
    source_upgraded = (
        _COST_SOURCE_RANK.get(incoming_source, 0)
        > _COST_SOURCE_RANK.get(existing_source, 0)
    )

    for key, value in incoming.items():
        if key in {"tokens_in", "tokens_out"}:
            if value is None:
                continue
            if merged.get(key) is None:
                merged[key] = value
            continue
        if key == "cost_usd":
            if value is None:
                continue
            if merged.get(key) is None or source_upgraded:
                merged[key] = value
            continue
        if key in {"usage_known", "cost_known", "priced"}:
            # Allow upgrade of cost_known/priced when source authority increases.
            if key in {"cost_known", "priced"} and source_upgraded and value:
                merged[key] = True
            else:
                merged[key] = bool(merged.get(key)) or bool(value)
            continue
        if key == "cost_source":
            if source_upgraded and value:
                merged[key] = value
            elif merged.get(key) in (None, "", "unknown") and value:
                merged[key] = value
            continue
        if key == "tokens_estimated":
            if value is not None and "tokens_estimated" not in merged:
                merged[key] = value
            continue
        if key == "provenance_error":
            continue
        if merged.get(key) in (None, "") and value not in (None, ""):
            merged[key] = value

    has_usage = (
        merged.get("tokens_in") is not None or merged.get("tokens_out") is not None
    )
    merged["usage_known"] = bool(merged.get("usage_known")) or has_usage

    source = str(merged.get("cost_source") or "unknown")
    has_cost = merged.get("cost_usd") is not None
    if source in {"plan", "reported", "real_cost_usd"} and has_cost:
        merged["cost_known"] = True
        merged["priced"] = True
        merged.pop("provenance_error", None)
    elif source == "unpriced":
        merged["cost_known"] = False
        merged["priced"] = False
        merged.pop("cost_usd", None)
    elif has_cost and bool(merged.get("cost_known")):
        merged["priced"] = bool(merged.get("priced", True))
        if source in (None, "", "unknown"):
            merged["cost_source"] = "reported"
    else:
        merged["cost_known"] = False
        merged["priced"] = False
        merged["cost_source"] = source if source in _COST_SOURCE_RANK else "unknown"
        if not has_cost:
            merged.pop("cost_usd", None)

    if merged.get("cost_source") == "plan" and merged.get("cost_known"):
        if merged.get("cost_usd") is None:
            merged["cost_usd"] = 0.0
        merged["priced"] = True
    return merged


def stamp_execution_provenance(
    artifact: Artifact,
    *,
    task: Any = None,
) -> Artifact:
    """Attach/merge ``payload.execution_provenance`` for typed peer artifacts."""
    if artifact.type not in _PROVENANCE_TYPES:
        return artifact
    payload = dict(artifact.payload or {})
    incoming = provenance_from_task_and_payload(
        task, payload, evidence=list(artifact.evidence or [])
    )
    payload[PROVENANCE_KEY] = _merge_provenance(
        payload.get(PROVENANCE_KEY), incoming
    )
    return replace(artifact, payload=payload, sha256=None)


def unknown_provenance_error(exc: BaseException) -> dict[str, Any]:
    """Fail-closed provenance stamp for hot paths that must not raise."""
    return {
        "adapter": None,
        "usage_known": False,
        "cost_known": False,
        "priced": False,
        "cost_source": "unknown",
        "provenance_error": type(exc).__name__,
    }


def stamp_execution_provenance_for_store(artifact: Artifact, store: Any) -> Artifact:
    """Store-seam helper: resolve the producing task when possible, then stamp."""
    if artifact.type not in _PROVENANCE_TYPES:
        return artifact
    task = None
    task_id = getattr(artifact, "task_id", None)
    if task_id and store is not None:
        getter = getattr(store, "get_task_by_id", None)
        if callable(getter):
            try:
                task = getter(task_id)
            except Exception:
                task = None
    return stamp_execution_provenance(artifact, task=task)
