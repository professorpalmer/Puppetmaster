"""Discover the model catalog the authenticated Cursor plan exposes.

Cursor is the one platform that lets us enumerate exactly which models the
user's plan can run (`Cursor.models.list()` via the SDK). Puppetmaster uses
that to:

* **Tag billing truthfully** — anything in the catalog is plan-billed (no
  marginal API spend), so the router can keep work inside the subscription.
* **Validate before dispatch** — a routed model id that isn't in the catalog
  is caught at preflight instead of failing mid-run.
* **Stop hand-maintaining availability** — the static registry becomes a
  capability/price *overlay* matched by id, not the source of truth for what
  exists.

The node-side enumeration lives in ``cursor_sdk_runner.mjs`` (``mode:
"list-models"``). This module shells out to it (or to an injected ``run`` for
tests) and reconciles the result with the registry.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec

CURSOR_RUNNER = Path(__file__).with_name("cursor_sdk_runner.mjs")

# (returncode, stdout, stderr) given (command, env).
CatalogRunner = Callable[[list[str], Mapping[str, str]], "tuple[int, str, str]"]

# Capability assigned to a discovered model with no registry overlay. Mid-tier
# and conservative: the user should tune it, but it won't masquerade as
# frontier or get starved as trivial in the meantime.
_DEFAULT_DISCOVERED_CAPABILITY = 60


class CursorDiscoveryError(RuntimeError):
    """Raised when the Cursor catalog cannot be enumerated."""


def _default_runner(command: list[str], env: Mapping[str, str]) -> "tuple[int, str, str]":
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(env),
        )
    except FileNotFoundError:
        return (127, "", "node not found")
    except subprocess.TimeoutExpired:
        return (124, "", "cursor catalog discovery timed out")
    return (completed.returncode, completed.stdout or "", completed.stderr or "")


def fetch_cursor_catalog(
    *,
    env: Optional[Mapping[str, str]] = None,
    run: Optional[CatalogRunner] = None,
    runner_path: Optional[Path] = None,
    node_command: str = "node",
) -> list[dict]:
    """Return the Cursor plan's model catalog as ``[{id, displayName, description}]``.

    Raises :class:`CursorDiscoveryError` on any failure (missing key, node not
    found, SDK error, malformed output) — callers fall back to the static
    registry rather than crashing the run.
    """
    base_env = dict(env if env is not None else os.environ)
    if not base_env.get("CURSOR_API_KEY"):
        raise CursorDiscoveryError(
            "CURSOR_API_KEY is not set — cannot enumerate the Cursor plan catalog."
        )
    base_env["PUPPETMASTER_CURSOR_INPUT"] = json.dumps({"mode": "list-models"})
    runner = runner_path or CURSOR_RUNNER
    runner_fn = run or _default_runner
    returncode, stdout, stderr = runner_fn([node_command, str(runner)], base_env)
    if returncode != 0:
        raise CursorDiscoveryError(
            f"cursor catalog discovery failed (rc={returncode}): {stderr.strip() or stdout.strip()}"
        )
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CursorDiscoveryError(f"cursor catalog returned non-JSON output: {exc}") from exc
    if not payload.get("ok"):
        raise CursorDiscoveryError("cursor catalog discovery returned ok=false")
    models = payload.get("models") or []
    return [m for m in models if isinstance(m, dict) and m.get("id")]


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def catalog_to_specs(
    catalog: list[dict],
    existing: list[ModelSpec],
) -> list[ModelSpec]:
    """Turn a discovered catalog into plan-billed :class:`ModelSpec`s.

    Capability/price are inherited from any existing cursor registry entry whose
    ``adapter_model_name`` matches the discovered id (the registry is a tuning
    overlay); unmatched models get a conservative mid-tier seed the user can
    edit. Every returned spec is ``billing="plan"`` — by definition it's in the
    plan catalog.
    """
    by_model_name = {
        spec.adapter_model_name: spec
        for spec in existing
        if spec.adapter == "cursor"
    }
    # Cross-adapter capability inheritance: a model exposed by the Cursor plan
    # (e.g. "claude-opus-4-8") usually has a known capability from its native
    # adapter entry (claude-code/opus-4-8). Inherit the highest known
    # capability for the same model name so a plan-billed frontier model is
    # ranked like the frontier it is — not stuck at the conservative seed.
    cap_by_name: dict[str, ModelSpec] = {}
    for spec in existing:
        current = cap_by_name.get(spec.adapter_model_name)
        if current is None or spec.capability_score > current.capability_score:
            cap_by_name[spec.adapter_model_name] = spec

    specs: list[ModelSpec] = []
    for item in catalog:
        model_id = str(item["id"])
        overlay = by_model_name.get(model_id)
        if overlay is not None:
            specs.append(
                ModelSpec(
                    id=overlay.id,
                    adapter="cursor",
                    adapter_model_name=model_id,
                    capability_score=overlay.capability_score,
                    input_per_mtok_usd=overlay.input_per_mtok_usd,
                    output_per_mtok_usd=overlay.output_per_mtok_usd,
                    context_window=overlay.context_window,
                    billing="plan",
                    tags=sorted(set(overlay.tags) | {"discovered"}),
                    notes=overlay.notes,
                    enabled=overlay.enabled,
                )
            )
        else:
            display = item.get("displayName") or model_id
            kin = cap_by_name.get(model_id)
            if kin is not None:
                capability = kin.capability_score
                context_window = kin.context_window
                inherited_tags = sorted(
                    {t for t in kin.tags if t not in {"cursor"}} | {"cursor", "discovered"}
                )
                note = (
                    f"Discovered from the Cursor plan catalog ({display}); capability "
                    f"inherited from {kin.id}. Plan-billed (no marginal API spend)."
                )
            else:
                capability = _DEFAULT_DISCOVERED_CAPABILITY
                context_window = 0
                inherited_tags = ["cursor", "discovered"]
                note = (
                    f"Discovered from the Cursor plan catalog ({display}). "
                    "Plan-billed (no marginal API spend). Tune capability_score "
                    "to rank it correctly against your other models."
                )
            specs.append(
                ModelSpec(
                    id=f"cursor/{_slug(model_id)}",
                    adapter="cursor",
                    adapter_model_name=model_id,
                    capability_score=capability,
                    input_per_mtok_usd=0.0,
                    output_per_mtok_usd=0.0,
                    context_window=context_window,
                    billing="plan",
                    tags=inherited_tags,
                    notes=note,
                )
            )
    return specs


def merge_catalog_into_registry(
    existing: list[ModelSpec],
    catalog: list[dict],
) -> "tuple[list[ModelSpec], dict]":
    """Reconcile a discovered catalog with ``existing`` registry entries.

    Cursor entries are replaced by their discovered, plan-billed counterparts;
    cursor entries that are NOT in the live catalog are dropped (the plan no
    longer exposes them); non-cursor entries pass through untouched. Returns the
    merged registry plus a small report describing what changed.
    """
    discovered = catalog_to_specs(catalog, existing)
    discovered_model_names = {s.adapter_model_name for s in discovered}

    non_cursor = [s for s in existing if s.adapter != "cursor"]
    existing_cursor_names = {
        s.adapter_model_name for s in existing if s.adapter == "cursor"
    }
    dropped = sorted(existing_cursor_names - discovered_model_names)
    added = sorted(discovered_model_names - existing_cursor_names)

    merged = non_cursor + discovered
    report = {
        "discovered_count": len(discovered),
        "added": added,
        "dropped_stale_cursor_models": dropped,
        "non_cursor_preserved": len(non_cursor),
    }
    return merged, report


def model_in_catalog(model_name: str, catalog: list[dict]) -> bool:
    """True if ``model_name`` (an adapter model id) is in the live catalog.

    Used by preflight to reject a routed cursor model that the plan no longer
    exposes — turning a mid-run "invalid model" into a clean preflight block.
    """
    return any(str(item.get("id")) == model_name for item in catalog)
