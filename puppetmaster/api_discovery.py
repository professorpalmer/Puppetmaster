"""Discover model catalogs from the API-billed platforms that expose one.

Cursor is special-cased in :mod:`puppetmaster.cursor_discovery` (plan-billed,
SDK enumeration). The other two platforms Puppetmaster routes to *also* expose
a model-list endpoint, so "can't enumerate non-Cursor catalogs" is only half
true:

* **OpenAI** — ``GET /v1/models`` (Bearer ``OPENAI_API_KEY``). Backs both the
  ``openai`` adapter and the ``codex`` CLI adapter.
* **Anthropic** — ``GET /v1/models`` (``x-api-key`` + ``anthropic-version``).
  Backs the ``claude-code`` adapter's underlying models.

Both are API-key authenticated, so enumerating them tells us *what exists*, not
how the adapter is billed at runtime (claude-code is usually OAuth/plan even
when an Anthropic key is present for discovery). Discovered entries therefore
inherit capability/price from any matching registry entry and otherwise get a
conservative, clearly-noted seed the user tunes. Every network dependency is
injectable so the whole thing is unit-testable without keys or network.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec, normalize_model_token
from puppetmaster.openai_security import DEFAULT_OPENAI_BASE_URL, validate_openai_base_url

# Injectable HTTP getter: (url, headers) -> (status, body_text). Defaults to a
# real urllib GET; tests pass a stub.
HttpGetter = Callable[[str, Mapping[str, str]], "tuple[int, str]"]

_DEFAULT_DISCOVERED_CAPABILITY = 60

_ANTHROPIC_VERSION = "2023-06-01"


class ApiDiscoveryError(RuntimeError):
    """Raised when a platform model catalog cannot be enumerated."""


def _default_getter(url: str, headers: Mapping[str, str]) -> "tuple[int, str]":
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return (response.status, response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return (exc.code, body)
    except Exception as exc:  # pragma: no cover - network failure
        raise ApiDiscoveryError(str(exc)) from exc


def fetch_openai_models(
    *,
    env: Optional[Mapping[str, str]] = None,
    getter: Optional[HttpGetter] = None,
) -> list[dict]:
    """Return OpenAI's model catalog as ``[{id, displayName}]``."""
    env = env if env is not None else os.environ
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        raise ApiDiscoveryError("OPENAI_API_KEY is not set — cannot enumerate OpenAI models.")
    base = env.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")
    base_url_error = validate_openai_base_url(base)
    if base_url_error is not None:
        raise ApiDiscoveryError(base_url_error)
    get = getter or _default_getter
    status, body = get(f"{base}/models", {"Authorization": f"Bearer {api_key}"})
    return _parse_models_response("openai", status, body)


def fetch_anthropic_models(
    *,
    env: Optional[Mapping[str, str]] = None,
    getter: Optional[HttpGetter] = None,
) -> list[dict]:
    """Return Anthropic's model catalog as ``[{id, displayName}]``."""
    env = env if env is not None else os.environ
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ApiDiscoveryError(
            "ANTHROPIC_API_KEY is not set — cannot enumerate Anthropic models. "
            "(Claude Code itself may still run via OAuth; this is for discovery only.)"
        )
    base = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    get = getter or _default_getter
    status, body = get(
        f"{base}/models",
        {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION},
    )
    return _parse_models_response("anthropic", status, body)


def _parse_models_response(platform: str, status: int, body: str) -> list[dict]:
    if status != 200:
        raise ApiDiscoveryError(f"{platform} models endpoint returned HTTP {status}: {body[:200]}")
    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise ApiDiscoveryError(f"{platform} models endpoint returned non-JSON: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ApiDiscoveryError(f"{platform} models endpoint had no 'data' array")
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        out.append(
            {
                "id": str(item["id"]),
                "displayName": item.get("display_name") or item.get("id"),
            }
        )
    return out


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def catalog_to_specs(
    adapter: str,
    billing: str,
    catalog: list[dict],
    existing: list[ModelSpec],
) -> list[ModelSpec]:
    """Turn a discovered API catalog into :class:`ModelSpec`s for ``adapter``.

    Capability/price/context are inherited from an existing registry entry for
    the same adapter+model (the registry is a tuning overlay), then from any
    same-named entry on a sibling adapter, and finally a conservative seed the
    user edits. ``billing`` is the caller-provided posture for the adapter."""
    same_adapter = {
        spec.adapter_model_name: spec for spec in existing if spec.adapter == adapter
    }
    same_adapter_normalized = {
        normalize_model_token(spec.adapter_model_name): spec
        for spec in existing
        if spec.adapter == adapter and normalize_model_token(spec.adapter_model_name)
    }
    cap_by_name: dict[str, ModelSpec] = {}
    for spec in existing:
        current = cap_by_name.get(spec.adapter_model_name)
        if current is None or spec.capability_score > current.capability_score:
            cap_by_name[spec.adapter_model_name] = spec

    specs: list[ModelSpec] = []
    for item in catalog:
        model_id = str(item["id"])
        overlay = same_adapter.get(model_id) or same_adapter_normalized.get(
            normalize_model_token(model_id)
        )
        if overlay is not None:
            specs.append(
                ModelSpec(
                    id=overlay.id,
                    adapter=adapter,
                    adapter_model_name=model_id,
                    capability_score=overlay.capability_score,
                    input_per_mtok_usd=overlay.input_per_mtok_usd,
                    output_per_mtok_usd=overlay.output_per_mtok_usd,
                    context_window=overlay.context_window,
                    billing=overlay.billing if overlay.billing != "unknown" else billing,
                    tags=sorted(set(overlay.tags) | {"discovered"}),
                    notes=overlay.notes,
                    enabled=overlay.enabled,
                )
            )
            continue
        display = item.get("displayName") or model_id
        kin = cap_by_name.get(model_id)
        if kin is not None:
            capability = kin.capability_score
            context_window = kin.context_window
            tags = sorted({t for t in kin.tags if t != adapter} | {adapter, "discovered"})
            note = (
                f"Discovered from the {adapter} catalog ({display}); capability "
                f"inherited from {kin.id}. Set pricing to route on cost."
            )
        else:
            capability = _DEFAULT_DISCOVERED_CAPABILITY
            context_window = 0
            tags = [adapter, "discovered"]
            note = (
                f"Discovered from the {adapter} catalog ({display}). Tune "
                "capability_score and pricing so the router ranks/charges it correctly."
            )
        specs.append(
            ModelSpec(
                id=f"{adapter}/{_slug(model_id)}",
                adapter=adapter,
                adapter_model_name=model_id,
                capability_score=capability,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                context_window=context_window,
                billing=billing,
                tags=tags,
                notes=note,
            )
        )
    return specs


def merge_api_catalog_into_registry(
    adapter: str,
    billing: str,
    existing: list[ModelSpec],
    catalog: list[dict],
) -> "tuple[list[ModelSpec], dict]":
    """Reconcile a discovered API catalog with ``existing`` registry entries.

    Unlike Cursor discovery, API catalogs are large and noisy (every legacy
    model, embeddings, TTS, …), so we do NOT drop registry entries that aren't
    in the catalog — we only *add* newly-discovered ones and refresh overlays.
    Non-matching adapters and hand-tuned entries are preserved untouched."""
    discovered = catalog_to_specs(adapter, billing, catalog, existing)
    discovered_by_name = {s.adapter_model_name: s for s in discovered}

    existing_names = {
        s.adapter_model_name for s in existing if s.adapter == adapter
    }
    added = sorted(discovered_by_name.keys() - existing_names)

    merged: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in existing:
        if spec.adapter == adapter and spec.adapter_model_name in discovered_by_name:
            merged.append(discovered_by_name[spec.adapter_model_name])
            seen.add(spec.adapter_model_name)
        else:
            merged.append(spec)
    for name, spec in discovered_by_name.items():
        if name not in seen:
            merged.append(spec)

    report = {
        "adapter": adapter,
        "discovered_count": len(discovered),
        "added": added,
        "refreshed": sorted(seen),
        "preserved": len([s for s in existing if s.adapter != adapter]),
    }
    return merged, report
