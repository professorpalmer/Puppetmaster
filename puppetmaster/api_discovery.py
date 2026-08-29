"""Discover model catalogs from the API-billed platforms that expose one.

Cursor is special-cased in :mod:`puppetmaster.cursor_discovery` (plan-billed,
SDK enumeration). The other two platforms Puppetmaster routes to *also* expose
a model-list endpoint, so "can't enumerate non-Cursor catalogs" is only half
true:

* **OpenAI** — ``GET /v1/models`` (Bearer ``OPENAI_API_KEY``). Backs both the
  ``openai`` adapter and the ``codex`` CLI adapter.
* **Anthropic** — ``GET /v1/models`` (``x-api-key`` + ``anthropic-version``).
  Backs the ``claude-code`` adapter's underlying models.
* **OpenRouter** — ``GET /api/v1/models`` (Bearer ``OPENROUTER_API_KEY``).
  Backs the keys-only ``agentic`` adapter with ``provider=openrouter``, and is
  the one catalog that also publishes per-model price and context window, so
  discovered rows land cost-routable instead of price-zero.

All three are API-key authenticated, so enumerating them tells us *what exists*, not
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
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec, normalize_model_token
from puppetmaster.openai_security import DEFAULT_OPENAI_BASE_URL, validate_openai_base_url

# Injectable HTTP getter: (url, headers) -> (status, body_text). Defaults to a
# real urllib GET; tests pass a stub.
HttpGetter = Callable[[str, Mapping[str, str]], "tuple[int, str]"]

_DEFAULT_DISCOVERED_CAPABILITY = 60

_ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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


def fetch_openrouter_models(
    *,
    env: Optional[Mapping[str, str]] = None,
    getter: Optional[HttpGetter] = None,
    tools_only: bool = True,
) -> list[dict]:
    """Return OpenRouter's catalog with live pricing and context windows.

    A Puppetmaster worker is a tool-use loop, so ``tools_only`` keeps only
    models advertising ``tools`` in ``supported_parameters`` — one without tool
    calling can never do more than a degraded run.
    """
    env = env if env is not None else os.environ
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ApiDiscoveryError(
            "OPENROUTER_API_KEY is not set — cannot enumerate OpenRouter models."
        )
    base = env.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    get = getter or _default_getter
    status, body = get(f"{base}/models", {"Authorization": f"Bearer {api_key}"})
    if status != 200:
        raise ApiDiscoveryError(
            f"openrouter models endpoint returned HTTP {status}: {body[:200]}"
        )
    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise ApiDiscoveryError(
            f"openrouter models endpoint returned non-JSON: {exc}"
        ) from exc
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ApiDiscoveryError("openrouter models endpoint had no 'data' array")

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        if tools_only and "tools" not in (item.get("supported_parameters") or []):
            continue
        architecture = item.get("architecture") or {}
        if "text" not in (architecture.get("output_modalities") or ["text"]):
            continue
        pricing = item.get("pricing") or {}
        out.append(
            {
                "id": str(item["id"]),
                "displayName": item.get("name") or item["id"],
                "input_per_mtok_usd": _per_mtok(pricing.get("prompt")),
                "output_per_mtok_usd": _per_mtok(pricing.get("completion")),
                "context_window": _openrouter_context_window(item),
                "tags": _openrouter_tags(item),
            }
        )
    return out


def _per_mtok(price_per_token: Any) -> float:
    """OpenRouter quotes USD per token; the registry stores USD per Mtok.

    Dynamically-priced router models (``openrouter/auto``) quote ``-1``; the
    registry has no representation for "depends on where it lands", so those
    become 0.0 (unpriced) rather than a negative cost the router would treat as
    the cheapest option in the catalog.
    """
    try:
        per_mtok = round(float(price_per_token) * 1_000_000, 6)
    except (TypeError, ValueError):
        return 0.0
    return per_mtok if per_mtok > 0 else 0.0


def _openrouter_context_window(item: Mapping[str, Any]) -> int:
    raw = item.get("context_length") or (item.get("top_provider") or {}).get(
        "context_length"
    )
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _openrouter_tags(item: Mapping[str, Any]) -> list[str]:
    tags = {"openrouter"}
    params = item.get("supported_parameters") or []
    if "reasoning" in params or "reasoning_effort" in params:
        tags.add("reasoning")
    if _openrouter_context_window(item) >= 200_000:
        tags.add("long-context")
    if "image" in ((item.get("architecture") or {}).get("input_modalities") or []):
        tags.add("vision")
    try:
        # A negative quote means dynamic pricing (openrouter/auto), not free.
        if float((item.get("pricing") or {}).get("prompt")) == 0.0:
            tags.add("free")
    except (TypeError, ValueError):
        pass
    return sorted(tags)


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
    *,
    id_namespace: Optional[str] = None,
    payload_defaults: Optional[Mapping[str, Any]] = None,
) -> list[ModelSpec]:
    """Turn a discovered API catalog into :class:`ModelSpec`s for ``adapter``.

    Capability/price/context are inherited from an existing registry entry for
    the same adapter+model (the registry is a tuning overlay), then from any
    same-named entry on a sibling adapter, and finally a conservative seed the
    user edits. ``billing`` is the caller-provided posture for the adapter.

    A catalog that publishes its own price / context / tags (OpenRouter) fills
    those in for a new row and backfills an existing row only where the user
    left the value unset, so hand-tuned numbers still win. ``id_namespace``
    prefixes generated ids (``openrouter/openai-gpt-4o``) so two catalogs
    feeding one adapter cannot collide, and ``payload_defaults`` are stamped on
    generated rows so a bare ``--model`` pin carries its wire config."""
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

    catalog_label = id_namespace or adapter
    specs: list[ModelSpec] = []
    for item in catalog:
        model_id = str(item["id"])
        overlay = same_adapter.get(model_id) or same_adapter_normalized.get(
            normalize_model_token(model_id)
        )
        catalog_tags = set(item.get("tags") or [])
        catalog_input = float(item.get("input_per_mtok_usd") or 0.0)
        catalog_output = float(item.get("output_per_mtok_usd") or 0.0)
        catalog_context = int(item.get("context_window") or 0)
        if overlay is not None:
            specs.append(
                replace(
                    overlay,
                    adapter=adapter,
                    adapter_model_name=model_id,
                    billing=overlay.billing if overlay.billing != "unknown" else billing,
                    tags=sorted(set(overlay.tags) | catalog_tags | {"discovered"}),
                    input_per_mtok_usd=overlay.input_per_mtok_usd or catalog_input,
                    output_per_mtok_usd=overlay.output_per_mtok_usd or catalog_output,
                    context_window=overlay.context_window or catalog_context,
                    payload_defaults={
                        **dict(payload_defaults or {}),
                        **dict(overlay.payload_defaults or {}),
                    },
                )
            )
            continue
        display = item.get("displayName") or model_id
        kin = cap_by_name.get(model_id)
        if kin is not None:
            capability = kin.capability_score
            context_window = catalog_context or kin.context_window
            tags = sorted(
                {t for t in kin.tags if t != adapter}
                | catalog_tags
                | {adapter, "discovered"}
            )
            note = (
                f"Discovered from the {catalog_label} catalog ({display}); capability "
                f"inherited from {kin.id}. Set pricing to route on cost."
            )
        else:
            capability = _DEFAULT_DISCOVERED_CAPABILITY
            context_window = catalog_context
            tags = sorted(catalog_tags | {adapter, "discovered"})
            note = (
                f"Discovered from the {catalog_label} catalog ({display}). Tune "
                "capability_score and pricing so the router ranks/charges it correctly."
            )
        specs.append(
            ModelSpec(
                id=f"{id_namespace or adapter}/{_slug(model_id)}",
                adapter=adapter,
                adapter_model_name=model_id,
                capability_score=capability,
                input_per_mtok_usd=catalog_input,
                output_per_mtok_usd=catalog_output,
                context_window=context_window,
                billing=billing,
                tags=tags,
                notes=note,
                payload_defaults=dict(payload_defaults or {}),
            )
        )
    return specs


def merge_api_catalog_into_registry(
    adapter: str,
    billing: str,
    existing: list[ModelSpec],
    catalog: list[dict],
    *,
    id_namespace: Optional[str] = None,
    payload_defaults: Optional[Mapping[str, Any]] = None,
) -> "tuple[list[ModelSpec], dict]":
    """Reconcile a discovered API catalog with ``existing`` registry entries.

    Unlike Cursor discovery, API catalogs are large and noisy (every legacy
    model, embeddings, TTS, …), so we do NOT drop registry entries that aren't
    in the catalog — we only *add* newly-discovered ones and refresh overlays.
    Non-matching adapters and hand-tuned entries are preserved untouched."""
    discovered = catalog_to_specs(
        adapter,
        billing,
        catalog,
        existing,
        id_namespace=id_namespace,
        payload_defaults=payload_defaults,
    )
    discovered_by_identity = {
        normalize_model_token(s.adapter_model_name): s for s in discovered
    }

    existing_identities = {
        normalize_model_token(s.adapter_model_name)
        for s in existing
        if s.adapter == adapter
    }
    added = sorted(
        spec.adapter_model_name
        for identity, spec in discovered_by_identity.items()
        if identity not in existing_identities
    )

    merged: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in existing:
        identity = normalize_model_token(spec.adapter_model_name)
        if spec.adapter == adapter and identity in discovered_by_identity:
            merged.append(discovered_by_identity[identity])
            seen.add(identity)
        else:
            merged.append(spec)
    for identity, spec in discovered_by_identity.items():
        if identity not in seen:
            merged.append(spec)

    report = {
        "adapter": adapter,
        "discovered_count": len(discovered),
        "added": added,
        "refreshed": sorted(
            discovered_by_identity[identity].adapter_model_name for identity in seen
        ),
        "preserved": len([s for s in existing if s.adapter != adapter]),
    }
    return merged, report
