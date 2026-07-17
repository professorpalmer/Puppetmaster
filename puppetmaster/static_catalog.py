"""Curated model catalogs for platforms that can't self-enumerate.

Cursor exposes a queryable plan catalog (:mod:`puppetmaster.cursor_discovery`),
and OpenAI/Anthropic expose ``/v1/models`` (:mod:`puppetmaster.api_discovery`).
The two **CLI agent loops** — Claude Code and Codex — do *not* offer a queryable
model list, so historically their registry tiers had to be hand-maintained and
they never got Cursor's "plan-first, zero-config" treatment.

This module closes that gap with a **hand-curated catalog** of the models each
of those platforms ships, plus the same first-run auto-merge Cursor gets:

* When the platform is authenticated via a **subscription** (Claude Code OAuth,
  Codex/ChatGPT login) — i.e. billing == ``plan`` — the curated models are
  merged in as **plan-billed** (zeroed marginal cost) so frontier work routes
  through the subscription the user already pays for instead of a per-token
  account.
* When authenticated via a raw API key (``api``), the same catalog can be
  merged with reference per-token pricing (manual ``models discover`` path).

The curated lists are the one place to update when these platforms ship new
models. Keep capability/price/context in sync with the starter registry.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from puppetmaster.model_registry import (
    DISCOVERY_SOURCE_TO_ADAPTER,
    ModelSpec,
    catalog_content_hash,
)

# Hand-maintained catalog of what each non-enumerable platform offers. The
# ``input``/``output`` prices are the public per-token reference rates. They
# remain populated for subscription entries so the router can rank finite
# shared-plan usage by nominal consumption; ``ModelSpec.marginal_cost_usd``
# still reports $0 for plan-billed calls.
CURATED_CATALOGS: dict[str, list[dict]] = {
    "claude-code": [
        {
            "model": "claude-haiku-4-5",
            "capability": 55,
            "input": 1.0,
            "output": 5.0,
            "context": 200_000,
            "tags": ["tools", "claude", "cheap", "fast", "vision", "reading", "code"],
        },
        {
            "model": "claude-sonnet-4-5",
            "capability": 82,
            "input": 3.0,
            "output": 15.0,
            "context": 200_000,
            "tags": ["tools", "claude", "balanced", "vision", "code", "reasoning"],
        },
        {
            "model": "claude-opus-4-6",
            "capability": 88,
            "input": 5.0,
            "output": 25.0,
            "context": 200_000,
            "tags": ["tools", "claude", "quality", "vision", "code", "reasoning"],
        },
        {
            "model": "claude-opus-4-7",
            "capability": 98,
            "input": 5.0,
            "output": 25.0,
            "context": 200_000,
            "tags": ["tools", "claude", "frontier", "vision", "detailed-vision", "reasoning", "code"],
        },
        {
            "model": "claude-opus-4-8",
            "capability": 99,
            "input": 5.0,
            "output": 25.0,
            "context": 1_000_000,
            "tags": ["tools", 
                "claude",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
        },
    ],
    "codex": [
        {
            "model": "gpt-5.4-mini",
            "capability": 72,
            "input": 0.75,
            "output": 4.5,
            "context": 400_000,
            "tags": ["tools", "codex", "balanced", "vision", "code", "agent-loop"],
        },
        {
            "model": "gpt-5.4",
            "capability": 87,
            "input": 2.5,
            "output": 15.0,
            "context": 1_000_000,
            "tags": ["tools", "codex", "quality", "vision", "code", "reasoning", "agent-loop", "long-context"],
        },
        {
            "model": "gpt-5.6-luna",
            "capability": 91,
            "input": 1.0,
            "output": 6.0,
            "context": 1_050_000,
            "tags": ["tools", "codex", "balanced", "vision", "code", "agent-loop", "long-context"],
        },
        {
            "model": "gpt-5.6-terra",
            "capability": 98,
            "input": 2.5,
            "output": 15.0,
            "context": 1_050_000,
            "tags": ["tools", 
                "codex",
                "quality",
                "vision",
                "code",
                "reasoning",
                "agent-loop",
                "long-context",
            ],
        },
        {
            "model": "gpt-5.6-sol",
            "capability": 100,
            "input": 5.0,
            "output": 30.0,
            "context": 1_050_000,
            "tags": ["tools", 
                "codex",
                "frontier",
                "vision",
                "reasoning",
                "code",
                "agent-loop",
                "long-context",
            ],
        },
        {
            "model": "gpt-5.5",
            "capability": 97,
            "input": 5.0,
            "output": 30.0,
            "context": 1_000_000,
            "tags": ["tools", 
                "codex",
                "frontier",
                "vision",
                "reasoning",
                "code",
                "agent-loop",
                "long-context",
            ],
        },
    ],
    # Hermes is API-billed: each entry routes through the user's own provider
    # key (Gemini / Anthropic / OpenAI). ``payload_defaults.provider`` is the
    # critical field — a bare model name routes to unconfigured OpenRouter, so
    # the router must stamp the explicit Hermes provider alongside the model.
    # Prices are the public per-token reference rates for the underlying model.
    # Standalone direct-API worker catalog. Each entry routes through the
    # AgenticAdapter's own tool loop against the stamped provider's HTTP API
    # using the user's key -- no external agent CLI. ``payload_defaults.provider``
    # is the critical field (same pattern as Hermes): it selects the provider
    # descriptor (key env + base URL + wire protocol). Prices are public
    # per-token reference rates; billing is API (out-of-pocket). The key-aware
    # router filter drops any entry whose provider has no usable credential, so
    # a fresh install offers exactly what the user's keys unlock.
    "agentic": [
        {
            "model": "gemini-2.5-flash",
            "capability": 60,
            "input": 0.30,
            "output": 2.5,
            "context": 1_000_000,
            "tags": ["tools", "agentic", "gemini", "cheap", "fast", "vision", "code", "long-context"],
            "payload_defaults": {"provider": "gemini"},
        },
        {
            "model": "gemini-2.5-pro",
            "capability": 84,
            "input": 1.25,
            "output": 10.0,
            "context": 1_000_000,
            "tags": ["tools", "agentic", "gemini", "balanced", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "gemini"},
        },
        {
            "model": "claude-haiku-4-5",
            "capability": 55,
            "input": 1.0,
            "output": 5.0,
            "context": 200_000,
            "tags": ["tools", "agentic", "anthropic", "cheap", "fast", "vision", "code"],
            "payload_defaults": {"provider": "anthropic"},
        },
        {
            "model": "claude-sonnet-4-5",
            "capability": 82,
            "input": 3.0,
            "output": 15.0,
            "context": 200_000,
            "tags": ["tools", "agentic", "anthropic", "balanced", "vision", "code", "reasoning"],
            "payload_defaults": {"provider": "anthropic"},
        },
        {
            "model": "claude-opus-4-8",
            "capability": 99,
            "input": 15.0,
            "output": 75.0,
            "context": 200_000,
            "tags": ["tools", "agentic", "anthropic", "frontier", "quality", "vision", "code", "reasoning"],
            "payload_defaults": {"provider": "anthropic"},
        },
        {
            "model": "gpt-5",
            "capability": 90,
            "input": 1.25,
            "output": 10.0,
            "context": 400_000,
            "tags": ["tools", "agentic", "openai", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-luna",
            "capability": 90,
            "input": 1.0,
            "output": 6.0,
            "context": 1_050_000,
            "tags": ["tools", "agentic", "openai", "balanced", "fast", "vision", "code", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-terra",
            "capability": 97,
            "input": 2.5,
            "output": 15.0,
            "context": 1_050_000,
            "tags": ["tools", "agentic", "openai", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-sol",
            "capability": 99,
            "input": 5.0,
            "output": 30.0,
            "context": 1_050_000,
            "tags": ["tools", "agentic", "openai", "frontier", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.5",
            "capability": 96,
            "input": 5.0,
            "output": 30.0,
            "context": 400_000,
            "tags": ["tools", "agentic", "openai", "frontier", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
    ],
    # Hermes is API-billed via the external hermes CLI (kept for users who have
    # it installed). Prefer the ``agentic`` catalog above for standalone runs.
    "hermes": [
        {
            "model": "gemini-2.5-flash",
            "capability": 60,
            "input": 0.30,
            "output": 2.5,
            "context": 1_000_000,
            "tags": ["tools", "hermes", "gemini", "cheap", "fast", "vision", "code", "long-context"],
            "payload_defaults": {"provider": "gemini"},
        },
        {
            "model": "gemini-2.5-pro",
            "capability": 84,
            "input": 1.25,
            "output": 10.0,
            "context": 1_000_000,
            "tags": ["tools", "hermes", "gemini", "balanced", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "gemini"},
        },
        {
            "model": "claude-sonnet-4-5",
            "capability": 82,
            "input": 3.0,
            "output": 15.0,
            "context": 200_000,
            "tags": ["tools", "hermes", "anthropic", "balanced", "vision", "code", "reasoning"],
            "payload_defaults": {"provider": "anthropic"},
        },
        {
            "model": "gpt-5",
            "capability": 90,
            "input": 1.25,
            "output": 10.0,
            "context": 400_000,
            "tags": ["tools", "hermes", "openai", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-luna",
            "capability": 90,
            "input": 1.0,
            "output": 6.0,
            "context": 1_050_000,
            "tags": ["tools", "hermes", "openai", "balanced", "fast", "vision", "code", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-terra",
            "capability": 97,
            "input": 2.5,
            "output": 15.0,
            "context": 1_050_000,
            "tags": ["tools", "hermes", "openai", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.6-sol",
            "capability": 99,
            "input": 5.0,
            "output": 30.0,
            "context": 1_050_000,
            "tags": ["tools", "hermes", "openai", "frontier", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "gpt-5.5",
            "capability": 96,
            "input": 5.0,
            "output": 30.0,
            "context": 400_000,
            "tags": ["tools", "hermes", "openai", "frontier", "quality", "vision", "code", "reasoning", "long-context"],
            "payload_defaults": {"provider": "openai-api"},
        },
        {
            "model": "claude-opus-4-8",
            "capability": 99,
            "input": 15.0,
            "output": 75.0,
            "context": 200_000,
            "tags": ["tools", "hermes", "anthropic", "frontier", "quality", "vision", "code", "reasoning"],
            "payload_defaults": {"provider": "anthropic"},
        },
    ],
}

# Map an adapter to the discovery-meta source key (and the `models discover
# --source` name). Kept distinct from the adapter id so `claude-code` reads as
# `claude` on the CLI / in the sidecar, matching the existing `anthropic` style.
ADAPTER_TO_SOURCE = {"claude-code": "claude", "codex": "codex"}
SOURCE_TO_ADAPTER = dict(DISCOVERY_SOURCE_TO_ADAPTER)


def curated_catalog(adapter: str) -> list[dict]:
    """The curated model list for ``adapter`` (empty if none is curated)."""
    return CURATED_CATALOGS.get(adapter, [])


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def curated_to_specs(
    adapter: str,
    billing: str,
    existing: list[ModelSpec],
    *,
    allowed_providers: Optional[set] = None,
) -> list[ModelSpec]:
    """Turn the curated catalog for ``adapter`` into :class:`ModelSpec`s.

    A model already in the registry keeps its (possibly user-tuned) id,
    capability, context, and tags — we only restamp ``billing`` and, for
    plan billing, zero the marginal price. A model not yet in the registry is
    seeded from the curated entry. ``plan`` billing means a subscription covers
    it, so input/output prices are 0; ``api`` keeps the reference per-token
    rates so cost routing still works.

    When ``allowed_providers`` is given, a curated entry whose
    ``payload_defaults.provider`` is not in the set is skipped — so a Hermes
    model is only seeded when its provider has a usable credential. Entries
    without a pinned provider are always kept (the filter is a no-op for
    catalogs like Cursor/Claude that don't stamp one). ``None`` disables
    filtering entirely (the default — preserves every existing caller).
    """
    plan = billing == "plan"
    by_name = {s.adapter_model_name: s for s in existing if s.adapter == adapter}
    specs: list[ModelSpec] = []
    for item in curated_catalog(adapter):
        if allowed_providers is not None:
            provider = (item.get("payload_defaults") or {}).get("provider")
            if provider is not None and provider not in allowed_providers:
                continue
        model = str(item["model"])
        prior = by_name.get(model)
        capability = prior.capability_score if prior else int(item["capability"])
        context = (
            prior.context_window
            if (prior and prior.context_window)
            else int(item.get("context", 0))
        )
        base_tags = list(prior.tags) if prior else list(item.get("tags", [adapter]))
        tags = set(base_tags) | {"curated"}
        if plan:
            tags.add("plan-billed")
        spec_id = prior.id if prior else f"{adapter}/{_slug(model)}"
        if plan:
            note = (
                f"Curated {adapter} catalog entry, plan-billed: the authenticated "
                "subscription covers it, so marginal cost is $0 and the router can "
                "keep frontier work inside the plan you already pay for."
            )
        else:
            note = (
                f"Curated {adapter} catalog entry, API-billed at the reference "
                "per-token rate. Tune capability_score / prices to match your account."
            )
        # A curated entry may pin payload defaults (e.g. Hermes needs an
        # explicit ``provider`` stamped alongside the model so a routed worker
        # reaches the right credential/wire protocol). Preserve any prior,
        # user-tuned defaults; the curated value seeds new entries.
        payload_defaults = (
            dict(prior.payload_defaults)
            if (prior and prior.payload_defaults)
            else dict(item.get("payload_defaults", {}))
        )
        specs.append(
            ModelSpec(
                id=spec_id,
                adapter=adapter,
                adapter_model_name=model,
                capability_score=capability,
                input_per_mtok_usd=float(item["input"]),
                output_per_mtok_usd=float(item["output"]),
                context_window=context,
                billing=billing,
                tags=sorted(tags),
                notes=note,
                enabled=prior.enabled if prior else True,
                payload_defaults=payload_defaults,
            )
        )
    return specs


def merge_curated_into_registry(
    adapter: str,
    billing: str,
    existing: list[ModelSpec],
    *,
    allowed_providers: Optional[set] = None,
) -> "tuple[list[ModelSpec], dict]":
    """Reconcile the curated catalog for ``adapter`` into ``existing``.

    Refresh-or-add, never drop (like the API merge): a curated model already in
    the registry is replaced by its restamped counterpart; new ones are
    appended; every other adapter and any hand-tuned entry is preserved.

    ``allowed_providers`` filters provider-stamped entries to those with a
    usable credential (see :func:`curated_to_specs`); the filtered-out models
    are reported under ``skipped`` so callers can warn the user. ``None``
    (default) disables filtering.
    """
    discovered = curated_to_specs(
        adapter, billing, existing, allowed_providers=allowed_providers
    )
    by_name = {s.adapter_model_name: s for s in discovered}
    existing_names = {s.adapter_model_name for s in existing if s.adapter == adapter}
    added = sorted(by_name.keys() - existing_names)

    merged: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in existing:
        if spec.adapter == adapter and spec.adapter_model_name in by_name:
            merged.append(by_name[spec.adapter_model_name])
            seen.add(spec.adapter_model_name)
        else:
            merged.append(spec)
    for name, spec in by_name.items():
        if name not in seen:
            merged.append(spec)

    skipped: list[dict] = []
    if allowed_providers is not None:
        for item in curated_catalog(adapter):
            provider = (item.get("payload_defaults") or {}).get("provider")
            if provider is not None and provider not in allowed_providers:
                skipped.append({"model": str(item["model"]), "provider": provider})

    report = {
        "adapter": adapter,
        "source": ADAPTER_TO_SOURCE.get(adapter, adapter),
        "discovered_count": len(discovered),
        "added": added,
        "refreshed": sorted(seen),
        "skipped": skipped,
        "billing": billing,
    }
    return merged, report


def ensure_subscription_plan_catalog(
    registry_path: "Path",
    *,
    adapters: "tuple[str, ...]" = ("claude-code", "codex"),
    billing_detector: Optional[Callable[[str], object]] = None,
    min_capability: Optional[int] = None,
) -> dict:
    """First-run guarantee that a subscription-authenticated CLI agent loop
    (Claude Code OAuth, Codex/ChatGPT) contributes a plan-billed frontier.

    This is the Claude Code / Codex analog of
    :func:`puppetmaster.cursor_discovery.ensure_cursor_plan_catalog`. Cursor is
    preferred (it self-enumerates); this kicks in for users without a Cursor
    plan so they still get plan-first routing instead of falling off to a
    per-token account. Idempotent (per-source discovery marker) and never
    raises — a failure degrades to ``{"action": "unavailable", ...}``.

    Resolution per ``adapters`` (priority order):

    * If the registry already has a plan-billed frontier → no-op.
    * Skip any source already discovered once (sidecar meta).
    * For the first adapter whose billing is a healthy ``plan`` subscription →
      merge its curated catalog as plan-billed, persist registry + meta, done.
    * Else → no-op.
    """
    from puppetmaster.cursor_discovery import (
        PLAN_FRONTIER_MIN_CAPABILITY,
        has_plan_frontier,
    )
    from puppetmaster.model_registry import (
        load_registry,
        read_discovery_meta,
        save_registry,
        write_discovery_meta,
    )

    cap = min_capability if min_capability is not None else PLAN_FRONTIER_MIN_CAPABILITY

    if billing_detector is None:
        from puppetmaster.platform_billing import detect_adapter_billing

        billing_detector = lambda adapter: detect_adapter_billing(adapter)  # noqa: E731

    try:
        registry = load_registry(registry_path)
        if has_plan_frontier(registry, min_capability=cap):
            return {"action": "skip", "reason": "plan_frontier_present"}

        meta = read_discovery_meta(registry_path)
        for adapter in adapters:
            source = ADAPTER_TO_SOURCE.get(adapter, adapter)
            if isinstance(meta, dict) and meta.get(source):
                continue
            status = billing_detector(adapter)
            if not (
                getattr(status, "healthy", False)
                and getattr(status, "billing", "unknown") == "plan"
            ):
                continue
            merged, report = merge_curated_into_registry(adapter, "plan", registry)
            save_registry(merged, registry_path)
            catalog = curated_catalog(adapter)
            write_discovery_meta(
                source,
                report["discovered_count"],
                registry_path,
                model_ids=[item["model"] for item in catalog if item.get("model")],
                catalog_hash=catalog_content_hash(catalog),
            )
            return {
                "action": "discovered",
                "adapter": adapter,
                "source": source,
                "discovered_count": report["discovered_count"],
                "added": report["added"],
            }
        return {"action": "skip", "reason": "no_subscription_adapter"}
    except Exception as exc:  # never block a run on catalog enumeration
        return {"action": "unavailable", "error": str(exc)}
