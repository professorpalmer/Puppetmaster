"""Recommend registry rows from Agent Arena Pareto intersected with one stream.

A stream (lane) is adapter plus, when the adapter is multi-provider,
``payload_defaults.provider``. Examples that must never share a catalog:

- Marionette: ``agentic`` + ``openai-codex`` auth + that provider's models
- Puppetmaster inside Codex: the ``codex`` adapter
- Puppetmaster in Cursor: the ``cursor`` adapter and that plan catalog

GPT-5.6 Sol on Cursor does not mean GPT-5.4 mini is callable on Codex.
Prior-generation skips and workhorse picks are per-lane only.

The packaged Pareto snapshot is the default ranking authority. This module
does not scrape arena.ai on the routing hot path.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from puppetmaster.model_registry import ModelSpec, normalize_model_token

PARETO_RECOMMEND_ENV = "PUPPETMASTER_PARETO_RECOMMEND"
GENERATION_FILTER_ENV = "PUPPETMASTER_GENERATION_FILTER"
USER_TOGGLE_AUTHORITIES = frozenset({"user", "marionette"})
_PACKAGED = Path("baselines") / "arena-agent-pareto-v1.json"
_MULTI_PROVIDER_ADAPTERS = frozenset({"agentic", "hermes"})


def pareto_recommend_enabled() -> bool:
    raw = (os.environ.get(PARETO_RECOMMEND_ENV) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def generation_filter_enabled() -> bool:
    raw = (os.environ.get(GENERATION_FILTER_ENV) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def spec_lane(spec: ModelSpec) -> tuple[str, str]:
    """Identity of the hooked stream that can actually invoke this row.

    Cursor / Codex / OpenAI / Claude Code are one-adapter streams.
    Agentic and Hermes fan out by provider (openai-codex vs openrouter vs
    openai-api). Mixing those catalogs is how a Cursor Sol pick poisons a
    Codex swarm.
    """
    adapter = str(spec.adapter or "").strip()
    provider = ""
    if adapter in _MULTI_PROVIDER_ADAPTERS:
        provider = str((spec.payload_defaults or {}).get("provider") or "").strip()
    return (adapter, provider)


def default_pareto_snapshot_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / _PACKAGED,
        here.parents[1] / "docs" / _PACKAGED,
        Path.cwd() / "docs" / _PACKAGED,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_pareto_snapshot(path: Optional[Path] = None) -> dict:
    snapshot_path = path or default_pareto_snapshot_path()
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list):
        raise ValueError("Pareto snapshot must be an object with a rows array")
    return raw


def _token_set(values: Iterable[str]) -> set[str]:
    out = set()
    for value in values:
        token = normalize_model_token(str(value or ""))
        if token:
            out.add(token)
    return out


def spec_identity_tokens(spec: ModelSpec) -> set[str]:
    return _token_set((spec.adapter_model_name,))


def _row_tokens(row: dict) -> set[str]:
    return _token_set(row.get("identities") or [])


def spec_in_available_set(
    spec: ModelSpec,
    available_ids: Optional[Iterable[str]],
) -> bool:
    """True when this row is live on its own stream.

    Registry ids (``cursor/gpt-5.6-sol``) match only that spec.id.
    Bare names (``gpt-5.6-sol``) match ``adapter_model_name`` on this spec
    only — they never copy Cursor Sol into a Codex or agentic-codex lane.
    """
    if available_ids is None:
        return spec.is_routable
    wanted = {str(item).strip() for item in available_ids if str(item).strip()}
    if not wanted:
        return False
    if spec.id in wanted:
        return True
    tokens = spec_identity_tokens(spec)
    for item in wanted:
        if "/" in item:
            continue
        if normalize_model_token(item) in tokens:
            return True
    return False


def _lane_available(
    specs: Iterable[ModelSpec],
    available_ids: Optional[Iterable[str]],
) -> dict[tuple[str, str], set[str]]:
    """Live identities grouped by stream. Never union Cursor with Codex."""
    lanes: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        if not spec_in_available_set(spec, available_ids):
            continue
        lanes.setdefault(spec_lane(spec), set()).update(spec_identity_tokens(spec))
    return lanes


def _effort_for_adapter(adapter: str, arena_effort: str) -> str:
    effort = (arena_effort or "").strip().lower()
    if not effort:
        return ""
    if adapter == "codex" and effort in ("xhigh", "max"):
        return "high"
    if adapter == "cursor":
        return ""
    if adapter == "hermes" and effort == "max":
        return "xhigh"
    return effort


def _stamp_effort(spec: ModelSpec, arena_effort: str) -> dict:
    from puppetmaster.cli.commands_models import model_payload_defaults_for_effort
    from puppetmaster.cli.guidance import _EFFORT_TOKEN_MULTIPLIERS

    mapped = _effort_for_adapter(spec.adapter, arena_effort)
    if not mapped:
        return {}
    try:
        defaults = model_payload_defaults_for_effort(spec.adapter, mapped)
    except ValueError:
        return {}
    payload = dict(spec.payload_defaults or {})
    payload.update(defaults)
    multiplier = float(_EFFORT_TOKEN_MULTIPLIERS.get(mapped, spec.output_token_multiplier))
    tags = [tag for tag in spec.tags if not str(tag).startswith("effort:")]
    tags.append("effort:" + mapped)
    return {
        "payload_defaults": payload,
        "output_token_multiplier": multiplier,
        "tags": tags,
    }


def workhorse_row(snapshot: dict, available: set[str]) -> Optional[dict]:
    candidates = []
    for row in snapshot.get("rows") or []:
        if not row.get("pareto_optimal"):
            continue
        if float(row.get("net_improvement_pct") or 0) <= 0:
            continue
        if _row_tokens(row) & available:
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("cost_per_task_usd") or 0))


def prior_generation_rules(snapshot: Optional[dict] = None) -> list[dict]:
    data = snapshot if snapshot is not None else load_pareto_snapshot()
    rules = []
    for row in data.get("prior_generation") or []:
        identities = _token_set(row.get("identities") or [])
        successors = _token_set(row.get("successors") or [])
        if identities and successors:
            rules.append(
                {
                    "identities": identities,
                    "successors": successors,
                    "reason": str(row.get("reason") or "prior generation"),
                }
            )
    return rules


def filter_prior_generation(
    candidates: list[ModelSpec],
    *,
    snapshot: Optional[dict] = None,
) -> tuple[list[ModelSpec], list[tuple[ModelSpec, str]]]:
    """Drop exact prior-generation ids when a successor is eligible *in lane*.

    Cursor GPT-5.6 Sol must not retire an agentic/Codex GPT-5 row, and the
    reverse. Fail-open when the filter would empty a lane.
    """
    if not generation_filter_enabled() or not candidates:
        return list(candidates), []
    rules = prior_generation_rules(snapshot)
    if not rules:
        return list(candidates), []
    present_by_lane: dict[tuple[str, str], set[str]] = {}
    for spec in candidates:
        present_by_lane.setdefault(spec_lane(spec), set()).update(
            spec_identity_tokens(spec)
        )
    kept: list[ModelSpec] = []
    rejected: list[tuple[ModelSpec, str]] = []
    dropped_lanes: dict[tuple[str, str], int] = {}
    for spec in candidates:
        tokens = spec_identity_tokens(spec)
        lane = spec_lane(spec)
        present = present_by_lane.get(lane) or set()
        drop = False
        reason = ""
        for rule in rules:
            if tokens & rule["identities"] and (present & rule["successors"]):
                drop = True
                reason = rule["reason"]
                break
        if drop:
            rejected.append((spec, reason))
            dropped_lanes[lane] = dropped_lanes.get(lane, 0) + 1
        else:
            kept.append(spec)
    kept_lanes = {spec_lane(spec) for spec in kept}
    for spec, _reason in list(rejected):
        if spec_lane(spec) not in kept_lanes:
            kept.append(spec)
            rejected = [item for item in rejected if item[0] is not spec]
    if not kept:
        return list(candidates), []
    return kept, rejected


def _can_overwrite_score(spec: ModelSpec) -> bool:
    source = str((spec.score_provenance or {}).get("source") or "")
    return source in ("", "none", "arena_pareto", "community_baseline", "manual")


def _can_auto_disable(spec: ModelSpec) -> bool:
    authority = str(spec.disabled_authority or "").strip().lower()
    return authority not in USER_TOGGLE_AUTHORITIES


def apply_pareto_recommendations(
    specs: Iterable[ModelSpec],
    *,
    available_ids: Optional[Iterable[str]] = None,
    snapshot: Optional[dict] = None,
    stamp_effort: bool = True,
    disable_prior_generation: bool = True,
    overwrite_scores: bool = False,
) -> tuple[list[ModelSpec], dict]:
    """Stamp each stream from Pareto intersected with that stream's live ids.

    ``available_ids`` is the Marionette Settings allowlist or a PM-without-
    Marionette discover set. Prefer registry ids (``cursor/gpt-5.6-sol``).
    Bare names only count on the spec they already belong to. Ranking may
    be global; availability is always ``(adapter, provider)``.
    """
    existing = list(specs)
    if not pareto_recommend_enabled():
        return existing, {"action": "skip", "reason": "disabled"}
    data = snapshot if snapshot is not None else load_pareto_snapshot()
    lanes = _lane_available(existing, available_ids)
    workhorses_by_lane = {}
    for lane, available in lanes.items():
        horse = workhorse_row(data, available)
        if horse:
            workhorses_by_lane[lane] = horse
    rows_by_token: dict[str, dict] = {}
    for row in data.get("rows") or []:
        for token in _row_tokens(row):
            rows_by_token[token] = row

    updated: list[ModelSpec] = []
    stamped = []
    workhorse_ids = []
    disabled = []
    for spec in existing:
        lane = spec_lane(spec)
        available = lanes.get(lane) or set()
        tokens = spec_identity_tokens(spec)
        matched_row = None
        for token in tokens:
            matched_row = rows_by_token.get(token)
            if matched_row:
                break
        changes: dict[str, Any] = {}
        tags = list(spec.tags)
        in_lane = bool(tokens & available)
        if matched_row and in_lane:
            if "pareto" not in tags:
                tags.append("pareto")
            if matched_row.get("pareto_optimal") and "pareto-frontier" not in tags:
                tags.append("pareto-frontier")
            horse = workhorses_by_lane.get(lane)
            if horse and tokens & _row_tokens(horse):
                if "pareto-workhorse" not in tags:
                    tags.append("pareto-workhorse")
                workhorse_ids.append(spec.id)
            if stamp_effort and matched_row.get("effort"):
                changes.update(_stamp_effort(spec, str(matched_row.get("effort") or "")))
                tags = list(changes.get("tags") or tags)
            if (
                overwrite_scores
                and _can_overwrite_score(spec)
                and matched_row.get("recommended_capability") is not None
            ):
                changes["capability_score"] = int(matched_row["recommended_capability"])
            changes["score_provenance"] = {
                "source": "arena_pareto",
                "bundle_id": data.get("bundle_id"),
                "bundle_version": data.get("version"),
                "calibrated_at": data.get("published"),
                "arena_name": matched_row.get("arena_name"),
                "lane": list(lane),
                "notes": data.get("workhorse_rule"),
            }
            stamped.append(spec.id)
        if disable_prior_generation and _can_auto_disable(spec):
            for rule in prior_generation_rules(data):
                if tokens & rule["identities"] and (available & rule["successors"]):
                    changes["enabled"] = False
                    changes["disabled_reason"] = rule["reason"]
                    changes["disabled_authority"] = "arena_pareto"
                    disabled.append(spec.id)
                    break
        if tags != list(spec.tags):
            changes["tags"] = tags
        updated.append(replace(spec, **changes) if changes else spec)

    return updated, {
        "action": "applied",
        "published": data.get("published"),
        "source_url": data.get("source_url"),
        "workhorse_by_lane": {
            "%s/%s" % lane if lane[1] else lane[0]: row.get("arena_name")
            for lane, row in workhorses_by_lane.items()
        },
        "workhorse_ids": workhorse_ids,
        "stamped": stamped,
        "disabled_prior_generation": disabled,
    }


def maybe_apply_pareto_recommendations(
    specs: Iterable[ModelSpec],
    **kwargs: Any,
) -> tuple[list[ModelSpec], dict]:
    """Best-effort stamp. Catalog writes must not fail a discover/save."""
    existing = list(specs)
    try:
        return apply_pareto_recommendations(existing, **kwargs)
    except Exception as exc:
        return existing, {"action": "skip", "reason": "error", "error": str(exc)}
