"""Task-aware model router.

Given a task and a user-owned ``ModelSpec`` registry, decide which
model to invoke. The router is built around three pillars:

1. **Transparent classification.** A pure-function heuristic assigns a
   capability-needed score 0..100 to each task, based on role,
   instruction length, and content signals. The score is recorded on
   the routing artifact so users can see *why* a task went where.
2. **User-controlled policy.** ``balanced`` (default), ``cheap``,
   ``absolute-cheapest``, ``quality``, and ``escalating`` are the built-in
   policies. Users
   pin per-task overrides via ``payload.min_capability``,
   ``payload.max_cost_usd``, and ``payload.required_tags``.
3. **Auditable decisions.** Every routing decision lists the rejected
   alternatives and *why* each was rejected. No black boxes.

This module deliberately does **not** call any LLM. It picks specs;
the adapter actually runs the model.
"""
from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec, enabled_specs, model_id_allowed
from puppetmaster.scorecards import (
    card_capability,
    effective_capability_score,
    is_capability_sufficient,
    receipt_quality_tuple,
    resolve_score_authority,
    role_card_override_note,
    source_rank,
)

logger = logging.getLogger(__name__)

CACHE_AFFINITY_ENV = "PUPPETMASTER_CACHE_AFFINITY"
CACHE_AFFINITY_POLICIES = frozenset({"balanced", "cheap"})


def cache_affinity_enabled() -> bool:
    """True unless the operator disables sibling model stickiness."""
    raw = (os.environ.get(CACHE_AFFINITY_ENV) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

_ROUTING_OVERRIDES_CACHE: dict[str, tuple[float, dict]] = {}
_ROUTING_OVERRIDES_PARSE_WARNED: set[str] = set()

ROLE_TAXONOMY_PATH = Path(__file__).with_name("routing_roles.json")
_ROLE_TAXONOMY_SCHEMA_VERSION = 1


class RoleTaxonomyError(RuntimeError):
    """Raised when the packaged routing-role authority cannot be trusted."""


def _normalized_role_key(role: object) -> str:
    """Return the stable comparison form used by taxonomy roles and aliases."""
    text = str(role or "").strip().casefold()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _taxonomy_int(value: object, *, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoleTaxonomyError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise RoleTaxonomyError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


@lru_cache(maxsize=8)
def _read_role_taxonomy(path_text: str, mtime_ns: int, size: int) -> dict:
    del mtime_ns, size  # cache-key freshness only
    path = Path(path_text)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RoleTaxonomyError(
            f"Routing role taxonomy is unavailable at {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RoleTaxonomyError(
            f"Routing role taxonomy at {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise RoleTaxonomyError("Routing role taxonomy root must be an object")
    schema_version = _taxonomy_int(
        raw.get("schema_version"),
        field_name="schema_version",
        minimum=_ROLE_TAXONOMY_SCHEMA_VERSION,
        maximum=_ROLE_TAXONOMY_SCHEMA_VERSION,
    )
    taxonomy_version = raw.get("taxonomy_version")
    if not isinstance(taxonomy_version, str) or not taxonomy_version.strip():
        raise RoleTaxonomyError("taxonomy_version must be a non-empty string")

    unknown = raw.get("unknown_role")
    if not isinstance(unknown, dict):
        raise RoleTaxonomyError("unknown_role must be an object")
    unknown_canonical = _normalized_role_key(unknown.get("canonical_role"))
    if not unknown_canonical:
        raise RoleTaxonomyError(
            "unknown_role.canonical_role must be a non-empty role name"
        )
    unknown_capability = _taxonomy_int(
        unknown.get("base_capability"),
        field_name="unknown_role.base_capability",
        minimum=0,
        maximum=100,
    )
    if not isinstance(unknown.get("tool_loop"), bool):
        raise RoleTaxonomyError("unknown_role.tool_loop must be a boolean")
    unknown_output = unknown.get("output_tokens", 1500)
    _taxonomy_int(
        unknown_output,
        field_name="unknown_role.output_tokens",
        minimum=1,
        maximum=1_000_000,
    )

    roles = raw.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise RoleTaxonomyError("roles must be a non-empty object")

    validated_roles: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    for role_name, profile in roles.items():
        canonical = _normalized_role_key(role_name)
        if not canonical or canonical != role_name:
            raise RoleTaxonomyError(
                f"roles key {role_name!r} must already be a canonical role name"
            )
        if canonical == unknown_canonical:
            raise RoleTaxonomyError(
                f"unknown role {unknown_canonical!r} must not also appear in roles"
            )
        if not isinstance(profile, dict):
            raise RoleTaxonomyError(f"roles.{canonical} must be an object")
        capability = _taxonomy_int(
            profile.get("base_capability"),
            field_name=f"roles.{canonical}.base_capability",
            minimum=0,
            maximum=100,
        )
        tool_loop = profile.get("tool_loop")
        if not isinstance(tool_loop, bool):
            raise RoleTaxonomyError(f"roles.{canonical}.tool_loop must be a boolean")
        output_tokens = profile.get("output_tokens", 1500)
        _taxonomy_int(
            output_tokens,
            field_name=f"roles.{canonical}.output_tokens",
            minimum=1,
            maximum=1_000_000,
        )
        raw_aliases = profile.get("aliases", [])
        if not isinstance(raw_aliases, list) or any(
            not isinstance(alias, str) or not _normalized_role_key(alias)
            for alias in raw_aliases
        ):
            raise RoleTaxonomyError(
                f"roles.{canonical}.aliases must be a list of non-empty strings"
            )
        validated_roles[canonical] = {
            "base_capability": capability,
            "tool_loop": tool_loop,
            "output_tokens": output_tokens,
            "aliases": [_normalized_role_key(alias) for alias in raw_aliases],
        }

    for canonical, profile in validated_roles.items():
        for alias in [canonical, *profile["aliases"]]:
            if alias == unknown_canonical:
                raise RoleTaxonomyError(
                    f"role alias {alias!r} conflicts with the reserved "
                    "unknown_role canonical identity"
                )
            existing = aliases.get(alias)
            if existing is not None and existing != canonical:
                raise RoleTaxonomyError(
                    f"role alias {alias!r} is ambiguous between "
                    f"{existing!r} and {canonical!r}"
                )
            aliases[alias] = canonical

    return {
        "schema_version": schema_version,
        "taxonomy_version": taxonomy_version.strip(),
        "unknown_role": {
            "canonical_role": unknown_canonical,
            "base_capability": unknown_capability,
            "tool_loop": unknown["tool_loop"],
            "output_tokens": unknown_output,
        },
        "roles": validated_roles,
        "aliases": aliases,
    }


def load_role_taxonomy(path: Optional[Path] = None) -> dict:
    """Load and validate the versioned routing-role authority.

    Missing, unreadable, malformed, or unsupported authority fails closed with
    :class:`RoleTaxonomyError`; routing never silently falls back to a hidden
    Python table.
    """
    authority_path = Path(path) if path is not None else ROLE_TAXONOMY_PATH
    try:
        stat = authority_path.stat()
    except OSError as exc:
        raise RoleTaxonomyError(
            f"Routing role taxonomy is unavailable at {authority_path}: {exc}"
        ) from exc
    resolved = str(authority_path.resolve())
    try:
        taxonomy = _read_role_taxonomy(resolved, stat.st_mtime_ns, stat.st_size)
    except RoleTaxonomyError as exc:
        if resolved.casefold() in str(exc).casefold():
            raise
        raise RoleTaxonomyError(
            f"Routing role taxonomy at {resolved} is invalid: {exc}"
        ) from exc
    # The cached validated authority is private. Public callers receive a
    # detached object so accidental mutation cannot alter later routing.
    return deepcopy(taxonomy)


def normalize_routing_role(role: object) -> str:
    """Map a display/custom role alias to its canonical routing role."""
    taxonomy = load_role_taxonomy()
    role_key = _normalized_role_key(role)
    if not role_key:
        role_key = "explore"
    return taxonomy["aliases"].get(
        role_key, taxonomy["unknown_role"]["canonical_role"]
    )


def _role_profile(role: object) -> tuple[str, dict, dict]:
    taxonomy = load_role_taxonomy()
    role_key = _normalized_role_key(role) or "explore"
    canonical = taxonomy["aliases"].get(role_key)
    if canonical is None:
        return taxonomy["unknown_role"]["canonical_role"], taxonomy["unknown_role"], taxonomy
    return canonical, taxonomy["roles"][canonical], taxonomy


def _load_routing_overrides() -> dict:
    """Read ``~/.pmharness/routing.json`` if present, else an empty dict.

    Never raises: a missing/corrupt file simply means "no overrides". Kept in
    one place so the classifier and ``route_task`` read the same source.
    """
    routing_path = os.path.expanduser("~/.pmharness/routing.json")
    try:
        mtime = os.path.getmtime(routing_path) if os.path.exists(routing_path) else -1.0
    except OSError:
        mtime = -1.0
    cached = _ROUTING_OVERRIDES_CACHE.get(routing_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        if mtime < 0:
            data: dict = {}
        else:
            with open(routing_path) as handle:
                data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        if routing_path not in _ROUTING_OVERRIDES_PARSE_WARNED:
            logger.warning(
                "Ignoring unreadable routing overrides at %s: %s",
                routing_path,
                exc,
            )
            _ROUTING_OVERRIDES_PARSE_WARNED.add(routing_path)
        data = {}
    result = data if isinstance(data, dict) else {}
    _ROUTING_OVERRIDES_CACHE[routing_path] = (mtime, result)
    return result


# ----- Task signals --------------------------------------------------------


@dataclass(frozen=True)
class TaskSignals:
    """Summary of a task used for routing decisions.

    Keep this small and explicit — the orchestrator constructs one of
    these from a ``WorkerSpec`` (or a free-form instruction in the
    ``puppetmaster_route_task`` MCP path).
    """

    instruction: str
    role: str = "explore"
    payload_size_chars: int = 0
    explicit_min_capability: Optional[int] = None
    # Ceiling on the classifier output (cost guardrail). Unlike
    # ``explicit_min_capability`` -- which FORCES the need to an exact value and
    # therefore flattens every task to the same score -- the ceiling lets the
    # classifier differentiate tasks normally and only clips the top, so cheap
    # tasks still route to cheap models while expensive ones stop at the cap.
    explicit_max_capability: Optional[int] = None
    explicit_max_cost_usd: Optional[float] = None
    required_tags: list[str] = field(default_factory=list)
    estimated_tokens_in: Optional[int] = None
    estimated_tokens_out: Optional[int] = None
    # Adapter-owned prompt material (system instructions, tool declarations,
    # repository summaries, etc.) that is known before dispatch.  Keep this
    # separate from payload character counting so callers can declare exact
    # token overhead without manufacturing a string placeholder.
    adapter_enrichment_tokens: int = 0
    # When true, capability sufficiency becomes a hard routing constraint.
    # The default preserves the historical strongest-available fallback.
    strict_capability: bool = False
    # Cost-containment knobs (the "stay inside the plan you already pay for"
    # default). ``prefer_plan_billed`` makes a subscription-covered model win
    # over an out-of-pocket API model at equal-or-sufficient capability.
    # ``allow_api_billing`` is the hard gate: when False, the router refuses
    # to spend on api-billed models at all (plan-only).
    prefer_plan_billed: bool = True
    allow_api_billing: bool = True
    # Platform lock: when set, only models whose adapter is in this set are
    # eligible. ``None`` means "no restriction" (every adapter allowed), which
    # is the default for unlocked users. Populated from ``platform_lock`` by
    # the signal builders so the restriction applies everywhere routing runs.
    allowed_adapters: Optional[frozenset[str]] = None
    # Explicit allowlist of model identities for auto-routing. Accepts registry
    # ids (``cursor/grok-4-5``), adapter model names (``grok-4.5``), and
    # slug-equivalent forms. ``None`` means unrestricted; an explicit empty
    # set (``frozenset()``) fails closed; a non-empty set restricts selection
    # and same-adapter reroute to those identities only.
    allowed_model_ids: Optional[frozenset[str]] = None
    # Same-job sibling stickiness: keep a previous worker's model when it still
    # meets capability so the shared job-brief prefix stays provider-cacheable.
    # This avoids a model switch; it does not port KV cache across models.
    prefer_model_id: Optional[str] = None


# ----- Classifier ----------------------------------------------------------


# Capability bases, aliases, and tool-loop classification are intentionally
# absent here. ``routing_roles.json`` is the versioned packaged authority; its
# loader validates the complete table before any routing decision.
TOOLS_TAG = "tools"

_HARD_SIGNAL_PATTERNS = [
    (re.compile(r"\baudit\b"), 10),
    (re.compile(r"\bsecurity\b"), 15),
    (re.compile(r"\bperformance\b"), 10),
    (re.compile(r"\bperf\b"), 10),
    (re.compile(r"\bcross[-\s]?repo\b"), 10),
    (re.compile(r"\bevery (file|function|module|repo)\b"), 10),
    (re.compile(r"\bdesign\b"), 8),
    (re.compile(r"\barchitect"), 8),
    (re.compile(r"\brefactor\b"), 5),
    (re.compile(r"\bcomplex\b"), 5),
    (re.compile(r"\bnon[-\s]?trivial\b"), 5),
]

_EASY_SIGNAL_PATTERNS = [
    (re.compile(r"\btypo\b"), -15),
    (re.compile(r"\bcomment\b(?!.*delete)"), -5),
    (re.compile(r"\brename\b"), -5),
    (re.compile(r"\bformat\b"), -5),
    (re.compile(r"\blint\b"), -5),
]

# Vision signals. When ANY of these match the instruction, the router
# (1) bumps the capability score (vision tasks are harder), and (2)
# adds ``vision`` as a required tag automatically — so the picked
# model must declare vision support in its tags. ``detailed-vision``
# tasks (screenshots, diagrams, OCR) get an extra bump on top.
_VISION_SIGNAL_PATTERNS = [
    (re.compile(r"\bimage\b"), 8),
    (re.compile(r"\bimages\b"), 8),
    (re.compile(r"\bphoto\b"), 8),
    (re.compile(r"\bvisual(ly)?\b"), 8),
    (re.compile(r"\bvision\b"), 10),
    (re.compile(r"\bscreenshot\b"), 10),
    (re.compile(r"\bdiagram\b"), 10),
    (re.compile(r"\bui mock(up)?\b"), 10),
    (re.compile(r"\bocr\b"), 12),
    (re.compile(r"\bchart\b"), 6),
]

_DETAILED_VISION_PATTERNS = [
    re.compile(r"\bdetailed (image|visual|vision|diagram|chart|screenshot)\b"),
    re.compile(r"\bocr\b"),
    re.compile(r"\bread (the|this) (screenshot|image|diagram|chart|ui mock(up)?)\b"),
    re.compile(
        r"\b(extract|describe) (every|all) (element|detail)s? in (the|this) (image|screenshot|diagram)\b"
    ),
    re.compile(
        r"\b(every|all) (element|detail)s? (in|of) (the|this) (image|screenshot|diagram)\b"
    ),
    re.compile(r"\bocr every (detail|element)\b"),
]


def classify_capability_needed(task: TaskSignals) -> int:
    """Return capability score 0..100 needed to handle ``task`` well.

    Pure function. Same input → same output. Users override via
    ``task.explicit_min_capability`` (which we honor without modification)
    or cap the classifier output with ``task.explicit_max_capability``.
    """
    canonical_role, profile, _taxonomy = _role_profile(task.role)
    if task.explicit_min_capability is not None:
        forced = max(0, min(100, task.explicit_min_capability))
        if task.explicit_max_capability is not None:
            forced = min(forced, max(0, min(100, task.explicit_max_capability)))
        return forced

    role_base_score = profile["base_capability"]
    overrides = _load_routing_overrides().get("overrides", {})
    if isinstance(overrides, dict) and canonical_role in overrides:
        candidate = overrides[canonical_role]
        # Only honor a real numeric override (bools are ints in Python but are
        # never a meaningful capability score, so reject them too). A bad value
        # — e.g. {"overrides": {"audit": "high"}} — is ignored so the later
        # arithmetic can't blow up on a non-numeric base.
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            role_base_score = max(0, min(100, int(candidate)))

    score = role_base_score

    instruction_lower = task.instruction.lower()
    for pattern, weight in _HARD_SIGNAL_PATTERNS:
        if re.search(pattern, instruction_lower):
            score += weight
    for pattern, weight in _EASY_SIGNAL_PATTERNS:
        if re.search(pattern, instruction_lower):
            score += weight
    for pattern, weight in _VISION_SIGNAL_PATTERNS:
        if re.search(pattern, instruction_lower):
            score += weight
    if has_detailed_vision_signal(task.instruction):
        score += 12

    # Long instructions usually mean harder problems.
    if len(task.instruction) > 2000:
        score += 10
    elif len(task.instruction) > 800:
        score += 5

    # Big payloads (lots of code stuffed in) also lean harder.
    if task.payload_size_chars > 20_000:
        score += 10
    elif task.payload_size_chars > 5_000:
        score += 5

    # Ceiling tracks the capability_score of the current frontier flagship
    # in the starter registry (Claude Fable 5 @ 100). Keeping the max
    # need at the top model's score means the absolute-hardest tasks demand
    # — and therefore route to — the flagship, instead of saturating one
    # notch below it. Bump this in lockstep when a stronger model lands.
    score = max(5, min(100, score))
    if task.explicit_max_capability is not None:
        score = min(score, max(0, min(100, task.explicit_max_capability)))
    return score


def has_vision_signal(instruction: str) -> bool:
    """True if the instruction mentions images, screenshots, or visual input.

    Public so the router (and tests) can decide whether to auto-add
    ``vision`` to a task's ``required_tags``.
    """
    lower = instruction.lower()
    for pattern, _ in _VISION_SIGNAL_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def has_detailed_vision_signal(instruction: str) -> bool:
    """True for the harder vision subclass: OCR / detailed diagrams / charts."""
    lower = instruction.lower()
    for pattern in _DETAILED_VISION_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def task_needs_tool_calling(task: TaskSignals) -> bool:
    """True for roles that run an agentic-style tool loop (search/edit/submit)."""
    _canonical_role, profile, _taxonomy = _role_profile(task.role)
    return profile["tool_loop"]


# ----- Token estimation ----------------------------------------------------


def _validated_explicit_token_estimate(value: object, *, field_name: str) -> int:
    """Return an authoritative token override or fail before routing math."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TokenEstimateCalibration:
    """Explicit, immutable estimate correction derived from measured usage."""

    adapter: str
    model_id: str
    canonical_role: str
    source: str
    sample_count: int
    multiplier: float

    def to_artifact_payload(self) -> dict:
        return {
            "adapter": self.adapter,
            "model_id": self.model_id,
            "canonical_role": self.canonical_role,
            "source": self.source,
            "sample_count": self.sample_count,
            "multiplier": self.multiplier,
        }


def calibration_from_measurements(
    measurements: Iterable[Mapping[str, object]],
    *,
    adapter: str,
    model_id: str,
    role: str,
    source: str,
) -> TokenEstimateCalibration:
    """Build a read-only calibration value from attributable measurements.

    The multiplier is the aggregate measured-input / aggregate estimated-input
    ratio.  Aggregate weighting avoids a tiny request influencing the result as
    much as a large request.  Approximate records explicitly marked
    ``actual_tokens_measured=False`` are excluded; invalid rows are ignored and
    an all-invalid input fails instead of inventing a neutral multiplier.

    This function only returns data.  It never writes routing overrides or
    mutates model scores/policy, so applying measured history remains an
    explicit choice at the call site.
    """
    adapter_text = str(adapter or "").strip()
    model_text = str(model_id or "").strip()
    source_text = str(source or "").strip()
    if not adapter_text or not model_text or not source_text:
        raise ValueError("calibration adapter, model_id, and source are required")

    estimated_total = 0
    actual_total = 0
    sample_count = 0
    for row in measurements:
        if not isinstance(row, Mapping):
            continue
        if row.get("actual_tokens_measured") is False:
            continue
        estimated = row.get("estimated_tokens_in")
        actual = row.get("actual_tokens_in")
        if (
            isinstance(estimated, bool)
            or not isinstance(estimated, (int, float))
            or isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not isfinite(float(estimated))
            or not isfinite(float(actual))
            or estimated <= 0
            or actual <= 0
        ):
            continue
        estimated_total += int(estimated)
        actual_total += int(actual)
        sample_count += 1
    if sample_count == 0 or estimated_total <= 0:
        raise ValueError("calibration requires at least one measured token sample")

    return TokenEstimateCalibration(
        adapter=adapter_text,
        model_id=model_text,
        canonical_role=normalize_routing_role(role),
        source=source_text,
        sample_count=sample_count,
        multiplier=actual_total / estimated_total,
    )


def estimate_tokens_in(
    task: TaskSignals,
    *,
    calibration: Optional[TokenEstimateCalibration] = None,
) -> int:
    if task.estimated_tokens_in is not None:
        base = _validated_explicit_token_estimate(
            task.estimated_tokens_in,
            field_name="estimated_tokens_in",
        )
    else:
        # ~4 chars/token is the standard rough heuristic.
        text_chars = len(task.instruction) + task.payload_size_chars
        base = max(500, text_chars // 4 + 500)  # system + tools baseline
    enrichment = task.adapter_enrichment_tokens
    if isinstance(enrichment, bool) or not isinstance(enrichment, int):
        enrichment = 0
    estimate = max(0, int(base)) + max(0, enrichment)
    if calibration is not None:
        estimate = int(round(estimate * calibration.multiplier))
    return max(0, estimate)


def estimate_tokens_out(task: TaskSignals) -> int:
    if task.estimated_tokens_out is not None:
        return _validated_explicit_token_estimate(
            task.estimated_tokens_out,
            field_name="estimated_tokens_out",
        )
    _canonical_role, profile, _taxonomy = _role_profile(task.role)
    return profile["output_tokens"]


# ----- Routing -------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """One model picked for one task, plus the audit trail.

    Persisted as an ``ArtifactType.ROUTING`` artifact so the user can
    inspect *why* each task ran where and what alternatives were
    considered.
    """

    model: ModelSpec
    policy: str
    capability_needed: int
    estimated_tokens_in: int
    estimated_tokens_out: int
    estimated_cost_usd: float
    reason: str
    rejected: list[tuple[ModelSpec, str]] = field(default_factory=list)
    nominal_cost_usd: float = 0.0
    # Savings accounting (Rule 1: snapshot the baseline at decision time so the
    # ledger never compares a stored cost against a recomputed/drifted one).
    # ``baseline`` = what this task would have cost on the strongest model the
    # user could have used (highest-capability enabled + platform-allowed),
    # at the same token estimate.
    baseline_cost_usd: float = 0.0
    baseline_nominal_cost_usd: float = 0.0
    baseline_model_id: str = ""
    allowed_model_ids: Optional[list[str]] = None
    role: str = ""
    canonical_role: str = ""
    taxonomy_version: str = ""
    effective_capability_score: Optional[int] = None
    score_source: str = ""
    score_provenance: dict = field(default_factory=dict)
    sample_count: Optional[int] = None
    predicted_quality: Optional[float] = None
    predicted_latency_p50_ms: Optional[float] = None
    token_estimate_calibration: Optional[TokenEstimateCalibration] = None
    # Optional counterfactual evidence. This is deliberately metadata on the
    # already-computed production decision; it never supplies dispatch fields.
    shadow_routing: Optional[dict] = None

    def to_artifact_payload(self) -> dict:
        payload = {
            "model_id": self.model.id,
            "adapter": self.model.adapter,
            "adapter_model_name": self.model.adapter_model_name,
            "billing": self.model.billing,
            "policy": self.policy,
            "capability_needed": self.capability_needed,
            "capability_score": self.model.capability_score,
            "estimated_tokens_in": self.estimated_tokens_in,
            "estimated_tokens_out": self.estimated_tokens_out,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "nominal_cost_usd": round(self.nominal_cost_usd, 6),
            "baseline_cost_usd": round(self.baseline_cost_usd, 6),
            "baseline_nominal_cost_usd": round(self.baseline_nominal_cost_usd, 6),
            "baseline_model_id": self.baseline_model_id,
            "reason": self.reason,
            "rejected": [
                {"id": spec.id, "reason": why} for spec, why in self.rejected
            ],
        }
        if self.allowed_model_ids:
            payload["allowed_model_ids"] = list(self.allowed_model_ids)
        if self.model.payload_defaults:
            payload["payload_defaults"] = self.model.payload_defaults
        if self.role:
            payload["role"] = self.role
        if self.canonical_role:
            payload["canonical_role"] = self.canonical_role
        if self.taxonomy_version:
            payload["taxonomy_version"] = self.taxonomy_version
        if self.effective_capability_score is not None:
            payload["effective_capability_score"] = self.effective_capability_score
        if self.score_source:
            payload["score_source"] = self.score_source
        if self.score_provenance:
            payload["score_provenance"] = dict(self.score_provenance)
        if self.sample_count is not None:
            payload["sample_count"] = self.sample_count
        if self.predicted_quality is not None:
            payload["predicted_quality"] = self.predicted_quality
        if self.predicted_latency_p50_ms is not None:
            payload["predicted_latency_p50_ms"] = self.predicted_latency_p50_ms
        if self.token_estimate_calibration is not None:
            payload["token_estimate_calibration"] = (
                self.token_estimate_calibration.to_artifact_payload()
            )
        if self.shadow_routing is not None:
            payload["shadow_routing"] = dict(self.shadow_routing)
        return payload


class NoEligibleModelError(RuntimeError):
    """Raised when the policy + constraints exclude every registered model."""


VALID_POLICIES = {
    "balanced",
    "cheap",
    "absolute-cheapest",
    "quality",
    "escalating",
}


def _route_task_once(
    task: TaskSignals,
    registry: Iterable[ModelSpec],
    *,
    policy: Optional[str] = None,
    calibration: Optional[TokenEstimateCalibration] = None,
    local_receipts: Optional[Iterable] = None,
) -> RoutingDecision:
    """Pick a model for ``task`` from ``registry`` using ``policy``.

    Raises :class:`NoEligibleModelError` for *hard* constraint failures: an
    empty/all-disabled registry, a platform lock that excludes every model, or
    a ``max_cost_usd`` cap nothing can satisfy.

    Capability is a soft preference by default: when no model meets the needed
    score, cost-aware policies deliberately fall back to the strongest model
    available with an explicit reason. Set ``strict_capability=True`` to make
    the score a hard gate. Required tags remain independently hard constraints.
    """
    canonical_role = normalize_routing_role(task.role)

    # An explicitly-passed policy always wins. Only when the caller leaves it
    # unspecified (``policy is None``) do we consult the saved routing.json
    # preference, falling back to ``balanced`` when none is configured.
    if policy is None:
        saved_policy = _load_routing_overrides().get("routing_policy")
        policy = saved_policy if saved_policy in VALID_POLICIES else "balanced"

    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {VALID_POLICIES}")

    candidates = enabled_specs(registry)
    if not candidates:
        raise NoEligibleModelError(
            "No enabled models in registry. Run `puppetmaster models init` "
            "to write a starter ~/.puppetmaster/models.json, then edit it."
        )

    base_tokens_in = estimate_tokens_in(task)
    tokens_out = estimate_tokens_out(task)
    need = classify_capability_needed(task)

    def _candidate_calibration(
        spec: ModelSpec,
    ) -> Optional[TokenEstimateCalibration]:
        if calibration is None:
            return None
        if calibration.canonical_role != canonical_role:
            return None
        if calibration.adapter != spec.adapter or calibration.model_id != spec.id:
            return None
        return calibration

    def _tokens_in_for(spec: ModelSpec) -> int:
        scoped = _candidate_calibration(spec)
        if scoped is None:
            return base_tokens_in
        return estimate_tokens_in(task, calibration=scoped)

    # Auto-add vision tags when the instruction needs vision. The user
    # can still pin explicit tags via TaskSignals.required_tags; we
    # union with the auto-detected ones so explicit choices stay in.
    effective_required_tags = set(task.required_tags)
    if has_vision_signal(task.instruction):
        effective_required_tags.add("vision")
    if has_detailed_vision_signal(task.instruction):
        effective_required_tags.add("detailed-vision")

    rejected: list[tuple[ModelSpec, str]] = []

    # Explicit model allowlist: when the caller (or ~/.pmharness/routing.json)
    # constrains auto-routing to a set of model identities, drop everything
    # else before platform/cost filters so disallowed models can never win
    # selection or same-adapter reroute.
    effective_allowed_models = _effective_allowed_model_ids(task)
    if effective_allowed_models is not None:
        if not effective_allowed_models:
            raise NoEligibleModelError(
                "allowed_model_ids is explicitly empty — no model may be "
                "selected. Add at least one model identity to allowed_model_ids "
                "/ allowed_models, or omit the key to allow unrestricted routing."
            )
        after_allowlist: list[ModelSpec] = []
        allowed_sorted = sorted(effective_allowed_models)
        for spec in candidates:
            if model_id_allowed(spec, effective_allowed_models):
                after_allowlist.append(spec)
            else:
                rejected.append(
                    (
                        spec,
                        f"model not in allowed_model_ids {allowed_sorted}",
                    )
                )
        if not after_allowlist:
            raise NoEligibleModelError(
                "No enabled model matches allowed_model_ids "
                f"{allowed_sorted}. Enable at least one listed model in "
                "`~/.puppetmaster/models.json`, widen the allowlist, or clear "
                "allowed_model_ids / allowed_models."
            )
        candidates = after_allowlist
    _allowed_for_artifact = (
        sorted(effective_allowed_models)
        if effective_allowed_models is not None
        else None
    )

    # Platform lock first: a disabled platform must never be selected, so drop
    # its models before any other consideration with a clear reason.
    if task.allowed_adapters is not None:
        after_platform: list[ModelSpec] = []
        for spec in candidates:
            if spec.adapter in task.allowed_adapters:
                after_platform.append(spec)
            else:
                rejected.append(
                    (
                        spec,
                        f"adapter {spec.adapter!r} not in platform lock "
                        f"{sorted(task.allowed_adapters)}",
                    )
                )
        if not after_platform:
            raise NoEligibleModelError(
                "No model in registry uses an enabled platform "
                f"{sorted(task.allowed_adapters)}. Adjust with "
                "`puppetmaster platform enable <adapter>`."
            )
        candidates = after_platform

    # Key-aware filter for standalone (agentic) models: a direct-API model can
    # only run if its stamped provider actually has a usable credential right
    # now. This is the direct-API analogue of the platform lock -- it drops a
    # model whose provider key is absent so routing never picks one that would
    # 401 on first call. Only 'agentic' specs are filtered (hermes and the CLI
    # adapters keep their own availability logic); non-agentic candidates pass
    # through untouched, so a mixed registry is unaffected.
    try:
        from puppetmaster.providers import available_providers as _available_providers
        _providers_ready: Optional[set] = _available_providers()
    except Exception:
        _providers_ready = None
    if _providers_ready is not None and any(s.adapter == "agentic" for s in candidates):
        after_keys: list[ModelSpec] = []
        for spec in candidates:
            provider = (spec.payload_defaults or {}).get("provider")
            if spec.adapter == "agentic" and provider and provider not in _providers_ready:
                rejected.append((spec, f"provider {provider!r} has no usable API key"))
                continue
            after_keys.append(spec)
        if not after_keys:
            raise NoEligibleModelError(
                "No standalone (agentic) model has a usable provider credential. "
                "Set a provider key (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "GEMINI_API_KEY, OPENROUTER_API_KEY) or verify AWS Bedrock "
                "(AWS_PROFILE/default ~/.aws, AWS_ACCESS_KEY_ID/"
                "AWS_SECRET_ACCESS_KEY, or AWS_BEARER_TOKEN_BEDROCK — then "
                "`puppetmaster models discover --source agentic --probe`), "
                "or enable another platform."
            )
        candidates = after_keys

    # Tag filter first — cheap to evaluate, gives a clean reason on rejection.
    after_tags: list[ModelSpec] = []
    for spec in candidates:
        if effective_required_tags and not effective_required_tags.issubset(set(spec.tags)):
            rejected.append(
                (
                    spec,
                    f"missing required tags: {sorted(effective_required_tags - set(spec.tags))}",
                )
            )
            continue
        after_tags.append(spec)

    if not after_tags:
        raise NoEligibleModelError(
            f"No model in registry has all required tags {sorted(effective_required_tags)}"
        )

    # Context capacity is a hard execution constraint, not a ranking hint.
    # A zero/unknown registry value stays compatible with existing catalogs;
    # any known positive window must hold both enriched input and the reserved
    # output budget.  Apply this before cost/capability policy ranking so an
    # attractive but impossible model can never win.
    after_context: list[ModelSpec] = []
    for spec in after_tags:
        candidate_tokens_in = _tokens_in_for(spec)
        required_context = candidate_tokens_in + tokens_out
        window = spec.context_window
        if window > 0 and required_context > window:
            rejected.append(
                (
                    spec,
                    "context overflow: estimated input "
                    f"{candidate_tokens_in} + reserved output {tokens_out} = "
                    f"{required_context} tokens exceeds context window {window}",
                )
            )
            continue
        after_context.append(spec)

    if not after_context:
        raise NoEligibleModelError(
            "No model in registry can fit its estimated request context "
            f"(base input {base_tokens_in}, reserved output {tokens_out}; "
            "see per-model context rejection reasons)."
        )

    # Cost budget filter. This is a hard marginal-spend cap; plan-billed
    # models remain $0 here even though their nominal usage rate is used for
    # ranking below.
    after_cost: list[ModelSpec] = []
    for spec in after_context:
        est = spec.marginal_cost_usd(_tokens_in_for(spec), tokens_out)
        if (
            task.explicit_max_cost_usd is not None
            and est > task.explicit_max_cost_usd
        ):
            rejected.append(
                (
                    spec,
                    f"estimated ${est:.4f} exceeds budget ${task.explicit_max_cost_usd:.4f}",
                )
            )
            continue
        after_cost.append(spec)

    if not after_cost:
        raise NoEligibleModelError(
            "No model in registry fits the cost budget for this task."
        )

    # Soft tools preference for agentic tool-loop roles: when *any* eligible
    # candidate carries the ``tools`` tag, require it and drop untagged
    # (often reasoning-only) models. Fail-open when nothing is tagged so
    # untagged registries keep prior behavior.
    if task_needs_tool_calling(task):
        tools_capable = [s for s in after_cost if TOOLS_TAG in set(s.tags)]
        if tools_capable:
            for spec in after_cost:
                if TOOLS_TAG not in set(spec.tags):
                    rejected.append(
                        (
                            spec,
                            f"missing required tag {TOOLS_TAG!r} for "
                            f"agentic tool-loop role {task.role!r}",
                        )
                    )
            after_cost = tools_capable

    # Billing gate: when the caller forbids out-of-pocket API spend, drop every
    # model that isn't covered by a subscription the user already pays for.
    # ``unknown``-billing models are treated as not-plan here (we don't bill an
    # account we can't confirm is contained). Runtime detection upgrades
    # ``unknown`` -> ``plan`` before routing when it can prove a subscription.
    if not task.allow_api_billing:
        after_billing: list[ModelSpec] = []
        for spec in after_cost:
            if spec.is_plan_billed:
                after_billing.append(spec)
            else:
                rejected.append(
                    (
                        spec,
                        f"api billing disabled (allow_api_billing=False); "
                        f"model is {spec.billing}-billed, not plan-covered",
                    )
                )
        if not after_billing:
            raise NoEligibleModelError(
                "allow_api_billing=False but no plan-billed (subscription-covered) "
                "model is eligible for this task. Enable API billing, add a "
                "plan-billed model (e.g. run `puppetmaster models discover`), or "
                "lower the task's capability need."
            )
        after_cost = after_billing

    # Same-lane prior-generation skip (issue #107). Cursor GPT-5.6 Sol must
    # not retire a Codex or agentic-codex GPT-5 row. Fail-open per lane.
    from puppetmaster.pareto_recommend import filter_prior_generation

    after_cost, generation_rejected = filter_prior_generation(after_cost)
    for spec, reason in generation_rejected:
        rejected.append(
            (
                spec,
                "prior generation on this adapter/provider lane: " + reason,
            )
        )
    if not after_cost:
        raise NoEligibleModelError(
            "No model in registry remains after prior-generation filtering."
        )

    # Snapshot the savings baseline from the strongest model that was *actually
    # eligible for this task* — i.e. the final candidate set the pick is drawn
    # from, after every hard constraint (platform lock, required tags, cost cap,
    # billing gate). Computing it from the same set the pick comes from is what
    # keeps the ledger honest: a constrained run can't be credited against (or
    # penalised by) a model it could never have run. Stored on the decision so
    # the ledger compares like-for-like later instead of recomputing against a
    # possibly-changed registry.
    receipts = tuple(local_receipts or ())

    def _authority(spec: ModelSpec):
        return resolve_score_authority(
            spec,
            canonical_role,
            receipts=receipts,
            candidates=after_cost,
        )

    def _cap(spec: ModelSpec) -> int:
        return _authority(spec).effective

    def _sufficient(spec: ModelSpec) -> bool:
        return is_capability_sufficient(_authority(spec), need)

    def _overlay_note(spec: ModelSpec) -> str:
        return role_card_override_note(
            spec, canonical_role, receipts=receipts, candidates=after_cost
        )

    def _stale(spec: ModelSpec) -> int:
        return 1 if _authority(spec).decayed else 0

    def _quality_key(spec: ModelSpec):
        # Capability stays the quality number. Sibling decay is what makes
        # a stale editorial 90 lose to a live receipt; source_rank is not a
        # global first key (that would leak a receipt across adapters).
        auth = _authority(spec)
        return (
            _cap(spec),
            receipt_quality_tuple(auth),
            1 if spec.is_plan_billed else 0,
        )

    _baseline_model = max(after_cost, key=_cap)
    _baseline_tokens_in = _tokens_in_for(_baseline_model)
    _baseline_cost = _baseline_model.marginal_cost_usd(
        _baseline_tokens_in, tokens_out
    )
    _baseline_nominal = _baseline_model.estimate_cost_usd(
        _baseline_tokens_in, tokens_out
    )
    _baseline_id = _baseline_model.id

    # Tie-break helper: when ``prefer_plan_billed`` is on, a subscription-covered
    # model sorts ahead of an out-of-pocket one at equal cost/capability, so
    # spend stays inside the user's plan whenever a plan model is good enough.
    def _plan_rank(spec: ModelSpec) -> int:
        if not task.prefer_plan_billed:
            return 0
        return 0 if spec.is_plan_billed else 1

    def _routing_cost(spec: ModelSpec) -> float:
        return spec.routing_cost_usd(_tokens_in_for(spec), tokens_out)

    sufficient = [s for s in after_cost if _sufficient(s)]
    if task.strict_capability and not sufficient:
        strongest = max(after_cost, key=_cap)
        raise NoEligibleModelError(
            "Strict capability requirement failed: no eligible model meets "
            f"capability {need}; strongest available is {strongest.id} "
            f"at {_cap(strongest)}."
        )

    prefer = (task.prefer_model_id or "").strip()
    if (
        prefer
        and cache_affinity_enabled()
        and policy in CACHE_AFFINITY_POLICIES
        and sufficient
    ):
        wanted = frozenset([prefer])
        sticky = None
        for spec in sufficient:
            if spec.id == prefer or model_id_allowed(spec, wanted):
                sticky = spec
                break
        if sticky is not None:
            pick = sticky
            reason = (
                "cache affinity: prefer_model_id="
                f"{pick.id} still meets capability need ({need}); "
                "keep shared job-brief prefix"
                + _overlay_note(pick)
            )
            for spec in after_cost:
                if spec.id != pick.id:
                    rejected.append(
                        (spec, f"cache affinity kept sibling {pick.id}")
                    )
            return _decision(
                pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
                _baseline_cost, _baseline_id, _baseline_nominal,
                allowed_model_ids=_allowed_for_artifact,
                role=task.role,
                calibration=_candidate_calibration(pick),
                local_receipts=receipts,
                candidates=after_cost,
            )

    if policy == "cheap":
        # ``cheap`` means cheapest *sufficient*.  The old unconditional-lowest
        # behavior remains available only through the conspicuous
        # ``absolute-cheapest`` opt-in below.
        if sufficient:
            pick = min(
                sufficient,
                key=lambda s: (
                    _stale(s),
                    _routing_cost(s),
                    _plan_rank(s),
                    _cap(s),
                ),
            )
            reason = (
                "policy=cheap: cheapest sufficient model whose "
                f"capability_score ({_cap(pick)}) >= needed ({need})"
                + _overlay_note(pick)
            )
            for spec in after_cost:
                if spec.id == pick.id:
                    continue
                if not _sufficient(spec):
                    rejected.append(
                        (
                            spec,
                            f"capability_score {_cap(spec)} < needed {need}"
                            + _overlay_note(spec),
                        )
                    )
                else:
                    rejected.append(
                        (spec, f"cheapest sufficient alternative {pick.id} chosen")
                    )
        else:
            # Preserve the established compatible fallback when strict mode is
            # not requested, but make the capability gap unmistakable.
            pick = max(after_cost, key=_quality_key)
            reason = (
                f"policy=cheap: NO model meets capability need ({need}); "
                "falling back to highest-capability available "
                f"({pick.id} @ {_cap(pick)})"
                + _overlay_note(pick)
            )
            for spec in after_cost:
                if spec.id != pick.id:
                    rejected.append((spec, f"lower capability_score {_cap(spec)}"))
        return _decision(
            pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
            _baseline_cost, _baseline_id, _baseline_nominal,
            allowed_model_ids=_allowed_for_artifact,
            role=task.role,
            calibration=_candidate_calibration(pick),
            local_receipts=receipts,
            candidates=after_cost,
        )

    if policy == "absolute-cheapest":
        absolute_pool = sufficient if task.strict_capability else after_cost
        pick = min(
            absolute_pool,
            key=lambda s: (
                _routing_cost(s),
                _plan_rank(s),
                _cap(s),
            ),
        )
        reason = (
            "policy=absolute-cheapest: explicit opt-in to absolute lowest "
            "nominal per-call usage cost"
            + (
                "; strict capability floor also enforced"
                if task.strict_capability
                else ""
            )
            + _overlay_note(pick)
        )
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append((spec, f"absolute-cheapest alternative {pick.id} chosen"))
        return _decision(
            pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
            _baseline_cost, _baseline_id, _baseline_nominal,
            allowed_model_ids=_allowed_for_artifact,
            role=task.role,
            calibration=_candidate_calibration(pick),
            local_receipts=receipts,
            candidates=after_cost,
        )

    if policy == "quality":
        # Highest capability wins; plan-billed breaks ties so we don't reach
        # for an out-of-pocket model when an equally-capable plan one exists.
        pick = max(after_cost, key=_quality_key)
        reason = (
            "policy=quality: highest capability_score"
            + _overlay_note(pick)
        )
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append((spec, f"higher-capability {pick.id} chosen"))
        return _decision(
            pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
            _baseline_cost, _baseline_id, _baseline_nominal,
            allowed_model_ids=_allowed_for_artifact,
            role=task.role,
            calibration=_candidate_calibration(pick),
            local_receipts=receipts,
            candidates=after_cost,
        )

    if policy == "escalating":
        # Start with the cheapest *sufficient* model, then list the remaining
        # models as an ordered escalation chain. The pick must be chosen among
        # models that actually clear the capability bar — sorting the whole set
        # by capability first would let a barely-sufficient but expensive model
        # win over a cheaper higher-capability one, contradicting "cheapest
        # sufficient". So filter to sufficient first, then order by
        # (plan_rank, nominal usage cost, capability_score).
        def _escalation_key(spec: ModelSpec):
            return (
                _plan_rank(spec),
                _routing_cost(spec),
                _cap(spec),
            )

        sufficient = sorted(
            (s for s in after_cost if _sufficient(s)),
            key=_escalation_key,
        )
        if sufficient:
            pick = sufficient[0]
        else:
            # Nothing clears the bar; escalate to the strongest available.
            pick = max(after_cost, key=_quality_key)
        reason = (
            "policy=escalating: start with cheapest sufficient; "
            "rejected list is the ordered escalation chain"
            + _overlay_note(pick)
        )
        # Order the escalation chain cheapest-first so the orchestrator retries
        # up the ladder, with any insufficient models trailing the sufficient
        # ones (they only get tried after every sufficient option is exhausted).
        escalation_order = sorted(
            after_cost,
            key=lambda s: (0 if _cap(s) >= need else 1, *_escalation_key(s)),
        )
        for spec in escalation_order:
            if spec.id != pick.id:
                rejected.append((spec, "escalation candidate"))
        return _decision(
            pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
            _baseline_cost, _baseline_id, _baseline_nominal,
            allowed_model_ids=_allowed_for_artifact,
            role=task.role,
            calibration=_candidate_calibration(pick),
            local_receipts=receipts,
            candidates=after_cost,
        )

    # balanced (default)
    if sufficient:
        # Pick the cheapest sufficient model using nominal usage rates. This
        # keeps the router cost-aware even when Cursor reports every
        # subscription-covered call as $0 marginal spend.
        pick = min(
            sufficient,
            key=lambda s: (
                _stale(s),
                _plan_rank(s),
                _routing_cost(s),
                _cap(s),
            ),
        )
        plan_note = (
            " (plan-billed, in-subscription)" if pick.is_plan_billed else ""
        )
        reason = (
            f"policy=balanced: cheapest sufficient model whose capability_score "
            f"({_cap(pick)}) >= needed ({need}){plan_note}"
            f"{_overlay_note(pick)}"
        )
        pick_cost = _routing_cost(pick)
        for spec in after_cost:
            if spec.id != pick.id:
                if not _sufficient(spec):
                    rejected.append(
                        (
                            spec,
                            f"capability_score {_cap(spec)} < needed {need}"
                            + _overlay_note(spec),
                        )
                    )
                else:
                    spec_cost = _routing_cost(spec)
                    if spec_cost > pick_cost:
                        rejected.append(
                            (
                                spec,
                                f"sufficient capability but pricier than {pick.id} "
                                f"(${spec_cost:.4f} vs ${pick_cost:.4f})",
                            )
                        )
                    else:
                        # Same estimated cost as the pick — the tie-break is
                        # capability right-sizing: prefer the lower
                        # capability_score so frontier models stay reserved
                        # for tasks that actually need them.
                        rejected.append(
                            (
                                spec,
                                f"same estimated cost as {pick.id} "
                                f"(${spec_cost:.4f}) but higher capability than "
                                f"needed; {pick.id} is right-sized for need {need}",
                            )
                        )
    else:
        # Nothing meets the bar; surface the best we have rather than
        # silently failing — but the reason makes the gap obvious.
        pick = max(after_cost, key=_quality_key)
        reason = (
            f"policy=balanced: NO model meets capability need ({need}); "
            f"falling back to highest-capability available "
            f"({pick.id} @ {_cap(pick)}). Consider adding a stronger "
            f"model to your registry or lowering payload.min_capability."
            f"{_overlay_note(pick)}"
        )
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append(
                    (spec, f"lower capability_score {_cap(spec)}")
                )
    return _decision(
        pick, policy, need, _tokens_in_for(pick), tokens_out, reason, rejected,
        _baseline_cost, _baseline_id, _baseline_nominal,
        allowed_model_ids=_allowed_for_artifact,
        role=task.role,
        calibration=_candidate_calibration(pick),
        local_receipts=receipts,
        candidates=after_cost,
    )


def route_task(
    task: TaskSignals,
    registry: Iterable[ModelSpec],
    *,
    policy: Optional[str] = None,
    calibration: Optional[TokenEstimateCalibration] = None,
    shadow_policy: Optional[str] = None,
    local_receipts: Optional[Iterable] = None,
) -> RoutingDecision:
    """Pick the production model and optionally attach counterfactual evidence.

    Shadow routing is opt-in and non-interfering: the production decision is
    computed first and returned unchanged except for an evidence-only payload.
    The counterfactual route cannot replace its model, adapter, or policy.
    """
    models = tuple(registry)
    production = _route_task_once(
        task,
        models,
        policy=policy,
        calibration=calibration,
        local_receipts=local_receipts,
    )
    if shadow_policy is None:
        return production
    if shadow_policy not in VALID_POLICIES:
        raise ValueError(
            f"unknown shadow_policy {shadow_policy!r}; expected one of {VALID_POLICIES}"
        )
    counterfactual = _route_task_once(
        task,
        models,
        policy=shadow_policy,
        calibration=calibration,
        local_receipts=local_receipts,
    )
    from puppetmaster.shadow_routing import shadow_evidence

    return replace(
        production,
        shadow_routing=shadow_evidence(
            production_model_id=production.model.id,
            counterfactual_model_id=counterfactual.model.id,
            policy=shadow_policy,
        ),
    )


def _effective_allowed_model_ids(task: TaskSignals) -> Optional[frozenset[str]]:
    """Resolve the effective model allowlist for a routing decision.

    Per-task ``TaskSignals.allowed_model_ids`` wins. When unset, honor a
    global ``allowed_model_ids`` / ``allowed_models`` list from
    ``~/.pmharness/routing.json`` so operators can constrain every auto-route
    without threading the flag on every worker payload.
    """
    if task.allowed_model_ids is not None:
        return frozenset(
            str(item).strip()
            for item in task.allowed_model_ids
            if str(item).strip()
        )
    saved = _load_routing_overrides()
    raw = saved.get("allowed_model_ids")
    if raw is None:
        raw = saved.get("allowed_models")
    if raw is None:
        return None
    return frozenset(
        item.strip() for item in _coerce_str_list(raw) if item.strip()
    )


def _decision_score_fields(
    pick: ModelSpec,
    role: str,
    *,
    receipts: Optional[Iterable] = None,
    candidates: Optional[Iterable[ModelSpec]] = None,
) -> dict:
    """Provenance fields for a pick; empty cards stay omitted-friendly."""
    authority = resolve_score_authority(
        pick, role, receipts=receipts, candidates=candidates
    )
    return {
        "effective_capability_score": authority.effective,
        "score_source": authority.source,
        "score_provenance": dict(authority.provenance),
        "sample_count": authority.sample_count,
        "predicted_quality": authority.predicted_quality,
        "predicted_latency_p50_ms": authority.predicted_latency_p50_ms,
    }


def _decision(
    pick: ModelSpec,
    policy: str,
    need: int,
    tokens_in: int,
    tokens_out: int,
    reason: str,
    rejected: list[tuple[ModelSpec, str]],
    baseline_cost_usd: float = 0.0,
    baseline_model_id: str = "",
    baseline_nominal_cost_usd: float = 0.0,
    allowed_model_ids: Optional[list[str]] = None,
    role: str = "",
    calibration: Optional[TokenEstimateCalibration] = None,
    local_receipts: Optional[Iterable] = None,
    candidates: Optional[Iterable[ModelSpec]] = None,
) -> RoutingDecision:
    canonical_role, _profile, taxonomy = _role_profile(role)
    score_fields = _decision_score_fields(
        pick,
        canonical_role,
        receipts=local_receipts,
        candidates=candidates,
    )
    return RoutingDecision(
        model=pick,
        policy=policy,
        capability_needed=need,
        estimated_tokens_in=tokens_in,
        estimated_tokens_out=tokens_out,
        estimated_cost_usd=pick.marginal_cost_usd(tokens_in, tokens_out),
        nominal_cost_usd=pick.estimate_cost_usd(tokens_in, tokens_out),
        reason=reason,
        rejected=rejected,
        baseline_cost_usd=baseline_cost_usd,
        baseline_nominal_cost_usd=baseline_nominal_cost_usd,
        baseline_model_id=baseline_model_id,
        allowed_model_ids=allowed_model_ids,
        role=role,
        canonical_role=canonical_role,
        taxonomy_version=taxonomy["taxonomy_version"],
        token_estimate_calibration=calibration,
        **score_fields,
    )


# ----- WorkerSpec -> TaskSignals helper -----------------------------------


_TRUE_STRINGS = {"true", "1", "yes"}
_FALSE_STRINGS = {"false", "0", "no"}


def _coerce_bool(value: object, default: bool) -> bool:
    """Strictly read a flag from payload.

    Accepts only real bools or the literal strings ``true``/``false``/``1``/
    ``0``/``yes``/``no`` (case-insensitive). Anything ambiguous — including
    ``bool('false') is True`` traps — falls back to ``default`` rather than
    silently flipping a billing/plan gate the wrong way.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default


def _coerce_str_list(value: object) -> list[str]:
    """Strictly read a list-of-strings from payload.

    A real ``list``/``tuple`` keeps its string elements; a single ``str`` is
    treated as one element (so ``"vision"`` does NOT explode into characters
    the way ``list("vision")`` would). Anything else yields an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _coerce_nonnegative_int(value: object, default: int = 0) -> int:
    """Read an integer token/count declaration without bool coercion traps."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(0, value)


def _nested_payload_string_chars(
    value: object, _seen: Optional[set[int]] = None
) -> int:
    """Count strings recursively in JSON-like payload context.

    Worker payloads are normally acyclic JSON values, but direct Python callers
    can supply self-referential containers.  Track container identities so
    estimation remains total and never recurses forever.
    """
    if isinstance(value, str):
        return len(value)
    if _seen is None:
        _seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in _seen:
            return 0
        _seen.add(identity)
        return sum(
            _nested_payload_string_chars(item, _seen) for item in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in _seen:
            return 0
        _seen.add(identity)
        return sum(_nested_payload_string_chars(item, _seen) for item in value)
    return 0


def signals_from_worker_spec(spec, *, instruction_override: Optional[str] = None) -> TaskSignals:
    """Build a :class:`TaskSignals` from a ``workers.WorkerSpec``.

    Honors per-task overrides in ``spec.payload``:

    * ``min_capability`` — int 0..100, forces classifier output
    * ``max_capability`` — int 0..100, ceiling on classifier output
    * ``max_cost_usd`` — float, hard cap
    * ``required_tags`` — list[str], all must be on the model's tags
    * ``allowed_model_ids`` / ``allowed_models`` — list[str], explicit model
      allowlist (registry ids or adapter model names)
    * ``estimated_tokens_in`` / ``estimated_tokens_out`` — override heuristic
    * ``adapter_enrichment_tokens`` — exact additive adapter prompt overhead
    * ``strict_capability`` — fail closed when no eligible model clears need
    """
    payload = getattr(spec, "payload", {}) or {}
    instruction = instruction_override or getattr(spec, "instruction", "") or ""
    payload_size_chars = _nested_payload_string_chars(payload)

    # A per-task override wins; otherwise inherit the user's platform lock.
    # A bare ``allowed_adapters: "openai"`` must mean the single adapter, not
    # frozenset('openai') == {'o','p','e','n','a','i'}.
    allowed = payload.get("allowed_adapters")
    if allowed is not None:
        allowed_adapters: Optional[frozenset[str]] = frozenset(_coerce_str_list(allowed))
    else:
        from puppetmaster.platform_lock import active_allowlist

        allowed_adapters = active_allowlist()

    raw_allowed_models = payload.get("allowed_model_ids")
    if raw_allowed_models is None:
        raw_allowed_models = payload.get("allowed_models")
    allowed_model_ids: Optional[frozenset[str]]
    if raw_allowed_models is not None:
        allowed_model_ids = frozenset(
            item.strip()
            for item in _coerce_str_list(raw_allowed_models)
            if item.strip()
        )
    else:
        allowed_model_ids = None

    return TaskSignals(
        instruction=instruction,
        role=getattr(spec, "role", "explore") or "explore",
        payload_size_chars=payload_size_chars,
        explicit_min_capability=payload.get("min_capability"),
        explicit_max_capability=payload.get("max_capability"),
        explicit_max_cost_usd=payload.get("max_cost_usd"),
        required_tags=_coerce_str_list(payload.get("required_tags")),
        estimated_tokens_in=payload.get("estimated_tokens_in"),
        estimated_tokens_out=payload.get("estimated_tokens_out"),
        adapter_enrichment_tokens=_coerce_nonnegative_int(
            payload.get("adapter_enrichment_tokens")
        ),
        strict_capability=_coerce_bool(payload.get("strict_capability"), False),
        prefer_plan_billed=_coerce_bool(payload.get("prefer_plan_billed"), True),
        allow_api_billing=_coerce_bool(payload.get("allow_api_billing"), True),
        allowed_adapters=allowed_adapters,
        allowed_model_ids=allowed_model_ids,
        prefer_model_id=(
            str(payload.get("prefer_model_id") or "").strip() or None
        ),
    )
