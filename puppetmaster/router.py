"""Task-aware model router.

Given a task and a user-owned ``ModelSpec`` registry, decide which
model to invoke. The router is built around three pillars:

1. **Transparent classification.** A pure-function heuristic assigns a
   capability-needed score 0..100 to each task, based on role,
   instruction length, and content signals. The score is recorded on
   the routing artifact so users can see *why* a task went where.
2. **User-controlled policy.** ``balanced`` (default), ``cheap``,
   ``quality``, and ``escalating`` are the built-in policies. Users
   pin per-task overrides via ``payload.min_capability``,
   ``payload.max_cost_usd``, and ``payload.required_tags``.
3. **Auditable decisions.** Every routing decision lists the rejected
   alternatives and *why* each was rejected. No black boxes.

This module deliberately does **not** call any LLM. It picks specs;
the adapter actually runs the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from puppetmaster.model_registry import ModelSpec, enabled_specs


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
    explicit_max_cost_usd: Optional[float] = None
    required_tags: list[str] = field(default_factory=list)
    estimated_tokens_in: Optional[int] = None
    estimated_tokens_out: Optional[int] = None
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


# ----- Classifier ----------------------------------------------------------


_ROLE_BASE_SCORE = {
    # Cheap, deterministic checks. A 50-capability model is overkill.
    "verify-runtime": 25,
    "shell": 20,
    "demo": 25,
    # Read-only exploration. Mid-tier is fine.
    "explore": 50,
    "review": 55,
    "plan": 60,
    # Anything that writes code or proposes patches. Need real ability.
    "implement": 75,
    "refactor": 75,
    "patch": 75,
    "fix": 70,
    "test-coverage-reviewer": 60,
    # Hard reasoning. Reserve frontier models.
    "architect": 85,
    "audit": 85,
    "security-review": 90,
    "decision-explainer": 70,
    "conflict-auditor": 75,
    "pipeline-mapper": 65,
}

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
    ``task.explicit_min_capability`` (which we honor without modification).
    """
    if task.explicit_min_capability is not None:
        return max(0, min(100, task.explicit_min_capability))

    score = _ROLE_BASE_SCORE.get(task.role, 50)

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
    return max(5, min(100, score))


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


# ----- Token estimation ----------------------------------------------------


def estimate_tokens_in(task: TaskSignals) -> int:
    if task.estimated_tokens_in is not None:
        return task.estimated_tokens_in
    # ~4 chars/token is the standard rough heuristic.
    text_chars = len(task.instruction) + task.payload_size_chars
    return max(500, text_chars // 4 + 500)  # +500 for system + tools overhead


def estimate_tokens_out(task: TaskSignals) -> int:
    if task.estimated_tokens_out is not None:
        return task.estimated_tokens_out
    # Output budgets by role. Pure heuristic, override per task if needed.
    by_role = {
        "verify-runtime": 300,
        "shell": 200,
        "demo": 500,
        "explore": 1500,
        "review": 1500,
        "plan": 2000,
        "implement": 3000,
        "refactor": 3000,
        "patch": 3000,
        "architect": 5000,
        "audit": 5000,
        "security-review": 5000,
    }
    return by_role.get(task.role, 1500)


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
    # Savings accounting (Rule 1: snapshot the baseline at decision time so the
    # ledger never compares a stored cost against a recomputed/drifted one).
    # ``baseline`` = what this task would have cost on the strongest model the
    # user could have used (highest-capability enabled + platform-allowed),
    # at the same token estimate.
    baseline_cost_usd: float = 0.0
    baseline_model_id: str = ""

    def to_artifact_payload(self) -> dict:
        return {
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
            "baseline_cost_usd": round(self.baseline_cost_usd, 6),
            "baseline_model_id": self.baseline_model_id,
            "reason": self.reason,
            "rejected": [
                {"id": spec.id, "reason": why} for spec, why in self.rejected
            ],
        }


class NoEligibleModelError(RuntimeError):
    """Raised when the policy + constraints exclude every registered model."""


VALID_POLICIES = {"balanced", "cheap", "quality", "escalating"}


def route_task(
    task: TaskSignals,
    registry: Iterable[ModelSpec],
    *,
    policy: str = "balanced",
) -> RoutingDecision:
    """Pick a model for ``task`` from ``registry`` using ``policy``.

    Raises :class:`NoEligibleModelError` for *hard* constraint failures: an
    empty/all-disabled registry, a platform lock that excludes every model, or
    a ``max_cost_usd`` cap nothing can satisfy.

    Capability is treated as a *soft* preference, not a hard gate: when no model
    meets the needed capability score, ``balanced`` deliberately falls back to
    the highest-capability model available (with a reason that makes the gap
    explicit) rather than raising — so a task still runs on the best option on
    hand instead of failing closed. Use ``payload.min_capability`` /
    ``payload.required_tags`` if you need a hard floor.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {VALID_POLICIES}")

    candidates = enabled_specs(registry)
    if not candidates:
        raise NoEligibleModelError(
            "No enabled models in registry. Run `puppetmaster models init` "
            "to write a starter ~/.puppetmaster/models.json, then edit it."
        )

    tokens_in = estimate_tokens_in(task)
    tokens_out = estimate_tokens_out(task)
    need = classify_capability_needed(task)

    # Auto-add vision tags when the instruction needs vision. The user
    # can still pin explicit tags via TaskSignals.required_tags; we
    # union with the auto-detected ones so explicit choices stay in.
    effective_required_tags = set(task.required_tags)
    if has_vision_signal(task.instruction):
        effective_required_tags.add("vision")
    if has_detailed_vision_signal(task.instruction):
        effective_required_tags.add("detailed-vision")

    rejected: list[tuple[ModelSpec, str]] = []

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

    # Cost budget filter.
    after_cost: list[ModelSpec] = []
    for spec in after_tags:
        est = spec.estimate_cost_usd(tokens_in, tokens_out)
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

    # Snapshot the savings baseline from the strongest model that was *actually
    # eligible for this task* — i.e. the final candidate set the pick is drawn
    # from, after every hard constraint (platform lock, required tags, cost cap,
    # billing gate). Computing it from the same set the pick comes from is what
    # keeps the ledger honest: a constrained run can't be credited against (or
    # penalised by) a model it could never have run. Stored on the decision so
    # the ledger compares like-for-like later instead of recomputing against a
    # possibly-changed registry.
    _baseline_model = max(after_cost, key=lambda s: s.capability_score)
    _baseline_cost = _baseline_model.estimate_cost_usd(tokens_in, tokens_out)
    _baseline_id = _baseline_model.id

    # Tie-break helper: when ``prefer_plan_billed`` is on, a subscription-covered
    # model sorts ahead of an out-of-pocket one at equal cost/capability, so
    # spend stays inside the user's plan whenever a plan model is good enough.
    def _plan_rank(spec: ModelSpec) -> int:
        if not task.prefer_plan_billed:
            return 0
        return 0 if spec.is_plan_billed else 1

    if policy == "cheap":
        # Tie-break: plan-billed first (no marginal spend), then lower
        # capability_score (reserve big models for tasks that need them;
        # "cheap" implies "small").
        pick = min(
            after_cost,
            key=lambda s: (
                s.estimate_cost_usd(tokens_in, tokens_out),
                _plan_rank(s),
                s.capability_score,
            ),
        )
        reason = "policy=cheap: lowest per-call estimated cost"
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append(
                    (spec, f"cheaper alternative {pick.id} chosen"),
                )
        return _decision(pick, policy, need, tokens_in, tokens_out, reason, rejected, _baseline_cost, _baseline_id)

    if policy == "quality":
        # Highest capability wins; plan-billed breaks ties so we don't reach
        # for an out-of-pocket model when an equally-capable plan one exists.
        pick = max(
            after_cost,
            key=lambda s: (s.capability_score, 1 if s.is_plan_billed else 0),
        )
        reason = "policy=quality: highest capability_score"
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append((spec, f"higher-capability {pick.id} chosen"))
        return _decision(pick, policy, need, tokens_in, tokens_out, reason, rejected, _baseline_cost, _baseline_id)

    if policy == "escalating":
        # For escalating we still return one decision (the cheapest
        # sufficient model), but ordered alternatives are listed in
        # `rejected` so the orchestrator can retry up the chain.
        sorted_by_cap = sorted(
            after_cost,
            key=lambda s: (
                s.capability_score,
                _plan_rank(s),
                s.estimate_cost_usd(tokens_in, tokens_out),
            ),
        )
        sufficient = [s for s in sorted_by_cap if s.capability_score >= need]
        pick = sufficient[0] if sufficient else sorted_by_cap[-1]
        reason = (
            "policy=escalating: start with cheapest sufficient; "
            "rejected list is the ordered escalation chain"
        )
        for spec in sorted_by_cap:
            if spec.id != pick.id:
                rejected.append((spec, "escalation candidate"))
        return _decision(pick, policy, need, tokens_in, tokens_out, reason, rejected, _baseline_cost, _baseline_id)

    # balanced (default)
    sufficient = [s for s in after_cost if s.capability_score >= need]
    if sufficient:
        # Tie-break: when several sufficient models cost the same
        # (e.g. multiple $0 Cursor-plan models), pick the LOWEST
        # capability_score among them — that's the right-sized model
        # for the task. Picking the highest would waste capability.
        pick = min(
            sufficient,
            key=lambda s: (
                _plan_rank(s),
                s.estimate_cost_usd(tokens_in, tokens_out),
                s.capability_score,
            ),
        )
        plan_note = (
            " (plan-billed, in-subscription)" if pick.is_plan_billed else ""
        )
        reason = (
            f"policy=balanced: cheapest sufficient model whose capability_score "
            f"({pick.capability_score}) >= needed ({need}){plan_note}"
        )
        pick_cost = pick.estimate_cost_usd(tokens_in, tokens_out)
        for spec in after_cost:
            if spec.id != pick.id:
                if spec.capability_score < need:
                    rejected.append(
                        (
                            spec,
                            f"capability_score {spec.capability_score} < needed {need}",
                        )
                    )
                else:
                    spec_cost = spec.estimate_cost_usd(tokens_in, tokens_out)
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
        pick = max(after_cost, key=lambda s: s.capability_score)
        reason = (
            f"policy=balanced: NO model meets capability need ({need}); "
            f"falling back to highest-capability available "
            f"({pick.id} @ {pick.capability_score}). Consider adding a stronger "
            f"model to your registry or lowering payload.min_capability."
        )
        for spec in after_cost:
            if spec.id != pick.id:
                rejected.append(
                    (spec, f"lower capability_score {spec.capability_score}")
                )
    return _decision(pick, policy, need, tokens_in, tokens_out, reason, rejected, _baseline_cost, _baseline_id)


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
) -> RoutingDecision:
    return RoutingDecision(
        model=pick,
        policy=policy,
        capability_needed=need,
        estimated_tokens_in=tokens_in,
        estimated_tokens_out=tokens_out,
        estimated_cost_usd=pick.estimate_cost_usd(tokens_in, tokens_out),
        reason=reason,
        rejected=rejected,
        baseline_cost_usd=baseline_cost_usd,
        baseline_model_id=baseline_model_id,
    )


# ----- WorkerSpec -> TaskSignals helper -----------------------------------


def signals_from_worker_spec(spec, *, instruction_override: Optional[str] = None) -> TaskSignals:
    """Build a :class:`TaskSignals` from a ``workers.WorkerSpec``.

    Honors per-task overrides in ``spec.payload``:

    * ``min_capability`` — int 0..100, forces classifier output
    * ``max_cost_usd`` — float, hard cap
    * ``required_tags`` — list[str], all must be on the model's tags
    * ``estimated_tokens_in`` / ``estimated_tokens_out`` — override heuristic
    """
    payload = getattr(spec, "payload", {}) or {}
    instruction = instruction_override or getattr(spec, "instruction", "") or ""
    payload_str = ""
    for value in payload.values():
        if isinstance(value, str):
            payload_str += value

    # A per-task override wins; otherwise inherit the user's platform lock.
    allowed = payload.get("allowed_adapters")
    if allowed is not None:
        allowed_adapters: Optional[frozenset[str]] = frozenset(allowed)
    else:
        from puppetmaster.platform_lock import active_allowlist

        allowed_adapters = active_allowlist()

    return TaskSignals(
        instruction=instruction,
        role=getattr(spec, "role", "explore") or "explore",
        payload_size_chars=len(payload_str),
        explicit_min_capability=payload.get("min_capability"),
        explicit_max_cost_usd=payload.get("max_cost_usd"),
        required_tags=list(payload.get("required_tags") or []),
        estimated_tokens_in=payload.get("estimated_tokens_in"),
        estimated_tokens_out=payload.get("estimated_tokens_out"),
        prefer_plan_billed=bool(payload.get("prefer_plan_billed", True)),
        allow_api_billing=bool(payload.get("allow_api_billing", True)),
        allowed_adapters=allowed_adapters,
    )
