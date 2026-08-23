from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import AbstractSet, Callable, Mapping, Optional

from puppetmaster.adapters import get_adapter, verification_artifact
from puppetmaster.models import AgentRun, Artifact, Task, TaskStatus, now_iso

# Adapters that bill an LLM provider and therefore benefit from a pre-dispatch
# auth/billing gate. ``local``/``shell`` run no model, so they're never gated.
_PREFLIGHTABLE_ADAPTERS = {
    "agentic",
    "cursor",
    "claude-code",
    "codex",
    "openai",
}

# Failure classes that are about *this adapter's account* (auth, billing,
# quota, missing CLI/key, or a preflight block) rather than the task itself.
# When a worker hits one of these, the right move is to fail the task
# truthfully and re-route it to a different funded adapter — not to bury a
# COMPLETE status over a doomed run, and not to nuke the whole job. The
# orchestrator's auto-fallback loop keys off this set; the worker runtime
# uses it to convert an adapter-detected billing/auth rejection into a
# truthful FAILED task status.
RECOVERABLE_FAILURES = frozenset(
    {
        "billing_or_quota",
        "preflight_blocked",
        "missing_api_key",
        "not_authenticated",
        "missing_cli",
        "auth",
        "authentication",
        "model_unavailable",
        # Cursor SDK terminal status:error — often a model/plan rejection that
        # surfaces without a clearer taxonomy. Treat as recoverable so one
        # same-adapter alternate can be tried for auto-routed work.
        "run_status_error",
        # Dispatch-config failures: the task reached an adapter that cannot
        # run it (no routed model / unknown provider slug / missing SDK
        # package). These are about the routing/config, never the task --
        # without them here a worker's instant fast-fail was recorded
        # COMPLETE and a fully dead swarm rendered as a healthy green
        # "done" run at $0 (field reports: no_model from a credential-less
        # router, then sdk_not_installed from a cursor worker in an env
        # without @cursor/sdk).
        "no_model",
        "unknown_provider",
        "sdk_not_installed",
        # Transient provider pressure after in-adapter key retries; prefer a
        # different model (often same adapter) before dying the task.
        "rate_limit",
        # Model (or endpoint) never emitted tool_calls — not a task bug; the
        # next eligible tool-capable model may recover.
        "no_tool_calls",
    }
)

# Failures where the *model* (not the adapter account) is the problem — allow
# same-adapter re-route to another registry model when no other platform is
# funded. Billing/auth failures still require a different adapter.
SAME_ADAPTER_MODEL_REROUTE = frozenset(
    {
        "model_unavailable",
        "run_status_error",
        "no_model",
        "unknown_provider",
        "rate_limit",
        "no_tool_calls",
    }
)


@dataclass(frozen=True)
class RoleSpec:
    """Structured analysis assignment with explicit scope boundaries."""

    name: str
    instruction: str
    source_scope: Optional[list[str]] = None
    negative_scope: Optional[list[str]] = None


@dataclass(frozen=True)
class WorkerSpec:
    role: str
    instruction: str
    adapter: str = "local"
    payload: dict = field(default_factory=dict)
    depends_on_roles: list[str] = field(default_factory=list)


def normalize_role_specs(roles: Optional[list[object]], goal: str) -> tuple[list[RoleSpec], bool]:
    """Normalize legacy role names and structured assignments once.

    Returns ``(specs, duplicated_legacy_roles)``.  A missing role list is one
    analysis assignment; callers that explicitly pass several bare names get a
    visible warning because those names do not decompose a shared goal.
    """
    if not roles:
        return [RoleSpec("analysis", goal)], False
    structured: list[RoleSpec] = []
    legacy_count = 0
    for raw in roles:
        if isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("role") or "").strip()
            instruction = str(raw.get("instruction") or raw.get("goal") or "").strip()
            if not name or not instruction:
                raise ValueError("structured roles require name and instruction")
            source = raw.get("source_scope") or raw.get("sources")
            negative = raw.get("negative_scope") or raw.get("exclude")
            structured.append(
                RoleSpec(
                    name,
                    instruction,
                    [str(item) for item in source] if isinstance(source, list) else None,
                    [str(item) for item in negative] if isinstance(negative, list) else None,
                )
            )
            continue
        name = str(raw).strip()
        if name:
            legacy_count += 1
            structured.append(RoleSpec(name, goal))
    return structured, legacy_count > 1 and len(structured) > 1


# Shared by DEFAULT_WORKERS and write_generated_swarm_config so analysis
# swarms keep read-only intent when auto_route lands on an edit-capable
# adapter (claude-code → permission_mode=plan; codex → sandbox read-only).
# Without these, a Claude-only lock routing the "implement" *planning*
# role to claude-code incorrectly takes acceptEdits and trips worktree
# / dirty-tree guards.
ANALYSIS_NO_EDIT_PAYLOAD = {
    "read_only": True,
    "sandbox": "read-only",
    "dangerously_bypass_approvals_and_sandbox": False,
}

# Built-in worker specs ship with ``auto_route: True`` so any swarm
# started via the MCP ``puppetmaster_start_*`` tools or ``python -m
# puppetmaster run`` consults the router and lets each role land on
# the appropriate tier (cheap for explore/test, mid for architect,
# frontier for redteam/audit). If the user has not run
# ``puppetmaster models init``, the orchestrator quietly skips routing
# and falls back to the spec's declared adapter — opting in is safe.
_DEFAULT_AUTO_ROUTE_PAYLOAD = {
    "auto_route": True,
    **ANALYSIS_NO_EDIT_PAYLOAD,
}

# Per-role routing hints for analysis swarms (OMP ``modelRoles.task``
# analogue). Policies only — never pin frontier model ids. Unknown roles
# get no stamp; the orchestrator then falls back to ``balanced``.
DEFAULT_ROLE_ROUTING_POLICY: dict[str, str] = {
    "explore": "cheap",
    "test": "cheap",
    "architect": "balanced",
    "plan": "balanced",
    "implement": "balanced",  # analysis *plan* role in DEFAULT_WORKERS
    "redteam": "quality",
    "review": "quality",
    "audit": "quality",
}


def default_routing_policy_for_role(role: str) -> Optional[str]:
    """Return the built-in ``routing_policy`` for an analysis swarm role, if any."""
    return DEFAULT_ROLE_ROUTING_POLICY.get(role)


def analysis_auto_route_payload(role: str) -> dict:
    """Default auto-route payload for an analysis swarm role.

    Always preserves ``auto_route=True`` and ``ANALYSIS_NO_EDIT_PAYLOAD``.
    Stamps ``routing_policy`` from ``DEFAULT_ROLE_ROUTING_POLICY`` when the
    role is known so explore/test stay cheap and review/redteam stay quality
    without pinning models.
    """
    payload = dict(_DEFAULT_AUTO_ROUTE_PAYLOAD)
    policy = default_routing_policy_for_role(role)
    if policy:
        payload["routing_policy"] = policy
    return payload


DEFAULT_WORKERS = [
    WorkerSpec(
        role="explore",
        instruction="Map the problem, extract constraints, and emit evidenced findings.",
        payload=analysis_auto_route_payload("explore"),
    ),
    WorkerSpec(
        role="architect",
        instruction="Choose the smallest viable architecture and record explicit decisions.",
        payload=analysis_auto_route_payload("architect"),
        depends_on_roles=["explore"],
    ),
    WorkerSpec(
        # NB: this analysis-swarm role produces an implementation *plan*
        # (decision + patch-plan artifacts), not edits. The swarm runs
        # read-only; use an implement verb / an edit-capable adapter to land
        # real code. The startup mode banner makes this explicit at runtime.
        role="implement",
        instruction=(
            "Produce an implementation plan as artifacts (decisions + patch "
            "plan), not prose blobs. This is an analysis role: do not expect "
            "files to be edited."
        ),
        payload=analysis_auto_route_payload("implement"),
        depends_on_roles=["architect"],
    ),
    WorkerSpec(
        role="redteam",
        instruction=(
            "Adversarially review the repository's code for real failure modes, "
            "stale assumptions, and missing verification, citing concrete "
            "files/functions. Analyze the code itself — never treat your own "
            "instructions or the artifact contract as the subject. If the "
            "codebase is small or sound and you find no real weakness, return an "
            "empty result rather than inventing one."
        ),
        payload=analysis_auto_route_payload("redteam"),
        depends_on_roles=["implement"],
    ),
    WorkerSpec(
        role="test",
        instruction="Convert claims into checks and verification results.",
        payload=analysis_auto_route_payload("test"),
        depends_on_roles=["implement"],
    ),
]


class LocalWorker:
    """Task executor used by real worker processes."""

    def __init__(self, role: str, worker_id: Optional[str] = None) -> None:
        self.role = role
        self.worker_id = worker_id or f"local-{role}"

    def run(self, task: Task, goal: str) -> tuple[AgentRun, list[Artifact]]:
        from puppetmaster.model_registry import AmbiguousModelPinError

        try:
            task = self._normalize_cursor_pin(task)
        except AmbiguousModelPinError as exc:
            run = AgentRun(
                job_id=task.job_id,
                task_id=task.id,
                role=task.role,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                completed_at=now_iso(),
            )
            blocked = verification_artifact(
                task=task,
                worker_id=self.worker_id,
                adapter=task.adapter,
                check=f"preflight: can {task.adapter} run this task?",
                result="blocked",
                confidence=0.95,
                evidence=[f"adapter:{task.adapter}", "preflight:ambiguous_model_pin"],
                payload={
                    "failure": "preflight_blocked",
                    "reason": f"ambiguous model pin: {exc}",
                    "model": (task.payload or {}).get("model"),
                },
            )
            return run, [blocked]

        blocked = self._preflight(task)
        if blocked is not None:
            run = AgentRun(
                job_id=task.job_id,
                task_id=task.id,
                role=task.role,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                completed_at=now_iso(),
            )
            return run, [blocked]

        run = AgentRun(
            job_id=task.job_id,
            task_id=task.id,
            role=task.role,
            worker_id=self.worker_id,
            status=TaskStatus.COMPLETE,
            completed_at=now_iso(),
        )
        if task.adapter == "cursor":
            return run, get_adapter("cursor").run(task, goal, self.worker_id)
        if task.adapter == "shell":
            return run, get_adapter("shell").run(task, goal, self.worker_id)
        return run, get_adapter(task.adapter).run(task, goal, self.worker_id)

    def _normalize_cursor_pin(self, task: Task) -> Task:
        """Stamp an explicit Cursor pin before preflight and adapter dispatch."""
        payload = task.payload or {}
        model = payload.get("model")
        if task.adapter != "cursor" or not model or payload.get("pinned_model"):
            return task
        from puppetmaster.model_registry import apply_cursor_model_pin

        return replace(task, payload=apply_cursor_model_pin(dict(payload), str(model)))

    def _preflight(self, task: Task) -> Optional[Artifact]:
        """Fast, no-network auth/billing gate before dispatching a paid worker.

        Catches the cheap-to-detect failures up front — missing CLI/key,
        unauthenticated adapter, or an api-billed adapter when the task forbids
        out-of-pocket spend — so we emit a clear blocked artifact instead of
        burning a dispatch. Catalog validation and live token probes are
        deliberately skipped here to keep dispatch latency near zero; opt out
        entirely with ``payload.skip_preflight``.
        """
        if task.payload.get("skip_preflight"):
            return None
        if task.adapter not in _PREFLIGHTABLE_ADAPTERS:
            return None

        from puppetmaster.model_registry import AmbiguousModelPinError
        from puppetmaster.preflight import preflight_check

        payload = task.payload or {}
        model = payload.get("model")
        catalog_fetcher = None
        if task.adapter == "cursor" and (
            model or payload.get("live_preflight")
        ):
            from puppetmaster.cursor_discovery import fetch_cursor_catalog

            catalog_fetcher = fetch_cursor_catalog
        try:
            result = preflight_check(
                task.adapter,
                model,
                allow_api_billing=bool(payload.get("allow_api_billing", True)),
                live=bool(payload.get("live_preflight")),
                catalog_fetcher=catalog_fetcher,
            )
        except AmbiguousModelPinError as exc:
            return verification_artifact(
                task=task,
                worker_id=self.worker_id,
                adapter=task.adapter,
                check=f"preflight: can {task.adapter} run this task?",
                result="blocked",
                confidence=0.95,
                evidence=[f"adapter:{task.adapter}", "preflight:ambiguous_model_pin"],
                payload={
                    "failure": "preflight_blocked",
                    "reason": f"ambiguous model pin: {exc}",
                    "model": model,
                },
            )
        if result.ok:
            return None
        return verification_artifact(
            task=task,
            worker_id=self.worker_id,
            adapter=task.adapter,
            check=f"preflight: can {task.adapter} run this task?",
            result="blocked",
            confidence=0.95,
            evidence=[f"adapter:{task.adapter}", *result.evidence],
            payload={
                "failure": "preflight_blocked",
                "reason": result.reason,
                "billing": result.billing,
                "model": model,
            },
        )


# Adapters that can actually modify the working tree. ``local``/``shell`` and
# the analyze-only paths (cursor without implement mode, openai) only ever emit
# artifacts, so a swarm built solely from those is read-only no matter how its
# roles are named — including the analysis swarm's "implement" role, which
# writes an implementation *plan*, not code.
_EDIT_CAPABLE_ADAPTERS = frozenset({"agentic", "claude-code", "codex"})


def spec_explicitly_no_edit(spec: WorkerSpec) -> bool:
    """True when a worker payload declares an adapter-independent no-edit run."""
    payload = spec.payload or {}
    return bool(
        payload.get("read_only")
        or payload.get("no_edit")
        or payload.get("dry_run")
    )


def spec_edits_files(spec: WorkerSpec) -> bool:
    """True when ``spec`` can leave file changes behind (vs. emit-only)."""
    payload = spec.payload or {}
    if spec_explicitly_no_edit(spec):
        return False
    if payload.get("mode") == "implement" or payload.get("implement"):
        return True
    return spec.adapter in _EDIT_CAPABLE_ADAPTERS


def swarm_mode(specs: list[WorkerSpec]) -> str:
    """Classify a swarm as ``"edit"`` (some worker can change files) or
    ``"analysis"`` (read-only — emits artifacts only)."""
    return "edit" if any(spec_edits_files(spec) for spec in specs) else "analysis"


def spec_has_side_effects(spec: WorkerSpec) -> bool:
    """True when a worker acts on the world *beyond the repository*.

    A browser worker drives a live site — navigation, logins, form fills — so it
    is read-only on the *repo* (``spec_edits_files`` is False, ``swarm_mode``
    stays ``"analysis"``, no clean-tree guard) yet is an **acting agent** with
    external side effects. Treat it with an implement-style approval posture, not
    the swarm's "this is just read-only analysis" assumption. Keyed off an
    explicit ``payload.side_effecting`` flag, with a defensive fallback that also
    catches a hand-rolled spec that wired the ``browser`` toolset directly.
    """
    payload = spec.payload or {}
    if payload.get("side_effecting"):
        return True
    # Agentic browser workers often opt in via allow_browser rather than
    # (or in addition to) a Hermes-style toolsets string.
    if payload.get("allow_browser"):
        return True
    toolsets = payload.get("toolsets")
    if isinstance(toolsets, str):
        return "browser" in {part.strip() for part in toolsets.split(",")}
    return False


def swarm_is_acting(specs: list[WorkerSpec]) -> bool:
    """True when any worker in the swarm has external side effects."""
    return any(spec_has_side_effects(spec) for spec in specs)


# Full-edit, PATCH-producing adapters in preference order. Cursor/Claude are the
# daily-driver editors; Codex and Hermes are full-edit too, so a host locked to
# either can still implement. This is the single source of truth — the MCP
# ``start_implement`` verb, the CLI, and the lightweight ``edit`` verb all import
# it instead of re-declaring the order.
IMPLEMENT_ADAPTER_PRIORITY = (
    "cursor",
    "claude-code",
    "codex",
    "hermes",
    "antigravity",
    "agentic",
)

# Platforms that can run a read-only analysis worker. This is deliberately a
# capability set, not a priority list: generic review never chooses the first
# enabled platform. It requires an explicit request or the user's configured
# default reviewer.
REVIEW_ADAPTERS = (
    "agentic",
    "cursor",
    "claude-code",
    "codex",
    "openai",
    "hermes",
    "antigravity",
)


class NoImplementAdapterError(RuntimeError):
    """Raised when no implement-capable adapter is available/enabled.

    Carries the resolved ``enabled`` set and the (optional) ``requested`` adapter
    so callers can render a precise, actionable error without re-deriving state.
    """

    def __init__(
        self,
        message: str,
        *,
        enabled: AbstractSet[str],
        requested: Optional[str] = None,
        fix: Optional[str] = None,
    ):
        super().__init__(message)
        self.enabled = enabled
        self.requested = requested
        self.fix = fix


class NoReviewAdapterError(RuntimeError):
    """Raised when no review-capable adapter is available/enabled.

    Carries the resolved ``enabled`` set, the (optional) ``requested`` adapter,
    and the (optional) ``configured_default`` so callers can render a precise,
    actionable error without re-deriving state.
    """

    def __init__(
        self,
        message: str,
        *,
        enabled: AbstractSet[str],
        requested: Optional[str] = None,
        configured_default: Optional[str] = None,
    ):
        super().__init__(message)
        self.enabled = enabled
        self.requested = requested
        self.configured_default = configured_default


# Keys-only agentic path (Grok Bot contained hosts / no Cursor SDK). These
# env vars are enough to run the in-process tool loop — no external CLI.
AGENTIC_PROVIDER_KEY_ENVS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)

CURSOR_ONLY_LOCK_FIX = (
    "enable agentic (`puppetmaster platform enable agentic`), unlock the "
    "platform lock, or install Cursor (`CURSOR_API_KEY` + "
    "`puppetmaster install-cursor-mcp`)"
)


def agentic_provider_keys_visible(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when a keys-only agentic provider credential is in ``env``."""
    probe = env if env is not None else os.environ
    return any((probe.get(name) or "").strip() for name in AGENTIC_PROVIDER_KEY_ENVS)


# Actionable, per-adapter "how to make me runnable" hints, surfaced when the
# auto pick finds nothing available so the user knows exactly what to install.
_ADAPTER_AVAILABILITY_HINT: dict[str, str] = {
    "cursor": "cursor (set CURSOR_API_KEY and run `puppetmaster install-cursor-mcp` to bootstrap @cursor/sdk)",
    "claude-code": "claude-code (`npm install -g @anthropic-ai/claude-code`, then authenticate)",
    "codex": "codex (`npm install -g @openai/codex`, then `codex login`)",
    "hermes": "hermes (install the NousResearch hermes CLI on PATH)",
    "antigravity": "antigravity (install the Google Antigravity CLI `agy` on PATH)",
    "agentic": (
        "agentic (set a provider API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "GEMINI_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY — no external CLI)"
    ),
}


def adapter_is_available(
    name: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    root: Optional["os.PathLike[str]"] = None,
) -> bool:
    """Cheap, no-network probe: can ``name``'s worker actually be dispatched?

    Closes the gap between "enabled by the platform lock" and "runnable on this
    machine" so the auto pick never silently lands on an adapter whose CLI or
    credentials are absent (the Cursor-without-CURSOR_API_KEY footgun). Reuses
    doctor's detection helpers so availability stays consistent with what
    ``puppetmaster doctor`` reports.

    * ``local``/``shell`` run no provider, so they're always available.
    * ``cursor`` needs both ``CURSOR_API_KEY`` and the @cursor/sdk runner.
    * ``claude-code`` / ``codex`` / ``hermes`` need their CLI resolvable.
    * ``openai`` needs ``OPENAI_API_KEY``.
    """
    env = env if env is not None else os.environ
    if name in ("local", "shell"):
        return True
    from pathlib import Path

    from puppetmaster import diagnostics

    if name == "cursor":
        probe_root = Path(root) if root is not None else Path.cwd()
        return bool(env.get("CURSOR_API_KEY")) and diagnostics._cursor_sdk_installed(probe_root)
    if name == "claude-code":
        return diagnostics._claude_code_installed()
    if name == "codex":
        return diagnostics._codex_cli_installed()
    if name == "hermes":
        return diagnostics._hermes_cli_installed()
    if name == "antigravity":
        return diagnostics._antigravity_installed()
    if name == "openai":
        return bool(env.get("OPENAI_API_KEY"))
    if name == "agentic":
        if agentic_provider_keys_visible(env):
            return True
        from puppetmaster.providers import available_providers

        return bool(available_providers(env))
    return False


def pick_implement_adapter(
    enabled: AbstractSet[str],
    requested: Optional[str] = None,
    *,
    is_available: Optional[Callable[[str], bool]] = None,
) -> str:
    """Resolve the full-edit adapter to use, honoring the platform lock.

    * ``requested`` set → validate it's implement-capable AND enabled, else raise.
    * ``requested`` unset → pick the first enabled adapter in priority order that
      is also actually *runnable* on this machine (CLI/credentials present), so a
      permissive default (no lock = every adapter "enabled") never silently lands
      on Cursor when the user has only Claude/Codex/Hermes configured.

    An explicit choice is never second-guessed for availability: a set ``requested``
    or a lock pinned to exactly one implement-capable adapter is honored verbatim
    (and fails later with a precise reason if its tooling is missing) rather than
    being swapped out here.

    Raising (vs. returning ``None``) keeps the single decision point honest: every
    caller gets the same precise failure with the same ``enabled``/``requested``
    context, instead of each re-inventing the error message.
    """
    if requested:
        adapter = str(requested)
        if adapter not in IMPLEMENT_ADAPTER_PRIORITY:
            raise NoImplementAdapterError(
                f"adapter {adapter!r} cannot implement. Implement-capable: "
                f"{', '.join(IMPLEMENT_ADAPTER_PRIORITY)}.",
                enabled=enabled,
                requested=adapter,
            )
        if adapter not in enabled:
            raise NoImplementAdapterError(
                f"adapter {adapter!r} is disabled by the platform lock.",
                enabled=enabled,
                requested=adapter,
            )
        return adapter
    candidates = [a for a in IMPLEMENT_ADAPTER_PRIORITY if a in enabled]
    if not candidates:
        raise NoImplementAdapterError(
            "No implement-capable platform is enabled. Enable one of "
            f"{', '.join(IMPLEMENT_ADAPTER_PRIORITY)} via the platform lock.",
            enabled=enabled,
        )
    available = is_available or adapter_is_available
    # A lock pinned to exactly one implement-capable adapter is an explicit
    # choice — honor it verbatim, except a cursor-only lock with no Cursor
    # must fail closed with the Grok Bot / agentic unlock fix rather than
    # dispatching a worker that cannot start.
    if len(candidates) == 1:
        only = candidates[0]
        if only == "cursor" and not available("cursor"):
            raise NoImplementAdapterError(
                "cursor-only platform lock but Cursor is not runnable on this "
                f"machine. Fix: {CURSOR_ONLY_LOCK_FIX}.",
                enabled=enabled,
                fix=CURSOR_ONLY_LOCK_FIX,
            )
        return only
    runnable = next((a for a in candidates if available(a)), None)
    if runnable is not None:
        return runnable
    hints = ", ".join(_ADAPTER_AVAILABILITY_HINT.get(a, a) for a in candidates)
    raise NoImplementAdapterError(
        "No implement-capable platform is runnable on this machine — these are "
        f"enabled but their CLI/credentials are missing: {', '.join(candidates)}. "
        f"Install/configure one of: {hints}.",
        enabled=enabled,
    )


# Analysis-swarm preference when the caller did not pin an adapter. Cursor
# stays first *when it is actually runnable*; otherwise agentic (keys-only)
# is the Grok Bot contained default. Do not invent a grok-bot adapter.
SWARM_ADAPTER_PRIORITY = (
    "cursor",
    "agentic",
    "claude-code",
    "codex",
    "hermes",
    "openai",
    "antigravity",
)


def pick_swarm_adapter(
    enabled: AbstractSet[str],
    requested: Optional[str] = None,
    *,
    is_available: Optional[Callable[[str], bool]] = None,
    allow_local_demo: bool = False,
) -> str:
    """Resolve the analysis-swarm adapter, honoring lock + runnability.

    Missing Cursor is never a sole failure when agentic is enabled and has a
    provider key. A cursor-only lock with no Cursor fails closed with
    ``CURSOR_ONLY_LOCK_FIX``.
    """
    if allow_local_demo:
        return "local"
    if requested:
        adapter = str(requested)
        if adapter == "local":
            return "local"
        if adapter not in SWARM_ADAPTER_PRIORITY:
            raise NoImplementAdapterError(
                f"adapter {adapter!r} cannot run an analysis swarm.",
                enabled=enabled,
                requested=adapter,
            )
        if adapter in enabled or adapter not in (
            "agentic",
            "cursor",
            "claude-code",
            "codex",
            "openai",
            "hermes",
            "antigravity",
        ):
            return adapter
        raise NoImplementAdapterError(
            f"adapter {adapter!r} is disabled by the platform lock.",
            enabled=enabled,
            requested=adapter,
        )
    candidates = [a for a in SWARM_ADAPTER_PRIORITY if a in enabled]
    if not candidates:
        raise NoImplementAdapterError(
            "No enabled non-demo analysis adapter is available.",
            enabled=enabled,
            fix=(
                "Enable an analysis adapter, pass adapter/config explicitly, "
                "or set allow_local_demo=true for deterministic tests."
            ),
        )
    available = is_available or adapter_is_available
    if len(candidates) == 1:
        only = candidates[0]
        if only == "cursor" and not available("cursor"):
            raise NoImplementAdapterError(
                "cursor-only platform lock but Cursor is not runnable on this "
                f"machine. Fix: {CURSOR_ONLY_LOCK_FIX}.",
                enabled=enabled,
                fix=CURSOR_ONLY_LOCK_FIX,
            )
        return only
    runnable = next((a for a in candidates if available(a)), None)
    if runnable is not None:
        return runnable
    hints = ", ".join(_ADAPTER_AVAILABILITY_HINT.get(a, a) for a in candidates)
    raise NoImplementAdapterError(
        "No analysis adapter is runnable on this machine — these are "
        f"enabled but their CLI/credentials are missing: {', '.join(candidates)}. "
        f"Install/configure one of: {hints}.",
        enabled=enabled,
    )


def pick_review_adapter(
    enabled: AbstractSet[str],
    requested: Optional[str] = None,
    configured_default: Optional[str] = None,
    *,
    is_available: Optional[Callable[[str], bool]] = None,
) -> str:
    """Resolve the review adapter to use, honoring the platform lock.

    Resolution order:
    1. ``requested`` (explicit adapter/platform selection) → validate it's
       review-capable AND enabled, else raise.
    2. ``configured_default`` (user's configured default reviewer) → validate
       it's review-capable, enabled, AND available, else raise with clear
       guidance (no silent fallback).
    3. Fail closed with an actionable error listing enabled valid reviewer
       platforms and how to configure a default.

    Review-capable means any adapter in ``REVIEW_ADAPTERS``. An explicit
    ``requested`` adapter is allowed to reach its platform-specific preflight;
    a configured default is checked here so an unavailable default fails with
    precise guidance instead of silently falling back.

    Raising (vs. returning ``None``) keeps the single decision point honest:
    every caller gets the same precise failure with the same context.
    """
    # Explicit request wins and is never second-guessed.
    if requested:
        adapter = str(requested)
        if adapter not in REVIEW_ADAPTERS:
            raise NoReviewAdapterError(
                f"adapter {adapter!r} cannot review. Review-capable: "
                f"{', '.join(REVIEW_ADAPTERS)}.",
                enabled=enabled,
                requested=adapter,
                configured_default=configured_default,
            )
        if adapter not in enabled:
            raise NoReviewAdapterError(
                f"adapter {adapter!r} is disabled by the platform lock.",
                enabled=enabled,
                requested=adapter,
                configured_default=configured_default,
            )
        return adapter

    # Configured default is authoritative when present — fail clearly if it's
    # disabled or unavailable instead of silently falling back.
    if configured_default:
        adapter = str(configured_default)
        if adapter not in REVIEW_ADAPTERS:
            raise NoReviewAdapterError(
                f"configured default reviewer {adapter!r} is not review-capable. "
                f"Review-capable: {', '.join(REVIEW_ADAPTERS)}. "
                f"Fix: run `puppetmaster platform reviewer <platform>`.",
                enabled=enabled,
                configured_default=adapter,
            )
        if adapter not in enabled:
            raise NoReviewAdapterError(
                f"configured default reviewer {adapter!r} is disabled by the platform lock. "
                f"Fix: run `puppetmaster platform enable {adapter}` or choose a different "
                f"default reviewer with `puppetmaster platform reviewer <platform>`.",
                enabled=enabled,
                configured_default=adapter,
            )
        available = is_available or adapter_is_available
        if not available(adapter):
            hint = _ADAPTER_AVAILABILITY_HINT.get(adapter, adapter)
            raise NoReviewAdapterError(
                f"configured default reviewer {adapter!r} is enabled but not available "
                f"on this machine (CLI/credentials missing). Install/configure: {hint}.",
                enabled=enabled,
                configured_default=adapter,
            )
        return adapter

    # No explicit request and no configured default → fail closed with actionable guidance.
    candidates = [a for a in REVIEW_ADAPTERS if a in enabled]
    if not candidates:
        raise NoReviewAdapterError(
            "No review-capable platform is enabled and no default reviewer is configured. "
            f"Enable one of {', '.join(REVIEW_ADAPTERS)} and configure it as your "
            f"default reviewer: `puppetmaster platform enable <adapter> && "
            f"puppetmaster platform reviewer <adapter>`.",
            enabled=enabled,
            configured_default=configured_default,
        )

    # Enabled platforms exist but no default is configured — fail closed.
    raise NoReviewAdapterError(
        f"No default reviewer configured. Enabled review-capable platforms: "
        f"{', '.join(candidates)}. Configure one as your default: "
        f"`puppetmaster platform reviewer <adapter>`.",
        enabled=enabled,
        configured_default=configured_default,
    )


def build_edit_payload(
    *,
    instruction: str,
    cwd: str,
    adapter: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    timeout_seconds: int = 300,
    routing_policy: str = "cheap",
    auto_route: bool = True,
    disable_codegraph: bool = False,
    allowed_model_ids: Optional[list[str]] = None,
) -> dict:
    """Build the worker payload for a lightweight, in-place single edit.

    This is the ``edit`` verb's defining policy — the deliberate divergence from
    ``start_implement``:

    * ``mode="implement"`` so the adapter actually edits (→ ``swarm_mode`` == edit).
    * ``allow_dirty=True`` + ``allow_non_worktree=True``: edit the working tree
      in place, no isolated worktree and no clean-tree guard. A single edit
      should land where you're working, fast. The PATCH artifact is still
      captured from the git diff, so the change is fully reviewable/revertable.
    * ``auto_route=True`` + ``routing_policy="cheap"`` by default: the cheapest
      sufficient model handles a mechanical edit, instead of burning a frontier
      model. A pinned ``model`` overrides routing.
    * CodeGraph stays ON (unless explicitly disabled): locate the exact edit site
      structurally instead of grepping.
    """
    payload: dict = {
        "prompt": instruction,
        "cwd": cwd,
        "mode": "implement",
        "timeout_seconds": timeout_seconds,
        "allow_dirty": True,
        "allow_non_worktree": True,
    }
    if disable_codegraph:
        payload["disable_codegraph"] = True
    if model:
        # An explicit model pin wins over routing — don't auto-route around it.
        if adapter == "cursor":
            from puppetmaster.model_registry import apply_cursor_model_pin

            payload = apply_cursor_model_pin(payload, model)
        else:
            payload["model"] = model
    elif auto_route:
        payload["auto_route"] = True
        payload["allowed_adapters"] = [adapter]
        if routing_policy:
            payload["routing_policy"] = routing_policy
    if allowed_model_ids is not None:
        payload["allowed_model_ids"] = list(allowed_model_ids)
    if provider:
        payload["provider"] = provider
    return payload


def build_edit_spec(
    *,
    instruction: str,
    adapter: str,
    cwd: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    timeout_seconds: int = 300,
    routing_policy: str = "cheap",
    auto_route: bool = True,
    disable_codegraph: bool = False,
    allowed_model_ids: Optional[list[str]] = None,
) -> WorkerSpec:
    """A single full-edit worker spec tuned for snappy in-place single edits."""
    payload = build_edit_payload(
        instruction=instruction,
        cwd=cwd,
        adapter=adapter,
        model=model,
        provider=provider,
        timeout_seconds=timeout_seconds,
        routing_policy=routing_policy,
        auto_route=auto_route,
        disable_codegraph=disable_codegraph,
        allowed_model_ids=allowed_model_ids,
    )
    return WorkerSpec(
        role=f"edit-{adapter}",
        instruction=instruction,
        adapter=adapter,
        payload=payload,
    )


def specs_for_roles(roles: Optional[list[str]] = None) -> list[WorkerSpec]:
    if roles is None:
        return list(DEFAULT_WORKERS)
    if not roles:
        return []
    known = {spec.role: spec for spec in DEFAULT_WORKERS}
    return [
        known.get(role, WorkerSpec(role=role, instruction=f"Run the {role} worker."))
        for role in roles
    ]

