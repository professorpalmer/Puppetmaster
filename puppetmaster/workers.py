from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from puppetmaster.adapters import get_adapter, verification_artifact
from puppetmaster.models import AgentRun, Artifact, Task, TaskStatus, now_iso

# Adapters that bill an LLM provider and therefore benefit from a pre-dispatch
# auth/billing gate. ``local``/``shell`` run no model, so they're never gated.
_PREFLIGHTABLE_ADAPTERS = {"cursor", "claude-code", "codex", "openai"}

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
        "missing_cli",
        "auth",
        "authentication",
    }
)


@dataclass(frozen=True)
class WorkerSpec:
    role: str
    instruction: str
    adapter: str = "local"
    payload: dict = field(default_factory=dict)
    depends_on_roles: list[str] = field(default_factory=list)


# Built-in worker specs ship with ``auto_route: True`` so any swarm
# started via the MCP ``puppetmaster_start_*`` tools or ``python -m
# puppetmaster run`` consults the router and lets each role land on
# the appropriate tier (cheap for explore/test, mid for architect,
# frontier for redteam/audit). If the user has not run
# ``puppetmaster models init``, the orchestrator quietly skips routing
# and falls back to the spec's declared adapter — opting in is safe.
_DEFAULT_AUTO_ROUTE_PAYLOAD = {"auto_route": True}


DEFAULT_WORKERS = [
    WorkerSpec(
        role="explore",
        instruction="Map the problem, extract constraints, and emit evidenced findings.",
        payload=dict(_DEFAULT_AUTO_ROUTE_PAYLOAD),
    ),
    WorkerSpec(
        role="architect",
        instruction="Choose the smallest viable architecture and record explicit decisions.",
        payload=dict(_DEFAULT_AUTO_ROUTE_PAYLOAD),
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
        payload=dict(_DEFAULT_AUTO_ROUTE_PAYLOAD),
        depends_on_roles=["architect"],
    ),
    WorkerSpec(
        role="redteam",
        instruction="Find failure modes, stale assumptions, and missing verification.",
        payload=dict(_DEFAULT_AUTO_ROUTE_PAYLOAD),
        depends_on_roles=["implement"],
    ),
    WorkerSpec(
        role="test",
        instruction="Convert claims into checks and verification results.",
        payload=dict(_DEFAULT_AUTO_ROUTE_PAYLOAD),
        depends_on_roles=["implement"],
    ),
]


class LocalWorker:
    """Task executor used by real worker processes."""

    def __init__(self, role: str, worker_id: Optional[str] = None) -> None:
        self.role = role
        self.worker_id = worker_id or f"local-{role}"

    def run(self, task: Task, goal: str) -> tuple[AgentRun, list[Artifact]]:
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

        from puppetmaster.preflight import preflight_check

        model = task.payload.get("model")
        catalog_fetcher = None
        if task.adapter == "cursor" and (
            model or task.payload.get("live_preflight")
        ):
            from puppetmaster.cursor_discovery import fetch_cursor_catalog

            catalog_fetcher = fetch_cursor_catalog
        result = preflight_check(
            task.adapter,
            model,
            allow_api_billing=bool(task.payload.get("allow_api_billing", True)),
            live=bool(task.payload.get("live_preflight")),
            catalog_fetcher=catalog_fetcher,
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
_EDIT_CAPABLE_ADAPTERS = frozenset({"claude-code", "codex"})


def spec_edits_files(spec: WorkerSpec) -> bool:
    """True when ``spec`` can leave file changes behind (vs. emit-only)."""
    payload = spec.payload or {}
    if payload.get("mode") == "implement" or payload.get("implement"):
        return True
    return spec.adapter in _EDIT_CAPABLE_ADAPTERS


def swarm_mode(specs: list[WorkerSpec]) -> str:
    """Classify a swarm as ``"edit"`` (some worker can change files) or
    ``"analysis"`` (read-only — emits artifacts only)."""
    return "edit" if any(spec_edits_files(spec) for spec in specs) else "analysis"


def specs_for_roles(roles: Optional[list[str]] = None) -> list[WorkerSpec]:
    if not roles:
        return DEFAULT_WORKERS
    known = {spec.role: spec for spec in DEFAULT_WORKERS}
    return [
        known.get(role, WorkerSpec(role=role, instruction=f"Run the {role} worker."))
        for role in roles
    ]

