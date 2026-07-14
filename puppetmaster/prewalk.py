"""OMP-style plan-then-cheap ``prewalk`` handoff.

A *prewalk* is a two-worker DAG: a quality-routed read-only ``plan`` worker
emits durable decision/plan artifacts, then a cheap-routed edit-capable
``implement`` worker (``depends_on_roles=["plan"]``) applies that plan.

This reuses the existing orchestrator DAG (``depends_on_roles``) and auto_route
policies — no new scheduler. Routing/savings stay honest: plan stamps a
quality ROUTING artifact, implement stamps a cheap one.

Inspired by oh-my-pi ``--prewalk`` (strong model plans, cheaper model implements),
adapted to Puppetmaster's durable artifacts and worker specs.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

from puppetmaster.workers import ANALYSIS_NO_EDIT_PAYLOAD, WorkerSpec

PLAN_ROLE = "plan"
IMPLEMENT_ROLE = "implement"

DEFAULT_PLAN_TIMEOUT_SECONDS = 900
DEFAULT_IMPLEMENT_TIMEOUT_SECONDS = 900

PREWALK_PLAN_SECTION_HEADER = "Upstream plan from the plan worker (APPLY THIS):"

_PLAN_ARTIFACT_TYPES = frozenset({"decision", "plan"})


def format_plan_artifacts_for_injection(
    artifacts: Iterable[Any],
) -> str:
    """Format plan/decision artifacts into text for implement-worker injection.

    Accepts ``Artifact`` objects or plain dicts (``type`` / ``payload``). Only
    decision and plan payloads are included; other artifact types are skipped.
    Returns an empty string when nothing usable is present.
    """
    blocks: list[str] = []
    for artifact in artifacts:
        kind, payload = _artifact_type_and_payload(artifact)
        if kind not in _PLAN_ARTIFACT_TYPES:
            # Some decision-shaped payloads ride on other types; still accept
            # an explicit plan/decision body when present.
            if not (
                isinstance(payload, dict)
                and (
                    payload.get("decision")
                    or payload.get("plan")
                    or payload.get("steps")
                )
            ):
                continue
        block = _format_one_plan_payload(payload if isinstance(payload, dict) else {})
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "\n\n".join(blocks)


def inject_plan_into_prompt(prompt: str, artifacts: Iterable[Any]) -> str:
    """Inject formatted plan/decision artifact text into ``prompt``.

    When the prompt has no plan section yet, prepends one. When it already
    carries the ``PREWALK_PLAN_SECTION_HEADER`` placeholder from
    ``build_prewalk_specs``, replaces that stub with the real plan text. If
    real plan content is already present under the header, leaves the prompt
    unchanged (avoids double-injection). No-op when formatting yields nothing.
    """
    plan_text = format_plan_artifacts_for_injection(artifacts)
    if not plan_text:
        return prompt
    body = (prompt or "").strip()
    section = f"{PREWALK_PLAN_SECTION_HEADER}\n{plan_text}"
    if not body:
        return section
    idx = body.find(PREWALK_PLAN_SECTION_HEADER)
    if idx < 0:
        return f"{section}\n\n{body}"
    after_header = body[idx + len(PREWALK_PLAN_SECTION_HEADER) :].lstrip("\n")
    first_line = after_header.split("\n", 1)[0].strip() if after_header else ""
    if first_line.startswith(
        ("Decision:", "Why:", "Plan:", "Plan steps:", "Files:", "Constraints:", "Notes:")
    ):
        return body
    before = body[:idx]
    if "\n\n" in after_header:
        remainder = after_header.split("\n\n", 1)[1]
    else:
        remainder = ""
    parts: list[str] = []
    if before.strip():
        parts.append(before.rstrip("\n"))
    parts.append(section)
    if remainder.strip():
        parts.append(remainder.lstrip("\n"))
    return "\n\n".join(parts)


def build_prewalk_specs(
    goal: str,
    cwd: str,
    *,
    plan_adapter: str = "local",
    implement_adapter: str = "local",
    plan_model: Optional[str] = None,
    implement_model: Optional[str] = None,
    plan_timeout_seconds: int = DEFAULT_PLAN_TIMEOUT_SECONDS,
    implement_timeout_seconds: int = DEFAULT_IMPLEMENT_TIMEOUT_SECONDS,
    plan_routing_policy: str = "quality",
    implement_routing_policy: str = "cheap",
    auto_route: bool = True,
    allow_dirty: bool = False,
    allow_non_worktree: bool = False,
    disable_codegraph: bool = False,
    disable_memory: bool = False,
) -> list[WorkerSpec]:
    """Build the plan → implement WorkerSpec DAG for a prewalk job.

    * ``plan`` — auto_route + quality (or pinned model), read-only analysis
      payload (``ANALYSIS_NO_EDIT_PAYLOAD``). Emits decision/plan artifacts.
    * ``implement`` — ``depends_on_roles=["plan"]``, auto_route + cheap (or
      pinned model), edit-capable (``mode=implement``, NOT analysis-no-edit).
      Instruction requires applying the upstream plan artifacts.
    """
    goal_text = (goal or "").strip()
    if not goal_text:
        raise ValueError("build_prewalk_specs: goal must be non-empty")
    cwd_text = (cwd or "").strip() or "."

    plan_instruction = (
        "Produce a concrete implementation plan as durable decision/plan "
        "artifacts (ordered steps, files to touch, constraints). Do not edit "
        "files — this is a read-only planning role."
    )
    plan_prompt = (
        f"{plan_instruction}\n\n"
        f"Goal:\n{goal_text}\n\n"
        "Return Puppetmaster artifact JSON with an artifacts array. Prefer "
        "type=decision payloads that include decision, why, and an ordered "
        "plan/steps list with concrete file paths."
    )
    plan_payload: dict[str, Any] = {
        "prompt": plan_prompt,
        "cwd": cwd_text,
        "timeout_seconds": int(plan_timeout_seconds),
        **ANALYSIS_NO_EDIT_PAYLOAD,
    }
    if disable_codegraph:
        plan_payload["disable_codegraph"] = True
    if disable_memory:
        plan_payload["disable_memory"] = True
    _apply_routing(
        plan_payload,
        adapter=plan_adapter,
        model=plan_model,
        routing_policy=plan_routing_policy,
        auto_route=auto_route,
        pin_adapter=False,
    )

    implement_instruction = (
        "Implement mode: apply the upstream plan worker's decision/plan "
        "artifacts for this job. Do NOT re-plan from scratch — follow the "
        "recorded steps, edit the files they name, and verify. If a step is "
        "ambiguous, prefer the smallest change that satisfies the plan."
    )
    implement_prompt = (
        f"{implement_instruction}\n\n"
        f"Goal:\n{goal_text}\n\n"
        f"{PREWALK_PLAN_SECTION_HEADER}\n"
        "(Read this job's upstream plan/decision artifacts from the plan "
        "worker and apply them exactly. The plan worker completed before you "
        "were unblocked.)"
    )
    implement_payload: dict[str, Any] = {
        "prompt": implement_prompt,
        "cwd": cwd_text,
        "mode": "implement",
        "timeout_seconds": int(implement_timeout_seconds),
        "prewalk": True,
        "allow_dirty": bool(allow_dirty),
        "allow_non_worktree": bool(allow_non_worktree),
    }
    if disable_codegraph:
        implement_payload["disable_codegraph"] = True
    if disable_memory:
        implement_payload["disable_memory"] = True
    _apply_routing(
        implement_payload,
        adapter=implement_adapter,
        model=implement_model,
        routing_policy=implement_routing_policy,
        auto_route=auto_route,
        # Keep implement on an edit-capable adapter when one was chosen.
        pin_adapter=implement_adapter not in ("", "local"),
    )

    return [
        WorkerSpec(
            role=PLAN_ROLE,
            instruction=plan_instruction,
            adapter=plan_adapter,
            payload=plan_payload,
        ),
        WorkerSpec(
            role=IMPLEMENT_ROLE,
            instruction=implement_instruction,
            adapter=implement_adapter,
            payload=implement_payload,
            depends_on_roles=[PLAN_ROLE],
        ),
    ]


def _apply_routing(
    payload: dict[str, Any],
    *,
    adapter: str,
    model: Optional[str],
    routing_policy: str,
    auto_route: bool,
    pin_adapter: bool,
) -> None:
    if model:
        payload["model"] = model
        return
    if not auto_route:
        return
    payload["auto_route"] = True
    if routing_policy:
        payload["routing_policy"] = routing_policy
    if pin_adapter and adapter:
        payload["allowed_adapters"] = [adapter]


def _artifact_type_and_payload(artifact: Any) -> tuple[str, Any]:
    if isinstance(artifact, dict):
        raw_type = artifact.get("type", "")
        payload = artifact.get("payload") or {}
    else:
        raw_type = getattr(artifact, "type", "") or ""
        payload = getattr(artifact, "payload", None) or {}
    kind = getattr(raw_type, "value", raw_type)
    return str(kind).strip().lower(), payload


def _format_one_plan_payload(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    decision = payload.get("decision")
    if decision:
        lines.append(f"Decision: {str(decision).strip()}")
    why = payload.get("why")
    if why:
        lines.append(f"Why: {str(why).strip()}")
    plan = payload.get("plan")
    steps = payload.get("steps")
    ordered: Optional[Sequence[Any]] = None
    if isinstance(plan, (list, tuple)):
        ordered = plan
    elif isinstance(steps, (list, tuple)):
        ordered = steps
    elif isinstance(plan, str) and plan.strip():
        lines.append(f"Plan: {plan.strip()}")
    if ordered:
        lines.append("Plan steps:")
        for index, step in enumerate(ordered, start=1):
            lines.append(f"  {index}. {_format_plan_step(step)}")
    # Catch remaining useful keys without dumping the whole payload.
    for key in ("files", "constraints", "notes"):
        value = payload.get(key)
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(str(item).strip() for item in value if str(item).strip())
            if rendered:
                lines.append(f"{key.capitalize()}: {rendered}")
        else:
            lines.append(f"{key.capitalize()}: {str(value).strip()}")
    return "\n".join(lines).strip()


def _format_plan_step(step: Union[str, dict, Any]) -> str:
    if isinstance(step, dict):
        for key in ("step", "action", "instruction", "text", "summary"):
            if step.get(key):
                text = str(step[key]).strip()
                files = step.get("files") or step.get("paths")
                if files:
                    if isinstance(files, (list, tuple)):
                        file_text = ", ".join(str(f) for f in files)
                    else:
                        file_text = str(files)
                    return f"{text} [{file_text}]"
                return text
        return str(step)
    return str(step).strip()
