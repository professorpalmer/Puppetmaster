"""Continuous planner loop: handoff-up, intent before fan-out, work vs snapshot.

Kernel-owned. Workers do not coordinate. Nested job starts stay refused.
Planner tasks never code; they emit same-job children and consume handoffs.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, TypedDict

from puppetmaster.models import ArtifactType, Task, TaskStatus

SCHEMA_VERSION = 1
PLANNER_ROLES = frozenset({"planner", "subplanner"})
WORK_LANE = "work"
SNAPSHOT_LANE = "snapshot"
REASON_INTENT_SPEC = "intent_spec_required"
DEFAULT_FANOUT_MAX = 20
DEFAULT_MAX_ITERATIONS = 4
INTENT_KIND = "intent_spec"
SCOPE_COMPLETE_KIND = "scope_complete"
_TERMINAL = frozenset({TaskStatus.COMPLETE, TaskStatus.FAILED})


class HandoffPayload(TypedDict):
    done: str
    deviations: list[str]
    concerns: list[str]


class IntentSpec(TypedDict):
    schema_version: int
    kind: str
    architecture: str
    out_of_scope: list[str]
    dependency_philosophy: str
    resource_timeouts: str
    fanout_min: int
    fanout_max: int


def _string_list(value: Any, key: str) -> list[str]:
    items = value.get(key)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError(f"{key} must be a list of strings")
    return items


def validate_handoff(value: Any) -> HandoffPayload:
    if not isinstance(value, dict) or not isinstance(value.get("done"), str):
        raise ValueError("handoff requires a done string")
    _string_list(value, "deviations")
    _string_list(value, "concerns")
    return value


def validate_intent_spec(value: Any) -> IntentSpec:
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("expected schema_version=1 object")
    if value.get("kind") != INTENT_KIND:
        raise ValueError("kind must be intent_spec")
    for key in ("architecture", "dependency_philosophy", "resource_timeouts"):
        if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
            raise ValueError(f"{key} must be a nonempty string")
    _string_list(value, "out_of_scope")
    try:
        fanout_min = int(value.get("fanout_min"))
        fanout_max = int(value.get("fanout_max"))
    except (TypeError, ValueError):
        raise ValueError("fanout_min and fanout_max must be integers") from None
    if type(value.get("fanout_min")) is bool or type(value.get("fanout_max")) is bool:
        raise ValueError("fanout_min and fanout_max must be integers")
    if fanout_min < 1 or fanout_max < fanout_min or fanout_max > 100:
        raise ValueError("fanout range must satisfy 1 <= min <= max <= 100")
    return value


def _payload(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    payload = getattr(value, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _role_of(task: Any) -> str:
    return str(getattr(task, "role", "") or "").strip().lower().replace("_", "-")


def is_planner_task(task: Any) -> bool:
    if _role_of(task) in PLANNER_ROLES:
        return True
    return bool(_payload(task).get("continuous_planner"))


def parent_task_id_of(task: Any) -> Optional[str]:
    payload = _payload(task)
    raw = payload.get("parent_task_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    depends = getattr(task, "depends_on", None) or []
    if isinstance(depends, (list, tuple)) and depends:
        first = str(depends[0] or "").strip()
        return first or None
    return None


def parse_handoff(artifact: Any) -> Optional[HandoffPayload]:
    payload = _payload(artifact)
    if "deviations" in payload or "concerns" in payload or "done" in payload:
        done = payload.get("done")
        if not isinstance(done, str) or not done.strip():
            claim = payload.get("claim") or payload.get("summary") or payload.get("change")
            done = str(claim).strip() if isinstance(claim, str) else ""
        candidate = {
            "done": done,
            "deviations": payload.get("deviations", []),
            "concerns": payload.get("concerns", []),
        }
        try:
            return validate_handoff(candidate)
        except ValueError:
            return None
    return None


def parse_intent_spec(payload: Any) -> Optional[IntentSpec]:
    body = _payload(payload)
    if body.get("kind") != INTENT_KIND:
        return None
    candidate = dict(body)
    candidate.setdefault("schema_version", SCHEMA_VERSION)
    try:
        return validate_intent_spec(candidate)
    except ValueError:
        return None


def intent_spec_for_job(artifacts: Sequence[Any]) -> Optional[IntentSpec]:
    for artifact in artifacts:
        kind = getattr(artifact, "type", None)
        if kind not in {ArtifactType.DECISION, "decision", ArtifactType.FINDING, "finding"}:
            continue
        parsed = parse_intent_spec(artifact)
        if parsed is not None:
            return parsed
    return None


def planner_emitted_scope_complete(artifacts: Sequence[Any], planner_id: str) -> bool:
    for artifact in artifacts:
        if getattr(artifact, "task_id", None) != planner_id:
            continue
        payload = _payload(artifact)
        if payload.get("kind") == SCOPE_COMPLETE_KIND:
            return True
        if str(payload.get("decision") or "").strip().lower() == SCOPE_COMPLETE_KIND:
            return True
    return False


def children_of(tasks: Sequence[Task], parent_id: str) -> list[Task]:
    found = []
    for task in tasks:
        if parent_task_id_of(task) == parent_id:
            found.append(task)
    return found


def _next_iteration(parent: Task) -> int:
    payload = _payload(parent)
    raw = payload.get("planner_iteration")
    try:
        current = int(raw)
    except (TypeError, ValueError):
        current = 0
    return current + 1


def _max_iterations(parent: Task) -> int:
    payload = _payload(parent)
    raw = payload.get("max_planner_iterations")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_ITERATIONS
    if value < 1:
        return DEFAULT_MAX_ITERATIONS
    return min(value, 32)


def follow_up_limit_for(task: Any, default: int = 4) -> int:
    if is_planner_task(task):
        intent = parse_intent_spec(_payload(task))
        if intent is not None:
            return max(default, int(intent["fanout_max"]))
        return max(default, DEFAULT_FANOUT_MAX)
    return default


def planner_instruction_from_handoffs(parent: Task, handoffs: Iterable[HandoffPayload]) -> str:
    lines = [
        "Consume worker handoffs and continue the same scope. Do not code.",
        "Emit 8-20 disjoint same-job tasks via enqueue_subtasks, enqueue a snapshot "
        "lane task before closing, or emit a decision with kind=scope_complete.",
        "Constraints: no TODOs, no partial implementations, no nested job starts.",
        f"Parent planner: {parent.id} (iteration {_next_iteration(parent) - 1}).",
        "Handoffs:",
    ]
    count = 0
    for handoff in handoffs:
        count += 1
        if count > 40:
            lines.append("- (further handoffs truncated)")
            break
        concerns = "; ".join(handoff["concerns"][:8]) or "none"
        deviations = "; ".join(handoff["deviations"][:8]) or "none"
        lines.append(
            f"- done={handoff['done'][:300]} | deviations={deviations[:300]} | concerns={concerns[:300]}"
        )
    if count == 0:
        lines.append("- (no typed handoffs; inspect artifacts and the tree)")
    return "\n".join(lines)


def maybe_requeue_planner(store: Any, completed_task: Task) -> Optional[Task]:
    """Enqueue the next planner iteration after its children all finish.

    Best-effort: never raises into the completion hot path. Does nothing when
    the completed task is itself a planner, the parent is not a planner, any
    sibling is still live, the parent already closed the scope, or the
    iteration cap is reached.
    """
    try:
        return _maybe_requeue_planner(store, completed_task)
    except Exception:
        return None


def _maybe_requeue_planner(store: Any, completed_task: Task) -> Optional[Task]:
    if is_planner_task(completed_task):
        return None
    parent_id = parent_task_id_of(completed_task)
    if not parent_id:
        return None
    parent = store.get_task_by_id(parent_id)
    if parent is None or not is_planner_task(parent):
        return None
    job_id = completed_task.job_id
    tasks = store.list_tasks(job_id)
    siblings = children_of(tasks, parent.id)
    live = [
        task
        for task in siblings
        if task.id != completed_task.id
        and task.status not in _TERMINAL
        and not is_planner_task(task)
    ]
    if live:
        return None
    artifacts = store.list_artifacts(job_id)
    if planner_emitted_scope_complete(artifacts, parent.id):
        return None
    nxt = _next_iteration(parent)
    if nxt > _max_iterations(parent):
        store._emit_enqueue_refused(
            job_id,
            "planner_iteration_cap",
            parent_task_id=parent.id,
            extra={"iteration": nxt, "max_planner_iterations": _max_iterations(parent)},
        )
        return None
    existing = [
        task
        for task in siblings
        if is_planner_task(task) and int(_payload(task).get("planner_iteration") or 0) == nxt
    ]
    if existing:
        return existing[0]
    handoffs = []
    for task in siblings:
        if is_planner_task(task):
            continue
        for artifact in artifacts:
            if getattr(artifact, "task_id", None) != task.id:
                continue
            parsed = parse_handoff(artifact)
            if parsed is not None:
                handoffs.append(parsed)
    instruction = planner_instruction_from_handoffs(parent, handoffs)
    root_id = str(_payload(parent).get("planner_root_id") or parent.id)
    child_payload = {
        "continuous_planner": True,
        "planner_iteration": nxt,
        "planner_root_id": root_id,
        "max_planner_iterations": _max_iterations(parent),
        "lane": WORK_LANE,
        "mode": "analysis",
    }
    cwd = _payload(parent).get("cwd") or _payload(completed_task).get("cwd")
    if cwd:
        child_payload["cwd"] = cwd
    return store.enqueue_subtask(
        job_id,
        parent_task_id=parent.id,
        role="planner",
        instruction=instruction,
        adapter=parent.adapter,
        payload=child_payload,
        created_by="continuous_plan",
        actor="coordinator",
    )
