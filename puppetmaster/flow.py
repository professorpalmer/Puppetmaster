"""Flow graphs: a pilot writes one graph, the runtime walks it durably.

The pilot turns a goal into a graph of nodes and edges once, starts it, and is
woken only when the run finishes, gets stuck, or waits at a gate. It never
spends a model turn on launching, polling or hand-offs. Every agent or judge
node runs as an ordinary durable Puppetmaster job, so leases, file claims,
write-scope gates, receipts and recovery all apply per node.

Graph shape::

    {"id": "fix-and-review", "entry": "build", "cwd": ".",
     "defaults": {"adapter": "codex", "model": "gpt-5.6-luna"},
     "limits": {"maxSteps": 60, "maxLoops": 3},
     "nodes": [{"id": "build", "kind": "agent", "role": "code",
                "task": "...", "files": ["pkg/a.py"]},
               {"id": "check", "kind": "shell", "command": "python -m pytest -q"},
               {"id": "review", "kind": "judge", "task": "Review the change."},
               {"id": "done", "kind": "end"}],
     "edges": [{"from": "build", "to": "check"},
               {"from": "check", "to": "review", "when": "ok"},
               {"from": "check", "to": "build", "when": "fail"},
               {"from": "review", "to": "done", "when": "PASS"},
               {"from": "review", "to": "build", "when": "FAIL"}]}

Routing: the first matching edge in declaration order wins. A back-edge is a
loop and may be taken at most ``max`` (default ``limits.maxLoops``) times;
past that the run is ``stuck``, never an unbounded spin.

What the runtime adds beyond walking:

- Exactly-once nodes. Each node visit has an in-flight record and a
  deterministic launch key, so a walker that crashes mid-node adopts the job it
  started instead of buying it again.
- Session continuity. A node that runs again (a judge sent it back, or a
  ``continue_from`` follow-up run) resumes its own provider session and gets
  only the delta, not the whole task.
- ``map`` fan-out. One node expands a list into per-item child flows, each
  walked by its own process with bounded concurrency. On a later visit only
  the items the feedback names (or that failed) run again, and each resumes its
  own session. The pilot's context stays the same size at any item count.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.models import ArtifactType, JobStatus, TaskStatus, now_iso
from puppetmaster.proc_identity import pid_reused, process_identity
from puppetmaster.swarm_reasoning import EFFORTS, WorkerEffortError, operator_effort_profile
from puppetmaster.worker_verdict import parse_terminal_verdict

NODE_KINDS = ("agent", "judge", "parallel", "map", "shell", "gate", "set", "end")
# Reasoning effort, cheapest first; escalation walks up this ladder.
LANES = ("explore", "code", "judge")
_CONTROL_KINDS = ("set", "end")
TERMINAL_STATUSES = ("done", "failed", "stuck", "stopped")
WAKE_STATUSES = TERMINAL_STATUSES + ("waiting", "interrupted")
RESTARTABLE_STATUSES = ("failed", "stuck", "stopped", "interrupted")
DEFAULT_MAX_STEPS = 60
DEFAULT_MAX_LOOPS = 3
# Model nodes run for minutes. A 30 s lease rides out write-contention stalls
# that expired 5 s leases on healthy workers; crash detection waits at most 30 s.
DEFAULT_LEASE_SECONDS = 30
DEFAULT_MAP_CONCURRENCY = 8
_SPAWN_WINDOW_SECONDS = 30
MAX_MAP_DEPTH = 3
_GRAPH_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NODE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_RUN_ID = re.compile(r"^flow_[0-9a-f]{12}$")
_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")
_STATE_PREDICATE = re.compile(r"^state\.([A-Za-z0-9_.-]+)\s*(>=|<=|!=|~=|=|>|<)\s*(.*)$")
# A value substituted into a shell command must be inert in sh and cmd.exe and
# must not start with "-" (no option injection); anything else reaches the
# command through PM_FLOW_CONTEXT instead.
_SHELL_SAFE = re.compile(r"^(?:[A-Za-z0-9_./:=@+,][A-Za-z0-9_./:=@+,-]*)?$")
_OUTPUT_CHARS = 6000
_JUDGE_VERDICT = (
    "End your reply with exactly one line: `VERDICT: PASS - <summary>` or "
    "`VERDICT: FAIL - <what is wrong, file:line>` or "
    "`VERDICT: PARTIAL - <what could not be verified>`."
)
_BUILD_VERDICT = (
    "When you finish, end your reply with exactly one line: "
    "`VERDICT: PASS - <what you built and checked>` or "
    "`VERDICT: FAIL - <what still fails>` or "
    "`VERDICT: PARTIAL - <what you could not finish or verify>`."
)
_READ_ONLY = {"read_only": True, "sandbox": "read-only", "permission_mode": "plan"}
_WRITE = {"sandbox": "workspace-write", "permission_mode": "acceptEdits", "allow_dirty": True}


class FlowError(ValueError):
    """A graph or run request that cannot be honored."""


# --------------------------------------------------------------------------
# Validation


def validate_graph(graph: Any, *, child: bool = False, depth: int = 0,
                   defaults: Optional[dict] = None) -> list[str]:
    """Every problem with ``graph``; an empty list means it can run."""
    if not isinstance(graph, dict):
        return ["graph must be a JSON object"]
    problems: list[str] = []
    if not child and not _GRAPH_ID.fullmatch(str(graph.get("id") or "")):
        problems.append("id must be kebab-case")
    for key in ("defaults", "limits", "state"):
        if key in graph and not isinstance(graph[key], dict):
            problems.append(f"{key} must be an object")
    if "cwd" in graph and not isinstance(graph["cwd"], str):
        problems.append("cwd must be a string")
    own_defaults = graph.get("defaults") if isinstance(graph.get("defaults"), dict) else {}
    if "payload" in own_defaults and not isinstance(own_defaults["payload"], dict):
        problems.append("defaults.payload must be an object")
        own_defaults = {key: value for key, value in own_defaults.items() if key != "payload"}
    problems.extend(_effort_problems(own_defaults, "defaults"))
    lanes = own_defaults.get("lanes")
    if lanes is not None:
        if not isinstance(lanes, dict) or not set(lanes) <= set(LANES):
            problems.append(f"defaults.lanes must map {', '.join(LANES)} to an effort")
        else:
            for lane, effort in lanes.items():
                if effort not in EFFORTS:
                    problems.append(f"defaults.lanes.{lane} must be one of {', '.join(EFFORTS)}")
    merged = _merged_defaults(defaults, own_defaults)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return problems + ["nodes must be a non-empty list"]
    by_id: dict[str, dict] = {}
    for node in nodes:
        if not isinstance(node, dict):
            problems.append("every node must be an object")
            continue
        nid = node.get("id")
        if not isinstance(nid, str) or not _NODE_ID.fullmatch(nid):
            problems.append(f"node id {nid!r} must match {_NODE_ID.pattern}")
            continue
        if nid in by_id:
            problems.append(f"duplicate node id {nid!r}")
        by_id[nid] = node
        problems.extend(_node_problems(node, merged, child=child, depth=depth))
    entry = graph.get("entry")
    if not isinstance(entry, str) or entry not in by_id:
        problems.append(f"entry {entry!r} is not a node")
    for node in by_id.values():
        if node.get("kind") == "parallel":
            problems.extend(_parallel_problems(node, by_id))
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        return problems + ["edges must be a list"]
    for edge in edges:
        if not isinstance(edge, dict):
            problems.append("every edge must be an object")
            continue
        for end in ("from", "to"):
            if not isinstance(edge.get(end), str) or edge.get(end) not in by_id:
                problems.append(f"edge {end} {edge.get(end)!r} is not a node")
        when = edge.get("when", "always")
        if not _valid_when(when):
            problems.append(f"edge when {when!r} is not a known condition")
        if "max" in edge and not _positive_int(edge["max"]):
            problems.append("edge max must be a positive integer")
    limits = graph.get("limits") if isinstance(graph.get("limits"), dict) else {}
    for key in ("maxSteps", "maxLoops"):
        if key in limits and not _positive_int(limits[key]):
            problems.append(f"limits.{key} must be a positive integer")
    return problems


def _effort_problems(obj: dict, where: str) -> list[str]:
    problems = []
    if "effort" in obj and obj["effort"] not in EFFORTS:
        problems.append(f"{where} effort must be one of {', '.join(EFFORTS)}")
    if "escalate" in obj and not isinstance(obj["escalate"], bool):
        problems.append(f"{where} escalate must be true or false")
    try:
        profile = operator_effort_profile()
    except WorkerEffortError as exc:
        return problems + [str(exc)]
    if profile.enforced:
        # Refuse at validation, not at the node that would launch it.
        requested = [obj.get("effort")]
        if isinstance(obj.get("lanes"), dict):
            requested.extend(obj["lanes"].values())
        for effort in requested:
            if effort in EFFORTS and effort != profile.effort:
                problems.append(f"{where} effort {effort} conflicts with the enforced worker effort {profile.effort}")
        if obj.get("escalate") is True:
            problems.append(f"{where} escalate conflicts with the enforced worker effort {profile.effort}")
    return problems


def node_effort(node: dict, defaults: dict, visit: int) -> Optional[str]:
    """Reasoning effort for one visit of an agent or judge node, or None (adapter default).

    The node's own ``effort`` wins over its lane (explore, code, judge) in
    ``defaults.lanes``, which wins over ``defaults.effort``. With ``escalate``,
    each repair visit thinks one step harder than the last, so a first attempt
    can run cheap and only work that failed a check pays for more.
    """
    lane = "judge" if node.get("kind") == "judge" else (
        "explore" if node.get("role") == "explore" else "code")
    lanes = defaults.get("lanes") if isinstance(defaults.get("lanes"), dict) else {}
    effort = node.get("effort") or lanes.get(lane) or defaults.get("effort")
    escalate = node.get("escalate", defaults.get("escalate", False))
    if escalate is True and visit > 1:
        base = EFFORTS.index(effort) if effort in EFFORTS else EFFORTS.index("medium")
        effort = EFFORTS[min(base + visit - 1, len(EFFORTS) - 1)]
    return effort if effort in EFFORTS else None


def _node_problems(node: dict, defaults: dict, *, child: bool, depth: int) -> list[str]:
    nid, kind = node["id"], node.get("kind")
    if kind not in NODE_KINDS:
        return [f"node {nid!r} kind {kind!r} must be one of {', '.join(NODE_KINDS)}"]
    problems: list[str] = []
    retries = node.get("retries", 0)
    if not (isinstance(retries, int) and not isinstance(retries, bool) and retries >= 0):
        problems.append(f"node {nid!r} retries must be a non-negative integer")
    for key in ("task", "command", "question", "summary", "saveAs", "revise", "key", "feedback", "cwd",
                "adapter", "model"):
        if key in node and not isinstance(node[key], str):
            problems.append(f"node {nid!r} {key} must be a string")
    if "timeout_seconds" in node and not _positive_int(node["timeout_seconds"]):
        problems.append(f"node {nid!r} timeout_seconds must be a positive integer")
    if "payload" in node and not isinstance(node["payload"], dict):
        problems.append(f"node {nid!r} payload must be an object")
    problems.extend(_effort_problems(node, f"node {nid!r}"))
    if kind in ("agent", "judge"):
        if not str(node.get("task") or "").strip():
            problems.append(f"{kind} {nid!r} needs a task")
        if not (node.get("adapter") or defaults.get("adapter")):
            problems.append(f"{kind} {nid!r} needs an adapter (node or defaults)")
        if kind == "agent" and node.get("role", "code") not in ("code", "explore"):
            problems.append(f"agent {nid!r} role must be code or explore")
        files = node.get("files")
        if files is not None and not (isinstance(files, str) or (
                isinstance(files, list) and all(isinstance(path, str) for path in files))):
            problems.append(f"agent {nid!r} files must be a list of paths or a template")
    elif kind == "parallel":
        branches = node.get("branches")
        if not isinstance(branches, list) or not branches or not all(isinstance(b, str) for b in branches):
            problems.append(f"parallel {nid!r} needs branches: a list of node ids")
    elif kind == "map":
        problems.extend(_map_problems(node, defaults, depth))
    elif kind == "shell":
        if not str(node.get("command") or "").strip():
            problems.append(f"shell {nid!r} needs a command")
        if "timeoutMs" in node and not _positive_int(node["timeoutMs"]):
            problems.append(f"shell {nid!r} timeoutMs must be a positive integer")
    elif kind == "gate":
        if child:
            problems.append(f"gate {nid!r} cannot run inside a map item flow")
        if not str(node.get("question") or "").strip():
            problems.append(f"gate {nid!r} needs a question")
        options = node.get("options", [])
        if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
            problems.append(f"gate {nid!r} options must be a list of strings")
    elif kind == "set":
        if not isinstance(node.get("values"), dict):
            problems.append(f"set {nid!r} needs a values object")
    elif kind == "end" and node.get("status", "pass") not in ("pass", "fail"):
        problems.append(f"end {nid!r} status must be pass or fail")
    return problems


def _parallel_problems(node: dict, by_id: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    owned: dict[str, str] = {}
    branches = node.get("branches")
    if not isinstance(branches, list) or not all(isinstance(b, str) for b in branches):
        return problems
    for branch in branches:
        target = by_id.get(branch)
        if target is None:
            problems.append(f"parallel {node['id']!r} branch {branch!r} is not a node")
            continue
        if target.get("kind") not in ("agent", "judge"):
            problems.append(f"parallel {node['id']!r} branch {branch!r} must be an agent or judge")
            continue
        if target.get("kind") == "agent" and target.get("role", "code") == "code":
            files = target.get("files")
            if not files:
                problems.append(f"parallel code branch {branch!r} needs files "
                                "(concurrent writers need disjoint scopes)")
            elif isinstance(files, list):
                for path in files:
                    if path in owned:
                        problems.append(f"parallel branches {owned[path]!r} and {branch!r} both own {path!r}")
                    owned[path] = branch
    return problems


def _map_problems(node: dict, defaults: dict, depth: int) -> list[str]:
    nid = node["id"]
    problems: list[str] = []
    if not isinstance(node.get("items"), (list, str)):
        problems.append(f"map {nid!r} items must be a list or a template")
    if ("node" in node) == ("graph" in node):
        return problems + [f"map {nid!r} needs exactly one of node or graph"]
    if depth + 1 > MAX_MAP_DEPTH:
        return problems + [f"map {nid!r} nests deeper than {MAX_MAP_DEPTH}"]
    if "concurrency" in node and not _positive_int(node["concurrency"]):
        problems.append(f"map {nid!r} concurrency must be a positive integer")
    if "feedback" in node and not isinstance(node["feedback"], str):
        problems.append(f"map {nid!r} feedback must be a template string")
    policy = node.get("pass", "all")
    if not (policy == "all" or (isinstance(policy, (int, float)) and not isinstance(policy, bool)
                                and 0 < policy <= 1)):
        problems.append(f"map {nid!r} pass must be \"all\" or a fraction in (0, 1]")
    shape = _child_graph_shape(node)
    if not isinstance(shape, dict):
        return problems + [f"map {nid!r} node/graph must be an object"]
    problems.extend(f"map {nid!r}: {problem}" for problem in
                    validate_graph(shape, child=True, depth=depth + 1, defaults=defaults))
    for item_node in shape.get("nodes") or []:
        if (isinstance(item_node, dict) and item_node.get("kind") == "agent"
                and item_node.get("role", "code") == "code" and not item_node.get("files")):
            problems.append(
                f"map {nid!r} code node {item_node.get('id')!r} needs files "
                "(items write concurrently, so each needs a disjoint scope)"
            )
    return problems


def _valid_when(when: Any) -> bool:
    if not isinstance(when, str):
        return False
    return (
        when in ("always", "ok", "fail", "PASS", "FAIL", "PARTIAL")
        or when.startswith("answer=")
        or when.startswith("out~=")
        or _STATE_PREDICATE.fullmatch(when) is not None
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _merged_defaults(parent: Optional[dict], own: Any) -> dict:
    merged = dict(parent or {})
    if isinstance(own, dict):
        combined = {
            key: {**(merged.get(key) or {}), **(own.get(key) or {})}
            for key in ("payload", "lanes")
            if isinstance(merged.get(key) or {}, dict) and isinstance(own.get(key) or {}, dict)
        }
        merged.update(own)
        merged.update({key: value for key, value in combined.items() if value})
    return merged


def back_edges(graph: dict) -> set[int]:
    """Indices of edges that close a cycle (DFS from the entry, in declaration order).

    Only these default to the ``maxLoops`` budget. A forward edge inside a loop
    is already bounded by its back-edge, so it never trips the budget first and
    blames the wrong edge.
    """
    edges = graph.get("edges") or []
    adjacency: dict[str, list[tuple[int, str]]] = {}
    for index, edge in enumerate(edges):
        adjacency.setdefault(edge["from"], []).append((index, edge["to"]))
    found: set[int] = set()
    seen: set[str] = set()

    def visit(start: str) -> None:
        # Iterative DFS: a long chain must not hit Python's recursion limit.
        seen.add(start)
        on_path = {start}
        stack = [(start, iter(adjacency.get(start, [])))]
        while stack:
            node, children = stack[-1]
            advanced = False
            for index, target in children:
                if target in on_path:
                    found.add(index)
                elif target not in seen:
                    seen.add(target)
                    on_path.add(target)
                    stack.append((target, iter(adjacency.get(target, []))))
                    advanced = True
                    break
            if not advanced:
                on_path.discard(node)
                stack.pop()

    entry = graph.get("entry")
    if entry:
        visit(entry)
    for node in adjacency:
        if node not in seen:
            visit(node)
    return found


# --------------------------------------------------------------------------
# Routing and templates


@dataclass
class NodeOutcome:
    ok: bool
    output: str = ""
    verdict: Optional[str] = None
    reason: str = ""
    error: Optional[str] = None
    answer: Optional[str] = None
    files: list[str] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    items: Optional[dict] = None
    # Durable worker logs that explain a failed launch (startup tracebacks).
    logs: list[str] = field(default_factory=list)

    def brief(self) -> dict:
        brief = {"ok": self.ok, "verdict": self.verdict, "reason": self.reason,
                 "error": self.error, "files": self.files, "task_id": self.task_id}
        if self.logs:
            brief["logs"] = self.logs
        return brief


def edge_matches(when: str, outcome: NodeOutcome, state: dict) -> bool:
    if when == "always":
        return True
    if when == "ok":
        return outcome.ok
    if when == "fail":
        return not outcome.ok
    if when in ("PASS", "FAIL", "PARTIAL"):
        return outcome.verdict == when
    if when.startswith("answer="):
        return outcome.answer is not None and _norm(outcome.answer) == _norm(when[len("answer="):])
    if when.startswith("out~="):
        return when[len("out~="):] in (outcome.output or "")
    match = _STATE_PREDICATE.fullmatch(when)
    if match is None:
        return False
    key, op, expected = match.group(1), match.group(2), match.group(3).strip()
    actual = _dig(state, key)
    if op == "~=":
        return expected in _text(actual)
    left, right = _number(actual), _number(expected)
    if left is not None and right is not None:
        return {"=": left == right, "!=": left != right, ">": left > right,
                "<": left < right, ">=": left >= right, "<=": left <= right}[op]
    if op in ("=", "!="):
        return (_text(actual) == expected) == (op == "=")
    return False


def aggregate_verdict(verdicts: list[Optional[str]], oks: list[bool]) -> Optional[str]:
    """FAIL if any FAIL; PARTIAL if any PARTIAL or any branch failed; else PASS."""
    if "FAIL" in verdicts:
        return "FAIL"
    if "PARTIAL" in verdicts or not all(oks):
        return "PARTIAL"
    if any(verdict == "PASS" for verdict in verdicts):
        return "PASS"
    return None


_MISSING = object()


def _task_digest(task: str) -> str:
    """Identity of a rendered node task, so a resumed session can tell that it changed."""
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]


def render(template: Any, run: "FlowRun", prev: str = "", *, shell: bool = False) -> str:
    """Substitute ``{{...}}`` placeholders; unknown names are left as written.

    ``shell=True`` refuses any substituted value that is not inert in sh and
    cmd.exe, so model output can never become a command.
    """
    def value(match: re.Match) -> str:
        resolved = _resolve(match.group(1), run, prev)
        if resolved is _MISSING:
            return match.group(0)
        text = _text(resolved)
        if shell and not _SHELL_SAFE.fullmatch(text):
            raise FlowError(
                f"{{{{{match.group(1)}}}}} is not safe in a shell command; "
                "read it from the PM_FLOW_CONTEXT file instead"
            )
        return text

    return _TEMPLATE.sub(value, str(template))


def render_value(value: Any, run: "FlowRun", prev: str = "") -> Any:
    """Like :func:`render`, but a field that is exactly one placeholder keeps its type."""
    if isinstance(value, str):
        match = _TEMPLATE.fullmatch(value.strip())
        if match:
            resolved = _resolve(match.group(1), run, prev)
            if resolved is not _MISSING and not isinstance(resolved, str):
                return resolved
        return render(value, run, prev)
    if isinstance(value, list):
        return [render_value(item, run, prev) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, run, prev) for key, item in value.items()}
    return value


def _resolve(name: str, run: "FlowRun", prev: str) -> Any:
    if name == "input":
        return run.input
    if name == "prev":
        return prev
    if name == "answer":
        return run.answer or ""
    head, _, rest = name.partition(".")
    if head == "out" and rest:
        return run.outputs.get(rest, "")
    if head in ("verdict", "reason") and rest:
        return (run.results.get(rest) or {}).get(head) or ""
    if head == "files" and rest:
        return "\n".join((run.results.get(rest) or {}).get("files") or [])
    if head == "state" and rest:
        found = _dig(run.state, rest)
        return "" if found is None else found
    if head in ("item", "index", "key"):
        found = _dig(run.state, name)
        return "" if found is None else found
    return _MISSING


def _dig(source: Any, dotted: str) -> Any:
    current = source
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold()


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Durable run record


@dataclass
class FlowRun:
    run_id: str
    graph: dict
    input: str = ""
    status: str = "running"
    reason: str = ""
    current: Optional[str] = None
    steps: list[dict] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    # Per node (or branch) id: the latest ok/verdict/reason/files, and for a
    # map node the per-item results that targeted repair selects from.
    results: dict[str, dict] = field(default_factory=dict)
    # Per node (or branch) id: the job and task whose provider session a later
    # visit or a continued run resumes.
    sessions: dict[str, dict] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    gate: Optional[dict] = None
    # The node visit in progress: launch attempts, jobs and child runs, so a
    # restarted walker adopts them instead of starting them again.
    inflight: Optional[dict] = None
    pid: Optional[int] = None
    backend: str = "sqlite"
    worker_mode: str = "subprocess"
    parent: Optional[dict] = None
    continued_from: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "FlowRun":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


def flows_root(state_dir: Path) -> Path:
    return Path(state_dir) / "flows"


def run_path(state_dir: Path, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(str(run_id or "")):
        raise FlowError(f"unknown flow run id {run_id!r}")
    return flows_root(state_dir) / "runs" / run_id / "run.json"


def save_run(state_dir: Path, run: FlowRun) -> None:
    """Atomically replace the run record (the caller holds the run lock or owns the run)."""
    from puppetmaster.fs_permissions import write_private_text

    run.updated_at = now_iso()
    text = json.dumps(run.to_dict(), indent=1, default=str)
    path = run_path(state_dir, run.run_id)
    for attempt in range(40):
        try:
            write_private_text(path, text, lock=False)
            return
        except PermissionError:
            # Windows refuses os.replace while a reader holds the file open.
            if attempt == 39:
                raise
            time.sleep(0.05)


def load_run(state_dir: Path, run_id: str) -> FlowRun:
    path = run_path(state_dir, run_id)
    for attempt in range(40):
        try:
            return FlowRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise FlowError(f"no flow run {run_id}") from None
        except (PermissionError, ValueError):
            if attempt == 39:
                raise
            time.sleep(0.05)
    raise FlowError(f"cannot read flow run {run_id}")


def list_runs(state_dir: Path, limit: int = 20, *, include_children: bool = False) -> list[dict]:
    root = flows_root(state_dir) / "runs"
    if not root.is_dir():
        return []
    rows = []
    paths = sorted(root.glob("flow_*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            run = FlowRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if run.parent and not include_children:
            continue
        rows.append({"run_id": run.run_id, "graph": run.graph.get("id"), "status": run.status,
                     "reason": run.reason, "steps": len(run.steps), "updated_at": run.updated_at})
        if len(rows) >= limit:
            break
    return rows


def saved_graph_path(state_dir: Path, graph_id: str) -> Path:
    if not _GRAPH_ID.fullmatch(str(graph_id or "")):
        raise FlowError(f"graph id {graph_id!r} must be kebab-case")
    return flows_root(state_dir) / "graphs" / f"{graph_id}.json"


def save_graph(state_dir: Path, graph: dict) -> Path:
    from puppetmaster.fs_permissions import write_private_text

    problems = validate_graph(graph)
    if problems:
        raise FlowError("; ".join(problems))
    path = saved_graph_path(state_dir, graph["id"])
    write_private_text(path, json.dumps(graph, indent=2))
    return path


def load_graph(state_dir: Path, ref: Any, *, base: Optional[str] = None) -> dict:
    """A graph object as given, else from a JSON file path, else a saved graph id.

    A relative path resolves against ``base`` (the caller's workspace), not
    the process directory, which for an MCP server is unrelated to the user.
    """
    if isinstance(ref, dict):
        return ref
    candidate = Path(str(ref)).expanduser()
    if base and not candidate.is_absolute():
        candidate = Path(base).expanduser() / candidate
    try:
        if candidate.suffix == ".json" or candidate.is_file():
            graph = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            path = saved_graph_path(state_dir, str(ref))
            if not path.is_file():
                raise FlowError(f"no saved graph {ref!r}")
            graph = json.loads(path.read_text(encoding="utf-8"))
    except FlowError:
        raise
    except OSError as exc:
        raise FlowError(f"cannot read graph {ref!r}: {exc}") from None
    except ValueError as exc:
        raise FlowError(f"graph {ref!r} is not valid JSON: {exc}") from None
    if not isinstance(graph, dict):
        raise FlowError(f"graph {ref!r} must be a JSON object")
    return graph


def new_run(state_dir: Path, graph: dict, input_text: str = "", *,
            state: Optional[dict] = None, parent: Optional[dict] = None,
            continue_from: Optional[str] = None, cwd: Optional[str] = None,
            backend: str = "sqlite", worker_mode: str = "subprocess") -> FlowRun:
    """Create a run. ``continue_from`` carries a prior run's sessions so its nodes resume."""
    child = parent is not None
    depth = int((parent or {}).get("depth", 0))
    problems = validate_graph(graph, child=child, depth=depth)
    if problems:
        raise FlowError("; ".join(problems))
    graph = copy.deepcopy(graph)
    # Resolve the workspace once, against the caller's workspace: a relative
    # graph cwd resolved against the process directory (an MCP server's) sent
    # write-mode workers into an unrelated tree.
    base = Path(cwd or os.getcwd()).expanduser()
    own = Path(graph.get("cwd") or ".").expanduser()
    graph["cwd"] = str((own if own.is_absolute() else base / own).resolve())
    seeded: dict[str, Any] = dict(graph.get("state") or {})
    sessions: dict[str, dict] = {}
    results: dict[str, dict] = {}
    if continue_from:
        prior = load_run(state_dir, continue_from)
        if prior.status not in TERMINAL_STATUSES:
            # Two live runs resuming the same provider sessions would interleave them.
            raise FlowError(f"cannot continue {continue_from}: it is still {prior.status}")
        sessions = copy.deepcopy(prior.sessions)
        results = copy.deepcopy(prior.results)
        seeded.update(prior.state)
        backend, worker_mode = prior.backend, prior.worker_mode
    seeded.update(state or {})
    run = FlowRun(run_id="flow_" + uuid.uuid4().hex[:12], graph=graph, input=input_text,
                  current=graph["entry"], state=seeded, sessions=sessions, results=results,
                  parent=parent, continued_from=continue_from, backend=backend,
                  worker_mode=worker_mode)
    save_run(state_dir, run)
    return run


# --------------------------------------------------------------------------
# Walk


NodeExecutor = Callable[[dict, "FlowRun", str], NodeOutcome]


def _lock(state_dir: Path, run_id: str, timeout: float):
    from puppetmaster.interprocess_lock import InterProcessFileLock

    return InterProcessFileLock.for_target(run_path(state_dir, run_id), timeout=timeout)


def walk(state_dir: Path, run_id: str, *, execute: Optional[NodeExecutor] = None,
         answer: Optional[str] = None, restart: bool = False, reset_loops: bool = False,
         extra_steps: int = 0) -> FlowRun:
    """Advance ``run_id`` until it finishes, gets stuck, or waits at a gate.

    Resumable and single-walker: the run lock admits one walker, state is
    persisted around every node, and an interrupted run resumes where it
    stopped. ``restart`` reopens a failed, stuck or stopped run.
    """
    lock = _lock(state_dir, run_id, timeout=0.5)
    try:
        lock.acquire()
    except TimeoutError:
        raise FlowError(f"flow run {run_id} is already being walked") from None
    try:
        run = load_run(state_dir, run_id)
        if not _prepare(state_dir, run, answer=answer, restart=restart,
                        reset_loops=reset_loops, extra_steps=extra_steps):
            return run
        run.pid = os.getpid()
        save_run(state_dir, run)
        if run.parent is None:
            _mark_walking(state_dir, run)
        executor = execute or JobNodeExecutor(state_dir, backend=run.backend, worker_mode=run.worker_mode)
        return _walk(state_dir, run, executor)
    finally:
        _clear_marker(state_dir, run_id, "walker.pid")
        _unmark_walking(run_id)
        lock.release()


def _mark_walking(state_dir: Path, run: FlowRun) -> None:
    from puppetmaster.fs_permissions import write_private_text
    from puppetmaster.state import walking_runs_dir

    try:
        write_private_text(walking_runs_dir() / f"{run.run_id}.json",
                           json.dumps({"state_dir": str(Path(state_dir).resolve()),
                                       "cwd": run.graph.get("cwd")}), lock=False)
    except OSError:
        pass


def _unmark_walking(run_id: str) -> None:
    from puppetmaster.state import walking_runs_dir

    try:
        (walking_runs_dir() / f"{run_id}.json").unlink()
    except OSError:
        pass


def prepare_resume(state_dir: Path, run_id: str, *, answer: Optional[str] = None,
                   restart: bool = False, reset_loops: bool = False,
                   extra_steps: int = 0) -> tuple[FlowRun, bool]:
    """Apply a gate answer or restart now, before a detached walker is spawned.

    Returns ``(run, should_walk)``. Doing the transition synchronously means a
    waiter never sees the stale ``waiting`` or terminal status in the window
    before the walker starts.
    """
    lock = _lock(state_dir, run_id, timeout=5)
    try:
        lock.acquire()
    except TimeoutError:
        return load_run(state_dir, run_id), False  # a walker is already on it
    try:
        run = load_run(state_dir, run_id)
        should_walk = _prepare(state_dir, run, answer=answer, restart=restart,
                               reset_loops=reset_loops, extra_steps=extra_steps)
        if should_walk:
            run.pid = None
            save_run(state_dir, run)
        return run, should_walk
    finally:
        lock.release()


def _prepare(state_dir: Path, run: FlowRun, *, answer: Optional[str], restart: bool,
             reset_loops: bool, extra_steps: int) -> bool:
    """Move a run to ``running`` if it may walk now; persist a rejected answer's reason."""
    if run.status == "waiting":
        if answer is None:
            return False
        options = (run.gate or {}).get("options") or []
        if options and _norm(answer) not in {_norm(str(option)) for option in options}:
            run.reason = f'answer "{answer}" is not one of: {", ".join(map(str, options))}'
            save_run(state_dir, run)
            return False
        run.gate = {**(run.gate or {}), "answer": answer}
        _clear_marker(state_dir, run.run_id, "stop")
    elif run.status in TERMINAL_STATUSES:
        if not restart or run.status not in RESTARTABLE_STATUSES:
            return False
        if extra_steps:
            run.graph.setdefault("limits", {})["maxSteps"] = len(run.steps) + int(extra_steps)
        if reset_loops:
            run.edge_counts = {}
        if run.inflight:
            # The interrupted visit continues as a new attempt: fresh launch
            # keys, so a job cut by the stop is not handed back as the result.
            run.inflight["attempt"] = int(run.inflight.get("attempt", 0)) + 1
            for key in ("branches", "children", "kept", "chosen", "unowned", "shell"):
                run.inflight.pop(key, None)
        _clear_marker(state_dir, run.run_id, "stop")
        _clear_marker(state_dir, run.run_id, "cut")
    elif restart:
        _clear_marker(state_dir, run.run_id, "stop")
    run.status, run.reason = "running", ""
    return True


def _walk(state_dir: Path, run: FlowRun, execute: NodeExecutor) -> FlowRun:
    graph = run.graph
    nodes = {node["id"]: node for node in graph["nodes"]}
    limits = graph.get("limits") or {}
    max_steps = int(limits.get("maxSteps", DEFAULT_MAX_STEPS))
    max_loops = int(limits.get("maxLoops", DEFAULT_MAX_LOOPS))
    loops = back_edges(graph)
    while True:
        if _marker(state_dir, run.run_id, "stop").exists():
            return _finish(state_dir, run, "stopped", "stopped by request")
        if len(run.steps) >= max_steps:
            return _finish(state_dir, run, "stuck", f"step-limit {max_steps} reached")
        node = nodes.get(run.current or "")
        if node is None:
            return _finish(state_dir, run, "failed", f'node "{run.current}" vanished')
        kind, prev = node["kind"], _prev(run)
        step: dict[str, Any] = {"node": node["id"], "kind": kind, "started_at": now_iso()}
        if kind == "gate":
            pending = run.gate if (run.gate or {}).get("node") == node["id"] else None
            if pending is None or pending.get("answer") is None:
                run.gate = {"node": node["id"], "question": render(node["question"], run, prev),
                            "options": node.get("options") or []}
                run.status, run.pid = "waiting", None
                run.reason = f'waiting at gate "{node["id"]}"'
                save_run(state_dir, run)
                return run
            run.answer, run.gate = str(pending["answer"]), None
            outcome = NodeOutcome(ok=True, output=run.answer, answer=run.answer)
        elif kind in _CONTROL_KINDS:
            outcome = _control_outcome(node, run, prev)
        else:
            if (run.inflight or {}).get("node") != node["id"]:
                visit = run.visits.get(node["id"], 0) + 1
                run.visits[node["id"]] = visit
                run.inflight = {"node": node["id"], "visit": visit, "attempt": 0,
                                "started_at": step["started_at"], "job_ids": []}
                save_run(state_dir, run)
            step["started_at"] = run.inflight["started_at"]
            step["visit"] = run.inflight["visit"]
            outcome = _execute_with_retries(state_dir, node, run, execute, prev)
            if _marker(state_dir, run.run_id, "stop").exists():
                # A stopped node's forced failure is not a result: do not record
                # it, spend loop budget, or route. Keep the in-flight record so a
                # restart continues this visit as a new attempt.
                return _finish(state_dir, run, "stopped", "stopped by request")
            run.inflight = None
            save_run(state_dir, run)
            _clear_marker(state_dir, run.run_id, "cut")
        step.update({"finished_at": now_iso(), "ok": outcome.ok, "verdict": outcome.verdict,
                     "reason": outcome.reason[:500], "error": outcome.error, "job_ids": outcome.job_ids,
                     "usage": outcome.usage, "preview": (outcome.output or "")[:300]})
        if outcome.items is not None:
            step["items"] = _item_counts(outcome.items)
        if outcome.logs:
            step["logs"] = outcome.logs
        run.steps.append(step)
        _add_usage(run.usage, outcome.usage)
        run.outputs[node["id"]] = (outcome.output or "")[-_OUTPUT_CHARS:]
        result = {**outcome.brief(), "visit": step.get("visit")}
        if outcome.items is not None:
            result["items"] = outcome.items
        run.results[node["id"]] = result
        if node.get("saveAs"):
            run.state[str(node["saveAs"])] = outcome.output
        if kind == "end":
            status = node.get("status", "pass")
            failed = [] if "status" in node else _failed_work(run)
            if failed:
                # An unconditional edge carried a failed node to an implicit
                # end; reporting that run as a pass woke the pilot with
                # nothing built. An explicit status still wins.
                return _finish(state_dir, run, "failed", "; ".join(failed)[:600])
            return _finish(state_dir, run, "done" if status == "pass" else "failed",
                           render(node.get("summary") or status, run, prev))
        nxt = _pick_edge(graph, node, outcome, run, max_loops, loops)
        if isinstance(nxt, tuple):
            return _finish(state_dir, run, *nxt)
        run.current = nxt
        save_run(state_dir, run)


def _pick_edge(graph: dict, node: dict, outcome: NodeOutcome, run: FlowRun,
               max_loops: int, loops: set[int]):
    if node["kind"] == "judge" and outcome.ok and outcome.verdict is None:
        # A judge that forgets its verdict line is stuck, never a quiet pass.
        return ("stuck", f'no-edge: judge "{node["id"]}" gave no verdict')
    outgoing = [(index, edge) for index, edge in enumerate(graph.get("edges") or [])
                if edge["from"] == node["id"]]
    if not outgoing:
        if outcome.ok:
            return ("done", f'ended at "{node["id"]}"')
        return ("failed", f'node-failed "{node["id"]}": {outcome.error or "failed"}')
    for index, edge in outgoing:
        if not edge_matches(edge.get("when", "always"), outcome, run.state):
            continue
        limit = edge.get("max") or (max_loops if index in loops else None)
        key = str(index)
        if limit is not None and run.edge_counts.get(key, 0) >= limit:
            return ("stuck", f'edge {edge["from"]}->{edge["to"]} taken {limit} times')
        run.edge_counts[key] = run.edge_counts.get(key, 0) + 1
        return edge["to"]
    if not outcome.ok:
        return ("failed", f'node-failed "{node["id"]}": {outcome.error or "failed"}')
    context = f" (verdict {outcome.verdict})" if outcome.verdict else ""
    if outcome.answer is not None:
        context = f' (answer "{outcome.answer}")'
    return ("stuck", f'no-edge from "{node["id"]}"{context}')


def _execute_with_retries(state_dir: Path, node: dict, run: FlowRun,
                          execute: NodeExecutor, prev: str) -> NodeOutcome:
    """Retry transport failures; a definite verdict or a user cut is an answer, not a failure."""
    attempts = 1 + int(node.get("retries", 0) or 0)
    job_ids: list[str] = []
    usage: dict[str, Any] = {}
    while True:
        try:
            outcome = execute(node, run, prev)
        except FlowError as exc:
            outcome = NodeOutcome(ok=False, error=str(exc))
        except Exception as exc:  # a node that raises is a failed node, not a dead walker
            outcome = NodeOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
        job_ids.extend(job for job in outcome.job_ids if job not in job_ids)
        _add_usage(usage, outcome.usage)
        reason = cut_reason(state_dir, run)
        if reason is not None:
            outcome.ok, outcome.verdict = False, None
            outcome.error = f"cut off by the user: {reason}"
            break
        if _marker(state_dir, run.run_id, "stop").exists():
            break
        if outcome.ok or outcome.verdict is not None or run.inflight["attempt"] + 1 >= attempts:
            break
        run.inflight["attempt"] += 1
        save_run(state_dir, run)
    outcome.job_ids, outcome.usage = job_ids, usage
    return outcome


def _control_outcome(node: dict, run: FlowRun, prev: str) -> NodeOutcome:
    """``set`` and ``end`` are bookkeeping the walker does itself, never a model call."""
    if node["kind"] == "set":
        for key, value in (node.get("values") or {}).items():
            run.state[str(key)] = render_value(value, run, prev)
    return NodeOutcome(ok=True, output=prev)


def _prev(run: FlowRun) -> str:
    return run.outputs.get(run.steps[-1]["node"], "") if run.steps else run.input


def _add_usage(total: dict, part: dict) -> None:
    for key, value in (part or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _finish(state_dir: Path, run: FlowRun, status: str, reason: str) -> FlowRun:
    run.status, run.reason, run.pid = status, reason, None
    save_run(state_dir, run)
    return run


def _marker(state_dir: Path, run_id: str, name: str) -> Path:
    return run_path(state_dir, run_id).with_name(name)


def _clear_marker(state_dir: Path, run_id: str, name: str) -> None:
    try:
        _marker(state_dir, run_id, name).unlink()
    except FileNotFoundError:
        pass


def walker_alive(state_dir: Path, run_id: str) -> bool:
    """True while a walker holds the run, or a walker spawned for it is still starting.

    A free run lock means nobody is walking, whatever ``run.pid`` says: that pid
    may since belong to another process. The spawner's marker covers the gap
    before a new walker takes the lock, for the start-up window only, and only
    while the pid still names the process that was spawned.
    """
    from puppetmaster.liveness import _pid_alive

    lock = _lock(state_dir, run_id, timeout=0)
    try:
        lock.acquire()
    except TimeoutError:
        return True
    lock.release()
    try:
        spawned = json.loads(_marker(state_dir, run_id, "walker.pid").read_text(encoding="utf-8"))
        pid = int(spawned["pid"])
        if time.time() - float(spawned.get("at", 0)) >= _SPAWN_WINDOW_SECONDS:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return pid != os.getpid() and _pid_alive(pid) and not pid_reused(pid, spawned.get("proc"))


def refresh_liveness(state_dir: Path, run_id: str) -> FlowRun:
    """Mark a ``running`` run whose walker is gone as ``interrupted``."""
    run = load_run(state_dir, run_id)
    if run.status != "running" or _recently_updated(run) or walker_alive(state_dir, run_id):
        return run
    lock = _lock(state_dir, run_id, timeout=0)
    try:
        lock.acquire()
    except TimeoutError:
        return run
    try:
        run = load_run(state_dir, run_id)
        if run.status == "running":
            run.status, run.pid = "interrupted", None
            run.reason = "the walker exited before the run finished"
            save_run(state_dir, run)
        return run
    finally:
        lock.release()


def _recently_updated(run: FlowRun, seconds: float = 10.0) -> bool:
    """A run moved to ``running`` moments ago may still be waiting for its walker to start."""
    from puppetmaster.models import parse_iso

    try:
        age = time.time() - parse_iso(run.updated_at).timestamp()
    except Exception:
        return False
    return 0 <= age < seconds


def request_stop(state_dir: Path, run_id: str) -> FlowRun:
    """Stop the run and cut what it has in flight; an idle run stops at once.

    The request is a marker file, not a ``run.json`` write, so the walker
    persisting the node it is finishing cannot overwrite it.
    """
    run = load_run(state_dir, run_id)
    if run.status in TERMINAL_STATUSES:
        return run
    _marker(state_dir, run_id, "stop").write_text(now_iso(), encoding="utf-8")
    _cut_inflight(state_dir, run)
    lock = _lock(state_dir, run_id, timeout=0.2)
    try:
        lock.acquire()
    except TimeoutError:
        return run  # a walker holds the run; it sees the marker before its next node
    try:
        run = load_run(state_dir, run_id)
        if run.status not in TERMINAL_STATUSES:
            run = _finish(state_dir, run, "stopped", "stopped by request")
        return run
    finally:
        lock.release()


def cut_node(state_dir: Path, run_id: str, reason: str = "") -> FlowRun:
    """Fail the node in flight now: its edges see ``fail`` with the reason.

    The marker names the node visit it targets, so a cut that lands just as
    that node finishes can never fail the next one.
    """
    run = load_run(state_dir, run_id)
    if not run.inflight:
        raise FlowError(f"flow run {run_id} has no node in flight")
    _marker(state_dir, run_id, "cut").write_text(json.dumps({
        "node": run.inflight.get("node"), "visit": run.inflight.get("visit"),
        "reason": reason or "no reason given"}), encoding="utf-8")
    _cut_inflight(state_dir, run)
    return run


def _root_run_id(state_dir: Path, run_id: str) -> str:
    """The top run of a map item's child run (claims are kept once per checkout run)."""
    seen = set()
    while run_id not in seen:
        seen.add(run_id)
        parent = (load_run(state_dir, run_id).parent or {}).get("run_id")
        if not parent:
            return run_id
        run_id = str(parent)
    raise FlowError(f"flow run {run_id} has a parent cycle")


def claim_paths(state_dir: Path, run_id: str, paths: list) -> list[str]:
    """Record shared paths the pilot writes while the run works in its checkout.

    A worker's write_scope gate does not charge the worker for a claimed path
    unless the worker's own events named it. Claims add up and last for the
    run. Returns every claim of the run.
    """
    from puppetmaster.fs_permissions import write_private_text
    from puppetmaster.interprocess_lock import InterProcessFileLock

    new = [str(path).strip() for path in paths or [] if str(path).strip()]
    if not new:
        raise FlowError("claim needs one or more paths or globs")
    target = _marker(state_dir, _root_run_id(state_dir, run_id), "pilot_claims.json")
    with InterProcessFileLock.for_target(target, timeout=10):
        claims = sorted(set(_read_claims(target)) | set(new))
        write_private_text(target, json.dumps(claims), lock=False)
    return claims


def pilot_claims(state_dir: Path, run_id: str) -> list[str]:
    """Every path or glob the pilot claimed in this run's checkout; [] if none."""
    try:
        return _read_claims(_marker(state_dir, _root_run_id(state_dir, run_id), "pilot_claims.json"))
    except FlowError:
        return []


def _read_claims(path: Path) -> list[str]:
    try:
        claims = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(item) for item in claims if isinstance(item, str)] if isinstance(claims, list) else []


def cut_reason(state_dir: Path, run: FlowRun) -> Optional[str]:
    """The reason of a cut aimed at the visit in flight, else None."""
    try:
        marker = json.loads(_marker(state_dir, run.run_id, "cut").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    inflight = run.inflight or {}
    if marker.get("node") != inflight.get("node") or marker.get("visit") != inflight.get("visit"):
        return None
    return str(marker.get("reason") or "no reason given")


def _interruption(state_dir: Path, run: FlowRun) -> Optional[str]:
    """Why the node in flight must stop now: a stop request, or a cut of this visit."""
    if _marker(state_dir, run.run_id, "stop").exists():
        return "stopped by request"
    reason = cut_reason(state_dir, run)
    return None if reason is None else f"cut off by the user: {reason}"


def _cut_inflight(state_dir: Path, run: FlowRun) -> None:
    """Cooperatively stop every running task and child run of the node in flight."""
    inflight = run.inflight or {}
    for child_id in (inflight.get("children") or {}).values():
        try:
            request_stop(state_dir, child_id)
        except FlowError:
            pass
    _cut_jobs(state_dir, run.backend, list(inflight.get("job_ids") or []))


_OPEN_TASK_STATUSES = (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED)


def open_work(state_dir: Path, run: FlowRun) -> list[dict]:
    """Work of the run's in-flight node, at any depth, that can still write.

    A stop is cooperative. The run can show ``stopped`` while a worker of its
    node, or of a map item's child run, still runs. Each entry is a child run
    that is not terminal, or a task that is queued, running, or blocked.
    """
    from puppetmaster.store_factory import create_store

    found: list[dict] = []
    pending, seen = [run], set()
    while pending:
        current = pending.pop()
        if current.run_id in seen:
            continue
        seen.add(current.run_id)
        inflight = current.inflight or {}
        for child_id in (inflight.get("children") or {}).values():
            try:
                child = load_run(state_dir, child_id)
            except FlowError:
                continue
            if child.status not in TERMINAL_STATUSES:
                found.append({"run_id": child.run_id, "status": child.status})
            pending.append(child)
        job_ids = list(inflight.get("job_ids") or [])
        if not job_ids:
            continue
        try:
            store = create_store(current.backend, state_dir)
            for job_id in job_ids:
                for task in store.list_tasks(job_id):
                    if task.status in _OPEN_TASK_STATUSES:
                        found.append({"run_id": current.run_id, "job_id": job_id, "task_id": task.id,
                                      "role": task.role, "status": str(task.status.value)})
        except Exception as exc:
            found.append({"run_id": current.run_id, "error": f"{type(exc).__name__}: {exc}"})
    return found


def _recut_open_work(state_dir: Path, work: list[dict]) -> None:
    """Cut again what a stopped run left open; ``cut_task`` is idempotent.

    The in-flight watch ends with the walker, so a cut that failed on a busy
    store got no retry after the stop.
    """
    by_run: dict[str, list[str]] = {}
    for entry in work:
        if entry.get("job_id"):
            by_run.setdefault(entry["run_id"], []).append(entry["job_id"])
    for run_id, job_ids in by_run.items():
        try:
            backend = load_run(state_dir, run_id).backend
        except FlowError:
            continue
        _cut_jobs(state_dir, backend, sorted(set(job_ids)))


def wait_for_event(state_dir: Path, run_id: str, timeout_seconds: float = 0.0,
                   poll_seconds: float = 0.5) -> FlowRun:
    """Block until the run finishes, gets stuck, waits at a gate, or loses its walker.

    A stopped run wakes only when its open work settles (or at the timeout):
    ``stopped`` alone does not prove that its workers stopped writing.
    """
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    recut_at = 0.0
    while True:
        run = refresh_liveness(state_dir, run_id)
        if run.status in WAKE_STATUSES:
            if run.status != "stopped":
                return run
            work = open_work(state_dir, run)
            if not work:
                return run
            if time.monotonic() >= recut_at:
                recut_at = time.monotonic() + 5
                _recut_open_work(state_dir, work)
        if deadline is not None and time.monotonic() >= deadline:
            return run
        time.sleep(poll_seconds)


def run_summary(run: FlowRun, *, since: int = 0) -> dict:
    """The compact view a pilot wakes up to; ``since`` returns only newer steps."""
    keep = ("node", "kind", "visit", "ok", "verdict", "reason", "error", "items", "job_ids")
    steps = [
        {key: step[key] for key in keep if step.get(key) not in (None, "", [], {})}
        for step in run.steps[max(0, since):]
    ]
    summary: dict[str, Any] = {
        "run_id": run.run_id,
        "graph": run.graph.get("id"),
        "status": run.status,
        "reason": run.reason,
        "current": run.current,
        "steps_total": len(run.steps),
        "next_since": len(run.steps),
        "steps": steps,
        "usage": run.usage,
    }
    if run.gate:
        summary["gate"] = run.gate
    def problem(step: dict) -> bool:
        return not step.get("ok") or step.get("verdict") in ("FAIL", "PARTIAL")

    # A done run can still end on a judge's FAIL (a node with no edges ends the
    # run); the pilot must see it either way.
    if run.steps and (run.status != "done" or problem(run.steps[-1])):
        failing = next((step for step in reversed(run.steps) if problem(step)), None)
        if failing:
            summary["last_problem"] = {key: failing[key] for key in ("node", "verdict", "reason", "error")
                                       if failing.get(key)}
    return summary


def spawn_background_walk(state_dir: Path, run_id: str, *, answer: Optional[str] = None,
                          restart: bool = False, popen: Optional[Callable[..., Any]] = None) -> int:
    """Walk the run in a detached process; the caller returns at once.

    ``popen`` replaces ``subprocess.Popen`` (same signature); see JobNodeExecutor.
    """
    popen = popen or subprocess.Popen
    run = load_run(state_dir, run_id)
    command = [sys.executable, "-m", "puppetmaster", "--state-dir", str(state_dir),
               "--backend", run.backend, "flow", "resume", run_id]
    if answer is not None:
        command.append(f"--answer={answer}")
    if restart:
        command.append("--restart")
    log = run_path(state_dir, run_id).with_name("walker.log")
    env = dict(os.environ)
    # The walker passes explicit launch keys; an inherited key would collide.
    env.pop("PUPPETMASTER_LAUNCH_KEY", None)
    with open(log, "ab") as handle:
        kwargs: dict[str, Any] = {"stdout": handle, "stderr": handle,
                                  "stdin": subprocess.DEVNULL, "env": env}
        if os.name == "nt":
            # DETACHED_PROCESS | NEW_PROCESS_GROUP, breaking away from the
            # host's job object so the walker outlives the pilot that started it.
            flags = 0x00000008 | 0x00000200
            try:
                process = popen(command, creationflags=flags | 0x01000000, **kwargs)
            except OSError:
                process = popen(command, creationflags=flags, **kwargs)
        else:
            process = popen(command, start_new_session=True, **kwargs)
    _marker(state_dir, run_id, "walker.pid").write_text(json.dumps({
        "pid": process.pid, "at": time.time(), "proc": process_identity(process.pid)}), encoding="utf-8")
    return process.pid


def _fail_unwalked(state_dir: Path, run_id: str, reason: str) -> None:
    """Fail a run only if no walker holds it; never race a slow-starting walker's writes."""
    lock = _lock(state_dir, run_id, timeout=0)
    try:
        lock.acquire()
    except TimeoutError:
        return
    try:
        child = load_run(state_dir, run_id)
        if child.status not in TERMINAL_STATUSES:
            _finish(state_dir, child, "failed", reason)
    finally:
        lock.release()


def _item_counts(items: dict) -> dict:
    counts: dict[str, int] = {}
    for item in items.values():
        status = item.get("status") or "?"
        counts[status] = counts.get(status, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Node execution on Puppetmaster jobs


def _startup_logs(store, job_id: str, role: str) -> list[str]:
    """The durable startup-failure logs a worker of ``role`` left for this job."""
    try:
        task_dir = store.job_dir(job_id) / "tasks"
        return sorted(str(path) for path in task_dir.glob(f"startup_error-worker-{role}-*.log"))
    except OSError:
        return []


class JobNodeExecutor:
    """Run agent, judge, parallel and map nodes as Puppetmaster jobs; shell locally.

    ``popen`` is the process factory for shell nodes and the default walker
    spawn: it takes ``subprocess.Popen``'s arguments and returns a Popen-like
    object (``pid``, ``poll``, ``wait``, ``kill``, ``returncode``). A host uses
    it to bind each new process group to its own runtime before the command
    runs. The keyword arguments always start a new session (process group on
    Windows), so the group a host binds is the one stop and timeout kill.
    """

    def __init__(self, state_dir: Path, *, backend: str = "sqlite", worker_mode: str = "subprocess",
                 spawn: Optional[Callable[[Path, str], Any]] = None, poll_seconds: float = 1.0,
                 popen: Optional[Callable[..., Any]] = None) -> None:
        self.state_dir = Path(state_dir)
        self.backend = backend
        self.worker_mode = worker_mode
        self.popen = popen or subprocess.Popen
        self.spawn = spawn or (lambda state_dir, run_id: spawn_background_walk(
            state_dir, run_id, popen=self.popen))
        self.poll_seconds = poll_seconds

    def __call__(self, node: dict, run: FlowRun, prev: str) -> NodeOutcome:
        kind = node["kind"]
        if kind in ("agent", "judge"):
            return self._run_members(run, [(node["id"], node)], prev, combine=False)
        if kind == "parallel":
            nodes = {item["id"]: item for item in run.graph["nodes"]}
            return self._run_members(run, [(branch, nodes[branch]) for branch in node["branches"]],
                                     prev, combine=True)
        if kind == "map":
            return self._run_map(node, run, prev)
        if kind == "shell":
            return self._run_shell(node, run, prev)
        return NodeOutcome(ok=False, error=f"unsupported node kind {kind!r}")

    # -- agents, judges and parallel branches ------------------------------

    def _run_members(self, run: FlowRun, members: list[tuple[str, dict]], prev: str, *,
                     combine: bool) -> NodeOutcome:
        inflight = run.inflight
        visit, attempt = inflight["visit"], inflight["attempt"]
        done: dict[str, dict] = inflight.setdefault("branches", {})
        if combine and visit > 1 and not done:
            prior = {key: run.results[key] for key, _ in members if key in run.results}
            chosen = select_targets([key for key, _ in members], prior, _feedback(run, prev), repair=True)
            for key, _ in members:
                if key not in chosen and key in prior:
                    done[key] = {**prior[key], "output": run.outputs.get(key, ""), "kept": True}
        # A branch already settled in this attempt (a walker crashed after
        # recording it) is reused, never relaunched under the same launch key.
        todo = [(key, node) for key, node in members
                if not (done.get(key) or {}).get("ok") and (done.get(key) or {}).get("attempt") != attempt]
        if todo:
            specs = [self._spec(key, node, run, prev, visit) for key, node in todo]
            launch_key = f"flow:{run.run_id}:{run.current}:{visit}:{attempt}"
            goal = specs[0].instruction if len(specs) == 1 else (run.input or specs[0].instruction)
            job_id, error, store = self._launch(run, specs, launch_key, goal)
            for (key, node), spec in zip(todo, specs):
                if job_id is None:
                    outcome = NodeOutcome(ok=False, error=error or "job was not created")
                else:
                    outcome = task_outcome(store, job_id, spec.role)
                    if error and not outcome.ok:
                        # The launch error is the cause; the task's own state
                        # ("no task for <role>", "task queued") only hid it.
                        state = outcome.error
                        outcome.error = error if not state or state == error else f"{error} [task: {state}]"
                        outcome.logs = _startup_logs(store, job_id, spec.role)
                    if outcome.task_id:
                        run.sessions[key] = {"job_id": job_id, "task_id": outcome.task_id,
                                             "adapter": spec.adapter, "run_id": run.run_id,
                                             "visit": visit, "attempt": attempt,
                                             "task_sha": _task_digest(render(node["task"], run, prev))}
                    scope = spec.payload.get("write_scope")
                    if scope:
                        # A shared workspace's snapshot also holds siblings' edits.
                        outcome.files = [path for path in outcome.files if _in_scope(path, scope)]
                if combine and node["kind"] == "judge" and outcome.ok and outcome.verdict is None:
                    outcome.verdict = "PARTIAL"  # a judge branch without a verdict never counts as a pass
                    outcome.reason = outcome.reason or "judge gave no verdict"
                done[key] = {**outcome.brief(), "output": outcome.output[-_OUTPUT_CHARS:],
                             "usage": outcome.usage, "job_id": job_id, "attempt": attempt}
                if combine:
                    run.results[key] = {**outcome.brief(), "visit": visit}
                    run.outputs[key] = outcome.output[-_OUTPUT_CHARS:]
            save_run(self.state_dir, run)
        jobs = list(inflight.get("job_ids") or [])
        if not combine:
            branch = done[members[0][0]]
            return NodeOutcome(ok=bool(branch.get("ok")), output=branch.get("output") or "",
                               verdict=branch.get("verdict"), reason=branch.get("reason") or "",
                               error=branch.get("error"), files=branch.get("files") or [],
                               job_ids=jobs, usage=branch.get("usage") or {}, task_id=branch.get("task_id"),
                               logs=branch.get("logs") or [])
        branches = [done[key] for key, _ in members]
        oks = [bool(branch.get("ok")) for branch in branches]
        transport = any(not branch.get("ok") and branch.get("verdict") is None for branch in branches)
        usage: dict[str, Any] = {}
        for branch in branches:
            if not branch.get("kept"):
                _add_usage(usage, branch.get("usage") or {})
        failing = [f"{key}: {done[key].get('reason') or done[key].get('error') or done[key].get('verdict')}"
                   for key, _ in members
                   if not done[key].get("ok") or done[key].get("verdict") in ("FAIL", "PARTIAL")]
        return NodeOutcome(
            ok=all(oks),
            output="\n\n".join(f"## {key}\n{done[key].get('output') or ''}" for key, _ in members),
            # A branch that failed without a verdict is a transport failure:
            # leave the verdict open so a retry reruns just those branches.
            verdict=None if transport else aggregate_verdict([b.get("verdict") for b in branches], oks),
            reason="; ".join(failing)[:1000],
            error=None if all(oks) else "branch failed: " + ", ".join(
                key for key, _ in members if not done[key].get("ok")),
            files=sorted({path for branch in branches for path in branch.get("files") or []}),
            job_ids=jobs, usage=usage,
        )

    def _spec(self, key: str, node: dict, run: FlowRun, prev: str, visit: int):
        from puppetmaster.workers import WorkerSpec

        defaults = run.graph.get("defaults") or {}
        judge = node["kind"] == "judge"
        read_only = judge or node.get("role") == "explore"
        task = render(node["task"], run, prev)
        payload: dict[str, Any] = {**(defaults.get("payload") or {}), **(node.get("payload") or {})}
        if read_only:
            payload.update(_READ_ONLY)
            payload["terminal_verdict"] = True
            if judge:
                task = f"{task}\n\n{_JUDGE_VERDICT}"
            else:
                revision = self._revision_context(run, prev, visit)
                if revision:
                    task = f"{task}\n\n{revision}"
        else:
            payload.update(_WRITE)
            files = render_value(node.get("files"), run, prev)
            if isinstance(files, str):
                files = [line.strip() for line in files.splitlines() if line.strip()]
            if files:
                payload["write_scope"] = [str(path) for path in files]
            peers = (run.parent or {}).get("peer_scopes")
            if peers:
                # Sibling map items edit the same workspace in their own jobs.
                payload["peer_write_scopes"] = list(peers)
            revision = self._revision_context(run, prev, visit)
            if revision:
                # A resumed session gets the delta alone; a fresh start (resume
                # unavailable or turned off) must still see what to fix.
                task = f"{task}\n\n{revision}"
            task = f"{task}\n\n{_BUILD_VERDICT}"
        payload["cwd"] = run.graph["cwd"]
        model = node.get("model") or defaults.get("model")
        if model:
            payload["model"] = model
        timeout = node.get("timeout_seconds") or defaults.get("timeout_seconds")
        if timeout:
            payload["timeout_seconds"] = int(timeout)
        effort = node_effort(node, defaults, visit)
        if effort:
            payload["reasoning_effort"] = effort
        # Keep provider sessions so a later visit or a follow-up run can resume them.
        payload["ephemeral"] = False
        payload["flow"] = {"run_id": run.run_id, "node": key, "visit": visit}
        session = run.sessions.get(key)
        if session and node.get("resume", True) is not False:
            payload["resume_from"] = {"job_id": session["job_id"], "task_id": session["task_id"]}
            payload["resume_prompt"] = self._delta(node, run, prev, visit, task, session=session)
        return WorkerSpec(role=key, instruction=task,
                          adapter=node.get("adapter") or defaults["adapter"], payload=payload)

    @staticmethod
    def _revision_context(run: FlowRun, prev: str, visit: int) -> str:
        if visit > 1:
            feedback = prev.strip() or "(the previous step gave no details)"
            return (f"This is revision {visit}; earlier work is already in the files. "
                    f"Feedback to act on:\n{feedback}")
        if run.continued_from:
            return f"Follow-up request on earlier work that is already in the files:\n{run.input}"
        return ""

    @staticmethod
    def _delta(node: dict, run: FlowRun, prev: str, visit: int, task: str, *,
               session: Optional[dict] = None) -> str:
        """What a resumed session needs: the change, not the whole task again."""
        closing = _JUDGE_VERDICT if node["kind"] == "judge" else _BUILD_VERDICT
        if node.get("revise"):
            return f"{render(node['revise'], run, prev)}\n\n{closing}"
        if session and session.get("run_id") == run.run_id and session.get("visit") == visit:
            # Same run and visit: an earlier attempt failed or was interrupted.
            # (A continued run or a map repair copies sessions from another run,
            # whose visit numbers are not this run's.)
            return ("Your previous attempt at this task stopped before it finished. Your work so far "
                    f"is on disk; continue the task and finish.\n\n{task}")
        if node["kind"] == "judge":
            return ("The work changed since your last review. Review it again against the same "
                    "task: confirm the problems you flagged are fixed and look for regressions.\n\n"
                    f"{task}")
        if visit > 1:
            feedback = prev.strip() or "(the previous step gave no details)"
            return (f"Revision {visit} of your task. Your earlier work in this session is already on "
                    f"disk.\n\nFeedback to act on:\n{feedback}\n\nFix what the feedback asks, rerun "
                    f"the checks, and finish.\n\n{_BUILD_VERDICT}")
        # A continued run can change the node task, give no input, or both. The
        # resumed session saw only its old task, so send the current one when
        # it changed (or is unknown), or when there is no request to act on.
        request = run.input.strip()
        base = render(node["task"], run, prev)
        recorded = (session or {}).get("task_sha")
        parts = [f"Follow-up request:\n{run.input}"] if request else []
        if recorded != _task_digest(base) or not request:
            if recorded is None:
                label = "Your current task (it can differ from your earlier task):"
            elif recorded != _task_digest(base):
                label = "Your task changed. The updated task:"
            else:
                label = "Do the same task again. Confirm that it is done and fix what is not:"
            parts.append(f"{label}\n{base}")
        context = f"\n\nContext from the previous step:\n{prev}" if prev and prev != run.input else ""
        return ("\n\n".join(parts) + f"{context}\n\nYour earlier work in this session is on "
                f"disk. Do the work above, rerun the checks, and finish.\n\n{_BUILD_VERDICT}")

    def _launch(self, run: FlowRun, specs: list, launch_key: str, goal: str):
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.store import LaunchConflictError
        from puppetmaster.store_factory import create_store

        store = create_store(self.backend, self.state_dir)
        jobs: list[str] = run.inflight["job_ids"]
        known = len(jobs)

        def created(job) -> None:
            if job.id not in jobs:
                jobs.append(job.id)
                save_run(self.state_dir, run)

        lease = int((run.graph.get("defaults") or {}).get("lease_seconds", DEFAULT_LEASE_SECONDS))
        orchestrator = Orchestrator(store)
        if _marker(self.state_dir, run.run_id, "stop").exists() or cut_reason(self.state_dir, run) is not None:
            return None, "stopped before launch", store
        error = None
        watch = _InflightWatch(self.state_dir, run, self.backend)
        watch.start()
        try:
            orchestrator.run(goal, specs=specs, launch_key=launch_key, lease_seconds=lease,
                             worker_mode=self.worker_mode, on_job_created=created,
                             label=f"flow {run.graph.get('id')}: {run.current}",
                             origin="flow", session_id=run.run_id)
        except LaunchConflictError as exc:
            return None, f"launch key conflict, the node changed under an interrupted run: {exc}", store
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            watch.stop()
        if len(jobs) == known and not self._adopting(run, launch_key, store):
            return None, error or "job was not created", store
        job_id = jobs[-1]
        if store.get_job(job_id).status not in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
            # An earlier walker started this job and died; finish it instead of buying it again.
            try:
                orchestrator.adopt(job_id, lease_seconds=lease, worker_mode=self.worker_mode, specs=specs)
            except Exception as exc:
                adopt_error = f"{type(exc).__name__}: {exc}"
                # Keep the launch's own error first: adoption failing too must
                # not replace the cause the launch already reported.
                error = adopt_error if error is None else f"{error}; then adopting the job: {adopt_error}"
        return job_id, error, store

    @staticmethod
    def _adopting(run: FlowRun, launch_key: str, store: Any) -> bool:
        """True when this attempt's job already existed (a restarted walker re-launched it)."""
        jobs = run.inflight.get("job_ids") or []
        if not jobs:
            return False
        try:
            return store.get_job(jobs[-1]).launch_key == launch_key
        except Exception:
            return False

    # -- map: per-item child flows ------------------------------------------

    def _run_map(self, node: dict, run: FlowRun, prev: str) -> NodeOutcome:
        inflight = run.inflight
        visit = inflight["visit"]
        template = node["items"]
        if isinstance(template, str):
            match = _TEMPLATE.fullmatch(template.strip())
            raw = _resolve(match.group(1), run, prev) if match else None
            if match and (raw is _MISSING or raw is None or raw == "" or (
                    match.group(1).startswith("state.") and _dig(run.state, match.group(1)[6:]) is None)):
                # A typo must not become "0/0 items passed".
                raise FlowError(f"map {node['id']!r} items {template!r} resolved to nothing")
        items = render_value(template, run, prev)
        if isinstance(items, str):
            try:
                items = json.loads(items) if items.strip() else []
            except ValueError:
                raise FlowError(f"map {node['id']!r} items did not render to a JSON list") from None
        if not isinstance(items, list):
            raise FlowError(f"map {node['id']!r} items must be a list")
        keys = [item_key(node, item, index) for index, item in enumerate(items)]
        if len(set(keys)) != len(keys):
            raise FlowError(f"map {node['id']!r} item keys are not unique")
        prior = dict((run.results.get(node["id"]) or {}).get("items") or {})
        children: dict[str, str] = inflight.setdefault("children", {})
        repeat = bool(prior) and (visit > 1 or bool(run.continued_from))
        if node.get("feedback"):
            feedback = render(node["feedback"], run, prev)
        else:
            feedback = _feedback(run, prev) if visit > 1 else run.input
        if "kept" not in inflight:
            chosen = (select_targets(keys, prior, feedback, repair=visit > 1) if repeat else set(keys))
            inflight["kept"] = [key for key in keys if key not in chosen]
            inflight["chosen"] = [key for key in keys if key in chosen]
            if repeat:
                owned = {path for item in prior.values() for path in item.get("files") or []}
                owned.update(path for paths in _item_scopes(child_graph_for(node, run), keys, items).values()
                             for path in paths)
                inflight["unowned"] = unowned_paths(feedback, owned)
            save_run(self.state_dir, run)
        child_graph = child_graph_for(node, run)
        depth = int((run.parent or {}).get("depth", 0)) + 1
        scopes = _item_scopes(child_graph, keys, items)
        _check_disjoint(node["id"], scopes)
        interrupted = _interruption(self.state_dir, run)
        if interrupted and not children:
            return NodeOutcome(ok=False, error=interrupted)
        created = False
        for index, (key, item) in enumerate(zip(keys, items)):
            if interrupted or key not in inflight["chosen"] or key in children:
                continue
            peers = sorted({path for other, owned in scopes.items() if other != key for path in owned})
            continued = (prior.get(key) or {}).get("run_id") if repeat else None
            owned_files = list(scopes.get(key) or []) + list((prior.get(key) or {}).get("files") or [])
            child_input = item_feedback(feedback, key, owned_files) if repeat else run.input
            child = new_run(self.state_dir, child_graph, child_input,
                            state={"item": item, "index": index, "key": key},
                            parent={"run_id": run.run_id, "node": node["id"], "key": key, "depth": depth,
                                    "peer_scopes": peers},
                            continue_from=continued, backend=run.backend, worker_mode=run.worker_mode)
            children[key] = child.run_id
            created = True
        if created:
            # One save after the loop: children are not walked until recorded,
            # so a crash mid-loop leaves only unwalked records behind.
            save_run(self.state_dir, run)
        self._drive_children(run, node, children)
        merged: dict[str, dict] = {key: prior[key] for key in inflight.get("kept") or [] if key in prior}
        usage: dict[str, Any] = {}
        for key, child_id in children.items():
            child = load_run(self.state_dir, child_id)
            merged[key] = _item_result(child)
            _add_usage(usage, child.usage)
        ordered = {key: merged[key] for key in keys if key in merged}
        outcome = _map_outcome(node, ordered, usage)
        unowned = inflight.get("unowned") or []
        if unowned:
            # No item will repair these; say so instead of looping on them silently.
            note = "feedback names files no item owns: " + ", ".join(unowned[:10])
            outcome.output = f"{outcome.output}\n{note}"
            outcome.reason = f"{outcome.reason}; {note}" if outcome.reason else note
        return outcome

    def _drive_children(self, run: FlowRun, node: dict, children: dict[str, str]) -> None:
        concurrency = int(node.get("concurrency", DEFAULT_MAP_CONCURRENCY))
        started: dict[str, float] = {}
        spawns: dict[str, int] = {}
        finished: dict[str, str] = {}
        stopping = False
        while True:
            if not stopping and _interruption(self.state_dir, run):
                stopping = True
                for child_id in children.values():
                    if child_id not in finished:
                        request_stop(self.state_dir, child_id)
            states = dict(finished)
            for child_id in children.values():
                if child_id not in finished:
                    states[child_id] = refresh_liveness(self.state_dir, child_id).status
                    if states[child_id] in TERMINAL_STATUSES:
                        finished[child_id] = states[child_id]
            if all(status in TERMINAL_STATUSES for status in states.values()):
                return
            now = time.monotonic()
            live = {child_id for child_id, status in states.items()
                    if status == "running" and (now - started.get(child_id, -1e9) < 15
                                                or walker_alive(self.state_dir, child_id))}
            for child_id, status in states.items():
                if stopping or len(live) >= concurrency:
                    break
                if status in TERMINAL_STATUSES or child_id in live:
                    continue
                if spawns.get(child_id, 0) >= 4:
                    _fail_unwalked(self.state_dir, child_id, "the item walker kept exiting")
                    continue
                spawns[child_id] = spawns.get(child_id, 0) + 1
                started[child_id] = now
                self.spawn(self.state_dir, child_id)
                live.add(child_id)
            time.sleep(self.poll_seconds)

    # -- shell ----------------------------------------------------------------

    def _run_shell(self, node: dict, run: FlowRun, prev: str) -> NodeOutcome:
        """Run a shell node; a stop, a cut or the timeout kills its process tree.

        Shell nodes are at-least-once: the exit status of a command whose walker
        died is lost, so a resumed walker settles that orphan and runs it again.
        """
        from puppetmaster.fs_permissions import write_private_text

        command = render(node["command"], run, prev, shell=True)
        cwd = run.graph["cwd"]
        if node.get("cwd"):
            cwd = str((Path(run.graph["cwd"]) / render(node["cwd"], run, prev, shell=True)).resolve())
        timeout = float(node.get("timeoutMs", 600_000)) / 1000.0
        inflight = run.inflight
        previous = inflight.get("shell") or {}
        if previous.get("pid"):
            interrupted = self._settle_orphan(int(previous["pid"]), previous.get("proc"), run, timeout)
            if interrupted:
                return NodeOutcome(ok=False, error=interrupted)
        directory = run_path(self.state_dir, run.run_id).parent
        stem = f"shell-{node['id']}-{inflight['visit']}-{inflight['attempt']}"
        context = directory / f"{stem}.context.json"
        write_private_text(context, json.dumps({
            "input": run.input, "prev": prev, "state": run.state, "outputs": run.outputs,
            "results": run.results, "run_id": run.run_id,
        }, default=str))
        log = directory / f"{stem}.log"
        env = {**os.environ, "PM_FLOW_CONTEXT": str(context), "PM_FLOW_RUN_ID": run.run_id}
        kwargs: dict[str, Any] = {"shell": True, "cwd": cwd, "env": env, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000200  # NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with open(log, "wb") as sink:
            process = self.popen(command, stdout=sink, stderr=subprocess.STDOUT, **kwargs)
            inflight["shell"] = {"pid": process.pid, "proc": process_identity(process.pid)}
            save_run(self.state_dir, run)
            deadline = time.monotonic() + timeout
            interrupted = None
            while process.poll() is None:
                interrupted = (f"timed out after {timeout:.0f}s" if time.monotonic() >= deadline
                               else _interruption(self.state_dir, run))
                if interrupted:
                    if not _kill_tree(process.pid):
                        process.kill()
                    process.wait()
                    break
                time.sleep(0.1)
        output = log.read_bytes()[-_OUTPUT_CHARS * 4:].decode("utf-8", errors="replace")[-_OUTPUT_CHARS:]
        if interrupted:
            return NodeOutcome(ok=False, output=output, error=interrupted)
        ok = process.returncode == 0
        verdict = parse_terminal_verdict(output)
        return NodeOutcome(ok=ok, output=output, error=None if ok else f"exit {process.returncode}",
                           verdict=verdict.verdict if verdict else None,
                           reason=verdict.reason if verdict else "")

    def _settle_orphan(self, pid: int, recorded: Optional[str], run: FlowRun,
                       timeout: float) -> Optional[str]:
        """Deal with the command a dead walker left running; return a stop or cut reason.

        A process that is provably that command is killed, so two copies never
        run at once. A pid that now names another process is left alone. One
        that cannot be told apart is waited out, as long as nobody stops or cuts.
        """
        from puppetmaster.liveness import _pid_alive

        if not _pid_alive(pid):
            return None
        current = process_identity(pid)
        if recorded and current:
            if current == recorded:
                _kill_tree(pid)
                gone = time.monotonic() + 5
                while _pid_alive(pid) and process_identity(pid) == recorded and time.monotonic() < gone:
                    time.sleep(0.05)
            return None
        deadline = time.monotonic() + timeout
        while _pid_alive(pid) and time.monotonic() < deadline:
            interrupted = _interruption(self.state_dir, run)
            if interrupted:
                return interrupted
            time.sleep(0.5)
        return None


class _InflightWatch:
    """Cut the node's own job the moment a stop or a cut for this visit lands.

    ``request_stop`` and ``cut_node`` cut the jobs they can see; this closes
    the window where the job or its tasks did not exist yet when they looked.
    """

    def __init__(self, state_dir: Path, run: FlowRun, backend: str, interval: float = 1.0) -> None:
        import threading

        self.state_dir, self.run, self.backend, self.interval = Path(state_dir), run, backend, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="flow-inflight-watch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if (_marker(self.state_dir, self.run.run_id, "stop").exists()
                    or cut_reason(self.state_dir, self.run) is not None):
                _cut_jobs(self.state_dir, self.backend, list(self.run.inflight.get("job_ids") or []))


def _cut_jobs(state_dir: Path, backend: str, job_ids: list[str]) -> None:
    """Cut each open task of these jobs. A failed cut is retried by the next call.

    The in-flight watch calls this each second while a stop or cut is in
    effect, and ``cut_task`` finishes a half-done cut, so a busy store only
    delays the cut. Each failure goes to stderr (the walker log): a cut that
    fails with no trace leaves a worker that runs on after a stop.
    """
    try:
        from puppetmaster.store_factory import create_store

        store = create_store(backend, state_dir)
    except Exception as exc:
        print(f"[flow] cut: store unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    for job_id in job_ids:
        try:
            tasks = store.list_tasks(job_id)
        except Exception as exc:
            print(f"[flow] cut: cannot list {job_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for task in tasks:
            if task.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED):
                continue
            try:
                store.cut_task(job_id, task.id)
            except Exception as exc:
                print(f"[flow] cut of {job_id}/{task.id} failed, will retry: {type(exc).__name__}: {exc}",
                      file=sys.stderr)


def _in_scope(path: str, scope: list) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(path, str(glob)) or path == str(glob).rstrip("/") for glob in scope)


def _kill_tree(pid: int) -> bool:
    """Kill a shell node's command and everything it started; it leads its own group."""
    if os.name == "nt":
        from puppetmaster.win_process import kill_process_tree

        return kill_process_tree(pid)
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def item_key(node: dict, item: Any, index: int) -> str:
    template = node.get("key")
    if template:
        probe = FlowRun(run_id="flow_000000000000", graph={}, state={"item": item, "index": index})
        raw = render(template, probe)
    elif isinstance(item, dict) and any(isinstance(item.get(name), (str, int)) and not isinstance(item.get(name), bool)
                                        for name in ("id", "key", "name")):
        raw = next(str(item[name]) for name in ("id", "key", "name")
                   if isinstance(item.get(name), (str, int)) and not isinstance(item.get(name), bool))
    elif isinstance(item, (str, int)) and not isinstance(item, bool):
        raw = str(item)
    else:
        raw = str(index)
    key = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")[:48]
    return key or str(index)


def item_feedback(feedback: str, key: str, files: list[str]) -> str:
    """The lines of shared feedback that concern one item; all of it when none do.

    A repair delta is smaller and stays inside the item's own files when the
    item does not read every other item's problems.
    """
    lines = (feedback or "").splitlines()
    nameable = len(key) >= 2 and not key.isdigit()
    pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(key)}(?![A-Za-z0-9_-])") if nameable else None
    mine = [line for line in lines
            if (pattern is not None and pattern.search(line)) or any(path and path in line for path in files)]
    return "\n".join(mine) if mine else (feedback or "")


def _feedback(run: FlowRun, prev: str) -> str:
    """Feedback for targeted repair; a node's own digest (a self-loop) names every item, so it is not feedback."""
    return "" if run.steps and run.steps[-1]["node"] == run.current else prev


def select_targets(keys: list[str], prior: dict[str, dict], feedback: str, *,
                   repair: bool = True) -> set[str]:
    """Which items or branches run again.

    A repair visit reruns the ones the feedback names plus the ones that
    failed; a follow-up reruns the ones the request names, or all of them.
    A key is matched by name only when it is at least two characters and not
    purely digits (``a`` and ``12`` collide with prose and line numbers); any
    key is matched through the files it owns.
    """
    text = feedback or ""
    named = set()
    for key in keys:
        files = [path for path in (prior.get(key) or {}).get("files") or [] if path]
        nameable = len(key) >= 2 and not key.isdigit()
        mentioned = nameable and re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(key)}(?![A-Za-z0-9_-])", text) is not None
        if mentioned or any(path in text for path in files):
            named.add(key)
    if not repair:
        return named or set(keys)
    failing = set()
    for key in keys:
        entry = prior.get(key)
        if entry is None or not entry.get("ok") or entry.get("verdict") in ("FAIL", "PARTIAL"):
            failing.add(key)
    return (named | failing) or set(keys)


_PATH_MENTION = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,8})(?::\d+)?")


def unowned_paths(feedback: str, owned: set) -> list[str]:
    """File paths the feedback mentions that no item declared or changed."""
    found: list[str] = []
    for match in _PATH_MENTION.finditer(feedback or ""):
        path = match.group(1)
        if "/" in path and path not in owned and path not in found:
            found.append(path)
    return found


def child_graph_for(node: dict, run: FlowRun) -> dict:
    shape = _child_graph_shape(node)
    defaults = _merged_defaults(run.graph.get("defaults"), shape.get("defaults"))
    slug = re.sub(r"[^a-z0-9]+", "-", node["id"].lower()).strip("-") or "map"
    return {
        "id": f"{run.graph.get('id', 'flow')}-{slug}",
        "cwd": run.graph["cwd"],
        "defaults": defaults,
        "limits": shape.get("limits") or {},
        "entry": shape["entry"],
        "nodes": shape["nodes"],
        "edges": shape.get("edges") or [],
    }


def _check_disjoint(node_id: str, scopes: dict[str, list[str]]) -> None:
    """Map items write one workspace concurrently: their scopes must not overlap."""
    from puppetmaster.conflicts import scopes_overlap

    owner: dict[str, str] = {}
    globbed: list[tuple[str, list[str]]] = []
    for key, paths in scopes.items():
        for path in paths:
            if path in owner and owner[path] != key:
                raise FlowError(f"map {node_id!r} items {owner[path]!r} and {key!r} both own {path!r}")
            owner[path] = key
        if any(ch in path for path in paths for ch in "*?["):
            globbed.append((key, paths))
    for key, paths in globbed:
        for other, other_paths in scopes.items():
            if other != key and scopes_overlap(paths, other_paths):
                raise FlowError(f"map {node_id!r} items {key!r} and {other!r} have overlapping files")


def _item_scopes(child_graph: dict, keys: list[str], items: list) -> dict[str, list[str]]:
    """Each item's declared write scope: the rendered ``files`` of its code nodes."""
    scopes: dict[str, list[str]] = {}
    for index, (key, item) in enumerate(zip(keys, items)):
        probe = FlowRun(run_id="flow_000000000000", graph={}, state={"item": item, "index": index, "key": key})
        owned: list[str] = []
        for item_node in child_graph.get("nodes") or []:
            if item_node.get("kind") == "agent" and item_node.get("role", "code") == "code":
                files = render_value(item_node.get("files"), probe)
                if isinstance(files, str):
                    files = [line.strip() for line in files.splitlines() if line.strip()]
                owned.extend(str(path) for path in files or [])
        scopes[key] = owned
    return scopes


def _child_graph_shape(node: dict) -> Any:
    if "graph" in node:
        return node["graph"]
    template = node.get("node")
    if not isinstance(template, dict):
        return template
    single = {**template, "id": template.get("id") or "item"}
    return {"entry": single["id"], "nodes": [single], "edges": []}


_WORK_KINDS = ("agent", "judge", "parallel", "map", "shell")


def _failed_work(run: FlowRun) -> list[str]:
    """Work nodes whose latest result failed. An earlier visit's FAIL that a
    later visit superseded does not count; a final FAIL or PARTIAL verdict does."""
    kinds = {node["id"]: node.get("kind") for node in run.graph.get("nodes") or []}
    return [f"{node_id}: {(result.get('error') or result.get('reason') or result.get('verdict') or 'failed')[:160]}"
            for node_id, result in run.results.items()
            if kinds.get(node_id) in _WORK_KINDS
            and (not result.get("ok") or result.get("verdict") in ("FAIL", "PARTIAL"))]


def _item_result(child: FlowRun) -> dict:
    """An item passes only when its flow is done and every work node's latest result passed.

    A node can fail (a write-scope gate, a crashed task) while an unconditional
    edge carries the flow on to its end; that must not count as a pass.
    """
    done = child.status == "done"
    failed = _failed_work(child)
    ok = done and not failed
    verdict = "PASS" if ok else "FAIL"
    reason = "" if ok else ("; ".join(failed) if done else child.reason)
    files = sorted({path for result in child.results.values() for path in result.get("files") or []})
    last = child.outputs.get(child.steps[-1]["node"], "") if child.steps else ""
    return {"run_id": child.run_id, "status": child.status, "ok": ok, "verdict": verdict,
            "reason": reason[:300], "files": files, "output": last[-400:]}


def _map_outcome(node: dict, items: dict[str, dict], usage: dict) -> NodeOutcome:
    total = len(items)
    passed = [key for key, item in items.items() if item.get("ok") and item.get("verdict") != "FAIL"]
    policy = node.get("pass", "all")
    met = len(passed) == total if policy == "all" else (total == 0 or len(passed) / total >= float(policy))
    failing = [key for key in items if key not in passed]
    lines = [f"map {node['id']}: {len(passed)}/{total} items passed"]
    for key in failing[:20]:
        item = items[key]
        lines.append(f"- {key}: {item.get('status')} {item.get('reason') or item.get('verdict') or ''}".rstrip())
    if len(failing) > 20:
        lines.append(f"(+{len(failing) - 20} more failing items)")
    verdicts = [item.get("verdict") for item in items.values()]
    # The wake summary carries this reason, so name the failing items there.
    named = "; ".join(f"{key} ({(items[key].get('reason') or items[key].get('status') or '')[:120]})"
                      for key in failing[:5])
    shortfall = f"{len(failing)} of {total} items did not pass" + (f": {named}" if named else "")
    return NodeOutcome(
        ok=met,
        output="\n".join(lines),
        verdict="PASS" if met else ("FAIL" if "FAIL" in verdicts else "PARTIAL"),
        reason="" if met else shortfall,
        error=None if met else shortfall,
        files=sorted({path for item in items.values() for path in item.get("files") or []}),
        usage=usage, items=items,
    )


def task_outcome(store: Any, job_id: str, role: str) -> NodeOutcome:
    """One task's latest attempt: ok from its status, verdict and report from its receipts."""
    from puppetmaster.usage import select_usage_records

    task = next((task for task in store.list_tasks(job_id) if task.role == role), None)
    if task is None:
        return NodeOutcome(ok=False, error=f"no task for {role}")
    artifacts = [artifact for artifact in store.list_artifacts(job_id) if artifact.task_id == task.id]
    receipts = [artifact for artifact in artifacts if artifact.type == ArtifactType.VERIFICATION
                and (artifact.payload or {}).get("kind") != "worker_verdict"
                and (artifact.payload or {}).get("adapter")]
    def rank(artifact: Any) -> tuple:
        result = str((artifact.payload or {}).get("result") or "")
        return (str(artifact.created_at or ""), result in ("passed", "recorded", "ok"))

    # Timestamps have one-second resolution: a fallback that succeeded in the
    # same second as a failed first attempt must still be the one that counts.
    latest = max(receipts, key=rank) if receipts else None
    # Only the latest attempt speaks for the task; an earlier fallback attempt's
    # report or verdict would otherwise leak into the outcome.
    cutoff = str(latest.created_at or "") if latest is not None else ""
    current = [artifact for artifact in artifacts if str(artifact.created_at or "") >= cutoff]
    verdict, reason = None, ""
    for artifact in sorted(current, key=lambda item: str(item.created_at or "")):
        payload = artifact.payload or {}
        if artifact.type == ArtifactType.VERIFICATION and payload.get("kind") == "worker_verdict":
            verdict, reason = payload.get("verdict") or verdict, str(payload.get("reason") or reason)
    last_message = ""
    if latest is not None:
        payload = latest.payload or {}
        last_message = str(payload.get("last_message") or payload.get("result_text") or "")
        if not last_message and payload.get("adapter") in ("shell", "local"):
            last_message = str(payload.get("stdout") or "")
    if verdict is None:
        parsed = parse_terminal_verdict(last_message)
        if parsed:
            verdict, reason = parsed.verdict, parsed.reason
    reports = [str((artifact.payload or {}).get("report") or (artifact.payload or {}).get("claim") or "")
               for artifact in current if artifact.type == ArtifactType.FINDING]
    files = sorted({str(path) for artifact in current if artifact.type == ArtifactType.PATCH
                    for path in (artifact.payload or {}).get("files") or []})
    record = select_usage_records(artifacts).get(task.id) or {}
    usage: dict[str, Any] = {key: record[key] for key in ("tokens_in", "tokens_out", "tokens_cached")
                             if record.get(key)}
    if isinstance(record.get("real_cost_usd"), (int, float)):
        usage["cost_usd"] = float(record["real_cost_usd"])
    ok = task.status == TaskStatus.COMPLETE
    error = None
    if not ok:
        cut = (task.payload or {}).get("failure_cut")
        detail = (latest.payload or {}).get("failure") if latest is not None else None
        error = "cut off by the user" if cut else f"task {task.status}" + (f": {detail}" if detail else "")
    output = "\n".join(text for text in reports if text) or last_message
    return NodeOutcome(ok=ok, output=output[-_OUTPUT_CHARS:], verdict=verdict, reason=reason[:1000],
                       error=error, files=files, usage=usage, task_id=task.id)
