"""Append-only event log of a flow run, and a read-only tap on it.

Each run directory holds ``events.jsonl`` next to ``run.json``. The walker
writes one line for each fact about the run: the run started, a step
finished, the run waits at a gate, the run reached a terminal status. The
log is a projection of ``run.json``: each event has a deterministic ``key``
and ``sync`` appends only the keys that are not in the file yet, so a
restarted walker fills a gap and never writes an event two times.

Lineage: each event names its ``parent`` event. A step's parent is the step
before it, and the step records the edge that led to it (``via``: from node,
``when`` and edge index), so a repair step points back at the failed check
that sent the run to it. A map step lists its item runs (``children``); each
item run's ``run.started`` event names the parent run, node and key.

The log is for people and tools outside the walk. No worker or judge prompt
reads it, and a reader never opens the job store: ``read`` and ``follow``
read this file and ``run.json`` only. A failed append is logged and dropped;
it never fails a walk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from puppetmaster.models import now_iso

logger = logging.getLogger(__name__)

EVENTS_VERSION = 1
EVENTS_FILE = "events.jsonl"
SCHEMA_FILE = Path(__file__).with_name("schemas") / "flow-event.v1.json"

_REASON_CHARS = 500
_FILE_MODE = 0o600

# path -> (size, {key: id}, last seq). Valid while the file size matches.
_CACHE: dict[str, tuple[int, dict[str, str], int]] = {}


def events_path(state_dir: Path, run_id: str) -> Path:
    from puppetmaster.flow import run_path

    return run_path(state_dir, run_id).with_name(EVENTS_FILE)


def question_digest(question: Any) -> str:
    return "sha256:" + hashlib.sha256(str(question or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Write


def sync(state_dir: Path, run: Any) -> None:
    """Append every event of ``run`` that the log does not hold yet. Never raises."""
    try:
        _sync(state_dir, run)
    except Exception as exc:  # the tap is best effort; the walk is not
        logger.warning("flow events: sync of %s failed: %s", getattr(run, "run_id", "?"), exc)


def _sync(state_dir: Path, run: Any) -> None:
    from puppetmaster.flow import TERMINAL_STATUSES
    from puppetmaster.interprocess_lock import InterProcessFileLock

    path = events_path(state_dir, run.run_id)
    with InterProcessFileLock.for_target(path, timeout=5):
        ids, seq = _load(path)
        pending: list[dict] = []

        def add(kind: str, key: str, parent: Optional[str], fields: dict) -> None:
            nonlocal seq
            if key in ids:
                return
            seq += 1
            event = {"v": EVENTS_VERSION, "id": f"{run.run_id}.{seq}", "seq": seq,
                     "run_id": run.run_id, "type": kind, "key": key, "at": now_iso(),
                     "parent": parent, **fields}
            ids[key] = event["id"]
            pending.append(event)

        add("run.started", "run", None, _started_fields(run))
        for index, step in enumerate(run.steps):
            parent = ids.get(f"step:{index - 1}") if index else ids["run"]
            add("step", f"step:{index}", parent, _step_fields(index, step))
        last = ids.get(f"step:{len(run.steps) - 1}") if run.steps else ids["run"]
        if run.status == "waiting" and run.gate:
            add("gate.waiting", f"gate:{run.gate.get('node')}:{len(run.steps)}", last, {
                "node": run.gate.get("node"),
                "question_digest": question_digest(run.gate.get("question")),
                "options": [str(option) for option in run.gate.get("options") or []],
            })
        if run.status in TERMINAL_STATUSES:
            add("run.finished", f"end:{len(run.steps)}:{run.status}", last, {
                "status": run.status, "reason": str(run.reason or "")[:_REASON_CHARS],
                "steps": len(run.steps), "usage": _numbers(run.usage),
            })
        if pending:
            _append(path, pending, ids, seq)


def _started_fields(run: Any) -> dict:
    parent = run.parent or None
    return {
        "graph_id": run.graph.get("id"),
        "entry": run.graph.get("entry"),
        "parent_run": ({key: parent.get(key) for key in ("run_id", "node", "key")}
                       if parent else None),
        "continued_from": run.continued_from,
    }


def _step_fields(index: int, step: dict) -> dict:
    via = step.get("via")
    approval = step.get("approval")
    return {
        "index": index,
        "node": step.get("node"),
        "kind": step.get("kind"),
        "visit": step.get("visit"),
        "ok": bool(step.get("ok")),
        "verdict": step.get("verdict"),
        "reason": str(step.get("reason") or "")[:_REASON_CHARS],
        "error": step.get("error"),
        "job_ids": [str(job) for job in step.get("job_ids") or []],
        "usage": _numbers(step.get("usage")),
        "via": dict(via) if isinstance(via, dict) else None,
        "items": dict(step["items"]) if isinstance(step.get("items"), dict) else None,
        "children": dict(step["children"]) if isinstance(step.get("children"), dict) else None,
        "approval": dict(approval) if isinstance(approval, dict) else None,
        "started_at": step.get("started_at"),
        "finished_at": step.get("finished_at"),
    }


def _numbers(usage: Any) -> dict:
    return {str(key): value for key, value in (usage or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def _load(path: Path) -> tuple[dict[str, str], int]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        _CACHE.pop(str(path), None)
        return {}, 0
    cached = _CACHE.get(str(path))
    if cached is not None and cached[0] == size:
        return dict(cached[1]), cached[2]
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        # A writer died in the middle of a line: drop the torn tail so the
        # next append starts on its own line.
        data = data[:data.rfind(b"\n") + 1]
        with open(path, "r+b") as handle:
            handle.truncate(len(data))
    ids: dict[str, str] = {}
    seq = 0
    for line in data.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("key"), str):
            ids[event["key"]] = str(event.get("id"))
            seq = max(seq, int(event.get("seq") or 0))
    _CACHE[str(path)] = (len(data), dict(ids), seq)
    return ids, seq


def _append(path: Path, events: list[dict], ids: dict[str, str], seq: int) -> None:
    text = "".join(json.dumps(event, separators=(",", ":"), default=str) + "\n" for event in events)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, _FILE_MODE)
    try:
        os.write(fd, text.encode("utf-8"))
        size = os.fstat(fd).st_size
    finally:
        os.close(fd)
    _CACHE[str(path)] = (size, ids, seq)


# --------------------------------------------------------------------------
# Read (the tap)


def read(state_dir: Path, run_id: str, *, since: int = 0) -> list[dict]:
    """Events of ``run_id`` with ``seq`` above ``since``. Reads the log file only."""
    return list(_parse(events_path(state_dir, run_id), since))


def _parse(path: Path, since: int) -> Iterator[dict]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn last line; the next sync drops it
        if isinstance(event, dict) and int(event.get("seq") or 0) > since:
            yield event


def follow(state_dir: Path, run_id: str, *, since: int = 0, timeout: float = 0.0,
           interval: float = 0.25) -> Iterator[dict]:
    """Yield events as the walker writes them.

    Ends when ``run.json`` shows a terminal status and the log holds the
    finish event for that many steps, or after ``timeout`` seconds (0 = no
    limit). A run that waits at a gate keeps the tap open. The finish event
    of an earlier attempt does not end a restarted run. Reads
    ``events.jsonl`` and ``run.json`` only.
    """
    from puppetmaster.flow import TERMINAL_STATUSES, load_run

    deadline = time.monotonic() + timeout if timeout > 0 else None
    seen = since
    grace: Optional[float] = None
    while True:
        run = load_run(state_dir, run_id)
        finished = False
        for event in _parse(events_path(state_dir, run_id), seen):
            seen = max(seen, int(event.get("seq") or 0))
            yield event
            if event.get("key") == f"end:{len(run.steps)}:{run.status}":
                finished = True
        if finished:
            return
        now = time.monotonic()
        if run.status in TERMINAL_STATUSES:
            # The walker saves run.json before it appends: give the finish
            # event a moment, then stop (another process wrote the stop).
            grace = grace if grace is not None else now + 2.0
            if now >= grace:
                return
        if deadline is not None and now >= deadline:
            return
        time.sleep(interval)


# --------------------------------------------------------------------------
# Schema


def schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def check(event: Any, spec: Optional[dict] = None) -> list[str]:
    """Problems of one event against the published schema (empty: valid)."""
    spec = spec or schema()
    return _check(event, spec, spec, "$")


_TYPES = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _check(value: Any, node: dict, root: dict, where: str) -> list[str]:
    """The JSON Schema subset the event schema uses: $ref, oneOf, const, enum,
    type, required, properties, additionalProperties, items, minimum."""
    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        return _check(value, root["$defs"][name], root, where)
    if "oneOf" in node:
        matches = [option for option in node["oneOf"] if not _check(value, option, root, where)]
        if len(matches) == 1:
            return []
        if not matches:
            kind = value.get("type") if isinstance(value, dict) else None
            for option in node["oneOf"]:
                target = root["$defs"].get(str(option.get("$ref", "")).rsplit("/", 1)[-1], option)
                if (target.get("properties") or {}).get("type", {}).get("const") == kind:
                    return _check(value, option, root, where)
            return [f"{where}: matches no event shape"]
        return [f"{where}: matches {len(matches)} event shapes"]
    problems: list[str] = []
    types = node.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_TYPES[name](value) for name in names):
            return [f"{where}: expected {' or '.join(names)}"]
    if "const" in node and value != node["const"]:
        problems.append(f"{where}: expected {node['const']!r}")
    if "enum" in node and value not in node["enum"]:
        problems.append(f"{where}: {value!r} is not one of {node['enum']}")
    if "minimum" in node and _TYPES["number"](value) and value < node["minimum"]:
        problems.append(f"{where}: below {node['minimum']}")
    if isinstance(value, dict):
        props = node.get("properties") or {}
        for key in node.get("required") or []:
            if key not in value:
                problems.append(f"{where}: missing {key!r}")
        for key, item in value.items():
            if key in props:
                problems.extend(_check(item, props[key], root, f"{where}.{key}"))
            elif node.get("additionalProperties") is False:
                problems.append(f"{where}: unknown field {key!r}")
            elif isinstance(node.get("additionalProperties"), dict):
                problems.extend(_check(item, node["additionalProperties"], root, f"{where}.{key}"))
    if isinstance(value, list) and isinstance(node.get("items"), dict):
        for index, item in enumerate(value):
            problems.extend(_check(item, node["items"], root, f"{where}[{index}]"))
    return problems


def check_lines(lines: list[str]) -> list[dict]:
    """Check JSONL text: one report row for each line with a problem."""
    spec = schema()
    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            rows.append({"line": number, "problems": [f"not JSON: {exc}"]})
            continue
        problems = check(event, spec)
        if problems:
            rows.append({"line": number, "id": event.get("id") if isinstance(event, dict) else None,
                         "problems": problems})
    return rows
