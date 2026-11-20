"""Compare runs of one flow: path shape, revisits, cost and wall time.

A run's shape is the sequence of nodes it visited. Two runs of the same graph
on the same request take the same path when nothing went wrong; a repair
loop, a reroute or a skipped review shows up as an edit in that sequence.
The score is ``1 - edit_distance / longer_length`` (1.0: the same path).

Reads the run records only (``run.json``). It never opens the job store
and is never part of a walk.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Optional


def compare(runs: list[Any]) -> dict:
    """The first run is the baseline; every other run is compared with it."""
    profiles = [profile(run) for run in runs]
    base = profiles[0]
    return {
        "baseline": base,
        "runs": [{**other, "vs_baseline": _delta(base, other)} for other in profiles[1:]],
    }


def profile(run: Any) -> dict:
    path = [str(step.get("node")) for step in run.steps]
    visits: dict[str, int] = {}
    for node in path:
        visits[node] = visits.get(node, 0) + 1
    items = {"total": 0, "not_done": 0, "reused": 0}
    for step in run.steps:
        # Map step counts are per item status, plus "reused".
        for status, count in (step.get("items") or {}).items():
            if status == "reused":
                items["reused"] += int(count)
                continue
            items["total"] += int(count)
            if status != "done":
                items["not_done"] += int(count)
    return {
        "run_id": run.run_id,
        "graph_id": run.graph.get("id"),
        "graph_digest": graph_digest(run.graph),
        "status": run.status,
        "steps": len(path),
        "path": path,
        "revisits": sum(count - 1 for count in visits.values() if count > 1),
        "visits": visits,
        "items": items,
        "usage": {key: value for key, value in (run.usage or {}).items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)},
        "wall_seconds": wall_seconds(run),
    }


def graph_digest(graph: dict) -> str:
    """Digest of the graph's logic; the resolved ``cwd`` differs between checkouts."""
    body = {key: value for key, value in graph.items() if key != "cwd"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def wall_seconds(run: Any) -> Optional[float]:
    if not run.steps:
        return None
    start = _time(run.steps[0].get("started_at"))
    end = _time(run.steps[-1].get("finished_at"))
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 3)


def shape_score(a: list[str], b: list[str]) -> float:
    longer = max(len(a), len(b))
    if longer == 0:
        return 1.0
    return round(1.0 - edit_distance(a, b) / longer, 4)


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def _delta(base: dict, other: dict) -> dict:
    nodes = sorted(set(base["visits"]) | set(other["visits"]))
    visits = {node: other["visits"].get(node, 0) - base["visits"].get(node, 0) for node in nodes}
    usage = {}
    for key in sorted(set(base["usage"]) & set(other["usage"])):
        if base["usage"][key]:
            usage[key] = round(other["usage"][key] / base["usage"][key], 4)
    wall = None
    if base["wall_seconds"] and other["wall_seconds"] is not None:
        wall = round(other["wall_seconds"] / base["wall_seconds"], 4)
    return {
        "same_graph": base["graph_digest"] == other["graph_digest"],
        "shape_score": shape_score(base["path"], other["path"]),
        "revisits": other["revisits"] - base["revisits"],
        "visits": {node: change for node, change in visits.items() if change},
        "usage_ratio": usage,
        "wall_ratio": wall,
    }


def _time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
