"""Attempt-local Codex usage from the session rollout file.

``turn.completed.usage`` on ``codex exec resume`` is session-cumulative, and
it omits the request made just before a compaction. The rollout keeps one
``token_usage_record`` per original request (``response_id``) tagged with its
``turn_id``; summing the unique records of the one turn this attempt started
is exact. ``turn_token_usage`` / ``thread_token_usage`` are running snapshots:
they only cross-check the sum, never add to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def rollout_path(home: Optional[Path], session_id: Optional[str]) -> Optional[Path]:
    if home is None or not session_id:
        return None
    root = Path(home) / "sessions"
    if not root.is_dir():
        return None
    return next(root.glob(f"*/*/*/rollout-*-{session_id}.jsonl"), None)


def _records(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("payload"), dict):
            yield record


def turn_ids(path: Optional[Path]) -> Optional[frozenset]:
    """Turns started in the rollout so far; None when there is no rollout."""
    if path is None or not path.is_file():
        return None
    return frozenset(
        str(r["payload"]["turn_id"])
        for r in _records(path)
        if r.get("type") == "event_msg"
        and r["payload"].get("type") == "task_started"
        and r["payload"].get("turn_id")
    )


def count(value) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _sum(rows: list) -> dict:
    # An unknown counter on any request leaves the total unknown, never zero.
    total = {}
    for field in FIELDS:
        values = [count(row.get(field)) for row in rows]
        total[field] = None if not values or None in values else sum(values)
    return total


def attempt_usage(path: Optional[Path], baseline: Optional[frozenset]) -> dict:
    """Usage of the single turn this attempt added to the rollout.

    ``baseline`` is the set of turn ids before launch (empty for a new
    session). Anything but exactly one new turn is unlinked: the counters are
    NULL with a reason, never a session total.
    """
    now = turn_ids(path)
    if now is None:
        return unlinked("rollout_missing")
    new = sorted(now - (baseline or frozenset()))
    if len(new) != 1:
        return unlinked("rollout_no_new_turn" if not new else "rollout_ambiguous_turns")
    turn = new[0]
    requests = {}
    snapshot = None
    for record in _records(path):
        payload = record["payload"]
        if record.get("type") != "token_usage_record" or str(payload.get("turn_id")) != turn:
            continue
        response_id = payload.get("response_id")
        usage = payload.get("usage")
        if not response_id or not isinstance(usage, dict):
            continue
        # A replayed record is the same request, not a second one.
        requests.setdefault(str(response_id), usage)
        if isinstance(payload.get("turn_token_usage"), dict):
            snapshot = payload["turn_token_usage"]
    if not requests:
        return unlinked("rollout_turn_without_requests", turn=turn)
    total = _sum(list(requests.values()))
    partial = sorted(f for f, v in total.items() if v is None)
    conflicts = []
    if snapshot is not None:
        conflicts = sorted(
            f for f in FIELDS
            if total[f] is not None and count(snapshot.get(f)) is not None
            and count(snapshot.get(f)) != total[f]
        )
    if (total["cached_input_tokens"] is not None and total["input_tokens"] is not None
            and total["cached_input_tokens"] > total["input_tokens"]):
        conflicts.append("cached_exceeds_input")
    return {
        "usage_scope": "attempt",
        "usage_provenance": "rollout_token_usage_records",
        "rollout_turn_id": turn,
        "rollout_request_count": len(requests),
        "usage": total,
        "usage_partial_fields": partial,
        "usage_conflicts": conflicts,
    }


def unlinked(reason: str, turn: Optional[str] = None) -> dict:
    return {
        "usage_scope": "unknown",
        "usage_provenance": None,
        "usage_unlinked_reason": reason,
        "rollout_turn_id": turn,
        "rollout_request_count": 0,
        "usage": {field: None for field in FIELDS},
        "usage_partial_fields": list(FIELDS),
        "usage_conflicts": [],
    }
