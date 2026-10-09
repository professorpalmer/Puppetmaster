"""Attribute a changed path to the worker that plausibly wrote it.

A full-edit adapter measures its run by diffing the tree it started from
against the tree it left. In a shared checkout that delta also contains
whatever a concurrent writer did in the same window — a user saving a file, a
pilot, another session, a host process appending to a log — and the
``write_scope`` gate then fails a worker for edits it never made.

The worker's own event stream is the missing evidence. Codex ``--json`` emits
``file_change`` items naming the paths it edited and ``command_execution``
items carrying the command text; Claude Code's stream-json output carries
``tool_use`` blocks with the same information. An adapter distills that into
the paths its run referenced and records them in its verification / PATCH
payload, so :mod:`puppetmaster.gates` judges against a path list instead of
parsing adapter logs.

Attribution is deliberately asymmetric: a path the events reference is the
worker's, and so is any new out-of-scope file, because "I cannot see who wrote
it" is not proof that somebody else did. Only a path that was already dirty or
untracked when the worker started, and that the worker's events never name, is
reported as a concurrent change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

REFERENCED_PATHS_KEY = "worker_referenced_paths"
BASELINE_DIRTY_KEY = "baseline_dirty_paths"
TRUNCATED_KEY = "worker_references_truncated"

# A worker that edits or shells out thousands of times still has to fit under
# the artifact byte cap. Past the cap the evidence has a hole in it, so the
# gate stops excusing anything.
_MAX_REFERENCED_PATHS = 2000

# Shell punctuation around an argument: `>>log.txt`, `'cat`, `foo.py;`.
_TOKEN_EDGES = "\"'`()[]{}<>;|&,:=$*"


def path_tokens(command: object) -> list[str]:
    """The path-like arguments in a command line.

    A token counts when it carries a ``/`` or a ``.`` — enough to name a file.
    A bare word like ``tests`` does not: it is a directory argument far more
    often than a write target, and treating it as one blames a worker that
    merely ran the suite.
    """
    text = _command_text(command)
    if not text:
        return []
    found: list[str] = []
    for raw in re.split(r"\s+", text):
        token = raw.strip(_TOKEN_EDGES)
        if not token or token.startswith("-"):
            continue
        if "/" not in token and "." not in token:
            continue
        found.append(token)
    return found


class WorkerReferences:
    """The paths a worker's own event stream names."""

    __slots__ = ("paths", "truncated")

    def __init__(
        self, paths: Optional[list[str]] = None, truncated: bool = False
    ) -> None:
        self.paths = list(paths or [])
        self.truncated = truncated


def normalize_path(path: object, cwd: Optional[Path] = None) -> str:
    """A worker-reported path as a repo-relative POSIX path, best effort."""
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    if cwd is not None:
        candidate = Path(text)
        if candidate.is_absolute():
            try:
                text = candidate.resolve().relative_to(Path(cwd).resolve()).as_posix()
            except (ValueError, OSError):
                text = candidate.as_posix()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def codex_event_references(
    events: list[dict[str, Any]], cwd: Optional[Path] = None
) -> Optional[WorkerReferences]:
    """Paths a Codex ``--json`` event stream says this run touched.

    Returns ``None`` when the stream carried no structured items at all — an
    adapter with nothing to consult keeps the gate's existing behavior.
    """
    structured = False
    paths: list[str] = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type") or "")
        if kind.startswith("item.") or kind == "turn.completed":
            structured = True
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "file_change":
            paths.extend(_file_change_paths(item, cwd))
        elif item_type == "command_execution":
            paths.extend(
                normalize_path(token, cwd) for token in path_tokens(item.get("command"))
            )
    if not structured:
        return None
    return path_references(paths)


def claude_tool_use_references(
    stdout: object, cwd: Optional[Path] = None
) -> Optional[WorkerReferences]:
    """Paths Claude Code's ``tool_use`` blocks say this run touched.

    Returns ``None`` unless the output actually carried tool-use blocks: the
    default ``--output-format json`` reports only the final result, which is no
    evidence about who wrote what.
    """
    paths: list[str] = []
    found = False
    for block in _claude_tool_use_blocks(stdout):
        found = True
        payload = block.get("input")
        if not isinstance(payload, dict):
            continue
        for key in ("file_path", "path", "notebook_path", "filePath"):
            value = payload.get(key)
            if isinstance(value, str):
                paths.append(normalize_path(value, cwd))
        paths.extend(
            normalize_path(token, cwd) for token in path_tokens(payload.get("command"))
        )
    if not found:
        return None
    return path_references(paths)


def attribution_payload(
    references: Optional[WorkerReferences], before: Any
) -> dict[str, Any]:
    """Payload fields the ``write_scope`` gate needs, or ``{}`` with no stream."""
    if references is None:
        return {}
    get = getattr(before, "get", None)
    dirty: list[object] = []
    if callable(get):
        dirty = list(get("changed_files") or []) + list(get("untracked_files") or [])
    payload: dict[str, Any] = {
        REFERENCED_PATHS_KEY: references.paths,
        BASELINE_DIRTY_KEY: sorted({normalize_path(path) for path in dirty} - {""}),
    }
    if references.truncated:
        payload[TRUNCATED_KEY] = True
    return payload


def path_is_referenced(path: str, references: WorkerReferences) -> bool:
    """True when the worker's own events name ``path``."""
    target = normalize_path(path)
    if not target:
        return False
    return any(_same_path(target, known) for known in references.paths)


def path_was_dirty_before(path: str, baseline: list[str]) -> bool:
    """True when ``path`` was already modified or untracked at worker start.

    ``git status --short`` reports an untracked directory as a single entry, so
    a baseline entry also covers everything beneath it.
    """
    target = normalize_path(path)
    for raw in baseline:
        known = normalize_path(raw)
        if not known:
            continue
        if _same_path(target, known) or target.startswith(known + "/"):
            return True
    return False


def path_references(paths: list[str]) -> WorkerReferences:
    unique = sorted({path for path in paths if path})
    return WorkerReferences(
        paths=unique[:_MAX_REFERENCED_PATHS],
        truncated=len(unique) > _MAX_REFERENCED_PATHS,
    )


def _file_change_paths(item: dict[str, Any], cwd: Optional[Path]) -> list[str]:
    changes = item.get("changes")
    if isinstance(changes, dict):
        candidates: list[object] = list(changes.keys())
    elif isinstance(changes, list):
        candidates = [
            change.get("path") if isinstance(change, dict) else change
            for change in changes
        ]
    else:
        candidates = []
    for key in ("path", "file_path"):
        if isinstance(item.get(key), str):
            candidates.append(item[key])
    return [normalize_path(candidate, cwd) for candidate in candidates]


def _command_text(command: object) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return ""


def _claude_tool_use_blocks(stdout: object) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in _json_objects(stdout):
        _collect_tool_use(event, blocks)
    return blocks


def _json_objects(stdout: object) -> list[Any]:
    text = "" if stdout is None else str(stdout)
    try:
        return [json.loads(text)]
    except (json.JSONDecodeError, TypeError):
        pass
    parsed: list[Any] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{") and not line.startswith("["):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


def _collect_tool_use(node: Any, blocks: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "tool_use":
            blocks.append(node)
        for value in node.values():
            _collect_tool_use(value, blocks)
    elif isinstance(node, list):
        for value in node:
            _collect_tool_use(value, blocks)


def _same_path(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    return left.endswith("/" + right) or right.endswith("/" + left)
