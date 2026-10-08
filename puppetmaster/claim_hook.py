"""Claim the files that a pilot writes while one of its flows walks.

A host hook runs this module after each file edit by the pilot:

* Claude Code ``PostToolUse`` on ``Write|Edit|MultiEdit|NotebookEdit``;
* Codex ``PostToolUse`` on ``apply_patch``;
* Cursor ``afterFileEdit``.

When a flow walks over the checkout that holds the file, the hook adds the
path to the pilot claims of that flow (``flow claim``). The write_scope gate
then does not charge the path to a worker whose own events never named it.

The hook does nothing in a Puppetmaster worker, because the gate must see the
writes of a worker. It does nothing when no flow walks. It never fails the
host: an error ends as a no-op, and the hook writes nothing to stdout.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from puppetmaster.worker_fence import is_worker_process

# The file headers of an apply_patch envelope (Codex).
_PATCH_PATH = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.MULTILINE)
_PATH_KEYS = ("file_path", "notebook_path", "path")


def edited_paths(payload: Mapping[str, Any]) -> list[Path]:
    """The absolute paths that one host edit event wrote."""
    base = Path(str(payload.get("cwd") or os.getcwd()))
    tool_input = payload.get("tool_input")
    names: list[str] = []
    if isinstance(tool_input, Mapping):
        names += [tool_input[key] for key in _PATH_KEYS if isinstance(tool_input.get(key), str)]
        patch = tool_input.get("command") or tool_input.get("input") or tool_input.get("patch")
        if isinstance(patch, str):
            names += _PATCH_PATH.findall(patch)
    if isinstance(payload.get("file_path"), str):
        names.append(payload["file_path"])
    paths = []
    for name in names:
        path = Path(name).expanduser()
        paths.append(path if path.is_absolute() else base / path)
    return paths


def claim_edits(payload: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Claim the edited paths in each walking flow whose checkout holds them.

    Returns the claimed paths, relative to the checkout root.
    """
    if is_worker_process(env):
        return []
    from puppetmaster.state import walking_runs_dir

    try:
        markers = sorted(walking_runs_dir().glob("flow_*.json"))
    except OSError:
        return []
    if not markers:
        return []
    paths = [path.resolve() for path in edited_paths(payload)]
    if not paths:
        return []
    from puppetmaster import flow

    claimed: list[str] = []
    for marker in markers:
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            state_dir, root = Path(info["state_dir"]), _checkout_root(Path(info["cwd"]).resolve())
        except (OSError, ValueError, KeyError, TypeError):
            continue
        relative = [path.relative_to(root).as_posix() for path in paths if _inside(path, root)]
        if not relative:
            continue
        try:
            run = flow.load_run(state_dir, marker.stem)
        except (flow.FlowError, OSError, ValueError):
            run = None
        if run is None or run.status != "running":
            # The walk ended, or its walker crashed and the run is now interrupted.
            _unlink(marker)
            continue
        try:
            flow.claim_paths(state_dir, run.run_id, relative)
        except (flow.FlowError, OSError, TimeoutError):
            continue
        claimed += relative
    return claimed


def _checkout_root(cwd: Path) -> Path:
    """The git top level that holds ``cwd``; the gate reads paths relative to it."""
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return cwd


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root and ".git" not in path.relative_to(root).parts


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def run(stdin=None, env: Optional[Mapping[str, str]] = None) -> int:
    """Read one host hook payload from stdin and claim its paths. Always exits 0."""
    try:
        payload = json.loads((stdin or sys.stdin).read() or "{}")
        if isinstance(payload, Mapping):
            claim_edits(payload, env)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(run())
