"""Change observation for a worker directory that is not a Git work tree.

Git attributes a run's delta from the tree it started on. Without Git, a
manifest of every file's size and mtime, taken before launch and compared
after the worker stops, gives the same attributed delta: added, modified and
deleted paths (a rename is a delete plus an add).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from puppetmaster.state import state_dir_from_env

# Caches and tool state that no task owns. The gate also skips bytecode.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".puppetmaster", ".codegraph",
})
MAX_FILES = 50_000

Manifest = dict  # relative posix path -> [size, mtime_ns]


def _excluded_roots(root: Path) -> list[Path]:
    """The live Puppetmaster state dir changes during every run; it is not the worker's."""
    raw = state_dir_from_env()
    if not raw:
        return []
    try:
        state = Path(raw).resolve()
        state.relative_to(root)
    except (OSError, ValueError):
        return []
    return [state]


def directory_manifest(root: Path) -> Optional[Manifest]:
    """Size and mtime of each file under ``root``. None when it cannot bound the walk."""
    try:
        root = Path(root).resolve()
    except OSError:
        return None
    if not root.is_dir():
        return None
    excluded = _excluded_roots(root)
    manifest: Manifest = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        here = Path(current)
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIP_DIRS and not any(here / d == ex for ex in excluded)
        )
        for name in files:
            path = here / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            manifest[path.relative_to(root).as_posix()] = [stat.st_size, stat.st_mtime_ns]
            if len(manifest) > MAX_FILES:
                return None
    return manifest


def manifest_delta(before: Manifest, after: Manifest) -> dict:
    """Paths the run added, modified and deleted, each sorted."""
    added = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    modified = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    return {"added": added, "modified": modified, "deleted": deleted}


def delta_status_text(delta: dict) -> str:
    """Name-status lines in the shape of `git diff --name-status`."""
    lines = [f"A\t{p}" for p in delta["added"]]
    lines += [f"M\t{p}" for p in delta["modified"]]
    lines += [f"D\t{p}" for p in delta["deleted"]]
    return "\n".join(sorted(lines, key=lambda line: line[2:]))
