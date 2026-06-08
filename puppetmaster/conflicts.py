"""Cross-worker write-conflict prediction (B3/C1).

When several tasks run in parallel against the same tree, the single biggest
manual cost is hand-merging the same hot files every wave and chasing the
cross-wave regressions that causes. If each task declares a ``write_scope``
(the globs it intends to touch), we can predict — *before* dispatching — which
tasks are aimed at overlapping territory, and warn (or serialize) instead of
discovering the collision after the fact.

The overlap test is a deliberately conservative path-prefix heuristic: two
globs overlap when the concrete directory prefix of one is an ancestor of (or
equal to) the other's. That over-reports a little (better a false warning than
a silent collision) and never needs to enumerate the filesystem.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Sequence

_WILDCARD_CHARS = set("*?[")


def _glob_prefix(glob: str) -> str:
    """The concrete directory prefix of ``glob`` — everything before the first
    path segment that contains a wildcard. ``src/api/**/*.py`` -> ``src/api``."""
    parts: list[str] = []
    for part in PurePosixPath(glob.strip()).parts:
        if any(char in part for char in _WILDCARD_CHARS):
            break
        parts.append(part)
    return "/".join(parts).rstrip("/")


def _prefixes_overlap(first: str, second: str) -> bool:
    """True when one prefix is an ancestor directory of (or equal to) the other.

    An empty prefix means "matches anywhere" (e.g. ``**/*.py``), which overlaps
    everything — the conservative, collision-avoiding default.
    """
    if first == second:
        return True
    if first == "" or second == "":
        return True
    return first.startswith(second + "/") or second.startswith(first + "/")


def scopes_overlap(first: Iterable[str], second: Iterable[str]) -> bool:
    """True when any glob in ``first`` could touch the same files as any in ``second``."""
    first_prefixes = [_glob_prefix(g) for g in first if str(g).strip()]
    second_prefixes = [_glob_prefix(g) for g in second if str(g).strip()]
    return any(
        _prefixes_overlap(a, b) for a in first_prefixes for b in second_prefixes
    )


def predict_write_conflicts(
    scoped_tasks: Sequence[tuple[str, Sequence[str]]],
) -> list[dict]:
    """Predict pairwise write conflicts among ``(task_id, write_scope)`` pairs.

    Returns one record per overlapping pair: ``{"tasks": [id_a, id_b],
    "scopes": [scope_a, scope_b]}``. Tasks without a declared scope are skipped
    (nothing to reason about). Order-independent; each pair reported once.
    """
    declared = [
        (task_id, [str(g) for g in (scope or []) if str(g).strip()])
        for task_id, scope in scoped_tasks
    ]
    declared = [(task_id, scope) for task_id, scope in declared if scope]

    conflicts: list[dict] = []
    for i in range(len(declared)):
        id_a, scope_a = declared[i]
        for j in range(i + 1, len(declared)):
            id_b, scope_b = declared[j]
            if scopes_overlap(scope_a, scope_b):
                conflicts.append(
                    {"tasks": sorted([id_a, id_b]), "scopes": [scope_a, scope_b]}
                )
    return conflicts
