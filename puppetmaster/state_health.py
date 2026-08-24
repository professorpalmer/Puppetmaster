"""Warn-only detection of a forked / wrong-project state directory.

``state.default_state_dir`` hashes ``_git_root(cwd) or cwd``. When a session
is opened one level *above* the real repository — say ``C:\\Projects\\foo``
holding the actual checkout at ``C:\\Projects\\foo\\Foo`` — the ``or``
fallback fires and hashes the wrapper folder instead. One logical project
then silently forks into two state dirs: a near-empty one the dashboard
shows, and the busy one every job actually wrote to.

This module *detects* that split and nothing else. It never changes which
directory anything resolves to (repointing a hash would orphan live job
history); it only reports a verdict a caller can turn into a warning.

Everything here is deliberately cheap: directory stats only, no SQLite, and
no ``git`` subprocess when the caller supplies ``state_dir``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from puppetmaster import state


# Verdict vocabulary. Closed set: callers may switch on these and nothing
# else ever comes out of ``diagnose_state_dir``.
VERDICT_OK = "ok"
VERDICT_NEW = "new"
VERDICT_NOT_PROJECT_SCOPED = "not-project-scoped"
VERDICT_WORKSPACE_IS_GIT_ROOT = "workspace-is-git-root"
VERDICT_NOT_CWD_DERIVED = "not-cwd-derived"
VERDICT_FORKED_PROJECT = "forked-project"
VERDICT_NAME_TWIN = "name-twin"
VERDICT_AMBIGUOUS = "ambiguous"
VERDICT_UNAVAILABLE = "unavailable"

VERDICTS = frozenset(
    {
        VERDICT_OK,
        VERDICT_NEW,
        VERDICT_NOT_PROJECT_SCOPED,
        VERDICT_WORKSPACE_IS_GIT_ROOT,
        VERDICT_NOT_CWD_DERIVED,
        VERDICT_FORKED_PROJECT,
        VERDICT_NAME_TWIN,
        VERDICT_AMBIGUOUS,
        VERDICT_UNAVAILABLE,
    }
)

# Verdicts that mean "the user is probably looking at the wrong dir".
SUSPECT_VERDICTS = frozenset(
    {VERDICT_FORKED_PROJECT, VERDICT_NAME_TWIN, VERDICT_AMBIGUOUS}
)

# Verdicts reached by an early exit: a deliberate choice or a legitimately
# separate project, never a fork.
GUARD_VERDICTS = frozenset(
    {
        VERDICT_NOT_PROJECT_SCOPED,
        VERDICT_WORKSPACE_IS_GIT_ROOT,
        VERDICT_NOT_CWD_DERIVED,
    }
)

# Child directories that are never a sibling checkout worth probing.
_SKIP_CHILDREN = frozenset(
    {"node_modules", "__pycache__", "venv", ".venv", "dist", "build", "target"}
)

# ``dashboard.py`` strips this same suffix to show a short project label; the
# name-twin check compares the surviving slug.
_DIGEST_SUFFIX_RE = re.compile(r"-[0-9a-f]{12}$")

# "Materially busier" gates a candidate into the report at all.
_MATERIAL_MULTIPLE = 2
_MATERIAL_MARGIN = 2
# "Far busier" is the stricter bar a same-name twin must clear before we
# contradict an active dir that already has real history in it.
_FAR_MULTIPLE = 5
_FAR_MARGIN = 5

_MAX_WARNING_CHARS = 140
_MAX_EVIDENCE_CANDIDATES = 3


@dataclass(frozen=True)
class StateDirSummary:
    """Cheap, SQLite-free activity snapshot of one project state dir."""

    path: Path
    job_count: int
    last_activity: Optional[float]
    truncated: bool = False


@dataclass(frozen=True)
class CandidateStateDir:
    """A sibling checkout whose state dir looks like the real project."""

    workspace: Path
    state_dir: Path
    job_count: int
    last_activity: Optional[float]
    name_twin: bool


@dataclass(frozen=True)
class StateDirDiagnosis:
    """Verdict plus the evidence that produced it. Never raised, never acted on."""

    verdict: str
    state_dir: Path
    active: StateDirSummary
    candidates: tuple[CandidateStateDir, ...]
    evidence: tuple[str, ...]

    @property
    def suspect(self) -> bool:
        return self.verdict in SUSPECT_VERDICTS


def summarize_project_state_dir(
    path: Union[Path, str], *, max_jobs: int = 4096
) -> StateDirSummary:
    """Summarize ``path`` by counting ``jobs/*`` directories and their mtimes.

    Deliberately does **not** open the store. The SQLite store keeps each
    job on disk under ``<state_dir>/jobs/<job_id>/``, so a directory listing
    is enough to tell how busy a project is — and it avoids spinning up the
    WAL writer just to answer a diagnostic question (same rationale as
    ``state.find_state_dir_for_job``).

    Never raises: an unreadable or missing directory yields zero jobs and no
    last-activity timestamp.
    """
    root = Path(path)
    jobs_dir = root / "jobs"
    count = 0
    latest: Optional[float] = None
    truncated = False
    try:
        with os.scandir(jobs_dir) as entries:
            for entry in entries:
                if count >= max_jobs:
                    truncated = True
                    break
                try:
                    if not entry.is_dir():
                        continue
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                count += 1
                if latest is None or mtime > latest:
                    latest = mtime
    except OSError:
        return StateDirSummary(path=root, job_count=0, last_activity=None)
    return StateDirSummary(
        path=root, job_count=count, last_activity=latest, truncated=truncated
    )


def diagnose_state_dir(
    cwd: Optional[Union[Path, str]] = None,
    state_dir: Optional[Union[Path, str]] = None,
    *,
    near_empty_max: int = 1,
    max_children: int = 512,
) -> StateDirDiagnosis:
    """Decide whether the active state dir is a fork of a nested repo's dir.

    Passing ``state_dir`` keeps the whole probe git-free: only filesystem
    stats are consulted. Omitting it resolves the default, which does shell
    out to ``git`` exactly as ``state.default_state_dir`` always has.
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        return _diagnose(base, state_dir, near_empty_max, max_children)
    except OSError as exc:
        active = Path(state_dir) if state_dir is not None else base
        return StateDirDiagnosis(
            verdict=VERDICT_UNAVAILABLE,
            state_dir=active,
            active=StateDirSummary(path=active, job_count=0, last_activity=None),
            candidates=(),
            evidence=(
                f"verdict:{VERDICT_UNAVAILABLE}",
                f"error:{type(exc).__name__}: {exc}",
            ),
        )


def short_warning(diagnosis: StateDirDiagnosis) -> Optional[str]:
    """One sentence, basenames only, for a status line. None when not suspect.

    Never emits an absolute path: the caller decides how much of the
    filesystem to reveal, and this string is safe to print anywhere.
    """
    if not diagnosis.suspect or not diagnosis.candidates:
        return None
    best = diagnosis.candidates[0]
    sentence = (
        f"{diagnosis.state_dir.name} has {diagnosis.active.job_count} job(s) "
        f"but nested repo {best.workspace.name} has {best.job_count} — "
        f"not a git repo here; see puppetmaster projects."
    )
    if len(sentence) > _MAX_WARNING_CHARS:
        sentence = sentence[: _MAX_WARNING_CHARS - 1].rstrip() + "\u2026"
    return sentence


def _diagnose(
    base: Path,
    state_dir: Optional[Union[Path, str]],
    near_empty_max: int,
    max_children: int,
) -> StateDirDiagnosis:
    pinned = state_dir is not None
    active_path = Path(state_dir) if pinned else state.default_state_dir(base)
    active = summarize_project_state_dir(active_path)

    # Guard 1 -- NOT-PROJECT-SCOPED. An explicit --state-dir or a
    # PUPPETMASTER_STATE_DIR env var is a deliberate choice; only the hashed
    # projects/ layout can fork.
    if not _same_path(active_path.parent, state.projects_root()):
        return StateDirDiagnosis(
            verdict=VERDICT_NOT_PROJECT_SCOPED,
            state_dir=active_path,
            active=active,
            candidates=(),
            evidence=(
                f"verdict:{VERDICT_NOT_PROJECT_SCOPED}",
                f"active:{active_path.name}",
                f"active_jobs:{active.job_count}",
                f"pinned:{str(pinned).lower()}",
            ),
        )

    # Guard 2 -- WORKSPACE-IS-GIT-ROOT. When cwd *is* the repo root its state
    # dir is correct by definition, and a nested repo (submodule, vendored
    # checkout, worktree) is legitimately its own project.
    if (base / ".git").exists():
        return StateDirDiagnosis(
            verdict=VERDICT_WORKSPACE_IS_GIT_ROOT,
            state_dir=active_path,
            active=active,
            candidates=(),
            evidence=(
                f"verdict:{VERDICT_WORKSPACE_IS_GIT_ROOT}",
                f"active:{active_path.name}",
                f"active_jobs:{active.job_count}",
                "cwd_is_git_root:true",
            ),
        )

    expected = state.project_state_dir_for(base)

    # Guard 3 -- NOT-CWD-DERIVED. Exact, git-free proof that the or-fallback
    # actually fired. Stays quiet when cwd is a subdirectory of a repo (the
    # dir came from the git root, not from cwd) and when a job-ownership
    # pivot picked the dir.
    if active_path.name != expected.name:
        return StateDirDiagnosis(
            verdict=VERDICT_NOT_CWD_DERIVED,
            state_dir=active_path,
            active=active,
            candidates=(),
            evidence=(
                f"verdict:{VERDICT_NOT_CWD_DERIVED}",
                f"active:{active_path.name}",
                f"active_jobs:{active.job_count}",
                f"expected_for_cwd:{expected.name}",
            ),
        )

    candidates, scanned, truncated = _scan_siblings(base, active, max_children)

    # Guard 4 / 6 -- a populated active dir is fine unless a same-slug twin
    # is *far* busier. near_empty_max defaults to 1 because the real incident
    # had exactly one stale job in the wrong dir; a "zero jobs" rule would
    # have missed it entirely.
    populated = active.job_count > near_empty_max
    twins = tuple(c for c in candidates if c.name_twin)
    if twins:
        verdict = VERDICT_NAME_TWIN
    elif populated:
        verdict = VERDICT_OK
    elif not candidates:
        verdict = VERDICT_NEW
    elif len(candidates) == 1:
        verdict = VERDICT_FORKED_PROJECT
    else:
        verdict = VERDICT_AMBIGUOUS

    evidence = [
        f"verdict:{verdict}",
        f"active:{active_path.name}",
        f"active_jobs:{active.job_count}",
        f"cwd:{base.name}",
        "cwd_is_git_root:false",
        f"expected_for_cwd:{expected.name}",
        f"scanned_children:{scanned}",
        f"candidates:{len(candidates)}",
    ]
    for candidate in candidates[:_MAX_EVIDENCE_CANDIDATES]:
        evidence.append(f"candidate:{candidate.state_dir.name}={candidate.job_count}")
    if truncated:
        evidence.append(f"truncated:{max_children}")
    if twins:
        evidence.append(f"name_twin:{twins[0].state_dir.name}")

    return StateDirDiagnosis(
        verdict=verdict,
        state_dir=active_path,
        active=active,
        candidates=candidates,
        evidence=tuple(evidence),
    )


def _scan_siblings(
    base: Path, active: StateDirSummary, max_children: int
) -> tuple[tuple[CandidateStateDir, ...], int, bool]:
    """Depth-1 scan of ``base`` for nested checkouts with busier state dirs."""
    found: list[CandidateStateDir] = []
    scanned = 0
    truncated = False
    active_slug = _project_slug(active.path.name).lower()
    try:
        with os.scandir(base) as entries:
            for entry in entries:
                if scanned >= max_children:
                    truncated = True
                    break
                scanned += 1
                name = entry.name
                if name.startswith(".") or name in _SKIP_CHILDREN:
                    continue
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                child = Path(entry.path)
                # A .git *file* means a worktree or submodule; both count.
                if not (child / ".git").exists():
                    continue
                candidate_dir = state.project_state_dir_for(child)
                summary = summarize_project_state_dir(candidate_dir)
                material = _busier(
                    summary.job_count,
                    active.job_count,
                    _MATERIAL_MULTIPLE,
                    _MATERIAL_MARGIN,
                )
                if not material:
                    continue
                far = _busier(
                    summary.job_count, active.job_count, _FAR_MULTIPLE, _FAR_MARGIN
                )
                twin = far and _project_slug(candidate_dir.name).lower() == active_slug
                found.append(
                    CandidateStateDir(
                        workspace=child,
                        state_dir=candidate_dir,
                        job_count=summary.job_count,
                        last_activity=summary.last_activity,
                        name_twin=twin,
                    )
                )
    except OSError:
        # A partially-read listing is still usable evidence; report what we got.
        pass
    found.sort(key=lambda c: (-c.job_count, c.state_dir.name))
    return tuple(found), scanned, truncated


def _busier(candidate_jobs: int, active_jobs: int, multiple: int, margin: int) -> bool:
    if candidate_jobs <= 0:
        return False
    return candidate_jobs >= max(active_jobs * multiple, active_jobs + margin)


def _project_slug(dir_name: str) -> str:
    return _DIGEST_SUFFIX_RE.sub("", dir_name) or dir_name


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )
