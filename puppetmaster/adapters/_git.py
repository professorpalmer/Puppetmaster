from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from puppetmaster.models import Artifact, Task

from ._facade import facade

_GIT_SUBPROCESS_TIMEOUT = 30


@dataclass
class GitSnapshot:
    """Typed git worktree snapshot for diff attribution and dirty-tree guards."""

    sha: str
    is_worktree: bool
    changed_files: list[str]
    untracked_files: list[str]
    diff: str
    tree: Optional[str] = None
    worker_changed_files: Optional[list[str]] = None
    worker_untracked_files: Optional[list[str]] = None
    worker_diff: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        if not hasattr(self, key):
            return default
        value = getattr(self, key)
        return default if value is None else value

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value


def _git_text(value: object) -> str:
    """Coerce subprocess stdout/stderr to str. On Windows without PYTHONUTF8,
    ``text=True`` can leave ``stdout`` as None after a decode failure in the
    reader thread — never call ``.strip()`` on the raw attribute."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_git(cwd: Path, args: list[str], *, strip: bool = True) -> str:
    try:
        completed = facade("subprocess").run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_GIT_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ""
    stdout = _git_text(completed.stdout)
    if strip:
        return stdout.strip() if completed.returncode == 0 else ""
    return stdout if completed.returncode == 0 else ""


def git_snapshot(cwd: Path, *, base_tree: Optional[str] = None) -> GitSnapshot:
    """Capture a diff-attributable snapshot of the working tree.

    Captures changes **against HEAD** (not just working-tree-vs-index) so that
    *staged* edits are seen by dirty-tree gating and included in dirty-state
    checks, and synthesizes no-index patches for *untracked* files. When
    ``base_tree`` is provided, also records a PM-attributable diff from that
    pre-worker tree to the current worktree. Also reports whether ``cwd`` is
    inside a git work tree so full-edit adapters can refuse to run outside a
    repo (where diffs are unattributable)."""
    inside = git_output(cwd, ["rev-parse", "--is-inside-work-tree"]) == "true"
    root = git_worktree_root(cwd) if inside else cwd
    sha = git_output(root, ["rev-parse", "HEAD"])
    untracked = git_untracked_files(root)
    if sha:
        # HEAD exists: `git diff HEAD` covers both staged and unstaged tracked
        # changes; plain `git diff` would silently drop staged edits.
        changed = git_lines(root, ["diff", "HEAD", "--name-only"])
        diff = git_diff_output(root, ["diff", "HEAD", "--binary"])
    else:
        # No commit yet (or detached/empty): union the staged and unstaged
        # diffs so nothing tracked is missed.
        changed = sorted(
            set(git_lines(root, ["diff", "--name-only"]))
            | set(git_lines(root, ["diff", "--cached", "--name-only"]))
        )
        diff = git_diff_output(root, ["diff", "--binary"]) + git_diff_output(
            root, ["diff", "--cached", "--binary"]
        )
    untracked_diff = git_untracked_diff(root, untracked)
    if untracked_diff:
        diff = (diff.rstrip("\n") + "\n" + untracked_diff) if diff.strip() else untracked_diff
    tree = facade("git_worktree_tree")(root) if inside else ""
    worker_diff = ""
    worker_changed: list[str] = []
    if base_tree and tree:
        worker_changed = git_lines(root, ["diff", "--name-only", base_tree, tree, "--"])
        worker_diff = git_diff_output(root, ["diff", "--binary", base_tree, tree, "--"])
    snapshot = GitSnapshot(
        sha=sha or "uncommitted",
        is_worktree=inside,
        changed_files=changed,
        untracked_files=untracked,
        diff=diff,
    )
    if tree:
        snapshot = GitSnapshot(
            sha=snapshot.sha,
            is_worktree=snapshot.is_worktree,
            changed_files=snapshot.changed_files,
            untracked_files=snapshot.untracked_files,
            diff=snapshot.diff,
            tree=tree,
        )
    if base_tree:
        worker_changed_set = set(worker_changed)
        snapshot = GitSnapshot(
            sha=snapshot.sha,
            is_worktree=snapshot.is_worktree,
            changed_files=snapshot.changed_files,
            untracked_files=snapshot.untracked_files,
            diff=snapshot.diff,
            tree=snapshot.tree,
            worker_changed_files=worker_changed,
            worker_untracked_files=[path for path in untracked if path in worker_changed_set],
            worker_diff=worker_diff,
        )
    return snapshot


def git_worktree_root(cwd: Path) -> Path:
    root = git_output(cwd, ["rev-parse", "--show-toplevel"])
    return Path(root) if root else cwd


def git_worktree_tree(cwd: Path) -> str:
    """Write the current worktree state to a temporary Git tree.

    Uses a throwaway index so staged/user index state is never modified. The tree
    includes tracked changes and untracked, non-ignored files, matching the files
    Puppetmaster can later report in a PATCH artifact.
    """
    sha = git_output(cwd, ["rev-parse", "HEAD"])
    fd, index_path = tempfile.mkstemp(prefix="puppetmaster-index-")
    os.close(fd)
    env = {**os.environ, "GIT_INDEX_FILE": index_path}
    try:
        _utf8 = {"text": True, "encoding": "utf-8", "errors": "replace"}
        if sha:
            read = facade("subprocess").run(
                ["git", "read-tree", sha],
                cwd=cwd,
                env=env,
                capture_output=True,
                check=False,
                timeout=_GIT_SUBPROCESS_TIMEOUT,
                **_utf8,
            )
        else:
            read = facade("subprocess").run(
                ["git", "read-tree", "--empty"],
                cwd=cwd,
                env=env,
                capture_output=True,
                check=False,
                timeout=_GIT_SUBPROCESS_TIMEOUT,
                **_utf8,
            )
        if read.returncode != 0:
            return ""
        add = facade("subprocess").run(
            ["git", "add", "-A", "--", "."],
            cwd=cwd,
            env=env,
            capture_output=True,
            check=False,
            timeout=_GIT_SUBPROCESS_TIMEOUT,
            **_utf8,
        )
        if add.returncode != 0:
            return ""
        written = facade("subprocess").run(
            ["git", "write-tree"],
            cwd=cwd,
            env=env,
            capture_output=True,
            check=False,
            timeout=_GIT_SUBPROCESS_TIMEOUT,
            **_utf8,
        )
        return _git_text(written.stdout).strip() if written.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""
    finally:
        try:
            os.unlink(index_path)
        except OSError:
            pass


def git_output(cwd: Path, args: list[str]) -> str:
    return _run_git(cwd, args, strip=True)


def git_diff_output(cwd: Path, args: list[str]) -> str:
    """Like :func:`git_output` but does not strip — diff bytes are significant
    and a trailing context newline can matter for patch application."""
    return _run_git(cwd, args, strip=False)


def git_lines(cwd: Path, args: list[str]) -> list[str]:
    output = git_output(cwd, args)
    return [line for line in output.splitlines() if line.strip()]


def git_untracked_diff(cwd: Path, untracked: list[str]) -> str:
    """Synthesize unified diffs for untracked files via ``git diff --no-index``.

    ``git diff`` never reports untracked files, so a run that only *creates*
    new files would otherwise produce an empty PATCH. ``--no-index`` exits 1
    when the files differ (the normal case here), so we can't reuse
    :func:`git_output`, which discards non-zero output."""
    chunks: list[str] = []
    for rel in untracked:
        try:
            if not (cwd / rel).is_file():
                continue  # skip directories / submodules / special files
        except OSError:
            continue
        try:
            completed = facade("subprocess").run(
                ["git", "diff", "--binary", "--no-index", "--", os.devnull, rel],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=_GIT_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            continue
        # 0 == identical (no output), 1 == differs (the diff we want), >1 == error.
        stdout = _git_text(completed.stdout)
        if completed.returncode in (0, 1) and stdout.strip():
            chunks.append(stdout)
    return "".join(chunks)


def git_untracked_files(cwd: Path) -> list[str]:
    output = git_output(cwd, ["status", "--short"])
    files = []
    for line in output.splitlines():
        if line.startswith("?? "):
            files.append(line[3:])
    return files


def worktree_guard(
    task: Task, worker_id: str, adapter: str, cwd: Path, before: dict
) -> Optional[list[Artifact]]:
    """Return a blocked-artifact list when a full-edit run is pointed outside a
    git work tree, else ``None``.

    Outside a repo, :func:`git_snapshot` reports ``sha='uncommitted'`` with no
    dirty state, so an editing agent would run with no dirty-tree gating and no
    reliable diff attribution — and could modify files anywhere. Callers can
    opt out with ``payload.allow_non_worktree=true`` for deliberately
    sandboxed/non-repo runs."""
    if before.get("is_worktree", True):
        return None
    if task.payload.get("allow_non_worktree", False):
        return None
    from ._base import verification_artifact

    return [
        verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter=adapter,
            check=task.instruction,
            result="blocked",
            confidence=0.85,
            evidence=[f"adapter:{adapter}", "status:not-a-worktree"],
            payload={
                "failure": "not_a_worktree",
                "message": (
                    f"{adapter} full-edit runs require cwd to be inside a git work tree "
                    "so Puppetmaster can gate on a clean tree and attribute the resulting "
                    "diff. Fix: run `git init` in the directory (restores diff capture), "
                    "point cwd at an existing repo, or set allow_non_worktree=true "
                    "(CLI: --allow-non-worktree) to run without diff attribution."
                ),
                "cwd": str(cwd),
            },
        )
    ]

