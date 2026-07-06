from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from puppetmaster.codegraph import inject_worker_cli_env
from puppetmaster.fs_permissions import mkdir_private, write_private_text
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.ports import apply_worktree_ports
from puppetmaster.redaction import redact_secrets

from ._facade import facade
from ._streaming import (
    StreamedProcess,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    _resolve_sidecar_state_dir,
    capture_subprocess_stdout,
    run_streamed_subprocess,
)

_PATCH_INLINE_CHARS = 20000


def build_patch_payload(
    *,
    task: Task,
    before: dict,
    after: dict,
    status: str,
    change: str,
    sidecar_name: str,
) -> dict[str, Any]:
    """Build a PATCH artifact payload from before/after git snapshots.

    Centralizes three previously-missing safeguards for full-edit runs:

    - **Redaction**: the unified diff is scrubbed for secrets before it is
      persisted (a diff that adds a key to a ``.env`` would otherwise leak it).
    - **No silent truncation**: large diffs are spooled in full to a sidecar
      and only an excerpt is kept inline, with explicit ``diff_truncated`` /
      ``diff_total_chars`` metadata, instead of slicing to the last 20k chars
      and producing an unapplyable patch.
    - **Untracked visibility**: untracked files are surfaced alongside changed
      files so a run that only creates new files is still attributable.
    """
    raw_diff = str(
        after.get("worker_diff")
        if after.get("worker_diff") is not None
        else after.get("diff") or ""
    )
    diff_source = diff_source_payload(before, after)
    redacted = redact_secrets(raw_diff) or ""
    total = len(redacted)
    truncated = total > _PATCH_INLINE_CHARS
    inline = redacted[-_PATCH_INLINE_CHARS:] if truncated else redacted
    payload: dict[str, Any] = {
        "change": change,
        "files": after.get("worker_changed_files", after.get("changed_files", [])),
        "untracked_files": after.get("worker_untracked_files", after.get("untracked_files", [])),
        "unified_diff": inline,
        **diff_source,
        "diff_total_chars": total,
        "diff_truncated": truncated,
        "base_sha": before.get("sha"),
        "head_sha": after.get("sha"),
        "base_tree": before.get("tree"),
        "head_tree": after.get("tree"),
        "status": status,
        "revert": (
            "Review the diff, then use git restore / git checkout or your VCS "
            "workflow to revert unwanted changes."
        ),
    }
    if truncated:
        payload["unified_diff_sidecar_path"] = _spool_patch_sidecar(
            task=task, sidecar_name=sidecar_name, diff=redacted
        )
    return payload


def snapshot_has_diff(snapshot: dict) -> bool:
    """True when a git snapshot reports any tracked or untracked diff."""
    return bool(
        str(snapshot.get("diff") or "").strip()
        or snapshot.get("changed_files")
        or snapshot.get("untracked_files")
    )


def diff_source_payload(before: dict, after: dict) -> dict[str, bool]:
    """Label whether a diff existed before the worker and whether it changed."""
    worker_diff = after.get("worker_diff")
    if worker_diff is not None:
        worker_diff_present = bool(str(worker_diff).strip())
    elif facade("snapshot_has_diff")(before):
        worker_diff_present = False
    else:
        worker_diff_present = facade("snapshot_has_diff")(after)
    return {
        "baseline_diff_present": facade("snapshot_has_diff")(before),
        "worker_diff_present": worker_diff_present,
    }


def dirty_worktree_paths_note(
    changed_files: object,
    untracked_files: object,
    *,
    limit: int = 10,
) -> str:
    """A short, named sample of the paths that make the tree dirty.

    The clean-tree guard is correct to block, but a bare "dirty worktree"
    reads like a real failure when the only offender is a stray ``__pycache__/``.
    Naming the paths in the message (not just the payload fields) lets a user
    see at a glance whether it's junk or real work without drilling in.
    """
    labeled = [f"{path} (modified)" for path in (changed_files or [])]
    labeled += [f"{path} (untracked)" for path in (untracked_files or [])]
    if not labeled:
        return ""
    shown = labeled[:limit]
    overflow = len(labeled) - len(shown)
    suffix = f" (+{overflow} more)" if overflow > 0 else ""
    return " Offending paths: " + ", ".join(shown) + suffix + "."


def make_patch_artifact(
    task: Task,
    worker_id: str,
    before: dict,
    after: dict,
    *,
    adapter: str,
    status: str,
    change: str,
    sidecar_name: str,
    evidence_adapter: Optional[str] = None,
) -> Artifact:
    label = evidence_adapter or adapter
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.PATCH,
        created_by=worker_id,
        confidence=0.8 if status == "applied" else 0.5,
        evidence=[f"adapter:{label}", f"base:{before['sha']}"],
        payload=build_patch_payload(
            task=task,
            before=before,
            after=after,
            status=status,
            change=change,
            sidecar_name=sidecar_name,
        ),
    )


def dirty_worktree_guard(
    task: Task,
    worker_id: str,
    adapter: str,
    before: dict,
    *,
    adapter_label: Optional[str] = None,
    extra_message: str = "",
) -> Optional[list[Artifact]]:
    if task.payload.get("allow_dirty", False):
        return None
    if not (before["changed_files"] or before["untracked_files"]):
        return None
    label = adapter_label or adapter
    return [
        verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter=adapter,
            check=task.instruction,
            result="blocked",
            confidence=0.8,
            evidence=[f"adapter:{label}", "status:dirty-repo"],
            payload={
                "failure": "dirty_worktree",
                "message": (
                    f"{label} full-edit runs require a clean working tree by default "
                    "so Puppetmaster can attribute the resulting diff correctly. Commit, "
                    "stash, use a worktree, or set payload.allow_dirty=true."
                    + extra_message
                    + dirty_worktree_paths_note(
                        before["changed_files"], before["untracked_files"]
                    )
                ),
                "changed_files": before["changed_files"],
                "untracked_files": before["untracked_files"],
                **diff_source_payload(before, {}),
            },
        )
    ]


def missing_cli_artifact(
    task: Task,
    worker_id: str,
    adapter: str,
    executable: object,
    message: str,
) -> list[Artifact]:
    return [
        verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter=adapter,
            check=task.instruction,
            result="blocked",
            confidence=0.45,
            evidence=[f"adapter:{adapter}", "status:missing-cli"],
            payload={
                "failure": "missing_cli",
                "message": message,
                "executable": executable,
            },
        )
    ]


def failure_verification(
    task: Task,
    worker_id: str,
    adapter: str,
    evidence: list[str],
    failure: str,
    stderr: str,
    *,
    returncode: Optional[int] = None,
    confidence: float = 0.55,
    extra: Optional[dict[str, Any]] = None,
) -> Artifact:
    payload: dict[str, Any] = {
        "failure": failure,
        "returncode": returncode,
        "stderr": _redacted_tail(stderr, _STDOUT_TAIL_CHARS),
    }
    if extra:
        payload.update(extra)
    return verification_artifact(
        task=task,
        worker_id=worker_id,
        adapter=adapter,
        check=task.instruction,
        result="failed",
        confidence=confidence,
        evidence=evidence,
        payload=payload,
    )


class FullEditWorkerAdapter:
    """Shared git snapshot + worktree/dirty guards for full-edit adapter runs."""

    name: str

    @staticmethod
    def guard_full_edit_run(
        task: Task,
        worker_id: str,
        adapter: str,
        cwd: Path,
        *,
        adapter_label: Optional[str] = None,
        extra_dirty_message: str = "",
    ) -> tuple[Optional[list[Artifact]], dict]:
        before = facade("git_snapshot")(cwd)
        blocked = facade("worktree_guard")(task, worker_id, adapter, cwd, before)
        if blocked is not None:
            return blocked, before
        dirty = dirty_worktree_guard(
            task,
            worker_id,
            adapter,
            before,
            adapter_label=adapter_label,
            extra_message=extra_dirty_message,
        )
        if dirty is not None:
            return dirty, before
        return None, before


@dataclass
class CliInvocation:
    """Prepared CLI subprocess state for :class:`CliWorkerAdapter`."""

    command: list[str]
    sidecar_name: str
    env: Optional[dict] = None
    subprocess_kwargs: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class CliWorkerAdapter(FullEditWorkerAdapter):
    """Shared snapshot → guard → CLI invoke → snapshot → artifact lifecycle."""

    default_timeout_seconds: int = 600

    def _run_cli_lifecycle(
        self,
        task: Task,
        goal: str,
        worker_id: str,
        *,
        pre_guard: bool = False,
    ) -> list[Artifact]:
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        timeout_seconds = int(
            task.payload.get("timeout_seconds", self.default_timeout_seconds)
        )
        before: dict = {}
        if pre_guard:
            blocked, before = self._early_run_guard(task, worker_id, cwd)
            if blocked is not None:
                return blocked

        executable_label, resolved = self._resolve_cli_executable(task)
        if resolved is None:
            return self._missing_cli(task, worker_id, executable_label)

        prepared = self._prepare_cli_invocation(
            task, goal, worker_id, cwd, resolved
        )
        if isinstance(prepared, list):
            return prepared

        if not pre_guard:
            blocked, before = self._apply_pre_run_guards(task, worker_id, cwd, prepared)
            if blocked is not None:
                return blocked
        elif not before:
            before = facade("git_snapshot")(cwd)

        completed = self._invoke_cli(task, prepared, cwd, timeout_seconds)
        after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
        return self._finalize_cli_run(
            task, worker_id, goal, prepared, before, after, completed
        )

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        raise NotImplementedError

    def _missing_cli(
        self, task: Task, worker_id: str, executable_label: str
    ) -> list[Artifact]:
        raise NotImplementedError

    def _prepare_cli_invocation(
        self,
        task: Task,
        goal: str,
        worker_id: str,
        cwd: Path,
        resolved: str,
    ) -> Union[list[Artifact], CliInvocation]:
        raise NotImplementedError

    def _early_run_guard(
        self, task: Task, worker_id: str, cwd: Path
    ) -> tuple[Optional[list[Artifact]], dict]:
        return None, {}

    def _apply_pre_run_guards(
        self,
        task: Task,
        worker_id: str,
        cwd: Path,
        prepared: CliInvocation,
    ) -> tuple[Optional[list[Artifact]], dict]:
        return self.guard_full_edit_run(
            task,
            worker_id,
            self.name,
            cwd,
            adapter_label=prepared.extras.get("adapter_label"),
            extra_dirty_message=prepared.extras.get("extra_dirty_message", ""),
        )

    def _invoke_cli(
        self,
        task: Task,
        prepared: CliInvocation,
        cwd: Path,
        timeout_seconds: int,
    ) -> StreamedProcess:
        env = prepared.env
        if env is None:
            env = inject_worker_cli_env(apply_worktree_ports(os.environ.copy(), cwd))
        kwargs = dict(prepared.subprocess_kwargs)
        kwargs.setdefault("cwd", str(cwd))
        return facade("run_streamed_subprocess")(
            command=prepared.command,
            env=env,
            task=task,
            sidecar_name=prepared.sidecar_name,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

    def _finalize_cli_run(
        self,
        task: Task,
        worker_id: str,
        goal: str,
        prepared: CliInvocation,
        before: dict,
        after: dict,
        completed: StreamedProcess,
    ) -> list[Artifact]:
        raise NotImplementedError


def _spool_patch_sidecar(*, task: Task, sidecar_name: str, diff: str) -> Optional[str]:
    """Write the full (already-redacted) diff to a sidecar file next to the
    task's other spooled output. Returns the path, or ``None`` if no state dir
    is in scope or the write fails."""
    state_dir = _resolve_sidecar_state_dir()
    if state_dir is None:
        return None
    try:
        sidecar_dir = state_dir / "jobs" / task.job_id / "tasks" / task.id
        mkdir_private(sidecar_dir)
        sidecar_path = sidecar_dir / f"{sidecar_name}.patch"
        write_private_text(sidecar_path, diff)
        return str(sidecar_path)
    except OSError:
        return None


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    status: str
    description: str
    requires: list[str]


class WorkerAdapter(Protocol):
    name: str

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        """Execute a task and return structured artifacts."""


def verification_artifact(
    *,
    task: Task,
    worker_id: str,
    adapter: str,
    check: str,
    result: str,
    confidence: float,
    evidence: list[str],
    payload: dict,
    evaluator_slot: str = "",
    evaluator_version: int = 0,
) -> Artifact:
    body = {
        "adapter": adapter,
        "check": check,
        "result": result,
        **payload,
    }
    if evaluator_slot:
        body["evaluator_slot"] = evaluator_slot
    if evaluator_version:
        body["evaluator_version"] = evaluator_version
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.VERIFICATION,
        created_by=worker_id,
        confidence=confidence,
        evidence=evidence,
        payload=body,
    )


def command_parts(command: object) -> list[str]:
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return command
    if isinstance(command, str):
        # POSIX-mode shlex treats backslashes as escapes, which mangles Windows
        # paths like ``C:\Users\me\claude.exe`` into an unresolvable token. Split
        # in non-POSIX mode on Windows so native paths survive intact.
        return shlex.split(command, posix=(os.name != "nt"))
    raise ValueError("provider executable must be a string or list of strings")


def resolve_command(executable: str) -> Optional[str]:
    path = Path(executable).expanduser()
    if path.exists():
        return str(path)
    return shutil.which(executable)


def tool_list(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _should_emit_patch_artifact(before: dict, after: dict) -> bool:
    """True when a PATCH can be attributed to the worker run."""
    worker_diff = after.get("worker_diff")
    if worker_diff is not None:
        return bool(str(worker_diff).strip())
    if facade("snapshot_has_diff")(before):
        return False
    return bool(str(after.get("diff") or "").strip())

