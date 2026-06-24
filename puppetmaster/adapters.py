from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union

from puppetmaster.codegraph import (
    enrich_prompt_with_codegraph,
    inject_worker_cli_env,
    repo_file_census,
    scrub_foreign_interpreter_env,
)
from puppetmaster.fs_permissions import mkdir_private, open_private, write_private_text
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.openai_security import (
    DEFAULT_OPENAI_BASE_URL,
    validate_openai_base_url_for_task,
)
from puppetmaster.ports import apply_worktree_ports
from puppetmaster.redaction import redact_secrets
from puppetmaster.usage import token_usage


# Default truncation budgets. Match what the codebase used pre-spool so existing
# verification / risk artifacts keep their inline excerpts; the new sidecar file
# preserves whatever falls in the middle so nothing is silently dropped.
_STDOUT_HEAD_CHARS = 1000
_STDOUT_TAIL_CHARS = 8000

# Inline budget for a PATCH artifact's unified diff. Larger diffs are spooled to
# a sidecar (full, redacted) and only an excerpt is kept inline so the JSON
# payload stays bounded but the patch is never silently lost.
_PATCH_INLINE_CHARS = 20000


def _coerce_text(value: object) -> str:
    """Normalize subprocess output to ``str``. ``TimeoutExpired.stdout`` may be
    bytes (or None) depending on how the child was captured, while a normal
    ``CompletedProcess`` yields str under ``text=True``."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _redacted_tail(value: object, limit: int) -> str:
    """Redact secrets, then keep the last ``limit`` chars for an inline excerpt."""
    text = redact_secrets(_coerce_text(value)) or ""
    return text[-limit:]


def _resolve_sidecar_state_dir() -> Optional[Path]:
    """Locate the active Puppetmaster state directory for sidecar spooling.

    Returns ``None`` if no state dir is in scope (e.g. direct adapter unit
    tests). Falling back to the default state dir would write logs into a
    workspace-hashed path that may not own the job, so we only honor an
    explicit ``PUPPETMASTER_STATE_DIR`` env var (which ``worker_runtime``
    exports after resolving its --state-dir flag).
    """
    raw = os.environ.get("PUPPETMASTER_STATE_DIR")
    if not raw:
        return None
    try:
        return Path(raw)
    except (TypeError, ValueError):
        return None


def capture_subprocess_stdout(
    *,
    text: str,
    task: Task,
    sidecar_name: str,
    head_chars: int = _STDOUT_HEAD_CHARS,
    tail_chars: int = _STDOUT_TAIL_CHARS,
) -> dict[str, Any]:
    """Build the stdout-capture metadata dict for an adapter artifact payload.

    Returns a dict with explicit truncation markers and (when the content
    exceeds head+tail and a state dir is available) a sidecar log file that
    preserves the full subprocess output. The dict is meant to be merged
    into the artifact payload alongside the legacy ``stdout`` (tail) and
    ``stdout_excerpt`` (head) fields so older callers keep working.

    Keys returned:

    - ``stdout_total_chars`` (int): total length of ``text``.
    - ``stdout_truncated`` (bool): True when head+tail can't fit the full text.
    - ``stdout_head_excerpt`` (str): first N chars when truncated, else full text.
    - ``stdout_tail_excerpt`` (str): last N chars when truncated, else "".
    - ``stdout_sidecar_path`` (str | None): absolute path to the spooled
      sidecar file when truncated and the spool succeeded, else None.
    - ``stdout_sidecar_error`` (str, optional): only set when spooling was
      attempted but failed (filesystem error).

    The text is secret-redacted before any excerpt or sidecar is produced, so
    an agent transcript that echoes an API key never lands in persisted state.
    """
    text = redact_secrets(text) or ""
    total = len(text)
    truncated = total > (head_chars + tail_chars)
    result: dict[str, Any] = {
        "stdout_total_chars": total,
        "stdout_truncated": truncated,
        "stdout_head_excerpt": text[:head_chars] if truncated else text,
        "stdout_tail_excerpt": text[-tail_chars:] if truncated else "",
    }
    if not truncated:
        return result

    state_dir = _resolve_sidecar_state_dir()
    if state_dir is None:
        result["stdout_sidecar_path"] = None
        return result
    try:
        sidecar_dir = state_dir / "jobs" / task.job_id / "tasks" / task.id
        mkdir_private(sidecar_dir)
        sidecar_path = sidecar_dir / f"{sidecar_name}.log"
        write_private_text(sidecar_path, text)
        result["stdout_sidecar_path"] = str(sidecar_path)
    except OSError as exc:
        result["stdout_sidecar_path"] = None
        result["stdout_sidecar_error"] = repr(exc)
    return result


@dataclass
class StreamedProcess:
    """Result of a streamed subprocess run (mirrors the subset of
    ``CompletedProcess`` the adapters use, plus liveness metadata)."""

    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    live_log_path: Optional[str] = None
    elapsed_seconds: float = 0.0


def run_streamed_subprocess(
    *,
    command: list[str],
    env: Optional[dict],
    task: Task,
    sidecar_name: str,
    timeout_seconds: int,
    cwd: Optional[str] = None,
    heartbeat_seconds: float = 30.0,
    start_new_session: bool = False,
) -> StreamedProcess:
    """Run ``command`` while teeing its output to a live sidecar log.

    A long agent run (e.g. ``cursor --implement``) used to produce a 0-byte log
    for minutes and then flush everything at exit, making "working" and "hung"
    indistinguishable without external ``pgrep``/``find -mmin`` heuristics. This
    streams stdout/stderr line-by-line to ``<task>/<sidecar_name>_live.log`` as
    they arrive and writes a ``still working`` heartbeat every
    ``heartbeat_seconds`` of the run, so the log visibly grows. Returns separate
    stdout/stderr buffers so existing artifact payloads are unchanged.
    """
    import threading
    import time as _time

    state_dir = _resolve_sidecar_state_dir()
    live_handle = None
    live_path: Optional[Path] = None
    if state_dir is not None:
        try:
            sidecar_dir = state_dir / "jobs" / task.job_id / "tasks" / task.id
            mkdir_private(sidecar_dir)
            live_path = sidecar_dir / f"{sidecar_name}_live.log"
            live_handle = open(
                open_private(live_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC),
                "w",
                encoding="utf-8",
                errors="replace",
                closefd=True,
            )
        except OSError:
            live_handle = None

    write_lock = threading.Lock()

    def _write_live(line: str) -> None:
        if live_handle is None:
            return
        try:
            redacted = redact_secrets(line) or line
            with write_lock:
                live_handle.write(redacted)
                live_handle.flush()
        except Exception:
            pass

    popen_kwargs: dict[str, Any] = {}
    if start_new_session:
        # Hermes tears down its own process group on exit and has been observed
        # signal-killing a parent shell loop. Launching in a fresh session keeps
        # that teardown confined to the child and away from Puppetmaster.
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        # Close stdin: an agent CLI launched in a non-interactive worker must
        # never block forever waiting on terminal input (a silent "stall").
        # Callers that previously passed input="" rely on this EOF behavior.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **popen_kwargs,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _reader(stream, buffer: list[str], tag: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                buffer.append(line)
                _write_live(line if line.endswith("\n") else line + "\n")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    threads = [
        threading.Thread(target=_reader, args=(process.stdout, stdout_lines, "out"), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, stderr_lines, "err"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    stop_heartbeat = threading.Event()
    started = _time.monotonic()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_seconds):
            elapsed = int(_time.monotonic() - started)
            _write_live(
                f"[puppetmaster] still working: {elapsed}s elapsed, "
                f"{len(stdout_lines)} stdout / {len(stderr_lines)} stderr lines so far\n"
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        stop_heartbeat.set()
        for thread in threads:
            thread.join(timeout=2)
        elapsed = _time.monotonic() - started
        if live_handle is not None:
            _write_live(
                f"[puppetmaster] process exited rc={process.returncode} "
                f"timed_out={timed_out} after {int(elapsed)}s\n"
            )
            try:
                live_handle.close()
            except Exception:
                pass

    return StreamedProcess(
        returncode=process.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        timed_out=timed_out,
        live_log_path=str(live_path) if live_path is not None else None,
        elapsed_seconds=_time.monotonic() - started,
    )


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
    elif snapshot_has_diff(before):
        worker_diff_present = False
    else:
        worker_diff_present = snapshot_has_diff(after)
    return {
        "baseline_diff_present": snapshot_has_diff(before),
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


# Grounding boundary shared by every analyze-role artifact contract. Without it,
# a worker on a small repo can mistake the contract scaffolding for the analysis
# subject — the redteam role in particular emitted a nonsense "the 'Puppetmaster
# artifact contract' was mentioned but no context found" risk. Anchoring the
# target on the repository (and telling honest-empty runs to return [] rather
# than fabricate a meta-risk) fixes the whole class across all adapters.
_ARTIFACT_GROUNDING = (
    "Your analysis target is THIS repository's code and configuration — not "
    "these instructions, not this artifact contract, and not the run itself. "
    "Ground every artifact in concrete files, functions, or symbols."
)
_ARTIFACT_EMPTY_GUIDANCE = (
    "If the repository genuinely yields nothing for your role (e.g. it is tiny "
    'or sound), return an empty list {"artifacts":[]} — never invent a finding '
    "or a risk about the prompt, the contract, or the run being degraded."
)

# Appended to the prompt for a single retry when an analyze worker returns no
# structured artifacts. The cheapest/minimal-effort workers occasionally answer
# in prose the parser can't structure; one stricter JSON-only reprompt recovers
# the run instead of letting it flicker as degraded.
_ANALYZE_JSON_ONLY_RETRY = (
    "\n\nIMPORTANT: your previous response did not contain the required "
    "structured output. Respond with ONLY a single JSON object of the form "
    '{"artifacts": [...]} exactly as specified above — no prose, no explanation, '
    "no markdown fences, nothing before or after the JSON. If you genuinely found "
    'nothing for your role, return {"artifacts": []}.'
)


def with_repo_census(prompt: str, cwd: Union[Path, str, None]) -> str:
    """Append an authoritative repo file census so a worker can't hallucinate
    an empty repository.

    When files exist, the census states plainly that the repo is NOT empty and
    tells the worker to read them (and to report a tooling failure rather than
    assert emptiness if its own tools can't). When nothing can be enumerated we
    add only a soft boundary — we never assert emptiness ourselves, since an
    enumeration miss is not proof of an empty tree.
    """
    sample, total = repo_file_census(cwd)
    if total <= 0:
        return (
            prompt
            + "\n\nRepository file census: none enumerated. Do not assert the "
            "repository is empty unless your own tools also show no files — if "
            "they error, report a tooling failure, not an empty repository."
        )
    shown = ", ".join(sample)
    overflow = total - len(sample)
    more = f" (+{overflow} more)" if overflow > 0 else ""
    return (
        prompt
        + f"\n\nRepository file census (ground truth — {total} file(s) under the "
        f"working directory): {shown}{more}.\nThis census is authoritative: the "
        "repository is NOT empty. Read the relevant files before reporting. Never "
        "claim the repo is empty or 'starting from scratch' when files are listed "
        "here; if your own tools cannot read them, report a tooling failure, not "
        "an empty repository."
    )


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


class LocalAdapter:
    name = "local"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        role_map = {
            "explore": self._explore,
            "architect": self._architect,
            "implement": self._implement,
            "redteam": self._redteam,
            "test": self._test,
        }
        return role_map.get(task.role, self._explore)(task, goal, worker_id)

    def _explore(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by=worker_id,
                confidence=0.9,
                evidence=["concept:independent-workers", "concept:structured-artifacts"],
                payload={
                    "claim": "The swarm should share durable state, not inherited transcript.",
                    "goal": goal,
                    "implication": "Workers can stay small, replayable, and independently replaceable.",
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by=worker_id,
                confidence=0.84,
                evidence=["concept:redis-gunicorn-analogy"],
                payload={
                    "claim": "Redis/Gunicorn is a useful operating model for agent swarms.",
                    "implication": "Use streams, locks, worker runs, and artifacts as first-class objects.",
                },
            ),
        ]

    def _architect(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.DECISION,
                created_by=worker_id,
                confidence=0.88,
                evidence=["mvp:local-filesystem", "mvp:no-runtime-dependencies"],
                payload={
                    "decision": "Start with a file-backed store that mirrors Redis key spaces.",
                    "why": "It makes jobs, streams, artifacts, and memory inspectable before adding Redis.",
                    "next": "Swap the store adapter for Redis Streams when live coordination matters.",
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.DECISION,
                created_by=worker_id,
                confidence=0.86,
                evidence=["rule:state-not-context"],
                payload={
                    "decision": "Synthesis reads JSON artifacts only.",
                    "why": "The final agent should not need raw worker transcripts to reconstruct state.",
                    "goal": goal,
                },
            ),
        ]

    def _implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.PATCH,
                created_by=worker_id,
                confidence=0.82,
                evidence=["module:orchestrator", "module:store", "module:stitcher"],
                payload={
                    "change": "Create a CLI that spawns role-specific workers and writes artifacts.",
                    "files": [
                        "puppetmaster/orchestrator.py",
                        "puppetmaster/store.py",
                        "puppetmaster/stitcher.py",
                    ],
                    "goal": goal,
                },
            )
        ]

    def _redteam(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.RISK,
                created_by=worker_id,
                confidence=0.8,
                evidence=["risk:artifact-quality"],
                payload={
                    "risk": "If workers emit vague prose, stitching collapses into summary theater.",
                    "mitigation": "Validate artifact type, confidence, evidence, and payload shape.",
                    "goal": goal,
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.RISK,
                created_by=worker_id,
                confidence=0.76,
                evidence=["risk:write-conflicts"],
                payload={
                    "risk": "Independent workers can collide when touching the same scope.",
                    "mitigation": "Use locks for file scopes and promote memory only after synthesis.",
                },
            ),
        ]

    def _test(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.VERIFICATION,
                created_by=worker_id,
                confidence=0.81,
                evidence=["test:job-produces-artifacts", "test:stitcher-promotes-memory"],
                payload={
                    "check": "A run should produce tasks, artifacts, a stitched summary, and promoted memory.",
                    "result": "designed",
                    "goal": goal,
                },
            )
        ]


class ShellAdapter:
    name = "shell"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        command = task.payload.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError("shell adapter requires payload.command as a list of strings")

        timeout_seconds = int(task.payload.get("timeout_seconds", 30))
        cwd = task.payload.get("cwd")
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(cwd)) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="shell",
                    check=task.instruction,
                    result="failed",
                    confidence=0.7,
                    evidence=[f"command:{' '.join(command)}", "timeout"],
                    payload={
                        "returncode": None,
                        "stdout": _redacted_tail(exc.stdout, 4000),
                        "stderr": _redacted_tail(exc.stderr, 4000),
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]
        return [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="shell",
                check=task.instruction,
                result="passed" if completed.returncode == 0 else "failed",
                confidence=0.95 if completed.returncode == 0 else 0.65,
                evidence=[f"command:{' '.join(command)}"],
                payload={
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, 4000),
                    "stderr": _redacted_tail(completed.stderr, 4000),
                },
            )
        ]


class CursorAdapter:
    name = "cursor"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        # Two execution modes share the same Cursor SDK runner. ``analyze`` (the
        # default) steers the agent to return findings/risks and never touches
        # the tree; ``implement`` lets it edit files and captures the resulting
        # diff as a PATCH artifact — the same full-edit contract Claude Code and
        # Codex already honour, so a cursor-only platform lock can still ship code.
        if task.payload.get("mode") == "implement" or task.payload.get("implement"):
            return self._run_implement(task, goal, worker_id)
        return self._run_analyze(task, goal, worker_id)

    def _run_implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        model = task.payload.get("model", "default")

        # Fail fast on a dirty tree before spending any work (codegraph, agent).
        before = git_snapshot(cwd)
        blocked = worktree_guard(task, worker_id, "cursor", cwd, before)
        if blocked is not None:
            return blocked
        if not task.payload.get("allow_dirty", False) and (
            before["changed_files"] or before["untracked_files"]
        ):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="cursor",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.8,
                    evidence=["adapter:cursor-sdk", "status:dirty-repo"],
                    payload={
                        "failure": "dirty_worktree",
                        "message": (
                            "Cursor implement runs require a clean working tree by default "
                            "so Puppetmaster can attribute the resulting diff correctly. Commit, "
                            "stash, use a worktree, or set payload.allow_dirty=true. For focused "
                            "edits on a dirty tree (docs, tests), use puppetmaster_edit — it edits "
                            "in place and needs no clean tree."
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

        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(self._implement_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )

        runner = Path(__file__).with_name("cursor_sdk_runner.mjs")
        environment = inject_worker_cli_env(os.environ.copy())
        apply_worktree_ports(environment, cwd)
        environment["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
            {"prompt": prompt, "cwd": str(cwd), "model": model},
            sort_keys=True,
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 900))
        # Stream output to a live sidecar log so a multi-minute agent run is
        # visibly making progress (no more 0-byte-log-then-flush ambiguity).
        completed = run_streamed_subprocess(
            command=["node", str(runner)],
            env=environment,
            task=task,
            sidecar_name="cursor_implement",
            timeout_seconds=timeout_seconds,
        )
        if completed.timed_out:
            after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
            artifacts: list[Artifact] = [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="cursor",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=["adapter:cursor-sdk", "mode:implement", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                        "timeout_seconds": timeout_seconds,
                        "live_log": completed.live_log_path,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                    },
                )
            ]
            # A timed-out implement run may still have edited files; surface the
            # partial diff so the work isn't silently lost.
            if _should_emit_patch_artifact(before, after):
                artifacts.append(
                    self._patch_artifact(task, worker_id, before, after, status="failed")
                )
            return artifacts

        after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
        failure = classify_cursor_failure(completed.stderr + completed.stdout)
        cursor_status, result_text = cursor_result_text(completed.stdout)
        usage = token_usage(
            sdk_usage=sdk_usage_from_stdout(completed.stdout),
            prompt_text=prompt,
            output_text=completed.stdout,
        )
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout, task=task, sidecar_name="cursor_implement_stdout", tail_chars=12000
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr, task=task, sidecar_name="cursor_implement_stderr"
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="cursor",
                check=task.instruction,
                result="passed" if completed.returncode == 0 else "failed",
                confidence=0.9 if completed.returncode == 0 else 0.55,
                evidence=(
                    ["adapter:cursor-sdk", "mode:implement", f"node:{sys.platform}"]
                    + (["context:codegraph"] if codegraph_used else [])
                ),
                payload={
                    "failure": None if completed.returncode == 0 else failure,
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, 12000),
                    "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "live_log": completed.live_log_path,
                    "model": model,
                    "cwd": str(cwd),
                    "cursor_status": cursor_status,
                    "base_sha": before["sha"],
                    "head_sha": after["sha"],
                    "changed_files": after["changed_files"],
                    "untracked_files": after["untracked_files"],
                    **diff_source_payload(before, after),
                    **usage,
                },
            )
        ]
        # The agent's final message is its report (root cause, files touched,
        # verification). Persist it as artifacts so the stitched summary and
        # quality verdict see the work instead of calling the run degraded.
        if completed.returncode == 0:
            artifacts.extend(
                implement_report_artifacts(
                    task, worker_id, result_text, adapter="cursor-sdk"
                )
            )
        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                self._patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    status="applied" if completed.returncode == 0 else "failed",
                )
            )
        return artifacts

    @staticmethod
    def _patch_artifact(task: Task, worker_id: str, before, after, *, status: str) -> Artifact:
        return Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.PATCH,
            created_by=worker_id,
            confidence=0.8 if status == "applied" else 0.5,
            evidence=["adapter:cursor-sdk", f"base:{before['sha']}"],
            payload=build_patch_payload(
                task=task,
                before=before,
                after=after,
                status=status,
                change="Cursor agent modified repository files.",
                sidecar_name="cursor_implement",
            ),
        )

    def _run_analyze(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model", "default")
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(
                self._structured_prompt(base_prompt),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = with_repo_census(prompt, cwd)
        runner = Path(__file__).with_name("cursor_sdk_runner.mjs")
        environment = inject_worker_cli_env(os.environ.copy())
        apply_worktree_ports(environment, cwd)
        environment["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
            {"prompt": prompt, "cwd": cwd, "model": model},
            sort_keys=True,
        )
        try:
            completed = subprocess.run(
                ["node", str(runner)],
                capture_output=True,
                text=True,
                timeout=int(task.payload.get("timeout_seconds", 300)),
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            stdout = exc.stdout or ""
            stdout_capture = capture_subprocess_stdout(
                text=stdout,
                task=task,
                sidecar_name="cursor_stdout_timeout",
            )
            stderr_capture = capture_subprocess_stdout(
                text=stderr,
                task=task,
                sidecar_name="cursor_stderr_timeout",
            )
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="cursor",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=["adapter:cursor-sdk", "timeout"],
                    payload={
                        "returncode": None,
                        "stdout": stdout[-_STDOUT_TAIL_CHARS:],
                        "stderr": stderr[-_STDOUT_TAIL_CHARS:],
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "model": model,
                        "failure": classify_cursor_failure(stderr + stdout),
                    },
                )
            ]
        failure = classify_cursor_failure(completed.stderr + completed.stdout)
        status, result_text = cursor_result_text(completed.stdout)
        usage = token_usage(
            sdk_usage=sdk_usage_from_stdout(completed.stdout),
            prompt_text=prompt,
            output_text=result_text or completed.stdout,
        )
        parsed_artifacts = (
            cursor_result_artifacts(task, worker_id, result_text)
            if completed.returncode == 0
            else []
        )
        # Salvage (#3): structured content can sit in raw stdout even when the
        # SDK's `result` field was empty (a dry-run that printed but never
        # "finished") or the run exited non-zero. Parse raw stdout before
        # declaring the run degraded, so a real review isn't lost to a manual
        # log read.
        if not parsed_artifacts:
            salvaged = cursor_result_artifacts(task, worker_id, completed.stdout)
            if salvaged:
                parsed_artifacts = salvaged
        degraded = completed.returncode == 0 and not parsed_artifacts
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="cursor_stdout",
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="cursor_stderr",
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="cursor",
                check=task.instruction,
                result=(
                    "degraded"
                    if degraded
                    else "passed"
                    if completed.returncode == 0
                    else "failed"
                ),
                confidence=0.65 if degraded else 0.9 if completed.returncode == 0 else 0.55,
                evidence=(
                    ["adapter:cursor-sdk", f"node:{sys.platform}"]
                    + (["context:codegraph"] if codegraph_used else [])
                ),
                payload={
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-_STDOUT_TAIL_CHARS:],
                    "stderr": completed.stderr[-_STDOUT_TAIL_CHARS:],
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "model": model,
                    "cursor_status": status,
                    "failure": (
                        "empty_or_unstructured_cursor_result"
                        if degraded
                        else failure
                        if completed.returncode != 0
                        else None
                    ),
                    **usage,
                },
            )
        ]
        if degraded:
            artifacts.append(
                cursor_degraded_artifact(
                    task,
                    worker_id,
                    result_text,
                    stdout_capture=stdout_capture,
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts

    @staticmethod
    def _implement_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Implement mode: you are running as a full-edit Puppetmaster worker "
                "inside the user's repository. Actually make the code changes — create, "
                "edit, and delete files as needed to complete the task end to end. Do not "
                "just describe a plan or return findings.",
                "Keep the change focused on the task; run any obvious local checks you can. "
                "Puppetmaster captures the resulting git diff as a PATCH artifact, so leave "
                "the working tree containing your final intended changes.",
                _IMPLEMENT_REPORT_CONTRACT,
            ]
        )

    @staticmethod
    def _structured_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Puppetmaster artifact contract:",
                "Return only JSON, with no markdown wrapper, in this shape:",
                '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
                "Allowed artifact types:",
                '- finding: requires "claim", "evidence", "confidence".',
                '- risk: requires "risk", "mitigation", "evidence", "confidence".',
                '- decision: requires "decision", "why", "evidence", "confidence".',
                _ARTIFACT_GROUNDING,
                _ARTIFACT_EMPTY_GUIDANCE,
            ]
        )


DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-4-8"


# Bedrock model identifiers are inference-profile / ARN shaped, never the short
# Cursor-style names the registry carries ("claude-opus-4-8"). The `claude` CLI
# on Bedrock rejects a short name outright — the failure mode that took down a
# worker (invalid model id) before it ever ran. Match a real Bedrock id so we
# can forward it untouched and refuse to forward anything else.
_BEDROCK_MODEL_ID = re.compile(
    r"^(arn:aws[\w-]*:bedrock:|(?:[a-z]{2}(?:-[a-z]+)?\.)?anthropic\.)"
)


def is_bedrock_model_id(model: object) -> bool:
    """True when ``model`` is a Bedrock model id / inference-profile ARN.

    Bedrock expects ``anthropic.claude-...-v1:0``, a region-prefixed inference
    profile (``us.anthropic.claude-...``), or a full ``arn:aws:bedrock:...`` ARN
    — never a short ``claude-opus-4-8`` name.
    """
    if not model:
        return False
    return bool(_BEDROCK_MODEL_ID.match(str(model).strip()))


def resolve_claude_code_model(
    payload: "Optional[dict]" = None,
    *,
    env: "Optional[Any]" = None,
    home: Optional[Path] = None,
) -> "tuple[Optional[str], Optional[str]]":
    """Pick the model id to hand the ``claude`` CLI, Bedrock-aware.

    Returns ``(model, note)``. ``model`` is ``None`` when ``--model`` must be
    omitted so the CLI uses its own ``ANTHROPIC_MODEL`` / configured default;
    ``note`` is a diagnostic to record as evidence, or ``None``.

    Off Bedrock, behavior is unchanged — the requested model or the default. On
    Bedrock we never forward a non-Bedrock short name (precisely what the CLI
    rejects). Precedence: an explicit Bedrock override (``payload.bedrock_model``
    or ``ANTHROPIC_MODEL``) > a requested id already Bedrock-shaped > omit
    ``--model`` with a clear, actionable note.
    """
    payload = payload or {}
    env = env if env is not None else os.environ
    requested = payload.get("model") or DEFAULT_CLAUDE_CODE_MODEL

    from puppetmaster.platform_billing import _claude_bedrock_enabled

    home_path = home if home is not None else Path.home()
    if not _claude_bedrock_enabled(env, home_path):
        return str(requested), None

    override = payload.get("bedrock_model") or env.get("ANTHROPIC_MODEL")
    if override:
        return str(override), None
    if is_bedrock_model_id(requested):
        return str(requested), None
    return (
        None,
        (
            f"CLAUDE_CODE_USE_BEDROCK is on but {str(requested)!r} is not a Bedrock "
            "model id; omitting --model so the CLI uses its configured Bedrock "
            "default. Set ANTHROPIC_MODEL (or payload.bedrock_model) to a Bedrock "
            "inference-profile, e.g. us.anthropic.claude-opus-4-1-20250805-v1:0."
        ),
    )


class ClaudeCodeAdapter:
    name = "claude-code"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = with_report_contract(task.payload.get("prompt") or task.instruction)
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(base_prompt, task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 600))
        executable = task.payload.get("executable") or os.environ.get("CLAUDE_CODE_COMMAND") or "claude"
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.45,
                    evidence=["adapter:claude-code", "status:missing-cli"],
                    payload={
                        "failure": "missing_cli",
                        "message": (
                            "Claude Code CLI was not found. Install it or set "
                            "CLAUDE_CODE_COMMAND / payload.executable."
                        ),
                        "executable": executable,
                    },
                )
            ]

        model_for_cli, model_note = resolve_claude_code_model(task.payload)
        command = build_claude_code_command(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=model_for_cli,
            output_format=task.payload.get("output_format", "json"),
            permission_mode=task.payload.get("permission_mode", "acceptEdits"),
            allowed_tools=task.payload.get("allowed_tools"),
            disallowed_tools=task.payload.get("disallowed_tools"),
            extra_args=task.payload.get("extra_args", []),
        )
        before = git_snapshot(cwd)
        blocked = worktree_guard(task, worker_id, "claude-code", cwd, before)
        if blocked is not None:
            return blocked
        if not task.payload.get("allow_dirty", False) and (
            before["changed_files"] or before["untracked_files"]
        ):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.8,
                    evidence=["adapter:claude-code", "status:dirty-repo"],
                    payload={
                        "failure": "dirty_worktree",
                        "message": (
                            "Claude Code full-edit runs require a clean working tree by default "
                            "so Puppetmaster can attribute resulting diffs correctly. Commit, stash, "
                            "use a worktree, or set payload.allow_dirty=true. For focused edits on a "
                            "dirty tree (docs, tests), use puppetmaster_edit — it edits in place and "
                            "needs no clean tree."
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
        # Stream output to a live sidecar log + heartbeat so a long Claude Code
        # run is visibly alive instead of looking hung behind a flat, silent
        # blocking wait. The streamed runner closes stdin, so the CLI can never
        # wedge waiting on terminal input.
        completed = run_streamed_subprocess(
            command=command,
            env=inject_worker_cli_env(apply_worktree_ports(os.environ.copy(), cwd)),
            task=task,
            sidecar_name="claude_implement",
            timeout_seconds=timeout_seconds,
            cwd=str(cwd),
        )
        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
            after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
            stdout_capture = capture_subprocess_stdout(
                text=stdout,
                task=task,
                sidecar_name="claude_stdout_timeout",
            )
            stderr_capture = capture_subprocess_stdout(
                text=stderr,
                task=task,
                sidecar_name="claude_stderr_timeout",
            )
            artifacts: list[Artifact] = [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="failed",
                    confidence=0.6,
                    evidence=["adapter:claude-code", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": _redacted_tail(stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "live_log": completed.live_log_path,
                        "timeout_seconds": timeout_seconds,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                    },
                )
            ]
            # A timed-out run may have already edited files; surface the partial
            # diff so the work isn't silently stranded in the tree.
            if _should_emit_patch_artifact(before, after):
                artifacts.append(
                    Artifact(
                        job_id=task.job_id,
                        task_id=task.id,
                        type=ArtifactType.PATCH,
                        created_by=worker_id,
                        confidence=0.5,
                        evidence=["adapter:claude-code", f"base:{before['sha']}", "timeout"],
                        payload=build_patch_payload(
                            task=task,
                            before=before,
                            after=after,
                            status="failed",
                            change="Claude Code modified repository files before timing out.",
                            sidecar_name="claude_implement_timeout",
                        ),
                    )
                )
            return artifacts

        after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
        # Claude Code stdout often carries long edit transcripts. Give the
        # head/tail more room than Cursor (12k tail vs 8k) but still spool the
        # full transcript so middle bytes survive.
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="claude_stdout",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="claude_stderr",
        )
        usage = token_usage(
            sdk_usage=sdk_usage_from_stdout(completed.stdout),
            prompt_text=prompt,
            output_text=completed.stdout,
        )
        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="claude-code",
            check=task.instruction,
            result="passed" if completed.returncode == 0 else "failed",
            confidence=0.9 if completed.returncode == 0 else 0.55,
            evidence=(
                [
                    "adapter:claude-code",
                    f"permission_mode:{task.payload.get('permission_mode', 'acceptEdits')}",
                ]
                + (["context:codegraph"] if codegraph_used else [])
                + (["bedrock:model-omitted"] if model_note else [])
            ),
            payload={
                "failure": None if completed.returncode == 0 else classify_claude_code_failure(completed.stderr + completed.stdout),
                "returncode": completed.returncode,
                "stdout": _redacted_tail(completed.stdout, 12000),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "live_log": completed.live_log_path,
                "cwd": str(cwd),
                "permission_mode": task.payload.get("permission_mode", "acceptEdits"),
                **({"bedrock_model_note": model_note} if model_note else {}),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
                **usage,
            },
        )
        artifacts = [verification]
        # Same reporting gap as the cursor implement path: Claude's final
        # message (the `result` field of its --output-format json envelope)
        # carried the worker's report but never became an artifact, so swarm
        # findings and implement reports alike vanished from synthesis.
        if completed.returncode == 0:
            _, result_text = cursor_result_text(completed.stdout)
            artifacts.extend(
                implement_report_artifacts(
                    task, worker_id, result_text, adapter="claude-code"
                )
            )
        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if completed.returncode == 0 else 0.5,
                    evidence=["adapter:claude-code", f"base:{before['sha']}"],
                    payload=build_patch_payload(
                        task=task,
                        before=before,
                        after=after,
                        status="applied" if completed.returncode == 0 else "failed",
                        change="Claude Code modified repository files.",
                        sidecar_name="claude_implement",
                    ),
                )
            )
        return artifacts


def _should_emit_patch_artifact(before: dict, after: dict) -> bool:
    """True when a PATCH can be attributed to the worker run."""
    worker_diff = after.get("worker_diff")
    if worker_diff is not None:
        return bool(str(worker_diff).strip())
    if snapshot_has_diff(before):
        return False
    return bool(str(after.get("diff") or "").strip())


DEFAULT_CODEX_MODEL = "gpt-5.4-mini"


class CodexAdapter:
    """Shells out to the official OpenAI Codex CLI (``codex exec --json``).

    Codex is the closest OpenAI-side analog to the Claude Code CLI: a
    coding-agent loop that reads a prompt, plans, calls tools (file edits,
    shell, search), and produces a final agent message. This adapter mirrors
    :class:`ClaudeCodeAdapter` for subprocess + git-snapshot + sidecar-spool
    semantics, and additionally parses Codex's structured JSONL event stream
    to extract real ``input_tokens`` / ``output_tokens`` from
    ``turn.completed.usage`` — telemetry the ``claude`` CLI does not surface.

    Default execution mode is non-interactive: ``--ephemeral``,
    ``--skip-git-repo-check``, ``approval_policy="never"``, sandbox
    ``workspace-write``. The agent runs in an isolated session, can edit
    files in ``cwd``, and never blocks on approval prompts. Callers can
    downgrade to ``--sandbox read-only`` for review-style tasks via
    ``payload.sandbox`` or opt in to
    ``--dangerously-bypass-approvals-and-sandbox`` for environments that are
    already externally sandboxed.
    """

    name = "codex"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(self._structured_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = with_repo_census(prompt, cwd)
        timeout_seconds = int(task.payload.get("timeout_seconds", 600))
        executable = task.payload.get("executable") or os.environ.get("CODEX_COMMAND") or "codex"
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="codex",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.45,
                    evidence=["adapter:codex", "status:missing-cli"],
                    payload={
                        "failure": "missing_cli",
                        "message": (
                            "Codex CLI was not found. Install it with "
                            "`npm install -g @openai/codex`, then `printenv "
                            "OPENAI_API_KEY | codex login --with-api-key`, or "
                            "set CODEX_COMMAND / payload.executable."
                        ),
                        "executable": executable,
                    },
                )
            ]

        model = str(task.payload.get("model") or DEFAULT_CODEX_MODEL)
        sandbox = str(task.payload.get("sandbox") or "workspace-write")
        approval_policy = str(task.payload.get("approval_policy") or "never")
        bypass = bool(task.payload.get("dangerously_bypass_approvals_and_sandbox", False))
        ephemeral = bool(task.payload.get("ephemeral", True))
        skip_git_repo_check = bool(task.payload.get("skip_git_repo_check", True))

        command = build_codex_exec_command(
            executable=[resolved, *command_base[1:]],
            prompt=prompt,
            model=model,
            cwd=cwd,
            sandbox=sandbox,
            approval_policy=approval_policy,
            ephemeral=ephemeral,
            skip_git_repo_check=skip_git_repo_check,
            dangerously_bypass=bypass,
            extra_args=task.payload.get("extra_args", []),
        )

        before = git_snapshot(cwd)
        # Read-only sandbox can't mutate the worktree, so dirty-tree gating
        # and worktree-boundary checks are unnecessary. For write-capable
        # sandboxes we mirror Claude Code: require a git work tree and a clean
        # tree by default so resulting diffs are attributable to this task.
        write_capable = sandbox != "read-only" or bypass
        if write_capable:
            blocked = worktree_guard(task, worker_id, "codex", cwd, before)
            if blocked is not None:
                return blocked
        if (
            write_capable
            and not task.payload.get("allow_dirty", False)
            and (before["changed_files"] or before["untracked_files"])
        ):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="codex",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.8,
                    evidence=["adapter:codex", "status:dirty-repo"],
                    payload={
                        "failure": "dirty_worktree",
                        "message": (
                            "Codex full-edit runs require a clean working tree by default "
                            "so Puppetmaster can attribute resulting diffs correctly. Commit, "
                            "stash, use a worktree, set payload.allow_dirty=true, or pass "
                            "payload.sandbox='read-only' for review-only tasks. For focused edits "
                            "on a dirty tree (docs, tests), use puppetmaster_edit — it edits in "
                            "place and needs no clean tree."
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

        # Stream output to a live sidecar log + heartbeat so a long `codex exec`
        # run is visibly alive instead of looking hung behind a flat, silent
        # blocking wait — the symptom that made Codex runs read as "stalled".
        # The streamed runner closes stdin, so codex can never wedge on input.
        completed = run_streamed_subprocess(
            command=command,
            env=inject_worker_cli_env(apply_worktree_ports(os.environ.copy(), cwd)),
            task=task,
            sidecar_name="codex_exec",
            timeout_seconds=timeout_seconds,
            cwd=str(cwd),
        )
        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
            after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
            stdout_capture = capture_subprocess_stdout(
                text=stdout,
                task=task,
                sidecar_name="codex_stdout_timeout",
                tail_chars=12000,
            )
            stderr_capture = capture_subprocess_stdout(
                text=stderr,
                task=task,
                sidecar_name="codex_stderr_timeout",
            )
            artifacts: list[Artifact] = [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="codex",
                    check=task.instruction,
                    result="failed",
                    confidence=0.6,
                    evidence=["adapter:codex", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "model": model,
                        "sandbox": sandbox,
                        "approval_policy": approval_policy,
                        "stdout": _redacted_tail(stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "live_log": completed.live_log_path,
                        "timeout_seconds": timeout_seconds,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                    },
                )
            ]
            if _should_emit_patch_artifact(before, after):
                artifacts.append(
                    Artifact(
                        job_id=task.job_id,
                        task_id=task.id,
                        type=ArtifactType.PATCH,
                        created_by=worker_id,
                        confidence=0.5,
                        evidence=["adapter:codex", f"base:{before['sha']}", "timeout"],
                        payload=build_patch_payload(
                            task=task,
                            before=before,
                            after=after,
                            status="failed",
                            change="Codex modified repository files before timing out.",
                            sidecar_name="codex_implement_timeout",
                        ),
                    )
                )
            return artifacts

        after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)

        events = parse_codex_events(completed.stdout)
        usage = next(
            (
                ev.get("usage", {})
                for ev in events
                if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict)
            ),
            {},
        )
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        cached_tokens = int(usage.get("cached_input_tokens") or 0)
        reasoning_tokens = int(usage.get("reasoning_output_tokens") or 0)

        last_message = last_codex_agent_message(events)
        turn_failed = any(ev.get("type") == "turn.failed" for ev in events)
        turn_failure_message = next(
            (
                str((ev.get("error") or {}).get("message") or "")
                for ev in events
                if ev.get("type") == "turn.failed"
            ),
            "",
        )
        thread_id = next(
            (ev.get("thread_id") for ev in events if ev.get("type") == "thread.started"),
            None,
        )

        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="codex_events",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="codex_stderr",
        )
        last_message_capture = capture_subprocess_stdout(
            text=last_message,
            task=task,
            sidecar_name="codex_last_message",
        )

        parsed_artifacts = cursor_result_artifacts(task, worker_id, last_message, adapter="codex")
        process_failed = completed.returncode != 0 or turn_failed
        unstructured = not process_failed and not parsed_artifacts and bool(last_message.strip())
        # A write-capable run (implement-style) normally reports in prose —
        # that's its report, not a degradation. Only read-only runs, whose
        # whole job is structured findings, count as degraded on prose.
        degraded = unstructured and not write_capable

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="codex",
            check=task.instruction,
            result=(
                "failed"
                if process_failed
                else "degraded"
                if degraded
                else "passed"
            ),
            confidence=(
                0.55 if process_failed else 0.65 if degraded else 0.9
            ),
            evidence=(
                [
                    "adapter:codex",
                    f"model:{model}",
                    f"sandbox:{sandbox}",
                    f"approval_policy:{approval_policy}",
                ]
                + (["context:codegraph"] if codegraph_used else [])
                + (["bypass:dangerously-bypass-approvals-and-sandbox"] if bypass else [])
            ),
            payload={
                "returncode": completed.returncode,
                "model": model,
                "sandbox": sandbox,
                "approval_policy": approval_policy,
                "ephemeral": ephemeral,
                "thread_id": thread_id,
                "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "live_log": completed.live_log_path,
                "last_message": _redacted_tail(last_message, _STDOUT_TAIL_CHARS),
                "last_message_capture": last_message_capture,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_in + tokens_out,
                "cached_input_tokens": cached_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "turn_failed": turn_failed,
                "turn_failure_message": turn_failure_message,
                "cwd": str(cwd),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
                "failure": (
                    None
                    if not process_failed
                    else "codex_turn_failed"
                    if turn_failed
                    else classify_codex_failure(
                        completed.stderr + "\n" + completed.stdout + "\n" + turn_failure_message
                    )
                ),
            },
        )
        artifacts: list[Artifact] = [verification]
        if unstructured and write_capable:
            artifacts.extend(
                implement_report_artifacts(task, worker_id, last_message, adapter="codex")
            )
        elif degraded:
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.RISK,
                    created_by=worker_id,
                    confidence=0.85,
                    evidence=["adapter:codex", "result:empty-or-unstructured"],
                    payload={
                        "risk": "Codex call completed without structured Puppetmaster findings.",
                        "mitigation": (
                            "Treat this swarm as degraded; rerun with a stricter prompt or "
                            "inspect the repo directly before implementation."
                        ),
                        "stdout_excerpt": (redact_secrets(last_message) or "")[:_STDOUT_HEAD_CHARS],
                        "last_message_capture": last_message_capture,
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if not process_failed else 0.5,
                    evidence=["adapter:codex", f"base:{before['sha']}"],
                    payload=build_patch_payload(
                        task=task,
                        before=before,
                        after=after,
                        status="applied" if not process_failed else "failed",
                        change="Codex modified repository files.",
                        sidecar_name="codex_implement",
                    ),
                )
            )
        return artifacts

    @staticmethod
    def _structured_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Puppetmaster artifact contract:",
                "When you are finished, emit ONLY a single JSON object as your final agent message "
                "(no prose around it, no markdown fences), in this shape:",
                '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
                "Allowed artifact types:",
                '- finding: requires "claim", "evidence", "confidence".',
                '- risk: requires "risk", "mitigation", "evidence", "confidence".',
                '- decision: requires "decision", "why", "evidence", "confidence".',
                _ARTIFACT_GROUNDING,
                _ARTIFACT_EMPTY_GUIDANCE,
                "You may still use your tools to read files and inspect code along the way; just "
                "make sure the FINAL agent message is the JSON object described above.",
            ]
        )


def build_codex_exec_command(
    *,
    executable: Union[str, list[str]] = "codex",
    prompt: str,
    model: Optional[str] = None,
    cwd: Optional[Path] = None,
    sandbox: str = "workspace-write",
    approval_policy: str = "never",
    ephemeral: bool = True,
    skip_git_repo_check: bool = True,
    dangerously_bypass: bool = False,
    extra_args: object = None,
) -> list[str]:
    command = command_parts(executable)
    command.append("exec")
    command.extend(["--json"])
    command.extend(["-c", f'approval_policy="{approval_policy}"'])
    if sandbox:
        command.extend(["--sandbox", sandbox])
    if dangerously_bypass:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    if ephemeral:
        command.append("--ephemeral")
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if cwd is not None:
        command.extend(["-C", str(cwd)])
    if model:
        command.extend(["-m", str(model)])
    if extra_args:
        command.extend(command_parts(extra_args))
    command.append(prompt)
    return command


def parse_codex_events(stdout: str) -> list[dict[str, Any]]:
    """Parse Codex's ``--json`` event stream from captured stdout.

    Codex CLI mixes a couple of human-readable banner lines into the stream
    on non-TTY stdin (notably "Reading additional input from stdin..." and
    occasional ``ERROR``-tagged warnings from the websocket layer). Skip
    anything that does not start with ``{`` and tolerate JSON decode
    failures so a single malformed line never loses the whole turn.
    """
    events: list[dict[str, Any]] = []
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def last_codex_agent_message(events: list[dict[str, Any]]) -> str:
    """Return the most recent ``item.completed`` of type ``agent_message``.

    Codex emits multiple ``item.completed`` events per turn (tool calls,
    reasoning summaries, the final agent message); we only want the final
    user-visible reply.
    """
    for ev in reversed(events):
        if ev.get("type") != "item.completed":
            continue
        item = ev.get("item") or {}
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agent_message":
            text = item.get("text")
            if text is None:
                continue
            return str(text)
    return ""


def classify_codex_failure(output: str) -> str:
    lowered = (output or "").lower()
    if "not logged in" in lowered or "codex login" in lowered:
        return "not_authenticated"
    if "missing bearer" in lowered or "401" in lowered or "unauthorized" in lowered:
        return "not_authenticated"
    if "rate limit" in lowered or "429" in lowered:
        return "rate_limit"
    if "billing" in lowered or "quota" in lowered or "credit" in lowered:
        return "billing_or_quota"
    if "model_not_found" in lowered:
        return "model_unavailable"
    if "model" in lowered and ("unavailable" in lowered or "not found" in lowered or "invalid" in lowered):
        return "model_unavailable"
    if "command not found" in lowered:
        return "missing_cli"
    if "approval" in lowered and ("denied" in lowered or "rejected" in lowered):
        return "approval_denied"
    if "sandbox" in lowered and ("denied" in lowered or "blocked" in lowered):
        return "sandbox_denied"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "network" in lowered or "dns" in lowered or "connect" in lowered:
        return "network_error"
    return "unknown"


# Hermes toolsets are passed verbatim to ``hermes chat -t`` and select the
# tools a worker may call. They deliberately EXCLUDE the ``memory`` and
# ``session_search`` toolsets: both read state from prior Hermes sessions and
# would let one Puppetmaster task observe another (or the user's interactive
# history), breaking swarm isolation. The stock ``coding`` alias bundles both,
# so we enumerate granular toolsets instead. Memory injection into the system
# prompt is separately suppressed via ``--ignore-rules`` (see
# ``build_hermes_chat_command``); together they make each worker hermetic.
DEFAULT_HERMES_ANALYZE_TOOLSETS = "file,web,vision"
DEFAULT_HERMES_IMPLEMENT_TOOLSETS = "file,terminal,code_execution,web,vision"
# Reasoning-effort levels Hermes accepts for ``agent.reasoning_effort`` and maps
# to each provider's native reasoning knob (OpenAI ``reasoning_effort``,
# Anthropic thinking budget, Gemini thinking). Mirrors Hermes'
# ``hermes_constants.VALID_REASONING_EFFORTS`` exactly — Hermes' own parser
# rejects anything outside this set (including ``"none"``, which silently falls
# back to the medium default), so the router and adapter use the same vocabulary.
VALID_HERMES_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_HERMES_ENV_CREDENTIAL_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

# Map each Hermes provider that appears in Puppetmaster's curated Hermes catalog
# (see ``static_catalog.CURATED_CATALOGS["hermes"]``) to the API-key env vars
# that make it callable. The router stamps these provider names via
# ``payload_defaults.provider``; seeding a model whose provider has no
# credential would route a worker to a guaranteed runtime failure, so the
# discover/wizard paths filter against ``available_hermes_providers()``.
_HERMES_PROVIDER_CREDENTIAL_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai-api": ("OPENAI_API_KEY",),
}


def build_hermes_chat_command(
    *,
    executable: Union[str, list[str]] = "hermes",
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_turns: Optional[int] = None,
    toolsets: object = None,
    yolo: bool = False,
    source: str = "tool",
    quiet: bool = True,
    cli: bool = True,
    ignore_rules: bool = True,
    safe_mode: bool = False,
    extra_args: object = None,
) -> list[str]:
    """Build a headless ``hermes chat`` invocation for Puppetmaster workers.

    ``ignore_rules`` defaults to ``True`` so each worker runs hermetically:
    Hermes's auto-injected AGENTS.md/SOUL.md/.cursorrules and — critically —
    its cross-session **memory** tool are skipped. Without this, a fact stored
    by one task ("remember codeword BANANA42") leaks into unrelated later tasks,
    which would corrupt swarm isolation and replayability. Puppetmaster injects
    its own repo context (CodeGraph, report contract, per-task memory), so the
    native Hermes injection is redundant as well as unsafe here.
    """
    command = command_parts(executable)
    command.extend(["chat", "-q", prompt])
    if quiet:
        command.append("-Q")
    command.extend(["--source", source])
    if cli:
        command.append("--cli")
    if ignore_rules:
        command.append("--ignore-rules")
    if safe_mode:
        command.append("--safe-mode")
    if yolo:
        command.append("--yolo")
    if model:
        command.extend(["-m", str(model)])
    if provider:
        command.extend(["--provider", str(provider)])
    if max_turns is not None:
        command.extend(["--max-turns", str(max_turns)])
    if toolsets:
        command.extend(["-t", tool_list(toolsets)])
    if extra_args:
        command.extend(command_parts(extra_args))
    return command


# Hermes worker runs use ``--source tool`` so they're tagged as third-party
# integration sessions. Hermes still PERSISTS every such session to its state DB
# (and the desktop sessions panel lists them regardless of source), so a busy
# swarm leaves a trail of throwaway worker sessions under random worktree cwds.
# After a worker finishes we prune the ended ``source=tool`` sessions via
# Hermes' OWN CLI (``hermes sessions prune``), which deletes cascade-correctly
# (messages + compression_locks + on-disk transcripts) and — critically — only
# touches ENDED sessions, so a sibling worker still running in the same swarm is
# never disturbed. We never touch Hermes' SQLite directly: that would couple us
# to its private schema and risk orphaning rows on a refactor.
_HERMES_SESSION_PRUNE_ENV = "PUPPETMASTER_HERMES_PRUNE_SESSIONS"


def _hermes_session_cleanup_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True unless the user opts out. Worker sessions are pure clutter, so the
    cleanup defaults ON; set ``PUPPETMASTER_HERMES_PRUNE_SESSIONS=0`` to keep
    them (e.g. to debug a worker by resuming its session)."""
    env = env if env is not None else os.environ
    raw = env.get(_HERMES_SESSION_PRUNE_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def prune_hermes_tool_sessions(
    executable: object,
    *,
    source: str = "tool",
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Best-effort prune of ended Hermes worker sessions tagged ``source``.

    Shells out to ``hermes sessions prune --source <source> --older-than 0
    --yes``. Race-safe (Hermes only prunes ended sessions) and never raises —
    session hygiene must never fail a worker run. No-op when the cleanup is
    disabled, the source isn't the worker tag, or the CLI can't be resolved.
    """
    if not _hermes_session_cleanup_enabled(env):
        return
    # Only ever prune the worker tag — never a real user source like ``cli``.
    if source != "tool":
        return
    try:
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return
        subprocess.run(
            [resolved, *command_base[1:], "sessions", "prune",
             "--source", source, "--older-than", "0", "--yes"],
            capture_output=True,
            text=True,
            timeout=30,
            start_new_session=True,
        )
    except Exception:
        # Cleanup is best-effort: a missing CLI, timeout, or any other failure
        # must not affect the worker's result.
        return



@contextlib.contextmanager
def hermes_reasoning_effort_env(base_env: dict, effort: object):
    """Yield a subprocess env that runs ``hermes chat`` at ``effort`` reasoning.

    Hermes has no ``hermes chat`` flag for reasoning effort; the headless source
    of truth is ``agent.reasoning_effort`` in the loaded ``config.yaml`` (read at
    every CLI startup in Hermes' ``cli.py``). The only knob that redirects which
    config file Hermes loads is ``HERMES_HOME``. So to set per-task effort
    *without mutating the user's real ``~/.hermes``* (which would be unsafe for
    parallel swarm workers), we point ``HERMES_HOME`` at an ephemeral home that
    symlinks every entry of the real home — preserving ``auth.json``, ``.env``,
    sessions, and MCP servers verbatim — except ``config.yaml``, which is
    rewritten with ``agent.reasoning_effort: <effort>`` merged in. The temp home
    lives only for the subprocess run and is removed on exit (only symlinks plus
    the one rewritten config are deleted; real state is never touched).

    Degrades to the unmodified ``base_env`` (default effort) when ``effort`` is
    empty/invalid, PyYAML is unavailable (it ships only with the ``hermes``
    extra), or the real home can't be read. A routing knob must never fail a
    worker.
    """
    level = str(effort or "").strip().lower()
    if level not in VALID_HERMES_REASONING_EFFORTS:
        yield base_env
        return
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - hosts without the hermes extra
        yield base_env
        return

    real_home = Path(
        base_env.get("HERMES_HOME")
        or os.environ.get("HERMES_HOME")
        or (Path.home() / ".hermes")
    )
    if not real_home.is_dir():
        yield base_env
        return

    tmp_home = Path(tempfile.mkdtemp(prefix="pm-hermes-effort-"))
    try:
        for entry in real_home.iterdir():
            # config.yaml is rewritten below; sessions/ is deliberately NOT
            # symlinked so an effort-run is hermetic — a symlinked sessions/
            # would write-through to the user's real ~/.hermes/sessions/. It
            # gets its own throwaway empty dir instead.
            if entry.name in ("config.yaml", "sessions"):
                continue
            try:
                os.symlink(entry, tmp_home / entry.name)
            except OSError:
                # A single un-symlinkable entry shouldn't sink the run; the
                # ones that matter (auth.json, .env) are simple files.
                pass

        # Real empty sessions dir so Hermes has somewhere to write session
        # files without touching the user's personal session store.
        try:
            (tmp_home / "sessions").mkdir(exist_ok=True)
        except OSError:
            pass

        config: dict = {}
        cfg_file = real_home / "config.yaml"
        if cfg_file.is_file():
            try:
                loaded = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config = loaded
            except (OSError, yaml.YAMLError):
                config = {}

        agent_cfg = config.get("agent")
        if not isinstance(agent_cfg, dict):
            agent_cfg = {}
        agent_cfg["reasoning_effort"] = level
        config["agent"] = agent_cfg

        effort_config = tmp_home / "config.yaml"
        effort_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        try:
            os.chmod(effort_config, 0o600)
        except OSError:
            pass

        run_env = dict(base_env)
        run_env["HERMES_HOME"] = str(tmp_home)
        yield run_env
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def _hermes_present_credential_keys() -> set:
    """Return the set of known credential env keys Hermes can see.

    Unions the process environment (which the adapter passes through unchanged)
    with any ``KEY=value`` assignments in ``~/.hermes/.env``. Only non-empty
    values count.
    """
    present = {key for key in _HERMES_ENV_CREDENTIAL_KEYS if os.environ.get(key)}
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.is_file():
        try:
            text = env_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for key in _HERMES_ENV_CREDENTIAL_KEYS:
            if re.search(rf"^\s*{re.escape(key)}\s*=\s*\S+", text, re.MULTILINE):
                present.add(key)
    return present


def _hermes_oauth_providers() -> set:
    """Return Hermes provider names with OAuth state in ``~/.hermes/auth.json``."""
    auth_file = Path.home() / ".hermes" / "auth.json"
    if not auth_file.is_file():
        return set()
    try:
        payload = json.loads(auth_file.read_text(encoding="utf-8", errors="replace") or "{}")
    except (OSError, json.JSONDecodeError):
        return set()
    providers = payload.get("providers")
    if isinstance(providers, dict):
        return {str(name).lower() for name in providers}
    return set()


def hermes_credentials_available() -> bool:
    """True when Hermes can likely reach a provider without inlining secrets.

    Checks ``~/.hermes/.env`` for common API keys, OAuth state in
    ``~/.hermes/auth.json``, and keys already present in the process
    environment (which the adapter passes through unchanged).
    """
    return bool(_hermes_present_credential_keys()) or bool(_hermes_oauth_providers())


def available_hermes_providers() -> set:
    """Return the set of Hermes providers that have a usable credential.

    A provider qualifies when an API-key env var that satisfies it is present
    (process env or ``~/.hermes/.env``) or it has OAuth state in
    ``~/.hermes/auth.json``. Used to filter Puppetmaster's curated Hermes
    catalog down to models that can actually be called, so the router never
    picks a Hermes model whose provider is unconfigured.
    """
    present_keys = _hermes_present_credential_keys()
    available = {
        provider
        for provider, keys in _HERMES_PROVIDER_CREDENTIAL_ENV.items()
        if any(key in present_keys for key in keys)
    }
    available |= _hermes_oauth_providers()
    return available


class HermesAdapter:
    """Shells out to the NousResearch Hermes CLI (``hermes chat``).

    Mirrors :class:`CodexAdapter` / :class:`ClaudeCodeAdapter` for subprocess,
    git-snapshot, sidecar-spool, and PATCH attribution semantics. Hermes has
    two operational quirks Puppetmaster must respect:

    - **Process-group isolation**: Hermes kills its own process group on exit.
      Runs always use ``start_new_session=True`` so teardown cannot reach the
      orchestrator parent.
    - **Unreliable exit codes**: A non-zero exit after a successful edit is
      common (provider flakiness, pgroup teardown). Implement-mode success is
      determined from the captured git diff; analyze-mode success is parsed
      from stdout.
    """

    name = "hermes"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        try:
            if task.payload.get("mode") == "implement" or task.payload.get("implement"):
                return self._run_implement(task, goal, worker_id)
            return self._run_analyze(task, goal, worker_id)
        finally:
            # Worker sessions are throwaway (--source tool). Prune the ended ones
            # so they don't pile up in Hermes' session store / desktop panel.
            # In ``finally`` so it runs whether the worker passed, failed, or
            # raised — but it only deletes ENDED sessions, so a sibling worker
            # still running in the same swarm is never touched. Best-effort.
            prune_hermes_tool_sessions(
                task.payload.get("executable")
                or os.environ.get("HERMES_COMMAND")
                or "hermes",
                source=str(task.payload.get("source", "tool")),
            )

    def _run_implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = with_report_contract(task.payload.get("prompt") or task.instruction)
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_skills(
                prompt_with_memory(CursorAdapter._implement_prompt(base_prompt), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 900))
        executable = task.payload.get("executable") or os.environ.get("HERMES_COMMAND") or "hermes"
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="hermes",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.45,
                    evidence=["adapter:hermes", "status:missing-cli"],
                    payload={
                        "failure": "missing_cli",
                        "message": (
                            "Hermes CLI was not found. Install it or set "
                            "HERMES_COMMAND / payload.executable."
                        ),
                        "executable": executable,
                    },
                )
            ]

        command = build_hermes_chat_command(
            executable=[resolved, *command_base[1:]],
            prompt=prompt,
            model=task.payload.get("model"),
            provider=task.payload.get("provider"),
            max_turns=task.payload.get("max_turns"),
            toolsets=task.payload.get("toolsets", DEFAULT_HERMES_IMPLEMENT_TOOLSETS),
            yolo=bool(task.payload.get("yolo", True)),
            source=str(task.payload.get("source", "tool")),
            quiet=bool(task.payload.get("quiet", True)),
            cli=bool(task.payload.get("cli", True)),
            ignore_rules=bool(task.payload.get("ignore_rules", True)),
            safe_mode=bool(task.payload.get("safe_mode", False)),
            extra_args=task.payload.get("extra_args", []),
        )

        before = git_snapshot(cwd)
        blocked = worktree_guard(task, worker_id, "hermes", cwd, before)
        if blocked is not None:
            return blocked
        if not task.payload.get("allow_dirty", False) and (
            before["changed_files"] or before["untracked_files"]
        ):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="hermes",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.8,
                    evidence=["adapter:hermes", "status:dirty-repo"],
                    payload={
                        "failure": "dirty_worktree",
                        "message": (
                            "Hermes full-edit runs require a clean working tree by default "
                            "so Puppetmaster can attribute resulting diffs correctly. Commit, "
                            "stash, use a worktree, or set payload.allow_dirty=true. For focused "
                            "edits on a dirty tree (docs, tests), use puppetmaster_edit — it edits "
                            "in place and needs no clean tree."
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

        # Hermes spawns a foreign Python interpreter; scrub the parent's
        # PYTHONPATH/PYTHONHOME so it can't import Puppetmaster's site-packages
        # and crash on a version clash (e.g. stale python-dotenv).
        worker_env = scrub_foreign_interpreter_env(
            apply_worktree_ports(os.environ.copy(), cwd)
        )
        with hermes_reasoning_effort_env(
            worker_env, task.payload.get("reasoning_effort")
        ) as run_env:
            completed = run_streamed_subprocess(
                command=command,
                env=run_env,
                task=task,
                sidecar_name="hermes_implement",
                timeout_seconds=timeout_seconds,
                cwd=str(cwd),
                start_new_session=True,
            )
        if completed.timed_out:
            after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
            stdout_capture = capture_subprocess_stdout(
                text=completed.stdout,
                task=task,
                sidecar_name="hermes_stdout_timeout",
                tail_chars=12000,
            )
            stderr_capture = capture_subprocess_stdout(
                text=completed.stderr,
                task=task,
                sidecar_name="hermes_stderr_timeout",
            )
            artifacts: list[Artifact] = [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="hermes",
                    check=task.instruction,
                    result="failed",
                    confidence=0.6,
                    evidence=["adapter:hermes", "mode:implement", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "live_log": completed.live_log_path,
                        "timeout_seconds": timeout_seconds,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                    },
                )
            ]
            if _should_emit_patch_artifact(before, after):
                artifacts.append(
                    self._patch_artifact(task, worker_id, before, after, status="failed")
                )
            return artifacts

        after = git_snapshot(cwd, base_tree=str(before.get("tree") or "") or None)
        has_work = _should_emit_patch_artifact(before, after)
        process_failed = completed.returncode != 0 and not has_work
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="hermes_stdout",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="hermes_stderr",
        )
        usage = token_usage(
            prompt_text=prompt,
            output_text=completed.stdout,
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="hermes",
                check=task.instruction,
                result="passed" if not process_failed else "failed",
                confidence=0.9 if not process_failed else 0.55,
                evidence=(
                    ["adapter:hermes", "mode:implement"]
                    + (["context:codegraph"] if codegraph_used else [])
                    + (["exit:ignored-after-diff"] if has_work and completed.returncode != 0 else [])
                ),
                payload={
                    "failure": (
                        None
                        if not process_failed
                        else classify_hermes_failure(completed.stderr + completed.stdout)
                    ),
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, 12000),
                    "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "live_log": completed.live_log_path,
                    "cwd": str(cwd),
                    "model": task.payload.get("model"),
                    "provider": task.payload.get("provider"),
                    "has_work": has_work,
                    "base_sha": before["sha"],
                    "head_sha": after["sha"],
                    "changed_files": after["changed_files"],
                    "untracked_files": after["untracked_files"],
                    **diff_source_payload(before, after),
                    **usage,
                },
            )
        ]
        if not process_failed:
            artifacts.extend(
                implement_report_artifacts(
                    task, worker_id, completed.stdout, adapter="hermes"
                )
            )
        if has_work:
            artifacts.append(
                self._patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    status="applied" if not process_failed else "failed",
                )
            )
        return artifacts

    def _run_analyze(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_skills(
                prompt_with_memory(CodexAdapter._structured_prompt(base_prompt), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = with_repo_census(prompt, cwd)
        timeout_seconds = int(task.payload.get("timeout_seconds", 600))
        executable = task.payload.get("executable") or os.environ.get("HERMES_COMMAND") or "hermes"
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="hermes",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.45,
                    evidence=["adapter:hermes", "status:missing-cli"],
                    payload={
                        "failure": "missing_cli",
                        "message": (
                            "Hermes CLI was not found. Install it or set "
                            "HERMES_COMMAND / payload.executable."
                        ),
                        "executable": executable,
                    },
                )
            ]

        def _invoke_hermes(run_prompt: str, sidecar: str):
            command = build_hermes_chat_command(
                executable=[resolved, *command_base[1:]],
                prompt=run_prompt,
                model=task.payload.get("model"),
                provider=task.payload.get("provider"),
                max_turns=task.payload.get("max_turns"),
                toolsets=task.payload.get("toolsets", DEFAULT_HERMES_ANALYZE_TOOLSETS),
                yolo=False,
                source=str(task.payload.get("source", "tool")),
                quiet=bool(task.payload.get("quiet", True)),
                cli=bool(task.payload.get("cli", True)),
                ignore_rules=bool(task.payload.get("ignore_rules", True)),
                safe_mode=bool(task.payload.get("safe_mode", False)),
                extra_args=task.payload.get("extra_args", []),
            )
            # Hermes spawns a foreign Python interpreter; scrub the parent's
            # PYTHONPATH/PYTHONHOME so it can't import Puppetmaster's
            # site-packages and crash on a version clash (e.g. stale
            # python-dotenv).
            worker_env = scrub_foreign_interpreter_env(
                apply_worktree_ports(os.environ.copy(), cwd)
            )
            with hermes_reasoning_effort_env(
                worker_env, task.payload.get("reasoning_effort")
            ) as run_env:
                return run_streamed_subprocess(
                    command=command,
                    env=run_env,
                    task=task,
                    sidecar_name=sidecar,
                    timeout_seconds=timeout_seconds,
                    cwd=str(cwd),
                    start_new_session=True,
                )

        completed = _invoke_hermes(prompt, "hermes_analyze")
        if completed.timed_out:
            stdout_capture = capture_subprocess_stdout(
                text=completed.stdout,
                task=task,
                sidecar_name="hermes_stdout_timeout",
            )
            stderr_capture = capture_subprocess_stdout(
                text=completed.stderr,
                task=task,
                sidecar_name="hermes_stderr_timeout",
            )
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="hermes",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=["adapter:hermes", "mode:analyze", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "live_log": completed.live_log_path,
                        "timeout_seconds": timeout_seconds,
                        "model": task.payload.get("model"),
                        "provider": task.payload.get("provider"),
                    },
                )
            ]

        def _parse(text_completed) -> list:
            text = text_completed.stdout.strip()
            found = cursor_result_artifacts(task, worker_id, text, adapter="hermes")
            if not found:
                found = cursor_result_artifacts(
                    task, worker_id, text_completed.stdout, adapter="hermes"
                )
            return found

        result_text = completed.stdout.strip()
        parsed_artifacts = _parse(completed)
        has_structured = bool(parsed_artifacts)
        process_failed = completed.returncode != 0 and not has_structured
        degraded = not process_failed and not has_structured

        # One stricter JSON-only reprompt before accepting a degrade: a clean run
        # that returned prose the parser couldn't structure (the minimal-effort
        # flicker) usually recovers on a single retry. Gated by analyze_retry
        # (default on); never retries a process failure or a timeout.
        retry_recovered = False
        retry_attempted = False
        if degraded and bool(task.payload.get("analyze_retry", True)):
            retry_attempted = True
            retry_completed = _invoke_hermes(
                prompt + _ANALYZE_JSON_ONLY_RETRY, "hermes_analyze_retry"
            )
            if not retry_completed.timed_out:
                retry_parsed = _parse(retry_completed)
                if retry_parsed:
                    completed = retry_completed
                    result_text = retry_completed.stdout.strip()
                    parsed_artifacts = retry_parsed
                    has_structured = True
                    process_failed = retry_completed.returncode != 0
                    degraded = False
                    retry_recovered = True
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="hermes_stdout",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="hermes_stderr",
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="hermes",
                check=task.instruction,
                result=(
                    "failed"
                    if process_failed
                    else "degraded"
                    if degraded
                    else "passed"
                ),
                confidence=0.55 if process_failed else 0.65 if degraded else 0.9,
                evidence=(
                    ["adapter:hermes", "mode:analyze"]
                    + (["context:codegraph"] if codegraph_used else [])
                    + (["exit:ignored-after-parse"] if has_structured and completed.returncode != 0 else [])
                    + (["retry:recovered"] if retry_recovered else [])
                    + (["retry:exhausted"] if retry_attempted and not retry_recovered else [])
                ),
                payload={
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, 12000),
                    "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "live_log": completed.live_log_path,
                    "model": task.payload.get("model"),
                    "provider": task.payload.get("provider"),
                    "cwd": str(cwd),
                    "failure": (
                        None
                        if not process_failed and not degraded
                        else (
                            "empty_or_unstructured_hermes_result"
                            if degraded
                            else classify_hermes_failure(completed.stderr + completed.stdout)
                        )
                    ),
                },
            )
        ]
        if degraded:
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.RISK,
                    created_by=worker_id,
                    confidence=0.85,
                    evidence=["adapter:hermes", "result:empty-or-unstructured"],
                    payload={
                        "risk": "Hermes call completed without structured Puppetmaster findings.",
                        "mitigation": (
                            "Treat this swarm as degraded; rerun with a stricter prompt or "
                            "inspect the repo directly before implementation."
                        ),
                        "stdout_excerpt": (redact_secrets(result_text) or "")[:_STDOUT_HEAD_CHARS],
                        "stdout_capture": stdout_capture,
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts

    @staticmethod
    def _patch_artifact(task: Task, worker_id: str, before, after, *, status: str) -> Artifact:
        return Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.PATCH,
            created_by=worker_id,
            confidence=0.8 if status == "applied" else 0.5,
            evidence=["adapter:hermes", f"base:{before['sha']}"],
            payload=build_patch_payload(
                task=task,
                before=before,
                after=after,
                status=status,
                change="Hermes modified repository files.",
                sidecar_name="hermes_implement",
            ),
        )


def classify_hermes_failure(output: str) -> str:
    lowered = (output or "").lower()
    if "command not found" in lowered or (
        "no such file or directory" in lowered and "hermes" in lowered
    ):
        return "missing_cli"
    if (
        "api key" in lowered
        or "not authenticated" in lowered
        or "authentication" in lowered
        or "unauthorized" in lowered
        or "401" in lowered
        or "please login" in lowered
        or "hermes login" in lowered
        or "missing credentials" in lowered
        or "no provider" in lowered
        or "provider credentials" in lowered
    ):
        return "not_authenticated"
    if "verification" in lowered and ("failed" in lowered or "required" in lowered):
        return "not_authenticated"
    if "context length" in lowered or "maximum context" in lowered or "context window" in lowered:
        return "context_length_exceeded"
    if "rate limit" in lowered or "429" in lowered:
        return "rate_limit"
    if "billing" in lowered or "quota" in lowered or "credit" in lowered:
        return "billing_or_quota"
    if "model" in lowered and (
        "unavailable" in lowered
        or "not found" in lowered
        or "invalid" in lowered
        or "does not exist" in lowered
        or "404" in lowered
    ):
        return "model_unavailable"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "network" in lowered or "dns" in lowered or "connect" in lowered:
        return "network_error"
    return "unknown"


class UnconfiguredProviderAdapter:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter=self.name,
                check=task.instruction,
                result="blocked",
                confidence=0.4,
                evidence=[f"adapter:{self.name}", "status:not-configured"],
                payload={
                    "message": (
                        f"{self.description} is a provider stub. Add a concrete adapter "
                        "implementation before using it for live work."
                    )
                },
            )
        ]


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
) -> Artifact:
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.VERIFICATION,
        created_by=worker_id,
        confidence=confidence,
        evidence=evidence,
        payload={
            "adapter": adapter,
            "check": check,
            "result": result,
            **payload,
        },
    )


# Promoted memory is injected into every worker prompt, so verbose or duplicated
# statements (e.g. a prior task's full instruction promoted as a "decision") are
# a per-worker token tax paid on every dispatch. Distill at the injection
# boundary: dedupe identical statements and cap each to a fact-sized snippet.
_MEMORY_MAX_ITEMS = 5
_MEMORY_STATEMENT_MAX_CHARS = 280


def _truncate_statement(statement: str) -> str:
    collapsed = " ".join(statement.split())
    if len(collapsed) <= _MEMORY_STATEMENT_MAX_CHARS:
        return collapsed
    return collapsed[: _MEMORY_STATEMENT_MAX_CHARS - 1].rstrip() + "…"


def _distill_memory_lines(retrieved: list) -> list[str]:
    """Dedupe promoted memory and cap each statement so a handful of verbose
    prior decisions can't balloon every worker prompt with thousands of tokens
    of duplicated instructions. Full statements remain in the memory store; only
    the injected copy is trimmed."""
    lines: list[str] = []
    seen: set[str] = set()
    for memory in retrieved:
        statement = str(memory.get("statement", "")).strip()
        if not statement:
            continue
        key = " ".join(statement.lower().split())
        if key in seen:
            continue
        seen.add(key)
        scope = memory.get("scope", "memory")
        lines.append(f"- [{scope}] {_truncate_statement(statement)}")
        if len(lines) >= _MEMORY_MAX_ITEMS:
            break
    return lines


def prompt_with_memory(prompt: str, task: Task) -> str:
    retrieved = task.payload.get("retrieved_memory") or []
    if not retrieved:
        return prompt
    distilled = _distill_memory_lines(retrieved)
    if not distilled:
        return prompt
    lines = [
        prompt,
        "",
        "Relevant promoted Puppetmaster memory (distilled facts/decisions):",
    ]
    lines.extend(distilled)
    lines.append("")
    lines.append("Use this as retrieved context, but verify claims before relying on them.")
    return "\n".join(lines)


def prompt_with_skills(prompt: str, task: Task) -> str:
    """Append the orchestrator-selected live-skill packet to a worker prompt.

    The mirror image of :func:`prompt_with_memory`: the trusted planner fills
    ``task.payload["injected_skills"]`` (a list of ``{"name", "body"}``) and the
    worker merely renders it. This is the return leg of the puppetmaster-learn
    flywheel (skill -> worker). It injects skill BODIES only — never the
    persona/rules layer, which ``--ignore-rules`` keeps suppressed — so the
    worker's access surface is unchanged. No-op when nothing was injected.
    """
    injected = task.payload.get("injected_skills") or []
    if not injected:
        return prompt
    from puppetmaster.skill_injection import render_skill_packet

    packet = render_skill_packet(injected)
    if not packet:
        return prompt
    return "\n".join([prompt, "", packet])


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


def build_claude_code_command(
    *,
    prompt: str,
    executable: Union[str, list[str]] = "claude",
    model: object = None,
    output_format: str = "json",
    permission_mode: str = "acceptEdits",
    allowed_tools: object = None,
    disallowed_tools: object = None,
    extra_args: object = None,
) -> list[str]:
    command = command_parts(executable)
    command.extend(["--print", prompt, "--output-format", output_format])
    if model:
        command.extend(["--model", str(model)])
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    if allowed_tools:
        command.extend(["--allowedTools", tool_list(allowed_tools)])
    if disallowed_tools:
        command.extend(["--disallowedTools", tool_list(disallowed_tools)])
    if extra_args:
        command.extend(command_parts(extra_args))
    return command


def tool_list(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def git_snapshot(cwd: Path, *, base_tree: Optional[str] = None) -> dict[str, object]:
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
    tree = git_worktree_tree(root) if inside else ""
    worker_diff = ""
    worker_changed: list[str] = []
    if base_tree and tree:
        worker_changed = git_lines(root, ["diff", "--name-only", base_tree, tree, "--"])
        worker_diff = git_diff_output(root, ["diff", "--binary", base_tree, tree, "--"])
    snapshot = {
        "sha": sha or "uncommitted",
        "is_worktree": inside,
        "changed_files": changed,
        "untracked_files": untracked,
        "diff": diff,
    }
    if tree:
        snapshot["tree"] = tree
    if base_tree:
        worker_changed_set = set(worker_changed)
        snapshot["worker_changed_files"] = worker_changed
        snapshot["worker_untracked_files"] = [path for path in untracked if path in worker_changed_set]
        snapshot["worker_diff"] = worker_diff
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
        if sha:
            read = subprocess.run(
                ["git", "read-tree", sha],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            read = subprocess.run(
                ["git", "read-tree", "--empty"],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        if read.returncode != 0:
            return ""
        add = subprocess.run(
            ["git", "add", "-A", "--", "."],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if add.returncode != 0:
            return ""
        written = subprocess.run(
            ["git", "write-tree"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return written.stdout.strip() if written.returncode == 0 else ""
    finally:
        try:
            os.unlink(index_path)
        except OSError:
            pass


def git_output(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_diff_output(cwd: Path, args: list[str]) -> str:
    """Like :func:`git_output` but does not strip — diff bytes are significant
    and a trailing context newline can matter for patch application."""
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else ""


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
        completed = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", os.devnull, rel],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        # 0 == identical (no output), 1 == differs (the diff we want), >1 == error.
        if completed.returncode in (0, 1) and completed.stdout.strip():
            chunks.append(completed.stdout)
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


def sdk_usage_from_stdout(stdout: str) -> Any:
    """Pull a top-level SDK ``usage`` object out of JSON stdout, if any.

    Shared by the Cursor and Claude Code adapters, whose runners both emit a
    single JSON result object carrying ``usage``. Returns ``None`` when stdout
    isn't JSON or has no usage, so the caller falls back to an approximation."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        return payload.get("usage")
    return None


def cursor_result_text(stdout: str) -> tuple[Optional[str], str]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None, stdout.strip()
    if not isinstance(payload, dict):
        return None, stdout.strip()
    if "result" not in payload and (
        "artifacts" in payload or "findings" in payload or "risks" in payload or "decisions" in payload
    ):
        return None, stdout.strip()
    result = payload.get("result")
    return (
        str(payload.get("status")) if payload.get("status") is not None else None,
        str(result).strip() if result is not None else "",
    )


_REPORT_TAIL_CHARS = 20000

_IMPLEMENT_REPORT_CONTRACT = (
    "Reporting contract: when you are done, end your final message with a short "
    "report — what you changed and why, the files you touched, and exactly what "
    "you ran to verify it. Puppetmaster persists that report as a durable "
    "artifact; without it the run looks like it did nothing."
)


def with_report_contract(prompt: str) -> str:
    """Append the implement reporting contract unless the prompt already
    carries a structured artifact contract (swarm review/plan prompts do)."""
    if "Puppetmaster artifact contract" in prompt or _IMPLEMENT_REPORT_CONTRACT in prompt:
        return prompt
    return f"{prompt}\n\n{_IMPLEMENT_REPORT_CONTRACT}"


def implement_report_artifacts(
    task: Task,
    worker_id: str,
    result_text: str,
    *,
    adapter: str,
) -> list[Artifact]:
    """Turn a full-edit worker's final message into durable artifacts.

    Field report (the v0.9.40 CI fix): a cursor implement worker correctly
    diagnosed both failures and shipped the right patch, but its final report
    lived only in a stdout tail — the stitched summary showed "Findings: None"
    and a perfectly good run read as degraded. Structured JSON parses into
    typed artifacts; anything else is preserved verbatim as a FINDING so the
    report survives synthesis instead of dying in a log nobody reads.
    """
    parsed = cursor_result_artifacts(task, worker_id, result_text, adapter=adapter)
    if parsed:
        return parsed
    report = redact_secrets(result_text or "").strip()
    if not report:
        return []
    headline = next(
        (line.strip().lstrip("#").strip() for line in report.splitlines() if line.strip()),
        "Worker report",
    )
    return [
        Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.FINDING,
            created_by=worker_id,
            confidence=0.75,
            evidence=[f"adapter:{adapter}", "report:final-message"],
            payload={
                "claim": headline[:300],
                "report": report[-_REPORT_TAIL_CHARS:],
            },
        )
    ]


def cursor_result_artifacts(
    task: Task,
    worker_id: str,
    result_text: str,
    *,
    adapter: str = "cursor-sdk",
) -> list[Artifact]:
    payload = parse_cursor_artifact_payload(result_text)
    if payload is None:
        return []

    raw_artifacts = []
    if isinstance(payload, list):
        raw_artifacts = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("artifacts"), list):
            raw_artifacts = payload["artifacts"]
        else:
            raw_artifacts.extend(_typed_items(payload, "findings", "finding"))
            raw_artifacts.extend(_typed_items(payload, "risks", "risk"))
            raw_artifacts.extend(_typed_items(payload, "decisions", "decision"))

    artifacts = []
    for item in raw_artifacts:
        artifact = cursor_artifact_from_item(task, worker_id, item, adapter=adapter)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def parse_cursor_artifact_payload(result_text: str) -> Optional[Any]:
    text = result_text.strip()
    if not text:
        return None
    for candidate in [text, _strip_json_fence(text), _json_object_slice(text)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Final fallback: a valid JSON object/array followed by a brace-bearing
    # trailer (a Hermes -Q "[session abc | 1,240 tokens | $0.003]" footer, a
    # courtesy "Done {ok}", or a second JSON blob) makes every greedy
    # candidate above fail — the slice over-reaches into the trailer and
    # json.loads raises. raw_decode parses one complete value from the first
    # opener and ignores trailing junk, recovering the real payload without
    # salvaging genuine prose (which has no parseable opener → still None).
    return _json_prefix_decode(text)


def cursor_artifact_from_item(
    task: Task,
    worker_id: str,
    item: object,
    *,
    adapter: str = "cursor-sdk",
) -> Optional[Artifact]:
    if not isinstance(item, dict):
        return None
    artifact_type = str(item.get("type") or "").lower().strip()
    if artifact_type in {"findings", "swarm.finding"}:
        artifact_type = "finding"
    if artifact_type not in {"finding", "risk", "decision"}:
        return None

    evidence = _string_list(item.get("evidence")) or [f"adapter:{adapter}"]
    confidence = _confidence(item.get("confidence"))
    payload = {key: value for key, value in item.items() if key not in {"type", "evidence", "confidence"}}

    if artifact_type == "finding":
        claim = payload.get("claim") or payload.get("finding") or payload.get("summary")
        if not claim:
            return None
        payload["claim"] = str(claim)
        kind = ArtifactType.FINDING
    elif artifact_type == "risk":
        risk = payload.get("risk") or payload.get("claim") or payload.get("summary")
        if not risk:
            return None
        payload["risk"] = str(risk)
        payload["mitigation"] = str(payload.get("mitigation") or "Review and verify before implementation.")
        kind = ArtifactType.RISK
    else:
        decision = payload.get("decision") or payload.get("claim") or payload.get("summary")
        if not decision:
            return None
        payload["decision"] = str(decision)
        payload["why"] = str(payload.get("why") or "Recommended by automated analysis.")
        kind = ArtifactType.DECISION

    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=kind,
        created_by=worker_id,
        confidence=confidence,
        evidence=evidence,
        payload=payload,
    )


def cursor_degraded_artifact(
    task: Task,
    worker_id: str,
    result_text: str,
    *,
    adapter: str = "cursor-sdk",
    stdout_capture: Optional[dict[str, Any]] = None,
) -> Artifact:
    payload: dict[str, Any] = {
        "risk": f"{adapter} completed without structured Puppetmaster findings.",
        "mitigation": "Treat this swarm as degraded; rerun with a stricter prompt or inspect the repo directly before implementation.",
        "stdout_excerpt": result_text[:_STDOUT_HEAD_CHARS],
    }
    if stdout_capture is None:
        stdout_capture = capture_subprocess_stdout(
            text=result_text,
            task=task,
            sidecar_name="cursor_stdout",
        )
    payload["stdout_capture"] = stdout_capture
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.RISK,
        created_by=worker_id,
        confidence=0.85,
        evidence=[f"adapter:{adapter}", "cursor-result:empty-or-unstructured"],
        payload=payload,
    )


def _typed_items(payload: dict[str, Any], key: str, artifact_type: str) -> list[dict[str, Any]]:
    items = payload.get(key)
    if not isinstance(items, list):
        return []
    return [{**item, "type": item.get("type", artifact_type)} for item in items if isinstance(item, dict)]


def _strip_json_fence(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    lines = stripped.splitlines()
    if len(lines) < 3:
        return None
    return "\n".join(lines[1:-1]).strip()


def _json_object_slice(text: str) -> Optional[str]:
    starts = [index for index in [text.find("{"), text.find("[")] if index >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else None


def _json_prefix_decode(text: str) -> Optional[Any]:
    """Decode the first complete JSON object/array, ignoring any trailing junk.

    Scans for each ``{``/``[`` opener and lets ``raw_decode`` consume exactly
    one JSON value from there, so a valid payload followed by a non-JSON trailer
    still parses. Only structured containers count as artifact payloads — a bare
    scalar that happens to sit after a brace (or genuine prose with no parseable
    opener) yields ``None`` so real degrades aren't masked.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            decoded, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, (dict, list)):
            return decoded
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _confidence(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.75
    return min(1.0, max(0.0, parsed))


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAIAdapter:
    """Calls the OpenAI Chat Completions API directly via OPENAI_API_KEY.

    Mirrors CursorAdapter semantics: enrich the prompt with CodeGraph + promoted
    memory, demand a strict JSON artifact contract, then parse and emit the same
    finding/risk/decision artifacts that the rest of Puppetmaster reasons over.
    """

    name = "openai"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model") or DEFAULT_OPENAI_MODEL
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(self._structured_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = with_repo_census(prompt, cwd)

        api_key = task.payload.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = (
            task.payload.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        organization = task.payload.get("openai_organization") or os.environ.get(
            "OPENAI_ORG_ID"
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 300))
        evidence_base = [
            "adapter:openai",
            f"model:{model}",
        ] + (["context:codegraph"] if codegraph_used else [])

        if not api_key:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.5,
                    evidence=evidence_base + ["missing_api_key"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "missing_api_key",
                        "stderr": (
                            "OPENAI_API_KEY is not set. Export it or pass openai_api_key "
                            "in the task payload."
                        ),
                    },
                )
            ]

        base_url_error = validate_openai_base_url_for_task(base_url, task)
        if base_url_error is not None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.5,
                    evidence=evidence_base + ["untrusted_base_url"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "untrusted_base_url",
                        "stderr": base_url_error,
                    },
                )
            ]

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        # Optional knobs — only included when the caller explicitly opts in.
        # OpenAI's GPT-5+ family deprecated `max_tokens` in favor of
        # `max_completion_tokens`. We send the new name by default and let
        # the caller force the legacy name for OpenAI-compatible providers
        # that still require it.
        max_completion = task.payload.get(
            "max_output_tokens",
            task.payload.get("max_completion_tokens", task.payload.get("max_tokens")),
        )
        if max_completion is not None:
            if task.payload.get("legacy_max_tokens"):
                body["max_tokens"] = int(max_completion)
            else:
                body["max_completion_tokens"] = int(max_completion)
        if "temperature" in task.payload:
            body["temperature"] = float(task.payload["temperature"])
        if task.payload.get("reasoning_effort"):
            body["reasoning_effort"] = str(task.payload["reasoning_effort"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "puppetmaster-openai-adapter",
        }
        if organization:
            headers["OpenAI-Organization"] = str(organization)

        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # 4xx/5xx
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + [f"http_status:{exc.code}"],
                    payload={
                        "returncode": exc.code,
                        "model": model,
                        "failure": classify_openai_failure(err_body, exc.code),
                        "stderr": _redacted_tail(err_body, 8000),
                    },
                )
            ]
        except (socket.timeout, TimeoutError):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["timeout"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "timeout",
                        "stderr": f"OpenAI request exceeded {timeout_seconds}s",
                    },
                )
            ]
        except urllib.error.URLError as exc:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["network_error"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "network_error",
                        "stderr": str(exc),
                    },
                )
            ]

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["malformed_response"],
                    payload={
                        "returncode": status_code,
                        "model": model,
                        "failure": "malformed_response",
                        "stderr": _redacted_tail(raw_body, 8000),
                    },
                )
            ]

        choices = data.get("choices") or []
        message = (
            choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        )
        result_text = str(message.get("content") or "").strip()
        finish_reason = (
            choices[0].get("finish_reason") if isinstance(choices, list) and choices else None
        )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        parsed_artifacts = cursor_result_artifacts(task, worker_id, result_text, adapter="openai")
        degraded = not parsed_artifacts and bool(result_text)
        failed = finish_reason not in (None, "stop", "length") and not parsed_artifacts

        stdout_capture = capture_subprocess_stdout(
            text=result_text,
            task=task,
            sidecar_name="openai_response",
        )

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="openai",
            check=task.instruction,
            result=(
                "failed"
                if failed
                else "degraded"
                if degraded
                else "passed"
            ),
            confidence=0.55 if failed else 0.65 if degraded else 0.9,
            evidence=evidence_base + [f"finish:{finish_reason}"] if finish_reason else evidence_base,
            payload={
                "returncode": status_code,
                "model": model,
                "finish_reason": finish_reason,
                "stdout": result_text[-_STDOUT_TAIL_CHARS:],
                "stderr": "",
                "stdout_capture": stdout_capture,
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
                "tokens_total": total_tokens,
                "failure": (
                    "empty_or_unstructured_openai_result"
                    if degraded
                    else None
                    if not failed
                    else "unexpected_finish_reason"
                ),
            },
        )
        artifacts: list[Artifact] = [verification]
        if degraded:
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.RISK,
                    created_by=worker_id,
                    confidence=0.85,
                    evidence=["adapter:openai", "result:empty-or-unstructured"],
                    payload={
                        "risk": "OpenAI call completed without structured Puppetmaster findings.",
                        "mitigation": (
                            "Treat this swarm as degraded; rerun with a stricter prompt or "
                            "inspect the repo directly before implementation."
                        ),
                        "stdout_excerpt": result_text[:_STDOUT_HEAD_CHARS],
                        "stdout_capture": stdout_capture,
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts

    @staticmethod
    def _structured_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Puppetmaster artifact contract:",
                "Return only JSON, with no markdown wrapper, in this shape:",
                '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
                "Allowed artifact types:",
                '- finding: requires "claim", "evidence", "confidence".',
                '- risk: requires "risk", "mitigation", "evidence", "confidence".',
                '- decision: requires "decision", "why", "evidence", "confidence".',
                _ARTIFACT_GROUNDING,
                _ARTIFACT_EMPTY_GUIDANCE,
            ]
        )


def classify_openai_failure(body: str, http_status: Optional[int] = None) -> str:
    if http_status == 401:
        return "missing_api_key"
    if http_status == 403:
        return "forbidden"
    if http_status == 404:
        return "model_unavailable"
    if http_status == 429:
        return "rate_limit"
    if http_status is not None and 500 <= http_status < 600:
        return "openai_server_error"
    lowered = (body or "").lower()
    if "api key" in lowered or "authorization" in lowered:
        return "missing_api_key"
    if "rate limit" in lowered:
        return "rate_limit"
    if "model_not_found" in lowered:
        return "model_unavailable"
    if "model" in lowered and ("not found" in lowered or "unavailable" in lowered):
        return "model_unavailable"
    if "context length" in lowered or "maximum context" in lowered:
        return "context_length_exceeded"
    return "unknown"


ADAPTERS: dict[str, WorkerAdapter] = {
    "local": LocalAdapter(),
    "shell": ShellAdapter(),
    "cursor": CursorAdapter(),
    "claude-code": ClaudeCodeAdapter(),
    "openai": OpenAIAdapter(),
    "codex": CodexAdapter(),
    "hermes": HermesAdapter(),
}


ADAPTER_INFO = [
    AdapterInfo(
        name="local",
        status="built-in",
        description="Deterministic structured artifacts for demo/runtime roles.",
        requires=[],
    ),
    AdapterInfo(
        name="shell",
        status="built-in",
        description="Runs bounded shell commands and emits verification artifacts.",
        requires=[],
    ),
    AdapterInfo(
        name="cursor",
        status="optional",
        description="Runs local Cursor SDK one-shot agents.",
        requires=["node", "npm install", "CURSOR_API_KEY"],
    ),
    AdapterInfo(
        name="claude-code",
        status="optional",
        description="Runs the Claude Code CLI in non-interactive full-edit mode.",
        requires=["claude CLI", "Claude Code auth"],
    ),
    AdapterInfo(
        name="openai",
        status="optional",
        description="Calls the OpenAI Chat Completions API directly with OPENAI_API_KEY.",
        requires=["OPENAI_API_KEY"],
    ),
    AdapterInfo(
        name="codex",
        status="optional",
        description=(
            "Runs the official OpenAI Codex CLI (`codex exec --json`) "
            "non-interactively. Captures billing-grade token counts from the "
            "structured event stream and emits Puppetmaster artifacts plus a "
            "PATCH artifact when the agent edits files."
        ),
        requires=[
            "codex CLI (`npm install -g @openai/codex`)",
            "OPENAI_API_KEY or `codex login`",
        ],
    ),
    AdapterInfo(
        name="hermes",
        status="optional",
        description=(
            "Runs the NousResearch Hermes CLI (`hermes chat`) headlessly for "
            "analyze and full-edit implement modes. Launches in an isolated "
            "process session and attributes file edits via git diff rather "
            "than exit code."
        ),
        requires=[
            "hermes CLI on PATH",
            "provider credential in ~/.hermes/.env or `hermes login` OAuth",
        ],
    ),
]


def get_adapter(name: str) -> WorkerAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"unsupported adapter: {name}")
    return ADAPTERS[name]


def classify_cursor_failure(output: str) -> str:
    lowered = output.lower()
    if "cursor_api_key" in lowered or "api key" in lowered:
        return "missing_api_key"
    if "cannot find package" in lowered or "@cursor/sdk" in lowered and "not found" in lowered:
        return "sdk_not_installed"
    if (
        "forbidden-model" in lowered
        or ("forbidden" in lowered and "model" in lowered)
        or ("unknown" in lowered and "model" in lowered)
        or (
            "model" in lowered
            and ("unavailable" in lowered or "not found" in lowered or "invalid" in lowered)
        )
    ):
        return "model_unavailable"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "status" in lowered and "error" in lowered:
        return "run_status_error"
    return "unknown"


def classify_claude_code_failure(output: str) -> str:
    lowered = output.lower()
    if "credit balance" in lowered or "billing" in lowered or "quota" in lowered:
        return "billing_or_quota"
    if (
        "not_found_error" in lowered
        or "permission_error" in lowered
        or (
            "model" in lowered
            and ("unavailable" in lowered or "invalid" in lowered or "not found" in lowered)
        )
        or ("permission" in lowered and "model" in lowered)
        or ("not allowed" in lowered and "model" in lowered)
        or ("denied" in lowered and "model" in lowered)
    ):
        return "model_unavailable"
    if "command not found" in lowered:
        return "missing_cli"
    if "auth" in lowered or "login" in lowered or "api key" in lowered:
        return "not_authenticated"
    if "permission" in lowered or "not allowed" in lowered or "denied" in lowered:
        return "permission_denied"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "unknown"
