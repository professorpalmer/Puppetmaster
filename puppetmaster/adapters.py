from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.models import Artifact, ArtifactType, Task


# Default truncation budgets. Match what the codebase used pre-spool so existing
# verification / risk artifacts keep their inline excerpts; the new sidecar file
# preserves whatever falls in the middle so nothing is silently dropped.
_STDOUT_HEAD_CHARS = 1000
_STDOUT_TAIL_CHARS = 8000


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
    """
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
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / f"{sidecar_name}.log"
        sidecar_path.write_text(text, encoding="utf-8", errors="replace")
        result["stdout_sidecar_path"] = str(sidecar_path)
    except OSError as exc:
        result["stdout_sidecar_path"] = None
        result["stdout_sidecar_error"] = repr(exc)
    return result


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
                        "stdout": (exc.stdout or "")[-4000:],
                        "stderr": (exc.stderr or "")[-4000:],
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
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            )
        ]


class CursorAdapter:
    name = "cursor"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model", "default")
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            self._prompt_with_memory(
                self._structured_prompt(base_prompt),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        runner = Path(__file__).with_name("cursor_sdk_runner.mjs")
        environment = os.environ.copy()
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
        parsed_artifacts = (
            cursor_result_artifacts(task, worker_id, result_text)
            if completed.returncode == 0
            else []
        )
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
                "Use concrete file/function evidence. If there are no concrete findings, return a risk artifact explaining why the run is degraded.",
            ]
        )

    @staticmethod
    def _prompt_with_memory(prompt: str, task: Task) -> str:
        retrieved = task.payload.get("retrieved_memory") or []
        if not retrieved:
            return prompt
        lines = [
            prompt,
            "",
            "Relevant promoted Puppetmaster memory:",
        ]
        for memory in retrieved[:5]:
            lines.append(
                f"- [{memory.get('scope', 'memory')}] {memory.get('statement', '')}"
            )
        lines.append("")
        lines.append("Use this as retrieved context, but verify claims before relying on them.")
        return "\n".join(lines)


DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-4-8"


class ClaudeCodeAdapter:
    name = "claude-code"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
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

        command = build_claude_code_command(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=task.payload.get("model") or DEFAULT_CLAUDE_CODE_MODEL,
            output_format=task.payload.get("output_format", "json"),
            permission_mode=task.payload.get("permission_mode", "acceptEdits"),
            allowed_tools=task.payload.get("allowed_tools"),
            disallowed_tools=task.payload.get("disallowed_tools"),
            extra_args=task.payload.get("extra_args", []),
        )
        before = git_snapshot(cwd)
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
                            "use a worktree, or set payload.allow_dirty=true."
                        ),
                        "changed_files": before["changed_files"],
                        "untracked_files": before["untracked_files"],
                    },
                )
            ]
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                input="",
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
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
            return [
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
                        "stdout": stdout[-_STDOUT_TAIL_CHARS:],
                        "stderr": stderr[-_STDOUT_TAIL_CHARS:],
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]

        after = git_snapshot(cwd)
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
            ),
            payload={
                "failure": None if completed.returncode == 0 else classify_claude_code_failure(completed.stderr + completed.stdout),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-_STDOUT_TAIL_CHARS:],
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "cwd": str(cwd),
                "permission_mode": task.payload.get("permission_mode", "acceptEdits"),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
            },
        )
        artifacts = [verification]
        if after["diff"].strip():
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if completed.returncode == 0 else 0.5,
                    evidence=["adapter:claude-code", f"base:{before['sha']}"],
                    payload={
                        "change": "Claude Code modified tracked repository files.",
                        "files": after["changed_files"],
                        "unified_diff": after["diff"][-20000:],
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "status": "applied" if completed.returncode == 0 else "failed",
                        "revert": "Review the diff, then use git restore / git checkout or your VCS workflow to revert unwanted changes.",
                    },
                )
            )
        return artifacts


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
        # is unnecessary. For write-capable sandboxes we mirror Claude Code:
        # refuse to run on a dirty tree by default so resulting diffs are
        # attributable to this Puppetmaster task, not pre-existing churn.
        if (
            sandbox != "read-only"
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
                            "payload.sandbox='read-only' for review-only tasks."
                        ),
                        "changed_files": before["changed_files"],
                        "untracked_files": before["untracked_files"],
                    },
                )
            ]

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                input="",
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
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
            return [
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
                        "stdout": stdout[-_STDOUT_TAIL_CHARS:],
                        "stderr": stderr[-_STDOUT_TAIL_CHARS:],
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]

        after = git_snapshot(cwd)

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

        parsed_artifacts = cursor_result_artifacts(task, worker_id, last_message)
        process_failed = completed.returncode != 0 or turn_failed
        # "degraded" mirrors OpenAIAdapter / CursorAdapter: the process
        # succeeded but didn't return parseable Puppetmaster artifacts.
        degraded = not process_failed and not parsed_artifacts and bool(last_message.strip())

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
                "stdout": completed.stdout[-_STDOUT_TAIL_CHARS:],
                "stderr": completed.stderr[-_STDOUT_TAIL_CHARS:],
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "last_message": last_message[-_STDOUT_TAIL_CHARS:],
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
        if degraded:
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
                        "stdout_excerpt": last_message[:_STDOUT_HEAD_CHARS],
                        "last_message_capture": last_message_capture,
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        if after["diff"].strip():
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if not process_failed else 0.5,
                    evidence=["adapter:codex", f"base:{before['sha']}"],
                    payload={
                        "change": "Codex modified tracked repository files.",
                        "files": after["changed_files"],
                        "unified_diff": after["diff"][-20000:],
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "status": "applied" if not process_failed else "failed",
                        "revert": (
                            "Review the diff, then use git restore / git checkout or your "
                            "VCS workflow to revert unwanted changes."
                        ),
                    },
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
                "Use concrete file/function evidence. If there are no concrete findings, return a "
                "risk artifact explaining why the run is degraded. You may still use Codex's tools "
                "to read files and edit code along the way; just make sure the FINAL agent message "
                "is the JSON object described above.",
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


def prompt_with_memory(prompt: str, task: Task) -> str:
    retrieved = task.payload.get("retrieved_memory") or []
    if not retrieved:
        return prompt
    lines = [
        prompt,
        "",
        "Relevant promoted Puppetmaster memory:",
    ]
    for memory in retrieved[:5]:
        lines.append(f"- [{memory.get('scope', 'memory')}] {memory.get('statement', '')}")
    lines.append("")
    lines.append("Use this as retrieved context, but verify claims before relying on them.")
    return "\n".join(lines)


def command_parts(command: object) -> list[str]:
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return command
    if isinstance(command, str):
        return shlex.split(command)
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


def git_snapshot(cwd: Path) -> dict[str, object]:
    return {
        "sha": git_output(cwd, ["rev-parse", "HEAD"]) or "uncommitted",
        "changed_files": git_lines(cwd, ["diff", "--name-only"]),
        "untracked_files": git_untracked_files(cwd),
        "diff": git_output(cwd, ["diff", "--binary"]) or "",
    }


def git_output(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_lines(cwd: Path, args: list[str]) -> list[str]:
    output = git_output(cwd, args)
    return [line for line in output.splitlines() if line.strip()]


def git_untracked_files(cwd: Path) -> list[str]:
    output = git_output(cwd, ["status", "--short"])
    files = []
    for line in output.splitlines():
        if line.startswith("?? "):
            files.append(line[3:])
    return files


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


def cursor_result_artifacts(task: Task, worker_id: str, result_text: str) -> list[Artifact]:
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
        artifact = cursor_artifact_from_item(task, worker_id, item)
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
    return None


def cursor_artifact_from_item(task: Task, worker_id: str, item: object) -> Optional[Artifact]:
    if not isinstance(item, dict):
        return None
    artifact_type = str(item.get("type") or "").lower().strip()
    if artifact_type in {"findings", "swarm.finding"}:
        artifact_type = "finding"
    if artifact_type not in {"finding", "risk", "decision"}:
        return None

    evidence = _string_list(item.get("evidence")) or ["adapter:cursor-sdk"]
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
        payload["why"] = str(payload.get("why") or "Recommended by Cursor SDK analysis.")
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
    stdout_capture: Optional[dict[str, Any]] = None,
) -> Artifact:
    payload: dict[str, Any] = {
        "risk": "Cursor SDK completed without structured Puppetmaster findings.",
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
        evidence=["adapter:cursor-sdk", "cursor-result:empty-or-unstructured"],
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


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
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
                        "stderr": err_body[-8000:],
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
                        "stderr": raw_body[-8000:],
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

        parsed_artifacts = cursor_result_artifacts(task, worker_id, result_text)
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
                "Use concrete file/function evidence. If there are no concrete findings, return a risk artifact explaining why the run is degraded.",
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
    if "model" in lowered and ("unavailable" in lowered or "not found" in lowered or "invalid" in lowered):
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
    if "command not found" in lowered or "not found" in lowered:
        return "missing_cli"
    if "auth" in lowered or "login" in lowered or "api key" in lowered:
        return "not_authenticated"
    if "permission" in lowered or "not allowed" in lowered or "denied" in lowered:
        return "permission_denied"
    if "model" in lowered and ("unavailable" in lowered or "invalid" in lowered or "not found" in lowered):
        return "model_unavailable"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "unknown"

