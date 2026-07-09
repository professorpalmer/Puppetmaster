from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.failure import classify_codex_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.redaction import redact_secrets

from ._base import (
    CliInvocation,
    CliWorkerAdapter,
    build_patch_payload,
    command_parts,
    diff_source_payload,
    make_patch_artifact,
    missing_cli_artifact,
    verification_artifact,
)
from ._git import git_snapshot
from ._facade import facade
from ._prompts import (
    build_structured_prompt,
    prompt_with_memory,
)
from ._streaming import (
    StreamedProcess,
    _STDOUT_HEAD_CHARS,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
)
from ._base import _should_emit_patch_artifact
from .cursor import (
    cursor_result_artifacts,
    implement_report_artifacts,
)

DEFAULT_CODEX_MODEL = "gpt-5.4-mini"


class CodexAdapter(CliWorkerAdapter):
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
    default_timeout_seconds = 600

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        executable = task.payload.get("executable") or os.environ.get("CODEX_COMMAND") or "codex"
        command_base = command_parts(executable)
        resolved = facade("resolve_command")(command_base[0])
        if resolved is None:
            return str(executable), None
        return str(executable), resolved

    def _missing_cli(
        self, task: Task, worker_id: str, executable_label: str
    ) -> list[Artifact]:
        return missing_cli_artifact(
            task,
            worker_id,
            "codex",
            executable_label,
            (
                "Codex CLI was not found. Install it with "
                "`npm install -g @openai/codex`, then `printenv "
                "OPENAI_API_KEY | codex login --with-api-key`, or "
                "set CODEX_COMMAND / payload.executable."
            ),
        )

    def _prepare_cli_invocation(
        self,
        task: Task,
        goal: str,
        worker_id: str,
        cwd: Path,
        resolved: str,
    ) -> Union[list[Artifact], CliInvocation]:
        base_prompt = task.payload.get("prompt") or task.instruction
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(
                facade("with_repo_census")(
                    build_structured_prompt(base_prompt, final_message_note=True),
                    cwd,
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        executable = task.payload.get("executable") or os.environ.get("CODEX_COMMAND") or "codex"
        command_base = command_parts(executable)
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
        write_capable = sandbox != "read-only" or bypass
        return CliInvocation(
            command=command,
            sidecar_name="codex_exec",
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model": model,
                "sandbox": sandbox,
                "approval_policy": approval_policy,
                "bypass": bypass,
                "ephemeral": ephemeral,
                "write_capable": write_capable,
                "extra_dirty_message": (
                    " Or pass payload.sandbox='read-only' for review-only tasks. For focused edits "
                    "on a dirty tree (docs, tests), use puppetmaster_edit — it edits in place and "
                    "needs no clean tree."
                ),
            },
        )

    def _apply_pre_run_guards(
        self,
        task: Task,
        worker_id: str,
        cwd: Path,
        prepared: CliInvocation,
    ) -> tuple[Optional[list[Artifact]], dict]:
        if not prepared.extras.get("write_capable", True):
            return None, facade("git_snapshot")(cwd)
        return super()._apply_pre_run_guards(task, worker_id, cwd, prepared)

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
        model = str(prepared.extras.get("model") or DEFAULT_CODEX_MODEL)
        sandbox = str(prepared.extras.get("sandbox") or "workspace-write")
        approval_policy = str(prepared.extras.get("approval_policy") or "never")
        bypass = bool(prepared.extras.get("bypass"))
        ephemeral = bool(prepared.extras.get("ephemeral", True))
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        write_capable = bool(prepared.extras.get("write_capable", True))
        timeout_seconds = int(task.payload.get("timeout_seconds", self.default_timeout_seconds))
        cwd = Path(task.payload.get("cwd") or ".").resolve()

        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
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
                make_patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    adapter="codex",
                    status="applied" if not process_failed else "failed",
                    change="Codex modified repository files.",
                    sidecar_name="codex_implement",
                )
            )
        return artifacts


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

