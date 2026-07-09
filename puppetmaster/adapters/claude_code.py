from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.failure import classify_claude_code_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.usage import token_usage

from ._base import (
    CliInvocation,
    CliWorkerAdapter,
    build_patch_payload,
    command_parts,
    diff_source_payload,
    make_patch_artifact,
    missing_cli_artifact,
    tool_list,
    verification_artifact,
)
from ._base import _should_emit_patch_artifact
from ._facade import facade
from ._prompts import (
    TASK_INSTRUCTION_HEADER,
    prompt_with_memory,
    with_job_brief,
    with_report_contract,
)
from ._streaming import (
    StreamedProcess,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
)
from .cursor import (
    cursor_result_text,
    implement_report_artifacts,
    sdk_usage_from_stdout,
)

DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-4-8"


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

    home_path = home if home is not None else facade("Path").home()
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


class ClaudeCodeAdapter(CliWorkerAdapter):
    name = "claude-code"
    default_timeout_seconds = 600

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        executable = (
            task.payload.get("executable")
            or os.environ.get("CLAUDE_CODE_COMMAND")
            or "claude"
        )
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
            "claude-code",
            executable_label,
            (
                "Claude Code CLI was not found. Install it or set "
                "CLAUDE_CODE_COMMAND / payload.executable."
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
        # Marker seam so memory / CodeGraph land before the per-task instruction
        # (static-first prefix caching). Report contract injects before the marker.
        raw_instruction = task.payload.get("prompt") or task.instruction
        base_prompt = with_report_contract(
            f"{TASK_INSTRUCTION_HEADER}\n{raw_instruction}"
        )
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            with_job_brief(prompt_with_memory(base_prompt, task), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        executable = (
            task.payload.get("executable")
            or os.environ.get("CLAUDE_CODE_COMMAND")
            or "claude"
        )
        command_base = command_parts(executable)
        model_for_cli, model_note = resolve_claude_code_model(task.payload)
        read_only_intent = bool(task.payload.get("read_only")) or (
            task.payload.get("sandbox") == "read-only"
        )
        if "permission_mode" in task.payload:
            effective_permission_mode = str(task.payload["permission_mode"])
        elif read_only_intent:
            effective_permission_mode = "plan"
        else:
            effective_permission_mode = "acceptEdits"
        write_capable = effective_permission_mode != "plan"
        command = facade("build_claude_code_command")(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=model_for_cli,
            output_format=task.payload.get("output_format", "json"),
            permission_mode=effective_permission_mode,
            allowed_tools=task.payload.get("allowed_tools"),
            disallowed_tools=task.payload.get("disallowed_tools"),
            extra_args=task.payload.get("extra_args", []),
        )
        return CliInvocation(
            command=command,
            sidecar_name="claude_implement",
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model_note": model_note,
                "permission_mode": effective_permission_mode,
                "write_capable": write_capable,
                "extra_dirty_message": (
                    " For focused edits on a dirty tree (docs, tests), use puppetmaster_edit — it edits "
                    "in place and needs no clean tree."
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
        prompt = str(prepared.extras.get("prompt") or "")
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        model_note = prepared.extras.get("model_note")
        permission_mode = str(
            prepared.extras.get("permission_mode")
            or task.payload.get("permission_mode", "acceptEdits")
        )
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        timeout_seconds = int(
            task.payload.get("timeout_seconds", self.default_timeout_seconds)
        )
        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
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
                    f"permission_mode:{permission_mode}",
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
                "permission_mode": permission_mode,
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

