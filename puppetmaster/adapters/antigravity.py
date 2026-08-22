from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.failure import classify_antigravity_failure
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
    cursor_result_artifacts,
    implement_report_artifacts,
)

DEFAULT_ANTIGRAVITY_MODEL = "gemini-3.7-flash"
DEFAULT_ANTIGRAVITY_EFFORT = "high"
MODELS_REQUIRING_EFFORT = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro",
)


def resolve_antigravity_model(payload: Optional[Mapping[str, object]] = None) -> tuple[str, Optional[str]]:
    """Determine effective model and reasoning effort for an Antigravity CLI run."""
    payload = payload or {}
    model = str(payload.get("model") or DEFAULT_ANTIGRAVITY_MODEL)
    effort = payload.get("effort")
    if effort is not None:
        return model, str(effort)
    if any(prefix in model.lower() for prefix in MODELS_REQUIRING_EFFORT):
        return model, DEFAULT_ANTIGRAVITY_EFFORT
    return model, None


class AntigravityAdapter(CliWorkerAdapter):
    """Worker adapter for Google Antigravity CLI (`agy`).

    Executes non-interactive print sessions with structured JSON telemetry,
    supporting both `plan` (read-only audit/analysis) and `accept-edits`
    (full-edit code generation with patch attribution) modes.
    """

    name = "antigravity"
    default_timeout_seconds = 600

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        executable = (
            task.payload.get("executable")
            or os.environ.get("AGY_COMMAND")
            or os.environ.get("ANTIGRAVITY_COMMAND")
            or "agy"
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
            "antigravity",
            executable_label,
            (
                "Antigravity CLI (agy) was not found on PATH. Install it or set "
                "AGY_COMMAND / ANTIGRAVITY_COMMAND / payload.executable."
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
        raw_instruction = str(task.payload.get("prompt") or task.instruction or "")
        base_prompt = with_report_contract(
            f"{TASK_INSTRUCTION_HEADER}\n{raw_instruction}"
        )
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            with_job_brief(prompt_with_memory(base_prompt, task), task),
            task_description=str(task.payload.get("codegraph_task") or task.instruction or goal),
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        executable = (
            task.payload.get("executable")
            or os.environ.get("AGY_COMMAND")
            or os.environ.get("ANTIGRAVITY_COMMAND")
            or "agy"
        )
        command_base = command_parts(executable)
        model, effort = resolve_antigravity_model(task.payload)

        read_only_intent = bool(task.payload.get("read_only")) or (
            task.payload.get("sandbox") == "read-only"
        )
        mode = str(task.payload.get("mode") or ("plan" if read_only_intent else "accept-edits"))
        write_capable = mode != "plan"
        dangerously_skip = bool(task.payload.get("dangerously_skip_permissions", True))

        command = build_antigravity_command(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=model,
            mode=mode,
            effort=effort,
            cwd=cwd,
            dangerously_skip_permissions=dangerously_skip,
            extra_args=task.payload.get("extra_args"),
        )
        return CliInvocation(
            command=command,
            sidecar_name="antigravity_run",
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model": model,
                "effort": effort,
                "mode": mode,
                "write_capable": write_capable,
                "extra_dirty_message": (
                    " For focused edits on a dirty tree, use puppetmaster_edit — it edits "
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
        model = str(prepared.extras.get("model") or DEFAULT_ANTIGRAVITY_MODEL)
        effort = prepared.extras.get("effort")
        mode = str(prepared.extras.get("mode") or "accept-edits")
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        prompt = str(prepared.extras.get("prompt") or "")
        cwd = Path(task.payload.get("cwd") or ".").resolve()

        if completed.timed_out:
            return self._handle_timeout(task, worker_id, before, after, completed, model, mode)

        parsed_json = self._parse_agy_output(completed.stdout)
        response_text = str(parsed_json.get("response") or completed.stdout or "").strip()
        agy_status = str(parsed_json.get("status") or "")
        agy_error = str(parsed_json.get("error") or "")
        conversation_id = str(parsed_json.get("conversation_id") or "")
        usage_data = parsed_json.get("usage") if isinstance(parsed_json.get("usage"), dict) else {}

        tokens_in = int(usage_data.get("input_tokens") or 0)
        tokens_out = int(usage_data.get("output_tokens") or 0)
        reasoning_tokens = int(usage_data.get("thinking_tokens") or 0)
        cached_tokens = int(usage_data.get("cache_read_tokens") or 0)

        process_failed = completed.returncode != 0 or agy_status == "ERROR"
        classified_failure = None
        if process_failed:
            combined_err = f"{completed.stderr}\n{completed.stdout}\n{agy_error}"
            classified_failure = classify_antigravity_failure(combined_err)

        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout, task=task, sidecar_name="antigravity_stdout", tail_chars=12000
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr, task=task, sidecar_name="antigravity_stderr"
        )

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="antigravity",
            check=task.instruction,
            result="failed" if process_failed else "passed",
            confidence=0.55 if process_failed else 0.9,
            evidence=(
                ["adapter:antigravity", f"model:{model}", f"mode:{mode}"]
                + ([f"effort:{effort}"] if effort else [])
                + (["context:codegraph"] if codegraph_used else [])
            ),
            payload={
                "returncode": completed.returncode,
                "model": model,
                "mode": mode,
                "effort": effort,
                "conversation_id": conversation_id,
                "agy_status": agy_status,
                "failure": classified_failure,
                "stdout": _redacted_tail(completed.stdout, 12000),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "live_log": completed.live_log_path,
                "cwd": str(cwd),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_in + tokens_out,
                "cached_input_tokens": cached_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
            },
        )
        artifacts: list[Artifact] = [verification]

        if not process_failed and response_text:
            artifacts.extend(implement_report_artifacts(task, worker_id, response_text, adapter="antigravity"))

        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                make_patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    adapter="antigravity",
                    status="applied" if not process_failed else "failed",
                    change="Antigravity modified repository files.",
                    sidecar_name="antigravity_implement",
                )
            )
        return artifacts

    def _handle_timeout(
        self,
        task: Task,
        worker_id: str,
        before: dict,
        after: dict,
        completed: StreamedProcess,
        model: str,
        mode: str,
    ) -> list[Artifact]:
        timeout_seconds = int(task.payload.get("timeout_seconds", self.default_timeout_seconds))
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout, task=task, sidecar_name="antigravity_stdout_timeout"
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr, task=task, sidecar_name="antigravity_stderr_timeout"
        )
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="antigravity",
                check=task.instruction,
                result="failed",
                confidence=0.6,
                evidence=["adapter:antigravity", "timeout"],
                payload={
                    "failure": "timeout",
                    "returncode": None,
                    "model": model,
                    "mode": mode,
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
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.5,
                    evidence=["adapter:antigravity", f"base:{before['sha']}", "timeout"],
                    payload=build_patch_payload(
                        task=task,
                        before=before,
                        after=after,
                        status="failed",
                        change="Antigravity modified repository files before timing out.",
                        sidecar_name="antigravity_implement_timeout",
                    ),
                )
            )
        return artifacts

    def _parse_agy_output(self, stdout: str) -> dict[str, object]:
        """Extract root JSON object from agy --output-format json output."""
        text = (stdout or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {}


def build_antigravity_command(
    *,
    prompt: str,
    executable: Union[str, list[str]] = "agy",
    model: Optional[str] = None,
    mode: str = "accept-edits",
    effort: Optional[str] = None,
    cwd: Optional[Path] = None,
    dangerously_skip_permissions: bool = True,
    disable_slash_commands: bool = True,
    extra_args: object = None,
) -> list[str]:
    """Build the non-interactive ``agy --output-format json`` argv."""
    command = command_parts(executable)
    command.extend(["--output-format", "json"])
    if mode in ("accept-edits", "plan"):
        command.extend(["--mode", mode])
    if dangerously_skip_permissions and mode == "accept-edits":
        command.append("--dangerously-skip-permissions")
    if disable_slash_commands:
        command.append("--disable-slash-commands")
    if model:
        command.extend(["--model", str(model)])
    if effort:
        command.extend(["--effort", str(effort)])
    if cwd is not None:
        command.extend(["--add-dir", str(cwd)])
    if extra_args:
        command.extend(command_parts(extra_args))
    command.append(f"-p={prompt}")
    return command
