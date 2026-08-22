from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional, Union

from puppetmaster.failure import classify_antigravity_failure
from puppetmaster.models import Artifact, ArtifactType, Task

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
from .cursor import implement_report_artifacts

DEFAULT_ANTIGRAVITY_MODEL = "gemini-3.7-flash"
DEFAULT_ANTIGRAVITY_EFFORT = "high"
MODELS_REQUIRING_EFFORT = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
)
_EFFORT_SLUG_SUFFIXES = ("-high", "-medium", "-low")


def _model_slug_encodes_effort(model: Optional[str]) -> bool:
    """True when the model slug already ends in an effort suffix."""
    if not model:
        return False
    return str(model).lower().endswith(_EFFORT_SLUG_SUFFIXES)


def resolve_antigravity_model(payload: Optional[Mapping[str, object]] = None) -> tuple[str, Optional[str]]:
    """Determine effective model and reasoning effort for an Antigravity CLI run.

    Slugs that already encode effort (``gemini-3.7-flash-high``) must not also
    receive ``--effort`` — the CLI treats that as a double application.
    """
    payload = payload or {}
    model = str(payload.get("model") or DEFAULT_ANTIGRAVITY_MODEL)
    if _model_slug_encodes_effort(model):
        return model, None
    if not any(prefix in model.lower() for prefix in MODELS_REQUIRING_EFFORT):
        return model, None
    effort = payload.get("effort")
    if effort is not None:
        return model, str(effort)
    return model, DEFAULT_ANTIGRAVITY_EFFORT


def resolve_antigravity_mode(payload: Optional[Mapping[str, object]] = None) -> str:
    """Map CLI/MCP verbs onto agy ``--mode`` values."""
    payload = payload or {}
    raw = str(payload.get("mode") or "").strip().lower()
    read_only_intent = bool(payload.get("read_only")) or (
        payload.get("sandbox") == "read-only"
    )
    if raw in ("plan", "analyze"):
        return "plan"
    if raw in ("accept-edits", "implement"):
        return "accept-edits"
    return "plan" if read_only_intent else "accept-edits"


def resolve_antigravity_skip_permissions(
    payload: Optional[Mapping[str, object]] = None,
    *,
    mode: Optional[str] = None,
) -> bool:
    """Headless accept-edits must skip prompts; plan never does.

    Live ``agy`` 1.1.18 keeps ``toolPermission=request-review`` even after
    ``--mode accept-edits``. Headless then auto-denies ``command`` / write
    tools (``jetski: … permission that headless mode cannot prompt for``).
    Opt out with ``payload.dangerously_skip_permissions=false``.
    """
    payload = payload or {}
    resolved_mode = mode or resolve_antigravity_mode(payload)
    if resolved_mode != "accept-edits":
        return False
    return payload.get("dangerously_skip_permissions") is not False


def antigravity_stdin_data(prompt: str) -> str:
    """One stream-json user event; the prompt never goes on argv."""
    return json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"


def _with_agy_workspace_path(prompt: str, cwd: Path) -> str:
    """agy tool shells start in scratch; pin the git workspace as an abs path."""
    workspace = str(Path(cwd).expanduser().resolve())
    note = (
        "Git workspace (absolute path): {workspace}. "
        "agy list_dir/run_command start in a private scratch directory, not "
        "this workspace. Read and write repository files with write_to_file / "
        "replace_file_content / view_file using that absolute path. "
        "Do not search the whole filesystem."
    ).format(workspace=workspace)
    return note + "\n\n" + prompt


def _workspace_add_dir(cwd: Optional[Path]) -> Optional[str]:
    if cwd is None:
        return None
    try:
        resolved = str(Path(cwd).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not resolved:
        return None
    return resolved


def _unwrap_agy_envelope(parsed: object) -> dict[str, object]:
    """Normalize a json envelope or a stream-json ``result`` event."""
    if not isinstance(parsed, dict):
        return {}
    inner = parsed.get("result")
    if parsed.get("event") == "result" and isinstance(inner, dict):
        return inner
    return parsed


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
        mode = resolve_antigravity_mode(task.payload)
        write_capable = mode != "plan"
        dangerously_skip = resolve_antigravity_skip_permissions(
            task.payload, mode=mode
        )
        timeout_seconds = int(
            task.payload.get("timeout_seconds", self.default_timeout_seconds)
        )
        stdin_data = antigravity_stdin_data(
            _with_agy_workspace_path(prompt, cwd)
        )

        command = build_antigravity_command(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=model,
            mode=mode,
            effort=effort,
            cwd=cwd,
            dangerously_skip_permissions=dangerously_skip,
            timeout_seconds=timeout_seconds,
            extra_args=task.payload.get("extra_args"),
        )
        return CliInvocation(
            command=command,
            sidecar_name="antigravity_run",
            subprocess_kwargs={"stdin_data": stdin_data},
            extras={
                "prompt": prompt,
                "stdin_data": stdin_data,
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
        """Parse a json envelope, NDJSON ``result`` events, or a trailing object."""
        text = (stdout or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            unwrapped = _unwrap_agy_envelope(parsed)
            if unwrapped:
                return unwrapped
        except (json.JSONDecodeError, TypeError):
            pass

        last_result: dict[str, object] = {}
        last_object: dict[str, object] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            last_object = parsed
            if parsed.get("event") == "result":
                last_result = _unwrap_agy_envelope(parsed)
        if last_result:
            return last_result
        if last_object:
            return _unwrap_agy_envelope(last_object)

        start = text.rfind("{")
        if start >= 0:
            try:
                parsed = json.loads(text[start:])
                unwrapped = _unwrap_agy_envelope(parsed)
                if unwrapped:
                    return unwrapped
            except (json.JSONDecodeError, TypeError):
                pass
        return {}


def build_antigravity_command(
    *,
    prompt: str = "",
    executable: Union[str, list[str]] = "agy",
    model: Optional[str] = None,
    mode: str = "accept-edits",
    effort: Optional[str] = None,
    cwd: Optional[Path] = None,
    dangerously_skip_permissions: bool = False,
    disable_slash_commands: bool = True,
    timeout_seconds: Optional[int] = None,
    extra_args: object = None,
) -> list[str]:
    """Build the non-interactive ``agy`` stream-json argv.

    The prompt is **not** part of the command: ``--input-format stream-json``
    reads one ``{"event":"user",...}`` line from stdin (see
    ``antigravity_stdin_data`` / ``CliInvocation.subprocess_kwargs``). Putting
    the prompt on ``-p=`` hits Windows ``CreateProcess`` 32767 / WinError 206.
    Live ``agy`` 1.1.18 accepts ``--add-dir`` as a CLI flag; the adapter passes
    the task workspace once. ``extra_args`` cannot inject additional dirs.
    """
    del prompt
    if mode in ("implement", "analyze"):
        mode = "accept-edits" if mode == "implement" else "plan"
    command = command_parts(executable)
    command.extend(["--input-format", "stream-json"])
    command.extend(["--output-format", "stream-json"])
    if mode in ("accept-edits", "plan"):
        command.extend(["--mode", mode])
    add_dir = _workspace_add_dir(cwd)
    if add_dir:
        command.extend(["--add-dir", add_dir])
    if dangerously_skip_permissions and mode == "accept-edits":
        command.append("--dangerously-skip-permissions")
    if disable_slash_commands:
        command.append("--disable-slash-commands")
    if model:
        command.extend(["--model", str(model)])
    if effort and any(
        prefix in str(model or "").lower() for prefix in MODELS_REQUIRING_EFFORT
    ) and not _model_slug_encodes_effort(model):
        command.extend(["--effort", str(effort)])
    if timeout_seconds is not None:
        command.extend(["--print-timeout", "{}s".format(int(timeout_seconds))])
    if extra_args:
        command.extend(_sanitize_antigravity_extra_args(extra_args))
    return command


_RESERVED_EXTRA_VALUE_FLAGS = frozenset(
    {
        "-p",
        "--print",
        "--mode",
        "--input-format",
        "--output-format",
        "--print-timeout",
        "--add-dir",
    }
)
_RESERVED_EXTRA_FLAG_ONLY = frozenset({"--dangerously-skip-permissions"})
_RESERVED_EXTRA_PREFIXES = (
    "-p=",
    "--print=",
    "--mode=",
    "--input-format=",
    "--output-format=",
    "--print-timeout=",
    "--add-dir=",
)


def _sanitize_antigravity_extra_args(extra_args: object) -> list[str]:
    """Drop extra_args that would rewrite the headless contract or flip mode."""
    parts = command_parts(extra_args)
    safe: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part in _RESERVED_EXTRA_FLAG_ONLY:
            index += 1
            continue
        if part in _RESERVED_EXTRA_VALUE_FLAGS:
            index += 1
            if index < len(parts) and not parts[index].startswith("-"):
                index += 1
            continue
        if any(part.startswith(prefix) for prefix in _RESERVED_EXTRA_PREFIXES):
            index += 1
            continue
        safe.append(part)
        index += 1
    return safe
