from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.failure import claude_code_diagnosis, classify_claude_code_failure, json_output_diagnostic
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.usage import token_usage
from puppetmaster.worker_attribution import (
    attribution_payload,
    claude_tool_use_references,
)
from puppetmaster.worker_resume import resolved_resume, task_resume_record

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
    build_cli_review_prompt,
    prompt_with_memory,
    wants_review_contract,
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

DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-5"


_BEDROCK_MODEL_ID = re.compile(
    r"^(?:"
    r"arn:aws[\w-]*:bedrock:"
    # Foundation / Marketplace ids: provider.model (amazon., deepseek., zai., …)
    r"|(?:amazon|anthropic|cohere|deepseek|meta|minimax|mistral|moonshot|"
    r"moonshotai|openai|qwen|zai)\."
    # Cross-region inference profiles: us.anthropic.… / eu.meta.…
    r"|(?:[a-z]{2}(?:-[a-z0-9]+)*)\.[a-z0-9][a-z0-9.-]*\."
    r")"
)


def is_bedrock_model_id(model: object) -> bool:
    """True when ``model`` is a Bedrock foundation / inference-profile id or ARN.

    Accepts any provider-shaped id (``amazon.nova-…``, ``deepseek.v3.2``,
    ``zai.glm-5``, ``us.anthropic.claude-…``) and ``arn:aws:bedrock:…`` ARNs.
    Rejects short Claude Code names like ``claude-opus-4-8`` (no provider dot)
    and OpenRouter-style ``org/model`` slugs.
    """
    if not model:
        return False
    text = str(model).strip()
    if text.startswith("arn:aws") and ":bedrock:" in text:
        return True
    if "/" in text:
        return False
    return bool(_BEDROCK_MODEL_ID.match(text))


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
        resume_record = task_resume_record(task.payload, "claude-code")
        resume = resolved_resume(resume_record, "claude-code")
        if resume is not None and task.payload.get("resume_prompt"):
            # The resumed session already holds the original task; send only what changed.
            raw_instruction = str(task.payload["resume_prompt"])
        if wants_review_contract(task.payload):
            base_prompt = build_cli_review_prompt(task, prompt=raw_instruction)
        else:
            base_prompt = with_report_contract(
                f"{TASK_INSTRUCTION_HEADER}\n{raw_instruction}"
            )
        disable_codegraph = bool(task.payload.get("disable_codegraph", False))
        # A builder gets task-scoped CodeGraph context; the job-wide goal brief
        # only dilutes it.
        shared_brief = disable_codegraph or not _claude_write_capable(task.payload)
        if resume is not None:
            # The resumed session already holds memory and CodeGraph context.
            prompt, codegraph_used = with_job_brief(base_prompt, task, shared_brief=shared_brief), False
        else:
            prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
                with_job_brief(prompt_with_memory(base_prompt, task), task, shared_brief=shared_brief),
                task_description=task.payload.get("codegraph_task") or task.instruction or goal,
                cwd=cwd,
                disabled=disable_codegraph,
            )
        executable = (
            task.payload.get("executable")
            or os.environ.get("CLAUDE_CODE_COMMAND")
            or "claude"
        )
        command_base = command_parts(executable)
        model_for_cli, model_note = resolve_claude_code_model(task.payload)
        effective_permission_mode = _claude_permission_mode(task.payload)
        write_capable = effective_permission_mode != "plan"
        command_kwargs: dict[str, Any] = {}
        if resume is not None:
            command_kwargs["resume_session_id"] = str(resume["session_id"])
        command = facade("build_claude_code_command")(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=model_for_cli,
            output_format=task.payload.get("output_format", "json"),
            permission_mode=cli_permission_mode(effective_permission_mode),
            allowed_tools=implement_allowed_tools(task.payload, write_capable=write_capable),
            disallowed_tools=read_only_disallowed_tools(task.payload, write_capable=write_capable),
            extra_args=task.payload.get("extra_args", []),
            **command_kwargs,
        )
        return CliInvocation(
            command=command,
            sidecar_name="claude_implement",
            # The prompt travels on stdin, not argv — see build_claude_code_command.
            subprocess_kwargs={"stdin_data": prompt},
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model_note": model_note,
                "permission_mode": effective_permission_mode,
                "resume": resume_record,
                "resumed": resume is not None,
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

    def _invoke_cli(
        self,
        task: Task,
        prepared: CliInvocation,
        cwd: Path,
        timeout_seconds: int,
    ) -> StreamedProcess:
        result = super()._invoke_cli(task, prepared, cwd, timeout_seconds)
        # A resumed (forked) session's result reports the session's cost to
        # date; its usage counts are this run's own.
        result.session_cumulative_cost = bool(prepared.extras.get("resumed"))
        return result

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
        resume_record = prepared.extras.get("resume")
        resume_fields = {"resume": resume_record} if resume_record else {}
        resumed = bool(prepared.extras.get("resumed"))
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        timeout_seconds = int(
            task.payload.get("timeout_seconds", self.default_timeout_seconds)
        )
        # Tool-use blocks name the files this run edited and the commands it
        # ran, so the write_scope gate can tell its writes from a concurrent
        # writer's in a shared checkout. The default json output format carries
        # none, and then the gate keeps judging the whole delta.
        attribution = attribution_payload(
            claude_tool_use_references(completed.stdout, cwd), before
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
                    evidence=["adapter:claude-code", "timeout"] + (["context:resumed"] if resumed else []),
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "session_id": claude_session_id_from_stdout(stdout),
                        **resume_fields,
                        "stdout": _redacted_tail(stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "live_log": completed.live_log_path,
                        "attempt_id": getattr(completed, "attempt_id", None),
                        "dispatch_receipt": getattr(completed, "dispatch_receipt", None),
                        "timeout_seconds": timeout_seconds,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                        **attribution,
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
                + (["context:resumed"] if resumed else [])
                + (["bedrock:model-omitted"] if model_note else [])
            ),
            payload={
                "failure": None if completed.returncode == 0 else classify_claude_code_failure(completed.stderr + "\n" + json_output_diagnostic(completed.stdout, claude_code_diagnosis)),
                "returncode": completed.returncode,
                "session_id": claude_session_id_from_stdout(completed.stdout),
                **resume_fields,
                "stdout": _redacted_tail(completed.stdout, 12000),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "live_log": completed.live_log_path,
                "attempt_id": getattr(completed, "attempt_id", None),
                "dispatch_receipt": getattr(completed, "dispatch_receipt", None),
                "cwd": str(cwd),
                "permission_mode": permission_mode,
                **({"bedrock_model_note": model_note} if model_note else {}),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
                **attribution,
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


# A headless full-edit run has nobody to answer a permission prompt, so under
# acceptEdits every shell command without an allow rule is refused: the worker
# could edit but never typecheck, test or commit its own work, and handed back
# unverified, uncommitted diffs. These rules let it verify and commit locally;
# anything else (push, network, rm) stays refused. An explicit allowed_tools
# payload replaces the default.
IMPLEMENT_VERIFY_TOOLS = (
    "Bash(cd:*)", "Bash(ls:*)", "Bash(pwd)",
    "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)", "Bash(git show:*)",
    "Bash(git add:*)", "Bash(git commit:*)", "Bash(git restore:*)", "Bash(git stash list:*)",
    "Bash(npx tsc:*)", "Bash(npx vitest:*)", "Bash(npx eslint:*)", "Bash(npx vite build:*)",
    "Bash(npm test:*)", "Bash(npm run:*)", "Bash(pnpm test:*)", "Bash(pnpm run:*)", "Bash(yarn test:*)",
    "Bash(pytest:*)", "Bash(python -m pytest:*)", "Bash(python3 -m pytest:*)",
    "Bash(python -m unittest:*)", "Bash(python3 -m unittest:*)",
    "Bash(.venv/bin/python -m pytest:*)", "Bash(.venv/bin/python -m unittest:*)",
    "Bash(uv run:*)", "Bash(make test:*)", "Bash(go test:*)", "Bash(go vet:*)",
    "Bash(cargo test:*)", "Bash(cargo check:*)",
)


# Read-only workers do not run in Claude Code's plan mode: plan mode writes the
# worker's report to ~/.claude/plans and may leave stdout with only a pointer,
# losing the report and littering the user's plans folder. ``dontAsk`` with a
# read-only allowlist denies everything else and keeps the report on stdout.
READ_ONLY_CLI_PERMISSION_MODE = "dontAsk"
READ_ONLY_TOOLS = ("Read", "Grep", "Glob")
READ_ONLY_DENIED_TOOLS = ("Edit", "Write", "NotebookEdit")


def cli_permission_mode(permission_mode: str) -> str:
    """The ``--permission-mode`` flag for a Puppetmaster permission mode."""
    return READ_ONLY_CLI_PERMISSION_MODE if permission_mode == "plan" else permission_mode


def implement_allowed_tools(payload: dict, *, write_capable: bool) -> object:
    explicit = payload.get("allowed_tools")
    if explicit is not None:
        return explicit
    return list(IMPLEMENT_VERIFY_TOOLS if write_capable else READ_ONLY_TOOLS)


def read_only_disallowed_tools(payload: dict, *, write_capable: bool) -> object:
    requested = payload.get("disallowed_tools")
    if write_capable:
        return requested
    if isinstance(requested, str):
        names = [name.strip() for name in requested.split(",") if name.strip()]
    else:
        names = [str(name) for name in requested or []]
    return list(dict.fromkeys([*names, *READ_ONLY_DENIED_TOOLS]))


def claude_session_id_from_stdout(stdout: Optional[str]) -> Optional[str]:
    """The provider ``session_id`` from ``--output-format json`` or ``stream-json`` stdout.

    For stream-json the final ``result`` event wins over the ``system`` init
    event, which carries the same id when the run completes.
    """
    text = "" if stdout is None else str(stdout)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        session_id = payload.get("session_id")
        return str(session_id) if session_id else None
    found: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or not event.get("session_id"):
            continue
        if event.get("type") == "result":
            return str(event["session_id"])
        if found is None and event.get("type") == "system":
            found = str(event["session_id"])
    return found


def build_claude_code_command(
    *,
    prompt: Optional[str] = None,
    executable: Union[str, list[str]] = "claude",
    model: object = None,
    output_format: str = "json",
    permission_mode: str = "acceptEdits",
    allowed_tools: object = None,
    disallowed_tools: object = None,
    extra_args: object = None,
    resume_session_id: Optional[str] = None,
) -> list[str]:
    """Build the non-interactive ``claude --print`` argv.

    ``resume_session_id`` continues a prior session with ``--fork-session`` so
    the earlier session record stays intact and this attempt gets its own id.

    ``--print`` with no prompt positional makes the CLI read its prompt from
    stdin, which the caller supplies via
    ``CliInvocation.subprocess_kwargs['stdin_data']``. Keeping the prompt out of
    argv is what makes a large enriched prompt spawnable on Windows, where
    ``CreateProcess`` rejects any command line past 32767 characters with
    ``[WinError 206]``. ``prompt`` is accepted and ignored so this exported
    signature stays source-compatible.
    """
    command = command_parts(executable)
    command.extend(["--print", "--output-format", output_format])
    if resume_session_id:
        command.extend(["--resume", str(resume_session_id), "--fork-session"])
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


def _claude_permission_mode(payload: dict) -> str:
    if "permission_mode" in payload:
        return str(payload["permission_mode"])
    if payload.get("read_only") or payload.get("sandbox") == "read-only":
        return "plan"
    return "acceptEdits"


def _claude_write_capable(payload: dict) -> bool:
    return _claude_permission_mode(payload) != "plan"
