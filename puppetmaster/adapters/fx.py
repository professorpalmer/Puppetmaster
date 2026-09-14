"""Puppetmaster worker adapter for fx, the inline terminal coding agent.

fx (``vercel-labs/fx``) is a native coding agent for the terminal. Its
non-interactive surface is a single command::

    fx ask --json [--auto|--full-access] [--no-save] [--system TEXT] <prompt>

with the prompt accepted on stdin when no prompt argument is given, and exactly
one JSON object emitted on stdout::

    {"output","final_output","exit_code","model","session_id","steps",
     "tool_calls","usage":{"input_tokens","output_tokens"}}

Two fx-specific properties shape this adapter:

- **Structured usage is native.** Unlike ``claude``, fx reports real
  ``input_tokens`` / ``output_tokens`` in its JSON result, so token accounting is
  parsed rather than estimated, and no JSONL event stream needs scanning (fx emits
  one object, not one line per event, unlike ``codex exec --json``).
- **fx owns its own configuration.** There is no ``--model`` flag; the model comes
  from ``FX_MODEL`` or ``~/.fx/settings.json``. The adapter therefore forwards a
  requested model through the environment and records the model fx *reports*
  rather than the one it requested.
- **A worker does not load MCP servers by default.** The worker inherits the
  operator's MCP profile, which inside an fx-hosted PM session includes the
  ``puppetmaster`` MCP server, so an unsuppressed worker could call back into PM
  and start more workers. The adapter sets fx's ``FX_DISABLE_MCP`` for every
  worker unless the caller passes ``payload.allow_mcp: true``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph, inject_worker_cli_env
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.ports import apply_worktree_ports

from ._base import (
    CliInvocation,
    CliWorkerAdapter,
    build_patch_payload,
    command_parts,
    diff_source_payload,
    missing_cli_artifact,
    verification_artifact,
)
from ._base import _should_emit_patch_artifact
from ._facade import facade
from ._prompts import (
    prompt_with_memory,
    structured_prompt_for_task,
    with_job_brief,
)
from ._streaming import (
    StreamedProcess,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
)
from .cursor import cursor_result_artifacts

# fx resolves its own default model from ~/.fx/settings.json. A hardcoded default
# here would silently override that user-owned configuration, so the adapter has
# none: absent payload.model, no FX_MODEL is sent and fx decides.
DEFAULT_FX_MODEL: Optional[str] = None

PERMISSION_MODES = ("auto", "full-access", "ask")

# fx is spawned by Puppetmaster, which may itself be running inside an fx session.
# That is the interesting case (fx delegates to PM swarms) but also the runaway
# case (fx worker calls puppetmaster_start_swarm, spawning more fx workers). The
# depth counter bounds it and names it instead of letting it recurse silently.
WORKER_DEPTH_ENV = "PUPPETMASTER_FX_WORKER_DEPTH"

# fx's own "load no MCP servers" switch (fx >= the --no-mcp change). It travels in
# the worker environment rather than argv on purpose: an env var is ignored by fx
# builds that predate it, while an unknown argv flag would be rejected outright,
# so an older fx keeps working unchanged.
DISABLE_MCP_ENV = "FX_DISABLE_MCP"


def resolve_fx_executable(task: Task) -> object:
    """Requested fx command, before PATH resolution."""
    return task.payload.get("executable") or os.environ.get("FX_COMMAND") or "fx"


def resolve_fx_permission_mode(task: Task) -> str:
    """Map the requested permission posture onto an fx ``ask`` flag.

    Unattended PM workers have no TTY, so fx's interactive ``ask`` posture would
    deadlock. ``auto`` is the default because fx routes unresolved actions through
    its own safety reviewer, which is strictly safer than bypassing policy.
    ``full-access`` is opt-in, never inferred from write capability.
    """
    requested = task.payload.get("permission_mode")
    if requested is None:
        return "full-access" if task.payload.get("full_access") else "auto"
    mode = str(requested)
    if mode not in PERMISSION_MODES:
        raise ValueError(
            f"unsupported fx permission_mode: {mode!r} "
            f"(expected one of {', '.join(PERMISSION_MODES)})"
        )
    return mode


def resolve_fx_worker_depth(env: Optional[dict] = None) -> int:
    """Current fx-worker nesting depth, 0 when this is the outermost run."""
    source = env if env is not None else os.environ
    raw = source.get(WORKER_DEPTH_ENV)
    try:
        depth = int(str(raw).strip()) if raw is not None else 0
    except (TypeError, ValueError):
        return 0
    return max(depth, 0)


def resolve_max_worker_depth(task: Task) -> int:
    raw = task.payload.get("max_worker_depth")
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return max(raw, 0)


def resolve_mcp_disabled(task: Task) -> bool:
    """Whether this worker should load MCP servers.

    Default is to disable them. A PM-spawned fx worker inherits the operator's MCP
    profile, and when PM was itself launched from an fx session that profile
    includes the ``puppetmaster`` MCP server, so the worker could call back into it
    and start more workers. The depth guard bounds that recursion downstream; this
    removes the surface entirely for the worker.

    Set ``payload.allow_mcp: true`` for a worker that legitimately needs MCP tools.
    Any non-boolean value falls back to the default rather than guessing.
    """
    raw = task.payload.get("allow_mcp")
    if isinstance(raw, bool):
        return not raw
    return True


def build_fx_command(
    *,
    executable: list[str],
    permission_mode: str = "auto",
    no_save: bool = True,
    system_prompt: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    extra_args: Optional[list] = None,
) -> list[str]:
    """Build the ``fx ask`` argv.

    The prompt is deliberately **not** included: it travels on stdin so a long
    structured worker prompt never lands in argv, where it would be visible to
    every process on the box and subject to argv length limits.
    """
    command = [*executable, "ask", "--json"]
    if permission_mode == "full-access":
        command.append("--full-access")
    elif permission_mode == "auto":
        command.append("--auto")
    elif permission_mode != "ask":
        raise ValueError(f"unsupported fx permission_mode: {permission_mode!r}")

    if resume_session_id:
        command += ["--resume-id", str(resume_session_id)]
    elif no_save:
        command.append("--no-save")

    if system_prompt:
        command += ["--system", str(system_prompt)]
    for arg in extra_args or []:
        command.append(str(arg))
    return command


def parse_fx_result(stdout: object) -> Optional[dict]:
    """Parse fx's single-object JSON result from stdout.

    fx writes the result object to stdout and keeps operational progress on
    stderr, so the common path is a clean ``json.loads``. A wrapper that prints a
    banner or trailer around the object still parses: the last balanced
    ``{...}`` slice is tried as a fallback.
    """
    text = "" if stdout is None else str(stdout).strip()
    if not text:
        return None
    direct = _load_json_object(text)
    if direct is not None:
        return direct
    sliced = _last_json_object_slice(text)
    if sliced is not None:
        return _load_json_object(sliced)
    return None


def _load_json_object(text: str) -> Optional[dict]:
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _last_json_object_slice(text: str) -> Optional[str]:
    end = text.rfind("}")
    if end == -1:
        return None
    start = text.find("{")
    while start != -1 and start < end:
        return text[start : end + 1]
    return None


def fx_usage_from_result(result: Optional[dict]) -> tuple[int, int]:
    """``(tokens_in, tokens_out)`` from fx's reported usage, or zeroes."""
    if not isinstance(result, dict):
        return 0, 0
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    tokens_in = usage.get("input_tokens")
    tokens_out = usage.get("output_tokens")
    return (
        int(tokens_in) if isinstance(tokens_in, int) and not isinstance(tokens_in, bool) else 0,
        int(tokens_out) if isinstance(tokens_out, int) and not isinstance(tokens_out, bool) else 0,
    )


def fx_report_text(result: Optional[dict], stdout: str) -> str:
    """The worker's final message, preferring ``final_output`` over ``output``."""
    if isinstance(result, dict):
        for key in ("final_output", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


class FxAdapter(CliWorkerAdapter):
    """Shells out to the fx CLI (``fx ask --json``).

    fx is a terminal-native coding agent with its own tool loop, skills, session
    store, and permission reviewer. This adapter runs it non-interactively as a PM
    worker, inherits snapshot, clean-tree, timeout, and output-budget handling
    from :class:`CliWorkerAdapter`, and adds fx-specific translation: flag
    mapping, ``FX_MODEL`` forwarding, the JSON result contract, and a bounded
    nesting guard for the fx-inside-PM-inside-fx case.
    """

    name = "fx"
    default_timeout_seconds = 900
    # fx owns mutable state under ~/.fx/sessions, history.jsonl, and usage.jsonl.
    # Long-lived isolated state that PM can safely sandbox per run does not exist,
    # so isolation is declared "none" rather than inferred.
    state_isolation = "none"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        executable = resolve_fx_executable(task)
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
            self.name,
            executable_label,
            (
                "The fx CLI was not found on PATH. Install it from "
                "https://fx.sh (or your package manager), or point "
                "FX_COMMAND / payload.executable at an fx binary."
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
        depth = resolve_fx_worker_depth()
        if depth > resolve_max_worker_depth(task):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter=self.name,
                    check=task.instruction,
                    result="blocked",
                    confidence=0.9,
                    evidence=[
                        f"adapter:{self.name}",
                        "status:nested-fx-worker",
                        f"depth:{depth}",
                    ],
                    payload={
                        "failure": "nested_fx_worker",
                        "message": (
                            "Refusing to spawn an fx worker from inside an fx "
                            "worker. Puppetmaster was launched by an fx session "
                            f"that was itself spawned by Puppetmaster (depth {depth}), "
                            "so a worker here would recurse. Dispatch from the outer "
                            "session, or raise payload.max_worker_depth deliberately."
                        ),
                        "depth": depth,
                        "max_worker_depth": resolve_max_worker_depth(task),
                    },
                )
            ]

        base_prompt = task.payload.get("prompt") or task.instruction
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(
                facade("with_repo_census")(
                    with_job_brief(
                        structured_prompt_for_task(
                            task,
                            prompt=base_prompt,
                            final_message_note=True,
                        ),
                        task,
                    ),
                    cwd,
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )

        executable = resolve_fx_executable(task)
        command_base = command_parts(executable)
        if not command_base:
            return self._missing_cli(task, worker_id, str(executable))
        command_base = [resolved, *command_base[1:]]

        permission_mode = resolve_fx_permission_mode(task)
        model = task.payload.get("model") or DEFAULT_FX_MODEL
        save_session = bool(task.payload.get("save_session", False))
        resume_session_id = task.payload.get("resume_session_id")
        system_prompt = task.payload.get("system_prompt")
        command = build_fx_command(
            executable=command_base,
            permission_mode=permission_mode,
            no_save=not save_session,
            system_prompt=str(system_prompt) if system_prompt else None,
            resume_session_id=str(resume_session_id) if resume_session_id else None,
            extra_args=task.payload.get("extra_args", []),
        )

        env = inject_worker_cli_env(apply_worktree_ports(os.environ.copy(), cwd))
        env[WORKER_DEPTH_ENV] = str(depth + 1)
        mcp_disabled = resolve_mcp_disabled(task)
        if mcp_disabled:
            env[DISABLE_MCP_ENV] = "1"
        if model:
            env["FX_MODEL"] = str(model)

        return CliInvocation(
            command=command,
            sidecar_name="fx_ask",
            env=env,
            subprocess_kwargs={"stdin_data": prompt},
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model": str(model) if model else None,
                "permission_mode": permission_mode,
                "save_session": save_session,
                "resume_session_id": str(resume_session_id) if resume_session_id else None,
                # fx has no read-only flag on `ask`, so analyze runs are instructed
                # read-only rather than enforced read-only. Recorded honestly.
                "enforcement": "prompt-only",
                "depth": depth + 1,
                "mcp_disabled": mcp_disabled,
            },
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
        model_requested = prepared.extras.get("model")
        permission_mode = str(prepared.extras.get("permission_mode") or "auto")
        resume_session_id = prepared.extras.get("resume_session_id")
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        timeout_seconds = int(
            task.payload.get("timeout_seconds", self.default_timeout_seconds)
        )
        cwd = Path(task.payload.get("cwd") or ".").resolve()

        if completed.timed_out:
            return self._timeout_artifacts(
                task,
                worker_id,
                prepared,
                before,
                after,
                completed,
                model_requested=model_requested,
                permission_mode=permission_mode,
                timeout_seconds=timeout_seconds,
            )

        result = parse_fx_result(completed.stdout)
        tokens_in, tokens_out = fx_usage_from_result(result)
        report = fx_report_text(result, completed.stdout)
        session_id = ""
        reported_model = None
        reported_exit_code: Optional[int] = None
        if isinstance(result, dict):
            raw_session = result.get("session_id")
            session_id = raw_session if isinstance(raw_session, str) else ""
            raw_model = result.get("model")
            reported_model = raw_model if isinstance(raw_model, str) and raw_model else None
            raw_exit = result.get("exit_code")
            if isinstance(raw_exit, int) and not isinstance(raw_exit, bool):
                reported_exit_code = raw_exit

        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="fx_result",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="fx_stderr",
        )
        report_capture = capture_subprocess_stdout(
            text=report,
            task=task,
            sidecar_name="fx_final_output",
        )

        process_failed = completed.returncode != 0 or (
            reported_exit_code is not None and reported_exit_code != 0
        )
        # A write-capable run that produced no typed artifacts is *not* degraded:
        # the PATCH artifact is its real evidence, and prose-only reporting is
        # normal for a coding worker. This mirrors the codex/claude-code path.
        # fx has no enforced read-only mode, so write capability is a property of
        # the permission posture, and `ask` is the only non-writing posture.
        write_capable = permission_mode != "ask"
        missing_result = result is None and not process_failed
        unstructured = (
            not process_failed and not missing_result and bool(report.strip())
        )
        # A run that exits cleanly but emits nothing parseable cannot be
        # attributed, so it is always degraded.
        degraded = bool(missing_result or (unstructured and not write_capable))

        artifacts = cursor_result_artifacts(
            task, worker_id, report, adapter=self.name
        )

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter=self.name,
            check=task.instruction,
            result="failed" if process_failed else "degraded" if degraded else "passed",
            confidence=0.55 if process_failed else 0.65 if degraded else 0.9,
            evidence=(
                [f"adapter:{self.name}", f"permission_mode:{permission_mode}"]
                + ([f"model:{reported_model}"] if reported_model else [])
                + (["context:codegraph"] if codegraph_used else [])
                + ([f"session:{session_id}"] if session_id else [])
            ),
            payload={
                "returncode": completed.returncode,
                "reported_exit_code": reported_exit_code,
                "model_requested": model_requested,
                "model": reported_model,
                "permission_mode": permission_mode,
                "enforcement": prepared.extras.get("enforcement"),
                "save_session": bool(prepared.extras.get("save_session")),
                "resume_session_id": resume_session_id,
                "session_id": session_id or None,
                "worker_depth": prepared.extras.get("depth"),
                "mcp_disabled": bool(prepared.extras.get("mcp_disabled")),
                "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "report_capture": report_capture,
                "live_log": completed.live_log_path,
                "final_output": _redacted_tail(report, _STDOUT_TAIL_CHARS),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_in + tokens_out,
                "usage_reported": result is not None,
                "result_missing": missing_result,
                "cwd": str(cwd),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
                "failure": (
                    "fx_unparseable_result"
                    if missing_result
                    else "fx_exit_code"
                    if process_failed
                    else None
                ),
            },
        )
        artifacts.append(verification)

        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if not process_failed else 0.5,
                    evidence=[
                        f"adapter:{self.name}",
                        f"base:{before['sha']}",
                        *(("failed",) if process_failed else ()),
                    ],
                    payload=build_patch_payload(
                        task=task,
                        before=before,
                        after=after,
                        status="failed" if process_failed else "applied",
                        change=(
                            "fx worker modified repository files before failing."
                            if process_failed
                            else "fx worker modified repository files."
                        ),
                        sidecar_name="fx_implement",
                    ),
                )
            )
        return artifacts

    def _timeout_artifacts(
        self,
        task: Task,
        worker_id: str,
        prepared: CliInvocation,
        before: dict,
        after: dict,
        completed: StreamedProcess,
        *,
        model_requested: Optional[str],
        permission_mode: str,
        timeout_seconds: int,
    ) -> list[Artifact]:
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="fx_result_timeout",
            tail_chars=12000,
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="fx_stderr_timeout",
        )
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter=self.name,
                check=task.instruction,
                result="failed",
                confidence=0.6,
                evidence=[f"adapter:{self.name}", "timeout"],
                payload={
                    "failure": "timeout",
                    "returncode": None,
                    "model_requested": model_requested,
                    "permission_mode": permission_mode,
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
                    evidence=[f"adapter:{self.name}", f"base:{before['sha']}", "timeout"],
                    payload=build_patch_payload(
                        task=task,
                        before=before,
                        after=after,
                        status="failed",
                        change="fx modified repository files before timing out.",
                        sidecar_name="fx_implement_timeout",
                    ),
                )
            )
        return artifacts
