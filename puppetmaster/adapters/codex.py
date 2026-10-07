from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph, inject_worker_cli_env
from puppetmaster.ports import apply_worktree_ports
from puppetmaster.failure import UNKNOWN, classify_codex_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.redaction import redact_secrets
from puppetmaster.usage import selected_token_usage
from puppetmaster.session_lease import SessionLease, acquire_codex_thread
from puppetmaster.worker_attribution import attribution_payload, codex_event_references
from puppetmaster.worker_resume import resolved_resume, task_resume_record

from . import codex_rollout

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
    prompt_with_memory,
    build_cli_implement_prompt,
    build_cli_review_prompt,
    structured_prompt_for_task,
    wants_review_contract,
    with_job_brief,
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

_CODEX_NPM_ENTRYPOINT = Path("node_modules") / "@openai" / "codex" / "bin" / "codex.js"


def _is_windows() -> bool:
    return os.name == "nt"


def _configured_codex_executable(task: Task) -> object:
    return task.payload.get("executable") or os.environ.get("CODEX_COMMAND") or "codex"


def _default_codex_command_base(resolved: str) -> Optional[list[str]]:
    """Resolve the default Codex command without a Windows batch intermediary.

    Global npm installs expose ``codex.CMD`` on Windows. Launching that shim
    inserts ``cmd.exe`` between Puppetmaster and the long-lived Node process,
    weakening signal, pipe, and timeout ownership. Resolve the package's real
    JavaScript entrypoint and invoke it with Node directly instead.
    """
    resolved_path = Path(resolved)
    if not _is_windows() or resolved_path.suffix.lower() not in {".cmd", ".bat"}:
        return [resolved]

    entrypoint = resolved_path.parent / _CODEX_NPM_ENTRYPOINT
    node = facade("resolve_command")("node")
    if node is None or not entrypoint.is_file():
        return None
    return [node, str(entrypoint)]


def _codex_command_base(task: Task, resolved: str) -> Optional[list[str]]:
    """Build the launch prefix while preserving explicit user commands."""
    executable = _configured_codex_executable(task)
    command = command_parts(executable)
    if task.payload.get("executable") or os.environ.get("CODEX_COMMAND"):
        return [resolved, *command[1:]]
    return _default_codex_command_base(resolved)


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
        executable = _configured_codex_executable(task)
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
        resume_record = task_resume_record(task.payload, "codex")
        resume = resolved_resume(resume_record, "codex")
        lease = None
        if resume is not None:
            resume_record, lease = _lease_codex_thread(resume_record)
            resume = resolved_resume(resume_record, "codex")
        if resume is not None and task.payload.get("resume_prompt"):
            # The thread already holds the original task; send only what changed.
            base_prompt = str(task.payload["resume_prompt"])
        sandbox = str(task.payload.get("sandbox") or "workspace-write")
        bypass = bool(task.payload.get("dangerously_bypass_approvals_and_sandbox", False))
        write_capable = sandbox != "read-only" or bypass
        review_mode = not write_capable and wants_review_contract(task.payload)
        disable_codegraph = bool(task.payload.get("disable_codegraph", False))
        if write_capable:
            # A builder gets task-scoped CodeGraph context; the job-wide census
            # and goal brief only dilute it.
            task_prompt = with_job_brief(
                build_cli_implement_prompt(task, prompt=base_prompt),
                task,
                shared_brief=disable_codegraph,
            )
        elif review_mode:
            task_prompt = with_job_brief(build_cli_review_prompt(task, prompt=base_prompt), task)
        else:
            task_prompt = with_job_brief(
                structured_prompt_for_task(
                    task,
                    prompt=base_prompt,
                    final_message_note=True,
                ),
                task,
            )
        if resume is not None:
            # The resumed thread already holds memory, census and CodeGraph context.
            prompt, codegraph_used = task_prompt, False
        else:
            if not write_capable:
                task_prompt = facade("with_repo_census")(task_prompt, cwd)
            prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
                prompt_with_memory(task_prompt, task),
                task_description=task.payload.get("codegraph_task") or task.instruction or goal,
                cwd=cwd,
                disabled=disable_codegraph,
            )
        executable = _configured_codex_executable(task)
        command_base = _codex_command_base(task, resolved)
        if command_base is None:
            if lease is not None:
                lease.release()
            return self._missing_cli(task, worker_id, str(executable))
        # Unpinned: the user's configured Codex model, else Codex's own default.
        model = str(task.payload.get("model") or _codex_home().configured_model())
        approval_policy = str(task.payload.get("approval_policy") or "never")
        # A review-loop task may be repaired by resuming this thread, so keep it.
        ephemeral = bool(task.payload.get("ephemeral", not task.payload.get("review_loop")))
        skip_git_repo_check = bool(task.payload.get("skip_git_repo_check", True))
        if resume is not None:
            ephemeral = False
            command = build_codex_resume_command(
                executable=command_base,
                session_id=str(resume["session_id"]),
                model=model,
                sandbox=sandbox,
                approval_policy=approval_policy,
                skip_git_repo_check=skip_git_repo_check,
                dangerously_bypass=bypass,
                extra_args=task.payload.get("extra_args", []),
            )
        else:
            command = build_codex_exec_command(
                executable=command_base,
                model=model,
                cwd=cwd,
                sandbox=sandbox,
                approval_policy=approval_policy,
                ephemeral=ephemeral,
                skip_git_repo_check=skip_git_repo_check,
                dangerously_bypass=bypass,
                extra_args=task.payload.get("extra_args", []),
            )
        return CliInvocation(
            command=command,
            sidecar_name="codex_exec",
            # The prompt travels on stdin, not argv — see build_codex_exec_command.
            subprocess_kwargs={"stdin_data": prompt},
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model": model,
                "sandbox": sandbox,
                "approval_policy": approval_policy,
                "bypass": bypass,
                "ephemeral": ephemeral,
                "resume": resume_record,
                "resumed": resume is not None,
                "session_lease": lease,
                "write_capable": write_capable,
                # Builders and verdict reviewers answer in free text, not findings JSON.
                "report_mode": write_capable or review_mode,
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

    @staticmethod
    def _worker_home(prepared: CliInvocation) -> Optional[Path]:
        """The CODEX_HOME this run uses, or None for the inherited one (see codex_home)."""
        from puppetmaster import codex_home

        if not codex_home.enabled():
            return None
        resume = prepared.extras.get("resume") if prepared.extras.get("resumed") else None
        resuming = isinstance(resume, dict) and bool(resume.get("session_id"))
        if resuming:
            # A thread resumes in the home that holds it.
            found = codex_home.home_for_session(str(resume["session_id"]))
            home = found if found == codex_home.worker_home_root() else None
        else:
            try:
                home = codex_home.prepare()
            except Exception:
                # Best effort: any failure runs the worker in the user's own home.
                return None
        if home is not None and "exec" in prepared.command:
            try:
                codex_home.ensure_system_skills(home, prepared.command[:prepared.command.index("exec")])
            except Exception:
                pass  # Best effort: stock Codex still installs the bundle itself.
        if home is not None and not resuming and prepared.command[-2:] == list(STDIN_PROMPT):
            # Bounded workers never fork subagents, and must not accrue memories.
            prepared.command = [*prepared.command[:-2], "--disable", "memories",
                                "--disable", "multi_agent", *STDIN_PROMPT]
        return home

    def _invoke_cli(
        self,
        task: Task,
        prepared: CliInvocation,
        cwd: Path,
        timeout_seconds: int,
    ) -> StreamedProcess:
        if prepared.extras.get("resumed") or not bool((task.payload or {}).get("native_steer")):
            home = self._worker_home(prepared)
            from puppetmaster import codex_home

            if home is not None:
                env = dict(prepared.env) if prepared.env is not None else inject_worker_cli_env(
                    apply_worktree_ports(os.environ.copy(), cwd))
                env["CODEX_HOME"] = str(home)
                prepared.env = env
            rollout_home = home if home is not None else codex_home.user_home(prepared.env)
            resume_record = prepared.extras.get("resume") if prepared.extras.get("resumed") else None
            baseline = _rollout_baseline(rollout_home, resume_record)
            try:
                result = super()._invoke_cli(task, prepared, cwd, timeout_seconds)
            finally:
                if home is not None and home == codex_home.worker_home_root():
                    codex_home.sync_back()
            result.attempt_usage = _rollout_attempt_usage(
                rollout_home, resume_record, baseline, result.stdout)
            return result
        try:
            from puppetmaster.adapters.codex_session import run_codex_session
            from puppetmaster.state import resolve_state_dir
            from puppetmaster.steering import native_steer_items
            from puppetmaster.store_factory import create_store

            store = create_store("sqlite", resolve_state_dir())
            pending = lambda: native_steer_items(store, task)
            result = run_codex_session(
                command_prefix=prepared.command[:1],
                cwd=cwd,
                prompt=str(prepared.extras.get("prompt") or task.instruction or ""),
                model=str(prepared.extras.get("model") or _codex_home().configured_model()) or None,
                sandbox=str(prepared.extras.get("sandbox") or "workspace-write"),
                timeout=float(timeout_seconds),
                pending_steering=pending,
            )
            text = "\n".join(result.messages)
            return StreamedProcess(
                returncode=0 if result.status == "completed" else 1,
                stdout=text,
                stderr=result.error or "",
                timed_out=result.status == "timeout",
                spawn_error=None if result.status != "failed" or text else result.error,
            )
        except Exception:
            return super()._invoke_cli(task, prepared, cwd, timeout_seconds)

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
        model = str(prepared.extras.get("model") or "codex-default")
        sandbox = str(prepared.extras.get("sandbox") or "workspace-write")
        approval_policy = str(prepared.extras.get("approval_policy") or "never")
        bypass = bool(prepared.extras.get("bypass"))
        ephemeral = bool(prepared.extras.get("ephemeral", True))
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        write_capable = bool(prepared.extras.get("write_capable", True))
        report_mode = bool(prepared.extras.get("report_mode", write_capable))
        resume_record = prepared.extras.get("resume")
        resume_fields = {"resume": resume_record} if resume_record else {}
        resumed = bool(prepared.extras.get("resumed"))
        timeout_seconds = int(task.payload.get("timeout_seconds", self.default_timeout_seconds))
        cwd = Path(task.payload.get("cwd") or ".").resolve()

        events = parse_codex_events(completed.stdout)
        # Codex names the files it edited and the commands it ran, so the
        # write_scope gate can tell this run's writes from a concurrent
        # writer's in a shared checkout.
        attribution = attribution_payload(codex_event_references(events, cwd), before)

        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
            # A persisted thread survives the timeout; record it so a follow-up
            # can resume it. The resolver still checks the session is on disk.
            timeout_thread_id = observed_thread_id(events, resume_record if resumed else None)
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
                    evidence=["adapter:codex", "timeout"] + (["context:resumed"] if resumed else []),
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        **resume_fields,
                        "ephemeral": ephemeral,
                        "thread_id": timeout_thread_id,
                        "model": model,
                        "sandbox": sandbox,
                        "approval_policy": approval_policy,
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

        usage = next(
            (
                ev.get("usage", {})
                for ev in reversed(events)
                if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict)
            ),
            {},
        )
        accounting = _attempt_accounting(getattr(completed, "attempt_usage", None), usage, resumed)
        counts = accounting.pop("usage")
        tokens_in = counts["input_tokens"]
        tokens_out = counts["output_tokens"]

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
        thread_id = observed_thread_id(events, resume_record if resumed else None)

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
        failure = classify_codex_failure(
            completed.stderr + "\n" + codex_diagnostic_text(completed.stdout, events)
            + "\n" + turn_failure_message
        ) if process_failed else None
        if turn_failed and failure == UNKNOWN:
            failure = "codex_turn_failed"
        # For a build worker a bare terminal verdict is not a report; its
        # free-text report must still become an artifact.
        report_artifacts = [
            artifact for artifact in parsed_artifacts
            if not report_mode or (artifact.payload or {}).get("kind") != "worker_verdict"
        ]
        unstructured = not process_failed and not report_artifacts and bool(last_message.strip())
        degraded = unstructured and not report_mode

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
                + (["context:resumed"] if resumed else [])
                + (["bypass:dangerously-bypass-approvals-and-sandbox"] if bypass else [])
            ),
            payload={
                "returncode": completed.returncode,
                "model": model,
                "sandbox": sandbox,
                "approval_policy": approval_policy,
                "ephemeral": ephemeral,
                "thread_id": thread_id,
                **resume_fields,
                "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                "stdout_capture": stdout_capture,
                "stderr_capture": stderr_capture,
                "live_log": completed.live_log_path,
                "attempt_id": getattr(completed, "attempt_id", None),
                "dispatch_receipt": getattr(completed, "dispatch_receipt", None),
                "last_message": _redacted_tail(last_message, _STDOUT_TAIL_CHARS),
                "last_message_capture": last_message_capture,
                **selected_token_usage(_selected_input(counts, usage, accounting)),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": None if tokens_in is None or tokens_out is None else tokens_in + tokens_out,
                "cached_input_tokens": counts["cached_input_tokens"],
                "cache_write_input_tokens": counts["cache_write_input_tokens"],
                "reasoning_output_tokens": counts["reasoning_output_tokens"],
                **accounting,
                "turn_failed": turn_failed,
                "turn_failure_message": turn_failure_message,
                "cwd": str(cwd),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
                **diff_source_payload(before, after),
                **attribution,
                "failure": failure,
            },
        )
        artifacts: list[Artifact] = [verification]
        if unstructured and report_mode:
            # implement_report_artifacts re-parses the verdict itself.
            parsed_artifacts = []
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


# The stdin prompt positional. ``--`` ends option parsing first: ``--image``
# is variadic and would otherwise take the bare ``-`` as another image path.
STDIN_PROMPT = ("--", "-")


def _attempt_accounting(attempt: object, sdk_usage: dict, resumed: bool) -> dict:
    """This attempt's counters plus where they came from.

    The SDK's ``turn.completed`` usage is kept beside them under ``sdk_*`` and
    labeled with its own scope; it is never added to or subtracted from the
    attempt counters.
    """
    sdk = {field: codex_rollout.count(sdk_usage.get(field)) for field in codex_rollout.FIELDS}
    if isinstance(attempt, dict):
        out = {key: value for key, value in attempt.items()}
        out["usage"] = dict(attempt.get("usage") or {})
        out.setdefault("usage_unlinked_reason", None)
    elif not resumed:
        out = {
            "usage_scope": "attempt",
            "usage_provenance": "sdk_turn_completed" if sdk_usage else None,
            "usage_unlinked_reason": None if sdk_usage else "sdk_usage_missing",
            "rollout_turn_id": None,
            "rollout_request_count": None,
            "usage": dict(sdk),
            "usage_partial_fields": sorted(f for f, v in sdk.items() if v is None),
            "usage_disputed_fields": [],
            "usage_conflicts": [],
        }
    else:
        out = codex_rollout.unlinked("rollout_not_read")
    out["sdk_usage_scope"] = "session_cumulative" if resumed else "attempt"
    out.update({"sdk_" + field: value for field, value in sdk.items()})
    return out


def _selected_input(counts: dict, sdk_usage: dict, accounting: dict) -> dict:
    """Presence-preserving input/output counts; an estimate flag only rides SDK counts."""
    selected = {k: counts[k] for k in ("input_tokens", "output_tokens") if counts[k] is not None}
    if accounting.get("usage_provenance") == "sdk_turn_completed" and "tokens_estimated" in sdk_usage:
        selected["tokens_estimated"] = sdk_usage["tokens_estimated"]
    return selected


def _resumed_session_id(resume_record: object) -> Optional[str]:
    sid = resume_record.get("session_id") if isinstance(resume_record, dict) else None
    return str(sid) if sid else None


def _rollout_baseline(home: Path, resume_record: object) -> Optional[frozenset]:
    """Turn ids already in the resumed session's rollout before this attempt."""
    sid = _resumed_session_id(resume_record)
    if sid is None:
        return None
    try:
        return codex_rollout.turn_ids(codex_rollout.rollout_path(home, sid))
    except Exception:
        return None


def _rollout_attempt_usage(
    home: Path, resume_record: object, baseline: Optional[frozenset], stdout: str,
) -> Optional[dict]:
    """This attempt's usage from its rollout records, or None to keep the SDK's.

    A resumed run always answers (linked or NULL with a reason): its
    ``turn.completed`` usage is session-cumulative. A cold run's SDK usage is
    already attempt-local, so it is replaced only by a linked rollout sum.
    """
    sid = _resumed_session_id(resume_record)
    try:
        if sid is not None:
            return codex_rollout.attempt_usage(
                codex_rollout.rollout_path(home, sid), baseline or frozenset())
        thread_id = observed_thread_id(parse_codex_events(stdout or ""))
        path = codex_rollout.rollout_path(home, thread_id)
        if path is None:
            return None
        found = codex_rollout.attempt_usage(path, frozenset())
        return found if found.get("usage_scope") == "attempt" else None
    except Exception:
        # Best effort: accounting never fails a worker run.
        return codex_rollout.unlinked("rollout_unreadable") if sid is not None else None


def observed_thread_id(events: list[dict], resume: object = None) -> Optional[str]:
    """The one thread this run started, or None when missing or ambiguous.

    Several distinct ``thread.started`` ids, or a resumed run reporting a thread
    other than the one it resumed, record nothing rather than a guess.
    """
    ids = {str(ev["thread_id"]) for ev in events
           if ev.get("type") == "thread.started" and ev.get("thread_id")}
    if len(ids) != 1:
        return None
    thread_id = ids.pop()
    expected = resume.get("session_id") if isinstance(resume, dict) else None
    if expected and str(expected) != thread_id:
        return None
    return thread_id


def build_codex_exec_command(
    *,
    executable: Union[str, list[str]] = "codex",
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    cwd: Optional[Path] = None,
    sandbox: str = "workspace-write",
    approval_policy: str = "never",
    ephemeral: bool = True,
    skip_git_repo_check: bool = True,
    dangerously_bypass: bool = False,
    extra_args: object = None,
) -> list[str]:
    """Build the non-interactive ``codex exec`` argv.

    The prompt is **not** part of the command: the trailing ``-`` positional
    tells ``codex exec`` to read its instructions from stdin, and the caller
    feeds them through ``CliInvocation.subprocess_kwargs['stdin_data']``. An
    enriched Puppetmaster prompt routinely runs past Windows' 32767-character
    ``CreateProcess`` command-line cap, which fails the spawn outright with
    ``[WinError 206]``; stdin has no such limit. ``prompt`` is accepted and
    ignored so this exported signature stays source-compatible.
    """
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
    # Read the prompt from stdin. Never pass a prompt positional as well: codex
    # then appends the piped text as a separate `<stdin>` block instead of
    # treating it as the instruction.
    command.extend(STDIN_PROMPT)
    return command


def build_codex_resume_command(
    *,
    executable: Union[str, list[str]] = "codex",
    session_id: str,
    model: Optional[str] = None,
    sandbox: str = "workspace-write",
    approval_policy: str = "never",
    skip_git_repo_check: bool = True,
    dangerously_bypass: bool = False,
    extra_args: object = None,
) -> list[str]:
    """Build the ``codex exec resume <thread_id>`` argv.

    ``exec resume`` rejects ``--sandbox`` and ``-C``: the sandbox travels as a
    ``-c sandbox_mode`` override and cwd comes from the subprocess. It never
    takes ``--ephemeral``, so the resumed thread stays resumable for the next
    revision. The prompt is read from stdin via the trailing ``-- -``.
    """
    command = command_parts(executable)
    command.extend(["exec", "resume", str(session_id), "--json"])
    command.extend(["-c", f'approval_policy="{approval_policy}"'])
    if sandbox:
        command.extend(["-c", f'sandbox_mode="{sandbox}"'])
    if dangerously_bypass:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if model:
        command.extend(["-m", str(model)])
    if extra_args:
        command.extend(command_parts(extra_args))
    command.extend(STDIN_PROMPT)
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


def codex_diagnostic_text(stdout: str, events: list[dict[str, Any]]) -> str:
    """The parts of a ``--json`` stdout that diagnose a failure.

    Non-JSON lines (CLI banners and errors) and ``error`` event messages.
    ``item.*`` events are the worker's own transcript: an edit to a login
    module is not a logout.
    """
    lines = [raw for raw in (stdout or "").splitlines() if not raw.strip().startswith("{")]
    lines.extend(str(ev.get("message") or "") for ev in events if ev.get("type") == "error")
    return "\n".join(lines)


def last_codex_agent_message(events: list[dict[str, Any]]) -> str:
    """Return the final turn's most recent ``item.completed`` ``agent_message``.

    Codex emits multiple ``item.completed`` events per turn (tool calls,
    reasoning summaries, the final agent message); we only want the final
    user-visible reply. Items before the last ``turn.started`` (startup notices,
    or anything a resumed thread echoes) are not this run's reply.
    """
    start = max(
        (index for index, ev in enumerate(events) if ev.get("type") == "turn.started"),
        default=-1,
    )
    for ev in reversed(events[start + 1:]):
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


def _lease_codex_thread(record: dict) -> tuple[dict, Optional[SessionLease]]:
    """Hold the resumed thread for this run, or fall back to a fresh session.

    Contention is settled before any model call or side effect: the losing
    worker runs fresh with an ``unavailable`` record saying why.
    """
    session_id = str(record.get("session_id"))
    try:
        lease = acquire_codex_thread(session_id)
    except OSError as exc:
        return {**record, "status": "unavailable", "reason": f"codex thread lease unavailable: {exc}"}, None
    if lease is None:
        return {
            **record,
            "status": "unavailable",
            "reason": (
                f"codex thread {session_id} is being resumed by another live worker; "
                "codex resume continues a thread in place"
            ),
        }, None
    return record, lease


def _codex_home():
    from puppetmaster import codex_home

    return codex_home
