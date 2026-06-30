from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph, scrub_foreign_interpreter_env
from puppetmaster.failure import classify_hermes_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.ports import apply_worktree_ports
from puppetmaster.redaction import redact_secrets
from puppetmaster.usage import token_usage

from ._base import (
    CliInvocation,
    CliWorkerAdapter,
    command_parts,
    diff_source_payload,
    make_patch_artifact,
    missing_cli_artifact,
    resolve_command,
    tool_list,
    verification_artifact,
)
from ._facade import facade
from ._prompts import (
    _ANALYZE_JSON_ONLY_RETRY,
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    prompt_with_skills,
    with_repo_census,
    with_report_contract,
)
from ._streaming import (
    StreamedProcess,
    _STDOUT_HEAD_CHARS,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
    run_streamed_subprocess,
)
from ._base import _should_emit_patch_artifact
from .cursor import cursor_result_artifacts, implement_report_artifacts

DEFAULT_HERMES_ANALYZE_TOOLSETS = "file,web,vision"


DEFAULT_HERMES_IMPLEMENT_TOOLSETS = "file,terminal,code_execution,web,vision"


VALID_HERMES_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


_HERMES_ENV_CREDENTIAL_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


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
        resolved = facade("resolve_command")(command_base[0])
        if resolved is None:
            return
        facade("subprocess").run(
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
        or (facade("Path").home() / ".hermes")
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
    env_file = facade("Path").home() / ".hermes" / ".env"
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
    auth_file = facade("Path").home() / ".hermes" / "auth.json"
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


class HermesAdapter(CliWorkerAdapter):
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
            facade("prune_hermes_tool_sessions")(
                task.payload.get("executable")
                or os.environ.get("HERMES_COMMAND")
                or "hermes",
                source=str(task.payload.get("source", "tool")),
            )

    def _run_implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        executable = task.payload.get("executable") or os.environ.get("HERMES_COMMAND") or "hermes"
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
            "hermes",
            executable_label,
            (
                "Hermes CLI was not found. Install it or set "
                "HERMES_COMMAND / payload.executable."
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
        base_prompt = with_report_contract(task.payload.get("prompt") or task.instruction)
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_skills(
                prompt_with_memory(build_implement_prompt(base_prompt), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        executable = task.payload.get("executable") or os.environ.get("HERMES_COMMAND") or "hermes"
        command_base = command_parts(executable)
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
        return CliInvocation(
            command=command,
            sidecar_name="hermes_implement",
            subprocess_kwargs={"start_new_session": True},
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "extra_dirty_message": (
                    " For focused edits on a dirty tree (docs, tests), use puppetmaster_edit — it edits "
                    "in place and needs no clean tree."
                ),
            },
        )

    def _invoke_cli(
        self,
        task: Task,
        prepared: CliInvocation,
        cwd: Path,
        timeout_seconds: int,
    ) -> StreamedProcess:
        worker_env = scrub_foreign_interpreter_env(
            apply_worktree_ports(os.environ.copy(), cwd)
        )
        with hermes_reasoning_effort_env(
            worker_env, task.payload.get("reasoning_effort")
        ) as run_env:
            return facade("run_streamed_subprocess")(
                command=prepared.command,
                env=run_env,
                task=task,
                sidecar_name=prepared.sidecar_name,
                timeout_seconds=timeout_seconds,
                cwd=str(cwd),
                start_new_session=True,
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
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        timeout_seconds = int(task.payload.get("timeout_seconds", 900))
        if completed.timed_out:
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
                    make_patch_artifact(
                        task,
                        worker_id,
                        before,
                        after,
                        adapter="hermes",
                        status="failed",
                        change="Hermes modified repository files.",
                        sidecar_name="hermes_implement",
                    )
                )
            return artifacts

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
            prompt_text=str(prepared.extras.get("prompt") or ""),
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
                    "cwd": str(Path(task.payload.get("cwd") or ".").resolve()),
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
                make_patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    adapter="hermes",
                    status="applied" if not process_failed else "failed",
                    change="Hermes modified repository files.",
                    sidecar_name="hermes_implement",
                )
            )
        return artifacts

    def _run_analyze(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_skills(
                prompt_with_memory(build_structured_prompt(base_prompt, final_message_note=True), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = facade("with_repo_census")(prompt, cwd)
        timeout_seconds = int(task.payload.get("timeout_seconds", 600))
        executable = task.payload.get("executable") or os.environ.get("HERMES_COMMAND") or "hermes"
        command_base = command_parts(executable)
        resolved = facade("resolve_command")(command_base[0])
        if resolved is None:
            return missing_cli_artifact(
                task,
                worker_id,
                "hermes",
                executable,
                (
                    "Hermes CLI was not found. Install it or set "
                    "HERMES_COMMAND / payload.executable."
                ),
            )

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
                return facade("run_streamed_subprocess")(
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

