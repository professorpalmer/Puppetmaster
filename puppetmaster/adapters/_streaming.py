from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from puppetmaster.fs_permissions import mkdir_private, open_private, write_private_text
from puppetmaster.models import Task
from puppetmaster.redaction import redact_secrets

_STDOUT_HEAD_CHARS = 1000


_STDOUT_TAIL_CHARS = 8000


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
    # Set when the process could not be spawned at all (missing executable,
    # bad cwd, ...). Callers treat a non-None value as a hard adapter failure
    # rather than an empty-but-successful run.
    spawn_error: Optional[str] = None


def _kill_process_tree(process: "subprocess.Popen", started_new_session: bool) -> None:
    """Best-effort kill of a timed-out child *and its descendants*.

    Adapters that launch with ``start_new_session=True`` (e.g. Hermes) put the
    child in its own process group; grandchildren survive a bare
    ``process.kill()`` of the direct child. On POSIX we signal the whole group
    so nothing is orphaned. ``os.killpg`` / ``os.getpgid`` only exist on POSIX,
    so non-POSIX falls back to killing the direct child.
    """
    if started_new_session and os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            # Group already gone or unkillable — fall through to the direct kill.
            pass
    try:
        process.kill()
    except Exception:
        pass


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
        except OSError:
            pass

    # Workers run console-less; any git the agent CLI spawns down the chain
    # must never launch an interactive pager or credential prompt. On Windows
    # a console-less grandchild git + less allocates a visible terminal window
    # per invocation ("Press RETURN to continue" flood).
    env = dict(env) if env is not None else os.environ.copy()
    env.setdefault("PAGER", "cat")
    env["GIT_PAGER"] = "cat"
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    popen_kwargs: dict[str, Any] = {}
    if start_new_session:
        # Hermes tears down its own process group on exit and has been observed
        # signal-killing a parent shell loop. Launching in a fresh session keeps
        # that teardown confined to the child and away from Puppetmaster.
        popen_kwargs["start_new_session"] = True

    try:
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
    except OSError as exc:
        # Popen failed before any reader thread started (missing executable,
        # bad cwd, ...). Close the live-log handle we already opened so it can't
        # leak, and surface a structured failure instead of letting the OSError
        # escape the adapter.
        message = redact_secrets(f"{type(exc).__name__}: {exc}") or "spawn_error"
        if live_handle is not None:
            try:
                live_handle.write(f"[puppetmaster] spawn failed: {message}\n")
            except Exception:
                pass
            try:
                live_handle.close()
            except Exception:
                pass
        return StreamedProcess(
            returncode=None,
            stdout="",
            stderr=message,
            timed_out=False,
            live_log_path=str(live_path) if live_path is not None else None,
            spawn_error=message,
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
        # Kill the whole process group when we launched a new session, so
        # grandchildren (e.g. Hermes' own spawned tree) don't survive the
        # timeout as orphans; otherwise kill just the direct child.
        _kill_process_tree(process, start_new_session)
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

