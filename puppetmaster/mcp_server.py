from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.codegraph import (
    CODEGRAPH_COMMAND,
    CODEGRAPH_MISSING_HINT,
    CODEGRAPH_NATIVE_SQLITE_HINT,
    CodegraphLockBusy,
    acquire_codegraph_lock,
    codegraph_affected,
    codegraph_available,
    codegraph_context_command,
    codegraph_files_listing,
    codegraph_init_command,
    codegraph_lock_path,
    codegraph_native_sqlite_broken,
    codegraph_query,
    codegraph_status_command,
)
from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.mcp_registry import (
    HeartbeatThread,
    deregister as registry_deregister,
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    register as registry_register,
    summarize as registry_summarize,
)
from puppetmaster.state import resolve_state_dir
from puppetmaster.store_factory import create_store


JsonObject = dict[str, Any]
ASYNC_PROCESSES: list[subprocess.Popen] = []

# Concurrency primitives for the stdio loop.
#
# The original implementation ran tools synchronously on the same thread
# that read stdin, so any long-running handler (most painfully
# `puppetmaster_codegraph_init(index=true)` with its 600-second timeout)
# blocked every subsequent JSON-RPC message. Cursor's agent would treat
# that frozen stdio pipe as a dead transport and surface
# "Tool execution error. Not connected" even though the server process
# was alive and the underlying swarm/index job was still making progress.
#
# We now dispatch every incoming message to a worker thread. Each thread
# writes its response under a shared stdout lock so JSON-RPC frames stay
# whole. JSON-RPC over stdio does not require responses to be returned
# in request order — every response carries the original `id` — so
# concurrent handling is correct as long as we don't interleave bytes.
_STDOUT_LOCK = threading.Lock()
_DEFAULT_WORKER_COUNT = 8

# Tool-call keepalive defaults. The v0.5.0 thread pool fix kept the stdin
# loop responsive, but Cursor's MCP client appears to treat a stdio pipe
# with no bytes flowing during a single long-running tool call as a dead
# transport — at which point it closes the pipe and surfaces
# "Tool execution error. Not connected" even though the server is alive.
# We defeat that heuristic by emitting a JSON-RPC ``notifications/message``
# every N seconds while a tool is still running. The notification is a
# debug-level log that clients are free to ignore; the point is that any
# bytes flowing through stdio keep the transport alive.
_DEFAULT_KEEPALIVE_AFTER_SECONDS = 5.0
_DEFAULT_KEEPALIVE_INTERVAL_SECONDS = 10.0

# Input-staleness self-termination defaults. Cursor's MCP "lease" lifecycle
# can re-create a logical client without killing the previous Python
# server process, so we accumulate orphan servers across lease cycles.
# Each orphan still holds open SQLite handles and competes for the
# CodeGraph indexer lock. We close that loop by self-terminating any
# server that has not received an inbound JSON-RPC message in
# ``_DEFAULT_INPUT_STALE_SECONDS`` AND currently has zero in-flight tool
# calls. Active sessions are unaffected; only true orphans reap.
_DEFAULT_INPUT_STALE_SECONDS = 600.0  # 10 minutes
_DEFAULT_INPUT_STALE_CHECK_SECONDS = 30.0

# Idle MCP-pipe keepalive defaults. The v0.5.3 tool-call keepalive
# keeps bytes flowing during long handler executions, but the stdio
# pipe still goes silent between tool calls. Some Cursor builds appear
# to close MCP transports they consider "inactive", manifesting to the
# agent as a sudden `Tool execution error. Not connected` on a
# previously healthy session. v0.5.5 adds a server-wide idle ping that
# emits a small `notifications/message` every ~25s when no traffic has
# moved either direction for a while. Cost: ~150 bytes per ping; total
# bandwidth per hour ≈ 22 KB. Disabled with PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED.
_DEFAULT_IDLE_KEEPALIVE_INTERVAL_SECONDS = 25.0

# Module-level state for stdin-liveness tracking. Initialized at startup
# by ``main()`` and mutated by the stdin reader + tool-call dispatcher.
_INPUT_STATE_LOCK = threading.Lock()
_LAST_INBOUND_MESSAGE_AT = time.time()
_ACTIVE_TOOL_CALLS = 0
_SHUTDOWN_REQUESTED = threading.Event()


def _resolve_worker_count() -> int:
    raw = os.environ.get("PUPPETMASTER_MCP_WORKERS")
    if not raw:
        return _DEFAULT_WORKER_COUNT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_WORKER_COUNT
    return max(1, min(value, 64))


def _resolve_keepalive_seconds(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.5, value)


def _keepalive_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_MCP_KEEPALIVE_DISABLED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit_notification(notification: JsonObject) -> bool:
    """Serialize and write a JSON-RPC notification under the stdout lock.

    Returns False when the pipe is gone (BrokenPipeError or general OSError),
    so callers running on a daemon thread can stop trying. Notifications
    must never have an ``id`` field — that's how clients distinguish them
    from responses.
    """
    serialized = json.dumps(notification) + "\n"
    try:
        with _STDOUT_LOCK:
            sys.stdout.write(serialized)
            sys.stdout.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


class _ToolCallKeepalive:
    """Emit periodic JSON-RPC log notifications while a tool call is in flight.

    Lifecycle: caller constructs, calls :meth:`start`, runs the handler,
    then calls :meth:`stop` (or relies on the daemon thread reaping with
    the process). The background thread waits ``start_after_seconds``
    before emitting anything — short calls never produce keepalive
    traffic — then emits one notification per ``interval_seconds`` until
    stop is signalled or the pipe is closed.

    The notifications use the MCP-spec ``notifications/message`` method
    with a ``debug`` level, which clients are free to log or ignore. They
    intentionally carry no ``id`` field so Cursor doesn't try to match
    them against an outstanding request.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        request_id: Any,
        interval_seconds: Optional[float] = None,
        start_after_seconds: Optional[float] = None,
        emitter: Callable[[JsonObject], bool] = _emit_notification,
    ) -> None:
        self._tool_name = tool_name or "unknown"
        self._request_id = request_id
        self._interval = (
            interval_seconds
            if interval_seconds is not None
            else _resolve_keepalive_seconds(
                "PUPPETMASTER_MCP_KEEPALIVE_INTERVAL_SECONDS",
                _DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
            )
        )
        self._start_after = (
            start_after_seconds
            if start_after_seconds is not None
            else _resolve_keepalive_seconds(
                "PUPPETMASTER_MCP_KEEPALIVE_AFTER_SECONDS",
                _DEFAULT_KEEPALIVE_AFTER_SECONDS,
            )
        )
        self._emit = emitter
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"pm-mcp-keepalive-{self._tool_name}",
        )
        self._emitted = 0

    @property
    def emitted_count(self) -> int:
        """How many keepalive notifications have actually been written."""
        return self._emitted

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        self._stop.set()
        if wait:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        if self._stop.wait(self._start_after):
            return
        elapsed = self._start_after
        while True:
            if not self._safe_emit(elapsed):
                return
            if self._stop.wait(self._interval):
                return
            elapsed += self._interval

    def _safe_emit(self, elapsed: float) -> bool:
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": "debug",
                "logger": "puppetmaster",
                "data": {
                    "kind": "tool_call_progress",
                    "tool": self._tool_name,
                    "request_id": self._request_id,
                    "elapsed_seconds": round(elapsed, 1),
                    "message": (
                        f"Puppetmaster tool '{self._tool_name}' still running "
                        f"({elapsed:.0f}s elapsed)"
                    ),
                },
            },
        }
        if self._stop.is_set():
            return False
        ok = self._emit(notification)
        if ok:
            self._emitted += 1
        return ok


def _input_staleness_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_MCP_INPUT_STALE_DISABLED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_input_stale_seconds(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(10.0, value)


def _mark_inbound_message() -> None:
    """Record that a JSON-RPC message arrived on stdin.

    Called from the stdin reader for every parsed line so the
    staleness watcher can tell a live conversation apart from an
    orphan whose Cursor parent has stopped talking to it.
    """
    global _LAST_INBOUND_MESSAGE_AT
    with _INPUT_STATE_LOCK:
        _LAST_INBOUND_MESSAGE_AT = time.time()


def _tool_call_started() -> None:
    global _ACTIVE_TOOL_CALLS
    with _INPUT_STATE_LOCK:
        _ACTIVE_TOOL_CALLS += 1


def _tool_call_finished() -> None:
    global _ACTIVE_TOOL_CALLS
    with _INPUT_STATE_LOCK:
        _ACTIVE_TOOL_CALLS = max(0, _ACTIVE_TOOL_CALLS - 1)


def _input_state_snapshot() -> tuple[float, int]:
    with _INPUT_STATE_LOCK:
        return _LAST_INBOUND_MESSAGE_AT, _ACTIVE_TOOL_CALLS


class _InputStalenessWatcher(threading.Thread):
    """Self-terminate the MCP server when it looks like an orphan.

    Cursor's MCP "lease" lifecycle re-creates a logical client without
    killing the previous Python server process. The old process keeps
    running, holds SQLite handles, and competes for the CodeGraph
    indexer lock. Heartbeat thread can't detect this — heartbeat
    measures Python process liveness, not stdin liveness — so we
    measure inbound JSON-RPC traffic directly.

    A server is "orphaned" when both:
      - No stdin message has arrived in ``stale_after_seconds``.
      - There are zero in-flight tool calls.

    On detection we close stdin, which causes the ``for line in sys.stdin``
    loop in ``main()`` to terminate with EOF. The existing finally
    block then runs cleanly: deregister from the registry, shut down
    the executor, exit. No ``os._exit``, no leaked SQLite handles.
    """

    def __init__(
        self,
        *,
        stale_after_seconds: float,
        check_interval_seconds: float,
        on_shutdown: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True, name="puppetmaster-mcp-input-stale-watcher")
        self._stale_after = stale_after_seconds
        self._interval = check_interval_seconds
        self._on_shutdown = on_shutdown
        self._stop = threading.Event()
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - exercised via integration tests
        while not self._stop.wait(self._interval):
            last_msg, active = _input_state_snapshot()
            age = time.time() - last_msg
            if age < self._stale_after:
                continue
            if active > 0:
                continue
            self._triggered = True
            _SHUTDOWN_REQUESTED.set()
            try:
                self._on_shutdown()
            except Exception:
                pass
            return


def _idle_keepalive_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_idle_keepalive_interval() -> float:
    raw = os.environ.get("PUPPETMASTER_MCP_IDLE_KEEPALIVE_INTERVAL_SECONDS")
    if not raw:
        return _DEFAULT_IDLE_KEEPALIVE_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_IDLE_KEEPALIVE_INTERVAL_SECONDS
    # Refuse intervals shorter than 5s to avoid log spam.
    return max(5.0, value)


class _IdleKeepalive(threading.Thread):
    """Emit a small notification every N seconds to keep the stdio pipe live.

    Complement to :class:`_ToolCallKeepalive`: that one keeps bytes
    flowing while a specific handler runs. This one keeps bytes flowing
    when *nothing* is running, which is when Cursor's "transport looks
    idle" heuristic can otherwise close the pipe and surface
    ``Tool execution error. Not connected`` on the agent's very next
    call.

    The keepalive is suppressed while a tool call is in flight because
    the per-call keepalive already covers that window. It also
    suppresses if the server is shutting down, so we don't fight the
    finally block.
    """

    def __init__(
        self,
        *,
        interval_seconds: float,
        emitter: Callable[[JsonObject], bool] = _emit_notification,
    ) -> None:
        super().__init__(daemon=True, name="puppetmaster-mcp-idle-keepalive")
        self._interval = interval_seconds
        self._emit = emitter
        self._stop = threading.Event()
        self._emitted = 0

    @property
    def emitted_count(self) -> int:
        return self._emitted

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - timing covered via direct tests
        while not self._stop.wait(self._interval):
            if _SHUTDOWN_REQUESTED.is_set():
                return
            _, active = _input_state_snapshot()
            if active > 0:
                continue
            ok = self._emit(self._build_notification())
            if not ok:
                return
            self._emitted += 1

    def _build_notification(self) -> JsonObject:
        return {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": "debug",
                "logger": "puppetmaster",
                "data": {
                    "kind": "idle_keepalive",
                    "message": "puppetmaster mcp server idle, pipe live",
                    "interval_seconds": self._interval,
                },
            },
        }


def _request_clean_shutdown() -> None:
    """Interrupt the main stdio loop so it exits through the finally block.

    We deliberately do NOT call ``os._exit`` — sending SIGINT to ourselves
    raises ``KeyboardInterrupt`` in the main thread, which propagates out
    of ``for line in sys.stdin`` and lets the existing ``finally`` block
    in :func:`main` deregister from the MCP registry, stop the
    heartbeat/watcher threads, and shut down the executor. A hard exit
    would skip all that and leave a stale tracking file.

    Closing fd 0 (the obvious approach) does not reliably unblock
    CPython's buffered stdin reader on macOS — once the underlying
    ``read(2)`` is in flight, closing the fd from another thread may
    return EBADF on the next syscall but does not interrupt the
    currently blocked read.
    """
    try:
        import signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)
    except (OSError, ValueError):
        # ValueError if SIGINT isn't a valid signal (unlikely);
        # OSError if the process is in some weird signal state.
        # Fall back to closing stdin — better than leaving the
        # orphan running indefinitely.
        try:
            os.close(0)
        except OSError:
            pass


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: JsonObject
    handler: Callable[[JsonObject], JsonObject]


def main() -> int:
    # Sweep dead tracking files from prior server runs before we
    # advertise ourselves. This is how Cursor users escape the
    # "Restart MCP, see 3 orphan PIDs in `ps`" trap without typing
    # any pkill commands.
    try:
        registry_prune_dead()
    except Exception:
        pass

    registration_path = None
    heartbeat_thread: Optional[HeartbeatThread] = None
    try:
        registration_path = registry_register(
            workspace=str(Path.cwd()),
            version=_server_version(),
        )
        heartbeat_thread = HeartbeatThread(registration_path)
        heartbeat_thread.start()
    except Exception:
        # The registry is a diagnostic affordance; never refuse to
        # serve MCP traffic just because we can't write to it.
        registration_path = None
        heartbeat_thread = None

    # Reset the staleness counter at startup so a slow Cursor handshake
    # doesn't immediately look orphaned to the watcher.
    global _LAST_INBOUND_MESSAGE_AT, _ACTIVE_TOOL_CALLS
    with _INPUT_STATE_LOCK:
        _LAST_INBOUND_MESSAGE_AT = time.time()
        _ACTIVE_TOOL_CALLS = 0
    _SHUTDOWN_REQUESTED.clear()

    input_watcher: Optional[_InputStalenessWatcher] = None
    if not _input_staleness_disabled():
        input_watcher = _InputStalenessWatcher(
            stale_after_seconds=_resolve_input_stale_seconds(
                "PUPPETMASTER_MCP_INPUT_STALE_SECONDS",
                _DEFAULT_INPUT_STALE_SECONDS,
            ),
            check_interval_seconds=_resolve_input_stale_seconds(
                "PUPPETMASTER_MCP_INPUT_STALE_CHECK_SECONDS",
                _DEFAULT_INPUT_STALE_CHECK_SECONDS,
            ),
            on_shutdown=_request_clean_shutdown,
        )
        input_watcher.start()

    idle_keepalive: Optional[_IdleKeepalive] = None
    if not _idle_keepalive_disabled():
        idle_keepalive = _IdleKeepalive(
            interval_seconds=_resolve_idle_keepalive_interval(),
        )
        idle_keepalive.start()

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_resolve_worker_count(),
        thread_name_prefix="pm-mcp",
    )
    diag = os.environ.get("PUPPETMASTER_MCP_DIAG_EXIT") == "1"
    exit_reason = "stdin_eof"
    try:
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _mark_inbound_message()
                try:
                    executor.submit(_process_message_safely, message)
                except RuntimeError as exc:
                    exit_reason = f"executor_submit_failed: {exc}"
                    raise
        except KeyboardInterrupt:
            exit_reason = "sigint_clean_shutdown"
            # InputStalenessWatcher (or an actual Ctrl-C) signalled
            # SIGINT to trigger a clean shutdown. Fall through to the
            # finally block instead of letting Python print a traceback.
            pass
        except BaseException as exc:  # pragma: no cover - smoke-only
            exit_reason = f"main_loop_exception: {type(exc).__name__}: {exc}"
            if diag:
                import traceback
                traceback.print_exc(file=sys.stderr)
            raise
    finally:
        if diag:
            sys.stderr.write(f"[pm-mcp] main exiting; reason={exit_reason}\n")
            sys.stderr.flush()
        executor.shutdown(wait=False)
        if input_watcher is not None:
            input_watcher.stop()
        if idle_keepalive is not None:
            idle_keepalive.stop()
        if heartbeat_thread is not None:
            heartbeat_thread.stop()
        if registration_path is not None:
            try:
                registry_deregister(registration_path)
            except Exception:
                pass
    return 0


def _server_version() -> Optional[str]:
    """Best-effort lookup of the installed puppetmaster version."""
    try:
        from importlib.metadata import PackageNotFoundError, version as _version

        try:
            return _version("puppetmaster")
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _process_message_safely(message: JsonObject) -> None:
    """Run handle_message on a worker thread and write any response.

    Wraps every ``tools/call`` in a :class:`_ToolCallKeepalive` so a long
    handler doesn't starve the stdio pipe. The keepalive is torn down in
    ``finally`` so a slow handler that raises still gets cleaned up.
    """
    is_tool_call = message.get("method") == "tools/call"
    keepalive: Optional[_ToolCallKeepalive] = None
    if is_tool_call and not _keepalive_disabled():
        params = message.get("params") or {}
        tool_name = params.get("name") if isinstance(params, dict) else None
        keepalive = _ToolCallKeepalive(
            tool_name=str(tool_name or ""),
            request_id=message.get("id"),
        )
        keepalive.start()
    if is_tool_call:
        _tool_call_started()
    try:
        response = handle_message(message)
    except Exception as exc:
        response = error_response(message.get("id"), -32000, str(exc))
    finally:
        if keepalive is not None:
            keepalive.stop()
        if is_tool_call:
            _tool_call_finished()
    if response is None:
        return
    serialized = json.dumps(response) + "\n"
    with _STDOUT_LOCK:
        sys.stdout.write(serialized)
        sys.stdout.flush()


def handle_message(message: JsonObject) -> Optional[JsonObject]:
    method = message.get("method")
    request_id = message.get("id")

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "puppetmaster", "version": "0.2.0-beta.1"},
            }
        elif method == "tools/list":
            result = {"tools": [tool_to_json(tool) for tool in tools()]}
        elif method == "tools/call":
            params = message.get("params") or {}
            result = call_tool(str(params.get("name", "")), params.get("arguments") or {})
        else:
            return error_response(request_id, -32601, f"Unknown MCP method: {method}")
    except Exception as exc:
        return error_response(request_id, -32000, str(exc))

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def call_tool(name: str, arguments: JsonObject) -> JsonObject:
    registry = {tool.name: tool for tool in tools()}
    tool = registry.get(name)
    if tool is None:
        raise ValueError(f"Unknown Puppetmaster tool: {name}")
    return tool.handler(arguments)


def tools() -> list[McpTool]:
    return [
        McpTool(
            name="puppetmaster_doctor",
            description="Check Puppetmaster runtime, SQLite state, and provider adapter setup.",
            input_schema=base_schema(),
            handler=lambda args: run_cli(["doctor"], args),
        ),
        McpTool(
            name="puppetmaster_route_task",
            description=(
                "Show which model the Puppetmaster router would pick for a task, "
                "including estimated cost in USD and a list of rejected alternatives "
                "with the reason each was rejected. Pure decision tool — does not "
                "run the model. Use this to estimate spend before kicking off a swarm "
                "or to debug 'why did this go to model X?' artifacts."
            ),
            input_schema=route_task_schema(),
            handler=run_route_task,
        ),
        McpTool(
            name="puppetmaster_list_models",
            description=(
                "Print the user-owned LLM model registry (~/.puppetmaster/models.json "
                "by default). Shows model ids, adapters, capability scores, and per-token "
                "prices that the router uses."
            ),
            input_schema=list_models_schema(),
            handler=run_list_models,
        ),
        McpTool(
            name="puppetmaster_job_cost",
            description=(
                "Sum the estimated USD cost of every router decision for a job. "
                "Returns per-model breakdown + grand total. Use to answer "
                "'how much did this swarm cost?' or 'which model ate the most tokens?'. "
                "Estimates are based on user-asserted prices in the registry — "
                "Puppetmaster does not call a billing API."
            ),
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(
                ["cost", require_string(args, "job_id"), "--json"], args
            ),
        ),
        McpTool(
            name="puppetmaster_cursor_review",
            description="Run a Cursor SDK review worker through Puppetmaster and wait for completion.",
            input_schema=goal_schema(
                "Review this repo and identify risks, findings, and verification gaps."
            ),
            handler=lambda args: run_cursor(args, review=True),
        ),
        McpTool(
            name="puppetmaster_start_cursor_review",
            description="Start a Cursor SDK review worker asynchronously and return job_id immediately.",
            input_schema=goal_schema(
                "Review this repo and identify risks, findings, and verification gaps."
            ),
            handler=lambda args: start_cursor(args, review=True),
        ),
        McpTool(
            name="puppetmaster_cursor_plan",
            description="Run a Cursor SDK planning worker through Puppetmaster and wait for completion.",
            input_schema=goal_schema("Plan the next safe implementation slice for this repo."),
            handler=lambda args: run_cursor(args, plan=True),
        ),
        McpTool(
            name="puppetmaster_start_cursor_plan",
            description="Start a Cursor SDK planning worker asynchronously and return job_id immediately.",
            input_schema=goal_schema("Plan the next safe implementation slice for this repo."),
            handler=lambda args: start_cursor(args, plan=True),
        ),
        McpTool(
            name="puppetmaster_claude_implement",
            description="Run Claude Code as a full-edit Puppetmaster worker and wait for completion.",
            input_schema=claude_schema(),
            handler=run_claude,
        ),
        McpTool(
            name="puppetmaster_start_claude_implement",
            description="Start Claude Code as a full-edit worker asynchronously and return job_id immediately.",
            input_schema=claude_schema(),
            handler=start_claude,
        ),
        McpTool(
            name="puppetmaster_openai",
            description=(
                "Run an OpenAI Chat Completions worker through Puppetmaster and wait for "
                "completion. Uses OPENAI_API_KEY; returns structured artifacts."
            ),
            input_schema=openai_schema(),
            handler=run_openai,
        ),
        McpTool(
            name="puppetmaster_start_openai",
            description="Start an OpenAI worker asynchronously and return job_id immediately.",
            input_schema=openai_schema(),
            handler=start_openai,
        ),
        McpTool(
            name="puppetmaster_start_swarm",
            description="Start a local Puppetmaster swarm asynchronously and return job_id immediately.",
            input_schema=swarm_schema(),
            handler=start_swarm,
        ),
        McpTool(
            name="puppetmaster_start_cursor_swarm",
            description="Start a multi-role Cursor SDK analysis swarm asynchronously and return job_id immediately.",
            input_schema=cursor_swarm_schema(),
            handler=start_cursor_swarm,
        ),
        McpTool(
            name="puppetmaster_last_job",
            description="Return the most recent Puppetmaster job id.",
            input_schema=base_schema(),
            handler=lambda args: run_cli(["last"], args),
        ),
        McpTool(
            name="puppetmaster_status",
            description="Return task, artifact, and stale lease state for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["status", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_logs",
            description="Return readable Puppetmaster event logs for a job, defaulting to latest.",
            input_schema=job_schema(),
            handler=lambda args: run_cli(["logs"] + optional_job(args), args),
        ),
        McpTool(
            name="puppetmaster_live_artifacts",
            description="Return the live artifact feed for a job without waiting for final stitching.",
            input_schema=feed_schema(),
            handler=lambda args: run_feed(args),
        ),
        McpTool(
            name="puppetmaster_live_artifacts_follow",
            description=(
                "Long-poll for new artifacts since a cursor. Returns immediately when new "
                "artifacts arrive, or after timeout_seconds with an empty items array. "
                "Chain calls with the returned next_cursor for a push-feeling stream."
            ),
            input_schema=follow_schema(),
            handler=run_feed_follow,
        ),
        McpTool(
            name="puppetmaster_partial_summary",
            description="Return a live summary from current artifacts without waiting for final stitching.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["show", require_string(args, "job_id"), "--partial"], args),
        ),
        McpTool(
            name="puppetmaster_artifacts",
            description="Return structured JSON artifacts for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["artifacts", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_show",
            description="Return the stitched summary for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["show", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_codegraph_search",
            description=(
                "Find symbols by name using the local CodeGraph index. "
                "Bundles `codegraph query` so Cursor Agent only needs the Puppetmaster MCP."
            ),
            input_schema=codegraph_search_schema(),
            handler=run_codegraph_search,
        ),
        McpTool(
            name="puppetmaster_codegraph_context",
            description=(
                "Build task-relevant CodeGraph context (entry points, related symbols) "
                "without spawning a worker. Use for quick repo intel before editing."
            ),
            input_schema=codegraph_context_schema(),
            handler=run_codegraph_context,
        ),
        McpTool(
            name="puppetmaster_codegraph_affected",
            description=(
                "Resolve which test files are impacted by changed source files using "
                "CodeGraph's import graph. Great for targeted CI/test selection."
            ),
            input_schema=codegraph_affected_schema(),
            handler=run_codegraph_affected,
        ),
        McpTool(
            name="puppetmaster_codegraph_files",
            description="Return the indexed file structure from CodeGraph (faster than fs scans).",
            input_schema=codegraph_files_schema(),
            handler=run_codegraph_files,
        ),
        McpTool(
            name="puppetmaster_codegraph_status",
            description="Return CodeGraph index health and statistics for the target workspace.",
            input_schema=base_schema(),
            handler=run_codegraph_status,
        ),
        McpTool(
            name="puppetmaster_codegraph_init",
            description=(
                "Initialize CodeGraph in the target workspace (creates .codegraph/). "
                "Pass index=true to also build the full index in the background — the "
                "tool returns immediately with a run_id and log paths instead of "
                "blocking the MCP transport for minutes."
            ),
            input_schema=codegraph_init_schema(),
            handler=run_codegraph_init,
        ),
        McpTool(
            name="puppetmaster_codegraph_index",
            description=(
                "Kick off `codegraph index` in the background for the target workspace. "
                "Returns immediately with a run_id, pid, and stdout/stderr log paths. "
                "Poll progress with puppetmaster_codegraph_status. Only one indexer is "
                "allowed at a time per machine — concurrent calls fail fast with a clear "
                "lock-busy error so SQLite never gets clobbered."
            ),
            input_schema=base_schema(),
            handler=run_codegraph_index,
        ),
        McpTool(
            name="puppetmaster_repair_codegraph",
            description=(
                "Rebuild CodeGraph's better-sqlite3 native module for Cursor's bundled "
                "Node so the MCP server stops falling back to slow WASM SQLite. Run this "
                "whenever puppetmaster_codegraph_status or puppetmaster_doctor reports "
                "the native-SQLite-broken hint. Returns the rebuild stdout/stderr plus a "
                "verification of Backend: native."
            ),
            input_schema=repair_codegraph_schema(),
            handler=run_repair_codegraph,
        ),
        McpTool(
            name="puppetmaster_mcp_status",
            description=(
                "List every Puppetmaster MCP server tracked on this machine, with PID, "
                "workspace, age, last heartbeat, and alive/stale flags. Use this right "
                "after a `Tool execution error. Not connected` to see whether the swarm "
                "is still alive in a sibling server before tearing things down."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=run_mcp_status,
        ),
        McpTool(
            name="puppetmaster_mcp_cleanup",
            description=(
                "Prune dead tracking files for MCP servers that exited without cleanup, "
                "and (with kill_stale=true) SIGTERM/SIGKILL stale-but-alive Puppetmaster "
                "MCP servers whose Cursor parent appears to be gone. Never signals the "
                "current process. Returns the before/after registry snapshot."
            ),
            input_schema=mcp_cleanup_schema(),
            handler=run_mcp_cleanup,
        ),
    ]


def run_codegraph_search(args: JsonObject) -> JsonObject:
    payload = codegraph_query(
        require_string(args, "query"),
        cwd(args),
        kind=args.get("kind") if isinstance(args.get("kind"), str) else None,
        limit=int(args["limit"]) if args.get("limit") is not None else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_context(args: JsonObject) -> JsonObject:
    payload = codegraph_context_command(
        require_string(args, "task"),
        cwd(args),
        max_nodes=int(args.get("max_nodes") or 15),
        fmt=str(args.get("format") or "markdown"),
    )
    return codegraph_response(payload)


def run_codegraph_affected(args: JsonObject) -> JsonObject:
    files = args.get("files")
    if not isinstance(files, list) or not files:
        return tool_error("files must be a non-empty array of changed source paths.")
    payload = codegraph_affected(
        [str(item) for item in files if str(item).strip()],
        cwd(args),
        depth=int(args["depth"]) if args.get("depth") is not None else None,
        filter_pattern=str(args["filter"]) if args.get("filter") else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_files(args: JsonObject) -> JsonObject:
    payload = codegraph_files_listing(
        cwd(args),
        path=str(args["path"]) if args.get("path") else None,
        fmt=str(args["format"]) if args.get("format") else None,
        filter_pattern=str(args["filter"]) if args.get("filter") else None,
        max_depth=int(args["max_depth"]) if args.get("max_depth") is not None else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_status(args: JsonObject) -> JsonObject:
    payload = codegraph_status_command(cwd(args))
    if codegraph_native_sqlite_broken(payload.get("stdout", "") or payload.get("stderr", "")):
        payload = dict(payload)
        payload["native_sqlite_broken"] = True
        existing_hint = payload.get("hint") or ""
        payload["hint"] = (existing_hint + "\n\n" + CODEGRAPH_NATIVE_SQLITE_HINT).strip()
    return codegraph_response(payload)


def run_codegraph_init(args: JsonObject) -> JsonObject:
    """Initialize CodeGraph; when index=true, dispatch indexing in the background.

    Synchronous `codegraph init` is fast (creates `.codegraph/` and writes a
    small scaffold) and stays inline. When the caller asks for `index=true`
    we run init synchronously, then fork `codegraph index` as a detached
    subprocess so the MCP transport returns immediately. The agent polls
    `puppetmaster_codegraph_status` to watch indexing progress.
    """
    target_cwd = cwd(args)
    init_payload = codegraph_init_command(target_cwd, index=False)
    if not init_payload.get("ok", False):
        return codegraph_response(init_payload)
    if not args.get("index", False):
        return codegraph_response(init_payload)
    index_payload = _spawn_codegraph_indexer(args, target_cwd)
    init_payload = dict(init_payload)
    init_payload["index_run"] = index_payload
    return codegraph_response(init_payload)


def run_codegraph_index(args: JsonObject) -> JsonObject:
    """Dispatch `codegraph index` in the background; never block stdio."""
    target_cwd = cwd(args)
    if not codegraph_available():
        return codegraph_response(
            {
                "ok": False,
                "command": "codegraph index",
                "cwd": target_cwd,
                "error": CODEGRAPH_MISSING_HINT,
            }
        )
    payload = _spawn_codegraph_indexer(args, target_cwd)
    return codegraph_response(payload)


def run_repair_codegraph(args: JsonObject) -> JsonObject:
    """Rebuild better-sqlite3 against Cursor's bundled Node.

    Designed for agent self-healing: when puppetmaster_codegraph_status
    surfaces the native-SQLite-broken hint, the agent can call this tool
    directly with no arguments and get a structured success/failure
    payload back. We delegate to ``codegraph_repair.repair_codegraph_sqlite``
    so the CLI command and the MCP tool share identical behaviour.
    """
    verify_cwd = args.get("verify_cwd")
    if not verify_cwd:
        verify_cwd = cwd(args)
    cursor_node = args.get("cursor_node")
    codegraph_install = args.get("codegraph_install")
    npm_command = args.get("npm_command") or "npm"
    rebuild_timeout = args.get("rebuild_timeout_seconds") or 180
    verify = bool(args.get("verify", True))
    result = repair_codegraph_sqlite(
        cursor_node=cursor_node if isinstance(cursor_node, str) and cursor_node else None,
        codegraph_install=(
            codegraph_install
            if isinstance(codegraph_install, str) and codegraph_install
            else None
        ),
        npm_command=str(npm_command),
        rebuild_timeout_seconds=int(rebuild_timeout),
        verify=verify,
        verify_cwd=str(verify_cwd) if verify_cwd else None,
    )
    payload = result.to_payload()
    payload.setdefault("command", "npm rebuild better-sqlite3")
    return codegraph_response(payload)


def run_mcp_status(args: JsonObject) -> JsonObject:
    """Report every tracked Puppetmaster MCP server on this machine.

    The payload mirrors the CLI output so an agent can render it
    directly. We always prune dead tracking files first so the answer
    is fresh — there's no point reporting servers we already know are
    gone.
    """
    try:
        cleaned = registry_prune_dead()
    except Exception:
        cleaned = []
    snapshot = registry_summarize(registry_list_entries())
    snapshot["self_pid"] = os.getpid()
    snapshot["pruned_dead"] = [entry.to_payload() for entry in cleaned]
    snapshot["ok"] = True
    return _mcp_diagnostic_response(snapshot)


def run_mcp_cleanup(args: JsonObject) -> JsonObject:
    """Prune dead tracking files and optionally kill stale-but-alive MCP servers.

    Defensive defaults: ``kill_stale`` is opt-in, the current PID is
    never signalled, and we surface the before/after snapshots so the
    caller can see exactly what changed.
    """
    before = registry_summarize(registry_list_entries())
    try:
        pruned = registry_prune_dead()
    except Exception as exc:
        return _mcp_diagnostic_response(
            {
                "ok": False,
                "error": f"prune_dead failed: {exc}",
                "before": before,
            }
        )
    killed: list = []
    if bool(args.get("kill_stale", False)):
        stale_after = float(args.get("stale_after_seconds") or 300)
        try:
            killed_entries = registry_kill_stale(
                stale_after_seconds=stale_after,
                self_pid=os.getpid(),
            )
        except Exception as exc:
            return _mcp_diagnostic_response(
                {
                    "ok": False,
                    "error": f"kill_stale failed: {exc}",
                    "before": before,
                    "pruned": [entry.to_payload() for entry in pruned],
                }
            )
        killed = [entry.to_payload() for entry in killed_entries]
    after = registry_summarize(registry_list_entries())
    return _mcp_diagnostic_response(
        {
            "ok": True,
            "before": before,
            "after": after,
            "pruned": [entry.to_payload() for entry in pruned],
            "killed": killed,
            "self_pid": os.getpid(),
        }
    )


def _mcp_diagnostic_response(payload: JsonObject) -> JsonObject:
    """Wrap an MCP registry payload in the standard tool-result envelope."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": not payload.get("ok", False),
    }


def _spawn_codegraph_indexer(args: JsonObject, target_cwd: str) -> JsonObject:
    """Launch `codegraph index` as a detached background process.

    Returns a JSON payload with the run metadata. Acquires the
    per-repo indexer lock first — if another indexer is already
    running against the **same repo's** SQLite DB, we fail fast
    instead of stacking work onto a broken SQLite. Different repos
    (post-v0.5.5) run in parallel because they have separate DBs.
    The lock is released by an atexit hook in the launched process;
    a stale lock left by a killed indexer is auto-cleared on the
    next acquire by the staleness check.
    """
    lock_path = codegraph_lock_path(target_cwd)
    try:
        lock = acquire_codegraph_lock(lock_path=lock_path)
    except CodegraphLockBusy as exc:
        return {
            "ok": False,
            "command": "codegraph index",
            "cwd": target_cwd,
            "error": str(exc),
            "lock_path": str(exc.lock_path),
            "holder_pid": exc.holder_pid,
        }
    # We acquired the lock just to validate availability; the background
    # process will re-acquire and hold it for its lifetime. Release the
    # parent's handle now so the child can take ownership cleanly.
    lock.release()

    state_dir = str(mcp_state_dir(args))
    run_dir = Path(state_dir) / "mcp-runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"codegraph_index_{int(time.time() * 1000)}_{os.getpid()}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
    launcher = [
        sys.executable,
        "-m",
        "puppetmaster.codegraph_index_runner",
        target_cwd,
        str(lock_path),
    ]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        launcher,
        cwd=target_cwd,
        env=launcher_environment(args),
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    ASYNC_PROCESSES.append(process)
    stdout_handle.close()
    stderr_handle.close()
    return {
        "ok": True,
        "command": "codegraph index",
        "cwd": target_cwd,
        "run_id": run_id,
        "pid": process.pid,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "lock_path": str(lock_path),
        "next_steps": [
            "Call puppetmaster_codegraph_status to watch index progress.",
            f"Tail {stdout_path} for raw indexer output if status looks stuck.",
        ],
    }


def codegraph_response(payload: JsonObject) -> JsonObject:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": not payload.get("ok", False),
    }


def run_cursor(args: JsonObject, review: bool = False, plan: bool = False) -> JsonObject:
    return run_cli(cursor_command(args, review=review, plan=plan), args)


def start_cursor(args: JsonObject, review: bool = False, plan: bool = False) -> JsonObject:
    return start_cli(cursor_command(args, review=review, plan=plan), args)


def cursor_command(args: JsonObject, review: bool = False, plan: bool = False) -> list[str]:
    goal = require_string(args, "goal")
    command = ["cursor", goal, "--cwd", cwd(args), "--dry-run"]
    if review:
        command.append("--review")
    if plan:
        command.append("--plan")
    model = args.get("model")
    if model:
        command.extend(["--model", str(model)])
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds:
        command.extend(["--timeout-seconds", str(timeout_seconds)])
    return command


def run_claude(args: JsonObject) -> JsonObject:
    return run_cli(claude_command(args), args)


def start_claude(args: JsonObject) -> JsonObject:
    return start_cli(claude_command(args), args)


def claude_command(args: JsonObject) -> list[str]:
    goal = require_string(args, "goal")
    command = [
        "claude",
        goal,
        "--cwd",
        cwd(args),
        "--permission-mode",
        str(args.get("permission_mode") or "acceptEdits"),
    ]
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("allow_dirty"):
        command.append("--allow-dirty")
    return command


def run_openai(args: JsonObject) -> JsonObject:
    return run_cli(openai_command(args), args)


def start_openai(args: JsonObject) -> JsonObject:
    return start_cli(openai_command(args), args)


def openai_command(args: JsonObject) -> list[str]:
    goal = require_string(args, "goal")
    command = ["openai", goal, "--cwd", cwd(args)]
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("openai_base_url"):
        command.extend(["--base-url", str(args["openai_base_url"])])
    if args.get("openai_organization"):
        command.extend(["--organization", str(args["openai_organization"])])
    if args.get("max_output_tokens") is not None:
        command.extend(["--max-output-tokens", str(int(args["max_output_tokens"]))])
    if args.get("legacy_max_tokens"):
        command.append("--legacy-max-tokens")
    if args.get("temperature") is not None:
        command.extend(["--temperature", str(float(args["temperature"]))])
    if args.get("reasoning_effort"):
        command.extend(["--reasoning-effort", str(args["reasoning_effort"])])
    if args.get("disable_codegraph"):
        command.append("--disable-codegraph")
    return command


def start_swarm(args: JsonObject) -> JsonObject:
    goal = require_string(args, "goal")
    command = ["run", goal]
    roles = normalized_roles(args)
    adapter = args.get("adapter")
    if args.get("config"):
        command.extend(["--config", str(args["config"])])
    elif adapter:
        config_path = write_generated_swarm_config(args, roles or ["explore"], str(adapter))
        command.extend(["--config", str(config_path)])
    elif roles:
        if not args.get("allow_local_demo"):
            return tool_error(
                "Custom-role MCP swarms require a workflow config or adapter. "
                "Otherwise Puppetmaster would use the demo local adapter and return generic artifacts.",
                {
                    "roles": roles,
                    "fix": "Use puppetmaster_start_cursor_swarm, pass adapter='cursor', pass config, or set allow_local_demo=true for tests/demos.",
                },
            )
        command.append("--workers")
        command.extend(roles)
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    return start_cli(command, args)


def start_cursor_swarm(args: JsonObject) -> JsonObject:
    roles = normalized_roles(args) or [
        "pipeline-mapper",
        "decision-explainer",
        "conflict-auditor",
        "test-coverage-reviewer",
    ]
    config_path = write_generated_swarm_config(args, roles, "cursor")
    command = ["run", require_string(args, "goal"), "--config", str(config_path)]
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    return start_cli(command, args)


def normalized_roles(args: JsonObject) -> list[str]:
    roles = args.get("roles")
    if not isinstance(roles, list):
        return []
    return [str(role) for role in roles if str(role).strip()]


def write_generated_swarm_config(args: JsonObject, roles: list[str], adapter: str) -> Path:
    if adapter not in {"cursor", "local"}:
        raise ValueError(f"MCP swarm adapter is not supported yet: {adapter}")
    root = mcp_state_dir(args)
    config_dir = root / "mcp-configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"swarm_{int(time.time() * 1000)}_{os.getpid()}.json"
    goal = require_string(args, "goal")
    timeout_seconds = int(args.get("timeout_seconds") or 900)
    explicit_model = args.get("model")
    model = str(explicit_model or "default")
    # Auto-routing is ON by default for MCP swarms (matches DEFAULT_WORKERS and the
    # v0.6.0 docs). It is suppressed only if the caller (a) pinned a specific model
    # via the `model` arg, or (b) explicitly passed auto_route=false.
    auto_route_arg = args.get("auto_route")
    if auto_route_arg is not None:
        auto_route_enabled = bool(auto_route_arg)
    else:
        auto_route_enabled = not bool(explicit_model)
    routing_policy = args.get("routing_policy")
    max_cost_usd = args.get("max_cost_usd")
    min_capability = args.get("min_capability")
    required_tags = args.get("required_tags")
    workers = []
    for role in roles:
        prompt = (
            f"Role: {role}\n"
            f"Goal: {goal}\n\n"
            "Return structured findings with concrete file/function evidence. "
            "Do not modify files unless the user explicitly requested implementation. "
            "Return only Puppetmaster artifact JSON with an artifacts array."
        )
        payload: JsonObject = {"prompt": prompt, "cwd": cwd(args), "timeout_seconds": timeout_seconds}
        if adapter == "cursor":
            payload["model"] = model
        if auto_route_enabled:
            payload["auto_route"] = True
            if isinstance(routing_policy, str) and routing_policy:
                payload["routing_policy"] = routing_policy
            if isinstance(max_cost_usd, (int, float)):
                payload["max_cost_usd"] = float(max_cost_usd)
            if isinstance(min_capability, int):
                payload["min_capability"] = int(min_capability)
            if isinstance(required_tags, list) and required_tags:
                payload["required_tags"] = [str(tag) for tag in required_tags if str(tag).strip()]
        workers.append(
            {
                "role": role,
                "instruction": prompt,
                "adapter": adapter,
                "payload": payload,
            }
        )
    config_path.write_text(json.dumps({"lease_seconds": 10, "workers": workers}, indent=2), encoding="utf-8")
    return config_path


def route_task_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "instruction": {
                "type": "string",
                "description": "Task instruction text. Drives the capability classifier.",
            },
            "role": {
                "type": "string",
                "description": (
                    "Task role (e.g. explore, architect, implement, audit). "
                    "Drives the base capability score. Defaults to 'explore'."
                ),
                "default": "explore",
            },
            "policy": {
                "type": "string",
                "enum": ["balanced", "cheap", "quality", "escalating"],
                "default": "balanced",
                "description": (
                    "Routing policy. balanced=cheapest sufficient; cheap=lowest cost; "
                    "quality=highest capability; escalating=ordered chain for retries."
                ),
            },
            "min_capability": {
                "type": "integer",
                "description": "Force classifier output to this value (0..100).",
            },
            "max_cost_usd": {
                "type": "number",
                "description": "Hard cap on estimated per-call USD cost.",
            },
            "required_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only consider models whose tags include ALL of these.",
            },
            "registry_path": {
                "type": "string",
                "description": (
                    "Optional override for the models registry path. Defaults to "
                    "$PUPPETMASTER_MODELS_PATH or ~/.puppetmaster/models.json."
                ),
            },
        }
    )
    schema["required"] = ["instruction"]
    return schema


def list_models_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"]["registry_path"] = {
        "type": "string",
        "description": (
            "Optional override for the models registry path. Defaults to "
            "$PUPPETMASTER_MODELS_PATH or ~/.puppetmaster/models.json."
        ),
    }
    return schema


def run_route_task(args: JsonObject) -> JsonObject:
    """Run the router and return the decision as a JSON content block."""
    from puppetmaster.model_registry import default_registry_path, load_registry
    from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

    instruction = require_string(args, "instruction")
    role = (args.get("role") or "explore").strip() or "explore"
    policy = (args.get("policy") or "balanced").strip() or "balanced"
    registry_path_arg = args.get("registry_path")
    registry_path = (
        Path(str(registry_path_arg)).expanduser()
        if registry_path_arg
        else default_registry_path()
    )

    try:
        specs = load_registry(registry_path)
    except RuntimeError as exc:
        return {
            "content": [{"type": "text", "text": f"registry error: {exc}"}],
            "isError": True,
        }
    if not specs:
        msg = (
            f"No models registered at {registry_path}. "
            "Run `puppetmaster models init` to write a starter registry, "
            "then edit capability scores + prices to match your subscriptions."
        )
        return {"content": [{"type": "text", "text": msg}], "isError": True}

    signals = TaskSignals(
        instruction=instruction,
        role=role,
        explicit_min_capability=(
            int(args["min_capability"]) if "min_capability" in args and args["min_capability"] is not None else None
        ),
        explicit_max_cost_usd=(
            float(args["max_cost_usd"]) if "max_cost_usd" in args and args["max_cost_usd"] is not None else None
        ),
        required_tags=list(args.get("required_tags") or []),
    )

    try:
        decision = route_task(signals, specs, policy=policy)
    except NoEligibleModelError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    except ValueError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}

    payload = decision.to_artifact_payload()
    payload["registry_path"] = str(registry_path)
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def run_list_models(args: JsonObject) -> JsonObject:
    """Return the registry as JSON. Mirrors `puppetmaster models list --json`."""
    from dataclasses import asdict

    from puppetmaster.model_registry import default_registry_path, load_registry

    registry_path_arg = args.get("registry_path")
    registry_path = (
        Path(str(registry_path_arg)).expanduser()
        if registry_path_arg
        else default_registry_path()
    )
    try:
        specs = load_registry(registry_path)
    except RuntimeError as exc:
        return {
            "content": [{"type": "text", "text": f"registry error: {exc}"}],
            "isError": True,
        }
    payload = {
        "registry_path": str(registry_path),
        "models": [asdict(s) for s in specs],
    }
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def run_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(mcp_state_dir(args))
    process = subprocess.run(
        [sys.executable, "-m", "puppetmaster", "--state-dir", state_dir] + command,
        cwd=cwd(args),
        env=launcher_environment(args),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=int(args.get("runner_timeout_seconds") or 1800),
    )
    body = {
        "command": "python -m puppetmaster " + " ".join(command),
        "cwd": cwd(args),
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
        "isError": process.returncode != 0,
    }


def run_feed(args: JsonObject) -> JsonObject:
    command = ["feed", require_string(args, "job_id"), "--json"]
    if args.get("limit"):
        command.extend(["--limit", str(args["limit"])])
    return run_cli(command, args)


def run_feed_follow(args: JsonObject) -> JsonObject:
    from puppetmaster.cli import artifact_feed_since

    job_id = require_string(args, "job_id")
    since = int(args.get("since_cursor") or args.get("since") or 0)
    timeout_seconds = float(args.get("timeout_seconds") or 10.0)
    poll_interval = float(args.get("poll_interval_seconds") or 0.1)
    limit_value = args.get("limit")
    limit = int(limit_value) if limit_value is not None else None
    backend = str(args.get("backend") or "sqlite")

    state_dir = mcp_state_dir(args)
    store = create_store(backend, state_dir)

    items, cursor = artifact_feed_since(store, job_id, since=since, limit=limit)
    if not items:
        store.wait_for_events(
            job_id,
            since=cursor,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        )
        items, cursor = artifact_feed_since(store, job_id, since=since, limit=limit)

    body = {
        "job_id": job_id,
        "since_cursor": since,
        "next_cursor": cursor,
        "item_count": len(items),
        "items": items,
        "timed_out": len(items) == 0,
    }
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2, default=str)}], "isError": False}


def start_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(mcp_state_dir(args))
    run_dir = Path(state_dir) / "mcp-runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"mcp_{int(time.time() * 1000)}_{os.getpid()}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
    full_command = [
        sys.executable,
        "-m",
        "puppetmaster",
        "--state-dir",
        state_dir,
        "--emit-job-id-early",
    ] + command
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        full_command,
        cwd=cwd(args),
        env=launcher_environment(args),
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    ASYNC_PROCESSES.append(process)
    stdout_handle.close()
    stderr_handle.close()
    job_id = wait_for_job_id(stdout_path, stderr_path, process, timeout_seconds=5)
    body = {
        "run_id": run_id,
        "job_id": job_id,
        "pid": process.pid,
        "command": "python -m puppetmaster " + " ".join(command),
        "cwd": cwd(args),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "next_steps": [
            f"Call puppetmaster_status with job_id={job_id}",
            f"Call puppetmaster_logs with job_id={job_id}",
            f"Call puppetmaster_show with job_id={job_id} after completion",
        ],
    }
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2)}], "isError": False}


def wait_for_job_id(
    stdout_path: Path,
    stderr_path: Path,
    process: subprocess.Popen,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    pattern = re.compile(r"job_id:\s*(job_[A-Za-z0-9]+)")
    while time.monotonic() < deadline:
        if process.poll() is not None and not stdout_path.exists():
            break
        if stdout_path.exists():
            text = stdout_path.read_text(encoding="utf-8")
            match = pattern.search(text)
            if match:
                return match.group(1)
            if process.poll() is not None:
                break
        time.sleep(0.05)
    stderr = stderr_path.read_text(encoding="utf-8")[-1000:] if stderr_path.exists() else ""
    raise RuntimeError(
        f"started Puppetmaster process but did not receive early job_id; "
        f"pid={process.pid}; returncode={process.poll()}; stderr={stderr}"
    )


def launcher_environment(args: JsonObject) -> dict[str, str]:
    env = environment(args)
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else source_root
    )
    return env


def environment(args: JsonObject) -> dict[str, str]:
    env = os.environ.copy()
    if args.get("cursor_api_key"):
        env["CURSOR_API_KEY"] = str(args["cursor_api_key"])
    if args.get("anthropic_api_key"):
        env["ANTHROPIC_API_KEY"] = str(args["anthropic_api_key"])
    if args.get("claude_code_command"):
        env["CLAUDE_CODE_COMMAND"] = str(args["claude_code_command"])
    if args.get("openai_api_key"):
        env["OPENAI_API_KEY"] = str(args["openai_api_key"])
    if args.get("openai_base_url"):
        env["OPENAI_BASE_URL"] = str(args["openai_base_url"])
    if args.get("openai_organization"):
        env["OPENAI_ORG_ID"] = str(args["openai_organization"])
    return env


def cwd(args: JsonObject) -> str:
    return str(args.get("cwd") or os.getcwd())


def mcp_state_dir(args: JsonObject) -> Path:
    value = args.get("state_dir")
    return resolve_state_dir(str(value) if value else None, cwd=Path(cwd(args)))


def require_string(args: JsonObject, name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def optional_job(args: JsonObject) -> list[str]:
    job_id = args.get("job_id")
    return [str(job_id)] if job_id else []


def base_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "cwd": {"type": "string", "description": "Workspace/repository path."},
            "state_dir": {
                "type": "string",
                "description": (
                    "Optional Puppetmaster state directory, relative to cwd unless absolute. "
                    "Defaults to per-workspace app state outside the repository."
                ),
            },
            "runner_timeout_seconds": {
                "type": "integer",
                "description": "Maximum time to wait for the local Puppetmaster process.",
            },
        },
    }


def job_schema(required: bool = False) -> JsonObject:
    schema = base_schema()
    schema["properties"]["job_id"] = {"type": "string", "description": "Puppetmaster job id."}
    if required:
        schema["required"] = ["job_id"]
    return schema


def feed_schema() -> JsonObject:
    schema = job_schema(required=True)
    schema["properties"]["limit"] = {
        "type": "integer",
        "description": "Limit feed to the most recent N artifacts.",
    }
    return schema


def follow_schema() -> JsonObject:
    schema = job_schema(required=True)
    schema["properties"].update(
        {
            "since_cursor": {
                "type": "integer",
                "default": 0,
                "description": "Resume from this event cursor (use the previous next_cursor).",
            },
            "timeout_seconds": {
                "type": "number",
                "default": 10,
                "description": "Maximum seconds to block waiting for new artifacts.",
            },
            "poll_interval_seconds": {
                "type": "number",
                "default": 0.1,
                "description": "Internal poll interval; lower = lower latency, higher = lighter on storage.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional cap on the number of artifacts returned in one batch.",
            },
            "backend": {
                "type": "string",
                "enum": ["file", "sqlite"],
                "default": "sqlite",
                "description": "Coordination backend to open. Match the one used by your jobs.",
            },
        }
    )
    return schema


def goal_schema(default_goal: str) -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "goal": {
                "type": "string",
                "description": "Goal/prompt to send to the worker.",
                "default": default_goal,
            },
            "model": {"type": "string", "description": "Optional provider model name."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Worker timeout passed to the adapter.",
            },
            "cursor_api_key": {
                "type": "string",
                "description": "Optional Cursor API key. Prefer MCP env config instead.",
            },
        }
    )
    schema["required"] = ["goal"]
    return schema


def swarm_schema() -> JsonObject:
    schema = goal_schema("Review this repo and produce structured artifacts.")
    schema["properties"].update(
        {
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local worker roles to run.",
            },
            "config": {"type": "string", "description": "Optional workflow config path."},
            "adapter": {
                "type": "string",
                "enum": ["cursor", "local"],
                "description": "Adapter to use for generated role configs. Required for custom roles unless config or allow_local_demo is set.",
            },
            "allow_local_demo": {
                "type": "boolean",
                "default": False,
                "description": "Allow custom roles to use the deterministic local demo adapter.",
            },
            "worker_mode": {
                "type": "string",
                "enum": ["subprocess", "inline", "daemon"],
                "default": "subprocess",
                "description": "Worker execution mode.",
            },
            "auto_route": {
                "type": "boolean",
                "description": (
                    "Enable per-task model routing. Defaults to true when no `model` is pinned, "
                    "false otherwise. Pass true to force-route even with a pinned model, or false to disable."
                ),
            },
            "routing_policy": {
                "type": "string",
                "enum": ["balanced", "cheap", "quality", "escalating"],
                "description": "Routing policy for auto-routed workers. Defaults to 'balanced'.",
            },
            "max_cost_usd": {
                "type": "number",
                "description": "Hard cap on estimated per-call USD cost for auto-routed workers.",
            },
            "min_capability": {
                "type": "integer",
                "description": "Force classifier output to this value (0..100) for auto-routed workers.",
            },
            "required_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only consider models whose tags include ALL of these for auto-routed workers.",
            },
        }
    )
    return schema


def cursor_swarm_schema() -> JsonObject:
    schema = swarm_schema()
    schema["properties"].pop("adapter", None)
    schema["properties"].pop("allow_local_demo", None)
    schema["properties"]["worker_mode"]["default"] = "subprocess"
    return schema


def codegraph_search_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "query": {
                "type": "string",
                "description": "Symbol name or substring to search the local CodeGraph index for.",
            },
            "kind": {
                "type": "string",
                "description": "Optional symbol kind filter (e.g. function, class, method, route).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches to return.",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    schema["required"] = ["query"]
    return schema


def codegraph_context_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "task": {
                "type": "string",
                "description": "Natural-language task description; CodeGraph returns relevant entry points and symbols.",
            },
            "max_nodes": {
                "type": "integer",
                "default": 15,
                "description": "Upper bound on graph nodes returned in the context bundle.",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "json", "text"],
                "default": "markdown",
                "description": "Output format for the context bundle.",
            },
        }
    )
    schema["required"] = ["task"]
    return schema


def codegraph_affected_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Changed source file paths (relative to cwd) whose tests should be discovered.",
            },
            "depth": {
                "type": "integer",
                "description": "Max dependency traversal depth (CodeGraph default: 5).",
            },
            "filter": {
                "type": "string",
                "description": "Custom glob used to identify test files (e.g. 'tests/**/*.py').",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    schema["required"] = ["files"]
    return schema


def codegraph_files_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "path": {
                "type": "string",
                "description": "Optional sub-path to scope the file listing.",
            },
            "format": {
                "type": "string",
                "description": "CodeGraph format option for the listing (e.g. tree, list).",
            },
            "filter": {
                "type": "string",
                "description": "Glob filter applied to the listing.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth to display.",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    return schema


def codegraph_init_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"]["index"] = {
        "type": "boolean",
        "default": False,
        "description": "If true, run a full index immediately after initialization.",
    }
    return schema


def mcp_cleanup_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "kill_stale": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, SIGTERM/SIGKILL Puppetmaster MCP servers whose "
                    "heartbeat is older than stale_after_seconds. The current "
                    "process is never signalled."
                ),
            },
            "stale_after_seconds": {
                "type": "integer",
                "default": 300,
                "description": "Heartbeat age in seconds beyond which a server is considered stale.",
            },
        },
    }


def repair_codegraph_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "cursor_node": {
                "type": "string",
                "description": (
                    "Path to Cursor's bundled Node binary. Auto-detected from common "
                    "macOS/Linux/Windows install locations when omitted."
                ),
            },
            "codegraph_install": {
                "type": "string",
                "description": (
                    "Path to the global @colbymchenry/codegraph install directory. "
                    "Resolved from `npm root -g` when omitted."
                ),
            },
            "npm_command": {
                "type": "string",
                "default": "npm",
                "description": "npm binary to invoke (default: npm).",
            },
            "rebuild_timeout_seconds": {
                "type": "integer",
                "default": 180,
                "description": "Hard timeout for the npm rebuild step.",
            },
            "verify": {
                "type": "boolean",
                "default": True,
                "description": "Run `codegraph status` under Cursor's Node after the rebuild.",
            },
            "verify_cwd": {
                "type": "string",
                "description": (
                    "Workspace to run the verification `codegraph status` in. "
                    "Defaults to the cwd argument or process cwd."
                ),
            },
        }
    )
    return schema


def claude_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change and run focused tests.")
    schema["properties"].update(
        {
            "permission_mode": {
                "type": "string",
                "default": "acceptEdits",
                "description": "Claude Code permission mode.",
            },
            "allow_dirty": {
                "type": "boolean",
                "default": False,
                "description": "Allow Claude Code to run in a dirty working tree.",
            },
            "anthropic_api_key": {
                "type": "string",
                "description": "Optional Anthropic API key. Prefer MCP env config instead.",
            },
            "claude_code_command": {
                "type": "string",
                "description": "Optional Claude Code command, such as npx -y @anthropic-ai/claude-code.",
            },
        }
    )
    return schema


def openai_schema() -> JsonObject:
    schema = goal_schema("Produce structured findings/risks/decisions for the requested task.")
    schema["properties"].update(
        {
            "openai_api_key": {
                "type": "string",
                "description": "Optional OpenAI API key. Prefer MCP env config instead.",
            },
            "openai_base_url": {
                "type": "string",
                "description": "Override the OpenAI base URL (e.g. for OpenAI-compatible providers).",
            },
            "openai_organization": {
                "type": "string",
                "description": "Optional OpenAI organization id.",
            },
            "max_output_tokens": {
                "type": "integer",
                "description": "Cap on completion tokens. Off by default to let the model finish.",
            },
            "legacy_max_tokens": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Send the deprecated `max_tokens` field instead of "
                    "`max_completion_tokens`. Use for OpenAI-compatible providers."
                ),
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature override (only sent if provided).",
            },
            "reasoning_effort": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "xhigh"],
                "description": "Reasoning effort level for GPT-5+ models.",
            },
            "disable_codegraph": {
                "type": "boolean",
                "default": False,
                "description": "Skip CodeGraph context injection (e.g. for non-repo prompts).",
            },
            "worker_mode": {
                "type": "string",
                "enum": ["subprocess", "inline", "daemon"],
                "default": "inline",
                "description": "Worker execution mode.",
            },
        }
    )
    return schema


def tool_to_json(tool: McpTool) -> JsonObject:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def tool_error(message: str, payload: Optional[JsonObject] = None) -> JsonObject:
    body: JsonObject = {"error": message}
    if payload:
        body.update(payload)
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2)}], "isError": True}


def error_response(request_id: Any, code: int, message: str) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
