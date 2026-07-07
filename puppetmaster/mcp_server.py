from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.codegraph import (
    CODEGRAPH_MISSING_HINT,
    CODEGRAPH_NATIVE_SQLITE_HINT,
    CodegraphLockBusy,
    acquire_codegraph_lock,
    codegraph_affected,
    codegraph_available,
    codegraph_context_command,
    codegraph_files_listing,
    codegraph_freshness,
    codegraph_init_command,
    codegraph_lock_path,
    codegraph_native_sqlite_broken,
    codegraph_query,
    codegraph_status_command,
    maybe_autosync_codegraph,
)
from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.mcp_registry import (
    HeartbeatThread,
    deregister as registry_deregister,
    installed_puppetmaster_version,
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    register as registry_register,
    summarize as registry_summarize,
)
from puppetmaster.state import resolve_state_dir
from puppetmaster.store_factory import create_store
from puppetmaster.update_check import pypi_update_note, version_is_newer

# Snapshot the version this server process loaded at startup. A long-lived stdio
# MCP server keeps these modules in memory; an in-place `pip install -U` changes
# what's on disk but cannot reload us, so this constant stays the running value
# while `installed_puppetmaster_version()` reflects the disk. Comparing them is
# how we turn silent staleness into a visible per-call nudge.
from puppetmaster import __version__ as _SERVER_RUNNING_VERSION


JsonObject = dict[str, Any]

# Detached job/index launchers spawned with ``start_new_session=True``. The
# request path returns the moment a launcher reports its job id, so nothing
# inline ever waits on these children. Without an out-of-band reaper, each one
# becomes a ``<defunct>`` zombie parented to this server the instant it exits —
# an unbounded leak that, over a long Cursor session, can exhaust the PID table.
# ``_ASYNC_PROCESSES_LOCK`` guards the list because tool handlers append from
# pool threads while the reaper thread sweeps it.
ASYNC_PROCESSES: list[subprocess.Popen] = []
_ASYNC_PROCESSES_LOCK = threading.Lock()
_DEFAULT_REAP_INTERVAL_SECONDS = 15.0

# The JSON-RPC frame stream. ``main()`` calls ``_isolate_protocol_stdout`` to
# dup the real stdout fd into ``_PROTOCOL_STREAM`` and repoint fd 1 at stderr.
# From then on, only frame writes reach the MCP client; a stray ``print()``
# anywhere in this process (or in a child that inherits fd 1) lands in stderr
# instead. That distinction is load-bearing for OpenAI Codex: its rmcp client
# treats the first non-JSON-RPC byte on the pipe as a fatal transport error
# and never reconnects, while Cursor's client just skips unparseable lines.
# Outside ``main()`` (unit tests calling handlers directly) the stream stays
# None and frame writes fall back to ``sys.stdout``.
_PROTOCOL_STREAM: Optional[Any] = None


def _protocol_stream() -> Any:
    return _PROTOCOL_STREAM if _PROTOCOL_STREAM is not None else sys.stdout


def _isolate_protocol_stdout() -> None:
    """Reserve the real stdout for JSON-RPC frames; divert fd 1 to stderr.

    Best-effort: if the dup dance fails (exotic hosts, closed fds), keep
    serving on ``sys.stdout`` exactly as before rather than refusing to start.
    """
    global _PROTOCOL_STREAM
    if _PROTOCOL_STREAM is not None:
        return
    try:
        protocol_fd = os.dup(sys.stdout.fileno())
        stream = os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (OSError, ValueError) as exc:
        print(
            f"puppetmaster-mcp: stdout isolation unavailable ({exc}); "
            "stray prints may corrupt the protocol stream under Codex.",
            file=sys.stderr,
        )
        return
    _PROTOCOL_STREAM = stream


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
#
# Idle reaping is HOST-GATED: it only solves a Cursor problem, and only
# Cursor transparently respawns a reaped server on the next call. Claude
# Code and Codex mark the connection failed and stay dead until the user
# restarts their session (field reports: Codex "Transport closed", Claude
# Code "Connection status failed for some reason" after an idle gap). On
# those hosts — and on unknown hosts — a leaked-but-alive server is far
# cheaper than a dead transport, so idle reaping stays off. True orphans
# on every host are still caught by the parent-death check, which fires
# when this process is reparented to init because the spawning host died.
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

# Maximum seconds any single tool handler may *block* the MCP turn.
#
# Cursor's MCP client tolerates arbitrarily long tool calls as long as
# bytes keep flowing (the keepalive above). OpenAI Codex does NOT: its
# MCP client enforces a hard per-tool `tool_timeout_sec` (default 60s)
# that cancels the request regardless of progress/log notifications. A
# long-poll handler (`live_artifacts_follow`, `await_job`) that the
# caller asked to block for, say, 300s therefore reads as a dead tool on
# Codex even though the swarm is healthy — the exact "can't keep
# puppetmaster alive" symptom.
#
# We make long-poll handlers Codex-safe by capping their effective block
# to this budget, which sits comfortably under Codex's 60s default with
# margin for import/serialization overhead. The follow/await tools are
# *designed* to be re-called (they return `next_cursor` / `timed_out`),
# so capping just turns one 300s block into a sequence of short polls —
# the push-feeling stream still works, and it now works everywhere. The
# response is stamped `capped: true` whenever the requested block was
# shortened so callers (and tests) can see it happened. Tune or disable
# with PUPPETMASTER_MCP_MAX_BLOCK_SECONDS (0 = no cap).
_DEFAULT_MAX_BLOCK_SECONDS = 45.0

# Module-level state for stdin-liveness tracking. Initialized at startup
# by ``main()`` and mutated by the stdin reader + tool-call dispatcher.
_INPUT_STATE_LOCK = threading.Lock()
_LAST_INBOUND_MESSAGE_AT = time.time()
_ACTIVE_TOOL_CALLS = 0
_SHUTDOWN_REQUESTED = threading.Event()

# Client identity from the MCP `initialize` handshake (params.clientInfo).
# Codex sends ``{"name": "codex-mcp-client", "title": "Codex", ...}``; Cursor
# and Claude Code send their own names. This is the reliable, in-protocol way
# to know which host is driving us — Codex scrubs ``CODEX_*`` env vars from the
# MCP server it spawns, so env sniffing alone misses it.
_CLIENT_INFO: dict = {}


def _reap_async_processes_locked() -> int:
    """Poll every tracked launcher and drop the ones that have exited.

    ``Popen.poll()`` waitpid-reaps an exited child, clearing its zombie slot.
    The caller must hold ``_ASYNC_PROCESSES_LOCK``. Returns the count reaped.
    """
    survivors: list[subprocess.Popen] = []
    reaped = 0
    for process in ASYNC_PROCESSES:
        try:
            exited = process.poll() is not None
        except Exception:
            # A process we can no longer query is not worth tracking; treat it
            # as gone rather than holding a reference that never clears.
            exited = True
        if exited:
            reaped += 1
        else:
            survivors.append(process)
    ASYNC_PROCESSES[:] = survivors
    return reaped


def _reap_async_processes() -> int:
    with _ASYNC_PROCESSES_LOCK:
        return _reap_async_processes_locked()


def _track_async_process(process: subprocess.Popen) -> None:
    """Register a detached launcher for reaping.

    Sweeps already-exited launchers in the same critical section so a burst of
    short jobs can't pile up zombies between reaper ticks.
    """
    with _ASYNC_PROCESSES_LOCK:
        ASYNC_PROCESSES.append(process)
        _reap_async_processes_locked()


def _resolve_reap_interval() -> float:
    raw = os.environ.get("PUPPETMASTER_MCP_REAP_INTERVAL_SECONDS")
    if not raw:
        return _DEFAULT_REAP_INTERVAL_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_REAP_INTERVAL_SECONDS


class _AsyncProcessReaper(threading.Thread):
    """Daemon thread that waitpid-reaps detached job/index launchers.

    Each async job and codegraph index is spawned as a detached child and the
    request path returns immediately, so nothing inline waits on them. On exit
    they become ``<defunct>`` zombies parented to this server. This thread
    sweeps the tracked set on a fixed interval (and once more on shutdown) so
    exited launchers are reaped instead of accumulating.
    """

    def __init__(self, interval_seconds: float = _DEFAULT_REAP_INTERVAL_SECONDS) -> None:
        super().__init__(daemon=True, name="puppetmaster-mcp-process-reaper")
        self._interval = max(1.0, interval_seconds)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        # A final sweep so a launcher that finished during shutdown doesn't
        # outlive us as a zombie handed back to init.
        try:
            _reap_async_processes()
        except (OSError, ChildProcessError):
            pass

    def run(self) -> None:  # pragma: no cover - timing covered via direct tests
        while not self._stop.wait(self._interval):
            try:
                _reap_async_processes()
            except (OSError, ChildProcessError):
                pass


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


def _extract_progress_token(params: Any) -> Any:
    """Pull ``params._meta.progressToken`` from a ``tools/call`` request.

    Per the MCP spec a client opts into progress notifications by sending a
    ``progressToken`` (string or integer) inside ``params._meta``. When the
    client omits it we return ``None`` and the keepalive sends no progress
    frames — only clients that asked for progress receive it.
    """
    if not isinstance(params, dict):
        return None
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("progressToken")


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
            stream = _protocol_stream()
            stream.write(serialized)
            stream.flush()
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

    Two complementary signals are emitted each tick:

    * ``notifications/message`` (debug level) — a transport-liveness ping
      that Cursor's client treats as "the pipe is alive". Always sent.
    * ``notifications/progress`` — the MCP-spec mechanism for telling a
      client a specific in-flight request is still making progress. Only
      sent when the originating ``tools/call`` supplied a
      ``params._meta.progressToken``. Spec-compliant clients (OpenAI
      Codex/rmcp among them) use this to keep a long call from tripping
      their hard per-tool timeout — the gap that made Codex read healthy
      swarms as a dead tool. Cursor ignores it; Codex needs it.

    Both intentionally carry no ``id`` field so clients don't try to match
    them against an outstanding request.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        request_id: Any,
        progress_token: Any = None,
        interval_seconds: Optional[float] = None,
        start_after_seconds: Optional[float] = None,
        emitter: Callable[[JsonObject], bool] = _emit_notification,
    ) -> None:
        self._tool_name = tool_name or "unknown"
        self._request_id = request_id
        self._progress_token = progress_token
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
        if self._stop.is_set():
            return False
        message = {
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
        ok = self._emit(message)
        if not ok:
            return False
        self._emitted += 1
        # Spec-compliant progress for clients (Codex) that reset their
        # per-tool timeout on it. `progress` must strictly increase; we use
        # elapsed seconds and omit `total` because the duration is unknown.
        if self._progress_token is not None and not self._stop.is_set():
            progress = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": self._progress_token,
                    # MCP requires `progress` to strictly increase per token.
                    # `elapsed` grows by `interval` each tick; round only enough
                    # to drop float noise while preserving monotonicity.
                    "progress": round(elapsed, 3),
                    "message": (
                        f"Puppetmaster '{self._tool_name}' running "
                        f"({elapsed:.0f}s)"
                    ),
                },
            }
            self._emit(progress)
        return True


def _input_staleness_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_MCP_INPUT_STALE_DISABLED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _input_staleness_forced() -> bool:
    """``PUPPETMASTER_MCP_INPUT_STALE_FORCED=1`` re-enables idle reaping on
    hosts that don't transparently respawn (for users who'd rather restart
    their session than carry an idle server)."""
    raw = os.environ.get("PUPPETMASTER_MCP_INPUT_STALE_FORCED")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _host_transparently_respawns(env: Optional[dict] = None) -> bool:
    """True when the spawning host respawns a reaped MCP server on demand.

    Only Cursor does — its lease lifecycle both creates the orphan-server
    problem the idle reaper exists for and transparently respawns the server
    on the next tool call. Claude Code (``CLAUDECODE`` /
    ``CLAUDE_CODE_ENTRYPOINT``) and Codex mark the connection failed and stay
    dead, so an idle reap turns into a user-visible outage on those hosts.
    Detection is deliberately biased toward "no": any Claude/Codex marker
    wins over Cursor markers, and an unrecognized host never idle-reaps.
    """
    environ = env if env is not None else dict(os.environ)
    if environ.get("CLAUDECODE") or environ.get("CLAUDE_CODE_ENTRYPOINT"):
        return False
    if any(key.startswith("CODEX_") for key in environ):
        return False
    return any(key.startswith("CURSOR_") for key in environ)


def _client_is_codex(client_info: Optional[dict] = None) -> bool:
    """True when the MCP `initialize` handshake identifies the client as Codex.

    Codex's rmcp client sends ``clientInfo = {"name": "codex-mcp-client",
    "title": "Codex", ...}``. This is the reliable signal because Codex spawns
    the MCP server with a scrubbed environment (no ``CODEX_*`` vars), so env
    sniffing on the server side misses it.
    """
    info = client_info if client_info is not None else _CLIENT_INFO
    if not isinstance(info, dict):
        return False
    name = str(info.get("name") or "").lower()
    title = str(info.get("title") or "").lower()
    return "codex" in name or title == "codex"


def _host_enforces_tool_timeout(env: Optional[dict] = None) -> bool:
    """True when the spawning host hard-cancels a tool call after a fixed budget.

    OpenAI Codex (rmcp client) enforces a per-tool ``tool_timeout_sec`` (default
    60s, often 300s) and — unlike Cursor — does NOT reset it on our progress/log
    keepalives, because Codex never sends a ``progressToken`` for us to address
    progress to. A synchronous worker verb (cursor/claude/codex/openai
    review/plan/implement) routinely runs for minutes, so under Codex it gets
    cancelled mid-run and a healthy job reads as a failed tool.

    Primary signal is the ``initialize`` clientInfo (Codex identifies itself as
    ``codex-mcp-client``); the ``CODEX_*`` env markers are a fallback used only
    when the handshake identity hasn't been captured yet, so a known non-Codex
    client (Cursor/Claude Code) is never misread from stray env vars. Cursor and
    Claude Code don't impose this hard cap, so they keep inline-result behavior.
    """
    if _CLIENT_INFO:
        return _client_is_codex()
    environ = env if env is not None else dict(os.environ)
    return any(key.startswith("CODEX_") for key in environ)


def _sync_autodetach_disabled(env: Optional[dict] = None) -> bool:
    """``PUPPETMASTER_MCP_SYNC_AUTODETACH=0`` keeps synchronous worker verbs
    blocking inline even on a hard-timeout host — for users who raised their
    Codex ``tool_timeout_sec`` high enough to wait out a full worker run."""
    environ = env if env is not None else os.environ
    raw = environ.get("PUPPETMASTER_MCP_SYNC_AUTODETACH")
    if raw is None:
        return False
    return raw.strip().lower() in {"0", "false", "no", "off"}


def _should_autodetach_worker(args: JsonObject) -> bool:
    """Whether a synchronous worker verb should run async (return a job_id to
    poll) instead of blocking the MCP turn until the host kills the transport.

    Only fires on a host with a hard per-tool timeout (Codex), and only when the
    operator hasn't opted out. A per-call ``autodetach`` arg overrides both."""
    override = args.get("autodetach")
    if isinstance(override, bool):
        return override
    if _sync_autodetach_disabled():
        return False
    return _host_enforces_tool_timeout()


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

    All self-reaping is **host-gated** by ``reap_enabled`` (see ``main()``):
    on only — Cursor (or an explicit ``PUPPETMASTER_MCP_INPUT_STALE_FORCED``
    override). The reason is that the only authoritative owner-death signal
    for a stdio server is **stdin EOF**: the host holds the write end of our
    stdin, so when the host dies the pipe closes and the ``for line in
    sys.stdin`` loop in ``main()`` exits cleanly on its own — on every host,
    with no watcher involved. The watcher exists purely to clean up Cursor's
    *lease-cycle orphan*: a server Cursor abandons without closing stdin and
    then transparently respawns on the next call. A false reap there is
    invisible. On Claude Code / Codex a reaped server instead presents as a
    dead transport (Codex "Transport closed") until the user restarts, so on
    those hosts a leaked-but-alive server is far cheaper than a wrong reap —
    the watcher stays out of the way and stdin EOF does the cleanup.

    When reaping is enabled, two signals fire it:

    * **Parent death**: ``getppid() == 1`` — we were reparented to init.
    * **Idle staleness**: no stdin message in ``stale_after_seconds`` and
      zero in-flight tool calls.

    Neither is authoritative on its own. ``getppid() == 1`` in particular is
    a false positive when the host spawns us through an intermediate launcher
    that exits: the launcher's death reparents us to init while the real
    owner is alive and still holding the pipe. That is exactly why the signal
    is gated to respawn-hosts where a wrong reap costs nothing.

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
        reap_enabled: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="puppetmaster-mcp-input-stale-watcher")
        self._stale_after = stale_after_seconds
        self._interval = check_interval_seconds
        self._on_shutdown = on_shutdown
        self._reap_enabled = reap_enabled
        self._stop = threading.Event()
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _parent_is_dead() -> bool:
        if os.name != "posix":
            return False  # no reparenting signal on Windows; idle path only
        try:
            return os.getppid() == 1
        except OSError:
            return False

    def _should_reap(self) -> bool:
        # Host-gated: on hosts that surface a reaped server as a dead
        # transport (Codex "Transport closed", Claude Code), never self-reap.
        # Their authoritative owner-death signal is stdin EOF, handled by the
        # main loop — so a server reparented to init by an exiting launcher
        # while the owner is alive must NOT kill itself.
        if not self._reap_enabled:
            return False
        # Live handshake override: ``reap_enabled`` is frozen at startup from
        # env, but Codex scrubs CODEX_* from our env, so a Codex session
        # launched from a Cursor terminal leaks CURSOR_* and is misread as a
        # respawn host. Once the ``initialize`` handshake reveals a Codex
        # client, never self-reap regardless of the startup env guess.
        if _client_is_codex():
            return False
        if self._parent_is_dead():
            return True
        last_msg, active = _input_state_snapshot()
        return (time.time() - last_msg) >= self._stale_after and active == 0

    def run(self) -> None:  # pragma: no cover - exercised via integration tests
        while not self._stop.wait(self._interval):
            if not self._should_reap():
                continue
            self._triggered = True
            _SHUTDOWN_REQUESTED.set()
            try:
                self._on_shutdown()
            except (OSError, RuntimeError):
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


def _resolve_max_block_seconds() -> float:
    """The hard ceiling on how long a tool handler may block the MCP turn.

    Returns 0.0 when the cap is disabled (``PUPPETMASTER_MCP_MAX_BLOCK_SECONDS=0``),
    in which case :func:`_capped_block_seconds` is a passthrough. A malformed
    value falls back to the default rather than removing the protection.
    """
    raw = os.environ.get("PUPPETMASTER_MCP_MAX_BLOCK_SECONDS")
    if raw is None:
        return _DEFAULT_MAX_BLOCK_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MAX_BLOCK_SECONDS
    if value <= 0:
        return 0.0  # explicitly disabled
    return value


def _capped_block_seconds(requested: float) -> tuple[float, bool]:
    """Clamp a requested block duration to the Codex-safe ceiling.

    Returns ``(effective_seconds, was_capped)``. ``was_capped`` is True only
    when the cap is active *and* strictly shortened the requested value, so a
    caller asking for less than the ceiling never gets flagged.
    """
    cap = _resolve_max_block_seconds()
    if cap <= 0 or requested <= cap:
        return requested, False
    return cap, True


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
    # Hide Windows console windows for every child we spawn (workers,
    # git, codegraph) before any subprocess machinery runs.
    from puppetmaster.win_console import hide_child_consoles

    hide_child_consoles()

    # Claim fd 1 for JSON-RPC frames before anything else runs, so no
    # stray print — ours, a library's, or an fd-inheriting child's — can
    # corrupt the protocol stream (fatal and unrecoverable under Codex).
    _isolate_protocol_stdout()

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
            reap_enabled=_input_staleness_forced() or _host_transparently_respawns(),
        )
        input_watcher.start()

    idle_keepalive: Optional[_IdleKeepalive] = None
    if not _idle_keepalive_disabled():
        idle_keepalive = _IdleKeepalive(
            interval_seconds=_resolve_idle_keepalive_interval(),
        )
        idle_keepalive.start()

    reaper = _AsyncProcessReaper(interval_seconds=_resolve_reap_interval())
    reaper.start()

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
        reaper.stop()
        if input_watcher is not None:
            input_watcher.stop()
        if idle_keepalive is not None:
            idle_keepalive.stop()
        if heartbeat_thread is not None:
            heartbeat_thread.stop()
        if registration_path is not None:
            try:
                registry_deregister(registration_path)
            except OSError:
                pass
    return 0


def _server_version() -> Optional[str]:
    """Best-effort lookup of the installed puppetmaster version.

    The distribution is published as ``puppetmaster-ai`` (the bare
    ``puppetmaster`` name is held on PyPI), so try that first and fall back to
    the legacy name for local/editable installs registered either way.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version as _version

        for dist_name in ("puppetmaster-ai", "puppetmaster"):
            try:
                return _version(dist_name)
            except PackageNotFoundError:
                continue
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
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        tool_name = params.get("name")
        keepalive = _ToolCallKeepalive(
            tool_name=str(tool_name or ""),
            request_id=message.get("id"),
            progress_token=_extract_progress_token(params),
        )
        keepalive.start()
    if is_tool_call:
        _tool_call_started()
    try:
        response = handle_message(message)
    except Exception as exc:
        response = error_response(
            message.get("id"), -32000, _error_message_with_update_nudges(str(exc))
        )
    finally:
        if keepalive is not None:
            keepalive.stop()
        if is_tool_call:
            _tool_call_finished()
    if response is None:
        return
    serialized = json.dumps(response) + "\n"
    with _STDOUT_LOCK:
        stream = _protocol_stream()
        stream.write(serialized)
        stream.flush()


def handle_message(message: JsonObject) -> Optional[JsonObject]:
    method = message.get("method")
    request_id = message.get("id")

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            params = message.get("params") or {}
            client_info = params.get("clientInfo")
            if isinstance(client_info, dict):
                global _CLIENT_INFO
                _CLIENT_INFO = client_info
            result = {
                "protocolVersion": "2024-11-05",
                # `logging` advertises that we emit `notifications/message`
                # (the keepalive + idle pings); declaring it keeps those
                # frames spec-legal for strict clients like Codex.
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {
                    "name": "puppetmaster",
                    "version": _server_version() or "0.2.0-beta.1",
                },
            }
        elif method == "tools/list":
            result = {"tools": [tool_to_json(tool) for tool in tools()]}
        elif method == "tools/call":
            params = message.get("params") or {}
            result = call_tool(str(params.get("name", "")), params.get("arguments") or {})
        else:
            return error_response(request_id, -32601, f"Unknown MCP method: {method}")
    except Exception as exc:
        return error_response(request_id, -32000, _error_message_with_update_nudges(str(exc)))

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


_SERVER_UPDATE_CHECK_TTL_SECONDS = 30.0
_server_update_cache: dict[str, Any] = {"checked_at": 0.0, "note": None}
_server_update_lock = threading.Lock()


def reset_server_update_cache() -> None:
    """Drop the cached staleness verdict (used by tests; harmless in prod)."""
    with _server_update_lock:
        _server_update_cache["checked_at"] = 0.0
        _server_update_cache["note"] = None


def _error_message_with_update_nudges(message: str) -> str:
    """Append opt-in PyPI update awareness to MCP error messages."""
    pypi_note = pypi_update_note()
    if pypi_note:
        return f"{message}\n\n{pypi_note}"
    return message


def server_update_note(*, now: Optional[float] = None) -> Optional[str]:
    """One-line nudge when newer puppetmaster-ai code is on disk than this
    long-lived server loaded at startup.

    The stdio server can't hot-reload (the client owns the pipe), so silent
    staleness cost a manual kill/toggle/reconnect round-trip every upgrade.
    Surfacing it in every tool response makes it a visible nudge. Cached with a
    short TTL so the hot path doesn't re-scan package metadata on every call.
    """
    moment = time.time() if now is None else now
    with _server_update_lock:
        age = moment - _server_update_cache["checked_at"]
        if age < _SERVER_UPDATE_CHECK_TTL_SECONDS and _server_update_cache["checked_at"]:
            return _server_update_cache["note"]
    running = _SERVER_RUNNING_VERSION
    on_disk = installed_puppetmaster_version()
    note: Optional[str] = None
    if running and on_disk and on_disk != running and version_is_newer(on_disk, running):
        note = (
            f"Puppetmaster MCP server is running {running} but {on_disk} is "
            "installed on disk. Restart it — toggle the MCP server in your client, "
            "or run `puppetmaster mcp cleanup --kill-stale` — to load the new code."
        )
    with _server_update_lock:
        _server_update_cache["checked_at"] = moment
        _server_update_cache["note"] = note
    return note


def _attach_update_nudges(result: dict) -> None:
    """Push-style update awareness on tool responses (informational only)."""
    note = server_update_note()
    if note:
        result.setdefault("server_update_available", note)
    if result.get("isError"):
        pypi_note = pypi_update_note()
        if pypi_note:
            result.setdefault("pypi_update_available", pypi_note)


def call_tool(name: str, arguments: JsonObject) -> JsonObject:
    tool = _tool_registry().get(name)
    if tool is None:
        raise ValueError(f"Unknown Puppetmaster tool: {name}")
    result = tool.handler(arguments)
    # Push-style staleness nudge: every tool response carries a one-line warning
    # when a newer puppetmaster-ai is installed than this server is running, so a
    # daily-driver user doesn't have to run `mcp status` to discover it.
    if isinstance(result, dict):
        _attach_update_nudges(result)
    return result


# The tool list and name->tool map are static for the life of the process, but
# `tools/list` and every `tools/call` used to rebuild them (allocating the full
# list plus a fresh dict and all the lambdas) on each hot-path MCP request.
# Build once and cache.
_TOOLS_CACHE: Optional[list[McpTool]] = None
_TOOL_REGISTRY_CACHE: Optional[dict[str, McpTool]] = None


def tools() -> list[McpTool]:
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        _TOOLS_CACHE = _build_tools()
    return _TOOLS_CACHE


def _tool_registry() -> dict[str, McpTool]:
    global _TOOL_REGISTRY_CACHE
    if _TOOL_REGISTRY_CACHE is None:
        _TOOL_REGISTRY_CACHE = {tool.name: tool for tool in tools()}
    return _TOOL_REGISTRY_CACHE


def _build_tools() -> list[McpTool]:
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
                "run the model. Use this to estimate spend before kicking off a swarm, "
                "to debug 'why did this go to model X?' artifacts, or as a cheap "
                "delegate/inline gate: if a task scores as non-trivial here, prefer a "
                "Puppetmaster swarm over grinding through it inline."
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
                "Report a job's cost on two bases: ACTUAL measured spend "
                "(tokens actually consumed × registry price of the model each "
                "task ran on — works for pinned, auto-routed, or plan-billed "
                "runs) plus a counterfactual ('what this volume would cost on "
                "the flagship'), and the PRE-FLIGHT routing estimate when the "
                "job auto-routed. Returns per-model breakdowns + totals. Use to "
                "answer 'how much did this swarm cost?' or 'which model ate the "
                "most tokens?'. Prices come from the user-asserted registry — "
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
            name="puppetmaster_codex",
            description="Run a full-featured Codex CLI worker through Puppetmaster and wait for completion.",
            input_schema=codex_schema(),
            handler=run_codex,
        ),
        McpTool(
            name="puppetmaster_start_codex",
            description="Start a Codex CLI worker asynchronously and return job_id immediately.",
            input_schema=codex_schema(),
            handler=start_codex,
        ),
        McpTool(
            name="puppetmaster_agentic",
            description=(
                "Run a standalone provider-agnostic worker through Puppetmaster and wait "
                "for completion. Uses your provider API key directly (no external agent CLI); "
                "supports analyze (read-only) and implement (full-edit) modes."
            ),
            input_schema=agentic_schema(),
            handler=run_agentic,
        ),
        McpTool(
            name="puppetmaster_start_agentic",
            description=(
                "Start a standalone provider-agnostic worker asynchronously and return "
                "job_id immediately. Keys-only: calls the provider HTTP API directly."
            ),
            input_schema=agentic_schema(),
            handler=start_agentic,
        ),
        McpTool(
            name="puppetmaster_cursor_implement",
            description="Run Cursor as a full-edit Puppetmaster worker (edits files, captures a PATCH) and wait for completion.",
            input_schema=cursor_implement_schema(),
            handler=lambda args: run_cursor(args, implement=True),
        ),
        McpTool(
            name="puppetmaster_start_cursor_implement",
            description="Start Cursor as a full-edit worker asynchronously (edits files, captures a PATCH) and return job_id immediately.",
            input_schema=cursor_implement_schema(),
            handler=lambda args: start_cursor(args, implement=True),
        ),
        McpTool(
            name="puppetmaster_start_implement",
            description=(
                "PREFER over the built-in Task tool or an inline multi-file edit loop for "
                "any cross-cutting change. Start a full-edit implement worker on whichever "
                "platform you're locked to (cursor, claude-code, codex, hermes, or agentic), "
                "so implement isn't Claude-Code-only. Runs in a clean worktree and captures "
                "a PATCH artifact. Returns job_id immediately. Pass adapter to force one; "
                "otherwise the enabled platform is used."
            ),
            input_schema=implement_schema(),
            handler=start_implement,
        ),
        McpTool(
            name="puppetmaster_edit",
            description=(
                "PREFER over an inline single-file edit when the change benefits from "
                "CodeGraph to locate the site or from cheap-model routing to save frontier "
                "tokens. Lightweight SINGLE in-place edit: picks the cheapest sufficient "
                "model, uses CodeGraph, edits the working tree directly, and returns the "
                "diff synchronously (no job_id). Captures a reviewable PATCH artifact. "
                "Use start_implement instead for multi-file/coupled features that want an "
                "isolated worktree. Keep truly trivial edits (typo/rename/comment) inline."
            ),
            input_schema=edit_schema(),
            handler=run_edit,
        ),
        McpTool(
            name="puppetmaster_start_browser_swarm",
            description=(
                "Start a browser-QA swarm: N parallel Hermes workers, each driving a REAL "
                "browser against a LIVE site to capture real network payloads (the QA "
                "that mock-backend tests and read-only repo analysis cannot reach). Bakes "
                "in three guardrails: React-controlled-input native-event entry, "
                "network-truth (a 200 can hide an error body), and a strong-model "
                "capability floor (cheap models fail browser grounding and lie about it). "
                "ACTING AGENT — workers have external side effects (logins, form fills), so "
                "treat with implement-style approval. Requires the Hermes platform. Returns "
                "job_id immediately."
            ),
            input_schema=browser_swarm_schema(),
            handler=start_browser_swarm,
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
            description=(
                "PREFER over the built-in Task tool or your own grep/read exploration "
                "loop for broad investigation. Start a local Puppetmaster swarm "
                "asynchronously and return job_id immediately."
            ),
            input_schema=swarm_schema(),
            handler=start_swarm,
        ),
        McpTool(
            name="puppetmaster_start_cursor_swarm",
            description=(
                "PREFER over the built-in Task tool or an inline grep/read loop for any "
                "multi-file audit, review, or 'find all X' investigation. Start a "
                "multi-role Cursor SDK analysis swarm asynchronously and return job_id "
                "immediately."
            ),
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
            input_schema=status_schema(),
            handler=run_status,
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
            name="puppetmaster_await_job",
            description=(
                "Await sugar over the long-poll: block up to timeout_seconds for a job to "
                "reach a terminal state (complete/failed), then return its status and final "
                "summary. If it times out first, returns timed_out=true with the current "
                "status so you can immediately call again to keep awaiting (the MCP turn "
                "can't block forever). Prefer this over polling status in a loop."
            ),
            input_schema=await_schema(),
            handler=run_await_job,
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
            name="puppetmaster_dashboard",
            description=(
                "Open the live Puppetmaster web dashboard. Starts the "
                "zero-dependency local server (loopback by default) if one isn't "
                "already listening on the port and returns the URL to open — "
                "pass job_id to deep-link straight to one job. Use whenever the "
                "user asks to see/open/show the job dashboard, then open the "
                "returned URL in a browser tab for them. Pass mobile=true to "
                "serve a phone-reachable Tailscale/LAN address and get a QR to "
                "hand off (embed qr_image_path inline); stop=true tears the "
                "background server down. The server runs detached — no terminal "
                "to keep open."
            ),
            input_schema=dashboard_schema(),
            handler=run_dashboard,
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
                "MCP servers whose parent client appears to be gone. Never signals the "
                "current process. Returns the before/after registry snapshot."
            ),
            input_schema=mcp_cleanup_schema(),
            handler=run_mcp_cleanup,
        ),
        McpTool(
            name="puppetmaster_gc",
            description=(
                "Reap durable state for old terminal jobs (complete/failed/stalled) so "
                "per-project / per-worktree state dirs stop piling up. Dry-run by default "
                "— pass force=true to actually delete. Use all_projects=true to sweep "
                "every Puppetmaster state dir on the machine. Platform-agnostic: works "
                "across cursor, claude-code, codex, and openai jobs alike."
            ),
            input_schema=gc_schema(),
            handler=run_gc,
        ),
        McpTool(
            name="puppetmaster_rollup",
            description=(
                "Aggregate jobs/artifacts/cost/tokens across many worktree state dirs for "
                "one logical effort. Tag jobs via PUPPETMASTER_EFFORT_ID or `run --effort`, "
                "then roll them up here to get one ledger for an effort that spanned many "
                "worktrees. Pass effort_id to filter; all_projects=true is the usual case."
            ),
            input_schema=rollup_schema(),
            handler=run_rollup,
        ),
        McpTool(
            name="puppetmaster_gate",
            description=(
                "Replay the non-bypassable completion gates against a working tree, outside "
                "a worker run — the same engine the runtime enforces at task completion: "
                "require_diff / command oracle / monotonic ratchet (baseline only shrinks) / "
                "committed. Lets a parent agent or CI enforce drift/parity/commit "
                "post-conditions on demand. isError=true when any gate fails. Universal "
                "across every adapter."
            ),
            input_schema=gate_schema(),
            handler=run_gate,
        ),
    ]


def _attach_codegraph_freshness(payload: JsonObject, args: JsonObject) -> JsonObject:
    """Annotate a codegraph tool payload with index-freshness + self-heal.

    Surfaces a structured ``index_freshness`` block so the agent can *see*
    when the graph is behind the code (the otherwise-silent failure that makes
    a query "miss" code that exists), folds a loud warning into ``hint`` when
    stale, and kicks a deduped background re-sync. Best-effort: any probe error
    leaves the payload untouched."""
    try:
        target = cwd(args)
        freshness = codegraph_freshness(target)
    except Exception:
        return payload
    if freshness.state == "uninitialized":
        return payload
    annotated = dict(payload)
    annotated["index_freshness"] = freshness.to_payload()
    if freshness.is_stale:
        annotated["index_stale"] = True
        warning = freshness.warning_text()
        if warning:
            existing = annotated.get("hint") or ""
            annotated["hint"] = (existing + "\n\n" + warning).strip() if existing else warning
        maybe_autosync_codegraph(target, freshness)
    return annotated


def run_codegraph_search(args: JsonObject) -> JsonObject:
    payload = codegraph_query(
        require_string(args, "query"),
        cwd(args),
        kind=args.get("kind") if isinstance(args.get("kind"), str) else None,
        limit=int(args["limit"]) if args.get("limit") is not None else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(_attach_codegraph_freshness(payload, args))


def run_codegraph_context(args: JsonObject) -> JsonObject:
    payload = codegraph_context_command(
        require_string(args, "task"),
        cwd(args),
        max_nodes=int(args.get("max_nodes") or 15),
        fmt=str(args.get("format") or "markdown"),
    )
    return codegraph_response(_attach_codegraph_freshness(payload, args))


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
    return codegraph_response(_attach_codegraph_freshness(payload, args))


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
    if snapshot.get("code_stale"):
        snapshot["hint"] = (
            f"{snapshot['code_stale']} MCP server(s) are running pre-upgrade code "
            f"(installed {snapshot.get('installed_version')}). Restart them — toggle "
            "the MCP server in your client, or run `puppetmaster mcp cleanup "
            "--kill-stale` — so the new code takes effect."
        )
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
    try:
        stderr_handle = stderr_path.open("w", encoding="utf-8")
    except OSError:
        stdout_handle.close()
        raise
    try:
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
    except OSError:
        # Popen failed (e.g. bad executable); don't leak the open log handles.
        stdout_handle.close()
        stderr_handle.close()
        raise
    _track_async_process(process)
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


def run_cursor(
    args: JsonObject, review: bool = False, plan: bool = False, implement: bool = False
) -> JsonObject:
    if implement:
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return run_worker_cli(cursor_command(args, review=review, plan=plan, implement=implement), args)


def start_cursor(
    args: JsonObject, review: bool = False, plan: bool = False, implement: bool = False
) -> JsonObject:
    if implement:
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return start_cli(cursor_command(args, review=review, plan=plan, implement=implement), args)


def cursor_command(
    args: JsonObject, review: bool = False, plan: bool = False, implement: bool = False
) -> list[str]:
    goal = require_string(args, "goal")
    command = ["cursor", goal, "--cwd", cwd(args)]
    if implement:
        # Full-edit run: no --dry-run, let the agent modify the tree and capture
        # the diff as a PATCH artifact.
        command.append("--implement")
        if args.get("allow_dirty"):
            command.append("--allow-dirty")
        if args.get("allow_non_worktree"):
            command.append("--allow-non-worktree")
    else:
        command.append("--dry-run")
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
    disable_memory = args.get("disable_memory")
    if disable_memory is True or (disable_memory is None and (review or plan)):
        command.append("--disable-memory")
    return command


def _worktree_preflight(args: JsonObject) -> Optional[JsonObject]:
    """Fail a full-edit verb fast when its cwd is not inside a git work tree.

    The worker-level guard (:func:`puppetmaster.adapters.worktree_guard`)
    already blocks these runs, but only *after* a worker has spawned — the
    caller experiences a failed job it then has to investigate (field
    report: "use puppetmaster to..." in a non-git experiment dir). Checking
    at the verb means the agent learns the remediation in the same turn and
    no worker is wasted. Indeterminate results (no git binary, timeout)
    return ``None`` and defer to the worker-level guard rather than blocking
    a run the guard might allow.
    """
    if args.get("allow_non_worktree"):
        return None
    directory = cwd(args)
    try:
        completed = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode == 0 and completed.stdout.strip() == "true":
        return None
    return tool_error(
        "cwd is not inside a git work tree, and full-edit runs need one for "
        "clean-tree gating and diff attribution. Fix: run `git init` in the "
        "directory (restores diff capture), point cwd at an existing repo, or "
        "pass allow_non_worktree=true to run without diff attribution.",
        {"failure": "not_a_worktree", "cwd": directory},
    )


# Adapters that can run a full-edit "implement" worker, in the order the generic
# `puppetmaster_start_implement` verb prefers when several are enabled. Codex is
# last (cursor/claude are the daily drivers) but is a full-edit, PATCH-producing
# adapter, so a codex-only platform lock can still use the generic verb.
# Canonical order lives in puppetmaster.workers (single source of truth); aliased
# here for the local helpers that still reference the name.
from puppetmaster.workers import IMPLEMENT_ADAPTER_PRIORITY as _IMPLEMENT_ADAPTER_PRIORITY


def _implement_command(args: JsonObject, adapter: str) -> list[str]:
    if adapter == "cursor":
        return cursor_command(args, implement=True)
    if adapter == "claude-code":
        return claude_command(args)
    if adapter == "codex":
        return codex_command(args)
    if adapter == "hermes":
        return hermes_command(args, implement=True)
    if adapter == "agentic":
        return agentic_command(args, implement=True)
    raise ValueError(f"adapter {adapter!r} has no implement command")


def codex_command(args: JsonObject) -> list[str]:
    goal = require_string(args, "goal")
    command = ["codex", goal, "--cwd", cwd(args)]
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("sandbox"):
        command.extend(["--sandbox", str(args["sandbox"])])
    if args.get("approval_policy"):
        command.extend(["--approval-policy", str(args["approval_policy"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("allow_dirty"):
        command.append("--allow-dirty")
    if args.get("allow_non_worktree"):
        command.append("--allow-non-worktree")
    if args.get("executable"):
        command.extend(["--executable", str(args["executable"])])
    if args.get("dangerously_bypass_approvals_and_sandbox"):
        command.append("--dangerously-bypass-approvals-and-sandbox")
    if args.get("disable_codegraph"):
        command.append("--disable-codegraph")
    if args.get("disable_memory"):
        command.append("--disable-memory")
    return command


def hermes_command(args: JsonObject, implement: bool = True) -> list[str]:
    prompt = require_string(args, "goal")
    command = ["hermes", prompt, "--cwd", cwd(args)]
    command.extend(["--mode", "implement" if implement else "analyze"])
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("provider"):
        command.extend(["--provider", str(args["provider"])])
    if args.get("max_turns") is not None:
        command.extend(["--max-turns", str(args["max_turns"])])
    if args.get("toolsets"):
        command.extend(["--toolsets", str(args["toolsets"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("executable"):
        command.extend(["--executable", str(args["executable"])])
    if args.get("allow_dirty"):
        command.append("--allow-dirty")
    if args.get("allow_non_worktree"):
        command.append("--allow-non-worktree")
    if args.get("use_hermes_rules"):
        command.append("--use-hermes-rules")
    if args.get("disable_codegraph"):
        command.append("--disable-codegraph")
    return command


def start_implement(args: JsonObject) -> JsonObject:
    """Platform-agnostic implement: pick the full-edit adapter for whichever
    platform the user is locked to, so `implement` works no matter which single
    platform is enabled (not Claude-Code-only)."""
    from puppetmaster import platform_lock
    from puppetmaster.workers import NoImplementAdapterError, pick_implement_adapter

    enabled = platform_lock.enabled_adapters()
    try:
        adapter = pick_implement_adapter(enabled, args.get("adapter"))
    except NoImplementAdapterError as exc:
        details: JsonObject = {"enabled": sorted(exc.enabled)}
        if exc.requested is not None:
            details["requested"] = exc.requested
            details["fix"] = "puppetmaster platform enable " + exc.requested
        return tool_error(str(exc), details)
    blocked = _worktree_preflight(args)
    if blocked is not None:
        return blocked
    result = start_cli(_implement_command(args, adapter), args)
    if isinstance(result, dict):
        result.setdefault("implement_adapter", adapter)
    return result


def edit_command(args: JsonObject) -> list[str]:
    """Build the ``puppetmaster edit`` CLI invocation from MCP args."""
    instruction = require_string(args, "instruction")
    command = ["edit", instruction, "--cwd", cwd(args)]
    if args.get("adapter"):
        command.extend(["--adapter", str(args["adapter"])])
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("provider"):
        command.extend(["--provider", str(args["provider"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("routing_policy"):
        command.extend(["--routing-policy", str(args["routing_policy"])])
    if args.get("auto_route") is False:
        command.append("--no-auto-route")
    if args.get("disable_codegraph"):
        command.append("--disable-codegraph")
    if args.get("executable"):
        command.extend(["--executable", str(args["executable"])])
    return command


def run_edit(args: JsonObject) -> JsonObject:
    """Lightweight single in-place edit — runs SYNCHRONOUSLY and returns the diff.

    Unlike ``start_implement`` (async, isolated worktree, frontier-capable), this
    is the snappy verb for one focused change: cheapest sufficient model, CodeGraph
    to locate the site, edits the working tree in place, and blocks until the diff
    is ready so the caller sees the result immediately. A PATCH artifact is still
    captured for review/revert. Validate the adapter up front so a disabled/invalid
    platform fails with the same precise guidance as the implement verbs.
    """
    from puppetmaster import platform_lock
    from puppetmaster.workers import NoImplementAdapterError, pick_implement_adapter

    enabled = platform_lock.enabled_adapters()
    try:
        pick_implement_adapter(enabled, args.get("adapter"))
    except NoImplementAdapterError as exc:
        details: JsonObject = {"enabled": sorted(exc.enabled)}
        if exc.requested is not None:
            details["requested"] = exc.requested
            details["fix"] = "puppetmaster platform enable " + exc.requested
        return tool_error(str(exc), details)
    return run_cli(edit_command(args), args)


def browser_swarm_command(args: JsonObject) -> list[str]:
    """Build the ``puppetmaster browser`` CLI invocation from MCP args.

    ``tasks`` is required and may be a single string or a list of strings; each
    becomes one parallel browser worker.
    """
    raw_tasks = args.get("tasks")
    if isinstance(raw_tasks, str):
        tasks = [raw_tasks]
    elif isinstance(raw_tasks, (list, tuple)):
        tasks = [str(t) for t in raw_tasks if str(t).strip()]
    else:
        tasks = []
    if not tasks:
        raise ValueError("browser: 'tasks' must be a non-empty string or list of strings")
    command = ["browser", *tasks, "--cwd", cwd(args)]
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("provider"):
        command.extend(["--provider", str(args["provider"])])
    if args.get("toolsets"):
        command.extend(["--toolsets", str(args["toolsets"])])
    if args.get("min_capability") is not None:
        command.extend(["--min-capability", str(args["min_capability"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("routing_policy"):
        command.extend(["--routing-policy", str(args["routing_policy"])])
    if args.get("worker_mode"):
        command.extend(["--worker-mode", str(args["worker_mode"])])
    if args.get("executable"):
        command.extend(["--executable", str(args["executable"])])
    return command


def start_browser_swarm(args: JsonObject) -> JsonObject:
    """Start a browser-QA swarm asynchronously and return job_id immediately.

    Validates that Hermes (the only browser-capable adapter) is enabled before
    dispatching, so a platform-locked host gets a precise error instead of
    workers that silently cannot carry the toolset.
    """
    from puppetmaster import platform_lock
    from puppetmaster.browser import BROWSER_ADAPTER

    if not platform_lock.is_adapter_enabled(BROWSER_ADAPTER):
        return tool_error(
            f"the {BROWSER_ADAPTER!r} adapter is disabled by the platform lock, "
            "but it is the only adapter that can drive a browser.",
            {
                "enabled": sorted(platform_lock.enabled_adapters()),
                "fix": f"puppetmaster platform enable {BROWSER_ADAPTER}",
            },
        )
    try:
        command = browser_swarm_command(args)
    except ValueError as exc:
        return tool_error(str(exc))
    return start_cli(command, args)


def run_claude(args: JsonObject) -> JsonObject:
    blocked = _worktree_preflight(args)
    if blocked is not None:
        return blocked
    return run_worker_cli(claude_command(args), args)


def start_claude(args: JsonObject) -> JsonObject:
    blocked = _worktree_preflight(args)
    if blocked is not None:
        return blocked
    return start_cli(claude_command(args), args)


def _codex_is_write_capable(args: JsonObject) -> bool:
    # Mirrors the adapter: a read-only sandbox can't mutate the tree, so the
    # worktree requirement doesn't apply (unless the sandbox is bypassed).
    sandbox = str(args.get("sandbox") or "workspace-write")
    return sandbox != "read-only" or bool(
        args.get("dangerously_bypass_approvals_and_sandbox")
    )


def run_codex(args: JsonObject) -> JsonObject:
    if _codex_is_write_capable(args):
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return run_worker_cli(codex_command(args), args)


def start_codex(args: JsonObject) -> JsonObject:
    if _codex_is_write_capable(args):
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return start_cli(codex_command(args), args)


def _agentic_is_write_capable(args: JsonObject) -> bool:
    mode = str(args.get("mode") or "implement").strip().lower()
    return mode == "implement" or bool(args.get("implement"))


def run_agentic(args: JsonObject) -> JsonObject:
    if _agentic_is_write_capable(args):
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return run_worker_cli(agentic_command(args), args)


def start_agentic(args: JsonObject) -> JsonObject:
    if _agentic_is_write_capable(args):
        blocked = _worktree_preflight(args)
        if blocked is not None:
            return blocked
    return start_cli(agentic_command(args), args)


def agentic_command(args: JsonObject, implement: Optional[bool] = None) -> list[str]:
    goal = require_string(args, "goal")
    command = ["agentic", goal, "--cwd", cwd(args)]
    if implement is not None:
        command.extend(["--mode", "implement" if implement else "analyze"])
    elif args.get("mode"):
        command.extend(["--mode", str(args["mode"])])
    if args.get("provider"):
        command.extend(["--provider", str(args["provider"])])
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("max_turns") is not None:
        command.extend(["--max-turns", str(int(args["max_turns"]))])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("temperature") is not None:
        command.extend(["--temperature", str(float(args["temperature"]))])
    if args.get("reasoning_effort"):
        command.extend(["--reasoning-effort", str(args["reasoning_effort"])])
    if args.get("allow_dirty"):
        command.append("--allow-dirty")
    if args.get("allow_non_worktree"):
        command.append("--allow-non-worktree")
    if args.get("disable_codegraph"):
        command.append("--disable-codegraph")
    if args.get("disable_memory"):
        command.append("--disable-memory")
    _append_routing_cli_flags(command, args)
    return command


def _append_routing_cli_flags(command: list[str], args: JsonObject) -> None:
    if args.get("auto_route"):
        command.append("--auto-route")
    if args.get("routing_policy"):
        command.extend(["--routing-policy", str(args["routing_policy"])])
    if args.get("max_cost_usd") is not None:
        command.extend(["--max-cost-usd", str(args["max_cost_usd"])])
    if args.get("min_capability") is not None:
        command.extend(["--min-capability", str(args["min_capability"])])


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
    if args.get("allow_non_worktree"):
        command.append("--allow-non-worktree")
    if args.get("disable_memory"):
        command.append("--disable-memory")
    return command


def run_openai(args: JsonObject) -> JsonObject:
    return run_worker_cli(openai_command(args), args)


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
    if args.get("disable_memory"):
        command.append("--disable-memory")
    return command


def _append_label_flag(command: list[str], args: JsonObject) -> None:
    label = args.get("label")
    if label:
        command.extend(["--label", str(label)])


def _enabled_swarm_adapters() -> list[str]:
    """Analysis-capable adapters the user has enabled, in priority order.

    Used to give platform-neutral guidance (and a runnable shortlist) instead of
    hard-coding Cursor when a generic swarm needs an adapter.
    """
    from puppetmaster import platform_lock

    enabled = platform_lock.enabled_adapters()
    return [a for a in SWARM_ANALYSIS_ADAPTERS if a in enabled or a == "local"]


def start_swarm(args: JsonObject) -> JsonObject:
    goal = require_string(args, "goal")
    command = ["run", goal]
    roles = normalized_roles(args)
    adapter = args.get("adapter")
    if args.get("config"):
        command.extend(["--config", str(args["config"])])
    elif adapter:
        if str(adapter) not in SWARM_ANALYSIS_ADAPTERS:
            return tool_error(
                f"adapter {str(adapter)!r} cannot run an analysis swarm.",
                {
                    "adapter": str(adapter),
                    "supported_adapters": list(SWARM_ANALYSIS_ADAPTERS),
                    "fix": (
                        "Pass adapter=<one of the supported analysis adapters> "
                        "or use the matching platform-specific start verb."
                    ),
                },
            )
        config_path = write_generated_swarm_config(args, roles or ["explore"], str(adapter))
        command.extend(["--config", str(config_path)])
    elif roles:
        if not args.get("allow_local_demo"):
            enabled = _enabled_swarm_adapters()
            return tool_error(
                "Custom-role MCP swarms require a workflow config or an explicit adapter. "
                "Otherwise Puppetmaster would use the demo local adapter and return generic artifacts.",
                {
                    "roles": roles,
                    "enabled_adapters": enabled,
                    "fix": (
                        "Pass adapter=<your platform> (one of: "
                        f"{', '.join(enabled) or ', '.join(SWARM_ANALYSIS_ADAPTERS)}), "
                        "pass a config, or use the platform-specific start verb that "
                        "matches your platform. For tests/demos set allow_local_demo=true."
                    ),
                },
            )
        command.append("--workers")
        command.extend(roles)
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    _append_swarm_memory_flags(command, args)
    _append_label_flag(command, args)
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
    _append_label_flag(command, args)
    return start_cli(command, args)


def _append_swarm_memory_flags(command: list[str], args: JsonObject) -> None:
    if args.get("disable_memory") is False:
        command.append("--enable-memory")
    elif args.get("disable_memory", True):
        command.append("--disable-memory")


def normalized_roles(args: JsonObject) -> list[str]:
    roles = args.get("roles")
    if not isinstance(roles, list):
        return []
    return [str(role) for role in roles if str(role).strip()]


# Adapters that can back a generated *analysis* swarm: every adapter that can
# run a worker and emit artifacts. ``local`` is the deterministic demo backend;
# the rest run a real model in read-only/analyze mode. Edit-only concerns don't
# apply here — the generated workers are marked read_only — so any runnable
# adapter qualifies, not just cursor.
SWARM_ANALYSIS_ADAPTERS: tuple[str, ...] = (
    "agentic",
    "cursor",
    "local",
    "claude-code",
    "codex",
    "hermes",
    "openai",
)


def write_generated_swarm_config(args: JsonObject, roles: list[str], adapter: str) -> Path:
    if adapter not in SWARM_ANALYSIS_ADAPTERS:
        raise ValueError(
            f"adapter {adapter!r} cannot run an analysis swarm. Supported: "
            f"{', '.join(SWARM_ANALYSIS_ADAPTERS)}."
        )
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
        payload: JsonObject = {
            "prompt": prompt,
            "cwd": cwd(args),
            "timeout_seconds": timeout_seconds,
            # Generated MCP swarms are analysis workers: they emit structured
            # artifacts and must be able to review the caller's dirty diff.
            # If routing lands on an edit-capable adapter such as Codex, keep
            # it on the adapter's existing read-only/no-edit path.
            "read_only": True,
            "sandbox": "read-only",
            "dangerously_bypass_approvals_and_sandbox": False,
        }
        if adapter == "cursor":
            payload["model"] = model
        elif explicit_model:
            # An explicit pin must reach non-cursor adapters too (cursor already
            # carries a "default" model; the others only set a model when pinned).
            payload["model"] = str(explicit_model)
        if auto_route_enabled:
            payload["auto_route"] = True
            # An explicitly chosen non-cursor adapter must actually be dispatched
            # to: constrain routing to it so the router can't hop off the user's
            # pick. Cursor/local keep their historical unconstrained routing.
            if adapter not in ("cursor", "local"):
                payload["allowed_adapters"] = [adapter]
            if isinstance(routing_policy, str) and routing_policy:
                payload["routing_policy"] = routing_policy
            if isinstance(max_cost_usd, (int, float)):
                payload["max_cost_usd"] = float(max_cost_usd)
            if isinstance(min_capability, int):
                payload["min_capability"] = int(min_capability)
            if isinstance(required_tags, list) and required_tags:
                payload["required_tags"] = [str(tag) for tag in required_tags if str(tag).strip()]
        if args.get("disable_memory") is False:
            payload["disable_memory"] = False
        else:
            payload["disable_memory"] = True
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

    from puppetmaster.platform_lock import active_allowlist

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
        allowed_adapters=active_allowlist(),
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


def run_status(args: JsonObject) -> JsonObject:
    command = ["status", require_string(args, "job_id")]
    if args.get("compact"):
        command.append("--compact")
    return run_cli(command, args)


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


# Secret redaction lives in puppetmaster.redaction so adapters and the MCP
# server share one implementation. Re-exported here under the original name
# for backward compatibility with any caller importing it from mcp_server.
from puppetmaster.redaction import register_secret_values, redact_secrets  # noqa: E402


def _coerce_subprocess_text(value: object) -> str:
    """Normalize subprocess stdout/stderr to str. ``TimeoutExpired`` may carry
    bytes (or None) depending on how the child was captured."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_worker_cli(command: list[str], args: JsonObject) -> JsonObject:
    """Run a long worker verb, auto-detaching on hard-timeout hosts.

    Synchronous worker verbs (cursor/claude/codex/openai review/plan/implement)
    block the MCP turn for the whole worker run — up to 30 minutes. That's fine
    on Cursor, which tolerates long calls as bytes flow, but Codex hard-cancels
    any tool past its ``tool_timeout_sec`` and won't honor our keepalive (no
    ``progressToken``), so a healthy multi-minute job reads as a failed tool.

    On such a host we transparently switch to the async path: start the same
    worker detached and return its ``job_id`` immediately, with guidance to poll
    (``puppetmaster_await_job`` / ``status`` / ``show``). The work is identical
    and its artifacts persist — only the delivery changes from inline-blocking to
    poll-for-result, which is the only delivery Codex can actually complete.
    """
    if not _should_autodetach_worker(args):
        return run_cli(command, args)
    result = start_cli(command, args)
    if isinstance(result, dict) and not result.get("isError"):
        content = result.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            try:
                body = json.loads(content[0].get("text") or "{}")
            except (ValueError, TypeError):
                body = None
            if isinstance(body, dict):
                job_id = body.get("job_id")
                body["autodetached"] = True
                body["autodetach_reason"] = (
                    "This host (Codex) enforces a hard per-tool timeout and does "
                    "not honor MCP progress keepalives, so a synchronous worker "
                    "run would be cancelled mid-flight. Ran it asynchronously "
                    "instead; the job is live."
                )
                body["next_steps"] = [
                    f"Call puppetmaster_await_job with job_id={job_id} to block for the result",
                    f"Or puppetmaster_status / puppetmaster_show with job_id={job_id}",
                ]
                content[0]["text"] = json.dumps(body, indent=2)
    return result


def run_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(mcp_state_dir(args))
    timeout_seconds = int(args.get("runner_timeout_seconds") or 1800)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "puppetmaster", "--state-dir", state_dir] + command,
            cwd=cwd(args),
            env=launcher_environment(args),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # Surface a structured, redacted error instead of letting the
        # TimeoutExpired bubble up as an unstructured tool crash. Synchronous
        # implement/review calls can legitimately exceed the runner budget.
        body = {
            "command": "python -m puppetmaster " + " ".join(command),
            "cwd": cwd(args),
            "returncode": None,
            "failure": "timeout",
            "timeout_seconds": timeout_seconds,
            "stdout": redact_secrets(_coerce_subprocess_text(exc.stdout)),
            "stderr": redact_secrets(_coerce_subprocess_text(exc.stderr)),
        }
        return {
            "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
            "isError": True,
        }
    body = {
        "command": "python -m puppetmaster " + " ".join(command),
        "cwd": cwd(args),
        "returncode": process.returncode,
        "stdout": redact_secrets(process.stdout),
        "stderr": redact_secrets(process.stderr),
    }
    return {
        "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
        "isError": process.returncode != 0,
    }


def _dashboard_alive(host: str = "127.0.0.1", port: int = 8787) -> bool:
    """True when a dashboard already answers on ``host:port``."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/jobs", timeout=1.0
        ) as response:
            return response.status == 200
    except OSError:
        return False


def _spawn_dashboard_server(command: list[str], args: JsonObject) -> subprocess.Popen:
    """Launch the dashboard CLI detached from this MCP process."""
    return subprocess.Popen(
        command,
        cwd=cwd(args),
        env=launcher_environment(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def run_dashboard(args: JsonObject) -> JsonObject:
    """Ensure a dashboard server is running and return its URL (and QR for phones).

    Idempotent: an already-listening dashboard on the host:port is reused rather
    than spawning a duplicate. Defaults to loopback; ``mobile=true`` binds a
    phone-reachable Tailscale/LAN address (still unauthenticated + read-only) and
    returns a scannable QR so the pilot can hand off a link with zero setup.
    ``stop=true`` shuts down the detached background server."""
    from puppetmaster.dashboard import (
        qr_ascii,
        read_dashboard_runfile,
        resolve_mobile_host,
        stop_background_dashboard,
        write_dashboard_runfile,
        write_qr_png,
    )

    state_dir = str(mcp_state_dir(args))
    port = int(args.get("port") or 8787)
    job_id = args.get("job_id")
    job = job_id.strip() if isinstance(job_id, str) and job_id.strip() else None
    all_projects = bool(args.get("all_projects"))
    mobile = bool(args.get("mobile"))
    want_qr = bool(args.get("qr")) or mobile

    if bool(args.get("stop")):
        result = stop_background_dashboard(state_dir)
        result["state_dir"] = state_dir
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    host = "127.0.0.1"
    source = "loopback"
    if mobile:
        ip, source = resolve_mobile_host()
        if ip is None:
            body = {
                "error": "no phone-reachable address",
                "hint": (
                    "Bring Tailscale up (or join a LAN) so this host has a 100.x / "
                    "LAN IP, then retry. The user also installs Tailscale on the "
                    "phone and signs in to the same tailnet."
                ),
            }
            return {
                "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
                "isError": True,
            }
        host = ip

    url = f"http://{host}:{port}/" + (f"?job={job}" if job else "")

    already_running = _dashboard_alive(host, port)
    pid: Optional[int] = None
    if not already_running:
        command = [
            sys.executable,
            "-m",
            "puppetmaster",
            "--state-dir",
            state_dir,
            "dashboard",
            "--port",
            str(port),
            "--no-open",
        ]
        if host != "127.0.0.1":
            command += ["--host", host, "--allow-external"]
        if all_projects:
            command.append("--all-projects")
        if job:
            command.append(job)
        process = _spawn_dashboard_server(command, args)
        pid = process.pid
        deadline = time.time() + 10
        while time.time() < deadline and not _dashboard_alive(host, port):
            if process.poll() is not None:
                break
            time.sleep(0.2)
        if not _dashboard_alive(host, port):
            body = {
                "error": "dashboard failed to start",
                "host": host,
                "port": port,
                "state_dir": state_dir,
                "returncode": process.poll(),
                "hint": f"Try `python -m puppetmaster dashboard --port {port}` for the full error.",
            }
            return {
                "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
                "isError": True,
            }
        write_dashboard_runfile(
            state_dir,
            {
                "pid": pid,
                "host": host,
                "port": port,
                "url": url,
                "source": source,
                "all_projects": all_projects,
            },
        )
    elif mobile:
        # Keep the runfile pid current when reusing a server we can identify.
        tracked = read_dashboard_runfile(state_dir)
        if tracked and tracked.get("host") == host and int(tracked.get("port") or 0) == port:
            pid = tracked.get("pid")

    body: JsonObject = {
        "url": url,
        "host": host,
        "port": port,
        "source": source,
        "state_dir": state_dir,
        "all_projects": all_projects,
        "already_running": already_running,
        "started": not already_running,
        "pid": pid,
    }

    if want_qr:
        qr_path = str(Path(state_dir) / "dashboard-qr.png")
        if write_qr_png(url, qr_path):
            body["qr_image_path"] = qr_path
            body["qr_hint"] = (
                "Embed this image inline so the user can scan it: "
                f"![Scan to open the dashboard]({qr_path})"
            )
        else:
            ascii_art = qr_ascii(url)
            if ascii_art:
                body["qr_ascii"] = ascii_art
            else:
                body["qr_hint"] = (
                    "Install the QR extra for a scannable image: "
                    "pip install 'puppetmaster-ai[mobile]' — the URL above still works."
                )

    if mobile:
        body["note"] = (
            f"Phone-reachable over {source}. Unauthenticated + read-only — keep it "
            "to Tailscale or a trusted LAN. Hand the user the URL/QR; stop later "
            "with stop=true."
        )
    else:
        body["note"] = (
            "Reused the dashboard already listening on this port — it may be "
            "serving a different project's state dir or all-projects mode."
            if already_running
            else "Open the URL in a browser tab for the user."
        )
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2)}]}


def run_gc(args: JsonObject) -> JsonObject:
    command = ["gc", "--json"]
    if args.get("older_than_days") is not None:
        command.extend(["--older-than-days", str(args["older_than_days"])])
    if args.get("all_projects"):
        command.append("--all-projects")
    if args.get("force"):
        command.append("--force")
    return run_cli(command, args)


def run_rollup(args: JsonObject) -> JsonObject:
    command = ["rollup", "--json"]
    if isinstance(args.get("effort_id"), str) and args["effort_id"].strip():
        command.extend(["--effort", args["effort_id"]])
    if args.get("all_projects"):
        command.append("--all-projects")
    return run_cli(command, args)


def run_gate(args: JsonObject) -> JsonObject:
    command = ["gate", "--json"]
    if isinstance(args.get("gate_cwd"), str) and args["gate_cwd"].strip():
        command.extend(["--cwd", args["gate_cwd"]])
    if args.get("require_diff"):
        command.append("--require-diff")
    if isinstance(args.get("command"), str) and args["command"].strip():
        command.extend(["--command", args["command"]])
    if isinstance(args.get("ratchet_command"), str) and args["ratchet_command"].strip():
        command.extend(["--ratchet-command", args["ratchet_command"]])
    if isinstance(args.get("metric"), str) and args["metric"].strip():
        command.extend(["--metric", args["metric"]])
    if args.get("committed"):
        command.append("--committed")
    gates = args.get("gates")
    if gates is not None:
        command.extend(["--gates-json", json.dumps(gates)])
    return run_cli(command, args)


def run_feed(args: JsonObject) -> JsonObject:
    command = ["feed", require_string(args, "job_id"), "--json"]
    if args.get("limit"):
        command.extend(["--limit", str(args["limit"])])
    return run_cli(command, args)


def run_feed_follow(args: JsonObject) -> JsonObject:
    from puppetmaster.cli import artifact_feed_since

    job_id = require_string(args, "job_id")
    since = int(args.get("since_cursor") or args.get("since") or 0)
    requested_timeout = float(args.get("timeout_seconds") or 10.0)
    timeout_seconds, was_capped = _capped_block_seconds(requested_timeout)
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
    if was_capped:
        # Tell the caller the block was shortened so it knows to re-poll
        # rather than assuming the job produced nothing for `requested_timeout`.
        body["capped"] = True
        body["requested_timeout_seconds"] = requested_timeout
        body["effective_timeout_seconds"] = timeout_seconds
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2, default=str)}], "isError": False}


def run_await_job(args: JsonObject) -> JsonObject:
    from puppetmaster.cli import await_job_state
    from puppetmaster.stitcher import Stitcher

    job_id = require_string(args, "job_id")
    requested_timeout = float(args.get("timeout_seconds") or 25.0)
    timeout_seconds, was_capped = _capped_block_seconds(requested_timeout)
    poll_interval = float(args.get("poll_interval_seconds") or 0.25)
    backend = str(args.get("backend") or "sqlite")

    state_dir = mcp_state_dir(args)
    store = create_store(backend, state_dir)

    state = await_job_state(
        store,
        job_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval,
    )
    summary = ""
    if state["terminal"]:
        summary_path = store.job_dir(job_id) / "summaries" / "stitched.md"
        if summary_path.is_file():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            summary = Stitcher(store).preview(job_id)

    body = {**state, "summary": summary}
    if was_capped and not state["terminal"]:
        # Block shortened to stay under the client tool-timeout. The job is
        # still running; the caller should immediately await again.
        body["capped"] = True
        body["requested_timeout_seconds"] = requested_timeout
        body["effective_timeout_seconds"] = timeout_seconds
    return {
        "content": [{"type": "text", "text": json.dumps(body, indent=2, default=str)}],
        "isError": state["status"] in {"failed", "stalled"},
    }


def start_cli(command: list[str], args: JsonObject) -> JsonObject:
    if command and command[0] != "run":
        _append_label_flag(command, args)
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
    try:
        stderr_handle = stderr_path.open("w", encoding="utf-8")
    except OSError:
        stdout_handle.close()
        raise
    try:
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
    except OSError:
        # Popen failed before owning the fds; don't leak the open log handles.
        stdout_handle.close()
        stderr_handle.close()
        raise
    _track_async_process(process)
    stdout_handle.close()
    stderr_handle.close()
    try:
        job_id = wait_for_job_id(stdout_path, stderr_path, process, timeout_seconds=5)
    except BaseException:
        # The child was spawned but never reported a job id (startup crash or
        # parse timeout). Don't leave a detached full-edit agent running.
        try:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        raise
    body = {
        "run_id": run_id,
        "job_id": job_id,
        # `launcher_pid` is the detached launcher/orchestrator process, NOT the
        # durable worker doing the edits — that worker is a downstream child with
        # its own (shorter) lifetime and pid. Don't monitor progress by this pid;
        # use `job_id` with status/logs/feed. `pid` is kept as a back-compat alias.
        "launcher_pid": process.pid,
        "pid": process.pid,
        "pid_note": (
            "launcher_pid is the orchestrator launcher, not the worker; "
            "track progress via job_id (status/logs/feed), not this pid"
        ),
        "monitor_with": {"job_id": job_id, "use": "status/logs/feed"},
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
    # Read the log incrementally from a tracked byte offset instead of
    # re-reading the entire file every 50ms. Re-reading was O(n) per poll
    # (O(n^2) overall) once startup wrote a lot of output before the job id.
    offset = 0
    buffer = ""
    while time.monotonic() < deadline:
        if process.poll() is not None and not stdout_path.exists():
            break
        if stdout_path.exists():
            with stdout_path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                buffer += chunk
                match = pattern.search(buffer)
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
    argument_secrets: list[str] = []
    if args.get("cursor_api_key"):
        value = str(args["cursor_api_key"])
        env["CURSOR_API_KEY"] = value
        argument_secrets.append(value)
    if args.get("anthropic_api_key"):
        value = str(args["anthropic_api_key"])
        env["ANTHROPIC_API_KEY"] = value
        argument_secrets.append(value)
    if args.get("claude_code_command"):
        env["CLAUDE_CODE_COMMAND"] = str(args["claude_code_command"])
    if args.get("openai_api_key"):
        value = str(args["openai_api_key"])
        env["OPENAI_API_KEY"] = value
        argument_secrets.append(value)
    if args.get("openai_base_url"):
        env["OPENAI_BASE_URL"] = str(args["openai_base_url"])
    if args.get("openai_organization"):
        env["OPENAI_ORG_ID"] = str(args["openai_organization"])
    if args.get("codex_command"):
        env["CODEX_COMMAND"] = str(args["codex_command"])
    register_secret_values(argument_secrets)
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


def status_schema() -> JsonObject:
    schema = job_schema(required=True)
    schema["properties"]["compact"] = {
        "type": "boolean",
        "description": (
            "Omit high-churn prompt bodies from status JSON and replace them "
            "with deterministic char-count/SHA-256 refs."
        ),
    }
    return schema


def dashboard_schema() -> JsonObject:
    schema = job_schema()
    schema["properties"]["job_id"]["description"] = (
        "Optional job to deep-link; omit to land on the jobs index."
    )
    schema["properties"]["port"] = {
        "type": "integer",
        "default": 8787,
        "description": "Dashboard port (default 8787).",
    }
    schema["properties"]["all_projects"] = {
        "type": "boolean",
        "description": (
            "Aggregate jobs from every Puppetmaster project state dir on this "
            "machine instead of just the current project."
        ),
    }
    schema["properties"]["mobile"] = {
        "type": "boolean",
        "description": (
            "Serve on a phone-reachable Tailscale/LAN address (implies external "
            "bind) and return a scannable QR. Still unauthenticated + read-only — "
            "use only over Tailscale or a trusted LAN."
        ),
    }
    schema["properties"]["qr"] = {
        "type": "boolean",
        "description": (
            "Return a QR of the URL (a PNG path to embed inline, or ASCII "
            "fallback). Implied by mobile."
        ),
    }
    schema["properties"]["stop"] = {
        "type": "boolean",
        "description": "Stop the detached background dashboard tracked for this state dir.",
    }
    return schema


def gc_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "older_than_days": {
                "type": "number",
                "default": 7.0,
                "description": "Only reap terminal jobs finished more than N days ago.",
            },
            "all_projects": {
                "type": "boolean",
                "description": "Sweep every Puppetmaster project state dir on this machine.",
            },
            "force": {
                "type": "boolean",
                "description": "Actually delete. Omit for a dry-run that only reports.",
            },
        }
    )
    return schema


def rollup_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "effort_id": {
                "type": "string",
                "description": "Only include jobs tagged with this effort id. Omit for all.",
            },
            "all_projects": {
                "type": "boolean",
                "description": "Aggregate across every project state dir (usual for an effort).",
            },
        }
    )
    return schema


def gate_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "gate_cwd": {
                "type": "string",
                "description": "Working tree to evaluate gates against. Defaults to cwd.",
            },
            "require_diff": {
                "type": "boolean",
                "description": "Fail unless the tree has a non-empty diff (an edit happened).",
            },
            "command": {
                "type": "string",
                "description": "Oracle command; must exit 0 (e.g. the test/parity suite).",
            },
            "ratchet_command": {
                "type": "string",
                "description": "Command printing JSON metrics on stdout for the ratchet gate.",
            },
            "metric": {
                "type": "string",
                "description": "Metric key the ratchet enforces (monotonic; may only shrink).",
            },
            "committed": {
                "type": "boolean",
                "description": "Fail if the tree has uncommitted changes after the run.",
            },
            "gates": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Full gate specs ([{kind,...}]) for gates the flags don't cover.",
            },
        }
    )
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


def await_schema() -> JsonObject:
    schema = job_schema(required=True)
    schema["properties"].update(
        {
            "timeout_seconds": {
                "type": "number",
                "default": 25,
                "description": (
                    "Maximum seconds to block waiting for the job to finish. Bounded so "
                    "the MCP turn returns; if it times out, call again to keep awaiting."
                ),
            },
            "poll_interval_seconds": {
                "type": "number",
                "default": 0.25,
                "description": "Internal poll interval while blocked.",
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
            "label": {
                "type": "string",
                "description": (
                    "Short human-readable label (3-6 words) shown as the job's "
                    "headline on the dashboard and in `puppetmaster_jobs`. Set one "
                    "by default so runs stay scannable; when omitted, a title is "
                    "derived from the goal."
                ),
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
            "disable_memory": {
                "type": "boolean",
                "description": (
                    "Skip promoted shared-memory injection for a fresh perspective. "
                    "Evaluative worker roles skip memory by default."
                ),
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


def codex_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change and run focused tests.")
    schema["properties"].update(
        {
            "model": {
                "type": "string",
                "description": (
                    "Optional Codex model name. Defaults to gpt-5.4-mini when omitted."
                ),
            },
            "sandbox": {
                "type": "string",
                "enum": ["read-only", "workspace-write", "danger-full-access"],
                "description": "Codex sandbox mode. Defaults to workspace-write.",
            },
            "approval_policy": {
                "type": "string",
                "default": "never",
                "description": "Codex approval policy for non-interactive automation.",
            },
            "allow_dirty": {
                "type": "boolean",
                "default": False,
                "description": "Allow Codex to run in a dirty working tree.",
            },
            "allow_non_worktree": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Allow a write-capable run outside a git work tree "
                    "(no diff attribution; `git init` is usually better)."
                ),
            },
            "executable": {
                "type": "string",
                "description": "Optional Codex CLI executable or command override.",
            },
            "dangerously_bypass_approvals_and_sandbox": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Disable Codex sandbox and approval prompts. Only when externally sandboxed."
                ),
            },
            "disable_codegraph": {
                "type": "boolean",
                "default": False,
                "description": "Skip CodeGraph context injection.",
            },
            "codex_command": {
                "type": "string",
                "description": "Optional Codex CLI command override (e.g. path to codex binary).",
            },
        }
    )
    return schema


def claude_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change and run focused tests.")
    schema["properties"].update(
        {
            "model": {
                "type": "string",
                "description": (
                    "Optional Claude model name. Defaults to claude-opus-4-8 "
                    "(the frontier flagship) when omitted and no router model "
                    "is stamped."
                ),
            },
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
            "allow_non_worktree": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Allow the run outside a git work tree "
                    "(no diff attribution; `git init` is usually better)."
                ),
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


def cursor_implement_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change end to end and leave the diff in the tree.")
    schema["properties"].update(
        {
            "allow_dirty": {
                "type": "boolean",
                "default": False,
                "description": "Allow the implement run to start in a dirty working tree.",
            },
            "allow_non_worktree": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Allow the implement run outside a git work tree "
                    "(no diff attribution; `git init` is usually better)."
                ),
            },
        }
    )
    return schema


def implement_schema() -> JsonObject:
    schema = cursor_implement_schema()
    schema["properties"].update(
        {
            "adapter": {
                "type": "string",
                "enum": ["cursor", "claude-code", "codex", "hermes", "agentic"],
                "description": (
                    "Force a specific implement-capable platform. Omit to use whichever "
                    "platform the lock has enabled (cursor preferred, then claude-code, "
                    "then codex, then hermes, then agentic)."
                ),
            },
            "sandbox": {
                "type": "string",
                "description": "Codex sandbox mode (only used when the codex adapter runs).",
            },
            "permission_mode": {
                "type": "string",
                "default": "acceptEdits",
                "description": "Claude Code permission mode (only used when the claude-code adapter runs).",
            },
        }
    )
    return schema


def edit_schema() -> JsonObject:
    """Schema for the lightweight, synchronous in-place ``edit`` verb."""
    return {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "What to change, in plain language (one focused edit).",
            },
            "cwd": {
                "type": "string",
                "description": "Workspace/repo to edit in. Defaults to the server's cwd.",
            },
            "adapter": {
                "type": "string",
                "enum": ["cursor", "claude-code", "codex", "hermes", "agentic"],
                "description": (
                    "Force a full-edit adapter. Omit to use the highest-priority "
                    "adapter the platform lock enables."
                ),
            },
            "model": {
                "type": "string",
                "description": "Pin the model (overrides cheap auto-routing).",
            },
            "provider": {
                "type": "string",
                "description": "Inference provider (Hermes or agentic adapter).",
            },
            "routing_policy": {
                "type": "string",
                "enum": ["cheap", "balanced", "quality", "escalating"],
                "default": "cheap",
                "description": "Router policy when not pinning a model (default: cheap).",
            },
            "auto_route": {
                "type": "boolean",
                "default": True,
                "description": "Let the router pick the cheapest sufficient model. Set false to use the adapter default.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Adapter timeout (default 300).",
            },
            "disable_codegraph": {
                "type": "boolean",
                "description": "Skip CodeGraph context injection (e.g. non-repo edits).",
            },
            "executable": {
                "type": "string",
                "description": "Override the adapter executable / command.",
            },
        },
        "required": ["instruction"],
    }


def browser_swarm_schema() -> JsonObject:
    """Schema for the async browser-QA swarm verb."""
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "One or more QA missions. Each runs as its own parallel "
                    "browser worker against the live site."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Workspace/repo for context. Defaults to the server's cwd.",
            },
            "model": {
                "type": "string",
                "description": "Pin the Hermes model (overrides the strong-model routing floor).",
            },
            "provider": {
                "type": "string",
                "description": "Hermes provider (e.g. anthropic).",
            },
            "toolsets": {
                "type": "string",
                "description": "Override Hermes toolsets (default: file,web,vision,browser).",
            },
            "min_capability": {
                "type": "integer",
                "description": "Override the strong-model capability floor (default 80, 0..100).",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Per-worker timeout (default 1200; live browser flows are slow).",
            },
            "routing_policy": {
                "type": "string",
                "enum": ["cheap", "balanced", "quality", "escalating"],
                "default": "balanced",
                "description": "Router policy above the capability floor (default: balanced).",
            },
            "worker_mode": {
                "type": "string",
                "enum": ["subprocess", "inline", "daemon"],
                "default": "subprocess",
                "description": "subprocess (default) runs workers in parallel; inline serializes them.",
            },
            "executable": {
                "type": "string",
                "description": "Override the hermes executable / command.",
            },
        },
        "required": ["tasks"],
    }


def agentic_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change and run focused tests.")
    schema["properties"].update(
        {
            "mode": {
                "type": "string",
                "enum": ["analyze", "implement"],
                "default": "implement",
                "description": (
                    "analyze = read-only structured findings; implement = full-edit "
                    "with git-diff PATCH attribution."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Provider slug (openai, anthropic, gemini, openrouter, ...). "
                    "Routes credentials and wire protocol."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional provider model name.",
            },
            "max_turns": {
                "type": "integer",
                "description": "Cap on tool-use iterations (default 12).",
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature override (only sent if provided).",
            },
            "reasoning_effort": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "xhigh"],
                "description": "Reasoning effort level for OpenAI-style models.",
            },
            "allow_dirty": {
                "type": "boolean",
                "default": False,
                "description": "Allow the run in a dirty working tree.",
            },
            "allow_non_worktree": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Allow a write-capable run outside a git work tree "
                    "(no diff attribution; `git init` is usually better)."
                ),
            },
            "disable_codegraph": {
                "type": "boolean",
                "default": False,
                "description": "Skip CodeGraph context injection.",
            },
            "auto_route": {
                "type": "boolean",
                "description": (
                    "Enable per-task model routing. Defaults to true when no `model` "
                    "is pinned, false otherwise."
                ),
            },
            "routing_policy": {
                "type": "string",
                "enum": ["balanced", "cheap", "quality", "escalating"],
                "description": "Routing policy for auto-routed workers.",
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
