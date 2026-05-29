"""Process registry for Puppetmaster MCP servers.

Why this exists
---------------

Cursor spawns a long-lived stdio MCP server per workspace. The transport
between Cursor's MCP client and that server can drop for reasons we
don't control — Cursor reloads MCP settings, the user toggles the
server, an in-flight call exceeds an internal timeout, etc. When that
happens the agent sees ``Tool execution error. Not connected`` while
the underlying Puppetmaster process and its subprocess swarm are still
running. Cursor then spawns a fresh MCP server for the next session.

Repeat that cycle a few times in one day and you accumulate orphan
``python -m puppetmaster.mcp_server`` processes that nobody is
supervising. They waste memory, hold open SQLite handles, and contend
for the CodeGraph indexer lock. We need to:

1. Know which MCP servers are alive on this machine.
2. Detect orphans whose Cursor parent has gone away.
3. Clean them up safely without nuking a sibling Cursor window's
   legitimate server.

This module is the per-user state for that. Every Puppetmaster MCP
server writes a tracking file at startup, updates a heartbeat from a
background thread, and removes its file on a clean exit. A
hard-killed server leaves a stale file behind — the next server to
boot prunes it on startup, and ``puppetmaster mcp cleanup`` exposes
the same cleanup to humans and agents.

The registry is intentionally tiny: a directory of JSON files keyed
by PID. We never expose a single-file database because two MCP
servers writing into it simultaneously would race.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# How often a running server bumps its heartbeat.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0

# A server is "stale" when its heartbeat hasn't been touched in this
# window. Cursor restarts in seconds, not minutes, so anything older
# than 5 minutes that is still alive is almost certainly an orphan
# whose parent client is gone.
DEFAULT_STALE_AFTER_SECONDS = 300.0


@dataclass
class McpServerEntry:
    """A single MCP server tracked in the per-user registry."""

    pid: int
    workspace: Optional[str]
    started_at: float
    last_heartbeat: float
    transport: str = "stdio"
    version: Optional[str] = None
    path: Optional[str] = None  # absolute path to the tracking file

    def is_alive(self) -> bool:
        return _pid_alive(self.pid)

    def is_stale(self, *, now: Optional[float] = None, stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS) -> bool:
        current = now if now is not None else time.time()
        return (current - self.last_heartbeat) > stale_after_seconds

    def to_payload(self, *, now: Optional[float] = None) -> dict:
        current = now if now is not None else time.time()
        return {
            "pid": self.pid,
            "workspace": self.workspace,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "age_seconds": round(current - self.started_at, 3),
            "heartbeat_age_seconds": round(current - self.last_heartbeat, 3),
            "transport": self.transport,
            "version": self.version,
            "alive": self.is_alive(),
            "stale": self.is_stale(now=current),
            "path": self.path,
        }


def registry_dir() -> Path:
    """Per-user directory holding one JSON file per running MCP server.

    Overridable via ``PUPPETMASTER_MCP_REGISTRY_DIR`` so tests can
    redirect to a temp directory. The default lives under the same
    cache root we already use for the CodeGraph indexer lock so all
    cross-process Puppetmaster state stays in one place.
    """
    override = os.environ.get("PUPPETMASTER_MCP_REGISTRY_DIR")
    if override:
        directory = Path(override)
    else:
        directory = _default_cache_root() / "puppetmaster" / "mcp-servers"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _default_cache_root() -> Path:
    """Resolve the per-user cache root, respecting XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


def register(
    *,
    pid: Optional[int] = None,
    workspace: Optional[str] = None,
    version: Optional[str] = None,
    transport: str = "stdio",
) -> Path:
    """Write this server's tracking file and return its path.

    Idempotent: if a tracking file already exists for this PID (e.g.
    re-registration after a heartbeat thread restart) we overwrite it.
    """
    actual_pid = pid if pid is not None else os.getpid()
    now = time.time()
    payload = {
        "pid": actual_pid,
        "workspace": workspace,
        "started_at": now,
        "last_heartbeat": now,
        "transport": transport,
        "version": version,
    }
    path = registry_dir() / f"{actual_pid}.json"
    _atomic_write(path, payload)
    return path


def heartbeat(path: Path, *, now: Optional[float] = None) -> bool:
    """Bump ``last_heartbeat`` on an existing tracking file.

    Returns False if the file is gone (e.g. another process cleaned it
    up because it thought we were dead) so the heartbeat thread can
    re-register itself instead of dying silently.
    """
    if not path.exists():
        return False
    try:
        data = _read_entry(path)
    except (OSError, ValueError):
        return False
    if data is None:
        return False
    data["last_heartbeat"] = now if now is not None else time.time()
    _atomic_write(path, data)
    return True


def deregister(path: Path) -> None:
    """Remove a tracking file on clean shutdown. Best-effort."""
    try:
        path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except (OSError, TypeError):
        # Python <3.8 doesn't support missing_ok; fall back manually.
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def list_entries(*, include_stale: bool = True) -> list[McpServerEntry]:
    """Return every server tracked in the registry, alive or otherwise.

    Set ``include_stale=False`` to drop entries that look like orphans
    according to ``DEFAULT_STALE_AFTER_SECONDS``. Dead entries (PID
    gone) are always returned so callers can see what's been cleaned
    up and tell the user.
    """
    directory = registry_dir()
    entries: list[McpServerEntry] = []
    if not directory.exists():
        return entries
    now = time.time()
    for child in sorted(directory.glob("*.json")):
        try:
            data = _read_entry(child)
        except (OSError, ValueError):
            continue
        if data is None:
            continue
        try:
            entry = McpServerEntry(
                pid=int(data["pid"]),
                workspace=data.get("workspace"),
                started_at=float(data.get("started_at") or now),
                last_heartbeat=float(data.get("last_heartbeat") or now),
                transport=str(data.get("transport") or "stdio"),
                version=data.get("version"),
                path=str(child),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not include_stale and entry.is_stale(now=now):
            continue
        entries.append(entry)
    return entries


def prune_dead() -> list[McpServerEntry]:
    """Remove tracking files whose PIDs are no longer alive.

    Returns the entries we cleaned so callers can surface them. A
    cleaned entry is *not* alive — its file is already gone by the
    time we return it.
    """
    cleaned: list[McpServerEntry] = []
    for entry in list_entries():
        if entry.is_alive():
            continue
        if entry.path:
            deregister(Path(entry.path))
        cleaned.append(entry)
    return cleaned


def kill_stale(
    *,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    self_pid: Optional[int] = None,
    grace_seconds: float = 3.0,
) -> list[McpServerEntry]:
    """Send SIGTERM (then SIGKILL after a grace period) to stale-but-alive servers.

    Refuses to signal the current process. Anything that exits between
    SIGTERM and SIGKILL is fine: ``deregister`` will be called by the
    atexit hook, and we'll skip the SIGKILL.
    """
    me = self_pid if self_pid is not None else os.getpid()
    killed: list[McpServerEntry] = []
    targets: list[McpServerEntry] = []
    now = time.time()
    for entry in list_entries():
        if not entry.is_alive():
            continue
        if entry.pid == me:
            continue
        if entry.is_stale(now=now, stale_after_seconds=stale_after_seconds):
            targets.append(entry)
    if not targets:
        return killed
    for entry in targets:
        try:
            os.kill(entry.pid, signal.SIGTERM)
        except OSError:
            continue
        killed.append(entry)
    if grace_seconds > 0:
        time.sleep(grace_seconds)
    for entry in targets:
        if not _pid_alive(entry.pid):
            if entry.path:
                deregister(Path(entry.path))
            continue
        # Windows has no SIGKILL; os.kill(pid, SIGTERM) there maps to
        # TerminateProcess, which is the equivalent hard kill.
        hard_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
        try:
            os.kill(entry.pid, hard_kill)
        except OSError:
            pass
        if entry.path:
            deregister(Path(entry.path))
    return killed


class HeartbeatThread(threading.Thread):
    """Background thread that bumps a tracking file on a fixed cadence.

    Lifetime is bounded by the daemon thread flag — when the server's
    main thread exits, this thread is reaped automatically. The
    ``stop()`` method is the cooperative shutdown path for clean
    exits, so the final atexit ``deregister`` doesn't race with a
    final heartbeat write.
    """

    def __init__(
        self,
        registration_path: Path,
        *,
        interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(daemon=True, name="puppetmaster-mcp-heartbeat")
        self._path = registration_path
        self._interval = max(0.5, float(interval_seconds))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - exercised via integration tests
        while not self._stop_event.wait(self._interval):
            if not heartbeat(self._path):
                # File got cleaned up under us — write a fresh one so we
                # remain visible. This is unusual but cheap to handle.
                try:
                    register(pid=os.getpid(), workspace=_workspace_hint())
                except OSError:
                    return


def _workspace_hint() -> Optional[str]:
    """Best-effort workspace label for the current process."""
    try:
        return str(Path.cwd())
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    """POSIX-friendly liveness probe via signal 0.

    On Windows this falls back to a process-handle probe via ``os.kill``
    which raises ``OSError`` for dead PIDs and is a no-op for live ones.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # We can't signal it (different user), but it exists.
        return True
    except OSError:
        return False
    return True


def _atomic_write(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` via a temp file + rename so readers never see a half-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _read_entry(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not raw.strip():
        return None
    return json.loads(raw)


def summarize(entries: Iterable[McpServerEntry], *, now: Optional[float] = None) -> dict:
    """Render a registry snapshot as a JSON-safe payload for CLI/MCP output."""
    current = now if now is not None else time.time()
    rows = [entry.to_payload(now=current) for entry in entries]
    return {
        "now": current,
        "count": len(rows),
        "alive": sum(1 for row in rows if row["alive"]),
        "stale": sum(1 for row in rows if row["stale"] and row["alive"]),
        "dead": sum(1 for row in rows if not row["alive"]),
        "servers": rows,
    }
