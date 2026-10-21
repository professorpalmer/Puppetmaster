"""One shared ``codex app-server`` for a flow walk.

A ``codex exec`` worker starts its own Codex process: the Node launcher, the
native binary, a login-shell snapshot and git probes. Native Codex runs its
subagents as threads in one process. With the host on, the top flow walker
starts one ``codex app-server`` on a Unix socket, and every Codex agent and
judge of the walk (map items too) runs as a thread on it.

The walker exports the record path in ``PUPPETMASTER_CODEX_HOST_RECORD``.
Child walkers and workers inherit it. The adapter uses the host only while
the recorded process is alive; otherwise it runs ``codex exec`` as before.
POSIX only: the socket is AF_UNIX.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from puppetmaster.proc_identity import pid_reused, process_identity

OPT_IN_ENV = "PUPPETMASTER_CODEX_HOST"
RECORD_ENV = "PUPPETMASTER_CODEX_HOST_RECORD"
RECORD_NAME = "codex-host.json"
# Bounded workers never fork subagents, and must not accrue memories.
HOST_FLAGS = ("--disable", "memories", "--disable", "multi_agent")
_START_SECONDS = 20.0


def supported() -> bool:
    return hasattr(socket, "AF_UNIX") and os.name != "nt"


def requested(graph: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether a walk of ``graph`` runs its Codex workers on a shared host."""
    env = os.environ if env is None else env
    defaults = graph.get("defaults") if isinstance(graph.get("defaults"), dict) else {}
    if "codex_host" in defaults:
        return defaults.get("codex_host") is True and supported()
    return str(env.get(OPT_IN_ENV, "")).strip().lower() in ("1", "true", "on", "yes") and supported()


def _alive(record: Mapping[str, Any]) -> bool:
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not pid_reused(pid, record.get("proc"))


def read(path: Path) -> Optional[dict]:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def current(env: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """The live host this process belongs to, or None."""
    env = os.environ if env is None else env
    path = (env.get(RECORD_ENV) or "").strip()
    if not path or not supported():
        return None
    record = read(Path(path))
    return record if record is not None and _alive(record) and record.get("socket") else None


def start(run_dir: Path, command_base: list, env: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """Start the host for a walk and write its record in ``run_dir``.

    Returns None when the host cannot start; workers then use ``codex exec``.
    """
    from puppetmaster import codex_home
    from puppetmaster.codegraph import inject_worker_cli_env

    run_dir = Path(run_dir)
    reap(run_dir)
    host_env = inject_worker_cli_env(dict(os.environ if env is None else env))
    home: Optional[Path] = None
    if codex_home.enabled(host_env):
        try:
            home = codex_home.prepare(host_env)
        except Exception:
            home = None  # Best effort: the host then runs in the user's own home.
    if home is not None:
        host_env["CODEX_HOME"] = str(home)
        try:
            codex_home.ensure_system_skills(home, list(command_base))
        except Exception:
            pass  # Best effort: stock Codex still installs the bundle itself.
    link = run_dir / "codex-host.sock"
    try:
        link.unlink()
    except OSError:
        pass
    command = [*command_base, "app-server", "--listen", f"unix://{link}", *HOST_FLAGS]
    try:
        with open(run_dir / "codex-host.log", "ab") as log:
            process = subprocess.Popen(command, cwd=str(run_dir), env=host_env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
    except OSError:
        return None
    started = time.time()
    record = {"pid": process.pid, "proc": process_identity(process.pid), "link": str(link),
              "socket": None, "home": str(home) if home is not None else None,
              "started_at": started, "command": command}
    deadline = time.monotonic() + _START_SECONDS
    while time.monotonic() < deadline and process.poll() is None:
        # Codex links the given path to a short socket path; AF_UNIX paths are
        # limited to about 104 bytes, so clients connect to the target.
        target = os.path.realpath(link) if link.is_symlink() else str(link)
        if Path(target).is_socket():
            record["socket"] = target
            break
        time.sleep(0.05)
    if record["socket"] is None:
        _signal(record, signal.SIGKILL)
        return None
    record["ready_at"] = time.time()
    path = run_dir / RECORD_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)
    record["record"] = str(path)
    return record


def _signal(record: Mapping[str, Any], sig: int) -> None:
    if not _alive(record):
        return
    try:
        os.killpg(int(record["pid"]), sig)
    except OSError:
        try:
            os.kill(int(record["pid"]), sig)
        except OSError:
            pass


def _reap_child(pid: int) -> None:
    """Reap the host when this process started it: a zombie still answers kill(pid, 0)."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def stop(record: Optional[Mapping[str, Any]], grace: float = 5.0) -> None:
    """Stop the host process group and copy a refreshed login back."""
    if not record:
        return
    from puppetmaster import codex_home

    _signal(record, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _alive(record) and time.monotonic() < deadline:
        _reap_child(int(record["pid"]))
        time.sleep(0.05)
    _signal(record, signal.SIGKILL)
    _reap_child(int(record["pid"]))
    home = record.get("home")
    if home and Path(home) == codex_home.worker_home_root():
        try:
            codex_home.sync_back()
        except Exception:
            pass  # Best effort: the next worker start copies auth in again.
    path = record.get("record")
    if path:
        try:
            Path(path).unlink()
        except OSError:
            pass


def reap(run_dir: Path) -> None:
    """Stop a host left by a walker of this run that died."""
    path = Path(run_dir) / RECORD_NAME
    record = read(path)
    if record is not None:
        stop({**record, "record": str(path)}, grace=1.0)
