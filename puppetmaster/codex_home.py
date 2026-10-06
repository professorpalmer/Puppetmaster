"""A lean CODEX_HOME for Puppetmaster's Codex workers.

Every fresh ``codex exec`` process sends, on its first model call, a developer
and environment block built from CODEX_HOME: the user's skills list,
memories, plugin recommendations and MCP context. Its per-turn metadata
differs per process, so that block is never a prompt-cache hit; with a
typical home it was about 12.4k fresh input tokens per worker. Native
subagents fork an already-cached parent thread and pay about 3.3k. A worker
does a bounded task and needs none of that block, so it runs in a home that
keeps the user's settings (model, service tier, providers, profiles, trusted
projects) and global AGENTS.md, and drops MCP servers, plugins, hooks,
notify, skills and memories: about 2.6k fresh tokens on the first call.

auth.json is copied, never symlinked: Codex refreshes a ChatGPT login by
atomically replacing auth.json, which would orphan the user's refresh token.
A worker's refreshed auth.json is copied back only if the user's own file has
not changed since it was copied in.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional

# Tables a bounded worker must not inherit: they cost context or reach out.
_DROPPED_TABLE = re.compile(r"^\[\[?\s*(mcp_servers|plugins|marketplaces|memories|hooks)\b")
_DROPPED_TOP_KEYS = ("notify",)
_LOCK_STALE_SECONDS = 30.0


def enabled(env: Optional[dict] = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("PUPPETMASTER_CODEX_LEAN_HOME", "1")).strip().lower() not in ("0", "false", "off", "no")


def user_home(env: Optional[dict] = None) -> Path:
    env = os.environ if env is None else env
    explicit = (env.get("CODEX_HOME") or "").strip()
    return Path(explicit).expanduser() if explicit else Path.home() / ".codex"


def worker_home_root() -> Path:
    from puppetmaster.community_observations import puppetmaster_home

    return puppetmaster_home() / "codex-worker-home"


def lean_config(text: str) -> str:
    """The user's config.toml without the tables and keys a worker must not inherit."""
    out: list[str] = []
    dropping_table = False
    skipping_key_depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if skipping_key_depth:
            skipping_key_depth += stripped.count("[") - stripped.count("]")
            continue
        if stripped.startswith("["):
            dropping_table = bool(_DROPPED_TABLE.match(stripped))
            if dropping_table:
                continue
        if dropping_table:
            continue
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key in _DROPPED_TOP_KEYS and not out_in_table(out):
            depth = stripped.count("[") - stripped.count("]")
            skipping_key_depth = depth if depth > 0 else 0
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def out_in_table(lines: list[str]) -> bool:
    return any(line.strip().startswith("[") for line in lines)


def _digest(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_atomic(path: Path, data: bytes, mode: int = 0o600) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "_Lock":
        deadline = time.monotonic() + 10
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > _LOCK_STALE_SECONDS:
                        self.path.unlink()
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"codex worker home lock busy: {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def prepare(env: Optional[dict] = None, *, root: Optional[Path] = None) -> Optional[Path]:
    """Create or refresh the lean home; None means use the user's own home."""
    source = user_home(env)
    if not (source / "auth.json").is_file():
        return None
    home = root or worker_home_root()
    home.mkdir(parents=True, exist_ok=True)
    with _Lock(home / ".sync.lock"):
        try:
            config = (source / "config.toml").read_text(encoding="utf-8")
        except OSError:
            config = ""
        lean = lean_config(config).encode("utf-8")
        if _digest(home / "config.toml") != hashlib.sha256(lean).hexdigest():
            _write_atomic(home / "config.toml", lean)
        agents = source / "AGENTS.md"
        if agents.is_file():
            data = agents.read_bytes()
            if _digest(home / "AGENTS.md") != hashlib.sha256(data).hexdigest():
                _write_atomic(home / "AGENTS.md", data, 0o644)
        user_auth = _digest(source / "auth.json")
        synced = _read(home / ".auth.synced")
        if user_auth != synced:
            _write_atomic(home / "auth.json", (source / "auth.json").read_bytes())
            (home / ".auth.synced").write_text(user_auth or "", encoding="utf-8")
    return home


def sync_back(env: Optional[dict] = None, *, root: Optional[Path] = None) -> bool:
    """Copy a worker-refreshed auth.json back to the user's home, only if theirs is unchanged."""
    source = user_home(env)
    home = root or worker_home_root()
    try:
        with _Lock(home / ".sync.lock"):
            synced = _read(home / ".auth.synced")
            worker = _digest(home / "auth.json")
            if not synced or worker in (None, synced) or _digest(source / "auth.json") != synced:
                return False
            _write_atomic(source / "auth.json", (home / "auth.json").read_bytes())
            (home / ".auth.synced").write_text(worker or "", encoding="utf-8")
            return True
    except (OSError, TimeoutError):
        return False


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def session_homes(env: Optional[dict] = None) -> list[Path]:
    """Codex homes whose session stores may hold a worker's thread, lean first."""
    homes = [worker_home_root()] if enabled(env) else []
    return homes + [user_home(env)]


def home_for_session(session_id: str, env: Optional[dict] = None) -> Optional[Path]:
    for home in session_homes(env):
        root = home / "sessions"
        if root.is_dir() and next(root.glob(f"*/*/*/rollout-*-{session_id}.jsonl"), None) is not None:
            return home
    return None
