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

Stock Codex installs its builtin skill bundle into ``skills/.system`` at
startup when the bundle's marker is missing or stale: it removes the
directory and writes it again. Parallel workers starting in one fresh home
all do that at once, and the last one writes the marker over a partial
bundle that no later start repairs (49 files became 41 or 1 in a stock
reproduction). :func:`ensure_system_skills` installs the bundle once per
Codex binary, with stock ``codex debug prompt-input`` in a private home that
has no credentials, and swaps it into the worker home under the home lock.
Workers then find a current marker and skip the install.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
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
            from puppetmaster.rules import strip_block_from_text

            # The user's own rules apply to workers; Puppetmaster's
            # orchestration block tells a pilot to delegate and only costs
            # a bounded worker context.
            raw = agents.read_bytes()
            try:
                data = strip_block_from_text(raw.decode("utf-8"))[0].encode("utf-8")
            except UnicodeDecodeError:
                data = raw  # not UTF-8 (a cp1252 file on Windows): copy it as is
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


_SYSTEM_SKILLS = Path("skills") / ".system"
_SYSTEM_RECORD = ".system-skills.json"
# Stock install takes well under a second; stay inside the lock's stale window.
_SEED_TIMEOUT_SECONDS = _LOCK_STALE_SECONDS / 2


def _binary_identity(command: list[str]) -> Optional[str]:
    """Path, size and mtime of the Codex launcher, enough to notice an upgrade."""
    if not command:
        return None
    launcher = command[-1]
    found = launcher if Path(launcher).is_file() else shutil.which(launcher)
    if not found:
        return None
    try:
        path = Path(found).resolve()
        stat = path.stat()
    except OSError:
        return None
    return json.dumps([*command[:-1], str(path), stat.st_size, stat.st_mtime_ns])


def _bundle_digest(home: Path) -> Optional[str]:
    root = home / _SYSTEM_SKILLS
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update((_digest(path) or "").encode("ascii") + b"\0")
    return digest.hexdigest()


def _system_skills_current(home: Path, identity: str) -> bool:
    try:
        record = json.loads((home / _SYSTEM_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(record, dict) and record.get("codex") == identity
            and record.get("bundle") == _bundle_digest(home))


def ensure_system_skills(home: Path, command: list[str]) -> bool:
    """Install stock Codex's builtin skill bundle into ``home`` once, race-free.

    ``command`` is the Codex launcher (``["codex"]`` or ``["node", "codex.js"]``).
    The bundle comes from the same binary, so its marker is current and every
    worker start skips its own install. Best effort: False leaves the home to
    stock Codex.
    """
    identity = _binary_identity(command)
    if identity is None:
        return False
    if _system_skills_current(home, identity):
        return True
    with _Lock(home / ".sync.lock"):
        if _system_skills_current(home, identity):
            return True
        with tempfile.TemporaryDirectory(prefix="pm-codex-seed-") as tmp:
            seed = Path(tmp)
            env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
            env["CODEX_HOME"] = str(seed)
            try:
                subprocess.run([*command, "debug", "prompt-input", "."], cwd=seed, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=_SEED_TIMEOUT_SECONDS, check=True)
            except (OSError, subprocess.SubprocessError):
                return False
            bundle = seed / _SYSTEM_SKILLS
            if not any(bundle.glob(".*marker")):
                return False
            target = home / _SYSTEM_SKILLS
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".system.{os.getpid()}.new")
            retired = target.with_name(f".system.{os.getpid()}.old")
            shutil.rmtree(staged, ignore_errors=True)
            try:
                shutil.copytree(bundle, staged)
                if target.exists():
                    os.replace(target, retired)
                os.replace(staged, target)
            finally:
                shutil.rmtree(staged, ignore_errors=True)
                shutil.rmtree(retired, ignore_errors=True)
        _write_atomic(home / _SYSTEM_RECORD, json.dumps(
            {"codex": identity, "bundle": _bundle_digest(home)}).encode("utf-8"))
    return True


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


def configured_model(env: Optional[dict] = None) -> str:
    """The top-level ``model`` in the user's Codex config, or ''.

    An unpinned Codex worker runs the model the user's own Codex would pick;
    a hard-coded fallback went stale (gpt-5.4-mini returned model_unavailable
    and failed every node of an unpinned flow).
    """
    try:
        text = (user_home(env) / "config.toml").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            break
        match = re.match(r'model\s*=\s*"([^"]+)"', stripped)
        if match:
            return match.group(1)
    return ""
