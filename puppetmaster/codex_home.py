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

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from puppetmaster.interprocess_lock import InterProcessFileLock

# Tables a bounded worker must not inherit: they cost context or reach out.
_DROPPED_TABLES = ("mcp_servers", "plugins", "marketplaces", "memories", "hooks")
_DROPPED_TOP_KEYS = ("notify",)
# The header of a dropped table with a bare name. No value of an array or of an inline table has this form.
_DROPPED_BARE_HEADER = re.compile(r"^\[\[?[ \t]*(" + "|".join(_DROPPED_TABLES) + r")[ \t]*[.\]]")
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
# The escapes of a basic TOML string (TOML 1.0, and `\e` and `\xHH` of TOML 1.1).
_TOML_ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|x[0-9A-Fa-f]{2}|.)")
_TOML_SIMPLE = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", "e": "\x1b", '"': '"', "\\": "\\"}
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


def _toml_scan(line: str, depth: int, quote: Optional[str]) -> Tuple[int, Optional[str]]:
    """The state after one line of TOML: the depth of open arrays and inline tables, and
    the delimiter of an open multi-line string (or None).

    Brackets in a string or a comment do not count. A statement ends where the depth is 0
    and no string is open, so a line that starts with "[" is a table header only at such
    a point.
    """
    i, n = 0, len(line)
    while i < n:
        if quote:
            if quote == '"""' and line[i] == "\\":
                # In a basic multi-line string a backslash escapes the next character.
                i += 2
            elif line.startswith(quote, i):
                # The content of a multi-line string can end with one or two quote
                # characters, so the string closes at the end of a run of three to five.
                i += 3
                extra = 0
                while extra < 2 and i < n and line[i] == quote[0]:
                    i, extra = i + 1, extra + 1
                quote = None
            else:
                i += 1
            continue
        ch = line[i]
        if ch == "#":
            break
        if line.startswith('"""', i) or line.startswith("'''", i):
            quote = line[i:i + 3]
            i += 3
            continue
        if ch == '"':
            i += 1
            while i < n and line[i] != '"':
                i += 2 if line[i] == "\\" else 1
        elif ch == "'":
            end = line.find("'", i + 1)
            i = n if end == -1 else end
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(depth - 1, 0)
        i += 1
    return depth, quote


def _toml_unescape(match: "re.Match[str]") -> str:
    """The character of one escape. An escape that TOML does not have stays as it is."""
    code = match.group(1)
    if code in _TOML_SIMPLE:
        return _TOML_SIMPLE[code]
    if len(code) > 1 and int(code[1:], 16) <= 0x10FFFF:
        return chr(int(code[1:], 16))
    return match.group(0)


def _split_key(text: str) -> Tuple[Optional[str], str]:
    """The first part of the TOML key at the start of ``text``, and the text after that
    part: ``a`` for ``a = 1``, ``a.b = 1``, ``"a".b = 1`` and ``'a' = 1``. The key is
    None when the text does not start with a key.
    """
    text = text.lstrip(" \t")
    if text.startswith('"'):
        i = 1
        while i < len(text) and text[i] != '"':
            i += 2 if text[i] == "\\" else 1
        if i >= len(text):
            return None, ""
        return _TOML_ESCAPE.sub(_toml_unescape, text[1:i]), text[i + 1:]
    if text.startswith("'"):
        end = text.find("'", 1)
        return (None, "") if end == -1 else (text[1:end], text[end + 1:])
    match = _BARE_KEY.match(text)
    return (match.group(0), text[match.end():]) if match else (None, "")


def _first_key(text: str) -> Optional[str]:
    return _split_key(text)[0]


def _toml_string(text: str) -> str:
    """The value of the TOML string at the start of ``text``, or '' when the text does
    not start with a string that ends on the same line.
    """
    for delimiter in ('"""', "'''", '"', "'"):
        if text.startswith(delimiter):
            break
    else:
        return ""
    basic = delimiter[0] == '"'
    body = text[len(delimiter):]
    i = 0
    while i < len(body):
        if basic and body[i] == "\\":
            i += 2
        elif body.startswith(delimiter, i):
            if len(delimiter) == 3:
                # The content can end with one or two quote characters.
                extra = 0
                while extra < 2 and body.startswith(delimiter[0], i + 3):
                    i, extra = i + 1, extra + 1
            return _TOML_ESCAPE.sub(_toml_unescape, body[:i]) if basic else body[:i]
        else:
            i += 1
    return ""


def lean_config(text: str) -> str:
    """The user's config.toml without the tables and keys a worker must not inherit.

    A dropped table goes in each form that TOML has for it: a header
    (``[mcp_servers.x]``, ``[[hooks]]``, ``["mcp_servers".x]``), a key with dots at the
    top level (``mcp_servers.x.command = "c"``) and an inline table at the top level
    (``mcp_servers = { x = { command = "c" } }``). A table that only starts with such a
    name (``[mcp_servers-old]``) is another table and stays.
    """
    out: list[str] = []
    depth = 0
    quote: Optional[str] = None
    in_table = dropped_table = dropped_key = False
    # Only "\n" ends a line. ``str.splitlines`` also ends a line at a form feed and at
    # U+2028, which are content of a TOML string.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        stripped = line.strip(" \t")
        if depth and quote is None and _DROPPED_BARE_HEADER.match(stripped):
            # The file has an open bracket that never closes. The header of a dropped
            # table ends that value: a damaged file must not keep the table.
            depth = 0
        if depth == 0 and quote is None:
            # A new statement: a table header, a key, a comment or an empty line.
            dropped_key = False
            if stripped.startswith("["):
                in_table = True
                dropped_table = _first_key(stripped.lstrip("[")) in _DROPPED_TABLES
            elif not in_table:
                dropped_key = _first_key(stripped) in _DROPPED_TOP_KEYS + _DROPPED_TABLES
        depth, quote = _toml_scan(line, depth, quote)
        if not (dropped_table or dropped_key):
            out.append(line)
    return "\n".join(out) + "\n"


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


def _sync_lock(home: Path) -> InterProcessFileLock:
    return InterProcessFileLock.for_target(home / "sync", stale_after=_LOCK_STALE_SECONDS)


def prepare(env: Optional[dict] = None, *, root: Optional[Path] = None) -> Optional[Path]:
    """Create or refresh the lean home; None means use the user's own home."""
    source = user_home(env)
    if not (source / "auth.json").is_file():
        return None
    home = root or worker_home_root()
    home.mkdir(parents=True, exist_ok=True)
    with _sync_lock(home):
        try:
            # utf-8-sig: Codex reads a config.toml that starts with a byte order mark.
            config = (source / "config.toml").read_text(encoding="utf-8-sig")
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
        else:
            # The user removed their AGENTS.md, so the worker must not keep the old copy.
            try:
                (home / "AGENTS.md").unlink()
            except FileNotFoundError:
                pass
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
        with _sync_lock(home):
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
    with _sync_lock(home):
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
                try:
                    os.replace(staged, target)
                except OSError:
                    if retired.exists():
                        os.replace(retired, target)
                    raise
            except OSError:
                return False  # the record stays stale, so the next start retries
            finally:
                shutil.rmtree(staged, ignore_errors=True)
                if target.exists():
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


_ROLLOUT_STAMP = re.compile(r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-")


def rollout_file(root: Path, session_id: str) -> Optional[Path]:
    """The rollout file of exactly ``session_id`` below a ``sessions`` folder.

    A rollout is ``rollout-<timestamp>-<session id>.jsonl``. The glob alone
    also accepts an id that is only the end of a different session id, so the
    part before the id must be a timestamp. The id is not a glob pattern.
    """
    name_end = f"-{session_id}.jsonl"
    for path in root.glob(f"*/*/*/rollout-*{glob.escape(name_end)}"):
        if _ROLLOUT_STAMP.fullmatch(path.name[: len(path.name) - len(name_end) + 1]):
            return path
    return None


def home_for_session(session_id: str, env: Optional[dict] = None) -> Optional[Path]:
    for home in session_homes(env):
        root = home / "sessions"
        if root.is_dir() and rollout_file(root, session_id) is not None:
            return home
    return None


def configured_model(env: Optional[dict] = None) -> str:
    """The top-level ``model`` in the user's Codex config, or ''.

    The value is a TOML string on one line. Each other value gives ''.

    An unpinned Codex worker runs the model the user's own Codex would pick;
    a hard-coded fallback went stale (gpt-5.4-mini returned model_unavailable
    and failed every node of an unpinned flow).
    """
    try:
        text = (user_home(env) / "config.toml").read_text(encoding="utf-8-sig")
    except (OSError, ValueError):
        return ""
    depth = 0
    quote: Optional[str] = None
    for line in text.split("\n"):
        if depth == 0 and quote is None:
            # A new statement. The first table header ends the top level.
            stripped = line.strip(" \t")
            if stripped.startswith("["):
                break
            key, rest = _split_key(stripped)
            rest = rest.lstrip(" \t")
            if key == "model" and rest.startswith("="):
                return _toml_string(rest[1:].lstrip(" \t"))
        depth, quote = _toml_scan(line, depth, quote)
    return ""
