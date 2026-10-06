from __future__ import annotations

import hashlib
from collections import OrderedDict
import threading
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from puppetmaster.readonly import ReadUnavailable


STATE_DIR_ENV = "PUPPETMASTER_STATE_DIR"


def state_identity(path: Union[Path, str]) -> str:
    """Return a stable opaque identity for a resolved state directory."""
    resolved = str(Path(path).expanduser().resolve())
    return "state_" + hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def resolve_state_dir(value: Optional[Union[Path, str]] = None, cwd: Optional[Path] = None) -> Path:
    """Resolve Puppetmaster state without dirtying the target repository by default."""
    root = cwd or Path.cwd()
    if value:
        return _resolve_user_path(value, root)
    env_value = os.environ.get(STATE_DIR_ENV)
    if env_value:
        return _resolve_user_path(env_value, root)
    return default_state_dir(root)


def default_state_dir(cwd: Optional[Path] = None) -> Path:
    base = cwd or Path.cwd()
    workspace = _git_root(base) or base
    return project_state_dir_for(workspace)


def projects_root() -> Path:
    """Return the parent directory holding every project-scoped state dir."""
    return app_state_root() / "projects"


def project_state_dir_for(workspace: Union[Path, str]) -> Path:
    """Return the state dir ``workspace`` hashes to, without probing git.

    Split out of ``default_state_dir`` so a caller can ask "what state dir
    would *that* directory get?" for an arbitrary path — the identity check
    in ``state_health`` needs to map a sibling repo to its state dir without
    shelling out to git and without re-deriving the slug/digest formula.

    ``workspace`` is used verbatim (already the git root, or the cwd when the
    caller has no repo): this helper never resolves a repository for you.
    """
    resolved = Path(workspace).resolve()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name).strip("-") or "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return projects_root() / f"{slug}-{digest}"


APP_STATE_ROOT_ENV = "PUPPETMASTER_APP_STATE_ROOT"


def app_state_root() -> Path:
    override = os.environ.get(APP_STATE_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "puppetmaster"
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "puppetmaster"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "puppetmaster"
    return Path.home() / ".local" / "state" / "puppetmaster"


def list_project_state_dirs() -> list[Path]:
    """Return every project-scoped state directory currently on disk.

    The MCP server and CLI both compute a per-workspace state dir hashed
    from the resolved workspace path, so `puppetmaster show <job_id>`
    only finds jobs created from the same workspace by default. This
    helper lets callers iterate every known project to support cross-
    workspace job lookup without forcing users to memorize the hash.
    """
    root = projects_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def find_state_dir_for_job(job_id: str) -> Optional[Path]:
    """Weak lookup rejects readable duplicates; unavailable stores are skipped."""
    if not job_id:
        return None
    if (not isinstance(job_id, str) or len(job_id) > 1024
            or Path(job_id).name != job_id or job_id in {".", ".."} or "\\" in job_id):
        raise ValueError("invalid job_id")
    matches = _owning_state_dirs(list_project_state_dirs(), job_id)
    if len(matches) > 1:
        raise ValueError("ambiguous job_id; provide job_ref or state_dir")
    return matches[0] if matches else None


def _resolve_user_path(value: Union[Path, str], cwd: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else cwd / path


def _git_root(cwd: Path) -> Optional[Path]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return Path(output) if output else None


def resolve_job_state(*, job_id=None, job_ref=None, state_dir=None, cwd=None, default_dir=None) -> Path:
    """Resolve identity before choosing a store, including explicit directories."""
    from puppetmaster.models import JobRef
    if isinstance(job_ref, JobRef):
        job_ref = job_ref.as_dict()
    if job_ref is not None:
        if isinstance(job_ref, dict):
            job_ref = JobRef(**job_ref).as_dict()
        if not isinstance(job_ref, dict):
            raise ValueError("job_ref must contain job_id and state_id")
        if any(not isinstance(job_ref.get(k), str) or not job_ref[k].strip()
               for k in ("job_id", "state_id")):
            raise ValueError("job_ref must contain job_id and state_id")
        if job_id is not None and job_id != job_ref["job_id"]:
            raise ValueError("job_id conflicts with job_ref.job_id")
        job_id = job_ref["job_id"]
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()
                              or Path(job_id).name != job_id or job_id in {".", ".."}
                              or "/" in job_id or "\\" in job_id):
        raise ValueError("invalid job_id")
    resolved = Path(default_dir) if default_dir is not None and not state_dir else resolve_state_dir(state_dir, cwd=cwd)
    expected = job_ref.get("state_id") if job_ref else None
    if state_dir:
        if expected and state_identity(resolved) != expected:
            raise ValueError("job_ref.state_id does not match explicit state_dir")
        if job_id and not state_owns_job(resolved, job_id, job_ref=job_ref):
            raise ValueError("explicit state_dir does not own job_id")
        return resolved
    if not job_id:
        return resolved
    candidates = set(list_project_state_dirs()) | {resolved}
    matches = _owning_state_dirs(
        (p for p in candidates if not expected or state_identity(p) == expected),
        job_id, job_ref=job_ref, required_root=resolved)
    if len(matches) > 1:
        raise ValueError("ambiguous job_id; provide job_ref or state_dir")
    if matches:
        return matches[0]
    if expected:
        raise ValueError("job_ref.state_id has no matching owning store")
    return resolved


def resolve_metadata_state(*, job_ref=None, job_id=None, state_dir=None, cwd=None, default_dir=None) -> Path:
    """Select a metadata store by path identity; readers validate its snapshot."""
    if job_id is not None and (job_ref is None or job_id != job_ref.job_id):
        raise ValueError("bounded job selection requires matching job_ref")
    resolved = Path(default_dir) if default_dir is not None and not state_dir else resolve_state_dir(state_dir, cwd=cwd)
    if job_ref is None or state_identity(resolved) == job_ref.state_id:
        return resolved
    if state_dir:
        raise ValueError("job_ref.state_id does not match explicit state_dir")
    matches = [p for p in list_project_state_dirs() if state_identity(p) == job_ref.state_id]
    if len(matches) != 1:
        raise ValueError("job_ref.state_id has no unique matching store")
    return matches[0]


class _OwnershipReader:
    backend_name = "sqlite"

    def __init__(self):
        # One scan intentionally reopens its helper across different roots.
        self._readonly_reuse_key = object()


# Cache only proven unscoped SQL membership. Directory stamps are essential:
# creating then unlinking live WAL sidecars can leave the main DB unchanged.
_ownership_cache = OrderedDict()
_ownership_cache_lock = threading.Lock()


def _ownership_stamp(root):
    result = []
    for path in (root, *(root / ('state.sqlite3' + suffix)
                        for suffix in ('', '-wal', '-shm', '-journal'))):
        try:
            st = path.stat()
            result.append((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        except FileNotFoundError:
            result.append(None)
    return tuple(result)


def _owning_state_dirs(candidates, job_id, *, job_ref=None, required_root=None):
    """Weak discovery skips unavailable global candidates, never its target.

    An unreadable global store may contain another owner; weak IDs cannot
    certify uniqueness. Scoped references and the caller's target fail closed.
    """
    reader = _OwnershipReader()
    required = Path(required_root).resolve() if required_root is not None else None
    matches = []
    try:
        roots = {Path(p).resolve(): Path(p) for p in candidates}
        for canonical, root in sorted(roots.items()):
            try:
                owned = state_owns_job(root, job_id, job_ref=job_ref, _reader=reader)
            except sqlite3.OperationalError as exc:
                if job_ref is not None or canonical == required:
                    raise
                code = getattr(exc, 'sqlite_errorcode', None)
                locked = (isinstance(code, int) and (code & 0xff) in (5, 6)) if code is not None else str(exc) in (
                    'database is locked', 'database table is locked', 'database schema is locked')
                if not isinstance(exc, ReadUnavailable) and not locked:
                    raise
                continue
            if owned:
                matches.append(root)
        if len(matches) > 1:
            raise ValueError("ambiguous job_id; provide job_ref or state_dir")
        return matches
    finally:
        # Close through the slot so the cleanup registry drops its owner too;
        # a direct transport.close() left one registry entry per scan.
        slot = getattr(reader, '_readonly_slot', None)
        if slot is not None:
            slot.discard()
        else:
            transport = getattr(reader, '_readonly_transport', None)
            if transport is not None:
                transport.close()


def state_owns_job(root: Path, job_id: str, *, job_ref=None, _reader=None) -> bool:
    """Metadata-only ownership lookup; a leftover directory is not a SQL row."""
    from types import SimpleNamespace
    from puppetmaster.readonly import connect, selection
    database = root / "state.sqlite3"
    if database.exists():
        stamp = _ownership_stamp(root) if _reader is not None and job_ref is None else None
        key = (root, job_id)
        if stamp is not None:
            with _ownership_cache_lock:
                cached = _ownership_cache.get(key)
                if cached is not None and cached[0] == stamp:
                    _ownership_cache.move_to_end(key)
                    return cached[1]
        selected = _reader if _reader is not None else SimpleNamespace(backend_name="sqlite")
        selected.root = root
        selected._read_selection = selection(selected)
        connection = connect(selected, reuse=_reader is not None)
        try:
            if job_ref and job_ref.get("version", 1) == 2:
                from puppetmaster.identity import read_identity, StoreIdentityError
                if read_identity(connection, "sqlite") != job_ref["incarnation"]:
                    raise StoreIdentityError("stale JobRef: store incarnation changed")
            owned = connection.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone() is not None
            if selection(selected) != selected._read_selection:
                from puppetmaster.identity import StoreIdentityError
                raise StoreIdentityError("store replaced during ownership selection")
        finally:
            connection.close()
        if stamp is not None and _ownership_stamp(root) == stamp:
            with _ownership_cache_lock:
                _ownership_cache[key] = (stamp, owned)
                _ownership_cache.move_to_end(key)
                while len(_ownership_cache) > 4096:
                    _ownership_cache.popitem(last=False)
        return owned
    if job_ref and job_ref.get("version", 1) == 2:
        from puppetmaster.identity import reference_at, StoreIdentityError
        if reference_at(root, job_id).incarnation != job_ref["incarnation"]:
            raise StoreIdentityError("stale JobRef: store incarnation changed")
    return (root / "jobs" / job_id / "job.json").is_file()
