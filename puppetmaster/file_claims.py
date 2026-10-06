"""Durable, cooperative ownership claims for files in a Git workspace.

This is deliberately a small primitive.  It records who has agreed to edit a
file; callers remain responsible for enforcing that agreement before editing.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import time
from typing import Callable, Iterable, Iterator, Optional, Union

from puppetmaster.state import app_state_root


PathLike = Union[str, Path]
_MAX_OWNER_LENGTH = 256
_MAX_TTL_SECONDS = 365 * 24 * 60 * 60
_BUSY_TIMEOUT_MS = 2_000


class FileClaimConflict(RuntimeError):
    """Raised when a live claim belongs to another (or the same) claimant."""

    def __init__(self, path: str, owner: str) -> None:
        super().__init__("file is already claimed by %s: %s" % (owner, path))
        self.path = path
        self.owner = owner


@dataclass(frozen=True)
class FileClaim:
    """The opaque fencing token and metadata returned by an acquisition."""

    claim_id: str
    repo_identity: str
    path: str
    owner: str
    acquired_at: float
    renewed_at: float
    ttl_seconds: float
    managed: bool = False
    owner_pid: Optional[int] = None
    owner_start_identity: Optional[str] = None
    state: str = "active"


@dataclass(frozen=True)
class FileClaimAuditRecord:
    event: str
    repo_identity: str
    path: str
    claim_id: str
    old_owner: Optional[str]
    new_owner: Optional[str]
    occurred_at: float


def default_file_claim_db_path(workspace: Optional[PathLike] = None) -> Path:
    """Return the application-wide ledger shared by all state directories."""
    return app_state_root() / "file_claims.sqlite3"


class FileClaimRegistry:
    """SQLite-backed file claims shared by independent local processes."""

    def __init__(
        self, db_path: PathLike, *, clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def acquire(
        self, repo: PathLike, path: PathLike, owner: str, ttl_seconds: float,
        *, force: bool = False, managed: bool = False,
        owner_pid: Optional[int] = None, owner_start_identity: Optional[str] = None,
    ) -> FileClaim:
        """Acquire one path, or raise :class:`FileClaimConflict` if it is live.

        ``force`` fences cooperative tokens only; it does not terminate a live
        writer that has already ignored or outlived its token.
        """
        return self.acquire_many(
            repo, [path], owner, ttl_seconds, force=force, managed=managed,
            owner_pid=owner_pid, owner_start_identity=owner_start_identity,
        )[0]

    def acquire_many(
        self, repo: PathLike, paths: Iterable[PathLike], owner: str,
        ttl_seconds: float, *, force: bool = False, managed: bool = False,
        owner_pid: Optional[int] = None, owner_start_identity: Optional[str] = None,
    ) -> list[FileClaim]:
        """Acquire all paths atomically; a conflict leaves no new claims behind."""
        owner = _validate_owner(owner)
        ttl_seconds = _validate_ttl(ttl_seconds)
        repo_identity, normalized_paths = self._claim_keys(repo, paths)
        now = self._now()
        with self._transaction() as connection:
            # A claim on a directory fences every descendant, and vice versa.
            # Remove descendants requested alongside an ancestor before looking
            # at the database so callers never create redundant claims.
            normalized_paths = _dedupe_ancestors(normalized_paths)
            rows = self._live_overlapping_rows(connection, repo_identity, normalized_paths, now)
            if rows and not force:
                row = rows[0]
                raise FileClaimConflict(str(row["path"]), str(row["owner"]))
            if rows and force and any(bool(row["managed"]) for row in rows):
                raise FileClaimConflict(str(rows[0]["path"]), str(rows[0]["owner"]))
            for row in rows:
                connection.execute("DELETE FROM file_claims WHERE repo_identity=? AND path=?",
                                   (repo_identity, row["path"]))
                self._audit(connection, "stolen", repo_identity, str(row["path"]),
                            str(row["claim_id"]), str(row["owner"]), owner, now)
            claimed = []
            for normalized_path in normalized_paths:
                claim_id = secrets.token_urlsafe(24)
                connection.execute(
                    "INSERT INTO file_claims(repo_identity,path,claim_id,owner,acquired_at,renewed_at,ttl_seconds,managed,owner_pid,owner_start_identity,state) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (repo_identity, normalized_path, claim_id, owner, now, now, ttl_seconds,
                     int(managed), owner_pid, owner_start_identity, "active"),
                )
                self._audit(connection, "acquired", repo_identity, normalized_path,
                            claim_id, None, owner, now)
                row = connection.execute(
                    "SELECT * FROM file_claims WHERE repo_identity=? AND path=?",
                    (repo_identity, normalized_path),
                ).fetchone()
                claimed.append(_claim_from_row(row))
            return claimed

    def renew(
        self, repo: PathLike, path: PathLike, claim_id: str,
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """Renew a claim only when its current fencing token matches."""
        repo_identity, normalized = self._claim_key(repo, path)
        _validate_claim_id(claim_id)
        if ttl_seconds is not None:
            ttl_seconds = _validate_ttl(ttl_seconds)
        now = self._now()
        with self._transaction() as connection:
            row = self._row(connection, repo_identity, normalized)
            if row is None or (not bool(row["managed"]) and _expired(row, now)) or row["claim_id"] != claim_id:
                return False
            duration = float(ttl_seconds if ttl_seconds is not None else row["ttl_seconds"])
            connection.execute(
                "UPDATE file_claims SET renewed_at=?, ttl_seconds=? WHERE repo_identity=? AND path=?",
                (now, duration, repo_identity, normalized),
            )
            self._audit(connection, "renewed", repo_identity, normalized, claim_id,
                        str(row["owner"]), str(row["owner"]), now)
            return True

    def release(self, repo: PathLike, path: PathLike, claim_id: str) -> bool:
        """Release only the exact current token; stale tokens return ``False``."""
        repo_identity, normalized = self._claim_key(repo, path)
        _validate_claim_id(claim_id)
        now = self._now()
        with self._transaction() as connection:
            row = self._row(connection, repo_identity, normalized)
            if row is None or row["claim_id"] != claim_id:
                return False
            connection.execute(
                "DELETE FROM file_claims WHERE repo_identity=? AND path=?",
                (repo_identity, normalized),
            )
            self._audit(connection, "released", repo_identity, normalized, claim_id,
                        str(row["owner"]), None, now)
            return True

    def release_many(
        self, repo: PathLike, claims: Iterable[tuple[PathLike, str]],
    ) -> int:
        """Release matching tokens atomically, returning the number released."""
        repo_identity, normalized_claims = self._claim_token_keys(repo, claims)
        now = self._now()
        released = 0
        with self._transaction() as connection:
            for path, claim_id in normalized_claims:
                row = self._row(connection, repo_identity, path)
                if row is None or row["claim_id"] != claim_id:
                    continue
                connection.execute("DELETE FROM file_claims WHERE repo_identity=? AND path=?",
                                   (repo_identity, path))
                self._audit(connection, "released", repo_identity, path, claim_id,
                            str(row["owner"]), None, now)
                released += 1
        return released

    def audit_records(self) -> list[FileClaimAuditRecord]:
        """Return durable audit metadata, never file contents."""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM file_claim_audit ORDER BY id").fetchall()
        return [FileClaimAuditRecord(
            event=str(row["event"]), repo_identity=str(row["repo_identity"]),
            path=str(row["path"]), claim_id=str(row["claim_id"]),
            old_owner=row["old_owner"], new_owner=row["new_owner"],
            occurred_at=float(row["occurred_at"]),
        ) for row in rows]

    def repository_identity(self, repo: PathLike) -> tuple[str, Path]:
        """Return canonical Git common-dir identity and this worktree's root."""
        return _repository_identity(repo)

    def _claim_keys(self, repo: PathLike, paths: Iterable[PathLike]) -> tuple[str, list[str]]:
        identity, root = self.repository_identity(repo)
        normalized = []
        seen = set()
        for path in paths:
            value = _normalize_path(root, path)
            if value not in seen:
                seen.add(value)
                normalized.append(value)
        if not normalized:
            raise ValueError("paths must not be empty")
        return identity, normalized

    def _claim_key(self, repo: PathLike, path: PathLike) -> tuple[str, str]:
        identity, paths = self._claim_keys(repo, [path])
        return identity, paths[0]

    def _claim_token_keys(self, repo: PathLike, claims: Iterable[tuple[PathLike, str]]) -> tuple[str, list[tuple[str, str]]]:
        identity, root = self.repository_identity(repo)
        normalized = []
        for path, claim_id in claims:
            _validate_claim_id(claim_id)
            normalized.append((_normalize_path(root, path), claim_id))
        return identity, normalized

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS file_claims (
                    repo_identity TEXT NOT NULL,
                    path TEXT NOT NULL,
                    claim_id TEXT NOT NULL UNIQUE,
                    owner TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    renewed_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL,
                    managed INTEGER NOT NULL DEFAULT 0,
                    owner_pid INTEGER,
                    owner_start_identity TEXT,
                    state TEXT NOT NULL DEFAULT 'active',
                    PRIMARY KEY (repo_identity, path)
                );
                CREATE TABLE IF NOT EXISTS file_claim_audit (
                    id INTEGER PRIMARY KEY,
                    event TEXT NOT NULL,
                    repo_identity TEXT NOT NULL,
                    path TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    old_owner TEXT,
                    new_owner TEXT,
                    occurred_at REAL NOT NULL
                );
            """)
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(file_claims)")}
            for name, definition in (
                ("managed", "INTEGER NOT NULL DEFAULT 0"),
                ("owner_pid", "INTEGER"),
                ("owner_start_identity", "TEXT"),
                ("state", "TEXT NOT NULL DEFAULT 'active'"),
            ):
                if name not in columns:
                    connection.execute("ALTER TABLE file_claims ADD COLUMN %s %s" % (name, definition))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _row(connection: sqlite3.Connection, identity: str, path: str) -> Optional[sqlite3.Row]:
        return connection.execute("SELECT * FROM file_claims WHERE repo_identity=? AND path=?", (identity, path)).fetchone()

    def _live_overlapping_rows(self, connection: sqlite3.Connection, identity: str, paths: list[str], now: float) -> list[sqlite3.Row]:
        rows = []
        candidates = connection.execute(
            "SELECT * FROM file_claims WHERE repo_identity=? ORDER BY path", (identity,)
        ).fetchall()
        for row in candidates:
            path = str(row["path"])
            if not any(_paths_overlap(path, requested) for requested in paths):
                continue
            if bool(row["managed"]):
                if not _managed_owner_live(row):
                    connection.execute(
                        "UPDATE file_claims SET state='quarantined' WHERE repo_identity=? AND path=?",
                        (identity, path),
                    )
                    self._audit(connection, "quarantined", identity, path, str(row["claim_id"]),
                                str(row["owner"]), str(row["owner"]), now)
                rows.append(row)
            elif _expired(row, now):
                connection.execute("DELETE FROM file_claims WHERE repo_identity=? AND path=?", (identity, path))
                self._audit(connection, "expired", identity, path, str(row["claim_id"]),
                            str(row["owner"]), None, now)
            else:
                rows.append(row)
        return rows

    @staticmethod
    def _audit(connection: sqlite3.Connection, event: str, identity: str, path: str,
               claim_id: str, old_owner: Optional[str], new_owner: Optional[str], now: float) -> None:
        connection.execute(
            "INSERT INTO file_claim_audit(event,repo_identity,path,claim_id,old_owner,new_owner,occurred_at) VALUES(?,?,?,?,?,?,?)",
            (event, identity, path, claim_id, old_owner, new_owner, now),
        )

    def _now(self) -> float:
        now = self._clock()
        if not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now):
            raise ValueError("clock must return a finite number")
        return float(now)


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _repository_identity(repo: PathLike) -> tuple[str, Path]:
    candidate = Path(repo).expanduser().resolve()
    try:
        root_text = _git(candidate, "rev-parse", "--show-toplevel")
        common_text = _git(candidate, "rev-parse", "--git-common-dir")
    except (OSError, subprocess.CalledProcessError) as exc:
        # Non-Git workspaces still need a stable, canonical admission scope.
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError("repo must be an existing workspace directory") from exc
        return str(candidate), candidate
    root = Path(root_text).resolve()
    common = Path(common_text)
    if not common.is_absolute():
        common = candidate / common
    # The physical worktree is the isolation boundary. The common Git
    # directory is shared metadata, not a shared write target.
    return str(root), root


def _normalize_path(root: Path, path: PathLike) -> str:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        normalized = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes the workspace: %s" % path) from exc
    parts = Path(normalized).parts
    if parts and (parts[0] == ".git" or parts[0].lower() == ".git"):
        raise ValueError("Git internals cannot be claimed: %s" % path)
    # Git paths are case-sensitive on POSIX and case-insensitive on Windows.
    return normalized.lower() if os.name == "nt" else normalized


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left == "." or right == "." or left.startswith(right + "/") or right.startswith(left + "/")


def _dedupe_ancestors(paths: list[str]) -> list[str]:
    result = []
    for path in paths:
        if any(existing == "." or path == existing or path.startswith(existing + "/") for existing in result):
            continue
        result = [existing for existing in result if not (path == "." or existing.startswith(path + "/"))]
        if path not in result:
            result.append(path)
    return result


def _validate_owner(owner: str) -> str:
    if not isinstance(owner, str) or not owner.strip() or len(owner) > _MAX_OWNER_LENGTH or "\x00" in owner:
        raise ValueError("owner must be a nonempty string of at most 256 characters")
    return owner


def _validate_ttl(ttl_seconds: float) -> float:
    if (not isinstance(ttl_seconds, (int, float)) or isinstance(ttl_seconds, bool)
            or not math.isfinite(ttl_seconds) or ttl_seconds <= 0 or ttl_seconds > _MAX_TTL_SECONDS):
        raise ValueError("ttl_seconds must be a finite value between 0 and one year")
    return float(ttl_seconds)


def _validate_claim_id(claim_id: str) -> None:
    if not isinstance(claim_id, str) or not claim_id or len(claim_id) > 256 or "\x00" in claim_id:
        raise ValueError("claim_id must be a nonempty bounded string")


def _expired(row: sqlite3.Row, now: float) -> bool:
    return float(row["renewed_at"]) + float(row["ttl_seconds"]) <= now


def process_start_identity(pid: Optional[int] = None) -> Optional[str]:
    """Return a best-effort identity that prevents PID reuse from fencing."""
    target = int(pid or os.getpid())
    if target <= 0:
        return None
    try:
        stat = Path("/proc/%d/stat" % target).read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def _managed_owner_live(row: sqlite3.Row) -> bool:
    pid = row["owner_pid"]
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    recorded = row["owner_start_identity"]
    return recorded is None or recorded == process_start_identity(int(pid))


def _claim_from_row(row: sqlite3.Row) -> FileClaim:
    return FileClaim(
        claim_id=str(row["claim_id"]), repo_identity=str(row["repo_identity"]),
        path=str(row["path"]), owner=str(row["owner"]),
        acquired_at=float(row["acquired_at"]), renewed_at=float(row["renewed_at"]),
        ttl_seconds=float(row["ttl_seconds"]),
        managed=bool(row["managed"]),
        owner_pid=(int(row["owner_pid"]) if row["owner_pid"] is not None else None),
        owner_start_identity=row["owner_start_identity"],
        state=str(row["state"]),
    )
