"""User-global passive rate-limit harvest + preemptive admission.

Agentic HTTP paths call :func:`record_from_headers` (best-effort) after every
provider response. Admission consults the harvested remaining/reset before
dialing so workers can fail over *before* a hard 429 when the provider already
advertised ``remaining=0``.

State lives under ``~/.puppetmaster/rate_limits.sqlite3`` (WAL, no secrets).
Kill harvest with ``PUPPETMASTER_RATE_LIMIT_HARVEST=0``; kill only admission
with ``PUPPETMASTER_RATE_LIMIT_ADMISSION=0``.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from puppetmaster.failure import RATE_LIMIT
from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.rate_limit_headers import (
    RateLimitSnapshot,
    RateLimitWindow,
    parse_rate_limit_headers,
)

_ENV_ENABLED = "PUPPETMASTER_RATE_LIMIT_HARVEST"
_ENV_PATH = "PUPPETMASTER_RATE_LIMIT_PATH"
_ENV_ADMIT = "PUPPETMASTER_RATE_LIMIT_ADMISSION"

QUOTA_EXHAUSTED_BODY = "rate_limit_quota_exhausted"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_limits (
    admission_key TEXT NOT NULL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    requests_limit INTEGER,
    requests_remaining INTEGER,
    requests_reset_at REAL,
    tokens_limit INTEGER,
    tokens_remaining INTEGER,
    tokens_reset_at REAL,
    retry_after_at REAL,
    updated_at REAL NOT NULL,
    http_status INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rate_limits_updated
    ON rate_limits(updated_at);
"""

_HARVEST_KEY: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "puppetmaster_rate_limit_harvest_key",
    default=None,
)
_HARVEST_META: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "puppetmaster_rate_limit_harvest_meta",
    default=("", ""),
)

_STORE_LOCK = threading.RLock()
_STORE_CACHE: dict[str, "RateLimitStore"] = {}


def harvest_enabled() -> bool:
    """False when ``PUPPETMASTER_RATE_LIMIT_HARVEST`` is 0/false/off/no."""
    val = os.environ.get(_ENV_ENABLED, "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def admission_enabled() -> bool:
    """Admission defaults on with harvest; independent kill switch available."""
    if not harvest_enabled():
        return False
    val = os.environ.get(_ENV_ADMIT, "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def default_rate_limit_path() -> Path:
    override = (os.environ.get(_ENV_PATH) or "").strip()
    if override:
        return Path(override).expanduser()
    home = (os.environ.get("PUPPETMASTER_HOME") or "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "rate_limits.sqlite3"


@dataclass(frozen=True)
class RateLimitRecord:
    """One durable harvested row (no credential material)."""

    admission_key: str
    provider: str
    model: str
    requests_limit: Optional[int]
    requests_remaining: Optional[int]
    requests_reset_at: Optional[float]
    tokens_limit: Optional[int]
    tokens_remaining: Optional[int]
    tokens_reset_at: Optional[float]
    retry_after_at: Optional[float]
    updated_at: float
    http_status: Optional[int] = None

    def snapshot(self) -> RateLimitSnapshot:
        requests = None
        if (
            self.requests_limit is not None
            or self.requests_remaining is not None
            or self.requests_reset_at is not None
        ):
            requests = RateLimitWindow(
                kind="requests",
                limit=self.requests_limit,
                remaining=self.requests_remaining,
                reset_at=self.requests_reset_at,
            )
        tokens = None
        if (
            self.tokens_limit is not None
            or self.tokens_remaining is not None
            or self.tokens_reset_at is not None
        ):
            tokens = RateLimitWindow(
                kind="tokens",
                limit=self.tokens_limit,
                remaining=self.tokens_remaining,
                reset_at=self.tokens_reset_at,
            )
        return RateLimitSnapshot(
            requests=requests,
            tokens=tokens,
            retry_after_at=self.retry_after_at,
        )

    def is_blocking(self, *, now: Optional[float] = None) -> bool:
        return self.snapshot().is_exhausted(now=now)

    def blocking_reset_at(self, *, now: Optional[float] = None) -> Optional[float]:
        return self.snapshot().blocking_reset_at(now=now)


class RateLimitStore:
    """Small WAL SQLite store for concurrent worker harvests."""

    def __init__(self, path: Optional[Path] = None, *, busy_timeout_ms: int = 5000):
        self.path = Path(path) if path is not None else default_rate_limit_path()
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._init_lock = threading.Lock()
        self._initialized = False

    def _ensure(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            mkdir_private(self.path.parent)
            connection = self._connect()
            try:
                connection.executescript(_SCHEMA)
                connection.commit()
            finally:
                connection.close()
            chmod_private_file(self.path)
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=self.busy_timeout_ms / 1000.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL").fetchone()
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        self._ensure()
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, admission_key: str) -> Optional[RateLimitRecord]:
        key = (admission_key or "").strip()
        if not key:
            return None
        with self._session() as connection:
            row = connection.execute(
                "SELECT * FROM rate_limits WHERE admission_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def upsert(
        self,
        *,
        admission_key: str,
        provider: str = "",
        model: str = "",
        snapshot: RateLimitSnapshot,
        http_status: Optional[int] = None,
        now: Optional[float] = None,
    ) -> RateLimitRecord:
        key = (admission_key or "").strip()
        if not key:
            raise ValueError("admission_key is required")
        clock = time.time() if now is None else float(now)
        # Coalesce with the prior row so a Retry-After-only 429 does not wipe a
        # still-active tokens/requests window, and an older response cannot
        # clobber a newer one (updated_at guard).
        prior = self.get(key)
        if prior is not None and prior.updated_at > clock:
            return prior
        req = snapshot.requests
        tok = snapshot.tokens
        if prior is not None:
            if req is None and (
                prior.requests_limit is not None
                or prior.requests_remaining is not None
                or prior.requests_reset_at is not None
            ):
                req = RateLimitWindow(
                    kind="requests",
                    limit=prior.requests_limit,
                    remaining=prior.requests_remaining,
                    reset_at=prior.requests_reset_at,
                )
            if tok is None and (
                prior.tokens_limit is not None
                or prior.tokens_remaining is not None
                or prior.tokens_reset_at is not None
            ):
                tok = RateLimitWindow(
                    kind="tokens",
                    limit=prior.tokens_limit,
                    remaining=prior.tokens_remaining,
                    reset_at=prior.tokens_reset_at,
                )
        retry_after = snapshot.retry_after_at
        if retry_after is None and prior is not None:
            retry_after = prior.retry_after_at
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO rate_limits (
                    admission_key, provider, model,
                    requests_limit, requests_remaining, requests_reset_at,
                    tokens_limit, tokens_remaining, tokens_reset_at,
                    retry_after_at, updated_at, http_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(admission_key) DO UPDATE SET
                    provider=excluded.provider,
                    model=excluded.model,
                    requests_limit=excluded.requests_limit,
                    requests_remaining=excluded.requests_remaining,
                    requests_reset_at=excluded.requests_reset_at,
                    tokens_limit=excluded.tokens_limit,
                    tokens_remaining=excluded.tokens_remaining,
                    tokens_reset_at=excluded.tokens_reset_at,
                    retry_after_at=excluded.retry_after_at,
                    updated_at=excluded.updated_at,
                    http_status=excluded.http_status
                WHERE excluded.updated_at >= rate_limits.updated_at
                """,
                (
                    key,
                    (provider or "").strip().lower(),
                    (model or "").strip(),
                    None if req is None else req.limit,
                    None if req is None else req.remaining,
                    None if req is None else req.reset_at,
                    None if tok is None else tok.limit,
                    None if tok is None else tok.remaining,
                    None if tok is None else tok.reset_at,
                    retry_after,
                    clock,
                    http_status,
                ),
            )
        record = self.get(key)
        assert record is not None
        return record

    def list_blocking(self, *, now: Optional[float] = None, limit: int = 20) -> list[RateLimitRecord]:
        clock = time.time() if now is None else float(now)
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM rate_limits ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit) * 4),),
            ).fetchall()
        out: list[RateLimitRecord] = []
        for row in rows:
            record = _row_to_record(row)
            if record.is_blocking(now=clock):
                out.append(record)
            if len(out) >= limit:
                break
        return out


def _row_to_record(row: sqlite3.Row) -> RateLimitRecord:
    def _opt_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def _opt_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    return RateLimitRecord(
        admission_key=str(row["admission_key"]),
        provider=str(row["provider"] or ""),
        model=str(row["model"] or ""),
        requests_limit=_opt_int(row["requests_limit"]),
        requests_remaining=_opt_int(row["requests_remaining"]),
        requests_reset_at=_opt_float(row["requests_reset_at"]),
        tokens_limit=_opt_int(row["tokens_limit"]),
        tokens_remaining=_opt_int(row["tokens_remaining"]),
        tokens_reset_at=_opt_float(row["tokens_reset_at"]),
        retry_after_at=_opt_float(row["retry_after_at"]),
        updated_at=float(row["updated_at"]),
        http_status=_opt_int(row["http_status"]),
    )


def get_rate_limit_store(path: Optional[Path] = None) -> RateLimitStore:
    """Process-cached store for the default (or overridden) path."""
    resolved = Path(path) if path is not None else default_rate_limit_path()
    key = str(resolved)
    with _STORE_LOCK:
        store = _STORE_CACHE.get(key)
        if store is None:
            store = RateLimitStore(resolved)
            _STORE_CACHE[key] = store
        return store


def reset_rate_limit_store_cache() -> None:
    """Drop cached stores (tests)."""
    with _STORE_LOCK:
        _STORE_CACHE.clear()


@contextlib.contextmanager
def harvesting_rate_limits(
    admission_key: str,
    *,
    provider: str = "",
    model: str = "",
) -> Iterator[str]:
    """Bind the admission key for nested ``_post_json`` / ``_open_stream`` harvests."""
    key = (admission_key or "").strip()
    token_key = _HARVEST_KEY.set(key or None)
    token_meta = _HARVEST_META.set(((provider or "").strip().lower(), (model or "").strip()))
    try:
        yield key
    finally:
        _HARVEST_KEY.reset(token_key)
        _HARVEST_META.reset(token_meta)


def record_from_headers(
    headers: Any,
    *,
    http_status: Optional[int] = None,
    admission_key: Optional[str] = None,
    provider: str = "",
    model: str = "",
    now: Optional[float] = None,
) -> Optional[RateLimitRecord]:
    """Best-effort harvest. Never raises into the chat/provider hot path."""
    if not harvest_enabled():
        return None
    try:
        snapshot = parse_rate_limit_headers(
            headers, now=now, http_status=http_status
        )
        if snapshot is None:
            return None
        key = (admission_key or _HARVEST_KEY.get() or "").strip()
        if not key:
            return None
        meta_provider, meta_model = _HARVEST_META.get()
        return get_rate_limit_store().upsert(
            admission_key=key,
            provider=provider or meta_provider,
            model=model or meta_model,
            snapshot=snapshot,
            http_status=http_status,
            now=now,
        )
    except Exception:
        return None


def quota_admission_error(key: str, *, reset_at: Optional[float] = None) -> "Any":
    """Recoverable ProviderError so agentic failover / key rotation can run."""
    from puppetmaster.providers import ProviderError

    detail = f"provider quota exhausted for {key!r}"
    if reset_at is not None:
        detail += f" until {reset_at:.0f}"
    return ProviderError(
        detail,
        reason=RATE_LIMIT,
        status=429,
        body=QUOTA_EXHAUSTED_BODY,
    )


def is_quota_admission_error(error: Any) -> bool:
    """True when ``error`` was raised by harvested-quota admission."""
    return (
        getattr(error, "reason", None) == RATE_LIMIT
        and getattr(error, "status", None) == 429
        and (getattr(error, "body", None) or "") == QUOTA_EXHAUSTED_BODY
    )


def admit_or_raise(admission_key: str, *, now: Optional[float] = None) -> None:
    """Raise a recoverable rate-limit error when harvested remaining is zero."""
    if not admission_enabled():
        return
    key = (admission_key or "").strip()
    if not key:
        return
    try:
        record = get_rate_limit_store().get(key)
    except Exception:
        return
    if record is None:
        return
    clock = time.time() if now is None else float(now)
    reset = record.blocking_reset_at(now=clock)
    if reset is None:
        return
    raise quota_admission_error(key, reset_at=reset)


def doctor_rate_limit_summary(*, limit: int = 5) -> str:
    """One-line doctor detail: active exhausted windows, or idle OK."""
    if not harvest_enabled():
        return "off — PUPPETMASTER_RATE_LIMIT_HARVEST disabled"
    try:
        blocking = get_rate_limit_store().list_blocking(limit=limit)
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"
    if not blocking:
        return "ok — no exhausted harvested windows"
    parts = []
    for record in blocking[:limit]:
        label = record.provider or record.admission_key.split("\x1f", 1)[0] or "?"
        reset = record.blocking_reset_at()
        if record.model:
            label = f"{label}/{record.model}"
        if reset is not None:
            parts.append(f"{label} until {int(reset)}")
        else:
            parts.append(label)
    return "exhausted — " + "; ".join(parts)
