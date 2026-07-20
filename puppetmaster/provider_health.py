"""User-global invoke-health store for provider auto-routing gates.

Separates credential *presence* from *verified* runtime availability. Bedrock
in particular must not enter ``available_providers`` / router candidates merely
because ``~/.aws/default`` or env keys exist — only a current verified
Converse/runtime invoke (or an explicit probe) makes it auto-routable. A
terminal auth denial is sticky across processes for the same non-secret
credential fingerprint until TTL expiry or the fingerprint changes.

State lives under ``~/.puppetmaster`` (stdlib SQLite WAL), never inside a
project tree, and never stores raw secrets — only hashed fingerprints.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from puppetmaster.fs_permissions import chmod_private_file, mkdir_private

STATUS_VERIFIED = "verified"
STATUS_DENIED = "denied"
STATUS_UNKNOWN = "unknown"

# Bounded TTLs: verified stays routable for a workday slice; denied stays out
# long enough that every worker process sees the poison without forever-locking
# a credential that the user later repairs under the same fingerprint.
DEFAULT_VERIFIED_TTL_SECONDS = 6 * 60 * 60
DEFAULT_DENIED_TTL_SECONDS = 24 * 60 * 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS invoke_health (
    provider TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    region TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (provider, fingerprint, region)
);
"""

_STORE_LOCK = threading.RLock()
_STORE_CACHE: dict[str, "ProviderHealthStore"] = {}


def _ttl_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(1, value)


def verified_ttl_seconds() -> int:
    return _ttl_from_env(
        "PUPPETMASTER_PROVIDER_HEALTH_VERIFIED_TTL",
        DEFAULT_VERIFIED_TTL_SECONDS,
    )


def denied_ttl_seconds() -> int:
    return _ttl_from_env(
        "PUPPETMASTER_PROVIDER_HEALTH_DENIED_TTL",
        DEFAULT_DENIED_TTL_SECONDS,
    )


def default_provider_health_path() -> Path:
    """SQLite path for the user-global invoke-health store."""
    override = (os.environ.get("PUPPETMASTER_PROVIDER_HEALTH_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    home = (os.environ.get("PUPPETMASTER_HOME") or "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "provider_health.sqlite3"


@dataclass(frozen=True)
class InvokeHealthRecord:
    """One durable invoke-health row (no credential material)."""

    provider: str
    fingerprint: str
    region: str
    status: str
    updated_at: float
    expires_at: float
    detail: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= float(self.expires_at)

    @property
    def effective_status(self) -> str:
        if self.expired:
            return STATUS_UNKNOWN
        if self.status in (STATUS_VERIFIED, STATUS_DENIED):
            return self.status
        return STATUS_UNKNOWN


class ProviderHealthStore:
    """Small WAL SQLite store with atomic upserts for concurrent workers."""

    def __init__(self, path: Optional[Path] = None, *, busy_timeout_ms: int = 5000):
        self.path = Path(path) if path is not None else default_provider_health_path()
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
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
                connection.commit()
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

    @contextmanager
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

    def get(
        self,
        provider: str,
        fingerprint: str,
        region: str,
    ) -> Optional[InvokeHealthRecord]:
        provider = (provider or "").strip().lower()
        fingerprint = (fingerprint or "").strip()
        region = (region or "").strip() or "us-east-1"
        if not provider or not fingerprint:
            return None
        with self._session() as connection:
            row = connection.execute(
                "SELECT provider, fingerprint, region, status, updated_at, "
                "expires_at, detail FROM invoke_health "
                "WHERE provider = ? AND fingerprint = ? AND region = ?",
                (provider, fingerprint, region),
            ).fetchone()
        if row is None:
            return None
        return InvokeHealthRecord(
            provider=str(row["provider"]),
            fingerprint=str(row["fingerprint"]),
            region=str(row["region"]),
            status=str(row["status"]),
            updated_at=float(row["updated_at"]),
            expires_at=float(row["expires_at"]),
            detail=str(row["detail"] or ""),
        )

    def upsert(
        self,
        *,
        provider: str,
        fingerprint: str,
        region: str,
        status: str,
        ttl_seconds: int,
        detail: str = "",
        now: Optional[float] = None,
    ) -> InvokeHealthRecord:
        if status not in (STATUS_VERIFIED, STATUS_DENIED):
            raise ValueError(f"unsupported invoke-health status: {status!r}")
        provider = (provider or "").strip().lower()
        fingerprint = (fingerprint or "").strip()
        region = (region or "").strip() or "us-east-1"
        if not provider or not fingerprint:
            raise ValueError("provider and fingerprint are required")
        stamp = float(time.time() if now is None else now)
        expires = stamp + max(1, int(ttl_seconds))
        detail_text = str(detail or "")[:500]
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO invoke_health "
                "(provider, fingerprint, region, status, updated_at, expires_at, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(provider, fingerprint, region) DO UPDATE SET "
                "status = excluded.status, "
                "updated_at = excluded.updated_at, "
                "expires_at = excluded.expires_at, "
                "detail = excluded.detail",
                (
                    provider,
                    fingerprint,
                    region,
                    status,
                    stamp,
                    expires,
                    detail_text,
                ),
            )
        return InvokeHealthRecord(
            provider=provider,
            fingerprint=fingerprint,
            region=region,
            status=status,
            updated_at=stamp,
            expires_at=expires,
            detail=detail_text,
        )

    def mark_verified(
        self,
        *,
        provider: str,
        fingerprint: str,
        region: str,
        detail: str = "runtime_invoke_ok",
        now: Optional[float] = None,
    ) -> InvokeHealthRecord:
        return self.upsert(
            provider=provider,
            fingerprint=fingerprint,
            region=region,
            status=STATUS_VERIFIED,
            ttl_seconds=verified_ttl_seconds(),
            detail=detail,
            now=now,
        )

    def mark_denied(
        self,
        *,
        provider: str,
        fingerprint: str,
        region: str,
        detail: str = "terminal_auth_denied",
        now: Optional[float] = None,
    ) -> InvokeHealthRecord:
        return self.upsert(
            provider=provider,
            fingerprint=fingerprint,
            region=region,
            status=STATUS_DENIED,
            ttl_seconds=denied_ttl_seconds(),
            detail=detail,
            now=now,
        )

    def dump_rows(self) -> list[dict[str, Any]]:
        """Return all rows as plain dicts (tests / doctor; no secrets)."""
        with self._session() as connection:
            rows = connection.execute(
                "SELECT provider, fingerprint, region, status, updated_at, "
                "expires_at, detail FROM invoke_health ORDER BY provider, region"
            ).fetchall()
        return [dict(row) for row in rows]


def get_provider_health_store(path: Optional[Path] = None) -> ProviderHealthStore:
    """Process-cached store handle keyed by resolved path."""
    resolved = str((path if path is not None else default_provider_health_path()).resolve())
    with _STORE_LOCK:
        store = _STORE_CACHE.get(resolved)
        if store is None:
            store = ProviderHealthStore(Path(resolved))
            _STORE_CACHE[resolved] = store
        return store


def reset_provider_health_store_cache() -> None:
    """Drop cached store handles (tests / hermetic isolation)."""
    with _STORE_LOCK:
        _STORE_CACHE.clear()


def fingerprint_bedrock_credentials(creds: Any) -> str:
    """Non-secret fingerprint for a resolved :class:`BedrockCredentials`.

    Hashes identity material only (kind + access-key id / profile / bearer
    digest). Never persists the bearer token, secret key, or session token.
    """
    kind = str(getattr(creds, "kind", "") or "").strip().lower() or "unknown"
    parts = [f"kind:{kind}"]
    access_key_id = getattr(creds, "access_key_id", None)
    if access_key_id:
        # Hash the id so the DB never holds even the non-secret AKIA… string.
        parts.append(
            "akid:"
            + hashlib.sha256(str(access_key_id).encode("utf-8")).hexdigest()
        )
    profile = getattr(creds, "profile", None)
    if profile:
        parts.append(f"profile:{str(profile).strip()}")
    bearer = getattr(creds, "bearer_token", None)
    if bearer:
        parts.append(
            "bearer:"
            + hashlib.sha256(str(bearer).encode("utf-8")).hexdigest()
        )
    if len(parts) == 1:
        parts.append("empty")
    material = "|".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _auth_signal_text(error: Any) -> str:
    chunks = [
        str(getattr(error, "reason", "") or ""),
        str(getattr(error, "failure", "") or ""),
        str(getattr(error, "body", "") or ""),
        str(error or ""),
    ]
    return " ".join(chunks).lower()


def is_terminal_bedrock_auth_failure(error: Any) -> bool:
    """True for Bedrock runtime auth/authorization failures that should deny.

    401 and canonical ``not_authenticated`` always deny. 403 / ``forbidden``
    deny only when the existing auth-signal heuristics agree this is a provider
    authentication/authorization failure (not an ordinary application 403).
    Transient 5xx / timeout / network never deny.
    """
    from puppetmaster.failure import FORBIDDEN, NOT_AUTHENTICATED

    status = getattr(error, "status", None)
    failure = str(getattr(error, "failure", "") or "")
    reason = str(getattr(error, "reason", "") or "")

    if status == 401 or failure == NOT_AUTHENTICATED or reason == "not_authenticated":
        return True
    if reason in ("http_status:401",):
        return True

    if status == 403 or failure == FORBIDDEN or reason in ("http_status:403",):
        text = _auth_signal_text(error)
        markers = (
            "unauthorized",
            "not authorized",
            "accessdenied",
            "access denied",
            "unrecognizedclient",
            "invalidsignature",
            "expiredtoken",
            "security token",
            "invalid identity",
            "incomplete signature",
            "authentication",
            "authorization",
            "not authenticated",
            "invalidclienttokenid",
            "signatureexpired",
            "request expired",
        )
        return any(marker in text for marker in markers)
    return False


def bedrock_failure_for_recovery(error: Any) -> str:
    """Canonical failure class for orchestrator auto-fallback after Bedrock auth.

    Maps terminal auth ``forbidden`` onto ``not_authenticated`` so recovery can
    leave the denied Bedrock fingerprint without treating arbitrary non-Bedrock
    application 403s as globally retryable inside the provider client.
    """
    from puppetmaster.failure import FORBIDDEN, NOT_AUTHENTICATED

    failure = str(getattr(error, "failure", "") or "") or "unknown"
    if not is_terminal_bedrock_auth_failure(error):
        return failure
    # Auth 403 and bare forbidden become not_authenticated for recovery.
    if failure == FORBIDDEN or getattr(error, "status", None) == 403:
        return NOT_AUTHENTICATED
    if failure in ("", "unknown"):
        return NOT_AUTHENTICATED
    return failure


def read_bedrock_invoke_health(
    env: Optional[Mapping[str, str]] = None,
    *,
    creds: Any = None,
    store: Optional[ProviderHealthStore] = None,
    home: Optional[Path] = None,
) -> str:
    """Effective invoke-health status: ``verified`` / ``denied`` / ``unknown``."""
    from puppetmaster.bedrock import (
        resolve_bedrock_credentials,
        resolve_bedrock_region,
    )

    env = env if env is not None else os.environ
    if creds is None:
        creds = resolve_bedrock_credentials(env, home=home)
    if creds is None:
        return STATUS_UNKNOWN
    region = resolve_bedrock_region(env)
    fingerprint = fingerprint_bedrock_credentials(creds)
    health_store = store or get_provider_health_store()
    record = health_store.get("bedrock", fingerprint, region)
    if record is None:
        return STATUS_UNKNOWN
    return record.effective_status


def is_bedrock_auto_routable(
    env: Optional[Mapping[str, str]] = None,
    *,
    creds: Any = None,
    store: Optional[ProviderHealthStore] = None,
    home: Optional[Path] = None,
) -> bool:
    """True only when credentials exist and invoke health is currently verified."""
    return (
        read_bedrock_invoke_health(env, creds=creds, store=store, home=home)
        == STATUS_VERIFIED
    )


def record_bedrock_invoke_success(
    env: Optional[Mapping[str, str]] = None,
    *,
    creds: Any,
    store: Optional[ProviderHealthStore] = None,
    detail: str = "runtime_invoke_ok",
) -> Optional[InvokeHealthRecord]:
    from puppetmaster.bedrock import resolve_bedrock_region

    env = env if env is not None else os.environ
    if creds is None:
        return None
    health_store = store or get_provider_health_store()
    return health_store.mark_verified(
        provider="bedrock",
        fingerprint=fingerprint_bedrock_credentials(creds),
        region=resolve_bedrock_region(env),
        detail=detail,
    )


def record_bedrock_invoke_failure(
    env: Optional[Mapping[str, str]] = None,
    *,
    creds: Any,
    error: Any,
    store: Optional[ProviderHealthStore] = None,
) -> Optional[InvokeHealthRecord]:
    """Persist denied health for terminal Bedrock auth failures only."""
    from puppetmaster.bedrock import resolve_bedrock_region

    env = env if env is not None else os.environ
    if creds is None or not is_terminal_bedrock_auth_failure(error):
        return None
    status = getattr(error, "status", None)
    failure = getattr(error, "failure", None)
    detail = f"terminal_auth_denied:{status or failure or 'auth'}"
    health_store = store or get_provider_health_store()
    return health_store.mark_denied(
        provider="bedrock",
        fingerprint=fingerprint_bedrock_credentials(creds),
        region=resolve_bedrock_region(env),
        detail=detail,
    )


def bedrock_health_report(
    env: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[Path] = None,
    store: Optional[ProviderHealthStore] = None,
) -> dict[str, Any]:
    """Doctor/billing-facing Bedrock posture (never exposes secret values)."""
    from puppetmaster.bedrock import (
        resolve_bedrock_credentials,
        resolve_bedrock_region,
    )

    env = env if env is not None else os.environ
    creds = resolve_bedrock_credentials(env, home=home)
    region = resolve_bedrock_region(env)
    if creds is None:
        return {
            "credentials_present": False,
            "invoke_health": STATUS_UNKNOWN,
            "auto_routable": False,
            "region": region,
            "detail": "no AWS Bedrock credentials visible",
        }
    status = read_bedrock_invoke_health(
        env, creds=creds, store=store, home=home
    )
    if status == STATUS_VERIFIED:
        detail = "Bedrock credentials present and runtime-verified"
    elif status == STATUS_DENIED:
        detail = (
            "Bedrock credentials present but terminal auth denied "
            "(not auto-routable until repaired or fingerprint changes)"
        )
    else:
        detail = (
            "Bedrock credentials present but unverified "
            "(not auto-routable until a successful runtime invoke or "
            "`models discover --source agentic --probe`)"
        )
    evidence = list(getattr(creds, "evidence", ()) or ())
    return {
        "credentials_present": True,
        "invoke_health": status,
        "auto_routable": status == STATUS_VERIFIED,
        "region": region,
        "credential_kind": getattr(creds, "kind", None),
        "detail": detail,
        "evidence": evidence,
    }


def probe_bedrock_runtime(
    *,
    env: Optional[Mapping[str, str]] = None,
    model_id: Optional[str] = None,
    timeout: int = 20,
    store: Optional[ProviderHealthStore] = None,
) -> dict[str, Any]:
    """Bounded off-hot-path Converse probe; updates invoke health.

    Uses a tiny prompt/output budget. Never called from ``route_task``.
    """
    from puppetmaster.bedrock import (
        bedrock_chat,
        diversify_chat_model_ids,
        list_chat_model_ids,
        resolve_bedrock_credentials,
        resolve_bedrock_region,
    )

    env = env if env is not None else os.environ
    creds = resolve_bedrock_credentials(env)
    region = resolve_bedrock_region(env)
    report: dict[str, Any] = {
        "probed": True,
        "ok": False,
        "region": region,
        "model_id": None,
        "invoke_health": STATUS_UNKNOWN,
        "credentials_present": creds is not None,
    }
    if creds is None:
        report["error"] = "no_credentials"
        return report

    candidate = (model_id or "").strip()
    if not candidate:
        try:
            ids = list_chat_model_ids(env=env, timeout=min(timeout, 30))
        except Exception as exc:
            report["error"] = f"catalog:{exc!r}"
            report["invoke_health"] = read_bedrock_invoke_health(
                env, creds=creds, store=store
            )
            return report
        diversified = diversify_chat_model_ids(ids)
        # Prefer tiny / cheap ids for the probe when present.
        preferred = [
            mid
            for mid in diversified
            if any(
                token in mid.lower()
                for token in ("micro", "lite", "haiku", "nano", "flash")
            )
        ]
        candidate = (preferred or diversified or ids or [None])[0] or ""
    if not candidate:
        report["error"] = "no_candidate_model"
        report["invoke_health"] = read_bedrock_invoke_health(
            env, creds=creds, store=store
        )
        return report

    report["model_id"] = candidate
    try:
        bedrock_chat(
            model=candidate,
            messages=[{"role": "user", "content": "ping"}],
            extra={"max_tokens": 1},
            timeout=timeout,
            env=env,
        )
        report["ok"] = True
        report["invoke_health"] = STATUS_VERIFIED
        # bedrock_chat records verified health on success.
        return report
    except Exception as exc:
        report["error"] = repr(exc)
        report["invoke_health"] = read_bedrock_invoke_health(
            env, creds=creds, store=store
        )
        report["ok"] = False
        return report


def assert_no_secrets_in_health_state(store: Optional[ProviderHealthStore] = None) -> None:
    """Raise ``AssertionError`` if any persisted row looks like secret material."""
    import re

    health_store = store or get_provider_health_store()
    secretish = (
        "aws_secret",
        "secret_access",
        "session_token",
        "bearer_token",
        "aws_access_key",
    )
    # Raw IAM access-key ids look like AKIA… / ASIA… (not hex fingerprints).
    access_key_re = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
    for row in health_store.dump_rows():
        blob = " ".join(str(v) for v in row.values())
        lower = blob.lower()
        for marker in secretish:
            if marker in lower:
                raise AssertionError(f"secret-like material in health state: {marker}")
        if access_key_re.search(blob):
            raise AssertionError("raw access key id persisted in health state")
