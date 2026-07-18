"""Process-local provider circuit-breaker admission (reliability Slice 9).

Gates agentic provider calls after a short streak of consecutive *retryable*
failures (rate-limit / server / timeout / network). Closed by default; opens for
a cooldown; then allows exactly one half-open probe. Auth, malformed,
cancellation, and other non-retryable errors never trip the breaker.

Thread-safe, stdlib-only, Python 3.9+. Kill with ``PUPPETMASTER_PROVIDER_CIRCUIT=0``.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from puppetmaster.failure import RATE_LIMIT
from puppetmaster.providers import ProviderError, get_provider, is_retryable_provider_error, resolve_base_url

# Env kill switch + knobs (safe / clamped defaults).
_ENV_ENABLED = "PUPPETMASTER_PROVIDER_CIRCUIT"
_ENV_THRESHOLD = "PUPPETMASTER_PROVIDER_CIRCUIT_THRESHOLD"
_ENV_COOLDOWN = "PUPPETMASTER_PROVIDER_CIRCUIT_COOLDOWN_SECONDS"

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0
# Bound process-local breaker state: drop idle closed buckets, then hard-cap.
DEFAULT_MAX_BUCKETS = 256
DEFAULT_CLOSED_BUCKET_TTL_SECONDS = 300.0
_ABS_MIN_THRESHOLD = 1
_ABS_MAX_THRESHOLD = 100
_ABS_MIN_COOLDOWN = 1.0
_ABS_MAX_COOLDOWN = 600.0
_ABS_MIN_MAX_BUCKETS = 1
_ABS_MAX_MAX_BUCKETS = 10_000
_ABS_MIN_CLOSED_TTL = 1.0
_ABS_MAX_CLOSED_TTL = 86_400.0

_ENV_MAX_BUCKETS = "PUPPETMASTER_PROVIDER_CIRCUIT_MAX_BUCKETS"
_ENV_CLOSED_TTL = "PUPPETMASTER_PROVIDER_CIRCUIT_CLOSED_TTL_SECONDS"

_STATE_CLOSED = "closed"
_STATE_OPEN = "open"
_STATE_HALF_OPEN = "half_open"

Clock = Callable[[], float]


def circuit_enabled() -> bool:
    """False when ``PUPPETMASTER_PROVIDER_CIRCUIT`` is 0/false/off/no."""
    val = os.environ.get(_ENV_ENABLED, "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def failure_threshold() -> int:
    """Consecutive qualifying failures before opening; clamped to ``[1, 100]``."""
    raw = os.environ.get(_ENV_THRESHOLD, "").strip()
    if not raw:
        return DEFAULT_FAILURE_THRESHOLD
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_FAILURE_THRESHOLD
    return max(_ABS_MIN_THRESHOLD, min(value, _ABS_MAX_THRESHOLD))


def cooldown_seconds() -> float:
    """Open-state cooldown in seconds; clamped to ``[1, 600]``."""
    raw = os.environ.get(_ENV_COOLDOWN, "").strip()
    if not raw:
        return DEFAULT_COOLDOWN_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_COOLDOWN_SECONDS
    return max(_ABS_MIN_COOLDOWN, min(value, _ABS_MAX_COOLDOWN))


def max_buckets() -> int:
    """Hard cap on retained breaker buckets; clamped to ``[1, 10000]``."""
    raw = os.environ.get(_ENV_MAX_BUCKETS, "").strip()
    if not raw:
        return DEFAULT_MAX_BUCKETS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BUCKETS
    return max(_ABS_MIN_MAX_BUCKETS, min(value, _ABS_MAX_MAX_BUCKETS))


def closed_bucket_ttl_seconds() -> float:
    """Idle TTL for closed buckets before eviction; clamped to ``[1, 86400]``."""
    raw = os.environ.get(_ENV_CLOSED_TTL, "").strip()
    if not raw:
        return DEFAULT_CLOSED_BUCKET_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CLOSED_BUCKET_TTL_SECONDS
    return max(_ABS_MIN_CLOSED_TTL, min(value, _ABS_MAX_CLOSED_TTL))


def circuit_key(provider: str, model: str = "", *, base_url: str = "") -> str:
    """Stable admission key: provider + model + base URL (when known)."""
    return f"{(provider or '').strip().lower()}\x1f{(model or '').strip()}\x1f{(base_url or '').rstrip('/')}"


def resolve_circuit_key(provider: str, model: str = "") -> str:
    """Build a key, resolving the provider's effective base URL when registered."""
    base_url = ""
    desc = get_provider(provider)
    if desc is not None:
        try:
            base_url = resolve_base_url(desc)
        except Exception:
            base_url = (desc.base_url or "").rstrip("/")
    return circuit_key(provider, model, base_url=base_url)


def admission_blocked_error(key: str) -> ProviderError:
    """Canonical recoverable failure so existing failover / retry policy can run."""
    return ProviderError(
        f"provider circuit open for {key!r}",
        reason=RATE_LIMIT,
        status=429,
        body="circuit_breaker_open",
    )


def is_admission_blocked_error(error: ProviderError) -> bool:
    """True when ``error`` was raised by circuit admission (not a live provider 429)."""
    return (
        error.reason == RATE_LIMIT
        and error.status == 429
        and (error.body or "") == "circuit_breaker_open"
    )


@dataclass
class _Bucket:
    state: str = _STATE_CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    probe_in_flight: bool = False
    last_used: float = 0.0


class ProviderCircuitBreaker:
    """Process-local breaker keyed by :func:`circuit_key` values."""

    def __init__(
        self,
        *,
        clock: Optional[Clock] = None,
        threshold: Optional[int] = None,
        cooldown: Optional[float] = None,
        enabled: Optional[bool] = None,
        max_buckets: Optional[int] = None,
        closed_ttl: Optional[float] = None,
    ) -> None:
        self._clock: Clock = clock or time.monotonic
        self._threshold_override = threshold
        self._cooldown_override = cooldown
        self._enabled_override = enabled
        self._max_buckets_override = max_buckets
        self._closed_ttl_override = closed_ttl
        self._lock = threading.RLock()
        self._buckets: dict[str, _Bucket] = {}

    def _is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return circuit_enabled()

    def _threshold(self) -> int:
        if self._threshold_override is not None:
            return max(_ABS_MIN_THRESHOLD, min(int(self._threshold_override), _ABS_MAX_THRESHOLD))
        return failure_threshold()

    def _cooldown(self) -> float:
        if self._cooldown_override is not None:
            return max(_ABS_MIN_COOLDOWN, min(float(self._cooldown_override), _ABS_MAX_COOLDOWN))
        return cooldown_seconds()

    def _max_buckets(self) -> int:
        if self._max_buckets_override is not None:
            return max(
                _ABS_MIN_MAX_BUCKETS,
                min(int(self._max_buckets_override), _ABS_MAX_MAX_BUCKETS),
            )
        return max_buckets()

    def _closed_ttl(self) -> float:
        if self._closed_ttl_override is not None:
            return max(
                _ABS_MIN_CLOSED_TTL,
                min(float(self._closed_ttl_override), _ABS_MAX_CLOSED_TTL),
            )
        return closed_bucket_ttl_seconds()

    def _touch(self, bucket: _Bucket) -> None:
        bucket.last_used = self._clock()

    def _evict_stale_closed(self, *, now: float, protect_key: Optional[str] = None) -> None:
        """Drop closed buckets idle longer than the configured TTL."""
        ttl = self._closed_ttl()
        stale = [
            key
            for key, bucket in self._buckets.items()
            if key != protect_key
            and bucket.state == _STATE_CLOSED
            and not bucket.probe_in_flight
            and (now - bucket.last_used) >= ttl
        ]
        for key in stale:
            del self._buckets[key]

    def _evict_to_cap(
        self,
        *,
        protect_key: Optional[str] = None,
        target: Optional[int] = None,
    ) -> None:
        """Shrink the bucket map to ``target`` (default: hard cap).

        Prefers evicting the oldest closed (idle) buckets. Only if the map is
        still over capacity after that (all remaining keys are hot open /
        half-open) does it drop the oldest non-protected entries.
        """
        cap = self._max_buckets() if target is None else max(0, int(target))
        if len(self._buckets) <= cap:
            return

        def _oldest_closed() -> Optional[str]:
            oldest_key = None
            oldest_used = None
            for key, bucket in self._buckets.items():
                if key == protect_key:
                    continue
                if bucket.state != _STATE_CLOSED or bucket.probe_in_flight:
                    continue
                if oldest_used is None or bucket.last_used < oldest_used:
                    oldest_key = key
                    oldest_used = bucket.last_used
            return oldest_key

        while len(self._buckets) > cap:
            victim = _oldest_closed()
            if victim is None:
                break
            del self._buckets[victim]

        if len(self._buckets) <= cap:
            return

        # Last resort: every remaining bucket is open/half-open. Drop oldest.
        while len(self._buckets) > cap:
            oldest_key = None
            oldest_used = None
            for key, bucket in self._buckets.items():
                if key == protect_key:
                    continue
                if oldest_used is None or bucket.last_used < oldest_used:
                    oldest_key = key
                    oldest_used = bucket.last_used
            if oldest_key is None:
                break
            del self._buckets[oldest_key]

    def _bucket(self, key: str) -> _Bucket:
        now = self._clock()
        self._evict_stale_closed(now=now, protect_key=key)
        bucket = self._buckets.get(key)
        if bucket is None:
            # Free a slot before inserting so the map never exceeds the hard cap.
            if len(self._buckets) >= self._max_buckets():
                self._evict_to_cap(
                    protect_key=key,
                    target=self._max_buckets() - 1,
                )
            bucket = _Bucket(last_used=now)
            self._buckets[key] = bucket
            self._evict_to_cap(protect_key=key)
        else:
            self._touch(bucket)
            self._evict_to_cap(protect_key=key)
        return bucket

    def bucket_count(self) -> int:
        """Number of retained breaker buckets (tests / diagnostics)."""
        with self._lock:
            return len(self._buckets)

    def state_for(self, key: str) -> str:
        """Current state name for ``key`` (``closed`` when unseen)."""
        with self._lock:
            # Avoid creating a bucket solely for an unseen probe: report closed
            # without retaining idle state that would need later eviction.
            bucket = self._buckets.get(key)
            if bucket is None:
                return _STATE_CLOSED
            self._touch(bucket)
            return bucket.state

    def before_call(self, key: str) -> None:
        """Admit a call or raise a recoverable :class:`ProviderError`."""
        if not self._is_enabled():
            return
        with self._lock:
            bucket = self._bucket(key)
            if bucket.state == _STATE_CLOSED:
                return
            now = self._clock()
            if bucket.state == _STATE_OPEN:
                if (now - bucket.opened_at) < self._cooldown():
                    raise admission_blocked_error(key)
                bucket.state = _STATE_HALF_OPEN
                bucket.probe_in_flight = True
                return
            # half-open: exactly one probe
            if bucket.probe_in_flight:
                raise admission_blocked_error(key)
            bucket.probe_in_flight = True

    def record_success(self, key: str) -> None:
        """Close the breaker after a successful provider call."""
        if not self._is_enabled():
            return
        with self._lock:
            bucket = self._bucket(key)
            bucket.state = _STATE_CLOSED
            bucket.consecutive_failures = 0
            bucket.probe_in_flight = False
            bucket.opened_at = 0.0

    def record_failure(self, key: str, error: ProviderError) -> None:
        """Count a qualifying retryable failure; ignore non-qualifying errors."""
        if not self._is_enabled():
            return
        if is_admission_blocked_error(error):
            return
        if not is_retryable_provider_error(error):
            with self._lock:
                bucket = self._bucket(key)
                # Non-qualifying response from a half-open probe: endpoint is
                # reachable (auth/malformed/etc.) — close without tripping.
                if bucket.state == _STATE_HALF_OPEN:
                    bucket.state = _STATE_CLOSED
                    bucket.consecutive_failures = 0
                    bucket.probe_in_flight = False
                    bucket.opened_at = 0.0
            return
        with self._lock:
            bucket = self._bucket(key)
            bucket.consecutive_failures += 1
            bucket.probe_in_flight = False
            if (
                bucket.state == _STATE_HALF_OPEN
                or bucket.consecutive_failures >= self._threshold()
            ):
                bucket.state = _STATE_OPEN
                bucket.opened_at = self._clock()

    def release_admission(self, key: str) -> None:
        """Drop a half-open probe reservation without counting success/failure.

        Used when a call is abandoned (cancellation, unexpected exit) so the
        breaker cannot stick half-open forever.
        """
        if not self._is_enabled():
            return
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or bucket.state != _STATE_HALF_OPEN:
                return
            bucket.probe_in_flight = False
            # Return to open so the next caller waits out the remaining cooldown
            # (or immediately re-probes if the cooldown already elapsed).
            bucket.state = _STATE_OPEN
            self._touch(bucket)

    def reset(self) -> None:
        """Drop all bucket state (tests / process recycle helpers)."""
        with self._lock:
            self._buckets.clear()


_BREAKER_LOCK = threading.Lock()
_BREAKER: Optional[ProviderCircuitBreaker] = None


def get_provider_circuit_breaker() -> ProviderCircuitBreaker:
    """Process-wide singleton used by agentic admission."""
    global _BREAKER
    with _BREAKER_LOCK:
        if _BREAKER is None:
            _BREAKER = ProviderCircuitBreaker()
        return _BREAKER


def reset_provider_circuit_breaker(
    breaker: Optional[ProviderCircuitBreaker] = None,
) -> ProviderCircuitBreaker:
    """Replace the singleton (or clear it). Returns the active instance."""
    global _BREAKER
    with _BREAKER_LOCK:
        _BREAKER = breaker if breaker is not None else ProviderCircuitBreaker()
        return _BREAKER
