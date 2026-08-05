"""Pure parsers for OpenAI / Anthropic rate-limit response headers.

Lifted in spirit from codex-router's passive rate-limit discovery: most
OpenAI-compatible providers report remaining quota on every response via
``x-ratelimit-*`` headers; Anthropic uses ``anthropic-ratelimit-*``. Reading
them costs no extra request and needs no provider-specific balance endpoint.

This module never I/O's. Persistence and admission live in
:mod:`puppetmaster.rate_limit_state`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

_DURATION_PATTERN = re.compile(
    r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)m?s)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RateLimitWindow:
    """One requests- or tokens-window snapshot."""

    kind: str  # "requests" | "tokens"
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset_at: Optional[float] = None  # unix epoch seconds


@dataclass(frozen=True)
class RateLimitSnapshot:
    """Parsed windows from a single HTTP response."""

    requests: Optional[RateLimitWindow] = None
    tokens: Optional[RateLimitWindow] = None
    retry_after_at: Optional[float] = None  # from Retry-After, unix epoch

    def is_exhausted(self, *, now: Optional[float] = None) -> bool:
        """True when any known window reports remaining == 0 with a future reset."""
        return self.blocking_reset_at(now=now) is not None

    @property
    def exhausted(self) -> bool:
        """Wall-clock convenience for :meth:`is_exhausted`."""
        return self.is_exhausted()

    def blocking_reset_at(self, *, now: Optional[float] = None) -> Optional[float]:
        """Earliest future reset among exhausted windows, or ``None``."""
        clock = time.time() if now is None else float(now)
        candidates: list[float] = []
        for window in (self.tokens, self.requests):
            if window is None:
                continue
            if (
                window.remaining == 0
                and window.reset_at is not None
                and window.reset_at > clock
            ):
                candidates.append(float(window.reset_at))
        if self.retry_after_at is not None and self.retry_after_at > clock:
            candidates.append(float(self.retry_after_at))
        return min(candidates) if candidates else None


def normalize_headers(headers: Any) -> dict[str, str]:
    """Lower-case header map from a urllib ``HTTPMessage``, mapping, or None."""
    if headers is None:
        return {}
    items = None
    if hasattr(headers, "items"):
        try:
            items = list(headers.items())
        except Exception:
            items = None
    if items is None and isinstance(headers, Mapping):
        items = list(headers.items())
    if not items:
        return {}
    out: dict[str, str] = {}
    for key, value in items:
        if key is None or value is None:
            continue
        out[str(key).strip().lower()] = str(value).strip()
    return out


def _finite_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _count(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or number < 0:
        return None
    return int(round(number))


def reset_at(value: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Normalize a reset header to unix epoch seconds.

    Accepts Go-style durations (``2m59s``), bare seconds, epoch seconds/ms, or
    absolute timestamps.
    """
    clock = time.time() if now is None else float(now)
    text = str(value or "").strip()
    if not text:
        return None

    bare = _finite_number(text)
    if bare is not None and re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        # Large enough to be an epoch: ms if >= 1e12, else seconds.
        if bare >= 1_000_000_000_000:
            return bare / 1000.0
        if bare >= 1_000_000_000:
            return bare
        if bare >= 0:
            return clock + bare
        return None

    match = _DURATION_PATTERN.match(text)
    if match and any(match.groups()):
        hours = _finite_number(match.group(1)) or 0.0
        minutes = _finite_number(match.group(2)) or 0.0
        seconds = _finite_number(match.group(3)) or 0.0
        if text.lower().endswith("ms"):
            delta = hours * 3600.0 + minutes * 60.0 + (seconds / 1000.0)
        else:
            delta = hours * 3600.0 + minutes * 60.0 + seconds
        return clock + delta

    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.timestamp()
    except Exception:
        pass
    return None


def _read(headers: Mapping[str, str], keys: list[str]) -> Optional[str]:
    for key in keys:
        value = headers.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _window(
    headers: Mapping[str, str],
    *,
    kind: str,
    limit_keys: list[str],
    remaining_keys: list[str],
    reset_keys: list[str],
    now: float,
) -> Optional[RateLimitWindow]:
    limit = _count(_read(headers, limit_keys))
    remaining = _count(_read(headers, remaining_keys))
    reset = reset_at(_read(headers, reset_keys), now=now)
    if limit is None and remaining is None and reset is None:
        return None
    return RateLimitWindow(kind=kind, limit=limit, remaining=remaining, reset_at=reset)


def parse_rate_limit_headers(
    headers: Any,
    *,
    now: Optional[float] = None,
    http_status: Optional[int] = None,
) -> Optional[RateLimitSnapshot]:
    """Parse rate-limit windows from response headers.

    Returns ``None`` when no recognized rate-limit signal is present. On HTTP
    429, also consults ``Retry-After`` so admission can wait even when the
    provider omits ``x-ratelimit-*``.
    """
    clock = time.time() if now is None else float(now)
    normalized = normalize_headers(headers)
    if not normalized and http_status != 429:
        return None

    requests = _window(
        normalized,
        kind="requests",
        limit_keys=[
            "x-ratelimit-limit-requests",
            "anthropic-ratelimit-requests-limit",
            "x-ratelimit-limit",
        ],
        remaining_keys=[
            "x-ratelimit-remaining-requests",
            "anthropic-ratelimit-requests-remaining",
            "x-ratelimit-remaining",
        ],
        reset_keys=[
            "x-ratelimit-reset-requests",
            "anthropic-ratelimit-requests-reset",
            "x-ratelimit-reset",
        ],
        now=clock,
    )
    tokens = _window(
        normalized,
        kind="tokens",
        limit_keys=[
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-input-tokens-limit",
        ],
        remaining_keys=[
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-input-tokens-remaining",
        ],
        reset_keys=[
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-tokens-reset",
            "anthropic-ratelimit-input-tokens-reset",
        ],
        now=clock,
    )

    retry_after_at = None
    if http_status == 429 or _read(normalized, ["retry-after"]) is not None:
        retry_after_at = reset_at(_read(normalized, ["retry-after"]), now=clock)
        # Bare Retry-After with no other signal still counts on 429.
        if http_status == 429 and retry_after_at is None:
            retry_after_at = clock + 30.0

    if requests is None and tokens is None and retry_after_at is None:
        return None
    return RateLimitSnapshot(
        requests=requests,
        tokens=tokens,
        retry_after_at=retry_after_at,
    )
