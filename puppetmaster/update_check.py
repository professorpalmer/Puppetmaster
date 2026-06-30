"""Opt-in PyPI update awareness for puppetmaster-ai.

The stdio MCP server cannot hot-reload or self-upgrade in-process. This module
only *detects* newer PyPI releases (when explicitly enabled) and produces
informational one-liners — never runs ``pip install``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any, Optional

from puppetmaster.mcp_registry import installed_puppetmaster_version

PYPI_DIST_NAME = "puppetmaster-ai"
PYPI_JSON_URL = "https://pypi.org/pypi/puppetmaster-ai/json"
PYPI_UPDATE_CHECK_ENV = "PUPPETMASTER_PYPI_UPDATE_CHECK"
PYPI_UPDATE_CHECK_TTL_SECONDS = 6 * 3600  # PyPI releases are infrequent
PYPI_FETCH_TIMEOUT_SECONDS = 3.0

_pypi_update_cache: dict[str, Any] = {"checked_at": 0.0, "note": None}
_pypi_update_lock = threading.Lock()


def reset_pypi_update_cache() -> None:
    """Drop the cached PyPI verdict (used by tests; harmless in prod)."""
    with _pypi_update_lock:
        _pypi_update_cache["checked_at"] = 0.0
        _pypi_update_cache["note"] = None


def _truthy_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def pypi_update_check_enabled() -> bool:
    """True when the opt-in PyPI update check env var is set."""
    return _truthy_env(os.environ.get(PYPI_UPDATE_CHECK_ENV))


def version_tuple(value: str) -> Optional[tuple[int, ...]]:
    """Leading-numeric dotted version as a tuple, or None if unparseable."""
    nums: list[int] = []
    for part in value.strip().split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return None
        nums.append(int(digits))
    return tuple(nums) or None


def version_is_newer(candidate: str, baseline: str) -> bool:
    """True when ``candidate`` sorts after ``baseline`` by numeric dotted parts."""
    parsed_candidate, parsed_baseline = version_tuple(candidate), version_tuple(baseline)
    if parsed_candidate is not None and parsed_baseline is not None:
        return parsed_candidate > parsed_baseline
    # Unparseable on either side: any difference is worth surfacing rather than
    # silently swallowing a real upgrade.
    return candidate != baseline


def _fetch_pypi_latest_version() -> Optional[str]:
    """Best-effort PyPI latest release; never raises."""
    try:
        request = urllib.request.Request(
            PYPI_JSON_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "puppetmaster-ai-update-check",
            },
        )
        with urllib.request.urlopen(request, timeout=PYPI_FETCH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        info = payload.get("info")
        if not isinstance(info, dict):
            return None
        version = info.get("version")
        if not isinstance(version, str) or not version.strip():
            return None
        return version.strip()
    except Exception:
        return None


def pypi_update_note(*, now: Optional[float] = None) -> Optional[str]:
    """One-line nudge when a newer puppetmaster-ai release exists on PyPI.

    Off by default; gated by :data:`PYPI_UPDATE_CHECK_ENV`. Cached with a long
    TTL so the MCP hot path never hits the network on every tool call. Any
    network or parse failure returns ``None`` silently.
    """
    if not pypi_update_check_enabled():
        return None

    moment = time.time() if now is None else now
    with _pypi_update_lock:
        age = moment - _pypi_update_cache["checked_at"]
        if age < PYPI_UPDATE_CHECK_TTL_SECONDS and _pypi_update_cache["checked_at"]:
            return _pypi_update_cache["note"]

    note: Optional[str] = None
    installed = installed_puppetmaster_version()
    latest = _fetch_pypi_latest_version()
    if installed and latest and version_is_newer(latest, installed):
        note = (
            f"puppetmaster-ai v{latest} is available (you have {installed}) — "
            "run `puppetmaster self-update` or `pip install -U puppetmaster-ai`."
        )

    with _pypi_update_lock:
        _pypi_update_cache["checked_at"] = moment
        _pypi_update_cache["note"] = note
    return note
