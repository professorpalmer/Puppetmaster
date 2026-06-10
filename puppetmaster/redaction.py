"""Shared secret-redaction helpers.

Lives in its own dependency-free module so both the MCP server (scrubbing
child-process output before returning it over the wire) and the worker
adapters (scrubbing stdout/stderr sidecars and PATCH diffs before they are
persisted to artifacts/state) can call the *same* implementation. Keeping a
single source of truth avoids the prior split where MCP responses were
redacted but adapter-persisted transcripts and diffs were not.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Optional

_REDACTED_PLACEHOLDER = "<redacted>"
_SECRET_KEY_SUFFIXES = ("_api_key", "_token", "_secret")
_EXPLICIT_SECRET_KEYS = frozenset(
    {"openai_api_key", "anthropic_api_key", "cursor_api_key"}
)

# Live values of these env vars are replaced wherever they appear verbatim.
_SECRET_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CURSOR_API_KEY",
    "OPENAI_ORG_ID",
)
_SECRET_SK = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_SECRET_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")
_SECRET_APIKEY = re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)[A-Za-z0-9._\-]{8,}")
_SECRET_GITHUB = re.compile(r"ghp_[A-Za-z0-9]{20,}")
_SECRET_AWS = re.compile(r"AKIA[0-9A-Z]{16}")
_SECRET_SLACK = re.compile(r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{20,}")
_SECRET_JWT = re.compile(
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
)

_registered_values: set[str] = set()
_registered_lock = threading.Lock()


def register_secret_value(value: Optional[str]) -> None:
    """Register an explicit secret value (e.g. MCP argument keys) for scrubbing."""
    if not value or len(value) < 6:
        return
    with _registered_lock:
        _registered_values.add(value)


def register_secret_values(values: Optional[list[str]]) -> None:
    for value in values or []:
        register_secret_value(value)


def clear_registered_secrets() -> None:
    with _registered_lock:
        _registered_values.clear()


def redact_secrets(text: Optional[str]) -> Optional[str]:
    """Scrub likely secrets from text before it is logged, returned over MCP,
    or persisted into an artifact/sidecar. Replaces the live values of known
    credential env vars plus common token shapes (``sk-...``, ``Bearer ...``,
    ``api_key=...``) so output that echoes a key doesn't leak it downstream.

    ``None``/empty input is returned unchanged so callers can pass optional
    fields through transparently.
    """
    if not text:
        return text
    redacted = text
    for var in _SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= 6:
            redacted = redacted.replace(value, f"<{var}:redacted>")
    with _registered_lock:
        registered = list(_registered_values)
    for value in registered:
        redacted = redacted.replace(value, "<secret:redacted>")
    redacted = _SECRET_SK.sub("sk-<redacted>", redacted)
    redacted = _SECRET_BEARER.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
    redacted = _SECRET_APIKEY.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
    redacted = _SECRET_GITHUB.sub("ghp_<redacted>", redacted)
    redacted = _SECRET_AWS.sub("AKIA<redacted>", redacted)
    redacted = _SECRET_SLACK.sub("xoxb-<redacted>", redacted)
    return _SECRET_JWT.sub("eyJ<redacted>", redacted)


def _is_secret_payload_key(key: str) -> bool:
    lower = key.lower()
    if lower in _EXPLICIT_SECRET_KEYS:
        return True
    return any(lower.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


def redact_payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with secret-bearing keys replaced for persistence."""
    if not payload:
        return payload
    return {
        key: (
            _REDACTED_PLACEHOLDER
            if _is_secret_payload_key(key) and value not in (None, "")
            else value
        )
        for key, value in payload.items()
    }
