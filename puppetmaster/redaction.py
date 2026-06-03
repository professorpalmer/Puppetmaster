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
from typing import Optional

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
    redacted = _SECRET_SK.sub("sk-<redacted>", redacted)
    redacted = _SECRET_BEARER.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
    return _SECRET_APIKEY.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
