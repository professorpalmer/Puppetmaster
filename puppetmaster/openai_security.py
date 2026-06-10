"""OpenAI API key transport safety — host allowlist validation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from puppetmaster.models import Task

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Hosts the OpenAI adapter may send the OPENAI_API_KEY to. A caller-supplied
# openai_base_url pointing elsewhere would exfiltrate the bearer token to an
# arbitrary "OpenAI-compatible" endpoint, so we refuse it unless explicitly
# trusted (allowlist env or per-task opt-in).
_OPENAI_DEFAULT_ALLOWED_HOSTS = frozenset({"api.openai.com"})
_OPENAI_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def openai_allowed_hosts() -> set[str]:
    hosts = set(_OPENAI_DEFAULT_ALLOWED_HOSTS)
    extra = os.environ.get("PUPPETMASTER_OPENAI_ALLOWED_HOSTS", "")
    for host in extra.replace(",", " ").split():
        cleaned = host.strip().lower()
        if cleaned:
            hosts.add(cleaned)
    return hosts


def validate_openai_base_url(
    base_url: str,
    *,
    allow_untrusted: bool = False,
) -> Optional[str]:
    """Return an error string when sending the API key to ``base_url`` is unsafe.

    Returns ``None`` when the URL is allowed. By default the key is only sent
    over HTTPS to an allowlisted host (``api.openai.com``; extend via
    ``PUPPETMASTER_OPENAI_ALLOWED_HOSTS``). A loopback host is permitted for
    local proxies/tests, and ``allow_untrusted=True`` is an explicit override.
    """
    if allow_untrusted:
        return None
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if not host:
        return f"could not parse a host from openai_base_url {base_url!r}"
    is_loopback = host in _OPENAI_LOOPBACK_HOSTS
    if parsed.scheme != "https" and not is_loopback:
        return f"refusing to send OPENAI_API_KEY to non-HTTPS base_url {base_url!r}"
    if is_loopback or host in openai_allowed_hosts():
        return None
    return (
        f"refusing to send OPENAI_API_KEY to untrusted host {host!r}. Add it to "
        "PUPPETMASTER_OPENAI_ALLOWED_HOSTS or set "
        "payload.openai_allow_untrusted_base_url=true to override."
    )


def validate_openai_base_url_for_task(base_url: str, task: "Task") -> Optional[str]:
    """Task-aware wrapper honoring ``payload.openai_allow_untrusted_base_url``."""
    return validate_openai_base_url(
        base_url,
        allow_untrusted=bool(task.payload.get("openai_allow_untrusted_base_url")),
    )
