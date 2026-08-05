from __future__ import annotations

"""ChatGPT Codex OAuth wire for agentic workers.

Same credential Marionette pilots use (``OPENAI_CODEX_TOKEN`` against
``https://chatgpt.com/backend-api/codex``). The ChatGPT Codex backend requires
``stream: true`` on every create — non-stream POSTs 400 — so callers must take
the SSE path. Headers mirror Hermes/Marionette originator requirements so
non-browser hosts are not Cloudflare-403'd.
"""

import base64
import json
import re
from typing import Any, Optional

PROVIDER_SLUG = "openai-codex"
API_KEY_ENV = "OPENAI_CODEX_TOKEN"
BASE_URL = "https://chatgpt.com/backend-api/codex"
BASE_URL_ENV = "OPENAI_CODEX_BASE_URL"
USER_AGENT = "codex_cli_rs/0.0.0 (Puppetmaster)"


def normalize_model_id(model: str) -> str:
    """Bare Codex model id (strip provider prefixes; gpt hyphen→dot)."""
    bare = (model or "").strip()
    if not bare:
        return ""
    if ":" in bare:
        bare = bare.split(":", 1)[1].strip() or bare
    while "/" in bare:
        head, rest = bare.split("/", 1)
        if head.lower() in {
            "openai-codex",
            "codex",
            "openai",
            "agentic",
            "native",
            "cursor",
        }:
            bare = rest.strip() or bare
            continue
        break
    # Cursor registry uses gpt-5-6-*; Codex Responses expects gpt-5.6-*.
    bare = re.sub(r"(gpt-\d+)-(\d+)", r"\1.\2", bare, count=1, flags=re.I)
    return bare


def cloudflare_headers(access_token: str) -> dict[str, str]:
    """Originator + optional ChatGPT-Account-ID from the JWT claims."""
    headers = {
        "User-Agent": USER_AGENT,
        "originator": "codex_cli_rs",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Accept": "text/event-stream",
    }
    try:
        parts = (access_token or "").split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload_b64))
            auth = claims.get("https://api.openai.com/auth") or {}
            acct = auth.get("chatgpt_account_id")
            if isinstance(acct, str) and acct.strip():
                headers["ChatGPT-Account-ID"] = acct.strip()
    except Exception:
        pass
    return headers


def driver_base_url(url: Optional[str] = None) -> str:
    """Normalize Codex base URL (no trailing slash; default chatgpt.com)."""
    raw = (url or BASE_URL).strip().rstrip("/")
    return raw or BASE_URL


def merge_request_headers(
    access_token: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Cloudflare/originator headers plus optional descriptor defaults."""
    out = cloudflare_headers(access_token)
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is None:
                continue
            # Never let a bare User-Agent overwrite the originator identity.
            if str(key).lower() == "user-agent":
                continue
            if str(key).lower() == "authorization":
                continue
            out[str(key)] = str(value)
    return out
