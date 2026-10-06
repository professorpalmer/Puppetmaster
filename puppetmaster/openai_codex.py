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
from typing import Any, Iterable, Optional

PROVIDER_SLUG = "openai-codex"
BASE_URL = "https://chatgpt.com/backend-api/codex"
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


# Codex ChatGPT OAuth wire allow-list (curated catalog + common aliases).
# Tests enforce parity with shipped Codex / openai-codex catalog rows, never
# user-editable registries or model names advertised by other providers.
# *-pro variants are remapped to the base id; anything else fail-closes.
CODEX_WIRE_MODELS = frozenset(
    {
        "gpt-6-astra",
        "gpt-5.3-codex",
        "gpt-5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    }
)

# gpt-5.6-luna-pro / gpt-5.6-sol-pro / gpt-5.6-terra-pro → base tier id.
_PRO_TIER_RE = re.compile(
    r"^(gpt-5(?:\.\d+)?-(?:luna|sol|terra))-pro$",
    re.IGNORECASE,
)

# Providers that must never carry Cary's Codex-class GPT-5* pins.
_FORBIDDEN_OPENAI_API_PROVIDERS = frozenset({"openai-api", "openai"})


class UnknownCodexModelError(ValueError):
    """Raised when an openai-codex model id is not remappable and not allow-listed."""


def is_codex_class_gpt_model(model: str) -> bool:
    """Recognize GPT-5+ family ids, including provider-qualified pins."""
    bare = normalize_model_id(model).lower()
    match = re.match(r"^gpt-(\d+)(?:$|[.-])", bare)
    return bool(match and int(match.group(1)) >= 5)


def remap_codex_pro_model(model: str) -> tuple[str, Optional[str]]:
    """Remap ``gpt-5.6-*-pro`` → base tier; return ``(wire_id, remapped_from)``.

    Prefer remap over hard-reject for ``*-pro`` so ``reasoning_effort`` (esp.
    ``max``) can still ride the base Codex model. Non-pro ids are returned
    unchanged with ``remapped_from=None``.
    """
    bare = normalize_model_id(model)
    if not bare:
        return "", None
    match = _PRO_TIER_RE.match(bare)
    if match:
        return match.group(1), bare
    return bare, None


def harden_codex_model_id(model: str) -> tuple[str, Optional[str]]:
    """Normalize + remap ``*-pro``, then fail-closed on unknown wire ids.

    Returns ``(wire_model, remapped_from)``. Never returns a ``*-pro`` id.
    """
    wire, remapped_from = remap_codex_pro_model(model)
    if not wire:
        raise UnknownCodexModelError(
            f"openai-codex model id is empty after normalizing {model!r}"
        )
    if wire.lower() not in {m.lower() for m in CODEX_WIRE_MODELS}:
        raise UnknownCodexModelError(
            f"openai-codex model {wire!r} (from {model!r}) is not supported "
            f"on ChatGPT Codex OAuth. Allowed: {', '.join(sorted(CODEX_WIRE_MODELS))}."
        )
    return wire, remapped_from


def openai_api_identity_provider(*identities: Any) -> Optional[str]:
    """Return ``openai`` / ``openai-api`` when it is a non-leaf identity segment.

    ``agentic/openai/gpt-5-6-sol`` → ``openai``.
    ``agentic/openai-api/gpt-5-6-sol`` → ``openai-api``.
    ``agentic/openai-codex/...``, ``agentic/gpt-5.6-luna``, and bare model
    ids return ``None``.
    """
    for raw in identities:
        if raw is None:
            continue
        text = str(raw).strip().replace("\\", "/")
        if not text:
            continue
        parts = [part for part in text.split("/") if part]
        if len(parts) < 2:
            continue
        for segment in parts[:-1]:
            slug = segment.lower()
            if slug in _FORBIDDEN_OPENAI_API_PROVIDERS:
                return slug
    return None


def refuse_openai_api_provider(
    provider: str,
    model: str,
    *,
    identities: Optional[Iterable[Any]] = None,
) -> Optional[str]:
    """If *provider* is openai-api for a Codex-class GPT model, return openai-codex.

    Cary's OpenAI models must always use Codex auth (``OPENAI_CODEX_TOKEN``),
    never ``openai-api`` / ``OPENAI_API_KEY``. Exact ``agentic/openai/`` and
    ``agentic/openai-api/`` identities stay on the declared API lane.
    Returns ``None`` when no remap is required.
    """
    if openai_api_identity_provider(*(identities or ()), model) is not None:
        return None
    slug = (provider or "").strip().lower()
    if slug not in _FORBIDDEN_OPENAI_API_PROVIDERS:
        return None
    if not is_codex_class_gpt_model(model):
        return None
    return PROVIDER_SLUG


def harden_agentic_openai_payload(payload: dict) -> dict:
    """Force Codex auth + remap ``*-pro`` for agentic GPT-5* pins.

    Merges under the caller's keys except ``provider`` / ``model`` when a
    harden rule applies. Preserves ``reasoning_effort`` unchanged. Exact
    ``agentic/openai/`` and ``agentic/openai-api/`` identities keep the
    declared API provider and skip Codex wire remaps.
    """
    merged = dict(payload or {})
    model = str(merged.get("model") or "").strip()
    provider = str(merged.get("provider") or "").strip().lower()
    if not model:
        return merged

    identities = (merged.get("pinned_model"), merged.get("router_model_id"))
    api_lane = openai_api_identity_provider(*identities, model)
    forced = refuse_openai_api_provider(provider, model, identities=identities)
    if forced is not None:
        merged["provider"] = forced
        provider = forced
    elif not provider and is_codex_class_gpt_model(model):
        if api_lane is not None:
            merged["provider"] = api_lane
            provider = api_lane
        else:
            # Bare GPT-5* agentic pins default to Codex OAuth, never openai-api.
            merged["provider"] = PROVIDER_SLUG
            provider = PROVIDER_SLUG

    if provider == PROVIDER_SLUG or (
        is_codex_class_gpt_model(model) and provider in ("", PROVIDER_SLUG)
    ):
        try:
            wire, remapped_from = harden_codex_model_id(model)
        except UnknownCodexModelError:
            # Leave model as-is for non-codex providers; for openai-codex
            # re-raise so dispatch fail-closes with a clear error.
            if provider == PROVIDER_SLUG:
                raise
            return merged
        if remapped_from:
            merged["model"] = wire
            merged["codex_pro_remapped_from"] = remapped_from
            if merged.get("pinned_adapter_model_name") == remapped_from:
                merged["pinned_adapter_model_name"] = wire
        elif wire != model and normalize_model_id(model) == wire:
            merged["model"] = wire
    return merged
