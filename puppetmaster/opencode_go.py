"""OpenCode Go: flat-namespace catalog, per-model endpoint routing, and quirks.

OpenCode Go (https://opencode.ai/docs/go/) is a subscription reseller of open
coding models. Two consequences shape this module:

1. The namespace is FLAT. Every model is a bare id whose dots are significant
   (``kimi-k2.7-code``, ``deepseek-v4-flash``). Aggregator prefixes
   (``moonshotai/kimi-k3``) or OpenCode config prefixes (``opencode-go/kimi-k3``)
   are rejected by the relay and must be stripped before the wire.

2. The WIRE PROTOCOL VARIES PER MODEL. The published endpoint table routes
   MiniMax and Qwen to Anthropic Messages, GPT to OpenAI Responses, and
   everything else to OpenAI Chat Completions. Puppetmaster's provider
   descriptor is provider-wide, so Go selects the wire per model here instead.

PROVENANCE: reasoning and completion-ceiling behaviors are adapted from the
Hermes Agent OpenCode Go provider plugin
(``plugins/model-providers/opencode-zen/``), MIT License, Copyright (c) Nous
Research, and Marionette's ``harness/opencode_go.py``. Transport stays
Puppetmaster's stdlib ``providers`` clients.
"""
from __future__ import annotations

import urllib.parse
from typing import Optional

PROVIDER_NAME = "opencode-go"
API_KEY_ENV = "OPENCODE_GO_API_KEY"
BASE_URL = "https://opencode.ai/zen/go/v1"

CHAT_COMPLETIONS = "chat_completions"
ANTHROPIC_MESSAGES = "anthropic_messages"
OPENAI_RESPONSES = "openai_responses"

# Curated fallback matching the published Go catalog order. DeepSeek V4 Flash
# is the current DeepSeek-V4-Flash-0731 build — older v3/v2 slugs were never
# Go models and must not reappear as aliases here.
CURATED_MODELS = (
    "grok-4.5",
    "glm-5.2",
    "glm-5.1",
    "gpt-5.6-luna",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m3",
    "minimax-m2.7",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "hy3",
)

# Endpoint table (https://opencode.ai/docs/go/). Prefix matching rather than an
# exact allow-list so a newly added sibling (qwen3.8-plus, minimax-m4) routes
# correctly before the curated list catches up.
_ANTHROPIC_MESSAGES_PREFIXES = ("minimax-", "qwen")
_OPENAI_RESPONSES_PREFIXES = ("gpt-",)

# Per-model completion ceiling. The relay's default of 262144 exceeds what
# Xiaomi serves for MiMo Pro (131072) and the request 400s.
_MODEL_MAX_TOKENS = {
    "mimo-v2.5-pro": 131072,
}

_MAXED_EFFORTS = frozenset({"xhigh", "max", "ultra"})
_NATIVE_EFFORTS = frozenset({"low", "medium", "high"})


def normalize_model_id(model: Optional[str]) -> str:
    """The bare Go model id for *model*.

    Strips an ``opencode-go/`` config prefix or a copied vendor namespace while
    preserving dots, which are part of the id (``kimi-k2.7-code``).
    """
    text = str(model or "").strip()
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].strip()


def is_retired_deepseek_go_model(model: Optional[str]) -> bool:
    """True when *model* is a retired DeepSeek V2/V3 slug the Go relay dropped."""
    bare = normalize_model_id(model).lower()
    return bare.startswith("deepseek-v2") or bare.startswith("deepseek-v3")


def api_mode_for_model(model: Optional[str]) -> str:
    """Which wire protocol the Go endpoint table assigns to *model*."""
    bare = normalize_model_id(model).lower()
    if not bare:
        return CHAT_COMPLETIONS
    if bare.startswith(_ANTHROPIC_MESSAGES_PREFIXES):
        return ANTHROPIC_MESSAGES
    if bare.startswith(_OPENAI_RESPONSES_PREFIXES):
        return OPENAI_RESPONSES
    return CHAT_COMPLETIONS


def supports_wire_in_puppetmaster(model: Optional[str]) -> bool:
    """True when Puppetmaster has a client for the model's Go endpoint.

    Chat Completions, Anthropic Messages, and OpenAI Responses (GPT 5.6 Luna)
    are all implemented. Retired DeepSeek V2/V3 slugs remain unsupported.
    """
    if is_retired_deepseek_go_model(model):
        return False
    return api_mode_for_model(model) in (
        CHAT_COMPLETIONS,
        ANTHROPIC_MESSAGES,
        OPENAI_RESPONSES,
    )


def unsupported_model_message(model: Optional[str]) -> str:
    """Fail-closed error text for a Go model Puppetmaster cannot route."""
    bare = normalize_model_id(model) or str(model or "").strip() or "<empty>"
    if is_retired_deepseek_go_model(bare):
        return (
            f"OpenCode Go model {bare!r} is retired (DeepSeek V2/V3). "
            "Use deepseek-v4-flash or deepseek-v4-pro."
        )
    return f"OpenCode Go model {bare!r} is not supported."


def driver_base_url(base_url: Optional[str] = None) -> str:
    """The ``/v1`` root every Puppetmaster Go client expects.

    Sync and stream helpers append only the endpoint segment (``/messages``,
    ``/chat/completions``). An Anthropic-routed config can persist a
    ``/v1``-stripped base; re-append it for opencode.ai hosts so the POST
    does not hit the marketing site. Custom non-opencode.ai relays are left
    exactly as given.
    """
    url = str(base_url or BASE_URL).strip().rstrip("/")
    if not url or url.endswith("/v1"):
        return url or BASE_URL
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except ValueError:
        host = ""
    if host == "opencode.ai" or host.endswith(".opencode.ai"):
        return url + "/v1"
    return url


def max_tokens_for_model(model: Optional[str], requested: Optional[int]) -> int:
    """Output ceiling for *model*, clamped to what the upstream vendor serves."""
    cap = _MODEL_MAX_TOKENS.get(normalize_model_id(model).lower())
    if not requested or requested <= 0:
        return cap or 0
    if cap is None:
        return int(requested)
    return min(int(requested), cap)


def _normalize_effort(effort: Optional[str]) -> str:
    text = str(effort or "").strip().lower()
    if not text:
        return ""
    if text in ("none", "off", "disabled", "false", "0"):
        return "none"
    return text


def _is_glm_5_2(bare: str) -> bool:
    return any(token in bare for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


def _is_kimi_k2(bare: str) -> bool:
    return bare.startswith("kimi-k2")


def _is_deepseek_thinking(bare: str) -> bool:
    if bare.startswith("deepseek-v") and not bare.startswith("deepseek-v3"):
        return True
    return bare == "deepseek-reasoner"


def reasoning_body_extras(model: Optional[str], effort: Optional[str] = None) -> dict:
    """Extra request-body fields Go needs to honor a reasoning level.

    Each family speaks a different dialect and rejects the others:

    - GLM-5.2 exposes ``reasoning_effort`` with only ``high`` and ``max``.
    - Kimi K2 and DeepSeek accept ``thinking`` OR ``reasoning_effort``, never both.
    - MiMo has no reasoning knob; its quirk is the completion ceiling.

    An unrecognized model or unset effort returns ``{}`` so the relay default stands.
    """
    bare = normalize_model_id(model).lower()
    if not bare:
        return {}
    level = _normalize_effort(effort)
    if not level:
        return {}

    if _is_glm_5_2(bare):
        if level == "none":
            return {}
        return {"reasoning_effort": "max" if level in _MAXED_EFFORTS else "high"}

    if _is_kimi_k2(bare):
        if level == "none":
            return {"thinking": {"type": "disabled"}}
        if level in _MAXED_EFFORTS:
            return {"reasoning_effort": "high"}
        if level in _NATIVE_EFFORTS:
            return {"reasoning_effort": level}
        return {"thinking": {"type": "enabled"}}

    if _is_deepseek_thinking(bare):
        if level == "none":
            return {"thinking": {"type": "disabled"}}
        if level in _MAXED_EFFORTS:
            return {"reasoning_effort": "max"}
        if level in _NATIVE_EFFORTS:
            return {"reasoning_effort": level}
        return {"thinking": {"type": "enabled"}}

    return {}
