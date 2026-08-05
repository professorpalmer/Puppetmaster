"""Provider registry for direct-API (standalone) workers.

Puppetmaster's standalone worker path calls provider HTTP APIs directly with
the user's own keys -- no external agent CLI (hermes / cursor-agent / claude /
codex) required. This module is the single source of truth for WHICH providers
exist, how to authenticate them, and which wire protocol they speak.

The descriptor shape is lifted from the Hermes provider-catalog methodology (a
per-provider record carrying its auth env vars + base URL), adapted to be
stdlib-only and to feed Puppetmaster's key-aware router: a model whose provider
has no usable credential is never offered to the router, so a fresh install
"just works" with whatever keys the user actually has.

Two wire protocols cover the entire set:

* ``"openai"`` -- the OpenAI Chat Completions shape. Covers OpenAI itself plus
  every OpenAI-compatible endpoint (OpenRouter, OpenCode Go chat models, xAI,
  DeepSeek, Groq, Mistral, Together, Nous, and local Ollama / LM Studio), and
  Google Gemini via its OpenAI-compatible endpoint -- each is just a different
  ``base_url`` + key.
* ``"anthropic"`` -- Anthropic's native ``/v1/messages`` shape (``tool_use``
  content blocks), which is different enough on the wire to warrant its own
  client. OpenCode Go MiniMax/Qwen models also use this path via the Go
  subscription base URL.

Both clients are normalized behind :func:`provider_chat`, which returns an
:class:`AssistantTurn`, so the agentic worker loop never branches on provider.
``provider=opencode-go`` additionally selects the wire per model — Chat
Completions, Anthropic Messages, or OpenAI Responses (GPT 5.6 Luna).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from puppetmaster.failure import (
    NETWORK_ERROR,
    OPENAI_SERVER_ERROR,
    RATE_LIMIT,
    SERVER_ERROR,
    TIMEOUT,
    classify_provider_failure,
)

_PROMPT_CACHE_OFF_VALUES = frozenset({"0", "false", "no", "off"})
# All Claude breakpoints (system + last tool + moving history) use 1h TTL by
# default — AGNT measured all-1h cheaper than hybrid 1h-stable + 5m-history
# (which double-writes history). Force all-5m (no ttl) via
# PUPPETMASTER_ANTHROPIC_CACHE_TTL=5m (or off/disabled synonyms).
_ANTHROPIC_CACHE_TTL_5M_VALUES = frozenset({
    "5m", "5", "off", "disabled", "0", "false", "no",
})
# OpenRouter Qwen/Alibaba slugs that require explicit ephemeral cache_control
# (automatic providers — OpenAI, Gemini, DeepSeek, Grok, Moonshot — are omitted).
_QWEN_EXPLICIT_CACHE_SLUG_FRAGMENTS = (
    "qwen-coder",
    "qwen-max",
    "qwen-plus",
    "qwen2.5-coder",
    "qwen3-coder",
)


def _anthropic_stable_cache_uses_1h() -> bool:
    raw = (os.environ.get("PUPPETMASTER_ANTHROPIC_CACHE_TTL") or "1h").strip().lower()
    return raw not in _ANTHROPIC_CACHE_TTL_5M_VALUES


def _anthropic_cache_control(*, stable: bool) -> dict:
    """Build an Anthropic cache_control marker.

    All Claude breakpoints — system, last tool schema, and moving history —
    get ``ttl:1h`` by default. Hybrid 1h-stable + 5m-history double-wrote
    history each turn; all-1h is cheaper (AGNT / Marionette parity). Dense
    loops can force all-5m (omit ``ttl``) via
    ``PUPPETMASTER_ANTHROPIC_CACHE_TTL=5m``.

    ``stable`` is retained for call-site clarity (system/tools vs history) but
    no longer gates TTL when 1h mode is on.
    """
    _ = stable  # all-1h policy; param kept for call-site intent
    if _anthropic_stable_cache_uses_1h():
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _openai_explicit_cache_kind(model: str) -> str:
    """Return ``claude``, ``qwen``, or empty string for OpenAI-wire models.

    OpenRouter auto-caches OpenAI/Gemini/DeepSeek/Grok/Moonshot — only Claude
    and Qwen/Alibaba need explicit ``cache_control`` breakpoints stamped here.
    """
    slug = (model or "").strip().lower()
    if not slug:
        return ""
    if slug.startswith("anthropic/") or "claude" in slug:
        return "claude"
    if slug.startswith("qwen/") or slug.startswith("alibaba/"):
        return "qwen"
    leaf = slug.rsplit("/", 1)[-1]
    if any(fragment in leaf for fragment in _QWEN_EXPLICIT_CACHE_SLUG_FRAGMENTS):
        return "qwen"
    return ""


def _openai_cache_control(kind: str, *, stable: bool) -> dict:
    """Build a cache_control marker for OpenAI-wire Claude/Qwen requests.

    Claude reuses the Anthropic all-1h TTL policy (or all-5m when forced).
    Qwen only accepts ``{"type": "ephemeral"}`` — never stamp a ``ttl``.
    """
    if kind == "qwen":
        return {"type": "ephemeral"}
    return _anthropic_cache_control(stable=stable)


# --- provider descriptors ---------------------------------------------------

@dataclass(frozen=True)
class ProviderDescriptor:
    """One direct-API provider Puppetmaster can call without an external CLI.

    ``api_key_env_vars`` is checked in order; the first set, non-empty value
    wins. A provider with an empty ``api_key_env_vars`` is a keyless local
    endpoint (Ollama / LM Studio) and is only considered "available" when one
    of its ``presence_env_vars`` is set, so we never route to a local server
    the user hasn't opted into.
    """

    slug: str
    wire: str  # "openai" | "anthropic"
    base_url: str
    base_url_env_var: str = ""
    api_key_env_vars: tuple[str, ...] = ()
    presence_env_vars: tuple[str, ...] = ()
    default_headers: dict[str, str] = field(default_factory=dict)
    label: str = ""

    @property
    def keyless(self) -> bool:
        return not self.api_key_env_vars


# The canonical set. Most providers speak the OpenAI wire; only Anthropic
# needs its native protocol. Add a new provider by adding a descriptor here --
# nothing else in the router or adapter needs to change.
PROVIDER_REGISTRY: dict[str, ProviderDescriptor] = {
    "openai": ProviderDescriptor(
        slug="openai",
        wire="openai",
        base_url="https://api.openai.com/v1",
        base_url_env_var="OPENAI_BASE_URL",
        api_key_env_vars=("OPENAI_API_KEY",),
        label="OpenAI",
    ),
    # Generic OpenAI-compatible endpoint driven purely by OPENAI_BASE_URL +
    # OPENAI_API_KEY. Mirrors Hermes's ``openai-api`` slug so catalog entries
    # that stamp provider=openai-api keep working.
    "openai-api": ProviderDescriptor(
        slug="openai-api",
        wire="openai",
        base_url="https://api.openai.com/v1",
        base_url_env_var="OPENAI_BASE_URL",
        api_key_env_vars=("OPENAI_API_KEY",),
        label="OpenAI (API)",
    ),
    "anthropic": ProviderDescriptor(
        slug="anthropic",
        wire="anthropic",
        base_url="https://api.anthropic.com/v1",
        base_url_env_var="ANTHROPIC_BASE_URL",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
        default_headers={"anthropic-version": "2023-06-01"},
        label="Anthropic",
    ),
    # Google Gemini via its OpenAI-compatible endpoint -- no separate client.
    "gemini": ProviderDescriptor(
        slug="gemini",
        wire="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        base_url_env_var="GEMINI_BASE_URL",
        api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        label="Google Gemini",
    ),
    "openrouter": ProviderDescriptor(
        slug="openrouter",
        wire="openai",
        base_url="https://openrouter.ai/api/v1",
        base_url_env_var="OPENROUTER_BASE_URL",
        api_key_env_vars=("OPENROUTER_API_KEY",),
        label="OpenRouter",
    ),
    # OpenCode Go subscription: flat model ids + per-model wire protocol
    # (chat/completions vs /messages vs /responses). Wire on the descriptor is
    # the chat/completions default; provider_chat selects the real path via
    # puppetmaster.opencode_go. Same OPENCODE_GO_API_KEY as Marionette.
    "opencode-go": ProviderDescriptor(
        slug="opencode-go",
        wire="openai",
        base_url="https://opencode.ai/zen/go/v1",
        base_url_env_var="OPENCODE_GO_BASE_URL",
        api_key_env_vars=("OPENCODE_GO_API_KEY",),
        label="OpenCode Go",
    ),
    # ChatGPT Codex OAuth (OPENAI_CODEX_TOKEN). Same plan credential Marionette
    # pilots use — agentic workers call chatgpt.com Responses (stream-required).
    "openai-codex": ProviderDescriptor(
        slug="openai-codex",
        wire="openai",
        base_url="https://chatgpt.com/backend-api/codex",
        base_url_env_var="OPENAI_CODEX_BASE_URL",
        api_key_env_vars=("OPENAI_CODEX_TOKEN",),
        label="ChatGPT Codex (OAuth)",
    ),
    "xai": ProviderDescriptor(
        slug="xai",
        wire="openai",
        base_url="https://api.x.ai/v1",
        base_url_env_var="XAI_BASE_URL",
        api_key_env_vars=("XAI_API_KEY",),
        label="xAI",
    ),
    "deepseek": ProviderDescriptor(
        slug="deepseek",
        wire="openai",
        base_url="https://api.deepseek.com/v1",
        base_url_env_var="DEEPSEEK_BASE_URL",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
        label="DeepSeek",
    ),
    "groq": ProviderDescriptor(
        slug="groq",
        wire="openai",
        base_url="https://api.groq.com/openai/v1",
        base_url_env_var="GROQ_BASE_URL",
        api_key_env_vars=("GROQ_API_KEY",),
        label="Groq",
    ),
    "mistral": ProviderDescriptor(
        slug="mistral",
        wire="openai",
        base_url="https://api.mistral.ai/v1",
        base_url_env_var="MISTRAL_BASE_URL",
        api_key_env_vars=("MISTRAL_API_KEY",),
        label="Mistral",
    ),
    "together": ProviderDescriptor(
        slug="together",
        wire="openai",
        base_url="https://api.together.xyz/v1",
        base_url_env_var="TOGETHER_BASE_URL",
        api_key_env_vars=("TOGETHER_API_KEY",),
        label="Together",
    ),
    "nous": ProviderDescriptor(
        slug="nous",
        wire="openai",
        base_url="https://inference-api.nousresearch.com/v1",
        base_url_env_var="NOUS_BASE_URL",
        api_key_env_vars=("NOUS_API_KEY", "HERMES_API_KEY"),
        label="Nous Research",
    ),
    # Z.AI (GLM) OpenAI-compatible PaaS — same keys Marionette keyed pilots use.
    "zai": ProviderDescriptor(
        slug="zai",
        wire="openai",
        base_url="https://api.z.ai/api/paas/v4",
        base_url_env_var="ZAI_BASE_URL",
        api_key_env_vars=("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"),
        label="Z.AI",
    ),
    # MiniMax Anthropic-compatible API (same base Marionette/SDK docs use).
    # resolve_base_url appends /v1 so _anthropic_chat's /messages suffix lands
    # on /anthropic/v1/messages like the Anthropic peer.
    "minimax": ProviderDescriptor(
        slug="minimax",
        wire="anthropic",
        base_url="https://api.minimax.io/anthropic",
        base_url_env_var="MINIMAX_BASE_URL",
        api_key_env_vars=("MINIMAX_API_KEY",),
        label="MiniMax",
    ),
    # NVIDIA NIM OpenAI-compatible integrate endpoint.
    "nvidia": ProviderDescriptor(
        slug="nvidia",
        wire="openai",
        base_url="https://integrate.api.nvidia.com/v1",
        base_url_env_var="NVIDIA_BASE_URL",
        api_key_env_vars=("NVIDIA_API_KEY",),
        label="NVIDIA NIM",
    ),
    # Keyless local endpoints: only "available" when the user points us at them
    # via a presence env var, so routing never assumes a local server is up.
    "ollama": ProviderDescriptor(
        slug="ollama",
        wire="openai",
        base_url="http://localhost:11434/v1",
        base_url_env_var="OLLAMA_BASE_URL",
        presence_env_vars=("OLLAMA_HOST", "OLLAMA_BASE_URL", "PUPPETMASTER_OLLAMA"),
        label="Ollama (local)",
    ),
    "lmstudio": ProviderDescriptor(
        slug="lmstudio",
        wire="openai",
        base_url="http://localhost:1234/v1",
        base_url_env_var="LMSTUDIO_BASE_URL",
        presence_env_vars=("LMSTUDIO_BASE_URL", "PUPPETMASTER_LMSTUDIO"),
        label="LM Studio (local)",
    ),
    # AWS Bedrock: Converse API (multi-provider) with SigV4 or bearer auth.
    # base_url is region-derived unless BEDROCK_BASE_URL overrides it —
    # see resolve_base_url / puppetmaster.bedrock.list_chat_model_ids.
    "bedrock": ProviderDescriptor(
        slug="bedrock",
        wire="anthropic",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        base_url_env_var="BEDROCK_BASE_URL",
        api_key_env_vars=("AWS_BEARER_TOKEN_BEDROCK",),
        label="AWS Bedrock",
    ),
}


def get_provider(slug: str) -> Optional[ProviderDescriptor]:
    """The descriptor for ``slug`` (case-insensitive), or ``None``."""
    if not slug:
        return None
    return PROVIDER_REGISTRY.get(str(slug).strip().lower())


def resolve_api_key(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """First non-empty API key among the descriptor's env vars, else ``None``."""
    env = env if env is not None else os.environ
    for name in desc.api_key_env_vars:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _numbered_env_names(name: str) -> "list[str]":
    """``NAME`` followed by ``NAME_2 .. NAME_9`` -- the convention for holding
    several rotating keys for one provider in the environment."""
    return [name] + [f"{name}_{i}" for i in range(2, 10)]


def provider_key_pool(
    provider: str, env: Optional[Mapping[str, str]] = None
) -> "list[str]":
    """All usable API keys for ``provider``, in rotation order.

    Unions every key env var the descriptor lists (e.g. Gemini's
    ``GEMINI_API_KEY`` + ``GOOGLE_API_KEY``) with numbered siblings
    (``OPENAI_API_KEY``, ``OPENAI_API_KEY_2``, ...), de-duplicated while
    preserving order. The adapter rotates through this pool on auth / rate-limit
    failures so one throttled or revoked key doesn't sink a worker that has
    another good key on hand. Empty for a keyless provider or an unknown slug.

    Bedrock is special: only ``AWS_BEARER_TOKEN_BEDROCK`` (+ numbered siblings)
    enter the pool. Access-key / ``~/.aws`` IAM auth is resolved inside
    ``bedrock_chat`` via SigV4 — never put an access-key id here (the agentic
    loop would pass it as ``api_key`` and Bedrock would treat it as a bearer).
    """
    desc = get_provider(provider)
    if desc is None:
        return []
    env = env if env is not None else os.environ
    if desc.slug == "bedrock":
        keys: list[str] = []
        seen: set[str] = set()
        for base in desc.api_key_env_vars:
            for name in _numbered_env_names(base):
                value = env.get(name)
                if value and value.strip() and value.strip() not in seen:
                    seen.add(value.strip())
                    keys.append(value.strip())
        return keys
    keys = []
    seen = set()
    for base in desc.api_key_env_vars:
        for name in _numbered_env_names(base):
            value = env.get(name)
            if value and value.strip() and value.strip() not in seen:
                seen.add(value.strip())
                keys.append(value.strip())
    return keys


def _minimax_driver_base_url(base_url: str) -> str:
    """Ensure MiniMax's Anthropic-compat root ends with ``/v1``.

    Descriptor / SDK docs use ``https://api.minimax.io/anthropic``; the
    Anthropic SDK appends ``/v1/messages``. Our driver only appends
    ``/messages``, so callers (and ``MINIMAX_BASE_URL`` overrides) get a
    trailing ``/v1`` when missing.
    """
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return "https://api.minimax.io/anthropic/v1"
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def resolve_base_url(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> str:
    """The provider base URL, honoring its override env var, trailing-slash trimmed.

    Bedrock derives ``https://bedrock-runtime.{region}.amazonaws.com`` from
    ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` / ``BEDROCK_REGION`` (default
    ``us-east-1``) unless ``BEDROCK_BASE_URL`` overrides it.
    """
    env = env if env is not None else os.environ
    override = env.get(desc.base_url_env_var) if desc.base_url_env_var else None
    if override and str(override).strip():
        url = str(override).strip().rstrip("/")
        if desc.slug == "opencode-go":
            from puppetmaster.opencode_go import driver_base_url

            return driver_base_url(url)
        if desc.slug == "openai-codex":
            from puppetmaster.openai_codex import driver_base_url

            return driver_base_url(url)
        if desc.slug == "minimax":
            return _minimax_driver_base_url(url)
        return url
    if desc.slug == "bedrock":
        from puppetmaster.bedrock import (
            bedrock_runtime_base_url,
            resolve_bedrock_region,
        )

        return bedrock_runtime_base_url(resolve_bedrock_region(env))
    if desc.slug == "opencode-go":
        from puppetmaster.opencode_go import driver_base_url

        return driver_base_url(desc.base_url)
    if desc.slug == "openai-codex":
        from puppetmaster.openai_codex import driver_base_url

        return driver_base_url(desc.base_url)
    if desc.slug == "minimax":
        return _minimax_driver_base_url(desc.base_url)
    return desc.base_url.rstrip("/")


def credentials_present(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> bool:
    """True when credential *presence* is visible for ``desc`` (not invoke-probed).

    For most providers this matches :func:`is_available`. Bedrock is special:
    a stale ``~/.aws/default`` profile counts as present but is not auto-routable
    until invoke-health is verified (see :func:`is_available`).
    """
    env = env if env is not None else os.environ
    if desc.slug == "bedrock":
        from puppetmaster.bedrock import bedrock_credentials_present

        return bedrock_credentials_present(env)
    if desc.keyless:
        return any(env.get(name, "").strip() for name in desc.presence_env_vars)
    return resolve_api_key(desc, env) is not None


def is_available(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> bool:
    """True when this provider is safe to *auto-route* with the current env.

    A keyed provider needs one of its API-key env vars set. A keyless local
    provider needs one of its presence env vars set (so we never route to a
    local server the user hasn't opted into). Bedrock requires both visible
    credentials *and* a current verified invoke-health record — presence of
    ``~/.aws/default`` / env keys alone is not enough. Explicit/pinned Bedrock
    calls still go through :func:`provider_chat` without this gate.
    """
    env = env if env is not None else os.environ
    if desc.slug == "bedrock":
        from puppetmaster.provider_health import is_bedrock_auto_routable

        return is_bedrock_auto_routable(env)
    return credentials_present(desc, env)


def disabled_providers(env: Optional[Mapping[str, str]] = None) -> set[str]:
    """Provider slugs explicitly disconnected via ``PUPPETMASTER_DISABLED_PROVIDERS``.

    Marionette uses this to preserve Settings disconnect state even when a stale
    shell env key is still present. Unset or empty → no providers disabled.
    """
    env = env if env is not None else os.environ
    raw = env.get("PUPPETMASTER_DISABLED_PROVIDERS", "").strip()
    if not raw:
        return set()
    return {slug.strip() for slug in raw.split(",") if slug.strip()}


def available_providers(env: Optional[Mapping[str, str]] = None) -> set[str]:
    """The set of provider slugs that are auto-routable with current credentials.

    This is the standalone analogue of Hermes's ``available_hermes_providers``:
    the router uses it to drop any direct-API model whose provider can't be
    reached, so a fresh install offers exactly the providers the user's keys
    unlock -- nothing more. Bedrock is included only when invoke-health is
    currently verified (not merely present, not denied).

    ``PUPPETMASTER_DISABLED_PROVIDERS`` (comma-separated slugs) excludes
    providers that have credentials but were disconnected in Marionette Settings.
    """
    env = env if env is not None else os.environ
    blocked = disabled_providers(env)
    return {
        slug
        for slug, desc in PROVIDER_REGISTRY.items()
        if slug not in blocked and is_available(desc, env)
    }


# --- normalized chat client -------------------------------------------------

@dataclass
class AssistantTurn:
    """One model turn, normalized across wire protocols.

    ``tool_calls`` is a list of ``{"id", "name", "arguments"}`` where
    ``arguments`` is a parsed dict. ``text`` is the assistant's prose (may be
    empty when the turn is purely tool calls).

    ``reasoning`` / ``reasoning_details`` carry provider reasoning blocks that
    some models (notably Meta Muse Spark) require echoed back on later turns so
    encrypted chain-of-thought continuity is preserved across tool loops.
    """

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    reasoning: str = ""
    reasoning_details: Optional[list] = None


class ProviderError(Exception):
    """A provider failure with both raw diagnostics and a canonical category."""

    def __init__(self, message: str, *, reason: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.reason = reason
        self.status = status
        self.body = body
        self.failure = classify_provider_failure(reason, status, body=body)


_RETRYABLE_PROVIDER_FAILURES = frozenset({
    NETWORK_ERROR,
    RATE_LIMIT,
    SERVER_ERROR,
    # Legacy observability literal must remain retryable when present on
    # ProviderError.failure (or when bridging older persisted classifications).
    OPENAI_SERVER_ERROR,
    TIMEOUT,
})
_PROVIDER_BACKOFF_BASE_SECONDS = 1.5
_PROVIDER_BACKOFF_MAX_SECONDS = 30.0


def is_retryable_provider_error(error: ProviderError) -> bool:
    """Return whether the canonical provider failure is transient."""
    return error.failure in _RETRYABLE_PROVIDER_FAILURES


def provider_retry_backoff_seconds(attempt: int) -> float:
    """Jittered exponential backoff for a zero-indexed provider retry."""
    ceiling = min(
        _PROVIDER_BACKOFF_MAX_SECONDS,
        _PROVIDER_BACKOFF_BASE_SECONDS * (2 ** max(0, attempt)),
    )
    return random.uniform(_PROVIDER_BACKOFF_BASE_SECONDS, ceiling)


def _post_json(url: str, *, headers: dict, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise ProviderError(
            _http_error_message(exc.code, err_body),
            reason=f"http_status:{exc.code}",
            status=exc.code,
            body=err_body,
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderError("request timed out", reason="timeout") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc), reason="network_error") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("malformed response", reason="malformed_response", body=raw[:8000]) from exc


def _http_error_message(status: int, body: str) -> str:
    """Human-readable HTTP error that preserves useful upstream detail.

    Keeps phrases like OpenCode's ``Request blocked by upstream provider`` in
    the exception message so classifiers and RISK artifacts can distinguish a
    vendor-side block from a genuinely invalid API key. Truncates and never
    echoes obvious secret-bearing query fragments.
    """
    text = (body or "").strip()
    if not text:
        return f"HTTP {status}"
    # Prefer a nested error.message when the body is JSON.
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    detail = ""
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("type") or "").strip()
        elif isinstance(err, str):
            detail = err.strip()
        if not detail:
            detail = str(parsed.get("message") or "").strip()
    if not detail:
        detail = text.replace("\n", " ").strip()
    if len(detail) > 400:
        detail = detail[:400] + "…"
    return f"HTTP {status}: {detail}"


def _openai_usage_fields(usage: dict) -> dict:
    """Normalize OpenAI-compatible usage into AssistantTurn.usage keys."""
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    out = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_tokens": int(cached or 0),
    }
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        out["cost_usd"] = float(cost)
    return out


def _openai_chat(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
) -> AssistantTurn:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    extra = dict(extra)
    force_tool = extra.pop("force_tool", None)
    # session_id is consumed by the OpenAI-wire cache helper for sticky routing;
    # keep it out of the generic extra merge so callers can pass it without
    # relying on the upstream accepting an unknown field on non-OpenRouter hosts.
    session_id = extra.pop("session_id", None)
    body.update(extra)
    if force_tool and tools:
        body["tool_choice"] = {"type": "function", "function": {"name": str(force_tool)}}
    if "openrouter.ai" in base_url:
        body["usage"] = {"include": True}
    cache_extra = {"session_id": session_id} if session_id is not None else {}
    body = _apply_openai_explicit_cache(
        body, model=model, base_url=base_url, extra=cache_extra,
    )
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _post_json(
        f"{base_url}/chat/completions",
        headers={"User-Agent": "puppetmaster-agentic", **auth, **headers},
        body=body,
        timeout=timeout,
    )
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    finish = choices[0].get("finish_reason") if choices else None
    tool_calls: list[dict] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (json.JSONDecodeError, TypeError):
            args = {"__raw__": raw_args}
        tool_calls.append({"id": call.get("id") or "", "name": fn.get("name") or "", "arguments": args})
    usage = data.get("usage") or {}
    reasoning_raw = message.get("reasoning")
    if reasoning_raw is None:
        reasoning_raw = message.get("reasoning_content")
    reasoning_text = str(reasoning_raw or "").strip()
    details = message.get("reasoning_details")
    if details is not None and not isinstance(details, list):
        details = [details]
    return AssistantTurn(
        text=str(message.get("content") or "").strip(),
        tool_calls=tool_calls,
        finish_reason=str(finish or ""),
        usage=_openai_usage_fields(usage),
        raw=data,
        reasoning=reasoning_text,
        reasoning_details=details if isinstance(details, list) else None,
    )


def _prompt_cache_enabled() -> bool:
    raw = (os.environ.get("PUPPETMASTER_PROMPT_CACHE") or "").strip().lower()
    return raw not in _PROMPT_CACHE_OFF_VALUES


def _mark_anthropic_cache_block(msg: dict, *, stable: bool = False) -> None:
    """Attach an ephemeral cache_control marker to a conversation message.

    Anthropic 400s on ``cache_control`` for empty text blocks, so only mark a
    block that carries real content. Whitespace-only text is skipped entirely;
    non-text blocks (tool_use / tool_result / image) may carry the marker.
    History callers pass ``stable=False`` for call-site clarity; under the
    default all-1h policy every Claude marker still gets ``ttl:1h``.
    """
    marker = _anthropic_cache_control(stable=stable)
    content = msg.get("content")
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            if last.get("type") == "text" and not str(last.get("text") or "").strip():
                return
            content[-1] = {**last, "cache_control": marker}
    elif isinstance(content, str):
        if not content.strip():
            return
        msg["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": marker,
        }]


def _apply_anthropic_cache_control(body: dict) -> dict:
    """Opt into Anthropic prompt caching via cache_control breakpoints.

    Marks up to four blocks (Anthropic's per-request cap): system + last tool
    + the second-to-last and last messages. All Claude markers use ``ttl:1h``
    by default (all-1h policy); ``PUPPETMASTER_ANTHROPIC_CACHE_TTL=5m`` forces
    all-5m (no ttl). Never raises -- on any failure the unmarked body is
    returned so a lost cache never fails the request.
    """
    if not _prompt_cache_enabled():
        return body
    try:
        out = copy.deepcopy(body)
        system = out.get("system")
        if isinstance(system, str) and system.strip():
            out["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": _anthropic_cache_control(stable=True),
            }]
        elif isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict):
                if not (last.get("type") == "text" and not str(last.get("text") or "").strip()):
                    system[-1] = {**last, "cache_control": _anthropic_cache_control(stable=True)}
            elif isinstance(last, str) and last.strip():
                system[-1] = {
                    "type": "text",
                    "text": last,
                    "cache_control": _anthropic_cache_control(stable=True),
                }

        tools = out.get("tools")
        if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
            tools[-1] = {
                **tools[-1],
                "cache_control": _anthropic_cache_control(stable=True),
            }

        messages = out.get("messages")
        if isinstance(messages, list) and messages:
            if len(messages) >= 2 and isinstance(messages[-2], dict):
                _mark_anthropic_cache_block(messages[-2], stable=False)
            if isinstance(messages[-1], dict):
                _mark_anthropic_cache_block(messages[-1], stable=False)
        return out
    except Exception:
        return body


def _mark_openai_cache_block(msg: dict, marker: dict) -> None:
    """Attach ``cache_control`` to an OpenAI chat-completions message.

    Content may be a string or a list of parts. Empty / whitespace-only text
    is skipped (providers reject markers on empty blocks). Non-text parts
    (e.g. tool_calls-only assistants) are left unmarked.
    """
    content = msg.get("content")
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            if last.get("type") == "text" and not str(last.get("text") or "").strip():
                return
            if last.get("type") not in (None, "text"):
                # Prefer marking a trailing text part when present.
                for idx in range(len(content) - 1, -1, -1):
                    part = content[idx]
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and str(part.get("text") or "").strip()
                    ):
                        content[idx] = {**part, "cache_control": marker}
                        return
                return
            content[-1] = {**last, "cache_control": marker}
        elif isinstance(last, str) and last.strip():
            content[-1] = {
                "type": "text",
                "text": last,
                "cache_control": marker,
            }
    elif isinstance(content, str):
        if not content.strip():
            return
        msg["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": marker,
        }]


def _openai_session_id_from_messages(messages: list) -> str:
    """Stable sticky-routing id from the early conversation prefix."""
    digest = hashlib.sha256()
    for msg in messages[:3]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    parts.append(part)
            content_key = "\n".join(parts)
        else:
            content_key = str(content or "")
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def _apply_openai_explicit_cache(
    body: dict,
    *,
    model: str,
    base_url: str,
    extra: Optional[dict] = None,
) -> dict:
    """Stamp explicit cache_control breakpoints on OpenAI-wire Claude/Qwen.

    Prefer per-block markers (Marionette parity / fine control) over OpenRouter's
    top-level ``cache_control`` automatic mode — do not set a top-level marker
    that would fight per-block TTL policy. Claude markers follow the all-1h
    Anthropic policy (or all-5m when forced); Qwen stays ephemeral-only.
    On OpenRouter, also set ``session_id`` for best-effort sticky routing
    across turns.

    Never raises — on any failure the unmarked body is returned.
    """
    if not _prompt_cache_enabled():
        return body
    kind = _openai_explicit_cache_kind(model)
    if not kind:
        return body
    try:
        out = copy.deepcopy(body)
        messages = out.get("messages")
        if not isinstance(messages, list):
            messages = []

        # System + last tool schema (Claude all-1h / Qwen ephemeral).
        # System lives inside ``messages`` on the OpenAI wire (unlike Anthropic's
        # top-level ``system``), so history markers below skip role=system.
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                _mark_openai_cache_block(
                    msg, _openai_cache_control(kind, stable=True),
                )
                break

        tools = out.get("tools")
        if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
            tools[-1] = {
                **tools[-1],
                "cache_control": _openai_cache_control(kind, stable=True),
            }

        # Moving history: last two non-system messages (Claude all-1h by
        # default; Qwen ephemeral-only).
        convo_idxs = [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") != "system"
        ]
        if convo_idxs:
            history_marker = _openai_cache_control(kind, stable=False)
            for idx in convo_idxs[-2:]:
                _mark_openai_cache_block(messages[idx], history_marker)

        if "openrouter.ai" in (base_url or ""):
            # Sticky routing so cache hits land on the same upstream worker.
            if "session_id" not in out:
                sid = None
                if isinstance(extra, dict):
                    raw_sid = extra.get("session_id")
                    if raw_sid is not None and str(raw_sid).strip():
                        sid = str(raw_sid).strip()
                if not sid:
                    sid = _openai_session_id_from_messages(messages)
                if sid:
                    out["session_id"] = sid
        return out
    except Exception:
        return body


def _anthropic_chat(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
) -> AssistantTurn:
    # Split the OpenAI-style system message out; Anthropic takes it top-level.
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    convo = [m for m in messages if m.get("role") != "system"]
    body: dict[str, Any] = {
        "model": model,
        "messages": _to_anthropic_messages(convo),
        "max_tokens": int(extra.get("max_tokens") or extra.get("max_completion_tokens") or 4096),
    }
    if system_parts:
        body["system"] = "\n\n".join(str(p) for p in system_parts)
    if tools:
        body["tools"] = [_to_anthropic_tool(t) for t in tools]
    force_tool = extra.get("force_tool")
    if force_tool and tools:
        body["tool_choice"] = {"type": "tool", "name": str(force_tool)}
    for key in ("temperature", "top_p", "stop_sequences"):
        if key in extra:
            body[key] = extra[key]
    body = _apply_anthropic_cache_control(body)
    auth = {"x-api-key": api_key} if api_key else {}
    data = _post_json(
        f"{base_url}/messages",
        headers={"User-Agent": "puppetmaster-agentic", "anthropic-version": "2023-06-01", **auth, **headers},
        body=body,
        timeout=timeout,
    )
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") or "",
                "name": block.get("name") or "",
                "arguments": block.get("input") or {},
            })
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=str(data.get("stop_reason") or ""),
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        },
        raw=data,
    )


def _to_anthropic_messages(convo: list[dict]) -> list[dict]:
    """Translate OpenAI-style messages to Anthropic content-block messages.

    Handles user/assistant prose, assistant ``tool_calls`` (-> ``tool_use``
    blocks), and ``role=tool`` results (-> a user message carrying a
    ``tool_result`` block, which is how Anthropic threads tool output).
    """
    out: list[dict] = []
    for msg in convo:
        role = msg.get("role")
        if role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or "",
                    "content": str(msg.get("content") or ""),
                }],
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": str(msg["content"])})
            for call in msg["tool_calls"]:
                fn = call.get("function") or call
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") or "",
                    "name": fn.get("name") or "",
                    "input": args,
                })
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": role or "user", "content": str(msg.get("content") or "")})
    return out


def _to_anthropic_tool(tool: dict) -> dict:
    """Translate an OpenAI function-tool spec to Anthropic's tool shape."""
    fn = tool.get("function") or tool
    return {
        "name": fn.get("name") or "",
        "description": fn.get("description") or "",
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _open_stream(url: str, *, headers: dict, body: dict, timeout: int):
    """POST ``body`` and return the raw streaming response for SSE iteration.

    Mirrors :func:`_post_json`'s error normalization so a streaming call raises
    the same classifiable :class:`ProviderError` on HTTP/transport failure.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps({**body, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **headers},
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise ProviderError(
            _http_error_message(exc.code, err_body),
            reason=f"http_status:{exc.code}",
            status=exc.code,
            body=err_body,
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderError("request timed out", reason="timeout") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc), reason="network_error") from exc


def _iter_sse_data(response) -> "Any":
    """Yield the payload of each ``data:`` SSE line from a streaming response."""
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line or not line.startswith("data:"):
            continue
        yield line[len("data:"):].strip()


def _openai_chat_stream(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
    on_delta: Optional[Callable[[str, str], None]],
) -> AssistantTurn:
    extra = dict(extra)
    force_tool = extra.pop("force_tool", None)
    session_id = extra.pop("session_id", None)
    body: dict[str, Any] = {
        "model": model, "messages": messages,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    if force_tool and tools:
        body["tool_choice"] = {"type": "function", "function": {"name": str(force_tool)}}
    body.update(extra)
    if "openrouter.ai" in base_url:
        body["usage"] = {"include": True}
    cache_extra = {"session_id": session_id} if session_id is not None else {}
    body = _apply_openai_explicit_cache(
        body, model=model, base_url=base_url, extra=cache_extra,
    )
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = _open_stream(
        f"{base_url}/chat/completions",
        headers={"User-Agent": "puppetmaster-agentic", **auth, **headers},
        body=body, timeout=timeout,
    )
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_details_acc: list = []
    tool_acc: dict[int, dict] = {}
    finish = ""
    usage: dict = {}
    try:
        for payload in _iter_sse_data(response):
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    text_parts.append(piece)
                    if on_delta:
                        on_delta("text", piece)
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                    if on_delta:
                        on_delta("reasoning", str(reasoning))
                details = delta.get("reasoning_details")
                if isinstance(details, list):
                    reasoning_details_acc.extend(details)
                elif details is not None:
                    reasoning_details_acc.append(details)
                for call in delta.get("tool_calls") or []:
                    idx = int(call.get("index") or 0)
                    slot = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    finally:
        response.close()
    tool_calls: list[dict] = []
    for idx in sorted(tool_acc):
        slot = tool_acc[idx]
        raw_args = slot["args"]
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {"__raw__": raw_args}
        tool_calls.append({"id": slot["id"] or "", "name": slot["name"] or "", "arguments": args})
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=str(finish or ""),
        usage=_openai_usage_fields(usage),
        raw={},
        reasoning="".join(reasoning_parts).strip(),
        reasoning_details=reasoning_details_acc or None,
    )


def _anthropic_chat_stream(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
    on_delta: Optional[Callable[[str, str], None]],
) -> AssistantTurn:
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    convo = [m for m in messages if m.get("role") != "system"]
    body: dict[str, Any] = {
        "model": model,
        "messages": _to_anthropic_messages(convo),
        "max_tokens": int(extra.get("max_tokens") or extra.get("max_completion_tokens") or 4096),
    }
    if system_parts:
        body["system"] = "\n\n".join(str(p) for p in system_parts)
    if tools:
        body["tools"] = [_to_anthropic_tool(t) for t in tools]
    force_tool = extra.get("force_tool")
    if force_tool and tools:
        body["tool_choice"] = {"type": "tool", "name": str(force_tool)}
    for key in ("temperature", "top_p", "stop_sequences"):
        if key in extra:
            body[key] = extra[key]
    body = _apply_anthropic_cache_control(body)
    auth = {"x-api-key": api_key} if api_key else {}
    response = _open_stream(
        f"{base_url}/messages",
        headers={"User-Agent": "puppetmaster-agentic", "anthropic-version": "2023-06-01", **auth, **headers},
        body=body, timeout=timeout,
    )
    text_parts: list[str] = []
    blocks: dict[int, dict] = {}
    finish = ""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    try:
        for payload in _iter_sse_data(response):
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "message_start":
                msg_usage = ((event.get("message") or {}).get("usage") or {})
                prompt_tokens = int(msg_usage.get("input_tokens") or 0)
                cached_tokens = int(msg_usage.get("cache_read_input_tokens") or 0)
                cache_write_tokens = int(msg_usage.get("cache_creation_input_tokens") or 0)
            elif etype == "content_block_start":
                idx = int(event.get("index") or 0)
                block = event.get("content_block") or {}
                blocks[idx] = {"type": block.get("type"), "id": block.get("id", ""),
                               "name": block.get("name", ""), "args": ""}
            elif etype == "content_block_delta":
                idx = int(event.get("index") or 0)
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    piece = delta.get("text") or ""
                    if piece:
                        text_parts.append(piece)
                        if on_delta:
                            on_delta("text", piece)
                elif delta.get("type") == "thinking_delta":
                    if on_delta and delta.get("thinking"):
                        on_delta("reasoning", str(delta["thinking"]))
                elif delta.get("type") == "input_json_delta":
                    slot = blocks.setdefault(idx, {"type": "tool_use", "id": "", "name": "", "args": ""})
                    slot["args"] += delta.get("partial_json") or ""
            elif etype == "message_delta":
                finish = str((event.get("delta") or {}).get("stop_reason") or finish)
                completion_tokens = int((event.get("usage") or {}).get("output_tokens") or completion_tokens)
            elif etype == "message_stop":
                break
    finally:
        response.close()
    tool_calls: list[dict] = []
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot.get("type") != "tool_use":
            continue
        raw_args = slot.get("args") or ""
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {"__raw__": raw_args}
        tool_calls.append({"id": slot.get("id") or "", "name": slot.get("name") or "", "arguments": args})
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        },
        raw={},
    )


# --- OpenAI Responses (stdlib) -----------------------------------------------

def _flatten_message_text(content: Any) -> str:
    """Coerce chat-message content (string or text-part list) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part:
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _messages_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI chat messages to Responses ``(instructions, input)``.

    System messages become top-level ``instructions``. Assistant tool-call turns
    expand into ``function_call`` items; tool results become
    ``function_call_output``. User/assistant prose stays as message items with
    the role-correct text part type.
    """
    instructions_parts: list[str] = []
    input_items: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "system":
            text = _flatten_message_text(content).strip()
            if text:
                instructions_parts.append(text)
            continue
        if role == "tool":
            output = content if isinstance(content, str) else (
                json.dumps(content) if content is not None else ""
            )
            input_items.append({
                "type": "function_call_output",
                "call_id": str(msg.get("tool_call_id") or msg.get("id") or ""),
                "output": output,
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            text = _flatten_message_text(content)
            if text:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    args_text = raw_args
                else:
                    args_text = json.dumps(raw_args or {})
                input_items.append({
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(fn.get("name") or ""),
                    "arguments": args_text or "{}",
                })
            continue
        text = _flatten_message_text(content)
        part_type = "output_text" if role == "assistant" else "input_text"
        wire_role = "user" if role == "user" else role
        input_items.append({
            "type": "message",
            "role": wire_role,
            "content": [{"type": part_type, "text": text}],
        })
    return "\n\n".join(instructions_parts), input_items


def _tools_to_responses(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Translate OpenAI chat function-tool specs to Responses flat tools."""
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" or "function" in tool:
            fn = tool.get("function") or {}
            out.append({
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        elif tool.get("name"):
            out.append({
                "type": "function",
                "name": tool.get("name") or "",
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            })
    return out or None


def _responses_usage_fields(usage: dict) -> dict:
    """Normalize Responses (or chat-shaped) usage into AssistantTurn.usage keys."""
    if not isinstance(usage, dict):
        usage = {}
    prompt = usage.get("input_tokens")
    if prompt is None:
        prompt = usage.get("prompt_tokens")
    completion = usage.get("output_tokens")
    if completion is None:
        completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    prompt_i = int(prompt or 0)
    completion_i = int(completion or 0)
    if total is None:
        total_i = prompt_i + completion_i
    else:
        total_i = int(total or 0)
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = 0
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    out = {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
        "cached_tokens": cached,
    }
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        out["cost_usd"] = float(cost)
    return out


def _parse_function_call_arguments(raw_args: Any) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"__raw__": raw_args}
        return parsed if isinstance(parsed, dict) else {"__raw__": raw_args}
    if raw_args is None:
        return {}
    return {"__raw__": raw_args}


def _reasoning_text_from_responses_item(item: dict) -> str:
    """Extract visible reasoning summary text from a Responses reasoning item."""
    parts: list[str] = []
    summary = item.get("summary")
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(part, str) and part:
                parts.append(part)
    elif isinstance(summary, str) and summary:
        parts.append(summary)
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "reasoning_text", "summary_text", "output_text", "text",
            ):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
    return "".join(parts)


def _assistant_turn_from_responses(data: dict) -> AssistantTurn:
    """Parse a non-stream (or assembled) Responses body into AssistantTurn."""
    if not isinstance(data, dict):
        raise ProviderError(
            "malformed responses body",
            reason="malformed_response",
            body=str(data)[:8000],
        )
    status = str(data.get("status") or "")
    err = data.get("error")
    if status == "failed" or err:
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("code") or err)[:800]
        elif err:
            detail = str(err)[:800]
        else:
            detail = "responses request failed"
        raise ProviderError(
            detail,
            reason="provider_error",
            body=json.dumps(data)[:8000] if data else detail,
        )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    saw_message = False
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or "")
        if itype == "message":
            # Skip commentary/analysis phases so progress/reasoning never
            # contaminates the final answer text.
            phase = item.get("phase")
            if isinstance(phase, str) and phase.strip().lower() in (
                "commentary", "analysis",
            ):
                if phase.strip().lower() == "analysis":
                    for part in item.get("content") or []:
                        if isinstance(part, dict) and part.get("type") in (
                            "output_text", "text",
                        ):
                            piece = part.get("text")
                            if isinstance(piece, str) and piece:
                                reasoning_parts.append(piece)
                continue
            saw_message = True
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in (
                    "output_text", "text",
                ):
                    piece = part.get("text")
                    if isinstance(piece, str) and piece:
                        text_parts.append(piece)
        elif itype == "function_call":
            tool_calls.append({
                "id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": _parse_function_call_arguments(item.get("arguments")),
            })
        elif itype == "reasoning" or itype.startswith("reasoning"):
            piece = _reasoning_text_from_responses_item(item)
            if piece:
                reasoning_parts.append(piece)

    if not text_parts and not saw_message and isinstance(data.get("output_text"), str):
        text_parts.append(data["output_text"])

    if tool_calls:
        finish = "tool_calls"
    elif status == "incomplete":
        finish = "length"
    elif status:
        finish = status
    else:
        finish = "stop"

    reasoning = "".join(reasoning_parts).strip()
    if not reasoning and isinstance(data.get("reasoning"), str):
        reasoning = data["reasoning"].strip()

    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=_responses_usage_fields(data.get("usage") or {}),
        raw=data,
        reasoning=reasoning,
    )


def _build_responses_body(
    *,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    extra: dict,
) -> dict:
    """Build a Responses create body (no ChatGPT-only metadata)."""
    extra = dict(extra)
    force_tool = extra.pop("force_tool", None)
    # Chat Completions leftovers that Responses does not accept.
    extra.pop("session_id", None)
    extra.pop("stream_options", None)
    max_tokens = extra.pop("max_tokens", None)
    max_completion = extra.pop("max_completion_tokens", None)
    instructions, input_items = _messages_to_responses_input(messages)
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
    }
    if instructions:
        body["instructions"] = instructions
    resp_tools = _tools_to_responses(tools)
    if resp_tools:
        body["tools"] = resp_tools
        if force_tool:
            body["tool_choice"] = {"type": "function", "name": str(force_tool)}
        else:
            body["tool_choice"] = "auto"
    ceiling = max_tokens if max_tokens is not None else max_completion
    if ceiling is not None:
        try:
            value = int(ceiling)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            body["max_output_tokens"] = value
    # Forward only known-safe extras (e.g. temperature). Never invent
    # ChatGPT client_metadata / session affinity fields.
    for key in ("temperature", "top_p", "metadata", "reasoning", "reasoning_effort"):
        if key in extra:
            body[key] = extra[key]
    return body


def _openai_responses_chat(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
) -> AssistantTurn:
    body = _build_responses_body(
        model=model, messages=messages, tools=tools, extra=extra,
    )
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _post_json(
        f"{base_url.rstrip('/')}/responses",
        headers={"User-Agent": "puppetmaster-agentic", **auth, **headers},
        body=body,
        timeout=timeout,
    )
    return _assistant_turn_from_responses(data)


def _openai_responses_chat_stream(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
    on_delta: Optional[Callable[[str, str], None]],
) -> AssistantTurn:
    body = _build_responses_body(
        model=model, messages=messages, tools=tools, extra=extra,
    )
    body["stream"] = True
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = _open_stream(
        f"{base_url.rstrip('/')}/responses",
        headers={"User-Agent": "puppetmaster-agentic", **auth, **headers},
        body=body,
        timeout=timeout,
    )
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    # output_index -> accumulating function_call slot
    tool_slots: dict[int, dict] = {}
    collected_items: list[dict] = []
    usage: dict = {}
    status = ""
    stream_error: Optional[str] = None
    try:
        for payload in _iter_sse_data(response):
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            etype = str(event.get("type") or "")

            if etype == "error":
                stream_error = str(
                    event.get("message") or event.get("error") or "stream error"
                )[:800]
                break

            if etype in (
                "response.output_text.delta",
                "response.content_part.delta",
            ) or etype.endswith("output_text.delta"):
                # Content-part deltas may carry nested delta objects; prefer
                # the string form used by standard Responses SSE.
                delta = event.get("delta")
                piece = ""
                if isinstance(delta, str):
                    piece = delta
                elif isinstance(delta, dict):
                    piece = str(delta.get("text") or delta.get("delta") or "")
                if piece:
                    text_parts.append(piece)
                    if on_delta:
                        on_delta("text", piece)
                continue

            if (
                "reasoning" in etype
                and "delta" in etype
            ) or etype in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                delta = event.get("delta")
                piece = delta if isinstance(delta, str) else ""
                if piece:
                    reasoning_parts.append(piece)
                    if on_delta:
                        on_delta("reasoning", piece)
                continue

            if etype == "response.output_item.added":
                item = event.get("item") or {}
                if not isinstance(item, dict):
                    continue
                idx = int(event.get("output_index") or 0)
                if str(item.get("type") or "") == "function_call":
                    slot = tool_slots.setdefault(
                        idx, {"id": "", "name": "", "args": ""},
                    )
                    if item.get("call_id") or item.get("id"):
                        slot["id"] = str(item.get("call_id") or item.get("id") or "")
                    if item.get("name"):
                        slot["name"] = str(item.get("name") or "")
                    if isinstance(item.get("arguments"), str) and item["arguments"]:
                        slot["args"] = item["arguments"]
                continue

            if etype == "response.function_call_arguments.delta" or (
                "function_call_arguments" in etype and etype.endswith(".delta")
            ):
                idx = int(event.get("output_index") or 0)
                slot = tool_slots.setdefault(idx, {"id": "", "name": "", "args": ""})
                piece = event.get("delta")
                if isinstance(piece, str) and piece:
                    slot["args"] += piece
                # Some relays stamp call_id/name on the delta event itself.
                if event.get("call_id"):
                    slot["id"] = str(event["call_id"])
                if event.get("name"):
                    slot["name"] = str(event["name"])
                continue

            if etype == "response.function_call_arguments.done":
                idx = int(event.get("output_index") or 0)
                slot = tool_slots.setdefault(idx, {"id": "", "name": "", "args": ""})
                args = event.get("arguments")
                if isinstance(args, str) and args:
                    slot["args"] = args
                if event.get("call_id"):
                    slot["id"] = str(event["call_id"])
                if event.get("name"):
                    slot["name"] = str(event["name"])
                continue

            if etype == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    collected_items.append(item)
                    idx = int(event.get("output_index") or 0)
                    if str(item.get("type") or "") == "function_call":
                        slot = tool_slots.setdefault(
                            idx, {"id": "", "name": "", "args": ""},
                        )
                        if item.get("call_id") or item.get("id"):
                            slot["id"] = str(
                                item.get("call_id") or item.get("id") or ""
                            )
                        if item.get("name"):
                            slot["name"] = str(item.get("name") or "")
                        if isinstance(item.get("arguments"), str):
                            slot["args"] = item["arguments"]
                continue

            if etype in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                resp_obj = event.get("response")
                if isinstance(resp_obj, dict):
                    if resp_obj.get("usage"):
                        usage = resp_obj["usage"]
                    status = str(resp_obj.get("status") or status)
                    if etype == "response.failed":
                        err = resp_obj.get("error") or resp_obj
                        if isinstance(err, dict):
                            stream_error = str(
                                err.get("message") or err.get("code") or err
                            )[:800]
                        else:
                            stream_error = str(err)[:800]
                    # Prefer terminal output items when present (complete
                    # function_call / message payloads).
                    terminal_output = resp_obj.get("output")
                    if isinstance(terminal_output, list) and terminal_output:
                        collected_items = [
                            i for i in terminal_output if isinstance(i, dict)
                        ]
                if not status:
                    status = etype.rsplit(".", 1)[-1]
                break

            if etype == "response.completed" or etype.endswith(".completed"):
                break
    finally:
        response.close()

    if stream_error:
        raise ProviderError(stream_error, reason="provider_error", body=stream_error)

    if collected_items:
        assembled = {
            "status": status or "completed",
            "output": collected_items,
            "usage": usage,
            "reasoning": "".join(reasoning_parts),
        }
        # Prefer delta-assembled answer text when output items omit it (some
        # relays only stream text via deltas and leave message content empty).
        turn = _assistant_turn_from_responses(assembled)
        if not turn.text and text_parts:
            turn.text = "".join(text_parts).strip()
        if not turn.reasoning and reasoning_parts:
            turn.reasoning = "".join(reasoning_parts).strip()
        if not turn.tool_calls and tool_slots:
            for idx in sorted(tool_slots):
                slot = tool_slots[idx]
                turn.tool_calls.append({
                    "id": slot.get("id") or "",
                    "name": slot.get("name") or "",
                    "arguments": _parse_function_call_arguments(slot.get("args")),
                })
            if turn.tool_calls and turn.finish_reason not in ("length", "incomplete"):
                turn.finish_reason = "tool_calls"
        if usage and not turn.usage.get("total_tokens"):
            turn.usage = _responses_usage_fields(usage)
        return turn

    tool_calls: list[dict] = []
    for idx in sorted(tool_slots):
        slot = tool_slots[idx]
        tool_calls.append({
            "id": slot.get("id") or "",
            "name": slot.get("name") or "",
            "arguments": _parse_function_call_arguments(slot.get("args")),
        })
    finish = "tool_calls" if tool_calls else (status or "completed")
    if finish == "incomplete":
        finish = "length"
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=_responses_usage_fields(usage),
        raw={},
        reasoning="".join(reasoning_parts).strip(),
    )


def _prepare_opencode_go_call(
    *,
    model: str,
    extra: dict,
    base_url: str,
) -> tuple[str, str, dict, str]:
    """Normalize a Go model and select wire extras; fail closed when unsupported.

    Returns ``(bare_model, api_mode, prepared_extra, healed_base_url)``.
    """
    from puppetmaster.opencode_go import (
        ANTHROPIC_MESSAGES,
        CHAT_COMPLETIONS,
        OPENAI_RESPONSES,
        api_mode_for_model,
        driver_base_url,
        is_retired_deepseek_go_model,
        max_tokens_for_model,
        normalize_model_id,
        reasoning_body_extras,
        unsupported_model_message,
    )

    bare = normalize_model_id(model)
    if not bare:
        raise ProviderError(
            unsupported_model_message(model),
            reason="unsupported_model",
        )
    if is_retired_deepseek_go_model(bare):
        raise ProviderError(
            unsupported_model_message(bare),
            reason="unsupported_model",
        )
    mode = api_mode_for_model(bare)
    prepared = dict(extra)
    # Go families speak distinct reasoning dialects; never forward a bare
    # OpenAI-style reasoning_effort that would 400 on thinking-only models.
    effort = prepared.pop("reasoning_effort", None)
    prepared.pop("thinking", None)
    extras = reasoning_body_extras(bare, effort)
    prepared.update(extras)
    requested = prepared.get("max_tokens") or prepared.get("max_completion_tokens")
    try:
        requested_int = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        requested_int = 0
    ceiling = max_tokens_for_model(bare, requested_int or None)
    if ceiling:
        prepared["max_tokens"] = ceiling
        prepared.pop("max_completion_tokens", None)
    url = driver_base_url(base_url)
    if mode not in (CHAT_COMPLETIONS, ANTHROPIC_MESSAGES, OPENAI_RESPONSES):
        raise ProviderError(
            unsupported_model_message(bare),
            reason="unsupported_model",
        )
    return bare, mode, prepared, url


def _openai_codex_chat(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    extra: dict,
    headers: dict,
    timeout: int,
    on_delta: Optional[Callable[[str, str], None]] = None,
) -> AssistantTurn:
    """ChatGPT Codex Responses call — always SSE (backend rejects non-stream)."""
    from puppetmaster.openai_codex import (
        driver_base_url,
        merge_request_headers,
        normalize_model_id,
    )

    if not api_key:
        raise ProviderError(
            "no OPENAI_CODEX_TOKEN for openai-codex",
            reason="not_authenticated",
        )
    bare = normalize_model_id(model)
    if not bare:
        raise ProviderError(
            f"empty openai-codex model id from {model!r}",
            reason="unsupported_model",
        )
    url = driver_base_url(base_url)
    req_headers = merge_request_headers(api_key, headers)
    return _openai_responses_chat_stream(
        base_url=url,
        api_key=api_key,
        model=bare,
        messages=messages,
        tools=tools,
        extra=dict(extra or {}),
        headers=req_headers,
        timeout=timeout,
        on_delta=on_delta,
    )


def _opencode_go_chat(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    extra: dict,
    headers: dict,
    timeout: int,
    on_delta: Optional[Callable[[str, str], None]] = None,
    stream: bool = False,
) -> AssistantTurn:
    """Dispatch an OpenCode Go call on the model's published endpoint.

    ``stream=True`` (used by :func:`provider_chat_streaming`) always takes the
    SSE client even when ``on_delta`` is omitted, matching non-Go providers.
    """
    from puppetmaster.opencode_go import ANTHROPIC_MESSAGES, OPENAI_RESPONSES

    bare, mode, prepared, url = _prepare_opencode_go_call(
        model=model, extra=extra, base_url=base_url,
    )
    use_stream = stream or on_delta is not None
    if mode == OPENAI_RESPONSES:
        if use_stream:
            return _openai_responses_chat_stream(
                base_url=url, api_key=api_key, model=bare, messages=messages,
                tools=tools, extra=prepared, headers=headers, timeout=timeout,
                on_delta=on_delta,
            )
        return _openai_responses_chat(
            base_url=url, api_key=api_key, model=bare, messages=messages,
            tools=tools, extra=prepared, headers=headers, timeout=timeout,
        )
    if mode == ANTHROPIC_MESSAGES:
        if use_stream:
            return _anthropic_chat_stream(
                base_url=url, api_key=api_key, model=bare, messages=messages,
                tools=tools, extra=prepared, headers=headers, timeout=timeout,
                on_delta=on_delta,
            )
        return _anthropic_chat(
            base_url=url, api_key=api_key, model=bare, messages=messages,
            tools=tools, extra=prepared, headers=headers, timeout=timeout,
        )
    if use_stream:
        return _openai_chat_stream(
            base_url=url, api_key=api_key, model=bare, messages=messages,
            tools=tools, extra=prepared, headers=headers, timeout=timeout,
            on_delta=on_delta,
        )
    return _openai_chat(
        base_url=url, api_key=api_key, model=bare, messages=messages,
        tools=tools, extra=prepared, headers=headers, timeout=timeout,
    )


def provider_chat_streaming(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
    on_delta: Optional[Callable[[str, str], None]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Mapping[str, str]] = None,
) -> AssistantTurn:
    """Streaming twin of :func:`provider_chat`.

    Streams the model turn over SSE, invoking ``on_delta(kind, text)`` for each
    text/reasoning chunk (``kind`` is ``"text"`` or ``"reasoning"``), and returns
    the same normalized :class:`AssistantTurn` once the turn completes -- so the
    agent loop is identical whether or not a caller wants live tokens. Falls back
    to the same provider resolution and error semantics as :func:`provider_chat`.
    """
    desc = get_provider(provider)
    if desc is None:
        raise ProviderError(f"unknown provider {provider!r}", reason="unknown_provider")
    env = env if env is not None else os.environ
    if desc.slug == "bedrock":
        from puppetmaster.bedrock import bedrock_chat

        return bedrock_chat(
            model=model,
            messages=messages,
            tools=tools,
            extra=dict(extra or {}),
            api_key=api_key,
            base_url=base_url or resolve_base_url(desc, env),
            timeout=timeout,
            env=env,
            on_delta=on_delta,
        )
    key = api_key if api_key is not None else resolve_api_key(desc, env)
    if key is None and not desc.keyless:
        raise ProviderError(
            f"no API key for provider {desc.slug!r} "
            f"(set one of {', '.join(desc.api_key_env_vars)})",
            reason="not_authenticated",
        )
    url = (base_url or resolve_base_url(desc, env)).rstrip("/")
    extra = dict(extra or {})
    if desc.slug == "opencode-go":
        return _opencode_go_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers),
            timeout=timeout, on_delta=on_delta, stream=True,
        )
    if desc.slug == "openai-codex":
        return _openai_codex_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers),
            timeout=timeout, on_delta=on_delta,
        )
    if desc.wire == "anthropic":
        return _anthropic_chat_stream(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers),
            timeout=timeout, on_delta=on_delta,
        )
    return _openai_chat_stream(
        base_url=url, api_key=key, model=model, messages=messages,
        tools=tools, extra=extra, headers=dict(desc.default_headers),
        timeout=timeout, on_delta=on_delta,
    )


def provider_chat(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Mapping[str, str]] = None,
) -> AssistantTurn:
    """Call ``provider`` directly and return a normalized :class:`AssistantTurn`.

    Resolves the key/base URL from the provider descriptor (unless overridden),
    dispatches on the provider's wire protocol, and normalizes the response so
    the caller never branches on provider. Raises :class:`ProviderError` on any
    HTTP/transport/parse failure. ``provider=bedrock`` uses the stdlib Bedrock
    Converse client (bearer or SigV4) — never Anthropic ``x-api-key``.
    ``provider=opencode-go`` selects chat/completions, /messages, or
    /responses per the published Go endpoint table for the model.
    ``provider=openai-codex`` always uses ChatGPT Codex Responses SSE
    (``OPENAI_CODEX_TOKEN``; non-stream creates are rejected by the backend).
    """
    desc = get_provider(provider)
    if desc is None:
        raise ProviderError(f"unknown provider {provider!r}", reason="unknown_provider")
    env = env if env is not None else os.environ
    if desc.slug == "bedrock":
        from puppetmaster.bedrock import bedrock_chat

        return bedrock_chat(
            model=model,
            messages=messages,
            tools=tools,
            extra=dict(extra or {}),
            api_key=api_key,
            base_url=base_url or resolve_base_url(desc, env),
            timeout=timeout,
            env=env,
        )
    key = api_key if api_key is not None else resolve_api_key(desc, env)
    if key is None and not desc.keyless:
        raise ProviderError(
            f"no API key for provider {desc.slug!r} "
            f"(set one of {', '.join(desc.api_key_env_vars)})",
            reason="not_authenticated",
        )
    url = (base_url or resolve_base_url(desc, env)).rstrip("/")
    extra = dict(extra or {})
    if desc.slug == "opencode-go":
        return _opencode_go_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers),
            timeout=timeout,
        )
    if desc.slug == "openai-codex":
        return _openai_codex_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers),
            timeout=timeout,
        )
    if desc.wire == "anthropic":
        return _anthropic_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers), timeout=timeout,
        )
    return _openai_chat(
        base_url=url, api_key=key, model=model, messages=messages,
        tools=tools, extra=extra, headers=dict(desc.default_headers), timeout=timeout,
    )
