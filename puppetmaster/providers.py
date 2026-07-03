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
  every OpenAI-compatible endpoint (OpenRouter, xAI, DeepSeek, Groq, Mistral,
  Together, Nous, and local Ollama / LM Studio), and Google Gemini via its
  OpenAI-compatible endpoint -- each is just a different ``base_url`` + key.
* ``"anthropic"`` -- Anthropic's native ``/v1/messages`` shape (``tool_use``
  content blocks), which is different enough on the wire to warrant its own
  client.

Both clients are normalized behind :func:`provider_chat`, which returns an
:class:`AssistantTurn`, so the agentic worker loop never branches on provider.
"""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


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
    """
    desc = get_provider(provider)
    if desc is None:
        return []
    env = env if env is not None else os.environ
    keys: list[str] = []
    seen: set[str] = set()
    for base in desc.api_key_env_vars:
        for name in _numbered_env_names(base):
            value = env.get(name)
            if value and value.strip() and value.strip() not in seen:
                seen.add(value.strip())
                keys.append(value.strip())
    return keys


def resolve_base_url(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> str:
    """The provider base URL, honoring its override env var, trailing-slash trimmed."""
    env = env if env is not None else os.environ
    override = env.get(desc.base_url_env_var) if desc.base_url_env_var else None
    return (override or desc.base_url).rstrip("/")


def is_available(
    desc: ProviderDescriptor, env: Optional[Mapping[str, str]] = None
) -> bool:
    """True when this provider can actually be called with the current env.

    A keyed provider needs one of its API-key env vars set. A keyless local
    provider needs one of its presence env vars set (so we never route to a
    local server the user hasn't opted into).
    """
    env = env if env is not None else os.environ
    if desc.keyless:
        return any(env.get(name, "").strip() for name in desc.presence_env_vars)
    return resolve_api_key(desc, env) is not None


def available_providers(env: Optional[Mapping[str, str]] = None) -> set[str]:
    """The set of provider slugs that have a usable credential/endpoint now.

    This is the standalone analogue of Hermes's ``available_hermes_providers``:
    the router uses it to drop any direct-API model whose provider can't be
    reached, so a fresh install offers exactly the providers the user's keys
    unlock -- nothing more.
    """
    env = env if env is not None else os.environ
    return {slug for slug, desc in PROVIDER_REGISTRY.items() if is_available(desc, env)}


# --- normalized chat client -------------------------------------------------

@dataclass
class AssistantTurn:
    """One model turn, normalized across wire protocols.

    ``tool_calls`` is a list of ``{"id", "name", "arguments"}`` where
    ``arguments`` is a parsed dict. ``text`` is the assistant's prose (may be
    empty when the turn is purely tool calls).
    """

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class ProviderError(Exception):
    """A provider HTTP/transport failure, carrying a classifiable reason."""

    def __init__(self, message: str, *, reason: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.reason = reason
        self.status = status
        self.body = body


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
            f"HTTP {exc.code}", reason=f"http_status:{exc.code}", status=exc.code, body=err_body
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderError("request timed out", reason="timeout") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc), reason="network_error") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("malformed response", reason="malformed_response", body=raw[:8000]) from exc


def _openai_chat(
    *, base_url: str, api_key: Optional[str], model: str, messages: list[dict],
    tools: Optional[list[dict]], extra: dict, headers: dict, timeout: int,
) -> AssistantTurn:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    extra = dict(extra)
    force_tool = extra.pop("force_tool", None)
    body.update(extra)
    if force_tool and tools:
        body["tool_choice"] = {"type": "function", "function": {"name": str(force_tool)}}
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
    return AssistantTurn(
        text=str(message.get("content") or "").strip(),
        tool_calls=tool_calls,
        finish_reason=str(finish or ""),
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        raw=data,
    )


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
            f"HTTP {exc.code}", reason=f"http_status:{exc.code}", status=exc.code, body=err_body
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
    body: dict[str, Any] = {
        "model": model, "messages": messages,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    if force_tool and tools:
        body["tool_choice"] = {"type": "function", "function": {"name": str(force_tool)}}
    body.update(extra)
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = _open_stream(
        f"{base_url}/chat/completions",
        headers={"User-Agent": "puppetmaster-agentic", **auth, **headers},
        body=body, timeout=timeout,
    )
    text_parts: list[str] = []
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
                if reasoning and on_delta:
                    on_delta("reasoning", str(reasoning))
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
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        raw={},
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
    try:
        for payload in _iter_sse_data(response):
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "message_start":
                prompt_tokens = int(((event.get("message") or {}).get("usage") or {}).get("input_tokens") or 0)
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
        },
        raw={},
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
    key = api_key if api_key is not None else resolve_api_key(desc, env)
    if key is None and not desc.keyless:
        raise ProviderError(
            f"no API key for provider {desc.slug!r} "
            f"(set one of {', '.join(desc.api_key_env_vars)})",
            reason="not_authenticated",
        )
    url = (base_url or resolve_base_url(desc, env)).rstrip("/")
    extra = dict(extra or {})
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
    HTTP/transport/parse failure.
    """
    desc = get_provider(provider)
    if desc is None:
        raise ProviderError(f"unknown provider {provider!r}", reason="unknown_provider")
    env = env if env is not None else os.environ
    key = api_key if api_key is not None else resolve_api_key(desc, env)
    if key is None and not desc.keyless:
        raise ProviderError(
            f"no API key for provider {desc.slug!r} "
            f"(set one of {', '.join(desc.api_key_env_vars)})",
            reason="not_authenticated",
        )
    url = (base_url or resolve_base_url(desc, env)).rstrip("/")
    extra = dict(extra or {})
    if desc.wire == "anthropic":
        return _anthropic_chat(
            base_url=url, api_key=key, model=model, messages=messages,
            tools=tools, extra=extra, headers=dict(desc.default_headers), timeout=timeout,
        )
    return _openai_chat(
        base_url=url, api_key=key, model=model, messages=messages,
        tools=tools, extra=extra, headers=dict(desc.default_headers), timeout=timeout,
    )
