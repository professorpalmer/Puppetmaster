"""AWS Bedrock client for agentic workers (stdlib only — no boto3).

IAM/SigV4 (or optional ``AWS_BEARER_TOKEN_BEDROCK``) against the caller's
account — not a single Bedrock API-key catalog. Chat uses the unified
**Converse** API so Anthropic, Amazon Nova, Meta, Mistral, DeepSeek, Z.AI,
Moonshot, Qwen, MiniMax, etc. share one request/response shape. Model ids are
account-specific; :func:`list_chat_model_ids` discovers what this credential
can see via ``ListFoundationModels`` + ``ListInferenceProfiles``.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import hmac
import json
import os
import socket
import struct
import urllib.error
import urllib.parse
import urllib.request
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional


_MISSING_CREDS_MSG = (
    "AWS Bedrock credentials not found — set AWS_PROFILE (or use the default "
    "profile in ~/.aws), AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or "
    "AWS_BEARER_TOKEN_BEDROCK."
)

_SHORT_MODEL_MSG = (
    "is not a Bedrock model id. Use a Bedrock foundation or inference profile id "
    "(e.g. amazon.nova-micro-v1:0, deepseek.v3.2, "
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0) or an "
    "arn:aws:bedrock:... ARN — short names like claude-opus-4-8 are rejected."
)

_NON_CHAT_MARKERS = (
    "embedding",
    "embed",
    "rerank",
    "canvas",
    "reel",
    "sonic",
    "tts",
    "whisper",
    "transcribe",
    "moderation",
    "guard",
    "image",
    "video",
    "speech",
    "music",
    "upscale",
    "stable-diffusion",
    "stability.",
    "twelvelabs",
    "pegasus",
    "marengo",
)

_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"
# InvokeModel requires application/json for Anthropic Claude (not the
# amazon.bedrock.anthropic.messages-v1+json media type).
_ANTHROPIC_MESSAGES_CONTENT_TYPE = "application/json"


@dataclass(frozen=True)
class BedrockCredentials:
    """Resolved AWS auth for a Bedrock InvokeModel call.

    ``kind`` is ``bearer``, ``access_key``, or ``profile`` (profile may still
    carry loaded access keys when ``~/.aws/credentials`` is parseable).
    """

    kind: str
    bearer_token: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    profile: Optional[str] = None
    evidence: tuple = ()


def resolve_bedrock_region(env: Optional[Mapping[str, str]] = None) -> str:
    """AWS region for bedrock-runtime, defaulting to ``us-east-1``."""
    env = env if env is not None else os.environ
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_REGION"):
        value = (env.get(name) or "").strip()
        if value:
            return value
    return "us-east-1"


def bedrock_runtime_base_url(region: str) -> str:
    """``https://bedrock-runtime.{region}.amazonaws.com``."""
    return f"https://bedrock-runtime.{region}.amazonaws.com"


def missing_bedrock_credentials_message() -> str:
    """Actionable remediation when no AWS creds are visible."""
    return _MISSING_CREDS_MSG


def bedrock_credentials_present(
    env: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[Path] = None,
) -> bool:
    """True when Bedrock credential *presence* is visible (not invoke-verified)."""
    return resolve_bedrock_credentials(env, home=home) is not None


def short_bedrock_model_message(model: object) -> str:
    """Actionable remediation when a short (non-Bedrock) model id is used."""
    return f"{model!r} {_SHORT_MODEL_MSG}"


def _aws_home(env: Mapping[str, str]) -> Path:
    override = (env.get("HOME") or env.get("USERPROFILE") or "").strip()
    if override:
        return Path(override)
    return Path.home()


def _load_shared_credentials(
    profile: str, home: Path
) -> "Optional[tuple[str, str, Optional[str]]]":
    """Read access key / secret / session from ``~/.aws/credentials`` (+ config).

    Returns ``(access_key_id, secret_access_key, session_token)`` or ``None``
    when the profile is missing or lacks static keys (SSO / credential_process).
    """
    cred_path = home / ".aws" / "credentials"
    if not cred_path.is_file():
        return None
    parser = ConfigParser()
    try:
        parser.read(str(cred_path))
    except OSError:
        return None
    section = profile if parser.has_section(profile) else None
    if section is None and profile == "default" and parser.has_section("default"):
        section = "default"
    if section is None:
        return None
    access = (parser.get(section, "aws_access_key_id", fallback="") or "").strip()
    secret = (parser.get(section, "aws_secret_access_key", fallback="") or "").strip()
    if not access or not secret:
        return None
    token = (parser.get(section, "aws_session_token", fallback="") or "").strip() or None
    return access, secret, token


def resolve_bedrock_credentials(
    env: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[Path] = None,
) -> Optional[BedrockCredentials]:
    """Presence-based AWS auth for Bedrock (no boto3, no network).

    Order: bearer token → access key + secret (+ optional session) → profile /
    ``~/.aws`` files. Profile entries with static keys are loaded for SigV4;
    profile/config presence alone still returns a credential handle so health
    checks stay honest (SSO may still fail at invoke time).

    The ``~/.aws`` file probe runs only for the live process env, an explicit
    ``home=``, or when ``HOME`` / ``USERPROFILE`` is present in ``env`` — so a
    hermetic ``available_providers({})`` does not inherit the developer's AWS
    config via ``Path.home()``.
    """
    env = env if env is not None else os.environ
    home_explicit = home is not None
    home_path = home if home is not None else _aws_home(env)

    bearer = (env.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    if bearer:
        return BedrockCredentials(
            kind="bearer",
            bearer_token=bearer,
            evidence=("aws_credentials:bearer_token",),
        )

    access = (env.get("AWS_ACCESS_KEY_ID") or "").strip()
    secret = (env.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if access and secret:
        session = (env.get("AWS_SESSION_TOKEN") or "").strip() or None
        return BedrockCredentials(
            kind="access_key",
            access_key_id=access,
            secret_access_key=secret,
            session_token=session,
            evidence=("aws_credentials:env_keys",),
        )

    profile = (env.get("AWS_PROFILE") or "").strip()
    allow_file_probe = (
        home_explicit
        or env is os.environ
        or bool((env.get("HOME") or env.get("USERPROFILE") or "").strip())
    )
    if profile:
        if allow_file_probe:
            loaded = _load_shared_credentials(profile, home_path)
            if loaded is not None:
                return BedrockCredentials(
                    kind="access_key",
                    access_key_id=loaded[0],
                    secret_access_key=loaded[1],
                    session_token=loaded[2],
                    profile=profile,
                    evidence=("aws_credentials:profile",),
                )
        return BedrockCredentials(
            kind="profile",
            profile=profile,
            evidence=("aws_credentials:profile",),
        )

    if not allow_file_probe:
        return None

    aws_dir = home_path / ".aws"
    for name in ("credentials", "config"):
        path = aws_dir / name
        if path.is_file() and path.stat().st_size > 0:
            loaded = _load_shared_credentials("default", home_path)
            if loaded is not None:
                return BedrockCredentials(
                    kind="access_key",
                    access_key_id=loaded[0],
                    secret_access_key=loaded[1],
                    session_token=loaded[2],
                    profile="default",
                    evidence=("aws_credentials:config_file",),
                )
            return BedrockCredentials(
                kind="profile",
                profile="default",
                evidence=("aws_credentials:config_file",),
            )
    return None


def require_bedrock_model_id(model: object) -> str:
    """Return a Bedrock-shaped model id or raise :class:`ProviderError`."""
    from puppetmaster.adapters.claude_code import is_bedrock_model_id
    from puppetmaster.providers import ProviderError

    text = str(model or "").strip()
    if not is_bedrock_model_id(text):
        raise ProviderError(
            short_bedrock_model_message(model),
            reason="invalid_model",
        )
    return text


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sigv4_signing_key(
    secret_key: str, date_stamp: str, region: str, service: str
) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


def sigv4_sign_headers(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    region: str,
    service: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: Optional[str] = None,
    amz_date: Optional[str] = None,
) -> dict:
    """Return request headers including AWS Signature Version 4 Authorization.

    ``amz_date`` is injectable for hermetic tests (``YYYYMMDDTHHMMSSZ``).
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    # SigV4 canonical URI double-encodes path segments (``:`` -> ``%3A`` on the
    # wire becomes ``%253A`` in the canonical request). ``quote(..., safe="/")``
    # re-encodes any ``%`` already present from ``invoke_model_url``.
    raw_path = parsed.path or "/"
    canonical_uri = urllib.parse.quote(raw_path, safe="/")
    canonical_query = parsed.query or ""

    if amz_date is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_date[:8]

    payload_hash = hashlib.sha256(body).hexdigest()
    signed: dict[str, str] = {
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }
    for key, value in headers.items():
        signed[key.lower()] = value
    if session_token:
        signed["x-amz-security-token"] = session_token

    signed_header_names = ";".join(sorted(signed))
    canonical_headers = "".join(
        f"{name}:{signed[name]}\n" for name in sorted(signed)
    )
    canonical_request = "\n".join([
        method.upper(),
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_header_names,
        payload_hash,
    ])
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signing_key = _sigv4_signing_key(secret_access_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_header_names}, Signature={signature}"
    )

    out = {key: value for key, value in headers.items()}
    out["Host"] = host
    out["X-Amz-Date"] = amz_date
    out["X-Amz-Content-Sha256"] = payload_hash
    out["Authorization"] = authorization
    if session_token:
        out["X-Amz-Security-Token"] = session_token
    return out


def invoke_model_url(base_url: str, model_id: str) -> str:
    """Bedrock Runtime ``InvokeModel`` URL for ``model_id`` (legacy Anthropic path)."""
    encoded = urllib.parse.quote(model_id, safe="")
    return f"{base_url.rstrip('/')}/model/{encoded}/invoke"


def converse_model_url(base_url: str, model_id: str) -> str:
    """Bedrock Runtime ``Converse`` URL for ``model_id``."""
    encoded = urllib.parse.quote(model_id, safe="")
    return f"{base_url.rstrip('/')}/model/{encoded}/converse"


def converse_stream_model_url(base_url: str, model_id: str) -> str:
    """Bedrock Runtime ``ConverseStream`` URL for ``model_id``."""
    encoded = urllib.parse.quote(model_id, safe="")
    return f"{base_url.rstrip('/')}/model/{encoded}/converse-stream"


def bedrock_control_base_url(region: str) -> str:
    """``https://bedrock.{region}.amazonaws.com`` (ListFoundationModels, etc.)."""
    return f"https://bedrock.{region}.amazonaws.com"


def _is_chat_capable_model_id(model_id: str) -> bool:
    mid = (model_id or "").lower()
    if not mid:
        return False
    return not any(marker in mid for marker in _NON_CHAT_MARKERS)


def _content_to_converse_blocks(content: Any) -> list[dict]:
    """Map OpenAI-ish message content to Converse content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, list):
        blocks: list[dict] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    blocks.append({"text": part})
            elif isinstance(part, dict):
                if "text" in part and part.get("text") is not None:
                    blocks.append({"text": str(part["text"])})
                elif part.get("type") == "text":
                    blocks.append({"text": str(part.get("text") or "")})
                elif "toolUse" in part or "toolResult" in part:
                    blocks.append(part)
        return blocks
    return [{"text": str(content)}]


def _openai_tools_to_converse(tools: Optional[list[dict]]) -> Optional[dict]:
    """OpenAI function-tools → Converse ``toolConfig``."""
    if not tools:
        return None
    specs: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            continue
        name = (fn.get("name") or "").strip()
        if not name:
            continue
        spec: dict[str, Any] = {
            "toolSpec": {
                "name": name,
                "inputSchema": {"json": fn.get("parameters") or {"type": "object"}},
            }
        }
        desc = (fn.get("description") or "").strip()
        if desc:
            spec["toolSpec"]["description"] = desc
        specs.append(spec)
    if not specs:
        return None
    return {"tools": specs}


def _bedrock_cache_point() -> dict:
    """Converse ``cachePoint`` block (default TTL; Bedrock stamps ttl in usage)."""
    return {"cachePoint": {"type": "default"}}


def apply_bedrock_converse_cache_points(body: dict) -> dict:
    """Stamp Converse ``cachePoint`` breakpoints for prompt-cache reads.

    Mirrors Anthropic ``cache_control`` placement (system + tools + last two
    messages) on the Bedrock Converse schema so Claude/Nova/etc. write and hit
    prompt caches. Honors ``PUPPETMASTER_PROMPT_CACHE`` (same kill switch as
    Anthropic/OpenAI-wire). Never raises — a failed stamp returns ``body``.
    """
    try:
        from puppetmaster.providers import _prompt_cache_enabled
    except Exception:
        return body
    if not _prompt_cache_enabled():
        return body
    try:
        out = copy.deepcopy(body)
        system = out.get("system")
        if isinstance(system, list) and system:
            if not any(isinstance(b, dict) and "cachePoint" in b for b in system):
                system.append(_bedrock_cache_point())

        tool_config = out.get("toolConfig")
        if isinstance(tool_config, dict) and tool_config.get("tools"):
            tool_config["cachePoint"] = {"type": "default"}

        messages = out.get("messages")
        if isinstance(messages, list) and messages:
            idxs = [len(messages) - 1]
            if len(messages) > 1:
                idxs.insert(0, len(messages) - 2)
            for idx in idxs:
                msg = messages[idx]
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list) or not content:
                    continue
                if any(isinstance(b, dict) and "cachePoint" in b for b in content):
                    continue
                content.append(_bedrock_cache_point())
        return out
    except Exception:
        return body


def build_converse_body(
    *,
    messages: list[dict],
    tools: Optional[list[dict]],
    extra: dict,
) -> dict:
    """Converse JSON body from OpenAI-shaped messages + tools."""
    system_parts = [
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "system" and m.get("content")
    ]
    convo: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            tool_use_id = str(msg.get("tool_call_id") or msg.get("id") or "")
            result_content = _content_to_converse_blocks(msg.get("content"))
            if not result_content:
                result_content = [{"text": str(msg.get("content") or "")}]
            convo.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": result_content,
                    }
                }],
            })
            continue
        if role == "assistant":
            blocks = _content_to_converse_blocks(msg.get("content"))
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = call.get("name") or fn.get("name") or ""
                raw_args = call.get("arguments")
                if raw_args is None:
                    raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args_obj = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        args_obj = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    args_obj = raw_args
                else:
                    args_obj = {}
                blocks.append({
                    "toolUse": {
                        "toolUseId": call.get("id") or "",
                        "name": name,
                        "input": args_obj,
                    }
                })
            if not blocks:
                blocks = [{"text": ""}]
            convo.append({"role": "assistant", "content": blocks})
            continue
        # user (default)
        blocks = _content_to_converse_blocks(msg.get("content"))
        if not blocks:
            blocks = [{"text": ""}]
        convo.append({"role": "user", "content": blocks})

    body: dict[str, Any] = {
        "messages": convo,
        "inferenceConfig": {
            "maxTokens": int(
                extra.get("max_tokens")
                or extra.get("max_completion_tokens")
                or 4096
            ),
        },
    }
    if system_parts:
        body["system"] = [{"text": "\n\n".join(system_parts)}]
    tool_config = _openai_tools_to_converse(tools)
    if tool_config:
        force_tool = extra.get("force_tool")
        if force_tool:
            tool_config["toolChoice"] = {"tool": {"name": str(force_tool)}}
        body["toolConfig"] = tool_config
    if "temperature" in extra:
        body["inferenceConfig"]["temperature"] = float(extra["temperature"])
    if "top_p" in extra:
        body["inferenceConfig"]["topP"] = float(extra["top_p"])
    # Pass-through for Claude extended thinking / other vendor fields.
    amrf = extra.get("additionalModelRequestFields")
    if isinstance(amrf, dict) and amrf:
        body["additionalModelRequestFields"] = dict(amrf)
    return apply_bedrock_converse_cache_points(body)


def build_anthropic_invoke_body(
    *,
    messages: list[dict],
    tools: Optional[list[dict]],
    extra: dict,
) -> dict:
    """Anthropic Messages JSON body for Bedrock ``InvokeModel`` (no ``model``)."""
    from puppetmaster.providers import _to_anthropic_messages, _to_anthropic_tool

    system_parts = [
        m["content"] for m in messages if m.get("role") == "system" and m.get("content")
    ]
    convo = [m for m in messages if m.get("role") != "system"]
    body: dict[str, Any] = {
        "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
        "messages": _to_anthropic_messages(convo),
        "max_tokens": int(
            extra.get("max_tokens") or extra.get("max_completion_tokens") or 4096
        ),
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
    return body


def _post_bedrock(
    url: str,
    *,
    headers: dict,
    body: dict,
    timeout: int,
) -> dict:
    from puppetmaster.providers import ProviderError

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
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
            f"HTTP {exc.code}",
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
        raise ProviderError(
            "malformed response", reason="malformed_response", body=raw[:8000]
        ) from exc


def _get_bedrock(
    url: str,
    *,
    headers: dict,
    timeout: int,
) -> dict:
    from puppetmaster.providers import ProviderError

    request = urllib.request.Request(url, headers=headers, method="GET")
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
            f"HTTP {exc.code}",
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
        raise ProviderError(
            "malformed response", reason="malformed_response", body=raw[:8000]
        ) from exc


def _auth_headers_for_request(
    *,
    method: str,
    url: str,
    body_bytes: bytes,
    region: str,
    creds: BedrockCredentials,
    content_type: Optional[str] = "application/json",
    accept: str = "application/json",
    amz_date: Optional[str] = None,
) -> dict:
    """Bearer or SigV4 headers for a Bedrock control/runtime request."""
    unsigned: dict[str, str] = {
        "Accept": accept,
        "User-Agent": "puppetmaster-agentic",
    }
    if content_type:
        unsigned["Content-Type"] = content_type
    if creds.kind == "bearer" and creds.bearer_token:
        unsigned["Authorization"] = f"Bearer {creds.bearer_token}"
        return unsigned
    if creds.access_key_id and creds.secret_access_key:
        return sigv4_sign_headers(
            method=method,
            url=url,
            headers=unsigned,
            body=body_bytes,
            region=region,
            service="bedrock",
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
            amz_date=amz_date,
        )
    from puppetmaster.providers import ProviderError

    raise ProviderError(
        missing_bedrock_credentials_message()
        + " (AWS_PROFILE / ~/.aws is present but has no static access keys "
        "Puppetmaster can load without boto3 — export "
        "AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID.)",
        reason="not_authenticated",
    )


def _resolve_call_credentials(
    *,
    api_key: Optional[str],
    env: Mapping[str, str],
) -> BedrockCredentials:
    from puppetmaster.providers import ProviderError

    if api_key and str(api_key).strip():
        return BedrockCredentials(
            kind="bearer",
            bearer_token=str(api_key).strip(),
            evidence=("aws_credentials:bearer_token",),
        )
    creds = resolve_bedrock_credentials(env)
    if creds is None:
        raise ProviderError(
            missing_bedrock_credentials_message(),
            reason="not_authenticated",
        )
    return creds


def list_chat_model_ids(
    *,
    env: Optional[Mapping[str, str]] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
    amz_date: Optional[str] = None,
) -> list[str]:
    """Discover chat-capable model ids visible to these AWS credentials.

    Unions ``ListFoundationModels`` + ``ListInferenceProfiles`` for the
    resolved region, filters out embed/image/video/speech-only ids, and
    returns a sorted unique list. Account allow-lists and marketplace
    entitlements make this set credential-specific — do not hardcode it.
    """
    env = env if env is not None else os.environ
    region = resolve_bedrock_region(env)
    creds = _resolve_call_credentials(api_key=api_key, env=env)
    control = bedrock_control_base_url(region)
    found: list[str] = []

    foundation_url = f"{control}/foundation-models"
    headers = _auth_headers_for_request(
        method="GET",
        url=foundation_url,
        body_bytes=b"",
        region=region,
        creds=creds,
        content_type=None,
        amz_date=amz_date,
    )
    try:
        data = _get_bedrock(foundation_url, headers=headers, timeout=timeout)
        for row in data.get("modelSummaries") or []:
            if not isinstance(row, dict):
                continue
            mid = (row.get("modelId") or "").strip()
            if not mid or not _is_chat_capable_model_id(mid):
                continue
            outputs = [str(x).upper() for x in (row.get("outputModalities") or [])]
            inputs = [str(x).upper() for x in (row.get("inputModalities") or [])]
            if outputs and "TEXT" not in outputs:
                continue
            if inputs and "TEXT" not in inputs and "SPEECH" in inputs:
                continue
            found.append(mid)
    except Exception:
        pass

    profiles_url = f"{control}/inference-profiles"
    headers = _auth_headers_for_request(
        method="GET",
        url=profiles_url,
        body_bytes=b"",
        region=region,
        creds=creds,
        content_type=None,
        amz_date=amz_date,
    )
    try:
        data = _get_bedrock(profiles_url, headers=headers, timeout=timeout)
        for row in data.get("inferenceProfileSummaries") or []:
            if not isinstance(row, dict):
                continue
            mid = (
                (row.get("inferenceProfileId") or row.get("modelId") or "")
            ).strip()
            if mid and _is_chat_capable_model_id(mid):
                found.append(mid)
    except Exception:
        pass

    # Stable unique order; prefer shorter foundation ids before duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for mid in sorted(found):
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def _assistant_turn_from_anthropic(data: dict):
    from puppetmaster.providers import AssistantTurn

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
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


def _assistant_turn_from_converse(data: dict):
    """Normalize a Converse ``Invoke`` JSON response to ``AssistantTurn``."""
    from puppetmaster.providers import AssistantTurn

    message = ((data.get("output") or {}).get("message") or {})
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block and block.get("text") is not None:
            text_parts.append(str(block.get("text") or ""))
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            tool_calls.append({
                "id": tool_use.get("toolUseId") or "",
                "name": tool_use.get("name") or "",
                "arguments": tool_use.get("input") or {},
            })
    usage = data.get("usage") or {}
    # Converse reports inputTokens as the uncached slice only. Cost meters and
    # Marionette treat tokens_cached as a subset of tokens_in, so fold cache
    # read/write into prompt_tokens (same shape as Anthropic Messages totals).
    uncached_in = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    cache_read = int(
        usage.get("cacheReadInputTokens")
        or usage.get("cacheReadInputTokenCount")
        or 0
    )
    cache_write = int(
        usage.get("cacheWriteInputTokens")
        or usage.get("cacheWriteInputTokenCount")
        or 0
    )
    prompt_tokens = uncached_in + cache_read + cache_write
    completion_tokens = int(
        usage.get("outputTokens") or usage.get("output_tokens") or 0
    )
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        finish_reason=str(data.get("stopReason") or data.get("stop_reason") or ""),
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(
                usage.get("totalTokens") or (prompt_tokens + completion_tokens)
            ),
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
        raw=data,
    )


# AWS Event Stream value type codes (botocore / Smithy binary framing).
_EVENTSTREAM_BOOL_TRUE = 0
_EVENTSTREAM_BOOL_FALSE = 1
_EVENTSTREAM_BYTE = 2
_EVENTSTREAM_SHORT = 3
_EVENTSTREAM_INT = 4
_EVENTSTREAM_LONG = 5
_EVENTSTREAM_BYTES = 6
_EVENTSTREAM_STRING = 7
_EVENTSTREAM_TIMESTAMP = 8
_EVENTSTREAM_UUID = 9


def encode_eventstream_message(
    headers: Mapping[str, Any],
    payload: bytes = b"",
) -> bytes:
    """Encode one AWS Event Stream message (CRC fields left as zero).

    Used by hermetic tests to build ConverseStream bodies without botocore.
    Runtime decoding does not require valid CRC32C.
    """
    header_buf = bytearray()
    for name, value in headers.items():
        name_bytes = str(name).encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(f"eventstream header name too long: {name!r}")
        header_buf.append(len(name_bytes))
        header_buf.extend(name_bytes)
        if isinstance(value, bool):
            header_buf.append(
                _EVENTSTREAM_BOOL_TRUE if value else _EVENTSTREAM_BOOL_FALSE
            )
        elif isinstance(value, int) and not isinstance(value, bool):
            header_buf.append(_EVENTSTREAM_INT)
            header_buf.extend(struct.pack(">i", int(value)))
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
            header_buf.append(_EVENTSTREAM_BYTES)
            header_buf.extend(struct.pack(">H", len(raw)))
            header_buf.extend(raw)
        else:
            raw = str(value).encode("utf-8")
            header_buf.append(_EVENTSTREAM_STRING)
            header_buf.extend(struct.pack(">H", len(raw)))
            header_buf.extend(raw)
    headers_len = len(header_buf)
    total_len = 16 + headers_len + len(payload)
    return (
        struct.pack(">II", total_len, headers_len)
        + b"\x00\x00\x00\x00"
        + bytes(header_buf)
        + payload
        + b"\x00\x00\x00\x00"
    )


def encode_converse_stream_event(event_type: str, body: dict) -> bytes:
    """Encode a ConverseStream ``:event-type`` message with a JSON payload."""
    return encode_eventstream_message(
        {
            ":message-type": "event",
            ":event-type": event_type,
            ":content-type": "application/json",
        },
        json.dumps(body).encode("utf-8"),
    )


def _parse_eventstream_headers(raw: bytes) -> dict[str, Any]:
    """Parse the header block of one AWS Event Stream message."""
    headers: dict[str, Any] = {}
    offset = 0
    length = len(raw)
    while offset < length:
        name_len = raw[offset]
        offset += 1
        name = raw[offset : offset + name_len].decode("utf-8", errors="replace")
        offset += name_len
        if offset >= length:
            break
        value_type = raw[offset]
        offset += 1
        if value_type in (_EVENTSTREAM_BOOL_TRUE, _EVENTSTREAM_BOOL_FALSE):
            headers[name] = value_type == _EVENTSTREAM_BOOL_TRUE
        elif value_type == _EVENTSTREAM_BYTE:
            headers[name] = struct.unpack_from(">b", raw, offset)[0]
            offset += 1
        elif value_type == _EVENTSTREAM_SHORT:
            headers[name] = struct.unpack_from(">h", raw, offset)[0]
            offset += 2
        elif value_type == _EVENTSTREAM_INT:
            headers[name] = struct.unpack_from(">i", raw, offset)[0]
            offset += 4
        elif value_type == _EVENTSTREAM_LONG:
            headers[name] = struct.unpack_from(">q", raw, offset)[0]
            offset += 8
        elif value_type == _EVENTSTREAM_BYTES:
            n = struct.unpack_from(">H", raw, offset)[0]
            offset += 2
            headers[name] = bytes(raw[offset : offset + n])
            offset += n
        elif value_type == _EVENTSTREAM_STRING:
            n = struct.unpack_from(">H", raw, offset)[0]
            offset += 2
            headers[name] = raw[offset : offset + n].decode("utf-8", errors="replace")
            offset += n
        elif value_type == _EVENTSTREAM_TIMESTAMP:
            headers[name] = struct.unpack_from(">q", raw, offset)[0]
            offset += 8
        elif value_type == _EVENTSTREAM_UUID:
            headers[name] = bytes(raw[offset : offset + 16])
            offset += 16
        else:
            break
    return headers


def _iter_eventstream_messages(byte_source) -> Iterator[tuple[dict[str, Any], bytes]]:
    """Yield ``(headers, payload)`` frames from an AWS Event Stream body.

    CRC32C prelude/message checksums are not validated (stdlib has no CRC32C);
    TLS already covers transport integrity for live calls.
    """
    pending = bytearray()

    def _pull(n: int) -> Optional[bytes]:
        while len(pending) < n:
            chunk = byte_source.read(max(8192, n - len(pending)))
            if not chunk:
                return None
            pending.extend(chunk)
        out = bytes(pending[:n])
        del pending[:n]
        return out

    while True:
        prelude = _pull(12)
        if prelude is None:
            if pending:
                from puppetmaster.providers import ProviderError

                raise ProviderError(
                    "truncated Bedrock event stream",
                    reason="malformed_response",
                )
            return
        total_len, headers_len = struct.unpack(">II", prelude[:8])
        if total_len < 16 or headers_len < 0 or headers_len > total_len - 16:
            from puppetmaster.providers import ProviderError

            raise ProviderError(
                "malformed Bedrock event stream framing",
                reason="malformed_response",
            )
        rest = _pull(total_len - 12)
        if rest is None:
            from puppetmaster.providers import ProviderError

            raise ProviderError(
                "truncated Bedrock event stream",
                reason="malformed_response",
            )
        headers_raw = rest[:headers_len]
        payload = rest[headers_len : len(rest) - 4]
        yield _parse_eventstream_headers(headers_raw), payload


def _open_bedrock_event_stream(
    url: str,
    *,
    headers: dict,
    body: dict,
    timeout: int,
):
    """POST ``body`` and return the raw streaming response for eventstream iteration."""
    from puppetmaster.providers import ProviderError

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
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
            f"HTTP {exc.code}",
            reason=f"http_status:{exc.code}",
            status=exc.code,
            body=err_body,
        ) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderError("request timed out", reason="timeout") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc), reason="network_error") from exc


def _iter_converse_stream_events(byte_source) -> Iterator[dict]:
    """Yield ConverseStream events as ``{event_type: payload_dict}`` maps."""
    from puppetmaster.providers import ProviderError

    for headers, payload in _iter_eventstream_messages(byte_source):
        message_type = str(headers.get(":message-type") or "event")
        if message_type == "exception":
            exc_type = str(headers.get(":exception-type") or "Exception")
            normalized_type = exc_type.replace("_", "").replace("-", "").lower()
            detail = ""
            if payload:
                try:
                    parsed = json.loads(payload.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    detail = str(parsed.get("message") or parsed.get("Message") or "")
                if not detail:
                    detail = payload.decode("utf-8", errors="replace")[:8000]
            if "accessdenied" in normalized_type or "unauthorized" in normalized_type:
                status = 403
            elif "throttl" in normalized_type:
                status = 429
            elif any(
                marker in normalized_type
                for marker in ("internalserver", "serviceunavailable", "modelstreamerror")
            ):
                status = 503
            elif "validation" in normalized_type:
                status = 400
            else:
                status = None
            raise ProviderError(
                detail or exc_type,
                reason=f"http_status:{status}" if status is not None else "provider_error",
                status=status,
                body=detail[:8000] if detail else None,
            )
        if message_type != "event":
            continue
        event_type = str(headers.get(":event-type") or "")
        if not event_type:
            continue
        if not payload:
            yield {event_type: {}}
            continue
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "malformed ConverseStream event",
                reason="malformed_response",
                body=payload.decode("utf-8", errors="replace")[:8000],
            ) from exc
        if not isinstance(data, dict):
            data = {}
        yield {event_type: data}


def _assistant_turn_from_converse_stream(
    events: Iterator[dict],
    *,
    on_delta: Optional[Callable[[str, str], None]] = None,
):
    """Fold ConverseStream events into an ``AssistantTurn``, firing deltas live."""
    blocks: dict[int, dict] = {}
    stop_reason = ""
    usage: dict = {}
    raw_events: list[dict] = []

    for event in events:
        raw_events.append(event)
        if "contentBlockStart" in event:
            start_evt = event["contentBlockStart"] or {}
            idx = int(start_evt.get("contentBlockIndex") or 0)
            start = start_evt.get("start") or {}
            tool_use = start.get("toolUse")
            if isinstance(tool_use, dict):
                blocks[idx] = {
                    "kind": "tool",
                    "id": str(tool_use.get("toolUseId") or ""),
                    "name": str(tool_use.get("name") or ""),
                    "args": "",
                }
            continue
        if "contentBlockDelta" in event:
            delta_evt = event["contentBlockDelta"] or {}
            idx = int(delta_evt.get("contentBlockIndex") or 0)
            delta = delta_evt.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if "text" in delta and delta.get("text") is not None:
                piece = str(delta.get("text") or "")
                slot = blocks.setdefault(idx, {"kind": "text", "text": ""})
                if slot.get("kind") != "text":
                    slot = {"kind": "text", "text": ""}
                    blocks[idx] = slot
                slot["text"] = str(slot.get("text") or "") + piece
                if on_delta and piece:
                    on_delta("text", piece)
            reasoning = delta.get("reasoningContent")
            if isinstance(reasoning, dict):
                thought = reasoning.get("text")
                if thought and on_delta:
                    on_delta("reasoning", str(thought))
            tool_delta = delta.get("toolUse")
            if isinstance(tool_delta, dict) and "input" in tool_delta:
                slot = blocks.setdefault(
                    idx, {"kind": "tool", "id": "", "name": "", "args": ""}
                )
                if slot.get("kind") != "tool":
                    slot = {"kind": "tool", "id": "", "name": "", "args": ""}
                    blocks[idx] = slot
                slot["args"] = str(slot.get("args") or "") + str(
                    tool_delta.get("input") or ""
                )
            continue
        if "messageStop" in event:
            stop_reason = str(
                (event.get("messageStop") or {}).get("stopReason") or stop_reason
            )
            continue
        if "metadata" in event:
            meta = event.get("metadata") or {}
            if isinstance(meta.get("usage"), dict):
                usage = meta["usage"]
            continue

    content: list[dict] = []
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot.get("kind") == "text":
            content.append({"text": str(slot.get("text") or "")})
            continue
        if slot.get("kind") != "tool":
            continue
        raw_args = str(slot.get("args") or "")
        try:
            args_obj = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args_obj = {"__raw__": raw_args}
        content.append({
            "toolUse": {
                "toolUseId": slot.get("id") or "",
                "name": slot.get("name") or "",
                "input": args_obj,
            }
        })

    data = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": usage,
        "streamEvents": raw_events,
    }
    return _assistant_turn_from_converse(data)


def _note_bedrock_invoke_outcome(
    env: Mapping[str, str],
    creds: BedrockCredentials,
    *,
    error: Optional[BaseException] = None,
) -> None:
    """Best-effort invoke-health update; never masks the caller error."""
    try:
        from puppetmaster.provider_health import (
            record_bedrock_invoke_failure,
            record_bedrock_invoke_success,
        )

        if error is None:
            record_bedrock_invoke_success(env, creds=creds)
            return
        record_bedrock_invoke_failure(env, creds=creds, error=error)
    except Exception:
        return


def bedrock_chat(
    *,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Mapping[str, str]] = None,
    on_delta: Optional[Callable[[str, str], None]] = None,
    amz_date: Optional[str] = None,
) -> Any:
    """Call Bedrock ``Converse`` and return a normalized ``AssistantTurn``.

    Works across providers on the caller's allow-list (Claude, Nova, Llama,
    DeepSeek, GLM, Moonshot, …) via one schema. Auth is IAM SigV4 or optional
    bearer. When ``on_delta`` is provided, delegates to
    :func:`bedrock_chat_stream` (``ConverseStream``) so text/reasoning deltas
    fire incrementally; otherwise posts non-stream ``Converse``.

    Successful runtime invokes mark Bedrock invoke-health verified; terminal
    auth failures (401 / auth-signal 403) mark it denied for the credential
    fingerprint. Catalog list calls never go through this path.
    """
    if on_delta is not None:
        return bedrock_chat_stream(
            model=model,
            messages=messages,
            tools=tools,
            extra=extra,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            env=env,
            on_delta=on_delta,
            amz_date=amz_date,
        )
    env = env if env is not None else os.environ
    region = resolve_bedrock_region(env)
    url_base = (base_url or bedrock_runtime_base_url(region)).rstrip("/")
    model_id = require_bedrock_model_id(model)
    extra = dict(extra or {})
    creds = _resolve_call_credentials(api_key=api_key, env=env)

    body = build_converse_body(messages=messages, tools=tools, extra=extra)
    url = converse_model_url(url_base, model_id)
    payload = json.dumps(body).encode("utf-8")
    headers = _auth_headers_for_request(
        method="POST",
        url=url,
        body_bytes=payload,
        region=region,
        creds=creds,
        content_type="application/json",
        amz_date=amz_date,
    )
    try:
        data = _post_bedrock(url, headers=headers, body=body, timeout=timeout)
        turn = _assistant_turn_from_converse(data)
    except Exception as exc:
        _note_bedrock_invoke_outcome(env, creds, error=exc)
        raise
    _note_bedrock_invoke_outcome(env, creds)
    return turn


def bedrock_chat_stream(
    *,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    extra: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Mapping[str, str]] = None,
    on_delta: Optional[Callable[[str, str], None]] = None,
    amz_date: Optional[str] = None,
) -> Any:
    """Call Bedrock ``ConverseStream`` and return a normalized ``AssistantTurn``.

    POSTs the same Converse body as :func:`bedrock_chat` to
    ``.../converse-stream``, parses the AWS Event Stream response, and invokes
    ``on_delta(kind, text)`` for each text/reasoning chunk (``kind`` is
    ``"text"`` or ``"reasoning"``). Cache read/write usage is folded into
    the final turn the same way as non-stream Converse.
    """
    env = env if env is not None else os.environ
    region = resolve_bedrock_region(env)
    url_base = (base_url or bedrock_runtime_base_url(region)).rstrip("/")
    model_id = require_bedrock_model_id(model)
    extra = dict(extra or {})
    creds = _resolve_call_credentials(api_key=api_key, env=env)

    body = build_converse_body(messages=messages, tools=tools, extra=extra)
    url = converse_stream_model_url(url_base, model_id)
    payload = json.dumps(body).encode("utf-8")
    headers = _auth_headers_for_request(
        method="POST",
        url=url,
        body_bytes=payload,
        region=region,
        creds=creds,
        content_type="application/json",
        accept="application/vnd.amazon.eventstream",
        amz_date=amz_date,
    )
    response = None
    try:
        response = _open_bedrock_event_stream(
            url, headers=headers, body=body, timeout=timeout
        )
        turn = _assistant_turn_from_converse_stream(
            _iter_converse_stream_events(response),
            on_delta=on_delta,
        )
    except Exception as exc:
        _note_bedrock_invoke_outcome(env, creds, error=exc)
        raise
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    _note_bedrock_invoke_outcome(env, creds)
    return turn


def diversify_chat_model_ids(
    model_ids: list[str],
    *,
    max_per_family: int = 3,
    max_total: int = 36,
) -> list[str]:
    """Pick a cross-provider subset of Bedrock chat ids for the agentic registry.

    Account catalogs are large; the router needs a diversified daily-driver set
    (Claude is not the only family). Prefers inference-profile ids when present.
    """
    prefer_prefix = ("us", "eu", "ap", "global")
    priority = [
        "anthropic",
        "amazon",
        "deepseek",
        "zai",
        "moonshotai",
        "moonshot",
        "qwen",
        "meta",
        "mistral",
        "minimax",
        "openai",
        "cohere",
        "ai21",
    ]

    def _family(model_id: str) -> str:
        parts = model_id.split(".")
        if len(parts) >= 3 and parts[0] in prefer_prefix:
            return parts[1]
        return parts[0] if parts else model_id

    def _rank(mid: str) -> tuple:
        head = mid.split(".", 1)[0]
        return (0 if head in prefer_prefix else 1, mid)

    by_family: dict[str, list[str]] = {}
    for mid in model_ids:
        if not mid or not _is_chat_capable_model_id(mid):
            continue
        # Drop context-window variants (…:24k) that duplicate the base id.
        if mid.lower().rstrip().endswith("k") and ":" in mid.split(".")[-1]:
            # e.g. amazon.nova-lite-v1:0:24k
            tail = mid.rsplit(":", 1)[-1].lower()
            if tail.endswith("k") and tail[:-1].isdigit():
                continue
        by_family.setdefault(_family(mid), []).append(mid)

    fam_order = [f for f in priority if f in by_family] + sorted(
        f for f in by_family if f not in priority
    )
    out: list[str] = []
    seen: set[str] = set()
    for fam in fam_order:
        for mid in sorted(by_family[fam], key=_rank)[:max_per_family]:
            if mid in seen:
                continue
            seen.add(mid)
            out.append(mid)
            if len(out) >= max_total:
                return out
    return out


def _bedrock_tier_for_model(model_id: str) -> str:
    n = model_id.lower()
    if any(x in n for x in ("lite", "nano", "haiku", "-8b", "flash-lite", "micro")):
        return "cheap"
    if any(x in n for x in ("opus", "ultra", "premier", "glm-5", "v3.2")):
        return "frontier"
    if any(x in n for x in ("flash", "mini", "fast")):
        return "cheap"
    return "balanced"


_BEDROCK_AGENTIC_TIERS = {
    "frontier": (92, 3.0, 15.0, 200000, ["frontier", "reasoning", "analysis"]),
    "balanced": (85, 3.0, 15.0, 200000, ["balanced", "fast", "vision"]),
    "cheap": (70, 0.8, 4.0, 200000, ["cheap", "fast", "vision"]),
}


def bedrock_agentic_model_specs(
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout: int = 30,
    model_ids: Optional[list[str]] = None,
) -> list:
    """Build agentic :class:`ModelSpec` entries from the account's Bedrock catalog.

    When ``model_ids`` is omitted, discovers via :func:`list_chat_model_ids`.
    """
    from puppetmaster.model_registry import ModelSpec

    env = env if env is not None else os.environ
    ids = list(model_ids) if model_ids is not None else list_chat_model_ids(
        env=env, timeout=timeout
    )
    specs = []
    for mid in diversify_chat_model_ids(ids):
        tier = _bedrock_tier_for_model(mid)
        cap, inp, out, ctx, tags = _BEDROCK_AGENTIC_TIERS.get(
            tier, _BEDROCK_AGENTIC_TIERS["balanced"]
        )
        family = mid.split(".")[1] if mid.startswith(
            ("us.", "eu.", "ap.", "global.")
        ) else mid.split(".")[0]
        specs.append(
            ModelSpec(
                id=f"agentic/{mid}",
                adapter="agentic",
                adapter_model_name=mid,
                capability_score=int(cap),
                input_per_mtok_usd=float(inp),
                output_per_mtok_usd=float(out),
                context_window=int(ctx),
                tags=["agentic", "bedrock", family] + list(tags),
                payload_defaults={"provider": "bedrock"},
                billing="api",
                notes="Discovered from AWS Bedrock ListFoundationModels / ListInferenceProfiles",
            )
        )
    return specs


def merge_bedrock_discovered_into_registry(
    existing: list,
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout: int = 30,
    allow_unverified_catalog: bool = False,
) -> "tuple[list, dict]":
    """Replace agentic Bedrock entries with a live account-specific catalog.

    ListFoundationModels / ListInferenceProfiles are catalog shape only: they
    never mark Bedrock auto-routable. Registry rows are rewritten only when
    invoke-health is currently verified (so ``--write`` cannot re-enable
    Bedrock from a stale default profile). Denied / unverified credentials
    preserve existing registry rows and disabled overlays unchanged.
    """
    from puppetmaster.provider_health import (
        STATUS_VERIFIED,
        bedrock_health_report,
        read_bedrock_invoke_health,
    )
    from puppetmaster.providers import get_provider

    env = env if env is not None else os.environ
    desc = get_provider("bedrock")
    health = bedrock_health_report(env)
    report: dict[str, Any] = {
        "added": 0,
        "removed": 0,
        "available": False,
        "discovered": 0,
        "credentials_present": bool(health.get("credentials_present")),
        "invoke_health": health.get("invoke_health") or "unknown",
    }
    if desc is None or not health.get("credentials_present"):
        return existing, report

    invoke_health = read_bedrock_invoke_health(env)
    report["invoke_health"] = invoke_health
    verified = invoke_health == STATUS_VERIFIED
    report["available"] = verified
    if not verified and not allow_unverified_catalog:
        # Preserve overlays / disabled rows; catalog list alone must not write.
        return existing, report

    try:
        live = list_chat_model_ids(env=env, timeout=timeout)
    except Exception as exc:
        report["error"] = repr(exc)
        return existing, report
    report["discovered"] = len(live)
    if not live:
        return existing, report
    if not verified:
        # Probe/dry-run catalog shape without mutating registry enablement.
        return existing, report

    new_specs = bedrock_agentic_model_specs(env=env, model_ids=live, timeout=timeout)
    # Preserve enabled (and other overlay knobs) from prior Bedrock rows so a
    # catalog refresh cannot silently re-enable models the operator disabled.
    prior_by_name = {
        getattr(spec, "adapter_model_name", ""): spec
        for spec in existing
        if getattr(spec, "adapter", None) == "agentic"
        and (getattr(spec, "payload_defaults", None) or {}).get("provider") == "bedrock"
    }
    from dataclasses import replace

    preserved_new = []
    for spec in new_specs:
        prior = prior_by_name.get(spec.adapter_model_name)
        if prior is None:
            preserved_new.append(spec)
            continue
        preserved_new.append(replace(spec, enabled=prior.enabled))
    kept = []
    removed = 0
    for spec in existing:
        provider = (getattr(spec, "payload_defaults", None) or {}).get("provider")
        if getattr(spec, "adapter", None) == "agentic" and provider == "bedrock":
            removed += 1
            continue
        kept.append(spec)
    report["removed"] = removed
    report["added"] = len(preserved_new)
    return kept + preserved_new, report
