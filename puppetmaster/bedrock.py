"""AWS Bedrock client for agentic workers (stdlib only — no boto3).

Lets standalone agentic workers call Claude (and later other) models on
Bedrock with the user's AWS auth: bearer token (``AWS_BEARER_TOKEN_BEDROCK``)
or SigV4-signed ``InvokeModel`` with access keys. Reuses Anthropic message /
tool shapes; model ids must be Bedrock-shaped (inference profile / ARN).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


_MISSING_CREDS_MSG = (
    "AWS Bedrock credentials not found — run `aws configure`, set "
    "AWS_BEARER_TOKEN_BEDROCK, or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
)

_SHORT_MODEL_MSG = (
    "is not a Bedrock model id. Set a Bedrock inference profile id "
    "(e.g. us.anthropic.claude-sonnet-4-5-20250929-v1:0) or an "
    "arn:aws:bedrock:... ARN — short names like claude-opus-4-8 are rejected."
)

_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"
# Content type used by Bedrock Claude Messages / InvokeModel.
_ANTHROPIC_MESSAGES_CONTENT_TYPE = (
    "application/amazon.bedrock.anthropic.messages-v1+json"
)


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
    canonical_uri = parsed.path or "/"
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
    """Bedrock Runtime ``InvokeModel`` URL for ``model_id``."""
    encoded = urllib.parse.quote(model_id, safe="")
    return f"{base_url.rstrip('/')}/model/{encoded}/invoke"


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
    """Call Bedrock ``InvokeModel`` and return a normalized ``AssistantTurn``.

    Bearer auth uses ``Authorization: Bearer`` (Claude Code Bedrock API-key
    posture). Access-key auth uses minimal SigV4. Streaming callers still hit
    this non-stream invoke (Bedrock event-stream is not SSE); ``on_delta`` is
    invoked once with the final text when provided.
    """
    from puppetmaster.providers import ProviderError

    env = env if env is not None else os.environ
    region = resolve_bedrock_region(env)
    url_base = (base_url or bedrock_runtime_base_url(region)).rstrip("/")
    model_id = require_bedrock_model_id(model)
    extra = dict(extra or {})

    creds = None
    if api_key and str(api_key).strip():
        creds = BedrockCredentials(
            kind="bearer",
            bearer_token=str(api_key).strip(),
            evidence=("aws_credentials:bearer_token",),
        )
    else:
        creds = resolve_bedrock_credentials(env)
    if creds is None:
        raise ProviderError(
            missing_bedrock_credentials_message(),
            reason="not_authenticated",
        )

    body = build_anthropic_invoke_body(messages=messages, tools=tools, extra=extra)
    url = invoke_model_url(url_base, model_id)
    payload = json.dumps(body).encode("utf-8")

    if creds.kind == "bearer" and creds.bearer_token:
        headers = {
            "Content-Type": _ANTHROPIC_MESSAGES_CONTENT_TYPE,
            "Accept": "application/json",
            "Authorization": f"Bearer {creds.bearer_token}",
            "User-Agent": "puppetmaster-agentic",
        }
        data = _post_bedrock(url, headers=headers, body=body, timeout=timeout)
    elif creds.access_key_id and creds.secret_access_key:
        unsigned = {
            "Content-Type": _ANTHROPIC_MESSAGES_CONTENT_TYPE,
            "Accept": "application/json",
            "User-Agent": "puppetmaster-agentic",
        }
        headers = sigv4_sign_headers(
            method="POST",
            url=url,
            headers=unsigned,
            body=payload,
            region=region,
            service="bedrock",
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
            amz_date=amz_date,
        )
        data = _post_bedrock(url, headers=headers, body=body, timeout=timeout)
    else:
        raise ProviderError(
            missing_bedrock_credentials_message()
            + " (AWS_PROFILE / ~/.aws is present but has no static access keys "
            "Puppetmaster can load without boto3 — export "
            "AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID.)",
            reason="not_authenticated",
        )

    turn = _assistant_turn_from_anthropic(data)
    if on_delta and turn.text:
        on_delta("text", turn.text)
    return turn
