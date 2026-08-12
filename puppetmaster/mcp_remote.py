"""Remote (HTTP) MCP transport for non-stdio pilots such as Grok Bot.

Grok Bot can only consume remote HTTP/SSE (or streamable HTTP) MCP connectors.
This module wraps the existing ``mcp_server.handle_message`` / tool handlers so
the durable job/artifact contracts stay identical — only the transport changes.

This is **not** a ``grok-bot`` worker adapter. Grok Bot is the pilot/client;
Puppetmaster remains the leased worker runtime. Do not invent a worker adapter
named ``grok-bot`` until Grok Bot publishes a real task-dispatch API.

Design notes:
- Zero new dependencies — stdlib ``http.server`` only (matches the dashboard /
  provider_proxy pattern and keeps the core package dependency-free).
- Streamable HTTP (MCP 2025-03-26) at ``/mcp`` is the primary path.
- Legacy HTTP+SSE (2024-11-05) at ``/sse`` + ``/message`` for older clients.
- Bearer auth is mandatory — anonymous remote job control is refused.
- Default tool scope is ``supervise`` (doctor / start review|plan|swarm /
  status|logs|artifacts|show). Full-edit / implement tools require
  ``--scope implement`` (or ``PUPPETMASTER_MCP_REMOTE_SCOPE=implement``).
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Optional
from urllib.parse import parse_qs, urlparse

from puppetmaster import __version__ as _PACKAGE_VERSION
from puppetmaster.mcp_server import (
    error_response,
    handle_message,
    tool_error,
    tools,
)

JsonObject = dict[str, Any]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8743
DEFAULT_SCOPE = "supervise"
MCP_ENDPOINT_PATH = "/mcp"
LEGACY_SSE_PATH = "/sse"
LEGACY_MESSAGE_PATH = "/message"
HEALTH_PATH = "/health"
TOKEN_ENV = "PUPPETMASTER_MCP_TOKEN"
SCOPE_ENV = "PUPPETMASTER_MCP_REMOTE_SCOPE"
HOST_ENV = "PUPPETMASTER_MCP_REMOTE_HOST"
PORT_ENV = "PUPPETMASTER_MCP_REMOTE_PORT"

# Default only when the client omits protocolVersion. Echo any non-empty
# requested version verbatim — do not allowlist-gate. Returning a different
# version (e.g. hardcoding 2024-11-05 while the client asked for 2025-03-26)
# makes strict remote clients disconnect after initialize — the Grok Bot
# "Failed to load MCP server / tools=0" failure mode.
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

# Remote tools/list payload hygiene for Plugins / Grok Bot. Oversized
# descriptions and loose schemas have been observed to flap the connector
# after a green initialize; keep the listed surface compact and strict.
REMOTE_TOOL_DESCRIPTION_MAX = 280
REMOTE_PROPERTY_DESCRIPTION_MAX = 120
_SIMPLE_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "description",
        "enum",
        "items",
        "default",
        "additionalProperties",
        "title",
        "const",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
        "pattern",
    }
)

# Tools that start workers which may edit the tree or act externally.
# Supervise-first remote surface omits these unless scope=implement.
IMPLEMENT_TOOL_NAMES = frozenset(
    {
        "puppetmaster_claude_implement",
        "puppetmaster_start_claude_implement",
        "puppetmaster_cursor_implement",
        "puppetmaster_start_cursor_implement",
        "puppetmaster_start_implement",
        "puppetmaster_edit",
        "puppetmaster_codex",
        "puppetmaster_start_codex",
        "puppetmaster_agentic",
        "puppetmaster_start_agentic",
        "puppetmaster_openai",
        "puppetmaster_start_openai",
        "puppetmaster_start_browser_swarm",
        "puppetmaster_start_prewalk",
        "puppetmaster_dashboard",
        "puppetmaster_reset_subgraph",
        "puppetmaster_gc",
        "puppetmaster_codegraph_init",
        "puppetmaster_codegraph_index",
        "puppetmaster_repair_codegraph",
        "puppetmaster_mcp_cleanup",
        "puppetmaster_gate",
    }
)

# Fail closed for the default remote scope. New tools are not remotely
# callable until they have been reviewed as safe for a read-only pilot.
SUPERVISE_TOOL_NAMES = frozenset(
    {
        "puppetmaster_doctor",
        "puppetmaster_route_task",
        "puppetmaster_list_models",
        "puppetmaster_job_cost",
        "puppetmaster_job_receipt",
        "puppetmaster_cursor_review",
        "puppetmaster_start_cursor_review",
        "puppetmaster_cursor_plan",
        "puppetmaster_start_cursor_plan",
        "puppetmaster_start_swarm",
        "puppetmaster_start_cursor_swarm",
        "puppetmaster_last_job",
        "puppetmaster_status",
        "puppetmaster_logs",
        "puppetmaster_live_artifacts",
        "puppetmaster_live_artifacts_follow",
        "puppetmaster_partial_summary",
        "puppetmaster_await_job",
        "puppetmaster_artifacts",
        "puppetmaster_job_graph",
        "puppetmaster_show",
        "puppetmaster_codegraph_search",
        "puppetmaster_codegraph_context",
        "puppetmaster_codegraph_affected",
        "puppetmaster_codegraph_files",
        "puppetmaster_codegraph_status",
        "puppetmaster_mcp_status",
        "puppetmaster_rollup",
    }
)

MAX_REMOTE_REQUEST_BYTES = 1024 * 1024


class RequestBodyTooLarge(ValueError):
    """Raised before reading a remote request body above the safety limit."""


VALID_SCOPES = frozenset({"supervise", "implement"})


@dataclass
class RemoteMcpConfig:
    """Runtime knobs for the remote MCP server."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str = ""
    scope: str = DEFAULT_SCOPE
    allow_origins: tuple[str, ...] = ()
    rate_limit_per_minute: int = 120
    audit_log_path: Optional[str] = None
    require_auth: bool = True
    # How long legacy GET /sse holds the stream open to flush /message replies.
    # Primary Grok Bot path is streamable HTTP /mcp; this is for older clients.
    legacy_sse_hold_seconds: float = 30.0
    # Streamable HTTP GET /mcp long-lived SSE stream (post-initialize).
    get_stream_keepalive_seconds: float = 15.0
    get_stream_max_seconds: float = 300.0

    def validate(self) -> None:
        if self.scope not in VALID_SCOPES:
            raise ValueError(
                f"Unknown remote MCP scope {self.scope!r}; "
                f"expected one of {sorted(VALID_SCOPES)}"
            )
        if self.require_auth and not (self.token or "").strip():
            raise ValueError(
                "Remote MCP refuses to start without a bearer token. "
                f"Pass --token / --token-file, or set {TOKEN_ENV}."
            )
        if self.rate_limit_per_minute < 0:
            raise ValueError("rate_limit_per_minute must be >= 0")
        if self.legacy_sse_hold_seconds < 0:
            raise ValueError("legacy_sse_hold_seconds must be >= 0")
        if self.get_stream_keepalive_seconds <= 0:
            raise ValueError("get_stream_keepalive_seconds must be > 0")
        if self.get_stream_max_seconds <= 0:
            raise ValueError("get_stream_max_seconds must be > 0")


@dataclass
class _Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    protocol_version: str = "2025-03-26"


@dataclass
class _LegacySseClient:
    session_id: str
    queue: "deque[JsonObject]" = field(default_factory=deque)
    closed: bool = False


class RateLimiter:
    """Sliding-window request limiter keyed by client identity."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = int(limit_per_minute)
        self._hits: dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: Optional[float] = None) -> bool:
        if self.limit_per_minute <= 0:
            return True
        moment = time.time() if now is None else now
        window_start = moment - 60.0
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit_per_minute:
                return False
            bucket.append(moment)
            return True


class AuditLogger:
    """Append-only JSONL audit of remote MCP requests (no secrets)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: JsonObject) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            print(f"[puppetmaster-mcp-remote] {line}", file=sys.stderr)
            if not self.path:
                return
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                print(
                    f"[puppetmaster-mcp-remote] audit write failed: {exc}",
                    file=sys.stderr,
                )


def generate_token() -> str:
    """Cryptographically strong bearer token suitable for clipboard paste."""
    return secrets.token_urlsafe(32)


def resolve_token(
    *,
    explicit: Optional[str] = None,
    token_file: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> str:
    """Resolve a bearer token from flag, file, or environment (in that order)."""
    if explicit and explicit.strip():
        return explicit.strip()
    if token_file:
        with open(token_file, encoding="utf-8") as handle:
            raw = handle.read().strip()
        if not raw:
            raise ValueError(f"Token file {token_file!r} is empty")
        return raw
    environ = env if env is not None else os.environ
    from_env = (environ.get(TOKEN_ENV) or "").strip()
    if from_env:
        return from_env
    return ""


def resolve_scope(
    *,
    explicit: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> str:
    if explicit and explicit.strip():
        return explicit.strip().lower()
    environ = env if env is not None else os.environ
    return (environ.get(SCOPE_ENV) or DEFAULT_SCOPE).strip().lower() or DEFAULT_SCOPE


def tool_allowed(name: str, scope: str) -> bool:
    """Return True when *name* is visible/callable under *scope*."""
    if scope == "implement":
        return True
    return name in SUPERVISE_TOOL_NAMES


def filtered_tools(scope: str) -> list:
    return [tool for tool in tools() if tool_allowed(tool.name, scope)]


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Parse ``Authorization: Bearer <token>`` (case-insensitive scheme)."""
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2:
        return None
    scheme, value = parts
    if scheme.lower() != "bearer":
        return None
    value = value.strip()
    return value or None


def token_matches(provided: Optional[str], expected: str) -> bool:
    if not expected:
        return False
    if provided is None:
        return False
    return secrets.compare_digest(provided, expected)


def origin_allowed(origin: Optional[str], allow_origins: tuple[str, ...]) -> bool:
    """DNS-rebinding guard.

    Non-browser MCP clients typically omit Origin — those are allowed.
    When Origin is present it must match an allowlist entry, ``*``, or a
    localhost origin when the allowlist is empty (local PoC default).
    """
    if not origin:
        return True
    if "*" in allow_origins:
        return True
    if origin in allow_origins:
        return True
    if allow_origins:
        return False
    # Empty allowlist: only permit loopback browser origins. Parse the
    # hostname instead of using a prefix check, which would admit
    # ``127.0.0.1.evil.com`` and ``localhost.evil.com``.
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    return parsed.hostname.lower() in {"127.0.0.1", "localhost", "::1"}


def connector_snippet(*, base_url: str, token: str) -> JsonObject:
    """JSON a Grok Bot / remote MCP client can paste as a connector config."""
    mcp_url = base_url.rstrip("/") + MCP_ENDPOINT_PATH
    return {
        "name": "puppetmaster",
        "url": mcp_url,
        "transport": "streamable-http",
        "headers": {"Authorization": f"Bearer {token}"},
        "notes": (
            "Grok Bot is the pilot; Puppetmaster is the durable worker runtime. "
            "v1 remote surface defaults to supervise scope (no implement/edit). "
            "Paste the /mcp URL (streamable HTTP), not /sse, unless your client "
            "only speaks legacy HTTP+SSE."
        ),
    }


def negotiate_protocol_version(requested: Optional[str]) -> str:
    """Echo any non-empty client protocolVersion; default only if missing."""
    version = (requested or "").strip()
    if version:
        return version
    return DEFAULT_PROTOCOL_VERSION


def remote_initialize_capabilities() -> JsonObject:
    """Minimal capabilities proven to load Grok Bot Plugins (tools=50)."""
    return {
        # Non-empty tools object: some remote clients treat bare `{}` as
        # "no tool support". listChanged=false is honest — we don't push
        # tools/list_changed notifications on this transport.
        "tools": {"listChanged": False},
        "logging": {},
    }


def _truncate_text(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


def _simplify_json_schema(node: Any, *, property_description_max: int) -> Any:
    """Keep only simple JSON Schema fields; truncate nested descriptions."""
    if isinstance(node, list):
        return [
            _simplify_json_schema(item, property_description_max=property_description_max)
            for item in node
        ]
    if not isinstance(node, dict):
        return node
    simplified: JsonObject = {}
    for key, value in node.items():
        if key not in _SIMPLE_SCHEMA_KEYS:
            continue
        if key == "description" and isinstance(value, str):
            simplified[key] = _truncate_text(value, property_description_max)
        elif key == "properties" and isinstance(value, dict):
            simplified[key] = {
                name: _simplify_json_schema(
                    prop, property_description_max=property_description_max
                )
                for name, prop in value.items()
                if isinstance(name, str) and isinstance(prop, dict)
            }
        elif key == "items":
            simplified[key] = _simplify_json_schema(
                value, property_description_max=property_description_max
            )
        elif key == "required" and isinstance(value, list):
            simplified[key] = [str(item) for item in value if isinstance(item, str)]
        else:
            simplified[key] = value
    return simplified


def remote_tool_to_json(tool: Any) -> JsonObject:
    """Compact, Plugins-safe tool descriptor for remote ``tools/list``.

    Truncates descriptions, strips exotic schema keys, forces
    ``additionalProperties: false``, and drops required names that are not
    present in properties. Stdio ``tool_to_json`` is intentionally untouched.
    """
    raw_schema = getattr(tool, "input_schema", None)
    schema_src = dict(raw_schema) if isinstance(raw_schema, dict) else {}
    schema = _simplify_json_schema(
        schema_src, property_description_max=REMOTE_PROPERTY_DESCRIPTION_MAX
    )
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        schema["properties"] = properties
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [name for name in required if name in properties]
        if not schema["required"]:
            schema.pop("required", None)
    elif "required" in schema:
        schema.pop("required", None)
    schema["additionalProperties"] = False
    return {
        "name": str(getattr(tool, "name", "") or ""),
        "description": _truncate_text(
            str(getattr(tool, "description", "") or ""),
            REMOTE_TOOL_DESCRIPTION_MAX,
        ),
        "inputSchema": schema,
    }


def build_remote_initialize_result(
    *,
    request_id: Any,
    params: JsonObject,
    scope: str,
) -> JsonObject:
    """Build the initialize JSON-RPC response for the remote transport.

    Intentionally does **not** call ``mcp_server.handle_message``: the stdio
    handler hardcodes ``protocolVersion: 2024-11-05`` and ``tools: {}``, which
    makes Grok Bot / Cursor remote clients abort after initialize (tools=0).

    Live-proven shape: echo protocolVersion, minimal capabilities
    (tools.listChanged + logging), no ``experimental`` / ``instructions``.
    ``scope`` is accepted for call-site stability but is not embedded here.
    """
    del scope  # advertised via filtered tools/list, not initialize fluff
    requested = params.get("protocolVersion")
    negotiated = negotiate_protocol_version(
        str(requested) if requested is not None else None
    )
    result: JsonObject = {
        "protocolVersion": negotiated,
        "capabilities": remote_initialize_capabilities(),
        "serverInfo": {
            "name": "puppetmaster-remote",
            "version": _PACKAGE_VERSION,
        },
    }
    # Belt-and-suspenders: never ship the stdio defaults even if a future
    # edit merges another result dict in.
    result["protocolVersion"] = negotiated
    caps = result.setdefault("capabilities", {})
    if not isinstance(caps, dict):
        caps = {}
        result["capabilities"] = caps
    tools_cap = caps.get("tools")
    if not isinstance(tools_cap, dict) or not tools_cap:
        caps["tools"] = {"listChanged": False}
    else:
        caps["tools"] = dict(tools_cap)
        caps["tools"].setdefault("listChanged", False)
    caps.setdefault("logging", {})
    # Never advertise experimental / instructions on this transport.
    result.pop("instructions", None)
    caps.pop("experimental", None)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
        "_puppetmaster_protocol_version": negotiated,
    }


def prefer_sse_response(accept_header: Optional[str]) -> bool:
    """Return True only when the client wants SSE and not JSON.

    Grok Bot Plugins (and many MCP HTTP clients) send
    ``Accept: application/json, text/event-stream`` but parse initialize as
    plain JSON. Preferring SSE whenever event-stream appears causes
    "Failed to load MCP server" despite a correct protocolVersion body.

    Rule: JSON when Accept includes ``application/json``, is empty, or
    includes ``*/*``; SSE only when ``text/event-stream`` is listed and
    ``application/json`` is not (and Accept is not empty/``*/*``).
    """
    accept = (accept_header or "").strip().lower()
    if not accept or "*/*" in accept:
        return False
    if "application/json" in accept:
        return False
    return "text/event-stream" in accept


class RemoteMcpState:
    """Shared mutable state for one listening remote MCP process."""

    def __init__(self, config: RemoteMcpConfig) -> None:
        config.validate()
        self.config = config
        self.sessions: dict[str, _Session] = {}
        self.legacy_clients: dict[str, _LegacySseClient] = {}
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.audit = AuditLogger(config.audit_log_path)
        self._lock = threading.Lock()

    def create_session(self, protocol_version: str = "2025-03-26") -> _Session:
        session = _Session(session_id=secrets.token_urlsafe(24), protocol_version=protocol_version)
        with self._lock:
            self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: Optional[str]) -> Optional[_Session]:
        if not session_id:
            return None
        with self._lock:
            return self.sessions.get(session_id)

    def drop_session(self, session_id: str) -> bool:
        with self._lock:
            return self.sessions.pop(session_id, None) is not None

    def create_legacy_client(self) -> _LegacySseClient:
        client = _LegacySseClient(session_id=str(uuid.uuid4()))
        with self._lock:
            self.legacy_clients[client.session_id] = client
        return client

    def get_legacy_client(self, session_id: Optional[str]) -> Optional[_LegacySseClient]:
        if not session_id:
            return None
        with self._lock:
            return self.legacy_clients.get(session_id)


def handle_remote_message(
    message: JsonObject,
    *,
    scope: str,
    state: Optional[RemoteMcpState] = None,
) -> Optional[JsonObject]:
    """JSON-RPC dispatch with remote scope filtering applied.

    Reuses the stdio server's handlers so job/artifact semantics stay identical.
    """
    method = message.get("method")
    request_id = message.get("id")

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        # NEVER route initialize through mcp_server.handle_message — that path
        # hardcodes protocolVersion 2024-11-05 and capabilities.tools {}.
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        response = build_remote_initialize_result(
            request_id=request_id,
            params=params,
            scope=scope,
        )
        negotiated = str(response.get("_puppetmaster_protocol_version") or DEFAULT_PROTOCOL_VERSION)
        if state is not None:
            session = state.create_session(protocol_version=negotiated)
            response["_puppetmaster_session_id"] = session.session_id
        return response

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [remote_tool_to_json(tool) for tool in filtered_tools(scope)]
            },
        }

    if method == "tools/call":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        name = str(params.get("name") or "")
        if not tool_allowed(name, scope):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": tool_error(
                    (
                        f"Tool {name!r} is outside the remote MCP scope "
                        f"{scope!r}. Restart with --scope implement (or set "
                        f"{SCOPE_ENV}=implement) to expose full-edit verbs. "
                        "See docs/GROK_BOT.md."
                    ),
                    {
                        "code": "remote_scope_denied",
                        "tool": name,
                        "scope": scope,
                        "fix": f"python -m puppetmaster mcp serve-remote --scope implement",
                    },
                ),
            }
        return handle_message(message)

    # Unknown / other methods — defer to the shared handler.
    return handle_message(message)


def make_handler(state: RemoteMcpState) -> type:
    """Build a ``BaseHTTPRequestHandler`` bound to *state*."""

    class RemoteMcpHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Quiet by default; structured audit covers the security trail.
            return

        def _peer_key(self) -> str:
            return self.client_address[0] if self.client_address else "unknown"

        def _audit(
            self,
            *,
            event: str,
            status: int,
            method: Optional[str] = None,
            tool: Optional[str] = None,
            detail: Optional[str] = None,
        ) -> None:
            state.audit.write(
                {
                    "ts": time.time(),
                    "event": event,
                    "status": status,
                    "peer": self._peer_key(),
                    "path": self.path,
                    "http_method": self.command,
                    "rpc_method": method,
                    "tool": tool,
                    "detail": detail,
                    "scope": state.config.scope,
                }
            )

        def _read_json_body(self) -> Any:
            raw_length = self.headers.get("Content-Length", "0") or "0"
            try:
                length = int(raw_length)
            except (TypeError, ValueError) as exc:
                raise ValueError("Content-Length must be an integer") from exc
            if length < 0:
                raise ValueError("Content-Length must be non-negative")
            if length > MAX_REMOTE_REQUEST_BYTES:
                raise RequestBodyTooLarge(
                    f"request body exceeds {MAX_REMOTE_REQUEST_BYTES} bytes"
                )
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))

        def _require_streamable_session(self, messages: list[JsonObject]) -> bool:
            """Require an existing session for every non-initialize request."""
            if all(message.get("method") == "initialize" for message in messages):
                return True
            session_id = (self.headers.get("Mcp-Session-Id") or "").strip()
            if not session_id:
                self._audit(
                    event="session_denied",
                    status=400,
                    detail="missing Mcp-Session-Id",
                )
                self._send_json(
                    400,
                    {
                        "error": "missing_session",
                        "detail": "Mcp-Session-Id is required after initialize",
                    },
                )
                return False
            if state.get_session(session_id) is None:
                self._audit(
                    event="session_denied",
                    status=404,
                    detail="unknown Mcp-Session-Id",
                )
                self._send_json(
                    404,
                    {
                        "error": "session_not_found",
                        "detail": "Unknown Mcp-Session-Id; re-initialize",
                    },
                )
                return False
            return True

        def _cors_headers(self) -> dict[str, str]:
            """CORS headers for browser-hosted MCP clients (Grok Bot / tunnels).

            ``Mcp-Session-Id`` MUST be exposed — without
            Access-Control-Expose-Headers the browser cannot read the session
            id from initialize, so the client re-initializes forever and never
            reaches tools/list (tools=0).
            """
            origin = self.headers.get("Origin")
            if not origin_allowed(origin, state.config.allow_origins):
                return {}
            if origin:
                allow_origin = origin
            elif "*" in state.config.allow_origins:
                allow_origin = "*"
            else:
                # No Origin header (non-browser MCP client) — nothing to add.
                return {}
            return {
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": (
                    "Authorization, Content-Type, Accept, Mcp-Session-Id, "
                    "MCP-Protocol-Version, X-Puppetmaster-Token, Last-Event-ID"
                ),
                "Access-Control-Expose-Headers": (
                    "Mcp-Session-Id, MCP-Protocol-Version, WWW-Authenticate, Retry-After"
                ),
                "Vary": "Origin",
            }

        def _send(
            self,
            code: int,
            body: bytes,
            *,
            content_type: str = "application/json",
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            headers: dict[str, str] = {}
            headers.update(self._cors_headers())
            if extra_headers:
                headers.update(extra_headers)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _send_json(
            self,
            code: int,
            payload: Any,
            *,
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            data = json.dumps(payload).encode("utf-8")
            self._send(code, data, content_type="application/json", extra_headers=extra_headers)

        def _send_sse_json(
            self,
            payload: Any,
            *,
            extra_headers: Optional[dict[str, str]] = None,
            event_id: Optional[str] = None,
        ) -> None:
            frame = f"event: message\ndata: {json.dumps(payload)}\n"
            if event_id:
                frame = f"id: {event_id}\n" + frame
            frame += "\n"
            body = frame.encode("utf-8")
            headers = {"Connection": "close"}
            if extra_headers:
                headers.update(extra_headers)
            self._send(
                200,
                body,
                content_type="text/event-stream",
                extra_headers=headers,
            )

        def _unauthorized(self, detail: str = "missing or invalid bearer token") -> None:
            self._audit(event="auth_denied", status=401, detail=detail)
            self._send_json(
                401,
                {
                    "error": "unauthorized",
                    "detail": detail,
                    "hint": (
                        "Send Authorization: Bearer <token>. "
                        f"Token comes from {TOKEN_ENV} or `mcp serve-remote --token`."
                    ),
                },
                extra_headers={"WWW-Authenticate": "Bearer"},
            )

        def _forbidden_origin(self) -> None:
            self._audit(event="origin_denied", status=403, detail=self.headers.get("Origin"))
            self._send_json(
                403,
                {
                    "error": "origin_denied",
                    "detail": "Origin header is not on the allowlist",
                    "hint": "Pass --allow-origin <origin> (or --allow-origin *) for tunnels/browsers.",
                },
            )

        def _rate_limited(self) -> None:
            self._audit(event="rate_limited", status=429)
            self._send_json(
                429,
                {
                    "error": "rate_limited",
                    "detail": (
                        f"Exceeded {state.config.rate_limit_per_minute} "
                        "requests/minute for this client"
                    ),
                },
                extra_headers={"Retry-After": "60"},
            )

        def _check_guards(self, *, require_auth: bool = True) -> bool:
            origin = self.headers.get("Origin")
            if not origin_allowed(origin, state.config.allow_origins):
                self._forbidden_origin()
                return False
            client_key = self._peer_key()
            if not state.rate_limiter.allow(client_key):
                self._rate_limited()
                return False
            if require_auth and state.config.require_auth:
                provided = extract_bearer_token(self.headers.get("Authorization"))
                if provided is None:
                    # Also accept X-Puppetmaster-Token for clients that can't set Authorization.
                    alt = (self.headers.get("X-Puppetmaster-Token") or "").strip()
                    provided = alt or None
                if not token_matches(provided, state.config.token):
                    self._unauthorized()
                    return False
            return True

        def do_OPTIONS(self) -> None:  # noqa: N802
            # Minimal CORS preflight for browser-based MCP clients / tunnels.
            if not origin_allowed(self.headers.get("Origin"), state.config.allow_origins):
                self._forbidden_origin()
                return
            cors = self._cors_headers()
            self.send_response(204)
            for key, value in cors.items():
                self.send_header(key, value)
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path == HEALTH_PATH:
                # Unauthenticated liveness — no job data, no token echo.
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "service": "puppetmaster-remote-mcp",
                        "version": _PACKAGE_VERSION,
                        "scope": state.config.scope,
                        "mcp": MCP_ENDPOINT_PATH,
                        "legacy_sse": LEGACY_SSE_PATH,
                    },
                )
                return

            if path == "/":
                self._send_json(
                    200,
                    {
                        "service": "puppetmaster-remote-mcp",
                        "version": _PACKAGE_VERSION,
                        "scope": state.config.scope,
                        "endpoints": {
                            "mcp": MCP_ENDPOINT_PATH,
                            "health": HEALTH_PATH,
                            "legacy_sse": LEGACY_SSE_PATH,
                            "legacy_message": LEGACY_MESSAGE_PATH,
                        },
                        "docs": "docs/GROK_BOT.md",
                        "auth": "Bearer token required on MCP endpoints",
                    },
                )
                return

            if path == MCP_ENDPOINT_PATH:
                if not self._check_guards():
                    return
                # Streamable HTTP: clients open GET /mcp as a long-lived SSE
                # stream after initialize. Closing immediately looks like a
                # dead server (Grok Bot "Failed to load MCP server" loop).
                accept = (self.headers.get("Accept") or "").lower()
                if "text/event-stream" not in accept:
                    self._send_json(
                        405,
                        {
                            "error": "method_not_allowed",
                            "detail": "GET /mcp requires Accept: text/event-stream",
                        },
                    )
                    return
                session_id = (self.headers.get("Mcp-Session-Id") or "").strip() or None
                if not session_id:
                    self._audit(
                        event="session_denied",
                        status=400,
                        detail="missing Mcp-Session-Id",
                    )
                    self._send_json(
                        400,
                        {
                            "error": "missing_session",
                            "detail": "Mcp-Session-Id is required after initialize",
                        },
                    )
                    return
                if state.get_session(session_id) is None:
                    self._send_json(
                        404,
                        {
                            "error": "session_not_found",
                            "detail": "Unknown Mcp-Session-Id; re-initialize",
                        },
                    )
                    return
                self._serve_mcp_get_stream(session_id=session_id)
                return

            if path == LEGACY_SSE_PATH:
                if not self._check_guards():
                    return
                client = state.create_legacy_client()
                endpoint = f"{LEGACY_MESSAGE_PATH}?sessionId={client.session_id}"
                # Legacy HTTP+SSE: hold the stream open and flush any responses
                # queued by POST /message. Primary Grok Bot path remains /mcp.
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                for key, value in self._cors_headers().items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(
                        f"event: endpoint\ndata: {endpoint}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    deadline = time.time() + float(state.config.legacy_sse_hold_seconds)
                    while time.time() < deadline and not client.closed:
                        while client.queue:
                            payload = client.queue.popleft()
                            frame = (
                                f"event: message\ndata: {json.dumps(payload)}\n\n"
                            ).encode("utf-8")
                            self.wfile.write(frame)
                            self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    client.closed = True
                self._audit(event="legacy_sse_open", status=200, detail=client.session_id)
                return

            self._send_json(404, {"error": "not_found", "path": path})

        def _serve_mcp_get_stream(self, *, session_id: Optional[str]) -> None:
            """Hold GET /mcp open with SSE comment keepalives until disconnect/timeout."""
            keepalive = float(state.config.get_stream_keepalive_seconds)
            max_hold = float(state.config.get_stream_max_seconds)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            # Long-lived stream: do not force Connection: close on open.
            for key, value in self._cors_headers().items():
                self.send_header(key, value)
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.end_headers()
            self._audit(
                event="mcp_get_stream_open",
                status=200,
                detail=f"session={session_id or '-'};keepalive={keepalive};max={max_hold}",
            )
            closed_reason = "max_hold"
            try:
                self.wfile.write(b": puppetmaster-remote-mcp stream open\n\n")
                self.wfile.flush()
                deadline = time.time() + max_hold
                next_ping = time.time() + keepalive
                while time.time() < deadline:
                    now = time.time()
                    if now >= next_ping:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        next_ping = now + keepalive
                    time.sleep(min(0.25, max(0.05, next_ping - time.time())))
            except (BrokenPipeError, ConnectionResetError, OSError):
                closed_reason = "client_disconnect"
            self._audit(
                event="mcp_get_stream_close",
                status=200,
                detail=f"session={session_id or '-'};reason={closed_reason}",
            )

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path != MCP_ENDPOINT_PATH:
                self._send_json(404, {"error": "not_found", "path": path})
                return
            if not self._check_guards():
                return
            session_id = self.headers.get("Mcp-Session-Id")
            if not session_id:
                self._send_json(400, {"error": "missing_session", "detail": "Mcp-Session-Id required"})
                return
            dropped = state.drop_session(session_id)
            self._audit(
                event="session_delete",
                status=200 if dropped else 404,
                detail=session_id,
            )
            if not dropped:
                self._send_json(404, {"error": "session_not_found"})
                return
            self._send(200, b"", content_type="text/plain")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path == MCP_ENDPOINT_PATH:
                self._handle_streamable_post()
                return
            if path == LEGACY_MESSAGE_PATH:
                self._handle_legacy_message(parsed)
                return
            self._send_json(404, {"error": "not_found", "path": path})

        def _handle_streamable_post(self) -> None:
            if not self._check_guards():
                return
            try:
                payload = self._read_json_body()
            except RequestBodyTooLarge as exc:
                self.close_connection = True
                self._audit(event="body_too_large", status=413, detail=str(exc))
                self._send_json(
                    413,
                    {"error": "request_too_large", "detail": str(exc)},
                )
                return
            except (ValueError, UnicodeDecodeError) as exc:
                self._audit(event="bad_json", status=400, detail=str(exc))
                self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
                return

            messages: list[JsonObject]
            if isinstance(payload, list):
                messages = [m for m in payload if isinstance(m, dict)]
            elif isinstance(payload, dict):
                messages = [payload]
            else:
                self._send_json(400, {"error": "invalid_body", "detail": "expected JSON object or array"})
                return

            if not self._require_streamable_session(messages):
                return

            # Notifications / responses only → 202.
            has_request = any("method" in m and "id" in m for m in messages)
            notification_only = all(
                ("method" in m and "id" not in m) or ("result" in m or "error" in m)
                for m in messages
            ) and not has_request

            if notification_only:
                for message in messages:
                    if "method" in message:
                        handle_remote_message(
                            message, scope=state.config.scope, state=state
                        )
                self._audit(event="notification", status=202)
                self._send(202, b"", content_type="text/plain")
                return

            # Single-request happy path: application/json response.
            # (SSE upgrade is optional; JSON is enough for Grok Bot tool loops.)
            if len(messages) == 1:
                message = messages[0]
                rpc_method = str(message.get("method") or "")
                tool_name = None
                if rpc_method == "tools/call":
                    params = message.get("params") if isinstance(message.get("params"), dict) else {}
                    tool_name = str(params.get("name") or "") or None
                try:
                    response = handle_remote_message(
                        message, scope=state.config.scope, state=state
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    response = error_response(message.get("id"), -32000, str(exc))
                if response is None:
                    self._audit(
                        event="rpc",
                        status=202,
                        method=rpc_method,
                        tool=tool_name,
                    )
                    self._send(202, b"", content_type="text/plain")
                    return
                session_id = response.pop("_puppetmaster_session_id", None)
                protocol_version = response.pop("_puppetmaster_protocol_version", None)
                if protocol_version is None and rpc_method == "initialize":
                    # Never let a refactor silently reintroduce stdio defaults.
                    result_obj = response.get("result") if isinstance(response, dict) else None
                    if isinstance(result_obj, dict):
                        protocol_version = result_obj.get("protocolVersion")
                headers: dict[str, str] = {}
                if session_id:
                    headers["Mcp-Session-Id"] = str(session_id)
                if protocol_version:
                    headers["MCP-Protocol-Version"] = str(protocol_version)
                audit_detail = None
                if rpc_method == "initialize" and isinstance(response.get("result"), dict):
                    result_obj = response["result"]
                    audit_detail = (
                        f"protocolVersion={result_obj.get('protocolVersion')};"
                        f"tools_cap={result_obj.get('capabilities', {}).get('tools')}"
                    )
                # Prefer JSON for dual-Accept clients (Grok Bot); SSE only if
                # Accept is event-stream without application/json.
                if prefer_sse_response(self.headers.get("Accept")):
                    self._audit(
                        event="rpc_sse",
                        status=200,
                        method=rpc_method,
                        tool=tool_name,
                        detail=audit_detail,
                    )
                    self._send_sse_json(response, extra_headers=headers or None, event_id="1")
                    return
                self._audit(
                    event="rpc",
                    status=200,
                    method=rpc_method,
                    tool=tool_name,
                    detail=audit_detail,
                )
                self._send_json(200, response, extra_headers=headers or None)
                return

            # Batch: return a JSON array of responses.
            responses: list[JsonObject] = []
            session_header: Optional[str] = None
            for message in messages:
                if "id" not in message and "method" in message:
                    handle_remote_message(message, scope=state.config.scope, state=state)
                    continue
                try:
                    response = handle_remote_message(
                        message, scope=state.config.scope, state=state
                    )
                except Exception as exc:  # pragma: no cover
                    response = error_response(message.get("id"), -32000, str(exc))
                if response is None:
                    continue
                sid = response.pop("_puppetmaster_session_id", None)
                if sid and not session_header:
                    session_header = str(sid)
                responses.append(response)
            headers = {"Mcp-Session-Id": session_header} if session_header else None
            self._audit(event="rpc_batch", status=200, detail=str(len(responses)))
            self._send_json(200, responses, extra_headers=headers)

        def _handle_legacy_message(self, parsed) -> None:
            if not self._check_guards():
                return
            query = parse_qs(parsed.query or "")
            session_id = (query.get("sessionId") or [None])[0]
            client = state.get_legacy_client(session_id)
            if client is None:
                self._send_json(
                    404,
                    {
                        "error": "unknown_session",
                        "detail": "Open GET /sse first to obtain a sessionId",
                    },
                )
                return
            try:
                payload = self._read_json_body()
            except RequestBodyTooLarge as exc:
                self.close_connection = True
                self._audit(event="body_too_large", status=413, detail=str(exc))
                self._send_json(
                    413,
                    {"error": "request_too_large", "detail": str(exc)},
                )
                return
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "invalid_body"})
                return
            try:
                response = handle_remote_message(
                    payload, scope=state.config.scope, state=state
                )
            except Exception as exc:  # pragma: no cover
                response = error_response(payload.get("id"), -32000, str(exc))
            if response is not None:
                response.pop("_puppetmaster_session_id", None)
                client.queue.append(response)
            # Legacy transport acknowledges the POST; response rides the SSE.
            # For PoC clients that only POST, also return the JSON body.
            self._audit(
                event="legacy_message",
                status=200,
                method=str(payload.get("method") or ""),
            )
            if response is None:
                self._send(202, b"", content_type="text/plain")
            else:
                self._send_json(200, response)

    return RemoteMcpHandler


def build_server(config: RemoteMcpConfig) -> tuple[ThreadingHTTPServer, RemoteMcpState]:
    """Construct (but do not serve) a bound remote MCP server."""
    state = RemoteMcpState(config)
    handler = make_handler(state)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    return server, state


def serve_remote(config: RemoteMcpConfig) -> int:
    """Run the remote MCP server until interrupted. Returns a process exit code."""
    server, _state = build_server(config)
    base = f"http://{config.host}:{server.server_address[1]}"
    snippet = connector_snippet(base_url=base, token=config.token)
    print(f"puppetmaster remote MCP listening on {base}{MCP_ENDPOINT_PATH}", file=sys.stderr)
    print(f"  scope: {config.scope}", file=sys.stderr)
    print(f"  health: {base}{HEALTH_PATH}", file=sys.stderr)
    print(f"  legacy SSE: {base}{LEGACY_SSE_PATH}", file=sys.stderr)
    if config.host in ("0.0.0.0", "::"):
        print(
            "  WARNING: bound on all interfaces. Prefer 127.0.0.1 + a secure "
            "tunnel (cloudflared/ngrok) unless you terminate TLS yourself.",
            file=sys.stderr,
        )
    print("  connector (redact before sharing):", file=sys.stderr)
    print(json.dumps(snippet, indent=2), file=sys.stderr)
    print(
        "  Docs: docs/GROK_BOT.md — Grok Bot is the pilot; Puppetmaster is the runtime.",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\npuppetmaster remote MCP stopped.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def config_from_env_and_args(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    token_file: Optional[str] = None,
    scope: Optional[str] = None,
    allow_origins: Optional[list[str]] = None,
    rate_limit_per_minute: Optional[int] = None,
    audit_log: Optional[str] = None,
    generate_if_missing: bool = True,
    env: Optional[dict[str, str]] = None,
) -> tuple[RemoteMcpConfig, bool]:
    """Build a config; returns ``(config, token_was_generated)``."""
    environ = env if env is not None else os.environ
    resolved_token = resolve_token(explicit=token, token_file=token_file, env=environ)
    generated = False
    if not resolved_token and generate_if_missing:
        resolved_token = generate_token()
        generated = True
    resolved_scope = resolve_scope(explicit=scope, env=environ)
    resolved_host = host or (environ.get(HOST_ENV) or DEFAULT_HOST)
    if port is not None:
        resolved_port = int(port)
    else:
        raw_port = (environ.get(PORT_ENV) or "").strip()
        resolved_port = int(raw_port) if raw_port else DEFAULT_PORT
    config = RemoteMcpConfig(
        host=resolved_host,
        port=resolved_port,
        token=resolved_token,
        scope=resolved_scope,
        allow_origins=tuple(allow_origins or ()),
        rate_limit_per_minute=(
            120 if rate_limit_per_minute is None else int(rate_limit_per_minute)
        ),
        audit_log_path=audit_log,
    )
    return config, generated


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry for ``python -m puppetmaster.mcp_remote`` / console script."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="puppetmaster-mcp-remote",
        description=(
            "Serve Puppetmaster MCP tools over streamable HTTP for remote pilots "
            "(Grok Bot). Stdio MCP is unchanged."
        ),
    )
    parser.add_argument("--host", default=None, help=f"Bind host (default {DEFAULT_HOST}).")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Bind port (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=f"Bearer token (or set {TOKEN_ENV}). Generated if omitted.",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Read bearer token from a file (chmod 600 recommended).",
    )
    parser.add_argument(
        "--scope",
        choices=sorted(VALID_SCOPES),
        default=None,
        help=(
            "Tool scope: supervise (default, no implement/edit) or implement "
            f"(full surface). Or set {SCOPE_ENV}."
        ),
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=None,
        dest="allow_origins",
        help="Allowed Origin header (repeatable). Use * for any (tunnel PoCs).",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max authenticated requests per client IP per minute (default 120; 0=off).",
    )
    parser.add_argument(
        "--audit-log",
        default=None,
        help="Append JSONL audit records to this path (also printed on stderr).",
    )
    parser.add_argument(
        "--print-token",
        action="store_true",
        help="Print the bearer token on stdout once at startup (for scripting).",
    )
    args = parser.parse_args(argv)

    try:
        config, generated = config_from_env_and_args(
            host=args.host,
            port=args.port,
            token=args.token,
            token_file=args.token_file,
            scope=args.scope,
            allow_origins=args.allow_origins,
            rate_limit_per_minute=args.rate_limit,
            audit_log=args.audit_log,
        )
        config.validate()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if generated:
        print(
            "Generated bearer token (also in connector JSON below). "
            f"Persist it via {TOKEN_ENV} or --token-file for stable reconnects.",
            file=sys.stderr,
        )
    if args.print_token:
        print(config.token)
    return serve_remote(config)


if __name__ == "__main__":
    raise SystemExit(main())
