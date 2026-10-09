"""Canonical adapter failure classification.

Every adapter maps CLI/API output to the same vocabulary so identical failures
bucket identically across providers (router recoverable routing, stitcher alerts,
verification artifacts).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

# Canonical failure category strings (artifact payload ``failure`` field).
NOT_AUTHENTICATED = "not_authenticated"
MISSING_CLI = "missing_cli"
RATE_LIMIT = "rate_limit"
BILLING_OR_QUOTA = "billing_or_quota"
MODEL_UNAVAILABLE = "model_unavailable"
APPROVAL_DENIED = "approval_denied"
SANDBOX_DENIED = "sandbox_denied"
TIMEOUT = "timeout"
NETWORK_ERROR = "network_error"
MALFORMED_RESPONSE = "malformed_response"
SERVER_ERROR = "server_error"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
PERMISSION_DENIED = "permission_denied"
FORBIDDEN = "forbidden"
# Legacy OpenAI adapter literal. Kept as its own string so persisted
# artifacts / dashboards matching ``openai_server_error`` keep working.
# Canonical retry / provider policy uses :data:`SERVER_ERROR`.
OPENAI_SERVER_ERROR = "openai_server_error"
SDK_NOT_INSTALLED = "sdk_not_installed"
RUN_STATUS_ERROR = "run_status_error"
UNKNOWN = "unknown"

Checker = Callable[[str], bool]
Rule = Tuple[Checker, str]


def _any(*substrings: str) -> Checker:
    def check(lowered: str) -> bool:
        return any(part in lowered for part in substrings)

    return check


def _all(*substrings: str) -> Checker:
    def check(lowered: str) -> bool:
        return all(part in lowered for part in substrings)

    return check


def _matches(pattern: str) -> Checker:
    compiled = re.compile(pattern)

    def check(lowered: str) -> bool:
        return compiled.search(lowered) is not None

    return check


# An explicit credential diagnosis, never a bare "auth"/"login" substring: those
# also occur in symbols, paths and quoted source (``production_authority``).
_AUTH_DIAGNOSIS = (
    r"\b(?:auth|authentication|authorization|login|log in|sign in|token|api[ _-]key)"
    r"[ _-]?(?:error|failed|failure|required|expired|revoked|invalid|missing)\b"
    r"|\b(?:please |re-?)(?:authenticate|log ?in|sign ?in)\b"
    r"|\b(?:invalid|incorrect|missing|no)[ _-](?:api[ _-]key|token|credentials?)\b"
    r"|\binvalid authentication\b"
    r"|\b(?:login|account|provider|credentials?) verification (?:failed|required)\b"
)

# Python traceback frames: ``File "...", line N, in name`` and the quoted
# source/caret lines under it. They name code, not the failure.
_TRACEBACK_FRAME = re.compile(r'^\s*File "[^"\n]*", line \d+(?:, in [^\n]*)?$')
# Node ``error.stack`` frames: ``    at fn (/path/index.js:401:17)``.
_NODE_STACK_FRAME = re.compile(r"^\s+at .*:\d+:\d+\)?$")


def _without_traceback_frames(text: str) -> str:
    kept = []
    in_frame = False
    for line in text.splitlines():
        if _TRACEBACK_FRAME.match(line):
            in_frame = True
            continue
        if in_frame and line.startswith("    "):
            continue
        in_frame = False
        if _NODE_STACK_FRAME.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _model_unavailable(lowered: str) -> bool:
    return "model" in lowered and (
        "unavailable" in lowered
        or "not found" in lowered
        or "invalid" in lowered
        or "does not exist" in lowered
        or "not supported" in lowered
        or "404" in lowered
    )


def _classify(output: str, rules: Sequence[Rule], *, default: str = UNKNOWN) -> str:
    lowered = _without_traceback_frames(output or "").lower()
    for checker, category in rules:
        if checker(lowered):
            return category
    return default


_BASE_RULES: Tuple[Rule, ...] = (
    (_any("command not found"), MISSING_CLI),
    (_any("not logged in", "codex login", "missing bearer", "unauthorized"), NOT_AUTHENTICATED),
    (_matches(r"(?<![\w.])401(?![\w.])"), NOT_AUTHENTICATED),
    (_any("not authenticated", "please login", "hermes login", "missing credentials"), NOT_AUTHENTICATED),
    (_any("cursor_api_key"), NOT_AUTHENTICATED),
    (_matches(_AUTH_DIAGNOSIS), NOT_AUTHENTICATED),
    (_any("context length", "maximum context", "context window"), CONTEXT_LENGTH_EXCEEDED),
    (_any("rate limit"), RATE_LIMIT),
    (_matches(r"(?<![\w.])429(?![\w.])"), RATE_LIMIT),
    (_any("billing", "quota", "credit"), BILLING_OR_QUOTA),
    (_any("model_not_found"), MODEL_UNAVAILABLE),
    (_model_unavailable, MODEL_UNAVAILABLE),
    (_all("approval", "denied"), APPROVAL_DENIED),
    (_all("approval", "rejected"), APPROVAL_DENIED),
    (_all("sandbox", "denied"), SANDBOX_DENIED),
    (_all("sandbox", "blocked"), SANDBOX_DENIED),
    (_any("timeout", "timed out"), TIMEOUT),
    (_any("network", "dns", "connect"), NETWORK_ERROR),
)

_ADAPTER_EXTRA_RULES: dict[str, Tuple[Rule, ...]] = {
    "cursor": (
        (_any("cannot find package"), SDK_NOT_INSTALLED),
        (_all("@cursor/sdk", "not found"), SDK_NOT_INSTALLED),
        (_any("forbidden-model"), MODEL_UNAVAILABLE),
        (_all("forbidden", "model"), MODEL_UNAVAILABLE),
        (_all("not permitted", "model"), MODEL_UNAVAILABLE),
        (_all("not allowed", "model"), MODEL_UNAVAILABLE),
        (_all("unavailable", "model"), MODEL_UNAVAILABLE),
        (_all("unknown", "model"), MODEL_UNAVAILABLE),
        # Generic Cursor SDK terminal status after more specific model rules.
        (_all("status", "error"), RUN_STATUS_ERROR),
    ),
    "claude-code": (
        (_any("not_found_error", "permission_error"), MODEL_UNAVAILABLE),
        (_all("permission", "model"), MODEL_UNAVAILABLE),
        (_all("not allowed", "model"), MODEL_UNAVAILABLE),
        (_all("denied", "model"), MODEL_UNAVAILABLE),
        (_any("permission", "not allowed", "denied"), PERMISSION_DENIED),
    ),
    "codex": (
        (_any("spend cap"), BILLING_OR_QUOTA),
    ),
    "hermes": (
        (_all("no such file or directory", "hermes"), MISSING_CLI),
        (_any("no provider", "provider credentials", "no inference provider is configured",
              "no api key found for provider", "provider resolution failed"), NOT_AUTHENTICATED),
    ),
    "openai": (),
    "antigravity": (
        (_any("not recognized as a known model", "invalid model selection"), MODEL_UNAVAILABLE),
        (_any("requires --effort", "invalid effort"), MODEL_UNAVAILABLE),
        (_any("not authenticated", "please authenticate", "login to", "oauth"), NOT_AUTHENTICATED),
        (_any("high demand", "error 503", "status: unavailable"), RATE_LIMIT),
        (_any("jetski:", "headless mode cannot prompt", "user denied permission"), PERMISSION_DENIED),
    ),
}


def classify_adapter_failure(adapter: str, output: str) -> str:
    extra = _ADAPTER_EXTRA_RULES.get(adapter, ())
    return _classify(output, (*extra, *_BASE_RULES))


def json_output_diagnostic(stdout: Optional[str],
                           diagnosis: Callable[[dict], Iterable[Any]]) -> str:
    """The parts of a JSON (or JSON-lines) stdout that diagnose a failure.

    Non-JSON lines are CLI banners and errors, and are kept. A JSON event is
    the worker's own output (an agent message, a tool call, a final answer)
    unless ``diagnosis`` names its error fields: an edit to a login module is
    not a logout, and a worker discussing rate limits did not hit one.
    """
    lines: list[str] = []
    for raw in (stdout or "").splitlines():
        if not raw.strip().startswith("{"):
            lines.append(raw)
            continue
        try:
            event = json.loads(raw)
        except ValueError:
            continue  # a truncated transcript event, not a diagnosis
        if isinstance(event, dict):
            lines.extend(str(part) for part in diagnosis(event) if part)
    return "\n".join(lines)


def _error_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("message") or value.get("type")
    if not isinstance(value, str):
        return ""
    # An error code (``rate_limit``, ``authentication_failed``) reads as words.
    return value.replace("_", " ") if re.fullmatch(r"[a-z_]+", value) else value


def claude_code_diagnosis(event: dict) -> list:
    """stream-json / json events: error results and explicit error fields."""
    parts = [_error_text(event.get("error"))]
    if event.get("type") == "result" and event.get("is_error"):
        parts += [event.get("subtype"), event.get("result")]
        parts += [_error_text(item) for item in event.get("errors") or ()]
    return parts


def cursor_diagnosis(event: dict) -> list:
    """The SDK bridge result: its status and errors; the result text only on error."""
    status = event.get("status")
    parts = [f"status: {status}" if status is not None else "",
             _error_text(event.get("error")), _error_text(event.get("message"))]
    if str(status).lower() == "error":
        parts.append(event.get("result"))
    return parts


def antigravity_diagnosis(event: dict) -> list:
    """agy JSON output: its error and status, never the response."""
    return [_error_text(event.get("error")), event.get("status")]


def hermes_diagnostic(stdout: Optional[str], stderr: Optional[str]) -> str:
    """The parts of a ``hermes chat -Q`` run that diagnose a failure.

    Once a turn ran, Hermes prints only the worker's answer to stdout, writes
    backend errors to stderr, and ends stderr with ``session_id: ...``. Before
    a turn (credential or provider setup), its own diagnostics go to stdout.
    """
    stderr = stderr or ""
    if re.search(r"^session_id: ", stderr, re.MULTILINE):
        return stderr
    return stderr + "\n" + (stdout or "")


def classify_fx_failure(stderr: str, result: Optional[dict]) -> Optional[str]:
    """A failure class from fx's stderr and its typed JSON fields, or None.

    The final answer in fx's JSON is the worker's own text and is never read.
    """
    result = result if isinstance(result, dict) else {}
    error = result.get("error")
    if isinstance(result.get("auth_failure"), dict) or error == "MissingCredentials":
        return NOT_AUTHENTICATED
    code = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", error).lower() if isinstance(error, str) else ""
    failure = classify_adapter_failure("fx", "\n".join(part for part in (stderr or "", code) if part))
    return None if failure == UNKNOWN else failure


def classify_antigravity_failure(output: str) -> str:
    return classify_adapter_failure("antigravity", output)


def classify_codex_failure(output: str) -> str:
    return classify_adapter_failure("codex", output)


def classify_hermes_failure(output: str) -> str:
    return classify_adapter_failure("hermes", output)


def classify_cursor_failure(output: str) -> str:
    return classify_adapter_failure("cursor", output)


def classify_claude_code_failure(output: str) -> str:
    return classify_adapter_failure("claude-code", output)


def classify_openai_failure(body: str, http_status: Optional[int] = None) -> str:
    if http_status == 401:
        return NOT_AUTHENTICATED
    if http_status == 403:
        return FORBIDDEN
    if http_status == 404:
        return MODEL_UNAVAILABLE
    if http_status == 429:
        return RATE_LIMIT
    if http_status is not None and 500 <= http_status < 600:
        # Preserve the historical observability literal for OpenAI adapter
        # verification artifacts; provider retry uses SERVER_ERROR instead.
        return OPENAI_SERVER_ERROR
    return classify_adapter_failure("openai", body)


def is_server_error_failure(failure: str) -> bool:
    """True for canonical ``server_error`` or legacy ``openai_server_error``."""
    return failure in (SERVER_ERROR, OPENAI_SERVER_ERROR)


# OpenCode Go (and similar relays) can return HTTP 401 AuthError when the
# upstream vendor blocks a request even though GET /models and the local key
# are fine. Treat as transient — not a dead/revoked key.
_UPSTREAM_BLOCK_PATTERNS = (
    "blocked by upstream provider",
    "request blocked by upstream",
)


def is_upstream_provider_block(message: Optional[str] = None) -> bool:
    """True when the body describes an upstream-vendor block, not a bad key."""
    msg = (message or "").lower()
    return any(pattern in msg for pattern in _UPSTREAM_BLOCK_PATTERNS)


def classify_provider_failure(
    reason: str,
    http_status: Optional[int] = None,
    body: str = "",
) -> str:
    """Bridge raw direct-provider errors into the canonical failure taxonomy.

    ``reason`` remains provider diagnostic data; callers use this normalized
    result for retry and routing decisions. ``body`` is consulted so OpenCode
    Go-style upstream blocks are not misclassified as ``not_authenticated``.
    """
    if http_status is None and reason.startswith("http_status:"):
        try:
            http_status = int(reason.partition(":")[2])
        except ValueError:
            pass

    # Relay 401 "blocked by upstream" is not a rejected local key — classify
    # as a retryable server/upstream failure so health state does not
    # disconnect a valid subscription credential.
    if is_upstream_provider_block(body) or is_upstream_provider_block(reason):
        return SERVER_ERROR

    if http_status == 401:
        return NOT_AUTHENTICATED
    if http_status == 402:
        return BILLING_OR_QUOTA
    if http_status == 403:
        return FORBIDDEN
    if http_status == 404:
        return MODEL_UNAVAILABLE
    if http_status == 429:
        return RATE_LIMIT
    if http_status is not None and 500 <= http_status < 600:
        return SERVER_ERROR
    if http_status is not None and 400 <= http_status < 500:
        failure = classify_adapter_failure("openai", body)
        if failure != UNKNOWN:
            return failure

    category_by_reason = {
        "not_authenticated": NOT_AUTHENTICATED,
        "timeout": TIMEOUT,
        "network_error": NETWORK_ERROR,
        "malformed_response": MALFORMED_RESPONSE,
        # Legacy artifact / dashboard literal still maps to canonical retry.
        "openai_server_error": SERVER_ERROR,
        "server_error": SERVER_ERROR,
        "unsupported_model": MODEL_UNAVAILABLE,
    }
    return category_by_reason.get(reason, reason or UNKNOWN)
