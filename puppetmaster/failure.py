"""Canonical adapter failure classification.

Every adapter maps CLI/API output to the same vocabulary so identical failures
bucket identically across providers (router recoverable routing, stitcher alerts,
verification artifacts).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

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

# Legacy alias still referenced in stitcher remediation and older artifacts.
MISSING_API_KEY = NOT_AUTHENTICATED

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


def _model_unavailable(lowered: str) -> bool:
    return "model" in lowered and (
        "unavailable" in lowered
        or "not found" in lowered
        or "invalid" in lowered
        or "does not exist" in lowered
        or "404" in lowered
    )


def _classify(output: str, rules: Sequence[Rule], *, default: str = UNKNOWN) -> str:
    lowered = (output or "").lower()
    for checker, category in rules:
        if checker(lowered):
            return category
    return default


_BASE_RULES: Tuple[Rule, ...] = (
    (_any("command not found"), MISSING_CLI),
    (_any("not logged in", "codex login", "missing bearer", "401", "unauthorized"), NOT_AUTHENTICATED),
    (_any("api key", "not authenticated", "authentication", "please login", "hermes login", "missing credentials"), NOT_AUTHENTICATED),
    (_all("verification", "failed"), NOT_AUTHENTICATED),
    (_all("verification", "required"), NOT_AUTHENTICATED),
    (_any("cursor_api_key"), NOT_AUTHENTICATED),
    (_any("auth", "login"), NOT_AUTHENTICATED),
    (_any("context length", "maximum context", "context window"), CONTEXT_LENGTH_EXCEEDED),
    (_any("rate limit", "429"), RATE_LIMIT),
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
    "codex": (),
    "hermes": (
        (_all("no such file or directory", "hermes"), MISSING_CLI),
        (_any("no provider", "provider credentials"), NOT_AUTHENTICATED),
    ),
    "openai": (),
}


def classify_adapter_failure(adapter: str, output: str) -> str:
    extra = _ADAPTER_EXTRA_RULES.get(adapter, ())
    return _classify(output, (*extra, *_BASE_RULES))


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
    if http_status == 403:
        return FORBIDDEN
    if http_status == 429:
        return RATE_LIMIT
    if http_status is not None and 500 <= http_status < 600:
        return SERVER_ERROR

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
