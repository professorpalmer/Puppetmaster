"""Detect, per platform/adapter, whether work bills against a subscription
the user already pays for ("plan") or an out-of-pocket provider key ("api").

This is the runtime half of Puppetmaster's cost-containment story. The
registry (:mod:`puppetmaster.model_registry`) carries a static ``billing``
hint per model, but the *real* answer depends on how each CLI/SDK is
authenticated on this machine right now:

* **Cursor** always bills through the Cursor plan when a ``CURSOR_API_KEY``
  is present — the SDK only exposes the account's own catalog.
* **Claude Code** bills the subscription when signed in via OAuth — detected
  by reading the real ``oauthAccount`` (seat tier / org) from ``~/.claude.json``
  (not mere file existence, which survives a logout) — per-token to the console
  account when ``ANTHROPIC_API_KEY`` is set — or per-token to the AWS account
  when ``CLAUDE_CODE_USE_BEDROCK`` is enabled with usable AWS credentials.
* **Codex** reads ``$CODEX_HOME/auth.json`` when ``CODEX_HOME`` is set, else
  ``~/.codex/auth.json`` (``auth_mode``/``tokens``) directly — an API key is
  out-of-pocket; a ChatGPT login is subscription-covered — falling back to
  ``codex login status`` only when that file is absent.

Every probe is a pure function with injectable ``env`` / ``home`` / ``run``
dependencies so the test suite can exercise each branch without real
credentials or network calls. The orchestrator uses the result to upgrade a
model's ``unknown`` billing to ``plan``/``api`` before routing, and the
preflight check uses ``healthy`` to refuse dispatching to an unauthenticated
adapter.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Optional

from puppetmaster.model_registry import ModelSpec

# A command runner returns (returncode, stdout, stderr). Injectable for tests.
CommandRunner = Callable[[list[str]], "tuple[int, str, str]"]


@dataclass(frozen=True)
class BillingStatus:
    """The detected billing posture for one adapter on this machine."""

    adapter: str
    billing: str  # "plan" | "api" | "unknown"
    healthy: bool  # True when the adapter has usable credentials
    detail: str
    evidence: list[str] = field(default_factory=list)

    @property
    def is_plan_billed(self) -> bool:
        return self.billing == "plan"


@dataclass(frozen=True)
class AuthContext:
    """Effective credential context used to detect one adapter's billing.

    This keeps provider-specific detectors honest about *which* environment
    and home directory they inspected without making any one provider's
    variable name (for example ``CODEX_HOME``) a Puppetmaster-wide concept.
    """

    env: Mapping[str, str]
    home: Path
    label: str = "process"


def auth_context(
    *,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    label: str = "process",
) -> AuthContext:
    return AuthContext(
        env=env if env is not None else os.environ,
        home=home if home is not None else Path.home(),
        label=label,
    )


def _default_runner(command: list[str]) -> "tuple[int, str, str]":
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return (127, "", "command not found")
    except subprocess.TimeoutExpired:
        return (124, "", "timed out")
    return (completed.returncode, completed.stdout or "", completed.stderr or "")


def detect_cursor_billing(
    env: Optional[Mapping[str, str]] = None,
) -> BillingStatus:
    """Cursor work always rides the Cursor plan when a key is configured."""
    env = env if env is not None else os.environ
    if env.get("CURSOR_API_KEY"):
        return BillingStatus(
            adapter="cursor",
            billing="plan",
            healthy=True,
            detail="CURSOR_API_KEY is set; work bills against the Cursor plan.",
            evidence=["cursor_api_key:set"],
        )
    return BillingStatus(
        adapter="cursor",
        billing="unknown",
        healthy=False,
        detail="CURSOR_API_KEY is not set — the Cursor adapter cannot run.",
        evidence=["cursor_api_key:missing"],
    )


def _read_claude_oauth(home: Path) -> "Optional[dict]":
    """Return the ``oauthAccount`` block from ``~/.claude.json`` if a real OAuth
    session is present, else None.

    ``~/.claude.json`` also stores onboarding/config state and survives a
    logout, so "the file exists" is NOT proof of authentication — we require an
    ``oauthAccount`` carrying an ``accountUuid`` or ``emailAddress``.
    """
    path = home / ".claude.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    oauth = data.get("oauthAccount") if isinstance(data, dict) else None
    if isinstance(oauth, dict) and (oauth.get("accountUuid") or oauth.get("emailAddress")):
        return oauth
    return None


def _is_truthy_env_value(value: Optional[str]) -> bool:
    """Return True for common truthy env strings ("1", "true"); False for off/empty."""
    if not value:
        return False
    normalized = value.strip().lower()
    if normalized in ("0", "false", ""):
        return False
    return normalized in ("1", "true")


def _claude_bedrock_enabled(env: Mapping[str, str], home: Path) -> bool:
    if _is_truthy_env_value(env.get("CLAUDE_CODE_USE_BEDROCK")):
        return True
    path = home / ".claude" / "settings.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    env_block = data.get("env")
    if not isinstance(env_block, dict):
        return False
    return _is_truthy_env_value(env_block.get("CLAUDE_CODE_USE_BEDROCK"))


def _detect_aws_credentials(env: Mapping[str, str], home: Path) -> "tuple[Optional[str], list[str]]":
    """Return (credential_kind, evidence) when AWS creds appear usable, else (None, [])."""
    if env.get("AWS_BEARER_TOKEN_BEDROCK"):
        return ("bearer_token", ["aws_credentials:bearer_token"])
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        return ("env_keys", ["aws_credentials:env_keys"])
    if env.get("AWS_PROFILE"):
        return ("profile", ["aws_credentials:profile"])
    aws_dir = home / ".aws"
    for name in ("credentials", "config"):
        path = aws_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return ("config_file", ["aws_credentials:config_file"])
    return (None, [])


def _detect_claude_bedrock(
    env: Mapping[str, str],
    home: Path,
) -> "Optional[BillingStatus]":
    """Return Bedrock billing posture when CLAUDE_CODE_USE_BEDROCK is on, else None."""
    if not _claude_bedrock_enabled(env, home):
        return None

    evidence = ["claude_bedrock:enabled"]
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    region_suffix = f" (region {region})" if region else ""

    cred_kind, cred_evidence = _detect_aws_credentials(env, home)
    if cred_kind is not None:
        evidence.extend(cred_evidence)
        detail = (
            "CLAUDE_CODE_USE_BEDROCK is enabled — Claude Code bills per-token to "
            f"the AWS account via Bedrock{region_suffix} (out-of-pocket)."
        )
        return BillingStatus(
            adapter="claude-code",
            billing="api",
            healthy=True,
            detail=detail,
            evidence=evidence,
        )

    evidence.append("aws_credentials:missing")
    return BillingStatus(
        adapter="claude-code",
        billing="api",
        healthy=False,
        detail=(
            "CLAUDE_CODE_USE_BEDROCK is set but no AWS credentials were detected — "
            "run `aws configure`, set AWS_BEARER_TOKEN_BEDROCK, or set "
            "AWS_ACCESS_KEY_ID (and AWS_SECRET_ACCESS_KEY)."
        ),
        evidence=evidence,
    )


def detect_claude_billing(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> BillingStatus:
    """Claude Code: OAuth subscription (plan) vs ANTHROPIC_API_KEY (api) vs Bedrock (api).

    Plan detection reads the real ``oauthAccount`` from ``~/.claude.json`` (and
    falls back to ``~/.claude/.credentials.json``) rather than trusting that the
    file merely exists — most users are on a Pro/Max/Team subscription, so this
    is the common path and it must not false-positive on a logged-out config.

    When ``CLAUDE_CODE_USE_BEDROCK`` is enabled (env or ``~/.claude/settings.json``),
    Bedrock wins over ``ANTHROPIC_API_KEY`` and bills per-token to the AWS account.
    """
    env = env if env is not None else os.environ
    home = home if home is not None else Path.home()
    bedrock = _detect_claude_bedrock(env, home)
    if bedrock is not None:
        return bedrock
    if env.get("ANTHROPIC_API_KEY"):
        return BillingStatus(
            adapter="claude-code",
            billing="api",
            healthy=True,
            detail=(
                "ANTHROPIC_API_KEY is set — Claude Code bills per-token to that "
                "console account (out-of-pocket)."
            ),
            evidence=["anthropic_api_key:set"],
        )
    oauth = _read_claude_oauth(home)
    if oauth is not None:
        seat = oauth.get("seatTier") or oauth.get("subscriptionType")
        org = oauth.get("organizationName")
        evidence = ["claude_oauth:account"]
        who = []
        if seat:
            evidence.append(f"seat_tier:{seat}")
            who.append(f"{seat} seat")
        if org:
            who.append(f"org '{org}'")
        suffix = f" ({', '.join(who)})" if who else ""
        return BillingStatus(
            adapter="claude-code",
            billing="plan",
            healthy=True,
            detail=(
                "Claude Code is signed in via OAuth" + suffix + " — work bills "
                "against the logged-in Anthropic subscription (no marginal API spend)."
            ),
            evidence=evidence,
        )
    # Fallback: some installs keep tokens only in ~/.claude/.credentials.json.
    creds = home / ".claude" / ".credentials.json"
    if creds.is_file() and creds.stat().st_size > 2:
        return BillingStatus(
            adapter="claude-code",
            billing="plan",
            healthy=True,
            detail=(
                "Claude Code OAuth credentials present (~/.claude/.credentials.json)"
                " — work bills against the logged-in Anthropic subscription."
            ),
            evidence=["claude_oauth:credentials"],
        )
    return BillingStatus(
        adapter="claude-code",
        billing="unknown",
        healthy=False,
        detail=(
            "No ANTHROPIC_API_KEY and no active ~/.claude OAuth session "
            "(run `claude` and sign in) — Claude Code is not authenticated."
        ),
        evidence=["claude_auth:missing"],
    )


def _codex_auth_path(env: Mapping[str, str], home: Path) -> tuple[Path, str]:
    codex_home = env.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json", "$CODEX_HOME/auth.json"
    return home / ".codex" / "auth.json", "~/.codex/auth.json"


def _read_codex_auth(path: Path, label: str) -> "Optional[BillingStatus]":
    """Detect Codex billing from Codex auth.json without a subprocess.

    The file is authoritative and fast: ``auth_mode == "apikey"`` (or a present
    ``OPENAI_API_KEY``) is out-of-pocket; ``auth_mode == "chatgpt"`` (or a
    ``tokens`` block) is subscription-covered. Returns None when the file is
    absent/unreadable so the caller can fall back to ``codex login status``.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    mode = str(data.get("auth_mode") or "").lower()
    has_key = bool(data.get("OPENAI_API_KEY"))
    has_tokens = bool(data.get("tokens"))
    if mode == "apikey" or (has_key and not has_tokens):
        return BillingStatus(
            adapter="codex",
            billing="api",
            healthy=True,
            detail=(
                f"Codex is authenticated with an OpenAI API key ({label}) "
                "— work bills per-token to that account (out-of-pocket)."
            ),
            evidence=["codex_auth:apikey", f"codex_auth_path:{label}"],
        )
    if mode == "chatgpt" or has_tokens:
        return BillingStatus(
            adapter="codex",
            billing="plan",
            healthy=True,
            detail=(
                f"Codex is signed in via a ChatGPT subscription ({label}) "
                "— work is covered by that plan (no marginal API spend)."
            ),
            evidence=["codex_auth:chatgpt", f"codex_auth_path:{label}"],
        )
    return None


def detect_codex_billing(
    run: Optional[CommandRunner] = None,
    codex_command: str = "codex",
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    context: Optional[AuthContext] = None,
) -> BillingStatus:
    """Codex: API key (api) vs ChatGPT subscription (plan).

    Reads ``$CODEX_HOME/auth.json`` first when CODEX_HOME is set, otherwise
    ``~/.codex/auth.json`` (fast, deterministic, no subprocess), and only
    falls back to parsing ``codex login status`` when the file is missing.
    """
    ctx = context or auth_context(env=env, home=home)
    auth_path, auth_label = _codex_auth_path(ctx.env, ctx.home)
    from_file = _read_codex_auth(auth_path, auth_label)
    if from_file is not None:
        return replace(from_file, evidence=[*from_file.evidence, f"auth_context:{ctx.label}"])
    run = run or _default_runner
    returncode, stdout, stderr = run([codex_command, "login", "status"])
    text = f"{stdout}\n{stderr}".lower()
    if returncode == 127 or "command not found" in text:
        return BillingStatus(
            adapter="codex",
            billing="unknown",
            healthy=False,
            detail="Codex CLI not found on PATH.",
            evidence=["codex_cli:missing", f"auth_context:{ctx.label}"],
        )
    if "not logged in" in text or "not authenticated" in text:
        return BillingStatus(
            adapter="codex",
            billing="unknown",
            healthy=False,
            detail="Codex is not logged in (run `codex login`).",
            evidence=["codex_login:none", f"auth_context:{ctx.label}"],
        )
    if "api key" in text:
        return BillingStatus(
            adapter="codex",
            billing="api",
            healthy=True,
            detail=(
                "Codex is logged in with an OpenAI API key — work bills "
                "per-token to that account (out-of-pocket)."
            ),
            evidence=["codex_login:api_key", f"auth_context:{ctx.label}"],
        )
    if "logged in" in text or "chatgpt" in text:
        return BillingStatus(
            adapter="codex",
            billing="plan",
            healthy=True,
            detail=(
                "Codex is logged in via a ChatGPT subscription — work is "
                "covered by that plan."
            ),
            evidence=["codex_login:chatgpt", f"auth_context:{ctx.label}"],
        )
    return BillingStatus(
        adapter="codex",
        billing="unknown",
        healthy=False,
        detail="Codex is not logged in (run `codex login`).",
        evidence=["codex_login:none", f"auth_context:{ctx.label}"],
    )


# Hermes is a meta-router CLI: it reaches an underlying provider (Anthropic,
# OpenAI, Gemini) via a provider API key or an OAuth login. Its billing posture
# is therefore the posture of whatever credential it's configured to use. These
# detectors are re-implemented here (rather than imported from ``adapters``) on
# purpose: ``adapters`` imports this module, so importing it back would be a
# cycle. They mirror ``adapters._hermes_present_credential_keys`` /
# ``_hermes_oauth_providers`` with injectable ``env``/``home`` for testability.
_HERMES_CREDENTIAL_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def _hermes_present_credential_keys(env: Mapping[str, str], home: Path) -> "set[str]":
    """Provider API-key env vars Hermes can see (process env ∪ ~/.hermes/.env)."""
    import re

    present = {key for key in _HERMES_CREDENTIAL_ENV_KEYS if env.get(key)}
    env_file = home / ".hermes" / ".env"
    if env_file.is_file():
        try:
            text = env_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for key in _HERMES_CREDENTIAL_ENV_KEYS:
            if re.search(rf"^\s*{re.escape(key)}\s*=\s*\S+", text, re.MULTILINE):
                present.add(key)
    return present


def _hermes_oauth_providers(home: Path) -> "set[str]":
    """Hermes provider names that carry OAuth state in ~/.hermes/auth.json."""
    auth_file = home / ".hermes" / "auth.json"
    if not auth_file.is_file():
        return set()
    try:
        payload = json.loads(auth_file.read_text(encoding="utf-8", errors="replace") or "{}")
    except (OSError, json.JSONDecodeError):
        return set()
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if isinstance(providers, dict):
        return {str(name).lower() for name in providers}
    return set()


def detect_hermes_billing(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> BillingStatus:
    """Hermes billing posture = the posture of its configured provider credential.

    Hermes routes to an underlying provider, so the question "will this cost me
    extra?" reduces to which credential it will use:

    * A provider **API key** (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
      ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, from the process env or
      ``~/.hermes/.env``) bills per-token to that provider — ``api``.
    * An **OAuth login** (``~/.hermes/auth.json``) is covered by that provider
      login — ``plan``.
    * Neither → ``unknown`` and unhealthy: Hermes cannot run.

    When both are configured we report ``api`` — at least one out-of-pocket path
    exists, and underselling the cost would be the wrong way to be wrong.
    """
    env = env if env is not None else os.environ
    home = home if home is not None else Path.home()

    keys = sorted(_hermes_present_credential_keys(env, home))
    oauth = sorted(_hermes_oauth_providers(home))
    evidence = [f"hermes_api_key:{key}" for key in keys]
    evidence += [f"hermes_oauth:{provider}" for provider in oauth]

    if not keys and not oauth:
        return BillingStatus(
            adapter="hermes",
            billing="unknown",
            healthy=False,
            detail=(
                "No Hermes provider credential found — set a provider API key "
                "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY) in "
                "~/.hermes/.env or run `hermes login`."
            ),
            evidence=evidence or ["hermes_credentials:missing"],
        )

    if keys:
        oauth_note = (
            f" OAuth login also present for: {', '.join(oauth)}." if oauth else ""
        )
        return BillingStatus(
            adapter="hermes",
            billing="api",
            healthy=True,
            detail=(
                f"Hermes has provider API key(s) configured ({', '.join(keys)}) — "
                f"those providers bill per-token (out-of-pocket).{oauth_note}"
            ),
            evidence=evidence,
        )

    return BillingStatus(
        adapter="hermes",
        billing="plan",
        healthy=True,
        detail=(
            f"Hermes is signed in via OAuth ({', '.join(oauth)}) — work is covered "
            "by that provider login (no marginal API-key spend)."
        ),
        evidence=evidence,
    )


# OpenAI raw-API and shell are not subscription-coverable.
def detect_openai_billing(
    env: Optional[Mapping[str, str]] = None,
) -> BillingStatus:
    env = env if env is not None else os.environ
    healthy = bool(env.get("OPENAI_API_KEY"))
    return BillingStatus(
        adapter="openai",
        billing="api",
        healthy=healthy,
        detail=(
            "OpenAI adapter bills per-token to OPENAI_API_KEY (out-of-pocket)."
            if healthy
            else "OPENAI_API_KEY is not set — the OpenAI adapter cannot run."
        ),
        evidence=["openai_api_key:" + ("set" if healthy else "missing")],
    )


def detect_agentic_billing(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> BillingStatus:
    """Agentic bills per-token to whichever provider API key is actually available.

    Unlike Hermes/Codex there is no external CLI or OAuth file to probe — the
    standalone worker calls provider HTTP APIs directly, so readiness reduces to
    ``available_providers()`` from :mod:`puppetmaster.providers`. Bedrock is
    reported as present-but-unverified / denied when credentials exist without
    a current verified invoke-health record (never "ready" on presence alone).
    """
    from puppetmaster.provider_health import bedrock_health_report
    from puppetmaster.providers import available_providers

    env = env if env is not None else os.environ
    providers = sorted(available_providers(env))
    bedrock = bedrock_health_report(env, home=home)
    healthy = bool(providers)
    evidence = [f"agentic_provider:{slug}" for slug in providers]
    if bedrock.get("credentials_present"):
        evidence.append(f"bedrock_invoke_health:{bedrock.get('invoke_health')}")
        evidence.append(
            "bedrock_auto_routable:"
            + ("yes" if bedrock.get("auto_routable") else "no")
        )
    if healthy:
        detail = (
            "Agentic adapter bills per-token to your provider API key(s) "
            f"(out-of-pocket). Ready providers: {', '.join(providers)}."
        )
        if bedrock.get("credentials_present") and not bedrock.get("auto_routable"):
            detail += f" Bedrock: {bedrock.get('detail')}."
    else:
        detail = (
            "No auto-routable provider credential — set OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, or "
            "OPENROUTER_API_KEY (no external CLI)."
        )
        if bedrock.get("credentials_present"):
            detail = (
                f"Bedrock: {bedrock.get('detail')}. Other providers: set "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
                "GOOGLE_API_KEY, or OPENROUTER_API_KEY."
            )
            evidence.append("agentic_providers:none_auto_routable")
        else:
            evidence.append("agentic_providers:none")
    return BillingStatus(
        adapter="agentic",
        billing="api",
        healthy=healthy,
        detail=detail,
        evidence=evidence,
    )


_DETECTORS: dict[str, Callable[..., BillingStatus]] = {
    "cursor": lambda **kw: detect_cursor_billing(env=kw.get("env")),
    "claude-code": lambda **kw: detect_claude_billing(
        env=kw.get("env"), home=kw.get("home")
    ),
    "codex": lambda **kw: detect_codex_billing(
        run=kw.get("run"), env=kw.get("env"), home=kw.get("home"), context=kw.get("context")
    ),
    "hermes": lambda **kw: detect_hermes_billing(
        env=kw.get("env"), home=kw.get("home")
    ),
    "openai": lambda **kw: detect_openai_billing(env=kw.get("env")),
    "agentic": lambda **kw: detect_agentic_billing(
        env=kw.get("env"), home=kw.get("home")
    ),
}


def detect_adapter_billing(
    adapter: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    run: Optional[CommandRunner] = None,
    context: Optional[AuthContext] = None,
) -> BillingStatus:
    """Detect the billing posture for ``adapter``.

    Unknown adapters resolve to a benign ``unknown``/healthy status so callers
    can treat them as pass-through (the mod never blocks a path it can't
    reason about).
    """
    detector = _DETECTORS.get(adapter)
    if detector is None:
        return BillingStatus(
            adapter=adapter,
            billing="unknown",
            healthy=True,
            detail=f"No billing detector for adapter {adapter!r}; treating as pass-through.",
            evidence=[f"adapter:{adapter}", "detector:none"],
        )
    return detector(env=env, home=home, run=run, context=context)


_BILLING_CACHE: dict[str, tuple[BillingStatus, float]] = {}


def _default_billing_ttl_seconds() -> int:
    raw = os.environ.get("PUPPETMASTER_BILLING_TTL_SECONDS")
    if raw is not None:
        return int(raw)
    return 300


def clear_billing_cache() -> None:
    """Clear the module-level billing detection cache (for tests)."""
    _BILLING_CACHE.clear()


def detect_adapter_billing_cached(
    adapter: str,
    *,
    ttl_seconds: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    run: Optional[CommandRunner] = None,
) -> BillingStatus:
    """Like :func:`detect_adapter_billing`, with a TTL cache per adapter."""
    if ttl_seconds is None:
        ttl_seconds = _default_billing_ttl_seconds()
    if ttl_seconds == 0:
        return detect_adapter_billing(adapter, env=env, home=home, run=run)

    now = time.monotonic()
    cached = _BILLING_CACHE.get(adapter)
    if cached is not None:
        status, stamped = cached
        if now - stamped < ttl_seconds:
            return status

    status = detect_adapter_billing(adapter, env=env, home=home, run=run)
    _BILLING_CACHE[adapter] = (status, now)
    return status


@dataclass(frozen=True)
class RegistryReconciliation:
    """Result of upgrading registry billing hints and filtering unhealthy adapters."""

    specs: list[ModelSpec]
    upgraded: list[dict[str, str]]
    dropped: list[dict[str, str]]


def reconcile_registry(
    specs: list[ModelSpec],
    *,
    detect: Optional[Callable[..., BillingStatus]] = None,
) -> RegistryReconciliation:
    """Upgrade ``unknown`` billing from runtime detection and drop unhealthy adapters."""
    if not specs:
        return RegistryReconciliation(specs=[], upgraded=[], dropped=[])

    detect_fn = detect or detect_adapter_billing_cached
    upgraded: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    upgraded_specs: list[ModelSpec] = []
    surviving: list[ModelSpec] = []

    for spec in specs:
        working = spec
        try:
            status = detect_fn(spec.adapter)
        except Exception:
            upgraded_specs.append(spec)
            surviving.append(spec)
            continue

        if status.billing in ("plan", "api") and spec.billing == "unknown":
            working = replace(spec, billing=status.billing)
            upgraded.append(
                {"model_id": spec.id, "from": spec.billing, "to": status.billing}
            )

        upgraded_specs.append(working)

        if not status.healthy:
            reason = (
                f"adapter {spec.adapter!r} has no usable credentials: {status.detail}"
            )
            dropped.append(
                {"model_id": spec.id, "adapter": spec.adapter, "reason": reason}
            )
        else:
            surviving.append(working)

    if not surviving:
        return RegistryReconciliation(
            specs=upgraded_specs, upgraded=upgraded, dropped=dropped
        )

    return RegistryReconciliation(specs=surviving, upgraded=upgraded, dropped=dropped)
