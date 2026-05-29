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
  (not mere file existence, which survives a logout) — or per-token to the
  console account when ``ANTHROPIC_API_KEY`` is set.
* **Codex** reads ``~/.codex/auth.json`` (``auth_mode``/``tokens``) directly —
  an API key is out-of-pocket; a ChatGPT login is subscription-covered —
  falling back to ``codex login status`` only when that file is absent.

Every probe is a pure function with injectable ``env`` / ``home`` / ``run``
dependencies so the test suite can exercise each branch without real
credentials or network calls. The orchestrator uses the result to upgrade a
model's ``unknown`` billing to ``plan``/``api`` before routing, and the
preflight check uses ``healthy`` to refuse dispatching to an unauthenticated
adapter.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

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
            detail="Cursor SDK authenticated; work bills against the Cursor plan.",
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
    import json

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


def detect_claude_billing(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> BillingStatus:
    """Claude Code: OAuth subscription (plan) vs ANTHROPIC_API_KEY (api).

    Plan detection reads the real ``oauthAccount`` from ``~/.claude.json`` (and
    falls back to ``~/.claude/.credentials.json``) rather than trusting that the
    file merely exists — most users are on a Pro/Max/Team subscription, so this
    is the common path and it must not false-positive on a logged-out config.
    """
    env = env if env is not None else os.environ
    home = home if home is not None else Path.home()
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


def _read_codex_auth(home: Path) -> "Optional[BillingStatus]":
    """Detect Codex billing from ``~/.codex/auth.json`` without a subprocess.

    The file is authoritative and fast: ``auth_mode == "apikey"`` (or a present
    ``OPENAI_API_KEY``) is out-of-pocket; ``auth_mode == "chatgpt"`` (or a
    ``tokens`` block) is subscription-covered. Returns None when the file is
    absent/unreadable so the caller can fall back to ``codex login status``.
    """
    import json

    path = home / ".codex" / "auth.json"
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
                "Codex is authenticated with an OpenAI API key (~/.codex/auth.json) "
                "— work bills per-token to that account (out-of-pocket)."
            ),
            evidence=["codex_auth:apikey"],
        )
    if mode == "chatgpt" or has_tokens:
        return BillingStatus(
            adapter="codex",
            billing="plan",
            healthy=True,
            detail=(
                "Codex is signed in via a ChatGPT subscription (~/.codex/auth.json) "
                "— work is covered by that plan (no marginal API spend)."
            ),
            evidence=["codex_auth:chatgpt"],
        )
    return None


def detect_codex_billing(
    run: Optional[CommandRunner] = None,
    codex_command: str = "codex",
    home: Optional[Path] = None,
) -> BillingStatus:
    """Codex: API key (api) vs ChatGPT subscription (plan).

    Reads ``~/.codex/auth.json`` first (fast, deterministic, no subprocess) and
    only falls back to parsing ``codex login status`` when the file is missing.
    """
    home = home if home is not None else Path.home()
    from_file = _read_codex_auth(home)
    if from_file is not None:
        return from_file
    run = run or _default_runner
    returncode, stdout, stderr = run([codex_command, "login", "status"])
    text = f"{stdout}\n{stderr}".lower()
    if returncode == 127 or "command not found" in text:
        return BillingStatus(
            adapter="codex",
            billing="unknown",
            healthy=False,
            detail="Codex CLI not found on PATH.",
            evidence=["codex_cli:missing"],
        )
    if "not logged in" in text or "not authenticated" in text:
        return BillingStatus(
            adapter="codex",
            billing="unknown",
            healthy=False,
            detail="Codex is not logged in (run `codex login`).",
            evidence=["codex_login:none"],
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
            evidence=["codex_login:api_key"],
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
            evidence=["codex_login:chatgpt"],
        )
    return BillingStatus(
        adapter="codex",
        billing="unknown",
        healthy=False,
        detail="Codex is not logged in (run `codex login`).",
        evidence=["codex_login:none"],
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


_DETECTORS: dict[str, Callable[..., BillingStatus]] = {
    "cursor": lambda **kw: detect_cursor_billing(env=kw.get("env")),
    "claude-code": lambda **kw: detect_claude_billing(
        env=kw.get("env"), home=kw.get("home")
    ),
    "codex": lambda **kw: detect_codex_billing(run=kw.get("run"), home=kw.get("home")),
    "openai": lambda **kw: detect_openai_billing(env=kw.get("env")),
}


def detect_adapter_billing(
    adapter: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    run: Optional[CommandRunner] = None,
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
    return detector(env=env, home=home, run=run)
