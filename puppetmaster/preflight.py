"""Universal pre-dispatch check: prove an adapter can actually run a model
*before* committing an expensive job to it.

This is the adapter-agnostic safety net for the failure mode that motivated
it: a Claude Code worker that returned in 391ms with "Credit balance is too
low" and zero edits, surfacing as a degraded run only after the fact. A
preflight catches that class of problem — unauthenticated CLI, depleted
account, out-of-plan model, model the plan no longer exposes — up front, with
a clear reason, so the orchestrator can re-route or fail fast instead of
burning a turn on a doomed dispatch.

It composes the two discovery primitives:

* :mod:`puppetmaster.platform_billing` — is this adapter authenticated, and
  does it bill a subscription (plan) or a raw key (api)?
* :mod:`puppetmaster.cursor_discovery` — for Cursor, is the routed model id
  actually in the plan's live catalog?

Every dependency is injectable so the whole thing is unit-testable without
real credentials or network.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from puppetmaster.platform_billing import (
    BillingStatus,
    CommandRunner,
    detect_adapter_billing,
)


@dataclass(frozen=True)
class PreflightResult:
    """Verdict on whether ``adapter``+``model`` is safe to dispatch."""

    ok: bool
    adapter: str
    model: Optional[str]
    billing: str
    reason: str
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "adapter": self.adapter,
            "model": self.model,
            "billing": self.billing,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


# Injectable catalog fetcher: returns the Cursor plan catalog list. Defaults to
# the real SDK-backed fetch; tests pass a stub.
CatalogFetcher = Callable[[], list]

# Injectable live prober: given (adapter, model) returns (returncode, stdout,
# stderr) from a minimal real call. Defaults to per-adapter 1-token probes;
# tests pass a stub so no real credentials/network are touched.
LiveProber = Callable[[str, Optional[str]], "tuple[int, str, str]"]


# Substrings that mean "your account can't actually run this", regardless of
# adapter. The static billing detector can't see these — only a real call can.
_LIVE_BILLING_MARKERS = (
    "credit balance is too low",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "billing",
    "payment required",
    "quota",
    "not_enough_credits",
    # Subscription/plan-side exhaustion (Cursor/ChatGPT/Max): a plan can be
    # authenticated and carry the model yet be out of monthly allowance or
    # rate-limited right now — a catalog ping can't see that, only a real call.
    "rate limit",
    "rate_limit",
    "usage limit",
    "usage_limit",
    "too many requests",
    "monthly limit",
    "limit reached",
    "out of credits",
    "429",
)
_LIVE_AUTH_MARKERS = (
    "unauthorized",
    "invalid api key",
    "authentication",
    "not logged in",
    "401",
    "403",
)


def classify_live_probe(adapter: str, returncode: int, output: str) -> Optional[str]:
    """Classify a live-probe result into a failure class, or None if it passed.

    Reuses the adapter-specific classifiers where they exist, then falls back
    to adapter-agnostic billing/auth marker matching. Returns one of
    ``billing_or_quota`` / ``auth`` / ``probe_failed`` / None."""
    lowered = output.lower()
    if returncode == 0 and not any(m in lowered for m in _LIVE_BILLING_MARKERS):
        if not any(m in lowered for m in _LIVE_AUTH_MARKERS):
            return None
    try:
        from puppetmaster.adapters import (
            classify_claude_code_failure,
            classify_codex_failure,
            classify_openai_failure,
        )

        if adapter == "claude-code":
            klass = classify_claude_code_failure(output)
        elif adapter == "codex":
            klass = classify_codex_failure(output)
        elif adapter == "openai":
            klass = classify_openai_failure(output)
        else:
            klass = None
    except Exception:
        klass = None
    if klass and klass != "unknown":
        return klass
    if any(m in lowered for m in _LIVE_BILLING_MARKERS):
        return "billing_or_quota"
    if any(m in lowered for m in _LIVE_AUTH_MARKERS):
        return "auth"
    if returncode != 0:
        return "probe_failed"
    return None


def _default_prober(adapter: str, model: Optional[str]) -> "tuple[int, str, str]":
    """Best-effort 1-token real call per adapter. Short timeout; never raises.

    These commands intentionally do the smallest possible real round-trip so a
    funded-looking-but-empty account (the exact case static detection misses)
    fails here instead of mid-job."""
    import subprocess

    try:
        if adapter == "claude-code":
            cmd = shlex.split(os.environ.get("CLAUDE_CODE_COMMAND", "claude"))
            cmd += ["-p", "Reply with: ok"]
            if model:
                cmd += ["--model", model]
        elif adapter == "codex":
            cmd = shlex.split(os.environ.get("CODEX_COMMAND", "codex"))
            cmd += ["exec", "Reply with: ok"]
            if model:
                cmd += ["-m", model]
        else:
            return (0, "", "no live probe for adapter")
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return (
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )
    except FileNotFoundError:
        return (127, "", "command not found")
    except subprocess.TimeoutExpired:
        return (124, "", "live probe timed out")
    except Exception as exc:  # pragma: no cover - defensive
        return (1, "", str(exc))


def _probe_cursor(
    model: Optional[str],
    env: Mapping[str, str],
    *,
    runner: Optional[Path] = None,
) -> "tuple[int, str, str]":
    """Minimal real generation through the Cursor SDK runner.

    A catalog ping proves the plan is authenticated and exposes the model, but
    NOT that the plan can serve a token *right now* — a Cursor subscription can
    be over its monthly allowance, rate-limited, or suspended and still
    enumerate models. This drives the same runner the adapter uses with a
    1-token prompt so that exhaustion surfaces here instead of mid-job."""
    import json
    import subprocess

    from puppetmaster.cursor_discovery import CURSOR_RUNNER

    base_env = dict(env)
    if not base_env.get("CURSOR_API_KEY"):
        return (1, "", "CURSOR_API_KEY not set")
    base_env["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
        {
            "prompt": "Reply with: ok",
            "model": model or "default",
            "cwd": base_env.get("PWD") or os.getcwd(),
        }
    )
    node = base_env.get("PUPPETMASTER_NODE", "node")
    runner_path = runner or CURSOR_RUNNER
    try:
        completed = subprocess.run(
            [node, str(runner_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=base_env,
        )
        return (completed.returncode, completed.stdout or "", completed.stderr or "")
    except FileNotFoundError:
        return (127, "", "node not found")
    except subprocess.TimeoutExpired:
        return (124, "", "cursor live probe timed out")
    except Exception as exc:  # pragma: no cover - defensive
        return (1, "", str(exc))


def _verdict_from_probe(
    adapter: str,
    model: Optional[str],
    billing: str,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    ok_reason: str,
) -> PreflightResult:
    """Shared classification of a live-probe round-trip into a verdict.

    Blocks ONLY on a definitive provider rejection (``billing_or_quota`` /
    ``auth``). A probe that couldn't reach a verdict (missing optional CLI/node,
    timeout, network blip, non-billing error) degrades to an "unverified" pass
    so ``--live`` never false-blocks an account the static checks already
    cleared."""
    failure = classify_live_probe(adapter, returncode, f"{stdout}\n{stderr}")
    if failure is None:
        return PreflightResult(
            ok=True, adapter=adapter, model=model, billing=billing,
            reason=ok_reason,
            evidence=["live_probe:ok"],
        )
    if failure not in _BLOCKING_FAILURES:
        return PreflightResult(
            ok=True, adapter=adapter, model=model, billing=billing,
            reason=(
                f"live probe could not reach a verdict ({failure}, rc={returncode}); "
                "leaving static result in place (unverified)"
            ),
            evidence=["live_probe:skipped_unverified"],
        )
    return PreflightResult(
        ok=False, adapter=adapter, model=model, billing=billing,
        reason=f"live probe failed ({failure}); account cannot serve this model",
        evidence=[f"live_probe:{failure}"],
    )


# A probe only blocks on a definitive provider rejection; everything else
# degrades to an unverified pass (see _verdict_from_probe).
_BLOCKING_FAILURES = {"billing_or_quota", "auth"}


def _probe_openai(model: Optional[str], env: Mapping[str, str]) -> "tuple[int, str, str]":
    """Minimal chat completion against the OpenAI API to confirm the key has
    usable balance (a key can be valid but out of quota).

    The GPT-5+ family rejects the legacy ``max_tokens`` parameter (it wants
    ``max_completion_tokens``), so a naive probe using ``max_tokens`` returns a
    400 ``unsupported_parameter`` on a *fully funded* account and would falsely
    block it. We therefore default to ``max_completion_tokens`` (matching
    ``OpenAIAdapter``) and only fall back to legacy ``max_tokens`` if a provider
    explicitly rejects the new name — that keeps OpenAI-compatible endpoints
    that predate the rename working too."""
    import json
    import urllib.error
    import urllib.request

    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        return (1, "", "OPENAI_API_KEY not set")
    try:
        from puppetmaster.adapters import DEFAULT_OPENAI_MODEL as _default_model
    except Exception:  # pragma: no cover - defensive
        _default_model = "gpt-5.4-mini"
    base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resolved_model = model or _default_model

    def _post(token_param: str) -> "tuple[int, str, str]":
        body = json.dumps(
            {
                "model": resolved_model,
                "messages": [{"role": "user", "content": "ok"}],
                token_param: 16,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return (0, response.read().decode("utf-8", "replace"), "")
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read().decode("utf-8", "replace")
            except Exception:
                payload = ""
            return (exc.code, payload, f"HTTP {exc.code}")
        except Exception as exc:
            return (1, "", str(exc))

    rc, out, err = _post("max_completion_tokens")
    # Older OpenAI-compatible servers don't know max_completion_tokens; only
    # then retry with the legacy name. A genuine billing/auth/quota rejection
    # is preserved (we only retry on the parameter-name mismatch).
    if rc == 400 and "max_completion_tokens" in out and "unsupported_parameter" in out:
        return _post("max_tokens")
    return (rc, out, err)


def live_probe(
    adapter: str,
    model: Optional[str] = None,
    *,
    prober: Optional[LiveProber] = None,
    catalog_fetcher: Optional[CatalogFetcher] = None,
    env: Optional[Mapping[str, str]] = None,
    billing: str = "unknown",
) -> PreflightResult:
    """Run a minimal real call to prove ``adapter`` can actually serve a token.

    This is the catch for the failure that started it all: an OAuth/API login
    that *looks* authenticated but whose account balance is $0. Static
    detection passes it; only a real call surfaces ``billing_or_quota``. For
    plan-billed Cursor (no marginal balance to exhaust) the probe is the
    catalog fetch + model-availability check. Fully injectable for tests."""
    env = env if env is not None else os.environ

    if adapter == "cursor":
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            model_in_catalog,
        )

        fetch = catalog_fetcher or (lambda: __import__(
            "puppetmaster.cursor_discovery", fromlist=["fetch_cursor_catalog"]
        ).fetch_cursor_catalog())
        try:
            catalog = fetch()
        except CursorDiscoveryError as exc:
            return PreflightResult(
                ok=False, adapter=adapter, model=model, billing=billing,
                reason=f"live probe failed: {exc}",
                evidence=["live_probe:catalog_error"],
            )
        if model and not model_in_catalog(model, catalog):
            return PreflightResult(
                ok=False, adapter=adapter, model=model, billing=billing,
                reason=f"live probe: model {model!r} not in Cursor plan catalog",
                evidence=["live_probe:model_not_in_catalog"],
            )
        # Catalog cleared auth + model availability; now do a real 1-token
        # generation so a plan that's out of monthly allowance / rate-limited /
        # suspended is caught here, not mid-job (the plan-path equivalent of the
        # OAuth-balance-$0 case that motivated all this).
        if prober is not None:
            returncode, stdout, stderr = prober(adapter, model)
        else:
            returncode, stdout, stderr = _probe_cursor(model, env)
        return _verdict_from_probe(
            adapter, model, billing, returncode, stdout, stderr,
            ok_reason="live probe ok (Cursor plan served a token)",
        )

    if prober is not None:
        returncode, stdout, stderr = prober(adapter, model)
    elif adapter == "openai":
        returncode, stdout, stderr = _probe_openai(model, env)
    else:
        returncode, stdout, stderr = _default_prober(adapter, model)

    return _verdict_from_probe(
        adapter, model, billing, returncode, stdout, stderr,
        ok_reason="live probe ok (1-token call succeeded)",
    )


def preflight_check(
    adapter: str,
    model: Optional[str] = None,
    *,
    allow_api_billing: bool = True,
    live: bool = False,
    prober: Optional[LiveProber] = None,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    run: Optional[CommandRunner] = None,
    catalog_fetcher: Optional[CatalogFetcher] = None,
    billing_status: Optional[BillingStatus] = None,
) -> PreflightResult:
    """Return a :class:`PreflightResult` for ``adapter`` (and ``model``).

    Blocks when: the adapter has no usable credentials; it would bill an API
    account but ``allow_api_billing`` is False; or (Cursor only) the routed
    model isn't in the live plan catalog. Catalog lookup failures degrade to a
    pass with a note — we never block a run just because discovery was
    unavailable.
    """
    status = billing_status or detect_adapter_billing(
        adapter, env=env, home=home, run=run
    )

    if not status.healthy:
        return PreflightResult(
            ok=False,
            adapter=adapter,
            model=model,
            billing=status.billing,
            reason=f"adapter not ready: {status.detail}",
            evidence=[*status.evidence, "preflight:unhealthy"],
        )

    if status.billing == "api" and not allow_api_billing:
        return PreflightResult(
            ok=False,
            adapter=adapter,
            model=model,
            billing=status.billing,
            reason=(
                "api billing disabled (allow_api_billing=False) but "
                f"{adapter} is api-billed: {status.detail}"
            ),
            evidence=[*status.evidence, "preflight:api_blocked"],
        )

    if adapter == "cursor" and model and catalog_fetcher is not None:
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            model_in_catalog,
        )

        try:
            catalog = catalog_fetcher()
        except CursorDiscoveryError as exc:
            return PreflightResult(
                ok=True,
                adapter=adapter,
                model=model,
                billing=status.billing,
                reason=f"plan-billed; catalog unverified ({exc})",
                evidence=[*status.evidence, "preflight:catalog_unavailable"],
            )
        if not model_in_catalog(model, catalog):
            available = ", ".join(sorted(str(m.get("id")) for m in catalog)[:8])
            return PreflightResult(
                ok=False,
                adapter=adapter,
                model=model,
                billing=status.billing,
                reason=(
                    f"model {model!r} is not in the Cursor plan catalog; "
                    f"available: {available}"
                ),
                evidence=[*status.evidence, "preflight:model_not_in_catalog"],
            )

    # Optional live 1-token probe: the only thing that catches a funded-looking
    # account whose balance is actually exhausted. Off by default (adds a real
    # call + latency); opt in per task via ``payload.live_preflight`` or the
    # ``--live`` CLI flag. A probe that itself can't run (e.g. missing optional
    # tooling) does not block — only an actual billing/auth/quota rejection does.
    if live:
        probe = live_probe(
            adapter,
            model,
            prober=prober,
            catalog_fetcher=catalog_fetcher,
            env=env,
            billing=status.billing,
        )
        if not probe.ok:
            return PreflightResult(
                ok=False,
                adapter=adapter,
                model=model,
                billing=status.billing,
                reason=probe.reason,
                evidence=[*status.evidence, *probe.evidence],
            )
        return PreflightResult(
            ok=True,
            adapter=adapter,
            model=model,
            billing=status.billing,
            reason=f"ready ({status.billing}-billed, live-probed): {status.detail}",
            evidence=[*status.evidence, *probe.evidence],
        )

    return PreflightResult(
        ok=True,
        adapter=adapter,
        model=model,
        billing=status.billing,
        reason=f"ready ({status.billing}-billed): {status.detail}",
        evidence=[*status.evidence, "preflight:ok"],
    )
