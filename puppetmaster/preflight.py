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


def preflight_check(
    adapter: str,
    model: Optional[str] = None,
    *,
    allow_api_billing: bool = True,
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

    return PreflightResult(
        ok=True,
        adapter=adapter,
        model=model,
        billing=status.billing,
        reason=f"ready ({status.billing}-billed): {status.detail}",
        evidence=[*status.evidence, "preflight:ok"],
    )
