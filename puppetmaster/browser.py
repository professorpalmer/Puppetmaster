"""First-class browser-swarm support.

A *browser swarm* is N independent Hermes workers, each carrying the ``browser``
toolset, dispatched in parallel against a live site to capture real network
payloads — the QA shape that read-only repo analysis cannot reach. Static
analysis and mock-backend tests only see the response shapes you stub; a browser
worker hitting the real backend captures the actual ones (e.g. an HTTP 200 whose
body is a ``<JAD_ERROR>`` or an empty payload), which is what makes the fix
correct instead of a guess.

This is deliberately **not** built on the MCP swarm specs. Those hardcode a
``file,web,vision`` (analyze) / ``file,terminal,code_execution,web,vision``
(implement) toolset list with no ``browser`` in it, and the cursor swarm adapter
has no browser at all. The only adapter that can drive a real browser is Hermes
(``hermes chat -t browser``), so a browser worker is always a Hermes worker with
the toolset explicitly threaded through ``payload.toolsets`` — which
:class:`puppetmaster.adapters.hermes.HermesAdapter` already honors.

Three hard-won guardrails are baked into every browser worker, because a naive
``-t browser`` user re-derives them painfully:

1. **Private-URL / local-engine fallback.** Hermes' cloud browser
   (``cloud_provider: browser-use``) lives outside your network and cannot reach
   a VPN-only host. ``auto_local_for_private_urls: true`` makes the engine fall
   back to a local browser for private targets. Without it, an internal box is
   simply unreachable and the whole run is dead on arrival.
2. **Model-capability floor.** Browser grounding — mapping "click the Search
   button" to the right DOM node across React re-renders — is hard, and weak
   models fail it *and then lie about it*, reporting a false "login failed" that
   looks like an app bug. So every browser worker carries a high
   ``min_capability`` floor; the router will never select a cheap model for it.
3. **React-controlled-input trap.** Login/Search inputs are usually
   React-controlled. Automation that sets ``input.value`` directly leaves
   React's state empty, so submit fires nothing and no network request goes out
   — a fake, perfectly reproducible "bug". Real users typing fire the synthetic
   events, so it works for them. Automation must use the native value setter and
   dispatch ``input``/``change`` events. Every browser prompt says so, up front.

A browser worker is read-only on the *repository* (it edits no files), so
``swarm_mode`` stays ``"analysis"`` and no clean-tree guard applies. But it is an
**acting agent with external side effects** (navigation, logins, form fills
against a live system), so it carries ``payload.side_effecting = True`` and the
orchestrator gives the run an acting-agent banner instead of the swarm's "this is
just read-only analysis" framing. See :func:`puppetmaster.workers.spec_has_side_effects`.
"""
from __future__ import annotations

from typing import Iterable, Optional

from puppetmaster.workers import WorkerSpec

# Browser plus the analyze-mode defaults: the worker still needs ``file`` to
# read repo context, ``web`` for plain HTTP, and ``vision`` to reason over
# screenshots it captures. ``browser`` is the addition the MCP swarm path
# strips. The hermes adapter passes this straight to ``hermes chat -t``.
BROWSER_TOOLSETS = "file,web,vision,browser"

# Capability floor (registry ``capability_score`` is 0..100). Browser grounding
# needs a strong model; this reserves the strong tier and keeps the router from
# ever picking a cheap model that would fail-and-lie (guardrail #2). Tuned to sit
# at the "implement/architect" band rather than the absolute ceiling so a host
# whose strongest Hermes model is just below flagship still routes sensibly.
BROWSER_MIN_CAPABILITY = 80

# Browser flows are slow — a single live SearchServlet round-trip can take ~85s,
# and a full login→search→open flow chains several. Default well above the
# analyze-mode 600s so a legitimately slow real-backend flow isn't killed
# mid-capture.
DEFAULT_BROWSER_TIMEOUT_SECONDS = 1200

# Only Hermes can drive a browser, so a browser worker is pinned to it. Routing
# still runs (to pick the strongest sufficient Hermes model), but never off-adapter.
BROWSER_ADAPTER = "hermes"

BROWSER_GUARDRAILS = (
    "You are a browser-QA worker driving a REAL browser against a LIVE site to "
    "capture real network payloads. Obey these hard-won rules:\n"
    "1. Inputs are typically React-controlled. NEVER set `input.value` directly "
    "— that leaves React's state empty so Login/Search submit nothing and no "
    "request fires (a fake, reproducible 'bug'). Use the native value setter "
    "(`Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "
    "'value').set`) and then dispatch bubbling `input` AND `change` events so "
    "React's onChange runs. Drive rows/buttons with real DOM MouseEvents "
    "(mousedown/mouseup/click), not synthetic `.click()` shortcuts where it "
    "matters.\n"
    "2. Judge success by the NETWORK, not the UI. Capture the exact request URL, "
    "params, HTTP status, AND response body for every key action. Application "
    "errors routinely arrive as HTTP 200 with an error body (e.g. an empty body "
    "or an error envelope), so a 2xx status is NOT proof of success — read the "
    "body.\n"
    "3. If a login or action appears to fail, FIRST suspect your own automation "
    "(rule 1) before reporting an app defect. Reproduce with native events "
    "before claiming a bug.\n"
    "4. The target may be a private/VPN-only host; the browser engine should "
    "fall back to a local engine for private URLs. If navigation cannot reach "
    "the host at all, report that as an environment/reachability problem, not an "
    "app bug.\n"
    "Report concrete, evidence-backed findings: for each failure include the "
    "verbatim request, status, and response body.\n\n"
    "TASK:\n"
)


def browser_prompt(task: str) -> str:
    """Prefix a QA task with the three-guardrail browser preamble."""
    return BROWSER_GUARDRAILS + (task or "").strip()


def build_browser_spec(
    instruction: str,
    cwd: str,
    *,
    role: str = "browser",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: Optional[str] = None,
    min_capability: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    routing_policy: Optional[str] = None,
    auto_route: bool = True,
    allow_api_billing: bool = True,
    executable: Optional[str] = None,
) -> WorkerSpec:
    """Build one browser-QA worker spec (a Hermes analyze worker with browser).

    The worker emits structured findings (analyze mode) — it does not edit the
    repo — but it acts on the live world, so the spec is marked
    ``side_effecting``. Routing is pinned to the Hermes adapter with a strong
    ``min_capability`` floor unless an explicit ``model`` is pinned (a pin always
    wins over routing). The three guardrails are baked into the prompt.
    """
    payload: dict = {
        "prompt": browser_prompt(instruction),
        "cwd": cwd,
        "toolsets": toolsets or BROWSER_TOOLSETS,
        "side_effecting": True,
        "timeout_seconds": int(
            timeout_seconds if timeout_seconds is not None else DEFAULT_BROWSER_TIMEOUT_SECONDS
        ),
    }
    if executable:
        payload["executable"] = executable
    if provider:
        payload["provider"] = provider
    if model:
        # An explicit pin wins over routing — don't auto-route around it.
        payload["model"] = model
    elif auto_route:
        payload["auto_route"] = True
        payload["allowed_adapters"] = [BROWSER_ADAPTER]
        payload["min_capability"] = int(
            min_capability if min_capability is not None else BROWSER_MIN_CAPABILITY
        )
        payload["allow_api_billing"] = bool(allow_api_billing)
        if routing_policy:
            payload["routing_policy"] = routing_policy
    return WorkerSpec(
        role=role,
        instruction=instruction,
        adapter=BROWSER_ADAPTER,
        payload=payload,
    )


def browser_swarm_specs(
    tasks: Iterable[str],
    cwd: str,
    **kwargs,
) -> list[WorkerSpec]:
    """Fan a browser swarm out as one independent worker per task.

    Each task is a self-contained QA mission with no cross-worker dependency, so
    the workers run fully in parallel (subprocess worker-mode) without colliding
    — the correct shape for genuinely independent slices. Roles are made unique
    (``browser-1``, ``browser-2``, …) so artifacts attribute cleanly per worker.
    """
    task_list = [t for t in (str(t).strip() for t in tasks) if t]
    if not task_list:
        raise ValueError("browser_swarm_specs: at least one non-empty task is required")
    single = len(task_list) == 1
    specs: list[WorkerSpec] = []
    for index, task in enumerate(task_list, start=1):
        role = "browser" if single else f"browser-{index}"
        specs.append(build_browser_spec(task, cwd, role=role, **kwargs))
    return specs
