# Four claims, four receipts

Every number here comes from a reproducible script in [`bench/`](../bench/).

## 1. Token cost — fixed on two axes

**On new work** — the v0.6.0+ router classifies each task's complexity (role + instruction signal patterns + payload size) and picks the cheapest model from your user-owned registry that can handle it. Every routing decision is an auditable `ROUTING` artifact: picked model, capability needed, estimated cost, and the full list of *rejected* alternatives with the reason each was rejected.

- [`bench/router_savings.py`](../bench/router_savings.py) — on a 6-task fixture, **35.1% cheaper** than pinning a frontier model. The wins come from *not* using a frontier model when the task doesn't need one.
- [`bench/router_live_ab.py`](../bench/router_live_ab.py) — live OpenAI A/B with real `usage.prompt_tokens`: **98.1–98.8% cheaper across consecutive runs** on a single explore task; wall-time savings between 68% and 88% per run.

**On follow-up work** — once a swarm completes, every artifact lives in SQLite. Follow-up questions are SQLite queries, not new agent runs.

- [`bench/followup_cost.py`](../bench/followup_cost.py) — **40 follow-up queries against a real completed swarm: 0 adapter calls, 0 tokens, \$0.00, avg 0.5 ms per query.** Hypothetical "always-frontier replay" baseline for the same 40 queries: **\$1.64** at Anthropic's current Opus 4.7 rate ($5/$25 per MTok).

Honest scope: this is the *follow-up reads are free* claim. If your follow-up needs new reasoning the swarm didn't produce, that's a new task and it costs tokens like any other.

## 2. Transcript — workers don't share one

The classic multi-subagent shape stuffs everything into one parent chat. Each subagent inherits stale context, results come back as prose, and the context window bloats until the important details are buried.

Puppetmaster does the opposite. Workers don't see each other's transcripts. They claim tasks by lease, emit **typed artifacts** with payloads + `evidence` + `confidence` + `sha256` integrity, and the final stitcher reads JSON — not raw worker stdout. The parent agent's context only sees what the stitcher publishes.

- Inspect a live swarm: `puppetmaster artifacts <job_id>` — the actual coordination surface, not a chat scrollback.
- Inspect a completed swarm without paying tokens: same command, milliseconds, $0 (receipt #1 above).
- Verify nothing is hand-waved: every artifact carries `created_by`, `created_at`, and a content `sha256`.

## 3. Graphing — credit [CodeGraph](https://github.com/colbymchenry/codegraph), wire it in cleanly

The "graph your directories" capability is **not** a Puppetmaster feature. It's [CodeGraph](https://github.com/colbymchenry/codegraph) — a separate project — and it deserves the credit. Puppetmaster's contribution is what happens after CodeGraph is installed: every worker auto-injects task-relevant CodeGraph context into its prompt before the model call, one shared `codegraph context` query seeds N parallel workers, and the resulting artifacts land in the same durable store. Puppetmaster works fine without CodeGraph; workers fall back to grep/read. Details in [docs/CODEGRAPH.md](CODEGRAPH.md).

## 4. Resilience — a dead provider doesn't kill the swarm (v0.9.0+)

The original failure that motivated this feature: a Claude Code worker hit a `$0` Anthropic balance and the job came back "degraded" with no useful output. Puppetmaster now treats that as a **recoverable** failure. The orchestrator classifies billing / quota / auth / missing-CLI failures, marks the task `FAILED` (not a silent `COMPLETE`), and **auto-re-routes it to the next funded adapter** under the same routing policy — preferring plan-billed models so the retry stays inside a subscription you already pay for.

Validated live end-to-end against real adapters (job `job_d82715bebc5d`):

```text
worker-implement (claude-code)  →  verification failure=billing_or_quota   [FAILED]
router-fallback                 →  reason="policy=balanced: cheapest sufficient
                                    model whose capability_score (78) >= needed (75)"
worker-implement (cursor/gpt-5.5)  →  verification result=passed  + finding @0.95  [COMPLETE]
```

The billing failure is also surfaced loudly in the stitched summary's **Alerts (action required)** section with concrete remediation, instead of being buried in a low-confidence verification line. Failure-classification, the `FAILED`-status propagation, and the reroute loop are covered by automated tests (`AutoFallbackTests`, `WorkerRuntimeFailureStatusTests`). A static + live `preflight` gate (`puppetmaster preflight --live`) catches missing keys, wrong billing mode, and even funded-looking-but-`$0` accounts *before* dispatch.
