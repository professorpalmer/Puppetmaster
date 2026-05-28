# Talking Points: what's defensible vs. what isn't

The LinkedIn pitch for Puppetmaster makes several quantitative-sounding claims. This file is the honest version, paired with the benchmark receipts that justify each line.

If you're about to post about Puppetmaster, lift phrasing from the **defensible** column. The **avoid** column is what gets fact-checked.

## The truth table

| Defensible (use this) | Avoid (overclaim) | Why / receipt |
|---|---|---|
| "Routes each task to the cheapest model that can handle it. On a live OpenAI A/B (real `usage.prompt_tokens` from the API), a simple task ran for **$0.000132** through the router vs **$0.006900** pinned to GPT-5.5 — **98.1% cheaper, 72.4% faster** per the 2026-05-28 receipt. Across 3 consecutive runs the cost ratio held at 98.1–98.7%, wall-time savings ranged 68–88%." | "Token costs? Solved." | `bench/router_live_ab.py` produces a real billing receipt. "Solved" overstates a 98% reduction on one task into a universal claim. The router does nothing about your IDE token costs; it controls per-task model selection for swarm workers. |
| "Follow-up questions about a *completed* swarm cost zero model tokens — they're SQLite queries against typed artifacts. Measured at 0.5 ms per query, 40 queries = $0.00." | "Token costs? Solved." (as a general claim) | `bench/followup_cost.py` proves this for the narrow case of follow-up reads. New reasoning still needs a new task — that's a fresh model call. |
| "Workers coordinate through durable state instead of a shared parent transcript, so the parent agent's context window doesn't bloat with worker chatter." | "Context issues? Solved." | Architectural fact. Visible in the SQLite store (`puppetmaster artifacts <job_id>`). Not a silver bullet for long-context tasks; it sidesteps one specific failure mode (transcript collapse across many subagents). |
| "On a 6-task fixture spanning easy/medium/hard, Puppetmaster's router was 35.1% cheaper than pinning the frontier model — and on hard tasks it correctly stayed on the frontier model. The savings come from *not* using a frontier model when the task doesn't need one." | "Faster, cheaper, more accurate than Cursor alone." | `bench/router_savings.py` measures this. Note that "frontier baseline" ≠ "Cursor alone" — Cursor's bundled models are $0 because they roll into your Cursor plan. The router's wins are largest against a user who pins a paid frontier model as their default. |
| "Routes by *task complexity* using a transparent classifier (role + payload + signal patterns) plus user-asserted capability scores per model. Every routing decision is logged as a `routing` artifact you can inspect with `puppetmaster artifacts`." | "Delegates to models based on complexity" *(no receipts)* | The classifier is in `puppetmaster/router.py`. Every routing decision is auditable; nothing is black-box. |
| "Pairs with [CodeGraph](https://github.com/colbymchenry/codegraph) — a separate project — for symbol-level repo context that gets pre-injected into every worker prompt." | "Graphs your directories to index them for 0 token cost follow ups." | CodeGraph is not a Puppetmaster feature. Credit it. Puppetmaster's contribution is the auto-injection plus the durable state that makes follow-up reads free. |
| "MIT licensed, runs locally, plays with Cursor SDK, Claude Code CLI, the OpenAI API, and the official OpenAI Codex CLI — four production adapters." | "Open source." | Same fact, less throwaway. v0.7.0 added the Codex CLI adapter; v0.6.1-beta.1 added the OpenAI Chat Completions adapter. |

## What is NOT currently measurable

These claims would be nice but cannot be defended without a real eval set. Don't make them yet.

- **"More accurate"** — no graded eval; the live A/B above returned equivalent answers but that's one task. The honest substitute is *"more structured"*: every worker emits typed artifacts with `evidence` + `confidence`.
- **"Faster on every task"** — wall-clock A/B exists for one OpenAI call. For multi-worker swarms, latency is bounded by the slowest worker; whether that's faster than a single big call depends on parallelism.
- **"Saves money on all workloads"** — false for any workload that needs frontier reasoning end-to-end. The router correctly picks frontier for audit/architect tasks, so the savings collapse for those.
- **"Replaces your agent"** — Puppetmaster orchestrates *your existing* agents (Cursor, Claude Code, OpenAI). It doesn't replace them.

## Reproducing the receipts

Every number in the truth table comes from a script in [`bench/`](bench/). All three write `bench/results/<harness>_<timestamp>.{md,json}` so the receipts are auditable.

```bash
# Dry-run — no API key, no money
python -m bench.router_savings

# Dry-run — uses your most recent completed Puppetmaster job
python -m bench.followup_cost --queries 10

# Live — requires OPENAI_API_KEY, costs roughly $0.02 of real spend
OPENAI_API_KEY=sk-... python -m bench.router_live_ab
```

If you re-run these with your own registry / your own task, please update the truth table above with the new numbers rather than the old ones. The whole point of receipts is that they're current.

## A rewritten LinkedIn-ready version

```
Puppetmaster routes each task to the cheapest model that can handle it.

Receipt: on a live OpenAI A/B with real billing tokens, a simple
explore task ran for $0.000132 through the router vs $0.006900
pinned to GPT-5.5 — 98.1% cheaper, 72.4% faster, equivalent answer.
Across 3 consecutive runs the cost ratio held 98.1–98.7%.

Routing is transparent: every decision is logged as an auditable
artifact. On hard tasks (audits, architecture), the router correctly
stays on the frontier model. The savings come from NOT using a
frontier model when the task doesn't need one.

Also: follow-up questions about a completed swarm cost zero model
tokens — they're SQLite queries against typed artifacts. 40 queries
benchmarked at 0.5 ms each, $0.00 total.

Open source (MIT), runs locally. Four production adapters live:
Cursor SDK, Claude Code CLI, OpenAI Chat Completions, and the
official OpenAI Codex CLI (v0.7.0). Reproducible benchmarks in bench/.

https://github.com/professorpalmer/Puppetmaster
```

Same shape, same energy, but every quantitative claim has a script behind it.
