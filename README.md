# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Puppetmaster turns Cursor (or Claude Code, or the OpenAI API, or the OpenAI Codex CLI) into an orchestrator that routes each task to the cheapest model that can handle it, stores worker outputs as typed SQLite artifacts so follow-ups cost zero tokens, and coordinates workers through durable state instead of a shared parent transcript.**

> Live OpenAI A/B with real billing tokens — same prompt, equivalent answer, one of 3 back-to-back runs:
> Pinned `gpt-5.5`: **\$0.006900 in 5.480 s** — Puppetmaster routed to `gpt-5.4-nano`: **\$0.000132 in 1.511 s** → **98.1% cheaper, 72.4% faster.** Reproduce: `OPENAI_API_KEY=... python -m bench.router_live_ab`.

## Install

```bash
pip install puppetmaster-ai
puppetmaster setup        # one-shot: doctor + models init + install-cursor-mcp + install-codex-mcp + install-rules
```

That's the whole install. `setup` runs every step idempotently, skips any tool that isn't present, and prints what it did at the end. Restart Cursor (or open a fresh Codex / Claude session) and the agent will see 32+ `puppetmaster_*` tools — plus an agent rule nudging it to reach for them on multi-file work.

**Pip name note:** PyPI lists this as [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/) because [PEP-503 name normalization](https://peps.python.org/pep-0503/#normalized-names) treats `puppetmaster` and `puppet-master` as the same name, and `puppet-master` is held by an [abandoned 2019 project](https://pypi.org/project/puppet-master/). The import name, CLI binary, GitHub repo, and brand all stay `puppetmaster`. Name-reassignment request in flight ([tracking doc](docs/PYPI_NAME_REQUEST.md)).

To run benchmarks or contribute, clone the repo instead:

```bash
git clone https://github.com/professorpalmer/Puppetmaster.git
cd Puppetmaster && python -m pip install -e . && npm install --package-lock=false --no-audit
OPENAI_API_KEY=... python -m bench.router_live_ab   # ~$0.01 of real spend, prints the ~98%-cheaper receipt
```

`bench/` and the Cursor extension source ship only with the cloned repo, not the pip wheel.

## What it does

Think **Redis/Gunicorn for agentic engineering**:

```text
Cursor Agent / Claude Code / OpenAI / Codex CLI / shell
        |
        v
Puppetmaster supervisor  ──>  task-aware model router (12 starter tiers)
        |
        v
independent worker processes  ──>  SQLite (typed artifacts, events, memory)
        |
        v
live artifact board  ──>  stitched summary  ──>  0-token follow-up reads
```

Puppetmaster is not trying to beat native IDE subagents at every tiny task. It is for the work that gets messy: long repo investigations, conflicting hypotheses, repeated handoffs, flaky memory, and code changes that need evidence, replay, and approval gates. The design rationale and the failure modes it fixes are in [docs/WHY.md](docs/WHY.md).

## Four claims, four receipts

Every number in this section comes from a reproducible script in [`bench/`](bench/). What is **not** defensible (and what we won't claim) lives in [TALKING_POINTS.md](TALKING_POINTS.md).

### 1. Token cost — fixed on two axes

**On new work** — the v0.6.0+ router classifies each task's complexity (role + instruction signal patterns + payload size) and picks the cheapest model from your user-owned registry that can handle it. Every routing decision is an auditable `ROUTING` artifact: picked model, capability needed, estimated cost, and the full list of *rejected* alternatives with the reason each was rejected.

- [`bench/router_savings.py`](bench/router_savings.py) — on a 6-task fixture, **35.1% cheaper** than pinning a frontier model. The wins come from *not* using a frontier model when the task doesn't need one.
- [`bench/router_live_ab.py`](bench/router_live_ab.py) — live OpenAI A/B with real `usage.prompt_tokens`: **98.1–98.7% cheaper across 3 consecutive runs** on a single explore task; wall-time savings between 68% and 88% per run.

**On follow-up work** — once a swarm completes, every artifact lives in SQLite. Follow-up questions are SQLite queries, not new agent runs.

- [`bench/followup_cost.py`](bench/followup_cost.py) — **40 follow-up queries against a real completed swarm: 0 adapter calls, 0 tokens, \$0.00, avg 0.5 ms per query.** Hypothetical "always-frontier replay" baseline for the same 40 queries: **\$1.64** at Anthropic's current Opus 4.7 rate ($5/$25 per MTok).

Honest scope: this is the *follow-up reads are free* claim. If your follow-up needs new reasoning the swarm didn't produce, that's a new task and it costs tokens like any other.

### 2. Transcript — workers don't share one

The classic multi-subagent shape stuffs everything into one parent chat. Each subagent inherits stale context, results come back as prose, and the context window bloats until the important details are buried.

Puppetmaster does the opposite. Workers don't see each other's transcripts. They claim tasks by lease, emit **typed artifacts** with payloads + `evidence` + `confidence` + `sha256` integrity, and the final stitcher reads JSON — not raw worker stdout. The parent agent's context only sees what the stitcher publishes.

- Inspect a live swarm: `puppetmaster artifacts <job_id>` — the actual coordination surface, not a chat scrollback.
- Inspect a completed swarm without paying tokens: same command, milliseconds, $0 (receipt #1 above).
- Verify nothing is hand-waved: every artifact carries `created_by`, `created_at`, and a content `sha256`.

### 3. Graphing — credit [CodeGraph](https://github.com/colbymchenry/codegraph), wire it in cleanly

The "graph your directories" capability is **not** a Puppetmaster feature. It's [CodeGraph](https://github.com/colbymchenry/codegraph) — a separate project — and it deserves the credit. Puppetmaster's contribution is what happens after CodeGraph is installed: every worker auto-injects task-relevant CodeGraph context into its prompt before the model call, one shared `codegraph context` query seeds N parallel workers, and the resulting artifacts land in the same durable store. Puppetmaster works fine without CodeGraph; workers fall back to grep/read. Details in [docs/CODEGRAPH.md](docs/CODEGRAPH.md).

### 4. Resilience — a dead provider doesn't kill the swarm (v0.9.0+)

The original failure that motivated this feature: a Claude Code worker hit a `$0` Anthropic balance and the job came back "degraded" with no useful output. Puppetmaster now treats that as a **recoverable** failure. The orchestrator classifies billing / quota / auth / missing-CLI failures, marks the task `FAILED` (not a silent `COMPLETE`), and **auto-re-routes it to the next funded adapter** under the same routing policy — preferring plan-billed models so the retry stays inside a subscription you already pay for.

Validated live end-to-end against real adapters (job `job_d82715bebc5d`):

```text
worker-implement (claude-code)  →  verification failure=billing_or_quota   [FAILED]
router-fallback                 →  reason="policy=balanced: cheapest sufficient
                                    model whose capability_score (78) >= needed (75)"
worker-implement (cursor/gpt-5.5)  →  verification result=passed  + finding @0.95  [COMPLETE]
```

The billing failure is also surfaced loudly in the stitched summary's **Alerts (action required)** section with concrete remediation, instead of being buried in a low-confidence verification line. Failure-classification, the `FAILED`-status propagation, and the reroute loop are covered by automated tests (`AutoFallbackTests`, `WorkerRuntimeFailureStatusTests`). A static + live `preflight` gate (`puppetmaster preflight --live`) catches missing keys, wrong billing mode, and even funded-looking-but-`$0` accounts *before* dispatch.

## Quickstart

After `pip install puppetmaster-ai && puppetmaster setup`, try one of these inside Cursor Agent or Codex:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

```text
Use Puppetmaster to start a cursor swarm for this repo and return the job id immediately.
Problem: users get logged out after refresh and token-refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

Or from the shell:

```bash
puppetmaster doctor
puppetmaster route "Security audit every endpoint" --role audit   # dry-run routing decision
puppetmaster cursor "Review this repo for release blockers" --review --dry-run
puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
puppetmaster show $(puppetmaster last)
```

More daily-driver prompts in [docs/DAILY_DRIVER.md](docs/DAILY_DRIVER.md).

## Adapters

Four production adapters live; eleven tiers in the starter registry (5 Cursor/Claude + 4 OpenAI + 2 Codex). Tier and pricing details in [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md); adapter wiring details in [docs/ADAPTERS.md](docs/ADAPTERS.md).

| Adapter | What it's for | Telemetry | Setup |
|---|---|---|---|
| `cursor` | Review / plan / dry-run via `@cursor/sdk` | tokens reported by SDK | `CURSOR_API_KEY` |
| `claude-code` | Full-edit workflows via the `claude` CLI | usage from CLI | `npm i -g @anthropic-ai/claude-code` + `ANTHROPIC_API_KEY` |
| `openai` | Direct Chat Completions (the most pricing-transparent path) | real `usage.prompt_tokens`/`completion_tokens` | `OPENAI_API_KEY` |
| `codex` | Full-edit via the OpenAI Codex CLI agent loop | `input_tokens` + `output_tokens` + `cached_input_tokens` + `reasoning_output_tokens` per turn | `npm i -g @openai/codex` + `codex login` |
| `shell` | Bounded verification commands | n/a | none |

## What works today

| Area | Status |
| --- | --- |
| Local runtime | Daily-driver beta: subprocess workers, task DAGs, leases, recovery, failure states |
| SQLite backend | Default, WAL mode, schema metadata, integrity checks, persisted events |
| Model router (v0.6.0+) | Task-aware routing; auditable `ROUTING` artifacts. Receipts: [`bench/`](bench/) |
| Billing-aware routing + auto-fallback (v0.9.0+) | Prefers plan-billed models; reroutes billing/quota/auth/missing-CLI failures to the next funded adapter. Validated live (claim #4) |
| Preflight gate (v0.9.0+) | Static checks (key/CLI/billing-mode) + optional live 1-token probe (`preflight --live`) catch zero-balance accounts before dispatch |
| Catalog discovery (v0.9.0+) | `models discover` enumerates Cursor / OpenAI / Anthropic catalogs; `doctor` nudges when a catalog goes stale |
| OpenTelemetry (optional, v0.9.0+) | Zero-cost unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set: per-task spans, job metrics, cross-process trace context. `pip install puppetmaster-ai[otel]` |
| Async await (v0.9.0+) | `puppetmaster await <job_id>` (CLI), `puppetmaster_await_job` (MCP), and a TypeScript blocking client in [`clients/typescript`](clients/typescript) |
| One-line MCP installers (v0.7.2+) | `install-cursor-mcp`, `install-codex-mcp` — resolve `sys.executable`, handshake before write, idempotent |
| One-line rule installer (v0.7.3+) | `install-rules` — Cursor `.mdc` + cross-tool `AGENTS.md` + global Codex/Claude rules, merge-don't-overwrite |
| `puppetmaster setup` (v0.7.3+) | One-shot wizard chaining doctor → models init → MCP installers → rules |
| Cursor Agent MCP | Async start tools, status polling, logs, live artifacts, partial summaries, routing tools |
| Cursor extension | Activity-bar control panel ([docs](docs/CURSOR_EXTENSION.md)) |
| Memory | Promoted memory retrieval into later worker context and prompts |
| CodeGraph | Optional shared repo intelligence ([docs](docs/CODEGRAPH.md)) |
| Patch workflow | Patch artifacts, path locks, approval/rejection events, dirty-worktree guard |
| Reproducible benchmarks | Six harnesses in [`bench/`](bench/), each with markdown + JSON receipts under `bench/results/` |

## Documentation

| Doc | What's in it |
|---|---|
| [docs/WHY.md](docs/WHY.md) | Design rationale: what shared-transcript subagents get wrong, what durable state fixes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Job / Task / Worker / Artifact / Stitcher / Memory object model |
| [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md) | Router policies, classifier, registry schema, the 12 starter tiers |
| [docs/CODEGRAPH.md](docs/CODEGRAPH.md) | CodeGraph integration, bundled MCP tools, cost comparison |
| [docs/ADAPTERS.md](docs/ADAPTERS.md) | All four production adapters + shell + how to add a new one |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Every CLI subcommand, workflow config schema, daemon mode |
| [docs/DAILY_DRIVER.md](docs/DAILY_DRIVER.md) | Prompt recipes for review, swarm, implement, post-job inspection |
| [docs/CURSOR_AGENT_MCP.md](docs/CURSOR_AGENT_MCP.md) | The MCP tool surface (32 tools) in detail |
| [docs/CURSOR_EXTENSION.md](docs/CURSOR_EXTENSION.md) | Activity-bar control panel install + features |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | `Tool execution error. Not connected`, CodeGraph SQLite ABI, state-dir auto-pivot, safety model |
| [docs/PRODUCTION.md](docs/PRODUCTION.md) | Operating notes for non-toy use |
| [docs/SECURITY.md](docs/SECURITY.md) | Secret handling + reporting |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to land a patch |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Versioned changes |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What's next |
| [docs/PYPI_NAME_REQUEST.md](docs/PYPI_NAME_REQUEST.md) | The bare-`puppetmaster` name reassignment effort |
| [TALKING_POINTS.md](TALKING_POINTS.md) | Truth-table separating "use this phrasing" from "avoid that overclaim" |

## Status

Puppetmaster is **daily-driver beta software**. The runtime contract is real, tests are automated, SQLite is the default backend, jobs fail closed, Cursor Agent MCP is live, the Cursor extension is installable, and Claude Code + Codex have both been validated as full-edit adapters that emit patch artifacts. It is credible for supervised local engineering workflows. It is not yet a hosted multi-user production service.

## License

MIT
