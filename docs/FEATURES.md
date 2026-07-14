# Features & adapters

A full map of what ships today, plus the adapter matrix. For the design rationale behind these, see [docs/WHY.md](WHY.md); for the proof behind the headline claims, see [docs/CLAIMS.md](CLAIMS.md).

## Adapters

Five production adapters live plus the keys-only `agentic` standalone worker; eleven tiers in the starter registry (5 Cursor/Claude + 4 OpenAI + 2 Codex). Tier and pricing details in [docs/MODEL_ROUTING.md](MODEL_ROUTING.md); adapter wiring details in [docs/ADAPTERS.md](ADAPTERS.md).

| Adapter | What it's for | Telemetry | Setup |
|---|---|---|---|
| `cursor` | Review / plan / dry-run via `@cursor/sdk` | tokens reported by SDK | `CURSOR_API_KEY` |
| `claude-code` | Full-edit workflows via the `claude` CLI | usage from CLI | `npm i -g @anthropic-ai/claude-code` + `ANTHROPIC_API_KEY` |
| `openai` | Direct Chat Completions (the most pricing-transparent path) | real `usage.prompt_tokens`/`completion_tokens` | `OPENAI_API_KEY` |
| `codex` | Full-edit via the OpenAI Codex CLI agent loop | `input_tokens` + `output_tokens` + `cached_input_tokens` + `reasoning_output_tokens` per turn | `npm i -g @openai/codex` + `codex login` |
| `hermes` | Analyze + full-edit via the NousResearch Hermes CLI (`hermes chat`); auto-injects CodeGraph context, parses typed artifacts | exit-code- and diff-based success (Hermes exit codes are unreliable) | `pipx install hermes-agent` (or any `hermes` on PATH) + `puppetmaster install-hermes-mcp` |
| `agentic` | Keys-only analyze + full-edit via direct provider HTTP APIs (no external CLI) | tool-loop artifacts + PATCH on implement; tokens + cache reads + `price_job` / savings | any provider API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`) or AWS Bedrock IAM / `AWS_BEARER_TOKEN_BEDROCK` (`provider=bedrock`, Converse + ConverseStream + live catalog) |
| `shell` | Bounded verification commands | n/a | none |

## What works today

| Area | Status |
| --- | --- |
| Local runtime | Daily-driver beta: subprocess workers, task DAGs, leases, recovery, failure states |
| SQLite backend | Default, WAL mode, schema metadata, integrity checks, persisted events |
| Model router (v0.6.0+) | Task-aware routing; auditable `ROUTING` artifacts. Receipts: [`bench/`](../bench/) |
| Billing-aware routing + auto-fallback (v0.9.0+) | Prefers plan-billed models; reroutes billing/quota/auth/missing-CLI failures to the next funded adapter. Validated live ([claim #4](CLAIMS.md)) |
| Preflight gate (v0.9.0+) | Static checks (key/CLI/billing-mode) + optional live 1-token probe (`preflight --live`) — real round-trips for **every** adapter incl. plan/subscription (Cursor generation, Claude/Codex CLI) catch a funded-looking-but-exhausted account before dispatch |
| Catalog discovery (v0.9.0+) | `models discover` enumerates Cursor / OpenAI / Anthropic catalogs; `doctor` nudges when a catalog goes stale |
| Plan-first auto-discovery (v0.9.0+) | First `auto_route` job auto-merges the authenticated subscription's frontier so hard tasks stay in-plan (`$0`). Cursor self-enumerates; Claude Code / Codex use curated catalogs (`models discover --source claude\|codex`) since their CLIs can't list models |
| OpenTelemetry (optional, v0.9.0+) | Zero-cost unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set: per-task spans, job metrics, cross-process trace context. `pip install puppetmaster-ai[otel]` |
| Async await (v0.9.0+) | `puppetmaster await <job_id>` (CLI), `puppetmaster_await_job` (MCP), and a TypeScript blocking client in [`clients/typescript`](../clients/typescript) |
| One-line MCP installers (v0.7.2+) | `install-cursor-mcp`, `install-codex-mcp`, `install-claude-mcp` — resolve `sys.executable`, handshake before write, idempotent |
| One-line rule installer (v0.7.3+) | `install-rules` — Cursor `.mdc` + cross-tool `AGENTS.md` + global Codex/Claude rules, merge-don't-overwrite |
| `puppetmaster setup` (v0.7.3+) | One-shot wizard chaining doctor → models init → MCP installers → rules |
| Cursor Agent MCP | Async start tools, status polling, logs, live artifacts, partial summaries, routing tools |
| Memory | Promoted memory retrieval into later worker context and prompts; Wave 10 weighted ranking (scope, query overlap, recency); Wave 11 MMR diversity rerank so injected memory stays relevant without near-duplicates |
| Memory cost accounting (v1.10.0+) | Every memory injection logs estimated tokens and USD to the local savings ledger (`python -m puppetmaster savings`); disable with `PUPPETMASTER_MEMORY_COST_LOG=0` |
| Degraded-run honesty (v1.10.0+) | Empty agentic swarms and max-turns-with-no-findings runs classify as degraded with mitigation advice instead of looking successful in stitched summaries |
| Run receipts | `puppetmaster receipt <job_id>` / `puppetmaster_job_receipt` reports objective run-efficiency metrics: degraded tasks, typed artifacts (finding/risk/decision/patch), empty/unstructured signals, stdout-salvage markers, token totals, and tokens per typed artifact |
| Windows console hygiene (v1.10.0+) | Process-wide `CREATE_NO_WINDOW` default for child subprocesses under console-less hosts (Cursor MCP, workers, CodeGraph index). Escape hatch: `PUPPETMASTER_SHOW_CONSOLES=1` |
| CodeGraph | Optional shared repo intelligence ([docs](CODEGRAPH.md)) |
| Patch workflow | Patch artifacts, path locks, approval/rejection events, dirty-worktree guard |
| Reproducible benchmarks | Six harnesses in [`bench/`](../bench/), each with markdown + JSON receipts under `bench/results/` |
| Local dashboard (v0.9.0+) | `puppetmaster dashboard [<job_id>]` — zero-dependency live web board (task graph, typed artifacts, cost, auto-fallback reroutes, alerts) served from durable state; no OTLP collector required |
| Cross-platform CI | GitHub Actions matrix runs the full suite on Linux / macOS / Windows (Python 3.9 + 3.12), **all three required and green.** Getting Windows there fixed real defects: a leaked sqlite handle (Windows mandatory locks), POSIX-mode path splitting that mangled `C:\…` executables, a `fcntl`-only CodeGraph lock that was a no-op on Windows (now `msvcrt`), and a `doctor` that could crash on a bad CLI shim |
| AWS Bedrock agentic (v1.19.0+) | First-class `provider=bedrock` via stdlib Converse API (IAM/BYOK); live account model discovery; prompt-cache and token/cost parity with other providers (v1.19.1+) |
| Bedrock ConverseStream (v1.19.3+) | Live text/reasoning/toolUse deltas on the agentic streaming path; real eventstream parsing via stdlib — no boto3 |
| Plan-then-cheap prewalk | `puppetmaster prewalk "<goal>"` / `puppetmaster_start_prewalk`: quality-routed read-only plan worker, then cheap edit-capable implement (`depends_on_roles`); honest ROUTING per stage — `puppetmaster savings` counts both legs (plan quality as deliberate spend, implement cheap as savings) without double-count |
| Swarm role routing defaults | Built-in analysis roles stamp per-role `routing_policy` under `auto_route` (explore/test → cheap, architect/plan → balanced, redteam/review/audit → quality) — no frontier model pins; MCP-generated swarms share the same map |

## Status

Puppetmaster is **daily-driver beta software**. The runtime contract is real, tests are automated, SQLite is the default backend, jobs fail closed, Cursor Agent MCP is live, and Claude Code + Codex have both been validated as full-edit adapters that emit patch artifacts. Hermes is wired as a fifth adapter (analyze + full-edit), with an end-to-end analyze run validated through the orchestrator (typed findings, CodeGraph context injection, stitched summary). It is credible for supervised local engineering workflows. It is not yet a hosted multi-user production service.
