# Features & adapters

A full map of what ships today, plus the adapter matrix. For the design rationale behind these, see [docs/WHY.md](WHY.md); for the proof behind the headline claims, see [docs/CLAIMS.md](CLAIMS.md).

## Adapters

Five production adapters live plus the keys-only `agentic` standalone worker; twelve tiers in the starter registry (6 Cursor/Claude + 4 OpenAI + 2 Codex, including the Grok 4.6 Cursor overlay added in v1.22.3). Tier and pricing details in [docs/MODEL_ROUTING.md](MODEL_ROUTING.md); adapter wiring details in [docs/ADAPTERS.md](ADAPTERS.md).

| Adapter | What it's for | Telemetry | Setup |
|---|---|---|---|
| `cursor` | Review / plan / dry-run via `@cursor/sdk` | tokens reported by SDK | `CURSOR_API_KEY` |
| `claude-code` | Full-edit workflows via the `claude` CLI | usage from CLI | `npm i -g @anthropic-ai/claude-code` + `ANTHROPIC_API_KEY` |
| `openai` | Direct Chat Completions (the most pricing-transparent path) | real `usage.prompt_tokens`/`completion_tokens` | `OPENAI_API_KEY` |
| `codex` | Full-edit via the OpenAI Codex CLI agent loop | `input_tokens` + `output_tokens` + `cached_input_tokens` + `reasoning_output_tokens` per turn | `npm i -g @openai/codex` + `codex login` |
| `hermes` | Analyze + full-edit via the NousResearch Hermes CLI (`hermes chat`); preferred live-site browser adapter (`hermes chat -t browser`); auto-injects CodeGraph context, parses typed artifacts | exit-code- and diff-based success (Hermes exit codes are unreliable) | `pipx install hermes-agent` (or any `hermes` on PATH) + `puppetmaster install-hermes-mcp` |
| `agentic` | Keys-only analyze + full-edit via direct provider HTTP APIs (no external CLI); browser-swarm fallback via stdlib CDP (OpenRouter / standalone keys) | tool-loop artifacts + PATCH on implement; tokens + cache reads + `price_job` / savings | any provider API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OPENCODE_GO_API_KEY`, `ZAI_API_KEY`/`GLM_API_KEY`/`Z_AI_API_KEY`, `MINIMAX_API_KEY`, `NVIDIA_API_KEY`) or AWS Bedrock IAM / `AWS_BEARER_TOKEN_BEDROCK` (`provider=bedrock`, Converse + ConverseStream + live catalog) |
| `shell` | Bounded verification commands | n/a | none |

## What works today

| Area | Status |
| --- | --- |
| Local runtime | Daily-driver beta: subprocess workers, task DAGs, leases, recovery, failure states |
| SQLite backend | Default, WAL mode, schema metadata, integrity checks, persisted events |
| Model router (v0.6.0+; role scorecards v1.22.8+) | Task-aware routing; auditable `ROUTING` artifacts. Optional per-role cards override `capability_score` for one role/adapter; provenance on `route` / ROUTING. Receipts: [`bench/`](../bench/) |
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
| Grok Bot remote MCP (v1.22.0+) | Streamable HTTP (+ legacy SSE) wrapping the same tool handlers for remote-only pilots; bearer auth; default supervise scope omits implement/edit — see [GROK_BOT.md](GROK_BOT.md). Not a `grok-bot` worker adapter |
| Browser swarm (v1.21+; agentic parity v1.22.6+) | Live-site QA via `puppetmaster browser` / `puppetmaster_start_browser_swarm` with React-input / network-truth / strong-model / acting-agent guardrails. Hermes is preferred; `--adapter agentic` uses the stdlib CDP engine (OpenRouter / standalone keys). Cursor/Claude/Codex still have no headless browser toolset |
| Durable autoresearch (v1.22.6+) | `puppetmaster research` coordinates fingerprinted claims, lease-backed harness runs, typed result publishing, zero-token artifact recall, leaderboards, and verification by re-run. v1 ships the deterministic CPU `ToyCompressionHarness`; GPU/nanochat runners can implement the same protocol later |
| Memory | Promoted memory retrieval into later worker context and prompts; Wave 10 weighted ranking (scope, query overlap, recency); Wave 11 MMR diversity rerank so injected memory stays relevant without near-duplicates |
| Memory cost accounting (v1.10.0+) | Every memory injection logs estimated tokens and USD to the local savings ledger (`python -m puppetmaster savings`); disable with `PUPPETMASTER_MEMORY_COST_LOG=0` |
| Degraded-run honesty (v1.10.0+) | Empty agentic swarms and max-turns-with-no-findings runs classify as degraded with mitigation advice instead of looking successful in stitched summaries |
| Run receipts | `puppetmaster receipt <job_id>` / `puppetmaster_job_receipt` reports objective run-efficiency metrics: degraded tasks, typed artifacts (finding/risk/decision/patch), empty/unstructured signals, stdout-salvage markers, token totals, and tokens per typed artifact |
| Windows console hygiene (v1.10.0+) | Process-wide `CREATE_NO_WINDOW` default for child subprocesses under console-less hosts (Cursor MCP, workers, CodeGraph index). Escape hatch: `PUPPETMASTER_SHOW_CONSOLES=1` |
| Windows stdin prompts (v1.22.7+) | Codex and Claude Code send the enriched prompt on stdin (`codex exec -`, `claude --print`) instead of argv, so `CreateProcess` no longer fails with WinError 206 past 32767 characters. Cursor already used `PUPPETMASTER_CURSOR_INPUT`. Hermes `chat -q` is unchanged until its stdin contract is confirmed |
| CodeGraph | Optional shared repo intelligence ([docs](CODEGRAPH.md)) |
| Patch workflow | Patch artifacts, path locks, approval/rejection events, dirty-worktree guard |
| Reproducible benchmarks | Six harnesses in [`bench/`](../bench/), each with markdown + JSON receipts under `bench/results/` |
| Local dashboard (v0.9.0+) | `puppetmaster dashboard [<job_id>]` — zero-dependency live web board (task graph, typed artifacts, cost, auto-fallback reroutes, alerts) served from durable state; no OTLP collector required |
| Shared verified context (v1.22.2+) | DeLM-inspired durable gists: admission-gated peer injection, selective gist→summary→raw unfold, bounded adaptive `enqueue_subtask`, dashboard **Frontier** + Gists panel; compounds with SQLite artifacts / provenance edges |
| Cross-platform CI | GitHub Actions matrix runs the full suite on Linux / macOS / Windows (Python 3.9 + 3.12), **all three required and green.** Getting Windows there fixed real defects: a leaked sqlite handle (Windows mandatory locks), POSIX-mode path splitting that mangled `C:\…` executables, a `fcntl`-only CodeGraph lock that was a no-op on Windows (now `msvcrt`), and a `doctor` that could crash on a bad CLI shim |
| AWS Bedrock agentic (v1.19.0+) | First-class `provider=bedrock` via stdlib Converse API (IAM/BYOK); live account model discovery; prompt-cache and token/cost parity with other providers (v1.19.1+) |
| OpenCode Go agentic (v1.21.9+) | First-class `provider=opencode-go` with `OPENCODE_GO_API_KEY` against the Go subscription endpoint; flat ids + per-model `/chat/completions`, `/messages`, or `/responses` (GPT 5.6 Luna) routing; upstream 401 blocks classified separately from invalid keys |
| Marionette-keyed providers (v1.21.12+) | First-class `provider=zai` / `minimax` / `nvidia` descriptors (`ZAI_API_KEY`·`GLM_API_KEY`·`Z_AI_API_KEY`, `MINIMAX_API_KEY`, `NVIDIA_API_KEY`) so keyed Marionette pilots can drive agentic workers |
| Passive rate-limit harvest (v1.21.13+) | Agentic HTTP paths harvest OpenAI/Anthropic `x-ratelimit-*` headers into `~/.puppetmaster/rate_limits.sqlite3` and preempt calls when `remaining=0` (Retry-After is diagnostic; multi-key rotation preserved). Doctor surfaces exhausted windows; `puppetmaster keys` rejects doubled pastes |
| Bedrock ConverseStream (v1.19.3+) | Live text/reasoning/toolUse deltas on the agentic streaming path; real eventstream parsing via stdlib — no boto3 |
| Verified Bedrock routing (v1.20.3+) | Auto-routing requires a successful runtime invoke, not credential-file presence. Credential-scoped health persists across processes; auth failures reroute away, while doctor/discovery report present, verified, or denied truthfully without network work on the routing hot path |
| Plan-then-cheap prewalk | `puppetmaster prewalk "<goal>"` / `puppetmaster_start_prewalk`: quality-routed read-only plan worker, then cheap edit-capable implement (`depends_on_roles`); honest ROUTING per stage — `puppetmaster savings` counts both legs (plan quality as deliberate spend, implement cheap as savings) without double-count |
| Swarm role routing defaults | Built-in analysis roles stamp per-role `routing_policy` under `auto_route` (explore/test → cheap, architect/plan → balanced, redteam/review/audit → quality) — no frontier model pins; MCP-generated swarms share the same map |
| Durable execution graph (v1.20.0+) | Typed provenance edges (`depends_on` / `produces` / `consumes`), SQLite v1→v2 migration + file-store lazy materialization, plan→implement→verify prewalk with edge handoff, hard-failure propagation, targeted reruns; read-only `puppetmaster graph` / `puppetmaster_job_graph` |
| Dependency-aware reusable validation (v1.21.3+) | Deterministic validation fingerprint (HEAD + scoped source bytes + rules/evaluator digest); additive `payload.validation` (`fresh`/`reused`/`stale`/`superseded`); fingerprint lookup + `record_derived_from`; CLI/MCP `reset-subgraph` with superseded labeling; `artifacts --refs` / MCP `refs=true` compact refs |
| Quality control (v1.21.5+) | Stitch-time contradiction conflicts with source-safe evidence checks; additive `payload.execution_provenance` on typed findings/risks/decisions/verifications (explicit unknown cost/usage); acceptance criteria preserved through adapter prompts with per-criterion verification status; v1.21.6 adds structured agentic `submit_findings` criterion rows, fail-closed evidence normalization, and non-raising verification fallback; v1.21.7 adds Codex flat artifact normalization with malformed/ambiguous-result guards |
| Cross-adapter model pins (v1.20.0+) | Explicit `--model` / MCP `model` pins apply on non-Cursor adapters (not Cursor-only); ambiguous Cursor pins return structured preflight/blocked errors. v1.22.5+: an exact enabled registry id wins over aliases so API vs Codex siblings that share an adapter model name stay distinguishable |
| Agentic pin defaults + max-turns submit (v1.20.8+) | Bare agentic `--model` merges registry `payload_defaults` (e.g. `provider=openrouter`); analyze grants one forced `submit_findings` after max_turns; empty silent runs stay degraded |
| Codex Responses effort map + unstructured promote (v1.21.14+) | Agentic openai-codex maps Chat Completions ``reasoning_effort`` into nested Responses ``reasoning`` (no bare-key 400); analyze promotes labeled/file-cited prose into typed FINDING/RISK/DECISION when submit is skipped. v1.22.5+: direct OpenAI Chat Completions sends ``none`` for GPT-5.6 tool requests (including alias ``gpt-5.6``) and omits the field on older GPT-5 |
| Model allowlists + bounded reroute (v1.20.2+) | CLI/MCP jobs can constrain routing to explicit model identities; empty allowlists fail closed, disabled catalog overlays remain disabled, and generic Cursor `status:error` gets at most one permitted same-adapter alternate |
| Windows UTF-8 / console safety (v1.20.0+) | Agent subprocess stdout/stderr decoded as UTF-8 with replace (avoids cp1252 reader crashes); existing `CREATE_NO_WINDOW` child-console hygiene unchanged |

## Durable execution graph

Jobs remain a task DAG scheduled by `Task.depends_on`. Alongside that scheduling field, the store persists typed provenance edges so downstream stages can resolve what an upstream task produced without reloading every artifact in the job:

| Edge | Meaning |
| --- | --- |
| `depends_on` | Task → task scheduling edge (kept in sync with `Task.depends_on`) |
| `produces` | Task → artifact (emitted when an artifact is saved) |
| `consumes` | Task → artifact (recorded when a worker resolves upstream outputs via edges) |
| `derived_from` | Artifact → artifact lineage when a finding/decision/verification is reused (`record_derived_from`) |

**Persistence.** SQLite schema version 2 adds a `graph_edges` table and migrates v1 databases on open (eager backfill of `depends_on` + `produces`). The file store lazy-materializes the same edges on first `job_graph` / ensure pass, with a crash-recoverable journal for `consumes` batches.

**Runtime.** Prewalk is plan → implement → verify: implement consumes plan artifacts via edges, verify consumes implement outputs the same way. Hard (non-recoverable) dependency failures cascade onto blocked descendants as terminal FAILED. Targeted subgraph reset clears selected tasks (and optionally their consumer closure) for rerun while retaining completed upstream work, artifacts, and edges — and refuses the whole reset if any selected task still holds an active (non-expired RUNNING) lease. Reset also stamps prior produced outputs with additive `payload.validation.status=superseded` so they are not presented as fresh.

**Reusable validation.** `puppetmaster.validation.compute_validation_fingerprint` builds a reuse key from HEAD + scoped working-tree bytes + rules/evaluator digests (separate from `Artifact.sha256`). Stores look up reusable FINDING/VERIFICATION/DECISION artifacts via `lookup_artifacts_by_validation_fingerprint` (bounded; excludes stale/superseded).

**Introspection.** Read-only snapshots: CLI `puppetmaster graph <job_id>` and MCP `puppetmaster_job_graph`. Mutating selective reset: CLI `puppetmaster reset-subgraph <job_id> --task <id>...` and MCP `puppetmaster_reset_subgraph`. Compact artifact listing: CLI `artifacts --refs` / MCP `refs=true`. No hosted or multi-user graph service is implied; this is local durable state for a single operator runtime.

## Status

Puppetmaster is **daily-driver beta software** (currently **v1.22.8**). The runtime contract is real, tests are automated, SQLite is the default backend, jobs fail closed, Cursor Agent MCP is live, Grok Bot can supervise via authenticated remote MCP (streamable HTTP; not a worker adapter), and Claude Code + Codex have both been validated as full-edit adapters that emit patch artifacts. Hermes is wired as a fifth adapter (analyze + full-edit) and remains the preferred live-site browser adapter; agentic/OpenRouter can drive the same browser swarm through the stdlib CDP engine. The 1.21.x–1.22.x line adds dependency-aware reusable validation, collision-proof centralized run IDs, deterministic quality control (contradiction conflicts, execution provenance, acceptance criteria, agentic criterion reporting), fail-fast model-backed auto-routing, Codex flat artifact normalization, Marionette-keyed Z.AI / MiniMax / NVIDIA NIM agentic providers, passive provider rate-limit harvest, the Grok Bot remote MCP pilot path, shared verified context (admitted gists / Frontier), and agentic browser-swarm parity without breaking file/SQLite/stdio-MCP contracts. It is credible for supervised local engineering workflows. It is not yet a hosted multi-user production service.
