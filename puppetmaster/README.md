# `puppetmaster/` source map

The Python package. This is an orientation map, not API docs — for the object model see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), for the CLI see [docs/CLI_REFERENCE.md](../docs/CLI_REFERENCE.md).

## Entry points

| Module | Role |
|---|---|
| `__main__.py` / `cli.py` | The `puppetmaster` CLI — every subcommand, arg wiring, read-only state-dir auto-pivot |
| `mcp_server.py` | The MCP server Cursor/Codex/Claude talk to: tool dispatch, long-poll, await, heartbeat |

## Orchestration runtime

| Module | Role |
|---|---|
| `orchestrator.py` | Builds the task DAG, routes, dispatches workers, runs auto-fallback, emits telemetry |
| `worker_runtime.py` | Per-worker process loop: lease a task, run the adapter, record status truthfully |
| `workers.py` | Worker/task specs, role definitions, recoverable-failure classification |
| `stitcher.py` | Reads typed artifacts (not stdout) into the final summary + Alerts section |

## Routing & models

| Module | Role |
|---|---|
| `router.py` | Task-aware model selection; billing-aware, policy-driven; writes `ROUTING` artifacts |
| `model_registry.py` | `ModelSpec`, the user registry at `~/.puppetmaster/models.json`, discovery metadata |
| `platform_billing.py` | Detects whether an adapter is plan-billed, API-billed, or unknown |
| `cursor_discovery.py` / `api_discovery.py` | Enumerate Cursor / OpenAI / Anthropic model catalogs |
| `preflight.py` | Static + live 1-token probe gating dispatch (catches `$0`-but-funded-looking accounts) |
| `adapters.py` | The `cursor` / `claude-code` / `openai` / `codex` / `shell` adapters + command builders |

## Storage & state

| Module | Role |
|---|---|
| `store.py` | `SwarmStore` abstract base — the storage seam |
| `sqlite_store.py` | Default SQLite implementation: WAL, O(1) cursor reads, durable events |
| `store_factory.py` | Backend selection (`file` / `sqlite`) |
| `models.py` | Job / Task / Artifact / AgentRun / MemoryRecord dataclasses + (de)serialization |
| `state.py` / `config.py` | State-dir resolution and config loading |

## Repo intelligence (CodeGraph)

| Module | Role |
|---|---|
| `codegraph.py` | Context injection, the cross-process advisory lock, search/context helpers |
| `codegraph_index_runner.py` / `codegraph_repair.py` | Background indexing + SQLite ABI repair |

## Integration & observability

| Module | Role |
|---|---|
| `installers.py` / `rules.py` | One-line MCP + agent-rule installers (`setup`, `install-*`) |
| `mcp_registry.py` | Tracks live MCP server processes; powers `mcp list` / `mcp cleanup` |
| `telemetry.py` | Optional OpenTelemetry (zero cost unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set) |
| `dashboard.py` | Zero-dependency local web board served from durable state |
| `diagnostics.py` | `doctor` checks (each crash-proofed into a Check result) |
