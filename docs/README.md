# Puppetmaster docs

The full documentation set. Start at the [project README](../README.md) for the 60-second tour and install; come here when you want depth.

## Start here

| Doc | What's in it |
|---|---|
| [WHY.md](WHY.md) | Design rationale: what shared-transcript subagents get wrong, what durable state fixes |
| [CLAIMS.md](CLAIMS.md) | The four headline claims with reproducible receipts from [`bench/`](../bench/) |
| [FEATURES.md](FEATURES.md) | Full feature matrix + the adapter table |
| [COMPARISON.md](COMPARISON.md) | How it differs from LangGraph / CrewAI / Claude Agent SDK / native subagents + "pick X instead if…" |
| [DAILY_DRIVER.md](DAILY_DRIVER.md) | Prompt recipes for review, swarm, implement, post-job inspection |

## Concepts & reference

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Job / Task / Worker / Artifact / Stitcher / Memory object model |
| [MODEL_ROUTING.md](MODEL_ROUTING.md) | Router policies, classifier, registry schema, the starter tiers |
| [ADAPTERS.md](ADAPTERS.md) | All four production adapters + shell + how to add a new one |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Every CLI subcommand, workflow config schema, daemon mode |
| [CURSOR_AGENT_MCP.md](CURSOR_AGENT_MCP.md) | The MCP tool surface in detail |
| [CURSOR_EXTENSION.md](CURSOR_EXTENSION.md) | Activity-bar control panel install + features |
| [CODEGRAPH.md](CODEGRAPH.md) | CodeGraph integration, bundled MCP tools, cost comparison |

## Operate & contribute

| Doc | What's in it |
|---|---|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | `Tool execution error. Not connected`, CodeGraph SQLite ABI, state-dir auto-pivot, safety model |
| [PRODUCTION.md](PRODUCTION.md) | Operating notes for non-toy use |
| [SECURITY.md](SECURITY.md) | Secret handling + reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to land a patch |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | Cutting a tagged release + PyPI upload |
| [CHANGELOG.md](CHANGELOG.md) | Versioned changes |
| [ROADMAP.md](ROADMAP.md) | What's next |
| [PYPI_NAME_REQUEST.md](PYPI_NAME_REQUEST.md) | The bare-`puppetmaster` name reassignment effort |

## Per-directory guides

| Directory | README |
|---|---|
| Source package | [`puppetmaster/`](../puppetmaster/README.md) |
| Benchmarks | [`bench/`](../bench/README.md) |
| Example workflow configs | [`examples/`](../examples/README.md) |
| Demo + GIF scripts | [`scripts/`](../scripts/README.md) |
| TypeScript await client | [`clients/typescript/`](../clients/typescript/README.md) |
| Cursor extension | [`cursor-extension/`](../cursor-extension/README.md) |
