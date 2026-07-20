# Puppetmaster docs

The full documentation set. Start at the [project README](../README.md) for the 60-second tour and install; come here when you want depth.

**Current release:** [v1.20.6](CHANGELOG.md#v1206) — cursor swarm `allowed_adapters` pin; see [CHANGELOG.md](CHANGELOG.md) for the full 1.20.x line.

## Start here

| Doc | What's in it |
|---|---|
| [WHY.md](WHY.md) | Design rationale: what shared-transcript subagents get wrong, what durable state fixes |
| [CLAIMS.md](CLAIMS.md) | The four headline claims with reproducible receipts from [`bench/`](../bench/) |
| [FEATURES.md](FEATURES.md) | Full feature matrix + the adapter table |
| [COMPARISON.md](COMPARISON.md) | How it differs from LangGraph / CrewAI / Claude Agent SDK / native subagents + "pick X instead if…" |
| [SECURITY.md](SECURITY.md) | Threat model: what it can do, what it touches, network egress, and how to run it safely |
| [DAILY_DRIVER.md](DAILY_DRIVER.md) | Prompt recipes for review, swarm, implement, post-job inspection |
| [DASHBOARD.md](DASHBOARD.md) | The live, zero-dependency web board: jobs index, routing rollup, and per-job detail |
| [MOBILE.md](MOBILE.md) | Watch swarms from your phone: Tailscale setup, agent/CLI start, QR handoff, troubleshooting |

## Research & external results

Reproducible evidence behind the durable-state thesis, including an independent third-party benchmark.

| Source | What's in it |
|---|---|
| [State, Not Tokens (paper site)](https://professorpalmer.github.io/durable-state-vs-context/) | The controlled JS→TS migration study + NL2Repo-Bench external validation: **91.1% mean test-pass, ~2.28× the ~40% published SOTA**, with honest limits (the K≈10–12 concurrency cap is a serving-platform property, not durable state) |
| [Zenodo record (citable, DOI)](https://doi.org/10.5281/zenodo.20709565) | Archived paper + self-contained source; concept DOI resolves to the latest version. Carries the model-attribution caveat and keeps packaging-bound tasks in the denominator |
| [SWE-bench Lite study](https://github.com/professorpalmer/swebench-pm) | 3-arm controlled cost/quality study on SWE-bench Lite (routing + CodeGraph + durable retries vs a frontier baseline) |

## Concepts & reference

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Job / Task / Worker / Artifact / Stitcher / Memory object model |
| [MODEL_ROUTING.md](MODEL_ROUTING.md) | Router policies, classifier, registry schema, the starter tiers |
| [ADAPTERS.md](ADAPTERS.md) | All production adapters (cursor, claude-code, openai, codex, hermes, agentic) + shell + how to add a new one |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Every CLI subcommand, workflow config schema, daemon mode |
| [CURSOR_AGENT_MCP.md](CURSOR_AGENT_MCP.md) | The MCP tool surface in detail |
| [CODEGRAPH.md](CODEGRAPH.md) | CodeGraph integration, bundled MCP tools, cost comparison |
| [OUTPUT_STYLE.md](OUTPUT_STYLE.md) | Optional Signal-maximizer worker output tiers (`terse` / `lithic`) |
| [COMPRESSION.md](COMPRESSION.md) | Why input-side compression isn't bundled + a bring-your-own recipe |

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

## Per-directory guides

| Directory | README |
|---|---|
| Source package | [`puppetmaster/`](../puppetmaster/README.md) |
| Benchmarks | [`bench/`](../bench/README.md) |
| Example workflow configs | [`examples/`](../examples/README.md) |
| Demo + GIF scripts | [`scripts/`](../scripts/README.md) |
| TypeScript await client | [`clients/typescript/`](../clients/typescript/README.md) |
