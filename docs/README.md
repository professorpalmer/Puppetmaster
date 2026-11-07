# Puppetmaster docs

This is the full documentation set. For the short tour and install, start at
the [project README](../README.md). For the changes in each release, see
[CHANGELOG.md](CHANGELOG.md).

## Get started

| Doc | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | Install, select platforms, run a first job, uninstall |
| [DAILY_DRIVER.md](DAILY_DRIVER.md) | Prompts for review, swarm, implement, and inspection |
| [GROK_BOT.md](GROK_BOT.md) | Grok Bot as a pilot over remote MCP. It is not a worker adapter |
| [PI.md](PI.md) | Pi as a pilot over stdio MCP. It is not a worker adapter |
| [OMP.md](OMP.md) | OMP (oh-my-pi) as a pilot. It is not a worker adapter |

## Run work

| Doc | What it covers |
|---|---|
| [FLOWS.md](FLOWS.md) | Flow graphs: one graph, one wake, resumable nodes, and `map` fan-out |
| [ADAPTERS.md](ADAPTERS.md) | Worker adapters (cursor, claude-code, openai, codex, hermes, antigravity, fx, agentic, shell) and how to add one |
| [MODEL_ROUTING.md](MODEL_ROUTING.md) | Router policies, classifier, registry schema, and starter tiers |
| [CONCURRENT_SESSIONS.md](CONCURRENT_SESSIONS.md) | Many sessions at once: project state, dashboard URLs, overrides, and isolated worktrees |
| [CODEGRAPH.md](CODEGRAPH.md) | Optional CodeGraph index, its MCP tools, and cost |
| [BUDGET_RESERVATIONS.md](BUDGET_RESERVATIONS.md) | Spend limits, CLI and MCP inputs, reservation lifecycle, and limits of enforcement |
| [FAILURE_POLICY.md](FAILURE_POLICY.md) | Task failure edges: `abort`, `continue`, `retry(n)` |
| [OUTPUT_STYLE.md](OUTPUT_STYLE.md) | Optional worker output style presets |
| [COMPRESSION.md](COMPRESSION.md) | Why input compression is not bundled, and a recipe to add your own |

## Watch and inspect

| Doc | What it covers |
|---|---|
| [DASHBOARD.md](DASHBOARD.md) | Live job board with no dependencies: job index, routing rollup, job detail |
| [MOBILE.md](MOBILE.md) | Watch jobs from a phone: Tailscale, start, QR handoff, troubleshooting |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Every CLI command, workflow config schema, and daemon mode |
| [CURSOR_AGENT_MCP.md](CURSOR_AGENT_MCP.md) | The MCP tool surface in detail |

## How it works

| Doc | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pilots, workers, and the job, task, artifact, stitcher, and memory objects |
| [WHY.md](WHY.md) | Why durable state, not one shared transcript |
| [COMPARISON.md](COMPARISON.md) | How it differs from LangGraph, CrewAI, the Claude Agent SDK, and native subagents |
| [FEATURES.md](FEATURES.md) | Pilots, adapters, and the full feature matrix |
| [STORE_CONTRACTS.md](STORE_CONTRACTS.md) | Embedding APIs for metadata, identity, completion, cancellation, and effects. Migration guarantees |
| [ATTEMPT_LEDGER.md](ATTEMPT_LEDGER.md) | Invocation facts, usage observations, and consumption reports |
| [ARTIFACT_STATUS.md](ARTIFACT_STATUS.md) | Artifact confidence against execution, grounding, claim support, and criterion status |

## Evidence

| Source | What it covers |
|---|---|
| [CLAIMS.md](CLAIMS.md) | The headline claims, with receipts you can run again from [`bench/`](../bench/) |
| [State, Not Tokens](https://professorpalmer.github.io/durable-state-vs-context/) | JS to TS migration study and NL2Repo-Bench: 91.1% mean test pass rate against a published baseline of about 40%. The page states the limits |
| [Zenodo record](https://doi.org/10.5281/zenodo.20709565) | Archived paper and source with a DOI |
| [SWE-bench Lite study](https://github.com/professorpalmer/swebench-pm) | Three-arm cost and quality study: routing, CodeGraph, and durable retries against a frontier baseline |
| [RESEARCH.md](RESEARCH.md) | Durable autoresearch: claims, leased runs, typed publication, and verification by a new run |

## Operate and contribute

| Doc | What it covers |
|---|---|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors: `Not connected`, CodeGraph SQLite ABI, state directory pivot |
| [PRODUCTION.md](PRODUCTION.md) | Notes for use past a local trial |
| [SECURITY.md](SECURITY.md) | Threat model, network egress, secret handling, and how to report an issue |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to land a patch |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | How to cut a tagged release and upload to PyPI |
| [CHANGELOG.md](CHANGELOG.md) | Changes in each release |
| [ROADMAP.md](ROADMAP.md) | Planned work |

## Directory guides

| Directory | Guide |
|---|---|
| Source package | [`puppetmaster/`](../puppetmaster/README.md) |
| Benchmarks | [`bench/`](../bench/README.md) |
| Example workflow configs | [`examples/`](../examples/README.md) |
| Demo and GIF scripts | [`scripts/`](../scripts/README.md) |
| TypeScript await client | [`clients/typescript/`](../clients/typescript/README.md) |
