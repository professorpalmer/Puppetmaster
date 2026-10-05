# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/pyproject.toml)

Puppetmaster runs multi-step engineering work through the agent tools you already use: Cursor, Grok Bot, Claude Code, Codex, Hermes, Antigravity (Gemini 3.7 / 3.6 / 3.5 / 3.1 Pro), or a provider API. It starts independent workers, routes tasks to an available model, and stores their typed results in SQLite so jobs can be inspected and resumed. It is aimed at developers who want durable state and reviewable output for repository investigations, audits, refactors, and implementations.

**Grok Bot:** only remote MCP, not the stdio server Cursor Agent uses. Puppetmaster is the durable worker runtime behind that chat — same jobs, artifacts, and `effort-index`, over streamable HTTP. See [Grok Bot](#grok-bot).

## Measured results

- **SWE-bench Lite:** 29% lower actual spend with cost routing and durable retries; 47–48% token-matched savings. This is a single-seed study and does not establish quality parity. [Study](https://github.com/professorpalmer/swebench-pm).
- **NL2Repo-Bench:** 91.1% mean pass rate, about 2.28× the published ~40% baseline. [Benchmark and methodology](https://professorpalmer.github.io/durable-state-vs-context/).

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/demo.gif" alt="Puppetmaster demo showing routing, worker fan-out, and a stitched summary" width="100%" />

## Contents

- [Install](#install)
- [Grok Bot](#grok-bot)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Evidence](#evidence)
- [More documentation](#more-documentation)
- [Uninstall](#uninstall)
- [Status](#status)
- [License](#license)

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # installs MCP tools, rules, and hooks
```

`setup` is idempotent, skips platforms that are not installed, and prints each change. It asks you to enable at least one adapter; for example:

```bash
puppetmaster setup --platforms cursor
# Pi TUI/pilot (not a worker adapter):
puppetmaster setup --platforms pi
# OMP / oh-my-pi TUI/pilot (not a worker adapter):
puppetmaster setup --platforms omp
```

Restart Cursor, Codex, Claude, Antigravity, Hermes, Pi, or OMP after setup. The host then has the `puppetmaster_*` MCP tools and, where supported, hooks that suggest delegation for larger tasks. Disable those hooks with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`. For CI, use `--platforms <comma-list>` or `--platforms all`. Add another adapter later with `puppetmaster platform enable <name>`.

Grok Bot does not use that stdio install. Start remote MCP and add the printed connector instead ([Grok Bot](#grok-bot)).

The built-in `agentic` adapter needs only a provider API key, so it can run without an external CLI. See [adapter setup](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ADAPTERS.md) for provider, Antigravity, and Hermes details.

## Grok Bot

Cursor's Grok Bot assistant attaches **remote** MCP connectors only (streamable HTTP / SSE). It cannot register `python -m puppetmaster.mcp_server` the way Cursor Agent, Claude Desktop, and Codex do. Serve the same tool handlers over HTTP and Grok Bot can start and watch durable jobs on this box:

```bash
export PUPPETMASTER_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m puppetmaster mcp serve-remote --scope supervise
# equivalent: puppetmaster-mcp-remote --scope supervise
# one-shot PoC (prints connector JSON): ./scripts/grok-bot-remote-poc.sh
```

In Grok Bot → Add MCP server, use the printed `/mcp` URL and `Authorization: Bearer <token>`. Use a TLS tunnel if the bot is off-box. Confirm tools load, then `puppetmaster_doctor` and `puppetmaster_start_implement` or `puppetmaster_start_agentic`.

When Cursor is not installed on that host, implement / prewalk / swarm pick keys-only **agentic** workers (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, …). There is no `grok-bot` adapter and no CreateAgent fleet. Default `--scope supervise` omits implement/edit; pass `--scope implement` only when you want the remote client to start full-edit workers.

Connector JSON, handshake notes, and auth: [GROK_BOT.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/GROK_BOT.md).

## Quickstart

Inside Cursor Agent, Grok Bot, or Codex:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

For a supervised change:

```text
Use Puppetmaster to start a review for this repo on my configured reviewer platform and return the job id immediately.
Problem: users get logged out after refresh and token-refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

From the shell:

```bash
puppetmaster doctor
puppetmaster route "Security audit every endpoint" --role audit
puppetmaster cursor "Review this repo for release blockers" --review --dry-run
puppetmaster platform reviewer codex
puppetmaster review "Review this repo for release blockers"
puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
puppetmaster show "$(puppetmaster last)"
```

To verify an installed Codex route on macOS, Linux, or Windows, run
`puppetmaster setup --verify-first-run codex/<model>` with an exact registry ID.
This opt-in check makes one live call in temporary state and returns nonzero if
it cannot prove delivery within 120 seconds. Ordinary setup does not make this
call. See [first-run verification](docs/CLI_REFERENCE.md).

More recipes are in [DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md) and [MODEL_ROUTING.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MODEL_ROUTING.md).

## How it works

Puppetmaster is a supervisor and job store for agent CLIs and provider adapters. Grok Bot, Cursor Agent, Pi, and OMP are *pilots* (they call MCP tools); cursor / claude-code / agentic / … are *adapters* (leased workers):

```text
pilots (MCP):  Cursor Agent / Grok Bot / Claude Desktop / Pi / OMP
workers:       cursor / claude-code / codex / hermes / antigravity / fx / agentic
                                |
                                v
      supervisor -> model router -> independent workers -> SQLite artifacts
                                                                |
                                                                v
                                                         stitched summary
```

Workers claim tasks, write artifacts containing payloads and evidence, and do not share one growing transcript. Follow-up inspection is a SQLite read at $0. A later model retrieves that working set; it does not inherit another model's provider KV cache. The parent agent receives the stitched result and can inspect the stored artifacts with:

```bash
puppetmaster artifacts <job_id>
python -m puppetmaster dashboard
```

A host pane (for example Marionette) can load a chrome-free job view at `http://127.0.0.1:<port>/?job=<id>&embed=1`.

[CodeGraph](https://github.com/colbymchenry/codegraph) is an optional structural code index. When installed, Puppetmaster adds task-relevant CodeGraph context before worker calls; otherwise workers use ordinary repository inspection. See [CODEGRAPH.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CODEGRAPH.md).

Puppetmaster sits above libraries such as LangGraph and CrewAI: those libraries help you build an agent, while Puppetmaster coordinates existing agent CLIs and adapters. See [WHY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md) and [COMPARISON.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md).

## Evidence

The repository includes reproducible benchmark scripts and their scope and caveats in [CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md). The receipts cover:

- routing fixture results and follow-up reads from completed SQLite artifacts;
- typed artifacts, evidence fields, and content hashes;
- CodeGraph context injection and adapter failure classification.

These are measurements of the included workflows, not guarantees for every repository or model. An independent durable-state benchmark is documented [here](https://professorpalmer.github.io/durable-state-vs-context/).

## More documentation

The [docs index](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/README.md) covers:

- [GROK_BOT.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/GROK_BOT.md) — Grok Bot as remote MCP pilot (streamable HTTP; not a worker adapter)
- [FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md) — adapters, pilots, and shipped features
- [FLOWS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FLOWS.md) — flow graphs: write one graph, wake on done, gate or stuck; resumable nodes and `map` fan-out
- [SECURITY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SECURITY.md) — safety and threat model
- [DASHBOARD.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DASHBOARD.md) — live job dashboard
- [CONCURRENT_SESSIONS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CONCURRENT_SESSIONS.md) — operating concurrent agent sessions, state scope, dashboard URLs, and worktrees
- [OUTPUT_STYLE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/OUTPUT_STYLE.md) and [COMPRESSION.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPRESSION.md) — output and context options
- [MOBILE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MOBILE.md) — watching jobs from a phone
- [RESEARCH.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/RESEARCH.md) — durable autoresearch claims, runs, publishing, and verification

## Uninstall

```bash
puppetmaster uninstall
pip uninstall puppetmaster-ai   # or: pipx uninstall puppetmaster-ai
```

`uninstall` removes Puppetmaster-owned MCP entries, hooks, and rules. It keeps `~/.puppetmaster/` and workspace `.codegraph/` unless you pass `--purge-state`; use `--dry-run` to preview.

## Status

Puppetmaster is a daily-driver beta at **v1.27.21**, suitable for supervised local engineering. v1.27.21 stops agentic implement workers that keep exploring without a successful mutation and preserves compact terminal diagnostics for no-diff failures. v1.27.20 persists an agentic catalog refresh that restamps existing rows so fallback reloads that bound epoch, and durable admission requires a verification that names the artifact id or exact claim/risk/decision text — worker `grounded` labels and a same-task pass with no pointer do not count. v1.27.19 admits an exact enabled `agentic/<provider>/…` registry id when that provider is ready, so a funded fallback such as `agentic/openai/gpt-5-6-sol` can dispatch even when `gpt-5.6-sol` is absent from a fresh agentic discover snapshot; bare names still need the catalog. v1.27.18 lets a generated auto-routed swarm leave a spent launch adapter after classified `billing_or_quota` (`adapter_lock=lane`) so that funded identity can complete the job; hard pins stay locked. v1.27.14 retries a locked file-backend projection writer on `save_task` (same helper as `save_run`) so crash recovery does not die with `ProjectionWriteAdmissionError` on Windows CI. v1.27.13 keeps OpenCode Go thinking models (`deepseek-flash` and peers) on the nudge-only submit path and retries a 400 whose body mentions `tool_choice`, so pinned Go swarms no longer die after a late forced submit. v1.27.12 makes `setup --verify-first-run` read the persisted Codex wire name instead of stripping `codex/` from the registry id, so dashed aliases such as `codex/gpt-5-6-sol` match dotted adapter names such as `gpt-5.6-sol`. v1.27.11 preserves DeepSeek `reasoning_content` through agentic tool calls so the next provider request is accepted, and keeps qualified model allowlists within their selected provider. v1.27.10 makes MCP `last_job` return a store-scoped bind (`last --json`) so a workspace last id cannot be mistaken for a Marionette `~/.pmharness` swarm. v1.27.9 stamps `x-opencode-session` on OpenCode Go worker HTTP so swarms stop dying with `MissingSessionID`. v1.27.8 keeps Darwin sqlite readonly helpers working when APFS updates ctime on a lock-open. v1.27.7 hardens openai-codex (remap *-pro, refuse openai-api, fail closed on HTTP 400). v1.27.6 adds orchestration durability (#167): session command ledger, run journal crash stamps, and WorkspaceScope freeze so MCP/CLI keep a stable primary store root. v1.27.4 adds an opt-in same-job continuous planner (`role=planner`): intent-spec before fan-out, typed worker handoffs, and kernel requeue until `scope_complete`. v1.27.1 adds chrome-free `/?job=&embed=1` densify mode for Marionette (and other) host panes. v1.27.0 added a community observation store and a separate role-preference file so StrongOrc-shaped priors can order already-sufficient models without rewriting `capability_score`. ROUTING names `community_observation` or `preference` when that layer chose the winner.

[Store contracts](docs/STORE_CONTRACTS.md) describe the embedding APIs and their limits. [Attempt accounting](docs/ATTEMPT_LEDGER.md) preserves retries and unknown usage; selected-result economics remain separate. [Budget reservations](docs/BUDGET_RESERVATIONS.md) enforce cumulative admission, though opaque provider overruns can exceed allowances. SQLite state migrates to schema v5, with stale projection triggers repaired at supervisor initialization. Stop long-lived Puppetmaster processes before the schema cutover and restart supervisors, workers, MCP servers, and dashboards on the new version. See the [feature matrix](docs/FEATURES.md) and [changelog](docs/CHANGELOG.md).

PyPI uses the package name [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/); the import name, CLI, and repository use `puppetmaster`.

## License

MIT
