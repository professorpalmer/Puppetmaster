# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/pyproject.toml)

Puppetmaster runs multi-step engineering work through the agent tools you already use: Cursor, Claude Code, Codex, Hermes, or a provider API. It starts independent workers, routes tasks to an available model, and stores their typed results in SQLite so jobs can be inspected and resumed. It is aimed at developers who want durable state and reviewable output for repository investigations, audits, refactors, and implementations.

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/demo.gif" alt="Puppetmaster demo showing routing, worker fan-out, and a stitched summary" width="100%" />

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # installs MCP tools, rules, and hooks
```

`setup` is idempotent, skips platforms that are not installed, and prints each change. It asks you to enable at least one adapter; for example:

```bash
puppetmaster setup --platforms cursor
```

Restart Cursor, Codex, Claude, or Hermes after setup. The host then has the `puppetmaster_*` MCP tools and, where supported, hooks that suggest delegation for larger tasks. Disable those hooks with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`. For CI, use `--platforms <comma-list>` or `--platforms all`. Add another adapter later with `puppetmaster platform enable <name>`.

The built-in `agentic` adapter needs only a provider API key, so it can run without an external CLI. See [adapter setup](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ADAPTERS.md) for provider and Hermes details.

## Quickstart

Inside Cursor Agent or Codex:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

For a supervised change:

```text
Use Puppetmaster to start a cursor swarm for this repo and return the job id immediately.
Problem: users get logged out after refresh and token-refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

From the shell:

```bash
puppetmaster doctor
puppetmaster route "Security audit every endpoint" --role audit
puppetmaster cursor "Review this repo for release blockers" --review --dry-run
puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
puppetmaster show "$(puppetmaster last)"
```

More recipes are in [DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md) and [MODEL_ROUTING.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MODEL_ROUTING.md).

## How it works

Puppetmaster is a supervisor and job store for agent CLIs and provider adapters:

```text
Cursor / Claude Code / Codex / Hermes / agentic
        |
        v
supervisor -> model router -> independent workers -> SQLite artifacts
                                                   |
                                                   v
                                            stitched summary
```

Workers claim tasks, write artifacts containing payloads and evidence, and do not share one growing transcript. The parent agent receives the stitched result and can inspect the stored artifacts with:

```bash
puppetmaster artifacts <job_id>
python -m puppetmaster dashboard
```

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

- [FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md) — adapters and shipped features
- [SECURITY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SECURITY.md) — safety and threat model
- [DASHBOARD.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DASHBOARD.md) — live job dashboard
- [OUTPUT_STYLE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/OUTPUT_STYLE.md) and [COMPRESSION.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPRESSION.md) — output and context options
- [MOBILE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MOBILE.md) — watching jobs from a phone

## Uninstall

```bash
puppetmaster uninstall
pip uninstall puppetmaster-ai   # or: pipx uninstall puppetmaster-ai
```

`uninstall` removes Puppetmaster-owned MCP entries, hooks, and rules. It keeps `~/.puppetmaster/` and workspace `.codegraph/` unless you pass `--purge-state`; use `--dry-run` to preview.

## Status

Puppetmaster is a daily-driver beta at **v1.21.8**, suitable for supervised local engineering rather than hosted multi-user use. The current release includes SQLite-backed jobs, Cursor Agent MCP support, full-edit adapters, constrained model routing, and AWS Bedrock support through the agentic adapter. See the [feature matrix](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md) and [CHANGELOG.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CHANGELOG.md) for the current details.

PyPI uses the package name [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/); the import name, CLI, and repository use `puppetmaster`.

## License

MIT
