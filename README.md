# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/pyproject.toml)

Puppetmaster runs engineering work as durable jobs. It is universal for any harness or CLI, or a plain provider API.

Your agent gives Puppetmaster a task. Puppetmaster starts independent workers, routes each task to a model, and stores each typed result in SQLite. You can inspect a job, resume it, or read its results later at no model cost.

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/demo.gif" alt="Puppetmaster demo showing routing, worker fan-out, and a stitched summary" width="100%" />

## Quick setup

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # installs MCP tools, rules, and hooks
```

Restart your agent tool. Then send it this prompt:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

No agent CLI? The built-in `agentic` adapter needs only a provider API key. For all steps, see [Setup](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SETUP.md).

## Index

### Get started

| Doc | What it covers |
|---|---|
| [Setup](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SETUP.md) | Install, select platforms, run a first job, uninstall |
| [Daily driver](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md) | Prompts for review, swarm, implement, and inspection |
| [Grok Bot](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/GROK_BOT.md) | Grok Bot as a pilot over remote MCP |
| [Pi](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/PI.md) / [OMP](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/OMP.md) | Pi and OMP as pilots |

### Run work

| Doc | What it covers |
|---|---|
| [Flows](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FLOWS.md) | Write one graph. Puppetmaster walks it and wakes you when it is done |
| [Adapters](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ADAPTERS.md) | Worker adapters and how to add one |
| [Model routing](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MODEL_ROUTING.md) | Router policies and the model registry |
| [Concurrent sessions](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CONCURRENT_SESSIONS.md) | State scope, worktrees, and many sessions at once |
| [CodeGraph](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CODEGRAPH.md) | Optional code index for worker context |
| [Budgets](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/BUDGET_RESERVATIONS.md) | Spend limits and how they are enforced |

### Watch and inspect

| Doc | What it covers |
|---|---|
| [Dashboard](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DASHBOARD.md) | Live job board |
| [Mobile](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MOBILE.md) | Watch jobs from a phone |
| [CLI reference](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLI_REFERENCE.md) | Every CLI command |
| [MCP tools](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CURSOR_AGENT_MCP.md) | The MCP tool surface |

### How it works

| Doc | What it covers |
|---|---|
| [Architecture](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ARCHITECTURE.md) | Pilots, workers, jobs, tasks, and artifacts |
| [Why](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md) | Why durable state, not one shared transcript |
| [Comparison](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md) | How it differs from LangGraph, CrewAI, and native subagents |
| [Store contracts](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/STORE_CONTRACTS.md) | Embedding APIs and their limits |

### Evidence

| Doc | What it covers |
|---|---|
| [Claims](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md) | Benchmark scripts, scope, and caveats |
| [SWE-bench Lite study](https://github.com/professorpalmer/swebench-pm) | 29% lower actual spend with cost routing and durable retries. One seed. Quality parity is not proven |
| [NL2Repo-Bench](https://professorpalmer.github.io/durable-state-vs-context/) | 91.1% mean pass rate against a published baseline of about 40% |
| [Research](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/RESEARCH.md) | Durable autoresearch protocol |

### Operate

| Doc | What it covers |
|---|---|
| [Troubleshooting](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/TROUBLESHOOTING.md) | Common errors and fixes |
| [Security](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SECURITY.md) | Threat model and safe use |
| [Production](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/PRODUCTION.md) | Notes for use past a local trial |
| [Changelog](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CHANGELOG.md) | Changes in each release |
| [Contributing](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CONTRIBUTING.md) | How to land a patch |
The full docs list is in the [docs index](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/README.md).

## License

MIT
