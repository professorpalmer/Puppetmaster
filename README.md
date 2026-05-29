# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Turn Cursor, Claude Code, the OpenAI API, or the Codex CLI into an orchestrator that routes every task to the cheapest model that can handle it, runs workers as independent processes, and stores their output as typed SQLite artifacts so follow-ups cost zero tokens.**

<img src="docs/demo.gif" alt="Puppetmaster 60-second demo: cost routing, swarm fan-out, stitched summary, $0 follow-ups" width="100%" />

> **💸 Cheaper — live OpenAI A/B, real billing tokens, same prompt, equivalent answer:** pinned `gpt-5.5` cost **\$0.0132**; Puppetmaster routed the same task to `gpt-5.4-nano` for **\$0.00016** — **98.8% cheaper and 81% faster.** Reproduce in ~$0.01 of spend: `OPENAI_API_KEY=... python -m bench.router_live_ab`.

> **🔁 Self-healing — a dead provider doesn't kill the swarm (proven live, job `job_d82715bebc5d`):** a `claude-code` worker hit a real **\$0 Anthropic balance** → classified `billing_or_quota` → marked **FAILED** → **auto-rerouted to `cursor/gpt-5.5`** (plan-billed, `$0`) → the funded adapter **completed the task.** No silent degraded run.

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # doctor + models init + MCP installers + agent rules, idempotent
```

That's the whole install. `setup` runs every step idempotently, skips any tool that isn't present, and prints what it did. Restart Cursor (or open a fresh Codex / Claude session) and the agent sees 32+ `puppetmaster_*` tools plus a rule nudging it to reach for them on multi-file work.

To run benchmarks or hack on it, clone instead — see [Contributing](docs/CONTRIBUTING.md). (`pipx` keeps the CLI in its own isolated environment, which is the recommended way to install a command-line app.)

## Index

**New here?** Watch the GIF above, run `pipx install puppetmaster-ai && puppetmaster setup`, then skim [What it does](#what-it-does).

| Want to… | Go to |
|---|---|
| Understand the design & what it fixes | [docs/WHY.md](docs/WHY.md) |
| Know how it differs from LangGraph / CrewAI / subagents | [docs/COMPARISON.md](docs/COMPARISON.md) |
| Know if it's safe to hand it your repo & plan | [docs/SECURITY.md](docs/SECURITY.md) |
| See the proof behind the claims | [docs/CLAIMS.md](docs/CLAIMS.md) · receipts in [`bench/`](bench/) |
| See everything that ships + adapters | [docs/FEATURES.md](docs/FEATURES.md) |
| Copy/paste prompts & shell recipes | [Quickstart](#quickstart) · [docs/DAILY_DRIVER.md](docs/DAILY_DRIVER.md) |
| Read the full docs set | [docs/README.md](docs/README.md) |
| Browse by directory | [`puppetmaster/`](puppetmaster/README.md) · [`bench/`](bench/README.md) · [`examples/`](examples/README.md) · [`scripts/`](scripts/README.md) · [`clients/typescript/`](clients/typescript/README.md) · [`cursor-extension/`](cursor-extension/README.md) |
| Know what's honest vs. overclaim | [TALKING_POINTS.md](TALKING_POINTS.md) |

## What it does

Think **Redis/Gunicorn for agentic engineering**:

```text
Cursor Agent / Claude Code / OpenAI / Codex CLI / shell
        |
        v
Puppetmaster supervisor  ──>  task-aware model router (auto-routes by cost)
        |
        v
independent worker processes  ──>  SQLite (typed artifacts, events, memory)
        |
        v
live artifact board  ──>  stitched summary  ──>  0-token follow-up reads
```

Puppetmaster isn't trying to beat native IDE subagents at every tiny task. It's for the work that gets messy: long repo investigations, conflicting hypotheses, repeated handoffs, flaky memory, and code changes that need evidence, replay, and approval gates. The rationale and failure modes it fixes are in [docs/WHY.md](docs/WHY.md).

**How it's different:** LangGraph, CrewAI, and the Claude Agent SDK are libraries you write code against to *build* an agent. Puppetmaster sits one layer up — it **orchestrates the agent CLIs you already pay for** (Cursor, Claude Code, Codex, OpenAI), routes each task to the cheapest sufficient model, keeps the spend inside your subscription, and self-heals when a provider is down. Full side-by-side + "pick X instead if…" in [docs/COMPARISON.md](docs/COMPARISON.md).

### The demo (no API keys)

The whole story in one command — local + shell adapters, nothing to configure:

```bash
./scripts/demo.sh                  # the 60-second tour (clean machine, no keys)
python -m puppetmaster dashboard   # live, zero-dependency web board for any job
```

It routes a task mix by cost, fans out a 6-role swarm as independent processes, reads the stitched summary, then proves follow-up reads cost **\$0.00**. Script + GIF source: [`scripts/`](scripts/README.md).

## Why it's credible — four claims, four receipts

Every number is reproducible from a script in [`bench/`](bench/). Full detail + caveats: [docs/CLAIMS.md](docs/CLAIMS.md).

1. **Cost is fixed on two axes.** New work auto-routes to the cheapest sufficient model (**35% cheaper** on a fixture; **98.8% cheaper** in a live OpenAI A/B). Follow-ups are SQLite reads, not new agent runs (**40 queries, \$0.00, 0.5 ms each**).
2. **Workers don't share a transcript.** They lease tasks and emit **typed artifacts** (payload + `evidence` + `confidence` + `sha256`); the stitcher reads JSON, not stdout. Inspect with `puppetmaster artifacts <job_id>`.
3. **Graphing is [CodeGraph](https://github.com/colbymchenry/codegraph)'s win, wired in cleanly.** Workers auto-inject task-relevant graph context before the model call; fall back to grep/read without it. ([docs/CODEGRAPH.md](docs/CODEGRAPH.md))
4. **A dead provider doesn't kill the swarm (v0.9.0+).** Billing/quota/auth/missing-CLI failures are marked `FAILED` and **auto-rerouted to the next funded adapter**, preferring plan-billed models. Validated live; surfaced loudly in the summary's Alerts section.

## Quickstart

After install, try one of these inside Cursor Agent or Codex:

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

More recipes in [docs/DAILY_DRIVER.md](docs/DAILY_DRIVER.md).

## Status

**Daily-driver beta.** Real runtime contract, automated tests, SQLite default backend, fail-closed jobs, live Cursor Agent MCP, installable Cursor extension, validated full-edit adapters. Credible for supervised local engineering; not yet a hosted multi-user service. Full feature matrix: [docs/FEATURES.md](docs/FEATURES.md).

**Pip name:** PyPI lists this as [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/) because [PEP-503 normalization](https://peps.python.org/pep-0503/#normalized-names) collides `puppetmaster` with an [abandoned 2019 `puppet-master`](https://pypi.org/project/puppet-master/). The import name, CLI, repo, and brand stay `puppetmaster`. ([tracking](docs/PYPI_NAME_REQUEST.md))

## License

MIT
