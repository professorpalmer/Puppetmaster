# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/pyproject.toml)

**Turn Cursor, Claude Code, the OpenAI API, or the Codex CLI into an orchestrator that routes every task to the cheapest model that can handle it, runs workers as independent processes, and stores their output as typed SQLite artifacts so follow-ups cost zero tokens.**

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/demo.gif" alt="Puppetmaster 60-second demo: cost routing, swarm fan-out, stitched summary, $0 follow-ups" width="100%" />

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/receipts.svg" alt="The receipts — Scenario A (best case, live OpenAI A/B): 98.8% cheaper, 88% faster. Scenario B (everyday mixed workload, dry-run): 35.1% cheaper overall." width="100%" />

> **💸 Reproduce the live A/B in ~$0.01 of spend** — `OPENAI_API_KEY=... python -m bench.router_live_ab`. Pinned `gpt-5.5` cost **\$0.0132**; Puppetmaster routed the same task to `gpt-5.4-nano` for **\$0.00016** (same prompt, equivalent answer). The 35.1% figure is a 6-task mixed-workload dry-run where the router *correctly* kept the frontier model on the 2 hard tasks — full method in [docs/CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md).

> **🔁 Self-healing — a dead provider doesn't kill the swarm (proven live, job `job_d82715bebc5d`):** a `claude-code` worker hit a real **\$0 Anthropic balance** → classified `billing_or_quota` → marked **FAILED** → **auto-rerouted to `cursor/gpt-5.5`** (plan-billed, `$0`) → the funded adapter **completed the task.** No silent degraded run.

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # doctor + models init + MCP installers + rules + auto-invocation hooks, idempotent
```

That's the whole install. `setup` runs every step idempotently, skips any tool that isn't present, and prints what it did. Restart Cursor (or open a fresh Codex / Claude session) and the agent sees 32+ `puppetmaster_*` tools, a rule nudging it to reach for them, **and deterministic auto-invocation hooks** that inject a "delegate now" directive on prompt submit and redirect genuinely broad shell searches (recursive `rg`/`grep`/`find`) plus `Task` fan-out to Puppetmaster — so you stop having to remind it. Read-only inspection (`git log`/`diff`, single-file greps, the native Grep/Glob tools) passes straight through; it's classifier-gated (trivial edits stay inline) and fully kill-switchable with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`. See [Auto-invocation](https://github.com/professorpalmer/Puppetmaster#auto-invocation).

To run benchmarks or hack on it, clone instead — see [Contributing](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CONTRIBUTING.md). (`pipx` keeps the CLI in its own isolated environment, which is the recommended way to install a command-line app.)

## Index

**New here?** Watch the GIF above, run `pipx install puppetmaster-ai && puppetmaster setup`, then skim [What it does](https://github.com/professorpalmer/Puppetmaster#what-it-does).

| Want to… | Go to |
|---|---|
| Understand the design & what it fixes | [docs/WHY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md) |
| Know how it differs from LangGraph / CrewAI / subagents | [docs/COMPARISON.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md) |
| Know if it's safe to hand it your repo & plan | [docs/SECURITY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SECURITY.md) |
| See the proof behind the claims | [docs/CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md) · receipts in [`bench/`](https://github.com/professorpalmer/Puppetmaster/tree/main/bench/) |
| See everything that ships + adapters | [docs/FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md) |
| Copy/paste prompts & shell recipes | [Quickstart](https://github.com/professorpalmer/Puppetmaster#quickstart) · [docs/DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md) |
| Read the full docs set | [docs/README.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/README.md) |
| Browse by directory | [`puppetmaster/`](https://github.com/professorpalmer/Puppetmaster/blob/main/puppetmaster/README.md) · [`bench/`](https://github.com/professorpalmer/Puppetmaster/blob/main/bench/README.md) · [`examples/`](https://github.com/professorpalmer/Puppetmaster/blob/main/examples/README.md) · [`scripts/`](https://github.com/professorpalmer/Puppetmaster/blob/main/scripts/README.md) · [`clients/typescript/`](https://github.com/professorpalmer/Puppetmaster/blob/main/clients/typescript/README.md) |

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

Puppetmaster isn't trying to beat native IDE subagents at every tiny task. It's for the work that gets messy: long repo investigations, conflicting hypotheses, repeated handoffs, flaky memory, and code changes that need evidence, replay, and approval gates. The rationale and failure modes it fixes are in [docs/WHY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md).

**How it's different:** LangGraph, CrewAI, and the Claude Agent SDK are libraries you write code against to *build* an agent. Puppetmaster sits one layer up — it **orchestrates the agent CLIs you already pay for** (Cursor, Claude Code, Codex, OpenAI), routes each task to the cheapest sufficient model, keeps the spend inside your subscription, and self-heals when a provider is down. Full side-by-side + "pick X instead if…" in [docs/COMPARISON.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md).

<p align="center">
<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/layer-above.jpg" alt="A layer above your agents, not another framework. Cursor / Claude Code / Codex / OpenAI feed into Puppetmaster (cost-aware router + supervisor), which fans out to independent worker processes, SQLite typed artifacts, and $0 follow-up reads. LangGraph / CrewAI are libraries you write code against to build an agent; Puppetmaster drives the agent CLIs you already use." width="100%" />
</p>

### The demo (no API keys)

The whole story in one command — local + shell adapters, nothing to configure:

```bash
./scripts/demo.sh                  # the 60-second tour (clean machine, no keys)
python -m puppetmaster dashboard   # live, zero-dependency web board for any job
```

It routes a task mix by cost, fans out a 6-role swarm as independent processes, reads the stitched summary, then proves follow-up reads cost **\$0.00**. Script + GIF source: [`scripts/`](https://github.com/professorpalmer/Puppetmaster/blob/main/scripts/README.md).

## Why it's credible — four claims, four receipts

Every number is reproducible from a script in [`bench/`](https://github.com/professorpalmer/Puppetmaster/tree/main/bench/). Full detail + caveats: [docs/CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md).

1. **Cost is fixed on two axes.** New work auto-routes to the cheapest sufficient model (**35% cheaper** on a fixture; **98.8% cheaper** in a live OpenAI A/B). Follow-ups are SQLite reads, not new agent runs (**40 queries, \$0.00, 0.5 ms each**).
2. **Workers don't share a transcript.** They lease tasks and emit **typed artifacts** (payload + `evidence` + `confidence` + `sha256`); the stitcher reads JSON, not stdout. Inspect with `puppetmaster artifacts <job_id>`.
3. **Graphing is [CodeGraph](https://github.com/colbymchenry/codegraph)'s win, wired in cleanly.** Workers auto-inject task-relevant graph context before the model call; fall back to grep/read without it. ([docs/CODEGRAPH.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CODEGRAPH.md))
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

More recipes in [docs/DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md).

## Recommended setup: a cheap chat model that delegates to Puppetmaster

The pattern that works best — and the one `puppetmaster setup` nudges your agent toward — is to **keep a cheap conversational model in your IDE chat window and let it hand the technical work to Puppetmaster.** You talk to the fast/cheap model for everything conversational; the moment a request is real engineering (multi-file investigation, a refactor, a review, an implementation), it starts a Puppetmaster job and drives it.

```text
You ── chat ──> cheap conversational model (fast, $-light)
                      │
                      │  "this is real work" → delegate
                      v
              Puppetmaster (routes to the cheapest sufficient model,
              runs durable workers, stores typed artifacts)
```

Why this is the shape to aim for:

- **Conversational asks stay instant and cheap.** "What does this do?", "summarize that", "thanks" never pay orchestration cost.
- **Technical work gets the full machine** — cost routing, independent workers, durable artifacts, replay, $0 follow-up reads — only when it's warranted.
- **Context compounds where it belongs.** Each delegated job leaves typed artifacts and promoted memory that later jobs reuse, instead of living in a chat scrollback that evaporates.

The agent rules installed by `puppetmaster setup` (Cursor `.mdc` + `AGENTS.md`) already encode this: *start a swarm for non-trivial work; use native tooling for trivial edits and conversational follow-ups.*

### Puppetmaster is not a chat layer — on purpose

Don't try to route **every** chat message through Puppetmaster "to capture context." It's a deliberate boundary, not a missing feature: spinning a durable worker for "hi" or "what's this function" inverts the entire value proposition (you'd add orchestration cost and latency to the cheapest turns), and it fights the IDE, which doesn't expose a clean intercept-every-message hook. Let the cheap model triage; let Puppetmaster do the work that deserves a job. The router-at-the-top decides what crosses that line.

## Auto-invocation

The hardest part of that pattern is getting the host agent to *actually* delegate without you reminding it every few turns. Rules help but decay with context distance. So `puppetmaster setup` also installs a layered, classifier-gated enforcement system — designed to fire automatically exactly when a task warrants a job, and stay out of the way otherwise.

- **The gate** (`puppetmaster should-delegate "<prompt>"`) reuses the router's existing pure-function classifier to answer delegate-vs-inline in microseconds — no LLM, no network. A trivial-task carve-out keeps typos/renames/one-liners/quick questions inline; a conservative threshold and a broad-scope override (audit/refactor/migrate/trace) catch real multi-file work.
- **Deterministic hooks** (`puppetmaster install-hooks`, run automatically by `setup`) write idempotent, non-destructive entries into Cursor's `.cursor/hooks.json` and Claude Code's `.claude/settings.json`: they inject a "delegate now" directive on prompt submit and **deny-redirect** genuinely broad shell searches (recursive `rg`/`grep`/`find`) plus built-in `Task` fan-out to the Puppetmaster/CodeGraph equivalent. Read-only inspection — `git log`/`show`/`diff`, listing a directory, single-file greps, and the native Grep/Glob tools — is explicitly carved out and passes through, because their scope isn't visible to the hook and hard-denying them would wedge legitimate work. They fail open — a hook can never wedge your session. Default scope is per-repo; `puppetmaster install-hooks --global` (or `setup --global-hooks`) writes user-level hooks (`~/.cursor/hooks.json`, `~/.claude/settings.json`) that cover every repo you open without re-running setup.
- **Optional proxy** (`puppetmaster proxy`) extends the same gate to OpenAI-compatible API-key/SDK clients that closed harnesses can't hook.

**Honest about the ceiling:** there is no universal *deterministic* invocation. Closed harnesses won't let anything sit on their LLM provider wire or force an MCP call, so the system is explicitly tiered — soft rules everywhere, hard hooks where the host exposes them (Cursor, Claude Code), proxy only for clients routed through it — and fully kill-switchable with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`. It enforces the "let the cheap model triage, let Puppetmaster do the real work" boundary above; it does not try to make Puppetmaster a chat layer.

## Status

**Daily-driver beta.** Real runtime contract, automated tests, SQLite default backend, fail-closed jobs, live Cursor Agent MCP, validated full-edit adapters. Credible for supervised local engineering; not yet a hosted multi-user service. Full feature matrix: [docs/FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md).

**Pip name:** PyPI lists this as [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/) because [PEP-503 normalization](https://peps.python.org/pep-0503/#normalized-names) collides `puppetmaster` with an [abandoned 2019 `puppet-master`](https://pypi.org/project/puppet-master/). The import name, CLI, repo, and brand stay `puppetmaster`. ([tracking](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/PYPI_NAME_REQUEST.md))

## License

MIT
