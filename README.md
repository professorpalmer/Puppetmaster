# Puppetmaster

[![PyPI](https://img.shields.io/pypi/v/puppetmaster-ai.svg)](https://pypi.org/project/puppetmaster-ai/)
[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/professorpalmer/Puppetmaster/blob/main/pyproject.toml)

Puppetmaster turns the agent CLIs you already pay for — Cursor, Claude Code (Anthropic or AWS Bedrock), the OpenAI API, the Codex CLI, or Hermes — into an orchestrator. Or run it with no external CLI at all: point the built-in `agentic` adapter at any provider API key (OpenAI, Anthropic, Gemini, OpenRouter) and it runs the whole tool-use loop itself — ideal for CI, containers, and headless servers. Either way it routes each task to the cheapest model that can handle it, runs workers as independent processes, and stores their output as typed SQLite artifacts, so follow-up reads cost zero tokens.

Community: join the Discord at https://discord.gg/VQmkmGtQnA

<img src="https://raw.githubusercontent.com/professorpalmer/Puppetmaster/main/docs/demo.gif" alt="Puppetmaster 60-second demo: cost routing, swarm fan-out, stitched summary, and zero-token follow-ups" width="100%" />

## Contents

- [Install](https://github.com/professorpalmer/Puppetmaster#install) and [Uninstall](https://github.com/professorpalmer/Puppetmaster#uninstall)
- [What it does](https://github.com/professorpalmer/Puppetmaster#what-it-does) — the object model, and how it differs from agent frameworks
- [Why it's credible](https://github.com/professorpalmer/Puppetmaster#why-its-credible) — four claims, four reproducible receipts
- [Quickstart](https://github.com/professorpalmer/Puppetmaster#quickstart) — prompts and shell recipes
- [Recommended setup](https://github.com/professorpalmer/Puppetmaster#recommended-setup) — a cheap chat model that delegates the real work
- [Auto-invocation](https://github.com/professorpalmer/Puppetmaster#auto-invocation) — how delegation fires without reminding the agent
- [Output style and compression](https://github.com/professorpalmer/Puppetmaster#output-style-and-compression)
- [Status](https://github.com/professorpalmer/Puppetmaster#status)

Documentation lives in [`docs/`](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/README.md):

- Design and rationale — [WHY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md)
- How it compares to LangGraph / CrewAI / native subagents — [COMPARISON.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md)
- The proof behind the claims — [CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md)
- Everything that ships, with the adapter table — [FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md)
- Safety and threat model — [SECURITY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/SECURITY.md)
- Prompt and shell recipes — [DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md)
- The live job dashboard — [DASHBOARD.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DASHBOARD.md)
- Watch swarms from your phone (Tailscale + QR) — [MOBILE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MOBILE.md)
- Model routing, architecture, adapters, CodeGraph, CLI — see the [docs index](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/README.md)

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # doctor + models init + MCP installers + rules + hooks — idempotent
```

That is the whole install. `setup` runs each step idempotently, skips any tool that isn't present, and prints what it did. Restart Cursor (or open a fresh Codex, Claude, or Hermes session) and the agent gains the `puppetmaster_*` MCP tools, a rule nudging it to use them, and [auto-invocation hooks](https://github.com/professorpalmer/Puppetmaster#auto-invocation) that delegate real work. The whole thing is kill-switchable with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`.

Setup starts with every platform off and asks you to enable at least one execution adapter (`--platforms cursor`, or an interactive pick). A single platform is the expected setup; enabling several is opt-in and unlocks cross-platform router fallback and free-tier hopping. Add more later with `puppetmaster platform enable <name>`. For CI, pass `--platforms <comma-list>` or `--platforms all`.

Hermes has an optional in-depth setup branch (learn flywheel, skill promotion, skill injection); see [ADAPTERS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ADAPTERS.md#hermes). The `agentic` adapter needs only a provider API key (no external CLI) — see [ADAPTERS.md#agentic](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/ADAPTERS.md#agentic) and [DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md#keys-only-recipe-no-external-cli). To run benchmarks or hack on the code, clone instead — see [CONTRIBUTING.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CONTRIBUTING.md).

## Uninstall

```bash
puppetmaster uninstall          # MCP registrations, hooks, rules — idempotent
pip uninstall puppetmaster-ai   # or: pipx uninstall puppetmaster-ai
```

`uninstall` removes only Puppetmaster-owned artifacts (its MCP entries, auto-invocation hooks, and rule files or marked blocks) and leaves other MCP servers untouched. Swarm state under `~/.puppetmaster/` and workspace `.codegraph/` are kept unless you pass `--purge-state`. Use `--dry-run` to preview and `--cwd` to target another workspace.

## What it does

Think of it as Redis or Gunicorn for agentic engineering: a supervisor in front of worker processes, with durable shared state.

```text
Cursor / Claude Code / OpenAI / Codex / Hermes / agentic (keys-only) / shell
        |
        v
Puppetmaster supervisor  ->  task-aware model router (routes by cost)
        |
        v
independent worker processes  ->  SQLite (typed artifacts, events, memory)
        |
        v
live artifact board  ->  stitched summary  ->  zero-token follow-up reads
```

It isn't meant to beat native IDE subagents at every small task. It's for work that gets messy: long repo investigations, conflicting hypotheses, repeated handoffs, flaky memory, and code changes that need evidence, replay, and approval gates.

LangGraph, CrewAI, and the Claude Agent SDK are libraries you write code against to build an agent. Puppetmaster sits one layer up — it drives the agent CLIs you already pay for, routes each task to the cheapest sufficient model, keeps spend inside your subscription, and self-heals when a provider goes down. The rationale and the side-by-side are in [WHY.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/WHY.md) and [COMPARISON.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPARISON.md).

You can see the whole story in one command, with no API keys:

```bash
./scripts/demo.sh                  # 60-second tour on a clean machine
python -m puppetmaster dashboard   # live web board for any job (see docs/DASHBOARD.md)
```

## Why it's credible

Every number is reproducible from a script in [`bench/`](https://github.com/professorpalmer/Puppetmaster/tree/main/bench/); full method and caveats in [CLAIMS.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/CLAIMS.md).

1. Cost is fixed on two axes. New work routes to the cheapest sufficient model (35% cheaper on a fixture; 98.8% cheaper in a live OpenAI A/B). Follow-ups are SQLite reads, not new runs (40 queries, $0.00, 0.5 ms each).
2. Workers don't share a transcript. They lease tasks and emit typed artifacts (payload, evidence, confidence, sha256); the stitcher reads JSON, not stdout. Inspect with `puppetmaster artifacts <job_id>`.
3. CodeGraph context is injected before the model call, so workers look up "where is X / what calls Y" structurally instead of grepping, and fall back to grep without it.
4. A dead provider doesn't kill the swarm. Billing, quota, auth, and missing-CLI failures are marked `FAILED` and rerouted to the next funded adapter, preferring plan-billed models. Validated live and surfaced in the summary's alerts.

Beyond those internal receipts, the durable-state thesis holds up on an **independent third-party benchmark**. On [NL2Repo-Bench](https://professorpalmer.github.io/durable-state-vs-context/) (build a full Python library from a natural-language spec, scored by the benchmark's own pytest suites), durable-state orchestration reaches a **91.1% mean test-pass rate — about 2.28× the ~40% published state of the art** — and solves 53% of libraries to a fully green upstream suite. This is a field/single-vs-swarm comparison with only the agent swapped (not the clean 3-arm control), and packaging-bound failures are kept in the denominator — full method, caveats, and the controlled JS→TS study are in the paper ([site](https://professorpalmer.github.io/durable-state-vs-context/) · [Zenodo/DOI](https://doi.org/10.5281/zenodo.20709565)). The [SWE-bench Lite cost/quality study](https://github.com/professorpalmer/swebench-pm) is a controlled 3-arm study where the CodeGraph-context + router arm lands about **47% cheaper than the frontier baseline at equal quality**.

## Quickstart

Inside Cursor Agent or Codex:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

```text
Use Puppetmaster to start a cursor swarm for this repo and return the job id immediately.
Problem: users get logged out after refresh and token-refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

From the shell:

```bash
puppetmaster doctor
puppetmaster route "Security audit every endpoint" --role audit   # dry-run routing decision
puppetmaster cursor "Review this repo for release blockers" --review --dry-run
puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
puppetmaster show $(puppetmaster last)
```

More recipes — including high/low effort model variants via `puppetmaster models setup` — are in [DAILY_DRIVER.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/DAILY_DRIVER.md) and [MODEL_ROUTING.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/MODEL_ROUTING.md).

## Recommended setup

The pattern that works best, and the one `setup` nudges your agent toward, is to keep a cheap conversational model in your IDE chat window and let it hand the technical work to Puppetmaster.

```text
You -- chat --> cheap conversational model (fast, low cost)
                      |
                      |  "this is real work" -> delegate
                      v
              Puppetmaster (routes to the cheapest sufficient model,
              runs durable workers, stores typed artifacts)
```

Conversational asks stay instant and cheap. Real engineering — multi-file investigation, a refactor, a review, an implementation — gets the full machine: cost routing, independent workers, durable artifacts, replay, and zero-token follow-up reads. Context compounds in artifacts and promoted memory that later jobs reuse, instead of in a chat scrollback that evaporates.

Don't try to route every chat message through Puppetmaster to "capture context." Spinning a durable worker for "hi" or "what's this function" inverts the value: it adds orchestration cost to the cheapest turns, and it fights the IDE, which has no clean intercept-every-message hook. Let the cheap model triage; let Puppetmaster do the work that deserves a job.

## Auto-invocation

The hard part of that pattern is getting the host agent to actually delegate without reminders. `setup` installs a classifier-gated enforcement layer for it:

- A pure-function gate (`puppetmaster should-delegate "<prompt>"`) answers delegate-vs-inline in microseconds with no LLM and no network. Typos, renames, one-liners, and quick questions stay inline; audits, refactors, migrations, and other broad-scope work delegate.
- Deterministic hooks in Cursor's `.cursor/hooks.json` and Claude Code's `.claude/settings.json` inject a "delegate now" directive on prompt submit and redirect genuinely broad shell searches to the Puppetmaster/CodeGraph equivalent. Read-only inspection (git log/show/diff, listing a directory, single-file greps, native Grep/Glob) passes through. Hooks fail open and never wedge a session. Add `--global` to cover every repo.
- An optional proxy (`puppetmaster proxy`) extends the same gate to OpenAI-compatible clients that closed harnesses can't hook.

There is no universal deterministic invocation: closed harnesses won't let anything sit on their provider wire. So the system is tiered — soft rules everywhere, hard hooks where the host exposes them, proxy only for clients routed through it — and fully kill-switchable with `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`.

## Output style and compression

Workers can optionally write tighter prose. `PUPPETMASTER_OUTPUT_STYLE=terse` (or `lithic`, or per-task `payload.output_style`) constrains form, not reasoning, so it trades verbosity for readability and latency without lowering answer quality. See [OUTPUT_STYLE.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/OUTPUT_STYLE.md).

Puppetmaster does not bundle input-side context compressors (RTK, Headroom, caveman). We measured them: the net savings are small and the failure modes (a compressor dropping data the agent then re-reads) run the wrong way for a coding agent. [COMPRESSION.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/COMPRESSION.md) shows the evidence and how to wire one yourself if you want it.

## Status

Daily-driver beta, currently at v1.9.0. Real runtime contract, automated tests, SQLite backend, fail-closed jobs, a live Cursor Agent MCP, and validated full-edit adapters. Credible for supervised local engineering; not yet a hosted multi-user service. Full feature matrix in [FEATURES.md](https://github.com/professorpalmer/Puppetmaster/blob/main/docs/FEATURES.md).

PyPI lists the package as [`puppetmaster-ai`](https://pypi.org/project/puppetmaster-ai/); [PEP 503 normalization](https://peps.python.org/pep-0503/#normalized-names) collides `puppetmaster` with an abandoned 2019 package. The import name, CLI, repo, and brand stay `puppetmaster`.

## License

MIT
