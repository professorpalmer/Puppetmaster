# Security & threat model

The honest one-line posture: **Puppetmaster is a local supervisor that drives coding agents which can read your code, edit files, and run shell commands. Treat installing it like granting an autonomous agent access to your shell and your existing LLM subscriptions — because that's what it is.** This page is the precise version of that sentence: what it can do, what it touches, what it deliberately does *not* do, and how to run it safely.

It runs entirely on your machine. There is no Puppetmaster cloud, no account, no phone-home, and no analytics. The only network traffic is your workers talking to the LLM providers you already use (and anything *you* explicitly turn on).

## What Puppetmaster can do (i.e. the attack surface you're accepting)

- **Spawn worker subprocesses** — each task runs as an independent OS process invoking an agent CLI/SDK (`cursor` SDK, `claude` CLI, `codex` CLI) or the OpenAI API.
- **Read and edit files in the target repo** (`cwd`). Full-edit adapters can modify your working tree.
- **Run shell commands** — the Claude Code and Codex agent loops can execute shell and git as part of their work. That is their whole value and their whole risk.
- **Read your local auth state** — to detect billing posture it reads `~/.claude.json` (`oauthAccount`), `~/.codex/auth.json` (`auth_mode`/`tokens`), and the presence of `CURSOR_API_KEY` / `OPENAI_API_KEY`. It reads these to *classify* (plan vs api); it does not copy, transmit, or persist the secret values.
- **Write local state** — a per-project SQLite store (jobs, tasks, typed artifacts, events, memory). Find yours with `puppetmaster projects`.

If you wouldn't let an AI agent do those things unsupervised in a given repo, don't point Puppetmaster at that repo without the guardrails below.

## Credentials & secrets

- **Uses the auth you already have.** Adapters read provider credentials from your environment / the platform's own login at call time (`CURSOR_API_KEY`, `OPENAI_API_KEY`, Claude OAuth in `~/.claude`, Codex login in `~/.codex`). Puppetmaster never asks you to paste a key into it.
- **Keys are never inlined or stored.** The model registry (`~/.puppetmaster/models.json`) describes models — ids, capability, price, billing posture — **not** key values. Routing/billing artifacts and `doctor` output record key *presence* (`cursor_api_key:set`) and never the secret itself; the Claude detector surfaces seat tier / org but redacts tokens.
- **The MCP config is the one place a key can live.** If you put a provider key in your `.cursor/mcp.json` env block, treat that file as a secret (it's gitignored by default — keep it that way). Prefer relying on the platform's own login over inlining keys.
- **Don't commit runtime state.** Everything under `.puppetmaster/` and `.cursor/` is local and gitignored; never stage it.

## Code execution & the edit-permission model

Read-only work (review, plan, route, discovery, follow-up reads) never edits anything. Write-capable adapters gate edits explicitly:

| Adapter | Default | Read-only mode | Full-bypass (opt-in) |
|---|---|---|---|
| `cursor` | `--review` / `--plan` are read-only; implement edits via the SDK | use review/plan | — |
| `claude-code` | `--permission-mode acceptEdits` (edits in `cwd`, no prompts) | `payload.permission_mode="plan"` / read-only modes | — |
| `codex` | `--sandbox workspace-write`, `--approval-policy never` | `payload.sandbox="read-only"` | `payload.dangerously_bypass_approvals_and_sandbox=true` |

Built-in guardrails:

- **Dirty-tree refusal.** Write-capable Claude/Codex runs refuse to start on a dirty working tree by default, so any resulting diff is attributable to the task (override with `payload.allow_dirty=true`, or run in a worktree).
- **Approve-before-edit workflow.** The recommended pattern is review/plan → read the stitched summary → approve → implement. Don't auto-run implement on untrusted code.
- **`--dangerously-bypass-approvals-and-sandbox` is strictly opt-in** and intended only for environments that are *already* externally sandboxed (a container/VM you control). Never set it on your primary machine against an untrusted repo.

## Network egress (exactly what leaves your machine)

- **Your prompts + code context → the LLM providers your workers use** (Cursor, Anthropic, OpenAI), exactly as if you used those CLIs directly. Their terms govern that data.
- **Optional catalog discovery** → `GET /v1/models` on OpenAI/Anthropic and the Cursor SDK, only when you run `models discover` (or first-run auto-discovery). Curated Claude/Codex catalogs are local — no call.
- **Optional live preflight probe** → a single real 1-token call to the routed provider, only with `--live` / `payload.live_preflight`.
- **Optional OpenTelemetry** → only to the OTLP endpoint *you* set. **Off by default** (no endpoint, no export). There is no default telemetry and no vendor collector.

That's the complete list. No usage analytics, no license check, no background calls.

## Data at rest

- Jobs, tasks, typed artifacts, events, and memory live in a **local SQLite** store per project (WAL mode). Nothing is uploaded.
- Artifacts contain the worker's structured output (claims, evidence, diffs, confidence) — i.e. they can contain snippets of your code. They're as sensitive as the repo. Delete the state dir to purge.

## What Puppetmaster does NOT do

- No hosted service, no account, no login to "Puppetmaster."
- No phone-home, telemetry, or analytics by default.
- No remote control — the MCP server is a **local stdio** process owned by your editor; it does not open a network port.
- It does not exfiltrate keys, and it doesn't run workers you didn't start.

## Hardening recommendations

1. **Untrusted or high-stakes repo?** Use read-only modes (`cursor --review/--plan`, `codex payload.sandbox="read-only"`) until you've read the plan, then approve edits explicitly.
2. **Run implement work in a dedicated git worktree** so diffs are isolated and a bad run is `git worktree remove` away.
3. **Scope your provider tokens** to least privilege; prefer the platform's own login over inlining keys in MCP config.
4. **Only use `--dangerously-bypass...` inside a container/VM** you already trust to be sandboxed.
5. **Keep `.cursor/` and `.puppetmaster/` out of git** (default gitignored — leave it).
6. **Review the stitched Alerts section** — billing/auth/permission failures are surfaced there, not buried.

## Reporting a vulnerability

Found a security issue? Please open a GitHub security advisory (or a minimal private report) on the [repository](https://github.com/professorpalmer/Puppetmaster) rather than a public issue, and give maintainers a chance to fix it before disclosure.

## Honest limitations

- Puppetmaster orchestrates **autonomous agents that can edit code and run shell**. No amount of dirty-tree guarding changes the fundamental trust decision: you are letting an LLM act in your repo. The guardrails make it auditable and reversible; they don't make it risk-free.
- Anything you send to a provider is subject to **that provider's** data handling, not Puppetmaster's. Puppetmaster adds no privacy layer on top of the underlying CLIs.
- This is **daily-driver beta, single-author** software. It has tests and fail-closed defaults, but it has not had an independent security audit. Run it accordingly.
