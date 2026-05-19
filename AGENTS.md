# AGENTS.md

Repo-level guidance for AI coding agents (Cursor, Claude Code, Codex, etc.) working in this repository. Cursor users see the same directive via `.cursor/rules/puppetmaster-workflow.mdc`.

## Default subagent runtime: Puppetmaster

When this repo is open, **Puppetmaster's MCP tools are the primary path for fanning out work**. Do not roll your own grep/read exploration loops or invoke ad-hoc subagents when a Puppetmaster swarm is available.

## When to start a Puppetmaster swarm (do this by default)

For any request that involves more than a single trivial change, start a Puppetmaster swarm **before** spending tokens on your own exploration:

- Broad repo investigation, audit, or risk analysis ("review this", "find issues", "what could break").
- Multi-file refactors, migrations, or cross-cutting cleanups.
- Debugging that requires tracing call graphs, data flow, or test coverage across multiple files.
- Planning the next implementation slice when scope or risks are unclear.
- Comparing approaches, surfacing trade-offs, or producing decision artifacts.

Default routing:

1. `puppetmaster_start_cursor_swarm` — multi-role read-only analysis (the daily-driver entry point).
2. `puppetmaster_start_claude_implement` — durable, patch-producing implementation in a clean worktree.
3. `puppetmaster_start_cursor_review` / `puppetmaster_start_cursor_plan` — lightweight single-pass review or plan.

Start tools return a `job_id` immediately. Do **not** wait inside one long MCP call.

## When NOT to route through Puppetmaster

Use native tooling directly for:

- Trivial single-file edits with obvious intent (rename, add comment, fix typo).
- Questions answerable from the current visible file or recent context.
- Conversational follow-ups that don't change repo state.
- Anything explicitly framed as "just answer me" / "no swarm".

## How to drive a started swarm

1. Return the `job_id` to the user immediately, in one line.
2. Prefer `puppetmaster_live_artifacts_follow` (long-poll, push-style) over polling `puppetmaster_status` in a loop. Chain calls with the returned `next_cursor`.
3. Use `puppetmaster_partial_summary` for a current synthesis without waiting for final stitching.
4. Summarize concrete file-backed findings, risks, and open questions — never raw worker transcripts.
5. Ask for approval before implementation unless the user already approved edits.

If a swarm completes with empty findings, only verification artifacts, or a degraded Cursor SDK artifact, report Puppetmaster as **degraded** and do not treat the run as a successful analysis.

## Repo intelligence (CodeGraph)

Puppetmaster auto-injects CodeGraph context into every Cursor and Claude Code worker prompt when `.codegraph/` exists in the target repo. The verification artifact's `evidence` array will include `context:codegraph` when this happened.

For quick, direct repo lookups without spinning up a swarm, prefer the bundled tools: `puppetmaster_codegraph_search`, `puppetmaster_codegraph_context`, `puppetmaster_codegraph_affected`, `puppetmaster_codegraph_files`, `puppetmaster_codegraph_status`. If CodeGraph isn't initialized, call `puppetmaster_codegraph_init` once first.

## Coding conventions in this repo

- Python 3.9+. Match existing style; do not introduce new dependencies without a clear need.
- Tests live in `tests/test_puppetmaster.py`. New behavior gets a focused test; mock subprocess calls for adapter coverage.
- The `SwarmStore` abstract base is the storage seam — the SQLite implementation lives in `puppetmaster/sqlite_store.py` and overrides hot methods for O(1) cursor reads. Don't break that contract.
- Don't commit provider keys, `.cursor/mcp.json` user-specific paths, or anything in `.puppetmaster/` runtime state.
- Run `python -m pytest tests/test_puppetmaster.py -q` before suggesting a commit.

## MCP surface (quick reference)

Orchestration: `puppetmaster_doctor`, `puppetmaster_start_swarm`, `puppetmaster_start_cursor_swarm`, `puppetmaster_start_cursor_review`, `puppetmaster_start_cursor_plan`, `puppetmaster_start_claude_implement`, `puppetmaster_status`, `puppetmaster_logs`, `puppetmaster_live_artifacts`, `puppetmaster_live_artifacts_follow`, `puppetmaster_partial_summary`, `puppetmaster_artifacts`, `puppetmaster_show`, `puppetmaster_last_job`.

Bundled CodeGraph: `puppetmaster_codegraph_search`, `puppetmaster_codegraph_context`, `puppetmaster_codegraph_affected`, `puppetmaster_codegraph_files`, `puppetmaster_codegraph_status`, `puppetmaster_codegraph_init`.

## When MCP fails: fall back to the CLI

If any `puppetmaster_*` MCP tool returns `Tool execution error. Not connected`, the daemon is almost certainly still running — only Cursor's stdio link for this chat dropped. **Do not stop work.** Every MCP tool has a CLI equivalent that talks to the same SQLite state:

| MCP | CLI |
| --- | --- |
| `puppetmaster_doctor` | `python -m puppetmaster doctor` |
| `puppetmaster_status` | `python -m puppetmaster status <job_id>` |
| `puppetmaster_logs` | `python -m puppetmaster logs <job_id>` |
| `puppetmaster_live_artifacts` | `python -m puppetmaster feed <job_id>` |
| `puppetmaster_live_artifacts_follow` | `python -m puppetmaster feed <job_id> --follow` |
| `puppetmaster_partial_summary` | `python -m puppetmaster show <job_id> --partial` |
| `puppetmaster_artifacts` | `python -m puppetmaster artifacts <job_id>` |
| `puppetmaster_show` | `python -m puppetmaster show <job_id>` |
| `puppetmaster_last_job` | `python -m puppetmaster last` |
| `puppetmaster_jobs` | `python -m puppetmaster jobs [--all-projects]` |
| `puppetmaster_mcp_status` | `python -m puppetmaster mcp list` |
| `puppetmaster_mcp_cleanup` | `python -m puppetmaster mcp cleanup --kill-stale` |
| `puppetmaster_repair_codegraph` | `python -m puppetmaster repair-codegraph` |
| `puppetmaster_codegraph_status` | `codegraph status` |
| `puppetmaster_codegraph_search` | `codegraph search '<query>'` |
| `puppetmaster_codegraph_context` | `codegraph context '<task>' --max-nodes 15 --format markdown` |
| `puppetmaster_codegraph_init` | `codegraph init --index` |

Read-only commands (`show`/`artifacts`/`logs`/`feed`/`status`) auto-pivot to whichever project state dir owns the job — no need to export `PUPPETMASTER_STATE_DIR`. Only ask the user to restart MCP in Cursor Settings when `python -m puppetmaster mcp list` shows zero alive servers.
