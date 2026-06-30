# Cursor Agent MCP Integration

The MCP server gives Cursor's Agent chat access to Puppetmaster tools.

That is the integration path for using Puppetmaster from the Agent surface instead of manually typing terminal commands.

## What It Adds

Puppetmaster exposes these MCP tools:

- `puppetmaster_doctor`
- `puppetmaster_cursor_review`
- `puppetmaster_start_cursor_review`
- `puppetmaster_cursor_plan`
- `puppetmaster_start_cursor_plan`
- `puppetmaster_claude_implement`
- `puppetmaster_start_claude_implement`
- `puppetmaster_start_swarm`
- `puppetmaster_start_cursor_swarm`
- `puppetmaster_start_browser_swarm`
- `puppetmaster_last_job`
- `puppetmaster_status`
- `puppetmaster_logs`
- `puppetmaster_live_artifacts`
- `puppetmaster_live_artifacts_follow`
- `puppetmaster_partial_summary`
- `puppetmaster_artifacts`
- `puppetmaster_show`
- `puppetmaster_dashboard`
- `puppetmaster_codegraph_search`
- `puppetmaster_codegraph_context`
- `puppetmaster_codegraph_affected`
- `puppetmaster_codegraph_files`
- `puppetmaster_codegraph_status`
- `puppetmaster_codegraph_init`

Cursor's Agent can call those tools during a chat, so you can ask it to run a Puppetmaster review, plan, or Claude Code implementation from inside Cursor.

## Bundled CodeGraph (no second MCP)

The `puppetmaster_codegraph_*` tools proxy the most useful CodeGraph CLI commands (`codegraph query`, `codegraph context`, `codegraph affected`, `codegraph files`, `codegraph status`, `codegraph init`). With these tools registered, Cursor Agent gets both Puppetmaster orchestration and CodeGraph repo intelligence through a single MCP server.

Use them for quick repo lookups between (or instead of) swarm runs:

```text
Use puppetmaster_codegraph_search to find every implementation of UserService.
Use puppetmaster_codegraph_context for "fix login redirect bug" before opening files.
Use puppetmaster_codegraph_affected for the files I just changed so we only run impacted tests.
```

Each tool degrades cleanly: if the `codegraph` CLI is not installed, or the workspace is not initialized, the response is a non-fatal `isError: true` payload with a one-line fix-it hint. Run `puppetmaster_codegraph_init` once to bootstrap a workspace.

Power users who want CodeGraph's full MCP surface (`codegraph_callers`, `codegraph_callees`, `codegraph_impact`, `codegraph_node`) — only exposed through CodeGraph's own MCP server — can still run `codegraph serve --mcp` alongside Puppetmaster's MCP. Bundling covers the daily-driver case so two MCP entries are no longer required by default.

For anything longer than a quick health check, prefer the `puppetmaster_start_*` tools. They return a `job_id` immediately and let Cursor poll `puppetmaster_status`, `puppetmaster_logs`, and `puppetmaster_show` instead of holding one long MCP request open.

Use `puppetmaster_start_cursor_swarm` for real multi-role code analysis. It generates a temporary workflow config with each role backed by the Cursor SDK adapter. Bare custom roles on `puppetmaster_start_swarm` require an explicit config or adapter; otherwise Puppetmaster returns an error instead of silently running deterministic demo workers.

## Live Artifact Board

Puppetmaster is not meant to hide all useful work until the end. Workers write artifacts and events as they run. Cursor Agent can inspect the job mid-run:

- `puppetmaster_status`: current task/artifact counts
- `puppetmaster_logs`: event stream
- `puppetmaster_live_artifacts`: live evidence board from saved artifacts
- `puppetmaster_live_artifacts_follow`: long-poll for new artifacts since a cursor (push-style stream over MCP)
- `puppetmaster_partial_summary`: current synthesis from artifacts already emitted
- `puppetmaster_show`: final stitched summary after completion
- `puppetmaster_dashboard`: ensure the local web dashboard is serving (idempotent, loopback-only) and get its URL — ask the Agent to "open the Puppetmaster dashboard" and it opens the returned URL in a browser tab, optionally deep-linked to one job via `job_id`

Final stitching is the publishable report. The live artifact board is the shared coordination surface.

### Push-style live feed (no polling loop in the Agent)

`puppetmaster_live_artifacts_follow` blocks server-side until new artifacts arrive (or `timeout_seconds` elapses, default 10s) and returns a `next_cursor`. Chain calls with that cursor to get a push-feeling stream of new artifacts without the Agent running its own polling loop:

```text
1. Call puppetmaster_live_artifacts_follow with since_cursor=0.
   Server returns immediately with the current artifacts and next_cursor=N.
2. Call again with since_cursor=N.
   Server blocks up to timeout_seconds and returns the next batch (or {timed_out: true}).
3. Repeat with the latest next_cursor.
```

Internally this is a cursor-based read over the durable SQLite event log (or the file-backed `.jsonl` stream when `backend=file`). The store wakes up within ~50ms of the next artifact write, so latency feels real-time without taking on Redis or any extra daemon.

## How It Respects Independent Workers

MCP is the bridge into Cursor Agent. It is not a hook into Cursor's private model picker, composer internals, or native subagent scheduler.

The control flow is:

```text
Cursor Agent chat
  -> MCP start tool
  -> detached python -m puppetmaster job
  -> independent worker subprocesses
  -> SQLite events + artifacts as workers run
  -> Cursor polls status/logs/live_artifacts/partial_summary
  -> final stitched summary when complete
```

So Puppetmaster does not force Cursor's built-in subagents to use our process model. It gives the Agent a way to delegate work to Puppetmaster instead. After that handoff, the important guarantees come from the Puppetmaster runtime:

- Workers run as separate OS subprocesses.
- Workers claim tasks through leases instead of shared chat history.
- Workers communicate through durable state and typed artifacts.
- The stitcher summarizes artifacts back into Cursor instead of merging raw worker transcripts into the Agent chat.

That is the resource/context boundary: Cursor Agent is the operator, while Puppetmaster is the worker runtime.

## Local Config

This repo includes `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "puppetmaster": {
      "command": "python",
      "args": ["-m", "puppetmaster.mcp_server"],
      "env": {
        "CLAUDE_CODE_COMMAND": "npx -y @anthropic-ai/claude-code"
      }
    }
  }
}
```

Open the Puppetmaster repo in Cursor, then enable/reload MCP servers from Cursor settings if prompted.

MCP uses Puppetmaster's normal state resolution. Unless a tool call passes `state_dir`, jobs and MCP run logs go to per-workspace app state outside the repository. Ask Cursor Agent to call `puppetmaster_doctor` or run `python -m puppetmaster state` if you need the exact path.

## Provider Keys

Do not commit provider keys.

For local use, configure keys in your shell/Cursor environment:

```bash
export CURSOR_API_KEY="<your-cursor-api-key>"
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
export CLAUDE_CODE_COMMAND="npx -y @anthropic-ai/claude-code"
```

For a team or packaged install, configure those environment variables in Cursor's MCP server settings or use the extension's secret storage for panel-driven runs.

## Example Agent Prompts

Ask Cursor Agent:

```text
Use Puppetmaster to run doctor in this repo and tell me what is missing.
```

```text
Use Puppetmaster to start a Cursor review dry run for this repo. Return the job id immediately, then poll status and summarize the result when complete. Focus on release blockers.
```

```text
Use Puppetmaster to start a Cursor swarm for this repo with roles pipeline-mapper, decision-explainer, conflict-auditor, and test-coverage-reviewer. Return the job id immediately, then poll status/logs/live artifacts and summarize concrete file-backed findings as they arrive.
```

```text
Use Puppetmaster to start Claude Code implementation for the approved fix in a clean worktree. Return the job id immediately, poll status/logs, and show me the stitched summary when complete.
```

## Current Boundary

Cursor does not expose the internal Agent model picker or composer controls as a public extension API. MCP is the supported tool surface for Agent-chat integration.
