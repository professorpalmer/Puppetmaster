# Cursor Agent MCP Integration

The Cursor extension gives Puppetmaster a visual control panel.

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
- `puppetmaster_last_job`
- `puppetmaster_status`
- `puppetmaster_logs`
- `puppetmaster_artifacts`
- `puppetmaster_show`

Cursor's Agent can call those tools during a chat, so you can ask it to run a Puppetmaster review, plan, or Claude Code implementation from inside Cursor.

For anything longer than a quick health check, prefer the `puppetmaster_start_*` tools. They return a `job_id` immediately and let Cursor poll `puppetmaster_status`, `puppetmaster_logs`, and `puppetmaster_show` instead of holding one long MCP request open.

Use `puppetmaster_start_cursor_swarm` for real multi-role code analysis. It generates a temporary workflow config with each role backed by the Cursor SDK adapter. Bare custom roles on `puppetmaster_start_swarm` require an explicit config or adapter; otherwise Puppetmaster returns an error instead of silently running deterministic demo workers.

## How It Respects Independent Workers

MCP is the bridge into Cursor Agent. It is not a hook into Cursor's private model picker, composer internals, or native subagent scheduler.

The control flow is:

```text
Cursor Agent chat
  -> MCP start tool
  -> puppetmaster.mcp_server
  -> detached python -m puppetmaster job
  -> independent worker subprocesses
  -> SQLite/shared state/artifacts
  -> Cursor polls status/logs/show by job_id
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
Use Puppetmaster to start a Cursor swarm for this repo with roles pipeline-mapper, decision-explainer, conflict-auditor, and test-coverage-reviewer. Return the job id immediately, then poll status/logs and summarize concrete file-backed findings.
```

```text
Use Puppetmaster to start Claude Code implementation for the approved fix in a clean worktree. Return the job id immediately, poll status/logs, and show me the stitched summary when complete.
```

## Current Boundary

Cursor does not expose the internal Agent model picker or composer controls as a public extension API. MCP is the supported tool surface for Agent-chat integration, while the extension is the supported UI surface for a native control panel.
