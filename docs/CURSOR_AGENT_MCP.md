# Cursor Agent MCP Integration

The Cursor extension gives Puppetmaster a visual control panel.

The MCP server gives Cursor's Agent chat access to Puppetmaster tools.

That is the integration path for using Puppetmaster from the Agent surface instead of manually typing terminal commands.

## What It Adds

Puppetmaster exposes these MCP tools:

- `puppetmaster_doctor`
- `puppetmaster_cursor_review`
- `puppetmaster_cursor_plan`
- `puppetmaster_claude_implement`
- `puppetmaster_last_job`
- `puppetmaster_logs`
- `puppetmaster_artifacts`
- `puppetmaster_show`

Cursor's Agent can call those tools during a chat, so you can ask it to run a Puppetmaster review, plan, or Claude Code implementation from inside Cursor.

## How It Respects Independent Workers

MCP is the bridge into Cursor Agent. It is not a hook into Cursor's private model picker, composer internals, or native subagent scheduler.

The control flow is:

```text
Cursor Agent chat
  -> MCP tool call
  -> puppetmaster.mcp_server
  -> python -m puppetmaster
  -> independent worker subprocesses
  -> SQLite/shared state/artifacts
  -> stitched summary back to Cursor Agent
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
Use Puppetmaster to run a Cursor review dry run for this repo. Focus on release blockers.
```

```text
Use Puppetmaster with Claude Code to implement the approved fix in a clean worktree.
```

## Current Boundary

Cursor does not expose the internal Agent model picker or composer controls as a public extension API. MCP is the supported tool surface for Agent-chat integration, while the extension is the supported UI surface for a native control panel.
