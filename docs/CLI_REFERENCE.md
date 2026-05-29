# CLI Reference

Every Puppetmaster MCP tool has a matching CLI subcommand. These are
the ones you'll use day-to-day from a shell; the full list is also
visible via `python -m puppetmaster --help`.

## Setup and inspection

```bash
python -m puppetmaster setup            # one-shot: doctor + models init + install-cursor-mcp + install-codex-mcp + install-rules
python -m puppetmaster doctor           # 15 health checks
python -m puppetmaster adapters         # list available worker adapters (json)
python -m puppetmaster state            # print the resolved per-workspace state dir
python -m puppetmaster init             # create the local state store
python -m puppetmaster init-config --path puppetmaster.json
```

## MCP wiring (one-liner installers, v0.7.2+)

```bash
python -m puppetmaster install-cursor-mcp           # workspace .cursor/mcp.json
python -m puppetmaster install-cursor-mcp --global  # ~/.cursor/mcp.json
python -m puppetmaster install-codex-mcp            # codex mcp add ...
python -m puppetmaster install-rules                # write .cursor/rules/puppetmaster.mdc + AGENTS.md
python -m puppetmaster install-rules --global       # also ~/.codex/instructions.md and ~/.claude/CLAUDE.md
```

All four installers resolve `sys.executable`, run a `tools/list`
handshake before writing anything, are idempotent (re-run =
`unchanged`), and preserve existing user content in the target files.

## Running swarms

```bash
python -m puppetmaster run "Goal" --config examples/enterprise-workflow.json
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster cursor "Goal" --review --dry-run
python -m puppetmaster claude "Goal" --permission-mode acceptEdits
python -m puppetmaster codex "Goal"
python -m puppetmaster openai "Goal"
python -m puppetmaster crash-demo
```

## Routing / cost

```bash
python -m puppetmaster route "instruction" --role <role>     # dry-run, returns picked model + cost
python -m puppetmaster cost <job_id>                          # sum spend across all routing artifacts
python -m puppetmaster models init                            # write starter registry
python -m puppetmaster models list                            # show registered models
python -m puppetmaster models path                            # print resolved registry path
```

## Platform lock (restrict which platforms get used)

```bash
python -m puppetmaster platform status                        # show each platform on/off
python -m puppetmaster platform only cursor                   # lock to Cursor only (single-platform mode)
python -m puppetmaster platform enable claude-code codex      # turn platforms back on
python -m puppetmaster platform disable openai                # turn a platform off
python -m puppetmaster platform reset                         # clear the lock (all platforms on)
# ephemeral / CI override (wins over saved config):
PUPPETMASTER_ONLY_ADAPTERS=cursor python -m puppetmaster route "audit" --role audit
```

A disabled platform is never routed to, auto-discovered, or used for fallback. Default is everything-on. Lock state persists in `~/.puppetmaster/platform.json`.

## Inspection (read-only, auto-pivots across workspaces v0.5.4+)

```bash
python -m puppetmaster status <job_id>
python -m puppetmaster watch <job_id>
python -m puppetmaster events <job_id>
python -m puppetmaster feed [job_id] [--follow]
python -m puppetmaster artifacts <job_id>
python -m puppetmaster logs [job_id]
python -m puppetmaster open [job_id]
python -m puppetmaster last
python -m puppetmaster show <job_id>
python -m puppetmaster diff [job_id]
python -m puppetmaster jobs [--all-projects]
python -m puppetmaster projects
python -m puppetmaster memory
```

## Lifecycle

```bash
python -m puppetmaster rerun [job_id]
python -m puppetmaster approve <job_id-or-artifact-id>
python -m puppetmaster reject <job_id-or-artifact-id> --reason "why"
python -m puppetmaster clean --completed
python -m puppetmaster recover                # recover stale-leased tasks
python -m puppetmaster repair-codegraph       # rebuild CodeGraph's native SQLite binding for Cursor's bundled Node
```

## MCP server management

```bash
python -m puppetmaster mcp list               # show every tracked MCP server PID
python -m puppetmaster mcp cleanup --kill-stale
```

## Workflow config schema

A workflow config is JSON describing the worker DAG for one swarm:

```json
{
  "lease_seconds": 10,
  "workers": [
    {
      "role": "explore",
      "instruction": "Map the goal and emit evidenced findings."
    },
    {
      "role": "claude-implement",
      "instruction": "Use Claude Code to implement the requested change.",
      "adapter": "claude-code",
      "depends_on": ["explore"],
      "payload": {
        "prompt": "Implement the change and run focused tests.",
        "cwd": ".",
        "permission_mode": "acceptEdits",
        "allowed_tools": ["Read", "Edit", "MultiEdit", "Write", "Bash"],
        "timeout_seconds": 900,
        "allow_dirty": false
      }
    }
  ]
}
```

Examples (each ships in [`examples/`](../examples/)):

- [Enterprise Workflow](../examples/enterprise-workflow.json)
- [Cursor Live](../examples/cursor-live.json)
- [Cursor Review](../examples/cursor-review.json)
- [Cursor Dry-Run Implementation](../examples/cursor-dry-run-implementation.json)
- [Claude Code Full Edit](../examples/claude-code-full-edit.json)
- [Memory Reuse](../examples/memory-reuse.json)

## Daemon mode

For local swarms, keep Puppetmaster workers warm and let jobs hand
off work to them:

```bash
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster run "Review this repo" --worker-mode daemon
```

Daemon mode keeps the Puppetmaster worker loop alive across jobs.
Lease-based task claiming and artifacts are preserved; only the
worker process startup cost is amortized.

## Clean worktree for real edits

```bash
git worktree add /tmp/puppetmaster-work -b puppetmaster-work
python -m puppetmaster claude "Implement the approved fix" --cwd /tmp/puppetmaster-work --permission-mode acceptEdits
```
