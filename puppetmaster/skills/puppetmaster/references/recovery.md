# Recovery guidance

This file contains version- and transport-sensitive recovery, not the normal
workflow.

## MCP disconnected

Keep the existing `job_ref`. Query recent jobs before retrying a start; a lost
MCP response can coexist with a live detached job. Continue through CLI:

```powershell
python -m puppetmaster status <job_id>
python -m puppetmaster await <job_id> --timeout-seconds 45 --json
python -m puppetmaster show <job_id>
```

Restart MCP only when `python -m puppetmaster mcp list` shows no applicable live
server or a verified running/on-disk version mismatch. Never restart the job just
because its observation transport was lost.

## Windows launch failure

Modern internal swarm launches and default Codex/Claude worker prompts use
file/stdin transport instead of large argv values. On older versions, keep goals
compact or point workers at a workspace brief. Treat `WinError 206` as a
pre-worker failure and upgrade rather than retrying the same oversized command.

Default Codex npm shims are resolved to Node plus the real JavaScript entrypoint.
Explicit custom executable overrides remain the operator's responsibility.

## Unexpected analysis edit

Current analysis runs fail their delivery gate when a worker-attributable diff
appears. Stop only that job's process tree, preserve unrelated dirty work, and
restore only paths proven to belong to the worker. Never use a broad reset or
cleanup command as incident recovery.
