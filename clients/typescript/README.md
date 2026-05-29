# @puppetmaster/client (TypeScript)

True **blocking await** for Puppetmaster jobs from any Node/TS process — CI
steps, backend services, scripts. The MCP `puppetmaster_await_job` tool is a
*bounded* long-poll because Cursor's stdio transport can't hold a long call
open; this client has no such constraint. It drives the durable CLI
(`python -m puppetmaster await <job_id> --json`), which reads the same
SQLite/file-backed state the daemon writes, so it works from any process that
can see the state dir.

Zero runtime dependencies (`node:child_process` only).

## Usage

```ts
import { awaitJob } from "@puppetmaster/client";

// Block until the job finishes, then read its stitched summary.
const result = await awaitJob("job_abc123");
if (result.status === "complete") {
  console.log(result.summary);
} else {
  console.error("degraded/failed:", result.status);
}

// Bounded wait (returns timed_out: true if it doesn't finish in 60s).
const bounded = await awaitJob("job_abc123", { timeoutSeconds: 60 });

// Point at a specific state dir (e.g. a different project / machine mount).
await awaitJob("job_abc123", { env: { PUPPETMASTER_STATE_DIR: "/data/pm" } });
```

## Requirements

- `puppetmaster-ai` installed and importable by the `python` (default
  `python3`) the client spawns. Override with the `python` option.
- Access to the same Puppetmaster state dir that produced the job.

## API

`awaitJob(jobId, options?) => Promise<AwaitJobResult>`

`options`: `timeoutSeconds` (0 = block forever), `pollIntervalSeconds`,
`python`, `cwd`, `env`, `killAfterSeconds`.

`AwaitJobResult`: `{ job_id, status, terminal, timed_out, completed_at, summary }`.

A job that finishes as **failed** resolves normally (the await succeeded); the
promise only rejects when the CLI can't run or returns no parseable JSON.
