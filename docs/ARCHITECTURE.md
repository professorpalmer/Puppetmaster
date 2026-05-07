# Architecture

Puppetmaster treats agent swarms like distributed systems rather than group chats.

```mermaid
flowchart TD
  CLI["CLI"] --> Supervisor["Supervisor"]
  Supervisor --> Store["Coordination Store"]
  Supervisor --> WorkerProc["Worker Processes"]
  WorkerProc --> Adapter["Adapter Registry"]
  Adapter --> Artifacts["Structured Artifacts"]
  Store --> Stitcher["Stitcher"]
  Artifacts --> Stitcher
  Stitcher --> Summary["Summary"]
  Stitcher --> Memory["Promoted Memory"]
```

## Core Objects

- `Job`: one swarm run and user goal.
- `Task`: a role-specific unit of work, optionally dependent on other tasks.
- `AgentRun`: one attempt by one worker process.
- `Artifact`: structured worker output with evidence, confidence, payload, and `sha256`.
- `MemoryRecord`: promoted facts that future workers can retrieve.

## Runtime Flow

1. The CLI creates a `Job`.
2. The supervisor creates a task DAG.
3. Downstream tasks start as `blocked`.
4. Worker subprocesses claim ready tasks with leases.
5. Long-running workers heartbeat and renew leases.
6. Workers emit structured artifacts.
7. Stale leases can be recovered back to `queued`.
8. The stitcher reads artifacts only and writes `stitched.md`.

## Backends

The default backend is SQLite with WAL enabled. It stores jobs, tasks, runs, artifacts, memory, and events in the resolved Puppetmaster state directory.

By default that directory is outside the target repository, under per-user app state:

```text
macOS: ~/Library/Application Support/puppetmaster/projects/<workspace>-<hash>/
Linux: ~/.local/state/puppetmaster/projects/<workspace>-<hash>/
```

Use `python -m puppetmaster state` to print the resolved path. Use `--state-dir` or `PUPPETMASTER_STATE_DIR` when you intentionally want a different location, such as CI state or explicit repo-local `.puppetmaster/`.

The file backend remains useful for debugging because every object is a readable JSON file.

## Failure Model

Workers are allowed to die. The lease expires, the task becomes recoverable, and another worker can reclaim it. The crash demo exercises this path.

```bash
python -m puppetmaster crash-demo
```

## Design Rules

- Workers do not communicate directly.
- Durable state goes through the coordination store.
- Final synthesis reads artifacts, not transcripts.
- Artifacts require evidence and type-specific payload fields.
- Optional providers must fail as structured artifacts, not runtime crashes.

