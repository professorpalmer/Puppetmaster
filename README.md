# Puppetmaster

Puppetmaster is a provider-neutral control plane for agent swarms.

It runs independent worker processes, coordinates them through shared state, forces structured artifacts instead of prose blobs, and stitches the results into replayable memory. Think Redis/Gunicorn for AI work: leases, workers, events, locks, artifacts, and deterministic synthesis.

## Why

Most swarms behave like group chats:

```text
Parent context
  |- child agent
  |- child agent
  `- child agent
```

That breaks down when context gets long, assumptions go stale, and every worker returns an unstructured summary.

Puppetmaster uses a distributed-systems shape:

```text
Agent workers -> shared coordination state -> structured artifacts -> stitcher
```

The rule is simple:

> Agents should not share full transcript context. They should share durable state.

## Features

- Independent OS subprocess workers.
- SQLite or file-backed coordination stores.
- Task DAGs with `blocked`, `queued`, `running`, `complete`, and `failed` states.
- Leases, heartbeats, stale-task recovery, event streams, and locks.
- Provider-neutral adapter registry.
- Built-in `local` and `shell` adapters.
- Optional live `cursor` adapter through `@cursor/sdk`.
- Safe `claude-code` and `codex` adapter stubs for future integrations.
- Structured artifacts with validation, evidence, confidence, and content hashes.
- Deterministic stitching and promoted memory.

## 60-Second Demo

```bash
git clone <your-puppetmaster-repo-url>
cd Puppetmaster

python -m unittest discover -s tests -v
python -m puppetmaster doctor
python -m puppetmaster run "Enterprise workflow" --config examples/enterprise-workflow.json
```

Inspect the run:

```bash
python -m puppetmaster jobs
python -m puppetmaster watch <job_id> --ticks 1
python -m puppetmaster show <job_id>
```

Run the crash-recovery demo:

```bash
python -m puppetmaster crash-demo
```

## Cursor Adapter

The Cursor adapter is optional and proves the provider boundary with a live agent runtime:

```bash
npm install
export CURSOR_API_KEY="crsr_..."
python -m puppetmaster run "Cursor agent workflow" --config examples/cursor-live.json
```

`cursor` uses `Agent.prompt(...)` in local mode with `settingSources: []`, so each worker is an isolated one-shot agent run.

## CLI

```bash
python -m puppetmaster doctor
python -m puppetmaster adapters
python -m puppetmaster init-config --path puppetmaster.json
python -m puppetmaster run "Goal" --config examples/enterprise-workflow.json
python -m puppetmaster crash-demo
python -m puppetmaster status <job_id>
python -m puppetmaster events <job_id>
python -m puppetmaster artifacts <job_id>
python -m puppetmaster memory
```

SQLite is the default backend. Use `--backend file` when you want maximally inspectable JSON files for debugging.

## Workflow Config

```json
{
  "lease_seconds": 5,
  "workers": [
    {
      "role": "explore",
      "instruction": "Map the goal and emit evidenced findings."
    },
    {
      "role": "verify-runtime",
      "instruction": "Verify Python is available before deeper work.",
      "adapter": "shell",
      "depends_on": ["explore"],
      "payload": {
        "command": ["python", "--version"],
        "timeout_seconds": 10
      }
    }
  ]
}
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Adapters](docs/ADAPTERS.md)
- [Roadmap](docs/ROADMAP.md)
- [Enterprise Workflow Example](examples/enterprise-workflow.json)
- [Cursor Live Example](examples/cursor-live.json)

## Status

Puppetmaster is early alpha software. The runtime contract is real, tests are automated, and the Cursor adapter has been exercised locally. It is not production-stable yet.

## Security

Never commit API keys. Use `.env.example` as the template and set provider credentials in your shell or secret manager. If you paste a key into a public chat, rotate it before publishing.

## License

MIT

