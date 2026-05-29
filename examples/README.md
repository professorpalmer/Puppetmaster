# Example workflow configs

Each JSON file here is a ready-to-run workflow spec for `puppetmaster run --config <file>`. They are the fastest way to see a given adapter or coordination pattern without writing a spec by hand.

```bash
puppetmaster run "demo goal" --config examples/enterprise-workflow.json
```

| File | What it demonstrates |
|---|---|
| [`enterprise-workflow.json`](enterprise-workflow.json) | Multi-role swarm (plan + review + redteam + …) over the SQLite backend — the default daily-driver shape |
| [`memory-reuse.json`](memory-reuse.json) | Independent workers reusing promoted memory across tasks |
| [`cursor-review.json`](cursor-review.json) | Read-only review pass via the `cursor` adapter |
| [`cursor-dry-run-implementation.json`](cursor-dry-run-implementation.json) | Plan/implement dry-run that proposes edits without writing them |
| [`cursor-live.json`](cursor-live.json) | Live `cursor` adapter run (requires `CURSOR_API_KEY`) |
| [`claude-code-full-edit.json`](claude-code-full-edit.json) | Full-edit implementation via the `claude` CLI, emitting patch artifacts |

The [`transcripts/`](transcripts) folder holds captured example outputs for reference.

The workflow config schema (roles, adapters, payloads, routing overrides, DAG edges) is documented in [docs/CLI_REFERENCE.md](../docs/CLI_REFERENCE.md).
