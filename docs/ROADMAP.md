# Roadmap

## Near Term

- Memory retrieval into worker context.
- Human approval gates for patch application and memory promotion.
- Path-level locks around real repository edits.
- Live Cursor SDK integration test gated by `CURSOR_API_KEY`.
- Better watch output or a small terminal UI.

## Provider Integrations

- Claude Code adapter.
- Codex adapter.
- Cursor cloud-agent mode with durable resume.
- Generic HTTP adapter for internal agent services.

## Backends

- Redis Streams backend.
- Postgres backend.
- Store migration/export/import.

## Evaluation

Puppetmaster needs measurable comparisons before any serious research claim:

- Parent-child swarm vs artifact-stitching swarm.
- Context token usage.
- Task completion rate.
- Crash recovery.
- Hallucinated or unsupported claims.
- Reproducibility from stored artifacts.

## Release Goals

### `v0.1.0`

- Local SQLite runtime.
- Subprocess workers.
- Adapter contract.
- Shell and Cursor adapters.
- Crash recovery demo.
- CI and docs.

### `v0.2.0`

- Memory retrieval.
- Patch approval workflow.
- Provider adapter expansion.
- TUI/watch improvements.

