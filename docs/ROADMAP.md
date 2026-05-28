# Roadmap

## Production Runtime

- Tighten approval-gated patch application with generated unified diffs.
- Add optional worktree creation for apply flows, not just existing worktree targets.
- Add richer exit-code and JSON output coverage for automation.
- Add signed artifact manifests and tamper-evident event exports.

## Daily-Driver UX

- Live Cursor SDK integration test gated by `CURSOR_API_KEY`.
- Better watch output or a small terminal UI.
- Saved workflow presets for review, plan, implement, redteam, and verify loops.

## Research Work

- Measurable comparisons against parent-child swarms.
- Artifact quality scoring.
- Memory promotion policies and expiration.
- Multi-agent evaluation harnesses.

## Provider Integrations

- Claude Code adapter.
- ~~Codex adapter.~~ Shipped in v0.7.0 — `CodexAdapter` shells out to `codex exec --json` and captures billing-grade token counts from the structured event stream. See `docs/ADAPTERS.md#codex`.
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

- Daily-driver Cursor UX.
- Memory retrieval into worker context.
- Patch approval workflow with path locks.
- SQLite schema metadata and doctor checks.
- Provider adapter expansion.
- TUI/watch improvements.

