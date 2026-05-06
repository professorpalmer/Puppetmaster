# Changelog

## v0.2.0-beta.1

- Position Puppetmaster as a supervised daily-driver beta for Cursor swarm workflows.
- Add run-management commands for latest job lookup, logs, reruns, cleanup, patch artifact review, and approval events.
- Add dry-run Cursor workflow support.
- Add full-edit Claude Code CLI adapter and `puppetmaster claude` command.
- Harden job and worker failure handling.
- Add SQLite schema version metadata and doctor validation.
- Retrieve promoted memory into new task payloads.
- Expand production, security, contribution, and release documentation.

## v0.1.0

- Establish the core local swarm runtime.
- Add subprocess workers, structured artifacts, stitching, and memory promotion.
- Add file and SQLite coordination backends.
- Add shell and optional Cursor adapters.
- Add crash recovery demo, CI, and starter docs.

## Planned v0.2.0

- Broaden provider adapters beyond Cursor.
- Improve patch artifact generation and isolated apply flows.
- Add richer watch output and scripting-friendly JSON modes.
