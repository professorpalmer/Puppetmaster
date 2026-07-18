# Production Notes

Puppetmaster is currently a supervised local runtime, not a hosted production control plane.

## What Is Hardened

- Python 3.9 through 3.12 are covered by CI.
- Stable commands include `doctor`, `adapters`, `run`, `cursor`, `claude`, `jobs`, `last`, `status`, `watch`, `logs`, `show`, `artifacts`, `diff`, `approve`, `reject`, `rerun`, `recover`, and `clean`.
- Jobs move to `failed` when orchestration raises.
- Worker subprocesses are bounded by timeouts.
- Task leases recover stale workers.
- Tasks stop being reclaimed after the max-attempt threshold.
- SQLite uses WAL mode and records a schema version.
- `doctor` / `schema_status` run a bounded read-only `PRAGMA quick_check` and warn (never crash) on locked, missing, or corrupt local state.
- Artifacts require evidence, confidence, required payload keys, and content hashes.

## What Is Not Production Yet

- No remote multi-host Redis backend.
- No RBAC or multi-user isolation.
- Claude Code can make real edits when explicitly invoked; hands-off unattended operation is still not recommended.
- No hosted observability stack.
- No signed artifacts or tamper-evident event stream.

## Backing up local SQLite state (opt-in)

Coordination state lives at `{state_dir}/state.sqlite3` (WAL mode). Puppetmaster does **not** auto-export it. To take a consistent local copy — including pending WAL frames — use the opt-in helper (stdlib `sqlite3` backup API):

```python
from pathlib import Path
from puppetmaster.sqlite_store import SQLiteSwarmStore

store = SQLiteSwarmStore(Path(".puppetmaster"))
# Writes under {.puppetmaster}/backups/ by default; refuses overwrite + path escape.
backup_path = store.backup_to("state-backup.sqlite3")
```

Safety rules:

- Destination must resolve under `{state_dir}/backups` (or a `confine_under=` you pass).
- Existing destinations are never overwritten.
- Relative names are resolved inside the confine root.
- This is local-only; there is no Redis/Postgres export path.

`python -m puppetmaster doctor` reports `quick_check=…` on the `sqlite-state` row when a state DB is present.

## Deployment Guidance

Run Puppetmaster locally against trusted repositories. Keep provider keys in the environment, use dry-run Cursor workflows first, and review artifacts before accepting changes.
