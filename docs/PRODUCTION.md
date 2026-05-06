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
- Artifacts require evidence, confidence, required payload keys, and content hashes.

## What Is Not Production Yet

- No remote multi-host Redis backend.
- No RBAC or multi-user isolation.
- Claude Code can make real edits when explicitly invoked; hands-off unattended operation is still not recommended.
- No hosted observability stack.
- No signed artifacts or tamper-evident event stream.

## Deployment Guidance

Run Puppetmaster locally against trusted repositories. Keep provider keys in the environment, use dry-run Cursor workflows first, and review artifacts before accepting changes.
