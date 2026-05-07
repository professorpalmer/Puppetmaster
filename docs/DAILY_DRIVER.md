# Daily Driver

Puppetmaster is meant to make repeated Cursor swarm work inspectable and replayable.

## Recommended Loop

```bash
python -m puppetmaster doctor
python -m puppetmaster cursor "Review the current repo and propose the next patch" --review --dry-run
python -m puppetmaster cursor "Plan the next implementation slice" --plan --dry-run
python -m puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
python -m puppetmaster logs
python -m puppetmaster show $(python -m puppetmaster last)
python -m puppetmaster rerun
```

Use `--dry-run` for normal daily planning and review tasks. The Cursor adapter receives promoted memory from earlier runs, but it is instructed to verify retrieved claims before relying on them.

Use `puppetmaster claude` when you intentionally want Claude Code to make real edits. It defaults to Claude Code `acceptEdits` permission mode and records tracked diffs as Puppetmaster patch artifacts.

`puppetmaster cursor` and `puppetmaster claude` default to `--worker-mode inline` for daily-driver speed. That skips an extra Puppetmaster Python worker subprocess while preserving job state, task leases, artifacts, stitching, and the provider's own process boundary. Pass `--worker-mode subprocess` when strict worker process isolation matters more than latency.

## Operator Gates

Puppetmaster records patch artifacts and approval events, but it does not auto-apply repository changes yet.

```bash
python -m puppetmaster diff
python -m puppetmaster approve <job_id>
python -m puppetmaster reject <job_id> --reason "Needs narrower scope"
```

This keeps the daily-driver workflow serious without handing write access to every worker by default.

## Run Management

```bash
python -m puppetmaster jobs
python -m puppetmaster last
python -m puppetmaster status <job_id>
python -m puppetmaster logs [job_id]
python -m puppetmaster open [job_id]
python -m puppetmaster clean --completed
```

SQLite is the default backend. Use `--backend file` only when you want fully inspectable JSON state for debugging.
