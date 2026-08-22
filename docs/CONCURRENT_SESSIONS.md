# Concurrent sessions and dashboards

Puppetmaster is safe to use from several agent conversations and command
shells at once, provided each operation uses an intentional workspace and
state scope. This page is the operator contract for that situation.

## Project scope is the default

Without an explicit state override, runtime state is project-scoped: jobs,
artifacts, promoted memory, streams, and the ordinary dashboard belong to the
workspace selected for the operation. Separate repositories and separate Git
worktrees therefore receive separate default state directories.

For **write-side commands that accept `--cwd`**, the target workspace selects
the default state directory as well as the worker's execution directory:

```bash
python -m puppetmaster swarm "Review the service" --cwd /path/to/project
```

The job above belongs to `/path/to/project`, even when the invoking shell is
elsewhere. Run `python -m puppetmaster state` from that target workspace to
print its default state location.

This does not make separate conversations a security boundary. Processes under
the same user can still share machine resources and explicitly user-global
configuration. Treat project-scoped state as an operational isolation boundary,
not a multi-user tenancy guarantee.

## Explicit state overrides are deliberate sharing

`--state-dir` and `PUPPETMASTER_STATE_DIR` override the project default. Use
one only when sharing a state directory is intentional, such as a controlled
CI location. A shared override shares jobs, artifacts, promoted memory, and a
project-scoped dashboard.

For compatibility, a **relative** explicit `--state-dir` value or relative
`PUPPETMASTER_STATE_DIR` is still resolved from the launcher shell's current
directory, not from `--cwd`. Prefer an absolute override when scripting, or
change into the directory that should anchor a relative override:

```bash
python -m puppetmaster --state-dir C:/ci/puppetmaster-state swarm "Audit" --cwd C:/src/project
```

## Dashboard scope and port selection

`python -m puppetmaster dashboard` shows only the current project's state by
default. It does not discover every job on the machine. Use the aggregate view
only when that is what you intend:

```bash
python -m puppetmaster dashboard --all-projects --background
python -m puppetmaster jobs --all-projects
```

The dashboard's default port search starts at 8787 and automatically advances
when another listener owns a port. Always open the **URL printed by the
command or returned by the MCP tool**; do not reuse a hard-coded `:8787`
bookmark. An explicit `--port` is strict and fails when busy; add
`--port-search` if an explicitly requested starting port may auto-bump.

The background dashboard reuses only a running board with the same state
directory and scope. `--all-projects` is a different scope from the ordinary
project dashboard, so start and inspect it intentionally.

## Concurrent editing requires separate worktrees

Do not run two full-edit or in-place edit jobs against the same checkout at the
same time. The jobs can race after their clean-tree checks and each job can
mistake the other's edits for its own patch.

Use one write-capable job per checkout, or give each concurrent write job its
own Git worktree:

```bash
git worktree add ../project-fix-a -b fix-a
git worktree add ../project-fix-b -b fix-b
python -m puppetmaster claude "Implement fix A" --cwd ../project-fix-a --permission-mode acceptEdits
python -m puppetmaster claude "Implement fix B" --cwd ../project-fix-b --permission-mode acceptEdits
```

Read-only reviews and inspections can run concurrently. A write-capable job
may still share provider quota, model policy, and local CPU/memory with other
sessions; those are user-level resources rather than job-state leakage.

## Quick diagnosis

When a job is not on the dashboard you opened, first compare scopes instead of
assuming the job failed:

```bash
python -m puppetmaster state
python -m puppetmaster projects
python -m puppetmaster jobs --all-projects
python -m puppetmaster dashboard --all-projects --background
```

Use the exact URL printed by the final command. For a known job ID, read-only
commands such as `show`, `status`, `feed`, and `artifacts` can locate the
owning project state automatically unless an explicit state override is in
force.

See [DASHBOARD.md](DASHBOARD.md) for dashboard flags and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for state-directory diagnosis.
