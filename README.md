# Puppetmaster

[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Provider-neutral swarms for coding agents.**

Puppetmaster treats AI coding agents like distributed workers: independent processes, shared coordination state, structured artifacts, replayable memory, leases, recovery, and human approval gates.

Think **Redis/Gunicorn for agentic engineering work**.

```text
Cursor / Claude Code / Shell / Codex
        |
        v
independent workers -> SQLite coordination -> structured artifacts -> stitched memory
```

## Why This Exists

Most agent swarms still behave like group chats:

```text
One parent context
  |- child agent
  |- child agent
  `- child agent
```

That shape collapses under real work. Context bloats. Workers inherit stale assumptions. Results return as prose blobs. The final synthesis has to trust vibes.

Puppetmaster uses a systems shape instead:

- Workers do not share full transcript context.
- Workers claim tasks through leases.
- Workers emit typed JSON artifacts with evidence and confidence.
- A stitcher reads artifacts, not raw chats.
- Useful facts become promoted memory for later runs.
- Patch proposals and full-edit diffs are reviewable artifacts.

The rule:

> Agents should not share transcript history. They should share durable state.

## What Works Today

| Area | Status |
| --- | --- |
| Local runtime | Daily-driver beta: subprocess workers, task DAGs, leases, recovery, failure states |
| SQLite backend | Default backend with WAL mode, schema metadata, integrity checks, and persisted events |
| Cursor adapter | Live adapter through `@cursor/sdk`; best for review/plan/dry-run workflows |
| Claude Code adapter | Live full-edit adapter through Claude Code CLI; validated with real tracked diffs |
| Shell adapter | Built-in bounded command runner for verification |
| Memory | Promoted memory retrieval into later worker context and prompts |
| Patch workflow | Patch artifacts, path locks, approval/rejection events, dirty-worktree guard |
| Codex | Stubbed provider slot; next adapter target |

## Quickstart

```bash
git clone https://github.com/professorpalmer/Puppetmaster.git
cd Puppetmaster

python -m unittest discover -s tests -v
python -m puppetmaster doctor
python -m puppetmaster run "Map this repo" --config examples/enterprise-workflow.json
```

Inspect the run:

```bash
python -m puppetmaster last
python -m puppetmaster watch $(python -m puppetmaster last) --ticks 1
python -m puppetmaster show $(python -m puppetmaster last)
python -m puppetmaster logs
```

Prove failure recovery:

```bash
python -m puppetmaster crash-demo
```

## Live Adapters

### Cursor

Use Cursor for review, planning, and dry-run implementation workflows.

```bash
npm install
export CURSOR_API_KEY="<your-cursor-api-key>"

python -m puppetmaster cursor "Review this repo and propose the next patch" --review --dry-run
python -m puppetmaster cursor "Plan the next implementation slice" --plan --dry-run
```

The Cursor adapter runs isolated one-shot agents through `@cursor/sdk`.

### Claude Code

Use Claude Code when you want a real terminal coding agent to edit a clean repo or worktree.

```bash
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
export CLAUDE_CODE_COMMAND="npx -y @anthropic-ai/claude-code"

python -m puppetmaster claude \
  "Implement the approved change and run focused tests" \
  --permission-mode acceptEdits
```

Claude Code full-edit runs require a clean working tree by default. Puppetmaster blocks dirty repos unless you explicitly pass `--allow-dirty`, because otherwise patch artifacts could mix old local changes with agent changes.

When Claude Code edits tracked files, Puppetmaster records:

- a `verification` artifact with stdout/stderr, return code, model usage, and failure classification
- a `patch` artifact with changed files, base SHA, unified diff, and revert guidance

### Shell

Use `shell` for bounded verification steps:

```json
{
  "role": "verify-runtime",
  "instruction": "Verify Python is available.",
  "adapter": "shell",
  "payload": {
    "command": ["python", "--version"],
    "timeout_seconds": 10
  }
}
```

## Full Claude Code Smoke Test

This is the proof that Puppetmaster can orchestrate a full-edit coding agent and capture the resulting diff:

```bash
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
export CLAUDE_CODE_COMMAND="npx -y @anthropic-ai/claude-code"

tmp_root=$(mktemp -d)
tmp_repo="$tmp_root/repo"
tmp_state="$tmp_root/.puppetmaster"

mkdir "$tmp_repo"
git init "$tmp_repo"
printf 'before\n' > "$tmp_repo/hello.txt"
git -C "$tmp_repo" add hello.txt
git -C "$tmp_repo" commit -m init

python -m puppetmaster \
  --state-dir "$tmp_state" \
  claude "Change hello.txt so its entire contents are exactly: after. Do not modify any other file." \
  --cwd "$tmp_repo" \
  --permission-mode acceptEdits \
  --timeout-seconds 300

job_id=$(python -m puppetmaster --state-dir "$tmp_state" last)
python -m puppetmaster --state-dir "$tmp_state" artifacts "$job_id"
git -C "$tmp_repo" diff -- hello.txt
```

Expected result:

```diff
-before
+after
```

## Daily Driver Loop

```bash
python -m puppetmaster doctor
python -m puppetmaster adapters

python -m puppetmaster cursor "Review this repo" --review --dry-run
python -m puppetmaster claude "Implement the approved change" --permission-mode acceptEdits

python -m puppetmaster last
python -m puppetmaster show $(python -m puppetmaster last)
python -m puppetmaster logs
python -m puppetmaster diff
python -m puppetmaster approve <job_id>
```

For real repo edits, prefer a worktree:

```bash
git worktree add /tmp/puppetmaster-claude-test -b puppetmaster-claude-test
python -m puppetmaster claude "Implement X" --cwd /tmp/puppetmaster-claude-test --permission-mode acceptEdits
```

## CLI Reference

```bash
python -m puppetmaster doctor
python -m puppetmaster adapters
python -m puppetmaster init-config --path puppetmaster.json
python -m puppetmaster run "Goal" --config examples/enterprise-workflow.json
python -m puppetmaster cursor "Goal" --review --dry-run
python -m puppetmaster claude "Goal" --permission-mode acceptEdits
python -m puppetmaster crash-demo
python -m puppetmaster status <job_id>
python -m puppetmaster watch <job_id>
python -m puppetmaster events <job_id>
python -m puppetmaster artifacts <job_id>
python -m puppetmaster logs [job_id]
python -m puppetmaster open [job_id]
python -m puppetmaster last
python -m puppetmaster rerun [job_id]
python -m puppetmaster diff [job_id]
python -m puppetmaster approve <job_id-or-artifact-id>
python -m puppetmaster reject <job_id-or-artifact-id> --reason "why"
python -m puppetmaster clean --completed
python -m puppetmaster memory
```

SQLite is the default backend. Use `--backend file` when you want maximally inspectable JSON state.

## Workflow Config

```json
{
  "lease_seconds": 10,
  "workers": [
    {
      "role": "explore",
      "instruction": "Map the goal and emit evidenced findings."
    },
    {
      "role": "claude-implement",
      "instruction": "Use Claude Code to implement the requested change.",
      "adapter": "claude-code",
      "depends_on": ["explore"],
      "payload": {
        "prompt": "Implement the change and run focused tests.",
        "cwd": ".",
        "permission_mode": "acceptEdits",
        "allowed_tools": ["Read", "Edit", "MultiEdit", "Write", "Bash"],
        "timeout_seconds": 900,
        "allow_dirty": false
      }
    }
  ]
}
```

More examples:

- [Enterprise Workflow](examples/enterprise-workflow.json)
- [Cursor Live](examples/cursor-live.json)
- [Cursor Review](examples/cursor-review.json)
- [Cursor Dry-Run Implementation](examples/cursor-dry-run-implementation.json)
- [Claude Code Full Edit](examples/claude-code-full-edit.json)
- [Memory Reuse](examples/memory-reuse.json)

## State Model

Puppetmaster writes local state under `.puppetmaster/`:

```text
.puppetmaster/
  state.sqlite3
  jobs/
  memory/
  streams/
  locks/
```

Core objects:

- `Job`: one swarm run and user goal
- `Task`: role-specific work, optionally dependency-gated
- `AgentRun`: one worker attempt
- `Artifact`: structured output with payload, evidence, confidence, and `sha256`
- `MemoryRecord`: promoted fact retrieved by later workers

## Safety Model

Puppetmaster is powerful because it can orchestrate tools that edit code. The safety model is explicit:

- Cursor defaults toward review/plan/dry-run workflows.
- Claude Code is full-edit, but blocked on dirty worktrees by default.
- Patch outputs are artifacts with diffs and base SHAs.
- Approval/rejection is recorded in the event stream.
- Stale workers are recovered through leases.
- Failed provider calls become structured artifacts instead of mystery crashes.
- Secrets stay in environment variables, never config files.

If you paste a key into a terminal, chat, issue, screenshot, or transcript, rotate it before publishing.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Adapters](docs/ADAPTERS.md)
- [Daily Driver](docs/DAILY_DRIVER.md)
- [Production Notes](docs/PRODUCTION.md)
- [Security](docs/SECURITY.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Release Checklist](docs/RELEASE_CHECKLIST.md)
- [Changelog](docs/CHANGELOG.md)
- [Roadmap](docs/ROADMAP.md)

## Status

Puppetmaster is **daily-driver beta software**. The runtime contract is real, tests are automated, SQLite is the default backend, jobs fail closed, Cursor is live, and Claude Code has been validated as a full-edit adapter that emits patch artifacts.

It is credible for supervised local engineering workflows. It is not yet a hosted multi-user production service.

## License

MIT

