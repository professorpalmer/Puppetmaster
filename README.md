# Puppetmaster

[![CI](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml/badge.svg)](https://github.com/professorpalmer/Puppetmaster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Durable state for parallel coding-agent swarms.** Workers coordinate through typed artifacts in SQLite instead of passing transcripts around — so follow-up questions about a completed swarm cost zero model tokens.

Puppetmaster is a local runtime that lets Cursor, Claude Code, shell checks, and future providers run as independent workers with leases, replay, and structured outputs. It works **standalone** — just `pip install` and you have an orchestrator. Pair it with optional [CodeGraph](https://github.com/colbymchenry/codegraph) integration when you want cheaper per-worker context; the two stack on different axes (CodeGraph optimizes per-call context resolution, Puppetmaster optimizes per-session coordination + state).

Think **Redis/Gunicorn for agentic engineering**:

```text
Cursor Agent / Claude Code / shell / future providers
        |
        v
Puppetmaster supervisor
        |
        v
independent worker processes -> SQLite state -> live artifacts -> stitched summary
```

Puppetmaster is not trying to beat native IDE subagents at every tiny task. It is for the work that gets messy: long repo investigations, conflicting hypotheses, repeated handoffs, flaky memory, and code changes that need evidence, replay, and approval gates.

## The Problem

Most multi-agent coding workflows still use a fragile shape:

```text
One parent chat
  |- subagent
  |- subagent
  `- subagent
```

That works for demos. It breaks down during real repo work.

- The parent context bloats until the important details are buried.
- Subagents inherit stale assumptions from the same conversation.
- Results come back as prose blobs instead of evidence-backed records.
- There is no durable state, replay, lease, failure recovery, or memory promotion.
- A crashed or confused worker often becomes a mystery instead of an inspectable event.
- Full-edit agents can mix old local changes with new changes unless the workflow guards against it.

Puppetmaster is built around a different rule:

> Agents should not share transcript history. They should share durable state.

## What Puppetmaster Solves

### 1. Context Collapse

Workers do not coordinate by stuffing every thought into one parent conversation. They claim tasks, write structured artifacts, and let the stitcher summarize durable outputs back to the operator.

### 2. Subagent Resource Contention

Puppetmaster does not rely on one parent agent spawning children inside the same chat surface. It runs workers as separate local subprocesses, each with its own adapter invocation and lifecycle.

### 3. Vibe-Based Handoffs

Workers emit typed artifacts with payloads, evidence, confidence, source files, and `sha256` integrity. The final synthesis reads artifacts, not raw worker transcripts.

Artifacts are available as soon as they are emitted. The final stitch is the publishable synthesis, not the first moment the work becomes visible.

### 4. Lost Work and Dead Workers

Tasks are lease-based. Stale workers can be recovered. Jobs fail closed. Failures become events and verification artifacts instead of disappearing into chat history.

### 5. Unsafe Code Edits

Claude Code full-edit runs are blocked on dirty worktrees by default. When edits happen, Puppetmaster captures patch artifacts with changed files, base SHA, unified diff, and revert guidance.

### 6. No Long-Term Recall

Useful artifacts can be promoted into memory and retrieved by later workers. The next run does not need the entire old conversation to remember what mattered.

## What It Is

Puppetmaster is not another group-chat swarm. It is a local coordination runtime:

- `Job`: one user goal
- `Task`: role-specific work, optionally dependency-gated
- `Worker`: separate subprocess that claims work through a lease
- `Adapter`: Cursor SDK, Claude Code CLI, shell, or future provider
- `Artifact`: structured finding, decision, patch, verification result, risk, or memory summary
- `Stitcher`: final synthesis from artifacts only
- `Memory`: promoted facts for future retrieval

SQLite is the default coordination backend. WAL mode, schema metadata, integrity checks, task leases, retries, event streams, and patch artifacts are built in.

## Why Not Just Use Subagents?

Native IDE subagents are great for quick parallel help inside one product surface. Puppetmaster solves a different problem: making agent work durable and inspectable outside a single parent context.

| Native subagents | Puppetmaster |
| --- | --- |
| Fast for small tasks | Better for long, stateful investigations |
| Shared chat surface | Shared durable state |
| Transcript-heavy handoffs | Typed artifacts with evidence |
| Harder to replay | Jobs, events, artifacts, and summaries persist locally |
| Usually opaque failure model | Leases, recovery, logs, and failed-task artifacts |
| Final answer often hides process | Live artifact board while workers run |

The goal is not “one more chat.” The goal is a local runtime where the operator can start a swarm, get a `job_id`, watch artifacts appear, inspect partial summaries, and only then approve edits.

## What Works Today

| Area | Status |
| --- | --- |
| Local runtime | Daily-driver beta: subprocess workers, task DAGs, leases, recovery, failure states |
| SQLite backend | Default backend with WAL mode, schema metadata, integrity checks, and persisted events |
| Cursor Agent MCP | Async start tools, status polling, logs, live artifacts, partial summaries |
| Cursor extension | Activity-bar control panel for running Puppetmaster inside Cursor |
| Cursor adapter | Live adapter through `@cursor/sdk`; best for review/plan/dry-run workflows |
| Claude Code adapter | Live full-edit adapter through Claude Code CLI; validated with real tracked diffs |
| Shell adapter | Built-in bounded command runner for verification |
| Memory | Promoted memory retrieval into later worker context and prompts |
| CodeGraph | Optional shared repo intelligence: workers auto-inject CodeGraph context when available |
| Patch workflow | Patch artifacts, path locks, approval/rejection events, dirty-worktree guard |
| Codex | Stubbed provider slot; next adapter target |

## Install

```bash
git clone https://github.com/professorpalmer/Puppetmaster.git
cd Puppetmaster

python -m pip install -e .
npm install --package-lock=false --no-audit
python -m puppetmaster doctor
```

Run the local demo:

```bash
python -m puppetmaster run "Map this repo" --config examples/enterprise-workflow.json
python -m puppetmaster show $(python -m puppetmaster last)
```

Prove worker recovery:

```bash
python -m puppetmaster crash-demo
```

## Daily Driver Prompts

In Cursor Agent, with MCP enabled:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

```text
Use Puppetmaster to start a swarm for this repo and return the job id immediately.
Problem: users are getting logged out after refresh and token refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, and run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

For real multi-role analysis, prefer `puppetmaster_start_cursor_swarm` through Cursor Agent. It creates real Cursor SDK-backed worker roles. Bare custom roles on the generic `puppetmaster_start_swarm` require a config or adapter so Puppetmaster does not accidentally run deterministic demo workers.

While the job runs, ask Cursor Agent to inspect:

```text
Poll Puppetmaster status, live artifacts, and partial summary for <job_id>. Summarize concrete findings as they arrive.
```

After review/approval:

```text
Use Puppetmaster to start Claude Code implementation for the approved fix in a clean worktree. Return the job id immediately and poll status until complete.
```

From the CLI:

```bash
python -m puppetmaster doctor
python -m puppetmaster cursor "Review this repo for release blockers" --review --dry-run
python -m puppetmaster cursor "Plan the next safe implementation slice" --plan --dry-run
python -m puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
python -m puppetmaster show $(python -m puppetmaster last)
python -m puppetmaster logs
```

`cursor` and `claude` use inline orchestration by default to avoid an extra Python worker cold start. The provider still runs in its own process (`node` for Cursor SDK, Claude Code CLI for Claude), while Puppetmaster keeps the same job/task/artifact/lease state model. Use `--worker-mode subprocess` when you want the stricter worker-process boundary for a run.

For local swarms, you can keep Puppetmaster workers warm and let jobs hand off work to them:

```bash
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster run "Review this repo" --worker-mode daemon
```

Daemon mode keeps the Puppetmaster worker loop alive across jobs. It preserves lease-based task claiming and artifacts, while avoiding repeated worker process startup for local-role swarms.

For real edits, prefer a clean worktree:

```bash
git worktree add /tmp/puppetmaster-work -b puppetmaster-work
python -m puppetmaster claude "Implement the approved fix" --cwd /tmp/puppetmaster-work --permission-mode acceptEdits
```

## Works great with CodeGraph (optional)

Puppetmaster runs fine without CodeGraph — workers will fall back to grep/read for context discovery, and the orchestration / durable state / parallel-worker machinery is unchanged. When you pair it with [CodeGraph](https://github.com/colbymchenry/codegraph), every Cursor/Claude worker gets a pre-built repo map (symbols, refs, call graph) injected into its prompt instead of having to rediscover the codebase. The two tools optimize different axes and stack cleanly:

- **CodeGraph** = per-call context resolution. Static facts about your *code* (symbols, refs, routes).
- **Puppetmaster** = per-session coordination + state. Dynamic facts about the *agents' work* (tasks, leases, typed artifacts, replayable events).

Install CodeGraph globally and initialize it once per target repo:

```bash
npm install -g @colbymchenry/codegraph
cd your-target-repo
codegraph init -i
```

After that, Puppetmaster's `doctor` will show `codegraph ok`, and every Cursor/Claude worker run against that workspace will:

- query CodeGraph for task-relevant symbols, files, and routes
- inject the result into the worker prompt as authoritative starting context
- tag the resulting verification artifact with `context:codegraph` so the operator can confirm shared intelligence was used

Fully optional and graceful. If `codegraph` is not installed, or the target repo is not initialized, workers fall back to their normal exploration path with no error. Pass `disable_codegraph: true` in a task payload to skip CodeGraph for a specific worker.

Cursor Agent can also query CodeGraph directly through Puppetmaster's MCP — no second MCP server required for the daily-driver case. See [Bundled CodeGraph tools](#bundled-codegraph-tools-no-second-mcp) below.

### Cost: what changes when you switch to durable state

The whole point of Puppetmaster is that durable state turns repeated questions about the same task into a database read instead of another agent run. The benchmark below shows that effect against two baselines:

- **A. Agent only** — one agent (Cursor or Claude Code) doing the work alone, discovering the repo with grep/read/list. No shared state across sessions.
- **B. CodeGraph alone** — same agent, with CodeGraph's MCP installed; the agent issues `codegraph_explore` calls itself. Still no shared state across sessions.
- **C. Puppetmaster + CodeGraph** — Puppetmaster swarm with CodeGraph context pre-injected into every worker prompt, structured artifacts in a durable SQLite store, stitcher reads JSON not transcripts. Follow-up queries read SQLite, not the model.

Result, modelled from real measurements on this repo (`bench/three_way.py`, swarm of 4 workers, artifact sizes from a real past Puppetmaster run, $3/1M token input price):

#### Fresh task cost (one investigation)

| Config | Tokens | Cost |
|---|---|---|
| A. Agent only | ~30,695 | ~$0.0921 |
| B. CodeGraph alone | ~6,250 | ~$0.0187 |
| C. Puppetmaster + CodeGraph | ~21,231 | ~$0.0637 |

**On a single fresh task, Puppetmaster does not beat CodeGraph alone in raw tokens.** Puppetmaster is doing more work — N parallel workers and a stitcher pass instead of one agent — so its token bill is higher than a single agent with CodeGraph. That's an honest, measured trade-off, and you should know it before believing any "99% reduction" copy.

#### Session cost (1 swarm + K follow-up reads at $3/1M)

This is where Puppetmaster actually wins. Real workflows are not one-shot: you investigate, then ask follow-up questions about the same task. In Configs A and B every follow-up is a fresh agent re-run (no persisted state). In Config C, every follow-up is just SQLite — **0 model tokens.**

| Config | K=0 | K=1 | K=5 | K=10 | K=25 |
|---|---|---|---|---|---|
| A. Agent only | ~$0.0921 | ~$0.1842 | ~$0.5525 | ~$1.0129 | ~$2.3942 |
| B. CodeGraph alone | ~$0.0187 | ~$0.0375 | ~$0.1125 | ~$0.2062 | ~$0.4875 |
| **C. Puppetmaster + CodeGraph** | **~$0.0637** | **~$0.0637** | **~$0.0637** | **~$0.0637** | **~$0.0637** |

At K=25 follow-ups, **Puppetmaster + CodeGraph is ~7.6× cheaper than CodeGraph alone and ~38× cheaper than agent-only.** The crossover where C catches up to B is around K=3-4 in this dataset.

#### Where the savings come from

1. **Durable resume (Puppetmaster)** — the headline. Every follow-up read against a completed swarm is a SQLite query, costing 0 model tokens. This is what flatlines the C column above.
2. **Typed-artifact coordination (Puppetmaster)** — workers communicate through structured rows instead of raw transcripts; the stitcher reads JSON, not stdout.
3. **Amortized context query (CodeGraph + Puppetmaster)** — one `codegraph context` call seeds N workers in a swarm; B issues N separate `codegraph_explore` calls.
4. **Zero tool-call frames (CodeGraph + Puppetmaster)** — workers receive context inline in the initial prompt; no MCP round-trip envelope per worker.

The first two are Puppetmaster's standalone contribution and work even without CodeGraph (you'd just lose the cheap per-call context, so worker prompts get more expensive). The last two only show up when both are installed.

#### Reproduce on your own repo

```bash
npm install -g @colbymchenry/codegraph && codegraph init && codegraph index

# Three-way cost-structure benchmark
python -m bench.three_way --cwd . --workers 4 --artifacts-state /path/to/past/puppetmaster/state

# Just CodeGraph's prompt enrichment (A/B, no API key required)
python -m bench.codegraph_ab --cwd . --prompt @bench/prompts/example.txt --dry-run
```

See [`bench/README.md`](bench/README.md) for full methodology, what's measured vs. modelled, and the honest caveats (no live token billing yet — that's on the roadmap and needs SDK-side stream instrumentation).

## Cursor Integration

Puppetmaster ships with two Cursor integration surfaces.

### Default subagent routing (no more "Utilize Puppetmaster..." prompts)

This repo includes `.cursor/rules/puppetmaster-workflow.mdc` with `alwaysApply: true` and a top-level `AGENTS.md`. Together they tell Cursor Agent (and any agent that reads `AGENTS.md`) to route the following work through Puppetmaster **by default**, without the user having to invoke it explicitly:

- broad investigation, audit, or risk analysis
- multi-file refactors, migrations, cross-cutting cleanups
- debugging that spans call graphs or test coverage
- planning when scope or risks are unclear
- comparing approaches / producing decision artifacts

Native Cursor tooling is still used directly for trivial single-file edits, follow-up questions, and anything the user explicitly framed as "just answer, no swarm."

Copy `.cursor/rules/puppetmaster-workflow.mdc` and `AGENTS.md` into any repo where you want the same default behavior.

### Cursor Agent MCP

The MCP server lets Cursor Agent call Puppetmaster tools directly:

- `puppetmaster_doctor`
- `puppetmaster_start_swarm`
- `puppetmaster_start_cursor_swarm`
- `puppetmaster_start_cursor_review`
- `puppetmaster_start_cursor_plan`
- `puppetmaster_start_claude_implement`
- `puppetmaster_status`
- `puppetmaster_logs`
- `puppetmaster_live_artifacts`
- `puppetmaster_live_artifacts_follow`
- `puppetmaster_partial_summary`
- `puppetmaster_artifacts`
- `puppetmaster_show`
- `puppetmaster_codegraph_search`
- `puppetmaster_codegraph_context`
- `puppetmaster_codegraph_affected`
- `puppetmaster_codegraph_files`
- `puppetmaster_codegraph_status`
- `puppetmaster_codegraph_init`

The older blocking tools are still available for short calls, but the daily-driver path should use `puppetmaster_start_*`. Start tools return a `job_id` immediately, so Cursor does not keep one long MCP call open while workers run.

### Bundled CodeGraph tools (no second MCP)

Puppetmaster's MCP server bundles the most useful CodeGraph CLI commands so Cursor Agent only needs the Puppetmaster MCP to get both orchestration and repo intelligence:

| Tool | Wraps | Use for |
|---|---|---|
| `puppetmaster_codegraph_search` | `codegraph query` | Find symbols by name (`{query, kind?, limit?}`) |
| `puppetmaster_codegraph_context` | `codegraph context` | Pull task-relevant entry points and related symbols (`{task}`) |
| `puppetmaster_codegraph_affected` | `codegraph affected` | Resolve impacted tests from changed source files (`{files[]}`) |
| `puppetmaster_codegraph_files` | `codegraph files` | Inspect the indexed file structure without scanning the FS |
| `puppetmaster_codegraph_status` | `codegraph status` | Check index health and backend |
| `puppetmaster_codegraph_init` | `codegraph init` | Initialize CodeGraph in a workspace (`{index?: true}` to also build immediately) |

Every tool degrades cleanly: if the `codegraph` CLI is not installed or the workspace is not initialized, the response is a non-fatal `isError: true` payload with `error` set to a one-line fix-it hint, not a runtime crash.

Power users who want CodeGraph's full MCP surface (`codegraph_callers`, `codegraph_callees`, `codegraph_impact`, `codegraph_node`) — only available through its own MCP server — can still run `codegraph serve --mcp` alongside Puppetmaster's MCP. Bundling covers the daily-driver case so two MCP entries are no longer required by default.

For real multi-role code analysis from Cursor Agent, use `puppetmaster_start_cursor_swarm`. Bare custom roles on `puppetmaster_start_swarm` require a config or adapter; otherwise Puppetmaster fails fast instead of silently using the deterministic local demo adapter.

Workers emit artifacts as they run. You do not have to wait for the final stitched summary: use `puppetmaster_live_artifacts` for the live evidence board and `puppetmaster_partial_summary` for a current synthesis. For a push-feeling stream, use `puppetmaster_live_artifacts_follow` — it long-polls the durable SQLite event log and returns as soon as a new artifact lands (or after `timeout_seconds`), with a `next_cursor` Cursor Agent can chain to receive the next batch. Final stitching is the publishable report built from the same artifact stream.

CLI users can do the same with `python -m puppetmaster feed <job_id> --follow`, which streams new artifacts as they arrive without re-reading already-seen events.

Blocking tools:

- `puppetmaster_cursor_review`
- `puppetmaster_cursor_plan`
- `puppetmaster_claude_implement`
- `puppetmaster_last_job`

Example `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "puppetmaster": {
      "command": "python",
      "args": ["-m", "puppetmaster.mcp_server"],
      "env": {
        "CLAUDE_CODE_COMMAND": "npx -y @anthropic-ai/claude-code"
      }
    }
  }
}
```

MCP does not patch Cursor's private model picker or force Cursor's native subagents to change their resource model. It gives Cursor Agent a tool surface that invokes Puppetmaster. Once invoked, Puppetmaster owns the run: independent worker processes, SQLite coordination, structured artifacts, and a stitched result returned to Cursor.

See [Cursor Agent MCP](docs/CURSOR_AGENT_MCP.md).

### Cursor Extension

The extension adds a Puppetmaster activity-bar control panel:

- configure provider keys in Cursor secret storage
- run `doctor`
- launch Cursor review/plan dry runs
- launch Claude Code full-edit jobs
- inspect latest job, logs, and artifacts

Download the VSIX from the latest GitHub release or build it locally:

```bash
cd cursor-extension
npm run check
npx -y @vscode/vsce package --no-dependencies
```

Then run `Extensions: Install from VSIX...` in Cursor and choose the generated `.vsix`.

See [Cursor Extension](docs/CURSOR_EXTENSION.md).

## Live Adapters

### Cursor

Use Cursor for review, planning, and dry-run implementation workflows.

```bash
export CURSOR_API_KEY="<your-cursor-api-key>"

python -m puppetmaster cursor "Review this repo and propose the next patch" --review --dry-run
python -m puppetmaster cursor "Plan the next implementation slice" --plan --dry-run
```

The Cursor adapter runs isolated one-shot agents through `@cursor/sdk`.

### Claude Code

Use Claude Code when you want a real coding agent to edit a clean repo or worktree.

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

## CLI Reference

```bash
python -m puppetmaster doctor
python -m puppetmaster adapters
python -m puppetmaster init-config --path puppetmaster.json
python -m puppetmaster run "Goal" --config examples/enterprise-workflow.json
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster cursor "Goal" --review --dry-run
python -m puppetmaster claude "Goal" --permission-mode acceptEdits
python -m puppetmaster crash-demo
python -m puppetmaster status <job_id>
python -m puppetmaster watch <job_id>
python -m puppetmaster events <job_id>
python -m puppetmaster feed [job_id]
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

Examples:

- [Enterprise Workflow](examples/enterprise-workflow.json)
- [Cursor Live](examples/cursor-live.json)
- [Cursor Review](examples/cursor-review.json)
- [Cursor Dry-Run Implementation](examples/cursor-dry-run-implementation.json)
- [Claude Code Full Edit](examples/claude-code-full-edit.json)
- [Memory Reuse](examples/memory-reuse.json)

## State Model

By default, Puppetmaster keeps runtime state outside the repository so `git status` stays focused on source changes:

```text
macOS: ~/Library/Application Support/puppetmaster/projects/<workspace>-<hash>/
Linux: ~/.local/state/puppetmaster/projects/<workspace>-<hash>/
```

Print the resolved location:

```bash
python -m puppetmaster state
```

Override it when you intentionally want repo-local or CI-specific state:

```bash
python -m puppetmaster --state-dir .puppetmaster run "Map this repo"
PUPPETMASTER_STATE_DIR=.puppetmaster python -m puppetmaster doctor
```

The state directory contains:

```text
<state-dir>/
  state.sqlite3
  jobs/
  memory/
  streams/
  locks/
```

`.puppetmaster/` remains in `.gitignore` as a compatibility fallback for explicit local state.

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
- [Cursor Agent MCP](docs/CURSOR_AGENT_MCP.md)
- [Cursor Extension](docs/CURSOR_EXTENSION.md)
- [Daily Driver](docs/DAILY_DRIVER.md)
- [Production Notes](docs/PRODUCTION.md)
- [Security](docs/SECURITY.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Release Checklist](docs/RELEASE_CHECKLIST.md)
- [Changelog](docs/CHANGELOG.md)
- [Roadmap](docs/ROADMAP.md)

## Status

Puppetmaster is **daily-driver beta software**. The runtime contract is real, tests are automated, SQLite is the default backend, jobs fail closed, Cursor Agent MCP is live, the Cursor extension is installable, and Claude Code has been validated as a full-edit adapter that emits patch artifacts.

It is credible for supervised local engineering workflows. It is not yet a hosted multi-user production service.

## License

MIT

