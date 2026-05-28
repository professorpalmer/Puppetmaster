# Why Puppetmaster

The design rationale behind the runtime: what fails in the common
"one parent chat + subagents" shape, and what Puppetmaster does
differently.

## The problem with shared-transcript subagents

Most multi-agent coding workflows still use the fragile shape:

```text
One parent chat
  |- subagent
  |- subagent
  `- subagent
```

That works for demos. It breaks down during real repo work:

- The parent context bloats until the important details are buried.
- Subagents inherit stale assumptions from the same conversation.
- Results come back as prose blobs instead of evidence-backed records.
- There is no durable state, replay, lease, failure recovery, or memory promotion.
- A crashed or confused worker becomes a mystery instead of an inspectable event.
- Full-edit agents can mix old local changes with new changes unless the workflow guards against it.

Puppetmaster is built around the opposite rule:

> Agents should not share transcript history. They should share durable state.

## What Puppetmaster fixes

### 1. Context collapse

Workers do not coordinate by stuffing every thought into one parent
conversation. They claim tasks, write structured artifacts, and let
the stitcher summarize durable outputs back to the operator.

### 2. Subagent resource contention

Puppetmaster does not rely on one parent agent spawning children
inside the same chat surface. It runs workers as separate local
subprocesses, each with its own adapter invocation and lifecycle.

### 3. Vibe-based handoffs

Workers emit typed artifacts with payloads, evidence, confidence,
source files, and `sha256` integrity. The final synthesis reads
artifacts, not raw worker transcripts. Artifacts are available as
soon as they are emitted; the final stitch is the publishable
synthesis, not the first moment the work becomes visible.

### 4. Lost work and dead workers

Tasks are lease-based. Stale workers can be recovered. Jobs fail
closed. Failures become events and verification artifacts instead of
disappearing into chat history.

### 5. Unsafe code edits

Claude Code full-edit runs are blocked on dirty worktrees by default.
When edits happen, Puppetmaster captures patch artifacts with changed
files, base SHA, unified diff, and revert guidance.

### 6. No long-term recall

Useful artifacts can be promoted into memory and retrieved by later
workers. The next run does not need the entire old conversation to
remember what mattered.

## What Puppetmaster is

A local coordination runtime:

- `Job`: one user goal
- `Task`: role-specific work, optionally dependency-gated
- `Worker`: separate subprocess that claims work through a lease
- `Adapter`: Cursor SDK, Claude Code CLI, OpenAI Chat Completions, Codex CLI, shell
- `Artifact`: structured finding, decision, patch, verification result, risk, or memory summary
- `Stitcher`: final synthesis from artifacts only
- `Memory`: promoted facts for future retrieval

SQLite is the default coordination backend. WAL mode, schema metadata,
integrity checks, task leases, retries, event streams, and patch
artifacts are built in.

## Why not just use native IDE subagents?

Native IDE subagents are great for quick parallel help inside one
product surface. Puppetmaster solves a different problem: making
agent work durable and inspectable outside a single parent context.

| Native subagents | Puppetmaster |
| --- | --- |
| Fast for small tasks | Better for long, stateful investigations |
| Shared chat surface | Shared durable state |
| Transcript-heavy handoffs | Typed artifacts with evidence |
| Harder to replay | Jobs, events, artifacts, and summaries persist locally |
| Usually opaque failure model | Leases, recovery, logs, and failed-task artifacts |
| Final answer often hides process | Live artifact board while workers run |

The goal is not "one more chat." The goal is a local runtime where
the operator can start a swarm, get a `job_id`, watch artifacts
appear, inspect partial summaries, and only then approve edits.

Puppetmaster is not trying to beat native IDE subagents at every tiny
task. It is for the work that gets messy: long repo investigations,
conflicting hypotheses, repeated handoffs, flaky memory, and code
changes that need evidence, replay, and approval gates.
