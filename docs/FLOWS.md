# Flow graphs

A flow is a graph the pilot writes **once**. Puppetmaster walks it in the
background and wakes the pilot only when the run is done, failed, stuck,
interrupted, or waiting at a gate. The pilot spends no model turns launching
workers, polling, or handing work between them.

Every agent and judge node runs as an ordinary durable Puppetmaster job, so it
keeps leases, file claims, write-scope gates, receipts and crash recovery. The
runtime adds three things a subagent tree does not have:

- **Exactly-once nodes.** Each node visit is recorded before it starts and
  launched with a deterministic key. A walker that dies mid-node adopts the job
  it started when it resumes; it never buys the same work twice. (Shell nodes
  are the exception: they are at-least-once, see below.)
- **Session continuity.** A node that runs again resumes its own provider
  session and receives only the delta: the judge's feedback, or a follow-up
  request. This applies when a back-edge sends work back to it, and to every
  node of a run started with `--continue`. In a continued run, a node whose
  task text changed also gets the updated task. A continued run with no
  input sends each node its current task again.
- **`map` fan-out.** One node expands a list into per-item child flows. Each
  walks in its own process with bounded concurrency. A repair visit re-runs
  only the items the feedback names plus the items that failed. The pilot's
  context is the same size at 4 items or 1,000.

## A first flow

```json
{
  "id": "fix-and-review",
  "entry": "build",
  "cwd": ".",
  "defaults": {"adapter": "codex", "model": "gpt-5.6-luna"},
  "nodes": [
    {"id": "build", "kind": "agent", "role": "code", "files": ["pkg/stats.py"],
     "task": "Implement pkg/stats.py so tests/test_stats.py passes."},
    {"id": "check", "kind": "shell", "command": "python -m pytest -q tests/test_stats.py"},
    {"id": "review", "kind": "judge", "task": "Review pkg/stats.py for edge cases and clarity."},
    {"id": "done", "kind": "end", "summary": "stats accepted"}
  ],
  "edges": [
    {"from": "build", "to": "check"},
    {"from": "check", "to": "review", "when": "ok"},
    {"from": "check", "to": "build", "when": "fail"},
    {"from": "review", "to": "done", "when": "PASS"},
    {"from": "review", "to": "build", "when": "FAIL", "max": 2}
  ]
}
```

```bash
puppetmaster flow run fix-and-review.json --input "stats module"
puppetmaster flow wait <run_id>
```

When `check` fails, `build` resumes its own session with the test output as
feedback. When the judge says FAIL, `build` resumes again with the judge's
`file:line` findings. The judge resumes too, and is asked to confirm its earlier
problems are fixed.

## Nodes

| kind | what it does | ok / verdict |
| --- | --- | --- |
| `agent` | A worker job. `role: "code"` (default) edits within `files`; `role: "explore"` is read-only. Optional `adapter`, `model`, `timeout_seconds`, `payload`, `retries`, `saveAs`, `revise`, `resume`. | ok when the task completes; verdict from its final `VERDICT:` line |
| `judge` | A read-only reviewer. It lists each problem as `path:line - what`, then ends with `VERDICT: PASS|FAIL|PARTIAL - reason`. A judge with no verdict makes the run `stuck`, never a quiet pass. | verdict routes the edges |
| `parallel` | `branches: [ids]` of agent/judge nodes run as one job. Code branches need disjoint `files`. On a later visit only the failing or named branches re-run. | ok if all ok; verdict FAIL > PARTIAL > PASS |
| `map` | `items` (a list, or a template such as `{{state.regions}}`) × a `node` template or an item `graph`. `key` (template) names items; `concurrency` (default 8); `pass` is `"all"` or a fraction such as `0.95`; `feedback` (template, e.g. `{{out.consistency}}`) chooses which output targets a repair visit. | ok when the pass policy is met |
| `shell` | `command`, optional `cwd` and `timeoutMs`. Reads run data from the JSON file at `$PM_FLOW_CONTEXT`. A final `VERDICT:` line in its output routes like a judge. At-least-once: keep commands idempotent. | ok on exit 0 |
| `gate` | Pauses with `question` and `options`; `flow resume <run> --answer=X` continues. Not allowed inside map items. | routes on `answer=X` |
| `set` | `values` written into state (templated). | ok |
| `end` | `status: pass|fail`, `summary` (templated). | ends the run |

### Effort

`effort` on an agent or judge node (`low`, `medium`, `high`, `xhigh`) sets its
reasoning effort; adapters translate it (Codex `model_reasoning_effort`,
Claude Code `--effort`). `defaults.effort` covers every node, and
`defaults.lanes` maps roles to an effort, for example
`{"explore": "low", "code": "medium", "judge": "high"}`; judges use the `judge`
lane. A node's own `effort` wins over its lane, and a lane over
`defaults.effort`. `escalate: true` (on a node or in `defaults`) raises a node's effort one step on each
repair visit, so a cheap first build gets more reasoning only when a check or
judge sends it back.

A node with no effort (from itself, its lane or `defaults`) runs the worker
default: `medium`, or the operator's `PUPPETMASTER_WORKER_EFFORT` (`low`,
`medium`, `high`, `xhigh`) when the MCP server or CLI that creates the tasks
has it set. Any effort the graph names still wins over that default. With
`PUPPETMASTER_WORKER_EFFORT_POLICY=enforce`, every worker runs the operator
effort: a graph naming a different effort, a lane or `escalate` fails
validation, and any other conflicting pin is refused before launch. Each task
records `reasoning_effort` (effective), `requested_reasoning_effort` (the
caller's pin or null) and `reasoning_effort_source` (`caller`,
`operator_default`, `operator_enforced` or `swarm_default`). The setting
applies to every worker task, not only flows: agents, judges, and resumed
sessions alike. An invalid value fails task creation with the reason.

## Edges

`{"from", "to", "when", "max"}`. The first matching edge in declaration order
wins. `when` is one of:

- `always` (the default), `ok`, `fail`
- `PASS`, `FAIL`, `PARTIAL`: the node's verdict
- `answer=X`: a gate's answer (case- and Unicode-insensitive)
- `out~=text`: the node's output contains the text
- `state.k OP v` with `= != >= <= > < ~=`; numbers compare numerically

A back-edge (an edge that closes a cycle) may be taken at most `max` times,
default `limits.maxLoops` (3). Past that the run is `stuck`. Forward edges are
bounded only by `limits.maxSteps` (60). A failed node with no matching edge
fails the run. A node with edges but no match makes the run `stuck`. A node
with no edges ends the run.

## Templates

`{{input}}`, `{{prev}}` (the previous node's output), `{{answer}}`,
`{{out.<node>}}`, `{{verdict.<node>}}`, `{{reason.<node>}}`,
`{{files.<node>}}`, `{{state.<path>}}`, and inside a map item `{{item}}`,
`{{item.<path>}}`, `{{index}}`, `{{key}}`. A field that is exactly one
placeholder keeps its type, so `"files": "{{item.files}}"` stays a list.

Shell commands only accept substituted values made of letters, digits and
`_ . / : = @ + , -`. Anything else fails the node instead of running, because a
model's output must never become a command. Read arbitrary data from
`$PM_FLOW_CONTEXT` instead.

## Mass fan-out with `map`

```json
{"id": "regions", "kind": "map", "items": "{{state.regions}}", "concurrency": 16,
 "graph": {
   "entry": "build",
   "nodes": [
     {"id": "build", "kind": "agent", "files": ["regions/{{item.id}}.py", "renders/{{item.id}}/*"],
      "task": "Build region {{item.id}}: {{item.brief}}"},
     {"id": "check", "kind": "shell", "command": "python judge.py --region {{item.id}}"},
     {"id": "review", "kind": "judge", "task": "Review regions/{{item.id}}.py for craft."},
     {"id": "ok", "kind": "end"}],
   "edges": [
     {"from": "build", "to": "check"},
     {"from": "check", "to": "review", "when": "ok"},
     {"from": "check", "to": "build", "when": "fail", "max": 2},
     {"from": "review", "to": "ok", "when": "PASS"},
     {"from": "review", "to": "build", "when": "FAIL", "max": 2}]}}
```

Each item's judge starts the moment that item's build passes its check; there
is no barrier waiting for the slowest item. An item passes only when its flow
is done and the latest result of every work node in it is ok: a build that
failed a write-scope gate does not pass just because an unconditional edge
carried the flow to its end. Prefer `"when": "ok"` on edges out of a build. Follow `map` with an assembled-result
check and judge, and route the judge's FAIL back to `map`. On that visit only
the items whose keys or files the judge names, plus any item that failed, run
again. Each continues its own child flow, so its sessions resume with the
judge's feedback. Item keys that are plain numbers never match by name, since
`x.py:12` would otherwise select item `12`.

Each re-run item receives only the feedback lines that mention its key or its
files (all of the feedback when none do), so its repair delta stays small and
inside its own files.

A unit's `files` are everything it writes, including what its own check
generates: here `renders/{{item.id}}/*` for the renders its check writes. The
write-scope gate compares the tree before and after each worker, and in a
shared checkout that window also holds its siblings' writes. A path some
item declared belongs to that item's gate; an undeclared output is charged
to whichever worker's gate sees it. Keep per-unit output directories
disjoint, and do not give every unit one broad shared output glob: that
reintroduces contention. Gitignored outputs are never judged.

Shared files no item owns (a package `__init__.py`, a registry, a manifest)
need an owner too, and so do shared assembly outputs (an integrated render,
a combined scene file). Route the assembled judge's FAIL through an `integrate`
agent whose `files` are those shared paths, then back to `map`. When feedback
names a file no item owns, the map's wake says so instead of looping on it.

The pilot is also a writer in the checkout. A new file that the pilot writes
during a map (a preview, a scratch scene) appears in the delta of each
worker that runs at that time. The shared tree cannot show who wrote it, so
the gate charges it to each of those workers. Before you write such a file,
claim it:

```bash
puppetmaster flow claim <run_id> preview_world.py
```

The MCP tool uses `action: "claim"` with `run_id` and `paths`. A claim holds
for the full run, including its map items. A worker gate then lists the path
under `pilot_claimed` and does not fail. The worker's own events must not
name the path. If they name it, the worker wrote the file, and the gate
still fails. A claim covers only the paths and globs that it names. Another
new file that nobody claimed or declared still fails the gate.

The Puppetmaster hooks make these claims for you. After each file edit by
the pilot, the claim hook (`python -m puppetmaster.claim_hook`) claims the
file in each flow that walks over that checkout:

| Host | Hook event | Tools |
| --- | --- | --- |
| Claude Code | `PostToolUse` | `Write`, `Edit`, `MultiEdit`, `NotebookEdit` |
| Codex | `PostToolUse` | `apply_patch` |
| Cursor | `afterFileEdit` | file edits |

To install the hooks, run `puppetmaster install-hooks` (add `--global` for
Codex). Codex runs a new hook only after you review it once. The hook does
nothing when no flow walks, and it does nothing in a Puppetmaster worker. It
takes about 40 ms per edit and makes no model call. An edit hook cannot see
a file that a shell command writes. Claim such a file yourself.

## Reviews bind to the final source

A judge's PASS is about the files that it read. When the code nodes of a run
declare `files`, each judge of that run does three things:

1. Before it starts, it waits until no live edit claim of another worker
   overlaps those files. The wait ends after `defaults.review_wait_seconds`
   (default 900) or on a stop. If a writer still holds a claim at that time,
   a PASS becomes PARTIAL.
2. It records `reviewed` in its result: a digest of the files, their count,
   and (up to 200 files) a hash for each file. If the files changed during
   the review, a PASS becomes PARTIAL, and the reason names the changed paths.
3. When the run reaches a passing end, each PASS digest is checked again. If
   a reviewed file changed after the PASS, the run ends `failed` with a
   "stale review" reason that names the paths.

A judge in a map item reviews only that item's files, so the builds of the
other items do not make it wait. A run whose code nodes declare no `files`
has no binding.

## Follow-ups

```bash
puppetmaster flow run fix-and-review.json --continue <prior_run_id> --input "also handle empty input"
```

The new run inherits the prior run's sessions and results. Every node resumes
its own session with the follow-up request instead of starting fresh, and a map
re-runs only the items the request names, or all items, each resumed.

## Running and waking

| CLI | MCP `puppetmaster_flow` action |
| --- | --- |
| `flow validate <graph>` | `validate` |
| `flow save <graph>` (reuse by id) | `save` |
| `flow run <graph\|id> [--input] [--continue RUN] [--wait] [--foreground]` | `run` (always detached; `wait: true` blocks up to the MCP cap) |
| `flow status <run> [--since N]` | `status` |
| `flow wait <run> [--timeout S]` | `wait` |
| `flow resume <run> [--answer=X] [--restart] [--reset-loops] [--extra-steps N]` | `resume` |
| `flow stop <run>` | `stop` |
| `flow cut <run> [--reason]` | `cut` |
| `flow list` | `list` |

Every action returns the same compact summary: status, reason, the steps since
`since`, the last problem, usage, and `next_since` for the next call. Exit
codes: 0 done, 4 waiting at a gate, 3 still running, 1 failed, stuck, stopped
or interrupted, 2 invalid input.

A walker that dies leaves the run `interrupted`; `flow resume` picks it up, and
any job it had started is adopted rather than relaunched. A dead walker's pid
that the OS has since given to another process does not keep the run looking
alive: run locks, spawn markers and shell records store the process identity
next to the pid.

`stop` halts the run and cuts its in-flight tasks and item flows. A node that
finishes in the instant a stop lands is not recorded, so `flow resume --restart`
runs it again as a new attempt, which is a new job. `cut` fails only the node in
flight, so its `fail` edges take over; a cut map also stops its item flows.

Stop and cut are cooperative. A worker stops at its next check, and the run can
show `stopped` before its workers end. The reply of `stop` and `cut`, and the
summary of each stopped run, has these fields:

| Field | Meaning |
| --- | --- |
| `settled` | True when no task or item flow of the in-flight node can still write. |
| `open_work` | Each child run that is not terminal, and each task that is queued, running, or blocked. |
| `note` | Present while work is open: call `wait` before you treat the workers as stopped. |

`wait` on a stopped run returns when its open work settles, or at the timeout.
While it waits, it cuts the open tasks again (a cut is idempotent). A worker
that queues for edit admission sees the cut and exits without a launch.

Shell nodes are at-least-once. A command's exit status dies with its walker, so
a resumed walker kills the command the dead walker left running (when the pid
still names that command) and runs it again. A pid that now names another
process is left alone.

Runs live under `<state_dir>/flows/runs/<run_id>/`. Pass `--cwd` (CLI) or
`cwd` (MCP) so the run's state follows the workspace.
