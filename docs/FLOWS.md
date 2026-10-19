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

### Finish

A build worker (Codex, Claude Code) is told when it is done. The default,
`finish: "improve"`, says passing checks are the floor: when the task asks for
quality the checks do not measure and time remains, the worker improves the
work and runs the checks again. `finish: "criteria"` (on an agent node or in
`defaults`) replaces that one sentence: the work is done when every acceptance
criterion is met and every named check passes, and the worker then stops. Put
the criteria in the task as an `Acceptance criteria:` block of numbered lines.
Judges and explore agents ignore `finish`.

### What builders and judges see

A flow adds run context to worker tasks. Each item below is on by default.

- **Gate preview.** A build agent sees the shell commands and judge tasks
  that come after it on its success edges, in order, up to the next agent,
  map or end. It runs those checks itself and checks its work against the
  review before it reports. Set `preview: false` on the agent node to turn
  this off.
- **Review delta.** A judge on a later visit sees which of its reviewed files
  changed since its last review and which did not. Its earlier findings on
  unchanged files stand.
- **Unit evidence.** A judge after a map sees the item reviews of that map:
  each item judge's PASS, the files it covered, and if those files changed
  since. The judge spends its review on how the units fit together. It
  reviews a unit again when that unit's review is not current. Set
  `evidence: false` on the judge node to turn this off.
- **Item frame.** A build agent inside a map item learns its item key, that
  the map waits for every item before later steps, and the files that the
  other items own. A check or review line that fails only in another item's
  files is not its failure: it names that line in its report and gives the
  verdict of its own work. On a follow-up it learns that the request is for
  the whole map. The frame is always on. The item's own checks and review,
  and the steps after the map, still judge the result.

In a local replica of the Codex R138 graph (16 regions in 4 groups,
`gpt-6.1-sol` high), first-visit review failures went from 3 of 4 groups
(1.43.4, one run) to 0, 1, 2 and 0 of 4 (four runs). The world review went
from 165 s to 52 to 87 s.

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
In `files`, a list placeholder inside a list is flattened:
`["{{item.files}}", "reports/{{item.id}}/*"]` gives the item's paths plus the
report glob.

Shell commands only accept substituted values made of letters, digits and
`_ . / : = @ + , -`. Anything else fails the run at once, because a model's
output must never become a command, and a repair edge cannot change a
template. A map checks its items' shell commands before it starts any worker.
Read a list or other data from the JSON file at `$PM_FLOW_CONTEXT` instead:
`input`, `prev`, `outputs`, `results` and `state`. Inside a map item, the item
is at `["state"]["item"]`, for example
`json.load(open(os.environ["PM_FLOW_CONTEXT"]))["state"]["item"]["regions"]`.

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

Task scopes, sibling scopes and pilot claims use one match rule. A glob
matches as Python `fnmatch` does, so `*` can also match `/`. A `**/` segment
also matches zero directories: `renders/r1/**/*` covers `renders/r1/a.json`
and `renders/r1/x/a.json`, but not `renders/r2/a.json`. A plain path matches
itself.

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

### Builders fork one shared session

A fresh builder in each item reads the same shared inputs (contracts, rubric,
API, checks) before it writes a line. A map can opt in to `fork`: one agent
node reads the shared inputs once, and each item's builder starts from a copy
of that session.

```json
{"id": "prime", "kind": "agent", "role": "explore",
 "task": "Read brief.md, the contracts and the checks. Do not edit files. Summarize what a builder needs."},
{"id": "groups", "kind": "map", "items": "{{state.groups}}", "fork": {"from": "prime"},
 "graph": {"...": "..."}}
```

With an edge `prime -> groups`:

1. `fork.from` must name an agent node. Use a read-only role (`explore`) so
   the source session writes no files.
2. On its first visit, each item's code builder forks the source session.
   Codex runs `codex exec fork`, which leaves the source thread as it was,
   so all items can fork it at the same time. Claude Code always forks.
3. A repair visit resumes the builder's own forked session, not the source.
4. Judges never fork. A review never shares a builder's context.
5. When the source node has no recorded session (it did not run, or it ran
   on an adapter without sessions), builders start fresh as before.

The forked session counts its own turn only. The copied history is not
cached for the new session, so each fork pays once for the source's input
tokens at the uncached rate. Keep the source turn short: read and summarize,
do not explore.

## Reviews bind to the final source

A judge's PASS is about the files that it read. When the judge or the code
nodes of its run declare `files`, the judge does three things:

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
other items do not make it wait. A judge after a map (a world review) has no
code node in its own graph, so give it `files` for the assembled source. A
judge with no `files` in a run whose code nodes declare no `files` has no
binding.

## Follow-ups

```bash
puppetmaster flow run fix-and-review.json --continue <prior_run_id> --input "also handle empty input"
```

The new run inherits the prior run's sessions and results. Each inherited
result records `inherited_from`: the run that produced it (the first one in a
chain). The run summary lists each inherited result that this run did not
produce again under `inherited`, with its verdict and, for a review, if its
files are still `current`. An inherited PASS describes the prior run's files,
not this run's. Every node resumes
its own session with the follow-up request instead of starting fresh, and a map
re-runs only the items the request names, or all items, each resumed.

### Reuse unchanged items

A map can opt in to `reuse`. On a follow-up run, an unchanged item then keeps
its prior result and does not build or review again:

```json
{"id": "groups", "kind": "map", "items": "{{state.groups}}",
 "reuse": {"inputs": "{{item.inputs}}", "names": "{{item.regions}}"},
 "graph": {"...": "..."}}
```

- `inputs` (required): the files each item's result depends on, for example
  its contracts and the shared checker and rubric. Do not list files that
  change in every phase, such as phase records: they make every item run.
- `names` (optional): more names that identify an item in a follow-up
  request, in addition to its key and its files.

When the map finishes an item, it records a fingerprint of that item: its
task (item graph and item value), the digest of its inputs, and the digest
of its own files. On the first map visit of a follow-up run:

1. When the request names no item, every item runs. Such a request can apply
   to every item.
2. An item that the request names runs. A request names an item by its key,
   a `names` alias, a file in its declared scope, or a file it changed. A map
   `feedback` template can name more items.
3. Any other item is reused only when its prior result passed, its task is
   the same, every declared input exists and is unchanged, its own files are
   unchanged, it has at least one review, and each review is a PASS whose
   files are unchanged. Otherwise it runs. An item without a recorded
   fingerprint runs.
4. An item never reuses when its item graph reads the request (`{{input}}`,
   or `{{prev}}` in its entry node). An item never reuses when a request
   cannot name it: a one-character or all-digit key with no `names` alias
   and no files.

The position of an item counts only when an item template reads `index` or
`state.index`, or a node mentions `PM_FLOW_CONTEXT`. A reordered list then
keeps reuse. A trailing `**` in `inputs` names every file below it. Each item records `reuse`: `reused` and the
reason, and for a reused item the item run that built it (`source_run`) and
its parent run (`source_parent`). Chained reuse keeps the original source. The
map output counts reused and run items. A reused item adds no usage to the
new run. Nodes after the map, such as a world review, run as usual.

When the run reaches a passing end, or ends because no edge leaves a passing
node, each reused item is checked again: a
changed input, own file or reviewed file fails the run with a "reused result
is stale" reason.

A cold run with `reuse` only records fingerprints. A map without `reuse`
works as before, and drops reuse records from earlier runs.

## Running and waking

| CLI | MCP `puppetmaster_flow` action |
| --- | --- |
| `flow validate <graph>` | `validate` |
| `flow save <graph>` (reuse by id) | `save` |
| `flow run <graph\|id> [--input] [--continue RUN] [--wait] [--foreground] [--worker-mode inline\|subprocess]` | `run` (always detached; `wait: true` blocks up to the MCP cap) |
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

Workers run as threads inside the walker by default (`--worker-mode inline`).
This removes one worker process start per node visit. The walker's process
group holds every worker and its CLI, so a stop of that group stops them all.
`--worker-mode subprocess` starts one process per worker. Those processes can
outlive a walker that dies alone.

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
