# Troubleshooting

Common failure modes, why they happen, and the one-line fix for each.
Most of these are MCP-transport edge cases or environment quirks
(macOS Cursor + pyenv + Node ABI). If you hit something not covered
here, `puppetmaster doctor` should flag it directly — open an issue
with that output attached.

## `Tool execution error. Not connected` from Cursor

This is Cursor's MCP client telling you it lost the stdio transport
to the Puppetmaster MCP server — **not** that your swarm or jobs
died. **Do not invent a JSON workflow config.** Start the analysis
swarm with one CLI line and keep working:

```bash
python -m puppetmaster swarm "your goal here"
python -m puppetmaster feed <job_id> --follow
```

Common triggers:

- Heavy concurrent load (parallel Cursor SDK swarm + CodeGraph index
  + large status payloads in the same window).
- Cursor reloading MCP settings, toggling the server, or restarting
  Cursor itself.
- An in-flight tool call exceeding Cursor's internal timeout.

**Prevention layer (v0.5.3+):** every long-running tool call now
emits JSON-RPC `notifications/message` keepalive frames every 10
seconds after a 5-second grace period. Bytes flowing on the stdio
pipe defeat the "transport looks dead" heuristic in Cursor's MCP
client. Short calls pay zero protocol cost. Tune or disable with:

- `PUPPETMASTER_MCP_KEEPALIVE_AFTER_SECONDS` (default 5)
- `PUPPETMASTER_MCP_KEEPALIVE_INTERVAL_SECONDS` (default 10)
- `PUPPETMASTER_MCP_KEEPALIVE_DISABLED=1` (turn off entirely)

**Root-cause fix (v0.5.6+):** Pre-v0.5.6, parallel
`puppetmaster_doctor` calls (or any other tool that fanned out to
multiple `subprocess.run` invocations) could silently kill the MCP
server with `exit_code=0` because subprocess children inherited the
parent's stdin by default. Concurrent spawn pressure caused the
parent's `for line in sys.stdin` loop to receive a phantom EOF and
exit cleanly — looking from Cursor's side exactly like
`Tool execution error. Not connected`. Every subprocess call in the
server's code path now passes `stdin=subprocess.DEVNULL`, severing
the inheritance chain. Verified by `bench/mcp_stress.py` (run it any
time: 6 scenarios in ~90s).

**Self-healing layer (v0.5.4+):** Cursor's MCP client uses a "lease"
lifecycle that periodically re-creates the logical client without
killing the previous Python MCP server. Without the keepalive above,
that left one orphan server per lease cycle holding open SQLite
handles and competing for the CodeGraph indexer lock. The
`_InputStalenessWatcher` measures **inbound** JSON-RPC traffic
directly: if no stdin message has arrived in 10 minutes **and** there
are zero in-flight tool calls, the server closes stdin and exits
through the normal `finally` block. Active sessions are never
interrupted; only true orphans reap. Tune or disable:

- `PUPPETMASTER_MCP_INPUT_STALE_SECONDS` (default 600)
- `PUPPETMASTER_MCP_INPUT_STALE_CHECK_SECONDS` (default 30)
- `PUPPETMASTER_MCP_INPUT_STALE_DISABLED=1`

**Idle-pipe keepalive (v0.5.5+):** Some Cursor builds close MCP
transports that have been quiet for a while, even between successful
calls. The `_IdleKeepalive` thread emits a tiny
`notifications/message` every ~25s while no tool call is running, so
the stdio pipe is never silent long enough to look dead. Cost is
trivial (~22 KB/hour). Tune or disable:

- `PUPPETMASTER_MCP_IDLE_KEEPALIVE_INTERVAL_SECONDS` (default 25, min 5)
- `PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED=1`

**Agent-side CLI fallback (v0.5.5+):** When the transport drops
anyway (e.g., during the lease transition itself), the bundled
Cursor rule (`.cursor/rules/puppetmaster-workflow.mdc`) and
`AGENTS.md` instruct the AI agent to call the equivalent
`python -m puppetmaster ...` command via its shell tool instead of
giving up. Every MCP tool has a matching CLI; read-only commands
(show/artifacts/logs/feed/status) auto-pivot to the project state
dir that owns the job, so no manual `PUPPETMASTER_STATE_DIR` export
is needed.

## CodeGraph indexes for different repos now run concurrently

Pre-v0.5.5, Puppetmaster used a single machine-wide lock to
serialize **all** CodeGraph indexers, so running
`puppetmaster_codegraph_index` against `ff-data-engineering` would
block the same call for `ff-ios` with
`Another CodeGraph indexer is already running (pid 80417)` — even
though the two repos have separate SQLite databases that can't trash
each other.

v0.5.5 keys the lock on the resolved repo root path
(`codegraph-indexer-<repo>-<digest>.lock`). Different repos index in
parallel; the lock only fires when two indexers are actually pointed
at the same repo's DB. Stale-PID auto-clear handles the post-`kill -9`
case: if the recorded PID isn't alive, the new claimant takes over
instead of refusing forever. Manual
`rm /Users/.../codegraph-indexer*.lock` is no longer needed after a
runaway indexer dies.

If the transport still drops, the recovery layer below catches the
fallout. When this happens, in-flight Puppetmaster swarms keep
running in the background (durable state — see
`python -m puppetmaster jobs` from a shell to confirm), but you
typically end up with one or more orphan
`python -m puppetmaster.mcp_server` processes holding open SQLite
handles and contending for the CodeGraph indexer lock.

**Diagnose:**

```bash
python -m puppetmaster mcp list
# 3 tracked  (1 alive, 0 stale, 2 dead)
#    PID  STATE        AGE     HBEAT  WORKSPACE
#  12345  ok            12s        8s  /Users/you/repo
#  11111  dead        4231s     4231s  /Users/you/repo
#  11112  dead        4231s     4231s  /Users/you/repo
```

`puppetmaster doctor` also flags this automatically.

**Clean up:**

```bash
python -m puppetmaster mcp cleanup --kill-stale
```

Then restart the Puppetmaster MCP server in Cursor
(Settings → MCP → toggle off/on). Inside an agent session you can
call `puppetmaster_mcp_status` / `puppetmaster_mcp_cleanup` directly
— handy for letting the agent self-diagnose right after a reconnect.

Each running Puppetmaster MCP server registers itself in
`~/Library/Caches/puppetmaster/mcp-servers/<pid>.json` (or
`$XDG_CACHE_HOME/puppetmaster/mcp-servers/` on Linux) and updates a
heartbeat from a background thread, so dead and stale entries are
detectable without grepping `ps`.

## CodeGraph reports `database is locked` from MCP, but works fine in the terminal

This is the most common gotcha on macOS Cursor installs. CodeGraph's
native SQLite driver (`better-sqlite3`) is locked to a specific Node
ABI. You have **two** Node runtimes that touch the same global
CodeGraph install:

| Runtime | Typical Node | NODE_MODULE_VERSION |
| ------- | ------------ | ------------------- |
| Your shell (`/opt/homebrew/bin/node`) | v23.x | 131 |
| Cursor's bundled Node (`Cursor.app/.../helpers/node`) | v22.22.0 | 127 |

If you ran `npm rebuild better-sqlite3` in your shell, it built for
the **shell's** Node, which means Puppetmaster's MCP (running under
Cursor's Node) silently falls back to the slow WASM driver and
you'll see `database is locked` / `unable to open database file`.
`puppetmaster doctor` will flag this as `native better-sqlite3
broken; codegraph is on slow WASM fallback.`

**One-command fix:**

```bash
python -m puppetmaster repair-codegraph
```

It auto-detects Cursor's bundled Node, locates the global CodeGraph
install, runs `npm rebuild better-sqlite3` with Cursor's Node on
PATH, and verifies the backend reports as native. Then restart the
Puppetmaster MCP server in Cursor (Settings → MCP → toggle off/on).

You can also call it from inside the agent itself via the
`puppetmaster_repair_codegraph` MCP tool — useful if an agent hits
the WASM fallback mid-session and can self-heal.

**v0.9.7 makes this fully automatic and kills the bare-shell footgun.**
Two changes: (1) every CodeGraph call routed through Puppetmaster
(`run_codegraph_cli`, used by the `puppetmaster_codegraph_*` MCP tools
and the CLI passthrough) now detects the `NODE_MODULE_VERSION` /
"compiled against a different Node.js version" native-load error,
rebuilds `better-sqlite3` against Cursor's Node **once per process**,
and retries — so you no longer have to run `repair-codegraph` by hand.
(2) A new `python -m puppetmaster codegraph <args>` passthrough runs the
CLI under Cursor's bundled Node. **Stop calling a bare `codegraph …`
from your shell** — that picks up your shell's Node (wrong ABI) and dies
with the native-load error. Use `python -m puppetmaster codegraph status`
/ `search` / `context` / `init --index` instead, which is also the
correct MCP-down fallback. Disable the auto-rebuild with
`PUPPETMASTER_CODEGRAPH_AUTOHEAL=0`.

**Tradeoff:** `better-sqlite3` is ABI-specific. Rebuilding for
Cursor's Node 22 may break native SQLite in your terminal (Node 23)
until you rebuild again with the shell's Node. For day-to-day Cursor
use, optimize for Cursor's Node. If you upgrade Cursor and the
bundled Node ABI changes, re-run `puppetmaster repair-codegraph`.

**v0.5.4 makes this self-correcting at runtime.** Puppetmaster
invokes `codegraph` by explicitly running its `codegraph.js`
entrypoint **under Cursor's bundled Node** whenever both are
discoverable (via the new `resolve_codegraph_invocation()` helper),
regardless of which Node sits first on `$PATH`. That eliminates the
failure mode where a stray shell shim under Homebrew Node spins up
an indexer in WASM mode and locks the DB for hours. The
`puppetmaster doctor` `codegraph` check now also verifies against
the runtime Puppetmaster actually uses — not whichever shim happens
to be on PATH — so you get `ok (verified under Cursor's bundled
Node)` instead of a misleading `warn` when MCP is healthy.

Escape hatches for weird installs:

- `PUPPETMASTER_CODEGRAPH_NODE` — full path to the Node binary to use.
- `PUPPETMASTER_CODEGRAPH_JS` — full path to `codegraph.js`.

Both must be set together; auto-detection runs otherwise.

## `puppetmaster adapters` says `cursor: configured=false`, but my swarms work

You're probably running it from a workspace where you don't have
`@cursor/sdk` installed locally. The Puppetmaster MCP loads the SDK
from the **package install dir's** `node_modules`, not from your
cwd — so the swarm worked fine while diagnostics lied. v0.5.4 fixes
the detection: `_cursor_sdk_installed` now checks both the workspace
and the package install dir, and reports the location it found:

```text
ok  cursor-sdk   @cursor/sdk installed (/Users/.../Puppetmaster/node_modules/@cursor/sdk)
```

`PUPPETMASTER_HOME` is an explicit escape hatch if your install lives
somewhere unusual.

## `puppetmaster show <job_id>` fails from any cwd other than the workspace that ran the job

Pre-v0.5.4, each workspace had its own per-project SQLite state dir
hashed from the resolved git root. If you ran a swarm in
`/Users/you/ff-ios` and later tried `puppetmaster show job_X` from
`/tmp`, it would fail with `job not found` even though the job was
alive.

v0.5.4 auto-pivots. Read-only commands (`show`, `artifacts`, `diff`,
`feed`, `logs`, `events`, `status`, `memory`, `open`) scan every
project state dir on the machine and use the one that owns the job,
with a single `note:` on stderr telling you which it picked:

```text
$ cd /tmp && python -m puppetmaster show job_4fc8c7148d65
note: job job_4fc8c7148d65 not in current workspace state dir; using /Users/.../projects/Puppetmaster-7b41939e66e6
# Puppetmaster Stitched Summary
...
```

Two new commands round it out:

- `python -m puppetmaster projects` — lists every project state dir
  on this machine with job counts and last activity.
- `python -m puppetmaster jobs --all-projects` — flattens jobs from
  every project into one stream with a project column.

Write-side commands (`run`, `cursor`, `claude`, `daemon`, ...)
intentionally do *not* pivot. Those always use the caller's
workspace state. Explicit `--state-dir` or `$PUPPETMASTER_STATE_DIR`
overrides also disable the pivot.

## Safety model

Puppetmaster can orchestrate tools that edit code. The safety model
is explicit:

- Cursor defaults toward review/plan/dry-run workflows.
- Claude Code is full-edit, but blocked on dirty worktrees by default.
- Patch outputs are artifacts with diffs and base SHAs.
- Approval/rejection is recorded in the event stream.
- Stale workers are recovered through leases.
- Failed provider calls become structured artifacts instead of
  mystery crashes.
- Secrets stay in environment variables, never config files.

If you paste a key into a terminal, chat, issue, screenshot, or
transcript, rotate it before publishing.

## State directory layout

By default, Puppetmaster keeps runtime state outside the repository
so `git status` stays focused on source changes:

```text
macOS: ~/Library/Application Support/puppetmaster/projects/<workspace>-<hash>/
Linux: ~/.local/state/puppetmaster/projects/<workspace>-<hash>/
```

Print the resolved location:

```bash
python -m puppetmaster state
```

Override it when you intentionally want repo-local or CI-specific
state:

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

`.puppetmaster/` remains in `.gitignore` as a compatibility fallback
for explicit local state.

Core objects:

- `Job`: one swarm run and user goal
- `Task`: role-specific work, optionally dependency-gated
- `AgentRun`: one worker attempt
- `Artifact`: structured output with payload, evidence, confidence,
  and `sha256`
- `MemoryRecord`: promoted fact retrieved by later workers

## Windows: blank console flashes beside browser / login

Owned Puppetmaster worker and helper subprocesses pass `CREATE_NO_WINDOW`
via `puppetmaster/win_console.py` so they do not allocate a visible console
when the parent is a console-less Electron/backend process.

If a **blank terminal still flashes** during Cursor Agent login or MCP tool
use, it is usually a **Cursor Node grandchild** (MCP server spawned by the
Agent CLI). That spawn path must set `windowsHide` / `CREATE_NO_WINDOW` inside
Cursor's Node runtime — Python `creationflags` on the top-level `agent`
process cannot reach those grandchildren. Marionette's own `agent login`
path uses `CREATE_NO_WINDOW` + direct `node`/`index.js` exec (not
`CREATE_NEW_CONSOLE`).
