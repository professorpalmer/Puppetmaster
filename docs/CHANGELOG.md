# Changelog

## v0.6.0-beta.2

- **Auto-routing is now the default on built-in swarm workers.** `DEFAULT_WORKERS` (used by `puppetmaster_start_cursor_swarm`, `puppetmaster_start_swarm`, and `python -m puppetmaster run`) now ship with `payload.auto_route = True`. Any swarm started through Puppetmaster's MCP surface now consults the router automatically — no per-spec opt-in required. If `~/.puppetmaster/models.json` doesn't exist yet, the orchestrator emits one `router.registry_empty` event and falls back to the spec's declared adapter, so this change is safe even before running `models init`.
- **Vision-aware routing.** Classifier now scans the instruction for vision signals (image, screenshot, diagram, chart, OCR, visual, photo) and auto-adds `vision` to the required tags. "Detailed" vision tasks (OCR / "every detail of the diagram" / explicit `detailed` qualifiers) also add `detailed-vision`. Models in the registry that don't carry those tags get filtered out with a clear rejection reason, so a vision request never lands on a text-only model.
- **Starter registry rebuilt to match the user mental model.** `puppetmaster models init` now writes four tiers: `cursor/composer-2-5` (fast / cheap / reading), `cursor/gpt-5-5` (balanced), `claude-code/opus-4-6` (high-quality, vision), `claude-code/opus-4-7` (frontier, detailed-vision). `adapter_model_name` defaults to model ids that already work today (`composer-1`, `gpt-5`, `claude-opus-4-5`); each entry's `notes` field flags what to edit when newer versions land. Tier ids stay stable.
- **Balanced-policy tie-break flipped.** When two sufficient models cost the same (common with `$0` Cursor-plan models), the router now picks the **lower** capability_score instead of the higher one — right-sizing the model to the task instead of wasting capability. Matches user intent ("easy tasks get the cheap tier even when expensive tiers are technically 'free' under my Cursor plan").
- **New `puppetmaster cost <job_id>` CLI + `puppetmaster_job_cost` MCP tool.** Sums `estimated_cost_usd` across all `ROUTING` artifacts for a job and breaks it down per-model. Answers "how much did this swarm cost?" with the same data the router already wrote.
- Tests: 162 passing (added 9 covering vision detection, detailed-vision detection, auto-required vision tag, starter-registry tier structure + monotone capability, balanced tie-break, default-workers opt-in, empty-registry pass-through, and the cost command).

## v0.6.0-beta.1

- **Intelligent model orchestration.** Puppetmaster now ships a task-aware **model router** that picks the right LLM for each task instead of pinning one model per adapter. Every swarm role can opt in by setting `payload.auto_route = true`; the orchestrator then:
  1. Runs a transparent capability classifier over the task's role + instruction + payload to assign a 0..100 score (e.g. `verify-runtime` ≈ 25, `explore` ≈ 50, `implement` ≈ 75, `audit/security-review` ≈ 90+).
  2. Picks a `ModelSpec` from the user-owned registry using one of four policies — `balanced` (default, cheapest sufficient), `cheap` (lowest cost regardless of fit), `quality` (highest capability), `escalating` (ordered chain for retries).
  3. Stamps the chosen `adapter` + model name into the task so the existing adapter pipeline runs the routed model with zero plumbing changes.
  4. Persists an `ArtifactType.ROUTING` artifact recording the chosen model, the classifier output, the estimated USD cost, and the full list of rejected alternatives with the reason each was rejected — so every routing decision is fully auditable in `puppetmaster show` / `puppetmaster artifacts`.
- **User-owned model registry.** Lives at `~/.puppetmaster/models.json` (override with `$PUPPETMASTER_MODELS_PATH`). Users describe their own models, prices, and asserted capability scores; Puppetmaster never hardcodes model names or fetches live prices. `puppetmaster models init` writes a starter registry; `puppetmaster models list` prints it as a table.
- **`puppetmaster route "<instruction>"` CLI** and `puppetmaster_route_task` / `puppetmaster_list_models` MCP tools. Dry-run a routing decision before kicking off a swarm — see which model would run, the estimated cost, and why the cheaper alternatives were rejected.
- **Per-task overrides:** `payload.min_capability` (force classifier output), `payload.max_cost_usd` (hard budget cap), `payload.required_tags` (e.g. `["long-context"]`), and `payload.routing_policy` (one of `balanced`/`cheap`/`quality`/`escalating`).
- **Honest scope:** v1 routes to adapters we already have (`claude-code`, `cursor`) — those alone cover Sonnet/Haiku/Opus and any Cursor SDK model variant. Raw HTTP adapters for additional providers (Gemini, DeepSeek, Kimi, OpenAI direct) slot in as new `adapter` values; the registry + router/classifier framework doesn't need to change. Pricing and capability scores stay user-asserted — Puppetmaster makes the **decision** transparent, not the **value judgments**.
- Tests: 153 passing (added 18 covering classifier behavior across roles, all four policies, tag filters, cost budget caps, registry round-trip, env override, CLI `route --json`, and end-to-end orchestrator integration that emits a `ROUTING` artifact tied to the real task id).

## v0.5.6-beta.1

- **Root-cause fix for "MCP drops on robust questions".** Found via a new live-fire stress harness (`bench/mcp_stress.py`) that drives the real MCP server over stdio with six load patterns. The failing pattern: `puppetmaster_doctor` (which fans out to multiple `subprocess.run` calls for `git`/`node`/`npm`/`codegraph status`) called in parallel from many requests would silently kill the server with `exit_code=0` and `stdin EOF`. Every `subprocess.run`/`subprocess.Popen` in the server-touched code path was inheriting the parent's fd 0 by default. Under concurrent spawn pressure, that inheritance somehow caused the parent's `for line in sys.stdin` loop to receive a phantom EOF and exit cleanly — looking from Cursor's side exactly like "Tool execution error. Not connected". The fix: every subprocess call in `codegraph.py`, `codegraph_repair.py`, `codegraph_index_runner.py`, `state.py`, `diagnostics.py`, and `mcp_server.py` now passes `stdin=subprocess.DEVNULL`, so children never touch the parent's stdin.
- **New stress harness for regression detection.** `python -m bench.mcp_stress` runs six scenarios against a real-spawned MCP server: 20 parallel doctor calls, 64 simultaneous senders, large payload, doctor + codegraph mixed, sustained 5-rps traffic, and idle->busy->idle transitions for the idle keepalive. All six pass on v0.5.6. Run anytime to verify the production server behavior in <90s.
- **New `PUPPETMASTER_MCP_DIAG_EXIT=1` env flag** that prints the main-loop exit reason on stderr (`stdin_eof`, `sigint_clean_shutdown`, `main_loop_exception: ...`). Off by default; turn on temporarily when an orphan server exits unexpectedly to learn why.
- New regression test (`test_parallel_doctor_calls_do_not_kill_mcp_server`) spawns the real server over stdio, sends 30 parallel doctor calls from 30 sender threads, and asserts every response returns and the server stays alive. This is the unit-level guard for the v0.5.6 fix.
- Tests: 135 passing (added 1 regression). The stress harness was the diagnostic that surfaced the bug; the unit test is the day-to-day guard.

## v0.5.5-beta.1

- **CodeGraph indexes are now per-repo, not machine-wide.** The previous lock was over-eager: each repo has its own SQLite DB at `<repo>/.codegraph/codegraph.db`, so two indexers on two different repos can't trash each other's data. The lock now keys on the resolved repo root path hash (`codegraph-indexer-<repo>-<digest>.lock`), so `ff-data-engineering`, `ff-ios`, and `ios-qa-agent-1` can index in parallel. The lock only blocks legitimate overlap on the same SQLite DB. Legacy callers passing no `repo_root` still get the global lock file for backwards compatibility.
- **Stale-PID auto-clear on lock acquire.** If the lock file records a PID that is no longer alive (`os.kill(pid, 0)` raises `ProcessLookupError`), the new claimant truncates the file and retries the `flock` once before reporting busy. This fixes the post-`kill -9` failure mode where a runaway `codegraph init --index` leaves a lock file pointing at a dead PID and every subsequent indexer attempt sees `Another CodeGraph indexer is already running (pid 80417)` indefinitely. Lock files owned by a *different* user are never blown away (treated as alive by design).
- **Idle MCP-pipe keepalive.** Complements v0.5.3's per-tool-call keepalive: a new `_IdleKeepalive` thread emits a small `notifications/message` every 25s when no tool call is in flight and the server is otherwise idle. Some Cursor builds close the stdio MCP transport for a chat that has been quiet for a while, producing `Tool execution error. Not connected` on the agent's very next call — even though the daemon is healthy. Bytes flowing on the pipe defeat that heuristic. Frames suppress automatically while any tool call is running (the per-call keepalive already handles that window) and stop on `BrokenPipeError`. Cost: ~150 bytes per frame, ~22 KB/hour. Tunable via `PUPPETMASTER_MCP_IDLE_KEEPALIVE_INTERVAL_SECONDS` (default 25, minimum 5) and `PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED=1`.
- **Agents now know how to fall back to the CLI when MCP drops.** The Cursor rule (`.cursor/rules/puppetmaster-workflow.mdc`) and `AGENTS.md` both gained a "When MCP fails" section with a complete mapping from every `puppetmaster_*` MCP tool to its `python -m puppetmaster ...` CLI equivalent (plus the bundled `codegraph` commands). The rule explicitly tells agents to **not** stop work on `Tool execution error. Not connected`, and to only ask the user to restart MCP in Cursor Settings if `python -m puppetmaster mcp list` shows zero alive servers. Read-only CLI commands auto-pivot to the project state dir that owns the job (v0.5.4-beta.2), so the fallback works from any cwd with zero manual state-dir wrangling.
- Tests: 134 passing (added 9 covering per-repo lock paths, parallel-different-repo claims, same-repo busy, stale-PID auto-clear, the `_pid_is_alive` helper, idle keepalive emission, suppression-during-tool-call, broken-pipe stop, env disable). End-to-end smoke-tested: `notifications/message` `idle_keepalive` frames arrive on the wire at the configured interval while the server is idle.

## v0.5.4-beta.2

- **Adapter detection now looks where the SDK actually lives.** `puppetmaster adapters` and `puppetmaster doctor` previously reported `cursor: configured=false` from any workspace other than the Puppetmaster install dir, because `_cursor_sdk_installed` only checked `<cwd>/node_modules/@cursor/sdk` — but `cursor_sdk_runner.mjs` resolves the SDK from the Puppetmaster *package* dir at runtime. The check now searches both the workspace and the package install (and an explicit `PUPPETMASTER_HOME` escape hatch), so diagnostics agree with what the swarm actually does.
- **Cross-workspace job lookup for read-only CLI commands.** `puppetmaster show`, `artifacts`, `diff`, `feed`, `logs`, `events`, `status`, `memory`, and `open` now auto-pivot to the project state dir that owns a given `job_id` if it isn't present in the current workspace's state dir. No more `PUPPETMASTER_STATE_DIR=~/Library/.../ff-data-engineering-cfbfad67d9fc python -m puppetmaster show ...` — just paste the job id from any cwd. Explicit `--state-dir` or `$PUPPETMASTER_STATE_DIR` overrides still take precedence (write-side commands always use the caller's workspace, never pivot).
- **New `puppetmaster projects` command + `puppetmaster jobs --all-projects` flag.** `projects` lists every Puppetmaster project state dir on this machine with job counts and last activity. `jobs --all-projects` flattens jobs from every project into one stream with a project column. Together they make "which workspace owns this job?" a one-liner.
- **Orphan self-termination via stdin staleness watcher.** Cursor's MCP "lease" lifecycle periodically re-creates the logical client without killing the previous Python server, so before this release Puppetmaster could accumulate one MCP server per lease cycle — each still holding SQLite handles and competing for the CodeGraph indexer lock. The new `_InputStalenessWatcher` measures inbound JSON-RPC traffic directly (not just process liveness like the heartbeat thread): if no stdin message arrives for 10 minutes **and** there are zero in-flight tool calls, the watcher closes stdin so the main loop sees EOF and exits through the normal `finally` block (deregister, stop heartbeat, shut down executor). No `os._exit`, no leaked tracking files. Tunable via `PUPPETMASTER_MCP_INPUT_STALE_SECONDS`, `PUPPETMASTER_MCP_INPUT_STALE_CHECK_SECONDS`, and `PUPPETMASTER_MCP_INPUT_STALE_DISABLED`.
- **Explicit Cursor-Node invocation for CodeGraph.** New `resolve_codegraph_invocation()` in `puppetmaster.codegraph` returns `[cursor_node, codegraph.js]` whenever Cursor's bundled Node and the global `@colbymchenry/codegraph` install are both discoverable. Every subprocess that previously invoked the bare `codegraph` shim — `codegraph_context`, `codegraph_status_line`, `run_codegraph_cli`, and the background `codegraph_index_runner` — now invokes CodeGraph under Cursor's Node first, with a transparent fallback to the PATH shim when Cursor isn't installed (Linux, headless CI). This closes the 18-hour-runaway-indexer class of bug, where a stray shell invocation under Homebrew Node would fall back to WASM SQLite, lock the DB, and burn 99% CPU for hours. `PUPPETMASTER_CODEGRAPH_NODE` / `PUPPETMASTER_CODEGRAPH_JS` env vars are an escape hatch for non-standard installs.
- **Doctor distinguishes shell-Node WASM from Cursor-Node WASM.** `puppetmaster doctor`'s `codegraph` check used to warn whenever the shim on PATH reported WASM, even after `repair-codegraph` had successfully rebuilt better-sqlite3 for Cursor's Node — leading to misleading "broken" diagnoses when MCP was actually healthy. The check now runs against the same invocation Puppetmaster uses at runtime (`resolve_codegraph_invocation()`) and reports `ok (verified under Cursor's bundled Node)` when the runtime that matters is healthy.
- New tests covering invocation-resolution under Cursor Node, environment overrides, the fallback path, staleness-watcher trigger/hold-off/reset, env-disable, dispatcher counter, doctor's Cursor-Node disambiguation, SDK detection from the package install dir, and cross-workspace job lookup. Total suite at 125 tests, all passing.

## v0.5.3-beta.1

- **Tool-call keepalive notifications.** Every long-running MCP `tools/call` now spawns a per-call `_ToolCallKeepalive` daemon thread that emits MCP-spec `notifications/message` frames every 10 seconds (after a 5-second grace period) for as long as the handler runs. Short calls pay zero protocol cost; the goal is to keep bytes flowing on the stdio pipe so Cursor's MCP client stops treating slow tool calls as a dead transport. This is the prevention complement to v0.5.2's cleanup — together they cover both "stop the drop from happening" and "recover cleanly when it does."
- All keepalive frames are unidentified JSON-RPC notifications (no `id`) and carry a structured `data.kind = tool_call_progress` payload with the tool name, request id, and elapsed time so logs and progress UIs can render them.
- Tunable via `PUPPETMASTER_MCP_KEEPALIVE_AFTER_SECONDS`, `PUPPETMASTER_MCP_KEEPALIVE_INTERVAL_SECONDS`, and `PUPPETMASTER_MCP_KEEPALIVE_DISABLED`. Default behaviour is on and conservative.
- Broken-pipe detection: if the emitter sees a dead stdio (`BrokenPipeError`/`OSError`), the keepalive thread shuts down cleanly instead of spinning. Post-response races are also closed via a stop-check inside the `_STDOUT_LOCK`.
- README troubleshooting section now distinguishes the prevention layer (this release) from the recovery layer (v0.5.2 mcp registry + cleanup tools).

## v0.5.2-beta.1

- **MCP server process registry + orphan cleanup.** Every Puppetmaster MCP server now writes a tracking file (PID, workspace, started_at, last_heartbeat) at startup, bumps its heartbeat from a daemon thread, and removes its file on a clean exit. New servers prune dead tracking files on startup so orphans from prior `Tool execution error. Not connected` cycles get reaped automatically — no more `pkill -f puppetmaster.mcp_server` from a shell.
- New `puppetmaster/mcp_registry.py` module with `register()`, `heartbeat()`, `deregister()`, `list_entries()`, `prune_dead()`, `kill_stale()`, and a `HeartbeatThread`. Per-user storage lives under `~/Library/Caches/puppetmaster/mcp-servers/` on macOS and `$XDG_CACHE_HOME/puppetmaster/mcp-servers/` on Linux; overridable via `PUPPETMASTER_MCP_REGISTRY_DIR`.
- New CLI surface: `python -m puppetmaster mcp list` (with `--json`) and `python -m puppetmaster mcp cleanup` (`--kill-stale`, `--stale-after-seconds`, `--json`). Lists every tracked server with alive/stale/dead flags and reclaims dead entries; `--kill-stale` SIGTERMs (and SIGKILLs after a grace period) servers whose heartbeat is older than the threshold. The current PID is never signalled.
- New MCP tools `puppetmaster_mcp_status` and `puppetmaster_mcp_cleanup` so an agent can self-diagnose and self-heal right after a transport reconnect without you running anything in a shell.
- `puppetmaster doctor` now includes an `mcp-servers` check that warns when dead tracking files or stale-but-alive servers are present, and points at the new cleanup command. This is the smoking gun behind a stale `Tool execution error. Not connected` symptom.
- README: new "`Tool execution error. Not connected` from Cursor" Troubleshooting section explaining the failure mode (Cursor closes the stdio transport while the swarm keeps running), the diagnostic command, the cleanup command, and the durable-state implication that orphan reconnects are recoverable.

## v0.5.1-beta.1

- **Self-healing CodeGraph SQLite repair.** New `python -m puppetmaster repair-codegraph` CLI command and matching `puppetmaster_repair_codegraph` MCP tool that auto-detect Cursor's bundled Node, locate the global `@colbymchenry/codegraph` install, and `npm rebuild better-sqlite3` against the right runtime — then verify the backend reports as native. This closes the most common Cursor footgun: a shell-Node `npm rebuild` builds for the wrong ABI (Homebrew Node 23 vs. Cursor's bundled Node 22), so the Puppetmaster MCP keeps falling back to slow WASM SQLite and emits spurious `database is locked` errors.
- Rewrite `CODEGRAPH_NATIVE_SQLITE_HINT` and `puppetmaster doctor`'s warn message to explain the dual-Node ABI trap and point users at `puppetmaster repair-codegraph` instead of the generic `npm rebuild` command that lands on the wrong runtime.
- New `puppetmaster/codegraph_repair.py` module with `find_cursor_node()`, `find_codegraph_install()`, and `repair_codegraph_sqlite()` — covered by tests for macOS/Linux/Windows candidate paths, explicit overrides, rebuild failure surfacing, and verification parsing.
- README: new Troubleshooting section documenting the dual-Node-ABI gotcha, the ABI tradeoff (rebuilding for Cursor breaks the terminal until you rebuild back), and the one-command fix.

## v0.5.0-beta.1

- **Multi-threaded MCP request dispatch.** Replace the single-threaded `for line in sys.stdin` loop with a `ThreadPoolExecutor` so long-running tool calls never block the stdio transport. This fixes the "Tool execution error. Not connected" failure mode where a heavy CodeGraph call would silently freeze every other agent tool call. Worker count is configurable via `PUPPETMASTER_MCP_WORKERS` (default 8).
- **Background CodeGraph indexer.** `puppetmaster_codegraph_init(index=true)` now runs the fast init synchronously and dispatches the slow `codegraph index` to a detached subprocess via the new `puppetmaster/codegraph_index_runner.py` launcher, returning immediately with a `run_id`, `pid`, and stdout/stderr log paths. Added a dedicated `puppetmaster_codegraph_index` MCP tool for the standalone background-index case.
- **Global indexer lock.** A user-scoped lock file (`~/Library/Caches/puppetmaster/codegraph-indexer.lock` on macOS, `$XDG_CACHE_HOME/puppetmaster/...` on Linux, overridable with `PUPPETMASTER_CODEGRAPH_LOCK_DIR`) serializes CodeGraph indexers so two parallel runs can never trash a SQLite database. Lock-busy returns a clear `CodegraphLockBusy` payload with the holder PID and a `pkill` hint.
- **Broken native-SQLite detection.** `puppetmaster doctor` and `puppetmaster_codegraph_status` now flag the `better-sqlite3` WASM fallback (Node ABI mismatch) and surface the `npm rebuild better-sqlite3` fix-it hint. `codegraph_native_sqlite_broken(status_output)` is the helper.
- Shorten `DEFAULT_INIT_TIMEOUT_SECONDS` from 600 → 60 since synchronous init is now bounded to the fast scaffold-creation step; indexing is always asynchronous.

## v0.4.0-beta.1

- Bundle six CodeGraph CLI commands into the Puppetmaster MCP (`puppetmaster_codegraph_search`, `_context`, `_affected`, `_files`, `_status`, `_init`) so Cursor Agent only needs one MCP server for both orchestration and repo intelligence.
- Auto-inject CodeGraph context into Cursor and Claude Code worker prompts when the target repo is initialized; tag verification artifacts with `context:codegraph` so the operator can confirm shared intelligence was used.
- Make MCP swarm starts asynchronous: `puppetmaster_start_swarm`, `_start_cursor_swarm`, and friends return a `job_id` immediately instead of holding the MCP call open.
- Add push-style live artifact streaming: `puppetmaster_live_artifacts_follow` (MCP) and `python -m puppetmaster feed <job> --follow` (CLI) long-poll the durable SQLite event cursor and return as soon as a new artifact lands.
- Add warm worker daemon mode (`python -m puppetmaster daemon --roles ...`) for local swarms that want to avoid repeated worker process startup.
- Make Puppetmaster the default subagent runtime: `.cursor/rules/puppetmaster-workflow.mdc` (`alwaysApply: true`) plus a top-level `AGENTS.md` direct Cursor Agent and Claude Code to route non-trivial work through Puppetmaster by default.
- Move runtime state out of the repository by default (macOS: `~/Library/Application Support/puppetmaster/projects/...`, Linux: `~/.local/state/puppetmaster/projects/...`). Override with `--state-dir` or `PUPPETMASTER_STATE_DIR`.
- Parse Cursor SDK swarm output into typed artifacts instead of leaving raw stdout blobs.
- Add cost-structure benchmark `bench/three_way.py` (Agent only vs CodeGraph alone vs Puppetmaster + CodeGraph) and prompt-enrichment harness `bench/codegraph_ab.py`. Numbers in the README and `bench/README.md` are reproducible from these scripts.
- Reposition the README so Puppetmaster's standalone identity (durable state for parallel coding-agent swarms) is the headline, with CodeGraph as an explicit optional integration.

## v0.3.0-beta.1

- Ship the Cursor / VS Code extension with a Puppetmaster activity-bar control panel.
- Add the Cursor Agent MCP server so Agent chat can call Puppetmaster tools directly.
- Add MCP tools for `doctor`, Cursor review/plan, Claude Code implementation, logs, artifacts, and partial summaries.
- Add the Claude Code full-edit adapter with dirty-worktree protection and patch artifact capture.
- Fix editable Python install for local source-checkout usage.
- Expand Cursor, MCP, Claude Code, and daily-driver docs.

## v0.2.0-beta.1

- Position Puppetmaster as a supervised daily-driver beta for Cursor swarm workflows.
- Add run-management commands for latest job lookup, logs, reruns, cleanup, patch artifact review, and approval events.
- Add dry-run Cursor workflow support.
- Add full-edit Claude Code CLI adapter and `puppetmaster claude` command.
- Harden job and worker failure handling.
- Add SQLite schema version metadata and doctor validation.
- Retrieve promoted memory into new task payloads.
- Expand production, security, contribution, and release documentation.

## v0.1.0

- Establish the core local swarm runtime.
- Add subprocess workers, structured artifacts, stitching, and memory promotion.
- Add file and SQLite coordination backends.
- Add shell and optional Cursor adapters.
- Add crash recovery demo, CI, and starter docs.

## Planned v0.2.0

- Broaden provider adapters beyond Cursor.
- Improve patch artifact generation and isolated apply flows.
- Add richer watch output and scripting-friendly JSON modes.
