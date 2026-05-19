# Changelog

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
