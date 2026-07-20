# AGENTS.md

Repo-level guidance for AI coding agents (Cursor, Claude Code, Codex, etc.) working in this repository. Cursor users see the same directive via `.cursor/rules/puppetmaster-workflow.mdc`.

## Default subagent runtime: Puppetmaster

When this repo is open, **Puppetmaster's MCP tools are the primary path for fanning out work**. Do not roll your own grep/read exploration loops or invoke ad-hoc subagents when a Puppetmaster swarm is available.

## When to start a Puppetmaster swarm (do this by default)

For any request that involves more than a single trivial change, start a Puppetmaster swarm **before** spending tokens on your own exploration:

- Broad repo investigation, audit, or risk analysis ("review this", "find issues", "what could break").
- Multi-file refactors, migrations, or cross-cutting cleanups.
- Debugging that requires tracing call graphs, data flow, or test coverage across multiple files.
- Planning the next implementation slice when scope or risks are unclear.
- Comparing approaches, surfacing trade-offs, or producing decision artifacts.

Default routing:

1. `puppetmaster_start_cursor_swarm` — multi-role read-only analysis (the daily-driver entry point).
2. `puppetmaster_start_implement` — durable, patch-producing implementation that runs on whatever platform the lock enables (cursor preferred, then claude-code). Use this as the default implement verb so it works regardless of platform lock. `puppetmaster_start_cursor_implement` / `puppetmaster_start_claude_implement` force a specific platform.
3. `puppetmaster_start_cursor_review` / `puppetmaster_start_cursor_plan` — lightweight single-pass review or plan.

Every implement verb runs full-edit in a clean worktree (clean-tree guard; set `allow_dirty` to override) and captures the resulting diff as a `PATCH` artifact.

Start tools return a `job_id` immediately. Do **not** wait inside one long MCP call.

## Match the verb to the task shape (single feature ≠ swarm)

A swarm is for **read-only, decomposable analysis** — explore, review, audit, plan, redteam — where independent roles can run in parallel without touching the same code. It is **not** the right shape for one coupled feature. Fanning out a single tightly-coupled change makes parallel workers re-ingest the same context and land commits that are unaware of each other, which produces conflicts, rework, and broken delivery — and erases any token savings.

So:

- **Implementing one feature / fixing one ticket → a single `puppetmaster_start_implement` worker** in a clean worktree. One worker keeps the change coherent and yields one reviewable `PATCH`. Reserve swarms for the explore/review/audit passes *around* the feature, not the edit itself.
- **A focused single edit, or any change that builds on uncommitted work → `puppetmaster_edit`** (in-place, `allow_dirty`, cheapest sufficient model + CodeGraph, reviewable `PATCH`). This is the verb for last-mile work — "finish the module I just wrote", "add tests for the code I just added". `puppetmaster_start_implement` branches off HEAD in an isolated worktree, so it **cannot see uncommitted modules** and would clobber or rebuild them; never reach for it when the work depends on dirty-tree state. Keep truly trivial edits (typo/rename/comment) inline.
- **Genuinely independent slices** (e.g. "add the same header to 30 unrelated endpoints") can fan out — but split by non-overlapping files so workers never collide.
- **Broad investigation / audit / "find all X" → a read-only swarm.** That is what it is built for.

Puppetmaster's edge is mixed workloads, heavy trivial sub-tasks, long-context horizons, and zero-token artifact recall across sessions — not winning a single hard implementation against one strong steered agent. Output-compression and context-hygiene tools (e.g. RTK) are **additive** to Puppetmaster, not competitors: let them shrink tool-output tokens while Puppetmaster owns orchestration and durable state.

## When NOT to route through Puppetmaster

Use native tooling directly for:

- Trivial single-file edits with obvious intent (rename, add comment, fix typo).
- Questions answerable from the current visible file or recent context.
- Conversational follow-ups that don't change repo state.
- Anything explicitly framed as "just answer me" / "no swarm".

## Browser swarms (live-site QA)

For QA that needs a **real browser against a live site** — capturing actual
network payloads instead of mocked ones — use the first-class browser verb, not
a read-only swarm. A read-only swarm can't reach it: the MCP swarm specs hardcode
a `file,web,vision` toolset list with no `browser`, and the cursor swarm adapter
has no browser at all. Only the **Hermes** adapter can drive a browser
(`hermes chat -t browser`).

- **Verb:** `puppetmaster_start_browser_swarm` (MCP) / `python -m puppetmaster browser "<task>" ["<task2>" ...]` (CLI). Each task becomes one parallel Hermes worker. Requires the Hermes platform enabled.
- **Single source of truth:** `puppetmaster/browser.py`. It bakes in three hard-won guardrails so callers don't re-derive them:
  1. **React-controlled inputs** — automation must use native value setters + dispatched `input`/`change` events, or submits fire nothing (a fake, reproducible "bug").
  2. **Network truth** — judge success by the captured request/response, not the UI; an HTTP 200 can carry an application error body.
  3. **Strong-model floor** — browser grounding needs a capable model; the spec carries a high `min_capability` (default 80) pinned to the Hermes adapter, because cheap models fail browser grounding *and lie about it*. Private/VPN-only hosts rely on Hermes' local-engine fallback for private URLs.
- **Safety posture:** a browser worker edits no repo files (`swarm_mode` stays `analysis`, no clean-tree guard) but is an **acting agent** with external side effects (logins, form fills). Specs carry `payload.side_effecting = True`; `workers.spec_has_side_effects` / `swarm_is_acting` drive an acting-agent banner. Treat with implement-style approval, never the swarm's "read-only, harmless" framing.
- The reusable playbook with the full gotcha derivation lives in the `jadui-browser-qa-swarm` skill.

## How to drive a started swarm

1. Return the `job_id` to the user immediately, in one line.
2. Prefer `puppetmaster_live_artifacts_follow` (long-poll, push-style) over polling `puppetmaster_status` in a loop. Chain calls with the returned `next_cursor`.
3. Use `puppetmaster_partial_summary` for a current synthesis without waiting for final stitching.
4. Summarize concrete file-backed findings, risks, and open questions — never raw worker transcripts.
5. Ask for approval before implementation unless the user already approved edits.

If a swarm completes with empty findings, only verification artifacts, or a degraded Cursor SDK artifact, report Puppetmaster as **degraded** and do not treat the run as a successful analysis.

## Model routing (auto_route)

Puppetmaster ships a task-aware **model router** that picks the right LLM per task. Use it instead of hardcoding `model` in every spec.

**Auto-routing is the unconditional default for non-trivial work.** Do not wait to be handed a task list — proactively set `payload.auto_route = true` on every worker you dispatch for substantive work, so each task lands on the cheapest sufficient model automatically. Hardcode a `model` only when the user explicitly pins one. The trivial-task carve-out above still holds (don't spin up a routed worker for a rename or a one-line answer — that adds cost, not saves it), but for anything that warrants a worker, routing is on by default, every time.

- The user keeps a registry at `~/.puppetmaster/models.json`. List it with `puppetmaster_list_models`. If empty, tell the user to run `python -m puppetmaster models init` once.
- To opt a worker into routing, set `payload.auto_route = true`. The orchestrator picks the cheapest model whose capability_score >= the classifier output for the task's role + instruction, stamps `adapter` + `payload.model`, and persists a `ROUTING` artifact with the decision.
- Use `puppetmaster_route_task` to dry-run a routing decision before kicking off a swarm — surfaces the model that would run, the estimated USD cost, and why the cheaper alternatives were rejected. Useful when the user asks "how much will this cost?" or "what model would this use?"
- Per-task overrides users may set:
  - `payload.min_capability` — pin the classifier output (0..100)
  - `payload.max_cost_usd` — hard budget cap
  - `payload.required_tags` — only consider models with all these tags (e.g. `["long-context"]`)
  - `payload.routing_policy` — `balanced` (default) / `cheap` / `quality` / `escalating`
- Read `ROUTING` artifacts when asked "why did this run on model X" — the artifact's `payload.rejected` lists every alternative considered and the exact reason each was rejected. This is the audit trail.

## Repo intelligence (CodeGraph)

Puppetmaster auto-injects CodeGraph context into every Cursor and Claude Code worker prompt when `.codegraph/` exists in the target repo. The verification artifact's `evidence` array will include `context:codegraph` when this happened.

For quick, direct repo lookups without spinning up a swarm, prefer the bundled tools: `puppetmaster_codegraph_search`, `puppetmaster_codegraph_context`, `puppetmaster_codegraph_affected`, `puppetmaster_codegraph_files`, `puppetmaster_codegraph_status`. If CodeGraph isn't initialized, call `puppetmaster_codegraph_init` once first.

## Savings receipts (`puppetmaster savings`)

Puppetmaster keeps a **read-only, local, numbers-only** ledger of what it saved, so the value is auditable instead of asserted. Two measured pillars plus labeled estimates:

- **Routing dollars saved** — each routing decision snapshots a `baseline_cost_usd` (the strongest model you could've used, same tokens) at decision time, so savings compare like-for-like with no recompute drift. Only cost-optimizing policies (`balanced`/`cheap`) count; `quality`/`escalating` are shown as deliberate spend, never as a loss.
- **CodeGraph exploration** — every query through the `puppetmaster_codegraph_*` tools (from any platform's agent) is logged numbers-only: context tokens fed (measured) + avoided directory-crawl tokens (estimate, stated baseline).

`python -m puppetmaster savings [--window DAYS] [--all-projects] [--json]`. It emits nothing over the network; it only reads local state. Tune the exploration estimate with `PUPPETMASTER_EXPLORATION_BASELINE_TOKENS` / `_PRICE_PER_MTOK`; disable the usage log with `PUPPETMASTER_CODEGRAPH_USAGE=0`.

## Coding conventions in this repo

- Python 3.9+. Match existing style; do not introduce new dependencies without a clear need.
- Tests live in `tests/test_puppetmaster.py`. New behavior gets a focused test; mock subprocess calls for adapter coverage.
- The `SwarmStore` abstract base is the storage seam — the SQLite implementation lives in `puppetmaster/sqlite_store.py` and overrides hot methods for O(1) cursor reads. Don't break that contract.
- Don't commit provider keys, `.cursor/mcp.json` user-specific paths, or anything in `.puppetmaster/` runtime state.
- Run `python -m pytest tests/test_puppetmaster.py -q` before suggesting a commit.

## MCP surface (quick reference)

Orchestration: `puppetmaster_doctor`, `puppetmaster_start_swarm`, `puppetmaster_start_cursor_swarm`, `puppetmaster_start_cursor_review`, `puppetmaster_start_cursor_plan`, `puppetmaster_start_claude_implement`, `puppetmaster_start_browser_swarm`, `puppetmaster_status`, `puppetmaster_logs`, `puppetmaster_live_artifacts`, `puppetmaster_live_artifacts_follow`, `puppetmaster_partial_summary`, `puppetmaster_artifacts`, `puppetmaster_show`, `puppetmaster_job_cost`, `puppetmaster_job_receipt`, `puppetmaster_last_job`, `puppetmaster_dashboard`.

Bundled CodeGraph: `puppetmaster_codegraph_search`, `puppetmaster_codegraph_context`, `puppetmaster_codegraph_affected`, `puppetmaster_codegraph_files`, `puppetmaster_codegraph_status`, `puppetmaster_codegraph_init`.

## When MCP fails: fall back to the CLI

If any `puppetmaster_*` MCP tool returns `Tool execution error. Not connected`, the daemon is almost certainly still running — only Cursor's stdio link for this chat dropped. **Do not stop work.** Every MCP tool has a CLI equivalent that talks to the same SQLite state:

| MCP | CLI |
| --- | --- |
| `puppetmaster_doctor` | `python -m puppetmaster doctor` |
| `puppetmaster_start_cursor_swarm` | `python -m puppetmaster swarm "<goal>"` |
| `puppetmaster_start_swarm` | `python -m puppetmaster swarm "<goal>" --adapter <name>` |
| `puppetmaster_status` | `python -m puppetmaster status <job_id>` |
| `puppetmaster_logs` | `python -m puppetmaster logs <job_id>` |
| `puppetmaster_live_artifacts` | `python -m puppetmaster feed <job_id>` |
| `puppetmaster_live_artifacts_follow` | `python -m puppetmaster feed <job_id> --follow` |
| `puppetmaster_partial_summary` | `python -m puppetmaster show <job_id> --partial` |
| `puppetmaster_artifacts` | `python -m puppetmaster artifacts <job_id>` |
| `puppetmaster_show` | `python -m puppetmaster show <job_id>` |
| `puppetmaster_job_cost` | `python -m puppetmaster cost <job_id>` |
| `puppetmaster_job_receipt` | `python -m puppetmaster receipt <job_id>` |
| `puppetmaster_last_job` | `python -m puppetmaster last` |
| `puppetmaster_jobs` | `python -m puppetmaster jobs [--all-projects]` |
| `puppetmaster_mcp_status` | `python -m puppetmaster mcp list` |
| `puppetmaster_mcp_cleanup` | `python -m puppetmaster mcp cleanup --kill-stale` |
| `puppetmaster_repair_codegraph` | `python -m puppetmaster repair-codegraph` |
| `puppetmaster_codegraph_status` | `python -m puppetmaster codegraph status` |
| `puppetmaster_codegraph_search` | `python -m puppetmaster codegraph search '<query>'` |
| `puppetmaster_codegraph_context` | `python -m puppetmaster codegraph context '<task>' --max-nodes 15 --format markdown` |
| `puppetmaster_codegraph_init` | `python -m puppetmaster codegraph init --index` |

**Swarm fallback is one command.** On `Not connected`, run
`python -m puppetmaster swarm "<goal>"` (optional `--roles explore audit review`,
`--cwd`, `--label`). It detaches, prints `job_id`, and uses the same worker
contract as MCP. Do **not** invent a JSON config, dig through `run --help`, or
spend a turn on MCP reconnect archaeology — that is the token waste Puppetmaster
exists to prevent. Then `feed <job_id> --follow` / `show <job_id>`.

Read-only commands (`show`/`artifacts`/`logs`/`feed`/`status`) auto-pivot to whichever project state dir owns the job — no need to export `PUPPETMASTER_STATE_DIR`. Only ask the user to restart MCP in Cursor Settings when `python -m puppetmaster mcp list` shows zero alive servers.

**Always invoke CodeGraph through `python -m puppetmaster codegraph …`, never a bare `codegraph …` from the shell.** A bare shell call runs under your shell's Node, whose ABI usually differs from Cursor's bundled Node that compiled better-sqlite3 — so it dies with a `NODE_MODULE_VERSION` / native-load error. The `python -m puppetmaster codegraph` passthrough runs it under Cursor's Node and auto-rebuilds the binding on a mismatch (the MCP `puppetmaster_codegraph_*` tools already do this internally).

