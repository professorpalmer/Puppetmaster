# CLI Reference

Every Puppetmaster MCP tool has a matching CLI subcommand. These are
the ones you'll use day-to-day from a shell; the full list is also
visible via `python -m puppetmaster --help`.

## Setup and inspection

```bash
python -m puppetmaster setup            # one-shot: doctor + models init + install-cursor-mcp + install-codex-mcp + install-claude-mcp + install-rules
python -m puppetmaster doctor           # 15 health checks
python -m puppetmaster adapters         # list available worker adapters (json)
python -m puppetmaster state            # print the resolved per-workspace state dir
python -m puppetmaster init             # create the local state store
python -m puppetmaster init-config --path puppetmaster.json
```

## MCP wiring (one-liner installers, v0.7.2+)

```bash
python -m puppetmaster install-cursor-mcp           # workspace .cursor/mcp.json
python -m puppetmaster install-cursor-mcp --global  # ~/.cursor/mcp.json
python -m puppetmaster install-codex-mcp            # codex mcp add ...
python -m puppetmaster install-claude-mcp           # claude mcp add --scope user ...
python -m puppetmaster install-codex-mcp --inherit-env OPENAI_API_KEY,CODEX_HOME
python -m puppetmaster install-codex-mcp --map-env CODEX_HOME=MY_CODEX_API_HOME
python -m puppetmaster install-codex-mcp --env-file ~/.config/puppetmaster/env.zsh
python -m puppetmaster install-rules                # write .cursor/rules/puppetmaster.mdc + AGENTS.md
python -m puppetmaster install-rules --global       # also ~/.codex/instructions.md and ~/.claude/CLAUDE.md
```

All four installers resolve `sys.executable`, run a `tools/list`
handshake before writing anything, are idempotent (re-run =
`unchanged`), and preserve existing user content in the target files.
For Codex MCP credentials, prefer a private env file (`chmod 600`) over
inline secrets in MCP JSON/TOML; env-file and secret-like inherited values are
loaded through a Puppetmaster-managed Python wrapper/private env file so values
are not printed or embedded in the wrapper. If your local variable name differs
from the provider's canonical variable, map it with `--map-env TARGET=SOURCE`.

## Running swarms

```bash
python -m puppetmaster run "Goal" --config examples/enterprise-workflow.json
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster cursor "Goal" --review --dry-run
python -m puppetmaster cursor "Goal" --implement           # full-edit: Cursor edits files, captures a PATCH (add --allow-dirty to skip the clean-tree guard)
python -m puppetmaster claude "Goal" --permission-mode acceptEdits
python -m puppetmaster codex "Goal"
python -m puppetmaster openai "Goal"
python -m puppetmaster crash-demo
```

## Routing / cost

```bash
python -m puppetmaster route "instruction" --role <role>     # dry-run, returns picked model + cost
python -m puppetmaster cost <job_id>                          # sum spend across all routing artifacts
python -m puppetmaster models init                            # write starter registry
python -m puppetmaster models list                            # show registered models
python -m puppetmaster models path                            # print resolved registry path
```

## Routing self-audit (recommend score changes from real history)

```bash
python -m puppetmaster audit                                 # per-model report + suggested score diff (dry-run)
python -m puppetmaster audit --window 7                       # only jobs from the last 7 days
python -m puppetmaster audit --json                           # machine-readable report + suggestions
python -m puppetmaster audit --apply                          # write the suggested score changes to models.json
```

Read-only by default. It aggregates the routing/escalation/verification artifacts already in your store and proposes a lower `capability_score` only for an **under-delivering** model (keeps getting escalated away from / finishing with low confidence) so harder work routes to a stronger model. A strong model doing trivial work is flagged `possibly-over-used` but never auto-adjusted — proving a cheaper model would've sufficed needs a counterfactual the audit doesn't run. The registry stays your assertion; nothing is written without `--apply`.

## Savings receipt (what Puppetmaster actually saved)

```bash
python -m puppetmaster savings                                # routing $ saved + CodeGraph savings (this project)
python -m puppetmaster savings --all-projects                 # aggregate across every workspace
python -m puppetmaster savings --window 30                    # last 30 days only
python -m puppetmaster savings --json                         # machine-readable
```

Read-only, local, **numbers-only** — emits nothing over the network. Two measured cost pillars: **routing dollars saved** (each decision snapshots a frontier `baseline_cost_usd` at decision time; only `balanced`/`cheap` count, `quality`/`escalating` are shown as deliberate spend) and **CodeGraph exploration** (context tokens fed, measured; avoided directory-crawl tokens, a clearly-labeled estimate). Plus two real-count lines that are never dollarized: **reliability** (tasks auto-recovered off a dead provider + tasks re-run for confidence, from existing fallback/escalation artifacts) and **$0 follow-up reads** (user-facing `show`/`artifacts`/`partial_summary`/`feed` reads served from durable state at zero model cost; recorded once per invocation at the command entry, never the store's internal reads, and the `feed --follow` long-poll is excluded so it can't inflate). Tune the codegraph estimate with `PUPPETMASTER_EXPLORATION_BASELINE_TOKENS` / `PUPPETMASTER_EXPLORATION_PRICE_PER_MTOK`; disable the codegraph usage log with `PUPPETMASTER_CODEGRAPH_USAGE=0` and the reads log with `PUPPETMASTER_READS_USAGE=0`.

For the **dollar headline** leadership tends to want, the report adds a `counterfactual` block: it prices the exact token volume you actually routed against a single reference model at metered API rates and subtracts what the work actually cost (`avoided_usd = naive_cost_usd - actual_cost_usd`). On a plan-billed setup the actual cost is ~$0, so the avoided figure ≈ the naive cost — a real, monotonically-growing number that answers *"what would this have cost if every task had run on `<reference>` at API rates?"* It is explicitly a **counterfactual, not cash off your bill**, and only as honest as the reference: pick a model leadership agrees you'd otherwise have used. The reference defaults to the highest-capability *priced* model in your registry; override with `PUPPETMASTER_COUNTERFACTUAL_MODEL=<model-id>`. If the chosen reference has no per-token price the figure is $0 (and the report says so rather than implying savings).

For org dashboards that want success metrics **without dollars**, the report also derives a `metrics` block of rates over explicit denominators (each rate is `null`, never a misleading `0`, when its denominator is empty; the `sample` sub-object exposes every denominator so small samples are visible): `capability_match_rate` (cost-optimizing tasks that ran a model *other than the strongest available* — discipline; judged by model identity so it stays meaningful for plan-billed shops where every model is $0), `escalation_rate` (tasks bumped a tier — calibration), `fallback_rate` (tasks failed over off a dead provider — reliability), `reuse_reads_per_job` ($0 result reads per job — leverage), and `context_tokens_per_job` (focused-context tokens fed per job — efficiency). Trend these over a `--window` rather than reading cumulative totals.

## Platform lock (restrict which platforms get used)

```bash
python -m puppetmaster platform status                        # show each platform on/off
python -m puppetmaster platform only cursor                   # lock to Cursor only (single-platform mode)
python -m puppetmaster platform enable claude-code codex      # turn platforms back on
python -m puppetmaster platform disable openai                # turn a platform off
python -m puppetmaster platform reset                         # clear the lock (all platforms on)
# ephemeral / CI override (wins over saved config):
PUPPETMASTER_ONLY_ADAPTERS=cursor python -m puppetmaster route "audit" --role audit
```

A disabled platform is never routed to, auto-discovered, or used for fallback. Default is everything-on. Lock state persists in `~/.puppetmaster/platform.json`.

## Inspection (read-only, auto-pivots across workspaces v0.5.4+)

```bash
python -m puppetmaster status <job_id>
python -m puppetmaster watch <job_id>
python -m puppetmaster events <job_id>
python -m puppetmaster feed [job_id] [--follow]
python -m puppetmaster artifacts <job_id>
python -m puppetmaster logs [job_id]
python -m puppetmaster open [job_id]
python -m puppetmaster last
python -m puppetmaster show <job_id>
python -m puppetmaster diff [job_id]
python -m puppetmaster jobs [--all-projects]
python -m puppetmaster projects
python -m puppetmaster memory
```

## Lifecycle

```bash
python -m puppetmaster rerun [job_id]
python -m puppetmaster approve <job_id-or-artifact-id>
python -m puppetmaster reject <job_id-or-artifact-id> --reason "why"
python -m puppetmaster clean --completed
python -m puppetmaster recover                # recover stale-leased tasks
python -m puppetmaster repair-codegraph       # rebuild CodeGraph's native SQLite binding for Cursor's bundled Node
```

## CodeGraph passthrough (ABI-safe)

```bash
python -m puppetmaster codegraph status                 # backend + index state
python -m puppetmaster codegraph search 'router'        # find symbols
python -m puppetmaster codegraph context 'add caching' --max-nodes 15 --format markdown
python -m puppetmaster codegraph init --index           # init + background index
python -m puppetmaster codegraph --cwd /path/to/repo files
python -m puppetmaster codegraph --timeout 60 search 'x' # cap the call at 60s
python -m puppetmaster codegraph -- --version            # pass codegraph's own flags after `--`
```

`--timeout SECONDS` caps the call; the default is **no timeout** so long ops
(`index`, `affected`) aren't cut off. Anything after a literal `--` is forwarded
verbatim to the CodeGraph CLI.

Use this instead of a bare `codegraph …` call from your shell. The passthrough
runs CodeGraph under **Cursor's bundled Node** (resolved via
`resolve_codegraph_invocation()`), not your shell's Node, so the native
`better-sqlite3` binding loads. If it still hits a Node-ABI mismatch
(`NODE_MODULE_VERSION` / "compiled against a different Node.js version"), it
**auto-rebuilds the binding against Cursor's Node once and retries** — the same
self-heal the `puppetmaster_codegraph_*` MCP tools now perform internally.
Disable the auto-rebuild with `PUPPETMASTER_CODEGRAPH_AUTOHEAL=0`. Everything
after the subcommand is forwarded verbatim to the codegraph CLI.

## MCP server management

```bash
python -m puppetmaster mcp list               # show every tracked MCP server PID
python -m puppetmaster mcp cleanup --kill-stale
```

## Workflow config schema

A workflow config is JSON describing the worker DAG for one swarm:

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

Examples (each ships in [`examples/`](../examples/)):

- [Enterprise Workflow](../examples/enterprise-workflow.json)
- [Cursor Live](../examples/cursor-live.json)
- [Cursor Review](../examples/cursor-review.json)
- [Cursor Dry-Run Implementation](../examples/cursor-dry-run-implementation.json)
- [Claude Code Full Edit](../examples/claude-code-full-edit.json)
- [Memory Reuse](../examples/memory-reuse.json)

## Daemon mode

For local swarms, keep Puppetmaster workers warm and let jobs hand
off work to them:

```bash
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster run "Review this repo" --worker-mode daemon
```

Daemon mode keeps the Puppetmaster worker loop alive across jobs.
Lease-based task claiming and artifacts are preserved; only the
worker process startup cost is amortized.

## Clean worktree for real edits

```bash
git worktree add /tmp/puppetmaster-work -b puppetmaster-work
python -m puppetmaster claude "Implement the approved fix" --cwd /tmp/puppetmaster-work --permission-mode acceptEdits
```
