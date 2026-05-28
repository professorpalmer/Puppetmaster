# CodeGraph integration

[CodeGraph](https://github.com/colbymchenry/codegraph) is a separate
project — credit goes to its author. The graphing capability that
shows up in Puppetmaster marketing is CodeGraph's, not
Puppetmaster's. Puppetmaster's contribution is what happens *after*
CodeGraph is installed:

- Every Cursor / Claude / OpenAI / Codex worker auto-injects
  task-relevant CodeGraph context into its prompt before the model
  call — no MCP round-trip per worker.
- One shared `codegraph context` query seeds N parallel workers in a
  swarm (vs N separate queries if each agent issues its own).
- The resulting artifacts (which now reference symbol-level evidence
  from CodeGraph) land in the same durable store, so follow-ups
  still cost zero tokens.
- The most-used CodeGraph CLI verbs are bundled directly into
  Puppetmaster's MCP — see [Bundled CodeGraph tools](#bundled-codegraph-tools-no-second-mcp)
  below — so Cursor Agent only needs one MCP for both orchestration
  and symbol intelligence.

The two tools optimize different axes and stack cleanly:

- **CodeGraph** = per-call context resolution. Static facts about
  your *code* (symbols, refs, routes).
- **Puppetmaster** = per-session coordination + state. Dynamic facts
  about the *agents' work* (tasks, leases, typed artifacts,
  replayable events).

## Install CodeGraph (optional)

```bash
npm install -g @colbymchenry/codegraph
cd your-target-repo
codegraph init -i
```

After that, Puppetmaster's `doctor` will show `codegraph ok`, and
every Cursor/Claude worker run against that workspace will:

- query CodeGraph for task-relevant symbols, files, and routes
- inject the result into the worker prompt as authoritative starting
  context
- tag the resulting verification artifact with `context:codegraph`
  so the operator can confirm shared intelligence was used

Fully optional and graceful. If `codegraph` is not installed, or the
target repo is not initialized, workers fall back to their normal
exploration path with no error. Pass `disable_codegraph: true` in a
task payload to skip CodeGraph for a specific worker.

## Bundled CodeGraph tools (no second MCP)

Puppetmaster's MCP server bundles the most useful CodeGraph CLI
commands so Cursor Agent only needs the Puppetmaster MCP to get both
orchestration and repo intelligence:

| Tool | Wraps | Use for |
|---|---|---|
| `puppetmaster_codegraph_search` | `codegraph query` | Find symbols by name (`{query, kind?, limit?}`) |
| `puppetmaster_codegraph_context` | `codegraph context` | Pull task-relevant entry points and related symbols (`{task}`) |
| `puppetmaster_codegraph_affected` | `codegraph affected` | Resolve impacted tests from changed source files (`{files[]}`) |
| `puppetmaster_codegraph_files` | `codegraph files` | Inspect the indexed file structure without scanning the FS |
| `puppetmaster_codegraph_status` | `codegraph status` | Check index health and backend |
| `puppetmaster_codegraph_init` | `codegraph init` | Initialize CodeGraph in a workspace (`{index?: true}` to also build immediately) |

Every tool degrades cleanly: if the `codegraph` CLI is not installed
or the workspace is not initialized, the response is a non-fatal
`isError: true` payload with `error` set to a one-line fix-it hint,
not a runtime crash.

Power users who want CodeGraph's full MCP surface
(`codegraph_callers`, `codegraph_callees`, `codegraph_impact`,
`codegraph_node`) — only available through its own MCP server — can
still run `codegraph serve --mcp` alongside Puppetmaster's MCP.
Bundling covers the daily-driver case so two MCP entries are no
longer required by default.

## Cost: what changes when you switch to durable state

> **Newer, more direct receipts** for the routing + durable-state
> claims live in [`bench/router_savings.py`](../bench/router_savings.py),
> [`bench/router_live_ab.py`](../bench/router_live_ab.py), and
> [`bench/followup_cost.py`](../bench/followup_cost.py) — see the
> [README opening section](../README.md#three-claims-three-receipts)
> and [TALKING_POINTS.md](../TALKING_POINTS.md). The Agent /
> CodeGraph / Puppetmaster three-way analysis below is older and
> broader (it models multi-worker swarm cost vs single-agent cost);
> both views are valid and they answer different questions.

The whole point of Puppetmaster is that durable state turns repeated
questions about the same task into a database read instead of
another agent run. The benchmark below shows that effect against two
baselines:

- **A. Agent only** — one agent (Cursor or Claude Code) doing the
  work alone, discovering the repo with grep/read/list. No shared
  state across sessions.
- **B. CodeGraph alone** — same agent, with CodeGraph's MCP
  installed; the agent issues `codegraph_explore` calls itself. Still
  no shared state across sessions.
- **C. Puppetmaster + CodeGraph** — Puppetmaster swarm with
  CodeGraph context pre-injected into every worker prompt, structured
  artifacts in a durable SQLite store, stitcher reads JSON not
  transcripts. Follow-up queries read SQLite, not the model.

Result, modelled from real measurements on this repo
(`bench/three_way.py`, swarm of 4 workers, artifact sizes from a real
past Puppetmaster run, $3/1M token input price):

### Fresh task cost (one investigation)

| Config | Tokens | Cost |
|---|---|---|
| A. Agent only | ~30,695 | ~$0.0921 |
| B. CodeGraph alone | ~6,250 | ~$0.0187 |
| C. Puppetmaster + CodeGraph | ~21,231 | ~$0.0637 |

**On a single fresh task, Puppetmaster does not beat CodeGraph alone
in raw tokens.** Puppetmaster is doing more work — N parallel workers
and a stitcher pass instead of one agent — so its token bill is
higher than a single agent with CodeGraph. That's an honest,
measured trade-off, and you should know it before believing any "99%
reduction" copy.

### Session cost (1 swarm + K follow-up reads at $3/1M)

This is where Puppetmaster actually wins. Real workflows are not
one-shot: you investigate, then ask follow-up questions about the
same task. In Configs A and B every follow-up is a fresh agent
re-run (no persisted state). In Config C, every follow-up is just
SQLite — **0 model tokens.**

| Config | K=0 | K=1 | K=5 | K=10 | K=25 |
|---|---|---|---|---|---|
| A. Agent only | ~$0.0921 | ~$0.1842 | ~$0.5525 | ~$1.0129 | ~$2.3942 |
| B. CodeGraph alone | ~$0.0187 | ~$0.0375 | ~$0.1125 | ~$0.2062 | ~$0.4875 |
| **C. Puppetmaster + CodeGraph** | **~$0.0637** | **~$0.0637** | **~$0.0637** | **~$0.0637** | **~$0.0637** |

At K=25 follow-ups, **Puppetmaster + CodeGraph is ~7.6× cheaper than
CodeGraph alone and ~38× cheaper than agent-only.** The crossover
where C catches up to B is around K=3-4 in this dataset.

### Where the savings come from

1. **Durable resume (Puppetmaster)** — the headline. Every follow-up
   read against a completed swarm is a SQLite query, costing 0 model
   tokens. This is what flatlines the C column above.
2. **Typed-artifact coordination (Puppetmaster)** — workers
   communicate through structured rows instead of raw transcripts;
   the stitcher reads JSON, not stdout.
3. **Amortized context query (CodeGraph + Puppetmaster)** — one
   `codegraph context` call seeds N workers in a swarm; B issues N
   separate `codegraph_explore` calls.
4. **Zero tool-call frames (CodeGraph + Puppetmaster)** — workers
   receive context inline in the initial prompt; no MCP round-trip
   envelope per worker.

The first two are Puppetmaster's standalone contribution and work
even without CodeGraph (you'd just lose the cheap per-call context,
so worker prompts get more expensive). The last two only show up
when both are installed.

### Reproduce on your own repo

```bash
npm install -g @colbymchenry/codegraph && codegraph init && codegraph index

# Three-way cost-structure benchmark
python -m bench.three_way --cwd . --workers 4 --artifacts-state /path/to/past/puppetmaster/state

# Just CodeGraph's prompt enrichment (A/B, no API key required)
python -m bench.codegraph_ab --cwd . --prompt @bench/prompts/example.txt --dry-run
```

See [`bench/README.md`](../bench/README.md) for full methodology, what's
measured vs. modelled, and the honest caveats (no live token billing
yet — that's on the roadmap and needs SDK-side stream instrumentation).
