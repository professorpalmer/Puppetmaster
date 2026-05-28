# Puppetmaster Benchmarks

Reproducible measurements you can run on any repo to validate Puppetmaster's marketing claims. Numbers cited in the main README and in [`TALKING_POINTS.md`](../TALKING_POINTS.md) come from running these harnesses locally — they're not synthetic.

Five scripts, five different questions:

| Script | Question it answers | API key? |
|---|---|---|
| `router_savings.py` | Does the v0.6.0 router actually pick cheaper models per task? | No (uses router math) |
| `router_live_ab.py` | If I pin a frontier model vs let Puppetmaster route, what's the **real** delta in `tokens_in/out` + wall-clock? | Yes (`OPENAI_API_KEY`) |
| `followup_cost.py` | After a swarm completes, do follow-up questions really cost 0 tokens? | No (reads SQLite) |
| `three_way.py` | What does Puppetmaster add on top of CodeGraph? (Agent-only vs CodeGraph alone vs Puppetmaster + CodeGraph) | No |
| `codegraph_ab.py` | What does CodeGraph add to a Cursor SDK worker? | Optional (`CURSOR_API_KEY` for live A/B) |
| `mcp_stress.py` | Does the MCP server hold up under concurrent load? | No (live MCP) |

If you only run one, run `router_live_ab.py` — it produces a real OpenAI billing receipt for under three cents of spend, and is the only one of the six that produces non-estimated dollar numbers.

## `router_savings.py` — does routing actually save money?

Dry-run benchmark. Runs the v0.6.0 router over a fixed 6-task fixture spanning easy / medium / hard tiers, then computes "Puppetmaster pick cost" vs "always frontier baseline cost" using the registry's pricing.

```bash
python -m bench.router_savings
```

What it proves: on a mixed-task workload, the router picks a cheaper-than-frontier model on routable tasks and correctly stays on the frontier model for hard tasks. Output is markdown + JSON under `bench/results/router_savings_<timestamp>.{md,json}`.

What it does **not** prove: any individual token count. The cost numbers come from the same heuristic estimates that drive routing itself, so this benchmark is internally consistent with the ROUTING artifacts your real swarms emit — but the dollars are directional, not billing-grade.

## `router_live_ab.py` — live OpenAI A/B with real token receipts

The only harness that gives billing-grade numbers. Runs the **same instruction** twice through `OpenAIAdapter`:

- **Arm A**: pinned to the strongest OpenAI model in your registry (`gpt-5.5` in the starter registry).
- **Arm B**: model picked by the Puppetmaster router under `balanced` policy, candidate pool constrained to `adapter=openai` so the A/B is apples-to-apples.

Both `tokens_in` and `tokens_out` are pulled from the live `usage.prompt_tokens` / `usage.completion_tokens` fields in the OpenAI API response — not estimated.

```bash
# Dry-run — show routing decision, no API call
python -m bench.router_live_ab --dry-run

# Live — real API call, ~2 cents of real spend at the default prompt
OPENAI_API_KEY=sk-... python -m bench.router_live_ab
```

Sample receipt (an actual run from the starter registry):

| Arm | Model | Wall (s) | tokens_in | tokens_out | $ (real) |
|---|---|---:|---:|---:|---:|
| A (always frontier) | `gpt-5.5` | 14.957 | 156 | 625 | $0.019530 |
| B (Puppetmaster) | `gpt-5.4-nano` | 2.491 | 156 | 131 | $0.000141 |

Delta: **99.3% cheaper, 83.3% faster**, equivalent finding artifacts.

What it does **not** prove: output *quality* is not graded automatically (the receipt prints both replies for human inspection). For trivial definitional tasks the nano model is sufficient; for complex tasks the router would pick a stronger model and the savings would be smaller.

## `followup_cost.py` — do follow-up reads really cost 0 tokens?

Runs against your most recently completed Puppetmaster job (or pass `--job-id`). Performs `--queries` × 4 different reads against the SQLite store — `get_job`, `list_artifacts`, `filter by finding`, `filter by verification` — and measures wall time per query.

```bash
python -m bench.followup_cost --queries 10
```

What it proves: every follow-up answer comes out of the durable state without invoking any adapter. The receipt reports `adapter_calls=0, tokens=0, cost=$0.00`, plus a hypothetical "if every follow-up was a fresh swarm replay" baseline column for context.

What it does **not** prove: that ALL follow-ups in a real workflow avoid LLM calls. If the user's follow-up needs reasoning the original swarm didn't capture, a new task has to run — and that costs tokens like any other.

## `three_way.py` — Agent vs CodeGraph alone vs Puppetmaster + CodeGraph

## `three_way.py` — Agent vs CodeGraph alone vs Puppetmaster + CodeGraph

A cost-structure model that maps three different deployment configurations onto the same investigation task, then reports bytes / tokens / dollars per fresh task and per session.

### Run it

```bash
# Default: 4 workers, $3/1M tokens, uses repo-local data only
python -m bench.three_way --cwd .

# With real past artifact data from one of your previous swarms:
python -m bench.three_way \
    --cwd . \
    --workers 4 \
    --artifacts-state ~/Library/Application\ Support/puppetmaster/projects/your-project
```

Output (markdown + JSON) lands in `bench/results/three_way_<timestamp>.{md,json}`.

### What it actually models

| Config | Per-worker prompt | Tool-call frames | Synthesis | Resume |
|---|---|---|---|---|
| A. Agent only | discovery_scan (~10% of repo) + agent output | many (grep/read/list) | agent self-summary | full re-run |
| B. CodeGraph alone | one `codegraph_explore` result + agent output | one MCP frame per worker | agent self-summary | full re-run |
| C. Puppetmaster + CodeGraph | inlined CodeGraph context + worker artifact output | **zero** (context is pre-injected) | stitcher reads structured artifacts | **read SQLite, 0 model tokens** |

The headline finding (on this repo): **for a single fresh task, Puppetmaster ≈ CodeGraph alone in raw tokens**. The real wins are on the **session axis** — every follow-up read against a Puppetmaster job is free at the model level, so cost flatlines while A and B keep paying per re-run.

### What's measured vs. what's modelled

Measured:

- Repo file count and total source bytes (walks the target).
- CodeGraph context size and query latency (real `codegraph context` call).
- Avg Puppetmaster artifact size by type, avg worker stdout size, avg artifacts per worker (real, from past `state.sqlite3` if `--artifacts-state` is supplied).

Modelled with stated constants (override with flags):

- Tool-call frame size (`~250 bytes`).
- Typical agent self-output per task (`~2,000 bytes`).
- Stitcher output (`~1,500 bytes`).
- Agent-only discovery scan ratio (`10%` of repo).
- Bytes → tokens (`÷ 4`).
- $ per 1M tokens (`--price-per-million`, default `$3.00`).

We do **not** capture exact per-task token billing from a live agent stream. That requires SDK-side instrumentation (on the roadmap).

## `codegraph_ab.py` — does CodeGraph actually save the agent work?

A/B benchmark that runs the **same Cursor SDK task twice** through Puppetmaster's `CursorAdapter`:

- **A**: with `payload.disable_codegraph=true` (raw prompt).
- **B**: with CodeGraph enabled (auto-injected entry points and related symbols).

Then compares wall-clock seconds, structured artifact yield, and stdout size. There is also a `--dry-run` mode that measures just the **prompt enrichment** — how many characters of pre-resolved context CodeGraph supplies for how much wall time — without spending any model tokens. That's the cheapest defensible signal that the integration is doing real work.

### Run the dry-run (no API key required)

```bash
# install + index the target repo for CodeGraph (one-time)
npm install -g @colbymchenry/codegraph
codegraph init && codegraph index

# run the dry-run benchmark
python -m bench.codegraph_ab --cwd . --prompt @bench/prompts/example.txt --dry-run
```

Output (markdown + JSON) is written to `bench/results/<timestamp>.{md,json}`.

### Run the live A/B (requires `CURSOR_API_KEY`)

```bash
export CURSOR_API_KEY=...
python -m bench.codegraph_ab --cwd . --prompt @bench/prompts/example.txt
```

The script will invoke the Cursor SDK twice — once with CodeGraph off, once with it on — and add a live Cursor SDK A/B table to the report.

### What we measure (and don't)

| Metric | Where it comes from | What it means |
|---|---|---|
| `raw_prompt_chars` | input prompt | baseline prompt size |
| `injected_context_chars` | `codegraph context <task>` | pre-resolved entry points / related symbols added to the worker prompt |
| `injection_ratio` | ratio of the two | how much "free" structured context the agent starts with |
| `codegraph_query_seconds` | wall clock around `codegraph context` | one-time cost to seed the agent |
| `arm.wall_seconds` | wall clock around `CursorAdapter.run` | total worker time per arm |
| `arm.structured_artifact_count` | non-verification artifacts emitted | useful findings/risks/decisions produced |
| `arm.stdout_chars` | Cursor SDK result body | proxy for the work the agent did to answer |
| `arm.codegraph_used` | adapter's `context:codegraph` evidence | sanity check that injection actually happened |

We do **not** currently capture exact model token counts. That requires SDK-side stream instrumentation, which is on the roadmap. Until then, treat the numbers as a directional A/B, not a billing receipt.

## Adding your own benchmark

Pattern to follow when adding a new benchmark:

1. Drop a single-file script in `bench/` that imports from `puppetmaster.*`.
2. Make it write `bench/results/<timestamp>.md` + `.json`.
3. Add a small `test_bench_*` block to `tests/test_puppetmaster.py` that exercises the dry-run path (no live API key dependency).
4. Link it from this README with a one-paragraph "what we measure / don't measure" section.
