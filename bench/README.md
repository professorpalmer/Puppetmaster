# Puppetmaster Benchmarks

Reproducible measurements you can run on any repo to validate the Puppetmaster + CodeGraph combo. Numbers in the main README come from running these harnesses locally — they're not synthetic.

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
