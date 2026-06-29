# Bring your own compression (input-side)

Short version: Puppetmaster does not bundle input-side context compressors
(RTK, Headroom, caveman, and similar), and it is not going to. This page
explains why, and shows how to wire one yourself if you want it. Doing it
yourself is the supported path on purpose.

> This is an unsupported, opt-in recipe. If you bolt a compressor onto your
> workers, you own its failure modes. Measure your own net savings and task
> accuracy before trusting it on real work.

## Why Puppetmaster doesn't own this

We evaluated it instead of guessing. The case against bundling is evidence, not
preference:

- **The net savings are tiny.** A replay over a real 614M-token / $926 corpus
  measured RTK + Headroom + caveman combined at **3.7% of total spend**. A live
  Hermes evaluation measured a lossless-only headroom path at **0.34% net** on
  ~3,000 real tool outputs. The advertised 60–95% numbers describe fewer tokens
  on specific redundant payloads, not a lower bill — the bill is dominated by
  cache writes and output tokens, which these tools don't touch.
- **The tail risk runs the wrong way for an agent.** Remove-and-retrieve
  schemes (e.g. Headroom's CCR) replace a blob with a marker and cache the
  original; a coding agent that is *using* that data retrieves it right back,
  putting it in context twice — more tokens, not fewer. Lossy text compression
  can also strip a stack-trace line or a path the agent needed, which triggers a
  re-read or a wrong turn. A small headline saving with a real chance of a
  larger total cost is a bad default.
- **Owning it makes us the blame surface.** A bundled "enable RTK" toggle turns
  every silent-degradation failure into a Puppetmaster bug report for a failure
  mode we didn't author, and adds a heavyweight dependency that can fight the
  prompt cache and our own summarization/compaction paths.

What we *do* own is the narrow, safe slice: **in-place lossless densification of
structured tool output** (rewriting a structured payload into fewer tokens with
zero information loss and no retrieve step). That's not a compression framework,
it's just good output formatting, and it's measured per case rather than turned
on blanket.

## Where the actual cost wins are

Before reaching for text compression, note that Puppetmaster already reduces
tokens *structurally*, which doesn't risk silent degradation:

- **Model routing** sends each task to the cheapest sufficient model.
- **CodeGraph** answers "where is X / what calls Y" from the symbol graph, so a
  worker never ingests a directory crawl in the first place.
- **Typed artifacts + `$0` recall** mean follow-ups are SQLite reads, not
  re-runs that re-send context.

If you also want to compress your own output prose, that's the supported,
in-tree feature — see [OUTPUT_STYLE.md](OUTPUT_STYLE.md).

## How to wire a compressor yourself

These tools are designed to be wired at the host/agent layer, not inside
Puppetmaster. Pick the integration that matches the tool:

- **RTK** (`github.com/rtk-ai/rtk`) — a CLI proxy for shell-command output. Wire
  it as a shell hook in your host agent (Cursor `.cursor/hooks.json`, Claude
  Code `.claude/settings.json`) so `git status`, `ls`, test runners, etc. are
  rewritten to their `rtk` equivalents before execution. It only affects shell
  output; built-in `Read`/`Grep`/`Glob` bypass it.
- **Headroom** (`github.com/chopratejas/headroom`) — runs as a proxy, library,
  middleware, or MCP server. Use its proxy/MCP mode at the host layer; keep its
  remove-and-retrieve (CCR) path off for coding work.
- **caveman** (`github.com/JuliusBrussee/caveman`) — an output-style skill. This
  overlaps with Puppetmaster's own [output style](OUTPUT_STYLE.md); prefer the
  in-tree feature so it composes with routing and the receipts.

General guidance if you do this:

- Keep a bypass switch wired (e.g. `RTK_DISABLE=1`) so a worker can recover the
  full, uncompressed output when it needs it.
- Prefer tools that fall back to raw output when a filter fails, and that are
  transparent about what they drop.
- **Measure net, not headline.** Run your real workload with the compressor on
  and off and compare total cost, latency, *and* task success — a cheaper prompt
  that fails the build is a loss.

## See also

- [OUTPUT_STYLE.md](OUTPUT_STYLE.md) — the supported, in-tree output-prose
  compression feature.
- [CLAIMS.md](CLAIMS.md) — how Puppetmaster measures its own cost numbers.
