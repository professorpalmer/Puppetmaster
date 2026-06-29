# Output style (Signal-maximizer)

An optional directive that tells workers to write tighter. Off by default.

It constrains **form, not reasoning**. The model still thinks as fully as it
needs to; only the emitted prose is compressed. So it does not lower answer
quality the way input/context compression can — there is no information removed
from what the worker *reads*, only from how it *writes*.

Be honest about the payoff: output tokens are a minority of an agentic bill
(turns, tool output, and prompt cache dominate), so the real win is **readability
and latency**, with a modest cost bonus on output-heavy roles. Treat it as a
style feature, not a cost lever.

## Enabling it

Globally, via env:

```bash
export PUPPETMASTER_OUTPUT_STYLE=terse   # or: lithic
```

Per task, via the spec payload (wins over the env for that one spec):

```json
{ "output_style": "terse" }
```

Precedence mirrors the skill/memory opt-in: an explicit `payload.output_style`
overrides the env, and a disabled value (`"off"`, `"none"`, `""`) opts a single
spec out even when the env default is on.

## The two tiers

### `terse` (safe; recommended when on)

- Start with the answer. No greetings, sign-offs, preambles, or postambles.
- No self-reference ("I'd be happy to", "let me", "as an AI").
- No filler ("basically", "essentially", "in order to", "it's worth noting").
- No transition padding ("furthermore", "additionally", "that said"). Sequence
  facts directly.
- Do not restate the question. No closing summary of what was just said.
- One claim per line. Never repeat a claim.
- Cut any word removable without changing meaning.
- Keep every fact, number, name, path, condition, and caveat that changes
  meaning. Never drop signal to save space.
- State genuine uncertainty as fact ("unconfirmed: X"). Never express false
  confidence and never pad with empty hedges.

That last rule is deliberately one line. Banning hedges and demanding honest
uncertainty are the same instruction — kept together so "no hedging" can never
be read as "sound confident when you aren't."

### `lithic` (aggressive; opt-in)

Everything in `terse`, plus: drop articles, copulas, and grammatical glue
wherever meaning survives (telegraphic style), while keeping code, identifiers,
paths, and quoted strings byte-exact.

The extra savings are marginal and the prose reads worse, so reserve `lithic`
for machine-consumed worker artifacts. Don't put a human-facing stitched summary
in `lithic`.

## What it does not do

- It does not compress what a worker reads (tool output, files, context). For
  why Puppetmaster does not bundle input-side compression, see
  [COMPRESSION.md](COMPRESSION.md).
- It does not touch reasoning/thinking tokens.
- It does not change routing. The directive is a fixed ~100-token block applied
  equally to every candidate model, so it does not bias which model is picked;
  `estimated_tokens_in` is intentionally left untouched.

## Implementation

`puppetmaster/output_style.py` holds the directive text and the resolver. The
orchestrator applies it in a single adapter-agnostic seam (`_with_output_style`,
mirroring memory/skill injection) by prepending the directive to each worker's
instruction before tasks are created. Every adapter funnels through
`payload["prompt"] or instruction`, so one seam covers all of them.
