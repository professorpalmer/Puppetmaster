# Design: promoted-memory hygiene (Wave 9)

## Problem: prompt-echo defect

Live job dispatch today injects promoted "memories" that are mostly worker role
prompts echoed back verbatim — for example
`Role: token-efficiency-reviewer\nGoal: ... Return only Puppetmaster artifact
JSON with an artifacts array.` — plus stale ship plans from jobs that already
executed. This wastes tokens on every dispatch and can mislead workers (a stale
plan says "push main and wait for CI").

Four root causes, all confirmed in code:

1. **Verification check = instruction.** Every adapter failure/timeout path
   stamps `check=task.instruction` on its `VERIFICATION` artifact
   (`adapters/_base.py` `verification_artifact` and each adapter's error
   paths). `Stitcher._statement_for` uses `payload["check"]` as the memory
   statement, so the whole worker prompt becomes a "memory statement".
2. **No promotion quality bar.** `Stitcher._promote_memories` promotes any
   artifact with `confidence >= 0.8` and a non-empty statement. No content
   filter.
3. **Unbounded store growth.** `FileStore.promote_memory` appends forever: no
   dedupe against existing promoted memory, no cap, no age-based expiry, no
   prune verb.
4. **Naive retrieval scoring.** `retrieve_memory` scores by term overlap, so
   long prompt-echo statements match almost any goal and crowd out real
   findings (`limit=5`).

## Promotion quality gate (Task B)

Before a worker artifact becomes promoted memory, the stitcher applies a
**promotion quality gate**:

- `_is_instruction_echo(statement, artifact)` rejects VERIFICATION statements
  that structurally match worker boilerplate: `Role:` prefix, known prompt
  markers (`Return only Puppetmaster artifact JSON`, etc.), or length over 600
  characters.
- VERIFICATION artifacts whose `result` is `failed`, `blocked`, or `degraded`
  are skipped — failed checks are job telemetry, not reusable knowledge.
- FINDING and DECISION promotion is unchanged.

Marker list lives in `_PROMPT_ECHO_MARKERS` so tests can extend it without
touching call sites.

## Store-side dedupe, cap, and expiry (Task C)

The memory directory must stop growing without bound and stale records must age
out of retrieval.

| Mechanism | Behavior |
|-----------|----------|
| **Dedupe** | `promote_memory` skips the write when the same `scope` and
  whitespace-normalized `statement` already exists (silent no-op). |
| **Cap** | After each write, enforce 200 records total; delete oldest by
  `created_at` until at cap. |
| **Age filter** | `retrieve_memory(..., max_age_days=N)` excludes records older
  than N days. Malformed `created_at` values are treated as fresh (never raise). |
| **Prune** | `prune_memory(scope=..., older_than_days=...)` deletes matching
  records and returns the count. No filters deletes all promoted memory. |

The orchestrator calls `retrieve_memory` with `max_age_days=14` by default
(`_MEMORY_MAX_AGE_DAYS`), overridable via `PUPPETMASTER_MEMORY_MAX_AGE_DAYS`
(0 or negative disables the age filter).

Both file and SQLite backends mirror these semantics.

## Memory CLI surface (Task D)

Inspect and clean promoted memory without hand-deleting JSON files:

```text
python -m puppetmaster memory              # human-readable, grouped by scope
python -m puppetmaster memory --json       # full JSON dump (backward compatible)
python -m puppetmaster memory --prune --scope swarm.findings
python -m puppetmaster memory --prune --older-than-days 30
python -m puppetmaster memory --prune --yes   # delete all (requires --yes)
```

`--state-dir` is global only (Wave 7 CLI pattern); the dispatch layer threads
the resolved `state_dir` through — subcommands do not re-parse it.

Unfiltered `--prune` without `--yes` refuses with a clear message and exit code
2. Empty store prints `No promoted memory.`

## Non-goals

- No embedding/vector retrieval.
- No LLM summarization of memories.
- No schema change to `MemoryRecord` beyond additive optional fields.
- No Redis/Postgres backend work in this wave.
- No changes to `_FRESH_JUDGMENT_ROLES` or memory-injection enablement.
- No evaluator registry, drafts, gates, or anchor changes from Waves 5-8.

## Wave 10: retrieval ranking

Wave 9 stopped garbage from entering promoted memory. Wave 10 makes retrieval
smart about what it **injects**. Before this wave, `retrieve_memory` ranked by
naive term overlap with confidence as the only tiebreak. Three defects:

1. **All scopes rank equally.** A `swarm.verification` record ("check X passed")
   outranks a `swarm.findings` insight whenever it shares more words with the
   goal — but verifications are job telemetry; findings and decisions are the
   reusable knowledge.
2. **Long statements win mechanically.** More words in the haystack means more
   term hits. Score was unnormalized by statement length.
3. **Recency is ignored inside the age window.** Wave 9's `max_age_days` filter
   drops records older than the window, but within the window a six-month-old
   decision ties with yesterday's on the same overlap.

### Weighted score formula (Task B)

Deterministic arithmetic only — no embeddings, no LLM calls.

| Factor | Rule |
|--------|------|
| **Scope weight** | `_SCOPE_WEIGHTS`: `swarm.findings` 1.0, `swarm.decisions` 1.0, `swarm.general` 0.7, `swarm.verification` 0.4; unknown scopes 0.7. |
| **Overlap** | `hits / max(1, len(query_terms))` — fraction of query terms matched in the haystack (`scope`, `statement`, `evidence`, `adapter`, `role`, `topic`); terms must be length > 2 (unchanged). |
| **Recency** | 1.0 when record age <= 7 days (`_RECENCY_FULL_DAYS`); linear decay to 0.5 at 56 days (`_RECENCY_FLOOR_DAYS`, `_RECENCY_FLOOR`); flat 0.5 beyond. Malformed `created_at` counts as fresh (same defensive parsing as Wave 9). |
| **Final score** | `overlap * scope_weight * recency` when the query has terms; `scope_weight * recency` when it does not. |
| **Tiebreak** | confidence descending, then `created_at` descending (newest first). |

Empty-query contract preserved: when the query yields no terms, return records
(up to `limit`) ranked by `scope_weight * recency` and confidence, including
zero-overlap records — the existing `if score > 0 or not terms` guard is unchanged.

### Injection floor (Task C)

The orchestrator passes `min_overlap=0.2` (`_MEMORY_MIN_OVERLAP`) to
`retrieve_memory`. Records whose overlap fraction falls below the floor are
excluded before ranking. Empty-term queries are exempt so the empty-query
contract holds. Override via `PUPPETMASTER_MEMORY_MIN_OVERLAP`; invalid values
fall back to the constant; negative values disable the floor.

### Non-goals (Wave 10)

- No embedding or vector search.
- No LLM reranking.
- No change to the retrieval `limit`, promotion rules, or Wave 9 hygiene.
- No new memory schema fields.
