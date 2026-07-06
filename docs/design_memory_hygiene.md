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

## After Wave 9 (roadmap)

- Marionette Round 8: consume L0-L3 memory-layer snapshots from Round 7
  (compaction advisor driven by layer pressure).
- Puppetmaster Wave 10 candidate: retrieval quality — scope-aware weighting so
  findings outrank verifications at injection time.
