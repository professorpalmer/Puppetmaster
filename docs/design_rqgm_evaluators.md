# Design: RQGM evaluator slots (v1)

## Problem

Puppetmaster already emits `VERIFICATION` artifacts and supports a `review`
completion gate (`puppetmaster/gates.py`), but evaluator criteria are ad hoc
per run. There is no durable evaluator slot, no version lineage, no frozen
epoch for a job, and no deterministic promotion battery. A swarm can therefore
change its implicit quality bar mid-flight or promote an evaluator on anecdotal
evidence.

This note defines the minimal kernel lifts inspired by
[The Red Queen Gödel Machine](https://arxiv.org/abs/2606.26294) — evaluators as
versioned citizens, not a full co-evolution research loop.

## Evaluator slot

An **evaluator slot** is a named, versioned spec stored on disk under
`{state_dir}/evaluators/registry.json`.

Fields (v1):

| Field | Type | Meaning |
|-------|------|---------|
| `slot_id` | str | Stable name (e.g. `test-verifier`, `redteam-reviewer`) |
| `version` | int | Monotonic per `slot_id`; never rewritten in place |
| `role` | str | Swarm role this evaluator covers (`test`, `redteam`, ...) |
| `instruction` | str | Human-readable criteria / prompt seed |
| `criteria` | object | Machine-readable thresholds (extensible dict) |
| `active` | bool | Whether this version is eligible for new epochs |
| `parent_version` | int or null | Lineage pointer for promotion |
| `promoted_at` | ISO str or null | When this version became active |

**Active resolution:** for each `slot_id`, the active spec is the highest
`version` where `active` is true. Registration appends a new row and
deactivates the prior active version for the same slot — history is append-only.

## Epoch freezing

When a job is created, the orchestrator snapshots the current active evaluator
set into a job-scoped `DECISION` artifact:

```json
{
  "kind": "evaluator_epoch",
  "decision": "freeze evaluator epoch at job start",
  "why": "RQGM epoch freezing v1",
  "evaluators": [
    {"slot_id": "test-verifier", "version": 1, "role": "test"}
  ]
}
```

Mid-job registry edits do **not** alter the stored epoch for that job. Only a
new job (or a future explicit epoch-advance API) picks up registry changes.
Lookup: `evaluator_epoch_for_job(store, job_id)` returns the latest epoch
payload or `{}`.

Hot path: snapshot failures are swallowed — job creation never blocks.

## Anchor sets

Promotion must not rely on a single lucky swarm. v1 uses a JSON **anchor
battery**: deterministic entries runnable through the **local** adapter only
(no LLM, no full swarm spawn).

Entry shape:

```json
{
  "id": "test-verifier-smoke",
  "goal": "verify artifact pipeline",
  "expect": {"min_verification_confidence": 0.8}
}
```

`run_anchor_battery` executes each anchor against the candidate slot's role,
scores pass/fail from returned artifact confidences, and returns
`pass_rate`. `promote_evaluator` runs the battery and registers a new active
version only when `pass_rate >= min_pass_rate`.

v1 does **not** use an LLM judge for promotion.

## Integration map

| Layer | Hook |
|-------|------|
| Registry | `puppetmaster/evaluators.py` — load/save/active/register/promote |
| Job start | `Orchestrator.run` / `run_crash_recovery_demo` after `create_job` — epoch snapshot artifact |
| Epoch read | `evaluator_epoch_for_job(store, job_id)` |
| Worker runtime | Before saving artifacts, stamp `evaluator_slot` + `evaluator_version` on `VERIFICATION` payloads when task `role` matches an epoch entry |
| Verification helper | `verification_artifact()` accepts optional slot/version kwargs |
| CLI | `python -m puppetmaster evaluators list|promote` |
| Review gate | Wave 7: read frozen epoch `criteria` when gate spec has no explicit `rubric` |
| Dashboard / CLI | Wave 7: `evaluator_epoch` snapshot field + `evaluators epoch <job_id>` |

## Wave 7: consuming the epoch

Wave 5 froze evaluator **lineage** (slot_id, version, role) at job start. That
was enough for artifact stamping but not for quality gates: the review gate
still judged diffs with a hardcoded rubric, and promoting a slot mid-registry
could change what a *new* job sees while old jobs stayed on stale metadata
only.

Wave 7 makes the epoch **full-fidelity** and **consumes** it:

### Full-fidelity epoch payloads

Each entry in the frozen `evaluators` list now carries:

```json
{
  "slot_id": "redteam-reviewer",
  "version": 2,
  "role": "review",
  "instruction": "Adversarially review for real failure modes.",
  "criteria": {"reject_partial": true, "min_confidence": 0.75}
}
```

Mid-job registry edits still do not alter a job's stored epoch. Wave 5 epochs
that lack `instruction`/`criteria` remain valid; callers treat missing
criteria as "no frozen rubric" and fall back.

### Review gate rubric resolution

When `_gate_review` runs, rubric text is chosen in strict order:

1. **Explicit gate spec** — `task.payload["review"]["rubric"]` or gate dict
   `rubric` wins (unchanged).
2. **Frozen epoch** — `evaluator_epoch_for_job(store, job_id)` then
   `epoch_evaluator_for_role(epoch, "review")`, falling back to the task's
   `role` when no `review` slot exists. Non-empty `criteria` render as one
   line per key (`- key: value`, sorted) after the frozen `instruction`.
3. **Default** — `_DEFAULT_REVIEW_RUBRIC` in `gates.py`.

Rubric *selection* is best-effort (lookup errors fall back to default). Gate
*execution* stays fail-closed. GATE artifact `detail` records
`rubric_source` (`spec` | `evaluator_epoch` | `default`) plus slot/version
when the epoch supplied the rubric.

### Lineage surfacing

- **Dashboard** — `build_job_snapshot` exposes `evaluator_epoch`: a list of
  `{slot_id, version, role}` (no criteria bodies). The job view renders compact
  `slot@vN (role)` chips alongside existing swarm metadata.
- **CLI** — `python -m puppetmaster evaluators epoch <job_id>` prints the
  frozen set (slot, version, role, criteria count) or
  `No evaluator epoch recorded.`

### Non-goals (Wave 7)

- Per-task evaluator overrides (epoch is job-scoped only).
- Automatic re-review when a slot is promoted mid-job (epoch freeze forbids it).
- Registry backends beyond JSON (Redis/Postgres deferred again).
- LLM-as-judge inside anchor promotion (stays deterministic).

## Non-goals (v1)

- Full RQGM co-evolution or genetic evaluator search
- LLM-as-judge promotion
- Marionette / harness changes
- New `ArtifactType` enum values (epoch uses `DECISION` + `payload.kind`)
- SQLite registry table (JSON file v1)
- YAML spec format
