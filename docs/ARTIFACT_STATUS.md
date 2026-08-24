# Artifact status vs confidence (#88)

One `confidence` float used to mix worker self-rating, adapter/process health,
gate bookkeeping, and implied evidence. v1.22.26 splits those meanings.

`confidence` remains on stored artifacts so old records and readers load.
It maps to `worker_self_rating` only. It is **not** an admission input and
is **not** a calibrated probability.

Do **not** invent `predicted_independent_gate_pass_probability` until a named
outcome has held-out calibration and published drift checks.

Marionette follows artifact JSON. This release is **additive**: new optional
fields plus the existing `confidence` key. Harness / SSE / session shapes are
unchanged. Do not pin or edit Marionette in this slice.

## Number vs explicit status

| Use | Needs a number? | What to use |
|---|---|---|
| Durable memory promotion | No | Independent support / `claim_support_status` / `criterion_status` |
| Shared gist admission | No | Same as memory. Self-rating cannot admit |
| Adapter/process health | No | `execution_status` (`completed` / `failed` / `degraded`) |
| Gate / verification outcome | No | `criterion_status` (`met` / `unmet`) from `result` / `passed` |
| Evidence presence | No | `grounding_status` (`cited` / `grounded` / `ungrounded`) |
| Worker self-assessment | Optional, non-authoritative | `worker_self_rating` (compat: `confidence`) |
| CLI / MCP / dashboard / effort-index | Status labels | Never `xx%` or a green bar for self-rating |
| Dedup / audit tie-break | Number OK as weak order | `confidence` / self-rating, documented as uncalibrated |
| Opt-in confidence escalation | Existing uncalibrated heuristic | Still `payload.min_confidence`; not admission; not a calibrated score |
| Future routing probability | Only after calibration | Named `predicted_independent_gate_pass_probability` — **not shipped** |

## Status vocabularies

- `execution_status`: `unknown` \| `running` \| `completed` \| `failed` \| `degraded`
- `grounding_status`: `unknown` \| `ungrounded` \| `cited` \| `grounded`
- `claim_support_status`: `unknown` \| `unsupported` \| `worker_asserted` \| `independently_supported`
- `criterion_status`: `unknown` \| `unmet` \| `met` \| `not_applicable`
- `worker_self_rating`: optional float in `[0, 1]`, non-authoritative

Worker JSON that sets `payload.claim_support_status` to `independently_supported`
(or `supported` / `verified`) is coerced to `worker_asserted`.

## Compat mapping

`artifact_from_dict` still requires `confidence`. Missing status fields are
inferred from type/payload (`VERIFICATION.result`, evidence present → `cited`).
The number is copied to `worker_self_rating` only. It never becomes
`independently_supported`.

## Inventory (PM)

### Producers

| Site | What it wrote | Now |
|---|---|---|
| Worker envelope (`adapters/_prompts.py`, agentic schema) | `confidence` on finding/risk/decision | Still accepted; stored as `confidence` + `worker_self_rating` |
| Adapter process paths (cursor, claude_code, codex, hermes, openai, agentic, antigravity, local, `_base`) | Hardcoded 0.9/0.55/0.65 from returncode / degrade | Still written to `confidence` (compat). Display uses `execution_status` inferred from verification `result` |
| `gates.py` | GATE artifact confidence 0.95/0.9 | Compat number; `criterion_status` from `passed` |
| `research.py` | Various research-loop scores | Compat number; not admission |
| `workers.py` | High self-rating demo artifacts | Compat number |
| `gist_admission.maybe_admit_finding_as_gist` | Copied finding confidence onto gist | Still copies for compat; **admission no longer keys off it** |

### Consumers (admission / authority)

| Site | Old | New |
|---|---|---|
| `gist_admission.maybe_admit_finding_as_gist` | `confidence >= 0.8` | Independent support only |
| `stitcher._promote_memories` | `confidence >= 0.8` | `durable_admission_allowed` |
| Shared-context gist filter | `admission=admitted` | Unchanged (admission is now harder to earn) |
| `orchestrator._reroute_low_confidence` | Opt-in numeric floor | Unchanged heuristic; not admission |
| `audit.py` mean confidence | Model audit bookkeeping | Unchanged; uncalibrated |
| `claim_conflicts` high-confidence filter | `confidence >= 0.8` | Still a weak peer filter (not admission) |
| Memory retrieval tie-break | confidence then recency | Unchanged weak order |

### Displays (PM side)

| Site | Old | New |
|---|---|---|
| Dashboard artifact tables | `confidence.toFixed(2)` | Status labels |
| Dashboard activity | `confidence: 0.95` | `status_label` |
| Dashboard swarm header | `VERIFICATION 80%` | `N passed / N unmet / N total` |
| CLI feed | `confidence=` | `execution_status` + `claim_support_status` |
| `effort-index` compact refs | `conf=` | Status fields; JSON still includes `confidence` for readers |
| MCP `artifacts --refs` | compact `confidence` | Additive statuses + legacy `confidence` |
| Stitched summary bullets | `confidence=0.95` | `format_status_label` |

Marionette may still render a green bar if it treats `confidence` as a
probability. PM does not change Mari. New fields are additive so Mari can
switch when it is ready.

## Calibration bar (future numerical score)

A later authoritative number must name its target (suggested:
`predicted_independent_gate_pass_probability`), publish labels, held-out
calibration, and drift checks. None of that ships in 1.22.26.

## Code

- `puppetmaster/artifact_status.py` — vocabularies, hydrate, admission helper
- Tests: `tests/test_artifact_status.py`, `tests/test_gist_admission.py`
