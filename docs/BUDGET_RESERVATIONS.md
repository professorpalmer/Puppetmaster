# Runtime budget reservations

`Job.budget_policy: Optional[BudgetPolicy]` is additive. Both store creation APIs
accept `budget_policy=`; JSON without it remains unbudgeted. Launch-key replay
with a different policy is a conflict. `payload.max_cost_usd` remains the router's
per-task marginal-cost estimate filter. WorkerRuntime now consults these APIs
through the shared invocation boundary before dispatching external work.

`puppetmaster.budget.BudgetPolicy` supports separate `max_usd`, `max_tokens_in`,
`max_tokens_out`, `max_attempts`, and `max_elapsed_seconds`. Seconds mean summed
invocation elapsed time, not wall time since job creation. Four parallel workers
with `timeout_seconds` 180 thus need a cap of 720, not 180 or 240. While an
earlier attempt has no settled elapsed time, the job total is indeterminate.

Admission has three outcomes:

| Case | Error | Result |
| --- | --- | --- |
| The total is known and within each cap | none | The attempt is admitted. |
| Only earlier attempts that have not settled make a total unknown | `BudgetUnsettled` | The attempt waits and tries again each second. It fails after `payload.budget_admission_wait_seconds` (default 900), on a lost lease, or on a cut. |
| A cap is exhausted, or an unknown can never settle (no allowance, or a settled unknown) | `BudgetAdmissionError` | The attempt fails at once. |

Thus a job under this cap runs its attempts one at a time, and its parallel
workers queue instead of failing. A wait holds no lock: each try takes the
dispatch lock, and reserve, adopt and the pending mark stay one atomic step,
so a sibling never counts a reservation beside an unsettled attempt. A child
enqueue is not refused for an unsettled sibling, because its dispatch waits.
Use a wall-clock timeout per worker, not this cap, to bound how long parallel
work takes. Token counts are the
invocation's input/output counts; cache counters are not added again. All limits
are optional, finite and nonnegative. An empty policy imposes no caps. Configure
a policy before dispatch; changing policy through ordinary job writes is not a
concurrent policy-update protocol.

`SwarmStore` and `SQLiteSwarmStore` expose the same contract:

- `reserve_dispatch(ExecutionAttempt, BudgetLiability)` admits one immutable
  invocation identity and allowance. The attempt is only embedded in the
  reservation; this does not claim execution occurred in the consumption ledger.
  An already recorded invocation cannot receive a new reservation retroactively.
- `adopt_dispatch(job_id, attempt_id, adoption_id=...)` durably moves reserved to
  dispatching. The adoption identity fences conflicting claimants. Exact replay
  is a recovery read, **not** authorization to invoke again.
- `reconcile_reservation(..., reconciliation_id=..., liability=..., final=...,
  evidence=...)` records cumulative invocation totals and source authority.
  Source-key replay is idempotent; changed facts conflict. A new source key may
  replace pending totals. Final, known marginal cost settles the reservation;
  incomplete coverage or unknown/partial cost leaves pending reconciliation.
  Settled facts cannot be revised by a new event. Unknown token/time metrics can
  still block their respective limits after dollar settlement.
- `release_undispatched(..., non_dispatch_proof=...)` only releases a reserved,
  never-adopted identity, with caller evidence. Recorded execution contradicts
  release. A timeout, lease expiry, reset or crash is not evidence of non-dispatch.
- `budget_snapshot(job_id)` returns policy, records and totals. Each metric has
  `total`, `known_subtotal` and `state`; an unknown total is `None`, not zero.

Reservation and consumption records share the immutable `ExecutionAttempt` facts
for `(job_id, attempt_id)`: task, run, invocation (`attempt_id`), start time,
adapter, model and optional provider. Provider defaults to `None` for legacy
records and is never inferred from billing. Exact attempt replay is idempotent;
conflicting facts raise `BudgetConflictError("invocation identity conflict")`.
Recording an attempt checks its reservation under the same lock/transaction.
Adoption, reservation replay, usage writes and reconciliation also reject
persisted cross-ledger conflicts from older versions, including recovery replays.
Recording execution after release is rejected. New reservations still cannot be
added retroactively to recorded invocations. Distinct invocation IDs remain
independent, including retries/fallbacks within one run; keys remain job-scoped.
Unreserved attempts retain the existing immutable consumption-ledger contract.

Admission counts reserved/dispatching allowances plus reconciled liability;
released reservations do not consume attempts or allowances. Pending coverage
blocks capped metrics even if a subtotal is known. Actual final consumption may
exceed an allowance or cap: reconciliation records that liability and subsequent
admission blocks. No settlement is rejected merely because real spend exceeded
the estimate. USD means API charge plus plan marginal charge; plan API-equivalent
estimates remain separate and cannot consume the USD cap. Plan no-marginal-cost
billing must explicitly provide `plan_marginal_usd=0`; unknown is not free.

Reconciliation is an explicit caller assertion of cumulative totals and finality,
not a new billing API or an automatic adjudicator of telemetry conflicts. It
must cover the whole invocation, including overlapping observations. Budget
snapshots never sum usage observations. Consumption reports, selected-result
usage, cost reports and receipts keep their existing behavior.

SQLite schema v4 adds `budget_reservations`, keyed by `(job_id, attempt_id)` with
a job foreign key, state constraint and job/state index. It deliberately has no
attempt foreign key because admission precedes execution. Migration from v3 is
transactional and invents no reservations. Every admission/lifecycle operation
uses `BEGIN IMMEDIATE`, or shares an enclosing completion writer transaction.
Rollback includes the reservation. Job deletion removes reservations; task and
subgraph resets do not. Reopening preserves reserved, dispatching and pending
states; a recovery owner must explicitly reconcile uncertain dispatches.

The file backend uses a job-wide 300-second expiring lock and atomic rename for
each reservation record. Contention raises for retry. It has no multi-record
transaction or fsync/power-loss guarantee; a writer paused beyond lock expiry can
race a reclaimer. Consumption writes share this job lock. Concurrent job deletion
and policy changes are not fenced by it. Use SQLite for strict admission serialization.

## Runtime enforcement coverage

`WorkerRuntime` binds the store, task, run and lease-loss signal to the invocation
scope. Persisted warm reuse and preflight blocks bypass invocation entirely.
CLI adapters (including Cursor's alternate launch and Hermes retries), OpenAI
HTTP requests, each agentic provider retry/failover, and the generic adapter
fallback reserve and adopt before the adapter/provider call. Each call gets a
fresh invocation nonce linked to its run. Repeated persistence callbacks use
that same identity; task resets do not reset consumption. Legacy jobs without a
policy keep best-effort telemetry and their previous dispatch behavior.

By default, invocation allowances are unknown, except explicit plan billing has
zero marginal USD. Thus attempt-only caps work without price metadata; API USD,
token and elapsed caps fail closed if no bounds are supplied. A task may provide
`payload.budget_allowance`, a `BudgetLiability` dictionary specifying conservative
**per-invocation** bounds for configured units. For example, an API task with
`billing="api"` can supply `{"billing":"api", "cost_state":"known",
"api_usd":1, "tokens_in":10000, "tokens_out":2000, "elapsed_seconds":120}`.
These are caller assertions, not inferred prices or router estimates. They must
bound every retry/failover using that task; billing must match the actual
invocation. Invalid or insufficient bounds stop dispatch. Plan API-equivalent
estimates cannot pay the marginal USD allowance.

Admission and adoption persistence errors propagate before the external call.
After adoption, the runtime writes pending uncertainty before calling the
provider. This deliberately serializes admission under capped consumption
metrics while an outcome is uncertain; attempt-only caps can admit concurrently.
A known lost lease before adoption releases the uninvoked reservation; after
adoption it is never released. A crash between lifecycle writes leaves the last
durable reserved/dispatching/pending state, never an automatic refund.

Provider return usage and CLI `type=result` usage events can assert final
coverage. Generic artifact usage and other CLI events remain observations, not
billing authority. Exactly one authoritative snapshot is used, never a sum of
overlapping snapshots. Measured zeros remain zeros; estimates cannot settle
actual billing. Exceptions, timeouts, output truncation, lease loss, missing or
ambiguous final telemetry remain pending. Reconciliation failures preserve both
the accepted result and the durable pending fence. Recovery can inspect
`budget_snapshot` and submit authoritative cumulative totals with a new
reconciliation ID. There is no automatic provider billing recovery poller.

The boundary covers calls owned by WorkerRuntime. Direct adapter/provider
library calls without a runtime scope have no job/store and remain unbudgeted.
Opaque CLI internal calls cannot be separately admitted. Caller-supplied bounds
are not provider-enforced limits, and actual consumption may exceed them;
subsequent admission then blocks. These limitations prevent a claim of an
absolute billed-dollar or token ceiling. Elapsed time measures the local
invocation lifetime; unknown remote work after a timeout stays pending.

## Public CLI and MCP launch inputs

All worker start surfaces (`run`, direct adapters, `swarm`, `review`, `edit`,
`prewalk`, and `browser`) accept these optional cumulative job limits:

| CLI flag | MCP argument | Unit |
| --- | --- | --- |
| `--budget-max-usd` | `budget_max_usd` | USD marginal charge |
| `--budget-max-tokens-in` | `budget_max_tokens_in` | Input tokens |
| `--budget-max-tokens-out` | `budget_max_tokens_out` | Output tokens |
| `--budget-max-attempts` | `budget_max_attempts` | Invocations, including retries |
| `--budget-max-elapsed-seconds` | `budget_max_elapsed_seconds` | Summed invocation seconds |

Public inputs must be positive and finite; token and attempt limits must be
integers. Zero is rejected at launch, while the internal `BudgetPolicy` still
supports zero for exhausted policies. Omit all five inputs to retain legacy
`budget_policy=None`. Omitted individual fields remain uncapped. A launch-key
retry resumes the same job only if its policy and launch request match; changing
or omitting a previously supplied policy fails closed.

`--max-cost-usd` / `max_cost_usd` is still a per-call **routing estimate filter**.
It does not set a cumulative job budget. For example, this MCP request sets both:

```json
{
  "goal": "Review budget input validation; return file-backed findings. Do not edit files.",
  "adapter": "codex",
  "launch_key": "budget-review-canary-001",
  "max_cost_usd": 0.10,
  "timeout_seconds": 120,
  "budget_max_attempts": 1,
  "budget_max_elapsed_seconds": 120
}
```

Pass this to `puppetmaster_start_swarm`. A public CLI/MCP launch refuses to
create a job when a requested cap has no bounded per-invocation allowance. The
error names the cap, adapter, and remediation. Elapsed caps derive a
conservative allowance from `timeout_seconds` / `--timeout-seconds` (capped at
the job limit). Output-token caps derive from `max_output_tokens` /
`max_tokens`. Plan billing may supply zero marginal USD. Input-token and API
USD caps require an explicit `payload.budget_allowance`. Unknown liability is
never treated as zero.

`budget_max_attempts` needs no allowance and is the safe first canary. This is
fail-closed, not permission to invent a price or usage value.

For a first live Codex canary, use a read-only review and cap invocation count,
which does not depend on provider pricing or token estimates:

```sh
python -m puppetmaster review \
  'Read puppetmaster/budget.py and return one file-backed finding. Do not edit files.' \
  --adapter codex --wait --cwd . --timeout-seconds 120 \
  --budget-max-attempts 1 --launch-key budget-codex-canary-001 \
  --label 'Codex budget canary'
```

This command invokes a real provider; run it deliberately. Adding
`--budget-max-elapsed-seconds 120` is valid because the command already sets
`--timeout-seconds 120`. Adding `--budget-max-tokens-in` or API
`--budget-max-usd` without `payload.budget_allowance` is rejected before the
job exists. Inspect `status` (the job's `budget_policy`), `wait`/`await` JSON,
or the dashboard job-detail JSON (`budget.policy`, `budget.totals`, and
reservations). The persisted policy survives reopening either store backend.

A budget gates subsequent admission; opaque provider overruns cannot guarantee
an absolute billed ceiling. Final consumption can exceed the reservation, and
reconciliation records that overrun. Plan/API/unknown billing semantics remain
unchanged: plan API-equivalent estimates are not billed USD, and unknown is not
zero. An attempts-only canary imposes no dollar or token ceiling.
