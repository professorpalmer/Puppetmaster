# Routing-quality evaluation

Puppetmaster's paired evaluator compares `balanced` routing with the strongest
eligible pinned baseline on the same versioned cases, packaged snapshot files,
canonical immutable snapshot digests, and repetitions. Each executor receipt
must attest the observed digest; missing or mismatched snapshot proof fails the
run. Arm order is reproducibly randomized from an explicit seed. Execution is
injected, so the harness can use Codex, Claude Code, Hermes, Antigravity, or
deterministic local fixtures without calling the OpenAI API directly.

The deterministic grader measures acceptance pass rate, criterion score,
unintended files, catastrophic failures, correction cycles, elapsed time,
retries, tokens, and nominal and marginal cost. Seeded failures are executable
negative vectors that must fail the same grader, and expected answers are not
disclosed in task instructions. Reports include paired deltas and uncertainty;
bounded finite-sample quality intervals do not collapse to zero merely because
a small sample tied. A configured non-inferiority margin can produce one of
three bounded results: noninferior, inferior, or inconclusive.

These are corpus-scoped measurements. A non-inferiority result supports only
the stated margin, corpus version, repetitions, and interval. An inconclusive
result supports no quality-preservation claim. Neither result proves that
routing improves quality without measured paired evidence.

Structural artifact presence remains useful process-health evidence, not semantic quality,
and cannot establish that an answer, finding, or patch is correct. Semantic
conclusions come only from the case's deterministic
acceptance criteria and observed results.

Shadow routing is opt-in. It records the production selection and a
counterfactual policy/model while explicitly recording
`production_selection_changed: false`; it never replaces the model dispatched
by the production policy. Auto-routed workers enable it with
`payload.shadow_policy` (for example, `quality`); the existing ROUTING artifact
then persists the counterfactual evidence alongside the unchanged production
selection.
