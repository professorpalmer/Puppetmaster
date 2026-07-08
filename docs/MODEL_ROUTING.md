# Intelligent Model Orchestration

Puppetmaster ships a task-aware **model router** that picks the right
LLM for each task instead of pinning one model per adapter. Cheap
models handle trivial work, capable models handle hard work, vision
tasks land on a vision-capable model, and you see exactly why.

Added in v0.6.0. Extended with the OpenAI tier in v0.6.1-beta.1 and
the Codex tier in v0.7.0.

## The three pillars

1. **A user-owned registry.** You describe your own models, prices,
   and asserted capability scores in `~/.puppetmaster/models.json`
   (override with `$PUPPETMASTER_MODELS_PATH`). No hardcoded model
   names, no live price fetching — your subscriptions, your numbers.
2. **A transparent classifier.** Pure-function heuristic that assigns
   a 0..100 capability-needed score from the task's role + instruction
   + payload (e.g. `verify-runtime` ≈ 25, `explore` ≈ 50, `implement`
   ≈ 75, `audit/security-review` ≈ 90+). Vision tasks auto-add a
   `vision` required-tag so non-vision models are filtered out
   cleanly. Override per-task with `payload.min_capability`.
3. **Four policies.** `balanced` (default — cheapest sufficient, ties
   broken toward right-sized smaller models), `cheap`, `quality`,
   `escalating` (ordered chain for retries). Override per-task with
   `payload.routing_policy`.

**Every routing decision is a durable artifact.** Picked model,
classifier output, estimated USD cost, and the full list of rejected
alternatives with the reason each was rejected — all stored as an
`ArtifactType.ROUTING` artifact tied to the task. Run
`puppetmaster artifacts <job_id>` to see why each task went where,
or `puppetmaster cost <job_id>` to sum spend across the run.

## Where it kicks in automatically (and where it doesn't)

This is the part to be honest about:

| Surface                                         | Auto-routes? |
| ----------------------------------------------- | ------------ |
| `puppetmaster_start_cursor_swarm` (MCP)         | **YES** — default workers ship with `auto_route: true`. |
| `puppetmaster_start_swarm` (MCP)                | **YES** — same default workers. |
| `puppetmaster_start_claude_implement` (MCP)     | Opt-in per call — pass a spec with `auto_route: true` or accept the default. |
| `python -m puppetmaster run`                    | **YES** for built-in workers; opt-in per spec in a custom config. |
| Cursor's main chat window (typing `@cursor`)    | **NO.** Cursor's own model picker chooses the chat model — Puppetmaster is not in that loop. The router applies *when Puppetmaster runs a swarm*, not when Cursor's agent is having a conversation with you. |
| Claude Code's main session                      | **NO** — same reason. Claude Code picks its own session model. |

In other words: **the router governs how Puppetmaster fans work out
across its swarm workers; it does not (and cannot) hijack the model
your IDE's primary chat agent uses.** If you want the cheap-tier
model for trivial chat work, set that as your IDE's default in Cursor
settings. The router is for *every task Puppetmaster delegates*,
which on a real workflow is far more model invocations than the chat
itself.

If you haven't run `puppetmaster models init` yet, auto-routing is a
clean no-op: the orchestrator emits one `router.registry_empty` event
per run, then falls back to each spec's declared adapter. Nothing
breaks.

## The four tiers in the starter registry

`puppetmaster models init` writes **12 tiered model entries** that
map directly to the "easy / balanced / high / extra-high" mental
model — 5 Cursor/Claude tiers, 4 OpenAI tiers, and 2 Codex tiers,
covering every cheap → frontier pairing across all four production
adapters. **The `adapter_model_name` values are the literal strings
each adapter passes through to its SDK / CLI today** (verified
against Cursor's runtime catalog, Anthropic's `claude` CLI, and
OpenAI's `codex` CLI as of v0.7.0): `composer-2.5`, `gpt-5.5`,
`claude-haiku-4-5`, `claude-opus-4-6`, `claude-opus-4-7`,
`claude-opus-4-8` for the
Cursor/Claude tier; `gpt-5.5` / `gpt-5.4` / `gpt-5.4-mini` /
`gpt-5.4-nano` for the OpenAI tier; `gpt-5.5` / `gpt-5.4-mini`
(routed through `codex exec --json`) for the Codex tier. When newer
versions land, edit `adapter_model_name` in
`~/.puppetmaster/models.json` and the tier ids stay stable:

| Tier ID                  | Adapter       | Mental model                                       | Tags |
| ------------------------ | ------------- | -------------------------------------------------- | ---- |
| `cursor/composer-2-5`    | `cursor`      | fast / cheap / reading (\$0 — bundled in Cursor plan) | `cheap`, `fast`, `reading`, `code` |
| `cursor/gpt-5-5`         | `cursor`      | balanced — \$0 via Cursor plan, GPT-5.5 quality     | `balanced`, `fast`, `vision` |
| `claude-code/haiku-4-5`  | `claude-code` | cheap on the Anthropic side (\$1 / \$5) — the cheap tier for Claude-Code-only users | `cheap`, `fast`, `vision`, `reading`, `code` |
| `claude-code/opus-4-6`   | `claude-code` | high-quality — \$5 / \$25 per MTok                  | `quality`, `vision`, `reasoning` |
| `claude-code/opus-4-7`   | `claude-code` | previous frontier — \$5 / \$25, superseded by 4.8  | `frontier`, `vision`, `detailed-vision`, `reasoning` |
| `claude-code/opus-4-8`   | `claude-code` | **frontier flagship (router's top tier)** — \$5 / \$25, 1M context, best for the hardest reasoning + detailed vision | `frontier`, `vision`, `detailed-vision`, `reasoning`, `code` |
| `openai/gpt-5-5`         | `openai`      | frontier via Responses API — \$5 / \$30 per MTok    | `frontier`, `vision`, `detailed-vision`, `reasoning`, `code` |
| `openai/gpt-5-4`         | `openai`      | workhorse — \$2.50 / \$15 per MTok                  | `quality`, `fast`, `vision`, `code`, `reasoning` |
| `openai/gpt-5-4-mini`    | `openai`      | balanced — \$0.75 / \$4.50 per MTok                 | `balanced`, `fast`, `vision`, `code` |
| `openai/gpt-5-4-nano`    | `openai`      | cheap reader — \$0.15 / \$0.90 per MTok             | `cheap`, `fast`, `reading` |
| `codex/gpt-5-5`          | `codex`       | frontier with the **Codex agent loop** (file edits, shell, search) — \$5 / \$30 per MTok | `frontier`, `vision`, `reasoning`, `code`, `agent-loop` |
| `codex/gpt-5-4-mini`     | `codex`       | balanced with the Codex agent loop — \$0.75 / \$4.50 per MTok | `balanced`, `vision`, `code`, `agent-loop` |

With the starter registry, balanced-policy routing lands roughly:

| Task                                            | Picked model |
| ----------------------------------------------- | ------------ |
| `format these files`                            | `cursor/composer-2-5` |
| `map the auth module`                           | `cursor/composer-2-5` |
| `add password reset endpoint`                   | `cursor/gpt-5-5` |
| `decision: which caching strategy fits`         | `claude-code/opus-4-6` |
| `security audit every endpoint`                 | `claude-code/opus-4-8` (frontier flagship — hardest tier) |
| `describe what you see in the screenshot`       | `cursor/gpt-5-5` (vision-tagged) |
| `OCR every detail of the diagram`               | `claude-code/opus-4-7` (detailed-vision; right-sized below the flagship) |
| `refactor every callsite of foo() and add tests` | `openai/gpt-5-4` (workhorse — cheaper than frontier, capable enough for cross-file refactor) |

## Quick start

```bash
# 1. Write the starter registry (6 Cursor/Claude tiers + 4 OpenAI tiers + 2 Codex tiers = 12)
python -m puppetmaster models init

# 2. Inspect the registry
python -m puppetmaster models list

# 3. Dry-run a routing decision before kicking off a swarm
python -m puppetmaster route "Security audit across every endpoint" --role audit
# picked: claude-code/opus-4-8  (adapter=claude-code, model_name=claude-opus-4-8)
# policy: balanced
# capability needed: 99  chosen capability: 99
# estimated tokens: in=510  out=5000  estimated cost: $0.127550
# why: policy=balanced: cheapest model whose capability_score (99) >= needed (99)
# rejected:
#   - cursor/composer-2-5:  capability_score 55 < needed 99
#   - cursor/gpt-5-5:       capability_score 78 < needed 99
#   - claude-code/haiku-4-5:capability_score 55 < needed 99
#   - claude-code/opus-4-6: capability_score 88 < needed 99
#   - claude-code/opus-4-7: capability_score 98 < needed 99
#   - openai/gpt-5-5:       capability_score 96 < needed 99
#   - openai/gpt-5-4:       capability_score 86 < needed 99
#   - openai/gpt-5-4-mini:  capability_score 70 < needed 99
#   - openai/gpt-5-4-nano:  capability_score 52 < needed 99

python -m puppetmaster route "Format these files" --role verify-runtime
# picked: cursor/composer-2-5  (adapter=cursor, model_name=composer-2.5)
# capability needed: 20  chosen capability: 55
# estimated cost: $0.000000  (Cursor-tier models bill through your Cursor plan)
```

## Wiring auto-routing into a swarm

Set `payload.auto_route = true` on any worker spec. The orchestrator
replaces the spec's `adapter` and stamps `payload.model` from the
router's decision before the task runs, and persists a `ROUTING`
artifact:

```python
from puppetmaster.workers import WorkerSpec

specs = [
    WorkerSpec(
        role="explore",
        instruction="Map the auth subsystem",
        payload={"auto_route": True},
    ),
    WorkerSpec(
        role="audit",
        instruction="Find auth bypasses in every endpoint",
        payload={"auto_route": True, "routing_policy": "quality"},
    ),
    WorkerSpec(
        role="verify-runtime",
        instruction="Run pytest and report results",
        payload={"auto_route": True, "max_cost_usd": 0.01},
    ),
]
```

After the run:

```bash
python -m puppetmaster artifacts <job_id> | jq '.[] | select(.type=="routing") | .payload'
# {
#   "model_id": "claude-code/opus-4-8",
#   "adapter": "claude-code",
#   "adapter_model_name": "claude-opus-4-8",
#   "policy": "balanced",
#   "capability_needed": 99,
#   "capability_score": 99,
#   "estimated_cost_usd": 0.127550,
#   "reason": "policy=balanced: cheapest model whose capability_score (99) >= needed (99)",
#   "rejected": [...]
# }
```

## Per-task overrides

| Override                       | Effect                                                                 |
| ------------------------------ | ---------------------------------------------------------------------- |
| `payload.min_capability` (int) | Force classifier output to this value (0..100).                        |
| `payload.max_cost_usd` (float) | Hard cap on estimated per-call USD cost. Models over budget are excluded with a clear rejection reason. |
| `payload.required_tags` (list) | Only consider models whose `tags` include ALL of these.                |
| `payload.routing_policy` (str) | One of `balanced` (default), `cheap`, `quality`, `escalating`.         |
| `payload.registry_path` (str)  | Use a different registry file for this task.                           |
| `payload.min_confidence` (float) | Confidence floor for mid-run escalation (0..1). Below it, the task is re-run one capability tier up. Off unless set (or `$PUPPETMASTER_ESCALATE_CONFIDENCE`). |

## Effort variants (and escalating effort)

The same model at different reasoning effort has very different cost and
capability, so effort variants are **plain registry rows** that compete in
routing like any other model. Create them with the wizard
(`puppetmaster models setup` → "effort variant") or one-shot:

```bash
python -m puppetmaster models set "claude-code/fable-5" effort=low
```

Adapters with an effort knob: `openai` / `agentic`
(`reasoning_effort`), `codex` (`-c model_reasoning_effort=...`),
`hermes` (config-level `reasoning_effort`), and `claude-code`
(`--effort low|medium|high|xhigh|max`, requires Claude Code >= 2.1.204).
`cursor` has no effort knob today. For a one-off manual run,
`puppetmaster claude "<prompt>" --effort low` sets it without touching
the registry.

**Escalating effort is composition, not a separate policy.** Register
two or three effort variants of the same model with honest capability
scores and output-token multipliers (e.g. `claude-code/fable-5-low` at
90 / 0.7x, `claude-code/fable-5` at 100 / 2x), and the existing
machinery does the rest:

- the `escalating` policy starts at the cheapest sufficient variant and
  orders the rest as the retry chain;
- confidence-based escalation (`payload.min_confidence`) and
  review-gate escalation re-dispatch onto the stronger variant when the
  low-effort run wasn't good enough.

There is deliberately no automatic low → medium → high → xhigh → max
ladder per task: "this failure was caused by too little effort" has no
reliable signal to key on, and a five-rung blind retry ladder multiplies
worst-case cost. Variants + the existing escalation triggers give the
same outcome with an audit trail.

## Confidence-based mid-run escalation (opt-in)

Upfront routing is a *prediction*: the classifier scores complexity from
the role + instruction *before* any work happens. Sometimes a task that
looks simple turns out hard once a worker is in it — and the cheap model
it was routed to will tell you, via a low-confidence `VERIFICATION`
artifact (its own self-assessment of the run).

When you set a confidence floor, the orchestrator acts on that signal: a
task that finished COMPLETE but below the floor is **re-dispatched to the
cheapest strictly-stronger funded model and re-run, before its result is
accepted**.

- **Enable globally:** `export PUPPETMASTER_ESCALATE_CONFIDENCE=0.7`
- **Enable per task:** `payload.min_confidence = 0.7`
- **Off by default** — escalation costs more, so it never fires unless you
  opt in. The default behavior (cheapest sufficient, accept the result) is
  unchanged.
- **Bounded** by `_MAX_ESCALATION_ATTEMPTS`; a task already on the top tier
  (no stronger model in the registry) is left as-is rather than looping.
- **Safe by construction:** only escalates router-placed tasks (never a
  model you pinned by hand), and respects the platform lock + billing
  posture exactly like auto-fallback (never reroutes onto a disabled or
  unfunded platform).
- **Auditable:** each escalation writes a `router-escalation` ROUTING
  artifact (`escalated_from_model`, `escalated_from_confidence`,
  `confidence_threshold`) and emits a `router.auto_escalate` event. The
  local dashboard's reroute panel shows it alongside fallbacks.

This is the counterpart to **auto-fallback**: fallback re-routes on
*failure* (a dead/unfunded provider), escalation re-routes on *low
confidence* (the cheap model ran but wasn't sure). Both reuse the same
bounded re-queue loop.

## Closing the loop: routing self-audit (`puppetmaster audit`)

Routing decisions, escalations, and verifications all land as durable
artifacts. `puppetmaster audit` reads them back and tells you whether your
capability scores still match reality — so the registry doesn't silently
"overcook" work as your model lineup or task mix drifts over weeks.

```bash
python -m puppetmaster audit                # dry-run report + suggested diff
python -m puppetmaster audit --window 7      # last 7 days only
python -m puppetmaster audit --apply         # write the suggested score changes
```

It prints, per model: how often it was the **initial pick**, mean/min
self-reported confidence, the rate it got **escalated away from**, and
estimated spend. Then it proposes score changes — but only for the one
case that's actually defensible:

- **`under-provisioned`** → a model that keeps getting picked and then
  escalated away from (or finishing below the confidence bar). The audit
  suggests **lowering its score** so the harder work routes to a stronger
  model instead, killing the cheap-then-expensive double-run. Bounded step
  (−5, or −10 when severe), floored so the model stays reachable for
  trivial work, and only after `MIN_SAMPLE` observations.
- **`possibly-over-used`** → a strong model confidently doing work that
  needed far less capability. This is **flagged, never auto-adjusted**:
  high confidence on a strong model doesn't prove a cheaper one would have
  succeeded — that needs a shadow run the audit deliberately doesn't
  perform. You get the signal; you decide.

**Recommend, don't autopilot.** Confidence and escalation rate are noisy,
self-reported, and gameable; a closed feedback loop on them risks a
ratchet that only ever raises cost. So `audit` is dry-run by default,
mutates `models.json` only with `--apply`, and the registry stays *your*
assertion. If you'd rather never touch scores automatically, just read the
report and ignore the diff — that's a supported mode, not a degraded one.

## Scope and honesty

Four production adapters plus the keys-only `agentic` standalone worker ship today: `cursor` (Cursor SDK via
`@cursor/sdk`), `claude-code` (Anthropic via the `claude` CLI),
`openai` (direct Chat Completions via `OPENAI_API_KEY`, added in
v0.6.1-beta.1), `codex` (official OpenAI Codex CLI via
`codex exec --json`, added in v0.7.0), and `agentic` (direct provider
HTTP APIs with your own key — no external CLI). Together they cover the
starter registry and curated catalogs. **`agentic` is the portable
keys-only path** when you have API keys but no vendor CLI installed;
vendor CLIs remain the ceiling for mature tool surfaces. Additional
providers slot in as new registry entries — the router/classifier
framework doesn't need to change.

Capability scores and prices stay **user-asserted**. Puppetmaster
makes the **decision** transparent (full audit trail of why each
task went where); it does not make the **value judgments** for you
(whether GPT-5.4 really is an 86, or whether Cursor's bundled models
should be treated as $0). Edit the registry to match your reality.

## MCP tools for routing

| MCP tool                       | What it does                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `puppetmaster_route_task`      | Dry-run the router on an instruction. Returns the picked model + cost + rejected alternatives. |
| `puppetmaster_list_models`     | Print the registry as JSON (path + each model spec).                          |
| `puppetmaster_job_cost`        | Sum spend (and rejection details) across every routing artifact in a job.   |
