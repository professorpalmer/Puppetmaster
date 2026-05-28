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

## Scope and honesty

Four production adapters ship today: `cursor` (Cursor SDK via
`@cursor/sdk`), `claude-code` (Anthropic via the `claude` CLI),
`openai` (direct Chat Completions via `OPENAI_API_KEY`, added in
v0.6.1-beta.1), and `codex` (official OpenAI Codex CLI via
`codex exec --json`, added in v0.7.0). Together they cover the entire
starter registry. **Raw HTTP adapters for additional providers
(Gemini, DeepSeek, Kimi) are not yet in.** They slot in cleanly as
new `adapter` values — the registry + router/classifier framework
doesn't need to change — but each one needs real validation against
its provider's API before it ships.

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
