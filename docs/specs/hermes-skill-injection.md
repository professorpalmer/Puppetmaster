# Hermes skill injection — routed workers inherit your live skills

Status: implemented (v0.9.81)
Owner: Puppetmaster
Related: `puppetmaster-learn` plugin (the other direction of the flywheel)

## Problem

A routed Hermes worker is a blank `hermes chat` with a task. It runs
`--ignore-rules`, so it sees **none** of the user's live Hermes skills, memory,
rules, or persona — and its toolset deliberately excludes `memory` /
`session_search` (`adapters.py:1888`). That isolation is correct and load-bearing:
it stops one swarm task observing another (or the user's interactive history).

But it means the worker doesn't know what *you* know. The `puppetmaster-learn`
plugin already bridges one direction — swarm → skill **candidate**. The missing
return leg is **skill → worker**: hand a routed worker the bodies of the user's
existing live skills so it works the way the user works.

## Non-goals

- **Not** dropping `--ignore-rules`. That re-opens cross-task rule bleed and
  drags the whole persona/rules layer in wholesale — the chainsaw. We use the
  scalpel instead.
- **Not** granting the worker a `skills` / `memory` / `session_search` toolset.
  The worker's access surface does not change at all.
- **Not** in-process execution, and **not** letting workers self-select skills.

## Core insight: a second consumer of an existing seam

This is **not a new mechanism**. `prompt_with_memory` (`adapters.py:2825`)
already injects orchestrator-selected context into a worker via the task
payload — the *trusted planner* fills `task.payload["retrieved_memory"]`
(a list, so it is **not** counted in routing's `payload_size_chars`, see
`router.py:645`), and the worker merely renders it. Skill injection is the
mirror image: `prompt_with_skills` reads `task.payload["injected_skills"]`.

Because the orchestrator — which already sees everything — does the selection
and hands the worker a curated packet, the isolation boundary is untouched.
We are not re-opening a door; we are handing over a sealed envelope.

## Design

### 1. Selection (the whole ballgame)

Selection is a semantic/keyword match of **task instruction → skill
frontmatter `description`**, run at dispatch, before routing. It does **not**
reuse CodeGraph (CodeGraph ranks code symbols; skills are prose markdown).

Selection quality is **load-bearing, not nice-to-have**, because of the cost
vector below — you can only afford to inject a tiny set, so picking the right
ones is the design, not the plumbing.

v1 ranks by keyword/token overlap on `name + description`. The ranking function
is intentionally swappable (embeddings are a v2 drop-in); the architectural
commitment is the **cap + the seam**, not the scoring function. v1 deliberately
takes **no embedding dependency** — the orchestrator runs in the 3.9
interpreter and skill descriptions are written to be discriminative.

### 2. The cap is a TOKEN BUDGET, not a skill count

Each worker is a fresh `hermes chat` loop, so an injected packet rides the
prompt on **every episode** — a 15-episode worker re-pays the bundle 15×. And
the usual escape hatch (inject names+descriptions, lazy-fetch bodies on demand)
is **closed**, because the worker has no skills toolset to fetch with. So we are
committed to injecting **full bodies of a tiny curated set**.

A count cap (N=2–3) only proxies the thing that matters — the per-turn cost.
Two fat skills can dwarf three lean ones. So the primary cap is a **packet token
budget**: add highest-ranked skills until the packet would exceed the budget;
a count cap is the secondary safety. This ties the cap directly to the per-turn
cost rationale and makes the routing-token bump (below) deterministic, because
the packet ceiling is known up front.

### 3. Cost honesty rides the existing router seam

`estimate_tokens_in` (`router.py:213`) returns `task.estimated_tokens_in` when
set. That single value feeds **both** the chosen model's cost **and** the
baseline snapshot (`router.py:448` `_baseline_model.marginal_cost_usd(tokens_in,
...)`), and the baseline is drawn from the same constrained candidate set the
pick comes from (documented honesty contract at `router.py:440–446`).

So the entire cost fix is: set
`payload["estimated_tokens_in"] = base + packet_tokens` **before** routing.
Because chosen-cost and baseline read the same `tokens_in`, the savings receipt
stays like-for-like **by construction** — you cannot drift the comparison.

What needs the packet size is **only** the pre-flight estimate + baseline.
The recorded ledger **actuals never drift**: they are measured per turn at
execution (`count_tokens` in the adapter), so they already count the
packet-inflated tokens automatically. Actuals self-correct; the estimate is the
only thing that needs telling.

Consequence to document, not a bug: a larger `tokens_in` also feeds the
cost-cap filter and the cost-ranked pick, so a fat packet **can change which
model a task routes to** (or trip a `max_cost_usd` cap). That is *correct* —
you should route on the true, packet-inflated cost — but it means
`puppetmaster route` answers differ with injection on/off.

### 4. Ordering (no chicken-and-egg)

```
discover → select → measure packet → set estimated_tokens_in → route → attach packet → build prompt
```

Selection is instruction→description, fully available at dispatch **before**
routing, so there is no circular dependency. Stated explicitly so nobody
"optimizes" select-after-route. In the orchestrator this is a
`_with_injected_skills` pass that runs right after `_with_retrieved_memory`
(both before `_create_tasks` → `_apply_auto_routing`).

### 5. Scoping enforcement: skill bodies only

`--ignore-rules` suppressed a *bundle*: rules + skills + memory + **persona
(SOUL.md)**. "Inject skills" must mean skill **bodies only** — the persona layer
stays suppressed, or workers start emitting persona meta-commentary instead of
doing the task. The implementation boundary is *skill body vs. everything else
`--ignore-rules` suppressed*, and it is **enforced, not assumed**: the
discovery step strips frontmatter and never reads SOUL.md / rules, and
`build_hermes_chat_command` keeps `--ignore-rules` set.

The keystone test is **structural** (assertable), not a tone check (not
assertable). It asserts on the constructed prompt/command:

- the prompt **contains** the selected skill bodies,
- it **does not contain** persona/rules/SOUL content,
- `--ignore-rules` is **still set** on the command,
- the injected set is **within the token budget / count cap**.

### 6. Storage coupling isolated to one function

`discover_hermes_skills` is the **single** place that knows both Hermes' skill
storage location (via `installers.hermes_skills_dir`, which respects
`$HERMES_HOME` and falls back to `~/.hermes/skills`) and the SKILL.md frontmatter
format. If Hermes reorganizes skill storage, the blast radius is this one
function. The failure is **observable, not silent**: when injection is enabled
but discovery returns zero, the orchestrator emits a `skills.none_discovered`
event so a storage reorg degrades loudly in the ledger instead of turning
injection into a permanent silent no-op.

### 7. Opt-in + provenance

- Off by default. Opt in with `PUPPETMASTER_INJECT_HERMES_SKILLS=1` or per-spec
  `payload["inject_skills"] = True`. Per-spec `inject_skills = False` always wins.
- Every injection emits a `skills.injected` event recording which skills landed
  in which job/role and the packet token size — so the learn-flywheel symmetry
  (swarm→skill and skill→worker) is auditable in **both** directions.

## Surfaces

- `puppetmaster/skill_injection.py` — discovery, selection, packet rendering,
  token budgeting (orchestrator-side; pure, injectable, no Hermes import).
- `adapters.prompt_with_skills` — mirror of `prompt_with_memory`; renders
  `payload["injected_skills"]` into the Hermes worker prompt only.
- `orchestrator._with_injected_skills` — the dispatch pass implementing the
  ordering above; sets `injected_skills` (list) + `estimated_tokens_in`.

## Future work

- v2 selector: embedding match over descriptions (swap the ranking function).
- Feed back which injected skills correlated with successful tasks (closes the
  loop with the learn flywheel's promotion signal).
- Upstream: a `hermes chat --reasoning-effort` flag would retire the temp-home
  workaround (unrelated to this spec; tracked separately).
