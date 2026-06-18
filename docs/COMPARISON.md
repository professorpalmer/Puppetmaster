# How Puppetmaster compares

Short version: **Puppetmaster is not another agent framework.** LangGraph, CrewAI, AutoGen/AG2, and the Claude Agent SDK are libraries you write code against to *build* an agent. Puppetmaster sits one layer up — it's a **supervisor that drives the agent CLIs/SDKs you already pay for** (Cursor, Claude Code, Codex, the OpenAI API, and Hermes), routes each task to the cheapest model that can handle it, keeps the spend inside your existing subscription, and stores every worker's output as a typed artifact so follow-ups cost zero tokens.

So this isn't "Puppetmaster vs LangGraph, who wins." They solve different problems and compose fine. This page is here to answer the only question that matters at a glance: *given what I'm trying to do, do I want this?*

## At a glance

| | **Puppetmaster** | LangGraph | CrewAI | Claude Agent SDK / Code subagents | Native IDE subagents (Cursor / Claude built-in) |
|---|---|---|---|---|---|
| What it fundamentally is | A cost-aware **orchestrator** over existing agent CLIs | A library to build **stateful agent graphs** | A library for **role-based agent crews** | An SDK + CLI to build/run **one Anthropic agent loop** | Built-in helpers that spawn subagents for a task |
| You interact by | MCP tools / CLI (no code to write) | Writing Python graph code | Writing Python role/task code | Writing code / using the CLI | Asking the IDE agent |
| Fan a single run across **multiple vendors** (Cursor + Claude + Codex + OpenAI) | **Yes** — that's the point | You wire it yourself | Per-agent LLM config | Anthropic only | The host vendor only |
| **Task → model cost routing** (auto-pick cheapest sufficient) | **Yes**, auditable per task | No (you choose models) | No (per-role models) | No | No |
| **Subscription / plan-billing containment** (keep spend in the plan you already pay for) | **Yes** — plan-first routing + detection | No | No | Plan or API, single vendor | Yes (it's the vendor's own plan) |
| Durable, replayable, **typed artifacts** → $0 follow-up reads | **Yes** (SQLite, payload+evidence+confidence+sha256) | Durable state / checkpointing (graph state) | Limited | Session/transcript | Transcript |
| **Auto-fallback** when a provider is unfunded/down | **Yes** — reroutes to the next funded adapter, proven live | You build retry/edges | Limited | — | — |
| Setup cost | `pipx install` + `setup` | Framework + your code | Framework + your code | SDK + your code | Zero |
| Maturity / ecosystem | Young, single-author, daily-driver beta | Large, production-proven, big ecosystem | Large, popular for prototyping | Backed by Anthropic | Backed by the IDE vendor |

Overlap is real and worth stating plainly: **LangGraph also gives you durable state and checkpointing**, and **Claude Code already spawns subagents**. The thing none of them centers on is *routing work across the agent CLIs you already subscribe to, by cost, with an auditable decision trail and a self-healing fallback when one provider is dead.* That's Puppetmaster's center of gravity.

## Pick something else if…

- **You're building a custom agent with bespoke control flow, tools, and memory in code** → use **LangGraph**. It's the production standard for stateful graphs, has a huge ecosystem (LangSmith observability), and is far more mature. Puppetmaster orchestrates agents; it isn't a framework for authoring one from primitives.
- **You want a fast role-based prototype** ("researcher → writer → editor") **and time-to-demo is everything** → use **CrewAI**. Its role/goal/backstory metaphor ships demos faster than anything here.
- **You're all-in on Anthropic and want one well-supported agent loop** → use the **Claude Agent SDK** / Claude Code subagents directly. If you never need other vendors or cross-vendor cost routing, Puppetmaster is overhead.
- **You just want to run one task with zero setup inside your IDE** → use the **native subagent**. Puppetmaster is for the messy work (long investigations, repeated handoffs, evidence + replay + approval gates), not a one-off edit.

## Choose Puppetmaster when…

- You already pay for **more than one** agent platform (Cursor *and* Claude *and*/or ChatGPT/Codex) and want a single run to use whichever is cheapest/sufficient — **without** burning a standalone API account.
- You want the **bill to stay inside the subscription** you already pay for, automatically, with detection that reads your real auth state.
- You need **auditable** per-task model decisions (why did this task run on that model? what did it reject and why?) — not a model picked by vibes.
- You want a swarm to **survive a dead/over-quota provider** instead of returning a degraded run.
- You want follow-up questions about a finished job to be **free** (durable typed artifacts, not a transcript replay).

## Honest caveats

- Puppetmaster is **young and single-author** (daily-driver beta), versus incumbents with thousands of stars and large teams. Maturity and ecosystem are theirs to win today.
- It's **narrower by design**: it shines for people already running multiple agent CLIs who care about cost and auditability. If that's not you, a framework or a native subagent is the simpler choice.
- The comparisons above are by **design center**, not a feature-by-feature audit; these tools move fast and overlap is growing. Verify current capabilities before betting on any of them.
