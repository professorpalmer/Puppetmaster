# ACP lessons for Puppetmaster (do not adopt as universal adapter)

Steal target: Comet/Zeron `docs/research/acp.md` (2026-08, pending-prompt
fix 2026-09-09). Useful as a **negative** and hardening checklist.

## Hard lessons

1. **Quiet settle is a lie.** A completed tool result, partial assistant
   message, or usage update does **not** prove `session/prompt` finished.
   Retiring a pending response future after N seconds of quiet lets the next
   prompt race an agent that is still busy (`Invalid request`, lost
   `error.data`). Prefer retaining the response future; bound handshake /
   child lifecycle instead.
2. **Managed installs beat `npx -y` on the hot path.** Cold npx can stall a
   first turn for minutes and encode fatal fs errors as `256 - errno` exits
   with empty stderr — the "Working forever" class.
3. **Ordering hazards.** Drain queued updates before emitting Done; EOF right
   after a final response is a clean finish, not a crash.
4. **Turn-boundary vs mid-turn steering.** Without a steering extension,
   steers queue as the next prompt. Do not pretendsend mid-turn when the wire
   cannot accept it.
5. **Resume quirks.** Cancelled turns may leave a user message with no reply;
   the next prompt after resume can answer both. Session already-open errors
   need a fresh-session fallback with honest context loss.

## Puppetmaster mapping

| ACP / Zeron concept | Puppetmaster seam |
| --- | --- |
| `session/prompt` pending future | Adapter stream + attempt ledger; do not quiet-fake complete |
| `session/cancel` | Scoped durable cancellation (`store_contracts`) + interrupt command |
| Steer mailbox | Session command ledger `steer` → follow-up / planner enqueue |
| Stall watchdog | `liveness.reap_stalled_jobs` |

Puppetmaster MCP/CLI remains the **engine**. Marionette / Automaton / Discord
OS are **viewports** — they must not own orchestration truth that the store /
command ledger already hold.

## Explicitly skipped

Promoting ACP to the universal harness for Claude/Codex/Cursor inside
Puppetmaster. Keep native wires ([harness.md](harness.md)).
