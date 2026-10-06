# Orchestration durability slice (Comet/Zeron → Puppetmaster)

Product steal for durable orchestration — **not** UI kit, not StrongOrc model
rank, not Loro day-one.

## Landed seams (this haul)

1. **Session command ledger** (`puppetmaster/session_commands.py`)
   - Kinds: `run` / `steer` / `interrupt` / `respond_input`
   - Rules: append-only entries, host-owned outcomes, mark-processed **before**
     execute, TTL / supersede / past-turn interrupt evaluation
   - Maps interrupt → cancellation request identity; run → task admission
2. **WorkspaceScope freeze** (`puppetmaster/workspace_scope.py`)
   - Primary store root frozen once per engine process (CLI bind / MCP main)
   - `create_store(..., mode="ensure")` refuses a foreign root while frozen
   - Cross-project **attach** (deferred) stays allowed
3. **Headless engine honesty**
   - MCP + CLI own orchestration state; viewports observe
   - See Architecture command-plane + research notes above

## Deferred

- Full Loro/CRDT session docs + ChatRoom/DeviceRoom
- Synced cloud `WorkspaceScope` profile
- ACP-universal adapter

## Tests

- `tests/test_session_commands.py`
- `tests/test_workspace_scope.py`
