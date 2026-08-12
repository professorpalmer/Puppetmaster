# Grok Bot harness (remote MCP pilot)

**Grok Bot is the pilot/client. Puppetmaster is the durable worker runtime.**

This is first-class harness support for Cursor's Grok Bot sidebar assistant product — not a fake `grok-bot` worker adapter. Grok Bot does not publish a task-dispatch API today, so Puppetmaster does **not** lease Grok Bot as a subprocess worker. Do not invent an adapter named `grok-bot` until that API exists.

## Why remote MCP

Grok Bot can consume MCP connectors that are **remote HTTP/SSE (or streamable HTTP)** only. It cannot register local stdio MCP servers like `python -m puppetmaster.mcp_server`. Marketplace plugins work because they are remote. Therefore the existing stdio JSON-RPC MCP — which Cursor / Claude Desktop / Codex use — is unreachable from Grok Bot even when you are "in" the platform.

The remote transport exposes the **same** MCP tool handlers (leases, jobs, artifacts, stitching, routing) over streamable HTTP, with bearer auth and a supervise-first scope. Stdio MCP is unchanged and remains the daily driver for Cursor.

```text
Grok Bot (pilot)
  -> HTTPS/SSE connector (Authorization: Bearer …)
  -> puppetmaster mcp serve-remote   (streamable HTTP /mcp)
  -> same tool handlers as stdio MCP
  -> detached python -m puppetmaster job
  -> independent worker subprocesses (cursor / claude / codex / …)
  -> SQLite events + artifacts
  -> Grok Bot polls status / logs / live_artifacts / show
```

## Quick start (local PoC)

```bash
# From a repo you want Puppetmaster to operate on:
export PUPPETMASTER_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m puppetmaster mcp serve-remote --scope supervise
# equivalent console script:
# puppetmaster-mcp-remote --scope supervise
```

The process prints a connector JSON on stderr:

```json
{
  "name": "puppetmaster",
  "url": "http://127.0.0.1:8743/mcp",
  "transport": "streamable-http",
  "headers": {
    "Authorization": "Bearer <token>"
  }
}
```

Print the connector without serving:

```bash
python -m puppetmaster mcp serve-remote --token "$PUPPETMASTER_MCP_TOKEN" --print-connector
```

### Reach Grok Bot from your laptop

Grok Bot needs a **reachable URL**. For a PoC, put a secure tunnel in front of loopback:

```bash
# terminal 1 — bind loopback only (default)
python -m puppetmaster mcp serve-remote --host 127.0.0.1 --port 8743

# terminal 2 — example with cloudflared (or ngrok)
cloudflared tunnel --url http://127.0.0.1:8743
```

Then point the Grok Bot connector at `https://<tunnel-host>/mcp` with the same bearer token. Prefer tunnels that terminate TLS. Do **not** bind `0.0.0.0` on a public interface without TLS and a strong token.

Toward a proper HTTPS service: run the same process behind any reverse proxy (Caddy, nginx, Traefik) that terminates TLS and forwards to `127.0.0.1:8743`. No Puppetmaster-hosted cloud is required or implied by this release.

## Auth model

| Piece | Behavior |
|---|---|
| Bearer token | Required. `Authorization: Bearer <token>` or `X-Puppetmaster-Token`. |
| Source | `--token`, `--token-file`, or `PUPPETMASTER_MCP_TOKEN`. Generated at startup if omitted. |
| Anonymous | Refused — there is no anonymous job control. |
| Origin | DNS-rebinding guard: missing Origin allowed (typical MCP clients); present Origin must match `--allow-origin` (or `*` / localhost defaults). |
| Rate limit | Default 120 req/min/IP (`--rate-limit`, `0` disables). |
| Audit | JSONL lines on stderr; optional `--audit-log PATH`. Records method/tool/peer/status — never the token. |

`/health` is intentionally unauthenticated and returns only liveness + version + scope (no job data).

## Tool scope (v1)

Default **`supervise`** — read-only / supervise-first:

| Allowed | Examples |
|---|---|
| Health / routing | `puppetmaster_doctor`, `puppetmaster_route_task`, `puppetmaster_list_models` |
| Start analysis | `puppetmaster_start_cursor_review`, `puppetmaster_start_cursor_plan`, `puppetmaster_start_cursor_swarm`, `puppetmaster_start_swarm`, `puppetmaster_start_prewalk` |
| Observe | `puppetmaster_status`, `puppetmaster_logs`, `puppetmaster_live_artifacts`, `puppetmaster_live_artifacts_follow`, `puppetmaster_partial_summary`, `puppetmaster_artifacts`, `puppetmaster_show`, `puppetmaster_await_job`, `puppetmaster_job_graph`, … |
| CodeGraph reads | `puppetmaster_codegraph_search`, `_context`, `_affected`, `_files`, `_status` |

**Omitted by default** (full-edit / side-effecting):

- `puppetmaster_start_implement` / `*_cursor_implement` / `*_claude_implement`
- `puppetmaster_edit`
- `puppetmaster_start_codex` / `_agentic` / `_openai` and their sync twins
- `puppetmaster_start_browser_swarm`
- `puppetmaster_reset_subgraph`, `puppetmaster_gc`
- `puppetmaster_codegraph_init` / `_index` / `puppetmaster_repair_codegraph`

Opt in explicitly:

```bash
python -m puppetmaster mcp serve-remote --scope implement
# or: PUPPETMASTER_MCP_REMOTE_SCOPE=implement
```

Treat `--scope implement` like giving a remote client the power to start full-edit workers on your machine. Prefer supervise + local CLI/stdio for implement until you trust the connector path.

## Pilot loop (same as Cursor Agent)

```text
1. puppetmaster_doctor
2. puppetmaster_start_cursor_swarm / _review / _plan  →  {job_id} immediately
3. puppetmaster_live_artifacts_follow (chain next_cursor)
   or puppetmaster_status / puppetmaster_logs
4. puppetmaster_partial_summary while running
5. puppetmaster_show when complete
6. Approve in chat before any implement (local CLI / stdio / --scope implement)
```

Do **not** hold one long MCP call open for a multi-minute swarm. Always use the `start_*` verbs and poll.

## Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/mcp` | POST | Streamable HTTP MCP (primary). JSON-RPC in, JSON (or SSE) out. |
| `/mcp` | GET | Optional idle SSE stream (`Accept: text/event-stream`). |
| `/mcp` | DELETE | End session (`Mcp-Session-Id`). |
| `/sse` | GET | Legacy HTTP+SSE (2024-11-05) session open. |
| `/message?sessionId=` | POST | Legacy HTTP+SSE client→server messages. |
| `/health` | GET | Liveness (no auth). |
| `/` | GET | Discovery JSON. |

## What is / is not in v1

**In**

- Streamable HTTP + legacy SSE wrapping the existing tool handlers
- Bearer auth, origin guard, rate limit, audit log hook
- Supervise-first scope with explicit implement opt-in
- CLI: `python -m puppetmaster mcp serve-remote` / `puppetmaster-mcp-remote`
- Local PoC + tunnel path; HTTPS via your reverse proxy

**Not in v1 (follow-ups)**

- Hosted Puppetmaster cloud / marketplace plugin packaging for Grok Bot
- A leased `grok-bot` worker adapter (blocked on a real dispatch API)
- Automatic Grok Bot UI config install (no local mcp.json equivalent yet)
- OAuth / per-user multi-tenant auth (single shared bearer for the operator)

## Security notes

Remote MCP **exposes job control** to whoever holds the token. That is stronger than "read a dashboard":

1. Bind `127.0.0.1` by default. Use a TLS-terminating tunnel or reverse proxy for off-box access.
2. Keep the bearer token in an env var or `chmod 600` file — never commit it.
3. Stay on `--scope supervise` until you need implement; approve edits the same way you would in Cursor Agent.
4. Rotate the token if a tunnel URL leaks.
5. Watch the stderr / `--audit-log` trail for unexpected peers or denied tools.

See also [SECURITY.md](SECURITY.md) — the "no remote control" claim now has this explicit, authenticated exception.

## Related docs

- [CURSOR_AGENT_MCP.md](CURSOR_AGENT_MCP.md) — full tool surface (stdio daily driver)
- [ADAPTERS.md](ADAPTERS.md) — worker adapters (Grok Bot is a *pilot*, not an adapter)
- [CLI_REFERENCE.md](CLI_REFERENCE.md) — `mcp serve-remote` flags
- [SECURITY.md](SECURITY.md) — threat model
