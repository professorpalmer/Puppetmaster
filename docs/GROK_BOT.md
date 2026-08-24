# Grok Bot harness (remote MCP pilot)

**Grok Bot is the pilot/client. Puppetmaster is the durable worker runtime.**

This is first-class harness support for Cursor's Grok Bot sidebar assistant product — not a fake `grok-bot` worker adapter. Grok Bot does not publish a task-dispatch API today, so Puppetmaster does **not** lease Grok Bot as a subprocess worker. Do not invent an adapter named `grok-bot` until that API exists.

## Why remote MCP

Grok Bot can consume MCP connectors that are **remote HTTP/SSE (or streamable HTTP)** only. It cannot register local stdio MCP servers like `python -m puppetmaster.mcp_server`. Marketplace plugins work because they are remote. Therefore the existing stdio JSON-RPC MCP — which Cursor / Claude Desktop / Codex use — is unreachable from Grok Bot even when you are "in" the platform.

The remote transport exposes the **same** MCP tool handlers (leases, jobs, artifacts, stitching, routing) over streamable HTTP, with bearer auth and a supervise-first scope. Stdio MCP is unchanged and remains the daily driver for Cursor.

```text
Grok Bot (pilot; no Cursor SDK required)
  -> HTTPS/SSE connector (Authorization: Bearer …)
  -> puppetmaster mcp serve-remote   (streamable HTTP /mcp)
  -> same tool handlers as stdio MCP
  -> detached python -m puppetmaster job
  -> agentic worker subprocesses on this box (keys-only, in-process tool loop)
  -> SQLite events + artifacts + effort-index
  -> Grok Bot polls status / logs / live_artifacts / show / effort_index
```

**Contained topology (preferred for Grok Bot).** Run remote MCP and the
workers on the *same* Grok Bot box (or any Puppetmaster host that has only
agentic provider keys). There is no `grok-bot` adapter and no CreateAgent
fleet. `start_implement` / `start_prewalk` / `start_swarm` pick **agentic**
when Cursor is not installed and a provider key is visible
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
or `OPENROUTER_API_KEY`). Cursor remains preferred only when the SDK *and*
`CURSOR_API_KEY` are actually runnable.


## Quick start (local PoC)

```bash
# From a repo you want Puppetmaster to operate on:
export PUPPETMASTER_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m puppetmaster mcp serve-remote --scope supervise
# equivalent console script:
# puppetmaster-mcp-remote --scope supervise
# one-shot helper (prints connector + starts server):
# ./scripts/grok-bot-remote-poc.sh   # or: make grok-bot-poc
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

### AddMcpServer steps (Grok Bot)

Grok Bot only accepts a **remote** MCP URL (not a local `command`/`args` stdio block).

1. Start the remote server on loopback. For a first live PoC that matches curl’s
   full tool list, use implement + open Origin (tighten later):
   ```bash
   export PUPPETMASTER_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
   python -m puppetmaster mcp serve-remote \
     --host 127.0.0.1 --port 8743 \
     --scope implement \
     --allow-origin '*'
   ```
   Daily-driver default remains `--scope supervise` (no implement/edit).
2. Make it reachable (laptop PoC — TLS tunnel in a second terminal):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8743
   # note the https://….trycloudflare.com hostname
   ```
3. In Grok Bot → **Add MCP server** / connector, set **exactly**:
   - **URL:** `https://<tunnel-host>/mcp`
     Path **must** be `/mcp` (Streamable HTTP). Do **not** paste `/sse`,
     `/health`, or the bare tunnel host — those are not the MCP endpoint.
   - **Headers:** `Authorization: Bearer <same PUPPETMASTER_MCP_TOKEN>`
   - **Transport:** streamable HTTP (if the UI only says “SSE/HTTP”, still use
     the `/mcp` URL — the server speaks Streamable HTTP there and keeps legacy
     `/sse` only for older clients)
4. Confirm load: Grok Bot should show tools > 0 (50 with `--scope implement`,
   ~28 with supervise). Then in chat (agentic default — **not** Cursor swarm):
   - `puppetmaster_doctor` — healthy when Cursor is missing and agentic keys
     are visible (`grok-bot-path` ok)
   - `puppetmaster_start_agentic` or `puppetmaster_start_implement` → `{job_id}`
     (picker falls to agentic; does not fail solely because Cursor is missing)
   - `puppetmaster_effort_index` (same artifact / effort-index contract)
   - `puppetmaster_status` / `puppetmaster_live_artifacts_follow` / `puppetmaster_show`

   `puppetmaster_start_cursor_swarm` is opt-in for hosts that actually have
   the Cursor SDK. Do not use it as the lead Grok Bot demo.

Equivalent connector JSON:

```json
{
  "name": "puppetmaster",
  "url": "https://<tunnel-host>/mcp",
  "transport": "streamable-http",
  "headers": {
    "Authorization": "Bearer <PUPPETMASTER_MCP_TOKEN>"
  }
}
```

**Handshake notes (live-proven: Plugins `connected` / tools=50):** the remote
server (1) echoes any non-empty client `protocolVersion` (default only if
missing — no allowlist), (2) returns minimal capabilities
`{"tools":{"listChanged":false},"logging":{}}` with **no** `experimental` /
`instructions`, (3) returns `application/json` when Accept lists JSON
(including dual `application/json, text/event-stream`; SSE only when
event-stream is present without JSON), (4) exposes `Mcp-Session-Id` via CORS
`Access-Control-Expose-Headers`, (5) holds **GET `/mcp`** open as a long-lived
SSE stream with keepalive comments, and (6) serves a compacted `tools/list`
via `remote_tool_to_json` (description ≤280, property description ≤120,
simple schema fields, `additionalProperties: false`, required ⊆ properties).
CI locks that sequence in `tests/test_mcp_remote_e2e.py`
(`GrokBotHandshakeRegressionTests`).

If Plugins still flaps after a green tools/list, try `--scope supervise` first
(~28 tools, smaller payload) before `--scope implement` (50).

Do **not** bind `0.0.0.0` on a public interface without TLS and a strong token.
Toward a proper HTTPS service: terminate TLS on Caddy/nginx/Traefik and forward
to `127.0.0.1:8743`. No Puppetmaster-hosted cloud is required or implied.

CI covers the HTTP loop without a tunnel (`tests/test_mcp_remote_e2e.py`).

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
| Start analysis | `puppetmaster_start_agentic`, `puppetmaster_start_implement`, `puppetmaster_start_swarm`, `puppetmaster_start_review` (Cursor-specific swarm/review/plan verbs are opt-in when the SDK is installed) |
| Observe | `puppetmaster_status`, `puppetmaster_logs`, `puppetmaster_live_artifacts`, `puppetmaster_live_artifacts_follow`, `puppetmaster_partial_summary`, `puppetmaster_artifacts`, `puppetmaster_show`, `puppetmaster_await_job`, `puppetmaster_job_graph`, … |
| CodeGraph reads | `puppetmaster_codegraph_search`, `_context`, `_affected`, `_files`, `_status` |

**Omitted by default** (full-edit / side-effecting):

- `puppetmaster_start_implement` / `*_cursor_implement` / `*_claude_implement`
- `puppetmaster_edit`
- `puppetmaster_start_codex` / `_agentic` / `_openai` and their sync twins
- `puppetmaster_start_browser_swarm`
- `puppetmaster_start_prewalk`, `puppetmaster_dashboard`, `puppetmaster_mcp_cleanup`, `puppetmaster_gate`
- `puppetmaster_reset_subgraph`, `puppetmaster_gc`
- `puppetmaster_codegraph_init` / `_index` / `puppetmaster_repair_codegraph`

The supervise list is an explicit safe allowlist. New tools stay hidden until
they are reviewed for read-only remote use.

Opt in explicitly:

```bash
python -m puppetmaster mcp serve-remote --scope implement
# or: PUPPETMASTER_MCP_REMOTE_SCOPE=implement
```

Treat `--scope implement` like giving a remote client the power to start full-edit workers on your machine. Prefer supervise + local CLI/stdio for implement until you trust the connector path.

## Pilot loop (same as Cursor Agent)

```text
1. puppetmaster_doctor          (healthy Grok Bot path: no Cursor + agentic keys)
2. puppetmaster_start_agentic / puppetmaster_start_implement  →  {job_id}
3. puppetmaster_effort_index    (or live_artifacts_follow / status / logs)
4. puppetmaster_partial_summary while running
5. puppetmaster_show when complete
6. Approve in chat before further implement if you started on supervise scope
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


## Pi TUI/pilot (stdio; v1.22.23+)

Grok Bot is a *remote HTTP* pilot. **Pi** is a *local TUI* pilot. Neither is a
worker adapter — do not invent a `pi` or `grok-bot` leased subprocess.

Pi has no built-in MCP. Install the in-repo package:

    puppetmaster install-pi-mcp
    # or: setup --platforms pi
    # or: pi install PATH_TO_puppetmaster/pi_package

That writes `~/.pi/agent/mcp.json` to the same stdio launch other hosts use
and lists the package in `~/.pi/agent/settings.json`. The extension registers
Puppetmaster tools inside Pi. Prefer start_implement / start_agentic /
start_prewalk, then effort_index / show. Artifacts, not transcripts. Size the
runtime, then nuke the job.

Doctor `pi-pilot` is healthy when the Pi CLI is visible, the package is loaded,
and Puppetmaster MCP is reachable.

## OMP / oh-my-pi TUI/pilot (stdio; v1.22.31+)

Grok Bot is a *remote HTTP* pilot. **Pi** and **OMP** are *local TUI* pilots.
None is a worker adapter — do not invent a `pi`, `omp`, `ohmypi`, or
`grok-bot` leased subprocess.

OMP already speaks native MCP. Install:

    puppetmaster install-omp-mcp
    # or: setup --platforms omp
    # or: setup --platforms ohmypi

That writes `~/.omp/agent/mcp.json` to the same stdio launch other hosts use
(`python -m puppetmaster.mcp_server`). No TypeScript extension. Prefer
start_implement / start_agentic / start_prewalk, then effort_index / show.
Artifacts, not transcripts. Size the runtime, then nuke the job.

Doctor `omp-pilot` is optional when OMP is not installed (missing CLI is not a
failure), ok when MCP is present, and warn when the install is partial.

## Related docs

- [CURSOR_AGENT_MCP.md](CURSOR_AGENT_MCP.md) — full tool surface (stdio daily driver)
- [ADAPTERS.md](ADAPTERS.md) — worker adapters (Grok Bot is a *pilot*, not an adapter)
- [CLI_REFERENCE.md](CLI_REFERENCE.md) — `mcp serve-remote` flags
- [SECURITY.md](SECURITY.md) — threat model
- [examples/grok-bot-remote-e2e/](../examples/grok-bot-remote-e2e/) — shippable fixture cwd for Plugins review/implement e2e
