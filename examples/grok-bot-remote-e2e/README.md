# Grok Bot remote MCP e2e fixture

Tiny Node workspace used as the **target cwd** when validating Grok Bot
Plugins against Puppetmaster's authenticated remote MCP transport
(`python -m puppetmaster mcp serve-remote`).

It is not a product sample and does not define new MCP APIs. Connector setup,
scopes, and the pilot loop live in [docs/GROK_BOT.md](../../docs/GROK_BOT.md).

## Why this exists

A live Grok Bot box once used a scratch workdir that accidentally tracked
`node_modules`, lacked a `.gitignore`, and left implement markers untracked.
This fixture is the shippable replacement: ignore rules, a real `main`, a
green `npm test`, and tracked e2e markers.

## Setup

```bash
cd examples/grok-bot-remote-e2e
npm install --package-lock=false --no-audit
npm test
```

`@cursor/sdk` uses the same major range as the repo root (`^1.0.26`) so a
Cursor-adapter worker can resolve the SDK when this directory is the job cwd.

## Pairing with remote MCP

1. From the Puppetmaster repo (or any install), start remote MCP per
   [GROK_BOT.md](../../docs/GROK_BOT.md) — typically
   `python -m puppetmaster mcp serve-remote --scope supervise` (or
   `--scope implement` when testing full-edit), then expose `/mcp` via a TLS
   tunnel if Grok Bot is off-box.
2. In Grok Bot → Add MCP server: URL `https://<tunnel-host>/mcp`, header
   `Authorization: Bearer <PUPPETMASTER_MCP_TOKEN>`, streamable HTTP.
3. Point review/implement jobs at **this directory** as `cwd` (or clone/copy
   it). Markers:
   - [`GROK_BOT_E2E.md`](GROK_BOT_E2E.md) — baseline / review path OK
   - [`GROK_BOT_IMPLEMENT_OK.md`](GROK_BOT_IMPLEMENT_OK.md) — tracked
     placeholder the implement job may update

Do **not** commit `node_modules/`, `.env`, or `.puppetmaster/` state — all are
gitignored here.
