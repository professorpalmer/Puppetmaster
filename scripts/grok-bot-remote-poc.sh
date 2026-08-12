#!/usr/bin/env bash
# Local PoC helper: serve Puppetmaster remote MCP on loopback and print the
# cloudflared one-liner for a Grok Bot connector. Not used by CI.
#
# Usage:
#   ./scripts/grok-bot-remote-poc.sh
#   PUPPETMASTER_MCP_TOKEN=... PORT=8743 ./scripts/grok-bot-remote-poc.sh
#
# Then in another terminal (optional tunnel):
#   cloudflared tunnel --url "http://127.0.0.1:${PORT}"
# Point Grok Bot AddMcpServer at https://<tunnel-host>/mcp with Bearer token.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${PUPPETMASTER_MCP_REMOTE_HOST:-127.0.0.1}"
PORT="${PUPPETMASTER_MCP_REMOTE_PORT:-${PORT:-8743}}"
SCOPE="${PUPPETMASTER_MCP_REMOTE_SCOPE:-supervise}"

if [[ -z "${PUPPETMASTER_MCP_TOKEN:-}" ]]; then
  PUPPETMASTER_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export PUPPETMASTER_MCP_TOKEN
  echo "Generated PUPPETMASTER_MCP_TOKEN (export it to reconnect with the same token)." >&2
fi

echo "=== Grok Bot remote MCP PoC ===" >&2
echo "Listening: http://${HOST}:${PORT}/mcp  scope=${SCOPE}" >&2
echo >&2
echo "AddMcpServer / connector:" >&2
python -m puppetmaster mcp serve-remote \
  --host "$HOST" \
  --port "$PORT" \
  --token "$PUPPETMASTER_MCP_TOKEN" \
  --scope "$SCOPE" \
  --print-connector >&2
echo >&2
echo "Optional tunnel (separate terminal):" >&2
echo "  cloudflared tunnel --url \"http://${HOST}:${PORT}\"" >&2
echo "Then set the connector URL to https://<tunnel-host>/mcp (same Bearer token)." >&2
echo "Docs: docs/GROK_BOT.md" >&2
echo >&2

exec python -m puppetmaster mcp serve-remote \
  --host "$HOST" \
  --port "$PORT" \
  --token "$PUPPETMASTER_MCP_TOKEN" \
  --scope "$SCOPE" \
  "$@"
