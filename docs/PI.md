# Pi TUI/pilot package

**Pi is the TUI/pilot. Puppetmaster is the durable worker runtime.**

This is first-class harness support for the Pi coding agent (`@earendil-works/pi-coding-agent`). Pi is **not** a leased worker adapter. Do not invent a `pi` adapter and do not start `pi` as a Puppetmaster subprocess worker. There is no `grok-bot` adapter either; Grok Bot remains a remote-MCP pilot ([GROK_BOT.md](GROK_BOT.md)).

Pi today has no native MCP client, no sub-agents, and no plan mode. The bundled `@puppetmaster/pi-pilot` package (`pi-package` keyword + extension + skill + prompt) registers Puppetmaster MCP tools over **stdio** so Pi can start disposable jobs and read artifacts.

## Install

Use install-pi-mcp or setup --platforms pi. Optional: pi install PATH_TO_PI_PACKAGE after the CLI is on PATH (Node 22).

install-pi-mcp is idempotent. It writes ~/.pi/agent/mcp.json (stdio MCP: python -m puppetmaster.mcp_server) and lists the bundled pi_package path in settings.json packages. Override with --path or PI_CODING_AGENT_DIR. Handshake unless --skip-handshake. --dry-run writes nothing. Restart Pi after install.

## Doctor

puppetmaster doctor includes pi-pilot: ok when CLI + package + MCP are present; warn when nothing is installed; error when the install is partial. Each non-ok state prints the exact fix.

## Pilot loop

1. Size: start_prewalk or route_task
2. Spawn one disposable job: start_implement or start_agentic
3. Recall: effort_index then show / artifact refs
4. Do not read worker transcripts
5. gc / nuke the finished job. Do not keep a worker warm.

Workers are cursor / claude-code / codex / hermes / antigravity / agentic — never pi.

## Uninstall

puppetmaster uninstall includes uninstall-pi-mcp.

## Tests

Hermetic: tests/test_pi_pilot_package.py (unittest, no pytest). Live E2E: tests/test_pi_e2e.py only when PI_LIVE_E2E=1, pi on PATH, and a provider key is visible. A skipped or refused live run is not a pass.
