# OMP / oh-my-pi TUI/pilot

**OMP is the TUI/pilot. Puppetmaster is the durable worker runtime.**

This is first-class harness support for [oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`). OMP is **not** a leased worker adapter. Do not invent an `omp` / `ohmypi` adapter and do not start `omp` as a Puppetmaster subprocess worker. There is no `pi` or `grok-bot` worker either.

OMP already speaks native MCP, so Puppetmaster writes `~/.omp/agent/mcp.json` only. No TypeScript extension. No vendored oh-my-pi.

## Install

Use `puppetmaster install-omp-mcp` or `setup --platforms omp` (alias: `ohmypi`).

`install-omp-mcp` is idempotent. It writes `~/.omp/agent/mcp.json` (stdio MCP: `python -m puppetmaster.mcp_server`). Override with `--path` or `OMP_AGENT_DIR`. A named profile (`OMP_PROFILE` / `PI_PROFILE`) writes `~/.omp/profiles/<name>/agent/mcp.json`. Handshake unless `--skip-handshake`. `--dry-run` writes nothing. Restart OMP after install.

## Doctor

`puppetmaster doctor` includes `omp-pilot`: ok when MCP is present; optional when nothing is installed (missing `omp` CLI is not a failure); warn when the install is partial. Never errors solely because the CLI is absent.

## Pilot loop

1. Size: start_prewalk or route_task
2. Spawn one disposable job: start_implement or start_agentic
3. Recall: effort_index then show / artifact refs
4. Do not read worker transcripts
5. gc / nuke the finished job. Do not keep a worker warm.

Workers are cursor / claude-code / codex / hermes / antigravity / agentic — never omp.

## Uninstall

`puppetmaster uninstall` includes `uninstall-omp-mcp`.

## Tests

Hermetic: `tests/test_omp_pilot_package.py` (unittest, no pytest, temp HOME, no network).
