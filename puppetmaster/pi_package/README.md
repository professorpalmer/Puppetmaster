# @puppetmaster/pi-pilot

Pi TUI/pilot package. **Pi is not a worker adapter.**
Do not lease `pi` as a subprocess. Pi stays the TUI.

The package registers Puppetmaster MCP tools over stdio
(`python -m puppetmaster.mcp_server`).

## Install

    puppetmaster install-pi-mcp
    # or
    puppetmaster setup --platforms pi

Both are idempotent. They write `~/.pi/agent/mcp.json` and list this
directory in `settings.json`. Then:

    pi install PATH_TO_THIS_PACKAGE   # optional refresh
    # restart pi / start a new session

## Skill

Prefer `start_implement` / `start_agentic` / `start_prewalk`, then
`effort_index` / `show`. Consume artifacts, not transcripts. Size the
runtime, then nuke the job.
