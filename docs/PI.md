# Pi TUI/pilot package

**Pi is the TUI/pilot. Puppetmaster is the durable worker runtime.**

This is first-class harness support for the Pi coding agent (`@earendil-works/pi-coding-agent`). Pi is **not** a leased worker adapter. Do not invent a `pi` adapter and do not start `pi` as a Puppetmaster subprocess worker. There is no `grok-bot` adapter either; Grok Bot remains a remote-MCP pilot ([GROK_BOT.md](GROK_BOT.md)).

Pi today has no native MCP client, no sub-agents, and no plan mode. The bundled `@puppetmaster/pi-pilot` package (`pi-package` keyword + extension + skill + prompt) registers Puppetmaster MCP tools over **stdio** so Pi can start disposable jobs and read artifacts.
