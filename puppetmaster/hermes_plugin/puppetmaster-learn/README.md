# puppetmaster-learn

A Hermes plugin that turns a finished [Puppetmaster](https://github.com/professorpalmer/Puppetmaster)
swarm into a Hermes skill candidate.

Puppetmaster runs durable worker jobs (swarms and flow graphs) for an agent.
A swarm that finds something or produces a patch holds knowledge that is
worth keeping. This plugin keeps it: when a Hermes session ends, it writes
that swarm's goal and findings as a `SKILL.md` that you can review and promote.

## What it does

When the session ends (`on_session_finalize`: quit, `/new`, a `-z` run, or a
gateway reset), the plugin:

1. Asks the Puppetmaster CLI for the latest job of the session's working
   directory (`puppetmaster last --json`).
2. Continues only when that job is `complete`, finished during this session,
   produced findings or a patch, and has no candidate yet.
3. Reads the job goal (`puppetmaster status <job>`) and the stitched summary
   (`puppetmaster show <job>`).
4. Writes a candidate and tells you where it is:

```
~/.hermes/skills-candidates/<YYYYMMDD>-<slug>/
  SKILL.md         # frontmatter (name, description), goal, findings, source job
  candidate.json   # job_id, goal, cwd, created_iso, source
```

The plugin never writes a live skill. To use a candidate, review it, then run
`puppetmaster skills promote-candidate <slug>` (or move the folder into
`~/.hermes/skills/` yourself). `puppetmaster skills list-candidates` lists them.

## Install

You need Puppetmaster (`pip install puppetmaster-ai` or `uv tool install puppetmaster-ai`)
with the `puppetmaster` command on `PATH`.

```sh
hermes plugins install puppetmaster-learn
hermes plugins enable puppetmaster-learn
```

`puppetmaster setup` can also copy and enable the plugin for you.

If Hermes starts with a short `PATH` (for example from a desktop launcher), the
plugin uses the interpreter in the `mcp_servers.puppetmaster` entry of Hermes
`config.yaml`, if that file exists. If it finds no Puppetmaster, it does nothing.

## Disclosure

- **Child processes.** At session end it runs up to three Puppetmaster CLI
  commands (`last --json`, `status`, `show`) in the session's working
  directory, each with a 15 second limit. These commands only read job state.
- **Reads outside its own data.** It reads Puppetmaster's local job store
  (through the CLI) and the `mcp_servers.puppetmaster` entry of Hermes
  `config.yaml`. It reads no credentials.
- **Writes.** It writes only under `<hermes home>/skills-candidates/`. The
  home follows the active Hermes profile.
- **No network, no background process, no telemetry.** The plugin makes no
  network calls and starts no thread or daemon. It does not start an agent or
  a Puppetmaster job.
- **Failures are silent.** Any error ends the hook with no message. Session
  teardown continues.
