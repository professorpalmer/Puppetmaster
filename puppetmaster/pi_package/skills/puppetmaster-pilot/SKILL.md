---
name: puppetmaster-pilot
description: "Pilot Puppetmaster from Pi. Pi is the TUI, not a leased worker. Use for start_implement, start_agentic, start_prewalk, effort-index, show, and nuking finished jobs."
---

# Puppetmaster from Pi

Pi stays the TUI / head seat. Never lease `pi` as a Puppetmaster worker
adapter. Spawn disposable Puppetmaster jobs; consume artifacts, not
transcripts; size the runtime; nuke the job.

## Verbs

- Coupled multi-file feature: `puppetmaster_start_implement`
- Keys-only contained path: `puppetmaster_start_implement` with `adapter: "agentic"`
- Size / preflight the work: `puppetmaster_route_task`
- Recall: `puppetmaster_show`
- Status: `puppetmaster_status` / `puppetmaster_artifacts` with refs
- Done: `puppetmaster gc` the finished job. Do not keep workers alive.

Do not read worker transcripts. Prefer compact refs and `show`.
 
