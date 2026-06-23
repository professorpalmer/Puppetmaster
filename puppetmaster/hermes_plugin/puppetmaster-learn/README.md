# puppetmaster-learn

A Hermes plugin that turns finished Puppetmaster swarms into reusable skills —
the Puppetmaster auto-/learn flywheel.

On `on_session_end`, it looks for the most recent **durable** Puppetmaster
swarm for the session's working directory (a job that reached `COMPLETE` and
produced at least one finding or patch artifact, finishing within ~30 minutes)
and distills it into a Hermes **skill candidate**.

## Opt-in

The plugin does nothing unless you opt in:

```sh
export PUPPETMASTER_LEARN=1   # also accepts: true, yes, on
```

## What it writes

It writes **candidates**, not live skills, to:

```
~/.hermes/skills-candidates/<YYYYMMDD>-<slug>/
  SKILL.md         # YAML frontmatter + distilled body, ready to review
  candidate.json   # provenance: job_id, goal, cwd, created_iso, source
```

(Honors `$HERMES_HOME` like the rest of Hermes.) Review a candidate, edit it,
and move it into `~/.hermes/skills/` yourself when you want it live. The plugin
never promotes candidates automatically.

## Guarantees

- **Best-effort.** All work runs in a background daemon thread and is wrapped
  so the hook can never raise or block session teardown.
- **No agent loop.** v1 only detects, distills deterministically, and logs —
  it never spawns an agent.
- **Idempotent.** A second session won't re-emit a candidate for a job that
  already has one.
- **Cross-interpreter safe.** The plugin runs in Hermes' Python and talks to
  Puppetmaster by shelling out to Puppetmaster's own interpreter; it never
  imports `puppetmaster`.
