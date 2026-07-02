# Daily Driver

Puppetmaster is meant to make repeated Cursor swarm work inspectable and replayable.

## Using it from the chat window (recommended)

If you live in the IDE chat and CLI isn't your thing, the recommended setup is to **let a cheap conversational model triage and delegate**: keep a fast/cheap model in the chat window for conversational asks, and instruct it to start Puppetmaster on anything that's real engineering (multi-file investigation, refactor, review, implementation). This is what the rules installed by `puppetmaster setup` already nudge the agent to do.

```text
You ── chat ──> cheap conversational model ──(real work?)──> Puppetmaster job
```

A one-line instruction you can drop in your agent's custom rules / system prompt:

```text
For any non-trivial coding, review, or investigation task, start Puppetmaster (a cursor swarm
or the appropriate verb), return the job id, then poll and summarize. Handle trivial edits and
conversational questions yourself without a job.
```

**Don't route every message through Puppetmaster.** It's intentionally *not* a conversational tool — a durable worker per chat turn would add cost and latency to the cheapest asks and fight the IDE's message routing. The win is the split: instant cheap chat for talk, the full durable machine for work. Context still compounds, because each delegated job leaves typed artifacts and promoted memory the next job reuses — you don't need to feed the chitchat through it to get that.

## Recommended Loop

```bash
python -m puppetmaster doctor
python -m puppetmaster cursor "Review the current repo and propose the next patch" --review --dry-run
python -m puppetmaster cursor "Plan the next implementation slice" --plan --dry-run
python -m puppetmaster claude "Implement the approved change and run focused tests" --permission-mode acceptEdits
python -m puppetmaster logs
python -m puppetmaster show $(python -m puppetmaster last)
python -m puppetmaster rerun
```

Use `--dry-run` for normal daily planning and review tasks. The Cursor adapter receives promoted memory from earlier runs, but it is instructed to verify retrieved claims before relying on them.

Use `puppetmaster claude` when you intentionally want Claude Code to make real edits. It defaults to Claude Code `acceptEdits` permission mode and records tracked diffs as Puppetmaster patch artifacts.

## Keys-only recipe (no external CLI)

When you have provider API keys but no Cursor/Claude/Codex/Hermes CLI installed, enable the `agentic` platform and drive work directly:

```bash
export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY
python -m puppetmaster platform enable agentic
python -m puppetmaster models discover --source agentic --write
python -m puppetmaster agentic "Audit the auth module for risks" --mode analyze --provider openai --model gpt-5.4-mini
python -m puppetmaster agentic "Fix the README typo" --mode implement --provider openai --model gpt-5.4-mini
```

No host restart, no MCP registration into an external agent — call `puppetmaster agentic` from the CLI or `puppetmaster_agentic` / `puppetmaster_start_agentic` from any MCP client that has Puppetmaster wired.

`puppetmaster cursor` and `puppetmaster claude` default to `--worker-mode inline` for daily-driver speed. That skips an extra Puppetmaster Python worker subprocess while preserving job state, task leases, artifacts, stitching, and the provider's own process boundary. Pass `--worker-mode subprocess` when strict worker process isolation matters more than latency.

For repeated local-role swarms, keep workers warm:

```bash
python -m puppetmaster daemon --roles explore architect implement redteam test
python -m puppetmaster run "Review this repo" --worker-mode daemon
```

The daemon watches running jobs, claims ready tasks by lease, and keeps processing until stopped. Use `--max-idle-seconds` or `--max-tasks` for bounded test/dev runs.

## Operator Gates

Puppetmaster records patch artifacts and approval events, but it does not auto-apply repository changes yet.

```bash
python -m puppetmaster diff
python -m puppetmaster approve <job_id>
python -m puppetmaster reject <job_id> --reason "Needs narrower scope"
```

This keeps the daily-driver workflow serious without handing write access to every worker by default.

## Run Management

```bash
python -m puppetmaster state
python -m puppetmaster jobs
python -m puppetmaster last
python -m puppetmaster status <job_id>
python -m puppetmaster logs [job_id]
python -m puppetmaster open [job_id]
python -m puppetmaster clean --completed
```

SQLite is the default backend. Runtime state is stored outside the repository by default so Puppetmaster jobs, logs, artifacts, and SQLite files do not inflate `git status`. Use `--state-dir .puppetmaster` or `PUPPETMASTER_STATE_DIR=.puppetmaster` only when you intentionally want repo-local state.

Use `--backend file` only when you want fully inspectable JSON state for debugging.
