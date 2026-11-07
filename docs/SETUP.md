# Setup

This page tells you how to install Puppetmaster, connect it to your agent
tool, run a first job, and remove it.

## Install

```bash
pipx install puppetmaster-ai     # or: pip install puppetmaster-ai
puppetmaster setup               # installs MCP tools, rules, and hooks
```

The PyPI package name is `puppetmaster-ai`. The import name, the CLI, and the
repository use `puppetmaster`.

`setup` is idempotent. It skips platforms that are not installed, and it
prints each change. It asks you to enable at least one adapter.

## Select platforms

To install for specified platforms only, use `--platforms`:

```bash
puppetmaster setup --platforms cursor
puppetmaster setup --platforms pi       # Pi as a pilot, not a worker adapter
puppetmaster setup --platforms omp      # OMP (oh-my-pi) as a pilot, not a worker adapter
```

- For CI, use `--platforms <comma-list>` or `--platforms all`.
- To add an adapter later, use `puppetmaster platform enable <name>`.

After setup, restart your agent tool (for example Cursor, Codex, Claude Code,
Antigravity, Hermes, Pi, or OMP). The tool then has the `puppetmaster_*` MCP
tools. Where the tool supports hooks, it also gets hooks that suggest
delegation for larger tasks. To disable those hooks, set
`PUPPETMASTER_AUTO_INVOKE_DISABLED=1`.

## Run without an agent CLI

The built-in `agentic` adapter needs only a provider API key. It does not need
an external CLI. For provider keys, Antigravity, and Hermes, see
[ADAPTERS.md](ADAPTERS.md).

Grok Bot does not use the stdio install. It uses remote MCP. See
[GROK_BOT.md](GROK_BOT.md).

## Run a first job

In your agent tool, send:

```text
Use Puppetmaster to run doctor in this repo and summarize what is missing.
```

For a supervised change, send a prompt with the problem, the constraints, and
an approval step:

```text
Use Puppetmaster to start a review for this repo on my configured reviewer platform and return the job id immediately.
Problem: users get logged out after refresh and token-refresh tests are flaky.
Constraints: keep the patch focused, preserve public API behavior, run relevant tests.
Do review/plan first. Poll status/logs by job id. Do not edit until you summarize findings and ask for approval.
```

From the shell:

```bash
puppetmaster doctor
puppetmaster route "Security audit every endpoint" --role audit
puppetmaster platform reviewer codex
puppetmaster review "Review this repo for release blockers"
puppetmaster show "$(puppetmaster last)"
puppetmaster artifacts <job_id>
python -m puppetmaster dashboard
```

For more prompts, see [DAILY_DRIVER.md](DAILY_DRIVER.md). For every command,
see [CLI_REFERENCE.md](CLI_REFERENCE.md).

## Verify a live route (optional)

To prove that an installed Codex route works, run:

```bash
puppetmaster setup --verify-first-run codex/<model>
```

Use an exact registry ID. This check makes one live call in temporary state.
It returns nonzero if it cannot prove delivery in 120 seconds. Ordinary setup
does not make this call. Details:
[CLI_REFERENCE.md](CLI_REFERENCE.md#verify-a-first-run).

## Uninstall

```bash
puppetmaster uninstall
pip uninstall puppetmaster-ai   # or: pipx uninstall puppetmaster-ai
```

`uninstall` removes the MCP entries, hooks, and rules that Puppetmaster owns.
It keeps `~/.puppetmaster/` and the workspace `.codegraph/` directory. To
remove them too, add `--purge-state`. To see the changes first, add
`--dry-run`.
