# Adapters

Puppetmaster adapters let different agent/tool providers plug into the same worker runtime.

Every adapter receives:

- `Task`
- `goal`
- `worker_id`

Every adapter returns structured `Artifact` objects.

## Built In

### `local`

Deterministic local artifacts for demos and contract tests.

### `shell`

Runs bounded shell commands and emits verification artifacts.

```json
{
  "role": "verify-runtime",
  "instruction": "Verify Python is available.",
  "adapter": "shell",
  "payload": {
    "command": ["python", "--version"],
    "timeout_seconds": 10
  }
}
```

### `cursor`

Runs a local Cursor SDK one-shot agent through `@cursor/sdk`.

Requirements:

- Node
- `npm install`
- `CURSOR_API_KEY`

```json
{
  "role": "repo-agent",
  "instruction": "Ask Cursor to inspect the repository.",
  "adapter": "cursor",
  "payload": {
    "prompt": "Inspect this repo and recommend the next engineering step.",
    "model": "default",
    "cwd": ".",
    "timeout_seconds": 300
  }
}
```

### `claude-code`

Runs the Claude Code CLI in non-interactive mode and allows real repository edits.

Requirements:

- Claude Code CLI installed as `claude`, or `CLAUDE_CODE_COMMAND` / `payload.executable` set.
- Claude Code authenticated locally.
- A reviewed workflow config, because this adapter can modify files.

Quick setup without a global install:

```bash
npx -y @anthropic-ai/claude-code --version
CLAUDE_CODE_COMMAND="npx -y @anthropic-ai/claude-code" puppetmaster claude "Review this repo" --permission-mode acceptEdits
```

If the adapter returns `failure=not_authenticated`, run Claude Code interactively and complete `/login`, then retry.
If it returns `failure=billing_or_quota`, Claude Code is authenticated but the configured billing/quota cannot run the request yet.
If it returns `failure=dirty_worktree`, run from a clean repo/worktree or set `payload.allow_dirty=true` after reviewing the existing diff.

Default permission mode is `acceptEdits`, which is intentionally edit-capable. Use `bypassPermissions` only in isolated worktrees or disposable sandboxes.

```json
{
  "role": "claude-implement",
  "instruction": "Use Claude Code to implement the requested change.",
  "adapter": "claude-code",
  "payload": {
    "prompt": "Implement the change and run the relevant tests.",
    "cwd": ".",
    "permission_mode": "acceptEdits",
    "allowed_tools": ["Read", "Edit", "Bash"],
    "timeout_seconds": 900
  }
}
```

If Claude Code edits tracked files, Puppetmaster records a `patch` artifact containing the resulting unified diff, changed files, base SHA, and revert guidance.

### `openai`

Calls the OpenAI Chat Completions API directly with `OPENAI_API_KEY` (or `OPENAI_BASE_URL` for compatible providers). Returns the same finding / risk / decision / verification artifact shape as the other adapters, plus `tokens_in` / `tokens_out` / `tokens_total` captured from the API's `usage` payload — the only adapter besides `codex` that gives you billing-grade token telemetry.

Requirements:

- `OPENAI_API_KEY`, or pass `payload.openai_api_key`.
- Optional: `OPENAI_BASE_URL` / `OPENAI_ORG_ID` / `payload.openai_organization`.

```json
{
  "role": "explore",
  "instruction": "Summarize the auth module.",
  "adapter": "openai",
  "payload": {
    "prompt": "Inspect the auth module and emit a finding per concrete risk.",
    "model": "gpt-5.4-mini",
    "cwd": ".",
    "timeout_seconds": 300
  }
}
```

### `codex`

Shells out to the official OpenAI Codex CLI (`codex exec --json`) — the OpenAI-side analog of the Claude Code CLI. The Codex CLI ships a real coding-agent loop (file edits, shell, search, tool use) on top of `gpt-5.5` / `gpt-5.4` / `gpt-5.4-mini`, so the adapter can act on the repo, not just answer.

This adapter is the most telemetry-rich of the four live adapters: it parses Codex's structured JSONL event stream (`--json`) and captures real `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_output_tokens`, and `thread_id` from `turn.completed.usage` into the verification artifact payload. The full event stream is spooled to a sidecar log so nothing is silently dropped.

Requirements:

- Codex CLI installed: `npm install -g @openai/codex` (binary: `codex`), or `CODEX_COMMAND` / `payload.executable` set to a custom path.
- Codex CLI authenticated locally. The simplest path is `printenv OPENAI_API_KEY | codex login --with-api-key` (or `codex login` for ChatGPT-account auth).
- A reviewed workflow config, because this adapter can modify files in the configured `cwd` under the default `workspace-write` sandbox.

For Puppetmaster MCP inside Codex, install selected credential env explicitly:

```bash
python -m puppetmaster install-codex-mcp --inherit-env OPENAI_API_KEY,CODEX_HOME
python -m puppetmaster install-codex-mcp --map-env CODEX_HOME=MY_CODEX_API_HOME
python -m puppetmaster install-codex-mcp --env-file ~/.config/puppetmaster/env.zsh
```

Keep secrets in a private env file (`chmod 600`) instead of inline MCP config. `CODEX_HOME` is Codex's canonical auth-home variable, not a Puppetmaster requirement; use `--map-env CODEX_HOME=YOUR_LOCAL_NAME` if your machine uses a different local variable name. `puppetmaster doctor --json` reports credential visibility and billing-source evidence without printing values, and Codex billing detection reads `$CODEX_HOME/auth.json` before falling back to `~/.codex/auth.json`.

Defaults are tuned for non-interactive automation: `approval_policy="never"`, `--sandbox workspace-write`, `--ephemeral`, `--skip-git-repo-check`. Use `payload.sandbox="read-only"` for explore / plan tasks that should never touch the worktree. Opt in to `payload.dangerously_bypass_approvals_and_sandbox=true` only when the surrounding environment is already externally sandboxed.

If the adapter returns `failure=not_authenticated`, run `printenv OPENAI_API_KEY | codex login --with-api-key` once.
If it returns `failure=dirty_worktree`, run from a clean tree, or set `payload.allow_dirty=true`, or downgrade to `payload.sandbox="read-only"`.
If it returns `failure=missing_cli`, install the CLI with `npm install -g @openai/codex`.

```json
{
  "role": "codex-implement",
  "instruction": "Use Codex to implement the requested change and run tests.",
  "adapter": "codex",
  "payload": {
    "prompt": "Implement the change and run the relevant tests.",
    "model": "gpt-5.4-mini",
    "cwd": ".",
    "sandbox": "workspace-write",
    "timeout_seconds": 900
  }
}
```

Like Claude Code, when Codex edits tracked files, Puppetmaster records a `patch` artifact alongside the verification artifact.

### `hermes`

Shells out to the NousResearch [Hermes](https://hermes-agent.nousresearch.com) CLI (`hermes chat`) — a personal AI agent with its own terminal, browser, memory, and skills. The adapter runs Hermes headlessly (`-q`/`--quiet`/`--cli`) as either an **analyze** worker (read-only findings) or a **full-edit** worker (`payload.mode="implement"`), mirroring the Claude Code / Codex subprocess, git-snapshot, sidecar-spool, and PATCH-attribution semantics.

Two Hermes quirks the adapter handles explicitly:

- **Process-group isolation.** Hermes kills its own process group on exit, so every run uses `start_new_session=True` — teardown can never reach the orchestrator parent.
- **Unreliable exit codes.** A non-zero exit after a successful edit is common (provider flakiness, pgroup teardown). Implement-mode success is therefore determined from the captured **git diff**, and analyze-mode success is parsed from stdout — not from the exit code alone.

Like the other CodeGraph-aware adapters, a Hermes worker gets task-relevant CodeGraph context auto-injected into its prompt when `.codegraph/` exists; the verification artifact's `evidence` then includes `context:codegraph`.

Requirements:

- Hermes CLI installed and on PATH (binary: `hermes`), or `HERMES_COMMAND` / `payload.executable` set to a custom path.
- Hermes authenticated with at least one inference provider (`~/.hermes/.env` API keys or `~/.hermes/auth.json` OAuth state). `puppetmaster doctor` reports Hermes credential visibility without printing values.

Register the Puppetmaster MCP inside Hermes so the Hermes agent can drive swarms:

```bash
python -m puppetmaster install-hermes-mcp
```

One command makes Hermes a **full auto-invocation host**: it registers the MCP server in `~/.hermes/config.yaml`, wires Hermes' native `pre_llm_call` / `pre_tool_call` shell hooks (so focused single edits auto-steer to `puppetmaster_edit` and broad work to a swarm), and installs the bundled `puppetmaster` skill into `~/.hermes/skills` (durable procedural knowledge — verb decision tree, CodeGraph-first flow, trust gate). All three steps are idempotent and non-destructive; an existing customized skill is left untouched unless you pass `--force`. `puppetmaster setup` does the same as part of first-run.

If the adapter returns `failure=missing_cli`, install Hermes or set `HERMES_COMMAND` / `payload.executable`.
If it returns `failure=dirty_worktree` in implement mode, run from a clean tree or set `payload.allow_dirty=true`.

```json
{
  "role": "hermes-analyze",
  "instruction": "Use Hermes to map the auth flow and emit evidenced findings.",
  "adapter": "hermes",
  "payload": {
    "cwd": ".",
    "toolsets": [],
    "timeout_seconds": 240
  }
}
```

When a Hermes implement worker edits tracked files, Puppetmaster records a `patch` artifact alongside the verification artifact.

## Adding A Provider

1. Implement a class with `run(task, goal, worker_id) -> list[Artifact]`.
2. Register it in `ADAPTERS` in `puppetmaster/adapters.py`.
3. Add `AdapterInfo` so `puppetmaster adapters` and `puppetmaster doctor` can surface setup requirements.
4. Add tests that prove provider failures become artifacts.

Provider adapters should avoid raw transcript dumps. Return claims, decisions, patches, risks, or verification results with evidence.
