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

## Provider Stubs

`codex` is intentionally present as a stub. It returns structured `blocked` verification artifacts until a concrete provider integration is added.

That lets configs reference future providers without breaking the runtime contract.

## Adding A Provider

1. Implement a class with `run(task, goal, worker_id) -> list[Artifact]`.
2. Register it in `ADAPTERS` in `puppetmaster/adapters.py`.
3. Add `AdapterInfo` so `puppetmaster adapters` and `puppetmaster doctor` can surface setup requirements.
4. Add tests that prove provider failures become artifacts.

Provider adapters should avoid raw transcript dumps. Return claims, decisions, patches, risks, or verification results with evidence.

