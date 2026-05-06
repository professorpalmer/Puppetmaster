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

## Provider Stubs

`claude-code` and `codex` are intentionally present as stubs. They return structured `blocked` verification artifacts until a concrete provider integration is added.

That lets configs reference future providers without breaking the runtime contract.

## Adding A Provider

1. Implement a class with `run(task, goal, worker_id) -> list[Artifact]`.
2. Register it in `ADAPTERS` in `puppetmaster/adapters.py`.
3. Add `AdapterInfo` so `puppetmaster adapters` and `puppetmaster doctor` can surface setup requirements.
4. Add tests that prove provider failures become artifacts.

Provider adapters should avoid raw transcript dumps. Return claims, decisions, patches, risks, or verification results with evidence.

