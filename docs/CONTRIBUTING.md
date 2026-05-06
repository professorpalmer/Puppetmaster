# Contributing

Puppetmaster values small, inspectable changes.

## Local Checks

```bash
python -m unittest discover -s tests -v
python -m puppetmaster doctor
python -m puppetmaster run "Contributor smoke test" --config examples/enterprise-workflow.json
python -m puppetmaster crash-demo
```

## Patch Expectations

- Keep worker outputs structured.
- Add tests for runtime, store, adapter, or CLI behavior changes.
- Do not commit secrets, local state, or generated lock files.
- Prefer dry-run examples for provider-backed agents.

## Commit Style

Use concise imperative messages, for example `Harden worker failure handling`.
