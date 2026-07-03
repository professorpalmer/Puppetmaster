# Release Checklist

## Versioning

Puppetmaster follows [semver](https://semver.org/) from **1.0.0** onward:

- **MAJOR** (`2.0.0`) — a breaking change to the public surface (CLI verbs, MCP
  tool names/inputs, on-disk state layout).
- **MINOR** (`1.1.0`) — a backwards-compatible new feature.
- **PATCH** (`1.0.1`) — a backwards-compatible fix or docs-only change.

Bump both `pyproject.toml` `version` and `puppetmaster/__init__.py` `__version__`
together at release time (they must match). The `0.9.x` line (through `0.9.102`)
predates this policy and used the patch field as a running build counter — do not
resume that; every bump now carries semver meaning.

Use this before sharing Puppetmaster publicly.

- CI passes on the default branch.
- `python -m unittest discover -s tests -v` passes locally.
- `python -m puppetmaster doctor` has no unexpected warnings.
- `python -m puppetmaster crash-demo` completes.
- Cursor adapter smoke test has been run locally with a real `CURSOR_API_KEY`.
- README status language matches the actual stability level.
- No secrets appear in git status or docs.
- Demo commands in README still work from a clean clone.
