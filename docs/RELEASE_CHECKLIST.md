# Release Checklist

Use this before sharing Puppetmaster publicly.

- CI passes on the default branch.
- `python -m unittest discover -s tests -v` passes locally.
- `python -m puppetmaster doctor` has no unexpected warnings.
- `python -m puppetmaster crash-demo` completes.
- Cursor adapter smoke test has been run locally with a real `CURSOR_API_KEY`.
- README status language matches the actual stability level.
- No secrets appear in git status or docs.
- Demo commands in README still work from a clean clone.
