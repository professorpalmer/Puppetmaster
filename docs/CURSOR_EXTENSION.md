# Cursor Extension

Puppetmaster includes a Cursor/VS Code extension under `cursor-extension/`.

The extension is the integrated control panel layer:

- Configure provider keys through Cursor secret storage.
- Run `doctor`.
- Start Cursor review/plan dry runs.
- Start Claude Code full-edit jobs.
- Inspect latest job, logs, and artifacts without typing terminal commands.

## Install Locally

```bash
cd cursor-extension
npm run check
npx -y @vscode/vsce package --no-dependencies
```

Then install the generated `.vsix` in Cursor with `Extensions: Install from VSIX...`.

## Runtime Requirement

The extension shells out to:

```bash
python -m puppetmaster
```

That means Puppetmaster must be importable in the selected Python environment. When developing this repo, opening `/Users/cary/Desktop/Puppetmaster` in Cursor is enough. For normal users, the packaged runtime should be installed first once Puppetmaster is published.

For source-checkout development from another workspace, either install Puppetmaster into the Python Cursor uses:

```bash
python -m pip install -e /Users/cary/Desktop/Puppetmaster
```

Or set `puppetmaster.runtimePath` in Cursor settings:

```json
{
  "puppetmaster.runtimePath": "/Users/cary/Desktop/Puppetmaster"
}
```

## Control Panel Workflow

1. Open the Puppetmaster activity-bar icon.
2. Click `Configure Provider Keys`.
3. Run `Doctor`.
4. Use `Cursor Review Dry Run` or `Cursor Plan Dry Run`.
5. Use `Claude Code Implement` from a clean worktree for real edits.
6. Inspect `Show Logs` and `Show Artifacts`.

The extension stores keys in Cursor secret storage and passes them to the runtime process as environment variables.
