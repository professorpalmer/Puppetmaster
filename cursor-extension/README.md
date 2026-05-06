# Puppetmaster Cursor Extension

This extension adds a Puppetmaster control panel to Cursor/VS Code.

It does not replace the runtime. It provides the integrated UI layer that calls the local `python -m puppetmaster` runtime from the current workspace.

## Features

- Activity-bar Puppetmaster panel.
- Provider key setup through Cursor secret storage.
- One-click `doctor`.
- Cursor review and plan dry runs.
- Claude Code full-edit runs.
- Last job, logs, and artifacts views.

## Development Install

From the repository root:

```bash
cd cursor-extension
npm run check
npx -y @vscode/vsce package --no-dependencies
```

Then in Cursor:

1. Open the command palette.
2. Run `Extensions: Install from VSIX...`.
3. Choose the generated `.vsix`.
4. Open the Puppetmaster icon in the activity bar.

## Required Runtime Setup

The extension runs `python -m puppetmaster` in the active workspace. For local development, open the Puppetmaster repo itself or install Puppetmaster into the Python environment used by `puppetmaster.pythonPath`.

Configure provider keys from the Puppetmaster panel. Keys are stored in Cursor secret storage, not in repo files.

## Notes

Cursor does not expose its model picker/control-panel internals as a general plugin API. This extension integrates Puppetmaster as a native activity-bar panel and command palette provider, which is the closest durable plugin surface available today.
