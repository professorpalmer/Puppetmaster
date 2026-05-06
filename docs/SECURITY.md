# Security

## Secrets

Never commit provider credentials. Use `.env.example` as a template and export `CURSOR_API_KEY` in your shell or secret manager.

If a key is pasted into a public place, rotate it before publishing the repository.

## Repository Writes

The default Cursor daily-driver posture is dry-run first. Cursor workers should emit findings, verification, and patch plans before a human approves the next step.

The `claude-code` adapter is different: it is a full-featured live adapter and can edit files through Claude Code. Run it in a clean working tree or isolated worktree when possible, review `git diff`, and keep permission mode at `acceptEdits` unless you intentionally want a broader sandbox escape hatch.

## Shell Adapter

The `shell` adapter runs local commands from workflow config. Treat config files as executable input and review them before running third-party workflows.

## Claude Code Adapter

The `claude-code` adapter shells out to your local Claude Code CLI. Treat prompts and workflow configs as code execution inputs. Puppetmaster records resulting diffs as artifacts, but Claude Code may already have changed files by the time artifacts are written.

## Reporting

Open a GitHub issue for security concerns until a dedicated disclosure channel exists. Do not include live secrets in issue text.
