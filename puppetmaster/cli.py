from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO

from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.installers import (
    CLAUDE_NEXT_STEPS_GUIDANCE,
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    HERMES_NEXT_STEPS_GUIDANCE,
    InstallResult,
    UninstallResult,
    ensure_cursor_sdk,
    install_claude_mcp,
    install_codex_mcp,
    install_cursor_mcp,
    install_hermes_mcp,
    resolve_claude_command,
    uninstall_claude_mcp,
    uninstall_codex_mcp,
    uninstall_cursor_mcp,
    uninstall_hermes_mcp,
)
from puppetmaster.rules import (
    VALID_TARGETS,
    RulesInstallResult,
    install_rules,
    uninstall_rules,
)
from puppetmaster.hook_installers import VALID_HOOK_TARGETS, install_hooks, uninstall_hooks
from puppetmaster.mcp_registry import (
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    summarize as registry_summarize,
)
from puppetmaster.redaction import redact_secrets
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.state import (
    find_state_dir_for_job,
    list_project_state_dirs,
    resolve_state_dir,
)
from puppetmaster.store_factory import create_store
from puppetmaster.stitcher import Stitcher
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec

# High-frequency lifecycle events that fire ~once/second per task/run and bury
# the real events in `logs` output. Collapsed to a one-line summary by default;
# `logs --all` (or an `--event-type` match) restores them.
_NOISY_LOG_EVENTS = {"task.lease_renewed", "run.heartbeat", "task.saved"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puppetmaster",
        description="Run independent agent workers that write structured artifacts.",
    )
    parser.add_argument(
        "--state-dir",
        help=(
            "Directory for jobs, streams, artifacts, locks, and promoted memory. "
            "Defaults to per-workspace app state outside the repository."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["file", "sqlite"],
        default="sqlite",
        help="Coordination backend.",
    )
    parser.add_argument(
        "--emit-job-id-early",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    subcommands = parser.add_subparsers(dest="command", required=True)

    def _add_routing_flags(adapter_parser: argparse.ArgumentParser) -> None:
        """Expose the model router on a direct adapter subcommand.

        Without these, a direct ``cursor``/``claude``/``codex``/``openai`` run
        bypassed routing entirely — no auto_route, no ROUTING artifact, no cost
        stamp — so "why this model / what did it cost" was unanswerable for
        anything but a swarm. ``--auto-route`` opts the single worker into the
        same router the swarms use, which stamps a ROUTING artifact.
        """
        group = adapter_parser.add_argument_group("routing")
        group.add_argument(
            "--auto-route",
            action="store_true",
            help="Let the router pick the model (emits a ROUTING artifact + cost stamp).",
        )
        group.add_argument(
            "--routing-policy",
            choices=["balanced", "cheap", "quality", "escalating"],
            help="Routing policy when --auto-route is set. Default: balanced.",
        )
        group.add_argument(
            "--max-cost-usd",
            type=float,
            help="Hard cap on the routed model's estimated USD/call.",
        )
        group.add_argument(
            "--min-capability",
            type=int,
            help="Force the routing capability floor (0..100).",
        )

    subcommands.add_parser("init", help="Create the local Puppetmaster state store.")
    subcommands.add_parser("state", help="Print the resolved Puppetmaster state directory.")
    doctor_parser = subcommands.add_parser("doctor", help="Check local runtime dependencies.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    subcommands.add_parser("adapters", help="List available worker adapters.")

    install_codex = subcommands.add_parser(
        "install-codex-mcp",
        help="Register Puppetmaster as an MCP server in the OpenAI Codex CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  puppetmaster install-codex-mcp --inherit-env OPENAI_API_KEY,CODEX_HOME\n"
            "  puppetmaster install-codex-mcp --map-env CODEX_HOME=MY_CODEX_API_HOME\n"
            "  puppetmaster install-codex-mcp --env-file ~/.config/puppetmaster/env.zsh\n"
            "\n"
            "Recommended: keep secrets in a private env file (`chmod 600`) rather "
            "than inline MCP JSON/TOML. Use --map-env when your local variable "
            "name differs from the provider's canonical variable."
        ),
    )
    install_codex.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing puppetmaster MCP entry even if it already matches.",
    )
    install_codex.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without modifying any config.",
    )
    install_codex.add_argument(
        "--skip-handshake",
        action="store_true",
        help="Do not spawn the MCP server to verify it responds before registering.",
    )
    install_codex.add_argument(
        "--codex",
        default=None,
        help="Override the `codex` CLI path (defaults to the first `codex` on PATH).",
    )
    install_codex.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set a selected MCP-server environment variable. Intended for "
            "non-secret values; repeat for multiple keys."
        ),
    )
    install_codex.add_argument(
        "--inherit-env",
        action="append",
        default=[],
        metavar="KEY[,KEY...]",
        help=(
            "Copy only these environment variables from the installer process "
            "into the MCP server environment."
        ),
    )
    install_codex.add_argument(
        "--env-file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Read selected MCP-server env from a shell-style env file, for "
            "example ~/.config/puppetmaster/env.zsh. File contents are never printed."
        ),
    )
    install_codex.add_argument(
        "--map-env",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help=(
            "Map a canonical MCP-server env key from a local installer env key, "
            "for example CODEX_HOME=MY_CODEX_API_HOME."
        ),
    )
    install_codex.add_argument(
        "--force-env",
        action="store_true",
        help="Allow requested env values to override existing puppetmaster MCP env keys.",
    )

    install_claude = subcommands.add_parser(
        "install-claude-mcp",
        help="Register Puppetmaster as a user-scope MCP server in Claude Code.",
    )
    install_claude.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing puppetmaster MCP entry even if it already matches.",
    )
    install_claude.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without modifying any config.",
    )
    install_claude.add_argument(
        "--skip-handshake",
        action="store_true",
        help="Do not spawn the MCP server to verify it responds before registering.",
    )
    install_claude.add_argument(
        "--claude",
        default=None,
        help=(
            "Override the Claude Code CLI command (defaults to `claude` on PATH, "
            "then CLAUDE_CODE_COMMAND; may be multi-word)."
        ),
    )

    install_hermes = subcommands.add_parser(
        "install-hermes-mcp",
        help="Register Puppetmaster as an MCP server in Hermes' config.yaml.",
    )
    install_hermes.add_argument(
        "--force",
        action="store_true",
        help="Replace any existing puppetmaster MCP entry even if it already matches.",
    )
    install_hermes.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without modifying any config.",
    )
    install_hermes.add_argument(
        "--skip-handshake",
        action="store_true",
        help="Do not spawn the MCP server to verify it responds before registering.",
    )
    install_hermes.add_argument(
        "--path",
        default=None,
        help="Explicit path to Hermes' config.yaml (defaults to $HERMES_HOME or ~/.hermes/config.yaml).",
    )

    install_cursor = subcommands.add_parser(
        "install-cursor-mcp",
        help="Register Puppetmaster as an MCP server in a Cursor mcp.json file.",
    )
    scope_group = install_cursor.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--global",
        dest="install_global",
        action="store_true",
        help="Install into ~/.cursor/mcp.json (visible in every workspace).",
    )
    scope_group.add_argument(
        "--workspace",
        dest="install_workspace",
        action="store_true",
        help="Install into <cwd>/.cursor/mcp.json (workspace-local). Default.",
    )
    install_cursor.add_argument(
        "--path",
        default=None,
        help="Explicit path to the Cursor mcp.json (overrides --global/--workspace).",
    )
    install_cursor.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the puppetmaster entry even if it already matches.",
    )
    install_cursor.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any file.",
    )
    install_cursor.add_argument(
        "--skip-handshake",
        action="store_true",
        help="Do not spawn the MCP server to verify it responds before registering.",
    )

    install_rules_parser = subcommands.add_parser(
        "install-rules",
        help="Write agent rule files (Cursor / AGENTS.md / Codex / Claude) that nudge hosts to reach for Puppetmaster on the right tasks.",
    )
    install_rules_parser.add_argument(
        "--target",
        default=None,
        help=(
            "Comma-separated subset of targets to install. "
            f"Valid: {', '.join(sorted(VALID_TARGETS))}. "
            "Default: auto-detect from cwd + tools on PATH."
        ),
    )
    install_rules_parser.add_argument(
        "--global",
        dest="rules_global",
        action="store_true",
        help="Also write user-level rules (~/.codex/instructions.md, ~/.claude/CLAUDE.md) when those tools are detected.",
    )
    install_rules_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-write rule blocks even when they already match the current version.",
    )
    install_rules_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any file.",
    )

    setup_parser = subcommands.add_parser(
        "setup",
        help="One-shot first-run: doctor + models init + install-cursor-mcp + install-codex-mcp + install-claude-mcp + install-hermes-mcp + install-rules. Skips steps where the tool isn't present.",
    )
    setup_parser.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip the opening doctor pass.",
    )
    setup_parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip `models init` (useful if you've already customized your registry).",
    )
    setup_parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Skip writing agent rule files.",
    )
    setup_parser.add_argument(
        "--skip-hooks",
        action="store_true",
        help="Skip installing deterministic auto-invocation hooks (.cursor/hooks.json, .claude/settings.json).",
    )
    setup_parser.add_argument(
        "--global-rules",
        action="store_true",
        help="Pass --global to install-rules.",
    )
    setup_parser.add_argument(
        "--global-hooks",
        action="store_true",
        help=(
            "Install user-level auto-invocation hooks (~/.cursor/hooks.json, "
            "~/.claude/settings.json) that cover every repo, instead of just this "
            "workspace."
        ),
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force through to MCP installers and rule installer.",
    )
    setup_parser.add_argument(
        "--platforms",
        default=None,
        help=(
            "Comma-separated platforms to enable out of the gate "
            "(e.g. 'cursor' or 'cursor,claude-code'); every other platform is "
            "disabled. Non-interactive — skips the platform prompt."
        ),
    )
    setup_parser.add_argument(
        "--skip-platforms",
        action="store_true",
        help="Skip the platform-lock selection step (leave the lock unchanged).",
    )

    uninstall_parser = subcommands.add_parser(
        "uninstall",
        help=(
            "Remove Puppetmaster host integrations (MCP registrations, hooks, rules) "
            "before `pip uninstall puppetmaster-ai`. Idempotent."
        ),
    )
    uninstall_parser.add_argument(
        "--cwd",
        default=".",
        help="Workspace root for project-scoped artifacts. Default: current dir.",
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every action without writing or deleting anything.",
    )
    uninstall_parser.add_argument(
        "--purge-state",
        action="store_true",
        help=(
            "Also remove ~/.puppetmaster/, <cwd>/.puppetmaster/, and <cwd>/.codegraph/. "
            "Left intact by default."
        ),
    )
    uninstall_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )

    init_config = subcommands.add_parser("init-config", help="Write a starter workflow config.")
    init_config.add_argument(
        "--path",
        default="puppetmaster.json",
        help="Destination path for the generated config.",
    )
    init_config.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file.",
    )

    run = subcommands.add_parser("run", help="Run a local swarm against a goal.")
    run.add_argument("goal", help="The swarm goal.")
    run.add_argument(
        "--effort",
        help="Tag this job with an effort id so it can be rolled up across "
        "worktrees later (sets PUPPETMASTER_EFFORT_ID for this run).",
    )
    run.add_argument(
        "--workers",
        nargs="+",
        help="Worker roles to run. Defaults to explore architect implement redteam test.",
    )
    run.add_argument("--config", help="Path to a Puppetmaster JSON workflow config.")
    run.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="subprocess",
        help="Use subprocess workers, inline workers, or wait for warm daemon workers.",
    )
    run.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection on all workers (fresh swarm perspective).",
    )
    run.add_argument(
        "--enable-memory",
        action="store_true",
        help="Force promoted shared-memory injection on all workers.",
    )

    jobs_parser = subcommands.add_parser("jobs", help="List known jobs.")
    jobs_parser.add_argument(
        "--all-projects",
        action="store_true",
        help=(
            "List jobs across every Puppetmaster project state dir on this "
            "machine instead of just the workspace's. Useful when you ran a "
            "swarm in one repo and want to find the job from another shell."
        ),
    )
    subcommands.add_parser("last", help="Print the most recent job id.")
    subcommands.add_parser(
        "projects",
        help=(
            "List every Puppetmaster project state dir on this machine with "
            "job counts and last activity. Helps you find which workspace "
            "owns a job without exporting PUPPETMASTER_STATE_DIR."
        ),
    )

    status = subcommands.add_parser("status", help="Show task, artifact, and stale lease state.")
    status.add_argument("job_id")
    status.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Omit high-churn prompt bodies from JSON output and replace them "
            "with deterministic char-count/SHA-256 refs."
        ),
    )

    watch = subcommands.add_parser("watch", help="Poll status for a job.")
    watch.add_argument("job_id")
    watch.add_argument("--interval", type=float, default=1.0)
    watch.add_argument("--ticks", type=int, default=5)

    recover = subcommands.add_parser("recover", help="Requeue stale leased tasks for a job.")
    recover.add_argument("job_id")

    events = subcommands.add_parser("events", help="Print the event stream for a job as JSON.")
    events.add_argument("job_id")

    logs = subcommands.add_parser("logs", help="Print event logs for a job, defaulting to the latest.")
    logs.add_argument("job_id", nargs="?")
    logs.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Include high-frequency heartbeat events (lease_renewed/run.heartbeat/"
        "task.saved). They are collapsed to a one-line summary by default.",
    )
    logs.add_argument(
        "--event-type",
        action="append",
        metavar="EVENT",
        help="Only show events whose name contains this substring. Repeatable; "
        "implies --all so a heartbeat type can be matched.",
    )

    feed = subcommands.add_parser("feed", help="Print live artifact feed for a job, defaulting to latest.")
    feed.add_argument("job_id", nargs="?")
    feed.add_argument("--limit", type=int, help="Limit feed to the most recent N artifacts.")
    feed.add_argument("--json", action="store_true", help="Print feed as JSON.")
    feed.add_argument(
        "--follow",
        action="store_true",
        help="Long-poll for new artifacts and stream them as they arrive.",
    )
    feed.add_argument(
        "--since",
        type=int,
        default=0,
        help="Resume the follow stream from this event cursor (defaults to 0).",
    )
    feed.add_argument(
        "--follow-timeout-seconds",
        type=float,
        default=0.0,
        help="Stop following after this many seconds of inactivity (0 = run until interrupted).",
    )
    feed.add_argument(
        "--follow-poll-seconds",
        type=float,
        default=0.1,
        help="Polling interval between cursor checks while following.",
    )

    open_cmd = subcommands.add_parser("open", help="Open or print a local job artifact path.")
    open_cmd.add_argument("job_id", nargs="?")
    open_cmd.add_argument(
        "--kind",
        choices=["summary", "state"],
        default="summary",
        help="Which path to open.",
    )

    show = subcommands.add_parser("show", help="Show the stitched or live summary for a job.")
    show.add_argument("job_id")
    show.add_argument(
        "--partial",
        action="store_true",
        help="Render a live summary from current artifacts without waiting for final stitching.",
    )

    finalize = subcommands.add_parser(
        "finalize",
        help=(
            "Force-stitch a job's artifacts into its summary and mark it "
            "complete. Use to recover a job whose orchestrator died before it "
            "could finalize (e.g. a wedged/stalled run)."
        ),
    )
    finalize.add_argument("job_id")

    reap = subcommands.add_parser(
        "reap",
        help=(
            "Scan running jobs and transition any whose orchestrator is dead "
            "(or wedged with no live lease) to 'stalled', requeuing their "
            "lease-expired tasks. Don't represent a dead job as running."
        ),
    )
    reap.add_argument(
        "--stall-after-seconds",
        type=int,
        default=None,
        help="No-progress window before a leaseless job is called stalled.",
    )
    reap.add_argument("--json", action="store_true", help="Emit JSON.")

    wait = subcommands.add_parser(
        "wait",
        help=(
            "Block until a job reaches a terminal state (complete/failed/"
            "stalled), then exit. Exit code is non-zero when the job did not "
            "complete cleanly. Runs the stalled-job reaper while waiting."
        ),
    )
    wait.add_argument("job_id")
    wait.add_argument(
        "--timeout",
        dest="timeout_seconds",
        type=float,
        default=0.0,
        help="Give up after N seconds (0 = block until terminal). Default: 0.",
    )
    wait.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.5,
        help="How often to re-check job state / run the reaper. Default: 0.5.",
    )
    wait.add_argument(
        "--stall-after-seconds",
        type=int,
        default=None,
        help="No-progress window before a leaseless job is called stalled.",
    )

    gc = subcommands.add_parser(
        "gc",
        help=(
            "Reap durable state for old terminal jobs (complete/failed/stalled) "
            "so per-project state dirs stop piling up. Dry-run by default."
        ),
    )
    gc.add_argument(
        "--older-than-days",
        type=float,
        default=7.0,
        help="Only reap terminal jobs finished more than N days ago. Default: 7.",
    )
    gc.add_argument(
        "--all-projects",
        action="store_true",
        help="Sweep every Puppetmaster project state dir on this machine.",
    )
    gc.add_argument(
        "--force",
        action="store_true",
        help="Actually delete. Without this, gc only reports what it would reap.",
    )
    gc.add_argument("--json", action="store_true", help="Emit JSON.")

    affected = subcommands.add_parser(
        "affected",
        help=(
            "Resolve which spec/test paths a set of changed files affects, using "
            "your own changed-file->spec mapping (declarative rules or a command). "
            "Puppetmaster supplies the blast radius; you supply the layout."
        ),
    )
    affected.add_argument(
        "--config",
        help="Path to a JSON mapping with 'rules' and/or 'command'.",
    )
    affected.add_argument(
        "--changed",
        nargs="*",
        help="Changed file paths. If omitted, read newline-separated paths from stdin.",
    )
    affected.add_argument(
        "--git-range",
        help="Derive changed files from `git diff --name-only <range>` (e.g. HEAD~1..HEAD).",
    )
    affected.add_argument(
        "--rule",
        action="append",
        default=[],
        help="Inline rule shorthand 'match=>spec1,spec2' (repeatable; merged with --config).",
    )
    affected.add_argument("--cwd", default=".", help="Repo root for globbing/command/git.")
    affected.add_argument("--json", action="store_true", help="Emit JSON instead of newline-separated paths.")

    rollup = subcommands.add_parser(
        "rollup",
        help=(
            "Aggregate jobs/artifacts/cost/tokens across many worktree state "
            "dirs for one logical effort. Tag jobs via PUPPETMASTER_EFFORT_ID "
            "or `run --effort`, then roll them up here."
        ),
    )
    rollup.add_argument(
        "--effort",
        help="Only include jobs tagged with this effort id. Omit to include all.",
    )
    rollup.add_argument(
        "--all-projects",
        action="store_true",
        help="Aggregate across every project state dir (the usual case for an "
        "effort that spanned multiple worktrees).",
    )
    rollup.add_argument("--json", action="store_true", help="Emit JSON.")

    gate = subcommands.add_parser(
        "gate",
        help=(
            "Replay the non-bypassable completion gates against a working tree, "
            "outside a worker run. Same engine the runtime enforces at task "
            "completion: require_diff / command oracle / monotonic ratchet / "
            "committed. Exits non-zero if any gate fails."
        ),
    )
    gate.add_argument(
        "--cwd",
        default=".",
        help="Working tree to evaluate gates against. Default: current dir.",
    )
    gate.add_argument(
        "--require-diff",
        action="store_true",
        help="Fail unless the tree has a non-empty diff (an edit happened).",
    )
    gate.add_argument(
        "--command",
        dest="gate_command",
        help="Oracle command; must exit 0 (e.g. the test/parity suite).",
    )
    gate.add_argument(
        "--ratchet-command",
        help="Command printing JSON metrics on stdout for the ratchet gate.",
    )
    gate.add_argument(
        "--metric",
        help="Metric key the ratchet enforces (monotonic; may only shrink).",
    )
    gate.add_argument(
        "--committed",
        action="store_true",
        help="Fail if the tree has uncommitted changes after the run.",
    )
    gate.add_argument(
        "--gates-json",
        help=(
            "Full gate spec as a JSON array of {kind,...} objects, for gates the "
            "convenience flags don't cover. Merged with any flags above."
        ),
    )
    gate.add_argument("--json", action="store_true", help="Emit JSON.")

    wait.add_argument("--json", action="store_true", help="Emit JSON.")
    wait.add_argument(
        "--summary",
        action="store_true",
        help="Also print the job summary when it finishes.",
    )

    dashboard_cmd = subcommands.add_parser(
        "dashboard",
        help=(
            "Serve a live, zero-dependency local web board for a job (or the "
            "job list) from durable state. No OTLP collector required."
        ),
    )
    dashboard_cmd.add_argument(
        "job_id",
        nargs="?",
        help="Job to open. Omit to land on the job list.",
    )
    dashboard_cmd.add_argument("--port", type=int, default=8787, help="Port (default 8787).")
    dashboard_cmd.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    dashboard_cmd.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open a browser tab.",
    )
    dashboard_cmd.add_argument(
        "--allow-external",
        action="store_true",
        help=(
            "Allow binding to a non-loopback host. The dashboard is "
            "unauthenticated, so this exposes job state to the network — only "
            "use on a trusted network."
        ),
    )
    dashboard_cmd.add_argument(
        "--all-projects",
        action="store_true",
        help="Show jobs from every Puppetmaster project state dir on this machine.",
    )

    await_cmd = subcommands.add_parser(
        "await",
        help=(
            "Block until a job reaches a terminal state (complete/failed), then "
            "print its final summary. True blocking await for the CLI/SDK path."
        ),
    )
    await_cmd.add_argument("job_id")
    await_cmd.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Give up after N seconds (0 = block until the job finishes). Default: 0.",
    )
    await_cmd.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.25,
        help="How often to re-check job state while blocked. Default: 0.25.",
    )
    await_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    artifacts = subcommands.add_parser("artifacts", help="Print artifacts for a job as JSON.")
    artifacts.add_argument("job_id")

    subcommands.add_parser("memory", help="Print promoted memory as JSON.")

    diff = subcommands.add_parser("diff", help="Print patch/diff artifacts for a job.")
    diff.add_argument("job_id", nargs="?")

    approve = subcommands.add_parser("approve", help="Approve patch artifacts for a job or artifact.")
    approve.add_argument("target", help="Job id or artifact id.")
    approve.add_argument(
        "--worktree",
        help="Optional existing worktree path where unified diffs should be applied.",
    )

    reject = subcommands.add_parser("reject", help="Reject patch artifacts for a job or artifact.")
    reject.add_argument("target", help="Job id or artifact id.")
    reject.add_argument("--reason", default="Rejected by operator.")

    rerun = subcommands.add_parser("rerun", help="Rerun the goal from a previous job.")
    rerun.add_argument("job_id", nargs="?")
    rerun.add_argument("--config", help="Path to a Puppetmaster JSON workflow config.")

    clean = subcommands.add_parser("clean", help="Delete completed or failed local job records.")
    clean.add_argument("--all", action="store_true", help="Delete all jobs.")
    clean.add_argument(
        "--completed",
        action="store_true",
        help="Delete complete and failed jobs only.",
    )

    daemon = subcommands.add_parser("daemon", help="Run warm workers that claim tasks from running jobs.")
    daemon.add_argument(
        "--roles",
        nargs="+",
        help="Optional task roles to claim. Defaults to any role.",
    )
    daemon.add_argument("--job-id", help="Optional job id to watch.")
    daemon.add_argument("--worker-id", help="Stable worker id prefix for this daemon.")
    daemon.add_argument("--lease-seconds", type=int, default=10)
    daemon.add_argument("--poll-seconds", type=float, default=0.25)
    daemon.add_argument(
        "--max-tasks",
        type=int,
        help="Exit after processing this many tasks. Useful for tests and one-shot warm workers.",
    )
    daemon.add_argument(
        "--max-idle-seconds",
        type=float,
        help="Exit after being idle this long. Omit to keep the daemon running.",
    )

    cursor = subcommands.add_parser("cursor", help="Run a Cursor daily-driver one-shot worker.")
    cursor.add_argument("prompt", help="Prompt for the Cursor worker.")
    cursor.add_argument("--cwd", default=str(Path.cwd()), help="Workspace for the Cursor SDK agent.")
    cursor.add_argument("--model", default="default")
    cursor.add_argument("--timeout-seconds", type=int, default=600)
    cursor.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="Cursor daily-driver runs default to inline orchestration to avoid an extra Python worker cold start.",
    )
    cursor.add_argument(
        "--no-edit",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Analysis-only: produce findings/plan/review artifacts but don't modify "
        "the working tree. The deliverable is still produced — '--dry-run' is the "
        "legacy alias and is a misnomer for plan/review, whose output IS the point.",
    )
    cursor.add_argument(
        "--review",
        action="store_true",
        help="Ask Cursor for repo findings, risks, and verification suggestions.",
    )
    cursor.add_argument(
        "--plan",
        action="store_true",
        help="Ask Cursor for decisions and a task graph without implementation.",
    )
    cursor.add_argument(
        "--implement",
        action="store_true",
        help="Full-edit mode: let the Cursor agent modify files and capture the diff as a PATCH artifact.",
    )
    cursor.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an --implement run to start in a dirty working tree.",
    )
    cursor.add_argument(
        "--allow-non-worktree",
        action="store_true",
        help="Allow an --implement run outside a git work tree (no diff attribution).",
    )
    cursor.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection for a fresh perspective.",
    )
    _add_routing_flags(cursor)

    claude = subcommands.add_parser("claude", help="Run a full-featured Claude Code worker.")
    claude.add_argument("prompt", help="Prompt for the Claude Code worker.")
    claude.add_argument("--cwd", default=str(Path.cwd()), help="Workspace for Claude Code.")
    claude.add_argument("--model", help="Claude model name to pass through.")
    claude.add_argument(
        "--permission-mode",
        default="acceptEdits",
        help="Claude Code permission mode. Defaults to acceptEdits for real repo edits.",
    )
    claude.add_argument("--allowed-tools", help="Comma-separated Claude Code allowed tools.")
    claude.add_argument("--disallowed-tools", help="Comma-separated Claude Code disallowed tools.")
    claude.add_argument("--executable", help="Claude Code executable or command.")
    claude.add_argument("--timeout-seconds", type=int, default=900)
    claude.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="Claude daily-driver runs default to inline orchestration while Claude Code remains a separate process.",
    )
    claude.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow Claude Code to run in a dirty working tree.",
    )
    claude.add_argument(
        "--allow-non-worktree",
        action="store_true",
        help="Allow Claude Code to run outside a git work tree (no diff attribution).",
    )
    claude.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection for a fresh perspective.",
    )
    _add_routing_flags(claude)

    openai = subcommands.add_parser(
        "openai",
        help="Run an OpenAI worker via the Chat Completions API (uses OPENAI_API_KEY).",
    )
    openai.add_argument("prompt", help="Prompt for the OpenAI worker.")
    openai.add_argument(
        "--cwd",
        default=str(Path.cwd()),
        help="Workspace path used for CodeGraph context enrichment.",
    )
    openai.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model id (gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, ...).",
    )
    openai.add_argument(
        "--base-url",
        help="Override the OpenAI base URL (e.g. for OpenAI-compatible providers).",
    )
    openai.add_argument(
        "--organization",
        help="Optional OpenAI organization id.",
    )
    openai.add_argument(
        "--max-output-tokens",
        type=int,
        help="Cap on completion tokens. Off by default to let the model finish.",
    )
    openai.add_argument(
        "--legacy-max-tokens",
        action="store_true",
        help=(
            "Send the deprecated `max_tokens` field instead of `max_completion_tokens`. "
            "Use for OpenAI-compatible providers that haven't migrated yet."
        ),
    )
    openai.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature override (only sent if provided).",
    )
    openai.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level for GPT-5+ models.",
    )
    openai.add_argument("--timeout-seconds", type=int, default=300)
    openai.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="OpenAI daily-driver runs default to inline orchestration.",
    )
    openai.add_argument(
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo prompts).",
    )
    openai.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection for a fresh perspective.",
    )
    _add_routing_flags(openai)

    codex = subcommands.add_parser(
        "codex",
        help="Run a full-featured Codex CLI worker (codex exec --json).",
    )
    codex.add_argument("prompt", help="Prompt for the Codex worker.")
    codex.add_argument("--cwd", default=str(Path.cwd()), help="Workspace for Codex.")
    codex.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Model passed to `codex exec -m` (gpt-5.5, gpt-5.4, gpt-5.4-mini, ...).",
    )
    codex.add_argument(
        "--sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
        help="Codex sandbox mode. Defaults to workspace-write for real repo edits.",
    )
    codex.add_argument(
        "--approval-policy",
        default="never",
        help="Codex approval policy (codex -c approval_policy=...). Defaults to 'never' for non-interactive automation.",
    )
    codex.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help="Disable Codex's sandbox AND approval prompts. Only safe when the surrounding environment is externally sandboxed.",
    )
    codex.add_argument("--executable", help="Override the codex executable / command.")
    codex.add_argument("--timeout-seconds", type=int, default=900)
    codex.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="Codex daily-driver runs default to inline orchestration while Codex remains a separate process.",
    )
    codex.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow Codex to run in a dirty working tree.",
    )
    codex.add_argument(
        "--allow-non-worktree",
        action="store_true",
        help="Allow Codex to run outside a git work tree (no diff attribution).",
    )
    codex.add_argument(
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo prompts).",
    )
    codex.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection for a fresh perspective.",
    )
    _add_routing_flags(codex)

    hermes = subcommands.add_parser(
        "hermes",
        help="Run a NousResearch Hermes CLI worker (hermes chat) headlessly.",
    )
    hermes.add_argument("prompt", help="Prompt for the Hermes worker.")
    hermes.add_argument("--cwd", default=str(Path.cwd()), help="Workspace for Hermes.")
    hermes.add_argument(
        "--mode",
        choices=["implement", "analyze"],
        default="implement",
        help="implement = full-edit with git-diff PATCH attribution; analyze = read-only structured findings.",
    )
    hermes.add_argument(
        "--model",
        help="Model passed to `hermes chat -m` (e.g. gemini-2.5-flash, claude-sonnet-4-5, gpt-5).",
    )
    hermes.add_argument(
        "--provider",
        help="Hermes provider (e.g. gemini, anthropic, openai-api). Routes credentials/wire protocol.",
    )
    hermes.add_argument(
        "--max-turns",
        type=int,
        help="Cap on Hermes tool-use iterations (`hermes chat --max-turns`).",
    )
    hermes.add_argument(
        "--toolsets",
        help="Override the comma-separated Hermes toolsets. Defaults exclude memory/session_search for worker isolation.",
    )
    hermes.add_argument("--executable", help="Override the hermes executable / command.")
    hermes.add_argument("--timeout-seconds", type=int, default=900)
    hermes.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="Hermes runs default to inline orchestration while Hermes remains a separate process.",
    )
    hermes.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow Hermes to run in a dirty working tree.",
    )
    hermes.add_argument(
        "--allow-non-worktree",
        action="store_true",
        help="Allow Hermes to run outside a git work tree (no diff attribution).",
    )
    hermes.add_argument(
        "--use-hermes-rules",
        action="store_true",
        help="Opt OUT of worker isolation: let Hermes inject its own AGENTS.md/memory and expose the memory tool. Off by default so workers stay hermetic.",
    )
    hermes.add_argument(
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo prompts).",
    )
    _add_routing_flags(hermes)

    demo = subcommands.add_parser("demo", help="Run the Puppetmaster concept demo.")
    demo.add_argument(
        "--goal",
        default="Build Puppetmaster: agentic swarms meets Redis/Gunicorn.",
    )

    crash_demo = subcommands.add_parser(
        "crash-demo",
        help="Run a subprocess swarm where one worker crashes and the task is recovered.",
    )
    crash_demo.add_argument(
        "--goal",
        default="Prove Puppetmaster can recover abandoned agent work.",
    )
    crash_demo.add_argument("--crash-role", default="implement")

    repair = subcommands.add_parser(
        "repair-codegraph",
        help=(
            "Rebuild CodeGraph's better-sqlite3 native module for Cursor's bundled "
            "Node so the MCP server stops falling back to slow WASM SQLite."
        ),
    )
    repair.add_argument(
        "--cursor-node",
        help="Path to Cursor's bundled Node binary. Auto-detected if omitted.",
    )
    repair.add_argument(
        "--codegraph-install",
        help=(
            "Path to the global @colbymchenry/codegraph install directory. "
            "Resolved from `npm root -g` if omitted."
        ),
    )
    repair.add_argument(
        "--npm-command",
        default="npm",
        help="npm binary to use (default: npm).",
    )
    repair.add_argument(
        "--rebuild-timeout-seconds",
        type=int,
        default=180,
        help="Hard timeout for the rebuild step (default: 180).",
    )
    repair.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the post-rebuild `codegraph status` check.",
    )
    repair.add_argument(
        "--verify-cwd",
        help=(
            "Target repo to run the verification `codegraph status` in. "
            "Defaults to the current working directory."
        ),
    )
    repair.add_argument(
        "--json",
        action="store_true",
        help="Print the full repair payload as JSON instead of a human-readable summary.",
    )

    codegraph_cmd = subcommands.add_parser(
        "codegraph",
        help=(
            "ABI-safe passthrough to the codegraph CLI. Runs under Cursor's "
            "bundled Node (not your shell's Node) so better-sqlite3 loads, and "
            "auto-rebuilds it on a Node ABI mismatch. Use this instead of a bare "
            "`codegraph ...` call from your shell."
        ),
    )
    codegraph_cmd.add_argument(
        "--cwd",
        help="Target repo to run codegraph in (default: current directory).",
    )
    codegraph_cmd.add_argument(
        "--timeout",
        type=int,
        default=0,
        help=(
            "Seconds before the codegraph call is killed. 0 (default) means no "
            "limit, so long operations like `index` / `affected` aren't cut off."
        ),
    )
    codegraph_cmd.add_argument(
        "cg_args",
        nargs=argparse.REMAINDER,
        help=(
            "codegraph subcommand and arguments, e.g. `search foo`, `status`, "
            "`context 'task'`. Prefix with `--` to pass codegraph's own flags, "
            "e.g. `codegraph -- --version`."
        ),
    )

    mcp = subcommands.add_parser(
        "mcp",
        help=(
            "Inspect or clean up tracked Puppetmaster MCP servers. Use after a "
            "`Tool execution error. Not connected` to see if orphan servers are left over."
        ),
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)

    mcp_list = mcp_sub.add_parser(
        "list",
        help="List every Puppetmaster MCP server tracked on this machine.",
    )
    mcp_list.add_argument(
        "--json",
        action="store_true",
        help="Print the full registry payload as JSON.",
    )

    mcp_doctor = mcp_sub.add_parser(
        "doctor",
        help=(
            "Diagnose a `Tool execution error. Not connected`: distinguish "
            "'daemon healthy, stdio pipe dropped (restart the MCP client)' from "
            "'no MCP server alive'."
        ),
    )
    mcp_doctor.add_argument("--json", action="store_true", help="Emit JSON.")

    mcp_cleanup = mcp_sub.add_parser(
        "cleanup",
        help=(
            "Prune dead tracking files; with --kill-stale, terminate stale-but-alive "
            "MCP servers whose parent client is gone."
        ),
    )
    mcp_cleanup.add_argument(
        "--kill-stale",
        action="store_true",
        help="SIGTERM (then SIGKILL after grace) stale-but-alive Puppetmaster MCP servers.",
    )
    mcp_cleanup.add_argument(
        "--stale-after-seconds",
        type=int,
        default=300,
        help="Heartbeat age that defines a stale server (default: 300).",
    )
    mcp_cleanup.add_argument(
        "--json",
        action="store_true",
        help="Print the before/after registry payload as JSON.",
    )

    models_cmd = subcommands.add_parser(
        "models",
        help=(
            "Manage the LLM model registry the router uses. Lives at "
            "~/.puppetmaster/models.json by default (override with "
            "$PUPPETMASTER_MODELS_PATH)."
        ),
    )
    models_sub = models_cmd.add_subparsers(dest="models_command", required=True)
    models_init = models_sub.add_parser(
        "init",
        help="Write a starter models.json with claude-code + cursor entries.",
    )
    models_init.add_argument("--registry-path", help="Override the registry path.")
    models_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing registry file.",
    )
    models_list = models_sub.add_parser(
        "list",
        help="Print the model registry.",
    )
    models_list.add_argument("--registry-path", help="Override the registry path.")
    models_list.add_argument("--json", action="store_true", help="Emit JSON.")
    models_path = models_sub.add_parser(
        "path",
        help="Print the resolved model registry path.",
    )
    models_path.add_argument("--registry-path", help="Override the registry path.")
    models_discover = models_sub.add_parser(
        "discover",
        help=(
            "Enumerate the Cursor plan's model catalog and reconcile it into the "
            "registry as plan-billed entries (requires CURSOR_API_KEY + node)."
        ),
    )
    models_discover.add_argument("--registry-path", help="Override the registry path.")
    models_discover.add_argument(
        "--source",
        choices=["cursor", "openai", "anthropic", "claude", "codex", "hermes", "all"],
        default="cursor",
        help=(
            "Which platform catalog to enumerate. cursor (plan, default), "
            "openai (GET /v1/models, needs OPENAI_API_KEY), anthropic "
            "(needs ANTHROPIC_API_KEY for discovery), claude / codex (curated "
            "catalogs for the CLI agent loops that can't self-enumerate; billed "
            "as your detected subscription/API posture), hermes (curated "
            "multi-provider catalog, API-billed via your own keys), or all."
        ),
    )
    models_discover.add_argument(
        "--write",
        action="store_true",
        help="Persist the merged registry (default: dry-run, just print the diff).",
    )
    models_discover.add_argument("--json", action="store_true", help="Emit JSON.")
    models_setup = models_sub.add_parser(
        "setup",
        help="Interactively manage the model registry without editing JSON.",
    )
    models_setup.add_argument("--registry-path", help="Override the registry path.")
    models_set = models_sub.add_parser(
        "set",
        help="Update one registry entry with key=value assignments.",
    )
    models_set.add_argument("--registry-path", help="Override the registry path.")
    models_set.add_argument("model_id", help="Model id to update.")
    models_set.add_argument("assignments", nargs="+", help="key=value updates.")

    platform_cmd = subcommands.add_parser(
        "platform",
        help=(
            "Restrict which platforms (adapters) Puppetmaster may use. Lock to "
            "the one plan you pay for, or toggle several on to bounce tiers."
        ),
    )
    platform_sub = platform_cmd.add_subparsers(dest="platform_command", required=True)
    platform_status = platform_sub.add_parser(
        "status", help="Show which platforms are enabled or disabled."
    )
    platform_status.add_argument("--registry-path", help="Override the registry path.")
    platform_status.add_argument("--json", action="store_true", help="Emit JSON.")
    platform_only = platform_sub.add_parser(
        "only", help="Enable exactly these platforms; disable all others."
    )
    platform_only.add_argument("adapters", nargs="+", help="cursor, claude-code, codex, openai")
    platform_only.add_argument("--registry-path", help="Override the registry path.")
    platform_enable = platform_sub.add_parser(
        "enable", help="Turn these platforms back on."
    )
    platform_enable.add_argument("adapters", nargs="+", help="cursor, claude-code, codex, openai")
    platform_enable.add_argument("--registry-path", help="Override the registry path.")
    platform_disable = platform_sub.add_parser(
        "disable", help="Turn these platforms off (never routed or discovered)."
    )
    platform_disable.add_argument("adapters", nargs="+", help="cursor, claude-code, codex, openai")
    platform_disable.add_argument("--registry-path", help="Override the registry path.")
    platform_reset = platform_sub.add_parser(
        "reset", help="Clear the lock; enable every platform again."
    )
    platform_reset.add_argument("--registry-path", help="Override the registry path.")

    preflight_cmd = subcommands.add_parser(
        "preflight",
        help=(
            "Check whether an adapter can actually run before dispatch: auth "
            "present, plan-vs-api billing, and (Cursor) model in the live catalog."
        ),
    )
    preflight_cmd.add_argument(
        "adapter",
        help="Adapter to check (cursor, claude-code, codex, openai).",
    )
    preflight_cmd.add_argument("--model", help="Model id to validate (Cursor catalog).")
    preflight_cmd.add_argument(
        "--no-api-billing",
        action="store_true",
        help="Treat api-billed adapters as blocked (plan-only).",
    )
    preflight_cmd.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run a real 1-token probe to catch a funded-looking account whose "
            "balance is actually exhausted (adds a real call + latency)."
        ),
    )
    preflight_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    audit_cmd = subcommands.add_parser(
        "audit",
        help=(
            "Analyze how routing actually behaved across past jobs and propose "
            "(never auto-apply) conservative capability-score adjustments."
        ),
    )
    audit_cmd.add_argument(
        "--window",
        type=float,
        default=None,
        metavar="DAYS",
        help="Only consider jobs created within this many days (default: all).",
    )
    audit_cmd.add_argument("--registry-path", help="Override the registry path.")
    audit_cmd.add_argument(
        "--apply",
        action="store_true",
        help="Write the suggested score changes to models.json (default: dry-run).",
    )
    audit_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    savings_cmd = subcommands.add_parser(
        "savings",
        help=(
            "Cumulative savings receipt: routing dollars saved (measured, "
            "policy-aware) + CodeGraph exploration savings. Read-only, local."
        ),
    )
    savings_cmd.add_argument(
        "--window",
        type=float,
        default=None,
        metavar="DAYS",
        help="Only count jobs/queries from the last N days (default: all time).",
    )
    savings_cmd.add_argument(
        "--all-projects",
        action="store_true",
        help="Aggregate across every workspace's state dir (default: just this one).",
    )
    savings_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    route_cmd = subcommands.add_parser(
        "route",
        help=(
            "Show which model the router would pick for an instruction, "
            "including estimated cost and rejected alternatives."
        ),
    )
    route_cmd.add_argument("instruction", help="The task instruction text.")
    route_cmd.add_argument(
        "--role",
        default="explore",
        help="Task role (drives the capability classifier). Default: explore.",
    )
    route_cmd.add_argument(
        "--policy",
        default="balanced",
        choices=["balanced", "cheap", "quality", "escalating"],
        help="Routing policy. Default: balanced.",
    )
    route_cmd.add_argument(
        "--min-capability",
        type=int,
        help="Force the classifier output to this value (0..100).",
    )
    route_cmd.add_argument(
        "--max-cost-usd",
        type=float,
        help="Hard cap on estimated USD cost per call.",
    )
    route_cmd.add_argument(
        "--required-tag",
        action="append",
        default=[],
        help="Filter to models whose tags include this. Repeat for multiple.",
    )
    route_cmd.add_argument(
        "--registry-path",
        help="Override the registry path.",
    )
    route_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    cost_cmd = subcommands.add_parser(
        "cost",
        help=(
            "Sum the estimated USD cost of router decisions for a job. "
            "Reads ROUTING artifacts that the orchestrator wrote at task "
            "creation. Auto-pivots across project state dirs."
        ),
    )
    cost_cmd.add_argument("job_id", help="Puppetmaster job id.")
    cost_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    delegate_cmd = subcommands.add_parser(
        "should-delegate",
        help=(
            "Run the classifier-backed invocation gate on a prompt: should the "
            "host agent delegate this to a Puppetmaster verb, or stay inline? "
            "Pure, local, no LLM. Pairs with the install-hooks deterministic hooks."
        ),
    )
    delegate_cmd.add_argument("prompt", help="The user prompt / task text.")
    delegate_cmd.add_argument("--role", default=None, help="Override the inferred task role.")
    delegate_cmd.add_argument(
        "--threshold", type=int, default=None, help="Delegate at/above this capability score."
    )
    delegate_cmd.add_argument("--json", action="store_true", help="Emit JSON.")

    gate_hook = subcommands.add_parser(
        "invocation-gate",
        help=(
            "Host-hook entry point: read a Cursor/Claude hook payload as JSON on "
            "stdin and print the host-specific verdict. Wired up by install-hooks; "
            "rarely run by hand."
        ),
    )
    gate_hook.add_argument("--host", default="cursor", help="cursor | claude")
    gate_hook.add_argument("--event", default="user-prompt", help="Host event name.")

    install_hooks_parser = subcommands.add_parser(
        "install-hooks",
        help=(
            "Install deterministic auto-invocation hooks into Cursor "
            "(.cursor/hooks.json) and Claude Code (.claude/settings.json). These "
            "inject a delegate directive on prompt submit and deny-redirect "
            "recursive shell searches + Task fan-out to Puppetmaster equivalents "
            "(read-only inspection and native Grep/Glob pass through). Default scope is "
            "this workspace; pass --global for user-level hooks covering every repo."
        ),
    )
    install_hooks_parser.add_argument(
        "--target",
        default=None,
        help=f"Comma-separated subset. Valid: {', '.join(sorted(VALID_HOOK_TARGETS))}. Default: both.",
    )
    install_hooks_parser.add_argument(
        "--global",
        dest="global_scope",
        action="store_true",
        help=(
            "Install user-level hooks (~/.cursor/hooks.json, ~/.claude/settings.json) "
            "that cover every repo you open, instead of just this workspace."
        ),
    )
    install_hooks_parser.add_argument("--force", action="store_true", help="Rewrite even if current.")
    install_hooks_parser.add_argument("--dry-run", action="store_true", help="Print without writing.")

    proxy_cmd = subcommands.add_parser(
        "proxy",
        help=(
            "Run a local OpenAI-compatible proxy that runs the invocation gate on "
            "inbound prompts — the deterministic enforcement path for API-key/SDK "
            "clients that closed harnesses can't offer."
        ),
    )
    proxy_cmd.add_argument("--host", default="127.0.0.1", help="Bind host.")
    proxy_cmd.add_argument("--port", type=int, default=8788, help="Bind port.")
    proxy_cmd.add_argument(
        "--mode",
        choices=["advise", "inject"],
        default="advise",
        help="advise: synthetic local delegate reply (no upstream). inject: forward to a vetted upstream.",
    )
    proxy_cmd.add_argument(
        "--upstream-base-url",
        default="",
        help="Required for inject mode; must be loopback or an allowlisted HTTPS host.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return _main(argv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _resolve_store_for_job(
    job_id: Optional[str],
    state_dir: Path,
    store,
    backend: str,
    explicit_state_dir: Optional[str],
):
    """Auto-pivot to the project that owns ``job_id`` when needed.

    Pre-fix, ``puppetmaster show job_X`` from a directory whose state
    dir didn't contain the job would emit a confusing "job not found"
    error, even though the job was alive in a sibling project's state
    dir. Users compensated by exporting
    ``PUPPETMASTER_STATE_DIR=~/Library/Application Support/...`` —
    which means they had to know the workspace hash. Now we scan
    every known project state dir for the job and pivot silently
    (with a single stderr note) when we find it elsewhere.

    Respects an explicit ``--state-dir`` or ``$PUPPETMASTER_STATE_DIR``
    override: if the user named a dir explicitly, we trust them and
    don't pivot.
    """
    if not job_id:
        return state_dir, store
    if explicit_state_dir or os.environ.get("PUPPETMASTER_STATE_DIR"):
        return state_dir, store
    if (state_dir / "jobs" / job_id).is_dir():
        return state_dir, store
    found = find_state_dir_for_job(job_id)
    if found is None or found.resolve() == state_dir.resolve():
        return state_dir, store
    sys.stderr.write(
        f"note: job {job_id} not in current workspace state dir; using {found}\n"
    )
    return found, create_store(backend, found)


def _main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = resolve_state_dir(args.state_dir)
    store = create_store(args.backend, state_dir)
    on_job_created = early_job_printer if args.emit_job_id_early else None

    # Read-only inspectors: pivot to whichever project state dir owns
    # the requested job_id. Write-side commands (run/cursor/claude/
    # daemon/...) intentionally do NOT pivot — those should always
    # use the caller's workspace state.
    if args.command in {
        "show",
        "artifacts",
        "diff",
        "memory",
        "feed",
        "logs",
        "events",
        "status",
        "open",
        "cost",
        "dashboard",
        "finalize",
        "wait",
    }:
        candidate_job_id = getattr(args, "job_id", None)
        state_dir, store = _resolve_store_for_job(
            candidate_job_id,
            state_dir,
            store,
            args.backend,
            args.state_dir,
        )

    if args.command == "init":
        store.init()
        print(f"Initialized Puppetmaster state at {store.root}")
        return 0

    if args.command == "state":
        print(store.root)
        return 0

    if args.command == "doctor":
        checks = run_doctor(Path.cwd(), state_dir)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    [
                        {
                            "name": check.name,
                            "status": check.status,
                            "detail": check.detail,
                            "evidence": check.evidence,
                        }
                        for check in checks
                    ],
                    indent=2,
                )
            )
            return 0
        for check in checks:
            print(f"{check.status:8} {check.name:16} {check.detail}")
        return 0

    if args.command == "install-codex-mcp":
        return _run_install_codex(args)

    if args.command == "install-claude-mcp":
        return _run_install_claude(args)

    if args.command == "install-hermes-mcp":
        return _run_install_hermes(args)

    if args.command == "install-cursor-mcp":
        return _run_install_cursor(args)

    if args.command == "install-rules":
        return _run_install_rules(args)

    if args.command == "setup":
        return _run_setup(args)

    if args.command == "uninstall":
        return _run_uninstall(args)

    if args.command == "repair-codegraph":
        return _run_repair_codegraph(args)

    if args.command == "codegraph":
        return _run_codegraph_passthrough(args)

    if args.command == "mcp":
        return _run_mcp_subcommand(args)

    if args.command == "models":
        return _run_models_subcommand(args)

    if args.command == "platform":
        return _run_platform_subcommand(args)

    if args.command == "route":
        return _run_route_command(args)

    if args.command == "should-delegate":
        return _run_should_delegate_command(args)

    if args.command == "invocation-gate":
        return _run_invocation_gate_command(args)

    if args.command == "install-hooks":
        return _run_install_hooks(args)

    if args.command == "proxy":
        return _run_proxy_command(args)

    if args.command == "audit":
        return _run_audit_command(args, store)

    if args.command == "savings":
        return _run_savings_command(args, state_dir)

    if args.command == "preflight":
        return _run_preflight_command(args)

    if args.command == "cost":
        return _run_cost_command(args, store)

    if args.command == "adapters":
        print(json.dumps(adapter_status(Path.cwd()), indent=2))
        return 0

    if args.command == "init-config":
        path = Path(args.path)
        if path.exists() and not args.force:
            raise SystemExit(f"{path} already exists; pass --force to overwrite")
        path.write_text(starter_config(), encoding="utf-8")
        print(f"wrote {path}")
        return 0

    if args.command == "run":
        from dataclasses import replace

        from puppetmaster.workers import specs_for_roles

        if getattr(args, "effort", None):
            os.environ["PUPPETMASTER_EFFORT_ID"] = args.effort
        if args.config:
            config = load_config(args.config)
            specs = config.workers
            lease_seconds = config.lease_seconds
        else:
            specs = specs_for_roles(args.workers)
            lease_seconds = 5
        if args.enable_memory:
            specs = [
                replace(spec, payload={**spec.payload, "disable_memory": False})
                for spec in specs
            ]
        elif args.disable_memory:
            specs = [
                replace(spec, payload={**spec.payload, "disable_memory": True})
                for spec in specs
            ]
        result = Orchestrator(store).run(
            args.goal,
            specs=specs,
            lease_seconds=lease_seconds,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "cursor":
        implement = getattr(args, "implement", False)
        prompt = cursor_prompt(
            args.prompt,
            review=args.review,
            plan=args.plan,
            dry_run=args.dry_run,
            implement=implement,
        )
        payload = {
            "prompt": prompt,
            "cwd": args.cwd,
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
        }
        if implement:
            payload["mode"] = "implement"
            payload["allow_dirty"] = getattr(args, "allow_dirty", False)
            payload["allow_non_worktree"] = getattr(args, "allow_non_worktree", False)
        if args.disable_memory or args.review or args.plan:
            payload["disable_memory"] = True
        payload.update(routing_payload_from_args(args, adapter="cursor"))
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="cursor",
                    instruction=args.prompt,
                    adapter="cursor",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "claude":
        payload = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "model": args.model,
            "permission_mode": args.permission_mode,
            "allowed_tools": args.allowed_tools,
            "disallowed_tools": args.disallowed_tools,
            "executable": args.executable,
            "timeout_seconds": args.timeout_seconds,
            "allow_dirty": args.allow_dirty,
            "allow_non_worktree": args.allow_non_worktree,
        }
        if args.disable_memory:
            payload["disable_memory"] = True
        payload.update(routing_payload_from_args(args, adapter="claude-code"))
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="claude-code",
                    instruction=args.prompt,
                    adapter="claude-code",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "openai":
        payload: dict[str, Any] = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
        }
        if args.base_url:
            payload["openai_base_url"] = args.base_url
        if args.organization:
            payload["openai_organization"] = args.organization
        if args.max_output_tokens is not None:
            payload["max_output_tokens"] = args.max_output_tokens
        if args.legacy_max_tokens:
            payload["legacy_max_tokens"] = True
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        if args.disable_codegraph:
            payload["disable_codegraph"] = True
        if args.disable_memory:
            payload["disable_memory"] = True
        payload.update(routing_payload_from_args(args, adapter="openai"))
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="openai",
                    instruction=args.prompt,
                    adapter="openai",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "codex":
        payload: dict[str, Any] = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "model": args.model,
            "sandbox": args.sandbox,
            "approval_policy": args.approval_policy,
            "timeout_seconds": args.timeout_seconds,
            "allow_dirty": args.allow_dirty,
            "allow_non_worktree": args.allow_non_worktree,
            "dangerously_bypass_approvals_and_sandbox": args.dangerously_bypass_approvals_and_sandbox,
        }
        if args.executable:
            payload["executable"] = args.executable
        if args.disable_codegraph:
            payload["disable_codegraph"] = True
        if args.disable_memory:
            payload["disable_memory"] = True
        if (
            args.sandbox == "read-only"
            and not args.dangerously_bypass_approvals_and_sandbox
        ):
            payload["read_only"] = True
        payload.update(routing_payload_from_args(args, adapter="codex"))
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="codex",
                    instruction=args.prompt,
                    adapter="codex",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "hermes":
        payload = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "mode": args.mode,
            "timeout_seconds": args.timeout_seconds,
            "allow_dirty": args.allow_dirty,
            "allow_non_worktree": args.allow_non_worktree,
        }
        if args.model:
            payload["model"] = args.model
        if args.provider:
            payload["provider"] = args.provider
        if args.max_turns is not None:
            payload["max_turns"] = args.max_turns
        if args.toolsets:
            payload["toolsets"] = args.toolsets
        if args.executable:
            payload["executable"] = args.executable
        if args.use_hermes_rules:
            payload["ignore_rules"] = False
        if args.disable_codegraph:
            payload["disable_codegraph"] = True
        payload.update(routing_payload_from_args(args, adapter="hermes"))
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role=f"hermes-{args.mode}",
                    instruction=args.prompt,
                    adapter="hermes",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        return finalize_cli_run(result)

    if args.command == "demo":
        result = Orchestrator(store).run(args.goal)
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        print("\n" + result.summary)
        return 0

    if args.command == "crash-demo":
        result = Orchestrator(store).run_crash_recovery_demo(
            args.goal,
            crash_role=args.crash_role,
        )
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        print(f"recovered_tasks: {result.recovered_tasks}")
        print("\n" + result.summary)
        return 0

    if args.command == "jobs":
        _reap_quietly(store)
        if getattr(args, "all_projects", False):
            for project in list_project_state_dirs():
                project_store = create_store(args.backend, project)
                try:
                    project_jobs = project_store.list_jobs()
                except Exception:
                    continue
                for job in project_jobs:
                    print(
                        f"{job.id}\t{job.status}\t{job.created_at}\t{project.name}\t{job.goal}"
                    )
            return 0
        for job in store.list_jobs():
            print(f"{job.id}\t{job.status}\t{job.created_at}\t{job.goal}")
        return 0

    if args.command == "projects":
        projects = list_project_state_dirs()
        if not projects:
            print("no Puppetmaster projects found on this machine yet")
            return 0
        for project in projects:
            jobs_dir = project / "jobs"
            job_count = (
                sum(1 for _ in jobs_dir.iterdir() if _.is_dir())
                if jobs_dir.is_dir()
                else 0
            )
            last_activity = (
                max(
                    (p.stat().st_mtime for p in jobs_dir.iterdir() if p.is_dir()),
                    default=None,
                )
                if jobs_dir.is_dir()
                else None
            )
            last_str = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_activity))
                if last_activity is not None
                else "(never)"
            )
            print(f"{project.name}\t{job_count} jobs\tlast: {last_str}\t{project}")
        return 0

    if args.command == "last":
        job = store.latest_job()
        if job is None:
            print("no jobs")
            return 1
        print(job.id)
        return 0

    if args.command == "status":
        # Surface a dead-but-"running" job as stalled before snapshotting, so
        # status never reports a wedged job as live.
        _reap_quietly(store)
        _warn_job_liveness(store, args.job_id)
        print(json.dumps(store.status_snapshot(args.job_id, compact=args.compact), indent=2))
        return 0

    if args.command == "watch":
        for _ in range(args.ticks):
            snapshot = store.status_snapshot(args.job_id)
            print_watch_snapshot(snapshot)
            if snapshot["job"]["status"] in {"complete", "failed"}:
                break
            time.sleep(args.interval)
        return 0

    if args.command == "recover":
        recovered = store.recover_stale_tasks(args.job_id)
        print(f"recovered: {len(recovered)}")
        for task in recovered:
            print(f"{task.id}\t{task.role}\tattempts={task.attempts}")
        return 0

    if args.command == "events":
        print(json.dumps(store.read_events(args.job_id), indent=2))
        return 0

    if args.command == "logs":
        job_id = args.job_id or require_latest_job_id(store)
        event_filters = [needle for needle in (args.event_type or []) if needle]
        show_all = args.all or bool(event_filters)
        suppressed: dict[str, int] = {}
        for event in store.read_events(job_id):
            name = event["event"]
            if event_filters and not any(needle in name for needle in event_filters):
                continue
            if not show_all and name in _NOISY_LOG_EVENTS:
                suppressed[name] = suppressed.get(name, 0) + 1
                continue
            print(f"{event['at']}\t{name}\t{json.dumps(event['payload'], sort_keys=True)}")
        if suppressed:
            collapsed = ", ".join(f"{name}={count}" for name, count in sorted(suppressed.items()))
            total = sum(suppressed.values())
            print(
                f"… collapsed {total} heartbeat event(s) [{collapsed}] — pass --all to show them",
                file=sys.stderr,
            )
        return 0

    if args.command == "feed":
        from puppetmaster import reads_log

        reads_log.record_read("feed", caller="cli")
        job_id = args.job_id or require_latest_job_id(store)
        if args.follow:
            return run_feed_follow(
                store,
                job_id,
                since=args.since,
                limit=args.limit,
                as_json=args.json,
                idle_timeout_seconds=args.follow_timeout_seconds,
                poll_interval_seconds=args.follow_poll_seconds,
            )
        items, _ = artifact_feed_since(
            store, job_id, since=args.since, limit=args.limit
        )
        if args.json:
            print(json.dumps(items, indent=2, default=str))
        else:
            for item in items:
                print_feed_item(item)
        return 0

    if args.command == "open":
        job_id = args.job_id or require_latest_job_id(store)
        path = (
            store.job_dir(job_id) / "summaries" / "stitched.md"
            if args.kind == "summary"
            else store.job_dir(job_id)
        )
        print(path)
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        return 0

    if args.command == "show":
        from puppetmaster import reads_log

        _warn_run_quality(store, args.job_id)
        if args.partial:
            reads_log.record_read("partial_summary", caller="cli")
            print(Stitcher(store).preview(args.job_id))
            return 0
        reads_log.record_read("show", caller="cli")
        path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
        if path.is_file():
            print(path.read_text(encoding="utf-8"))
            return 0
        # No stitched summary yet: degrade gracefully instead of crashing with a
        # raw "[Errno 2] No such file or directory" stack. Synthesize a live
        # summary from whatever artifacts exist and tell the user the job hasn't
        # finalized (and how to force it).
        job = store.get_job(args.job_id)
        sys.stderr.write(
            f"note: job {args.job_id} not finalized (state={job.status}); "
            "showing a live summary from current artifacts. "
            f"Run `puppetmaster finalize {args.job_id}` to force stitching.\n"
        )
        print(Stitcher(store).preview(args.job_id))
        return 0

    if args.command == "finalize":
        return _run_finalize_command(args, store)

    if args.command == "gc":
        return _run_gc_command(args, store)

    if args.command == "affected":
        return _run_affected_command(args)

    if args.command == "rollup":
        return _run_rollup_command(args, store)

    if args.command == "gate":
        return _run_gate_command(args, store)

    if args.command == "reap":
        return _run_reap_command(args, store)

    if args.command == "wait":
        return _run_wait_command(args, store)

    if args.command == "dashboard":
        from puppetmaster.dashboard import serve

        serve(
            state_dir,
            backend=args.backend,
            job_id=args.job_id,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            allow_external=getattr(args, "allow_external", False),
            all_projects=getattr(args, "all_projects", False),
        )
        return 0

    if args.command == "await":
        return _run_await_command(args, store)

    if args.command == "artifacts":
        from puppetmaster import reads_log

        reads_log.record_read("artifacts", caller="cli")
        artifacts = [artifact.__dict__ for artifact in store.list_artifacts(args.job_id)]
        print(json.dumps(artifacts, indent=2, default=str))
        return 0

    if args.command == "memory":
        print(json.dumps(store.list_memory(), indent=2))
        return 0

    if args.command == "diff":
        job_id = args.job_id or require_latest_job_id(store)
        patches = [
            artifact
            for artifact in store.list_artifacts(job_id)
            if str(artifact.type) == "patch"
        ]
        if not patches:
            print("no patch artifacts")
            return 0
        for artifact in patches:
            print(json.dumps(artifact.__dict__, indent=2, default=str))
        return 0

    if args.command == "approve":
        approved = approve_target(store, args.target, Path(args.worktree) if args.worktree else None)
        print(f"approved: {approved}")
        return 0

    if args.command == "reject":
        rejected = reject_target(store, args.target, args.reason)
        print(f"rejected: {rejected}")
        return 0

    if args.command == "rerun":
        source_job = store.get_job(args.job_id or require_latest_job_id(store))
        if args.config:
            config = load_config(args.config)
            result = Orchestrator(store).run(
                source_job.goal,
                specs=config.workers,
                lease_seconds=config.lease_seconds,
            )
        else:
            result = Orchestrator(store).run(source_job.goal)
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

    if args.command == "clean":
        if not args.all and not args.completed:
            raise ValueError("pass --completed or --all")
        deleted = 0
        for job in store.list_jobs():
            if args.all or str(job.status) in {"complete", "failed"}:
                store.delete_job(job.id)
                deleted += 1
        print(f"deleted: {deleted}")
        return 0

    if args.command == "daemon":
        completed = WorkerDaemon(
            store=store,
            roles=args.roles,
            worker_id=args.worker_id,
            job_id=args.job_id,
            lease_seconds=args.lease_seconds,
            poll_seconds=args.poll_seconds,
        ).run(max_tasks=args.max_tasks, max_idle_seconds=args.max_idle_seconds)
        print(f"processed_tasks: {completed}")
        return 0

    return 1


def print_run_result(job_id: str, artifact_count: int, summary_path: Path) -> None:
    print(f"job_id: {job_id}")
    print(f"artifacts: {artifact_count}")
    print(f"summary: {summary_path}")


def _warn_job_liveness(store: Any, job_id: str) -> None:
    """For a still-"running" job, print a loud liveness verdict to stderr so a
    wedged/dead orchestrator is obvious at a glance instead of hiding behind a
    quiet ``running`` status (#9)."""
    from puppetmaster.liveness import liveness_summary

    try:
        job = store.get_job(job_id)
        if job is None or str(job.status) not in {"running", "stitching"}:
            return
        summary = liveness_summary(store, job)
    except Exception:
        return
    if summary["verdict"] == "alive":
        return
    pid = summary.get("pid")
    sys.stderr.write(
        f"liveness: {summary['verdict']} — orchestrator pid={pid} "
        f"idle={summary['idle_seconds']}s, live_lease={summary['live_lease']}. "
        "Run `puppetmaster reap` to stall+requeue, or `recover` to retry tasks.\n"
    )


def _warn_run_quality(store: Any, job_id: str) -> None:
    """Print a one-line quality verdict to stderr when a job's artifacts look
    blocked/empty/degraded, so a reader of ``show`` is never silently handed an
    untrustworthy summary.

    A still-running job is exempt from the empty/degraded warning: implement
    workers stream no incremental artifacts, so a perfectly healthy in-flight
    job legitimately has no substantive artifacts yet. Calling that
    "low-confidence; verify before trusting" cries wolf. We instead emit a
    neutral in-progress note. A ``blocked`` verdict (a worker refused to run)
    is a real failure even mid-flight, so it still warns.
    """
    from puppetmaster.quality import assess_run_quality
    from puppetmaster.models import JobStatus

    try:
        verdict = assess_run_quality(store.list_artifacts(job_id))
    except Exception:
        return
    quality = verdict["quality"]
    if quality == "ok":
        return

    in_progress = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.STITCHING}
    try:
        status = store.get_job(job_id).status
    except Exception:
        status = None
    if quality in {"empty", "degraded"} and status in in_progress:
        sys.stderr.write(
            f"quality: in progress (state={status}) — no substantive artifacts yet. "
            "This is expected for a running implement job (artifacts land at the end); "
            "not a failure signal.\n"
        )
        return

    reason = "; ".join(verdict.get("reasons") or [])
    sys.stderr.write(
        f"quality: {quality} — {reason}. "
        "This run is low-confidence; verify before trusting it.\n"
    )


def finalize_cli_run(result: Any) -> int:
    """Print a run's mode banner, summary, and a built-in quality verdict, then
    return a shell exit code.

    A ``blocked`` run (a worker refused to run — e.g. dirty tree) or an
    ``empty`` run exits non-zero and prints a loud reason, so a "completed" job
    that did zero work can never masquerade as success. ``degraded`` is loud but
    non-fatal (exit 0) — the artifacts exist but shouldn't be trusted blindly.
    """
    from puppetmaster.quality import assess_run_quality

    print_mode_banner(result.mode)
    print_run_result(result.job.id, len(result.artifacts), result.summary_path)

    if result.mode == "edit":
        from puppetmaster.models import ArtifactType

        baseline_diff_present = any(
            bool((a.payload or {}).get("baseline_diff_present")) for a in result.artifacts
        )
        worker_diff_present = any(
            bool((a.payload or {}).get("worker_diff_present")) for a in result.artifacts
        )
        patch_artifact_emitted = any(
            a.type == ArtifactType.PATCH for a in result.artifacts
        )
        commit_present = any(
            a.type == ArtifactType.GATE
            and (a.payload or {}).get("kind") == "committed"
            and (a.payload or {}).get("passed") is True
            for a in result.artifacts
        )
        print(
            f"outcome: baseline_diff_present={baseline_diff_present} "
            f"worker_diff_present={worker_diff_present} "
            f"patch_artifact_emitted={patch_artifact_emitted} "
            f"commit_present={commit_present} "
            f"artifacts={len(result.artifacts)}",
            file=sys.stderr,
        )
        report_headline = next(
            (
                str(
                    (a.payload or {}).get("claim")
                    or (a.payload or {}).get("decision")
                    or ""
                ).strip()
                for a in result.artifacts
                if a.type in {ArtifactType.FINDING, ArtifactType.DECISION}
                and ((a.payload or {}).get("claim") or (a.payload or {}).get("decision"))
            ),
            None,
        )
        if report_headline:
            print(f"report: {report_headline}", file=sys.stderr)
            print(
                f"  full report: puppetmaster artifacts {result.job.id}",
                file=sys.stderr,
            )

    verdict = assess_run_quality(result.artifacts)
    quality = verdict["quality"]
    if quality == "ok":
        return 0

    reason = "; ".join(verdict.get("reasons") or [])
    if quality in {"blocked", "empty"}:
        print(
            f"puppetmaster: run {quality} — {reason}. "
            "Nothing was accomplished; not reporting success.",
            file=sys.stderr,
        )
        return 1
    print(
        f"puppetmaster: run quality=degraded — {reason}. "
        "Artifacts exist but treat this run as low-confidence.",
        file=sys.stderr,
    )
    return 0


def print_mode_banner(mode: str) -> None:
    """Print a one-line read-only / edit banner to stderr so the user is never
    surprised that an 'analysis' swarm wrote no files."""
    if mode == "edit":
        print(
            "puppetmaster: mode=edit — workers may modify files in the working tree.",
            file=sys.stderr,
        )
    else:
        print(
            "puppetmaster: mode=analysis (read-only) — no files will be edited; "
            "this run only emits artifacts.",
            file=sys.stderr,
        )


def routing_payload_from_args(args, *, adapter: str) -> dict:
    """Translate the shared ``--auto-route`` routing flags into payload keys the
    orchestrator's router understands. Empty unless ``--auto-route`` is set, so
    a direct adapter run is unchanged by default.

    Pins ``allowed_adapters`` to the invoked adapter so routing only picks a
    *model* within that platform — a direct ``cursor`` run never silently hops
    to claude-code."""
    if not getattr(args, "auto_route", False):
        return {}
    payload: dict[str, Any] = {"auto_route": True, "allowed_adapters": [adapter]}
    if getattr(args, "routing_policy", None):
        payload["routing_policy"] = args.routing_policy
    if getattr(args, "max_cost_usd", None) is not None:
        payload["max_cost_usd"] = args.max_cost_usd
    if getattr(args, "min_capability", None) is not None:
        payload["min_capability"] = args.min_capability
    return payload


def artifact_feed(store, job_id: str, limit: Optional[int] = None) -> list[dict]:
    items, _ = artifact_feed_since(store, job_id, since=0, limit=limit)
    return items


def artifact_feed_since(
    store,
    job_id: str,
    since: int = 0,
    limit: Optional[int] = None,
) -> tuple[list[dict], int]:
    """Return (items, next_cursor) of new ``artifact.saved`` events.

    ``since`` is the event cursor returned by a previous call (use 0 for a
    fresh read). ``next_cursor`` is the highest event id observed, regardless
    of whether it was an artifact event, so callers can resume reliably.
    """
    # Read events FIRST, then fetch only the artifacts those events reference.
    # The previous order (snapshot all artifacts, then read events) had a race:
    # an artifact saved between the two reads was missing from the snapshot, so
    # its event was skipped while the cursor still advanced past it — dropping
    # the artifact from the feed forever. save_artifact persists the row before
    # emitting its event, so an artifact named by an event we just read is
    # guaranteed to exist when we fetch it afterward.
    events = store.read_events_since(job_id, since=since)
    artifact_events: list[dict] = []
    cursor = since
    for event in events:
        event_id = event.get("id")
        if isinstance(event_id, int) and event_id > cursor:
            cursor = event_id
        if event.get("event") == "artifact.saved":
            artifact_events.append(event)

    needed_ids = [
        event.get("payload", {}).get("artifact_id") for event in artifact_events
    ]
    fetched = store.get_artifacts_by_ids(job_id, needed_ids)

    items: list[dict] = []
    seen: set = set()
    for event in artifact_events:
        artifact_id = event.get("payload", {}).get("artifact_id")
        artifact = fetched.get(artifact_id)
        if artifact is None or artifact_id in seen:
            continue
        seen.add(artifact_id)
        items.append(
            {
                "at": event["at"],
                "event": event["event"],
                "id": event.get("id"),
                "artifact": artifact.__dict__,
            }
        )
    if limit is not None:
        items = items[-limit:]
    return items, cursor


def print_feed_item(item: dict) -> None:
    artifact = item["artifact"]
    print(
        f"{item['at']}\t{artifact['type']}\t{artifact['id']}\t"
        f"task={artifact['task_id']}\tconfidence={artifact['confidence']}"
    )
    print(f"  {artifact_headline(artifact)}")


def run_feed_follow(
    store,
    job_id: str,
    *,
    since: int = 0,
    limit: Optional[int] = None,
    as_json: bool = False,
    idle_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> int:
    cursor = since
    initial_items, cursor = artifact_feed_since(store, job_id, since=cursor, limit=limit)
    for item in initial_items:
        emit_feed_item(item, as_json=as_json)

    poll_budget = max(0.05, poll_interval_seconds)
    block_seconds = max(poll_budget, 1.0)
    idle_deadline = (
        time.monotonic() + idle_timeout_seconds if idle_timeout_seconds > 0 else None
    )
    try:
        while True:
            events = store.wait_for_events(
                job_id,
                since=cursor,
                timeout_seconds=block_seconds,
                poll_interval=poll_budget,
            )
            if events:
                new_items, cursor = artifact_feed_since(
                    store, job_id, since=cursor, limit=None
                )
                for item in new_items:
                    emit_feed_item(item, as_json=as_json)
                idle_deadline = (
                    time.monotonic() + idle_timeout_seconds
                    if idle_timeout_seconds > 0
                    else None
                )
                continue
            if idle_deadline is not None and time.monotonic() >= idle_deadline:
                return 0
    except KeyboardInterrupt:
        return 0


def emit_feed_item(item: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(item, default=str), flush=True)
    else:
        print_feed_item(item)
        sys.stdout.flush()


def artifact_headline(artifact: dict) -> str:
    payload = artifact.get("payload", {})
    if not isinstance(payload, dict):
        return str(payload)
    for key in ["claim", "decision", "risk", "check", "change"]:
        if key in payload:
            return str(payload[key])
    return json.dumps(payload, sort_keys=True)


def early_job_printer(job) -> None:
    print(f"job_id: {job.id}", flush=True)


def _print_install_result(result: InstallResult, host: str) -> int:
    """Pretty-print an :class:`InstallResult` and return the appropriate exit code.

    Exit codes are deliberately compatible with shell-script automation:
    ``0`` for installed / unchanged / would_install (a successful no-op),
    ``1`` for any error so a CI step can fail fast on a broken install.
    """
    print(f"[install-{host}-mcp] status: {result.status}")
    print(f"[install-{host}-mcp] target: {redact_secrets(result.target)}")
    print(f"[install-{host}-mcp] python: {redact_secrets(result.python_executable)}")
    if result.handshake is not None:
        if result.handshake.ok:
            print(
                f"[install-{host}-mcp] handshake: OK ({result.handshake.tool_count} tools)"
            )
        else:
            print(f"[install-{host}-mcp] handshake: FAILED — {redact_secrets(result.handshake.error)}")
    for line in result.messages:
        print(f"[install-{host}-mcp] {redact_secrets(line)}")
    return 0 if result.status in {"installed", "unchanged", "would_install"} else 1


def _print_uninstall_mcp_result(result: UninstallResult, host: str) -> int:
    """Pretty-print an :class:`UninstallResult` and return the appropriate exit code."""
    print(f"[uninstall-{host}-mcp] status: {result.status}")
    print(f"[uninstall-{host}-mcp] target: {result.target}")
    for line in result.messages:
        print(f"[uninstall-{host}-mcp] {line}")
    return 0 if result.status in {"removed", "unchanged", "would_remove"} else 1


def _print_uninstall_rules_result(result: RulesInstallResult) -> int:
    print(f"[uninstall-rules] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[uninstall-rules] {outcome.target:<16} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[uninstall-rules] {' ' * 16} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[uninstall-rules] note: {msg}")
    return 1 if result.overall_status == "error" else 0


def _print_uninstall_hooks_result(result) -> int:
    print(f"[uninstall-hooks] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[uninstall-hooks] {outcome.target:<8} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[uninstall-hooks] {' ' * 8} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[uninstall-hooks] note: {msg}")
    return 1 if result.overall_status == "error" else 0


def _confirm_uninstall(*, yes: bool, dry_run: bool) -> bool:
    if yes or dry_run:
        return True
    if not sys.stdin.isatty():
        print(
            "error: refusing to uninstall without --yes in non-interactive mode",
            file=sys.stderr,
        )
        return False
    print(
        "This removes Puppetmaster MCP registrations, hooks, and rules "
        "from Cursor/Codex/Claude host configs."
    )
    try:
        answer = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _purge_uninstall_state(cwd: Path, *, dry_run: bool) -> list[tuple[str, str, str]]:
    """Remove optional state dirs when ``--purge-state`` is passed."""
    import shutil

    targets = [
        ("home-state", Path.home() / ".puppetmaster"),
        ("workspace-state", cwd / ".puppetmaster"),
        ("codegraph", cwd / ".codegraph"),
    ]
    outcomes: list[tuple[str, str, str]] = []
    for label, path in targets:
        if not path.exists():
            outcomes.append((label, str(path), "unchanged"))
            continue
        if dry_run:
            outcomes.append((label, str(path), "would_remove"))
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        outcomes.append((label, str(path), "removed"))
    return outcomes


def _run_uninstall(args) -> int:
    """Dispatch for ``puppetmaster uninstall`` — inverse of ``setup`` host wiring."""
    cwd = Path(getattr(args, "cwd", ".")).expanduser().resolve()
    dry_run = getattr(args, "dry_run", False)
    if not _confirm_uninstall(yes=getattr(args, "yes", False), dry_run=dry_run):
        return 1

    overall_rc = 0

    print("=== uninstall: cursor MCP (workspace + global) ===")
    for label, target in (
        ("workspace", cwd / ".cursor" / "mcp.json"),
        ("global", Path.home() / ".cursor" / "mcp.json"),
    ):
        result = uninstall_cursor_mcp(target_path=target.resolve(), dry_run=dry_run)
        overall_rc |= _print_uninstall_mcp_result(result, f"cursor-{label}")
    print()

    print("=== uninstall: codex MCP ===")
    codex_result = uninstall_codex_mcp(dry_run=dry_run)
    overall_rc |= _print_uninstall_mcp_result(codex_result, "codex")
    print()

    print("=== uninstall: claude MCP ===")
    claude_result = uninstall_claude_mcp(dry_run=dry_run)
    overall_rc |= _print_uninstall_mcp_result(claude_result, "claude")
    print()

    print("=== uninstall: hermes MCP ===")
    hermes_result = uninstall_hermes_mcp(dry_run=dry_run)
    overall_rc |= _print_uninstall_mcp_result(hermes_result, "hermes")
    print()

    print("=== uninstall: rules ===")
    rules_result = uninstall_rules(cwd=cwd, dry_run=dry_run)
    overall_rc |= _print_uninstall_rules_result(rules_result)
    print()

    print("=== uninstall: hooks (project + global scopes) ===")
    hooks_result = uninstall_hooks(cwd=cwd, dry_run=dry_run)
    overall_rc |= _print_uninstall_hooks_result(hooks_result)
    print()

    print("=== uninstall: stale MCP processes ===")
    if dry_run:
        print("[uninstall-mcp-processes] status: would_remove")
        print("[uninstall-mcp-processes] would run mcp cleanup --kill-stale")
    else:
        killed_entries = registry_kill_stale()
        if killed_entries:
            print("[uninstall-mcp-processes] status: removed")
            for entry in killed_entries:
                print(
                    f"[uninstall-mcp-processes] killed stale PID {entry.pid} "
                    f"({entry.workspace or '-'})"
                )
        else:
            print("[uninstall-mcp-processes] status: unchanged")
            print("[uninstall-mcp-processes] no stale Puppetmaster MCP processes")
    print()

    if getattr(args, "purge_state", False):
        print("=== uninstall: state purge (--purge-state) ===")
        for label, path, status in _purge_uninstall_state(cwd, dry_run=dry_run):
            print(f"[uninstall-state] {label:<16} status: {status}")
            print(f"[uninstall-state] {' ' * 16} target: {path}")
        print()
    else:
        print(
            "[uninstall-state] status: unchanged  "
            "(left ~/.puppetmaster/, <cwd>/.puppetmaster/, and .codegraph/ intact; "
            "pass --purge-state to remove)"
        )
        print()

    print("Host integrations removed. Last step: pip uninstall puppetmaster-ai")
    return overall_rc


def _run_install_codex(args) -> int:
    """Dispatch for ``puppetmaster install-codex-mcp``.

    Delegates to :func:`install_codex_mcp` and prints the sandbox
    guidance block on success so the user knows the first MCP call
    inside ``codex`` will surface an approval prompt.
    """
    result = install_codex_mcp(
        codex_executable=getattr(args, "codex", None),
        force=getattr(args, "force", False),
        force_env=getattr(args, "force_env", False),
        env=tuple(getattr(args, "env", []) or []),
        inherit_env=tuple(getattr(args, "inherit_env", []) or []),
        env_files=tuple(Path(p).expanduser() for p in (getattr(args, "env_file", []) or [])),
        map_env=tuple(getattr(args, "map_env", []) or []),
        dry_run=getattr(args, "dry_run", False),
        skip_handshake=getattr(args, "skip_handshake", False),
    )
    rc = _print_install_result(result, "codex")
    if result.status in {"installed", "unchanged"}:
        print()
        print("Next steps:")
        for line in CODEX_SANDBOX_GUIDANCE.splitlines():
            print(f"  {line}")
    return rc


def _run_install_claude(args) -> int:
    """Dispatch for ``puppetmaster install-claude-mcp``."""
    result = install_claude_mcp(
        claude_executable=getattr(args, "claude", None),
        force=getattr(args, "force", False),
        dry_run=getattr(args, "dry_run", False),
        skip_handshake=getattr(args, "skip_handshake", False),
    )
    rc = _print_install_result(result, "claude")
    if result.status in {"installed", "unchanged"}:
        print()
        print("Next steps:")
        for line in CLAUDE_NEXT_STEPS_GUIDANCE.splitlines():
            print(f"  {line}")
    return rc


def _seed_hermes_registry() -> None:
    """Seed credential-backed Hermes models into the router registry.

    Called from the ``setup`` wizard after the Hermes MCP install. Only models
    whose provider has a usable credential are added, so ``auto_route`` can pick
    Hermes immediately without ever landing on a provider the user can't call.
    Skipped models and the no-credential case are surfaced as actionable lines.
    Best-effort: any failure prints a note and never aborts the wizard.
    """
    try:
        from puppetmaster.adapters import available_hermes_providers
        from puppetmaster.model_registry import (
            default_registry_path,
            load_registry,
            save_registry,
        )
        from puppetmaster.static_catalog import merge_curated_into_registry

        registry_path = default_registry_path()
        if not registry_path.is_file():
            print("  registry  skipped — no registry yet; run `puppetmaster models init` first")
            return
        allowed = available_hermes_providers()
        existing = load_registry(registry_path)
        merged, report = merge_curated_into_registry(
            "hermes", "api", existing, allowed_providers=allowed
        )
        save_registry(merged, registry_path)
        if report["added"] or report["refreshed"]:
            print(
                f"  registry  seeded hermes models "
                f"(added={report['added']}, refreshed={report['refreshed']})"
            )
        for skip in report.get("skipped", []):
            print(
                f"  registry  skipped {skip['model']} — no credential for provider "
                f"'{skip['provider']}'"
            )
        if not allowed:
            print(
                "  registry  note: no Hermes provider credentials found "
                "(~/.hermes/.env or `hermes login`). Add a key and re-run "
                "`puppetmaster models discover --source hermes --write`."
            )
    except Exception as exc:  # never let registry seeding abort the wizard
        print(f"  registry  note: hermes registry seeding skipped ({exc!r})")


def _run_install_hermes(args) -> int:
    """Dispatch for ``puppetmaster install-hermes-mcp``."""
    explicit = getattr(args, "path", None)
    target_path = Path(explicit).expanduser().resolve() if explicit else None
    result = install_hermes_mcp(
        target_path=target_path,
        force=getattr(args, "force", False),
        dry_run=getattr(args, "dry_run", False),
        skip_handshake=getattr(args, "skip_handshake", False),
    )
    rc = _print_install_result(result, "hermes")
    if result.status in {"installed", "unchanged"}:
        print()
        print("Next steps:")
        for line in HERMES_NEXT_STEPS_GUIDANCE.splitlines():
            print(f"  {line}")
    return rc


def _run_install_cursor(args) -> int:
    """Dispatch for ``puppetmaster install-cursor-mcp``.

    Resolves the target ``mcp.json`` path from one of three signals:

    1. ``--path PATH`` overrides everything (used by tests and power users).
    2. ``--global`` writes to ``~/.cursor/mcp.json``.
    3. Otherwise default to ``<cwd>/.cursor/mcp.json`` (workspace-local).

    The workspace-local default mirrors the convention used by most
    Cursor projects of checking ``.cursor/mcp.json`` into the repo so
    teammates inherit the same MCP wiring.
    """
    explicit = getattr(args, "path", None)
    if explicit:
        target = Path(explicit).expanduser().resolve()
    elif getattr(args, "install_global", False):
        target = (Path.home() / ".cursor" / "mcp.json").resolve()
    else:
        target = (Path.cwd() / ".cursor" / "mcp.json").resolve()
    if not getattr(args, "dry_run", False):
        sdk = ensure_cursor_sdk(Path.cwd())
        print(f"[install-cursor-mcp] sdk {sdk.status}: {sdk.detail}")
    result = install_cursor_mcp(
        target_path=target,
        force=getattr(args, "force", False),
        dry_run=getattr(args, "dry_run", False),
        skip_handshake=getattr(args, "skip_handshake", False),
    )
    rc = _print_install_result(result, "cursor")
    if result.status in {"installed", "unchanged"}:
        print()
        print("Next steps:")
        for line in CURSOR_NEXT_STEPS_GUIDANCE.splitlines():
            print(f"  {line}")
    return rc


def _print_rules_result(result: RulesInstallResult) -> int:
    """Pretty-print a :class:`RulesInstallResult` and return an exit code.

    Exit codes mirror the MCP installers: 0 on installed / unchanged /
    would_install / skipped, 1 on any target error.
    """
    print(f"[install-rules] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[install-rules] {outcome.target:<14} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[install-rules] {' ' * 14} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[install-rules] note: {msg}")
    return 1 if result.overall_status == "error" else 0


def _run_install_rules(args) -> int:
    """Dispatch for ``puppetmaster install-rules``."""
    targets = None
    raw_target = getattr(args, "target", None)
    if raw_target:
        targets = [t.strip() for t in raw_target.split(",") if t.strip()]
    result = install_rules(
        cwd=Path.cwd(),
        targets=targets,
        install_global=getattr(args, "rules_global", False),
        dry_run=getattr(args, "dry_run", False),
        force=getattr(args, "force", False),
    )
    return _print_rules_result(result)


def _detected_platforms(root: Path) -> dict[str, bool]:
    """Which platform-billed adapters look *usable on this machine*.

    Presence probes, not auth audits: cursor needs ``CURSOR_API_KEY`` plus
    either its bundled SDK or an ``npm`` that setup can bootstrap it with
    (PyPI wheels can't ship ``node_modules``, so a fresh pip/pipx install
    legitimately lacks the SDK until the install-cursor-mcp step fetches
    it); claude-code and codex need their CLI resolvable; openai needs
    ``OPENAI_API_KEY``. Codex login state is deliberately not probed
    (subscription auth is opaque from here) — the billing checks in
    doctor cover that separately.
    """
    import shutil as _shutil

    from puppetmaster.diagnostics import (
        _claude_code_installed,
        _codex_cli_installed,
        _cursor_sdk_installed,
    )

    cursor_sdk_available = _cursor_sdk_installed(root) or _shutil.which("npm") is not None
    return {
        "cursor": bool(os.environ.get("CURSOR_API_KEY")) and cursor_sdk_available,
        "claude-code": _claude_code_installed(),
        "codex": _codex_cli_installed(),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }


def _setup_platform_step(args) -> int:
    """The `setup` wizard's platform-lock step.

    Defaults to *what is detected on this machine* instead of "everything on"
    (field report: a Claude-Code-only user got cursor recommendations because
    setup silently left all platforms enabled). Modes:

    * ``--platforms cursor,claude-code`` — explicit, always wins.
    * a TTY with no flag — interactive prompt; on first run, Enter applies
      the detected set.
    * non-interactive shell with no flag — first run locks to the detected
      set; an existing lock is respected and left unchanged.

    Returns non-zero only for an invalid *explicit* ``--platforms`` value; the
    interactive and skipped paths never fail the wizard on a typo.
    """
    from puppetmaster import platform_lock as pl

    known = pl.KNOWN_ADAPTERS
    detected = _detected_platforms(Path.cwd())
    detected_set = {a for a, present in detected.items() if present}

    def _show_state() -> None:
        enabled = pl.enabled_adapters()
        for adapter in known:
            mark = "on " if adapter in enabled else "off"
            note = "" if detected.get(adapter, True) else "   (not detected on this machine)"
            print(f"  [{mark}] {adapter}{note}")

    raw = getattr(args, "platforms", None)
    if raw is not None:
        wanted = {a.strip() for a in raw.split(",") if a.strip()}
        unknown = sorted(a for a in wanted if a not in known)
        if unknown:
            print(
                f"  error  unknown platform(s): {', '.join(unknown)}. "
                f"Known: {', '.join(known)}."
            )
            return 1
        valid = {a for a in wanted if a in known}
        if not valid:
            print("  error  --platforms named no known platform.")
            return 1
        pl.set_enabled(valid)
        print(f"  locked  routing restricted to: {', '.join(sorted(valid))}")
        undetected = sorted(a for a in valid if not detected.get(a, True))
        if undetected:
            print(
                f"  note: not detected on this machine: {', '.join(undetected)} — "
                "enabled anyway (explicit --platforms)"
            )
        _show_state()
        return 0

    if getattr(args, "skip_platforms", False):
        print("  skipped  (--skip-platforms) — platform lock left unchanged")
        _show_state()
        return 0

    first_run = not pl.platform_config_path().is_file()

    if not sys.stdin.isatty():
        if not first_run:
            print(
                "  unchanged  existing platform lock respected "
                "(non-interactive shell, no --platforms flag)"
            )
        elif detected_set:
            pl.set_enabled(detected_set)
            print(
                "  detected  locked to the platforms found on this machine: "
                f"{', '.join(sorted(detected_set))}"
            )
            print(
                "  note: adjust anytime with `puppetmaster platform "
                "enable/disable/only/reset` or `setup --platforms ...`"
            )
        else:
            print(
                "  skipped  no platforms detected on this machine — lock left "
                "at default (all enabled); set explicitly with --platforms"
            )
        _show_state()
        return 0

    print("Puppetmaster routes work across these platforms:")
    _show_state()
    if first_run and detected_set:
        print(f"Detected on this machine: {', '.join(sorted(detected_set))}")
        print(
            "Enter a comma-separated list of platforms to ENABLE (all others off),\n"
            "'all' to keep every platform on, or press Enter to use the detected set."
        )
    else:
        print(
            "Enter a comma-separated list of platforms to ENABLE (all others off),\n"
            "'all' to keep every platform on, or press Enter to leave unchanged."
        )
    try:
        answer = input("  platforms> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  skipped  no input — platform lock left unchanged")
        return 0

    if not answer:
        if first_run and detected_set:
            pl.set_enabled(detected_set)
            print(
                "  locked  routing restricted to detected platforms: "
                f"{', '.join(sorted(detected_set))}"
            )
        else:
            print("  unchanged  platform lock left as-is")
        _show_state()
        return 0
    if answer.lower() == "all":
        pl.reset()
        print("  reset  all platforms enabled")
        _show_state()
        return 0
    wanted = {a.strip() for a in answer.split(",") if a.strip()}
    unknown = sorted(a for a in wanted if a not in known)
    if unknown:
        print(
            f"  error  unknown platform(s): {', '.join(unknown)} — leaving "
            f"unchanged. Known: {', '.join(known)}."
        )
        return 0  # a typo shouldn't fail the whole wizard
    valid = {a for a in wanted if a in known}
    if not valid:
        print("  unchanged  no known platform named — left as-is")
        return 0
    pl.set_enabled(valid)
    print(f"  locked  routing restricted to: {', '.join(sorted(valid))}")
    _show_state()
    return 0


def _run_setup(args) -> int:
    """Dispatch for ``puppetmaster setup`` — one-shot first-run wizard.

    Chains the canonical install steps in dependency order:

    1. ``doctor`` — fail loudly if Puppetmaster's runtime is broken.
    2. ``platform lock`` — choose which platforms to route to out of the gate.
    3. ``models init`` — write the starter registry if missing.
    4. ``install-cursor-mcp`` — workspace .cursor/mcp.json.
    5. ``install-codex-mcp`` — only if ``codex`` is enabled *and* on PATH.
    6. ``install-claude-mcp`` — only if ``claude-code`` is enabled *and* the
       Claude CLI is resolvable (host-side registration, user scope).
    7. ``install-rules`` — soft agent nudges for whichever tools detected.
    8. ``install-hooks`` — deterministic auto-invocation hooks for Cursor +
       Claude Code (prompt-inject + native-tool deny-redirect).

    Each step is independent: a step's failure prints a clear error
    but does not abort the rest of the chain unless ``doctor`` reports
    that Python or sqlite is missing (in which case nothing else will
    work). The user can re-run after fixing whatever was reported.
    """
    cwd = Path.cwd()
    state_dir = resolve_state_dir(args.state_dir, cwd)
    overall_rc = 0

    if not getattr(args, "skip_doctor", False):
        print("=== step 1/9: doctor ===")
        checks = list(run_doctor(cwd, state_dir))
        for check in checks:
            print(f"  {check.status:8} {check.name:16} {check.detail}")
        criticals = [c for c in checks if c.status == "fail" and c.name in {"python", "sqlite"}]
        if criticals:
            print("\nCritical dependency missing — fix the above before re-running `setup`.")
            return 1
        print()
    else:
        print("=== step 1/9: doctor SKIPPED (--skip-doctor) ===\n")

    print("=== step 2/9: platform lock ===")
    if _setup_platform_step(args) != 0:
        overall_rc = 1
    print()

    if not getattr(args, "skip_models", False):
        print("=== step 3/9: models init ===")
        try:
            from puppetmaster.model_registry import (
                default_registry_path,
                save_registry,
                starter_registry,
            )

            registry_path = default_registry_path()
            if registry_path.is_file() and not getattr(args, "force", False):
                print(f"  unchanged  registry at {registry_path} already exists (use --force to overwrite)")
            else:
                save_registry(starter_registry(), registry_path)
                print(f"  installed  starter registry written to {registry_path}")
        except Exception as exc:
            print(f"  error  models init failed: {exc!r}")
            overall_rc = 1
        print()
    else:
        print("=== step 3/9: models init SKIPPED (--skip-models) ===\n")

    from puppetmaster import platform_lock as _pl
    enabled_adapters = _pl.enabled_adapters()

    print("=== step 4/9: install-cursor-mcp (workspace .cursor/mcp.json) ===")
    if "cursor" not in enabled_adapters:
        print(
            "  skipped  cursor platform disabled by the platform lock — not "
            "installing its MCP client (.cursor/mcp.json)"
        )
    else:
        sdk = ensure_cursor_sdk(cwd)
        print(f"  sdk {sdk.status}  {sdk.detail}")
        cursor_result = install_cursor_mcp(
            target_path=(cwd / ".cursor" / "mcp.json").resolve(),
            force=getattr(args, "force", False),
            dry_run=False,
            skip_handshake=False,
        )
        for line in cursor_result.messages:
            print(f"  {line}")
        if cursor_result.status not in {"installed", "unchanged", "would_install"}:
            overall_rc = 1
    print()

    print("=== step 5/9: install-codex-mcp ===")
    import shutil as _shutil
    if "codex" not in enabled_adapters:
        print("  skipped  codex platform disabled by the platform lock — not installing its MCP client")
    elif _shutil.which("codex") is None:
        print("  skipped  `codex` CLI not on PATH — install with `npm install -g @openai/codex` and re-run `puppetmaster install-codex-mcp` later")
    else:
        codex_result = install_codex_mcp(
            force=getattr(args, "force", False),
            dry_run=False,
            skip_handshake=False,
        )
        for line in codex_result.messages:
            print(f"  {line}")
        if codex_result.status not in {"installed", "unchanged", "would_install"}:
            overall_rc = 1
    print()

    print("=== step 6/9: install-claude-mcp ===")
    if "claude-code" not in enabled_adapters:
        print("  skipped  claude-code platform disabled by the platform lock — not installing its MCP client")
    elif resolve_claude_command() is None:
        print(
            "  skipped  Claude Code CLI not found — install with "
            "`npm install -g @anthropic-ai/claude-code` (or set CLAUDE_CODE_COMMAND) "
            "and re-run `puppetmaster install-claude-mcp` later"
        )
    else:
        claude_result = install_claude_mcp(
            force=getattr(args, "force", False),
            dry_run=False,
            skip_handshake=False,
        )
        for line in claude_result.messages:
            print(f"  {line}")
        if claude_result.status not in {"installed", "unchanged", "would_install"}:
            overall_rc = 1
    print()

    print("=== step 7/9: install-hermes-mcp + router registry ===")
    if "hermes" not in enabled_adapters:
        print("  skipped  hermes platform disabled by the platform lock — not installing its MCP client")
    elif _shutil.which("hermes") is None:
        print(
            "  skipped  `hermes` CLI not on PATH — install NousResearch hermes-agent "
            "and re-run `puppetmaster install-hermes-mcp` later"
        )
    else:
        hermes_result = install_hermes_mcp(
            force=getattr(args, "force", False),
            dry_run=False,
            skip_handshake=False,
        )
        for line in hermes_result.messages:
            print(f"  {line}")
        if hermes_result.status not in {"installed", "unchanged", "would_install"}:
            overall_rc = 1
        if not getattr(args, "skip_models", False):
            _seed_hermes_registry()
    print()

    if not getattr(args, "skip_rules", False):
        print("=== step 8/9: install-rules (soft agent nudges) ===")
        rules_result = install_rules(
            cwd=cwd,
            install_global=getattr(args, "global_rules", False),
            dry_run=False,
            force=getattr(args, "force", False),
            enabled_adapters=enabled_adapters,
        )
        for outcome in rules_result.outcomes:
            print(f"  {outcome.target:<14} {outcome.status:<14} {outcome.reason}")
        for msg in rules_result.messages:
            print(f"  note: {msg}")
        if rules_result.overall_status == "error":
            overall_rc = 1
    else:
        print("=== step 8/9: install-rules SKIPPED (--skip-rules) ===")
    print()

    if not getattr(args, "skip_hooks", False):
        hooks_scope = "global" if getattr(args, "global_hooks", False) else "project"
        print(f"=== step 9/9: install-hooks (deterministic auto-invocation, scope={hooks_scope}) ===")
        hooks_result = install_hooks(
            cwd=cwd,
            dry_run=False,
            force=getattr(args, "force", False),
            scope=hooks_scope,
            enabled_adapters=enabled_adapters,
        )
        for outcome in hooks_result.outcomes:
            print(f"  {outcome.target:<14} {outcome.status:<14} {outcome.reason}")
        for msg in hooks_result.messages:
            print(f"  note: {msg}")
        if hooks_result.overall_status == "error":
            overall_rc = 1
        scope_note = (
            "user-level (~/.cursor, ~/.claude) — covers every repo you open"
            if hooks_scope == "global"
            else "this workspace only — re-run with --global-hooks to cover every repo"
        )
        print(f"  note: scope is {scope_note}.")
        print(
            "  note: hooks inject a delegate directive on prompt-submit and "
            "deny-redirect recursive shell searches + Task fan-out (read-only "
            "inspection passes through). Disable anytime with "
            "PUPPETMASTER_AUTO_INVOKE_DISABLED=1."
        )
    else:
        print("=== step 9/9: install-hooks SKIPPED (--skip-hooks) ===")
    print()

    if overall_rc == 0:
        print("Setup complete. Restart Cursor (or open a new Codex / Claude session) to pick up the MCP server.")
    else:
        print("Setup completed with errors — see above. Individual `puppetmaster install-*` commands can be re-run after fixing.")
    return overall_rc


def _run_repair_codegraph(args) -> int:
    """CLI entrypoint for `python -m puppetmaster repair-codegraph`.

    Returns 0 when the rebuild succeeds and 1 otherwise. Output mode is
    JSON when ``--json`` is passed and a human-readable summary otherwise.
    """
    result = repair_codegraph_sqlite(
        cursor_node=args.cursor_node,
        codegraph_install=args.codegraph_install,
        npm_command=args.npm_command,
        rebuild_timeout_seconds=args.rebuild_timeout_seconds,
        verify=args.verify,
        verify_cwd=args.verify_cwd,
    )
    if args.json:
        print(json.dumps(result.to_payload(), indent=2))
        return 0 if result.ok else 1

    status = "ok" if result.ok else "fail"
    print(f"repair-codegraph: {status}")
    print(f"  message: {result.message}")
    if result.cursor_node_path:
        version = f" ({result.cursor_node_version})" if result.cursor_node_version else ""
        print(f"  cursor-node: {result.cursor_node_path}{version}")
    if result.codegraph_install_path:
        print(f"  codegraph: {result.codegraph_install_path}")
    if result.verify_backend:
        print(f"  verify: Backend: {result.verify_backend}")
    if result.next_steps:
        print("  next:")
        for step in result.next_steps:
            print(f"    - {step}")
    if not result.ok and result.rebuild_stderr.strip():
        print("  stderr (last 20 lines):")
        for line in result.rebuild_stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
    return 0 if result.ok else 1


def _hoist_global_codegraph_flags(
    cli_args: list[str],
) -> tuple[Optional[str], Optional[int], list[str]]:
    """Pull misplaced ``--cwd``/``--timeout`` out of codegraph passthrough args.

    Returns ``(cwd, timeout, remaining_args)``. Scanning stops at a literal
    ``--`` so codegraph's own flags (forwarded after ``--``) are never touched.
    Supports both ``--cwd X`` and ``--cwd=X`` spellings.
    """
    cwd: Optional[str] = None
    timeout: Optional[int] = None
    remaining: list[str] = []
    index = 0
    forwarding = False
    while index < len(cli_args):
        token = cli_args[index]
        if forwarding:
            remaining.append(token)
            index += 1
            continue
        if token == "--":
            forwarding = True
            remaining.append(token)
            index += 1
            continue
        if token in ("--cwd", "--timeout"):
            if index + 1 < len(cli_args):
                value = cli_args[index + 1]
                if token == "--cwd":
                    cwd = value
                else:
                    try:
                        timeout = int(value)
                    except (TypeError, ValueError):
                        timeout = None
                index += 2
                continue
            index += 1
            continue
        if token.startswith("--cwd="):
            cwd = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("--timeout="):
            try:
                timeout = int(token.split("=", 1)[1])
            except (TypeError, ValueError):
                timeout = None
            index += 1
            continue
        remaining.append(token)
        index += 1
    return cwd, timeout, remaining


def _run_codegraph_passthrough(args) -> int:
    """CLI entrypoint for `python -m puppetmaster codegraph <args>`.

    The whole point: invoke CodeGraph under Cursor's bundled Node via
    ``run_codegraph_cli`` (which resolves that Node and auto-rebuilds the
    native better-sqlite3 binding on an ABI mismatch) instead of a bare
    ``codegraph`` shell call that picks up the wrong Node and dies with a
    ``NODE_MODULE_VERSION`` native-load error. This is the durable fallback
    when the MCP transport is unavailable.
    """
    from puppetmaster.codegraph import run_codegraph_cli

    cli_args = list(args.cg_args or [])
    # Accept the global flags after the subcommand too: `codegraph init --cwd X`
    # is natural to type, but argparse REMAINDER captures `--cwd X` into cg_args,
    # so codegraph saw an unknown option. Hoist any misplaced --cwd/--timeout
    # (up to a literal `--`, which forwards the rest verbatim) onto the global
    # flags when they weren't already supplied before the subcommand.
    hoisted_cwd, hoisted_timeout, cli_args = _hoist_global_codegraph_flags(cli_args)
    if getattr(args, "cwd", None) is None and hoisted_cwd is not None:
        args.cwd = hoisted_cwd
    if not getattr(args, "timeout", 0) and hoisted_timeout is not None:
        args.timeout = hoisted_timeout
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    if not cli_args:
        print("usage: python -m puppetmaster codegraph <subcommand> [args...]", file=sys.stderr)
        print("examples: codegraph status | codegraph search 'router' | codegraph context 'task'", file=sys.stderr)
        return 2

    target = args.cwd or os.getcwd()
    # `status`, `init`, and `help` work before/while a workspace is indexed;
    # everything else needs an initialized `.codegraph/`. (codegraph's own
    # flags like `--version` arrive after a literal `--`, already stripped
    # above, so they pass through to the CLI rather than being gated here.)
    sub = cli_args[0]
    require_initialized = sub not in {"status", "init", "help"}

    timeout_seconds = args.timeout if getattr(args, "timeout", 0) else None
    result = run_codegraph_cli(
        cli_args,
        target,
        require_initialized=require_initialized,
        timeout_seconds=timeout_seconds,
    )

    autoheal = result.get("autoheal")
    if isinstance(autoheal, dict):
        verdict = "ok" if autoheal.get("ok") else "failed"
        print(
            f"[puppetmaster] codegraph native binding rebuilt against Cursor's Node ({verdict}).",
            file=sys.stderr,
        )

    if not result.get("ok"):
        error = result.get("error")
        if error:
            print(error, file=sys.stderr)
        if result.get("stdout"):
            sys.stdout.write(result["stdout"])
        if result.get("stderr"):
            sys.stderr.write(result["stderr"])
        return int(result.get("returncode") or 1)

    if result.get("stdout"):
        sys.stdout.write(result["stdout"])
    if result.get("stderr"):
        sys.stderr.write(result["stderr"])
    return 0


def _run_mcp_subcommand(args) -> int:
    """Dispatch the `python -m puppetmaster mcp ...` family of commands."""
    if args.mcp_command == "list":
        return _run_mcp_list(args)
    if args.mcp_command == "doctor":
        return _run_mcp_doctor(args)
    if args.mcp_command == "cleanup":
        return _run_mcp_cleanup(args)
    raise SystemExit(f"unknown mcp subcommand: {args.mcp_command}")


def _run_mcp_doctor(args) -> int:
    """Diagnose a `Tool execution error. Not connected`.

    The actual failure mode the rules document: the Puppetmaster daemon/MCP
    server is fine, but this chat's stdio pipe dropped. This separates that
    (restart the MCP client) from a genuinely dead server (no process tracked)
    so the user knows whether to restart or to fall back to the CLI and keep
    working."""
    snapshot = registry_summarize(registry_list_entries())
    alive = snapshot["alive"]
    stale = snapshot["stale"]
    dead = snapshot["dead"]
    tracked = snapshot["count"]

    if alive > 0:
        verdict = "stdio_pipe_dropped"
        headline = (
            f"{alive} MCP server(s) alive. The Puppetmaster daemon is healthy."
        )
        remedy = (
            "If your MCP client reports 'Tool execution error. Not connected', "
            "the stdio pipe for THIS chat dropped — reconnect Puppetmaster in "
            "your MCP client settings (or just keep using the `python -m puppetmaster` CLI; "
            "durable state is unaffected)."
        )
    elif tracked > 0:
        verdict = "servers_stale_or_dead"
        headline = (
            f"No alive MCP server (tracked={tracked}, stale={stale}, dead={dead})."
        )
        remedy = (
            "Run `python -m puppetmaster mcp cleanup --kill-stale` to clear "
            "orphans, then reconnect Puppetmaster in your MCP client settings."
        )
    else:
        verdict = "no_server"
        headline = "No Puppetmaster MCP server is tracked on this machine."
        remedy = (
            "Run the MCP installer for your host "
            "(`python -m puppetmaster install-codex-mcp` or "
            "`python -m puppetmaster install-cursor-mcp`), then reconnect "
            "Puppetmaster in your MCP client settings."
        )

    if args.json:
        print(
            json.dumps(
                {"verdict": verdict, "headline": headline, "remedy": remedy, **snapshot},
                indent=2,
            )
        )
        return 0
    print(f"verdict: {verdict}")
    print(headline)
    print(f"  -> {remedy}")
    # Non-zero only when there's no usable server at all.
    return 0 if alive > 0 else 1


def _registry_path_from_args(args) -> Optional[Path]:
    raw = getattr(args, "registry_path", None)
    return Path(raw).expanduser() if raw else None


def _run_platform_subcommand(args) -> int:
    """Dispatch `python -m puppetmaster platform ...`.

    The platform lock decides which adapters (cursor, claude-code, codex,
    openai) Puppetmaster may route to, auto-discover, or fall back onto. It is
    persisted next to the model registry; an empty lock means everything is on.
    """
    import json as _json

    from puppetmaster import platform_lock as pl

    path = _registry_path_from_args(args)
    sub = args.platform_command

    def _normalize(raw_adapters) -> tuple[set[str], list[str]]:
        wanted = {a.strip() for a in raw_adapters if a.strip()}
        unknown = sorted(a for a in wanted if a not in pl.KNOWN_ADAPTERS)
        return {a for a in wanted if a in pl.KNOWN_ADAPTERS}, unknown

    if sub in ("only", "enable", "disable"):
        valid, unknown = _normalize(args.adapters)
        if unknown:
            print(
                f"error: unknown platform(s): {', '.join(unknown)}. "
                f"Known: {', '.join(pl.KNOWN_ADAPTERS)}.",
                file=sys.stderr,
            )
            return 1
        if not valid:
            print("error: name at least one known platform.", file=sys.stderr)
            return 1
        if sub == "only":
            pl.set_enabled(valid, path)
        elif sub == "enable":
            pl.enable(valid, path)
        else:
            pl.disable(valid, path)
        # Fall through to status so the user always sees the resulting state.
    elif sub == "reset":
        pl.reset(path)

    enabled = pl.enabled_adapters(path)
    restricted = pl.is_restricted(path)
    env_override = bool((os.environ.get(pl.ONLY_ENV) or "").strip())

    if getattr(args, "json", False):
        print(
            _json.dumps(
                {
                    "enabled": sorted(enabled),
                    "disabled": sorted(set(pl.KNOWN_ADAPTERS) - enabled),
                    "restricted": restricted,
                    "env_override": env_override,
                    "config_path": str(pl.platform_config_path(path)),
                },
                indent=2,
            )
        )
        return 0

    if restricted:
        print(f"Platform lock ACTIVE — only: {', '.join(sorted(enabled)) or '(none)'}")
    else:
        print("Platform lock: off (all platforms enabled)")
    for adapter in pl.KNOWN_ADAPTERS:
        mark = "on " if adapter in enabled else "off"
        print(f"  [{mark}] {adapter}")
    if env_override:
        print(
            f"note: ${pl.ONLY_ENV} is set and overrides the saved config "
            "for this shell."
        )
    return 0


_OPENAI_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh")
_CODEX_EFFORT_LEVELS = ("low", "medium", "high")
_EFFORT_TOKEN_MULTIPLIERS = {
    "none": 0.7,
    "low": 0.7,
    "medium": 1.0,
    "high": 2.0,
    "xhigh": 3.0,
}


def model_payload_defaults_for_effort(adapter: str, effort: str) -> dict[str, Any]:
    """Translate a registry effort level into adapter payload defaults."""
    normalized = effort.strip().lower()
    if adapter == "openai":
        if normalized not in _OPENAI_EFFORT_LEVELS:
            raise ValueError(
                "openai effort must be one of "
                + ", ".join(_OPENAI_EFFORT_LEVELS)
            )
        return {"reasoning_effort": normalized}
    if adapter == "codex":
        if normalized not in _CODEX_EFFORT_LEVELS:
            raise ValueError("codex effort must be one of " + ", ".join(_CODEX_EFFORT_LEVELS))
        return {"extra_args": ["-c", f"model_reasoning_effort={normalized}"]}
    if adapter in ("claude-code", "cursor"):
        raise ValueError(
            f"{adapter} does not expose an effort knob through its CLI/SDK today."
        )
    raise ValueError(f"adapter {adapter!r} does not have known effort support")


def _payload_defaults_summary(payload_defaults: dict[str, Any]) -> str:
    if not payload_defaults:
        return "-"
    effort = payload_defaults.get("reasoning_effort")
    if effort:
        return f"effort={effort}"
    extra_args = payload_defaults.get("extra_args")
    if isinstance(extra_args, list):
        for arg in extra_args:
            if isinstance(arg, str) and arg.startswith("model_reasoning_effort="):
                return "effort=" + arg.split("=", 1)[1]
    return ",".join(sorted(payload_defaults))


def _parse_bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _json_assignment_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected JSON value, got {raw_value!r}") from exc


def _replace_model_spec(spec, **updates):
    from puppetmaster.model_registry import ModelSpec

    data = dataclasses.asdict(spec)
    data.update(updates)
    return ModelSpec(**data)


def _updated_spec_for_assignment(spec, key: str, value: str):
    if key == "capability_score":
        return _replace_model_spec(spec, capability_score=int(value))
    if key == "enabled":
        return _replace_model_spec(spec, enabled=_parse_bool_value(value))
    if key == "notes":
        return _replace_model_spec(spec, notes=value)
    if key == "billing":
        return _replace_model_spec(spec, billing=value)
    if key == "output_token_multiplier":
        return _replace_model_spec(spec, output_token_multiplier=float(value))
    if key == "effort":
        level = value.strip().lower()
        effort_defaults = model_payload_defaults_for_effort(spec.adapter, level)
        # Merge so unrelated defaults (e.g. temperature) survive an effort change,
        # and swap the effort:* tag to keep CLI and wizard entries consistent.
        payload_defaults = {**(spec.payload_defaults or {}), **effort_defaults}
        tags = [tag for tag in spec.tags if not tag.startswith("effort:")]
        tags.append(f"effort:{level}")
        return _replace_model_spec(spec, payload_defaults=payload_defaults, tags=tags)
    if key.startswith("payload_defaults."):
        payload_key = key[len("payload_defaults.") :]
        if not payload_key:
            raise ValueError("payload_defaults assignment needs a key")
        payload_defaults = dict(spec.payload_defaults or {})
        payload_defaults[payload_key] = _json_assignment_value(value)
        return _replace_model_spec(spec, payload_defaults=payload_defaults)
    raise ValueError(f"unknown models set key: {key}")


def _run_models_set(args, path: Path) -> int:
    from puppetmaster.model_registry import load_registry, save_registry

    try:
        specs = load_registry(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    by_id = {spec.id: i for i, spec in enumerate(specs)}
    if args.model_id not in by_id:
        print(f"error: unknown model id: {args.model_id}", file=sys.stderr)
        return 1

    index = by_id[args.model_id]
    updated = specs[index]
    try:
        for assignment in args.assignments:
            if "=" not in assignment:
                raise ValueError(f"expected key=value assignment, got {assignment!r}")
            key, value = assignment.split("=", 1)
            updated = _updated_spec_for_assignment(updated, key, value)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    specs[index] = updated
    save_registry(specs, path)
    print(json.dumps(dataclasses.asdict(updated), indent=2))
    return 0


class ModelRegistryWizard:
    def __init__(self, path: Path, stdin: TextIO, stdout: TextIO) -> None:
        self.path = path
        self.stdin = stdin
        self.stdout = stdout
        self.specs = []
        self.dirty = False

    def run(self) -> int:
        try:
            return self._run_menu_loop()
        except EOFError:
            self._write("")
            if self.dirty:
                self._write("Input closed — discarding unsaved changes.")
                return 1
            self._write("Input closed — nothing to save.")
            return 0

    def _run_menu_loop(self) -> int:
        from puppetmaster.model_registry import load_registry, save_registry, starter_registry

        if self.path.is_file():
            self.specs = load_registry(self.path)
        else:
            self._write(f"Registry not found at {self.path}.")
            if self._confirm("Write the starter registry now?", default=True):
                self.specs = starter_registry()
                self.dirty = True
            else:
                self.specs = []
        self.show_table()
        while True:
            self._write("")
            self._write("Choose: [1] effort variant  [2] edit field  [3] add model")
            self._write("        [4] remove entry     [5] show table  [q] save & quit")
            choice = self._prompt("> ").strip().lower()
            if choice == "1":
                self.add_effort_variant()
            elif choice == "2":
                self.edit_entry()
            elif choice == "3":
                self.add_model_entry()
            elif choice == "4":
                self.remove_entry()
            elif choice == "5":
                self.show_table()
            elif choice == "q":
                if self.dirty:
                    if self._confirm("Save changes?", default=True):
                        saved = save_registry(self.specs, self.path)
                        self._write(f"Saved registry to {saved}")
                        return 0
                    if self._confirm("Quit without saving?", default=False):
                        self._write("Discarded changes.")
                        return 0
                    continue
                self._write("No changes to save.")
                return 0
            else:
                self._write("Please choose 1, 2, 3, 4, 5, or q.")

    def show_table(self) -> None:
        self._write("")
        self._write(f"{len(self.specs)} model(s)  ({self.path})")
        self._write(
            f"{'#':>2}  {'ID':<28}  {'ADAPTER':<12}  {'MODEL':<18}  "
            f"{'CAP':>3}  {'BILLING':<7}  DEFAULTS"
        )
        for index, spec in enumerate(self.specs, 1):
            disabled = "" if spec.enabled else " [disabled]"
            self._write(
                f"{index:>2}  {spec.id:<28}  {spec.adapter:<12}  "
                f"{spec.adapter_model_name:<18}  {spec.capability_score:>3}  "
                f"{spec.billing:<7}  {_payload_defaults_summary(spec.payload_defaults)}{disabled}"
            )

    def add_effort_variant(self) -> None:
        base = self._choose_spec("Base model number")
        if base is None:
            return
        if base.adapter in ("claude-code", "cursor"):
            self._write(
                f"{base.adapter} does not expose an effort knob through its CLI/SDK today."
            )
            return
        levels = _OPENAI_EFFORT_LEVELS if base.adapter == "openai" else _CODEX_EFFORT_LEVELS
        self._write("Supported efforts: " + ", ".join(levels))
        effort = self._prompt_default("Effort level", "high").strip().lower()
        try:
            payload_defaults = model_payload_defaults_for_effort(base.adapter, effort)
        except ValueError as exc:
            self._write(f"error: {exc}")
            return

        suggested_id = f"{base.id}-{effort}"
        model_id = self._prompt_default("New model id", suggested_id).strip()
        if any(spec.id == model_id for spec in self.specs):
            self._write(f"error: a registry entry with id {model_id!r} already exists.")
            return
        self._write("Higher effort usually means higher capability and higher token burn.")
        capability_score = self._prompt_int(
            "Capability score", base.capability_score, minimum=0, maximum=100
        )
        multiplier_default = _EFFORT_TOKEN_MULTIPLIERS.get(effort, 1.0)
        multiplier_default *= base.output_token_multiplier
        self._write("Output token multiplier estimates extra hidden reasoning/output volume.")
        output_token_multiplier = self._prompt_float(
            "Output token multiplier", multiplier_default, minimum_exclusive=0.0
        )
        tags = list(base.tags)
        effort_tag = f"effort:{effort}"
        if effort_tag not in tags:
            tags.append(effort_tag)
        note = f"{effort.capitalize()}-effort variant of {base.id}."
        try:
            new_spec = _replace_model_spec(
                base,
                id=model_id,
                capability_score=capability_score,
                tags=tags,
                notes=note,
                payload_defaults=payload_defaults,
                output_token_multiplier=output_token_multiplier,
            )
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        self._write(
            f"Add {new_spec.id}: adapter={new_spec.adapter}, "
            f"defaults={new_spec.payload_defaults}, multiplier={new_spec.output_token_multiplier:g}"
        )
        if self._confirm("Add this entry?", default=True):
            self.specs.append(new_spec)
            self.dirty = True
            self._write(f"Added {new_spec.id}.")

    def edit_entry(self) -> None:
        spec = self._choose_spec("Entry number")
        if spec is None:
            return
        fields = ("capability_score", "tags", "notes", "enabled", "output_token_multiplier")
        self._write("Fields: " + ", ".join(fields))
        field = self._prompt("Field: ").strip()
        if field not in fields:
            self._write(f"Unknown editable field: {field}")
            return
        try:
            if field == "capability_score":
                value = self._prompt_int("Capability score", spec.capability_score, 0, 100)
                updated = _replace_model_spec(spec, capability_score=value)
            elif field == "tags":
                raw = self._prompt_default("Tags (comma-separated)", ",".join(spec.tags))
                tags = [item.strip() for item in raw.split(",") if item.strip()]
                updated = _replace_model_spec(spec, tags=tags)
            elif field == "notes":
                updated = _replace_model_spec(
                    spec, notes=self._prompt_default("Notes", spec.notes)
                )
            elif field == "enabled":
                updated = _replace_model_spec(
                    spec,
                    enabled=self._confirm("Enabled?", default=spec.enabled),
                )
            else:
                value = self._prompt_float(
                    "Output token multiplier",
                    spec.output_token_multiplier,
                    minimum_exclusive=0.0,
                )
                updated = _replace_model_spec(spec, output_token_multiplier=value)
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        self._write(f"Update {spec.id}: {field} -> {getattr(updated, field)!r}")
        if self._confirm("Apply this change?", default=True):
            self.specs[self.specs.index(spec)] = updated
            self.dirty = True

    def add_model_entry(self) -> None:
        self._write("Add a brand-new registry entry.")
        try:
            spec = self._build_new_model_spec()
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        if any(existing.id == spec.id for existing in self.specs):
            self._write(f"error: a registry entry with id {spec.id!r} already exists.")
            return
        self._write(f"Add {spec.id}: adapter={spec.adapter}, model={spec.adapter_model_name}")
        if self._confirm("Add this entry?", default=True):
            self.specs.append(spec)
            self.dirty = True

    def remove_entry(self) -> None:
        spec = self._choose_spec("Entry number to remove")
        if spec is None:
            return
        self._write(f"Remove {spec.id} ({spec.adapter}/{spec.adapter_model_name}).")
        if self._confirm("Remove this entry?", default=False):
            self.specs.remove(spec)
            self.dirty = True
            self._write(f"Removed {spec.id}.")

    def _build_new_model_spec(self):
        from puppetmaster.model_registry import ModelSpec

        model_id = self._prompt("ID: ").strip()
        adapter = self._prompt("Adapter: ").strip()
        adapter_model_name = self._prompt("Adapter model name: ").strip()
        capability_score = self._prompt_int("Capability score", 50, 0, 100)
        input_price = self._prompt_float("Input $/Mtok", 0.0, minimum_inclusive=0.0)
        output_price = self._prompt_float("Output $/Mtok", 0.0, minimum_inclusive=0.0)
        context_window = self._prompt_int("Context window tokens", 0, minimum=0)
        billing = self._prompt_default("Billing (plan/api/unknown)", "unknown")
        raw_tags = self._prompt_default("Tags (comma-separated)", adapter)
        tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
        notes = self._prompt_default("Notes", "")
        return ModelSpec(
            id=model_id,
            adapter=adapter,
            adapter_model_name=adapter_model_name,
            capability_score=capability_score,
            input_per_mtok_usd=input_price,
            output_per_mtok_usd=output_price,
            context_window=context_window,
            billing=billing,
            tags=tags,
            notes=notes,
        )

    def _choose_spec(self, prompt: str):
        if not self.specs:
            self._write("No registry entries yet.")
            return None
        while True:
            raw = self._prompt(f"{prompt}: ").strip()
            if not raw:
                return None
            try:
                index = int(raw)
            except ValueError:
                self._write("Enter a number, or blank to cancel.")
                continue
            if 1 <= index <= len(self.specs):
                return self.specs[index - 1]
            self._write("That number is not in the table.")

    def _prompt(self, prompt: str) -> str:
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            # Closed stdin must abort the wizard, not echo "" forever:
            # every prompt loop treats "" as retryable input.
            raise EOFError("input closed")
        return line.rstrip("\n")

    def _prompt_default(self, prompt: str, default) -> str:
        raw = self._prompt(f"{prompt} [{default}]: ")
        return str(default) if raw == "" else raw

    def _prompt_int(
        self, prompt: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None
    ) -> int:
        while True:
            raw = self._prompt_default(prompt, default)
            try:
                value = int(raw)
            except ValueError:
                self._write("Enter an integer.")
                continue
            if minimum is not None and value < minimum:
                self._write(f"Enter a value >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                self._write(f"Enter a value <= {maximum}.")
                continue
            return value

    def _prompt_float(
        self,
        prompt: str,
        default: float,
        minimum_inclusive: Optional[float] = None,
        minimum_exclusive: Optional[float] = None,
    ) -> float:
        while True:
            raw = self._prompt_default(prompt, default)
            try:
                value = float(raw)
            except ValueError:
                self._write("Enter a number.")
                continue
            if minimum_inclusive is not None and value < minimum_inclusive:
                self._write(f"Enter a value >= {minimum_inclusive:g}.")
                continue
            if minimum_exclusive is not None and value <= minimum_exclusive:
                self._write(f"Enter a value > {minimum_exclusive:g}.")
                continue
            return value

    def _confirm(self, prompt: str, default: bool = False) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            raw = self._prompt(prompt + suffix).strip().lower()
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            self._write("Please answer y or n.")

    def _write(self, message: str) -> None:
        print(message, file=self.stdout)


def _run_models_setup(args, path: Path) -> int:
    try:
        return ModelRegistryWizard(path, sys.stdin, sys.stdout).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_models_subcommand(args) -> int:
    """Dispatch `python -m puppetmaster models ...`.

    Three subcommands:

    * ``init`` — write a starter registry the user can edit.
    * ``list`` — show what's registered, including price + capability.
    * ``path`` — print the resolved registry path (handy in scripts).
    """
    from puppetmaster.model_registry import (
        default_registry_path,
        load_registry,
        save_registry,
        starter_registry,
    )

    path = _registry_path_from_args(args) or default_registry_path()

    if args.models_command == "path":
        print(path)
        return 0

    if args.models_command == "init":
        if path.is_file() and not args.force:
            print(
                f"error: {path} already exists; pass --force to overwrite",
                file=sys.stderr,
            )
            return 1
        save_registry(starter_registry(), path)
        print(f"wrote starter model registry to {path}")
        print("Edit capability_score / prices to match your subscriptions.")
        return 0

    if args.models_command == "list":
        try:
            specs = load_registry(path)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            from dataclasses import asdict

            print(json.dumps({"path": str(path), "models": [asdict(s) for s in specs]}, indent=2))
            return 0
        if not specs:
            print(f"No models registered (looked at {path}).")
            print("Run `puppetmaster models init` to write a starter registry.")
            return 0
        print(f"{len(specs)} model(s) registered  ({path})")
        print(
            f"  {'ID':<28}  {'ADAPTER':<12}  {'CAP':>3}  "
            f"{'IN $/Mtok':>10}  {'OUT $/Mtok':>10}  TAGS"
        )
        for spec in specs:
            tags = ",".join(spec.tags) if spec.tags else "-"
            disabled = "" if spec.enabled else "  [disabled]"
            print(
                f"  {spec.id:<28}  {spec.adapter:<12}  "
                f"{spec.capability_score:>3}  "
                f"{spec.input_per_mtok_usd:>10.3f}  "
                f"{spec.output_per_mtok_usd:>10.3f}  {tags}{disabled}"
            )
        return 0

    if args.models_command == "discover":
        return _run_models_discover(args, path)

    if args.models_command == "setup":
        return _run_models_setup(args, path)

    if args.models_command == "set":
        return _run_models_set(args, path)

    raise SystemExit(f"unknown models subcommand: {args.models_command}")


def _run_models_discover(args, path: Path) -> int:
    """Enumerate platform model catalogs and reconcile them into the registry.

    Cursor (plan, default) uses the SDK; OpenAI and Anthropic use their
    ``/v1/models`` endpoints. ``--source all`` runs every reachable source."""
    import json as _json

    from puppetmaster.model_registry import (
        load_registry,
        save_registry,
        starter_registry,
        write_discovery_meta,
    )

    try:
        registry = load_registry(path)
    except RuntimeError:
        registry = starter_registry()

    if args.source == "all":
        sources = ["cursor", "openai", "anthropic"]
        # Fold Hermes into the catch-all only when its CLI is actually present,
        # so users who don't run Hermes don't get hermes/* entries injected just
        # because they happen to have an OPENAI/ANTHROPIC/GEMINI key set. The
        # explicit `--source hermes` path always works regardless.
        from puppetmaster.diagnostics import _hermes_cli_installed

        if _hermes_cli_installed():
            sources.append("hermes")
    else:
        sources = [args.source]
    reports: list[dict] = []
    catalogs: dict[str, list] = {}
    errors: dict[str, str] = {}

    for source in sources:
        try:
            registry, report, catalog = _discover_one_source(source, registry)
        except _DiscoverSourceError as exc:
            errors[source] = str(exc)
            if args.source != "all":
                print(f"error: {exc}", file=sys.stderr)
                return 1
            continue
        reports.append(report)
        catalogs[source] = catalog

    if args.json:
        print(
            _json.dumps(
                {
                    "reports": reports,
                    "errors": errors,
                    "catalogs": catalogs,
                    "written": bool(args.write),
                    "registry_path": str(path),
                },
                indent=2,
            )
        )
    else:
        for report in reports:
            src = report.get("source") or report.get("adapter") or "cursor"
            print(f"[{src}] discovered {report['discovered_count']} model(s).")
            if report.get("added"):
                print(f"  + new: {', '.join(report['added'])}")
            if report.get("dropped_stale_cursor_models"):
                print(
                    f"  - dropped (no longer in plan): "
                    f"{', '.join(report['dropped_stale_cursor_models'])}"
                )
        for src, err in errors.items():
            print(f"[{src}] skipped: {err}")

    if args.write and reports:
        save_registry(registry, path)
        for report in reports:
            src = report.get("source") or report.get("adapter")
            write_discovery_meta(src, report["discovered_count"], path)
        if not args.json:
            print(f"Wrote merged registry to {path}")
    elif not args.json and reports:
        print("Dry run — pass --write to persist.")
    return 0 if reports or not errors else 1


class _DiscoverSourceError(RuntimeError):
    pass


def _discover_one_source(source: str, registry: list):
    """Fetch + merge one catalog source; returns (registry, report, catalog)."""
    if source == "cursor":
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            fetch_cursor_catalog,
            merge_catalog_into_registry,
        )

        try:
            catalog = fetch_cursor_catalog()
        except CursorDiscoveryError as exc:
            raise _DiscoverSourceError(str(exc)) from exc
        merged, report = merge_catalog_into_registry(registry, catalog)
        report["source"] = "cursor"
        return merged, report, catalog

    if source == "hermes":
        from puppetmaster.adapters import available_hermes_providers
        from puppetmaster.static_catalog import (
            curated_catalog,
            merge_curated_into_registry,
        )

        # Hermes always bills per-token to the user's own provider key (no
        # subscription posture to detect, so billing is unconditionally "api").
        # Seed only models whose provider has a usable credential so the router
        # never picks a Hermes model it can't actually call.
        allowed = available_hermes_providers()
        merged, report = merge_curated_into_registry(
            "hermes", "api", registry, allowed_providers=allowed
        )
        report["source"] = "hermes"
        report["available_providers"] = sorted(allowed)
        catalog = [
            {"id": item["model"]}
            for item in curated_catalog("hermes")
            if (item.get("payload_defaults") or {}).get("provider") in allowed
        ]
        return merged, report, catalog

    if source in ("claude", "codex"):
        from puppetmaster.platform_billing import detect_adapter_billing
        from puppetmaster.static_catalog import (
            SOURCE_TO_ADAPTER,
            curated_catalog,
            merge_curated_into_registry,
        )

        adapter = SOURCE_TO_ADAPTER[source]
        status = detect_adapter_billing(adapter)
        # Use the detected posture so prices/billing are truthful; fall back to
        # API-billed reference pricing when auth can't be determined, so the
        # curated entries are still usable rather than silently $0.
        billing = (
            status.billing
            if getattr(status, "healthy", False)
            and getattr(status, "billing", "unknown") in ("plan", "api")
            else "api"
        )
        merged, report = merge_curated_into_registry(adapter, billing, registry)
        catalog = [{"id": item["model"]} for item in curated_catalog(adapter)]
        report["source"] = source
        return merged, report, catalog

    from puppetmaster.api_discovery import (
        ApiDiscoveryError,
        fetch_anthropic_models,
        fetch_openai_models,
        merge_api_catalog_into_registry,
    )

    try:
        if source == "openai":
            catalog = fetch_openai_models()
            merged, report = merge_api_catalog_into_registry(
                "openai", "api", registry, catalog
            )
        elif source == "anthropic":
            catalog = fetch_anthropic_models()
            merged, report = merge_api_catalog_into_registry(
                "claude-code", "unknown", registry, catalog
            )
        else:
            raise _DiscoverSourceError(f"unknown source: {source}")
    except ApiDiscoveryError as exc:
        raise _DiscoverSourceError(str(exc)) from exc
    report["source"] = source
    return merged, report, catalog


def await_job_state(
    store,
    job_id: str,
    *,
    timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    """Block until ``job_id`` reaches a terminal state or the timeout elapses.

    Returns ``{status, terminal, timed_out, completed_at}``. ``timeout_seconds``
    of 0 blocks indefinitely (CLI/SDK path); a positive value bounds the wait
    (so the MCP path can return and be re-called). Uses the store's
    event-wait primitive between checks instead of busy-polling.
    """
    from puppetmaster.models import JobStatus

    terminal = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.STALLED}
    poll = max(0.05, poll_interval_seconds)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    cursor = 0
    while True:
        job = store.get_job(job_id)
        if job.status in terminal:
            return {
                "job_id": job_id,
                "status": str(job.status),
                "terminal": True,
                "timed_out": False,
                "completed_at": job.completed_at,
            }
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "job_id": job_id,
                "status": str(job.status),
                "terminal": False,
                "timed_out": True,
                "completed_at": job.completed_at,
            }
        block = poll if deadline is None else max(0.05, min(poll * 4, deadline - time.monotonic()))
        events = store.wait_for_events(
            job_id,
            since=cursor,
            timeout_seconds=max(0.05, block),
            poll_interval=poll,
        )
        # Advance the cursor past the events we just observed. Without this the
        # cursor stayed at 0, so once any event existed wait_for_events returned
        # immediately every iteration and the loop hot-spun (re-reading the
        # whole event stream) until the job reached a terminal state.
        for event in events:
            event_id = event.get("id")
            if isinstance(event_id, int) and event_id > cursor:
                cursor = event_id


def _reap_quietly(store) -> list[dict]:
    """Run the stalled-job reaper, swallowing any failure.

    Wired into read-side commands (status/jobs/wait) so a dead-but-"running"
    job is transitioned to stalled the next time anyone looks, without the user
    having to remember a separate command. Never raises into the caller."""
    try:
        from puppetmaster.liveness import reap_stalled_jobs

        return reap_stalled_jobs(store)
    except Exception:
        return []


def _run_finalize_command(args, store) -> int:
    """Force-stitch a job and mark it complete.

    Recovery path for a job whose orchestrator died after the workers finished
    but before it could stitch — exactly the run-swarm finalize gap. Stitching
    is idempotent (it just rewrites summaries/stitched.md from artifacts)."""
    from puppetmaster.models import JobStatus

    job = store.get_job(args.job_id)
    Stitcher(store).stitch(args.job_id)  # side effect: (re)writes summaries/stitched.md
    summary_path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
    # Only advance a non-terminal job to complete; never override an explicit
    # FAILED verdict.
    if job.status not in {JobStatus.COMPLETE, JobStatus.FAILED}:
        store.update_job_status(args.job_id, JobStatus.COMPLETE)
    print(f"finalized: {args.job_id}")
    print(f"summary: {summary_path}")
    return 0


def _run_reap_command(args, store) -> int:
    from puppetmaster.liveness import reap_stalled_jobs

    reaped = reap_stalled_jobs(store, stall_after_seconds=args.stall_after_seconds)
    if args.json:
        print(json.dumps(reaped, indent=2))
        return 0
    if not reaped:
        print("no stalled jobs found")
        return 0
    print(f"stalled: {len(reaped)}")
    for row in reaped:
        print(
            f"  {row['job_id']}\treason={row['reason']}\t"
            f"requeued_tasks={row['requeued_tasks']}"
        )
    return 0


def _gc_target_stores(args, store) -> list:
    """The stores `gc`/`rollup` should sweep: just this project, or every one."""
    if not getattr(args, "all_projects", False):
        return [store]
    stores = []
    for project in list_project_state_dirs():
        try:
            stores.append(create_store(args.backend, project))
        except Exception:
            continue
    return stores or [store]


def _run_gc_command(args, store) -> int:
    from puppetmaster.lifecycle import gc_terminal_jobs

    all_projects = getattr(args, "all_projects", False)
    active_root = _resolved_store_root(store)
    reaped: list[dict] = []
    protected_active = False
    for target in _gc_target_stores(args, store):
        # D1 (P0): a `gc --force --all-projects` sweep must never destroy the
        # active worktree's state out from under live work. Reap the active
        # project only when the user targets it explicitly (plain `gc --force`,
        # no --all-projects); under --all-projects we report it dry-run only.
        is_active_under_sweep = all_projects and _resolved_store_root(target) == active_root
        effective_force = args.force and not is_active_under_sweep
        if args.force and is_active_under_sweep:
            protected_active = True
        reaped.extend(
            gc_terminal_jobs(
                target, older_than_days=args.older_than_days, force=effective_force
            )
        )
    if args.json:
        print(json.dumps(
            {"reaped": reaped, "deleted": args.force, "protected_active_worktree": protected_active},
            indent=2,
        ))
        return 0
    if not reaped:
        print(f"gc: no terminal jobs older than {args.older_than_days}d to reap")
        return 0
    verb = "reaped" if args.force else "would reap (dry-run; pass --force)"
    print(f"gc: {verb} {len(reaped)} job(s):")
    for row in reaped:
        print(f"  {row['job_id']}\t{row['status']}\t{row['age_days']}d\t{row['goal'][:60]}")
    if not args.force:
        print("\n  Re-run with --force to delete this state.")
    if protected_active:
        print(
            "\n  note: skipped the active worktree's state under --all-projects "
            "(reported dry-run above). Run plain `gc --force` here to reap it.",
            file=sys.stderr,
        )
    return 0


def _resolved_store_root(store) -> Optional[str]:
    """Best-effort resolved filesystem root for a store, for active-worktree
    comparison. Returns None when it can't be determined."""
    try:
        return str(Path(store.root).resolve())
    except Exception:
        return None


def _parse_inline_rules(rule_args: list[str]) -> list[dict]:
    """Turn 'src/**/*.py=>tests/{stem}_test.py,tests/smoke.py' shorthand into
    mapping rules."""
    rules: list[dict] = []
    for raw in rule_args or []:
        if "=>" not in raw:
            raise SystemExit(f"affected: bad --rule (missing '=>'): {raw!r}")
        match, specs = raw.split("=>", 1)
        rules.append({
            "match": match.strip(),
            "specs": [s.strip() for s in specs.split(",") if s.strip()],
        })
    return rules


def _run_affected_command(args) -> int:
    from puppetmaster.affected import affected_specs, changed_files_from_git, load_mapping

    cwd = Path(args.cwd)
    mapping: dict[str, Any] = {}
    if args.config:
        mapping = load_mapping(args.config)
    inline_rules = _parse_inline_rules(args.rule)
    if inline_rules:
        mapping = dict(mapping)
        mapping["rules"] = list(mapping.get("rules") or []) + inline_rules
    if not mapping.get("rules") and not mapping.get("command"):
        raise SystemExit("affected: provide --config and/or --rule defining a mapping")

    if args.git_range:
        changed = changed_files_from_git(cwd, args.git_range)
    elif args.changed:
        changed = list(args.changed)
    else:
        changed = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]

    specs = affected_specs(changed, mapping, cwd=cwd)
    if args.json:
        print(json.dumps({"changed": changed, "affected_specs": specs}, indent=2))
    else:
        for spec in specs:
            print(spec)
    return 0


def _run_rollup_command(args, store) -> int:
    from puppetmaster.lifecycle import rollup_stores

    rollup = rollup_stores(_gc_target_stores(args, store), effort_id=args.effort)
    if args.json:
        print(json.dumps(rollup, indent=2))
        return 0
    scope = f"effort '{args.effort}'" if args.effort else "all jobs"
    print(f"rollup ({scope}):")
    print(f"  jobs:      {rollup['jobs']}  {rollup['jobs_by_status']}")
    print(f"  artifacts: {rollup['artifacts']}")
    print(f"  est. cost: ${rollup['estimated_cost_usd']:.6f} (pre-flight routing estimate)")
    usage = rollup["token_usage"]
    if usage["measured_runs"] or usage["estimated_runs"]:
        print(
            f"  tokens:    {usage['measured_tokens_in'] + usage['measured_tokens_out']:,} measured / "
            f"~{usage['estimated_tokens_in'] + usage['estimated_tokens_out']:,} estimated"
        )
    if not args.effort and rollup["efforts_seen"]:
        print(f"  efforts seen: {', '.join(rollup['efforts_seen'])}")
    return 0


def _gate_specs_from_args(args) -> list[dict]:
    """Translate the gate flags + --gates-json into the same spec list the
    runtime resolves from ``task.payload['gates']``."""
    specs: list[dict] = []
    if args.gates_json:
        parsed = json.loads(args.gates_json)
        if not isinstance(parsed, list):
            raise ValueError("--gates-json must be a JSON array of gate objects")
        specs.extend(s for s in parsed if isinstance(s, dict) and s.get("kind"))
    if args.require_diff:
        specs.append({"kind": "require_diff"})
    if args.gate_command:
        specs.append({"kind": "command", "command": args.gate_command})
    if args.ratchet_command or args.metric:
        if not (args.ratchet_command and args.metric):
            raise ValueError("--ratchet-command and --metric must be given together")
        specs.append(
            {"kind": "ratchet", "command": args.ratchet_command, "metric": args.metric}
        )
    if args.committed:
        specs.append({"kind": "committed"})
    return specs


def _run_gate_command(args, store) -> int:
    """Replay completion gates against a working tree outside a worker run, so
    a parent agent or CI can enforce the very same post-conditions the runtime
    applies at task completion. Exits non-zero when any gate fails."""
    from puppetmaster.gates import evaluate_task_gates
    from puppetmaster.models import Task

    try:
        specs = _gate_specs_from_args(args)
    except ValueError as exc:
        print(f"gate: {exc}")
        return 2
    if not specs:
        print(
            "gate: no gates specified. Pass --require-diff / --command / "
            "--ratchet-command+--metric / --committed / --gates-json."
        )
        return 2

    cwd = Path(args.cwd).resolve()
    task = Task(
        job_id="gate-replay",
        role="gate",
        instruction="gate replay",
        payload={"gates": specs, "cwd": str(cwd)},
    )
    evaluation = evaluate_task_gates(
        task, artifacts=[], store=store, worker_id="gate-replay", cwd=cwd
    )

    rows = [
        {
            "gate": result.name,
            "kind": result.kind,
            "passed": result.passed,
            "reason": result.reason,
        }
        for result in evaluation.results
    ]
    if args.json:
        print(json.dumps({"passed": evaluation.passed, "gates": rows}, indent=2))
    else:
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"  [{mark}] {row['gate']} ({row['kind']}): {row['reason']}")
        print(f"\ngate: {'all gates passed' if evaluation.passed else 'GATE FAILED'}")
    return 0 if evaluation.passed else 1


def _run_wait_command(args, store) -> int:
    """Block until a job reaches a terminal state, running the reaper between
    checks so a stalled job is detected (not waited on forever). Exits non-zero
    when the job did not complete cleanly."""
    poll = max(0.05, args.poll_interval_seconds)
    deadline = (
        time.monotonic() + args.timeout_seconds if args.timeout_seconds > 0 else None
    )
    terminal = {"complete", "failed", "stalled"}
    while True:
        try:
            from puppetmaster.liveness import reap_stalled_jobs

            reap_stalled_jobs(store, stall_after_seconds=args.stall_after_seconds)
        except Exception:
            pass
        job = store.get_job(args.job_id)
        status = str(job.status)
        if status in terminal:
            timed_out = False
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(poll)

    summary = ""
    if args.summary and status in {"complete", "stalled"}:
        summary_path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
        if summary_path.is_file():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            summary = Stitcher(store).preview(args.job_id)

    payload = {
        "job_id": args.job_id,
        "status": status,
        "terminal": status in terminal,
        "timed_out": timed_out,
        "completed_at": job.completed_at,
    }
    if args.json:
        print(json.dumps({**payload, "summary": summary}, indent=2, default=str))
    elif timed_out:
        print(
            f"timed out after {args.timeout_seconds}s; job {args.job_id} is {status}",
            file=sys.stderr,
        )
    else:
        print(f"job {args.job_id} reached terminal state: {status}")
        if summary:
            print()
            print(summary)
    # Exit non-zero on the bad terminal states (and on timeout) so scripts can
    # branch on it without parsing output.
    if timed_out or status in {"failed", "stalled"}:
        return 1
    return 0


def _run_await_command(args, store) -> int:
    state = await_job_state(
        store,
        args.job_id,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    summary = ""
    if state["terminal"]:
        summary_path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
        if summary_path.is_file():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            summary = Stitcher(store).preview(args.job_id)

    if args.json:
        print(json.dumps({**state, "summary": summary}, indent=2, default=str))
    else:
        if state["timed_out"]:
            print(f"timed out after {args.timeout_seconds}s; job {args.job_id} is {state['status']}")
        else:
            print(summary or f"job {args.job_id} finished: {state['status']}")
    return 0 if state["status"] not in {"failed", "stalled"} else 1


def _run_preflight_command(args) -> int:
    """Check an adapter's auth/billing posture (and Cursor model) before dispatch."""
    import json as _json

    from puppetmaster.preflight import preflight_check

    catalog_fetcher = None
    if args.adapter == "cursor" and args.model:
        from puppetmaster.cursor_discovery import fetch_cursor_catalog

        catalog_fetcher = fetch_cursor_catalog

    result = preflight_check(
        args.adapter,
        args.model,
        allow_api_billing=not args.no_api_billing,
        live=getattr(args, "live", False),
        catalog_fetcher=catalog_fetcher,
    )

    if args.json:
        print(_json.dumps(result.as_dict(), indent=2))
    else:
        status = "READY" if result.ok else "BLOCKED"
        print(f"{status:8} {result.adapter:12} billing={result.billing}")
        print(f"  {result.reason}")
    return 0 if result.ok else 1


def _run_audit_command(args, store) -> int:
    """Analyze past routing behavior and propose conservative score changes.

    Read-only by default: prints a per-model report (picks, mean confidence,
    escalation rate, spend) plus a suggested models.json diff. ``--apply``
    writes the suggested score changes; nothing is mutated otherwise.
    """
    import json as _json
    from dataclasses import asdict, replace

    from puppetmaster.audit import build_audit_report, collect_records
    from puppetmaster.model_registry import (
        default_registry_path,
        load_registry,
        save_registry,
    )

    registry_path = _registry_path_from_args(args) or default_registry_path()
    registry = load_registry(registry_path)
    scores = {s.id: s.capability_score for s in registry}
    specs_by_id = {s.id: s for s in registry}

    def actual_cost_fn(model_id: str, tokens_in: int, tokens_out: int) -> float:
        spec = specs_by_id.get(model_id)
        # Price actuals with the same marginal-cost basis the router used for its
        # estimate, so plan-billed models read $0 on both sides (honest parity).
        return spec.marginal_cost_usd(tokens_in, tokens_out) if spec else 0.0

    records, jobs_considered = collect_records(store, window_days=args.window)
    report = build_audit_report(
        records,
        scores,
        window_days=args.window,
        jobs_considered=jobs_considered,
        actual_cost_fn=actual_cost_fn,
    )

    if args.json:
        print(
            _json.dumps(
                {
                    "jobs_considered": report.jobs_considered,
                    "tasks_considered": report.tasks_considered,
                    "window_days": report.window_days,
                    "total_est_spend_usd": report.total_est_spend_usd,
                    "reconciliation": {
                        "tasks_with_actuals": report.tasks_with_actuals,
                        "total_est_tokens": report.total_est_tokens,
                        "total_actual_tokens": report.total_actual_tokens,
                        "token_drift_ratio": report.token_drift_ratio,
                        "total_actual_spend_usd": report.total_actual_spend_usd,
                        "cost_drift_ratio": report.cost_drift_ratio,
                    },
                    "models": [asdict(m) for m in report.models],
                    "suggestions": report.suggestions,
                },
                indent=2,
            )
        )
    else:
        window = f"{args.window:g}d" if args.window else "all time"
        print(
            f"Routing audit — {report.jobs_considered} jobs, "
            f"{report.tasks_considered} tasks ({window}); "
            f"est spend ${report.total_est_spend_usd:.4f}"
        )
        if report.tasks_considered == 0:
            print(
                "  No router-placed tasks found. Run some auto_route work first "
                "(routing decisions are recorded as durable artifacts)."
            )
        else:
            print(
                f"  {'model':<26}{'picks':>6}{'conf':>7}{'esc%':>7}{'drift':>8}  flags"
            )
            for m in report.models:
                if m.selections == 0 and m.runs_with_confidence == 0:
                    continue
                conf = f"{m.mean_confidence:.2f}" if m.mean_confidence is not None else "  -"
                esc = f"{m.escalated_away_rate:.0%}"
                drift = f"{m.token_drift_ratio:.2f}x" if m.token_drift_ratio is not None else "  -"
                print(
                    f"  {m.model_id:<26}{m.selections:>6}{conf:>7}{esc:>7}{drift:>8}  "
                    f"{', '.join(m.flags)}"
                )
            if report.tasks_with_actuals:
                tok = report.token_drift_ratio
                tok_str = f"{tok:.2f}x actual/est" if tok is not None else "n/a"
                line = (
                    f"  reconciled {report.tasks_with_actuals}/{report.tasks_considered} "
                    f"tasks: tokens {report.total_actual_tokens:,} actual vs "
                    f"{report.total_est_tokens:,} est ({tok_str})"
                )
                cost = report.cost_drift_ratio
                if cost is not None:
                    line += (
                        f"; metered cost ${report.total_actual_spend_usd:.4f} actual vs "
                        f"${report.total_est_spend_reconciled_usd:.4f} est ({cost:.2f}x)"
                    )
                else:
                    line += "; cost $0 (plan-billed — tokens are the real signal)"
                print(line)
            else:
                print(
                    "  No token usage recorded yet — estimate-vs-actual drift will "
                    "populate once routed runs report usage."
                )
        if report.suggestions:
            print("\nSuggested score changes (your assertion stays the source of truth):")
            for s in report.suggestions:
                print(f"  {s['model_id']}: {s['from_score']} -> {s['to_score']}")
                print(f"      {s['rationale']}")
        elif report.tasks_considered:
            print("\nNo score changes suggested — routing looks well-calibrated.")

    if args.apply:
        if not report.suggestions:
            print("\nNothing to apply.")
            return 0
        by_id = {s.id: s for s in registry}
        changed = 0
        for sug in report.suggestions:
            spec = by_id.get(sug["model_id"])
            if spec is None:
                continue
            by_id[sug["model_id"]] = replace(spec, capability_score=sug["to_score"])
            changed += 1
        if changed:
            save_registry(list(by_id.values()), registry_path)
            print(f"\nApplied {changed} score change(s) to {registry_path}.")
    return 0


def _run_savings_command(args, state_dir) -> int:
    """Print the cumulative savings receipt. Read-only; local; emits nothing."""
    import json as _json

    from puppetmaster.savings import COUNTERFACTUAL_MODEL_ENV, build_report
    from puppetmaster.state import list_project_state_dirs
    from puppetmaster.store_factory import create_store

    dirs = [state_dir]
    if getattr(args, "all_projects", False):
        seen = {state_dir.resolve()}
        for d in list_project_state_dirs():
            if d.resolve() not in seen and d.exists():
                seen.add(d.resolve())
                dirs.append(d)
    stores = []
    for d in dirs:
        try:
            stores.append(create_store(args.backend, d))
        except Exception:
            continue

    report = build_report(stores, window_days=args.window)
    routing = report["routing"]
    cg = report["codegraph"]
    heal = report["self_heal"]
    reads = report["reads"]
    metrics = report["metrics"]
    cf = report["counterfactual"]

    if args.json:
        from dataclasses import asdict

        print(
            _json.dumps(
                {
                    "window_days": report["window_days"],
                    "jobs_considered": report["jobs_considered"],
                    "routing": asdict(routing),
                    "routing_pct_cheaper": round(routing.pct_cheaper, 1),
                    "self_heal": asdict(heal),
                    "codegraph": cg,
                    "reads": reads,
                    "metrics": metrics,
                    "counterfactual": asdict(cf) if cf is not None else None,
                },
                indent=2,
            )
        )
        return 0

    window = f"last {args.window:g}d" if args.window else "all time"
    scope = "all projects" if getattr(args, "all_projects", False) else "this project"
    print(f"Puppetmaster savings — {window}, {scope} ({report['jobs_considered']} jobs)")
    print()
    print("MEASURED")
    print(
        f"  Routing (cost-optimizing tasks): saved ${routing.saved_usd:.4f} "
        f"of ${routing.baseline_usd:.4f} baseline ({routing.pct_cheaper:.0f}% cheaper) "
        f"across {routing.cost_optimizing_tasks} tasks"
    )
    print(
        f"    tasks routed to a $0-marginal (plan) model: {routing.plan_routed_tasks}"
    )
    if routing.deliberate_tasks:
        print(
            f"  Deliberate quality spend (by request): ${routing.deliberate_spend_usd:.4f} "
            f"over {routing.deliberate_tasks} tasks (not counted as savings)"
        )
    print(
        f"  CodeGraph: {cg['queries']} exploration queries served, "
        f"~{cg['context_tokens_fed']:,} focused-context tokens fed to agents"
    )
    print(
        f"  Reliability: {heal.fallbacks} task(s) auto-recovered off a dead/unfunded "
        f"provider, {heal.escalations} re-run for confidence (counts, not dollars)"
    )
    print(
        f"  $0 follow-up reads: {reads['reads']} result read(s) served from durable "
        f"state at zero model cost"
    )
    print()
    print(
        f"ESTIMATE (baseline: {cg['exploration_baseline_tokens']:,} tokens/query "
        f"a graph-less crawl would read, ${cg['input_price_per_mtok']:g}/Mtok input)"
    )
    print(
        f"  Avoided exploration: ~{cg['net_tokens_saved_est']:,} tokens "
        f"-> ~${cg['dollars_saved_est']:.4f} saved"
    )
    print()

    if cf is not None:
        print(
            f"COUNTERFACTUAL (vs running every routed task on {cf.reference_model_id} "
            f"at metered API rates)"
        )
        if cf.reference_priced:
            print(
                f"  Avoided spend: ${cf.avoided_usd:,.2f} "
                f"(naive ${cf.naive_cost_usd:,.2f} - actual ${cf.actual_cost_usd:,.2f}) "
                f"across {cf.tasks} routed task(s)"
            )
            print(
                "  This is a counterfactual, not cash off your bill — only as honest "
                f"as the assumption that you'd otherwise have run {cf.reference_model_id}."
            )
        else:
            print(
                f"  Reference {cf.reference_model_id} has no per-token price, so there is "
                "no metered counterfactual to compute ($0). Point "
                f"${COUNTERFACTUAL_MODEL_ENV} at a priced model to get this number."
            )
        print()

    def _pct(value):
        return "n/a" if value is None else f"{value * 100:.0f}%"

    sample = metrics["sample"]
    print("RATES (no $, for org dashboards — trend these over a --window)")
    print(
        f"  Capability-match rate: {_pct(metrics['capability_match_rate'])} "
        f"(right-sized below the strongest model; n={sample['cost_optimizing_judgeable']})"
    )
    print(
        f"  Escalation rate: {_pct(metrics['escalation_rate'])} | "
        f"Fallback rate: {_pct(metrics['fallback_rate'])} (n={sample['routed_tasks']} routed tasks)"
    )
    reuse = metrics["reuse_reads_per_job"]
    ctx = metrics["context_tokens_per_job"]
    print(
        f"  Reuse: {('n/a' if reuse is None else f'{reuse:g}')} result reads/job | "
        f"Context: {('n/a' if ctx is None else f'~{ctx:,.0f}')} focused tokens/job "
        f"(n={sample['jobs']} jobs)"
    )
    print()
    print("Notes")
    print("  - Routing $ is vs the strongest model you could have used, snapshotted at")
    print("    decision time (no recompute drift). Quality/escalating picks are spend you")
    print("    asked for, shown separately, never as a loss.")
    if routing.tasks_without_baseline:
        print(
            f"  - {routing.tasks_without_baseline} cost-optimizing tasks predate baseline "
            "tracking and are excluded from the $ figure."
        )
    print("  - 'Avoided exploration' is an estimate; tune with "
          "PUPPETMASTER_EXPLORATION_BASELINE_TOKENS / _PRICE_PER_MTOK.")
    return 0


def _run_route_command(args) -> int:
    """Run the router against a free-form instruction and print the decision.

    Use this to sanity-check capability scores and policies before
    putting them in front of a real swarm. Pairs with the
    ``puppetmaster_route_task`` MCP tool.
    """
    from puppetmaster.model_registry import default_registry_path, load_registry
    from puppetmaster.router import (
        NoEligibleModelError,
        TaskSignals,
        route_task,
    )

    path = _registry_path_from_args(args) or default_registry_path()
    try:
        specs = load_registry(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not specs:
        print(
            f"error: no models registered at {path}. "
            "Run `puppetmaster models init` first.",
            file=sys.stderr,
        )
        return 1

    from puppetmaster.platform_lock import active_allowlist

    signals = TaskSignals(
        instruction=args.instruction,
        role=args.role,
        explicit_min_capability=args.min_capability,
        explicit_max_cost_usd=args.max_cost_usd,
        required_tags=list(args.required_tag),
        allowed_adapters=active_allowlist(),
    )
    try:
        decision = route_task(signals, specs, policy=args.policy)
    except NoEligibleModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(decision.to_artifact_payload(), indent=2))
        return 0

    print(
        f"picked: {decision.model.id}  (adapter={decision.model.adapter}, "
        f"model_name={decision.model.adapter_model_name})"
    )
    print(f"policy: {decision.policy}")
    print(
        f"capability needed: {decision.capability_needed}  "
        f"chosen capability: {decision.model.capability_score}"
    )
    print(
        f"estimated tokens: in={decision.estimated_tokens_in}  "
        f"out={decision.estimated_tokens_out}  "
        f"estimated cost: ${decision.estimated_cost_usd:.6f}"
    )
    print(f"why: {decision.reason}")
    if decision.rejected:
        print("rejected:")
        for spec, why in decision.rejected:
            print(f"  - {spec.id}: {why}")
    return 0


def _run_should_delegate_command(args) -> int:
    """Dry-run the invocation gate against a prompt."""
    from puppetmaster.invocation_gate import should_delegate

    decision = should_delegate(
        args.prompt, role=args.role, threshold=args.threshold
    )
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return 0
    verdict = "DELEGATE" if decision.should_delegate else "inline"
    print(f"{verdict}  (capability {decision.capability_score}, role={decision.role})")
    print(f"verb:   {decision.suggested_verb}")
    print(f"why:    {decision.reason}")
    if decision.matched_signals:
        print(f"signals: {', '.join(decision.matched_signals)}")
    return 0


def _run_invocation_gate_command(args) -> int:
    """Host-hook entry point. Reads stdin JSON, prints host verdict, exits 0."""
    from puppetmaster.hook_runner import run as run_hook

    return run_hook(["--host", args.host, "--event", args.event])


def _run_install_hooks(args) -> int:
    """Dispatch for ``puppetmaster install-hooks``."""
    targets = None
    raw = getattr(args, "target", None)
    if raw:
        targets = [t.strip() for t in raw.split(",") if t.strip()]
    scope = "global" if getattr(args, "global_scope", False) else "project"
    result = install_hooks(
        cwd=Path.cwd(),
        targets=targets,
        dry_run=getattr(args, "dry_run", False),
        force=getattr(args, "force", False),
        scope=scope,
    )
    print(f"[install-hooks] overall: {result.overall_status} (scope={scope})")
    for outcome in result.outcomes:
        print(f"[install-hooks] {outcome.target:<8} {outcome.status:<14} {outcome.reason}")
        if outcome.path:
            print(f"[install-hooks] {' ' * 8} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[install-hooks] note: {msg}")
    return 1 if result.overall_status == "error" else 0


def _run_proxy_command(args) -> int:
    """Run the local OpenAI-compatible enforcement proxy."""
    from puppetmaster.provider_proxy import serve_proxy

    try:
        serve_proxy(
            host=args.host,
            port=args.port,
            mode=args.mode,
            upstream_base_url=args.upstream_base_url,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_cost_command(args, store) -> int:
    """Summarize estimated USD spend for a job from its ROUTING artifacts.

    The router writes one ``ArtifactType.ROUTING`` artifact per
    auto-routed task at task creation, with the chosen model + the
    estimated USD cost in ``payload.estimated_cost_usd``. This command
    sums them up and prints a per-model breakdown plus the grand total.

    These are **estimates** based on user-asserted prices in
    ``~/.puppetmaster/models.json`` — Puppetmaster doesn't call a
    billing API. They're useful for budgeting, not invoicing.
    """
    from puppetmaster.models import ArtifactType

    job_id = args.job_id
    artifacts = store.list_artifacts(job_id)
    # Only count the router's initial decision per task. Fallback/escalation
    # reroutes (created_by 'router-fallback' / 'router-escalation') emit their
    # own ROUTING artifacts; summing all of them double-counts a rerouted task.
    # Dedup by task_id mirrors savings.collect_routing_records.
    routing = []
    seen_router_tasks: set = set()
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        task_id = artifact.task_id
        if task_id:
            if task_id in seen_router_tasks:
                continue
            seen_router_tasks.add(task_id)
        routing.append(artifact)

    if not routing:
        msg = (
            f"No ROUTING artifacts on job {job_id}. Either the job didn't "
            "auto-route any tasks, or it predates the router (v0.6.0)."
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "job_id": job_id,
                        "total_estimated_cost_usd": 0.0,
                        "tasks": [],
                        "note": msg,
                    },
                    indent=2,
                )
            )
        else:
            print(msg)
        return 0

    by_model: dict[str, dict] = {}
    rows = []
    total = 0.0
    for artifact in routing:
        payload = artifact.payload or {}
        model_id = payload.get("model_id", "<unknown>")
        cost = float(payload.get("estimated_cost_usd") or 0.0)
        total += cost
        rows.append(
            {
                "task_id": artifact.task_id,
                "role": payload.get("role"),
                "model_id": model_id,
                "adapter": payload.get("adapter"),
                "policy": payload.get("policy"),
                "capability_needed": payload.get("capability_needed"),
                "estimated_cost_usd": cost,
            }
        )
        bucket = by_model.setdefault(model_id, {"calls": 0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["cost"] += cost

    if args.json:
        from puppetmaster.usage import aggregate_token_usage

        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "cost_basis": "preflight_routing_estimate",
                    "total_estimated_cost_usd": round(total, 6),
                    "by_model": {
                        mid: {
                            "calls": v["calls"],
                            "estimated_cost_usd": round(v["cost"], 6),
                        }
                        for mid, v in by_model.items()
                    },
                    "token_usage": aggregate_token_usage(artifacts),
                    "tasks": rows,
                },
                indent=2,
            )
        )
        return 0

    print(f"job {job_id}: estimated total cost = ${total:.6f}")
    print()
    print(f"  {'MODEL':<28}  {'CALLS':>5}  {'COST':>12}")
    for mid, v in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"]):
        print(f"  {mid:<28}  {v['calls']:>5}  ${v['cost']:>10.6f}")
    print()
    print(f"  {'TASK':<14}  {'ROLE':<14}  {'MODEL':<28}  {'COST':>12}")
    for row in rows:
        task_id = (row["task_id"] or "")[:14]
        role = (row["role"] or "")[:14]
        model_id = (row["model_id"] or "")[:28]
        print(
            f"  {task_id:<14}  {role:<14}  {model_id:<28}  "
            f"${row['estimated_cost_usd']:>10.6f}"
        )
    print()
    print(
        "  note: the figures above are PRE-FLIGHT ROUTING ESTIMATES (relative "
        "model cost), not measured consumption — do not read them as token volume."
    )
    _print_token_usage(artifacts)
    return 0


def _print_token_usage(artifacts) -> None:
    """Print measured-vs-estimated token consumption for a job's runs.

    Plan-billed runtimes (Cursor SDK) have $0 marginal cost, so a dollars-only
    ledger says nothing. Token counts are the honest measure of consumption;
    surface them, clearly split into measured vs char/4-estimated.
    """
    from puppetmaster.usage import aggregate_token_usage

    usage = aggregate_token_usage(artifacts)
    if usage["measured_runs"] == 0 and usage["estimated_runs"] == 0:
        return
    print()
    print("  token consumption (measured where the SDK reports usage):")
    if usage["measured_runs"]:
        print(
            f"    measured:  {usage['measured_tokens_in']:,} in / "
            f"{usage['measured_tokens_out']:,} out over {usage['measured_runs']} run(s)"
        )
    if usage["estimated_runs"]:
        print(
            f"    estimated: ~{usage['estimated_tokens_in']:,} in / "
            f"~{usage['estimated_tokens_out']:,} out over {usage['estimated_runs']} run(s) "
            "(char/4 approximation — SDK reported no usage)"
        )


def _run_mcp_list(args) -> int:
    snapshot = registry_summarize(registry_list_entries())
    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0
    if snapshot["count"] == 0:
        print("No Puppetmaster MCP servers tracked.")
        return 0
    print(
        f"{snapshot['count']} tracked  "
        f"({snapshot['alive']} alive, {snapshot['stale']} stale, "
        f"{snapshot['dead']} dead)"
    )
    print(
        f"  {'PID':>7}  {'STATE':<6}  {'AGE':>8}  {'HBEAT':>8}  "
        f"{'PPID':>7}  {'PARENT':<12}  WORKSPACE"
    )
    for row in snapshot["servers"]:
        if not row["alive"]:
            state = "dead"
        elif row["stale"]:
            state = "stale"
        else:
            state = "ok"
        workspace = row.get("workspace") or "-"
        parent_pid = row.get("parent_pid") or "-"
        parent_process = row.get("parent_process") or "-"
        print(
            f"  {row['pid']:>7}  {state:<6}  "
            f"{row['age_seconds']:>8.0f}s  "
            f"{row['heartbeat_age_seconds']:>8.0f}s  "
            f"{parent_pid:>7}  "
            f"{parent_process:<12}  "
            f"{workspace}"
        )
    return 0


def _run_mcp_cleanup(args) -> int:
    before = registry_summarize(registry_list_entries())
    pruned = registry_prune_dead()
    killed: list = []
    if args.kill_stale:
        killed_entries = registry_kill_stale(
            stale_after_seconds=float(args.stale_after_seconds),
        )
        killed = [entry.to_payload() for entry in killed_entries]
    after = registry_summarize(registry_list_entries())
    payload = {
        "before": before,
        "after": after,
        "pruned": [entry.to_payload() for entry in pruned],
        "killed": killed,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(
        f"before: {before['count']} tracked "
        f"({before['alive']} alive, {before['stale']} stale, {before['dead']} dead)"
    )
    print(f"pruned dead: {len(pruned)}")
    for entry in pruned:
        print(f"  - PID {entry.pid} ({entry.workspace or '-'})")
    if args.kill_stale:
        print(f"killed stale: {len(killed)}")
        for row in killed:
            print(f"  - PID {row['pid']} ({row.get('workspace') or '-'})")
    elif before["stale"]:
        print(
            f"note: {before['stale']} stale-but-alive server(s) detected; "
            "pass --kill-stale to terminate them."
        )
    print(
        f"after: {after['count']} tracked "
        f"({after['alive']} alive, {after['stale']} stale, {after['dead']} dead)"
    )
    return 0


def print_watch_snapshot(snapshot: dict) -> None:
    counts = ", ".join(
        f"{status}={count}" for status, count in sorted(snapshot["task_counts"].items())
    )
    print(
        f"{snapshot['job']['id']} {snapshot['job']['status']} "
        f"tasks[{counts}] artifacts={snapshot['artifact_count']} "
        f"stale={len(snapshot['stale_task_ids'])}"
    )


def require_latest_job_id(store) -> str:
    job = store.latest_job()
    if job is None:
        raise FileNotFoundError("no jobs found")
    return job.id


def artifact_job_id(store, artifact_id: str) -> str:
    job_id = store.get_artifact_job_id(artifact_id)
    if job_id is None:
        raise FileNotFoundError(f"artifact not found: {artifact_id}")
    return job_id


def patch_artifacts_for_target(store, target: str):
    try:
        store.get_job(target)
        artifacts = store.list_artifacts(target)
        return target, [artifact for artifact in artifacts if str(artifact.type) == "patch"]
    except FileNotFoundError:
        pass
    job_id = artifact_job_id(store, target)
    artifacts = [artifact for artifact in store.list_artifacts(job_id) if artifact.id == target]
    return job_id, artifacts


def approve_target(store, target: str, worktree: Optional[Path] = None) -> int:
    job_id, artifacts = patch_artifacts_for_target(store, target)
    if not artifacts:
        raise FileNotFoundError(f"no patch artifacts found for {target}")
    applied = 0
    for artifact in artifacts:
        files = artifact.payload.get("files", [])
        locks = [f"patch:{path}" for path in files if isinstance(path, str)]
        for lock in locks:
            if not store.acquire_lock(lock, f"approve:{artifact.id}"):
                raise RuntimeError(f"path lock unavailable: {lock}")
        try:
            diff = artifact.payload.get("unified_diff") or artifact.payload.get("diff")
            if diff:
                apply_patch_diff(diff, cwd=worktree or Path.cwd())
                applied += 1
            store.emit(
                job_id,
                "artifact.approved",
                {
                    "artifact_id": artifact.id,
                    "applied": bool(diff),
                    "worktree": str(worktree) if worktree else None,
                },
            )
        finally:
            for lock in locks:
                store.release_lock(lock)
    return len(artifacts)


def reject_target(store, target: str, reason: str) -> int:
    job_id, artifacts = patch_artifacts_for_target(store, target)
    if not artifacts:
        raise FileNotFoundError(f"no patch artifacts found for {target}")
    for artifact in artifacts:
        store.emit(
            job_id,
            "artifact.rejected",
            {"artifact_id": artifact.id, "reason": reason},
        )
    return len(artifacts)


def apply_patch_diff(diff: str, cwd: Path) -> None:
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=diff,
        text=True,
        capture_output=True,
        cwd=cwd,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"patch did not apply cleanly: {check.stderr.strip()}")
    applied = subprocess.run(
        ["git", "apply", "-"],
        input=diff,
        text=True,
        capture_output=True,
        cwd=cwd,
        check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"patch apply failed: {applied.stderr.strip()}")


def cursor_prompt(
    prompt: str,
    *,
    review: bool = False,
    plan: bool = False,
    dry_run: bool = False,
    implement: bool = False,
) -> str:
    lines = [prompt]
    if implement:
        lines.extend(
            [
                "",
                "Implement mode: you are a full-edit worker inside the user's repository. "
                "Actually make the code changes to complete the task end to end — create, "
                "edit, and delete files as needed. Do not just return a plan or findings; "
                "leave the working tree containing your final intended changes.",
            ]
        )
    if review:
        lines.extend(
            [
                "",
                "Review mode: inspect the repository and return findings, risks, evidence, and verification suggestions.",
            ]
        )
    if plan:
        lines.extend(
            [
                "",
                "Plan mode: produce implementation decisions, task graph suggestions, risks, and test strategy. Do not edit files.",
            ]
        )
    if dry_run:
        lines.extend(
            [
                "",
                "Dry-run constraint: do not modify files. Return findings, patch plan, risks, and verification commands as structured evidence.",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
