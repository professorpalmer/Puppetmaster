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
from puppetmaster import platform_lock as _platform_lock

_PLATFORM_ADAPTER_HELP = ", ".join(_platform_lock.KNOWN_ADAPTERS)
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
    install_hermes_plugin,
    install_hermes_skill,
    list_skill_candidates,
    promote_skill_candidate,
    resolve_claude_command,
    set_hermes_mcp_env,
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
from puppetmaster.hook_installers import (
    VALID_HOOK_TARGETS,
    install_hermes_hooks,
    install_hooks,
    uninstall_hermes_hooks,
    uninstall_hooks,
)
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

    def _add_label_argument(job_parser: argparse.ArgumentParser) -> None:
        job_parser.add_argument(
            "--label",
            default=None,
            help="Optional human-readable job label for the dashboard.",
        )

    subcommands.add_parser("init", help="Create the local Puppetmaster state store.")
    subcommands.add_parser("state", help="Print the resolved Puppetmaster state directory.")
    doctor_parser = subcommands.add_parser("doctor", help="Check local runtime dependencies.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    self_update_parser = subcommands.add_parser(
        "self-update",
        help="Upgrade puppetmaster-ai with pip (requires an MCP server restart afterward).",
    )
    self_update_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command that would run without executing it.",
    )
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
        help="Write agent rule files (Cursor / AGENTS.md / Codex / Claude / Hermes) that nudge hosts to reach for Puppetmaster on the right tasks.",
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
        help="Also write user-level rules (~/.codex/instructions.md, ~/.claude/CLAUDE.md, ~/.hermes/SOUL.md) when those tools are detected.",
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
        help="One-shot first-run: doctor + models init + install-cursor-mcp + install-codex-mcp + install-claude-mcp + install-hermes-mcp (+ Hermes hooks) + install-rules + install-hooks. Skips steps where the tool isn't present.",
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
    setup_parser.add_argument(
        "--skip-hermes-advanced",
        action="store_true",
        help=(
            "Skip the optional in-depth Hermes setup branch (learn flywheel, skill "
            "injection env knobs) during `setup` step 7."
        ),
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
    _add_label_argument(run)

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
    _add_label_argument(cursor)

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
    _add_label_argument(claude)

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
    _add_label_argument(openai)

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
    _add_label_argument(codex)

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
    _add_label_argument(hermes)

    agentic = subcommands.add_parser(
        "agentic",
        help=(
            "Run a standalone provider-agnostic worker (keys-only, no external CLI). "
            "Calls the provider HTTP API directly for analyze or implement modes."
        ),
    )
    agentic.add_argument("prompt", help="Prompt for the agentic worker.")
    agentic.add_argument("--cwd", default=str(Path.cwd()), help="Workspace for the worker.")
    agentic.add_argument(
        "--mode",
        choices=["implement", "analyze"],
        default="implement",
        help="implement = full-edit with git-diff PATCH attribution; analyze = read-only structured findings.",
    )
    agentic.add_argument(
        "--provider",
        help="Provider slug (openai, anthropic, gemini, openrouter, ...). Routes credentials/wire protocol.",
    )
    agentic.add_argument(
        "--model",
        help="Model name passed to the provider API.",
    )
    agentic.add_argument(
        "--max-turns",
        type=int,
        help="Cap on tool-use iterations (default 12).",
    )
    agentic.add_argument("--timeout-seconds", type=int, default=900)
    agentic.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature override.",
    )
    agentic.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level for OpenAI-style models.",
    )
    agentic.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="inline",
        help="Agentic runs default to inline orchestration.",
    )
    agentic.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow the worker to run in a dirty working tree.",
    )
    agentic.add_argument(
        "--allow-non-worktree",
        action="store_true",
        help="Allow the worker to run outside a git work tree (no diff attribution).",
    )
    agentic.add_argument(
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo prompts).",
    )
    agentic.add_argument(
        "--disable-memory",
        action="store_true",
        help="Skip promoted shared-memory injection for a fresh perspective.",
    )
    _add_routing_flags(agentic)
    _add_label_argument(agentic)

    edit = subcommands.add_parser(
        "edit",
        help=(
            "Single in-place edit: cheapest sufficient model + CodeGraph, edits "
            "the working tree directly, captures a reviewable PATCH. The "
            "lightweight verb between an inline edit and a full implement job."
        ),
    )
    edit.add_argument("instruction", help="What to change, in plain language.")
    edit.add_argument("--cwd", default=str(Path.cwd()), help="Workspace to edit in.")
    edit.add_argument(
        "--adapter",
        help=(
            "Force a full-edit adapter (cursor | claude-code | codex | hermes | agentic). "
            "Default: the highest-priority adapter the platform lock enables."
        ),
    )
    edit.add_argument(
        "--model",
        help="Pin the model (overrides cheap auto-routing).",
    )
    edit.add_argument(
        "--provider",
        help="Inference provider (Hermes or agentic adapter only; routes credentials).",
    )
    edit.add_argument("--timeout-seconds", type=int, default=300)
    edit.add_argument(
        "--routing-policy",
        default="cheap",
        choices=["cheap", "balanced", "quality", "escalating"],
        help="Router policy when not pinning --model (default: cheap).",
    )
    edit.add_argument(
        "--no-auto-route",
        dest="auto_route_edit",
        action="store_false",
        help="Disable routing; use the adapter's default model.",
    )
    edit.add_argument(
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo edits).",
    )
    edit.add_argument("--executable", help="Override the adapter executable / command.")
    _add_label_argument(edit)

    browser = subcommands.add_parser(
        "browser",
        help=(
            "Browser-QA swarm: N parallel Hermes workers, each driving a real "
            "browser against a live site to capture real network payloads. Bakes "
            "in the React-controlled-input, network-truth, and strong-model "
            "guardrails. ACTING AGENT — has external side effects."
        ),
    )
    browser.add_argument(
        "tasks",
        nargs="+",
        help="One or more QA missions; each runs as its own parallel browser worker.",
    )
    browser.add_argument(
        "--cwd", default=str(Path.cwd()), help="Workspace for repo context."
    )
    browser.add_argument(
        "--model",
        help="Pin the Hermes model (overrides the strong-model routing floor).",
    )
    browser.add_argument(
        "--provider",
        help="Hermes provider (e.g. anthropic). Routes credentials/wire protocol.",
    )
    browser.add_argument(
        "--toolsets",
        help="Override the Hermes toolsets (default: file,web,vision,browser).",
    )
    browser.add_argument(
        "--min-capability",
        type=int,
        help="Override the strong-model capability floor (default 80, 0..100).",
    )
    browser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
        help="Per-worker timeout (default 1200; live browser flows are slow).",
    )
    browser.add_argument(
        "--routing-policy",
        default="balanced",
        choices=["cheap", "balanced", "quality", "escalating"],
        help="Router policy above the capability floor (default: balanced).",
    )
    browser.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline", "daemon"],
        default="subprocess",
        help="subprocess (default) runs the workers in parallel; inline serializes them.",
    )
    browser.add_argument("--executable", help="Override the hermes executable / command.")
    _add_label_argument(browser)

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
        "--runtime-node",
        dest="cursor_node",
        help=(
            "Path to the Node binary to rebuild CodeGraph against. Auto-detected "
            "if omitted (PUPPETMASTER_CODEGRAPH_NODE, then Cursor's bundled Node, "
            "then `node` on PATH). Works for any harness, not just Cursor."
        ),
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
        choices=["cursor", "openai", "anthropic", "claude", "codex", "hermes", "agentic", "all"],
        default=None,
        help=(
            "Which platform catalog to enumerate. Default: derived from the "
            "platform lock — a single locked platform is discovered directly; "
            "otherwise every reachable source ('all'). cursor (plan, needs "
            "CURSOR_API_KEY + node), openai (GET /v1/models, needs OPENAI_API_KEY), "
            "anthropic (needs ANTHROPIC_API_KEY for discovery), claude / codex "
            "(curated catalogs for the CLI agent loops that can't self-enumerate; "
            "billed as your detected subscription/API posture), hermes (curated "
            "multi-provider catalog, API-billed via your own keys), agentic "
            "(curated keys-only catalog filtered by visible provider keys), or all."
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

    keys_cmd = subcommands.add_parser(
        "keys",
        help=(
            "Set agentic provider API keys into the Cursor MCP config so "
            "direct-API workers can reach them (no external CLI required). "
            "Run with no subcommand for an interactive wizard."
        ),
    )
    keys_cmd.add_argument(
        "--target",
        help="Path to the mcp.json to write (default: ~/.cursor/mcp.json).",
    )
    keys_cmd.add_argument(
        "--workspace",
        action="store_true",
        help="Write to <cwd>/.cursor/mcp.json instead of the global config.",
    )
    keys_sub = keys_cmd.add_subparsers(dest="keys_command", required=False)
    keys_status = keys_sub.add_parser(
        "status",
        help="Show which provider keys are visible/stored (values hidden).",
    )
    keys_status.add_argument("--target", help="Path to the mcp.json to inspect.")
    keys_status.add_argument(
        "--workspace", action="store_true", help="Inspect <cwd>/.cursor/mcp.json."
    )
    keys_set = keys_sub.add_parser(
        "set",
        help="Set one provider's key (prompts hidden, or read from --stdin).",
    )
    keys_set.add_argument("provider", help="Provider slug, e.g. openai, anthropic, gemini.")
    keys_set.add_argument("--target", help="Path to the mcp.json to write.")
    keys_set.add_argument(
        "--workspace", action="store_true", help="Write to <cwd>/.cursor/mcp.json."
    )
    keys_set.add_argument(
        "--stdin",
        action="store_true",
        help="Read the key value from stdin instead of a hidden prompt.",
    )

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
    platform_only.add_argument("adapters", nargs="+", help=_PLATFORM_ADAPTER_HELP)
    platform_only.add_argument("--registry-path", help="Override the registry path.")
    platform_enable = platform_sub.add_parser(
        "enable", help="Turn these platforms back on."
    )
    platform_enable.add_argument("adapters", nargs="+", help=_PLATFORM_ADAPTER_HELP)
    platform_enable.add_argument("--registry-path", help="Override the registry path.")
    platform_disable = platform_sub.add_parser(
        "disable", help="Turn these platforms off (never routed or discovered)."
    )
    platform_disable.add_argument("adapters", nargs="+", help=_PLATFORM_ADAPTER_HELP)
    platform_disable.add_argument("--registry-path", help="Override the registry path.")
    platform_reset = platform_sub.add_parser(
        "reset", help="Clear the lock; enable every platform again."
    )
    platform_reset.add_argument("--registry-path", help="Override the registry path.")

    skills_cmd = subcommands.add_parser(
        "skills",
        help=(
            "Review and promote Hermes skill CANDIDATES distilled by the "
            "puppetmaster-learn flywheel. Promotion is always explicit — nothing "
            "is ever auto-promoted into a live skill."
        ),
    )
    skills_sub = skills_cmd.add_subparsers(dest="skills_command", required=True)
    skills_list = skills_sub.add_parser(
        "list-candidates", help="List review-ready skill candidates."
    )
    skills_list.add_argument("--json", action="store_true", help="Emit JSON.")
    skills_promote = skills_sub.add_parser(
        "promote-candidate",
        help="Promote a reviewed candidate into a live Hermes skill (by slug or dir name).",
    )
    skills_promote.add_argument(
        "slug", help="Candidate slug (e.g. fix-the-thing) or exact directory name."
    )
    skills_promote.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing live skill that differs from the candidate.",
    )
    skills_promote.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be promoted without writing anything.",
    )

    preflight_cmd = subcommands.add_parser(
        "preflight",
        help=(
            "Check whether an adapter can actually run before dispatch: auth "
            "present, plan-vs-api billing, and (Cursor) model in the live catalog."
        ),
    )
    preflight_cmd.add_argument(
        "adapter",
        help=f"Adapter to check ({_PLATFORM_ADAPTER_HELP}).",
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
