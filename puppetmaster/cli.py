from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.installers import (
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    InstallResult,
    install_codex_mcp,
    install_cursor_mcp,
)
from puppetmaster.rules import (
    VALID_TARGETS,
    RulesInstallResult,
    install_rules,
)
from puppetmaster.mcp_registry import (
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    summarize as registry_summarize,
)
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

    subcommands.add_parser("init", help="Create the local Puppetmaster state store.")
    subcommands.add_parser("state", help="Print the resolved Puppetmaster state directory.")
    subcommands.add_parser("doctor", help="Check local runtime dependencies.")
    subcommands.add_parser("adapters", help="List available worker adapters.")

    install_codex = subcommands.add_parser(
        "install-codex-mcp",
        help="Register Puppetmaster as an MCP server in the OpenAI Codex CLI.",
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
        help="One-shot first-run: doctor + models init + install-cursor-mcp + install-codex-mcp + install-rules. Skips steps where the tool isn't present.",
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
        "--global-rules",
        action="store_true",
        help="Pass --global to install-rules.",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force through to MCP installers and rule installer.",
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
        "--dry-run",
        action="store_true",
        help="Instruct the agent to inspect and propose artifacts without editing files.",
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
        "--disable-codegraph",
        action="store_true",
        help="Skip CodeGraph context injection (e.g. for non-repo prompts).",
    )

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

    mcp_cleanup = mcp_sub.add_parser(
        "cleanup",
        help=(
            "Prune dead tracking files; with --kill-stale, terminate stale-but-alive "
            "MCP servers whose Cursor parent is gone."
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
        choices=["cursor", "openai", "anthropic", "claude", "codex", "all"],
        default="cursor",
        help=(
            "Which platform catalog to enumerate. cursor (plan, default), "
            "openai (GET /v1/models, needs OPENAI_API_KEY), anthropic "
            "(needs ANTHROPIC_API_KEY for discovery), claude / codex (curated "
            "catalogs for the CLI agent loops that can't self-enumerate; billed "
            "as your detected subscription/API posture), or all."
        ),
    )
    models_discover.add_argument(
        "--write",
        action="store_true",
        help="Persist the merged registry (default: dry-run, just print the diff).",
    )
    models_discover.add_argument("--json", action="store_true", help="Emit JSON.")

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
        for check in run_doctor(Path.cwd(), state_dir):
            print(f"{check.status:8} {check.name:16} {check.detail}")
        return 0

    if args.command == "install-codex-mcp":
        return _run_install_codex(args)

    if args.command == "install-cursor-mcp":
        return _run_install_cursor(args)

    if args.command == "install-rules":
        return _run_install_rules(args)

    if args.command == "setup":
        return _run_setup(args)

    if args.command == "repair-codegraph":
        return _run_repair_codegraph(args)

    if args.command == "mcp":
        return _run_mcp_subcommand(args)

    if args.command == "models":
        return _run_models_subcommand(args)

    if args.command == "platform":
        return _run_platform_subcommand(args)

    if args.command == "route":
        return _run_route_command(args)

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
        if args.config:
            config = load_config(args.config)
            result = Orchestrator(store).run(
                args.goal,
                specs=config.workers,
                lease_seconds=config.lease_seconds,
                worker_mode=args.worker_mode,
                on_job_created=on_job_created,
            )
        else:
            result = Orchestrator(store).run(
                args.goal,
                roles=args.workers,
                worker_mode=args.worker_mode,
                on_job_created=on_job_created,
            )
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

    if args.command == "cursor":
        prompt = cursor_prompt(args.prompt, review=args.review, plan=args.plan, dry_run=args.dry_run)
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="cursor",
                    instruction=args.prompt,
                    adapter="cursor",
                    payload={
                        "prompt": prompt,
                        "cwd": args.cwd,
                        "model": args.model,
                        "timeout_seconds": args.timeout_seconds,
                    },
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

    if args.command == "claude":
        result = Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role="claude-code",
                    instruction=args.prompt,
                    adapter="claude-code",
                    payload={
                        "prompt": args.prompt,
                        "cwd": args.cwd,
                        "model": args.model,
                        "permission_mode": args.permission_mode,
                        "allowed_tools": args.allowed_tools,
                        "disallowed_tools": args.disallowed_tools,
                        "executable": args.executable,
                        "timeout_seconds": args.timeout_seconds,
                        "allow_dirty": args.allow_dirty,
                    },
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
        )
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

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
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

    if args.command == "codex":
        payload: dict[str, Any] = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "model": args.model,
            "sandbox": args.sandbox,
            "approval_policy": args.approval_policy,
            "timeout_seconds": args.timeout_seconds,
            "allow_dirty": args.allow_dirty,
            "dangerously_bypass_approvals_and_sandbox": args.dangerously_bypass_approvals_and_sandbox,
        }
        if args.executable:
            payload["executable"] = args.executable
        if args.disable_codegraph:
            payload["disable_codegraph"] = True
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
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        return 0

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
        print(json.dumps(store.status_snapshot(args.job_id), indent=2))
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
        for event in store.read_events(job_id):
            print(f"{event['at']}\t{event['event']}\t{json.dumps(event['payload'], sort_keys=True)}")
        return 0

    if args.command == "feed":
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
        if args.partial:
            print(Stitcher(store).preview(args.job_id))
        else:
            path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
            print(path.read_text(encoding="utf-8"))
        return 0

    if args.command == "dashboard":
        from puppetmaster.dashboard import serve

        serve(
            state_dir,
            backend=args.backend,
            job_id=args.job_id,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return 0

    if args.command == "await":
        return _run_await_command(args, store)

    if args.command == "artifacts":
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
    artifacts = {artifact.id: artifact.__dict__ for artifact in store.list_artifacts(job_id)}
    items: list[dict] = []
    seen: set = set()
    cursor = since
    for event in store.read_events_since(job_id, since=since):
        event_id = event.get("id")
        if isinstance(event_id, int) and event_id > cursor:
            cursor = event_id
        if event.get("event") != "artifact.saved":
            continue
        artifact_id = event.get("payload", {}).get("artifact_id")
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact_id in seen:
            continue
        seen.add(artifact_id)
        items.append(
            {
                "at": event["at"],
                "event": event["event"],
                "id": event_id,
                "artifact": artifact,
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
    print(f"[install-{host}-mcp] target: {result.target}")
    print(f"[install-{host}-mcp] python: {result.python_executable}")
    if result.handshake is not None:
        if result.handshake.ok:
            print(
                f"[install-{host}-mcp] handshake: OK ({result.handshake.tool_count} tools)"
            )
        else:
            print(f"[install-{host}-mcp] handshake: FAILED — {result.handshake.error}")
    for line in result.messages:
        print(f"[install-{host}-mcp] {line}")
    return 0 if result.status in {"installed", "unchanged", "would_install"} else 1


def _run_install_codex(args) -> int:
    """Dispatch for ``puppetmaster install-codex-mcp``.

    Delegates to :func:`install_codex_mcp` and prints the sandbox
    guidance block on success so the user knows the first MCP call
    inside ``codex`` will surface an approval prompt.
    """
    result = install_codex_mcp(
        codex_executable=getattr(args, "codex", None),
        force=getattr(args, "force", False),
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


def _run_setup(args) -> int:
    """Dispatch for ``puppetmaster setup`` — one-shot first-run wizard.

    Chains the canonical install steps in dependency order:

    1. ``doctor`` — fail loudly if Puppetmaster's runtime is broken.
    2. ``models init`` — write the starter registry if missing.
    3. ``install-cursor-mcp`` — workspace .cursor/mcp.json.
    4. ``install-codex-mcp`` — only if ``codex`` is on PATH.
    5. ``install-rules`` — agent nudges for whichever tools detected.

    Each step is independent: a step's failure prints a clear error
    but does not abort the rest of the chain unless ``doctor`` reports
    that Python or sqlite is missing (in which case nothing else will
    work). The user can re-run after fixing whatever was reported.
    """
    cwd = Path.cwd()
    state_dir = resolve_state_dir(args.state_dir, cwd)
    overall_rc = 0

    if not getattr(args, "skip_doctor", False):
        print("=== step 1/5: doctor ===")
        checks = list(run_doctor(cwd, state_dir))
        for check in checks:
            print(f"  {check.status:8} {check.name:16} {check.detail}")
        criticals = [c for c in checks if c.status == "fail" and c.name in {"python", "sqlite"}]
        if criticals:
            print("\nCritical dependency missing — fix the above before re-running `setup`.")
            return 1
        print()
    else:
        print("=== step 1/5: doctor SKIPPED (--skip-doctor) ===\n")

    if not getattr(args, "skip_models", False):
        print("=== step 2/5: models init ===")
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
        print("=== step 2/5: models init SKIPPED (--skip-models) ===\n")

    print("=== step 3/5: install-cursor-mcp (workspace .cursor/mcp.json) ===")
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

    print("=== step 4/5: install-codex-mcp ===")
    import shutil as _shutil
    if _shutil.which("codex") is None:
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

    if not getattr(args, "skip_rules", False):
        print("=== step 5/5: install-rules (agent nudges) ===")
        rules_result = install_rules(
            cwd=cwd,
            install_global=getattr(args, "global_rules", False),
            dry_run=False,
            force=getattr(args, "force", False),
        )
        for outcome in rules_result.outcomes:
            print(f"  {outcome.target:<14} {outcome.status:<14} {outcome.reason}")
        for msg in rules_result.messages:
            print(f"  note: {msg}")
        if rules_result.overall_status == "error":
            overall_rc = 1
    else:
        print("=== step 5/5: install-rules SKIPPED (--skip-rules) ===")
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


def _run_mcp_subcommand(args) -> int:
    """Dispatch the `python -m puppetmaster mcp ...` family of commands."""
    if args.mcp_command == "list":
        return _run_mcp_list(args)
    if args.mcp_command == "cleanup":
        return _run_mcp_cleanup(args)
    raise SystemExit(f"unknown mcp subcommand: {args.mcp_command}")


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

    sources = (
        ["cursor", "openai", "anthropic"] if args.source == "all" else [args.source]
    )
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

    terminal = {JobStatus.COMPLETE, JobStatus.FAILED}
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
        store.wait_for_events(
            job_id,
            since=cursor,
            timeout_seconds=max(0.05, block),
            poll_interval=poll,
        )


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
    return 0 if state["status"] != "failed" else 1


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
    routing = [a for a in artifacts if a.type == ArtifactType.ROUTING]

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
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "total_estimated_cost_usd": round(total, 6),
                    "by_model": {
                        mid: {
                            "calls": v["calls"],
                            "estimated_cost_usd": round(v["cost"], 6),
                        }
                        for mid, v in by_model.items()
                    },
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
    return 0


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
        f"  {'PID':>7}  {'STATE':<6}  {'AGE':>8}  {'HBEAT':>8}  WORKSPACE"
    )
    for row in snapshot["servers"]:
        if not row["alive"]:
            state = "dead"
        elif row["stale"]:
            state = "stale"
        else:
            state = "ok"
        workspace = row.get("workspace") or "-"
        print(
            f"  {row['pid']:>7}  {state:<6}  "
            f"{row['age_seconds']:>8.0f}s  "
            f"{row['heartbeat_age_seconds']:>8.0f}s  "
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
    for job in store.list_jobs():
        if any(artifact.id == artifact_id for artifact in store.list_artifacts(job.id)):
            return job.id
    raise FileNotFoundError(f"artifact not found: {artifact_id}")


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


def cursor_prompt(prompt: str, *, review: bool = False, plan: bool = False, dry_run: bool = False) -> str:
    lines = [prompt]
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

