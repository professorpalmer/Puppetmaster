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
        f"{snapshot.get('code_stale', 0)} old-code, "
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
        elif row.get("code_stale"):
            state = "old"
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
    if snapshot.get("code_stale"):
        print(
            f"\n  {snapshot['code_stale']} server(s) marked 'old' are running "
            f"pre-upgrade code (installed {snapshot.get('installed_version')}). "
            "Restart them — toggle the MCP server in your client or run "
            "`puppetmaster mcp cleanup --kill-stale` — for the new code to load."
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
