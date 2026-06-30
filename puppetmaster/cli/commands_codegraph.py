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

from puppetmaster.cli.helpers import _print_codegraph_freshness


def _run_repair_codegraph(args) -> int:
    """CLI entrypoint for `python -m puppetmaster repair-codegraph`.

    Returns 0 when the rebuild succeeds and 1 otherwise. Output mode is
    JSON when ``--json`` is passed and a human-readable summary otherwise.
    """
    import puppetmaster.cli as cli

    result = cli.repair_codegraph_sqlite(
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
        print("examples: codegraph status | codegraph search 'router' | codegraph context 'task' | codegraph freshness", file=sys.stderr)
        return 2

    # `freshness` is a Puppetmaster-native check (not a codegraph subcommand):
    # report whether the index still matches the working tree.
    if cli_args[0] == "freshness":
        return _print_codegraph_freshness(args.cwd or os.getcwd())

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
