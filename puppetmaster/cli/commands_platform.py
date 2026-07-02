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

from puppetmaster.cli.helpers import _registry_path_from_args


def _run_skills_subcommand(args) -> int:
    """Dispatch `python -m puppetmaster skills ...`.

    Closes the puppetmaster-learn flywheel: list the candidates the learn plugin
    distilled, then promote a reviewed one into a live Hermes skill. Promotion is
    only ever this explicit human action — the plugin never auto-promotes.
    """
    sub = args.skills_command

    if sub == "list-candidates":
        candidates = list_skill_candidates()
        if getattr(args, "json", False):
            print(json.dumps(candidates, indent=2))
            return 0
        if not candidates:
            print(
                "No skill candidates yet. They appear here after a durable "
                "Puppetmaster swarm with PUPPETMASTER_LEARN enabled."
            )
            return 0
        print(f"{len(candidates)} skill candidate(s) — promote with "
              f"`puppetmaster skills promote-candidate <slug>`:")
        for candidate in candidates:
            goal = candidate["goal"] or "(no recorded goal)"
            print(f"  {candidate['slug']:40} {goal}")
            print(f"  {'':40} dir={candidate['dir']}  job={candidate['job_id'] or '-'}")
        return 0

    if sub == "promote-candidate":
        outcome = promote_skill_candidate(
            args.slug,
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        ok_statuses = {"promoted", "unchanged", "would_promote"}
        stream = sys.stdout if outcome.status in ok_statuses else sys.stderr
        print(f"promote-candidate: {outcome.status} — {outcome.reason}", file=stream)
        if outcome.target and outcome.status in ok_statuses:
            print(f"  live skill: {outcome.target}", file=stream)
        return 0 if outcome.status in ok_statuses else 1

    raise SystemExit(f"unknown skills subcommand: {sub}")

def _run_platform_subcommand(args) -> int:
    """Dispatch `python -m puppetmaster platform ...`.

    The platform lock decides which adapters Puppetmaster may route to,
    auto-discover, or fall back onto (see ``platform_lock.KNOWN_ADAPTERS``:
    agentic, cursor, claude-code, codex, openai, hermes). It is persisted next
    to the model registry; an empty lock means everything is on.
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
