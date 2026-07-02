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

from puppetmaster.cli.guidance import (
    CLAUDE_NEXT_STEPS_GUIDANCE,
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    HERMES_NEXT_STEPS_GUIDANCE,
)
from puppetmaster.cli.helpers import (
    _print_install_result,
    _print_rules_result,
    _print_uninstall_hooks_result,
    _print_uninstall_mcp_result,
    _print_uninstall_rules_result,
)


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
    hermes_hooks = uninstall_hermes_hooks(dry_run=dry_run)
    print(f"[uninstall-hermes-hooks] {hermes_hooks.status:<14} {hermes_hooks.reason}")
    if hermes_hooks.status == "error":
        overall_rc |= 1
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

def _agentic_providers_visible() -> set[str]:
    try:
        from puppetmaster.providers import available_providers

        return available_providers()
    except Exception:
        return set()

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


def _seed_agentic_registry() -> None:
    """Seed credential-backed agentic models into the router registry.

    Only models whose provider has a visible API key are added — agentic never
    injects uncallable keyless entries.
    """
    try:
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
        allowed = _agentic_providers_visible()
        existing = load_registry(registry_path)
        merged, report = merge_curated_into_registry(
            "agentic", "api", existing, allowed_providers=allowed
        )
        save_registry(merged, registry_path)
        if report["added"] or report["refreshed"]:
            print(
                f"  registry  seeded agentic models "
                f"(added={report['added']}, refreshed={report['refreshed']})"
            )
        for skip in report.get("skipped", []):
            print(
                f"  registry  skipped {skip['model']} — no credential for provider "
                f"'{skip['provider']}'"
            )
        if not allowed:
            print(
                "  registry  note: no provider API keys visible for agentic "
                "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / "
                "GOOGLE_API_KEY / OPENROUTER_API_KEY). Add a key and re-run "
                "`puppetmaster models discover --source agentic --write`."
            )
    except Exception as exc:  # never let registry seeding abort the wizard
        print(f"  registry  note: agentic registry seeding skipped ({exc!r})")

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
    # Wire Hermes' native shell hooks too, so a single command makes Hermes a
    # full auto-invocation host (MCP server + per-turn delegate hook). Skipped
    # only when the MCP step errored, since hooks without a server are no-ops.
    if result.status in {"installed", "unchanged", "would_install"}:
        hook_outcome = install_hermes_hooks(
            target_path=target_path,
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        print(
            f"[install-hermes-hooks] {hook_outcome.status:<14} {hook_outcome.reason}"
        )
        if hook_outcome.status == "error":
            rc = rc or 1
        # Install the bundled Hermes skill so Hermes has durable procedural
        # knowledge of Puppetmaster (verb decision tree, CodeGraph-first flow,
        # trust gate) — not just the per-turn hook nudge. Non-destructive: an
        # existing customized skill is left alone unless --force.
        skill_outcome = install_hermes_skill(
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        print(
            f"[install-hermes-skill] {skill_outcome.status:<14} {skill_outcome.reason}"
        )
        if skill_outcome.status == "error":
            rc = rc or 1
        rc = rc or _install_hermes_rule_status(
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            label="[install-hermes-rule]",
        )
        # Bundle the opt-in auto-/learn plugin so a finished swarm can distill
        # itself into a Hermes skill candidate. Non-destructive (won't clobber a
        # customized plugin without --force); inert until PUPPETMASTER_LEARN=1.
        plugin_outcome = install_hermes_plugin(
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        print(
            f"[install-hermes-plugin] {plugin_outcome.status:<14} {plugin_outcome.reason}"
        )
        if plugin_outcome.status == "error":
            rc = rc or 1
    if result.status in {"installed", "unchanged"}:
        print()
        print("Next steps:")
        for line in HERMES_NEXT_STEPS_GUIDANCE.splitlines():
            print(f"  {line}")
    return rc

def _install_hermes_rule_status(*, force: bool, dry_run: bool, label: str) -> int:
    """Install Hermes' persistent SOUL.md routing rule and print one status line."""
    result = install_rules(
        cwd=Path.cwd(),
        targets=["hermes_global"],
        install_global=True,
        force=force,
        dry_run=dry_run,
    )
    outcome = result.outcomes[0] if result.outcomes else None
    if outcome is None:
        print(f"{label} error          no hermes_global outcome returned")
        return 1
    print(f"{label} {outcome.status:<14} {outcome.reason}")
    return 1 if result.overall_status == "error" else 0

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

def _run_install_rules(args) -> int:
    """Dispatch for ``puppetmaster install-rules``.

    When no explicit ``--target`` is given, the auto-detected set is filtered to
    the platforms the persisted platform lock allows, and the Cursor rule is
    written only when Cursor is actually *indicated* — an existing ``.cursor/``
    directory or a lock that includes cursor — so a non-Cursor user in a plain
    git repo no longer gets a ``.cursor/rules/`` file. An explicit ``--target``
    always wins (filter bypassed), matching how ``setup`` passes its own filter.
    """
    cwd = Path.cwd()
    targets = None
    raw_target = getattr(args, "target", None)
    if raw_target:
        targets = [t.strip() for t in raw_target.split(",") if t.strip()]
    enabled_adapters = None
    if targets is None:
        from puppetmaster import platform_lock as _pl

        enabled_adapters = set(_pl.enabled_adapters())
        cursor_indicated = (cwd / ".cursor").exists() or (
            _pl.is_configured() and "cursor" in enabled_adapters
        )
        if not cursor_indicated:
            enabled_adapters.discard("cursor")
    result = install_rules(
        cwd=cwd,
        targets=targets,
        install_global=getattr(args, "rules_global", False),
        dry_run=getattr(args, "dry_run", False),
        force=getattr(args, "force", False),
        enabled_adapters=enabled_adapters,
    )
    return _print_rules_result(result)

def _detected_platforms(root: Path) -> dict[str, bool]:
    """Which platform-billed adapters look *usable on this machine*.

    Presence probes, not auth audits: cursor needs ``CURSOR_API_KEY`` plus
    either its bundled SDK or an ``npm`` that setup can bootstrap it with
    (PyPI wheels can't ship ``node_modules``, so a fresh pip/pipx install
    legitimately lacks the SDK until the install-cursor-mcp step fetches
    it);     claude-code and codex need their CLI resolvable; openai needs
    ``OPENAI_API_KEY``; agentic needs any provider API key visible to
    ``available_providers()``. Codex login state is deliberately not probed
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
        "agentic": bool(_agentic_providers_visible()),
        "cursor": bool(os.environ.get("CURSOR_API_KEY")) and cursor_sdk_available,
        "claude-code": _claude_code_installed(),
        "codex": _codex_cli_installed(),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "hermes": _shutil.which("hermes") is not None,
    }

def _setup_platform_step(args) -> int:
    """The `setup` wizard's platform-lock step — a forced-pick gate.

    Runtime default stays permissive (no ``platform.json`` = all adapters on).
    The wizard, however, starts with every platform shown OFF on first run and
    requires an explicit choice of at least one adapter before proceeding.

    Modes:

    * ``--platforms cursor,claude-code`` or ``--platforms all`` — explicit,
      always wins.
    * TTY, no flag — interactive loop until >=1 known adapter is locked.
    * non-interactive, no flag — respects an existing lock (grandfather);
      fails on first run with actionable guidance.
    """
    import puppetmaster.cli as cli

    from puppetmaster import platform_lock as pl

    known = pl.KNOWN_ADAPTERS
    detected = cli._detected_platforms(Path.cwd())
    detected_set = {a for a, present in detected.items() if present}
    configured = pl.is_configured()

    def _show_state(*, wizard_first_run: bool = False) -> None:
        enabled = set() if wizard_first_run else pl.enabled_adapters()
        for adapter in known:
            mark = "on " if adapter in enabled else "off"
            note = "" if detected.get(adapter, True) else "   (not detected on this machine)"
            print(f"  [{mark}] {adapter}{note}")

    raw = getattr(args, "platforms", None)
    if raw is not None:
        if raw.strip().lower() == "all":
            pl.reset()
            print("  reset  all platforms enabled")
            _show_state()
            return 0
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
        _show_state(wizard_first_run=not configured)
        return 0

    if not sys.stdin.isatty():
        if configured:
            print(
                "  unchanged  existing platform lock respected "
                "(non-interactive shell, no --platforms flag)"
            )
            _show_state()
            return 0
        print(
            "  error  no platform selected — re-run in an interactive terminal or "
            f"pass --platforms <comma-list>. Known: {', '.join(known)}."
        )
        _show_state(wizard_first_run=True)
        return 1

    print("Puppetmaster routes work across these platforms.")
    print(
        "Most users enable a single platform (e.g. just cursor). Enabling multiple "
        "platforms unlocks cross-platform router fallback/healing and free-tier "
        "hopping — opt in anytime with `puppetmaster platform enable <name>`."
    )
    _show_state(wizard_first_run=not configured)
    if not configured and detected_set:
        print(
            f"Detected on this machine: {', '.join(sorted(detected_set))} — "
            "you must still choose at least one to enable."
        )
    if not configured:
        prompt_hint = (
            "Enter a comma-separated list of platforms to ENABLE (all others off),\n"
            "or type 'all' to enable every platform."
        )
    else:
        prompt_hint = (
            "Enter a comma-separated list of platforms to ENABLE (all others off),\n"
            "'all' to keep every platform on, or press Enter to leave unchanged."
        )
    print(prompt_hint)

    reprompt = (
        'You must enable at least one platform to continue '
        '(or type "all" to enable every platform).'
    )

    while True:
        try:
            answer = input("  platforms> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(
                "\n  aborted  platform selection required — re-run setup in an "
                "interactive terminal or pass --platforms <comma-list>."
            )
            return 1

        if not answer:
            if configured:
                print("  unchanged  platform lock left as-is")
                _show_state()
                return 0
            print(f"  {reprompt}")
            continue

        if answer.lower() == "all":
            pl.reset()
            print("  reset  all platforms enabled")
            _show_state()
            return 0

        wanted = {a.strip() for a in answer.split(",") if a.strip()}
        unknown = sorted(a for a in wanted if a not in known)
        if unknown:
            print(
                f"  error  unknown platform(s): {', '.join(unknown)}. "
                f"Known: {', '.join(known)}."
            )
            continue
        valid = {a for a in wanted if a in known}
        if not valid:
            print(f"  {reprompt}")
            continue
        pl.set_enabled(valid)
        print(f"  locked  routing restricted to: {', '.join(sorted(valid))}")
        _show_state()
        return 0

def _setup_hermes_advanced(args) -> int:
    """Optional Hermes flywheel + skill-injection knobs (setup step 7 extension).

    Persists accepted toggles into ``mcp_servers.puppetmaster.env`` in Hermes'
    config.yaml. Non-interactive runs print the knobs instead of prompting.
    """
    if getattr(args, "skip_hermes_advanced", False):
        print("  [hermes-advanced] skipped (--skip-hermes-advanced)")
        return 0

    guidance = [
        "PUPPETMASTER_LEARN=1 — finished swarms distill into Hermes skill CANDIDATES "
        "(never auto-promoted). Review: `puppetmaster skills list-candidates`; promote: "
        "`puppetmaster skills promote-candidate <slug>`.",
        "PUPPETMASTER_INJECT_HERMES_SKILLS=1 — routed Hermes workers inherit your "
        "curated live skill bodies (per-turn token cost on every worker turn; tune with "
        "PUPPETMASTER_SKILL_TOKEN_BUDGET, default 1200).",
        "Set these on mcp_servers.puppetmaster.env in Hermes config.yaml, or re-run "
        "`puppetmaster setup` interactively.",
    ]

    if not sys.stdin.isatty():
        print("  [hermes-advanced] non-interactive — optional Hermes MCP env knobs:")
        for line in guidance:
            print(f"    {line}")
        return 0

    print("  [hermes-advanced] Optional Hermes flywheel (writes to mcp_servers.puppetmaster.env)")
    env_updates: dict[str, str] = {}

    try:
        learn_answer = input(
            "  Enable learn flywheel (PUPPETMASTER_LEARN=1)? [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  [hermes-advanced] skipped — no input")
        return 0
    if learn_answer in {"y", "yes"}:
        env_updates["PUPPETMASTER_LEARN"] = "1"
        print(
            "  learn  enabled — finished swarms write skill CANDIDATES under "
            "~/.hermes/skills-candidates/ (never auto-promoted)."
        )
        print(
            "  review  `puppetmaster skills list-candidates` then "
            "`puppetmaster skills promote-candidate <slug>` to promote a keeper."
        )
    else:
        print("  learn  left off (enable later via Hermes MCP env or re-run setup)")

    try:
        inject_answer = input(
            "  Enable skill injection (PUPPETMASTER_INJECT_HERMES_SKILLS=1)? "
            "Injected skill bodies ride every worker turn (per-turn token cost). [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  [hermes-advanced] skipped remaining prompts — no input")
        if env_updates:
            result = set_hermes_mcp_env(env_updates, dry_run=getattr(args, "dry_run", False))
            for line in result.messages:
                print(f"  [hermes-advanced] {line}")
            if result.status == "error":
                return 1
        return 0

    if inject_answer in {"y", "yes"}:
        env_updates["PUPPETMASTER_INJECT_HERMES_SKILLS"] = "1"
        print(
            "  inject  enabled — routed Hermes workers inherit curated live skill bodies."
        )
        try:
            budget_answer = input(
                "  Set PUPPETMASTER_SKILL_TOKEN_BUDGET (default 1200, Enter to keep default)? "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            budget_answer = ""
        if budget_answer:
            env_updates["PUPPETMASTER_SKILL_TOKEN_BUDGET"] = budget_answer
            print(f"  budget  PUPPETMASTER_SKILL_TOKEN_BUDGET={budget_answer}")
        else:
            print("  budget  using default (1200 tokens per worker turn)")
    else:
        print("  inject  left off (enable later via Hermes MCP env or re-run setup)")

    if not env_updates:
        return 0

    result = set_hermes_mcp_env(env_updates, dry_run=getattr(args, "dry_run", False))
    for line in result.messages:
        print(f"  [hermes-advanced] {line}")
    return 1 if result.status == "error" else 0

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
    import puppetmaster.cli as cli

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
    platform_rc = _setup_platform_step(args)
    if platform_rc != 0:
        print(
            "\nSetup aborted at platform lock — choose at least one platform "
            "and re-run (interactive terminal or --platforms <comma-list>)."
        )
        return 1

    from puppetmaster import platform_lock as _pl
    if not _pl.is_configured():
        print(
            "\nSetup aborted — no platform selected. Re-run in an interactive "
            "terminal or pass --platforms <comma-list>."
        )
        return 1
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

    if "agentic" in enabled_adapters:
        print("=== agentic (keys-only standalone worker) ===")
        providers = sorted(_agentic_providers_visible())
        if providers:
            print(f"  providers  ready: {', '.join(providers)}")
        else:
            print(
                "  providers  none visible — set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "GEMINI_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY"
            )
        if not getattr(args, "skip_models", False):
            _seed_agentic_registry()
        print()

    print("=== step 4/9: install-cursor-mcp (workspace .cursor/mcp.json) ===")
    if "cursor" not in enabled_adapters:
        print(
            "  skipped  cursor platform disabled by the platform lock — not "
            "installing its MCP client (.cursor/mcp.json)"
        )
    else:
        sdk = cli.ensure_cursor_sdk(cwd)
        print(f"  sdk {sdk.status}  {sdk.detail}")
        cursor_result = cli.install_cursor_mcp(
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
        codex_result = cli.install_codex_mcp(
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
        hermes_result = cli.install_hermes_mcp(
            force=getattr(args, "force", False),
            dry_run=False,
            skip_handshake=False,
        )
        for line in hermes_result.messages:
            print(f"  {line}")
        if hermes_result.status not in {"installed", "unchanged", "would_install"}:
            overall_rc = 1
        # Wire Hermes' native shell hooks too, so `setup` makes Hermes a full
        # auto-invocation host (MCP server + per-turn delegate hook) — parity
        # with the Cursor/Claude hooks installed in step 9, which only touch
        # those harnesses' hook files. Skipped when the MCP step errored (a hook
        # without a live server is a no-op) or when --skip-hooks is set.
        if (
            hermes_result.status in {"installed", "unchanged", "would_install"}
            and not getattr(args, "skip_hooks", False)
        ):
            hermes_hooks = cli.install_hermes_hooks(
                force=getattr(args, "force", False),
                dry_run=False,
            )
            print(
                f"  [hermes-hooks] {hermes_hooks.status:<14} {hermes_hooks.reason}"
            )
            if hermes_hooks.status == "error":
                overall_rc = 1
        elif getattr(args, "skip_hooks", False):
            print("  [hermes-hooks] skipped (--skip-hooks)")
        # Install the bundled Hermes skill (durable procedural knowledge), gated
        # by --skip-rules since a skill is the same category of artifact as the
        # agent rule files installed in step 8. Non-destructive (won't clobber a
        # customized skill without --force).
        if (
            hermes_result.status in {"installed", "unchanged", "would_install"}
            and not getattr(args, "skip_rules", False)
        ):
            hermes_skill = install_hermes_skill(force=getattr(args, "force", False))
            print(
                f"  [hermes-skill] {hermes_skill.status:<14} {hermes_skill.reason}"
            )
            if hermes_skill.status == "error":
                overall_rc = 1
            if _install_hermes_rule_status(
                force=getattr(args, "force", False),
                dry_run=getattr(args, "dry_run", False),
                label="  [install-hermes-rule]",
            ):
                overall_rc = 1
            hermes_plugin = install_hermes_plugin(force=getattr(args, "force", False))
            print(
                f"  [hermes-plugin] {hermes_plugin.status:<14} {hermes_plugin.reason}"
            )
            if hermes_plugin.status == "error":
                overall_rc = 1
        elif getattr(args, "skip_rules", False):
            print("  [hermes-skill] skipped (--skip-rules)")
            print("  [install-hermes-rule] skipped (--skip-rules)")
            print("  [hermes-plugin] skipped (--skip-rules)")
        if not getattr(args, "skip_models", False):
            cli._seed_hermes_registry()
        if (
            hermes_result.status in {"installed", "unchanged", "would_install"}
            and not getattr(args, "skip_rules", False)
        ):
            if _setup_hermes_advanced(args) != 0:
                overall_rc = 1
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
        print("Setup complete.")
        for line in _setup_next_steps(enabled_adapters):
            print(f"  {line}")
    else:
        print("Setup completed with errors — see above. Individual `puppetmaster install-*` commands can be re-run after fixing.")
    return overall_rc

def _setup_next_steps(enabled_adapters: set[str]) -> list[str]:
    """Next-steps lines tailored to the platform(s) the user actually enabled.

    Cursor/Codex/Claude/Hermes each pick up the MCP server differently (restart
    vs. fresh session); the local/file mode needs no host at all. Rendering from
    the enabled set means a non-Cursor user never sees a "restart Cursor" nudge
    that doesn't apply to them.
    """
    steps: list[str] = []
    if "cursor" in enabled_adapters:
        steps.append("cursor: restart Cursor (or open a fresh chat) to pick up the MCP server.")
    if "codex" in enabled_adapters:
        steps.append("codex: start a new `codex` session to pick up the MCP server.")
    if "claude-code" in enabled_adapters:
        steps.append("claude-code: start a new Claude Code session; verify with `claude mcp list`.")
    if "hermes" in enabled_adapters:
        steps.append("hermes: start a new Hermes session; verify with `hermes mcp list`.")
    if "openai" in enabled_adapters:
        steps.append("openai: set OPENAI_API_KEY; the API adapter needs no host restart.")
    if "agentic" in enabled_adapters:
        steps.append(
            "agentic: set a provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "GEMINI_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY). No external "
            "CLI or host restart — drive via `puppetmaster agentic` or "
            "`puppetmaster_agentic` / `puppetmaster_start_agentic` MCP verbs."
        )
    if not steps:
        # Lock excludes every host-backed adapter: only the local/file demo mode
        # is available, which runs straight from the CLI without a host.
        steps.append(
            "local/file mode: no host to restart — drive Puppetmaster directly "
            "via `python -m puppetmaster run …` (CLI / file workflows)."
        )
    steps.append(
        "Verify any host by asking its agent to call `puppetmaster_doctor`, or run "
        "`python -m puppetmaster doctor`."
    )
    return steps

def _run_self_update(args) -> int:
    """Explicit user-invoked upgrade — never called from the MCP server."""
    import puppetmaster.cli as cli

    cmd = [cli.sys.executable, "-m", "pip", "install", "-U", "puppetmaster-ai"]
    cmd_display = " ".join(cmd)
    if args.dry_run:
        print(cmd_display)
        return 0

    completed = cli.subprocess.run(cmd)
    if completed.returncode != 0:
        return completed.returncode

    print(
        "Successfully upgraded puppetmaster-ai. The long-lived MCP stdio server "
        "cannot reload in place — restart it: toggle the MCP server in your client, "
        "or run `puppetmaster mcp cleanup --kill-stale`."
    )
    return 0

def _run_install_hooks(args) -> int:
    """Dispatch for ``puppetmaster install-hooks``.

    With no explicit ``--target``, the default cursor+claude set is filtered to
    the platforms the persisted platform lock allows, so a non-Cursor user no
    longer gets a ``.cursor/hooks.json`` written just because that was the old
    default. An explicit ``--target`` bypasses the filter (explicit intent wins),
    matching how ``setup`` passes its own filter.
    """
    targets = None
    raw = getattr(args, "target", None)
    if raw:
        targets = [t.strip() for t in raw.split(",") if t.strip()]
    enabled_adapters = None
    if targets is None:
        from puppetmaster import platform_lock as _pl

        enabled_adapters = _pl.enabled_adapters()
    scope = "global" if getattr(args, "global_scope", False) else "project"
    result = install_hooks(
        cwd=Path.cwd(),
        targets=targets,
        dry_run=getattr(args, "dry_run", False),
        force=getattr(args, "force", False),
        scope=scope,
        enabled_adapters=enabled_adapters,
    )
    print(f"[install-hooks] overall: {result.overall_status} (scope={scope})")
    for outcome in result.outcomes:
        print(f"[install-hooks] {outcome.target:<8} {outcome.status:<14} {outcome.reason}")
        if outcome.path:
            print(f"[install-hooks] {' ' * 8} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[install-hooks] note: {msg}")
    return 1 if result.overall_status == "error" else 0
