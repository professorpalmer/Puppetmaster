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
from puppetmaster.state import (
    find_state_dir_for_job,
    list_project_state_dirs,
    resolve_state_dir,
)
from puppetmaster.store_factory import create_store
from puppetmaster.stitcher import Stitcher
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec

from puppetmaster.cli._parser import build_parser
from puppetmaster.cli.guidance import _NOISY_LOG_EVENTS
from puppetmaster.cli.helpers import (
    allowed_models_cli_list,
    approve_target,
    artifact_feed_since,
    cursor_prompt,
    early_job_printer,
    print_feed_item,
    print_run_result,
    print_watch_snapshot,
    reject_target,
    require_latest_job_id,
    routing_payload_from_args,
    run_deltas_follow,
    run_feed_follow,
    _warn_job_liveness,
    _warn_run_quality,
)
from puppetmaster.cli.commands_install import (
    _run_install_claude,
    _run_install_codex,
    _run_install_cursor,
    _run_install_hermes,
    _run_install_hooks,
    _run_install_rules,
    _run_self_update,
    _run_setup,
    _run_uninstall,
)
from puppetmaster.cli.commands_mcp import _run_mcp_subcommand
from puppetmaster.cli.commands_keys import run_keys_subcommand
from puppetmaster.cli.commands_models import _run_models_subcommand
from puppetmaster.cli.commands_evaluators import _run_evaluators_subcommand
from puppetmaster.cli.commands_platform import _run_platform_subcommand, _run_skills_subcommand
from puppetmaster.cli.commands_codegraph import _run_codegraph_passthrough, _run_repair_codegraph
from puppetmaster.cli.commands_research import _run_research_subcommand
from puppetmaster.cli.commands_jobs import (
    _reap_quietly,
    _run_await_command,
    _run_finalize_command,
    _run_gc_command,
    _run_reap_command,
    _run_wait_command,
)
from puppetmaster.cli.commands_gate import (
    _run_affected_command,
    _run_audit_command,
    _run_cost_command,
    _run_gate_command,
    _run_invocation_gate_command,
    _run_preflight_command,
    _run_proxy_command,
    _run_receipt_command,
    _run_rollup_command,
    _run_route_command,
    _run_savings_command,
    _run_should_delegate_command,
)


def main(argv: Optional[list[str]] = None) -> int:
    from puppetmaster.win_console import hide_child_consoles

    hide_child_consoles()
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


def _start_background_dashboard(
    args: argparse.Namespace,
    state_dir: Path,
    host: str,
    *,
    port: int,
    auto_port: bool,
    source: str,
    allow_external: bool,
) -> int:
    """Spawn a detached dashboard, wait for it to answer, and return the link.

    The detached child runs the ordinary foreground server in its own
    session, so it survives this process exiting.

    Reuse is identity-aware (``dashboard_serves``), not a bare liveness
    probe: something merely *answering* on ``host:port`` used to be enough to
    declare "reusing it" — even when it was a different project's dashboard,
    which strands the caller with a URL for state it doesn't own and no
    runfile of its own (``--status``/``--stop`` then fail on it). And because
    the child may bump to a port other than the one requested (``auto_port``,
    §3.1), *it* — not this parent — writes the runfile once it knows the real
    port (``--write-runfile``); the parent only ever reads that back.
    """
    from puppetmaster.dashboard import (
        clear_dashboard_runfile,
        dashboard_identity,
        dashboard_serves,
        pid_alive,
        read_dashboard_runfile,
        write_dashboard_runfile,
    )

    all_projects = getattr(args, "all_projects", False)

    def _reuse(existing_host: str, existing_port: int, existing_pid) -> int:
        url = f"http://{existing_host}:{existing_port}/" + (
            f"?job={args.job_id}" if args.job_id else ""
        )
        print(f"Dashboard already serving at {url} — reusing it.")
        # Closes the bonus defect where the old blind-reuse early return
        # preceded the runfile write entirely: without this, the caller that
        # "reused" the server had no runfile of its own, so its own
        # --status/--stop had nothing to work with.
        write_dashboard_runfile(
            state_dir,
            {
                "pid": existing_pid,
                "host": existing_host,
                "port": existing_port,
                "url": url,
                "all_projects": all_projects,
            },
        )
        _announce_background_dashboard(args, existing_host, existing_port, source)
        return 0

    # This project's own background dashboard may already be running — at
    # whatever port it actually bound to, which can be past `port` if it
    # bumped on a previous run. Check the tracked address first so a repeat
    # `--background` doesn't spawn a redundant second server for the same
    # project (and clobber the runfile the first one is still using).
    existing = read_dashboard_runfile(state_dir)
    if existing and pid_alive(int(existing.get("pid") or 0)):
        existing_host = existing.get("host", host)
        existing_port = int(existing.get("port") or port)
        if dashboard_serves(existing_host, existing_port, state_dir, all_projects=all_projects):
            return _reuse(existing_host, existing_port, existing.get("pid"))

    # Nothing tracked (or it's stale/foreign) — something may still identify
    # itself as this project's dashboard on the requested port specifically.
    if dashboard_serves(host, port, state_dir, all_projects=all_projects):
        identity = dashboard_identity(host, port)
        return _reuse(host, port, (identity or {}).get("pid"))

    # Clear any stale/foreign runfile so the readiness poll below can't
    # mistake it for the child we're about to spawn.
    clear_dashboard_runfile(state_dir)

    command = [
        sys.executable,
        "-m",
        "puppetmaster",
        "--state-dir",
        str(state_dir),
        "--backend",
        args.backend,
        "dashboard",
        "--port",
        str(port),
        "--no-open",
        "--write-runfile",
    ]
    if auto_port:
        command.append("--port-search")
    if host.strip().lower() not in {"127.0.0.1", "localhost", "::1"}:
        command += ["--host", host]
    if allow_external:
        command.append("--allow-external")
    if all_projects:
        command.append("--all-projects")
    if args.job_id:
        command.append(args.job_id)

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Poll the child's own runfile rather than re-probing `host:port` — the
    # child may have bumped, so a liveness/identity probe pinned to the
    # requested port can miss it entirely (or, worse, see a *different*
    # project's dashboard on that port and declare success).
    deadline = time.time() + 10
    child_info = None
    while time.time() < deadline:
        candidate = read_dashboard_runfile(state_dir)
        if candidate and candidate.get("pid") == process.pid:
            child_info = candidate
            break
        if process.poll() is not None:
            break
        time.sleep(0.2)

    if child_info is None or not dashboard_serves(
        child_info.get("host", host),
        int(child_info.get("port") or port),
        state_dir,
        all_projects=all_projects,
    ):
        print(
            "puppetmaster dashboard --background: server did not come up. Run "
            f"`python -m puppetmaster dashboard --port {port}` in the foreground "
            "to see the error.",
            file=sys.stderr,
        )
        return 1

    bound_host = child_info.get("host", host)
    bound_port = int(child_info.get("port") or port)
    print(f"Dashboard running in the background (pid {process.pid}).")
    _announce_background_dashboard(args, bound_host, bound_port, source)
    print("Stop it with: python -m puppetmaster dashboard --stop")
    return 0


def _announce_background_dashboard(
    args: argparse.Namespace, host: str, port: int, source: str
) -> None:
    """Print the phone banner (mobile) or a plain URL line (loopback)."""
    from puppetmaster.dashboard import print_mobile_banner

    if getattr(args, "mobile", False):
        print_mobile_banner(
            host, port, source, job_id=args.job_id, qr=getattr(args, "qr", False)
        )
    else:
        url = f"http://{host}:{port}/" + (f"?job={args.job_id}" if args.job_id else "")
        print(f"  {url}")


def _run_dashboard_command(args: argparse.Namespace, state_dir: Path) -> int:
    from puppetmaster.dashboard import (
        dashboard_alive,
        dashboard_serves,
        print_mobile_banner,
        read_dashboard_runfile,
        resolve_mobile_host,
        serve,
        stop_background_dashboard,
    )

    # argparse's --port default is now a sentinel (None), not 8787: only that
    # lets us tell "user typed --port 8787" apart from "user typed nothing"
    # (§3.3/§3.7(a)), which is what makes an explicit --port strict-fail
    # instead of silently landing on a different port than the one asked for.
    explicit_port = args.port is not None
    port = args.port if explicit_port else 8787
    auto_port = (not explicit_port) or getattr(args, "port_search", False)
    all_projects = getattr(args, "all_projects", False)

    if getattr(args, "stop", False):
        result = stop_background_dashboard(state_dir)
        if result.get("stopped"):
            where = f" ({result['url']})" if result.get("url") else ""
            print(f"Stopped the background dashboard{where}.")
            return 0
        print(
            f"No background dashboard to stop: {result.get('reason', 'none tracked')}.",
            file=sys.stderr,
        )
        return 0

    if getattr(args, "status", False):
        info = read_dashboard_runfile(state_dir)
        if info:
            info_host = info.get("host", "127.0.0.1")
            info_port = int(info.get("port") or port)
            if dashboard_serves(info_host, info_port, state_dir, all_projects=all_projects):
                print(
                    f"Background dashboard running: {info.get('url')} (pid {info.get('pid')})."
                )
            elif dashboard_alive(info_host, info_port):
                # Something answers at the tracked address, but it isn't
                # provably *this* project (or --all-projects scope doesn't
                # match) — the old bare-liveness check reported this as
                # "running" even when it was a different project's board.
                print(
                    "A dashboard is running at the tracked address, but it is "
                    "serving another project (or a different --all-projects "
                    "scope) — not this state dir. Clear it with `dashboard "
                    "--stop` and start a fresh one."
                )
            else:
                print(
                    "A background dashboard is tracked here but isn't answering "
                    "(stale). Clear it with `dashboard --stop`."
                )
        else:
            print("No background dashboard is running for this state dir.")
        return 0

    host = args.host
    allow_external = getattr(args, "allow_external", False)
    open_browser = not args.no_open
    source = "loopback"
    if getattr(args, "mobile", False):
        ip, source = resolve_mobile_host()
        if ip is None:
            print(
                "puppetmaster dashboard --mobile: could not detect a Tailscale "
                "or LAN address. Join a network (or bring Tailscale up) and "
                "retry, or set --host <ip> --allow-external manually.",
                file=sys.stderr,
            )
            return 1
        host = ip
        allow_external = True
        open_browser = False  # the browser is on the phone, not this host

    if getattr(args, "background", False):
        return _start_background_dashboard(
            args,
            state_dir,
            host,
            port=port,
            auto_port=auto_port,
            source=source,
            allow_external=allow_external,
        )

    # The mobile banner used to print before serve() bound the socket, so its
    # QR/URL named the *requested* port even when the bind later bumped past
    # it (§3.4 step 7). Deferring it to serve()'s on_bound callback means it
    # only ever prints the port that's actually listening.
    on_bound = None
    if getattr(args, "mobile", False):

        def on_bound(bound_host: str, bound_port: int) -> None:
            print_mobile_banner(
                bound_host,
                bound_port,
                source,
                job_id=args.job_id,
                qr=getattr(args, "qr", False),
            )

    try:
        serve(
            state_dir,
            backend=args.backend,
            job_id=args.job_id,
            host=host,
            port=port,
            open_browser=open_browser,
            allow_external=allow_external,
            all_projects=all_projects,
            auto_port=auto_port,
            on_bound=on_bound,
            runfile_state_dir=state_dir if getattr(args, "write_runfile", False) else None,
        )
    except OSError as exc:
        # bind_dashboard_server's strict-fail / cap-exhausted errors, e.g. an
        # explicit --port that's busy. main() only catches FileNotFoundError/
        # ValueError/RuntimeError (_dispatch.py:136-141), so an uncaught
        # OSError here would otherwise print a raw traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _main(argv: Optional[list[str]] = None) -> int:
    import puppetmaster.cli as cli

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
        "receipt",
        "artifacts",
        "graph",
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

    if args.command == "self-update":
        return _run_self_update(args)

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

    if args.command == "evaluators":
        return _run_evaluators_subcommand(args, state_dir=state_dir)

    if args.command == "keys":
        return run_keys_subcommand(args)

    if args.command == "platform":
        return _run_platform_subcommand(args)

    if args.command == "skills":
        return _run_skills_subcommand(args)

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

    if args.command == "receipt":
        return _run_receipt_command(args, store)

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
        # The orchestrator's watchdog reads `timeout_seconds` off each task
        # payload (Orchestrator._worker_wait_timeout). Nothing in the default
        # worker specs sets it, so an unstamped `run` silently falls back to
        # the 60s floor + 30s grace = 90s base / 270s hard cap -- shorter than
        # a single real agent turn. Stamp the caller's value onto every role.
        #
        # Applied on top of `--config` too, so an explicit flag outranks the
        # per-worker values a config declares -- the usual precedence, and what
        # MCP `start_swarm` relies on to honor its advertised `timeout_seconds`
        # when the caller supplied their own config. Skipped entirely when the
        # flags are absent, so a config-only run is untouched.
        timeout_seconds = getattr(args, "timeout_seconds", None)
        max_timeout_seconds = getattr(args, "max_timeout_seconds", None)
        if timeout_seconds is not None or max_timeout_seconds is not None:
            overrides: dict = {}
            if timeout_seconds is not None:
                overrides["timeout_seconds"] = int(timeout_seconds)
            if max_timeout_seconds is not None:
                overrides["max_timeout_seconds"] = int(max_timeout_seconds)
            specs = [
                replace(spec, payload={**spec.payload, **overrides})
                for spec in specs
            ]
        result = cli.Orchestrator(store).run(
            args.goal,
            specs=specs,
            lease_seconds=lease_seconds,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
            label=args.label,
        )
        return cli.finalize_cli_run(result)

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
            "timeout_seconds": args.timeout_seconds,
        }
        if args.model and args.model != "default":
            from puppetmaster.model_registry import (
                AmbiguousModelPinError,
                apply_cursor_model_pin,
            )

            try:
                payload.update(apply_cursor_model_pin({}, args.model))
            except AmbiguousModelPinError as exc:
                # Structured preflight/blocked contract (never a raw traceback).
                blocked = {
                    "ok": False,
                    "adapter": "cursor",
                    "model": args.model,
                    "billing": "unknown",
                    "reason": f"ambiguous model pin: {exc}",
                    "evidence": [
                        "adapter:cursor",
                        "preflight:ambiguous_model_pin",
                    ],
                    "failure": "preflight_blocked",
                    "result": "blocked",
                }
                print(json.dumps(blocked, indent=2))
                return 1
        else:
            payload["model"] = args.model
        if implement:
            payload["mode"] = "implement"
            payload["allow_dirty"] = getattr(args, "allow_dirty", False)
            payload["allow_non_worktree"] = getattr(args, "allow_non_worktree", False)
        if args.disable_memory or args.review or args.plan:
            payload["disable_memory"] = True
        payload.update(routing_payload_from_args(args, adapter="cursor"))
        result = cli.Orchestrator(store).run(
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
            label=args.label,
        )
        return cli.finalize_cli_run(result)

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
        if getattr(args, "effort", None):
            payload["extra_args"] = ["--effort", args.effort]
        payload.update(routing_payload_from_args(args, adapter="claude-code"))
        result = cli.Orchestrator(store).run(
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
            label=args.label,
        )
        return cli.finalize_cli_run(result)

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
        result = cli.Orchestrator(store).run(
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
            label=args.label,
        )
        return cli.finalize_cli_run(result)

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
        result = cli.Orchestrator(store).run(
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
            label=args.label,
        )
        return cli.finalize_cli_run(result)

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
        result = cli.Orchestrator(store).run(
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
            label=args.label,
        )
        return cli.finalize_cli_run(result)

    if args.command == "agentic":
        payload = {
            "prompt": args.prompt,
            "cwd": args.cwd,
            "mode": args.mode,
            "timeout_seconds": args.timeout_seconds,
            "allow_dirty": args.allow_dirty,
            "allow_non_worktree": args.allow_non_worktree,
        }
        if args.model:
            from puppetmaster.model_registry import (
                AmbiguousModelPinError,
                apply_agentic_model_pin,
            )

            try:
                payload.update(apply_agentic_model_pin({}, args.model))
            except AmbiguousModelPinError as exc:
                print(f"agentic: {exc}", file=sys.stderr)
                return 2
        # Explicit --provider wins over registry payload_defaults.provider.
        if args.provider:
            payload["provider"] = args.provider
        if args.max_turns is not None:
            payload["max_turns"] = args.max_turns
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        if args.disable_codegraph:
            payload["disable_codegraph"] = True
        if args.disable_memory:
            payload["disable_memory"] = True
        payload.update(routing_payload_from_args(args, adapter="agentic"))
        result = cli.Orchestrator(store).run(
            args.prompt,
            specs=[
                WorkerSpec(
                    role=f"agentic-{args.mode}",
                    instruction=args.prompt,
                    adapter="agentic",
                    payload=payload,
                )
            ],
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
            label=args.label,
        )
        return cli.finalize_cli_run(result)

    if args.command == "edit":
        from puppetmaster import platform_lock
        from puppetmaster.workers import (
            NoImplementAdapterError,
            build_edit_spec,
            pick_implement_adapter,
        )

        enabled = platform_lock.enabled_adapters()
        try:
            adapter = pick_implement_adapter(enabled, args.adapter)
        except NoImplementAdapterError as exc:
            print(f"edit: {exc}", file=sys.stderr)
            if exc.requested:
                print(
                    f"  enable it: puppetmaster platform enable {exc.requested}",
                    file=sys.stderr,
                )
            print(f"  enabled adapters: {', '.join(sorted(exc.enabled)) or '(none)'}", file=sys.stderr)
            return 2
        spec = build_edit_spec(
            instruction=args.instruction,
            adapter=adapter,
            cwd=args.cwd,
            model=args.model,
            provider=args.provider,
            timeout_seconds=args.timeout_seconds,
            routing_policy=args.routing_policy,
            auto_route=getattr(args, "auto_route_edit", True),
            disable_codegraph=args.disable_codegraph,
            allowed_model_ids=allowed_models_cli_list(args) or None,
        )
        if args.executable:
            spec.payload["executable"] = args.executable
        result = cli.Orchestrator(store).run(
            args.instruction,
            specs=[spec],
            lease_seconds=10,
            worker_mode="inline",
            on_job_created=on_job_created,
            label=args.label,
        )
        return cli.finalize_cli_run(result)

    if args.command == "prewalk":
        from puppetmaster import platform_lock
        from puppetmaster.prewalk import build_prewalk_specs
        from puppetmaster.workers import (
            NoImplementAdapterError,
            pick_implement_adapter,
        )

        enabled = platform_lock.enabled_adapters()
        try:
            implement_adapter = pick_implement_adapter(enabled, args.adapter)
        except NoImplementAdapterError as exc:
            print(f"prewalk: {exc}", file=sys.stderr)
            if exc.requested:
                print(
                    f"  enable it: puppetmaster platform enable {exc.requested}",
                    file=sys.stderr,
                )
            print(
                f"  enabled adapters: {', '.join(sorted(exc.enabled)) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        plan_adapter = args.plan_adapter or "local"
        verify_timeout = getattr(args, "verify_timeout_seconds", None)
        if verify_timeout is None:
            verify_timeout = args.timeout_seconds
        allowed_models = [
            str(item).strip()
            for item in (getattr(args, "allowed_models", None) or [])
            if str(item).strip()
        ]
        specs = build_prewalk_specs(
            args.goal,
            args.cwd,
            plan_adapter=plan_adapter,
            implement_adapter=implement_adapter,
            verify_adapter=getattr(args, "verify_adapter", None),
            plan_model=args.plan_model,
            implement_model=args.model,
            verify_model=getattr(args, "verify_model", None),
            plan_timeout_seconds=args.timeout_seconds,
            implement_timeout_seconds=args.timeout_seconds,
            verify_timeout_seconds=verify_timeout,
            auto_route=getattr(args, "auto_route_prewalk", True),
            allow_dirty=args.allow_dirty,
            allow_non_worktree=args.allow_non_worktree,
            disable_codegraph=args.disable_codegraph,
            disable_memory=args.disable_memory,
            allowed_model_ids=allowed_models or None,
        )
        result = cli.Orchestrator(store).run(
            args.goal,
            specs=specs,
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
            label=args.label,
        )
        return cli.finalize_cli_run(result)

    if args.command == "swarm":
        from dataclasses import replace

        from puppetmaster import platform_lock
        from puppetmaster.swarm_launch import (
            DEFAULT_SWARM_ROLES,
            build_analysis_swarm_specs,
            detach_analysis_swarm,
        )

        adapter = str(getattr(args, "adapter", None) or "cursor")
        if adapter != "local" and not platform_lock.is_adapter_enabled(adapter):
            print(
                f"swarm: adapter {adapter!r} is disabled by the platform lock.",
                file=sys.stderr,
            )
            print(f"  enable it: puppetmaster platform enable {adapter}", file=sys.stderr)
            return 2

        roles = list(args.roles) if getattr(args, "roles", None) else list(DEFAULT_SWARM_ROLES)
        disable_memory = not bool(getattr(args, "enable_memory", False))
        auto_route = None
        if getattr(args, "no_auto_route", False):
            auto_route = False
        elif getattr(args, "auto_route", None):
            auto_route = True

        if getattr(args, "wait", False):
            specs = build_analysis_swarm_specs(
                args.goal,
                roles,
                adapter=adapter,
                cwd=args.cwd,
                timeout_seconds=int(args.timeout_seconds),
                max_timeout_seconds=getattr(args, "max_timeout_seconds", None),
                model=args.model,
                auto_route=auto_route,
                routing_policy=getattr(args, "routing_policy", None),
                disable_memory=disable_memory,
            )
            if args.enable_memory:
                specs = [
                    replace(spec, payload={**spec.payload, "disable_memory": False})
                    for spec in specs
                ]
            result = cli.Orchestrator(store).run(
                args.goal,
                specs=specs,
                lease_seconds=10,
                worker_mode=args.worker_mode,
                on_job_created=on_job_created or early_job_printer,
                label=args.label,
            )
            return cli.finalize_cli_run(result)

        try:
            body = detach_analysis_swarm(
                goal=args.goal,
                roles=roles,
                adapter=adapter,
                state_dir=state_dir,
                cwd=args.cwd,
                timeout_seconds=int(args.timeout_seconds),
                max_timeout_seconds=getattr(args, "max_timeout_seconds", None),
                model=args.model,
                auto_route=auto_route,
                routing_policy=getattr(args, "routing_policy", None),
                disable_memory=disable_memory,
                label=args.label,
                worker_mode=args.worker_mode,
                backend=args.backend,
            )
        except Exception as exc:
            print(f"swarm: failed to start: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            print(json.dumps(body, indent=2))
        else:
            print(body["job_id"])
            print(
                f"# detached launcher_pid={body['launcher_pid']}  "
                f"next: python -m puppetmaster feed {body['job_id']} --follow",
                file=sys.stderr,
            )
        return 0

    if args.command == "research":
        return _run_research_subcommand(args, state_dir)

    if args.command == "browser":
        from puppetmaster.browser import (
            BrowserAdapterUnavailable,
            browser_swarm_specs,
            resolve_browser_adapter,
        )

        # Hermes is preferred; agentic/OpenRouter is the CDP fallback. Fail
        # fast with the exact remediation rather than dispatching workers
        # that can't carry the toolset.
        try:
            adapter = resolve_browser_adapter(getattr(args, "adapter", None))
        except BrowserAdapterUnavailable as exc:
            print(f"browser: {exc}", file=sys.stderr)
            if exc.fix:
                print(f"  enable it: {exc.fix}", file=sys.stderr)
            return 2
        specs = browser_swarm_specs(
            args.tasks,
            args.cwd,
            adapter=adapter,
            model=args.model,
            provider=args.provider,
            toolsets=args.toolsets,
            min_capability=args.min_capability,
            timeout_seconds=args.timeout_seconds,
            routing_policy=args.routing_policy,
            executable=args.executable,
        )
        goal = (
            args.tasks[0]
            if len(args.tasks) == 1
            else f"Browser-QA swarm ({len(args.tasks)} parallel workers)"
        )
        result = cli.Orchestrator(store).run(
            goal,
            specs=specs,
            lease_seconds=10,
            worker_mode=args.worker_mode,
            on_job_created=on_job_created,
            label=args.label,
        )
        return cli.finalize_cli_run(result)

    if args.command == "demo":
        result = cli.Orchestrator(store).run(args.goal)
        print_run_result(result.job.id, len(result.artifacts), result.summary_path)
        print("\n" + result.summary)
        return 0

    if args.command == "crash-demo":
        result = cli.Orchestrator(store).run_crash_recovery_demo(
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
                    line = f"{job.id}\t{job.status}\t{job.created_at}\t{project.name}"
                    if job.label:
                        line += f"\t{job.label}"
                    line += f"\t{job.goal}"
                    print(line)
            return 0
        for job in store.list_jobs():
            line = f"{job.id}\t{job.status}\t{job.created_at}"
            if job.label:
                line += f"\t{job.label}"
            line += f"\t{job.goal}"
            print(line)
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

    if args.command == "deltas":
        from puppetmaster import reads_log

        reads_log.record_read("deltas", caller="cli")
        job_id = args.job_id or require_latest_job_id(store)
        return run_deltas_follow(
            store,
            job_id,
            task_id=args.task_id,
            as_json=args.json,
            follow=args.follow,
            idle_timeout_seconds=args.follow_timeout_seconds,
            poll_interval_seconds=args.follow_poll_seconds,
        )

    if args.command == "eval":
        import dataclasses

        from puppetmaster.eval_harness import (
            adapter_apply_fn,
            builtin_cases,
            format_report,
            run_eval,
        )

        cases = builtin_cases()
        if args.cases:
            wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
            cases = [case for case in cases if case.name in wanted]
        if not cases:
            print("no matching eval cases", file=sys.stderr)
            return 1
        apply_fn = adapter_apply_fn(
            adapter=args.adapter, model=args.model, provider=args.provider,
            use_verify_loop=not args.no_verify_loop,
        )
        report = run_eval(cases, apply_fn, adapter=args.adapter, model=args.model)
        if args.json:
            payload = dataclasses.asdict(report)
            payload.update(
                passed=report.passed, total=report.total, pass_rate=report.pass_rate
            )
            print(json.dumps(payload, default=str, indent=2))
        else:
            print(format_report(report))
        return 0 if report.passed == report.total else 1

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
        return _run_dashboard_command(args, state_dir)

    if args.command == "await":
        return _run_await_command(args, store)

    if args.command == "artifacts":
        from puppetmaster import reads_log
        from puppetmaster.validation import compact_artifact_ref

        reads_log.record_read("artifacts", caller="cli")
        artifacts = store.list_artifacts(args.job_id)
        if getattr(args, "refs", False):
            payload = [compact_artifact_ref(artifact) for artifact in artifacts]
        else:
            payload = [artifact.__dict__ for artifact in artifacts]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "graph":
        from puppetmaster import reads_log

        reads_log.record_read("graph", caller="cli")
        print(json.dumps(store.job_graph(args.job_id), indent=2, default=str))
        return 0

    if args.command == "reset-subgraph":
        from puppetmaster.store import ActiveTaskLeaseError

        task_ids = list(getattr(args, "task_ids", None) or [])
        if not task_ids:
            print("error: pass at least one --task", file=sys.stderr)
            return 1
        try:
            reset = store.reset_subgraph(
                args.job_id,
                task_ids,
                include_descendants=not getattr(args, "no_descendants", False),
            )
        except ActiveTaskLeaseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        body = {
            "job_id": args.job_id,
            "reset_count": len(reset),
            "tasks": [
                {
                    "id": task.id,
                    "role": task.role,
                    "status": str(task.status),
                    "attempts": task.attempts,
                }
                for task in reset
            ],
            "superseded_artifact_ids": list(
                getattr(reset, "superseded_artifact_ids", []) or []
            ),
        }
        print(json.dumps(body, indent=2))
        return 0

    if args.command == "memory":
        records = store.list_memory()
        if getattr(args, "prune", False):
            scope = getattr(args, "scope", None)
            older_than_days = getattr(args, "older_than_days", None)
            if scope is None and older_than_days is None and not getattr(args, "yes", False):
                print(
                    "Refusing to prune all promoted memory without --yes. "
                    "Pass --scope, --older-than-days, or --yes to confirm."
                )
                return 2
            deleted = store.prune_memory(scope=scope, older_than_days=older_than_days)
            print(f"Pruned {deleted} memory record(s).")
            return 0
        if getattr(args, "json", False):
            print(json.dumps(records, indent=2, sort_keys=True))
            return 0
        if not records:
            print("No promoted memory.")
            return 0
        by_scope: dict[str, list[dict]] = {}
        for record in records:
            by_scope.setdefault(str(record.get("scope") or ""), []).append(record)
        for scope_name in sorted(by_scope):
            scoped = by_scope[scope_name]
            print(f"{scope_name}: {len(scoped)} record(s)")
            for record in scoped:
                statement = str(record.get("statement") or "")
                preview = statement[:120] + ("…" if len(statement) > 120 else "")
                confidence = record.get("confidence", 0.0)
                created_at = record.get("created_at") or "unknown"
                print(f"  [{confidence}] {preview} ({created_at})")
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
            result = cli.Orchestrator(store).run(
                source_job.goal,
                specs=config.workers,
                lease_seconds=config.lease_seconds,
                label=source_job.label,
            )
        else:
            result = cli.Orchestrator(store).run(source_job.goal, label=source_job.label)
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


if __name__ == "__main__":
    raise SystemExit(main())
