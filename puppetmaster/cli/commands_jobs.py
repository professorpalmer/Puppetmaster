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

    terminal = {
        JobStatus.COMPLETE,
        JobStatus.FAILED,
        JobStatus.STALLED,
        JobStatus.CANCELLED,
    }
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

    import puppetmaster.cli as cli

    all_projects = getattr(args, "all_projects", False)
    active_root = _resolved_store_root(store)
    reaped: list[dict] = []
    protected_active = False
    for target in cli._gc_target_stores(args, store):
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
