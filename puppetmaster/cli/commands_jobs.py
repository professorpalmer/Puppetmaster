from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
from puppetmaster.models import is_terminal_job_status
from puppetmaster.identity import make_ref
from puppetmaster.redaction import redact_secrets
from puppetmaster.readonly import ReadUnavailable, read_deadline
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

_LAST_GOAL_PREVIEW_CHARS = 160
_LAST_STORE_SCOPE = (
    "This id is from THIS state_dir only. Marionette app jobs often live in "
    "~/.pmharness; Cursor MCP last uses the workspace store. artifacts on a "
    "foreign job_id fail closed."
)


def last_job_bind(store, job, state_dir=None):
    """Structured bind for `last --json` / MCP last_job. Never a naked id."""
    job_id = str(getattr(job, "id", "") or "")
    goal = str(getattr(job, "goal", "") or "")
    root = Path(state_dir) if state_dir is not None else Path(getattr(store, "root", "") or "")
    try:
        root = root.resolve()
    except Exception:
        pass
    tasks = []
    artifacts = []
    try:
        tasks = list(store.list_tasks(job_id) or [])
    except Exception:
        tasks = []
    try:
        artifacts = list(store.list_artifacts(job_id) or [])
    except Exception:
        artifacts = []
    finding_count = 0
    for art in artifacts:
        kind = str(getattr(art, "type", "") or "").lower()
        if kind in {"finding", "risk", "decision"}:
            finding_count += 1
    return {
        "job_id": job_id,
        "status": str(getattr(job, "status", "") or ""),
        "state_dir": str(root),
        "goal_preview": goal[:_LAST_GOAL_PREVIEW_CHARS],
        "role_count": len(tasks),
        "finding_count": finding_count,
        "store_scope": _LAST_STORE_SCOPE,
    }


def read_job_state(store, job_id: str, *, timed_out: bool = False, stall_after_seconds=None,
                   reference=None) -> dict:
    """Return the canonical non-blocking state payload for one durable job."""
    if stall_after_seconds is None:
        _reap_quietly(store)
    else:
        _reap_quietly(store, stall_after_seconds=stall_after_seconds)
    job = store.get_job(job_id)
    snapshot = (
        store.status_snapshot(job_id, compact=True)
        if hasattr(store, "status_snapshot")
        else {}
    )
    return {
        "job_id": job_id,
        "status": str(job.status),
        "terminal": is_terminal_job_status(job.status),
        "timed_out": bool(timed_out),
        "completed_at": job.completed_at,
        "budget_policy": dataclasses.asdict(job.budget_policy) if job.budget_policy else None,
        "job_ref": (reference or getattr(store, "_legacy_read_ref", None) or store.job_ref(job_id)).as_dict(),
        "delivery": snapshot.get("delivery"),
        "progress": snapshot.get("progress"),
    }


def await_job_state(
    store,
    job_id: str,
    *,
    timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.25,
    stall_after_seconds=None,
) -> dict:
    """Block until ``job_id`` reaches a terminal state or the timeout elapses.

    Returns ``{status, terminal, timed_out, completed_at}``. ``timeout_seconds``
    of 0 blocks indefinitely (CLI/SDK path); a positive value bounds the wait
    (so the MCP path can return and be re-called). Uses the store's
    event-wait primitive between checks instead of busy-polling.
    """
    poll = max(0.05, poll_interval_seconds)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    cursor = 0
    reference = getattr(store, '_legacy_read_ref', None)
    if reference is None and getattr(store, '_incarnation', None) is not None:
        reference = make_ref(store.root, job_id, store._incarnation)
    state = dict(job_id=job_id, job_ref=reference.as_dict() if reference is not None else None, status='unavailable', terminal=False,
                 timed_out=False, completed_at=None)
    while True:
        try:
            with read_deadline(deadline):
                state = read_job_state(store, job_id, stall_after_seconds=stall_after_seconds,
                                       reference=reference)
        except ReadUnavailable:
            if deadline is not None and time.monotonic() >= deadline:
                return {**state, "timed_out": True}
            time.sleep(poll if deadline is None else min(poll, max(0, deadline - time.monotonic())))
            continue
        if state["terminal"]:
            return state
        if deadline is not None and time.monotonic() >= deadline:
            return {**state, "timed_out": True}
        block = poll if deadline is None else max(0, min(poll * 4, deadline - time.monotonic()))
        try:
            with read_deadline(deadline):
                events = store.wait_for_events(
                    job_id, since=cursor, timeout_seconds=block, poll_interval=poll)
        except ReadUnavailable:
            time.sleep(poll if deadline is None else min(poll, max(0, deadline - time.monotonic())))
            continue
        # Advance the cursor past the events we just observed. Without this the
        # cursor stayed at 0, so once any event existed wait_for_events returned
        # immediately every iteration and the loop hot-spun (re-reading the
        # whole event stream) until the job reached a terminal state.
        for event in events:
            event_id = event.get("id")
            if isinstance(event_id, int) and event_id > cursor:
                cursor = event_id

def _reap_quietly(store, *, stall_after_seconds=None) -> list[dict]:
    """Run the stalled-job reaper, swallowing any failure.

    Wired into read-side commands (status/jobs/wait) so a dead-but-"running"
    job is transitioned to stalled the next time anyone looks, without the user
    having to remember a separate command. Never raises into the caller."""
    try:
        from puppetmaster.liveness import reap_stalled_jobs

        kwargs = {} if stall_after_seconds is None else {'stall_after_seconds': stall_after_seconds}
        return reap_stalled_jobs(store, **kwargs)
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
    state = await_job_state(
        store, args.job_id, timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        stall_after_seconds=args.stall_after_seconds)
    status, terminal, timed_out = state['status'], state['terminal'], state['timed_out']

    summary = ""
    if args.summary and status in {"complete", "stalled"}:
        summary_path = store.job_dir(args.job_id) / "summaries" / "stitched.md"
        if summary_path.is_file():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            summary = Stitcher(store).preview(args.job_id)

    payload = state
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
    delivery = payload.get("delivery") or {}
    if timed_out or not delivery.get("successful", False):
        return 1
    return 0

AWAIT_SUMMARY_MODES = ("compact", "full", "none")


def await_summary_mode(explicit: Optional[str]) -> str:
    """``explicit`` when given (validated), else $PUPPETMASTER_AWAIT_SUMMARY, else compact."""
    if explicit is not None:
        mode = str(explicit).strip().lower()
        if mode not in AWAIT_SUMMARY_MODES:
            raise ValueError(f"summary must be one of {', '.join(AWAIT_SUMMARY_MODES)}")
        return mode
    env_mode = os.environ.get("PUPPETMASTER_AWAIT_SUMMARY", "").strip().lower()
    return env_mode if env_mode in AWAIT_SUMMARY_MODES else "compact"


def await_summary_body(store, job_id: str, state: dict, mode: str) -> dict:
    """Await response body: full stitched summary, compact digest + summary ref, or state only."""
    body: dict = {**state, "summary_mode": mode}
    if mode == "none":
        return body
    if not state["terminal"]:
        body["summary"] = ""
        return body
    summary_path: Optional[Path] = store.job_dir(job_id) / "summaries" / "stitched.md"
    if summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8")
    else:
        summary_path = None
        summary = Stitcher(store).preview(job_id)
    if mode == "full":
        body["summary"] = summary
        return body
    body["summary_ref"] = {
        "path": str(summary_path) if summary_path is not None else None,
        "chars": len(summary),
        "sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
    }
    try:
        body["digest"] = Stitcher(store).digest(job_id)
    except Exception as exc:
        body["digest_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return body


def _render_compact_await(body: dict) -> str:
    digest = body.get("digest") or {}
    lines = [f"job {body.get('job_id')} finished: {body.get('status')}"]
    counts = digest.get("counts") or {}
    if counts:
        lines.append("artifacts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    for row in digest.get("exceptions") or []:
        lines.append(f"exception: {row.get('role')} {row.get('result')}: {row.get('reason')}")
    for name in ("findings", "decisions", "risks", "conflicts"):
        for item in digest.get(name) or []:
            text = item.get("text") or item.get("reason") or ""
            lines.append(f"{name[:-1]}: {text}")
    if body.get("digest_error"):
        lines.append(f"digest unavailable: {body['digest_error']}")
    ref = body.get("summary_ref") or {}
    if ref.get("path"):
        lines.append(f"full summary: {ref['path']} ({ref.get('chars')} chars; --summary full prints it)")
    return "\n".join(lines)


def _run_await_command(args, store) -> int:
    try:
        mode = await_summary_mode(getattr(args, "summary", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    state = await_job_state(
        store,
        args.job_id,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    body = await_summary_body(store, args.job_id, state, mode)

    if args.json:
        print(json.dumps(body, indent=2, default=str))
    elif state["timed_out"]:
        print(f"timed out after {args.timeout_seconds}s; job {args.job_id} is {state['status']}")
    elif mode == "full":
        print(body.get("summary") or f"job {args.job_id} finished: {state['status']}")
    elif mode == "compact":
        print(_render_compact_await(body))
    else:
        print(f"job {args.job_id} finished: {state['status']}")
    if state["timed_out"]:
        return 1
    return 0 if (state.get("delivery") or {}).get("successful", False) else 1


def cmd_bounded_metadata(args, store, reference):
    from puppetmaster.models import to_jsonable
    if args.command == 'selected-economics':
        result = store.get_selected_economics(reference, expected_summary_revision=args.expected_summary_revision)
    else:
        kwargs = {name: getattr(args, name) for name in
                  ('cursor', 'limit', 'max_scan', 'max_bytes', 'status', 'origin', 'project_id', 'session_id')}
        kwargs['job_ref'] = reference
        if args.command == 'job-summary-changes':
            result = store.read_job_summary_changes(after_revision=args.after_revision, **kwargs)
        else:
            result = store.list_job_summaries(**kwargs)
    print(json.dumps(to_jsonable(result), ensure_ascii=True, separators=(',', ':')))
    return 0
