from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.mcp_registry import (
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    summarize as registry_summarize,
)
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.state import resolve_state_dir
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

    subcommands.add_parser("jobs", help="List known jobs.")
    subcommands.add_parser("last", help="Print the most recent job id.")

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


def _main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = resolve_state_dir(args.state_dir)
    store = create_store(args.backend, state_dir)
    on_job_created = early_job_printer if args.emit_job_id_early else None

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

    if args.command == "repair-codegraph":
        return _run_repair_codegraph(args)

    if args.command == "mcp":
        return _run_mcp_subcommand(args)

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
        for job in store.list_jobs():
            print(f"{job.id}\t{job.status}\t{job.created_at}\t{job.goal}")
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

