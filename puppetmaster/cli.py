from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.orchestrator import Orchestrator
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
        default=".puppetmaster",
        help="Directory for jobs, streams, artifacts, locks, and promoted memory.",
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
    store = create_store(args.backend, Path(args.state_dir))
    on_job_created = early_job_printer if args.emit_job_id_early else None

    if args.command == "init":
        store.init()
        print(f"Initialized Puppetmaster state at {store.root}")
        return 0

    if args.command == "doctor":
        for check in run_doctor(Path.cwd(), Path(args.state_dir)):
            print(f"{check.status:8} {check.name:16} {check.detail}")
        return 0

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
        feed = artifact_feed(store, job_id, limit=args.limit)
        if args.json:
            print(json.dumps(feed, indent=2, default=str))
        else:
            for item in feed:
                artifact = item["artifact"]
                print(
                    f"{item['at']}\t{artifact['type']}\t{artifact['id']}\t"
                    f"task={artifact['task_id']}\tconfidence={artifact['confidence']}"
                )
                print(f"  {artifact_headline(artifact)}")
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
    artifacts = {artifact.id: artifact.__dict__ for artifact in store.list_artifacts(job_id)}
    items = []
    seen = set()
    for event in store.read_events(job_id):
        if event.get("event") != "artifact.saved":
            continue
        artifact_id = event.get("payload", {}).get("artifact_id")
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact_id in seen:
            continue
        seen.add(artifact_id)
        items.append({"at": event["at"], "event": event["event"], "artifact": artifact})
    if limit is not None:
        return items[-limit:]
    return items


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

