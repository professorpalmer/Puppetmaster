"""``puppetmaster flow``: start a graph once, wake on done, gate or stuck.

The CLI and the MCP ``puppetmaster_flow`` tool share :func:`flow_action`, so
both surfaces return the same compact run summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Exit codes for scripts and background pilots.
EXIT_DONE, EXIT_PROBLEM, EXIT_USAGE, EXIT_RUNNING, EXIT_WAITING = 0, 1, 2, 3, 4


def add_flow_parser(subcommands: Any) -> None:
    flow = subcommands.add_parser(
        "flow",
        help=(
            "Run a flow graph: write one graph, start it, and be woken only when it "
            "finishes, gets stuck, or waits at a gate."
        ),
    )
    actions = flow.add_subparsers(dest="flow_action", required=True)

    def with_cwd(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--cwd", help="Workspace whose state holds the run (default: current directory).")
        return parser

    with_cwd(actions.add_parser("validate", help="Check a graph without running it.")).add_argument("graph")
    with_cwd(actions.add_parser("save", help="Save a graph under its id for reuse.")).add_argument("graph")

    run = with_cwd(actions.add_parser("run", help="Start a run in the background (or --foreground)."))
    run.add_argument("graph", help="Graph JSON file, saved graph id, or - for stdin.")
    run.add_argument("--input", default="", help="The run's {{input}}.")
    run.add_argument("--input-file", help="Read {{input}} from a file.")
    run.add_argument("--continue", dest="continue_from",
                     help="Continue a prior run: its nodes resume their own sessions with this input.")
    run.add_argument("--wait", action="store_true", help="Block until the run wakes the pilot.")
    run.add_argument("--timeout", type=float, default=0.0, help="With --wait: give up after N seconds.")
    run.add_argument("--foreground", action="store_true", help="Walk in this process instead of a detached one.")
    run.add_argument("--worker-mode", choices=["subprocess", "inline"], default="subprocess")

    status = with_cwd(actions.add_parser("status", help="Compact run summary."))
    status.add_argument("run_id")
    status.add_argument("--since", type=int, default=0, help="Only steps after this index.")

    wait = with_cwd(actions.add_parser("wait", help="Block until done, failed, stuck, waiting or interrupted."))
    wait.add_argument("run_id")
    wait.add_argument("--timeout", type=float, default=0.0)
    wait.add_argument("--since", type=int, default=0)

    resume = with_cwd(actions.add_parser("resume", help="Walk a run: answer a gate, resume, or restart."))
    resume.add_argument("run_id")
    resume.add_argument("--answer", help="Answer for the gate the run waits at.")
    resume.add_argument("--restart", action="store_true", help="Reopen a failed, stuck or stopped run.")
    resume.add_argument("--reset-loops", action="store_true", help="With --restart: reset loop budgets.")
    resume.add_argument("--extra-steps", type=int, default=0, help="With --restart: allow N more steps.")
    resume.add_argument("--background", action="store_true", help="Walk in a detached process.")

    with_cwd(actions.add_parser("stop", help="Stop a run and cut what it has in flight.")).add_argument("run_id")
    cut = with_cwd(actions.add_parser("cut", help="Fail the node in flight; its fail edges take over."))
    cut.add_argument("run_id")
    cut.add_argument("--reason", default="")

    listing = with_cwd(actions.add_parser("list", help="Recent runs."))
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--all", action="store_true", help="Include map item runs.")

    with_cwd(actions.add_parser("show", help="Full run record.")).add_argument("run_id")


def run_flow_command(args: argparse.Namespace, state_dir: Path) -> int:
    params = {key: value for key, value in vars(args).items() if value is not None}
    graph = params.get("graph")
    if graph == "-":
        params["graph"] = json.loads(sys.stdin.read())
    if params.get("input_file"):
        params["input"] = Path(params["input_file"]).read_text(encoding="utf-8")
    try:
        body, code = flow_action(state_dir, args.flow_action, params, backend=args.backend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(json.dumps(body, indent=2, default=str))
    return code


def flow_action(state_dir: Path, action: str, params: dict, *, backend: str = "sqlite",
                max_block_seconds: Optional[float] = None) -> tuple[dict, int]:
    """Run one flow action; returns ``(body, exit_code)``. Raises ValueError on bad input."""
    from puppetmaster import flow
    from puppetmaster.worker_fence import is_worker_process, nested_starts_allowed

    state_dir = Path(state_dir)
    if action in ("run", "resume") and is_worker_process() and not nested_starts_allowed():
        raise ValueError(
            "nested flow start refused (PUPPETMASTER_WORKER=1): this process is already a "
            "Puppetmaster worker. Override: PUPPETMASTER_ALLOW_NESTED=1"
        )
    base = params.get("cwd")
    if action == "validate":
        problems = flow.validate_graph(flow.load_graph(state_dir, params["graph"], base=base))
        return {"valid": not problems, "problems": problems}, EXIT_DONE if not problems else EXIT_USAGE
    if action == "save":
        path = flow.save_graph(state_dir, flow.load_graph(state_dir, params["graph"], base=base))
        return {"saved": str(path)}, EXIT_DONE
    if action == "list":
        return {"runs": flow.list_runs(state_dir, int(params.get("limit") or 20),
                                       include_children=bool(params.get("all")))}, EXIT_DONE
    if action == "run":
        graph = flow.load_graph(state_dir, params["graph"], base=base)
        run = flow.new_run(state_dir, graph, str(params.get("input") or ""),
                           continue_from=params.get("continue_from"), cwd=params.get("cwd"),
                           backend=backend, worker_mode=str(params.get("worker_mode") or "subprocess"))
        if params.get("foreground"):
            run = flow.walk(state_dir, run.run_id)
        else:
            flow.spawn_background_walk(state_dir, run.run_id)
            if params.get("wait"):
                run = flow.wait_for_event(state_dir, run.run_id,
                                          _block(params.get("timeout"), max_block_seconds))
            else:
                run = flow.load_run(state_dir, run.run_id)
        return _summary(state_dir, run), _exit(run.status)
    run_id = str(params.get("run_id") or "")
    if action == "status":
        run = flow.refresh_liveness(state_dir, run_id)
        return _summary(state_dir, run, since=int(params.get("since") or 0)), _exit(run.status)
    if action == "show":
        return flow.load_run(state_dir, run_id).to_dict(), EXIT_DONE
    if action == "wait":
        run = flow.wait_for_event(state_dir, run_id, _block(params.get("timeout"), max_block_seconds))
        return _summary(state_dir, run, since=int(params.get("since") or 0)), _exit(run.status)
    if action == "resume":
        if params.get("background"):
            run, should_walk = flow.prepare_resume(
                state_dir, run_id, answer=params.get("answer"), restart=bool(params.get("restart")),
                reset_loops=bool(params.get("reset_loops")), extra_steps=int(params.get("extra_steps") or 0))
            if should_walk:
                flow.spawn_background_walk(state_dir, run_id)
        else:
            run = flow.walk(state_dir, run_id, answer=params.get("answer"),
                            restart=bool(params.get("restart")),
                            reset_loops=bool(params.get("reset_loops")),
                            extra_steps=int(params.get("extra_steps") or 0))
        return _summary(state_dir, run), _exit(run.status)
    if action == "stop":
        run = flow.request_stop(state_dir, run_id)
        return _summary(state_dir, run), _exit(run.status)
    if action == "cut":
        run = flow.cut_node(state_dir, run_id, str(params.get("reason") or ""))
        return _summary(state_dir, run), _exit(run.status)
    raise ValueError(f"unknown flow action {action!r}")


def _summary(state_dir: Path, run: Any, since: int = 0) -> dict:
    from puppetmaster import flow

    body = flow.run_summary(run, since=since)
    body["state_dir"] = str(state_dir)
    return body


def _block(requested: Any, cap: Optional[float]) -> float:
    seconds = float(requested or 0)
    if cap is not None and cap > 0 and (seconds <= 0 or seconds > cap):
        return cap
    return seconds


def _exit(status: str) -> int:
    if status == "done":
        return EXIT_DONE
    if status == "waiting":
        return EXIT_WAITING
    if status == "running":
        return EXIT_RUNNING
    return EXIT_PROBLEM
