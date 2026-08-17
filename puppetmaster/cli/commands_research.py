"""CLI handlers for `python -m puppetmaster research ...`."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from puppetmaster.research import (
    DEFAULT_LAB_LABEL,
    TOY_HARNESS_ID,
    ResearchLab,
    analyze_swarm_from_artifacts,
)
from puppetmaster.store_factory import create_store


def _parse_config(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid --config JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("error: --config must be a JSON object")
    return value


def _lab_from_args(args) -> ResearchLab:
    store = create_store(getattr(args, "backend", "sqlite") or "sqlite", args.state_dir)
    store.init()
    worker_id = getattr(args, "worker_id", None) or None
    return ResearchLab(store=store, worker_id=worker_id) if worker_id else ResearchLab(store=store)


def _print_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    print(payload)


def _run_research_subcommand(args, state_dir: Path) -> int:
    """Dispatch nested `research` verbs."""
    # Ensure global --state-dir is visible to helpers that read args.state_dir.
    if not getattr(args, "state_dir", None):
        args.state_dir = str(state_dir)
    else:
        args.state_dir = str(args.state_dir)

    sub = args.research_command
    as_json = bool(getattr(args, "json", False))

    if sub == "init":
        lab = _lab_from_args(args)
        job = lab.init_lab(
            args.goal,
            label=getattr(args, "label", None) or DEFAULT_LAB_LABEL,
        )
        _print_payload(
            {"job_id": job.id, "label": job.label, "goal": job.goal, "status": str(job.status)},
            as_json=as_json,
        )
        return 0

    if sub == "announce":
        lab = _lab_from_args(args)
        art = lab.announce(args.job_id, args.message)
        _print_payload({"artifact_id": art.id, "decision": art.payload.get("decision")}, as_json=as_json)
        return 0

    if sub == "think":
        lab = _lab_from_args(args)
        view = analyze_swarm_from_artifacts(lab.store, args.job_id)
        if as_json:
            print(json.dumps(view, indent=2, sort_keys=True, default=str))
            return 0
        status = view.get("status") or {}
        print(f"job {status.get('job_id')} status={status.get('status')} "
              f"open_claims={status.get('open_claims')} artifacts={status.get('artifacts')}")
        print("hypotheses:")
        for item in view.get("hypotheses") or []:
            print(f"  - {item}")
        print("insights:")
        for item in view.get("insights") or []:
            print(f"  - {item}")
        print("leaderboard:")
        for row in view.get("leaderboard") or []:
            print(
                f"  - {row.get('bits_per_byte')}  {row.get('hypothesis')}  ({row.get('artifact_id')})"
            )
        return 0

    if sub == "claim":
        lab = _lab_from_args(args)
        config = _parse_config(getattr(args, "config", None))
        harness_id = getattr(args, "harness", None) or TOY_HARNESS_ID
        claimed = lab.claim_experiment(
            args.job_id,
            args.hypothesis,
            harness_id,
            config,
        )
        if claimed is None:
            print("claim refused: fingerprint already held by another worker", file=sys.stderr)
            return 1
        metrics = None
        if getattr(args, "run", False):
            metrics = dict(lab.run_claimed(args.job_id, task_id=claimed.id))
        payload = {
            "task_id": claimed.id,
            "fingerprint": claimed.payload.get("fingerprint"),
            "harness_id": claimed.payload.get("harness_id"),
            "config": claimed.payload.get("config"),
        }
        if metrics is not None:
            payload["metrics"] = metrics
        _print_payload(payload, as_json=as_json)
        return 0

    if sub == "publish":
        lab = _lab_from_args(args)
        metrics = _parse_config(getattr(args, "metrics", None))
        if not metrics and getattr(args, "run", False):
            metrics = dict(lab.run_claimed(args.job_id, task_id=args.task_id))
        if not metrics:
            print("error: provide --metrics JSON or pass --run", file=sys.stderr)
            return 2
        art = lab.publish_result(
            args.job_id,
            metrics,
            task_id=args.task_id,
            keep=not getattr(args, "no_keep", False),
        )
        _print_payload(
            {
                "artifact_id": art.id,
                "bits_per_byte": art.payload.get("bits_per_byte"),
                "keep": art.payload.get("keep"),
            },
            as_json=as_json,
        )
        return 0

    if sub == "insight":
        lab = _lab_from_args(args)
        art = lab.post_insight(args.job_id, args.insight, why=getattr(args, "why", "") or "")
        _print_payload({"artifact_id": art.id, "claim": art.payload.get("claim")}, as_json=as_json)
        return 0

    if sub == "hypothesis":
        lab = _lab_from_args(args)
        config = _parse_config(getattr(args, "config", None)) or None
        art = lab.publish_hypothesis(
            args.job_id,
            args.hypothesis,
            why=getattr(args, "why", "") or "",
            config=config,
            harness_id=getattr(args, "harness", None),
        )
        _print_payload(
            {"artifact_id": art.id, "decision": art.payload.get("decision")},
            as_json=as_json,
        )
        return 0

    if sub == "verify":
        lab = _lab_from_args(args)
        art = lab.verify_claim(args.job_id, args.artifact_id)
        _print_payload(
            {
                "artifact_id": art.id,
                "result": art.payload.get("result"),
                "passed": art.payload.get("passed"),
                "expected": art.payload.get("expected"),
                "actual": art.payload.get("actual"),
            },
            as_json=as_json,
        )
        return 0 if art.payload.get("passed") else 1

    if sub == "status":
        lab = _lab_from_args(args)
        print(json.dumps(lab.status(args.job_id), indent=2, sort_keys=True, default=str))
        return 0

    if sub == "leaderboard":
        lab = _lab_from_args(args)
        board = lab.leaderboard(args.job_id)
        rows = [
            {
                "rank": index,
                "artifact_id": art.id,
                "bits_per_byte": art.payload.get("bits_per_byte"),
                "hypothesis": art.payload.get("hypothesis"),
                "config": art.payload.get("config"),
            }
            for index, art in enumerate(board, start=1)
        ]
        if as_json:
            print(json.dumps(rows, indent=2, sort_keys=True, default=str))
            return 0
        if not rows:
            print("leaderboard empty")
            return 0
        for row in rows:
            print(
                f"{row['rank']}. {row['bits_per_byte']}  {row['hypothesis']}  "
                f"{json.dumps(row['config'], sort_keys=True)}  ({row['artifact_id']})"
            )
        return 0

    if sub == "demo":
        lab = _lab_from_args(args)
        brief = getattr(args, "brief_path", None)
        result = lab.run_demo(
            goal=getattr(args, "goal", None),
            brief_path=Path(brief) if brief else None,
            label=getattr(args, "label", None) or DEFAULT_LAB_LABEL,
        )
        if as_json:
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        else:
            print(f"demo complete job_id={result['job_id']}")
            print(f"brief={result['brief_path']}")
            for row in result.get("leaderboard") or []:
                print(
                    f"  {row.get('bits_per_byte')}  {row.get('hypothesis')}  "
                    f"{json.dumps(row.get('config') or {}, sort_keys=True)}"
                )
        return 0

    raise SystemExit(f"unknown research subcommand: {sub}")
