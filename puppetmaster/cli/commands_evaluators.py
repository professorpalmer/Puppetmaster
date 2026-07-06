from __future__ import annotations

import json
import os
from pathlib import Path

from puppetmaster.evaluators import active_evaluators, promote_evaluator
from puppetmaster.state import resolve_state_dir


def _state_dir_from_args(args) -> str:
    raw = getattr(args, "state_dir", None) or os.environ.get("PUPPETMASTER_STATE_DIR")
    return str(resolve_state_dir(raw))


def _default_anchor_set_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "docs" / "sample-anchor-set.json")


def _run_evaluators_list(args) -> int:
    state_dir = _state_dir_from_args(args)
    active = active_evaluators(state_dir)
    if getattr(args, "json", False):
        payload = {
            slot_id: {
                "version": spec.version,
                "role": spec.role,
                "instruction": spec.instruction,
                "criteria": spec.criteria,
            }
            for slot_id, spec in sorted(active.items())
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not active:
        print("No active evaluator slots.")
        return 0
    for slot_id, spec in sorted(active.items()):
        print(
            f"{slot_id} v{spec.version} role={spec.role} "
            f"instruction={spec.instruction!r}"
        )
    return 0


def _run_evaluators_promote(args) -> int:
    state_dir = _state_dir_from_args(args)
    anchor_path = getattr(args, "anchor_set", None) or _default_anchor_set_path()
    criteria_raw = getattr(args, "criteria_json", None) or "{}"
    try:
        criteria = json.loads(criteria_raw)
    except json.JSONDecodeError as exc:
        print(f"evaluators promote: invalid --criteria-json: {exc}")
        return 2
    if not isinstance(criteria, dict):
        print("evaluators promote: --criteria-json must be a JSON object")
        return 2
    try:
        spec = promote_evaluator(
            state_dir,
            args.slot_id,
            parent_version=getattr(args, "parent_version", None),
            instruction=args.instruction,
            criteria=criteria,
            anchor_path=anchor_path,
            min_pass_rate=float(getattr(args, "min_pass_rate", 1.0) or 1.0),
        )
    except ValueError as exc:
        print(f"evaluators promote: {exc}")
        return 1
    print(
        f"Promoted {spec.slot_id} to v{spec.version} "
        f"(role={spec.role}, parent={spec.parent_version})"
    )
    return 0


def _run_evaluators_subcommand(args) -> int:
    if args.evaluators_command == "list":
        return _run_evaluators_list(args)
    if args.evaluators_command == "promote":
        return _run_evaluators_promote(args)
    raise SystemExit(f"unknown evaluators subcommand: {args.evaluators_command}")
