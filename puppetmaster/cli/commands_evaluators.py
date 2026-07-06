from __future__ import annotations

import json
import os
from pathlib import Path

from puppetmaster.evaluators import (
    active_evaluators,
    clear_drafts,
    load_drafts,
    merge_draft_notes_into_criteria,
    promote_evaluator,
)
from puppetmaster.state import resolve_state_dir


def _state_dir_from_args(args, state_dir=None) -> str:
    if state_dir is not None:
        return str(state_dir)
    raw = getattr(args, "state_dir", None) or os.environ.get("PUPPETMASTER_STATE_DIR")
    return str(resolve_state_dir(raw))


def _default_anchor_set_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "docs" / "sample-anchor-set.json")


def _run_evaluators_list(args, state_dir=None) -> int:
    state_dir = _state_dir_from_args(args, state_dir)
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


def _run_evaluators_promote(args, state_dir=None) -> int:
    state_dir = _state_dir_from_args(args, state_dir)
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
    folded = 0
    if getattr(args, "from_drafts", False):
        criteria, folded = merge_draft_notes_into_criteria(state_dir, args.slot_id, criteria)
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
    if getattr(args, "from_drafts", False):
        cleared = clear_drafts(state_dir, args.slot_id)
        print(
            f"Promoted {spec.slot_id} to v{spec.version} "
            f"(role={spec.role}, parent={spec.parent_version}); "
            f"folded {folded} draft note(s), cleared {cleared} draft(s)."
        )
        return 0
    print(
        f"Promoted {spec.slot_id} to v{spec.version} "
        f"(role={spec.role}, parent={spec.parent_version})"
    )
    return 0


def _run_evaluators_drafts(args, state_dir=None) -> int:
    state_dir = _state_dir_from_args(args, state_dir)
    clear_slot = getattr(args, "clear", None)
    if clear_slot:
        removed = clear_drafts(state_dir, clear_slot)
        print(f"Cleared {removed} draft(s) for {clear_slot}.")
        return 0

    slot_filter = getattr(args, "slot_id", None)
    drafts = load_drafts(state_dir, slot_filter)
    if getattr(args, "json", False):
        print(json.dumps(drafts, indent=2, sort_keys=True))
        return 0
    if not drafts:
        print("No draft criteria recorded.")
        return 0

    if slot_filter:
        grouped = {slot_filter: drafts}
    else:
        grouped: dict[str, list] = {}
        for draft in drafts:
            slot = str(draft.get("slot_id") or "")
            grouped.setdefault(slot, []).append(draft)

    for slot_id in sorted(grouped):
        entries = grouped[slot_id]
        print(f"{slot_id}: {len(entries)} draft(s)")
        for entry in entries:
            severity = entry.get("severity") or "none"
            reasons = entry.get("reasons") or []
            reason_text = "; ".join(str(reason) for reason in reasons)
            job_id = entry.get("source_job_id") or ""
            print(f"  [{severity}] {reason_text} (job {job_id})")
    return 0


def _run_evaluators_epoch(args, state_dir=None) -> int:
    from puppetmaster.evaluators import evaluator_epoch_for_job
    from puppetmaster.state import find_state_dir_for_job
    from puppetmaster.store import SwarmStore

    job_id = args.job_id
    state_dir = Path(_state_dir_from_args(args, state_dir))
    explicit = bool(getattr(args, "state_dir", None) or os.environ.get("PUPPETMASTER_STATE_DIR"))
    if not (state_dir / "jobs" / job_id).is_dir() and not explicit:
        found = find_state_dir_for_job(job_id)
        if found is not None:
            state_dir = found
    if not (state_dir / "jobs" / job_id).is_dir():
        print("No evaluator epoch recorded.")
        return 0

    store = SwarmStore(state_dir)
    store.init()
    epoch = evaluator_epoch_for_job(store, job_id)
    evaluators = epoch.get("evaluators") or []
    if not evaluators:
        print("No evaluator epoch recorded.")
        return 0
    for entry in evaluators:
        if not isinstance(entry, dict):
            continue
        criteria = entry.get("criteria") or {}
        count = len(criteria) if isinstance(criteria, dict) else 0
        slot_id = str(entry.get("slot_id") or "")
        version = entry.get("version") or 0
        role = str(entry.get("role") or "")
        print(f"{slot_id} v{version} role={role} criteria={count}")
    return 0


def _run_evaluators_subcommand(args, state_dir=None) -> int:
    if args.evaluators_command == "list":
        return _run_evaluators_list(args, state_dir=state_dir)
    if args.evaluators_command == "promote":
        return _run_evaluators_promote(args, state_dir=state_dir)
    if args.evaluators_command == "drafts":
        return _run_evaluators_drafts(args, state_dir=state_dir)
    if args.evaluators_command == "epoch":
        return _run_evaluators_epoch(args, state_dir=state_dir)
    raise SystemExit(f"unknown evaluators subcommand: {args.evaluators_command}")
