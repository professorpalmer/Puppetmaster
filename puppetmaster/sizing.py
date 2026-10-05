"""Sizing gate: stay solo until a model's solo falloff, then hand off.

Below the falloff one session that holds the whole problem beats any fan-out;
past it, the remaining independent units go to workers. The falloff tracks
irreducible independent work, not a unit count, so it is read from the plan
and measured progress, never from prompt wording.

The decision is pure. Calibration comes from a JSON file the benchmark writes
(``PUPPETMASTER_SIZING_CALIBRATION`` or ``~/.puppetmaster/sizing-calibration.json``),
keyed by model; the built-in default is provisional.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

STAY_SOLO = "stay_solo"
HANDOFF = "handoff"
DELEGATE_UPFRONT = "delegate_upfront"

DONE_STATUSES = frozenset(("done", "completed", "complete"))
PARALLEL_TAG = "[parallel]"

# Provisional until the falloff sweep writes a calibration file: sol61 sizes
# 1-4 (solo wins) and the Sep 16 n=1 runs (16 coupled regions capped solo at
# 8/16 in 30 minutes).
DEFAULT_CALIBRATION: dict[str, Any] = {
    "solo_horizon_s": 1200.0,
    "solo_context_frac": 0.6,
    "min_independent_units": 4,
    "upfront_units": 16,
    "handoff_overhead_s": 90.0,
    "late_fraction": 0.8,
    "source": "built-in provisional default",
}


@dataclass(frozen=True)
class Unit:
    id: str
    status: str = "pending"
    independent: bool = False

    @property
    def done(self) -> bool:
        return self.status.strip().lower() in DONE_STATUSES


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    total_units: int
    done_units: int
    remaining_independent: int
    projected_solo_s: Optional[float] = None
    projected_parallel_s: Optional[float] = None
    projected_context_frac: Optional[float] = None
    calibration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_plan(raw: Any) -> list[Unit]:
    """Units from a plan list; accepts Marionette, Codex update_plan and TodoWrite shapes."""
    items = raw.get("plan") or raw.get("todos") or raw.get("units") if isinstance(raw, dict) else raw
    units = []
    for index, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("content") or item.get("step") or item.get("title") or "")
        units.append(Unit(
            id=str(item.get("id") or text or index),
            status=str(item.get("status") or "pending"),
            # Host plan tools have no independence field; pilots tag the text.
            independent=item.get("independent") is True or text.lstrip().lower().startswith(PARALLEL_TAG),
        ))
    return units


def _model_key(model: str) -> str:
    key = (model or "").strip().lower()
    key = key.rsplit("/", 1)[-1]
    return key.replace(".", "-").replace("_", "-")


def calibration_path(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    explicit = (env.get("PUPPETMASTER_SIZING_CALIBRATION") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    from puppetmaster.community_observations import puppetmaster_home

    return puppetmaster_home() / "sizing-calibration.json"


def load_calibration(model: str, *, path: Optional[Path] = None,
                     env: Optional[Mapping[str, str]] = None) -> dict:
    """Merged calibration for ``model``: built-in default, file default, file model entry."""
    merged = dict(DEFAULT_CALIBRATION)
    target = path if path is not None else calibration_path(env)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return merged
    if not isinstance(data, dict):
        return merged
    if isinstance(data.get("default"), dict):
        merged.update(data["default"])
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    wanted = _model_key(model)
    for name, entry in models.items():
        if _model_key(name) == wanted and isinstance(entry, dict):
            merged.update(entry)
            break
    return merged


def decide(units: Sequence[Unit], *, elapsed_s: float, context_frac: float = 0.0,
           calibration: Optional[Mapping[str, Any]] = None,
           handed_off: bool = False) -> Decision:
    """Stay solo, hand the remaining independent units off, or delegate upfront."""
    cal = dict(DEFAULT_CALIBRATION if calibration is None else calibration)
    total = len(units)
    done = sum(unit.done for unit in units)
    remaining = [unit for unit in units if not unit.done]
    independent = sum(unit.independent for unit in remaining)

    def result(action: str, reason: str, **extra: Any) -> Decision:
        return Decision(action, reason, total, done, independent, calibration=cal, **extra)

    if handed_off:
        return result(STAY_SOLO, "already handed off this turn")
    if independent < int(cal["min_independent_units"]):
        return result(STAY_SOLO, f"{independent} independent units remain; "
                                 f"fewer than {int(cal['min_independent_units'])} never fan out")
    if done == 0:
        if independent >= int(cal["upfront_units"]):
            return result(DELEGATE_UPFRONT, f"{independent} independent units before any work "
                                            f"(upfront bar {int(cal['upfront_units'])})")
        return result(STAY_SOLO, "no unit finished yet; measuring solo pace")
    if done / total >= float(cal["late_fraction"]):
        return result(STAY_SOLO, f"{done}/{total} done; too late for a handoff to pay")

    per_unit = max(float(elapsed_s), 0.0) / done
    remaining_solo = per_unit * len(remaining)
    projected_solo = float(elapsed_s) + remaining_solo
    # Coupled units stay with the pilot, so the parallel path still pays for them.
    coupled = len(remaining) - independent
    remaining_parallel = per_unit * (coupled + 1) + float(cal["handoff_overhead_s"])
    projected_context = (float(context_frac) / done) * total if context_frac > 0 else None
    numbers = dict(projected_solo_s=round(projected_solo, 1),
                   projected_parallel_s=round(float(elapsed_s) + remaining_parallel, 1),
                   projected_context_frac=None if projected_context is None else round(projected_context, 3))
    over_time = projected_solo > float(cal["solo_horizon_s"])
    over_context = projected_context is not None and projected_context > float(cal["solo_context_frac"])
    if not (over_time or over_context):
        return result(STAY_SOLO, "solo projected to finish inside its horizon", **numbers)
    if remaining_parallel >= remaining_solo:
        return result(STAY_SOLO, "projected overrun, but a handoff would not finish sooner", **numbers)
    why = "time" if over_time else "context"
    return result(HANDOFF, f"projected solo overrun ({why}); {independent} independent units "
                           f"can run in parallel", **numbers)


def advice(decision: Decision) -> str:
    """The steering text a host shows the pilot, or '' when it should keep going solo."""
    if decision.action == DELEGATE_UPFRONT:
        return (f"SIZING: {decision.remaining_independent} independent units before any work. "
                "Fan them out now: one Puppetmaster flow with a map node over the units "
                "(per-item check and judge), and keep coupled work yourself.")
    if decision.action == HANDOFF:
        solo = decision.projected_solo_s or 0.0
        parallel = decision.projected_parallel_s or 0.0
        return (f"SIZING: solo is projected to overrun ({decision.done_units}/{decision.total_units} "
                f"done; ~{solo / 60:.0f} min solo vs ~{parallel / 60:.0f} min with a handoff). "
                f"Finish the unit in hand, then hand the {decision.remaining_independent} "
                "remaining independent units to one Puppetmaster flow with a map node "
                "(per-item check and judge), and integrate when it wakes you.")
    return ""
