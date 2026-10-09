"""Versioned community observations (StrongOrc and peers).

These are not role-card capability numbers and never write
``capability_score``. The router joins them at route time on exact
identity and the effort the worker will run, and applies a
publication-style gate (mean delta plus CI).

The packaged StrongOrc ranking-v1 bundle applies when the user has no
observation store of their own.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from puppetmaster.swarm_reasoning import NO_EFFORT_ADAPTERS


WORKER_ROLES = frozenset({"implement", "explore", "review", "audit", "plan"})
REQUIRED_IDENTITY_FIELDS = (
    "registry_id",
    "adapter",
    "role",
    "effort",
    "provider",
    "track",
    "bank",
    "harness",
)
_STORE_NAME = "community-observations.json"
_PACKAGED_BUNDLE = Path(__file__).resolve().parent / "baselines" / "strongorc-ranking-v1.json"


@dataclass(frozen=True)
class CommunityObservation:
    """One published observation for one registry identity + role."""

    registry_id: str
    adapter: str
    role: str
    effort: str
    provider: str
    track: str
    bank: str
    harness: str
    pass_rate: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    sample_count: int = 0
    published: str = ""


def puppetmaster_home() -> Path:
    home = (os.environ.get("PUPPETMASTER_HOME") or "").strip()
    if home:
        return Path(home).expanduser()
    return Path.home() / ".puppetmaster"


def default_community_observations_path() -> Path:
    return puppetmaster_home() / _STORE_NAME


def default_example_bundle_path() -> Path:
    """The packaged StrongOrc ranking-v1 bundle (``models import-observations`` default)."""
    return _PACKAGED_BUNDLE


def _require_text(entry: dict, field: str) -> str:
    value = entry.get(field)
    text = str(value).strip() if value not in (None, "") else ""
    if not text:
        raise ValueError(f"observation missing {field}")
    return text


def _optional_unit_interval(value: Any, field: str) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"observation {field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observation {field} must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"observation {field} must be in 0..1")
    return number


def parse_observation_entry(entry: Any) -> CommunityObservation:
    if not isinstance(entry, dict):
        raise ValueError("observation entry must be an object")
    fields = {name: _require_text(entry, name) for name in REQUIRED_IDENTITY_FIELDS}
    track = fields["track"].strip().lower()
    role = fields["role"].strip()
    if track == "orchestrator" and role in WORKER_ROLES:
        raise ValueError(
            "orchestrator track cannot map to worker role "
            f"{role!r}; name an orchestrator role or drop the row"
        )
    pass_rate = _optional_unit_interval(entry.get("pass_rate"), "pass_rate")
    if pass_rate is None:
        raise ValueError("observation missing pass_rate")
    sample = entry.get("sample_count", 0)
    if isinstance(sample, bool) or not isinstance(sample, int):
        raise ValueError("observation sample_count must be an int")
    published = str(entry.get("published") or "").strip()
    return CommunityObservation(
        registry_id=fields["registry_id"],
        adapter=fields["adapter"],
        role=role,
        effort=fields["effort"],
        provider=fields["provider"],
        track=track,
        bank=fields["bank"],
        harness=fields["harness"],
        pass_rate=pass_rate,
        ci_low=_optional_unit_interval(entry.get("ci_low"), "ci_low"),
        ci_high=_optional_unit_interval(entry.get("ci_high"), "ci_high"),
        sample_count=sample,
        published=published,
    )


def parse_observation_bundle(bundle: Any) -> list[CommunityObservation]:
    if not isinstance(bundle, dict):
        raise ValueError("observation bundle must be an object")
    entries = bundle.get("entries")
    if not isinstance(entries, list):
        raise ValueError("observation bundle entries must be a list")
    return [parse_observation_entry(entry) for entry in entries]


def load_observation_bundle(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable observation bundle {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("observation bundle must be an object")
    return raw


def load_community_observations(
    path: Optional[Path] = None,
) -> list[CommunityObservation]:
    resolved = path or default_community_observations_path()
    if not resolved.is_file():
        if path is not None:
            return []
        resolved = _PACKAGED_BUNDLE
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        entries = raw.get("entries", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        return []
    out: list[CommunityObservation] = []
    for entry in entries:
        try:
            out.append(parse_observation_entry(entry))
        except ValueError:
            continue
    return out


def save_community_observations(
    observations: Iterable[CommunityObservation],
    path: Optional[Path] = None,
) -> Path:
    resolved = path or default_community_observations_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "not_ground_truth": True,
        "never_writes_capability_score": True,
        "entries": [asdict(item) for item in observations],
    }
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolved


def import_observations(
    bundle_path: Path,
    store_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> tuple[list[CommunityObservation], dict]:
    """Parse a bundle into the observation store. Never touches models.json."""
    bundle = load_observation_bundle(bundle_path)
    observations = parse_observation_bundle(bundle)
    report = {
        "count": len(observations),
        "dry_run": dry_run,
        "store": str(store_path or default_community_observations_path()),
    }
    if not dry_run:
        save_community_observations(observations, store_path)
    return observations, report


def match_observation(
    spec: Any,
    role: str,
    observations: Iterable[CommunityObservation],
    effort: str,
) -> Optional[CommunityObservation]:
    """Exact id + adapter + role + the effort the worker runs. No family-name smear.

    ``effort`` is the effective worker effort (caller pin, else the operator
    or swarm default). Catalog ``payload_defaults`` effort is not a pin and
    does not reach the worker, so it is not a join key. An adapter without
    per-run effort control runs no effort and joins no effort row.
    """
    wanted_id = str(getattr(spec, "id", "") or "")
    wanted_adapter = str(getattr(spec, "adapter", "") or "")
    wanted_effort = "" if wanted_adapter in NO_EFFORT_ADAPTERS else effort
    wanted_role = (role or "").strip()
    if not wanted_id or not wanted_adapter or not wanted_role:
        return None
    matched: list[CommunityObservation] = []
    for item in observations:
        if item.registry_id != wanted_id:
            continue
        if item.adapter != wanted_adapter:
            continue
        if item.role != wanted_role:
            continue
        if item.effort != wanted_effort:
            continue
        matched.append(item)
    if not matched:
        return None
    return matched[-1]


def _paired_delta_covers_zero(
    left: CommunityObservation,
    right: CommunityObservation,
) -> bool:
    if (
        left.ci_low is None
        or left.ci_high is None
        or right.ci_low is None
        or right.ci_high is None
    ):
        return False
    low = left.ci_low - right.ci_high
    high = left.ci_high - right.ci_low
    return low <= 0.0 <= high


def observation_beats(
    winner: CommunityObservation,
    other: CommunityObservation,
    *,
    min_delta: float = 0.10,
) -> bool:
    delta = winner.pass_rate - other.pass_rate
    if delta < min_delta:
        return False
    if _paired_delta_covers_zero(winner, other):
        return False
    return True


def community_gate(
    candidates: Iterable[Any],
    role: str,
    observations: Iterable[CommunityObservation],
    effort: str,
) -> Optional[Any]:
    """Return the unique gated winner among candidates, or None.

    A winner must beat at least one other observed candidate. A lone
    observation compares against nothing, so it does not override routing.
    """
    observed: list[tuple[Any, CommunityObservation]] = []
    materialized = list(observations)
    for spec in candidates:
        hit = match_observation(spec, role, materialized, effort)
        if hit is not None:
            observed.append((spec, hit))
    if len(observed) < 2:
        return None
    for spec, row in observed:
        if all(
            observation_beats(row, other)
            for other_spec, other in observed
            if other_spec is not spec
        ):
            return spec
    return None
