"""Role scorecards and community-baseline import for the model registry.

``capability_score`` on :class:`~puppetmaster.model_registry.ModelSpec` stays
the explicit manual fallback. A role card's ``capability`` (int 0-100) overrides
that fallback for one :class:`~puppetmaster.router.TaskSignals` role only.
Empty cards are bit-identical to today's routing.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional


_PACKAGED_BASELINE = Path("baselines") / "role-scorecards-v1.json"
_DOCS_BASELINE = Path("docs") / "baselines" / "role-scorecards-v1.json"
_PROVENANCE_SOURCE_COMMUNITY = "community_baseline"
ROLE_CARD_SCALE = "puppetmaster-capability-0-100"
MIN_QUALIFIED_SAMPLE_COUNT = 5
MAX_CALIBRATION_AGE_DAYS = 180


def default_community_baseline_path() -> Path:
    """Resolve the shipped community baseline from the wheel, repo, or CWD."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / _PACKAGED_BASELINE,
        here.parents[1] / _DOCS_BASELINE,
        Path.cwd() / _DOCS_BASELINE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _qualified_card(card: Any, *, today: Optional[date] = None) -> bool:
    """Whether a role card is current, reproducible routing authority.

    Incomplete cards remain useful evidence in the registry and audit output,
    but they cannot override the user's manual ``capability_score``.  The
    qualification fields are deliberately card-local: a model-wide provenance
    blob cannot make one role's stale or undersampled measurement authoritative.
    """
    if not isinstance(card, dict):
        return False
    sample_count = card.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < MIN_QUALIFIED_SAMPLE_COUNT
    ):
        return False
    if card.get("scale") != ROLE_CARD_SCALE:
        return False
    scale_version = card.get("scale_version")
    if not isinstance(scale_version, str) or not scale_version.strip():
        return False
    provenance = card.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        return False
    source = provenance.get("source")
    version = provenance.get("version")
    if not isinstance(source, str) or not source.strip():
        return False
    if not isinstance(version, str) or not version.strip():
        return False
    calibrated = card.get("last_calibrated")
    if not isinstance(calibrated, str):
        return False
    try:
        calibrated_on = date.fromisoformat(calibrated)
    except ValueError:
        return False
    reference = today or date.today()
    age_days = (reference - calibrated_on).days
    return 0 <= age_days <= MAX_CALIBRATION_AGE_DAYS


def card_capability(spec: Any, role: str) -> Optional[int]:
    """Return a qualified role-card capability, otherwise ``None``."""
    cards = getattr(spec, "role_scorecards", None) or {}
    if not isinstance(cards, dict):
        return None
    card = cards.get(role or "")
    if not _qualified_card(card):
        return None
    cap = card.get("capability")
    if isinstance(cap, bool) or not isinstance(cap, int):
        return None
    if 0 <= cap <= 100:
        return cap
    return None


def effective_capability_score(spec: Any, role: str) -> int:
    """Role-card capability when present and valid; else ``spec.capability_score``."""
    override = card_capability(spec, role)
    if override is not None:
        return override
    return int(spec.capability_score)


def role_card_override_note(spec: Any, role: str) -> str:
    """Reason suffix when a role card overrode the manual capability score."""
    override = card_capability(spec, role)
    if override is None:
        return ""
    manual = int(spec.capability_score)
    if override == manual:
        return ""
    return (
        f"; role={role} card capability {override} (manual fallback {manual})"
    )


def load_community_baseline(path: Path) -> dict:
    """Load and minimally validate a community scorecard bundle (JSON only)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"community baseline not found: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"community baseline is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("community baseline must be a JSON object")
    bundle_id = raw.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise ValueError("community baseline missing bundle_id")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("community baseline missing version")
    if raw.get("adapter_scoped") is not True:
        raise ValueError("community baseline adapter_scoped must be true")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ValueError("community baseline entries must be a list")
    return raw


def _identity_hit(spec: Any, entry: dict) -> bool:
    entry_id = entry.get("id")
    entry_name = entry.get("adapter_model_name")
    if isinstance(entry_id, str) and entry_id and spec.id == entry_id:
        return True
    if (
        isinstance(entry_name, str)
        and entry_name
        and spec.adapter_model_name == entry_name
    ):
        return True
    return False


def _copy_card(card: Any) -> Optional[dict]:
    if not isinstance(card, dict):
        return None
    return dict(card)


def _merge_role_cards(
    local: dict,
    incoming: dict,
    *,
    replace_cards: bool,
) -> "tuple[dict, int]":
    merged: dict = {}
    for role, card in (local or {}).items():
        copied = _copy_card(card)
        if copied is not None:
            merged[str(role)] = copied
    added = 0
    for role, card in (incoming or {}).items():
        copied = _copy_card(card)
        if copied is None:
            continue
        key = str(role)
        if key in merged and not replace_cards:
            continue
        if key not in merged or replace_cards:
            added += 1
        merged[key] = copied
    return merged, added


def _stamp_provenance(bundle: dict, entry: dict) -> dict:
    entry_prov = entry.get("score_provenance")
    notes = ""
    sample_count: Any = None
    if isinstance(entry_prov, dict):
        notes = entry_prov.get("notes") or ""
        sample_count = entry_prov.get("sample_count")
    if not notes:
        notes = bundle.get("notes") or ""
    if sample_count is None:
        sample_count = bundle.get("sample_count")
    provenance = {
        "source": _PROVENANCE_SOURCE_COMMUNITY,
        "bundle_id": bundle.get("bundle_id"),
        "bundle_version": bundle.get("version"),
        "calibrated_at": bundle.get("published"),
        "notes": notes,
    }
    if sample_count is not None:
        provenance["sample_count"] = sample_count
    return {k: v for k, v in provenance.items() if v not in (None, "")}


def import_community_baseline(
    specs: Iterable[Any],
    bundle: dict,
    *,
    replace_cards: bool = False,
) -> "tuple[list, dict]":
    """Overlay adapter-scoped community cards onto ``specs``.

    Match only when both adapter and (id or adapter_model_name) match. Never
    copies cards across adapters. Never overwrites ``capability_score``.
    Existing local cards win per role unless ``replace_cards``.
    """
    existing = list(specs)
    entries = bundle.get("entries") if isinstance(bundle, dict) else None
    if not isinstance(entries, list):
        raise ValueError("community baseline entries must be a list")

    by_id = {spec.id: spec for spec in existing}
    matched: list[str] = []
    skipped_adapter_mismatch: list[dict] = []
    skipped_no_match: list[dict] = []
    cards_added = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_adapter = entry.get("adapter")
        incoming_cards = entry.get("role_scorecards")
        if not isinstance(incoming_cards, dict):
            incoming_cards = {}
        identity_hits = [spec for spec in existing if _identity_hit(spec, entry)]
        adapter_hits = [
            spec for spec in identity_hits if spec.adapter == entry_adapter
        ]
        mismatches = [
            spec for spec in identity_hits if spec.adapter != entry_adapter
        ]
        for spec in mismatches:
            skipped_adapter_mismatch.append(
                {
                    "id": spec.id,
                    "adapter": spec.adapter,
                    "entry_adapter": entry_adapter,
                    "reason": (
                        f"adapter mismatch: registry adapter={spec.adapter!r} "
                        f"entry adapter={entry_adapter!r}"
                    ),
                }
            )
        if not adapter_hits:
            if not mismatches:
                skipped_no_match.append(
                    {
                        "id": entry.get("id"),
                        "adapter": entry_adapter,
                        "adapter_model_name": entry.get("adapter_model_name"),
                        "reason": "no matching registry entry",
                    }
                )
            continue
        provenance = _stamp_provenance(bundle, entry)
        for spec in adapter_hits:
            merged, added = _merge_role_cards(
                getattr(spec, "role_scorecards", None) or {},
                incoming_cards,
                replace_cards=replace_cards,
            )
            cards_added += added
            by_id[spec.id] = replace(
                spec,
                role_scorecards=merged,
                score_provenance=provenance,
            )
            if spec.id not in matched:
                matched.append(spec.id)

    new_specs = [by_id[spec.id] for spec in existing]
    report = {
        "matched": matched,
        "skipped_adapter_mismatch": skipped_adapter_mismatch,
        "skipped_no_match": skipped_no_match,
        "cards_added": cards_added,
    }
    return new_specs, report
