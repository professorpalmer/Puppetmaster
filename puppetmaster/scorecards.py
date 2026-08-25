"""Role scorecards and community-baseline import for the model registry.

``capability_score`` on :class:`~puppetmaster.model_registry.ModelSpec` stays
the explicit manual fallback. A role card's ``capability`` (int 0-100) overrides
that fallback for one :class:`~puppetmaster.router.TaskSignals` role only.
Empty cards are bit-identical to today's routing.

A later overlay (v1.22.32) lets one host-local latency/confidence/success
receipt beat a stale editorial ``capability_score`` for the same registry
id + adapter + role. Editorial scores decay when a newer sibling is already
running successfully here. Receipts never transfer across adapter, effort,
harness, or role. One receipt is one raw observation; nothing is averaged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
import re
from typing import Any, Iterable, Optional


_PACKAGED_BASELINE = Path("baselines") / "role-scorecards-v1.json"
_DOCS_BASELINE = Path("docs") / "baselines" / "role-scorecards-v1.json"
_PROVENANCE_SOURCE_COMMUNITY = "community_baseline"
ROLE_CARD_SCALE = "puppetmaster-capability-0-100"
MIN_QUALIFIED_SAMPLE_COUNT = 5
MAX_CALIBRATION_AGE_DAYS = 180
SCORE_SOURCE_ROLE_CARD = "role_card"
SCORE_SOURCE_LOCAL_RECEIPT = "local_receipt"
SCORE_SOURCE_MANUAL = "manual"
_PROVENANCE_SOURCE_LOCAL_RECEIPT = "local_receipt"


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


def effective_capability_score(
    spec: Any,
    role: str,
    *,
    receipts: Optional[Iterable["LocalReceipt"]] = None,
    candidates: Optional[Iterable[Any]] = None,
) -> int:
    """Role-card capability when present and valid; else overlay or fallback.

    Qualified role cards still win. Otherwise a successful host-local receipt
    for this exact registry id + adapter + role can overlay the editorial
    prior. A stale editorial score is decayed when a newer sibling is already
    running successfully here. Empty receipts keep ``spec.capability_score``.
    """
    return resolve_score_authority(
        spec, role, receipts=receipts, candidates=candidates
    ).effective


def role_card_override_note(
    spec: Any,
    role: str,
    *,
    receipts: Optional[Iterable["LocalReceipt"]] = None,
    candidates: Optional[Iterable[Any]] = None,
) -> str:
    """Reason suffix when a card, local receipt, or sibling decay changed the pick."""
    return resolve_score_authority(
        spec, role, receipts=receipts, candidates=candidates
    ).note


@dataclass(frozen=True)
class LocalReceipt:
    """One raw host-local observation for one registry identity + role.

    Not an average. Not a community card. Latency, confidence, and success
    are the observation; ``capability_score`` is only the editorial prior.
    """

    registry_id: str
    adapter: str
    role: str
    success: bool
    elapsed_seconds: Optional[float] = None
    confidence: Optional[float] = None
    effort: str = ""
    harness: str = ""
    observed_at: str = ""


@dataclass(frozen=True)
class ScoreAuthority:
    """Resolved routing authority for one spec + role."""

    effective: int
    source: str
    decayed: bool = False
    receipt: Optional[LocalReceipt] = None
    provenance: dict = field(default_factory=dict)
    note: str = ""
    sample_count: Optional[int] = None
    predicted_quality: Optional[float] = None
    predicted_latency_p50_ms: Optional[float] = None


def spec_effort(spec: Any) -> str:
    raw = (getattr(spec, "payload_defaults", None) or {}).get("reasoning_effort")
    return str(raw).strip() if raw not in (None, "") else ""


def spec_harness(spec: Any) -> str:
    raw = (getattr(spec, "payload_defaults", None) or {}).get("harness")
    if raw not in (None, ""):
        return str(raw).strip()
    return str(getattr(spec, "adapter", "") or "").strip()


def receipt_effort(receipt: LocalReceipt) -> str:
    return str(receipt.effort or "").strip()


def receipt_harness(receipt: LocalReceipt) -> str:
    if str(receipt.harness or "").strip():
        return str(receipt.harness).strip()
    return str(receipt.adapter or "").strip()


def model_family(adapter_model_name: str) -> str:
    """Conservative family key so gpt-5 and gpt-5.6-luna are siblings.

    Unrecognized names stay exact, so unrelated models never decay each other.
    """
    text = (adapter_model_name or "").strip().lower().replace("_", "-")
    match = re.match(r"(gpt-\d+)", text)
    if match:
        return match.group(1)
    return text


def _same_lane(spec: Any, role: str, receipt: LocalReceipt) -> bool:
    return (
        str(receipt.registry_id) == str(spec.id)
        and str(receipt.adapter) == str(spec.adapter)
        and str(receipt.role or "") == str(role or "")
        and receipt_effort(receipt) == spec_effort(spec)
        and receipt_harness(receipt) == spec_harness(spec)
    )


def _same_sibling_lane(left: Any, right: Any, role: str) -> bool:
    if str(left.id) == str(right.id):
        return False
    if str(left.adapter) != str(right.adapter):
        return False
    if spec_effort(left) != spec_effort(right):
        return False
    if spec_harness(left) != spec_harness(right):
        return False
    if model_family(left.adapter_model_name) != model_family(right.adapter_model_name):
        return False
    return True


def _receipt_sort_key(receipt: LocalReceipt) -> tuple:
    return (str(receipt.observed_at or ""),)


def latest_receipt_for(
    spec: Any,
    role: str,
    receipts: Optional[Iterable[LocalReceipt]],
) -> Optional[LocalReceipt]:
    """Return the newest matching observation. No averaging."""
    matched = [
        receipt
        for receipt in (receipts or [])
        if isinstance(receipt, LocalReceipt) and _same_lane(spec, role, receipt)
    ]
    if not matched:
        return None
    return max(matched, key=_receipt_sort_key)


def receipt_is_live(receipt: Optional[LocalReceipt]) -> bool:
    if receipt is None or not receipt.success:
        return False
    if receipt.elapsed_seconds is None or receipt.confidence is None:
        return False
    if isinstance(receipt.elapsed_seconds, bool) or isinstance(receipt.confidence, bool):
        return False
    try:
        elapsed = float(receipt.elapsed_seconds)
        confidence = float(receipt.confidence)
    except (TypeError, ValueError):
        return False
    return elapsed >= 0 and 0.0 <= confidence <= 1.0


def receipt_beats_editorial_sibling(
    live: LocalReceipt,
    sibling: Optional[LocalReceipt],
) -> bool:
    """A faster, higher-confidence live receipt beats a stale or worse sibling."""
    if not receipt_is_live(live):
        return False
    if sibling is None or not receipt_is_live(sibling):
        return True
    return (
        float(live.elapsed_seconds) < float(sibling.elapsed_seconds)
        and float(live.confidence) > float(sibling.confidence)
    )


def local_receipts_from_records(records: Iterable[Any]) -> list[LocalReceipt]:
    """Project audit records into one-observation receipts. No averages."""
    out: list[LocalReceipt] = []
    for record in records:
        registry_id = str(getattr(record, "model_id", "") or "")
        adapter = str(getattr(record, "adapter", "") or "")
        role = str(getattr(record, "role", "") or "")
        if not registry_id or not adapter:
            continue
        verification = str(getattr(record, "verification_result", "") or "").strip().lower()
        failed = verification in {"failed", "timeout", "error", "fail"}
        gate_passed = getattr(record, "gate_passed", None)
        success = (not failed) and gate_passed is not False
        elapsed = getattr(record, "elapsed_seconds", None)
        confidence = getattr(record, "confidence", None)
        if isinstance(elapsed, bool) or isinstance(confidence, bool):
            continue
        out.append(
            LocalReceipt(
                registry_id=registry_id,
                adapter=adapter,
                role=role,
                success=bool(success),
                elapsed_seconds=float(elapsed) if elapsed is not None else None,
                confidence=float(confidence) if confidence is not None else None,
                effort=str(getattr(record, "effort", "") or ""),
                harness=str(getattr(record, "harness", "") or adapter),
                observed_at=str(getattr(record, "evaluated_at", "") or ""),
            )
        )
    return out


def resolve_score_authority(
    spec: Any,
    role: str,
    *,
    receipts: Optional[Iterable[LocalReceipt]] = None,
    candidates: Optional[Iterable[Any]] = None,
) -> ScoreAuthority:
    """Resolve card vs local receipt vs editorial fallback for one spec."""
    materialized = [item for item in (receipts or []) if isinstance(item, LocalReceipt)]
    pool = list(candidates or [])
    override = card_capability(spec, role)
    manual = int(spec.capability_score)
    card = (getattr(spec, "role_scorecards", None) or {}).get(role or "")
    if not isinstance(card, dict):
        card = {}
    provenance = dict(getattr(spec, "score_provenance", None) or {})
    card_provenance = card.get("provenance") if override is not None else None
    if isinstance(card_provenance, dict):
        provenance.update(card_provenance)
    sample = card.get("sample_count") if override is not None else None
    if isinstance(sample, bool) or not isinstance(sample, int):
        sample = provenance.get("sample_count")
        if isinstance(sample, bool) or not isinstance(sample, int):
            sample = None
    quality = card.get("quality") if override is not None else None
    if isinstance(quality, bool) or not isinstance(quality, (int, float)):
        quality = None
    elif quality is not None:
        quality = float(quality)
    latency = card.get("latency_p50_ms") if override is not None else None
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        latency = None
    elif latency is not None:
        latency = float(latency)
    if override is not None:
        note = ""
        if override != manual:
            note = (
                f"; role={role} card capability {override} "
                f"(manual fallback {manual})"
            )
        return ScoreAuthority(
            effective=override,
            source=SCORE_SOURCE_ROLE_CARD,
            provenance=provenance,
            note=note,
            sample_count=sample,
            predicted_quality=quality,
            predicted_latency_p50_ms=latency,
        )

    own = latest_receipt_for(spec, role, materialized)
    sibling_live: Optional[tuple[Any, LocalReceipt]] = None
    for other in pool:
        if not _same_sibling_lane(spec, other, role):
            continue
        other_receipt = latest_receipt_for(other, role, materialized)
        if not receipt_is_live(other_receipt):
            continue
        if sibling_live is None or receipt_quality_tuple_from_receipt(
            other_receipt
        ) > receipt_quality_tuple_from_receipt(sibling_live[1]):
            sibling_live = (other, other_receipt)

    sibling_beats_own = (
        sibling_live is not None
        and receipt_beats_editorial_sibling(sibling_live[1], own)
    )

    if receipt_is_live(own) and not sibling_beats_own:
        # Own live receipt is the observation for this identity. It overlays
        # the editorial prior without inventing a new 0-100 number.
        note = (
            f"; role={role} local receipt {own.elapsed_seconds:g}s "
            f"confidence {own.confidence:g} (manual fallback {manual})"
        )
        return ScoreAuthority(
            effective=manual,
            source=SCORE_SOURCE_LOCAL_RECEIPT,
            receipt=own,
            provenance={
                "source": _PROVENANCE_SOURCE_LOCAL_RECEIPT,
                "version": "1",
                "elapsed_seconds": own.elapsed_seconds,
                "confidence": own.confidence,
                "success": True,
                "sample_count": 1,
                "effort": receipt_effort(own),
                "harness": receipt_harness(own),
            },
            note=note,
            sample_count=1,
            predicted_quality=float(own.confidence),
            predicted_latency_p50_ms=float(own.elapsed_seconds) * 1000.0,
        )

    if sibling_live is not None:
        sibling_spec, sibling_receipt = sibling_live
        sibling_effective = int(sibling_spec.capability_score)
        decayed = min(manual, max(0, sibling_effective - 1))
        note = (
            f"; editorial capability_score {manual} decayed to {decayed}; "
            f"sibling {sibling_spec.id} has a successful local receipt "
            f"{sibling_receipt.elapsed_seconds:g}s confidence "
            f"{sibling_receipt.confidence:g}"
        )
        return ScoreAuthority(
            effective=decayed,
            source=SCORE_SOURCE_MANUAL,
            decayed=True,
            receipt=sibling_receipt,
            provenance={
                "source": SCORE_SOURCE_MANUAL,
                "decayed": True,
                "decayed_from": manual,
                "decayed_to": decayed,
                "sibling_id": sibling_spec.id,
                "sibling_receipt": {
                    "source": _PROVENANCE_SOURCE_LOCAL_RECEIPT,
                    "elapsed_seconds": sibling_receipt.elapsed_seconds,
                    "confidence": sibling_receipt.confidence,
                    "success": True,
                    "sample_count": 1,
                },
            },
            note=note if decayed != manual else (
                f"; editorial capability_score {manual} stale; sibling "
                f"{sibling_spec.id} has a successful local receipt"
            ),
        )

    return ScoreAuthority(
        effective=manual,
        source=SCORE_SOURCE_MANUAL,
        provenance=provenance,
    )


def source_rank(authority: ScoreAuthority) -> int:
    """Higher rank outruns a stale editorial prior. Qualified cards stay first."""
    if authority.source == SCORE_SOURCE_ROLE_CARD:
        return 2
    if authority.source == SCORE_SOURCE_LOCAL_RECEIPT:
        return 1
    if authority.decayed:
        return -1
    return 0


def is_capability_sufficient(authority: ScoreAuthority, need: int) -> bool:
    """A live local receipt already did this role here; it meets the need."""
    if authority.source == SCORE_SOURCE_LOCAL_RECEIPT and receipt_is_live(authority.receipt):
        return True
    return authority.effective >= need


def receipt_quality_tuple_from_receipt(receipt: Optional[LocalReceipt]) -> tuple:
    """Higher is better. One observation; never an average."""
    if not receipt_is_live(receipt):
        return (0.0, 0.0)
    return (-float(receipt.elapsed_seconds), float(receipt.confidence))


def receipt_quality_tuple(authority: ScoreAuthority) -> tuple:
    """Higher is better. One observation; never an average."""
    if authority.source != SCORE_SOURCE_LOCAL_RECEIPT:
        return (0.0, 0.0)
    return receipt_quality_tuple_from_receipt(authority.receipt)


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
    """Match evidence to one exact registry id, never a shared wire name."""
    entry_id = entry.get("id")
    return (
        isinstance(entry_id, str)
        and bool(entry_id)
        and spec.id == entry_id
    )


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

    Match only when registry id and adapter both match. Never copies cards
    across registry variants or adapters. Never overwrites ``capability_score``.
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
