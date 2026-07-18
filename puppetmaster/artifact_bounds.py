"""Bounded artifact persistence before SQLite / file-store writes.

Large worker payloads (especially PATCH diffs and long FINDING text) must not
inflate the coordination store without bound. This module soft-caps serialized
artifact JSON bytes while preserving reviewability:

* Spill oversized text to a durable sidecar under
  ``{state_dir}/jobs/{job_id}/artifact_offload/`` when possible.
* Keep a deterministic head/tail preview inline (reuses ``tool_offload``
  helpers).
* Record truthful truncation / offload metadata so nothing is silently dropped.

Policy (never raises from the hot path):

* Kill switch: ``PUPPETMASTER_ARTIFACT_BOUNDS=0`` disables bounding entirely.
* Global cap: ``PUPPETMASTER_ARTIFACT_MAX_BYTES`` (default 262144).
* Optional PATCH caps: ``PUPPETMASTER_ARTIFACT_PATCH_DIFF_MAX_CHARS`` and
  ``PUPPETMASTER_ARTIFACT_PATCH_FILES_MAX``.
* Generic text field cap: ``PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS``.
* Write failure / missing state dir: last-resort soft truncate with an
  explicit truncated flag (same spirit as tool-output hard-cap fallback).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from puppetmaster.fs_permissions import mkdir_private, write_private_text
from puppetmaster.models import Artifact, ArtifactType, to_jsonable
from puppetmaster.tool_offload import head_tail_preview, soft_truncate

DEFAULT_MAX_ARTIFACT_BYTES = 262_144
DEFAULT_TEXT_FIELD_MAX_CHARS = 48_000
DEFAULT_PATCH_DIFF_MAX_CHARS = 20_000
DEFAULT_PATCH_FILES_MAX = 500
DEFAULT_HEAD_CHARS = 2_000
DEFAULT_TAIL_CHARS = 2_000
DEFAULT_EVIDENCE_MAX_ITEMS = 64
DEFAULT_EVIDENCE_ITEM_MAX_CHARS = 4_000
# Leave room for the post-bound sha256 stamp + JSON key overhead in the
# indent=2 persistence form so INSERT size stays under the configured cap.
SHA256_RESERVE_BYTES = 160
OFFLOAD_SUBDIR = "artifact_offload"
BOUNDS_KEY = "_persistence_bounds"

PathLike = Union[str, Path]

# Payload keys that must remain present for schema validation — never delete.
_REQUIRED_PAYLOAD_KEYS = {
    ArtifactType.FINDING: ("claim",),
    ArtifactType.DECISION: ("decision", "why"),
    ArtifactType.PATCH: ("change", "files"),
    ArtifactType.VERIFICATION: ("check", "result"),
    ArtifactType.RISK: ("risk", "mitigation"),
    ArtifactType.MEMORY_SUMMARY: ("summary",),
    ArtifactType.ROUTING: ("model_id", "adapter", "policy"),
    ArtifactType.GATE: ("gate", "passed"),
}


def bounds_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_ARTIFACT_BOUNDS", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def max_artifact_bytes() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_MAX_BYTES", DEFAULT_MAX_ARTIFACT_BYTES))


def text_field_max_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS", DEFAULT_TEXT_FIELD_MAX_CHARS))


def patch_diff_max_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_PATCH_DIFF_MAX_CHARS", DEFAULT_PATCH_DIFF_MAX_CHARS))


def patch_files_max() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_PATCH_FILES_MAX", DEFAULT_PATCH_FILES_MAX))


def preview_head_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS", DEFAULT_HEAD_CHARS))


def preview_tail_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS", DEFAULT_TAIL_CHARS))


def serialized_artifact_bytes(artifact: Artifact) -> int:
    """Byte length of the JSON form that persistence writes (indent=2)."""
    value = to_jsonable(replace(artifact, sha256=None))
    return len(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))


def _sanitize_field(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip())[:80]
    return cleaned.strip("-._") or "field"


def _offload_root(state_dir: Path, artifact: Artifact) -> Path:
    return state_dir / "jobs" / artifact.job_id / OFFLOAD_SUBDIR / artifact.id


def _spill_text(
    text: str,
    *,
    state_dir: Optional[Path],
    artifact: Artifact,
    field_name: str,
) -> Optional[str]:
    """Write full text to a sidecar. Returns absolute path or None on failure."""
    if state_dir is None:
        return None
    try:
        target = _offload_root(state_dir, artifact) / f"{_sanitize_field(field_name)}.txt"
        mkdir_private(target.parent)
        write_private_text(target, text)
        return str(target.resolve())
    except OSError:
        return None


def _preview(text: str) -> str:
    return head_tail_preview(
        text,
        head=preview_head_chars(),
        tail=preview_tail_chars(),
    )


def _ensure_bounds(payload: dict[str, Any]) -> dict[str, Any]:
    bounds = payload.get(BOUNDS_KEY)
    if not isinstance(bounds, dict):
        bounds = {
            "truncated": False,
            "original_bytes": 0,
            "stored_bytes": 0,
            "fields": {},
        }
        payload[BOUNDS_KEY] = bounds
    fields = bounds.get("fields")
    if not isinstance(fields, dict):
        bounds["fields"] = {}
    return bounds


def _record_field_bound(
    payload: dict[str, Any],
    field_name: str,
    *,
    original_chars: int,
    preview_chars: int,
    offload_path: Optional[str],
    reason: str,
) -> None:
    bounds = _ensure_bounds(payload)
    bounds["truncated"] = True
    bounds["fields"][field_name] = {
        "truncated": True,
        "original_chars": original_chars,
        "preview_chars": preview_chars,
        "offloaded": bool(offload_path),
        "offload_path": offload_path,
        "reason": reason,
    }


def _bound_string_field(
    payload: dict[str, Any],
    field_name: str,
    value: str,
    *,
    char_cap: int,
    state_dir: Optional[Path],
    artifact: Artifact,
    reason: str,
    trusted_originals: Optional[Dict[str, str]] = None,
) -> str:
    body = value if isinstance(value, str) else str(value)
    if trusted_originals is not None and field_name not in trusted_originals:
        # Remember the first full text we saw for this field so later shrink
        # passes can re-preview from memory — never from payload path metadata.
        trusted_originals[field_name] = body
    if char_cap > 0 and len(body) <= char_cap:
        return body

    # Prefer head/tail preview sized to the configured cap when possible.
    head = preview_head_chars()
    tail = preview_tail_chars()
    if char_cap > 0:
        # Leave room for the omission marker; fall back to soft truncate.
        budget = max(0, char_cap)
        if head + tail > budget:
            head = max(0, budget // 2)
            tail = max(0, budget - head)

    if head + tail > 0 and len(body) > head + tail:
        preview = head_tail_preview(body, head=head, tail=tail)
    else:
        preview = soft_truncate(body, char_cap if char_cap > 0 else DEFAULT_TEXT_FIELD_MAX_CHARS)

    if char_cap > 0 and len(preview) > char_cap:
        preview = soft_truncate(body, char_cap)

    offload_path = _spill_text(
        body,
        state_dir=state_dir,
        artifact=artifact,
        field_name=field_name,
    )
    _record_field_bound(
        payload,
        field_name,
        original_chars=len(body),
        preview_chars=len(preview),
        offload_path=offload_path,
        reason=reason if offload_path else f"{reason}; soft-truncated (no offload)",
    )
    return preview


def _bound_patch_files(payload: dict[str, Any], *, files_max: int) -> None:
    files = payload.get("files")
    if not isinstance(files, list) or files_max <= 0 or len(files) <= files_max:
        return
    total = len(files)
    payload["files"] = list(files[:files_max])
    payload["files_truncated"] = True
    payload["files_total"] = total
    bounds = _ensure_bounds(payload)
    bounds["truncated"] = True
    bounds["fields"]["files"] = {
        "truncated": True,
        "original_items": total,
        "kept_items": files_max,
        "offloaded": False,
        "offload_path": None,
        "reason": "patch files list cap",
    }


def _bound_evidence(
    evidence: list[str],
    *,
    payload: dict[str, Any],
    state_dir: Optional[Path],
    artifact: Artifact,
    trusted_originals: Optional[Dict[str, str]] = None,
) -> list[str]:
    if not evidence:
        return evidence
    item_cap = max(0, _env_int("PUPPETMASTER_ARTIFACT_EVIDENCE_ITEM_MAX_CHARS", DEFAULT_EVIDENCE_ITEM_MAX_CHARS))
    max_items = max(1, _env_int("PUPPETMASTER_ARTIFACT_EVIDENCE_MAX_ITEMS", DEFAULT_EVIDENCE_MAX_ITEMS))
    changed = False
    bounded: list[str] = []
    for index, item in enumerate(evidence):
        text = item if isinstance(item, str) else str(item)
        if item_cap > 0 and len(text) > item_cap:
            field = f"evidence[{index}]"
            text = _bound_string_field(
                payload,
                field,
                text,
                char_cap=item_cap,
                state_dir=state_dir,
                artifact=artifact,
                reason="evidence item cap",
                trusted_originals=trusted_originals,
            )
            changed = True
        bounded.append(text)
    if len(bounded) > max_items:
        overflow = len(bounded) - max_items
        bounded = bounded[:max_items]
        bounded[-1] = f"{bounded[-1]} (+{overflow} more evidence items truncated)"
        changed = True
        bounds = _ensure_bounds(payload)
        bounds["truncated"] = True
        bounds["fields"]["evidence"] = {
            "truncated": True,
            "original_items": len(evidence),
            "kept_items": max_items,
            "offloaded": False,
            "offload_path": None,
            "reason": "evidence list cap",
        }
    return bounded if changed else evidence


def _largest_string_fields(payload: dict[str, Any]) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    for key, value in payload.items():
        if key == BOUNDS_KEY:
            continue
        if isinstance(value, str):
            fields.append((key, value))
    fields.sort(key=lambda item: len(item[1]), reverse=True)
    return fields


def _apply_type_caps(
    artifact: Artifact,
    payload: dict[str, Any],
    *,
    state_dir: Optional[Path],
    trusted_originals: Optional[Dict[str, str]] = None,
) -> None:
    text_cap = text_field_max_chars()
    if artifact.type == ArtifactType.PATCH:
        diff_cap = patch_diff_max_chars()
        diff = payload.get("unified_diff")
        if isinstance(diff, str) and diff_cap > 0 and len(diff) > diff_cap:
            original_chars = len(diff)
            preview = _bound_string_field(
                payload,
                "unified_diff",
                diff,
                char_cap=diff_cap,
                state_dir=state_dir,
                artifact=artifact,
                reason="patch diff char cap",
                trusted_originals=trusted_originals,
            )
            payload["unified_diff"] = preview
            payload["diff_truncated"] = True
            payload.setdefault("diff_total_chars", original_chars)
            field_meta = _ensure_bounds(payload)["fields"].get("unified_diff") or {}
            offload_path = field_meta.get("offload_path")
            if offload_path and not payload.get("unified_diff_sidecar_path"):
                payload["unified_diff_sidecar_path"] = offload_path
        _bound_patch_files(payload, files_max=patch_files_max())

    # Soft-cap large top-level text fields for every artifact type.
    if text_cap > 0:
        for key, value in list(payload.items()):
            if key == BOUNDS_KEY or not isinstance(value, str):
                continue
            if key == "unified_diff" and artifact.type == ArtifactType.PATCH:
                continue  # already handled with the PATCH-specific cap
            if len(value) > text_cap:
                payload[key] = _bound_string_field(
                    payload,
                    key,
                    value,
                    char_cap=text_cap,
                    state_dir=state_dir,
                    artifact=artifact,
                    reason="text field char cap",
                    trusted_originals=trusted_originals,
                )


def _shrink_until_under_cap(
    artifact: Artifact,
    payload: dict[str, Any],
    evidence: list[str],
    *,
    state_dir: Optional[Path],
    max_bytes: int,
    trusted_originals: Optional[Dict[str, str]] = None,
) -> Tuple[dict[str, Any], list[str]]:
    """Aggressively shrink the largest remaining strings until under the byte cap."""
    if max_bytes <= 0:
        return payload, evidence
    originals = trusted_originals if trusted_originals is not None else {}

    # Iteratively reduce preview budgets for the largest fields.
    for _ in range(24):
        candidate = replace(artifact, payload=payload, evidence=evidence, sha256=None)
        size = serialized_artifact_bytes(candidate)
        if size <= max_bytes:
            break

        largest = _largest_string_fields(payload)
        if not largest:
            # Last resort: clamp evidence strings further.
            evidence = _bound_evidence(
                evidence,
                payload=payload,
                state_dir=state_dir,
                artifact=artifact,
                trusted_originals=originals,
            )
            candidate = replace(artifact, payload=payload, evidence=evidence, sha256=None)
            if serialized_artifact_bytes(candidate) <= max_bytes:
                break
            # Still oversized with no string fields left — stop rather than
            # destroying required schema keys.
            break

        # Shrink the top few offenders each pass so multi-field blobs converge.
        for key, value in largest[:3]:
            if len(value) <= 64:
                continue
            target = max(256, len(value) // 2)
            if target >= len(value):
                target = max(64, len(value) - 1)
            # Re-preview from the trusted in-memory original when we have one.
            # Never open ``_persistence_bounds.fields.*.offload_path`` (or any
            # other artifact-supplied filesystem path) — that metadata is
            # attacker-controlled on re-persist of untrusted payloads.
            source = originals.get(key, value)
            payload[key] = _bound_string_field(
                payload,
                key,
                source,
                char_cap=target,
                state_dir=state_dir,
                artifact=artifact,
                reason="serialized artifact byte cap",
                trusted_originals=originals,
            )
            if key == "unified_diff":
                payload["diff_truncated"] = True
                payload.setdefault("diff_total_chars", len(source))
                field_meta = _ensure_bounds(payload)["fields"].get("unified_diff") or {}
                offload_path = field_meta.get("offload_path")
                if offload_path and not payload.get("unified_diff_sidecar_path"):
                    payload["unified_diff_sidecar_path"] = offload_path

    return payload, evidence


def prepare_artifact_for_persist(
    artifact: Artifact,
    *,
    state_dir: Optional[PathLike] = None,
) -> Artifact:
    """Return an artifact safe to INSERT / write_json under configured caps.

    Idempotent for already-bounded artifacts under the same caps. Never raises.
    """
    if not bounds_enabled():
        return artifact

    max_bytes = max_artifact_bytes()
    effective_max = (
        max(0, max_bytes - SHA256_RESERVE_BYTES) if max_bytes > 0 else 0
    )
    root: Optional[Path] = Path(state_dir) if state_dir else None

    try:
        original_bytes = serialized_artifact_bytes(artifact)
        payload = dict(artifact.payload or {})
        evidence = list(artifact.evidence or [])

        needs_work = False
        if effective_max > 0 and original_bytes > effective_max:
            needs_work = True
        elif artifact.type == ArtifactType.PATCH:
            diff = payload.get("unified_diff")
            diff_cap = patch_diff_max_chars()
            files = payload.get("files")
            files_max = patch_files_max()
            if isinstance(diff, str) and diff_cap > 0 and len(diff) > diff_cap:
                needs_work = True
            if isinstance(files, list) and files_max > 0 and len(files) > files_max:
                needs_work = True
        text_cap = text_field_max_chars()
        if not needs_work and text_cap > 0:
            for key, value in payload.items():
                if key != BOUNDS_KEY and isinstance(value, str) and len(value) > text_cap:
                    needs_work = True
                    break

        if not needs_work:
            return artifact

        # Full field text captured during this prepare call only. Persistence
        # must never re-open paths from payload ``_persistence_bounds``.
        trusted_originals: Dict[str, str] = {}
        _apply_type_caps(
            artifact, payload, state_dir=root, trusted_originals=trusted_originals
        )
        evidence = _bound_evidence(
            evidence,
            payload=payload,
            state_dir=root,
            artifact=artifact,
            trusted_originals=trusted_originals,
        )
        payload, evidence = _shrink_until_under_cap(
            artifact,
            payload,
            evidence,
            state_dir=root,
            max_bytes=effective_max,
            trusted_originals=trusted_originals,
        )

        # Preserve required keys even if a buggy producer omitted them after
        # bounding — validate() still owns the hard failure for empty/missing.
        for key in _REQUIRED_PAYLOAD_KEYS.get(artifact.type, ()):
            if key not in payload and key in (artifact.payload or {}):
                payload[key] = artifact.payload[key]

        if not evidence and artifact.evidence:
            evidence = list(artifact.evidence[:1])

        bounds = _ensure_bounds(payload)
        bounds["original_bytes"] = original_bytes
        bounded = replace(
            artifact,
            payload=payload,
            evidence=evidence,
            sha256=None,  # re-hash after bounding
        )
        bounds["stored_bytes"] = serialized_artifact_bytes(bounded)
        bounds["truncated"] = True
        return replace(bounded, payload=payload)
    except Exception:
        # Bounding must never block persistence of a valid artifact.
        return artifact


def artifact_bounds_summary(artifact: Artifact) -> Dict[str, Any]:
    """Return the persistence-bounds metadata block, or an empty dict."""
    payload = artifact.payload or {}
    bounds = payload.get(BOUNDS_KEY)
    return dict(bounds) if isinstance(bounds, dict) else {}
