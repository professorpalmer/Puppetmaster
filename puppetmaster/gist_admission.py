"""Verified-gist admission for peer shared-context injection.

Wave 1 (DeLM-inspired): durable compact discoveries (``ArtifactType.GIST``)
are filtered at injection boundaries so peers only see admitted gists.
Pending/rejected gists remain in the store for tooling/MCP. No LLM call —
structural validation plus optional VERIFICATION accept is enough.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, List, Optional, Sequence

from puppetmaster.artifact_status import durable_admission_allowed
from puppetmaster.models import Artifact, ArtifactType

AdmissionStatus = str  # "pending" | "admitted" | "rejected"

GIST_ADMISSION_PENDING = "pending"
GIST_ADMISSION_ADMITTED = "admitted"
GIST_ADMISSION_REJECTED = "rejected"

# Legacy substantive types that remain injectable without a gist wrapper.
_LEGACY_SHARED_CONTEXT_TYPES = frozenset(
    {
        ArtifactType.FINDING,
        ArtifactType.DECISION,
        ArtifactType.PATCH,
        ArtifactType.RISK,
        ArtifactType.VERIFICATION,
    }
)

_MIN_FINDING_CONFIDENCE_FOR_GIST = 0.8


def _event_verifier_payload(verifier_result: Any) -> Any:
    if isinstance(verifier_result, (bool, str, int, float)) or verifier_result is None:
        return verifier_result
    if isinstance(verifier_result, dict):
        return dict(verifier_result)
    return str(verifier_result)


def _artifact_type(artifact: Any) -> Optional[ArtifactType]:
    raw = getattr(artifact, "type", None)
    if raw is None and isinstance(artifact, dict):
        raw = artifact.get("type")
    if raw is None:
        return None
    if isinstance(raw, ArtifactType):
        return raw
    try:
        return ArtifactType(str(raw))
    except ValueError:
        return None


def _payload(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        payload = artifact.get("payload") or {}
    else:
        payload = getattr(artifact, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def _structurally_valid_artifact(artifact: Any) -> bool:
    if isinstance(artifact, Artifact):
        try:
            artifact.validate()
            return True
        except ValueError:
            return False
    # Plain dicts (test / inline injection): require non-empty payload + evidence
    # when present, matching Artifact.validate's spirit without rehydration.
    payload = _payload(artifact)
    if not payload:
        return False
    evidence = (
        artifact.get("evidence")
        if isinstance(artifact, dict)
        else getattr(artifact, "evidence", None)
    )
    if evidence is not None and not evidence:
        return False
    return True


def _gist_admission(artifact: Any) -> str:
    value = _payload(artifact).get("admission")
    return str(value).strip().lower() if value is not None else ""


def _verifier_accepted(verifier_result: Any) -> bool:
    if verifier_result is True:
        return True
    if verifier_result is False or verifier_result is None:
        return False
    if isinstance(verifier_result, dict):
        for key in ("accepted", "result", "verdict", "status"):
            raw = verifier_result.get(key)
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered in ("accept", "accepted", "pass", "passed", "ok", "true"):
                    return True
                if lowered in ("reject", "rejected", "fail", "failed", "false"):
                    return False
    return bool(verifier_result)


def _raw_type_name(artifact: Any) -> str:
    raw = getattr(artifact, "type", None)
    if raw is None and isinstance(artifact, dict):
        raw = artifact.get("type")
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def is_admitted_for_shared_context(artifact: Any) -> bool:
    """Return True when ``artifact`` may be injected into peer shared context.

    Admitted gists pass. Pending/rejected gists are excluded. Legacy
    FINDING/DECISION/PATCH/RISK/VERIFICATION artifacts pass when structurally
    valid (backward compatible). Other non-gist types keep prior injection
    behavior (ROUTING/GATE/plan-shaped dicts, etc.).
    """
    kind = _artifact_type(artifact)
    if kind == ArtifactType.GIST or _raw_type_name(artifact) == "gist":
        if _gist_admission(artifact) != GIST_ADMISSION_ADMITTED:
            return False
        return _structurally_valid_artifact(artifact)
    if kind in _LEGACY_SHARED_CONTEXT_TYPES:
        return _structurally_valid_artifact(artifact)
    return True


def filter_shared_context_artifacts(artifacts: Iterable[Any]) -> List[Any]:
    """Keep only artifacts safe for peer prompt injection."""
    return [artifact for artifact in artifacts if is_admitted_for_shared_context(artifact)]


def admit_gist(
    store: Any,
    artifact: Artifact,
    *,
    verifier_result: Any = True,
) -> Artifact:
    """Mark a gist admitted (when verifier accepts) and emit ``gist.admitted``."""
    if _artifact_type(artifact) != ArtifactType.GIST:
        raise ValueError("admit_gist requires an ArtifactType.GIST artifact")
    if not _verifier_accepted(verifier_result):
        return reject_gist(store, artifact, verifier_result=verifier_result)
    payload = dict(artifact.payload or {})
    payload["admission"] = GIST_ADMISSION_ADMITTED
    payload.setdefault("level", "gist")
    updated = replace(artifact, payload=payload, sha256=None)
    updated.validate()
    store.save_artifact(updated)
    store.emit(
        updated.job_id,
        "gist.admitted",
        {
            "artifact_id": updated.id,
            "task_id": updated.task_id,
            "source_artifact_ids": list(payload.get("source_artifact_ids") or []),
            "verifier_result": _event_verifier_payload(verifier_result),
        },
    )
    return updated


def reject_gist(
    store: Any,
    artifact: Artifact,
    *,
    verifier_result: Any = False,
) -> Artifact:
    """Mark a gist rejected and emit ``gist.rejected``."""
    if _artifact_type(artifact) != ArtifactType.GIST:
        raise ValueError("reject_gist requires an ArtifactType.GIST artifact")
    payload = dict(artifact.payload or {})
    payload["admission"] = GIST_ADMISSION_REJECTED
    payload.setdefault("level", "gist")
    updated = replace(artifact, payload=payload, sha256=None)
    updated.validate()
    store.save_artifact(updated)
    store.emit(
        updated.job_id,
        "gist.rejected",
        {
            "artifact_id": updated.id,
            "task_id": updated.task_id,
            "source_artifact_ids": list(payload.get("source_artifact_ids") or []),
            "verifier_result": _event_verifier_payload(verifier_result),
        },
    )
    return updated


def _source_has_accepting_verification(
    store: Any,
    finding: Artifact,
) -> bool:
    """True when a same-task VERIFICATION accepts the finding's claim/id."""
    try:
        artifacts = store.list_artifacts(finding.job_id)
    except Exception:
        return False
    finding_id = finding.id
    claim = str((finding.payload or {}).get("claim") or "").strip()
    for artifact in artifacts:
        if _artifact_type(artifact) != ArtifactType.VERIFICATION:
            continue
        if getattr(artifact, "task_id", None) != finding.task_id:
            continue
        payload = _payload(artifact)
        result = str(payload.get("result") or "").strip().lower()
        if result not in ("accept", "accepted", "pass", "passed", "ok", "true"):
            continue
        evidence = list(getattr(artifact, "evidence", None) or [])
        check = str(payload.get("check") or "")
        haystack = " ".join([check, *evidence])
        if finding_id and finding_id in haystack:
            return True
        if claim and claim in haystack:
            return True
        # Same-task accept with no explicit pointer still counts as structural
        # acceptance for this wave (cheap default; no LLM).
        return True
    return False


def maybe_admit_finding_as_gist(
    store: Any,
    finding: Artifact,
    *,
    min_confidence: float = _MIN_FINDING_CONFIDENCE_FOR_GIST,
) -> Optional[Artifact]:
    """Materialize an admitted gist from a substantive FINDING when eligible.

    Called after a successful ``save_artifact`` of a FINDING. Self-rating /
    ``confidence`` / ``min_confidence`` never admit. Requires independent
    support (same-task accepting VERIFICATION or PM
    ``claim_support_status=independently_supported``). ``min_confidence`` is
    retained for call-compat only and is ignored.
    """
    if _artifact_type(finding) != ArtifactType.FINDING:
        return None
    _ = min_confidence  # explicitly unused — self-rating cannot admit
    if not durable_admission_allowed(finding, store=store):
        return None
    try:
        confidence = float(finding.confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    claim = str((finding.payload or {}).get("claim") or "").strip()
    if not claim:
        return None
    evidence_digests: List[str] = []
    if finding.sha256:
        evidence_digests.append(str(finding.sha256))
    payload: dict[str, Any] = {
        "claim": claim,
        "source_artifact_ids": [finding.id],
        "admission": GIST_ADMISSION_ADMITTED,
        "level": "gist",
        "evidence_digests": evidence_digests,
    }
    # Optional unfold pointers reserved for a later wave.
    if (finding.payload or {}).get("summary_ref"):
        payload["summary_ref"] = (finding.payload or {}).get("summary_ref")
    if (finding.payload or {}).get("raw_ref"):
        payload["raw_ref"] = (finding.payload or {}).get("raw_ref")
    gist = Artifact(
        job_id=finding.job_id,
        task_id=finding.task_id,
        type=ArtifactType.GIST,
        created_by=finding.created_by,
        confidence=confidence,
        evidence=list(finding.evidence or []) or [f"source:{finding.id}"],
        payload=payload,
    )
    # Prefer verification-backed admission when present; otherwise structural.
    verifier_ok = _source_has_accepting_verification(store, finding)
    try:
        gist.validate()
    except ValueError:
        return None
    if not verifier_ok:
        # Structural path: validate already passed and admission is admitted.
        pass
    store.save_artifact(gist)
    store.emit(
        gist.job_id,
        "gist.admitted",
        {
            "artifact_id": gist.id,
            "task_id": gist.task_id,
            "source_artifact_ids": [finding.id],
            "from_finding": finding.id,
            "confidence": confidence,
            "structural": not verifier_ok,
        },
    )
    return gist


def build_pending_gist(
    *,
    job_id: str,
    task_id: str,
    created_by: str,
    claim: str,
    source_artifact_ids: Sequence[str],
    confidence: float = 0.8,
    evidence: Optional[Sequence[str]] = None,
    evidence_digests: Optional[Sequence[str]] = None,
    summary_ref: Optional[str] = None,
    raw_ref: Optional[str] = None,
) -> Artifact:
    """Helper for callers/tests that need a pending gist shell."""
    payload: dict[str, Any] = {
        "claim": claim,
        "source_artifact_ids": list(source_artifact_ids),
        "admission": GIST_ADMISSION_PENDING,
        "level": "gist",
    }
    if evidence_digests is not None:
        payload["evidence_digests"] = list(evidence_digests)
    if summary_ref is not None:
        payload["summary_ref"] = summary_ref
    if raw_ref is not None:
        payload["raw_ref"] = raw_ref
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.GIST,
        created_by=created_by,
        confidence=confidence,
        evidence=list(evidence or [f"source:{sid}" for sid in source_artifact_ids] or ["gist:pending"]),
        payload=payload,
    )


# --- Selective unfold (Wave 2) -------------------------------------------------

CONTEXT_LEVEL_GIST = "gist"
CONTEXT_LEVEL_SUMMARY = "summary"
CONTEXT_LEVEL_RAW = "raw"
CONTEXT_LEVELS = frozenset(
    {CONTEXT_LEVEL_GIST, CONTEXT_LEVEL_SUMMARY, CONTEXT_LEVEL_RAW}
)

_SUMMARY_CLAIM_MAX_CHARS = 480


def normalize_context_level(level: Optional[str]) -> str:
    """Return a supported unfold level; unknown values fall back to gist."""
    value = str(level or CONTEXT_LEVEL_GIST).strip().lower()
    if value in CONTEXT_LEVELS:
        return value
    return CONTEXT_LEVEL_GIST


def _resolve_artifact(store: Any, job_id: str, artifact_or_id: Any) -> Optional[Artifact]:
    if isinstance(artifact_or_id, Artifact):
        return artifact_or_id
    artifact_id = str(artifact_or_id or "").strip()
    if not artifact_id:
        return None
    try:
        by_id = store.get_artifacts_by_ids(job_id, [artifact_id])
    except Exception:
        return None
    found = by_id.get(artifact_id) if isinstance(by_id, dict) else None
    if isinstance(found, Artifact):
        return found
    try:
        for item in store.list_artifacts(job_id):
            if getattr(item, "id", None) == artifact_id:
                return item
    except Exception:
        return None
    return None


def _truncate_summary(text: str, limit: int = _SUMMARY_CLAIM_MAX_CHARS) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)].rstrip() + "..."


def unfold_shared_context(
    store: Any,
    artifact: Any,
    *,
    level: str = CONTEXT_LEVEL_GIST,
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    """Unfold a shared-context artifact to gist / summary / raw detail.

    Default workers should stay on ``gist``. Summary expands ``summary_ref``
    when present, otherwise a bounded claim. Raw loads source artifacts by
    ``source_artifact_ids`` / ``raw_ref`` (and the artifact body itself).
    """
    resolved_level = normalize_context_level(level)
    kind = _artifact_type(artifact)
    payload = _payload(artifact)
    artifact_id = getattr(artifact, "id", None) or (
        artifact.get("id") if isinstance(artifact, dict) else None
    )
    resolved_job = job_id or getattr(artifact, "job_id", None) or (
        artifact.get("job_id") if isinstance(artifact, dict) else None
    )
    claim = str(payload.get("claim") or "").strip()
    base: dict[str, Any] = {
        "artifact_id": artifact_id,
        "type": str(kind) if kind is not None else _raw_type_name(artifact),
        "level": resolved_level,
        "claim": claim,
        "admission": payload.get("admission"),
        "source_artifact_ids": list(payload.get("source_artifact_ids") or []),
    }
    if resolved_level == CONTEXT_LEVEL_GIST:
        base["body"] = claim
        return base

    if resolved_level == CONTEXT_LEVEL_SUMMARY:
        summary_ref = payload.get("summary_ref")
        summary_text = ""
        if summary_ref and resolved_job:
            linked = _resolve_artifact(store, str(resolved_job), summary_ref)
            if linked is not None:
                linked_payload = _payload(linked)
                summary_text = str(
                    linked_payload.get("summary")
                    or linked_payload.get("claim")
                    or linked_payload.get("decision")
                    or ""
                ).strip()
        if not summary_text:
            summary_text = str(payload.get("summary") or claim).strip()
        base["body"] = _truncate_summary(summary_text)
        base["summary_ref"] = summary_ref
        return base

    # raw
    sources: List[dict[str, Any]] = []
    raw_ref = payload.get("raw_ref")
    source_ids = list(payload.get("source_artifact_ids") or [])
    if raw_ref and str(raw_ref) not in source_ids:
        source_ids.append(str(raw_ref))
    if resolved_job:
        for source_id in source_ids:
            linked = _resolve_artifact(store, str(resolved_job), source_id)
            if linked is None:
                sources.append({"id": source_id, "missing": True})
                continue
            sources.append(
                {
                    "id": linked.id,
                    "type": str(linked.type),
                    "payload": dict(linked.payload or {}),
                    "evidence": list(linked.evidence or []),
                    "confidence": linked.confidence,
                    "sha256": linked.sha256,
                }
            )
    base["body"] = claim
    base["raw_ref"] = raw_ref
    base["sources"] = sources
    if isinstance(artifact, Artifact):
        base["artifact_payload"] = dict(artifact.payload or {})
    elif isinstance(artifact, dict):
        base["artifact_payload"] = dict(payload)
    return base


def format_unfolded_for_injection(unfolded: dict[str, Any]) -> str:
    """Render an ``unfold_shared_context`` result for prompt injection."""
    if not isinstance(unfolded, dict):
        return ""
    level = normalize_context_level(unfolded.get("level"))
    claim = str(unfolded.get("claim") or unfolded.get("body") or "").strip()
    lines: List[str] = []
    if level == CONTEXT_LEVEL_GIST:
        if claim:
            lines.append(f"Gist: {claim}")
        return "\n".join(lines).strip()
    if level == CONTEXT_LEVEL_SUMMARY:
        body = str(unfolded.get("body") or claim).strip()
        if body:
            lines.append(f"Summary: {body}")
        return "\n".join(lines).strip()
    if claim:
        lines.append(f"Raw claim: {claim}")
    for source in unfolded.get("sources") or []:
        if not isinstance(source, dict):
            continue
        if source.get("missing"):
            lines.append(f"Source missing: {source.get('id')}")
            continue
        source_type = str(source.get("type") or "artifact")
        source_payload = source.get("payload") or {}
        snippet = ""
        if isinstance(source_payload, dict):
            for key in ("claim", "summary", "decision", "change", "check", "result"):
                if source_payload.get(key):
                    snippet = str(source_payload.get(key)).strip()
                    break
        lines.append(
            f"Source ({source_type} {source.get('id')}): {snippet or '(no text)'}"
        )
    return "\n".join(lines).strip()
