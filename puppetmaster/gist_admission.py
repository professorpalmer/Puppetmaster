"""Verified-gist admission for peer shared-context injection.

Wave 1 (DeLM-inspired): durable compact discoveries (``ArtifactType.GIST``)
are filtered at injection boundaries so peers only see admitted gists.
Pending/rejected gists remain in the store for tooling/MCP. No LLM call —
structural validation plus optional VERIFICATION accept is enough.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from puppetmaster.artifact_status import durable_admission_allowed
from puppetmaster.models import Artifact, ArtifactType

GIST_ADMISSION_ADMITTED = "admitted"

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


def _raw_type_name(artifact: Any) -> str:
    raw = getattr(artifact, "type", None)
    if raw is None and isinstance(artifact, dict):
        raw = artifact.get("type")
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def is_admitted_for_shared_context(
    artifact: Any,
    *,
    for_job_id: Optional[str] = None,
) -> bool:
    """Return True when ``artifact`` may be injected into peer shared context.

    Admitted gists pass. Pending/rejected gists are excluded. Legacy
    FINDING/DECISION/PATCH/RISK/VERIFICATION artifacts pass when structurally
    valid (backward compatible). Other non-gist types keep prior injection
    behavior (ROUTING/GATE/plan-shaped dicts, etc.).

    Coordination-protocol gists never inject. Cross-job inject only
    independently_supported host-admitted findings (worker_asserted must not
    leak across jobs).
    """
    from puppetmaster.metr_seams import (
        independently_supported_artifact,
        is_coordination_protocol_payload,
        is_cross_job_injectable,
    )

    if is_coordination_protocol_payload(artifact):
        return False
    if _payload(artifact).get("jev_injectable") is False:
        return False
    if not is_cross_job_injectable(artifact, for_job_id=for_job_id):
        return False
    if not _freshness_allows_shared_context(artifact):
        return False
    kind = _artifact_type(artifact)
    if kind == ArtifactType.GIST or _raw_type_name(artifact) == "gist":
        if _gist_admission(artifact) != GIST_ADMISSION_ADMITTED:
            return False
        if for_job_id:
            artifact_job = getattr(artifact, "job_id", None)
            if isinstance(artifact, dict):
                artifact_job = artifact.get("job_id") or artifact_job
            if artifact_job and str(artifact_job) != str(for_job_id):
                if not independently_supported_artifact(artifact):
                    return False
        return _structurally_valid_artifact(artifact)
    if kind in _LEGACY_SHARED_CONTEXT_TYPES:
        if for_job_id:
            artifact_job = getattr(artifact, "job_id", None)
            if isinstance(artifact, dict):
                artifact_job = artifact.get("job_id") or artifact_job
            if artifact_job and str(artifact_job) != str(for_job_id):
                if not independently_supported_artifact(artifact):
                    return False
        return _structurally_valid_artifact(artifact)
    return True


def _validation_status_for_admission(artifact: Any) -> Optional[str]:
    from puppetmaster.validation import validation_status_of

    if isinstance(artifact, Artifact):
        return validation_status_of(artifact)
    payload = _payload(artifact)
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return None
    status = validation.get("status")
    return str(status) if status is not None else None


def _freshness_allows_shared_context(artifact: Any) -> bool:
    from puppetmaster.validation import REUSABLE_VALIDATION_STATUSES

    status = _validation_status_for_admission(artifact)
    if status is None:
        return True
    return status in REUSABLE_VALIDATION_STATUSES


def filter_shared_context_artifacts(
    artifacts: Iterable[Any],
    *,
    for_job_id: Optional[str] = None,
    cwd: Optional[Union[str, Path]] = None,
    store: Any = None,
) -> List[Any]:
    """Keep only artifacts safe for peer prompt injection."""
    from puppetmaster.validation import refresh_cited_freshness

    refresh_cwd = cwd if cwd not in (None, "") else None
    admitted: List[Any] = []
    for artifact in artifacts:
        if refresh_cwd is not None:
            try:
                artifact = refresh_cited_freshness(artifact, refresh_cwd, store=store)
            except Exception:
                pass
        if is_admitted_for_shared_context(artifact, for_job_id=for_job_id):
            admitted.append(artifact)
    return admitted


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
    support (an accepting VERIFICATION that names this artifact id or
    exact claim/risk/decision text, or PM
    ``claim_support_status=independently_supported``).
    ``min_confidence`` is
    retained for call-compat only and is ignored.
    """
    if _artifact_type(finding) != ArtifactType.FINDING:
        return None
    _ = min_confidence  # explicitly unused — self-rating cannot admit
    from puppetmaster.metr_seams import is_coordination_protocol_payload

    if is_coordination_protocol_payload(finding):
        return None
    if not durable_admission_allowed(finding, store=store):
        return None
    try:
        confidence = float(finding.confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    claim = str((finding.payload or {}).get("claim") or "").strip()
    if not claim:
        return None
    try:
        from puppetmaster.jev.edges import apply_finding_admission

        if apply_finding_admission(store, finding) is False:
            return None
    except Exception:
        pass
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
