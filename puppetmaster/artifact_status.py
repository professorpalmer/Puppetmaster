"""Explicit artifact statuses vs uncalibrated confidence (#88).

One ``confidence`` float used to mix worker self-rating, adapter/process
health, gate bookkeeping, and implied evidence. This module splits those
meanings into named statuses. ``confidence`` remains on the artifact for
stored-record compatibility and is mapped to ``worker_self_rating``.

``worker_self_rating`` and legacy ``confidence`` are never admission inputs
and are not a calibrated probability. Do not invent
``predicted_independent_gate_pass_probability`` here.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from puppetmaster.models import ArtifactType

# --- Closed vocabularies (reviewable contract) --------------------------------

EXECUTION_UNKNOWN = "unknown"
EXECUTION_RUNNING = "running"
EXECUTION_COMPLETED = "completed"
EXECUTION_FAILED = "failed"
EXECUTION_DEGRADED = "degraded"
EXECUTION_STATUSES = frozenset(
    {
        EXECUTION_UNKNOWN,
        EXECUTION_RUNNING,
        EXECUTION_COMPLETED,
        EXECUTION_FAILED,
        EXECUTION_DEGRADED,
    }
)

GROUNDING_UNKNOWN = "unknown"
GROUNDING_UNGROUNDED = "ungrounded"
GROUNDING_CITED = "cited"
GROUNDING_GROUNDED = "grounded"
GROUNDING_STATUSES = frozenset(
    {
        GROUNDING_UNKNOWN,
        GROUNDING_UNGROUNDED,
        GROUNDING_CITED,
        GROUNDING_GROUNDED,
    }
)

CLAIM_SUPPORT_UNKNOWN = "unknown"
CLAIM_SUPPORT_UNSUPPORTED = "unsupported"
CLAIM_SUPPORT_WORKER_ASSERTED = "worker_asserted"
CLAIM_SUPPORT_INDEPENDENT = "independently_supported"
CLAIM_SUPPORT_STATUSES = frozenset(
    {
        CLAIM_SUPPORT_UNKNOWN,
        CLAIM_SUPPORT_UNSUPPORTED,
        CLAIM_SUPPORT_WORKER_ASSERTED,
        CLAIM_SUPPORT_INDEPENDENT,
    }
)

CRITERION_UNKNOWN = "unknown"
CRITERION_UNMET = "unmet"
CRITERION_MET = "met"
CRITERION_NOT_APPLICABLE = "not_applicable"
CRITERION_STATUSES = frozenset(
    {
        CRITERION_UNKNOWN,
        CRITERION_UNMET,
        CRITERION_MET,
        CRITERION_NOT_APPLICABLE,
    }
)

_PASSED = frozenset({"accept", "accepted", "pass", "passed", "ok", "true"})
_FAILED = frozenset({"reject", "rejected", "fail", "failed", "false", "blocked"})
_DEGRADED = frozenset({"degraded"})

# Worker-supplied labels that must not grant independent support.
_WORKER_SUPPORT_ALIASES = frozenset(
    {
        CLAIM_SUPPORT_INDEPENDENT,
        "supported",
        "verified",
        "independent",
        "independently-supported",
    }
)


def _payload_of(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        payload = artifact.get("payload") or {}
    else:
        payload = getattr(artifact, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def _type_of(artifact: Any) -> Optional[ArtifactType]:
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


def _norm(value: Any, allowed: frozenset[str], default: str) -> str:
    if value is None:
        return default
    raw = str(value).strip().lower().replace("-", "_")
    return raw if raw in allowed else default


def normalize_execution_status(value: Any) -> str:
    return _norm(value, EXECUTION_STATUSES, EXECUTION_UNKNOWN)


def normalize_grounding_status(value: Any) -> str:
    return _norm(value, GROUNDING_STATUSES, GROUNDING_UNKNOWN)


def normalize_claim_support_status(value: Any, *, from_worker_payload: bool = False) -> str:
    raw = None if value is None else str(value).strip().lower().replace("-", "_")
    if from_worker_payload and raw in _WORKER_SUPPORT_ALIASES:
        return CLAIM_SUPPORT_WORKER_ASSERTED
    return _norm(value, CLAIM_SUPPORT_STATUSES, CLAIM_SUPPORT_UNKNOWN)


def normalize_criterion_status(value: Any) -> str:
    return _norm(value, CRITERION_STATUSES, CRITERION_UNKNOWN)


def worker_self_rating_of(artifact: Any) -> Optional[float]:
    """Non-authoritative 0..1 self-rating. Never an admission input."""
    explicit = (
        artifact.get("worker_self_rating")
        if isinstance(artifact, dict)
        else getattr(artifact, "worker_self_rating", None)
    )
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            value = None
        else:
            if 0 <= value <= 1:
                return value
    raw = (
        artifact.get("confidence")
        if isinstance(artifact, dict)
        else getattr(artifact, "confidence", None)
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if 0 <= value <= 1:
        return value
    return None


def infer_execution_status(artifact: Any) -> str:
    existing = (
        artifact.get("execution_status")
        if isinstance(artifact, dict)
        else getattr(artifact, "execution_status", None)
    )
    if existing:
        return normalize_execution_status(existing)
    payload = _payload_of(artifact)
    if payload.get("execution_status"):
        return normalize_execution_status(payload.get("execution_status"))
    kind = _type_of(artifact)
    if kind == ArtifactType.VERIFICATION:
        result = str(payload.get("result") or "").strip().lower()
        if result in _PASSED:
            return EXECUTION_COMPLETED
        if result in _DEGRADED:
            return EXECUTION_DEGRADED
        if result in _FAILED:
            return EXECUTION_FAILED
    if kind == ArtifactType.GATE:
        passed = payload.get("passed")
        if passed is True:
            return EXECUTION_COMPLETED
        if passed is False:
            return EXECUTION_FAILED
    return EXECUTION_UNKNOWN


def infer_criterion_status(artifact: Any) -> str:
    existing = (
        artifact.get("criterion_status")
        if isinstance(artifact, dict)
        else getattr(artifact, "criterion_status", None)
    )
    if existing:
        return normalize_criterion_status(existing)
    payload = _payload_of(artifact)
    if payload.get("criterion_status"):
        return normalize_criterion_status(payload.get("criterion_status"))
    kind = _type_of(artifact)
    if kind == ArtifactType.VERIFICATION:
        result = str(payload.get("result") or "").strip().lower()
        if result in _PASSED:
            return CRITERION_MET
        if result in _FAILED or result in _DEGRADED:
            return CRITERION_UNMET
    if kind == ArtifactType.GATE:
        passed = payload.get("passed")
        if passed is True:
            return CRITERION_MET
        if passed is False:
            return CRITERION_UNMET
    return CRITERION_UNKNOWN


def infer_grounding_status(artifact: Any) -> str:
    existing = (
        artifact.get("grounding_status")
        if isinstance(artifact, dict)
        else getattr(artifact, "grounding_status", None)
    )
    if existing:
        return normalize_grounding_status(existing)
    payload = _payload_of(artifact)
    if payload.get("grounding_status"):
        return normalize_grounding_status(payload.get("grounding_status"))
    evidence = (
        artifact.get("evidence")
        if isinstance(artifact, dict)
        else getattr(artifact, "evidence", None)
    )
    if evidence:
        return GROUNDING_CITED
    return GROUNDING_UNKNOWN


def infer_claim_support_status(artifact: Any) -> str:
    existing = (
        artifact.get("claim_support_status")
        if isinstance(artifact, dict)
        else getattr(artifact, "claim_support_status", None)
    )
    if existing:
        return normalize_claim_support_status(existing, from_worker_payload=False)
    payload = _payload_of(artifact)
    if payload.get("claim_support_status"):
        return normalize_claim_support_status(
            payload.get("claim_support_status"), from_worker_payload=True
        )
    return CLAIM_SUPPORT_UNKNOWN


def hydrate_artifact_fields(artifact: Any) -> None:
    """Fill missing status fields on a frozen Artifact (compat mapping).

    Old records: ``confidence`` → ``worker_self_rating`` only. Never map a
    number into ``claim_support_status`` / independent support.
    """
    rating = worker_self_rating_of(artifact)
    if getattr(artifact, "worker_self_rating", None) is None and rating is not None:
        object.__setattr__(artifact, "worker_self_rating", rating)
    if not getattr(artifact, "execution_status", None):
        object.__setattr__(artifact, "execution_status", infer_execution_status(artifact))
    if not getattr(artifact, "grounding_status", None):
        object.__setattr__(artifact, "grounding_status", infer_grounding_status(artifact))
    if not getattr(artifact, "claim_support_status", None):
        object.__setattr__(
            artifact, "claim_support_status", infer_claim_support_status(artifact)
        )
    if not getattr(artifact, "criterion_status", None):
        object.__setattr__(artifact, "criterion_status", infer_criterion_status(artifact))


def status_fields(artifact: Any) -> dict[str, Any]:
    return {
        "execution_status": infer_execution_status(artifact),
        "grounding_status": infer_grounding_status(artifact),
        "claim_support_status": infer_claim_support_status(artifact),
        "criterion_status": infer_criterion_status(artifact),
        "worker_self_rating": worker_self_rating_of(artifact),
    }


def format_status_label(artifact: Any) -> str:
    """Human label for CLI/dashboard — statuses, not xx% / green bars."""
    fields = status_fields(artifact)
    parts = []
    for key in (
        "execution_status",
        "grounding_status",
        "claim_support_status",
        "criterion_status",
    ):
        value = fields.get(key)
        if value and value != "unknown":
            parts.append(f"{key}={value}")
    return " ".join(parts) if parts else "status=unknown"


def verification_accepts(artifact: Any) -> bool:
    if _type_of(artifact) != ArtifactType.VERIFICATION:
        return False
    payload = _payload_of(artifact)
    result = str(payload.get("result") or "").strip().lower()
    return result in _PASSED


def _points_at_finding(verification: Any, finding: Any) -> bool:
    payload = _payload_of(verification)
    evidence = list(
        verification.get("evidence")
        if isinstance(verification, dict)
        else getattr(verification, "evidence", None)
        or []
    )
    check = str(payload.get("check") or "")
    haystack = " ".join([check, *evidence])
    finding_id = (
        finding.get("id") if isinstance(finding, dict) else getattr(finding, "id", None)
    )
    claim = str(_payload_of(finding).get("claim") or "").strip()
    if finding_id and str(finding_id) in haystack:
        return True
    if claim and claim in haystack:
        return True
    return False


def finding_has_independent_support(
    finding: Any,
    *,
    peers: Optional[Sequence[Any]] = None,
    store: Any = None,
) -> bool:
    """True only for named independent support — never self-rating/confidence.

    Independent means a same-task accepting VERIFICATION in ``peers`` / the
    store, or a PM-persisted ``claim_support_status=independently_supported``.
    Worker payload aliases are coerced to ``worker_asserted`` and do not pass.
    """
    if infer_claim_support_status(finding) == CLAIM_SUPPORT_INDEPENDENT:
        return True
    if infer_grounding_status(finding) == GROUNDING_GROUNDED:
        return True
    task_id = (
        finding.get("task_id")
        if isinstance(finding, dict)
        else getattr(finding, "task_id", None)
    )
    job_id = (
        finding.get("job_id")
        if isinstance(finding, dict)
        else getattr(finding, "job_id", None)
    )
    candidates: list[Any] = list(peers or [])
    if store is not None and job_id:
        try:
            candidates.extend(store.list_artifacts(job_id))
        except Exception:
            pass
    seen: set[int] = set()
    pointed = False
    same_task_accept = False
    for artifact in candidates:
        marker = id(artifact)
        if marker in seen:
            continue
        seen.add(marker)
        if not verification_accepts(artifact):
            continue
        other_task = (
            artifact.get("task_id")
            if isinstance(artifact, dict)
            else getattr(artifact, "task_id", None)
        )
        if task_id and other_task != task_id:
            continue
        same_task_accept = True
        if _points_at_finding(artifact, finding):
            pointed = True
            break
    # Same-task accept with no pointer still counts (existing cheap default).
    return pointed or same_task_accept


def independently_supported_ids(artifacts: Iterable[Any]) -> set[str]:
    items = list(artifacts)
    supported: set[str] = set()
    for artifact in items:
        kind = _type_of(artifact)
        if kind not in (
            ArtifactType.FINDING,
            ArtifactType.DECISION,
            ArtifactType.RISK,
        ):
            continue
        if finding_has_independent_support(artifact, peers=items):
            ident = (
                artifact.get("id")
                if isinstance(artifact, dict)
                else getattr(artifact, "id", None)
            )
            if ident:
                supported.add(str(ident))
    return supported


def durable_admission_allowed(
    artifact: Any,
    *,
    peers: Optional[Sequence[Any]] = None,
    store: Any = None,
) -> bool:
    """Memory / gist admission: independent support only. Self-rating is ignored."""
    rating = worker_self_rating_of(artifact)
    _ = rating  # explicitly unused — self-rating cannot admit
    kind = _type_of(artifact)
    if kind == ArtifactType.VERIFICATION:
        return verification_accepts(artifact)
    if kind in (ArtifactType.FINDING, ArtifactType.DECISION, ArtifactType.RISK):
        return finding_has_independent_support(artifact, peers=peers, store=store)
    return False
