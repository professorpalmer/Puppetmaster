from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from puppetmaster.models import Artifact, ArtifactType, parse_iso
from puppetmaster.usage import aggregate_token_usage
from puppetmaster.scm_observe import derive_attention

# Useful operator-facing outputs (not transport/meta). PATCH counts so implement
# jobs are not scored as zero-typed when they only shipped a diff.
_TYPED = {
    ArtifactType.FINDING,
    ArtifactType.RISK,
    ArtifactType.DECISION,
    ArtifactType.PATCH,
    ArtifactType.GIST,
}
_EMPTY_MARKER = "empty-or-unstructured"
_EMPTY_FAILURE = "empty_or_unstructured"


def build_job_receipt(store: Any, job_id: str) -> dict[str, Any]:
    """Return objective run-efficiency telemetry for one job.

    This is deliberately not a stitched summary: no subjective accepted/rejected
    labels, no prose synthesis. It only aggregates durable job/task/artifact
    state so operators can spot context and transport tax at a glance.
    """
    job = store.get_job(job_id)
    tasks = store.list_tasks(job_id)
    artifacts = store.list_artifacts(job_id)
    token_usage = aggregate_token_usage(artifacts)
    by_type = Counter(str(artifact.type) for artifact in artifacts)
    typed_total = sum(by_type[str(kind)] for kind in _TYPED)
    degraded_tasks = _degraded_task_ids(artifacts)
    empty_tasks = _empty_or_unstructured_task_ids(artifacts)
    stdout_salvage = _stdout_salvage_count(artifacts)
    elapsed = _elapsed_seconds(job.created_at, job.completed_at)
    total_tokens = int(token_usage.get("total_tokens") or 0)
    drift = _estimate_drift(artifacts, total_tokens)
    return {
        "job_id": job_id,
        "status": str(job.status),
        "elapsed_seconds": elapsed,
        "tasks": {
            "total": len(tasks),
            "complete": _task_status_count(tasks, "complete"),
            "failed": _task_status_count(tasks, "failed"),
            "blocked": _task_status_count(tasks, "blocked"),
            "degraded": len(degraded_tasks),
        },
        "artifacts": {
            "total": len(artifacts),
            "typed_total": typed_total,
            "by_type": dict(sorted(by_type.items())),
        },
        "signals": {
            "empty_or_unstructured": len(empty_tasks),
            "stdout_salvage": stdout_salvage,
        },
        "tokens": token_usage,
        "estimate_drift": drift,
        "delivery": _delivery(store, job_id, tasks, artifacts),
        "host_observations": _host_observations(store, job_id, artifacts),
        "attention": derive_attention(store, job_id),
        "efficiency": {
            "tokens_per_typed_artifact": round(total_tokens / typed_total, 3) if typed_total else None,
            "degraded_rate": round(len(degraded_tasks) / len(tasks), 3) if tasks else 0.0,
        },
    }


def _estimate_drift(artifacts: list[Artifact], measured_tokens: int) -> dict[str, Any]:
    from puppetmaster.cost import final_routing_artifacts

    routes = final_routing_artifacts(artifacts)
    estimated = sum(
        int((artifact.payload or {}).get("estimated_tokens_in") or 0)
        + int((artifact.payload or {}).get("estimated_tokens_out") or 0)
        for artifact in routes.values()
    )
    return {
        "route_estimated_tokens": estimated,
        "measured_usage_tokens": measured_tokens,
        "token_ratio": round(measured_tokens / estimated, 6) if estimated else None,
    }


def _delivery(store: Any, job_id: str, tasks: list[Any], artifacts: list[Artifact]) -> dict[str, Any]:
    from puppetmaster.delivery import delivery_verdict
    from puppetmaster.quality import assess_run_quality

    quality = assess_run_quality(artifacts)
    stale = [task.id for task in tasks if store.is_task_stale(task)]
    return delivery_verdict(
        store.get_job(job_id).status,
        quality=quality.get("quality"),
        stale_tasks=stale,
        incomplete_tasks=any(str(task.status) != "complete" for task in tasks),
        required_artifacts=bool(artifacts),
    )


def _task_status_count(tasks: list[Any], status: str) -> int:
    return sum(1 for task in tasks if str(getattr(task, "status", "")) == status)


def _degraded_task_ids(artifacts: list[Artifact]) -> set[str]:
    return {
        artifact.task_id
        for artifact in artifacts
        if artifact.type == ArtifactType.VERIFICATION
        and str((artifact.payload or {}).get("result") or "").lower() == "degraded"
    }


def _empty_or_unstructured_task_ids(artifacts: list[Artifact]) -> set[str]:
    out: set[str] = set()
    for artifact in artifacts:
        payload = artifact.payload or {}
        evidence = " ".join(str(item) for item in (artifact.evidence or []))
        failure = str(payload.get("failure") or "")
        if _EMPTY_FAILURE in failure or _EMPTY_MARKER in evidence:
            out.add(artifact.task_id)
    return out


def _stdout_salvage_count(artifacts: list[Artifact]) -> int:
    count = 0
    for artifact in artifacts:
        payload = artifact.payload or {}
        evidence = " ".join(str(item) for item in (artifact.evidence or []))
        if _EMPTY_MARKER not in evidence and _EMPTY_FAILURE not in str(payload.get("failure") or ""):
            continue
        if any(key in payload for key in ("stdout_excerpt", "stdout_capture", "last_message_capture")):
            count += 1
    return count


def _host_observations(store: Any, job_id: str, artifacts: list[Artifact]) -> dict[str, Any]:
    """Host-observed land/ship/merge vs worker claims (v1.22.37)."""
    from puppetmaster.metr_seams import (
        HOST_DELIVERY_KINDS,
        delivery_claim_support_status,
        is_worker_delivery_claim,
        list_host_observations,
    )

    observations = list_host_observations(store, job_id)
    worker_claims = []
    for artifact in artifacts:
        if not is_worker_delivery_claim(artifact):
            continue
        worker_claims.append(
            {
                "artifact_id": artifact.id,
                "claim_support_status": delivery_claim_support_status(
                    artifact, store
                ),
            }
        )
    return {
        "kinds": sorted(HOST_DELIVERY_KINDS),
        "observed": observations,
        "worker_claims": worker_claims,
    }


def record_host_delivery_observation(
    store: Any,
    job_id: str,
    kind: str,
    *,
    evidence: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Idempotent host observation of shipped/merged/released/landed."""
    from puppetmaster.metr_seams import record_host_observation

    return record_host_observation(store, job_id, kind, evidence=evidence)


def _elapsed_seconds(created_at: str, completed_at: Optional[str]) -> Optional[float]:
    if not created_at or not completed_at:
        return None
    try:
        return round((parse_iso(completed_at) - parse_iso(created_at)).total_seconds(), 3)
    except Exception:
        return None
