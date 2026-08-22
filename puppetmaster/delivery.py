"""Shared lifecycle-to-delivery contract for CLI and MCP observers."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from puppetmaster.models import DeliveryVerdict, JobStatus, TaskStatus


def delivery_verdict(
    status: Union[JobStatus, str],
    *,
    quality: Optional[str] = None,
    stale_tasks: Iterable[str] = (),
    incomplete_tasks: bool = False,
    required_artifacts: bool = True,
) -> dict[str, Any]:
    """Return one conservative operator verdict without hiding raw status."""
    raw = str(status)
    stale = list(stale_tasks)
    if raw in {str(JobStatus.QUEUED), str(JobStatus.RUNNING), str(JobStatus.STITCHING)}:
        verdict = DeliveryVerdict.PENDING
    elif raw in {str(JobStatus.FAILED), str(JobStatus.STALLED), str(JobStatus.CANCELLED)}:
        verdict = DeliveryVerdict.BLOCKED
    elif (
        stale
        or incomplete_tasks
        or not required_artifacts
        or quality is None
        or quality in {"blocked", "empty"}
    ):
        verdict = DeliveryVerdict.BLOCKED
    elif quality in {"degraded", "untrusted"}:
        verdict = DeliveryVerdict.DEGRADED
    else:
        verdict = DeliveryVerdict.DELIVERED
    return {
        "verdict": str(verdict),
        "successful": verdict == DeliveryVerdict.DELIVERED,
        "status": raw,
        "quality": quality,
        "stale_task_ids": stale,
        "incomplete_tasks": bool(incomplete_tasks),
        "required_artifacts": bool(required_artifacts),
    }
