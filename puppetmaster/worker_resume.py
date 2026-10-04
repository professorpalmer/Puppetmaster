"""Resolve an opt-in worker session resume against recorded receipts.

A revision task (changed input for work a prior worker already did) is a new
task with its own attempt and receipts; only the provider conversation is
reused. The request lives on the task payload as ``resume_from`` (a prior
job/task/role in the same store) or ``resume_session_id`` (an id the caller
kept itself). Resolution is stamped back as ``payload["resume"]`` once, so
retries reuse the same record instead of re-resolving.
"""
from __future__ import annotations

from typing import Any, Optional

from puppetmaster.models import ArtifactType

RESUMABLE_ADAPTERS = {"codex": "thread_id", "claude-code": "session_id"}


def resume_requested(payload: Optional[dict]) -> bool:
    payload = payload or {}
    return bool(payload.get("resume_from") or payload.get("resume_session_id"))


def task_resume_record(payload: Optional[dict], adapter: str) -> Optional[dict]:
    """The resume record an adapter should honor, resolving explicit ids when unstamped.

    Tasks created through the orchestrator already carry ``payload["resume"]``.
    Direct adapter callers without a store can still pass ``resume_session_id``.
    """
    payload = payload or {}
    record = payload.get("resume")
    if isinstance(record, dict):
        return record
    return resolve_worker_resume(None, payload, adapter)


def resolved_resume(record: Optional[dict], adapter: str) -> Optional[dict]:
    """``record`` when it resumes a session on ``adapter``, else None for a fresh session."""
    if (
        isinstance(record, dict)
        and record.get("status") == "resolved"
        and record.get("adapter") == adapter
        and record.get("session_id")
    ):
        return record
    return None


def resolve_worker_resume(store: Any, payload: Optional[dict], adapter: str) -> Optional[dict]:
    """Return the ``resume`` record for a task payload, or None when not requested.

    Never raises: any lookup failure becomes an ``unavailable`` record so the
    worker falls back to a fresh session with an honest receipt.
    """
    payload = payload or {}
    existing = payload.get("resume")
    if isinstance(existing, dict) and existing.get("status") in {"resolved", "unavailable"}:
        return existing
    if not resume_requested(payload):
        return None
    try:
        return _resolve(store, payload, str(adapter or ""))
    except Exception as exc:
        return _unavailable(payload, f"resume lookup failed: {type(exc).__name__}: {exc}")


def _request_echo(payload: dict) -> dict:
    echo: dict[str, Any] = {}
    if payload.get("resume_from"):
        echo["resume_from"] = payload.get("resume_from")
    if payload.get("resume_session_id"):
        echo["resume_session_id"] = str(payload.get("resume_session_id"))
    if payload.get("resume_adapter"):
        echo["resume_adapter"] = str(payload.get("resume_adapter"))
    return echo


def _unavailable(payload: dict, reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, **_request_echo(payload)}


def _resolved(adapter: str, session_id: str, job_id: Optional[str], task_id: Optional[str]) -> dict:
    return {
        "status": "resolved",
        "adapter": adapter,
        "session_id": session_id,
        "from_job_id": job_id,
        "from_task_id": task_id,
    }


def _resolve(store: Any, payload: dict, adapter: str) -> dict:
    if adapter not in RESUMABLE_ADAPTERS:
        return _unavailable(payload, f"adapter {adapter!r} does not support session resume")

    explicit = str(payload.get("resume_session_id") or "").strip()
    if explicit:
        requested_adapter = str(payload.get("resume_adapter") or adapter)
        if requested_adapter != adapter:
            return _unavailable(
                payload,
                f"resume_adapter {requested_adapter!r} does not match task adapter {adapter!r}",
            )
        return _resolved(adapter, explicit, None, None)

    request = payload.get("resume_from")
    if not isinstance(request, dict) or not str(request.get("job_id") or "").strip():
        return _unavailable(payload, "resume_from must be an object with a job_id")
    job_id = str(request["job_id"]).strip()
    if store is None:
        return _unavailable(payload, "no state store available to resolve resume_from")
    task_id = str(request.get("task_id") or "").strip() or None
    role = str(request.get("role") or "").strip() or None

    try:
        store.get_job(job_id)
    except Exception:
        return _unavailable(payload, f"job {job_id!r} not found")
    tasks = list(store.list_tasks(job_id))
    if task_id:
        matches = [task for task in tasks if task.id == task_id]
    elif role:
        matches = [task for task in tasks if task.role == role]
    else:
        matches = [task for task in tasks if task.adapter == adapter]
    if not matches:
        target = f"task {task_id!r}" if task_id else f"role {role!r}" if role else f"a {adapter} task"
        return _unavailable(payload, f"{target} not found in job {job_id!r}")
    prior = max(matches, key=lambda task: str(task.updated_at or task.created_at or ""))
    if prior.adapter != adapter:
        return _unavailable(
            payload,
            f"prior task {prior.id!r} ran on {prior.adapter!r}, not {adapter!r}",
        )

    verifications = [
        artifact
        for artifact in store.list_artifacts(job_id)
        if artifact.task_id == prior.id
        and artifact.type == ArtifactType.VERIFICATION
        and (artifact.payload or {}).get("adapter") == adapter
    ]
    if not verifications:
        return _unavailable(payload, f"prior task {prior.id!r} has no {adapter} verification receipt")
    latest = max(verifications, key=lambda artifact: str(artifact.created_at or ""))
    receipt = latest.payload or {}
    if adapter == "codex" and receipt.get("ephemeral") is True:
        return _unavailable(
            payload,
            f"prior codex task {prior.id!r} ran ephemeral and was not persisted; "
            "launch with ephemeral=false to make it resumable",
        )
    session_id = str(receipt.get(RESUMABLE_ADAPTERS[adapter]) or "").strip()
    if not session_id:
        return _unavailable(
            payload,
            f"prior task {prior.id!r} recorded no {RESUMABLE_ADAPTERS[adapter]}",
        )
    return _resolved(adapter, session_id, job_id, prior.id)
