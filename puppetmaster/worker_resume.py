"""Resolve an opt-in worker session resume against recorded receipts.

A revision task (changed input for work a prior worker already did) is a new
task with its own attempt and receipts; only the provider conversation is
reused. The request lives on the task payload as ``resume_from`` (a prior
job/task/role in the same store) or ``resume_session_id`` (an id the caller
kept itself). Resolution is stamped back as ``payload["resume"]`` once, so
retries reuse the same record instead of re-resolving.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from puppetmaster.models import ArtifactType

RESUMABLE_ADAPTERS = {"codex": "thread_id", "claude-code": "session_id", "fx": "session_id"}
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


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
        if record.get("status") == "resolved" and record.get("adapter") != adapter:
            return {
                **record,
                "status": "unavailable",
                "reason": (
                    f"task now runs on {adapter!r}; the resolved session belongs to "
                    f"{record.get('adapter')!r}"
                ),
            }
        return record
    return resolve_worker_resume(None, payload, adapter)


def claim_resumed_session(record: Optional[dict], claimed: dict, role: str) -> Optional[dict]:
    """Allow one task per job to continue a codex thread; ``codex exec resume`` has no fork."""
    if not isinstance(record, dict) or record.get("status") != "resolved" or record.get("adapter") != "codex":
        return record
    session_id = str(record.get("session_id"))
    holder = claimed.get(session_id)
    if holder is not None:
        return {
            **record,
            "status": "unavailable",
            "reason": (
                f"codex thread {session_id} is already resumed by role {holder!r} in this "
                "job; codex resume continues a thread in place"
            ),
        }
    claimed[session_id] = role
    return record


def repair_resume_record(task: Any, artifacts: Any) -> Optional[dict]:
    """Resume a review-repaired task's own latest provider session, or None to run fresh.

    The repair is the same task continuing its own edit, so the record points at
    the session that produced the rejected diff. A session that cannot be
    resumed yields an ``unavailable`` record so the receipt says why it ran fresh.
    """
    payload = dict(getattr(task, "payload", None) or {})
    adapter = str(getattr(task, "adapter", "") or "")
    if payload.get("review_repair_resume") is False or adapter not in RESUMABLE_ADAPTERS:
        return None
    key = RESUMABLE_ADAPTERS[adapter]
    receipts = [
        artifact
        for artifact in artifacts
        if artifact.task_id == task.id
        and artifact.type == ArtifactType.VERIFICATION
        and (artifact.payload or {}).get("adapter") == adapter
        and str((artifact.payload or {}).get(key) or "").strip()
    ]
    if not receipts:
        return None
    receipt = max(receipts, key=lambda artifact: str(artifact.created_at or "")).payload or {}
    if adapter == "codex" and receipt.get("ephemeral") is True:
        return _unavailable(payload, "the rejected codex attempt ran ephemeral and was not persisted")
    return _resolved_or_missing(
        payload, adapter, str(receipt[key]).strip(), getattr(task, "job_id", None), task.id
    )


def session_on_disk(adapter: str, session_id: str) -> Optional[bool]:
    """Whether the CLI's local session store holds ``session_id``; None when there is no store to check."""
    if adapter == "codex":
        from puppetmaster import codex_home

        homes = codex_home.session_homes()
        if not any((home / "sessions").is_dir() for home in homes):
            return None
        return codex_home.home_for_session(session_id) is not None
    if adapter != "claude-code":
        # fx keeps no local session store that Puppetmaster can read.
        return None
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
    pattern = f"*/{session_id}.jsonl"
    if not root.is_dir():
        return None
    return next(root.glob(pattern), None) is not None


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


def _resolved_or_missing(
    payload: dict, adapter: str, session_id: str, job_id: Optional[str], task_id: Optional[str]
) -> dict:
    if not _SESSION_ID.match(session_id):
        return _unavailable(payload, f"invalid {adapter} session id {session_id!r}")
    if session_on_disk(adapter, session_id) is False:
        return _unavailable(
            payload,
            f"{adapter} session {session_id} is not in the local session store; it may have been pruned",
        )
    return _resolved(adapter, session_id, job_id, task_id)


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
        return _resolved_or_missing(payload, adapter, explicit, None, None)

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
        # A worker's VERDICT line is stored as a verification artifact of the
        # same adapter but carries no session id; it is not the receipt.
        and (artifact.payload or {}).get("kind") != "worker_verdict"
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
    return _resolved_or_missing(payload, adapter, session_id, job_id, prior.id)
