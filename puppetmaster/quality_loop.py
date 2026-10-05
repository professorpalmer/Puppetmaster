"""Opt-in post-edit cleanup and bounded same-adapter review repair.

Off by default. Review loops never multiply with model-tier auto-escalation:
a task with ``payload.review_loop`` is repaired in place (same adapter/model,
``allow_dirty``, reviewer reasons injected) and skipped by
``_reroute_failed_review``.
"""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Sequence

from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus, now_iso
from puppetmaster.worker_resume import repair_resume_record


DEFAULT_REVIEW_LOOP_LIMIT = 3
DEFAULT_CLEANUP_MAX_USD = 0.25


def review_loop_enabled(payload: Optional[dict]) -> bool:
    return bool((payload or {}).get("review_loop"))


def review_loop_limit(payload: Optional[dict]) -> int:
    raw = (payload or {}).get("review_loop_limit", DEFAULT_REVIEW_LOOP_LIMIT)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_REVIEW_LOOP_LIMIT
    return max(1, min(value, 10))


def cleanup_enabled(payload: Optional[dict]) -> bool:
    return bool((payload or {}).get("cleanup"))


def cleanup_max_usd(payload: Optional[dict]) -> float:
    raw = (payload or {}).get("cleanup_max_usd", DEFAULT_CLEANUP_MAX_USD)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CLEANUP_MAX_USD
    return max(0.0, value)


def review_reasons_from_artifacts(artifacts: Sequence[Artifact], task_id: str) -> List[str]:
    reasons: List[str] = []
    for artifact in artifacts:
        if artifact.task_id != task_id:
            continue
        payload = artifact.payload or {}
        if artifact.type == ArtifactType.GATE and payload.get("kind") == "review":
            if payload.get("passed"):
                continue
            for key in ("reason", "summary", "feedback", "notes"):
                text = str(payload.get(key) or "").strip()
                if text:
                    reasons.append(text)
            for item in payload.get("findings") or []:
                if isinstance(item, str) and item.strip():
                    reasons.append(item.strip())
                elif isinstance(item, dict):
                    text = str(item.get("claim") or item.get("reason") or "").strip()
                    if text:
                        reasons.append(text)
        if artifact.type == ArtifactType.FINDING and "review" in (artifact.created_by or ""):
            claim = str((payload.get("claim") or "")).strip()
            if claim:
                reasons.append(claim)
    return reasons


def maybe_requeue_review_repair(store: Any, job_id: str) -> List[Task]:
    """Requeue review-rejected tasks that opted into ``review_loop``."""
    artifacts = store.list_artifacts(job_id)
    rejected = _failed_review_ids(artifacts)
    repaired: List[Task] = []
    for task in store.list_tasks(job_id):
        if task.id not in rejected or task.status != TaskStatus.FAILED:
            continue
        payload = dict(task.payload or {})
        if not review_loop_enabled(payload):
            continue
        attempts = int(payload.get("review_loop_attempts") or 0)
        if attempts >= review_loop_limit(payload):
            continue
        reasons = review_reasons_from_artifacts(artifacts, task.id)
        payload["review_loop_attempts"] = attempts + 1
        payload["allow_dirty"] = True
        payload["review_repair"] = True
        payload["review_reasons"] = reasons
        resume = repair_resume_record(task, artifacts)
        if resume is None:
            payload.pop("resume", None)
        else:
            payload["resume"] = resume
        extra = ""
        if reasons:
            extra = "\n\nReviewer rejected the previous edit. Fix these:\n" + "\n".join(
                "- %s" % reason for reason in reasons
            )
        requeued = replace(
            task,
            status=TaskStatus.QUEUED,
            instruction=str(task.instruction or "") + extra,
            payload=payload,
            attempts=0,
            lease_owner=None,
            lease_expires_at=None,
            completed_at=None,
            updated_at=now_iso(),
        )
        store.save_task(requeued)
        store.emit(
            job_id,
            "quality.review_repair",
            {
                "task_id": task.id,
                "attempt": attempts + 1,
                "limit": review_loop_limit(payload),
                "reason_count": len(reasons),
                "resumed": bool(resume and resume.get("status") == "resolved"),
            },
        )
        repaired.append(requeued)
    return repaired


def run_cleanup_pass(
    task: Task,
    artifacts: List[Artifact],
    *,
    store: Any = None,
    runner: Optional[Any] = None,
) -> List[Artifact]:
    """Best-effort lint --fix on edited paths. Never fails the implement."""
    if not cleanup_enabled(task.payload):
        return artifacts
    paths = edited_paths(artifacts, task)
    if not paths:
        return artifacts + [_cleanup_verification(task, "skipped", "no edited paths", paths)]
    cwd = Path((task.payload or {}).get("cwd") or ".").resolve()
    command = list((task.payload or {}).get("cleanup_command") or ["ruff", "check", "--fix"])
    command = command + [str(path) for path in paths]
    execute = runner or _run_command
    try:
        completed = execute(command, cwd=str(cwd))
        ok = int(getattr(completed, "returncode", 1) or 0) == 0
        detail = _tail(getattr(completed, "stdout", "") or getattr(completed, "stderr", "") or "")
        status = "passed" if ok else "skipped"
        note = "lint --fix applied" if ok else "cleanup failed; implement kept"
    except Exception as exc:
        status = "skipped"
        note = "cleanup failed; implement kept"
        detail = str(exc)
    extra = _cleanup_verification(task, status, note, paths, detail=detail)
    if store is not None:
        try:
            store.emit(
                task.job_id,
                "quality.cleanup",
                {"task_id": task.id, "status": status, "paths": [str(p) for p in paths]},
            )
        except Exception:
            pass
    return artifacts + [extra]


def edited_paths(artifacts: Sequence[Artifact], task: Task) -> List[str]:
    paths: List[str] = []
    seen = set()
    for artifact in artifacts:
        if artifact.task_id != task.id:
            continue
        payload = artifact.payload or {}
        for key in ("changed_files", "untracked_files", "paths", "files"):
            for item in payload.get(key) or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    paths.append(text)
        change = payload.get("change")
        if isinstance(change, dict):
            for item in change.get("files") or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    paths.append(text)
    return paths


def _failed_review_ids(artifacts: Sequence[Artifact]) -> set:
    latest = {}
    for artifact in artifacts:
        payload = artifact.payload or {}
        if artifact.type != ArtifactType.GATE or payload.get("kind") != "review":
            continue
        previous = latest.get(artifact.task_id)
        if previous is None or artifact.created_at > previous[0]:
            latest[artifact.task_id] = (artifact.created_at, bool(payload.get("passed")))
    return {task_id for task_id, (_, passed) in latest.items() if not passed}


def _cleanup_verification(task: Task, result: str, note: str, paths: Sequence[str], detail: str = "") -> Artifact:
    from puppetmaster.adapters import verification_artifact

    return verification_artifact(
        task=task,
        worker_id="quality-cleanup",
        adapter=str(task.adapter or "local"),
        check="cleanup",
        result=result,
        confidence=0.7,
        evidence=["quality:cleanup"],
        payload={
            "kind": "cleanup",
            "note": note,
            "paths": list(paths),
            "detail": detail,
        },
    )


def _run_command(command: Sequence[str], *, cwd: str) -> Any:
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _tail(text: str, limit: int = 800) -> str:
    text = str(text or "")
    return text[-limit:]
