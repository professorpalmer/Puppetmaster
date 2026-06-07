"""Effort rollups (#7) and durable-state garbage collection (#8).

A single logical effort (a migration, a feature) routinely spans many git
worktrees, and each worktree hashes to its own per-project state dir. That
gives two problems this module solves:

- **Rollup (#7):** there's no single view of "what did this *effort* cost /
  produce" — you'd have to fan a loop over every worktree DB by hand. Tag jobs
  with an effort id (``PUPPETMASTER_EFFORT_ID`` / ``run --effort``) and
  :func:`rollup_stores` aggregates jobs, artifacts, estimated cost, and token
  usage across every project state dir.

- **GC (#8):** finished-job state just piles up — dozens of state dirs with no
  way to reap them. :func:`gc_terminal_jobs` deletes durable state for old
  *terminal* jobs (complete/failed/stalled). It is dry-run by default and never
  touches a live job or a user's git worktree, only the state Puppetmaster owns.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from puppetmaster.models import ArtifactType, JobStatus, now_iso, parse_iso

_TERMINAL_STATUSES = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.STALLED}
_EFFORT_FILENAME = "effort.json"


# --- effort tagging (#7) ------------------------------------------------------


def current_effort_id() -> Optional[str]:
    """The effort id for newly created jobs, from ``PUPPETMASTER_EFFORT_ID``."""
    value = (os.environ.get("PUPPETMASTER_EFFORT_ID") or "").strip()
    return value or None


def tag_job_effort(store: Any, job_id: str, effort_id: Optional[str]) -> None:
    """Stamp ``job_id`` with ``effort_id`` via a job-dir sidecar. Best-effort;
    never raises into job creation."""
    if not effort_id:
        return
    try:
        store.write_json(
            store.job_dir(job_id) / _EFFORT_FILENAME,
            {"effort_id": effort_id, "tagged_at": now_iso()},
        )
    except Exception:
        pass


def job_effort_id(store: Any, job_id: str) -> Optional[str]:
    """Read a job's effort tag, or ``None`` if it was never tagged."""
    try:
        path = store.job_dir(job_id) / _EFFORT_FILENAME
        if path.is_file():
            return store.read_json(path).get("effort_id")
    except Exception:
        pass
    return None


# --- garbage collection (#8) --------------------------------------------------


def _job_timestamp(job: Any) -> Optional[str]:
    return job.completed_at or job.created_at


def gc_terminal_jobs(
    store: Any,
    *,
    older_than_days: float = 7.0,
    force: bool = False,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Reap durable state for terminal jobs finished more than ``older_than_days``
    ago. Returns one record per reaped (or reapable, when ``force`` is False)
    job. Only acts on complete/failed/stalled jobs — a live job is never gc'd.
    """
    now = now or datetime.now(timezone.utc)
    cutoff_seconds = max(0.0, older_than_days) * 86400.0
    reaped: list[dict] = []
    for job in store.list_jobs():
        if job.status not in _TERMINAL_STATUSES:
            continue
        stamp = _job_timestamp(job)
        try:
            age_seconds = (now - parse_iso(stamp)).total_seconds()
        except (TypeError, ValueError):
            continue
        if age_seconds < cutoff_seconds:
            continue
        reaped.append(
            {
                "job_id": job.id,
                "status": str(job.status),
                "age_days": round(age_seconds / 86400.0, 1),
                "goal": job.goal,
                "deleted": force,
            }
        )
        if force:
            try:
                store.delete_job(job.id)
            except Exception:
                reaped[-1]["deleted"] = False
    return reaped


# --- effort rollup (#7) -------------------------------------------------------


def _job_estimated_cost(artifacts: list) -> float:
    """Sum the router's per-task estimated cost, deduped by task (mirrors the
    `cost` command so a rerouted task isn't double-counted)."""
    total = 0.0
    seen: set = set()
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        task_id = artifact.task_id
        if task_id and task_id in seen:
            continue
        if task_id:
            seen.add(task_id)
        total += float((artifact.payload or {}).get("estimated_cost_usd") or 0.0)
    return total


def rollup_stores(stores: Iterable[Any], *, effort_id: Optional[str] = None) -> dict:
    """Aggregate jobs/artifacts/cost/tokens across ``stores``.

    When ``effort_id`` is given, only jobs tagged with it are included; when it
    is ``None``, every job is included and the distinct effort ids that were
    seen are reported so a caller can discover them.
    """
    from puppetmaster.usage import aggregate_token_usage

    total_jobs = 0
    total_artifacts = 0
    total_cost = 0.0
    jobs_by_status: dict[str, int] = {}
    efforts_seen: set = set()
    all_artifacts: list = []

    for store in stores:
        try:
            jobs = store.list_jobs()
        except Exception:
            continue
        for job in jobs:
            tag = job_effort_id(store, job.id)
            if effort_id is not None and tag != effort_id:
                continue
            if tag:
                efforts_seen.add(tag)
            total_jobs += 1
            jobs_by_status[str(job.status)] = jobs_by_status.get(str(job.status), 0) + 1
            try:
                artifacts = store.list_artifacts(job.id)
            except Exception:
                artifacts = []
            total_artifacts += len(artifacts)
            all_artifacts.extend(artifacts)
            total_cost += _job_estimated_cost(artifacts)

    return {
        "effort_id": effort_id,
        "jobs": total_jobs,
        "jobs_by_status": jobs_by_status,
        "artifacts": total_artifacts,
        "estimated_cost_usd": round(total_cost, 6),
        "token_usage": aggregate_token_usage(all_artifacts),
        "efforts_seen": sorted(efforts_seen),
    }
