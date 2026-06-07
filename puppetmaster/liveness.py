"""Job liveness tracking and the stalled-job reaper.

A Puppetmaster job is driven by an orchestrator process. When that process is
backgrounded and reaped (``nohup ... &`` that the shell kills), or otherwise
dies mid-run, the job's ``job.json`` is left saying ``running`` forever even
though nothing is leasing its tasks. That is the worst kind of failure: a dead
job that lies about being alive.

This module closes that gap two ways:

* The orchestrator stamps a small ``orchestrator.json`` heartbeat into the job
  directory (pid + host + heartbeat time) so liveness is *checkable* rather than
  inferred.
* :func:`reap_stalled_jobs` scans live-looking jobs and, for any whose driver is
  provably gone (pid dead) or that has made no progress with no live lease for
  ``stall_after_seconds``, transitions it ``running -> stalled`` and emits a
  ``job.stalled`` event. Along the way it requeues lease-expired tasks so a
  still-live driver (or a fresh worker) can pick them up without a manual
  ``recover``.

Everything here is best-effort and never raises into a caller's hot path: a
reaper that itself crashes would be worse than a stale status line.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from puppetmaster.models import Job, JobStatus, TaskStatus, now_iso, parse_iso
from puppetmaster.store import SwarmStore

# How long a job may show no progress (no fresh orchestrator heartbeat, no new
# event) with no live task lease before the reaper calls it stalled. Generous
# by default so a legitimately slow single worker (a multi-minute agent run that
# only flushes at the end) is never reaped out from under itself.
DEFAULT_STALL_AFTER_SECONDS = 180

_HEARTBEAT_FILENAME = "orchestrator.json"


def stall_after_seconds_default() -> int:
    raw = os.environ.get("PUPPETMASTER_STALL_AFTER_SECONDS")
    if not raw:
        return DEFAULT_STALL_AFTER_SECONDS
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_STALL_AFTER_SECONDS
    return value if value > 0 else DEFAULT_STALL_AFTER_SECONDS


def _heartbeat_path(store: SwarmStore, job_id: str):
    return store.job_dir(job_id) / _HEARTBEAT_FILENAME


def record_orchestrator_heartbeat(
    store: SwarmStore, job_id: str, *, started: bool = False
) -> None:
    """Stamp (or refresh) the orchestrator liveness record for ``job_id``.

    Cheap enough to call on every worker batch. Never raises — a job that can't
    write its heartbeat simply falls back to event-staleness detection."""
    try:
        path = _heartbeat_path(store, job_id)
        existing = _read_record(store, job_id) if not started else None
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": (existing or {}).get("started_at") or now_iso(),
            "heartbeat_at": now_iso(),
        }
        store.write_json(path, record)
    except Exception:
        pass


def _read_record(store: SwarmStore, job_id: str) -> Optional[dict]:
    path = _heartbeat_path(store, job_id)
    try:
        if not path.exists():
            return None
        return store.read_json(path)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists. Signal 0 probes without delivering
    anything; EPERM means it exists but we don't own it (still alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class LivenessVerdict:
    dead: bool
    reason: str
    recovered_tasks: int = 0


def _has_live_lease(tasks, now: datetime) -> bool:
    for task in tasks:
        if task.status != TaskStatus.RUNNING or not task.lease_expires_at:
            continue
        try:
            if parse_iso(task.lease_expires_at) > now:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _latest_activity(store: SwarmStore, job: Job, record: Optional[dict]) -> datetime:
    """Most recent sign of life: orchestrator heartbeat, last event, or (as a
    floor) the job's creation time."""
    candidates: list[str] = [job.created_at]
    if record and record.get("heartbeat_at"):
        candidates.append(str(record["heartbeat_at"]))
    try:
        events = store.read_events(job.id)
        if events:
            candidates.append(str(events[-1].get("at")))
    except Exception:
        pass
    latest = None
    for raw in candidates:
        try:
            parsed = parse_iso(raw)
        except (TypeError, ValueError):
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest or datetime.now(timezone.utc)


def assess_job_liveness(
    store: SwarmStore,
    job: Job,
    *,
    stall_after_seconds: int,
    now: Optional[datetime] = None,
) -> LivenessVerdict:
    """Decide whether ``job`` is a dead-but-running job that should be stalled.

    Two independent signals, either of which is sufficient:

    * The orchestrator pid was recorded *on this host* and that process is gone.
    * No task holds a live lease AND nothing has happened (no heartbeat, no
      event) for longer than ``stall_after_seconds`` — covers a wedged or
      pid-recycled driver where the pid check alone wouldn't fire.
    """
    now = now or datetime.now(timezone.utc)
    record = _read_record(store, job.id)
    tasks = store.list_tasks(job.id)

    if record and record.get("host") == socket.gethostname():
        pid = record.get("pid")
        if isinstance(pid, int) and not _pid_alive(pid):
            return LivenessVerdict(dead=True, reason="orchestrator_pid_gone")

    if _has_live_lease(tasks, now):
        return LivenessVerdict(dead=False, reason="live_lease")

    idle_seconds = (now - _latest_activity(store, job, record)).total_seconds()
    if idle_seconds > stall_after_seconds:
        return LivenessVerdict(
            dead=True,
            reason=f"no_progress_for_{int(idle_seconds)}s",
        )
    return LivenessVerdict(dead=False, reason="recent_activity")


def liveness_summary(
    store: SwarmStore,
    job: Job,
    *,
    stall_after_seconds: Optional[int] = None,
    now: Optional[datetime] = None,
) -> dict:
    """A glanceable liveness report for a job: pid, whether it's alive, the
    heartbeat/idle age, and a human verdict. Lets ``status`` scream "this is
    dead" instead of quietly showing ``running`` for a wedged job (#9)."""
    stall_after = stall_after_seconds or stall_after_seconds_default()
    now = now or datetime.now(timezone.utc)
    record = _read_record(store, job.id) or {}
    tasks = store.list_tasks(job.id)
    pid = record.get("pid") if isinstance(record.get("pid"), int) else None
    same_host = record.get("host") == socket.gethostname()
    pid_alive = bool(pid and same_host and _pid_alive(pid))
    idle_seconds = int((now - _latest_activity(store, job, record)).total_seconds())
    live_lease = _has_live_lease(tasks, now)

    if pid and same_host and not pid_alive:
        verdict = "dead (orchestrator pid gone)"
    elif not live_lease and idle_seconds > stall_after:
        verdict = f"dead (no progress for {idle_seconds}s)"
    elif not live_lease and idle_seconds > stall_after // 2:
        verdict = f"stale (no progress for {idle_seconds}s; reaped at {stall_after}s)"
    else:
        verdict = "alive"

    return {
        "pid": pid,
        "host": record.get("host"),
        "pid_alive": pid_alive,
        "idle_seconds": idle_seconds,
        "live_lease": live_lease,
        "stall_after_seconds": stall_after,
        "verdict": verdict,
    }


def reap_stalled_jobs(
    store: SwarmStore,
    *,
    stall_after_seconds: Optional[int] = None,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Transition dead-but-running jobs to ``stalled`` and requeue their
    lease-expired tasks.

    Returns one dict per affected job describing what was done. Safe to call
    from any read-side command (status/jobs/wait) — it only acts on jobs whose
    driver is provably gone, and requeuing a stale task is idempotent.
    """
    stall_after = stall_after_seconds or stall_after_seconds_default()
    now = now or datetime.now(timezone.utc)
    reaped: list[dict] = []
    for job in store.list_jobs():
        if job.status not in {JobStatus.RUNNING, JobStatus.STITCHING}:
            continue
        # Auto-requeue any task whose lease expired (its worker is gone) so a
        # live driver can re-run it without a manual `recover` (#4).
        recovered = store.recover_stale_tasks(job.id)
        verdict = assess_job_liveness(
            store, job, stall_after_seconds=stall_after, now=now
        )
        if not verdict.dead:
            continue
        store.update_job_status(job.id, JobStatus.STALLED)
        store.emit(
            job.id,
            "job.stalled",
            {
                "reason": verdict.reason,
                "requeued_tasks": len(recovered),
                "stall_after_seconds": stall_after,
            },
        )
        reaped.append(
            {
                "job_id": job.id,
                "reason": verdict.reason,
                "requeued_tasks": len(recovered),
            }
        )
    return reaped
