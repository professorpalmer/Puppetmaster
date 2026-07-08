"""In-process cooperative job cancellation.

A host embedding the orchestrator (e.g. Marionette's backend running inline
agentic workers) calls :func:`request_cancel` with a job id; every agentic
worker on that job stops at its next cancellation point:

* mid-stream -- the delta sink raises :class:`JobCancelled`, aborting the
  provider HTTP stream within one chunk (near-instant), and
* per-turn -- the agent loop checks the flag before each provider call.

Python threads cannot be force-killed, so this registry is the kill switch:
cheap to check, safe to set from any thread, and keyed by job id so one flag
stops every worker in the swarm.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_cancelled: set = set()


class JobCancelled(Exception):
    """Raised inside a worker's stream to abort the in-flight provider call."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} cancelled")
        self.job_id = job_id


def request_cancel(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    with _lock:
        _cancelled.add(jid)


def is_cancelled(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    with _lock:
        return jid in _cancelled


def clear_cancel(job_id: str) -> None:
    with _lock:
        _cancelled.discard((job_id or "").strip())
