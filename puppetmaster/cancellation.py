"""Scoped durable cooperative cancellation, with legacy standalone flags.

Execution-scoped workers consult their owning store and immutable lease binding.
Unscoped string flags remain only for callers running adapters outside a store.
A cooperative stop never proves that a remote provider or domain effect stopped.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar

_context = ContextVar("cancellation_scope", default=None)


@contextmanager
def cancellation_scope(store, task):
    from puppetmaster.store_contracts import task_binding
    from puppetmaster.identity import make_ref, read_identity
    from puppetmaster.projections import connection
    # Execution is a supervisor operation. Use its normal store session, which
    # can read committed WAL while another worker holds a write reservation.
    with connection(store) as c:
        ref = make_ref(store.root, task.job_id, read_identity(c, store.backend_name))
        store.validate_job_ref(ref, connection=c, strict=True)
    binding = task_binding(task)
    key = (ref, binding)
    with _lock:
        _active[key] = _active.get(key, 0) + 1
    token = _context.set((store, ref, binding))
    try:
        yield
    finally:
        try:
            if store.cancellation_pending(ref, binding):
                store.observe_cancellation(ref, binding, cleanup="unknown")
        finally:
            _context.reset(token)
            with _lock:
                _active[key] -= 1
                if not _active[key]:
                    del _active[key]
                    _scoped_cancelled.discard(key)


_lock = threading.Lock()
_cancelled: set = set()
_active: dict = {}
_scoped_cancelled: set = set()


class JobCancelled(Exception):
    """Raised inside a worker's stream to abort the in-flight provider call."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} cancelled")
        self.job_id = job_id
        # What the stopped attempt left: attempt id, dispatch receipt, live
        # log, elapsed seconds and the single observed session id. The stream
        # layer fills it so the stop receipt keeps the attempt's linkage.
        self.partial: dict = {}


def request_cancel(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    with _lock:
        scoped = _context.get()
        if scoped is not None and scoped[1].job_id == jid:
            _scoped_cancelled.update(key for key in _active if key[0] == scoped[1])
        else:
            matches = [key for key in _active if key[0].job_id == jid]
            # An unqualified legacy ID cannot select between different stores.
            if len({key[0].state_id for key in matches}) == 1:
                _scoped_cancelled.update(matches)
            elif not matches:
                _cancelled.add(jid)


def is_cancelled(job_id: str) -> bool:
    scoped = _context.get()
    if scoped is not None:
        store, ref, binding = scoped
        with _lock:
            legacy = (ref, binding) in _scoped_cancelled
        return job_id == ref.job_id and (legacy or store.cancellation_pending(ref, binding))
    jid = (job_id or "").strip()
    if not jid:
        return False
    with _lock:
        return jid in _cancelled


def clear_cancel(job_id: str) -> None:
    with _lock:
        jid = (job_id or "").strip()
        _cancelled.discard(jid)
        _scoped_cancelled.difference_update([key for key in _scoped_cancelled if key[0].job_id == jid])


def check_cancellation():
    scoped = _context.get()
    if scoped is not None and is_cancelled(scoped[1].job_id):
        raise JobCancelled(scoped[1].job_id)
