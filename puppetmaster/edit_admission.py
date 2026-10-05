"""Cooperative admission for tasks whose adapters may edit a workspace.

This module is intentionally a seam: the runtime owns the context lifetime and
therefore closes admission only after the adapter process has stopped.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

from .file_claims import (
    FileClaim,
    FileClaimConflict,
    FileClaimRegistry,
    default_file_claim_db_path,
    process_start_identity,
)


# How long a worker may queue behind another worker's live claim before it
# gives up. A holder keeps its claim for its WHOLE run, and the adapter wall
# timeout is 900s (``agentic.DEFAULT_IMPLEMENT_TIMEOUT_SECONDS``), so a 30s
# wait could only ever fire while the holder was still legitimately working --
# turning "queue behind a writer" into a failed task and a failed swarm.
DEFAULT_ADMISSION_WAIT_SECONDS = 900.0


class EditAdmissionTimeout(TimeoutError):
    """The bounded wait elapsed before a conflicting claim was available."""


@dataclass
class EditAdmissionOwner:
    store: Any
    job_id: str
    task: Any
    generation: Any
    lease: Any
    claims: tuple[FileClaim, ...]
    _registry: FileClaimRegistry
    _cwd: Path
    _stop: threading.Event
    _lost: threading.Event
    _heartbeat: Optional[threading.Thread] = None
    _closed: bool = False
    # Seconds spent queued behind conflicting claims before admission.
    waited_seconds: float = 0.0

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def check(self) -> bool:
        """Return whether this owner still fences every claimed path."""
        if self._closed or self.lost:
            return False
        for claim in self.claims:
            if not self._registry.renew(self._cwd, claim.path, claim.claim_id):
                self._lost.set()
                return False
        return True

    def close(self) -> None:
        """Stop renewal and release exact fencing tokens; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join()
        self._registry.release_many(self._cwd, [(c.path, c.claim_id) for c in self.claims])
        if self.claims:
            _emit(self.store, self.job_id, "edit_admission.released", self.task, self.claims)

    def __enter__(self) -> "EditAdmissionOwner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def edit_admission(store: Any, task: Any, worker_id: str) -> EditAdmissionOwner:
    """Acquire the task's effective write scope and renew until context exit."""
    payload = dict(getattr(task, "payload", None) or {})
    job_id = str(getattr(task, "job_id", ""))
    cwd = Path(payload.get("cwd") or Path.cwd()).expanduser().resolve()
    if not _adapter_may_write(str(getattr(task, "adapter", "")), payload):
        return _owner(store, task, worker_id, FileClaimRegistry(default_file_claim_db_path()), cwd, ())

    registry = FileClaimRegistry(default_file_claim_db_path())
    scope = payload.get("write_scope")
    if scope is None:
        requested = ["."]
    elif isinstance(scope, (str, Path)):
        requested = [scope]
    elif isinstance(scope, (list, tuple, set)):
        requested = list(scope)
        if not requested:
            # An explicitly EMPTY write_scope means "this worker writes
            # nothing" -- it must not be promoted to ["."], which turns the
            # narrowest possible declaration into the widest possible lock.
            return _owner(store, task, worker_id, registry, cwd, ())
    else:
        raise ValueError("write_scope must be a path or list of paths")
    identity, normalized = registry._claim_keys(cwd, requested)
    del identity  # normalization is deliberately done by the primitive
    ttl = _positive_number(payload.get("edit_claim_ttl_seconds", 2.0), "edit_claim_ttl_seconds")
    timeout = _positive_number(
        payload.get(
            "edit_admission_wait_seconds",
            payload.get("claim_wait_seconds", DEFAULT_ADMISSION_WAIT_SECONDS),
        ),
        "edit_admission_wait_seconds",
    )
    started = time.monotonic()
    deadline = started + timeout
    owner = "%s:%s:%s" % (worker_id, job_id, getattr(task, "id", "task"))
    while True:
        if _cancelled(store, task, payload):
            raise EditAdmissionTimeout("edit admission cancelled while waiting")
        try:
            claims = registry.acquire_many(
                cwd, normalized, owner, ttl, managed=True, owner_pid=os.getpid(),
                owner_start_identity=process_start_identity(),
            )
            break
        except FileClaimConflict as exc:
            _emit(store, job_id, "edit_admission.waiting", task, (), worker_id=worker_id,
                  path=exc.path, owner=exc.owner)
            if time.monotonic() >= deadline:
                raise EditAdmissionTimeout(
                    "timed out waiting for edit admission: %s (held by %s, waited %.0fs)"
                    % (exc.path, getattr(exc, "owner", "unknown"), timeout)
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    admission = _owner(store, task, worker_id, registry, cwd, tuple(claims), ttl=ttl)
    admission.waited_seconds = round(time.monotonic() - started, 3)
    _emit(store, job_id, "edit_admission.acquired", task, admission.claims, worker_id=worker_id,
          waited_seconds=admission.waited_seconds)
    return admission


def _owner(store, task, worker_id, registry, cwd, claims, ttl=2.0):
    stop, lost = threading.Event(), threading.Event()
    owner = EditAdmissionOwner(
        store=store, job_id=str(getattr(task, "job_id", "")), task=task,
        generation=getattr(task, "generation", None), lease=getattr(task, "lease_id", None),
        claims=tuple(claims), _registry=registry, _cwd=cwd, _stop=stop, _lost=lost,
    )
    if claims:
        def renew_loop() -> None:
            while not stop.wait(max(0.05, ttl / 3.0)):
                if not owner.check():
                    _emit(store, owner.job_id, "edit_admission.lost", task, claims)
                    return
                _emit(store, owner.job_id, "edit_admission.renewed", task, claims)
        owner._heartbeat = threading.Thread(target=renew_loop, name="edit-admission", daemon=True)
        owner._heartbeat.start()
    return owner


def _adapter_may_write(adapter: str, payload: dict) -> bool:
    """Same write matrix as ``spec_edits_files`` / dirty-tree guards."""
    try:
        from puppetmaster.platform_lock import canonicalize_adapter
        adapter = canonicalize_adapter(adapter)
    except Exception:
        adapter = str(adapter).lower()
    from puppetmaster.write_intent import adapter_may_write
    return adapter_may_write(adapter, payload)


def _cancelled(store, task, payload) -> bool:
    callback = payload.get("edit_admission_cancelled")
    if callable(callback) and callback():
        return True
    try:
        from puppetmaster.cancellation import is_cancelled
        return bool(is_cancelled(str(getattr(task, "job_id", ""))))
    except Exception:
        return False


def _emit(store, job_id, event, task, claims, **extra) -> None:
    if store is None or not hasattr(store, "emit"):
        return
    payload = {
        "task_id": str(getattr(task, "id", "")),
        "worker_id": str(extra.pop("worker_id", "")),
        "paths": [claim.path for claim in claims],
        "generation": getattr(task, "generation", None),
        "lease_id": getattr(task, "lease_id", None),
        **extra,
    }
    try:
        store.emit(job_id, event, payload)
    except Exception:
        pass


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("%s must be positive" % name)
    return float(value)
