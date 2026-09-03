"""Supervisor host boot records and lifecycle-default event queries.

The supervisor writes ``host_boot.json`` once per process and fans
``host.started`` / ``host.recovered`` to live jobs. Workers never record.
``read_lifecycle_events`` wraps ``read_events_since`` without changing
store defaults. Fail-soft: recording and shutdown marking never raise.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from puppetmaster.models import JobStatus, new_id, now_iso
from puppetmaster.worker_fence import is_worker_process

HOST_EVENT_STARTED = "host.started"
HOST_EVENT_RECOVERED = "host.recovered"
HOST_LIFECYCLE_EVENTS = frozenset({HOST_EVENT_STARTED, HOST_EVENT_RECOVERED})
NOISY_TURN_EVENTS = frozenset({"task.lease_renewed", "run.heartbeat", "task.saved"})

HOST_BOOT_FILENAME = "host_boot.json"
JOB_STALLED_EVENT = "job.stalled"

_LIVE_JOB_STATUSES = frozenset({JobStatus.RUNNING, JobStatus.STITCHING})
_INCLUDE_MODES = frozenset({"lifecycle", "quiet", "all"})

# Process-idempotency: one boot record (and fanout) per store root per process.
_recorded_roots: set[str] = set()
_atexit_roots: set[str] = set()
_last_records: dict[str, "HostStartRecord"] = {}


@dataclass(frozen=True)
class HostStartRecord:
    kind: str
    reason: str
    boot_id: str
    pid: int
    host: str
    fanned_out: int
    skipped: int
    idempotent: bool
    started_at: str = ""


def reset_host_start_guard() -> None:
    """Tests only: allow another ``record_host_start`` in this process."""
    _recorded_roots.clear()
    _last_records.clear()


def classify_host_start(previous: Optional[dict[str, Any]]) -> tuple[str, str]:
    """Return ``(kind, reason)`` for a supervisor start given the last boot."""
    if previous is None or not isinstance(previous, dict):
        return HOST_EVENT_STARTED, "first"
    if previous.get("clean_shutdown"):
        return HOST_EVENT_STARTED, "reboot"
    return HOST_EVENT_RECOVERED, "crash"


def filter_events(
    events: Iterable[dict[str, Any]],
    include: str = "lifecycle",
) -> list[dict[str, Any]]:
    """Filter event dicts. Unknown ``include`` values fall back to lifecycle."""
    mode = include if include in _INCLUDE_MODES else "lifecycle"
    if mode == "all":
        return events if isinstance(events, list) else list(events)
    if mode == "quiet":
        return [
            event
            for event in events
            if _event_name(event) not in NOISY_TURN_EVENTS
        ]
    return [event for event in events if _is_lifecycle_event(_event_name(event))]


def read_lifecycle_events(
    store: Any,
    job_id: str,
    since: int = 0,
    include: str = "lifecycle",
) -> list[dict[str, Any]]:
    """Read events after ``since`` and apply :func:`filter_events`."""
    events = store.read_events_since(job_id, since)
    return filter_events(events, include=include)


def record_host_start(store: Any) -> Optional[HostStartRecord]:
    """Write a supervisor boot record and fan it out to live jobs.

    No-op when ``PUPPETMASTER_WORKER=1``. Process-idempotent. Never raises.
    """
    try:
        if is_worker_process():
            return None
        root = Path(store.root)
        key = str(root)
        if key in _recorded_roots:
            cached = _last_records.get(key)
            if cached is not None:
                return HostStartRecord(
                    kind=cached.kind,
                    reason=cached.reason,
                    boot_id=cached.boot_id,
                    pid=cached.pid,
                    host=cached.host,
                    fanned_out=cached.fanned_out,
                    skipped=cached.skipped,
                    idempotent=True,
                    started_at=cached.started_at,
                )
            boot = _read_boot_record(store, root) or {}
            return _record_from_boot(boot, fanned_out=0, skipped=0, idempotent=True)
        # Claim before any store call that re-enters ``init`` / ``ensure_schema``.
        _recorded_roots.add(key)
        previous = _read_boot_record(store, root)
        kind, reason = classify_host_start(previous)
        record = {
            "boot_id": new_id("boot"),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": now_iso(),
            "kind": kind,
            "reason": reason,
            "clean_shutdown": False,
        }
        _write_boot_record(store, root, record)
        _register_clean_shutdown(root)
        fanned_out, skipped = _fanout_host_event(store, kind, record)
        result = HostStartRecord(
            kind=kind,
            reason=reason,
            boot_id=str(record["boot_id"]),
            pid=int(record["pid"]),
            host=str(record["host"]),
            fanned_out=fanned_out,
            skipped=skipped,
            idempotent=False,
            started_at=str(record["started_at"]),
        )
        _last_records[key] = result
        return result
    except Exception:
        return None


def mark_clean_shutdown(store: Any) -> Optional[dict[str, Any]]:
    """Set ``clean_shutdown`` on the boot record. Fail-soft; never raises."""
    try:
        root = Path(store.root)
        record = _read_boot_record(store, root)
        if record is None:
            return None
        record["clean_shutdown"] = True
        _write_boot_record(store, root, record)
        return record
    except Exception:
        return None


def _event_name(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    name = event.get("event")
    if name is None:
        return ""
    return str(name)


def _is_lifecycle_event(name: str) -> bool:
    return name.startswith("host.") or name == JOB_STALLED_EVENT


def _read_boot_record(store: Any, root: Path) -> Optional[dict[str, Any]]:
    path = root / HOST_BOOT_FILENAME
    if not path.is_file():
        return None
    reader = getattr(store, "read_json", None)
    try:
        if callable(reader):
            data = reader(path)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_boot_record(store: Any, root: Path, record: dict[str, Any]) -> None:
    path = root / HOST_BOOT_FILENAME
    writer = getattr(store, "write_json", None)
    if callable(writer):
        writer(path, record)
        return
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _register_clean_shutdown(root: Path) -> None:
    key = str(root)
    if key in _atexit_roots:
        return
    _atexit_roots.add(key)
    atexit.register(mark_clean_shutdown, SimpleNamespace(root=root))


def _record_from_boot(
    boot: dict[str, Any],
    *,
    fanned_out: int,
    skipped: int,
    idempotent: bool,
) -> HostStartRecord:
    return HostStartRecord(
        kind=str(boot.get("kind") or HOST_EVENT_STARTED),
        reason=str(boot.get("reason") or "first"),
        boot_id=str(boot.get("boot_id") or ""),
        pid=int(boot.get("pid") or 0),
        host=str(boot.get("host") or ""),
        fanned_out=fanned_out,
        skipped=skipped,
        idempotent=idempotent,
        started_at=str(boot.get("started_at") or ""),
    )


def _fanout_host_event(store: Any, kind: str, record: dict[str, Any]) -> tuple[int, int]:
    try:
        jobs = store.list_jobs()
    except Exception:
        return 0, 0
    payload = {
        "boot_id": record.get("boot_id"),
        "pid": record.get("pid"),
        "host": record.get("host"),
        "started_at": record.get("started_at"),
        "kind": kind,
        "reason": record.get("reason"),
    }
    fanned_out = 0
    skipped = 0
    for job in jobs:
        status = getattr(job, "status", None)
        if status not in _LIVE_JOB_STATUSES and str(status) not in {"running", "stitching"}:
            continue
        job_id = getattr(job, "id", None)
        if not job_id:
            continue
        try:
            store.emit(str(job_id), kind, payload)
            fanned_out += 1
        except Exception:
            skipped += 1
    return fanned_out, skipped
