"""Durable, followable token-delta stream for agentic workers.

The in-process :mod:`_delta_bus` streams tokens to an *inline* host (e.g. the
Marionette harness) through a callback, but a Python callback can't cross a
subprocess boundary -- so a worker spawned by the Orchestrator or an MCP run
could not stream its tokens anywhere. This module closes that gap: it persists
the same ``(kind, text)`` deltas to an append-only NDJSON file under the job's
state dir, so any follower -- a CLI ``deltas --follow`` tail, an MCP live feed,
another process -- can watch a subprocess worker think in real time.

The file lives beside the existing streamed sidecar logs at
``<state_dir>/jobs/<job_id>/tasks/<task_id>/agentic_deltas.ndjson`` (the same
convention as :mod:`_streaming`), one JSON object per line::

    {"ts": 1720000000.0, "worker_id": "w1", "kind": "text", "text": "chunk"}

When no state dir is in scope (direct adapter unit tests, ad-hoc runs) the
writer is a no-op, so nothing changes for callers that don't opt in.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional

from puppetmaster.fs_permissions import mkdir_private, open_private
from puppetmaster.models import Task
from puppetmaster.redaction import redact_secrets

_DELTA_FILE = "agentic_deltas.ndjson"


def _resolve_state_dir() -> Optional[Path]:
    """The active state dir from ``PUPPETMASTER_STATE_DIR`` (exported by
    ``worker_runtime`` after resolving --state-dir), or ``None`` when no state
    dir is in scope -- mirroring :func:`_streaming._resolve_sidecar_state_dir`.
    """
    raw = os.environ.get("PUPPETMASTER_STATE_DIR")
    if not raw:
        return None
    try:
        return Path(raw)
    except (TypeError, ValueError):
        return None


def delta_file_path(state_dir: Path, job_id: str, task_id: str) -> Path:
    """The NDJSON delta path for a task, beside its streamed sidecar logs."""
    return Path(state_dir) / "jobs" / job_id / "tasks" / task_id / _DELTA_FILE


class DurableDeltaWriter:
    """Append-only NDJSON sink for a single worker's token deltas.

    Construction never raises: a filesystem error simply yields an inert writer
    whose :meth:`emit` is a no-op, so streaming a run can never crash it.
    """

    def __init__(self, path: Path, worker_id: str) -> None:
        self.path = path
        self._worker_id = worker_id
        self._handle = None
        try:
            mkdir_private(path.parent)
            self._handle = open(
                open_private(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND),
                "w", encoding="utf-8", errors="replace", closefd=True,
            )
        except OSError:
            self._handle = None

    @classmethod
    def for_task(cls, task: Task, worker_id: str) -> "Optional[DurableDeltaWriter]":
        """A writer for ``task`` when a state dir is in scope, else ``None``."""
        state_dir = _resolve_state_dir()
        if state_dir is None:
            return None
        return cls(delta_file_path(state_dir, task.job_id, task.id), worker_id)

    def emit(self, kind: str, text: str) -> None:
        """Append one redacted delta record. Silent on any write error."""
        if self._handle is None or not text:
            return
        record = {
            "ts": time.time(), "worker_id": self._worker_id,
            "kind": kind, "text": redact_secrets(text) or text,
        }
        try:
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()
        except OSError:
            pass

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


def iter_deltas(
    path: Path,
    *,
    follow: bool = False,
    idle_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> Iterator[dict]:
    """Yield delta records from an NDJSON delta file.

    Reads all currently-available records first. When ``follow`` is set it then
    tails the file for appended records until ``idle_timeout_seconds`` elapses
    with no new data (0 waits indefinitely) or the reader is interrupted. Safe
    to start before the file exists -- it waits for the worker to create it.
    """
    poll = max(0.02, poll_interval_seconds)
    offset = 0
    idle_deadline = (
        time.monotonic() + idle_timeout_seconds if idle_timeout_seconds > 0 else None
    )
    while True:
        produced = False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    if not line.endswith("\n"):
                        break  # partial trailing line -- reread from offset next pass
                    offset += len(line.encode("utf-8"))
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    produced = True
                    yield record
        except FileNotFoundError:
            pass
        if not follow:
            return
        if produced and idle_timeout_seconds > 0:
            idle_deadline = time.monotonic() + idle_timeout_seconds
        if idle_deadline is not None and time.monotonic() >= idle_deadline:
            return
        try:
            time.sleep(poll)
        except KeyboardInterrupt:
            return
