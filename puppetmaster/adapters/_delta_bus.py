"""A process-local token-delta bus for streaming agentic worker output.

Token streaming is a UX concern that must not couple the adapter to any
particular host. Rather than thread a callable through the JSON-serializable
``Task.payload`` (which cannot carry a function across a subprocess boundary),
an in-process caller -- e.g. the Marionette harness running inline workers --
registers a sink keyed by ``worker_id`` before the run, and the agentic adapter
looks it up at run start. When no sink is registered the adapter uses its normal
blocking path, so nothing changes for subprocess workers, the CLI, or tests.

The contract is deliberately tiny: a sink is ``callable(kind, text)`` where
``kind`` is ``"text"`` or ``"reasoning"`` and ``text`` is the incremental chunk.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

DeltaSink = Callable[[str, str], None]  # (kind, text)
BroadcastSink = Callable[[str, str, str], None]  # (worker_id, kind, text)

_lock = threading.Lock()
_sinks: "dict[str, DeltaSink]" = {}
_broadcast: "Optional[BroadcastSink]" = None


def register_delta_sink(worker_id: str, sink: DeltaSink) -> None:
    """Register a streaming sink for ``worker_id`` (overwrites any prior one)."""
    if not worker_id or sink is None:
        return
    with _lock:
        _sinks[worker_id] = sink


def unregister_delta_sink(worker_id: str) -> None:
    """Drop the sink for ``worker_id`` if present. Safe to call unconditionally."""
    with _lock:
        _sinks.pop(worker_id, None)


def set_broadcast_sink(sink: Optional[BroadcastSink]) -> None:
    """Set (or clear, with ``None``) a fallback sink used for any worker that has
    no per-worker sink. A swarm caller that can't know worker ids in advance --
    the Orchestrator mints them at run time -- registers a broadcast sink that
    receives every worker's deltas as ``(worker_id, kind, text)`` so tokens stay
    attributable to a specific worker/card even when several stream at once.
    """
    global _broadcast
    with _lock:
        _broadcast = sink


def delta_sink_for(worker_id: str) -> Optional[DeltaSink]:
    """Return a ``(kind, text)`` sink for ``worker_id``: a per-worker sink wins,
    else the broadcast sink bound to this ``worker_id``, else ``None`` (the
    adapter then uses its blocking, non-streaming path)."""
    with _lock:
        per_worker = _sinks.get(worker_id)
        broadcast = _broadcast
    if per_worker is not None:
        return per_worker
    if broadcast is not None:
        return lambda kind, text: broadcast(worker_id, kind, text)
    return None
