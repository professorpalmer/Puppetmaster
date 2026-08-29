from __future__ import annotations

from pathlib import Path
from typing import Union

from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


def create_store(
    backend: str,
    state_dir: Union[Path, str],
    *,
    mode: str = "deferred",
) -> SwarmStore:
    """Build a coordination store.

    ``mode`` applies to the SQLite backend only:
    - ``deferred`` (default): construct only. Supervisor APIs such as
      ``create_job`` call ``ensure_schema``; dashboard listing must not
      rewrite a corrupt ``state.sqlite3`` on open.
    - ``ensure``: create dirs, DDL, migrate immediately.
    - ``attach`` (workers): PRAGMAs + schema_version assert; never CREATE.
    """
    if backend == "file":
        return SwarmStore(state_dir)
    if backend == "sqlite":
        store = SQLiteSwarmStore(state_dir)
        store._open_mode = mode
        if mode == "attach":
            store.attach()
        elif mode == "ensure":
            store.ensure_schema()
        elif mode == "deferred":
            pass
        else:
            raise ValueError(f"unsupported store mode: {mode}")
        return store
    raise ValueError(f"unsupported backend: {backend}")


def create_worker_store(backend: str, state_dir: Union[Path, str]) -> SwarmStore:
    """Open a store the way a worker process must: attach only."""
    if backend == "file":
        return SwarmStore(state_dir)
    return create_store(backend, state_dir, mode="attach")
