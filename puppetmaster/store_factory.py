from __future__ import annotations

from pathlib import Path

from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


def create_store(backend: str, state_dir: Path | str) -> SwarmStore:
    if backend == "file":
        return SwarmStore(state_dir)
    if backend == "sqlite":
        return SQLiteSwarmStore(state_dir)
    raise ValueError(f"unsupported backend: {backend}")

