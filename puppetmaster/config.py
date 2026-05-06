from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from typing import Any

from puppetmaster.adapters import ADAPTERS
from puppetmaster.workers import WorkerSpec


@dataclass(frozen=True)
class SwarmConfig:
    workers: list[WorkerSpec]
    lease_seconds: int = 5


def load_config(path: Union[Path, str]) -> SwarmConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    workers = [_worker_spec_from_dict(item) for item in data.get("workers", [])]
    if not workers:
        raise ValueError("config requires at least one worker")
    return SwarmConfig(
        workers=workers,
        lease_seconds=int(data.get("lease_seconds", 5)),
    )


def _worker_spec_from_dict(data: dict[str, Any]) -> WorkerSpec:
    role = data.get("role")
    instruction = data.get("instruction")
    if not role or not instruction:
        raise ValueError("each worker requires role and instruction")
    adapter = data.get("adapter", "local")
    if adapter not in ADAPTERS:
        raise ValueError(f"unsupported adapter: {adapter}")
    return WorkerSpec(
        role=role,
        instruction=instruction,
        adapter=adapter,
        payload=data.get("payload", {}),
        depends_on_roles=data.get("depends_on", []),
    )

