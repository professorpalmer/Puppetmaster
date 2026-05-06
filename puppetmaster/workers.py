from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from puppetmaster.adapters import get_adapter
from puppetmaster.models import AgentRun, Artifact, Task, TaskStatus, now_iso


@dataclass(frozen=True)
class WorkerSpec:
    role: str
    instruction: str
    adapter: str = "local"
    payload: dict = field(default_factory=dict)
    depends_on_roles: list[str] = field(default_factory=list)


DEFAULT_WORKERS = [
    WorkerSpec(
        role="explore",
        instruction="Map the problem, extract constraints, and emit evidenced findings.",
    ),
    WorkerSpec(
        role="architect",
        instruction="Choose the smallest viable architecture and record explicit decisions.",
        depends_on_roles=["explore"],
    ),
    WorkerSpec(
        role="implement",
        instruction="Produce implementation artifacts and patch plans, not prose blobs.",
        depends_on_roles=["architect"],
    ),
    WorkerSpec(
        role="redteam",
        instruction="Find failure modes, stale assumptions, and missing verification.",
        depends_on_roles=["implement"],
    ),
    WorkerSpec(
        role="test",
        instruction="Convert claims into checks and verification results.",
        depends_on_roles=["implement"],
    ),
]


class LocalWorker:
    """Task executor used by real worker processes."""

    def __init__(self, role: str, worker_id: Optional[str] = None) -> None:
        self.role = role
        self.worker_id = worker_id or f"local-{role}"

    def run(self, task: Task, goal: str) -> tuple[AgentRun, list[Artifact]]:
        run = AgentRun(
            job_id=task.job_id,
            task_id=task.id,
            role=task.role,
            worker_id=self.worker_id,
            status=TaskStatus.COMPLETE,
            completed_at=now_iso(),
        )
        if task.adapter == "cursor":
            return run, get_adapter("cursor").run(task, goal, self.worker_id)
        if task.adapter == "shell":
            return run, get_adapter("shell").run(task, goal, self.worker_id)
        return run, get_adapter(task.adapter).run(task, goal, self.worker_id)


def specs_for_roles(roles: Optional[list[str]] = None) -> list[WorkerSpec]:
    if not roles:
        return DEFAULT_WORKERS
    known = {spec.role: spec for spec in DEFAULT_WORKERS}
    return [
        known.get(role, WorkerSpec(role=role, instruction=f"Run the {role} worker."))
        for role in roles
    ]

