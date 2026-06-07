from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def seconds_from_now(seconds: int) -> str:
    return datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + seconds,
        tz=timezone.utc,
    ).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class JobStatus(StringEnum):
    QUEUED = "queued"
    RUNNING = "running"
    STITCHING = "stitching"
    COMPLETE = "complete"
    FAILED = "failed"
    # A job whose orchestrator died (or whose work wedged) with no live worker
    # leasing tasks. Distinct from RUNNING so a dead job is never represented
    # as live, and distinct from FAILED so it stays recoverable.
    STALLED = "stalled"


class TaskStatus(StringEnum):
    BLOCKED = "blocked"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ArtifactType(StringEnum):
    FINDING = "finding"
    DECISION = "decision"
    PATCH = "patch"
    VERIFICATION = "verification"
    RISK = "risk"
    MEMORY_SUMMARY = "memory_summary"
    ROUTING = "routing"
    # The verdict of a non-bypassable completion gate (post-condition / drift
    # ratchet / commit check). A failed GATE forces the task to FAILED, so the
    # agent can never report COMPLETE over work that regressed a baseline or
    # left its output uncommitted.
    GATE = "gate"


@dataclass(frozen=True)
class Job:
    goal: str
    id: str = field(default_factory=lambda: new_id("job"))
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None


@dataclass(frozen=True)
class Task:
    job_id: str
    role: str
    instruction: str
    id: str = field(default_factory=lambda: new_id("task"))
    status: TaskStatus = TaskStatus.QUEUED
    adapter: str = "local"
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    attempts: int = 0
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None


@dataclass(frozen=True)
class AgentRun:
    job_id: str
    task_id: str
    role: str
    worker_id: str
    id: str = field(default_factory=lambda: new_id("run"))
    status: TaskStatus = TaskStatus.RUNNING
    started_at: str = field(default_factory=now_iso)
    heartbeat_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None


@dataclass(frozen=True)
class Artifact:
    job_id: str
    task_id: str
    type: ArtifactType
    created_by: str
    payload: dict[str, Any]
    confidence: float
    evidence: list[str]
    id: str = field(default_factory=lambda: new_id("artifact"))
    created_at: str = field(default_factory=now_iso)
    sha256: Optional[str] = None

    def validate(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("artifact confidence must be between 0 and 1")
        if not self.payload:
            raise ValueError("artifact payload must not be empty")
        if not self.evidence:
            raise ValueError(f"{self.type} artifacts require evidence")
        required_keys = {
            ArtifactType.FINDING: ["claim"],
            ArtifactType.DECISION: ["decision", "why"],
            ArtifactType.PATCH: ["change", "files"],
            ArtifactType.VERIFICATION: ["check", "result"],
            ArtifactType.RISK: ["risk", "mitigation"],
            ArtifactType.MEMORY_SUMMARY: ["summary"],
            ArtifactType.ROUTING: ["model_id", "adapter", "policy"],
            ArtifactType.GATE: ["gate", "passed"],
        }
        for key in required_keys.get(self.type, []):
            if key not in self.payload:
                raise ValueError(f"{self.type} artifacts require payload.{key}")


@dataclass(frozen=True)
class MemoryRecord:
    scope: str
    statement: str
    evidence: list[str]
    source_artifacts: list[str]
    confidence: float
    promoted: bool = True
    id: str = field(default_factory=lambda: new_id("memory"))
    created_at: str = field(default_factory=now_iso)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, StringEnum):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def job_from_dict(data: dict[str, Any]) -> Job:
    return Job(
        id=data["id"],
        goal=data["goal"],
        status=JobStatus(data["status"]),
        created_at=data["created_at"],
        completed_at=data.get("completed_at"),
    )


def task_from_dict(data: dict[str, Any]) -> Task:
    return Task(
        id=data["id"],
        job_id=data["job_id"],
        role=data["role"],
        instruction=data["instruction"],
        status=TaskStatus(data["status"]),
        adapter=data.get("adapter", "local"),
        payload=data.get("payload", {}),
        depends_on=data.get("depends_on", []),
        attempts=data.get("attempts", 0),
        lease_owner=data.get("lease_owner"),
        lease_expires_at=data.get("lease_expires_at"),
        created_at=data["created_at"],
        updated_at=data.get("updated_at", data["created_at"]),
        completed_at=data.get("completed_at"),
    )


def artifact_from_dict(data: dict[str, Any]) -> Artifact:
    return Artifact(
        id=data["id"],
        job_id=data["job_id"],
        task_id=data["task_id"],
        type=ArtifactType(data["type"]),
        created_by=data["created_by"],
        payload=data["payload"],
        confidence=data["confidence"],
        evidence=data["evidence"],
        created_at=data["created_at"],
        sha256=data.get("sha256"),
    )

