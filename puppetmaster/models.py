from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union
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
    # Deliberately stopped by an operator (cooperative cancellation or a host
    # UI cancel). Terminal, distinct from FAILED: nothing went wrong.
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETE,
        JobStatus.FAILED,
        JobStatus.STALLED,
        JobStatus.CANCELLED,
    }
)

# Cost-final jobs may own an immutable receipt. STALLED is recoverable
# and is terminal for liveness, not for selected-model economics.
COST_FINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETE,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


class DeliveryVerdict(StringEnum):
    """Operator-facing delivery result, distinct from raw lifecycle status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class JobRef:
    """Opaque continuation identity returned by asynchronous launchers."""

    job_id: str
    state_id: str

    def as_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "state_id": self.state_id}


def is_terminal_job_status(status: JobStatus) -> bool:
    """Return whether ``status`` represents a job that will do no more work."""
    return status in TERMINAL_JOB_STATUSES


def is_cost_final_job_status(status: JobStatus) -> bool:
    """True when ``status`` may own an immutable ``Job.cost_receipt``.

    ``stalled`` is recoverable and must never stamp or retain a receipt.
    """
    return status in COST_FINAL_JOB_STATUSES


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
    # Compact verified discovery admitted for peer shared-context injection.
    # Peers see only admission=admitted gists; pending/rejected stay in the
    # store for tooling/MCP but are filtered at injection boundaries.
    GIST = "gist"


@dataclass(frozen=True)
class Job:
    goal: str
    label: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("job"))
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    # Optional caller-supplied idempotency key.  Older persisted jobs simply
    # omit both fields and remain fully readable.
    launch_key: Optional[str] = None
    launch_fingerprint: Optional[str] = None
    # v1.22.37 additive METR seams. Coordinator outlives workers: persist the
    # original contract (criteria + granted authority) and wait/hold state.
    # Older records omit these keys and remain fully readable.
    acceptance_criteria: Optional[list[str]] = None
    granted_authority: Optional[Any] = None
    wait_reason: Optional[str] = None
    subgraph_owner: Optional[str] = None
    subgraph_hold: Optional[str] = None
    # Coordinator-stamped selected-model economics at cost-final status.
    # Older records omit the key. JSON on the job row only — not a new
    # ArtifactType or store table. STALLED must not keep a receipt.
    cost_receipt: Optional[dict[str, Any]] = None


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
    # Monotonic per-claim token. Stamped fresh on every successful claim so a
    # worker that reuses the same ``lease_owner`` id after its lease was
    # reclaimed cannot fence a *newer* claim's terminal write or renewal.
    lease_id: Optional[str] = None
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
    # Additive #88 status split. ``confidence`` stays for stored-record compat
    # and maps to worker_self_rating. Statuses are the authoritative labels.
    execution_status: Optional[str] = None
    grounding_status: Optional[str] = None
    claim_support_status: Optional[str] = None
    criterion_status: Optional[str] = None
    worker_self_rating: Optional[float] = None

    def __post_init__(self) -> None:
        from puppetmaster.artifact_status import hydrate_artifact_fields

        hydrate_artifact_fields(self)

    def validate(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("artifact confidence must be between 0 and 1")
        if self.worker_self_rating is not None and not 0 <= self.worker_self_rating <= 1:
            raise ValueError("artifact worker_self_rating must be between 0 and 1")
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
            ArtifactType.GIST: ["claim", "source_artifact_ids", "admission"],
        }
        for key in required_keys.get(self.type, []):
            if key not in self.payload:
                raise ValueError(f"{self.type} artifacts require payload.{key}")
        if self.type == ArtifactType.GIST:
            admission = self.payload.get("admission")
            if admission not in ("pending", "admitted", "rejected"):
                raise ValueError(
                    "gist artifacts require payload.admission in "
                    "{pending, admitted, rejected}"
                )
            source_ids = self.payload.get("source_artifact_ids")
            if not isinstance(source_ids, list):
                raise ValueError(
                    "gist artifacts require payload.source_artifact_ids as a list"
                )


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


class GraphEdgeType(StringEnum):
    """Typed provenance / scheduling relation between graph nodes."""

    DEPENDS_ON = "depends_on"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    DERIVED_FROM = "derived_from"


class GraphNodeKind(StringEnum):
    TASK = "task"
    ARTIFACT = "artifact"


def graph_edge_identity(
    job_id: str,
    edge_type: Union[GraphEdgeType, str],
    from_kind: Union[GraphNodeKind, str],
    from_id: str,
    to_kind: Union[GraphNodeKind, str],
    to_id: str,
) -> str:
    """Stable, idempotent edge id from the typed endpoint tuple."""
    raw = "|".join(
        (
            str(job_id),
            str(edge_type),
            str(from_kind),
            str(from_id),
            str(to_kind),
            str(to_id),
        )
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"edge_{digest}"


@dataclass(frozen=True)
class GraphEdge:
    """Persisted typed edge for task/task and task/artifact provenance.

    Identity is derived from ``(job_id, type, from_*, to_*)`` so upserts are
    idempotent across file and SQLite backends.
    """

    job_id: str
    type: GraphEdgeType
    from_kind: GraphNodeKind
    from_id: str
    to_kind: GraphNodeKind
    to_id: str
    id: str = ""
    created_at: str = field(default_factory=now_iso)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id:
            return
        object.__setattr__(
            self,
            "id",
            graph_edge_identity(
                self.job_id,
                self.type,
                self.from_kind,
                self.from_id,
                self.to_kind,
                self.to_id,
            ),
        )


def make_graph_edge(
    *,
    job_id: str,
    type: Union[GraphEdgeType, str],
    from_kind: Union[GraphNodeKind, str],
    from_id: str,
    to_kind: Union[GraphNodeKind, str],
    to_id: str,
    created_at: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> GraphEdge:
    edge_type = GraphEdgeType(str(type))
    src_kind = GraphNodeKind(str(from_kind))
    dst_kind = GraphNodeKind(str(to_kind))
    return GraphEdge(
        id=graph_edge_identity(
            job_id, edge_type, src_kind, from_id, dst_kind, to_id
        ),
        job_id=job_id,
        type=edge_type,
        from_kind=src_kind,
        from_id=from_id,
        to_kind=dst_kind,
        to_id=to_id,
        created_at=created_at or now_iso(),
        meta=dict(meta or {}),
    )


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


def _job_status_or_stalled(raw: Any) -> JobStatus:
    """Coerce an unknown persisted status to STALLED instead of raising.

    Stores are written by multiple host versions (harness UIs, older CLIs);
    one row with a status this build doesn't know must not poison every
    ``list_jobs()`` reader into an empty feed."""
    try:
        return JobStatus(raw)
    except ValueError:
        return JobStatus.STALLED


def job_from_dict(data: dict[str, Any]) -> Job:
    raw_receipt = data.get("cost_receipt")
    return Job(
        id=data["id"],
        goal=data["goal"],
        label=data.get("label"),
        status=_job_status_or_stalled(data["status"]),
        created_at=data["created_at"],
        completed_at=data.get("completed_at"),
        launch_key=data.get("launch_key"),
        launch_fingerprint=data.get("launch_fingerprint"),
        acceptance_criteria=data.get("acceptance_criteria"),
        granted_authority=data.get("granted_authority"),
        wait_reason=data.get("wait_reason"),
        subgraph_owner=data.get("subgraph_owner"),
        subgraph_hold=data.get("subgraph_hold"),
        cost_receipt=raw_receipt if isinstance(raw_receipt, dict) else None,
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
        lease_id=data.get("lease_id"),
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
        execution_status=data.get("execution_status"),
        grounding_status=data.get("grounding_status"),
        claim_support_status=data.get("claim_support_status"),
        criterion_status=data.get("criterion_status"),
        worker_self_rating=data.get("worker_self_rating"),
    )


def graph_edge_from_dict(data: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        id=data.get("id")
        or graph_edge_identity(
            data["job_id"],
            data["type"],
            data["from_kind"],
            data["from_id"],
            data["to_kind"],
            data["to_id"],
        ),
        job_id=data["job_id"],
        type=GraphEdgeType(data["type"]),
        from_kind=GraphNodeKind(data["from_kind"]),
        from_id=data["from_id"],
        to_kind=GraphNodeKind(data["to_kind"]),
        to_id=data["to_id"],
        created_at=data.get("created_at") or now_iso(),
        meta=data.get("meta") or {},
    )

