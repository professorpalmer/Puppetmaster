"""METR control seams: graph dispatcher, no cross-job board, host truth.

v1.22.37 strengthens existing enqueue / gist / receipt / lease seams. Workers
propose bounded same-job children; coordinators own HOLD/VETO, job completion,
and host-observed land/ship/merge. Additive JSON only.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Optional, Sequence

from puppetmaster.artifact_status import (
    CLAIM_SUPPORT_INDEPENDENT,
    CLAIM_SUPPORT_WORKER_ASSERTED,
    infer_claim_support_status,
)
from puppetmaster.models import ArtifactType, now_iso

WAIT_EXTERNAL = "waiting_external"
WAIT_USER = "waiting_user"
WAIT_REASONS = frozenset({WAIT_EXTERNAL, WAIT_USER})

HOLD_STATE = "hold"
VETO_STATE = "veto"
SUBGRAPH_HOLD_STATES = frozenset({HOLD_STATE, VETO_STATE})

REASON_WORKER_PROTOCOL = "worker_protocol_refused"
REASON_GATE_FAILED = "gate_failed"
REASON_CROSS_JOB = "cross_job_refused"
REASON_PARENT_MISMATCH = "parent_mismatch"
REASON_NEW_JOB = "new_job_refused"
REASON_SUBGRAPH_WRITER = "subgraph_writer_refused"
REASON_SUBGRAPH_HOLD = "subgraph_hold"
REASON_SUBGRAPH_VETO = "subgraph_veto"
REASON_WORKER_JOB_COMPLETE = "worker_job_complete_refused"

ACTOR_WORKER = "worker"
ACTOR_COORDINATOR = "coordinator"
ACTOR_HOST = "host"

COORDINATOR_ACTORS = frozenset({ACTOR_COORDINATOR, ACTOR_HOST, "orchestrator", "operator"})

PROTOCOL_KEYS = frozenset(
    {
        "recruit",
        "hold",
        "veto",
        "mailbox",
        "mailbox_protocol",
        "open_mailbox",
        "peer_recruit",
        "coordination_protocol",
        "recruit_peers",
    }
)
PROTOCOL_ROLES = frozenset(
    {
        "recruit",
        "hold",
        "veto",
        "mailbox",
        "mailbox_protocol",
        "coordinator-mailbox",
        "peer-recruit",
    }
)
_SHIP_ROLES = frozenset({"merge", "ship", "release", "land", "publish"})
HOST_DELIVERY_KINDS = frozenset({"shipped", "merged", "released", "landed"})
HOST_OBSERVATIONS_FILENAME = "host_observations.json"

_PROTOCOL_INSTRUCTION_RE = re.compile(
    r"\b("
    r"HOLD|VETO|"
    r"recruit(?:ing)?(?:\s+peers)?|"
    r"mailbox\s+protocol|"
    r"open(?:ing)?\s+(?:a\s+)?mailbox"
    r")\b",
    re.IGNORECASE,
)
_SHIP_INSTRUCTION_RE = re.compile(
    r"\b(merge|ship|release|land|publish)\b",
    re.IGNORECASE,
)


def normalize_actor(actor: Optional[str]) -> str:
    value = str(actor or "").strip().lower()
    if value in COORDINATOR_ACTORS:
        return ACTOR_COORDINATOR
    if value == ACTOR_WORKER or value.startswith("worker"):
        return ACTOR_WORKER
    if not value:
        return ACTOR_COORDINATOR
    return ACTOR_WORKER


def is_worker_actor(actor: Optional[str]) -> bool:
    return normalize_actor(actor) == ACTOR_WORKER


def normalize_wait_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower().replace("-", "_")
    if raw in WAIT_REASONS:
        return raw
    return None


def _payload_of(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = value.get("payload") if "payload" in value else value
    else:
        payload = getattr(value, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _role_of(value: Any) -> str:
    if isinstance(value, dict):
        return _lower_text(value.get("role"))
    return _lower_text(getattr(value, "role", None))


def is_coordination_protocol_text(text: Any) -> bool:
    body = str(text or "").strip()
    if not body:
        return False
    return _PROTOCOL_INSTRUCTION_RE.search(body) is not None


def is_coordination_protocol_payload(value: Any) -> bool:
    """True when a worker payload looks like recruit/HOLD/VETO/mailbox protocol."""
    if value is None:
        return False
    if isinstance(value, str):
        return is_coordination_protocol_text(value)
    payload = _payload_of(value)
    role = _role_of(value) or _lower_text(payload.get("role"))
    if role.replace("_", "-") in PROTOCOL_ROLES or role in PROTOCOL_ROLES:
        return True
    keys = {str(key).strip().lower().replace("-", "_") for key in payload.keys()}
    if keys & PROTOCOL_KEYS:
        return True
    for field in ("instruction", "goal", "claim", "protocol", "action", "command"):
        if is_coordination_protocol_text(payload.get(field)):
            return True
    if is_coordination_protocol_text(getattr(value, "instruction", None)):
        return True
    nested = payload.get("protocol") or payload.get("mailbox") or payload.get("recruit")
    if isinstance(nested, dict) and nested:
        return True
    if nested in (True, "true", "yes"):
        return True
    return False


def is_ship_or_merge_proposal(proposal: Any) -> bool:
    role = _role_of(proposal) or _lower_text(_payload_of(proposal).get("role"))
    if role in _SHIP_ROLES:
        return True
    payload = _payload_of(proposal)
    instruction = payload.get("instruction") or payload.get("goal") or ""
    if isinstance(proposal, dict):
        instruction = proposal.get("instruction") or proposal.get("goal") or instruction
    return _SHIP_INSTRUCTION_RE.search(str(instruction or "")) is not None


def proposal_requests_new_job(proposal: Any) -> bool:
    if not isinstance(proposal, dict):
        payload = _payload_of(proposal)
    else:
        payload = proposal
    if payload.get("create_job") or payload.get("new_job"):
        return True
    if payload.get("open_job") or payload.get("spawn_job"):
        return True
    return False


def proposal_foreign_job_id(proposal: Any, job_id: str) -> Optional[str]:
    if not isinstance(proposal, dict):
        payload = _payload_of(proposal)
    else:
        payload = proposal
    raw = payload.get("job_id") or payload.get("target_job_id")
    if raw is None:
        return None
    text = str(raw).strip()
    if text and text != job_id:
        return text
    return None


def proposal_foreign_parent_id(proposal: Any, parent_task_id: str) -> Optional[str]:
    if not isinstance(proposal, dict):
        return None
    raw = proposal.get("parent_task_id") or proposal.get("parent")
    if raw is None:
        return None
    text = str(raw).strip()
    if text and text != parent_task_id:
        return text
    return None


def gate_failed_for_task(store: Any, job_id: str, task_id: str) -> bool:
    try:
        artifacts = store.list_artifacts(job_id)
    except Exception:
        return False
    for artifact in artifacts:
        kind = getattr(artifact, "type", None)
        if kind != ArtifactType.GATE and str(kind) != "gate":
            continue
        if getattr(artifact, "task_id", None) != task_id:
            continue
        payload = _payload_of(artifact)
        if payload.get("passed") is False:
            return True
    return False


def artifact_is_failed_gate(artifact: Any) -> bool:
    kind = getattr(artifact, "type", None)
    if kind != ArtifactType.GATE and str(kind) != "gate":
        return False
    payload = _payload_of(artifact)
    return payload.get("passed") is False


def is_worker_asserted_finding(artifact: Any) -> bool:
    kind = getattr(artifact, "type", None)
    name = str(getattr(kind, "value", kind) or "").strip().lower()
    if name not in {"finding", "gist", "decision", "risk"}:
        return False
    status = infer_claim_support_status(artifact)
    return status != CLAIM_SUPPORT_INDEPENDENT


def independently_supported_artifact(artifact: Any) -> bool:
    return infer_claim_support_status(artifact) == CLAIM_SUPPORT_INDEPENDENT


def is_cross_job_injectable(
    artifact: Any,
    *,
    for_job_id: Optional[str],
) -> bool:
    """Cross-job inject only independently_supported, non-protocol findings."""
    if is_coordination_protocol_payload(artifact):
        return False
    artifact_job = getattr(artifact, "job_id", None)
    if isinstance(artifact, dict):
        artifact_job = artifact.get("job_id") or artifact_job
    if for_job_id and artifact_job and str(artifact_job) != str(for_job_id):
        return independently_supported_artifact(artifact)
    return True


def filter_cross_job_listing(
    artifacts: Iterable[Any],
    *,
    for_job_id: Optional[str] = None,
) -> list[Any]:
    """Drop coordination-protocol rows and worker-asserted cross-job leaks."""
    kept: list[Any] = []
    for artifact in artifacts:
        if is_coordination_protocol_payload(artifact):
            continue
        artifact_job = getattr(artifact, "job_id", None)
        if isinstance(artifact, dict):
            artifact_job = artifact.get("job_id") or artifact_job
        if for_job_id and artifact_job and str(artifact_job) != str(for_job_id):
            if not independently_supported_artifact(artifact):
                continue
        elif for_job_id is None:
            # Effort-index / listing across jobs: only host-admitted findings/gists.
            kind = getattr(artifact, "type", None)
            name = str(getattr(kind, "value", kind) or "").strip().lower()
            if name in {"finding", "gist", "decision", "risk"}:
                if not independently_supported_artifact(artifact):
                    continue
        kept.append(artifact)
    return kept


def is_worker_delivery_claim(artifact: Any) -> bool:
    payload = _payload_of(artifact)
    for kind in HOST_DELIVERY_KINDS:
        value = payload.get(kind)
        if value in (True, "true", "yes", kind, "ok"):
            return True
        status = _lower_text(payload.get("status") or payload.get("delivery"))
        if status == kind:
            return True
        claim = _lower_text(
            payload.get("claim") or payload.get("decision") or payload.get("change")
        )
        if re.search(rf"\b{kind}\b", claim):
            return True
    return False


def _observation_fingerprint(kind: str, evidence: Sequence[str]) -> str:
    raw = json.dumps(
        {"kind": kind, "evidence": list(evidence)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _observations_path(store: Any, job_id: str):
    return store.job_dir(job_id) / HOST_OBSERVATIONS_FILENAME


def list_host_observations(store: Any, job_id: str) -> list[dict[str, Any]]:
    path = _observations_path(store, job_id)
    try:
        if not path.is_file():
            return []
        data = store.read_json(path)
    except Exception:
        return []
    if isinstance(data, dict):
        rows = data.get("observations") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def host_observed_kind(store: Any, job_id: str, kind: str) -> bool:
    wanted = str(kind or "").strip().lower()
    for row in list_host_observations(store, job_id):
        if str(row.get("kind") or "").strip().lower() == wanted:
            return True
    return False


def record_host_observation(
    store: Any,
    job_id: str,
    kind: str,
    *,
    evidence: Optional[Sequence[str]] = None,
    source: str = ACTOR_HOST,
) -> dict[str, Any]:
    """Idempotent host observation (git SHA / CI / PyPI / gate).

    Repeating the same kind + evidence digest does not double-apply.
    """
    normalized = str(kind or "").strip().lower()
    if normalized not in HOST_DELIVERY_KINDS:
        raise ValueError(f"unsupported host observation kind: {kind!r}")
    evidence_list = [str(item) for item in (evidence or []) if str(item).strip()]
    fingerprint = _observation_fingerprint(normalized, evidence_list)
    existing = list_host_observations(store, job_id)
    for row in existing:
        if row.get("fingerprint") == fingerprint:
            return {**row, "idempotent": True}
        if (
            str(row.get("kind") or "").strip().lower() == normalized
            and list(row.get("evidence") or []) == evidence_list
        ):
            return {**row, "idempotent": True}
    observation = {
        "kind": normalized,
        "evidence": evidence_list,
        "source": source or ACTOR_HOST,
        "observed_at": now_iso(),
        "fingerprint": fingerprint,
        "claim_support_status": CLAIM_SUPPORT_INDEPENDENT,
    }
    payload = {"observations": existing + [observation]}
    store.write_json(_observations_path(store, job_id), payload)
    try:
        store.emit(
            job_id,
            "host.observed",
            {
                "kind": normalized,
                "fingerprint": fingerprint,
                "evidence": evidence_list,
            },
        )
    except Exception:
        pass
    return observation


def delivery_claim_support_status(artifact: Any, store: Any = None) -> str:
    """Worker ship/merge/release claims stay worker_asserted until host-observed."""
    if not is_worker_delivery_claim(artifact):
        return infer_claim_support_status(artifact)
    job_id = getattr(artifact, "job_id", None)
    if isinstance(artifact, dict):
        job_id = artifact.get("job_id") or job_id
    if store is not None and job_id:
        payload = _payload_of(artifact)
        for kind in HOST_DELIVERY_KINDS:
            if host_observed_kind(store, str(job_id), kind):
                status = _lower_text(payload.get("status") or payload.get("delivery"))
                if status == kind or payload.get(kind) or re.search(
                    rf"\b{kind}\b",
                    _lower_text(payload.get("claim") or payload.get("decision") or ""),
                ):
                    return CLAIM_SUPPORT_INDEPENDENT
    return CLAIM_SUPPORT_WORKER_ASSERTED


def subgraph_root_ids(task: Any, task_map: dict[str, Any]) -> set[str]:
    """Walk depends_on toward roots (cycle-safe). Isolated tasks are their own root."""
    roots: set[str] = set()
    seen: set[str] = set()
    stack = [task]
    while stack:
        current = stack.pop()
        current_id = getattr(current, "id", None)
        if not current_id or current_id in seen:
            continue
        seen.add(current_id)
        depends = list(getattr(current, "depends_on", None) or [])
        parents = [task_map[dep] for dep in depends if dep in task_map]
        if not parents:
            roots.add(current_id)
            continue
        stack.extend(parents)
    return roots


def foreign_active_writer(
    store: Any,
    task: Any,
    worker_id: str,
    *,
    task_map: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Return another worker's id if they hold an active lease in this subgraph."""
    job_id = getattr(task, "job_id", None)
    if not job_id:
        return None
    tasks = list(task_map.values()) if task_map else store.list_tasks(job_id)
    mapping = {item.id: item for item in tasks}
    mapping.setdefault(task.id, task)
    roots = subgraph_root_ids(task, mapping)
    selected: set[str] = set()
    for root_id in roots:
        selected |= store.consumer_closure(job_id, [root_id])
    selected.add(task.id)
    for item in tasks:
        if item.id not in selected:
            continue
        if not store.has_active_lease(item):
            continue
        owner = item.lease_owner
        if owner and owner != worker_id:
            return str(owner)
    return None
