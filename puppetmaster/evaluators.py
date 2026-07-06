from __future__ import annotations

"""RQGM v1: durable evaluator slot registry and epoch helpers."""

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

from puppetmaster.fs_permissions import write_private_text
from puppetmaster.models import ArtifactType, now_iso


@dataclass(frozen=True)
class EvaluatorSpec:
    slot_id: str
    version: int
    role: str
    instruction: str
    criteria: dict
    active: bool = True
    parent_version: Optional[int] = None
    promoted_at: Optional[str] = None


def registry_path(state_dir: str) -> str:
    return os.path.join(state_dir, "evaluators", "registry.json")


def _registry_dir(state_dir: str) -> Path:
    return Path(state_dir) / "evaluators"


def _spec_from_dict(raw: dict) -> EvaluatorSpec:
    if not isinstance(raw, dict):
        raise ValueError("evaluator entry must be an object")
    slot_id = str(raw.get("slot_id") or "").strip()
    if not slot_id:
        raise ValueError("evaluator missing slot_id")
    role = str(raw.get("role") or "").strip()
    if not role:
        raise ValueError(f"evaluator {slot_id!r} missing role")
    instruction = str(raw.get("instruction") or "").strip()
    criteria = raw.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise ValueError(f"evaluator {slot_id!r}: criteria must be an object")
    version_raw = raw.get("version", 1)
    version = int(version_raw)
    parent_raw = raw.get("parent_version")
    parent_version: Optional[int] = int(parent_raw) if parent_raw is not None else None
    promoted_raw = raw.get("promoted_at")
    promoted_at = str(promoted_raw) if promoted_raw else None
    return EvaluatorSpec(
        slot_id=slot_id,
        version=version,
        role=role,
        instruction=instruction,
        criteria=criteria,
        active=bool(raw.get("active", True)),
        parent_version=parent_version,
        promoted_at=promoted_at,
    )


def _spec_to_dict(spec: EvaluatorSpec) -> dict[str, Any]:
    data = asdict(spec)
    if data.get("parent_version") is None:
        data["parent_version"] = None
    if data.get("promoted_at") is None:
        data["promoted_at"] = None
    return data


def load_registry(state_dir: str) -> list[EvaluatorSpec]:
    path = registry_path(state_dir)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")
    entries = data.get("evaluators") or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: evaluators must be a list")
    return [_spec_from_dict(item) for item in entries]


def save_registry(state_dir: str, specs: list[EvaluatorSpec]) -> None:
    directory = _registry_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(registry_path(state_dir))
    payload = {"evaluators": [_spec_to_dict(spec) for spec in specs]}
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    write_private_text(tmp, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def active_evaluators(state_dir: str) -> dict[str, EvaluatorSpec]:
    active: dict[str, EvaluatorSpec] = {}
    for spec in load_registry(state_dir):
        if not spec.active:
            continue
        current = active.get(spec.slot_id)
        if current is None or spec.version > current.version:
            active[spec.slot_id] = spec
    return active


def register_evaluator(state_dir: str, spec: EvaluatorSpec) -> EvaluatorSpec:
    specs = load_registry(state_dir)
    updated: list[EvaluatorSpec] = []
    for existing in specs:
        if existing.slot_id == spec.slot_id and existing.active:
            updated.append(replace(existing, active=False))
        else:
            updated.append(existing)
    updated.append(spec)
    save_registry(state_dir, updated)
    return spec


def evaluator_epoch_for_job(store, job_id: str) -> dict:
    """Return the latest evaluator_epoch payload for a job, or {}."""
    try:
        artifacts = store.list_artifacts(job_id)
    except Exception:
        return {}
    latest_at = ""
    latest_payload: dict = {}
    for artifact in artifacts:
        if artifact.type != ArtifactType.DECISION:
            continue
        payload = getattr(artifact, "payload", None) or {}
        if payload.get("kind") != "evaluator_epoch":
            continue
        created_at = getattr(artifact, "created_at", "") or ""
        if created_at >= latest_at:
            latest_at = created_at
            latest_payload = dict(payload)
    return latest_payload


def load_anchor_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        entries = data.get("anchors") or data.get("entries") or []
        if isinstance(entries, list):
            return entries
    raise ValueError(f"{path}: anchor set must be a list or {{anchors: [...]}}")


def run_anchor_battery(state_dir: str, anchor_path: str, *, slot_id: str) -> dict:
    from puppetmaster.adapters.local import LocalAdapter
    from puppetmaster.models import Artifact, Task

    anchors = load_anchor_set(anchor_path)
    active = active_evaluators(state_dir)
    spec = active.get(slot_id)
    if spec is None:
        raise ValueError(f"unknown slot_id {slot_id!r}")

    adapter = LocalAdapter()
    results: list[dict] = []
    passed_count = 0
    for entry in anchors:
        if not isinstance(entry, dict):
            raise ValueError("anchor entry must be an object")
        anchor_id = str(entry.get("id") or "").strip()
        if not anchor_id:
            raise ValueError("anchor entry missing id")
        goal = str(entry.get("goal") or "")
        expect = entry.get("expect") or {}
        if not isinstance(expect, dict):
            raise ValueError(f"anchor {anchor_id!r}: expect must be an object")
        min_conf = float(expect.get("min_verification_confidence", 0.0))

        task = Task(
            job_id="anchor-battery",
            role=spec.role,
            instruction=goal or spec.instruction,
            adapter="local",
        )
        artifacts = adapter.run(task, goal, worker_id="anchor-battery")
        max_conf = 0.0
        for art in artifacts:
            if isinstance(art, Artifact):
                max_conf = max(max_conf, float(getattr(art, "confidence", 0.0) or 0.0))
        ok = max_conf >= min_conf
        if ok:
            passed_count += 1
        results.append(
            {
                "id": anchor_id,
                "passed": ok,
                "max_confidence": max_conf,
                "min_confidence": min_conf,
                "reason": "ok" if ok else f"max confidence {max_conf} below {min_conf}",
            }
        )

    total = len(results)
    pass_rate = (passed_count / total) if total else 0.0
    return {
        "passed": passed_count,
        "total": total,
        "pass_rate": pass_rate,
        "results": results,
    }


def promote_evaluator(
    state_dir: str,
    slot_id: str,
    *,
    parent_version: Optional[int],
    instruction: str,
    criteria: dict,
    anchor_path: str,
    min_pass_rate: float = 1.0,
) -> EvaluatorSpec:
    battery = run_anchor_battery(state_dir, anchor_path, slot_id=slot_id)
    if battery["pass_rate"] < min_pass_rate:
        raise ValueError(
            f"anchor battery pass rate {battery['pass_rate']:.2f} "
            f"below {min_pass_rate:.2f}"
        )

    active = active_evaluators(state_dir)
    current = active.get(slot_id)
    role = current.role if current is not None else slot_id
    existing = load_registry(state_dir)
    versions = [spec.version for spec in existing if spec.slot_id == slot_id]
    next_version = max(versions, default=0) + 1
    resolved_parent = parent_version
    if resolved_parent is None and current is not None:
        resolved_parent = current.version

    new_spec = EvaluatorSpec(
        slot_id=slot_id,
        version=next_version,
        role=role,
        instruction=instruction,
        criteria=criteria or {},
        active=True,
        parent_version=resolved_parent,
        promoted_at=now_iso(),
    )
    return register_evaluator(state_dir, new_spec)


def stamp_verification_artifacts(task, artifacts: list, epoch: dict) -> list:
    """Attach evaluator slot metadata to VERIFICATION artifacts when role matches."""
    from dataclasses import replace as dc_replace

    from puppetmaster.models import Artifact

    evaluators = epoch.get("evaluators") or []
    match = None
    for entry in evaluators:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") == task.role:
            match = entry
            break
    if match is None:
        return artifacts

    slot_id = str(match.get("slot_id") or "")
    version = int(match.get("version") or 0)
    if not slot_id or version <= 0:
        return artifacts

    stamped = []
    for art in artifacts:
        if not isinstance(art, Artifact) or art.type != ArtifactType.VERIFICATION:
            stamped.append(art)
            continue
        payload = dict(art.payload or {})
        payload["evaluator_slot"] = slot_id
        payload["evaluator_version"] = version
        stamped.append(dc_replace(art, payload=payload))
    return stamped
