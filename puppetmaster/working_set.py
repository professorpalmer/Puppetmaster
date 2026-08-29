"""Portable working-set cache: retrieve artifacts by index, do not stuff JSON.

Sibling workers share a job-stable mention of ``artifact_index.json`` in the
job brief (written once at job start). The index itself is a compact JSON
sidecar workers can read; full FINDING bodies stay in the store.

Warm-skip reuses labeled substantive artifacts when the current task is
read-only/analysis and a fingerprint that binds *both* scoped source bytes
and the task instruction matches. Scope-only reuse is forbidden: a different
question on the same files must not inherit the previous answer.

Kill switches:
- ``PUPPETMASTER_WORKING_SET=0`` — disable index + brief line + reuse
- ``PUPPETMASTER_WORKING_SET_REUSE=0`` — disable warm-skip only
- ``payload.skip_working_set_reuse`` — per-task opt out
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

from puppetmaster.models import Artifact, new_id, now_iso
from puppetmaster.validation import (
    SUBSTANTIVE_VALIDATION_TYPES,
    compact_artifact_ref,
    compute_validation_fingerprint,
    validation_status_of,
)

WORKING_SET_ENV = "PUPPETMASTER_WORKING_SET"
WORKING_SET_REUSE_ENV = "PUPPETMASTER_WORKING_SET_REUSE"
ARTIFACT_INDEX_FILENAME = "artifact_index.json"

# Stable job-brief one-liner. Path mention only — never a count or body dump
# so the cached prefix does not grow when the index is rebuilt.
WORKING_SET_BRIEF_LINE = (
    "Working-set artifact index: artifact_index.json "
    "(retrieve by filename; do not expect full FINDING bodies in this brief)."
)

_DISABLED = ("0", "false", "no", "off")
_ANALYSIS_ROLES = frozenset(
    {
        "analysis",
        "architect",
        "audit",
        "explore",
        "plan",
        "redteam",
        "review",
        "test",
    }
)


def working_set_enabled() -> bool:
    """Return False when the optional kill switch disables the working set."""
    raw = (os.environ.get(WORKING_SET_ENV) or "1").strip().lower()
    return raw not in _DISABLED


def working_set_reuse_enabled() -> bool:
    """Return False when warm-skip reuse is globally disabled."""
    if not working_set_enabled():
        return False
    raw = (os.environ.get(WORKING_SET_REUSE_ENV) or "1").strip().lower()
    return raw not in _DISABLED


def artifact_index_path(job_dir: Union[Path, str]) -> Path:
    return Path(job_dir) / ARTIFACT_INDEX_FILENAME


def working_set_brief_line() -> str:
    """Stable job-brief mention of the artifact index (path only)."""
    return WORKING_SET_BRIEF_LINE


def _usable_repo_cwd(raw: Any) -> Optional[Path]:
    if raw is None or raw == "":
        return None
    try:
        path = Path(str(raw))
        if path.is_dir():
            return path
    except Exception:
        return None
    return None


def _cwd_from_artifacts(artifacts: Iterable[Any]) -> Optional[Path]:
    for item in artifacts:
        payload = getattr(item, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        validation = payload.get("validation")
        if not isinstance(validation, dict):
            continue
        cwd = _usable_repo_cwd(validation.get("repo_root"))
        if cwd is not None:
            return cwd
    return None


def _cwd_from_store_job(store: Any, job_id: str) -> Optional[Path]:
    try:
        tasks = store.list_tasks(job_id)
    except Exception:
        return None
    for task in tasks:
        payload = getattr(task, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        cwd = _usable_repo_cwd(payload.get("cwd") or payload.get("workspace"))
        if cwd is not None:
            return cwd
    return None


def write_artifact_index(
    job_dir: Union[Path, str],
    artifacts: Iterable[Any],
    *,
    cwd: Optional[Union[Path, str]] = None,
    store: Any = None,
) -> Optional[Path]:
    """Persist compact refs for shared-context artifacts. Best-effort.

    Uses ``filter_shared_context_artifacts`` + ``compact_artifact_ref`` so the
    sidecar never contains full FINDING bodies. When ``cwd`` (or a repo root
    on the artifacts) is available, cited freshness is refreshed first;
    already-stale status is refused either way.
    """
    if not working_set_enabled():
        return None
    try:
        from puppetmaster.gist_admission import filter_shared_context_artifacts

        root = Path(job_dir)
        root.mkdir(parents=True, exist_ok=True)
        items = list(artifacts or [])
        job_id = None
        for item in items:
            job_id = getattr(item, "job_id", None)
            if job_id:
                break
        resolved_cwd = _usable_repo_cwd(cwd)
        if resolved_cwd is None:
            resolved_cwd = _cwd_from_artifacts(items)
        filtered = filter_shared_context_artifacts(
            items,
            for_job_id=job_id,
            cwd=resolved_cwd,
            store=store,
        )
        refs = []
        for artifact in filtered:
            try:
                refs.append(compact_artifact_ref(artifact))
            except Exception:
                continue
        payload = {"artifacts": refs}
        path = artifact_index_path(root)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
        return path
    except Exception:
        try:
            temp_path = Path(job_dir) / f".{ARTIFACT_INDEX_FILENAME}.{os.getpid()}.tmp"
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return None


def read_artifact_index(job_dir: Union[Path, str, None]) -> List[dict]:
    """Read compact artifact refs from ``job_dir``. Empty list on miss."""
    if job_dir is None:
        return []
    try:
        path = artifact_index_path(job_dir)
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            items = raw.get("artifacts")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []
    except Exception:
        return []


def rebuild_artifact_index(
    job_dir: Union[Path, str],
    store: Any,
    job_id: str,
) -> Optional[Path]:
    """Rebuild the sidecar from ``store.list_artifacts(job_id)``. Best-effort."""
    if not working_set_enabled():
        return None
    try:
        artifacts = store.list_artifacts(job_id)
        cwd = _cwd_from_store_job(store, job_id)
        return write_artifact_index(job_dir, artifacts, cwd=cwd, store=store)
    except Exception:
        return None


def _task_payload(task: Any) -> dict:
    payload = getattr(task, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def _task_cwd(task: Any) -> Optional[Path]:
    payload = _task_payload(task)
    raw = payload.get("cwd")
    if raw is None or raw == "":
        return None
    try:
        path = Path(str(raw))
        if not path.is_dir():
            return None
        return path
    except Exception:
        return None


def _task_source_scope(task: Any) -> Optional[List[str]]:
    payload = _task_payload(task)
    raw = payload.get("source_scope")
    if raw is None:
        raw = payload.get("sources")
    if not isinstance(raw, list) or not raw:
        return None
    scope = [str(item).strip() for item in raw if str(item).strip()]
    return scope or None


def _instruction_digest(task: Any) -> str:
    role = str(getattr(task, "role", "") or "")
    instruction = str(getattr(task, "instruction", "") or "")
    payload = _task_payload(task)
    prompt = str(payload.get("prompt") or instruction)
    memory_blob = ""
    memory = payload.get("retrieved_memory")
    if memory is not None:
        try:
            memory_blob = json.dumps(memory, sort_keys=True, default=str)
        except Exception:
            memory_blob = str(memory)
    material = f"{role}\n{instruction}\n{prompt}\n{memory_blob}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _combined_reuse_fingerprint(scope_fingerprint: str, instruction_digest: str) -> str:
    material = f"{scope_fingerprint}:{instruction_digest}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def reuse_fingerprint(task: Any) -> Optional[str]:
    """Combined validation + instruction fingerprint, or None (fail closed)."""
    try:
        cwd = _task_cwd(task)
        scope = _task_source_scope(task)
        if cwd is None or scope is None:
            return None
        result = compute_validation_fingerprint(cwd, scope, strict=False)
        if not result.complete or not result.fingerprint:
            return None
        return _combined_reuse_fingerprint(result.fingerprint, _instruction_digest(task))
    except Exception:
        return None


def _fresh_validation_payload(task: Any) -> Optional[dict]:
    """Build ``payload.validation`` for a newly produced artifact, or None."""
    try:
        cwd = _task_cwd(task)
        scope = _task_source_scope(task)
        if cwd is None or scope is None:
            return None
        result = compute_validation_fingerprint(cwd, scope, strict=False)
        if not result.complete:
            return None
        instruction_digest = _instruction_digest(task)
        combined = _combined_reuse_fingerprint(result.fingerprint, instruction_digest)
        payload = result.to_payload(status="fresh")
        payload["fingerprint"] = combined
        payload["scope_fingerprint"] = result.fingerprint
        payload["instruction_hash"] = instruction_digest
        return payload
    except Exception:
        return None


def stamp_fresh_validation(task: Any, artifacts: Sequence[Any]) -> List[Any]:
    """Stamp ``payload.validation`` (status=fresh) on unlabeled substantive artifacts.

    Best-effort: returns the original list when the fingerprint cannot be
    computed or stamping fails.
    """
    try:
        validation = _fresh_validation_payload(task)
        if validation is None:
            return list(artifacts)
        stamped: List[Any] = []
        for artifact in artifacts:
            try:
                if getattr(artifact, "type", None) not in SUBSTANTIVE_VALIDATION_TYPES:
                    stamped.append(artifact)
                    continue
                payload = dict(getattr(artifact, "payload", None) or {})
                if isinstance(payload.get("validation"), dict):
                    stamped.append(artifact)
                    continue
                payload["validation"] = dict(validation)
                stamped.append(replace(artifact, payload=payload, sha256=None))
            except Exception:
                stamped.append(artifact)
        return stamped
    except Exception:
        return list(artifacts)


def _is_prewalk_implement(task: Any) -> bool:
    payload = _task_payload(task)
    if not payload.get("prewalk"):
        return False
    if payload.get("mode") == "implement" or payload.get("implement"):
        return True
    return str(getattr(task, "role", "") or "") == "implement"


def _is_read_only_or_analysis(task: Any) -> bool:
    """True when warm-skip is allowed (analysis / read-only, not an edit)."""
    if _is_prewalk_implement(task):
        return False
    payload = _task_payload(task)
    if payload.get("read_only") or payload.get("no_edit") or payload.get("dry_run"):
        return True
    if payload.get("mode") == "implement" or payload.get("implement"):
        return False
    role = str(getattr(task, "role", "") or "")
    return role in _ANALYSIS_ROLES


def _reuse_job_ids(task: Any) -> List[str]:
    job_id = str(getattr(task, "job_id", "") or "")
    ids: List[str] = []
    if job_id:
        ids.append(job_id)
    extra = _task_payload(task).get("reuse_job_ids")
    if isinstance(extra, (list, tuple)):
        for item in extra:
            text = str(item or "").strip()
            if text and text not in ids:
                ids.append(text)
    return ids


def _clone_reused_artifact(source: Artifact, task: Any) -> Artifact:
    from puppetmaster.validation import with_validation_status

    cloned = replace(
        source,
        id=new_id("artifact"),
        job_id=str(getattr(task, "job_id", source.job_id) or source.job_id),
        task_id=str(getattr(task, "id", "") or source.task_id),
        created_at=now_iso(),
        sha256=None,
    )
    return with_validation_status(
        cloned,
        "reused",
        source_artifact_ids=[source.id],
    )


def maybe_reuse_artifacts(store: Any, task: Any) -> List[Artifact]:
    """Return reused clones when a warm-skip hit exists; else empty.

    Lookup is bounded to the current job plus optional ``payload.reuse_job_ids``.
    Only read-only/analysis tasks skip. Prewalk implement never skips.
    Unlabeled or incomplete fingerprints fail closed. Never raises.
    """
    try:
        if not working_set_reuse_enabled():
            return []
        payload = _task_payload(task)
        if payload.get("skip_working_set_reuse"):
            return []
        if not _is_read_only_or_analysis(task):
            return []
        fingerprint = reuse_fingerprint(task)
        if not fingerprint:
            return []
        job_ids = _reuse_job_ids(task)
        if not job_ids:
            return []
        hits = store.lookup_artifacts_by_validation_fingerprint(
            fingerprint,
            job_ids=job_ids,
        )
        if not hits:
            return []
        clones: List[Artifact] = []
        seen_sources = set()
        for hit in hits:
            source_id = getattr(hit, "id", None)
            if not source_id or source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            try:
                clones.append(_clone_reused_artifact(hit, task))
            except Exception:
                continue
        return clones
    except Exception:
        return []


def persist_reused_artifacts(
    store: Any,
    task: Any,
    artifacts: Sequence[Artifact],
    *,
    worker_id: Optional[str] = None,
) -> List[Artifact]:
    """Save reused clones and record DERIVED_FROM. Best-effort.

    Does not re-admit gists or enqueue follow-ups: the source artifact already
    did that. Repeating from a new parent_task_id would duplicate subtasks.
    """
    persisted: List[Artifact] = []
    for artifact in artifacts:
        try:
            if worker_id:
                artifact = replace(artifact, created_by=worker_id, sha256=None)
            store.save_artifact(artifact)
            persisted.append(artifact)
        except Exception:
            continue
        source_ids = []
        try:
            validation = (getattr(artifact, "payload", None) or {}).get("validation")
            if isinstance(validation, dict):
                source_ids = list(validation.get("source_artifact_ids") or [])
        except Exception:
            source_ids = []
        for source_id in source_ids:
            try:
                store.record_derived_from(
                    artifact.job_id,
                    str(source_id),
                    artifact.id,
                    meta={"reason": "working_set_reuse"},
                )
            except Exception:
                pass
        # Clones are lineage, not new discoveries. The source already admitted
        # gists and enqueued follow-ups; repeating that from a new parent_task_id
        # would duplicate subtasks.
        if validation_status_of(artifact) == "reused":
            continue
        try:
            from puppetmaster.gist_admission import maybe_admit_finding_as_gist

            maybe_admit_finding_as_gist(store, artifact)
        except Exception:
            pass
        try:
            store.maybe_enqueue_follow_ups_from_artifact(
                artifact,
                parent_task_id=getattr(task, "id", None),
                created_by=worker_id or getattr(artifact, "created_by", None),
            )
        except Exception:
            pass
    if persisted:
        try:
            from puppetmaster.working_set_usage import record_reuse

            record_reuse(artifact_count=len(persisted), caller="runtime")
        except Exception:
            pass
    return persisted
