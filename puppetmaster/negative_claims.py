"""Negative-claim fingerprints on GATE / GIST / host-observation artifacts.

Sketch B: persist ``payload.negative_claim`` on existing artifacts (and host
observation rows). Same failed GATE / rejected gist / ``ci_failed`` does not
re-enqueue while its cited scope is still fresh. Withdrawal is skip, not delete.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from puppetmaster.models import Artifact, ArtifactType

REASON_NEGATIVE_CLAIM = "negative_claim"
NEGATIVE_CLAIM_KINDS = frozenset({"gate", "gist", "ci_failed"})


def normalize_claim_text(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def normalize_scope(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def negative_claim_fingerprint(claim: Any, scope: Any) -> str:
    material = "%s\n%s" % (normalize_scope(scope), normalize_claim_text(claim))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def scope_from_paths(paths: Optional[Iterable[Any]]) -> str:
    rels = []
    seen = set()
    for raw in paths or ():
        text = str(raw or "").strip().replace("\\", "/")
        if not text or text in seen:
            continue
        seen.add(text)
        rels.append(text)
    return ",".join(sorted(rels))


def scope_from_source_ids(source_ids: Optional[Iterable[Any]]) -> str:
    parts = [str(item).strip() for item in (source_ids or ()) if str(item).strip()]
    return ",".join(parts)


def gate_negative_claim(name: Any, reason: Any) -> str:
    return ("%s %s" % (str(name or "").strip(), str(reason or "").strip())).strip()


def ci_failed_negative_claim(kind: Any, instruction: Any) -> str:
    return ("%s %s" % (str(kind or "").strip(), str(instruction or "").strip())).strip()


def _payload_of(artifact: Any) -> dict[str, Any]:
    payload = getattr(artifact, "payload", None) or {}
    return payload if isinstance(payload, dict) else {}


def _token_looks_like_path(token: Any) -> bool:
    text = str(token or "").strip().replace("\\", "/")
    if not text or text.startswith("gate:"):
        return False
    if "://" in text:
        return False
    if "/" in text:
        return True
    name = text.rsplit("/", 1)[-1]
    if "." in name and not name.startswith("."):
        suffix = name.rsplit(".", 1)[-1]
        if suffix.isalnum() and 1 <= len(suffix) <= 8:
            return True
    return False


def _paths_from_scope(scope: Any) -> list[str]:
    text = str(scope or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def scope_looks_like_paths(scope: Any) -> bool:
    parts = _paths_from_scope(scope)
    if not parts:
        return False
    return all(_token_looks_like_path(part) for part in parts)


def _source_scope_from_mapping(payload: dict[str, Any]) -> Optional[str]:
    for key in ("source_scope", "sources", "files", "write_scope"):
        raw = payload.get(key)
        if isinstance(raw, list) and raw:
            return scope_from_paths(raw)
    validation = payload.get("validation")
    if isinstance(validation, dict):
        vscope = validation.get("scope")
        if isinstance(vscope, list) and vscope:
            return scope_from_paths(vscope)
    return None


def gate_scope_for(artifact: Any, task: Any = None) -> str:
    if task is not None:
        task_payload = getattr(task, "payload", None) or {}
        if isinstance(task_payload, dict):
            scoped = _source_scope_from_mapping(task_payload)
            if scoped:
                return scoped
    payload = _payload_of(artifact)
    scoped = _source_scope_from_mapping(payload)
    if scoped:
        return scoped
    evidence = [
        item
        for item in (getattr(artifact, "evidence", None) or [])
        if _token_looks_like_path(item)
    ]
    if evidence:
        return scope_from_paths(evidence)
    return str(payload.get("kind") or payload.get("gate") or "gate")


def enqueue_negative_scope(
    artifact: Any,
    proposal: Optional[dict[str, Any]] = None,
    instruction: str = "",
) -> str:
    proposal = proposal or {}
    raw = proposal.get("scope")
    if isinstance(raw, (list, tuple)):
        scoped = scope_from_paths(raw)
        if scoped:
            return scoped
    if raw not in (None, ""):
        return str(raw)
    payload = _payload_of(artifact)
    scoped = _source_scope_from_mapping(payload)
    if scoped:
        return scoped
    evidence = [
        item
        for item in (getattr(artifact, "evidence", None) or [])
        if _token_looks_like_path(item)
    ]
    if evidence:
        return scope_from_paths(evidence)
    source_ids = payload.get("source_artifact_ids")
    if isinstance(source_ids, list) and source_ids:
        joined = scope_from_source_ids(source_ids)
        if joined:
            return joined
    kind = payload.get("kind") or payload.get("gate")
    if kind:
        return str(kind)
    return instruction


def resolve_negative_cwd(
    store: Any,
    artifact: Any,
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[Union[str, Path]]:
    if cwd not in (None, ""):
        return cwd
    payload = _payload_of(artifact)
    validation = payload.get("validation")
    if isinstance(validation, dict):
        repo_root = validation.get("repo_root")
        if repo_root and Path(str(repo_root)).is_dir():
            return str(repo_root)
    task_id = getattr(artifact, "task_id", None)
    if store is not None and task_id:
        try:
            task = store.get_task_by_id(task_id)
        except Exception:
            task = None
        if task is not None:
            raw = (getattr(task, "payload", None) or {}).get("cwd")
            if raw not in (None, "") and Path(str(raw)).is_dir():
                return str(raw)
    return None


def _source_digests_for_scope(
    scope: Any,
    cwd: Optional[Union[str, Path]],
) -> Optional[dict[str, str]]:
    if cwd in (None, "") or not scope_looks_like_paths(scope):
        return None
    from puppetmaster.validation import compute_validation_fingerprint

    try:
        result = compute_validation_fingerprint(
            cwd, _paths_from_scope(scope), strict=False
        )
    except Exception:
        return None
    digests = dict(result.source_digests or {})
    return digests or None


def stamp_negative_claim(
    artifact: Artifact,
    *,
    kind: str,
    claim: str,
    scope: str,
    cwd: Optional[Union[str, Path]] = None,
) -> Artifact:
    """Set ``payload.negative_claim`` on ``artifact``. Does not persist."""
    if kind not in NEGATIVE_CLAIM_KINDS:
        raise ValueError(
            "negative_claim kind must be one of %s; got %r"
            % (sorted(NEGATIVE_CLAIM_KINDS), kind)
        )
    negative: dict[str, Any] = {
        "fingerprint": negative_claim_fingerprint(claim, scope),
        "kind": kind,
        "claim": claim,
        "scope": scope,
        "retry_count": 0,
    }
    digests = _source_digests_for_scope(scope, cwd)
    if digests:
        negative["source_digests"] = digests
    payload = dict(artifact.payload or {})
    payload["negative_claim"] = negative
    return replace(artifact, payload=payload, sha256=None)


def stamp_failed_gate(
    artifact: Artifact,
    *,
    task: Any = None,
    cwd: Optional[Union[str, Path]] = None,
    store: Any = None,
) -> Artifact:
    """Stamp a failed GATE when it does not already carry a negative claim."""
    kind = getattr(artifact, "type", None)
    if kind != ArtifactType.GATE and str(kind) != "gate":
        return artifact
    payload = _payload_of(artifact)
    if payload.get("passed") is not False:
        return artifact
    if isinstance(payload.get("negative_claim"), dict):
        return artifact
    resolved_task = task
    resolved_cwd = cwd
    if resolved_task is None and store is not None:
        try:
            resolved_task = store.get_task_by_id(artifact.task_id)
        except Exception:
            resolved_task = None
    if resolved_cwd in (None, "") and resolved_task is not None:
        raw = (getattr(resolved_task, "payload", None) or {}).get("cwd")
        if raw not in (None, ""):
            resolved_cwd = raw
    name = payload.get("gate") or payload.get("kind") or "gate"
    reason = payload.get("reason") or payload.get("failed_reason") or "failed"
    return stamp_negative_claim(
        artifact,
        kind="gate",
        claim=gate_negative_claim(name, reason),
        scope=gate_scope_for(artifact, resolved_task),
        cwd=resolved_cwd,
    )


def is_scope_fresh(
    negative_payload: Any,
    cwd: Optional[Union[str, Path]] = None,
) -> bool:
    """True when the stamped scope is still current (skip repeats)."""
    if not isinstance(negative_payload, dict):
        return False
    digests = negative_payload.get("source_digests")
    if cwd in (None, "") or not isinstance(digests, dict) or not digests:
        return True
    from puppetmaster.validation import compute_validation_fingerprint

    paths = [str(key).strip() for key in digests.keys() if str(key).strip()]
    try:
        result = compute_validation_fingerprint(cwd, paths, strict=False)
    except Exception:
        return False
    if result.missing_paths or result.unreadable_paths:
        return False
    current = dict(result.source_digests or {})
    for rel, expected in digests.items():
        path = str(rel or "").strip()
        if not path or current.get(path) != str(expected):
            return False
    return True


def _host_negative_rows(store: Any, job_id: str) -> list[dict[str, Any]]:
    try:
        from puppetmaster.metr_seams import load_host_document

        document = load_host_document(store, job_id)
    except Exception:
        return []
    rows = []
    for row in list(document.get("observations") or []):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _artifact_from_host_row(
    job_id: str,
    row: dict[str, Any],
    negative: dict[str, Any],
) -> Artifact:
    claim = str(negative.get("claim") or row.get("kind") or "ci_failed")
    return Artifact(
        job_id=job_id,
        task_id=str(row.get("task_id") or "host"),
        type=ArtifactType.FINDING,
        created_by=str(row.get("source") or "host"),
        confidence=1.0,
        evidence=list(row.get("evidence") or ["host:observation"]),
        payload={"claim": claim, "negative_claim": dict(negative)},
    )


def find_fresh_negative(
    store: Any,
    job_id: str,
    fingerprint: str,
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[Artifact]:
    wanted = str(fingerprint or "")
    if not wanted:
        return None
    try:
        artifacts: Sequence[Any] = store.list_artifacts(job_id)
    except Exception:
        artifacts = []
    for artifact in artifacts:
        negative = _payload_of(artifact).get("negative_claim")
        if not isinstance(negative, dict):
            continue
        if str(negative.get("fingerprint") or "") != wanted:
            continue
        if is_scope_fresh(negative, cwd):
            return artifact
    for row in _host_negative_rows(store, job_id):
        negative = row.get("negative_claim")
        if not isinstance(negative, dict):
            continue
        if str(negative.get("fingerprint") or "") != wanted:
            continue
        if is_scope_fresh(negative, cwd):
            return _artifact_from_host_row(job_id, row, negative)
    return None


def should_skip_negative(
    store: Any,
    job_id: str,
    claim: str,
    scope: str,
    cwd: Optional[Union[str, Path]] = None,
) -> bool:
    fingerprint = negative_claim_fingerprint(claim, scope)
    return find_fresh_negative(store, job_id, fingerprint, cwd) is not None


def persist_ci_failed_negative(store: Any, job_id: str, fact: Any) -> None:
    """Stamp ``negative_claim`` on the first ``ci_failed`` host observation."""
    from puppetmaster.metr_seams import load_host_document, save_host_document

    kind = str(getattr(fact, "kind", "") or "").strip()
    if kind != "ci_failed":
        return
    claim = ci_failed_negative_claim(kind, getattr(fact, "instruction", ""))
    scope = str(getattr(fact, "key", "") or "")
    fingerprint = negative_claim_fingerprint(claim, scope)
    document = load_host_document(store, job_id)
    rows = list(document.get("observations") or [])
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("kind") or "").strip().lower() != "ci_failed":
            continue
        if isinstance(row.get("negative_claim"), dict):
            return
        row["negative_claim"] = {
            "fingerprint": fingerprint,
            "kind": "ci_failed",
            "claim": claim,
            "scope": scope,
            "retry_count": 0,
        }
        changed = True
        break
    if not changed:
        return
    document["observations"] = rows
    save_host_document(store, job_id, document)
