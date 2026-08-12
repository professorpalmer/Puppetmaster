"""Dependency-aware reusable validation protocol (additive).

Provides a deterministic validation fingerprint that binds repo HEAD, scoped
source/evidence file bytes (including dirty/untracked working-tree content),
and evaluator/rules digests. ``Artifact.sha256`` remains content integrity of
the artifact document and must not be overloaded for reuse keys.

Artifact.payload convention (optional; older artifacts load unchanged)::

    payload["validation"] = {
        "fingerprint": "<hex>",
        "status": "fresh" | "reused" | "stale" | "superseded",
        "head_sha": "...",
        "repo_root": "...",
        "scope": ["rel/path", ...],
        "source_digests": {"rel/path": "<hex>", ...},
        "source_digest": "<hex>",
        "rules_version": "...|null",
        "rules_digest": "<hex>",
        "evaluator_digest": "<hex>",
        "source_artifact_ids": ["artifact_...", ...],  # when reused
        "generation": 0,  # optional; bumped when prior outputs are superseded
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from puppetmaster.models import Artifact, ArtifactType

VALIDATION_STATUSES = frozenset({"fresh", "reused", "stale", "superseded"})
REUSABLE_VALIDATION_STATUSES = frozenset({"fresh", "reused"})
SUBSTANTIVE_VALIDATION_TYPES = frozenset(
    {
        ArtifactType.FINDING,
        ArtifactType.VERIFICATION,
        ArtifactType.DECISION,
        ArtifactType.GIST,
    }
)
DEFAULT_LOOKUP_LIMIT = 256
_EVIDENCE_SUMMARY_LIMIT = 8
_EVIDENCE_ITEM_MAX_CHARS = 240
_CONCISE_FIELD_MAX_CHARS = 240


class ValidationFingerprintError(ValueError):
    """Raised when fingerprint inputs are incomplete or escape the repo root."""

    def __init__(
        self,
        message: str,
        *,
        missing_paths: Optional[Sequence[str]] = None,
        unreadable_paths: Optional[Sequence[str]] = None,
        rejected_paths: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.missing_paths = list(missing_paths or [])
        self.unreadable_paths = list(unreadable_paths or [])
        self.rejected_paths = list(rejected_paths or [])


@dataclass(frozen=True)
class ValidationFingerprintResult:
    """Structured metadata plus a stable validation fingerprint."""

    fingerprint: str
    head_sha: str
    repo_root: str
    scope: list[str]
    source_digests: dict[str, str]
    source_digest: str
    rules_version: Optional[str]
    rules_digest: str
    evaluator_digest: str
    dirty_scoped: bool
    missing_paths: list[str] = field(default_factory=list)
    unreadable_paths: list[str] = field(default_factory=list)
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "head_sha": self.head_sha,
            "repo_root": self.repo_root,
            "scope": list(self.scope),
            "source_digests": dict(self.source_digests),
            "source_digest": self.source_digest,
            "rules_version": self.rules_version,
            "rules_digest": self.rules_digest,
            "evaluator_digest": self.evaluator_digest,
            "dirty_scoped": self.dirty_scoped,
            "missing_paths": list(self.missing_paths),
            "unreadable_paths": list(self.unreadable_paths),
            "complete": self.complete,
        }

    def to_payload(
        self,
        *,
        status: str = "fresh",
        source_artifact_ids: Optional[Sequence[str]] = None,
        generation: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build the additive ``Artifact.payload['validation']`` object."""
        if status not in VALIDATION_STATUSES:
            raise ValueError(
                f"validation status must be one of {sorted(VALIDATION_STATUSES)}; "
                f"got {status!r}"
            )
        if not self.complete:
            raise ValidationFingerprintError(
                "cannot stamp validation payload from incomplete fingerprint",
                missing_paths=self.missing_paths,
                unreadable_paths=self.unreadable_paths,
            )
        payload: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "status": status,
            "head_sha": self.head_sha,
            "repo_root": self.repo_root,
            "scope": list(self.scope),
            "source_digests": dict(self.source_digests),
            "source_digest": self.source_digest,
            "rules_version": self.rules_version,
            "rules_digest": self.rules_digest,
            "evaluator_digest": self.evaluator_digest,
            "dirty_scoped": self.dirty_scoped,
        }
        if source_artifact_ids is not None:
            payload["source_artifact_ids"] = [
                str(item) for item in source_artifact_ids if item
            ]
        if generation is not None:
            payload["generation"] = int(generation)
        return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _is_absolute_input_path(text: str) -> bool:
    """True for POSIX/Windows absolute inputs (before joining under a root)."""
    if not text:
        return False
    if text.startswith("/") or text.startswith("\\"):
        return True
    # Windows drive / UNC (also reject when running on POSIX hosts).
    if len(text) >= 2 and text[1] == ":":
        return True
    if text.startswith("//") or text.startswith("\\\\"):
        return True
    return Path(text).is_absolute()


def _normalize_rel_paths(paths: Sequence[str]) -> list[str]:
    """Normalize scoped/rules paths; reject absolute and ``..`` traversal."""
    normalized: list[str] = []
    seen: set[str] = set()
    rejected: list[str] = []
    for raw in paths:
        text = str(raw or "").strip().replace("\\", "/")
        if not text:
            continue
        if _is_absolute_input_path(text):
            rejected.append(text)
            continue
        while text.startswith("./"):
            text = text[2:]
        # Do not strip leading "/" into a relative path — absolute inputs are
        # rejected above. Collapse duplicate separators only.
        parts = [part for part in text.split("/") if part and part != "."]
        if any(part == ".." for part in parts):
            rejected.append(text)
            continue
        text = "/".join(parts)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    if rejected:
        raise ValidationFingerprintError(
            "validation paths must stay inside repo root (fail closed): "
            f"rejected={rejected!r}",
            rejected_paths=rejected,
        )
    return sorted(normalized)


def _resolve_contained_path(root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` under ``root``; refuse absolute/.. /symlink escapes."""
    text = str(rel_path or "").strip().replace("\\", "/")
    if not text or _is_absolute_input_path(text):
        raise ValidationFingerprintError(
            f"path escapes repo root: {rel_path!r}",
            rejected_paths=[str(rel_path)],
        )
    parts = [part for part in text.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValidationFingerprintError(
            f"path escapes repo root: {rel_path!r}",
            rejected_paths=[str(rel_path)],
        )
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts)
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValidationFingerprintError(
            f"path unresolvable under repo root: {rel_path!r}",
            rejected_paths=[str(rel_path)],
        ) from exc
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValidationFingerprintError(
            f"path escapes repo root: {rel_path!r}",
            rejected_paths=[str(rel_path)],
        )
    return resolved


def _read_scoped_file(
    root: Path, rel_path: str
) -> tuple[Optional[bytes], Optional[str]]:
    """Return ``(bytes, None)`` or ``(None, 'missing'|'unreadable')``.

    Paths are confined to ``root`` (absolute / ``..`` / symlink escapes raise
    :class:`ValidationFingerprintError` and never hash outside bytes).
    """
    path = _resolve_contained_path(root, rel_path)
    try:
        if not path.is_file():
            return None, "missing"
        return path.read_bytes(), None
    except OSError:
        return None, "unreadable"


def _digest_mapping(mapping: Mapping[str, str]) -> str:
    canonical = json.dumps(
        {key: mapping[key] for key in sorted(mapping)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical)


def _rules_digest(
    root: Path,
    rules_paths: Sequence[str],
    rules_version: Optional[str],
) -> tuple[str, list[str], list[str]]:
    missing: list[str] = []
    unreadable: list[str] = []
    parts: dict[str, str] = {}
    if rules_version is not None:
        parts["__rules_version__"] = str(rules_version)
    for rel in _normalize_rel_paths(list(rules_paths)):
        data, error = _read_scoped_file(root, rel)
        if error == "missing":
            missing.append(rel)
            continue
        if error == "unreadable":
            unreadable.append(rel)
            continue
        assert data is not None
        parts[rel] = _sha256_bytes(data)
    if not parts and not missing and not unreadable:
        return _sha256_text(""), missing, unreadable
    return _digest_mapping(parts), missing, unreadable


def _evaluator_digest_value(
    evaluator_version: Optional[str],
    evaluator_digest: Optional[str],
    evaluator: Optional[Mapping[str, Any]],
) -> str:
    if evaluator_digest is not None and str(evaluator_digest).strip():
        return str(evaluator_digest).strip()
    if evaluator is not None:
        canonical = json.dumps(evaluator, sort_keys=True, separators=(",", ":"))
        return _sha256_text(canonical)
    if evaluator_version is not None:
        return _sha256_text(str(evaluator_version))
    return _sha256_text("")


def compute_validation_fingerprint(
    cwd: Union[str, Path],
    scope: Sequence[str],
    *,
    rules_paths: Optional[Sequence[str]] = None,
    rules_version: Optional[str] = None,
    evaluator_version: Optional[str] = None,
    evaluator_digest: Optional[str] = None,
    evaluator: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> ValidationFingerprintResult:
    """Compute a deterministic validation fingerprint for ``scope`` under ``cwd``.

    Binds:

    - git ``HEAD`` sha (or ``uncommitted`` when unavailable)
    - content hashes of each scoped path from the working tree (dirty/untracked
      bytes included when those paths are in scope)
    - rules version/content digest
    - evaluator version/digest

    Missing or unreadable scoped (or rules) inputs are listed explicitly. With
    ``strict=True`` (default) an incomplete input set raises
    :class:`ValidationFingerprintError` (fail closed).
    """
    from puppetmaster.adapters._git import git_snapshot, git_worktree_root

    root = Path(cwd).resolve()
    if not root.is_dir():
        raise ValidationFingerprintError(f"cwd is not a directory: {root}")

    snapshot = git_snapshot(root)
    repo_root = git_worktree_root(root) if snapshot.is_worktree else root
    repo_root = Path(repo_root).resolve()
    head_sha = snapshot.sha or "uncommitted"

    scope_paths = _normalize_rel_paths(list(scope))
    source_digests: dict[str, str] = {}
    missing_paths: list[str] = []
    unreadable_paths: list[str] = []
    dirty_names = set(snapshot.changed_files or []) | set(snapshot.untracked_files or [])
    dirty_scoped = False

    for rel in scope_paths:
        data, error = _read_scoped_file(repo_root, rel)
        if error == "missing":
            missing_paths.append(rel)
            continue
        if error == "unreadable":
            unreadable_paths.append(rel)
            continue
        assert data is not None
        source_digests[rel] = _sha256_bytes(data)
        if rel in dirty_names:
            dirty_scoped = True

    rules_digest, rules_missing, rules_unreadable = _rules_digest(
        repo_root, list(rules_paths or ()), rules_version
    )
    missing_paths.extend(rules_missing)
    unreadable_paths.extend(rules_unreadable)

    evaluator_digest_value = _evaluator_digest_value(
        evaluator_version, evaluator_digest, evaluator
    )
    source_digest = _digest_mapping(source_digests)
    complete = not missing_paths and not unreadable_paths

    material = {
        "head_sha": head_sha,
        "source_digest": source_digest,
        "rules_digest": rules_digest,
        "evaluator_digest": evaluator_digest_value,
    }
    fingerprint = _sha256_text(
        json.dumps(material, sort_keys=True, separators=(",", ":"))
    )

    result = ValidationFingerprintResult(
        fingerprint=fingerprint,
        head_sha=head_sha,
        repo_root=str(repo_root),
        scope=scope_paths,
        source_digests=source_digests,
        source_digest=source_digest,
        rules_version=str(rules_version) if rules_version is not None else None,
        rules_digest=rules_digest,
        evaluator_digest=evaluator_digest_value,
        dirty_scoped=dirty_scoped,
        missing_paths=missing_paths,
        unreadable_paths=unreadable_paths,
        complete=complete,
    )
    if strict and not complete:
        raise ValidationFingerprintError(
            "validation fingerprint inputs incomplete (fail closed): "
            f"missing={missing_paths!r} unreadable={unreadable_paths!r}",
            missing_paths=missing_paths,
            unreadable_paths=unreadable_paths,
        )
    return result


def validation_payload_of(artifact: Artifact) -> Optional[dict[str, Any]]:
    """Return ``payload.validation`` when it is a dict; else ``None``."""
    payload = getattr(artifact, "payload", None) or {}
    if not isinstance(payload, dict):
        return None
    validation = payload.get("validation")
    return dict(validation) if isinstance(validation, dict) else None


def validation_status_of(artifact: Artifact) -> Optional[str]:
    validation = validation_payload_of(artifact)
    if not validation:
        return None
    status = validation.get("status")
    return str(status) if status is not None else None


def validation_fingerprint_of(artifact: Artifact) -> Optional[str]:
    validation = validation_payload_of(artifact)
    if not validation:
        return None
    fingerprint = validation.get("fingerprint")
    return str(fingerprint) if fingerprint else None


def is_reusable_validation_artifact(artifact: Artifact) -> bool:
    """True for substantive FINDING/VERIFICATION/DECISION with fresh|reused status.

    Artifacts without a validation block are treated as unlabeled legacy output
    and are not considered reusable via fingerprint lookup (fail closed for reuse).
    """
    if artifact.type not in SUBSTANTIVE_VALIDATION_TYPES:
        return False
    status = validation_status_of(artifact)
    if status is None:
        return False
    return status in REUSABLE_VALIDATION_STATUSES


def with_validation_status(
    artifact: Artifact,
    status: str,
    *,
    generation: Optional[int] = None,
    source_artifact_ids: Optional[Sequence[str]] = None,
) -> Artifact:
    """Return a copy with ``payload.validation.status`` updated (additive)."""
    from dataclasses import replace

    if status not in VALIDATION_STATUSES:
        raise ValueError(
            f"validation status must be one of {sorted(VALIDATION_STATUSES)}; "
            f"got {status!r}"
        )
    payload = dict(getattr(artifact, "payload", None) or {})
    validation = dict(payload.get("validation") or {})
    validation["status"] = status
    if generation is not None:
        validation["generation"] = int(generation)
    if source_artifact_ids is not None:
        validation["source_artifact_ids"] = [
            str(item) for item in source_artifact_ids if item
        ]
    payload["validation"] = validation
    # Clear sha256 so persist recomputes content integrity for the new document.
    return replace(artifact, payload=payload, sha256=None)


def _truncate_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    limit = max(0, int(max_chars))
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def compact_artifact_ref(
    artifact: Artifact, *, evidence_limit: int = _EVIDENCE_SUMMARY_LIMIT
) -> dict[str, Any]:
    """Stable compact artifact record without large payload bodies.

    Omits absolute ``repo_root`` and truncates both evidence *count* (via
    ``evidence_limit``) and each evidence string (``_EVIDENCE_ITEM_MAX_CHARS``).
    """
    payload = getattr(artifact, "payload", None) or {}
    if not isinstance(payload, dict):
        payload = {}
    concise: dict[str, Any] = {}
    for key in ("claim", "check", "decision", "result", "why", "admission"):
        if key in payload and payload[key] is not None:
            concise[key] = _truncate_text(payload[key], _CONCISE_FIELD_MAX_CHARS)
    evidence = list(getattr(artifact, "evidence", None) or [])
    bound = max(0, int(evidence_limit))
    evidence_summary = {
        "count": len(evidence),
        "items": [
            _truncate_text(item, _EVIDENCE_ITEM_MAX_CHARS) for item in evidence[:bound]
        ],
    }
    ref: dict[str, Any] = {
        "id": artifact.id,
        "type": str(artifact.type),
        "task_id": artifact.task_id,
        "sha256": artifact.sha256,
        "confidence": artifact.confidence,
        "created_at": artifact.created_at,
        "evidence_summary": evidence_summary,
    }
    ref.update(concise)
    validation = validation_payload_of(artifact)
    if validation is not None:
        # Keep validation metadata but drop bulky digests and absolute roots.
        compact_validation = {
            key: validation[key]
            for key in (
                "fingerprint",
                "status",
                "head_sha",
                "scope",
                "source_digest",
                "rules_version",
                "rules_digest",
                "evaluator_digest",
                "dirty_scoped",
                "source_artifact_ids",
                "generation",
            )
            if key in validation
        }
        ref["validation"] = compact_validation
    return ref


def filter_artifacts_by_validation_fingerprint(
    artifacts: Iterable[Artifact],
    fingerprint: str,
    *,
    types: Optional[Iterable[Union[ArtifactType, str]]] = None,
    include_statuses: Optional[Iterable[str]] = None,
    limit: int = DEFAULT_LOOKUP_LIMIT,
) -> list[Artifact]:
    """Filter artifacts for fingerprint-aware reusable lookup (bounded)."""
    wanted_fp = str(fingerprint or "").strip()
    if not wanted_fp:
        return []
    type_filter: Optional[set[str]] = None
    if types is not None:
        type_filter = {str(item) for item in types}
    else:
        type_filter = {str(item) for item in SUBSTANTIVE_VALIDATION_TYPES}
    status_filter = (
        {str(item) for item in include_statuses}
        if include_statuses is not None
        else set(REUSABLE_VALIDATION_STATUSES)
    )
    bound = max(0, int(limit))
    matched: list[Artifact] = []
    for artifact in artifacts:
        if len(matched) >= bound:
            break
        if str(artifact.type) not in type_filter:
            continue
        validation = validation_payload_of(artifact)
        if not validation:
            continue
        if str(validation.get("fingerprint") or "") != wanted_fp:
            continue
        status = str(validation.get("status") or "")
        if status not in status_filter:
            continue
        matched.append(artifact)
    return matched
