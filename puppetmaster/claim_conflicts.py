"""Deterministic stitch-time contradictory-peer detection.

Narrow checks only — ambiguous pairs stay unknown. High-confidence findings
that share an evidence locus or symbol and make incompatible claims are
removed from ordinary Findings and surfaced as visible Conflicts with
artifact ids, claims, evidence, downgraded confidence, and source-verification
status. Path:line evidence is resolved only under ``cwd``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from puppetmaster.models import Artifact, ArtifactType

_HIGH_CONFIDENCE = 0.8
_DOWNGRADED_CONFIDENCE = 0.35

_LOCK_TYPES = ("threading.Lock", "asyncio.Lock")

_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
_PATH_LINE_RE = re.compile(
    r"\b([\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,9}):(\d+)\b"
)

_MAX_SOURCE_BYTES = 1_048_576  # 1 MiB

_ALLOWED_CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".m",
        ".mm",
        ".scala",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".txt",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".css",
        ".scss",
        ".html",
        ".vue",
        ".svelte",
    }
)

_SENSITIVE_NAME_RE = re.compile(
    r"(?i)(?:^|\.)(?:env|pem|key)$|"
    r"(?:credential|secret|password|private.?key|id_rsa|passwd|token)"
)

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "uses",
        "used",
        "use",
        "when",
        "where",
        "which",
        "should",
        "would",
        "could",
        "must",
        "not",
        "none",
        "true",
        "false",
        "lock",
        "file",
        "line",
        "code",
        "function",
        "method",
        "class",
        "module",
        "adapter",
        "result",
        "status",
        "claim",
        "finding",
        "bug",
        "error",
        "same",
        "also",
        "than",
        "then",
        "via",
        "per",
        "its",
        "has",
        "have",
        "been",
        "are",
        "was",
        "were",
        "is",
        "in",
        "on",
        "at",
        "of",
        "to",
        "a",
        "an",
        "or",
        "by",
    }
)

_NON_LOCUS_PREFIXES = frozenset(
    {
        "adapter",
        "context",
        "result",
        "status",
        "mode",
        "base",
        "exit",
        "node",
        "retry",
        "check",
    }
)


@dataclass(frozen=True)
class ClaimConflict:
    artifact_ids: tuple[str, str]
    claims: tuple[str, str]
    evidence: tuple[tuple[str, ...], tuple[str, ...]]
    reason: str
    confidence: float
    source_verification: str  # supports | contradicts | unknown | mixed

    def to_summary_lines(self) -> list[str]:
        left_id, right_id = self.artifact_ids
        left_claim, right_claim = self.claims
        left_ev = ", ".join(self.evidence[0]) or "no evidence"
        right_ev = ", ".join(self.evidence[1]) or "no evidence"
        return [
            (
                f"- CONFLICT ({self.reason}): "
                f"{left_id} vs {right_id} "
                f"[source_verification={self.source_verification}; "
                f"confidence={self.confidence:.2f}]"
            ),
            f"  - {left_id}: {left_claim}",
            f"    evidence={left_ev}",
            f"  - {right_id}: {right_claim}",
            f"    evidence={right_ev}",
        ]


def _claim_text(artifact: Artifact) -> str:
    payload = artifact.payload or {}
    for key in ("claim", "risk", "decision"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _lock_type_in_text(text: str) -> Optional[str]:
    lowered = text.lower()
    found = None
    for lock_type in _LOCK_TYPES:
        if lock_type.lower() in lowered:
            if found and found != lock_type:
                return None  # ambiguous within one claim
            found = lock_type
    return found


def _lock_subjects_in_text(text: str) -> frozenset[str]:
    """Explicit lock variable names (e.g. ``_listener_lock``), not modules or types."""
    subjects: set[str] = set()
    for symbol in _symbols_in_text(text):
        lower = symbol.lower()
        if lower in {"lock", "locks"}:
            continue
        # snake_case ..._lock / bare _lock, or CamelCase ...Lock (not bare Lock).
        if lower.endswith("_lock") or lower == "_lock":
            subjects.add(symbol)
            continue
        if symbol.endswith("Lock") and symbol != "Lock" and symbol[0].isupper():
            subjects.add(symbol)
    return frozenset(subjects)


def _symbols_in_text(text: str) -> frozenset[str]:
    """Extract API-like identifiers (snake_case / CamelCase), not prose words."""
    symbols = set()
    for match in _SYMBOL_RE.finditer(text or ""):
        token = match.group(1)
        lower = token.lower()
        if lower in _STOPWORDS:
            continue
        if token in {"Lock", "Path", "Optional", "True", "False", "None"}:
            continue
        # Prefer API identifiers: snake_case, or mixed CamelCase, or known dotted
        # lock/type tails. Pure lowercase prose words are ignored.
        is_snake = "_" in token
        is_camel = any(ch.isupper() for ch in token[1:]) and token[0].isupper()
        if not (is_snake or is_camel):
            continue
        symbols.add(token)
    # Dotted lock forms are handled separately; also catch explicit foo.bar APIs.
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b", text or ""
    ):
        dotted = match.group(1)
        if dotted in _LOCK_TYPES:
            continue
        symbols.add(dotted.split(".")[-1])
    return frozenset(symbols)


def _path_line_refs(evidence: Sequence[str]) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for item in evidence or []:
        if not isinstance(item, str):
            continue
        head = item.split(":", 1)[0].strip().lower()
        if "." not in head and head in _NON_LOCUS_PREFIXES:
            continue
        for match in _PATH_LINE_RE.finditer(item):
            refs.append((match.group(1).replace("\\", "/"), int(match.group(2))))
    return refs


def _loci(evidence: Sequence[str]) -> frozenset[str]:
    return frozenset(path.lower() for path, _line in _path_line_refs(evidence))


def _line_loci(evidence: Sequence[str]) -> frozenset[tuple[str, int]]:
    """Exact (path, line) loci — never path-only."""
    return frozenset((path.lower(), line) for path, line in _path_line_refs(evidence))


def _overlapping_line_loci(
    left_evidence: Sequence[str], right_evidence: Sequence[str]
) -> bool:
    """True when both cite the same path with at least one shared line number."""
    return bool(_line_loci(left_evidence) & _line_loci(right_evidence))


def _compatible_subjects(left_claim: str, right_claim: str) -> bool:
    """Subjects are compatible when they share a symbol or both name API subjects."""
    left_symbols = _symbols_in_text(left_claim)
    right_symbols = _symbols_in_text(right_claim)
    if left_symbols & right_symbols:
        return True
    # Competing identifier claims at one locus (wrong symbol vs right symbol).
    if left_symbols and right_symbols:
        return True
    if _lock_type_in_text(left_claim) or _lock_type_in_text(right_claim):
        return True
    return False


def _symbol_line_map(
    claim: str, evidence: Sequence[str]
) -> dict[str, set[int]]:
    """Map symbols mentioned in the claim to line numbers cited in evidence."""
    symbols = _symbols_in_text(claim)
    refs = _path_line_refs(evidence)
    mapping: dict[str, set[int]] = {}
    if not symbols or not refs:
        return mapping
    for symbol in symbols:
        lines = {line for _path, line in refs}
        if lines:
            mapping[symbol] = lines
    return mapping


def _is_sensitive_path(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if _SENSITIVE_NAME_RE.search(name):
        return True
    return False


def _allowed_source_file(path: Path, cwd: Path) -> bool:
    """Reject escapes, secrets, unsupported/binary types, and oversized files."""
    try:
        cwd_resolved = cwd.resolve()
        if path.is_symlink():
            try:
                path.resolve().relative_to(cwd_resolved)
            except (OSError, ValueError):
                return False
        resolved = path.resolve()
        try:
            resolved.relative_to(cwd_resolved)
        except ValueError:
            return False
        if not resolved.is_file():
            return False
        if _is_sensitive_path(resolved):
            return False
        if resolved.suffix.lower() not in _ALLOWED_CODE_EXTENSIONS:
            return False
        if resolved.stat().st_size > _MAX_SOURCE_BYTES:
            return False
        return True
    except OSError:
        return False


def _resolve_under_cwd(path_text: str, cwd: Path) -> Optional[Path]:
    candidate = Path(path_text)
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
            cwd_resolved = cwd.resolve()
            resolved.relative_to(cwd_resolved)
            if _allowed_source_file(resolved, cwd):
                return resolved
            return None
        except (OSError, ValueError):
            return None
    # Prefer exact relative path; basename-only only when exactly one allowed match.
    direct = (cwd / candidate).resolve()
    try:
        direct.relative_to(cwd.resolve())
    except ValueError:
        return None
    if direct.is_file():
        return direct if _allowed_source_file(direct, cwd) else None
    name = candidate.name
    if name and "/" not in path_text.replace("\\", "/"):
        try:
            cwd_resolved = cwd.resolve()
            matches = sorted(
                {
                    child.resolve()
                    for child in cwd.rglob(name)
                    if child.is_file() and _allowed_source_file(child, cwd)
                }
            )
            # Ambiguous basename → unknown (do not guess).
            if len(matches) == 1:
                return matches[0]
        except OSError:
            return None
    return None


def _read_source_lines(resolved: Path) -> Optional[list[str]]:
    try:
        if resolved.stat().st_size > _MAX_SOURCE_BYTES:
            return None
        return resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def source_verify_claim(
    claim: str,
    evidence: Sequence[str],
    *,
    cwd: Optional[Path],
) -> str:
    """Return supports | contradicts | unknown for resolvable path:line evidence."""
    if cwd is None:
        return "unknown"
    try:
        root = Path(cwd)
        if not root.is_dir():
            return "unknown"
    except OSError:
        return "unknown"

    refs = _path_line_refs(evidence)
    if not refs:
        return "unknown"

    symbols = _symbols_in_text(claim)
    lock_type = _lock_type_in_text(claim)
    statuses: set[str] = set()

    for path_text, line_no in refs:
        resolved = _resolve_under_cwd(path_text, root)
        if resolved is None:
            statuses.add("unknown")
            continue
        lines = _read_source_lines(resolved)
        if lines is None:
            statuses.add("unknown")
            continue
        if line_no < 1 or line_no > len(lines):
            statuses.add("contradicts")
            continue
        line_text = lines[line_no - 1]
        if lock_type:
            if lock_type in line_text:
                statuses.add("supports")
            elif any(other in line_text for other in _LOCK_TYPES if other != lock_type):
                statuses.add("contradicts")
            elif "Lock" in line_text:
                statuses.add("contradicts")
            else:
                statuses.add("unknown")
        if symbols:
            hit = any(symbol in line_text for symbol in symbols)
            # Also accept dotted forms like threading.Lock already handled.
            if hit:
                statuses.add("supports")
            else:
                # Only contradict when the claim asserts a concrete symbol that
                # is absent from the cited line and the line is non-empty.
                if line_text.strip():
                    statuses.add("contradicts")
                else:
                    statuses.add("unknown")
        if not symbols and not lock_type:
            statuses.add("unknown")

    if not statuses:
        return "unknown"
    if statuses == {"supports"}:
        return "supports"
    if statuses == {"contradicts"}:
        return "contradicts"
    if "contradicts" in statuses and "supports" in statuses:
        return "mixed"
    if "contradicts" in statuses:
        return "contradicts"
    if "supports" in statuses:
        return "supports"
    return "unknown"


def _pair_incompatible(
    left: Artifact,
    right: Artifact,
    *,
    cwd: Optional[Path],
) -> Optional[tuple[str, str]]:
    """Return (reason, source_verification) when the pair is a deterministic conflict."""
    if left.confidence < _HIGH_CONFIDENCE or right.confidence < _HIGH_CONFIDENCE:
        return None
    left_claim = _claim_text(left)
    right_claim = _claim_text(right)
    if not left_claim or not right_claim:
        return None

    left_loci = _loci(left.evidence)
    right_loci = _loci(right.evidence)
    shared_locus = left_loci & right_loci
    left_symbols = _symbols_in_text(left_claim)
    right_symbols = _symbols_in_text(right_claim)
    shared_symbols = left_symbols & right_symbols
    shared_line = _overlapping_line_loci(left.evidence, right.evidence)

    if not shared_locus and not shared_symbols:
        return None

    left_lock = _lock_type_in_text(left_claim)
    right_lock = _lock_type_in_text(right_claim)
    if left_lock and right_lock and left_lock != right_lock:
        # Distinct lock types in one file are legitimate when they cite different
        # lines and name different subjects. Conflict only on overlapping
        # path:line evidence or a shared explicit lock subject (e.g. _listener_lock).
        shared_lock_subjects = (
            _lock_subjects_in_text(left_claim) & _lock_subjects_in_text(right_claim)
        )
        if shared_line or shared_lock_subjects:
            left_sv = source_verify_claim(left_claim, left.evidence, cwd=cwd)
            right_sv = source_verify_claim(right_claim, right.evidence, cwd=cwd)
            if left_sv == "supports" and right_sv == "contradicts":
                sv = "supports"
            elif right_sv == "supports" and left_sv == "contradicts":
                sv = "supports"
            elif left_sv == "contradicts" and right_sv == "contradicts":
                sv = "contradicts"
            elif left_sv == "unknown" and right_sv == "unknown":
                sv = "unknown"
            else:
                sv = "mixed" if {left_sv, right_sv} != {"unknown"} else "unknown"
            return ("incompatible_lock_types", sv)

    left_map = _symbol_line_map(left_claim, left.evidence)
    right_map = _symbol_line_map(right_claim, right.evidence)
    for symbol in shared_symbols:
        left_lines = left_map.get(symbol) or set()
        right_lines = right_map.get(symbol) or set()
        if left_lines and right_lines and left_lines.isdisjoint(right_lines):
            left_sv = source_verify_claim(left_claim, left.evidence, cwd=cwd)
            right_sv = source_verify_claim(right_claim, right.evidence, cwd=cwd)
            if left_sv == right_sv:
                sv = left_sv
            elif "contradicts" in {left_sv, right_sv} and "supports" in {
                left_sv,
                right_sv,
            }:
                sv = "mixed"
            else:
                sv = "unknown"
            return ("conflicting_symbol_lines", sv)

    # Shared exact path:line (or overlapping cited lines) + compatible subjects
    # with opposing source-verification — never path-only.
    if shared_line and _compatible_subjects(left_claim, right_claim):
        left_sv = source_verify_claim(left_claim, left.evidence, cwd=cwd)
        right_sv = source_verify_claim(right_claim, right.evidence, cwd=cwd)
        if {left_sv, right_sv} == {"supports", "contradicts"}:
            return ("source_verification_disagreement", "mixed")

    return None


def detect_contradictory_peers(
    artifacts: Sequence[Artifact],
    *,
    cwd: Optional[Path] = None,
) -> list[ClaimConflict]:
    """Find high-confidence incompatible FINDING peers before dedupe/render."""
    findings = [
        artifact
        for artifact in artifacts
        if artifact.type == ArtifactType.FINDING
        and artifact.confidence >= _HIGH_CONFIDENCE
        and _claim_text(artifact)
    ]
    conflicts: list[ClaimConflict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(findings):
        for right in findings[index + 1 :]:
            pair_key = tuple(sorted((left.id, right.id)))
            if pair_key in seen_pairs:
                continue
            result = _pair_incompatible(left, right, cwd=cwd)
            if result is None:
                continue
            reason, source_verification = result
            seen_pairs.add(pair_key)
            conflicts.append(
                ClaimConflict(
                    artifact_ids=(left.id, right.id),
                    claims=(_claim_text(left), _claim_text(right)),
                    evidence=(
                        tuple(left.evidence or ()),
                        tuple(right.evidence or ()),
                    ),
                    reason=reason,
                    confidence=_DOWNGRADED_CONFIDENCE,
                    source_verification=source_verification,
                )
            )
    return conflicts


def conflicting_artifact_ids(conflicts: Sequence[ClaimConflict]) -> frozenset[str]:
    ids: set[str] = set()
    for conflict in conflicts:
        ids.update(conflict.artifact_ids)
    return frozenset(ids)


def conflicts_as_payloads(conflicts: Sequence[ClaimConflict]) -> list[dict[str, Any]]:
    """Structured conflict records for tests / MCP consumers."""
    rows: list[dict[str, Any]] = []
    for conflict in conflicts:
        rows.append(
            {
                "artifact_ids": list(conflict.artifact_ids),
                "claims": list(conflict.claims),
                "evidence": [list(conflict.evidence[0]), list(conflict.evidence[1])],
                "reason": conflict.reason,
                "confidence": conflict.confidence,
                "source_verification": conflict.source_verification,
            }
        )
    return rows
