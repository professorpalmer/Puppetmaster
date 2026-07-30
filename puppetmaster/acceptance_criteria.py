"""Optional harness acceptance criteria preserved through prompt assembly.

Workers must not substitute a generic repo audit for explicit acceptance
criteria. Criteria are parsed only from an explicit ``Acceptance criteria:``
block or a structured field, bounded in count/length, and re-anchored after
``Your task:`` so job-brief / memory / CodeGraph inserts cannot displace them.

Verification artifacts report per-criterion status. Omitted criteria become
``unknown`` / ``not_reported`` — never silently ``passed``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

MAX_ACCEPTANCE_CRITERIA = 20
MAX_CRITERION_CHARS = 500

_CRITERIA_HEADER_RE = re.compile(
    r"(?im)^[ \t]*Acceptance criteria:[ \t]*\n"
)
_CRITERION_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*]|\d+[.)])[ \t]+(.+?)\s*$"
)


def normalize_acceptance_criteria(value: Any) -> list[str]:
    """Normalize a structured acceptance_criteria field into bounded strings."""
    if value is None:
        return []
    raw_items: list[Any]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        raw_items = [line.strip() for line in text.splitlines() if line.strip()]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        return []

    criteria: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        # Strip a leading bullet if a structured list already included markers.
        text = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", text).strip()
        if not text:
            continue
        if len(text) > MAX_CRITERION_CHARS:
            text = text[:MAX_CRITERION_CHARS].rstrip()
        criteria.append(text)
        if len(criteria) >= MAX_ACCEPTANCE_CRITERIA:
            break
    return criteria


def parse_acceptance_criteria_block(text: str) -> list[str]:
    """Parse only an explicit ``Acceptance criteria:`` block from free text."""
    if not text or not isinstance(text, str):
        return []
    match = _CRITERIA_HEADER_RE.search(text)
    if not match:
        return []
    rest = text[match.end() :]
    criteria: list[str] = []
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped:
            if criteria:
                break
            continue
        # A new markdown-style header ends the block.
        if re.match(r"^[A-Za-z].*:$", stripped) and not _CRITERION_LINE_RE.match(line):
            break
        item_match = _CRITERION_LINE_RE.match(line)
        if item_match:
            item = item_match.group(1).strip()
            if item:
                if len(item) > MAX_CRITERION_CHARS:
                    item = item[:MAX_CRITERION_CHARS].rstrip()
                criteria.append(item)
                if len(criteria) >= MAX_ACCEPTANCE_CRITERIA:
                    break
            continue
        # Indented continuation of the previous criterion.
        if criteria and line.startswith((" ", "\t")):
            cont = stripped
            combined = f"{criteria[-1]} {cont}".strip()
            if len(combined) > MAX_CRITERION_CHARS:
                combined = combined[:MAX_CRITERION_CHARS].rstrip()
            criteria[-1] = combined
            continue
        break
    return criteria


def acceptance_criteria_for_task(task: Any) -> list[str]:
    """Resolve criteria from structured payload/config, else instruction block."""
    payload = getattr(task, "payload", None) or {}
    if isinstance(payload, dict):
        structured = normalize_acceptance_criteria(payload.get("acceptance_criteria"))
        if structured:
            return structured
    # Top-level optional attribute for callers that set it directly.
    structured = normalize_acceptance_criteria(
        getattr(task, "acceptance_criteria", None)
    )
    if structured:
        return structured
    instruction = getattr(task, "instruction", None) or ""
    return parse_acceptance_criteria_block(str(instruction))


def format_acceptance_criteria_block(criteria: Sequence[str]) -> str:
    items = normalize_acceptance_criteria(list(criteria))
    if not items:
        return ""
    lines = ["Acceptance criteria:"]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _criteria_block_span(text: str, start: int) -> tuple[int, int]:
    """Return ``[header_start, block_end)`` for a criteria block at ``start``."""
    rest = text[start:]
    match = _CRITERIA_HEADER_RE.match(rest)
    if not match:
        return start, start
    body_start = start + match.end()
    end = body_start
    saw_item = False
    offset = 0
    for line in rest[match.end() :].splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            if saw_item:
                break
            offset += len(line)
            continue
        if re.match(r"^[A-Za-z].*:$", stripped) and not _CRITERION_LINE_RE.match(line):
            break
        if _CRITERION_LINE_RE.match(line):
            saw_item = True
            offset += len(line)
            end = body_start + offset
            continue
        if saw_item and line.startswith((" ", "\t")):
            offset += len(line)
            end = body_start + offset
            continue
        break
    if not saw_item:
        end = body_start
    return start, end


def _replace_acceptance_criteria_blocks(text: str, block: str) -> str:
    """Replace every explicit criteria block with ``block``; keep one copy."""
    if not text:
        return block
    spans: list[tuple[int, int]] = []
    for match in _CRITERIA_HEADER_RE.finditer(text):
        spans.append(_criteria_block_span(text, match.start()))
    if not spans:
        body = text.rstrip()
        if not body:
            return block
        return body + "\n\n" + block
    # Rebuild with the canonical block at the first span; drop later duplicates.
    parts: list[str] = []
    cursor = 0
    for index, (start, end) in enumerate(spans):
        parts.append(text[cursor:start])
        if index == 0:
            parts.append(block)
        cursor = end
        # Swallow a single trailing newline after removed duplicate blocks.
        if index > 0 and cursor < len(text) and text[cursor] == "\n":
            cursor += 1
    parts.append(text[cursor:])
    rebuilt = "".join(parts)
    # Collapse accidental triple blank lines introduced by span surgery.
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt).rstrip()
    return rebuilt


def ensure_acceptance_criteria_in_text(
    text: str, criteria: Sequence[str]
) -> str:
    """Ensure ``text`` carries the canonical Acceptance criteria block.

    Structured ``criteria`` win. A divergent existing explicit block is replaced
    deterministically; identical duplicate blocks are collapsed to one. Text
    outside the criteria block is left unchanged.
    """
    items = normalize_acceptance_criteria(list(criteria))
    if not items:
        return text
    block = format_acceptance_criteria_block(items)
    existing = parse_acceptance_criteria_block(text or "")
    header_count = len(_CRITERIA_HEADER_RE.findall(text or ""))
    if existing == items and header_count <= 1:
        return text
    if existing or header_count:
        return _replace_acceptance_criteria_blocks(text or "", block)
    body = (text or "").rstrip()
    if not body:
        return block
    return body + "\n\n" + block


def attach_acceptance_criteria_to_task_payload(task: Any) -> Any:
    """Return a task copy with ``payload.acceptance_criteria`` populated.

    Leaves ``instruction`` unchanged. Safe no-op when nothing to attach.
    """
    from dataclasses import replace

    criteria = acceptance_criteria_for_task(task)
    if not criteria:
        return task
    payload = dict(getattr(task, "payload", None) or {})
    if normalize_acceptance_criteria(payload.get("acceptance_criteria")):
        return task
    payload["acceptance_criteria"] = list(criteria)
    return replace(task, payload=payload)


def criterion_status_records(
    criteria: Sequence[str],
    reported: Optional[Iterable[Any]] = None,
) -> list[dict[str, Any]]:
    """Build verification criterion rows.

    Omitted / unmatched criteria are ``status=unknown`` with
    ``evidence=not_reported`` — never ``passed``.
    """
    items = normalize_acceptance_criteria(list(criteria))
    reported_map: dict[str, dict[str, Any]] = {}
    if reported:
        for entry in reported:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("criterion") or entry.get("text") or "").strip()
            if not key:
                continue
            status = str(entry.get("status") or "unknown").strip().lower()
            if status not in {"passed", "failed", "unknown", "not_reported"}:
                status = "unknown"
            evidence = entry.get("evidence")
            if evidence is None or evidence == "":
                evidence = "not_reported" if status in {"unknown", "not_reported"} else ""
            reported_map[key] = {
                "criterion": key,
                "status": "unknown" if status == "not_reported" else status,
                "evidence": evidence if evidence != "" else "not_reported",
            }

    rows: list[dict[str, Any]] = []
    for item in items:
        if item in reported_map:
            rows.append(reported_map[item])
        else:
            rows.append(
                {
                    "criterion": item,
                    "status": "unknown",
                    "evidence": "not_reported",
                }
            )
    return rows


def stamp_verification_acceptance_criteria(
    artifact: Any, task: Any
) -> Any:
    """Attach criterion status rows to a VERIFICATION artifact payload."""
    from dataclasses import replace

    from puppetmaster.models import Artifact, ArtifactType

    if not isinstance(artifact, Artifact) or artifact.type != ArtifactType.VERIFICATION:
        return artifact
    criteria = acceptance_criteria_for_task(task)
    if not criteria:
        return artifact
    payload = dict(artifact.payload or {})
    existing = payload.get("acceptance_criteria")
    reported = None
    if isinstance(existing, list) and existing and isinstance(existing[0], dict):
        reported = existing
    payload["acceptance_criteria"] = criterion_status_records(criteria, reported)
    return replace(artifact, payload=payload, sha256=None)
