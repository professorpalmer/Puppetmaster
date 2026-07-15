"""Hashline: compact, content-hash-anchored line edits for agentic workers.

Inspired by Oh My Pi / @oh-my-pi/hashline (MIT). Concepts and prompt economics
are ported; this is a stdlib Python implementation, not a TypeScript vendor.

Format (each file section)::

    [rel/path.py#A1B2]
    SWAP 2.=2:
    +replacement line
    DEL 3
    INS.POST 1:
    +inserted after line 1

Tags are 4-hex fingerprints of the full normalized file. Stale tags are
rejected before any write. Block ops (*.BLK) are not supported — use line
ranges. Kill switch: ``PUPPETMASTER_HASHLINE=0``.
"""
from __future__ import annotations

import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class HashlineError(Exception):
    """Base error for parse / apply failures."""


class UnsupportedError(HashlineError):
    """Raised for *.BLK ops (tree-sitter block ops are out of scope)."""


class StaleTagError(HashlineError):
    """Raised when a section tag does not match the live file snapshot."""


_SECTION_HEADER_RE = re.compile(
    r"^\[(?P<path>[^\]#]+)#(?P<tag>[0-9A-Fa-f]{4})\]\s*$"
)
_SWAP_RE = re.compile(r"^SWAP\s+(?P<start>[1-9]\d*)\.=(?P<end>[1-9]\d*)\s*:\s*$")
_DEL_RANGE_RE = re.compile(r"^DEL\s+(?P<start>[1-9]\d*)\.=(?P<end>[1-9]\d*)\s*$")
_DEL_ONE_RE = re.compile(r"^DEL\s+(?P<start>[1-9]\d*)\s*$")
_INS_PRE_RE = re.compile(r"^INS\.PRE\s+(?P<line>[1-9]\d*)\s*:\s*$")
_INS_POST_RE = re.compile(r"^INS\.POST\s+(?P<line>[1-9]\d*)\s*:\s*$")
_INS_HEAD_RE = re.compile(r"^INS\.HEAD\s*:\s*$")
_INS_TAIL_RE = re.compile(r"^INS\.TAIL\s*:\s*$")
_REM_RE = re.compile(r"^REM\s*$")
_MV_RE = re.compile(r'^MV\s+(?P<dest>"[^"]+"|\S+)\s*$')
_BLK_RE = re.compile(r"^(SWAP|DEL|INS)\.BLK(\.POST)?\b", re.IGNORECASE)


def hashline_enabled() -> bool:
    """True unless ``PUPPETMASTER_HASHLINE`` is an explicit off value (default on)."""
    val = os.environ.get("PUPPETMASTER_HASHLINE", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def fs_cache_enabled() -> bool:
    """True unless ``PUPPETMASTER_FS_CACHE`` is an explicit off value (default on)."""
    val = os.environ.get("PUPPETMASTER_FS_CACHE", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def normalize_text(text: str) -> str:
    """Strip UTF-8 BOM and normalize every line ending to LF."""
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _hash_normalized(text: str) -> str:
    """4-hex tag of full-file text (trailing [ \\t] per line trimmed, like OMP)."""
    trimmed = re.sub(r"[ \t]+(?=\n|$)", "", text)
    low16 = zlib.crc32(trimmed.encode("utf-8")) & 0xFFFF
    return f"{low16:04X}"


def content_tag(text: str) -> str:
    """Return the 4-hex content tag for ``text`` (BOM-stripped, LF-normalized)."""
    return _hash_normalized(normalize_text(text))


def detect_line_ending(content: str) -> str:
    """Return the first line-ending style in ``content`` (``\\r\\n`` or ``\\n``)."""
    crlf = content.find("\r\n")
    lf = content.find("\n")
    if lf < 0:
        return "\n"
    if crlf < 0:
        return "\n"
    return "\r\n" if crlf < lf else "\n"


def restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


@dataclass
class Snapshot:
    path: str
    text: str
    tag: str


class SnapshotStore:
    """In-memory path → version history binding tags to full normalized text."""

    def __init__(self, *, max_versions_per_path: int = 8) -> None:
        self._versions: Dict[str, List[Snapshot]] = {}
        self._max_versions = max_versions_per_path

    def record(self, path: str, text: str) -> str:
        """Record normalized ``text`` for ``path``; return its content tag."""
        path = path.replace("\\", "/")
        normalized = normalize_text(text)
        tag = _hash_normalized(normalized)
        history = self._versions.get(path) or []
        for existing in history:
            if existing.tag == tag and existing.text == normalized:
                if history[0] is not existing:
                    history = [existing] + [s for s in history if s is not existing]
                    self._versions[path] = history
                return tag
        snap = Snapshot(path=path, text=normalized, tag=tag)
        self._versions[path] = ([snap] + history)[: self._max_versions]
        return tag

    def resolve(self, path: str, tag: str) -> Optional[str]:
        """Return recorded normalized text for ``path`` + ``tag``, or None."""
        path = path.replace("\\", "/")
        tag_u = tag.upper()
        for snap in self._versions.get(path) or []:
            if snap.tag == tag_u:
                return snap.text
        return None

    def verify_live(self, path: str, tag: str, live_text: str) -> bool:
        """True when live normalized content matches the tag's recorded snapshot."""
        recorded = self.resolve(path, tag)
        if recorded is None:
            return False
        return normalize_text(live_text) == recorded and content_tag(live_text) == tag.upper()

    def invalidate(self, path: str) -> None:
        self._versions.pop(path.replace("\\", "/"), None)

    def relocate(self, src: str, dest: str) -> None:
        src = src.replace("\\", "/")
        dest = dest.replace("\\", "/")
        history = self._versions.pop(src, None)
        if not history:
            return
        relocated = [Snapshot(path=dest, text=s.text, tag=s.tag) for s in history]
        existing = self._versions.get(dest) or []
        seen = {s.tag for s in relocated}
        merged = relocated + [s for s in existing if s.tag not in seen]
        self._versions[dest] = merged[: self._max_versions]

    def clear(self) -> None:
        self._versions.clear()


@dataclass
class Op:
    kind: str  # swap|del|ins_pre|ins_post|ins_head|ins_tail|rem|mv
    start: int = 0
    end: int = 0
    body: List[str] = field(default_factory=list)
    dest: str = ""
    source_line: int = 0


@dataclass
class Section:
    path: str
    tag: str
    ops: List[Op]
    header_line: int = 1


@dataclass
class SectionResult:
    path: str
    tag: str
    op: str  # update|delete|move
    move_dest: str = ""


@dataclass
class ApplyResult:
    sections: List[SectionResult]
    touched: List[Path]


def format_numbered_read(rel_path: str, tag: str, lines: Sequence[str], *, start_line: int = 1) -> str:
    """Format a tagged, numbered read body: ``[path#TAG]`` then ``N:line`` rows."""
    header = f"[{rel_path}#{tag.upper()}]"
    body = [f"{start_line + i}:{line}" for i, line in enumerate(lines)]
    return "\n".join([header] + body)


def parse_patch(patch: str) -> List[Section]:
    """Parse a multi-section hashline patch into ``Section`` objects."""
    text = normalize_text(patch)
    if not text.strip():
        raise HashlineError("empty hashline patch")

    sections: List[Section] = []
    current: Optional[Section] = None
    pending: Optional[Op] = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is None or current is None:
            pending = None
            return
        if pending.kind in ("swap", "ins_pre", "ins_post", "ins_head", "ins_tail"):
            if not pending.body and pending.kind != "swap":
                raise HashlineError(
                    f"line {pending.source_line}: insert op requires at least one + body row"
                )
        current.ops.append(pending)
        pending = None

    lines = text.split("\n")
    for line_num, raw in enumerate(lines, 1):
        line = raw
        header = _SECTION_HEADER_RE.match(line)
        if header:
            flush_pending()
            if current is not None:
                sections.append(current)
            current = Section(
                path=header.group("path").strip(),
                tag=header.group("tag").upper(),
                ops=[],
                header_line=line_num,
            )
            continue

        if current is None:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raise HashlineError(
                f"line {line_num}: expected [PATH#TAG] section header, got {line!r}"
            )

        if _BLK_RE.match(line.strip()):
            raise UnsupportedError(
                "block ops (*.BLK) are not supported; use SWAP/DEL line ranges "
                "(e.g. SWAP N.=M: / DEL N.=M) instead of tree-sitter block ops"
            )

        if pending is not None and line.startswith("+"):
            pending.body.append(line[1:])
            continue

        # Blank lines between ops are separators (discard); inside a body they
        # must be written as a lone '+' row.
        if not line.strip():
            flush_pending()
            continue

        if line.lstrip().startswith("#") and pending is None:
            continue

        flush_pending()

        m = _SWAP_RE.match(line)
        if m:
            start, end = int(m.group("start")), int(m.group("end"))
            if end < start:
                raise HashlineError(f"line {line_num}: range {start}.={end} ends before it starts")
            pending = Op(kind="swap", start=start, end=end, source_line=line_num)
            continue

        m = _DEL_RANGE_RE.match(line)
        if m:
            start, end = int(m.group("start")), int(m.group("end"))
            if end < start:
                raise HashlineError(f"line {line_num}: range {start}.={end} ends before it starts")
            current.ops.append(Op(kind="del", start=start, end=end, source_line=line_num))
            continue

        m = _DEL_ONE_RE.match(line)
        if m:
            n = int(m.group("start"))
            current.ops.append(Op(kind="del", start=n, end=n, source_line=line_num))
            continue

        m = _INS_PRE_RE.match(line)
        if m:
            pending = Op(kind="ins_pre", start=int(m.group("line")), source_line=line_num)
            continue

        m = _INS_POST_RE.match(line)
        if m:
            pending = Op(kind="ins_post", start=int(m.group("line")), source_line=line_num)
            continue

        if _INS_HEAD_RE.match(line):
            pending = Op(kind="ins_head", source_line=line_num)
            continue

        if _INS_TAIL_RE.match(line):
            pending = Op(kind="ins_tail", source_line=line_num)
            continue

        if _REM_RE.match(line):
            current.ops.append(Op(kind="rem", source_line=line_num))
            continue

        m = _MV_RE.match(line)
        if m:
            dest = m.group("dest")
            if dest.startswith('"') and dest.endswith('"'):
                dest = dest[1:-1]
            current.ops.append(Op(kind="mv", dest=dest, source_line=line_num))
            continue

        raise HashlineError(
            f"line {line_num}: unrecognized hashline op {line!r}. "
            "Use SWAP N.=M:, DEL N.=M / DEL N, INS.PRE/POST/HEAD/TAIL, REM, or MV."
        )

    flush_pending()
    if current is not None:
        sections.append(current)
    if not sections:
        raise HashlineError("hashline patch has no [PATH#TAG] sections")
    return sections


def _apply_ops_to_lines(lines: List[str], ops: Sequence[Op]) -> List[str]:
    """Apply line ops using ORIGINAL line numbers (bottom-up splice)."""
    file_op = None
    line_ops: List[Op] = []
    for op in ops:
        if op.kind in ("rem", "mv"):
            if file_op is not None:
                raise HashlineError("only one file-level op (REM or MV) per section")
            file_op = op
            if op.kind == "rem" and line_ops:
                raise HashlineError("REM deletes the whole file and cannot be combined with line ops")
        else:
            if file_op is not None and file_op.kind == "rem":
                raise HashlineError("REM deletes the whole file and cannot be combined with line ops")
            line_ops.append(op)

    if file_op is not None and file_op.kind == "rem":
        return []  # caller deletes the file

    # Phantom trailing empty from split("\n") on newline-terminated text:
    # addressable for inserts, not a real delete target.
    n_lines = len(lines)
    phantom = n_lines if (n_lines > 1 and lines[-1] == "") else 0

    # Validate bounds against concrete content lines.
    content_len = n_lines - 1 if phantom else n_lines
    if content_len == 0 and n_lines == 1 and lines[0] == "":
        content_len = 0

    def _check_line(n: int, *, allow_phantom_insert: bool = False) -> None:
        if n < 1:
            raise HashlineError(f"Line {n} does not exist (file has {content_len} lines)")
        upper = n_lines if allow_phantom_insert else (content_len if content_len > 0 else n_lines)
        if content_len == 0 and n_lines == 1 and lines[0] == "":
            if not allow_phantom_insert and n != 1:
                raise HashlineError(f"Line {n} does not exist (file has 0 lines)")
            return
        if n > upper:
            raise HashlineError(f"Line {n} does not exist (file has {content_len} lines)")

    for op in line_ops:
        if op.kind == "swap":
            _check_line(op.start)
            _check_line(op.end)
        elif op.kind == "del":
            _check_line(op.start)
            _check_line(op.end)
        elif op.kind in ("ins_pre", "ins_post"):
            _check_line(op.start, allow_phantom_insert=True)

    # Expand into per-line edit buckets (original line numbers).
    before: Dict[int, List[str]] = {}
    after: Dict[int, List[str]] = {}
    replace: Dict[int, List[str]] = {}
    delete: set = set()
    bof: List[str] = []
    eof: List[str] = []

    for op in line_ops:
        if op.kind == "swap":
            for ln in range(op.start, op.end + 1):
                if ln in delete or ln in replace:
                    raise HashlineError(
                        f"anchor line {ln} is already targeted by another hunk"
                    )
                delete.add(ln)
            replace.setdefault(op.start, []).extend(op.body)
        elif op.kind == "del":
            for ln in range(op.start, op.end + 1):
                if phantom and ln == phantom:
                    continue
                if ln in delete:
                    raise HashlineError(
                        f"anchor line {ln} is already targeted by another hunk"
                    )
                delete.add(ln)
        elif op.kind == "ins_pre":
            before.setdefault(op.start, []).extend(op.body)
        elif op.kind == "ins_post":
            after.setdefault(op.start, []).extend(op.body)
        elif op.kind == "ins_head":
            bof.extend(op.body)
        elif op.kind == "ins_tail":
            eof.extend(op.body)

    work = list(lines)
    origins_touched = sorted(
        set(before) | set(after) | set(replace) | delete,
        reverse=True,
    )
    for ln in origins_touched:
        idx = ln - 1
        if idx < 0 or idx >= len(work):
            # Allow insert-after on last phantom / EOF edge cases already validated.
            if ln in after and idx == len(work):
                work.extend(after[ln])
            continue
        current = work[idx]
        pre = before.get(ln, [])
        rep = replace.get(ln, [])
        post = after.get(ln, [])
        if ln in delete:
            replacement = pre + rep + post
        else:
            replacement = pre + rep + [current] + post
        work[idx:idx + 1] = replacement

    if bof:
        if len(work) == 1 and work[0] == "":
            work = list(bof)
        else:
            work = list(bof) + work
    if eof:
        if len(work) == 1 and work[0] == "":
            work = list(eof)
        elif work and work[-1] == "":
            work = work[:-1] + list(eof) + [""]
        else:
            work = work + list(eof)

    return work


def _resolve_under_cwd(cwd: Path, rel: str) -> Path:
    expanded = Path(rel).expanduser()
    if expanded.is_absolute():
        target = expanded.resolve()
    else:
        target = (cwd / expanded).resolve()
    root = cwd.resolve()
    if root != target and root not in target.parents:
        raise HashlineError(f"path {rel!r} escapes the workspace")
    return target


def apply_section_text(normalized: str, ops: Sequence[Op]) -> str:
    """Apply section ops to LF-normalized text; return LF-normalized result."""
    lines = normalized.split("\n")
    new_lines = _apply_ops_to_lines(lines, ops)
    if any(op.kind == "rem" for op in ops):
        return ""
    return "\n".join(new_lines)


def apply_patch(cwd: Path, patch: str, store: SnapshotStore) -> ApplyResult:
    """Parse, preflight all sections, then write. Rejects stale tags; atomic.

    Preflight validates every section (tag, bounds, parse) before any file is
    written. On success, updates ``store`` with post-edit snapshots.
    """
    cwd = Path(cwd).resolve()
    sections = parse_patch(patch)

    prepared: List[Tuple[Section, Path, str, str, str, Optional[str], List[Op]]] = []
    # (section, abs_path, raw, bom, ending, after_text|None if rem, line_ops)

    for section in sections:
        # Canonicalize to forward slashes so Windows reads and model headers match.
        section.path = section.path.replace("\\", "/")
        abs_path = _resolve_under_cwd(cwd, section.path)
        if not abs_path.is_file():
            raise HashlineError(
                f"{section.path}: file does not exist (hashline only edits existing files; "
                "use write_file to create new ones)"
            )
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
        bom = "\ufeff" if raw.startswith("\ufeff") else ""
        ending = detect_line_ending(raw)
        normalized = normalize_text(raw)

        recorded = store.resolve(section.path, section.tag)
        if recorded is None:
            # Allow apply when the live file itself hashes to the tag (re-entry
            # without an explicit prior record), then bind it into the store.
            if content_tag(normalized) != section.tag.upper():
                raise StaleTagError(
                    f"{section.path}#{section.tag}: unknown or stale tag — "
                    "re-read the file and use the tag from the latest read"
                )
            store.record(section.path, normalized)
            recorded = normalized
        elif recorded != normalized:
            raise StaleTagError(
                f"{section.path}#{section.tag}: stale tag — live file no longer "
                "matches the snapshot; re-read before editing"
            )
        elif content_tag(normalized) != section.tag.upper():
            raise StaleTagError(
                f"{section.path}#{section.tag}: tag mismatch with live content; re-read"
            )

        rem_ops = [op for op in section.ops if op.kind == "rem"]
        mv_ops = [op for op in section.ops if op.kind == "mv"]
        line_ops = [op for op in section.ops if op.kind not in ("rem", "mv")]
        if rem_ops and (line_ops or mv_ops):
            raise HashlineError("REM cannot be combined with other ops in a section")
        if len(rem_ops) > 1 or len(mv_ops) > 1:
            raise HashlineError("only one REM or MV per section")

        if rem_ops:
            prepared.append((section, abs_path, raw, bom, ending, None, list(section.ops)))
            continue

        after = apply_section_text(normalized, line_ops)
        move_dest: Optional[str] = mv_ops[0].dest if mv_ops else None
        if move_dest is not None:
            move_dest = move_dest.replace("\\", "/")
            mv_ops[0].dest = move_dest
            dest_path = _resolve_under_cwd(cwd, move_dest)
            # Preflight dest parent exists or is creatable; reject self-clobber oddly.
            if dest_path.exists() and dest_path.resolve() != abs_path.resolve():
                raise HashlineError(f"MV destination already exists: {move_dest}")
        prepared.append((section, abs_path, raw, bom, ending, after, list(section.ops)))

    # Commit phase — all sections validated.
    results: List[SectionResult] = []
    touched: List[Path] = []

    for section, abs_path, raw, bom, ending, after, ops in prepared:
        rem = any(op.kind == "rem" for op in ops)
        mv = next((op for op in ops if op.kind == "mv"), None)
        rel = section.path

        if rem:
            abs_path.unlink()
            store.invalidate(rel)
            touched.append(abs_path)
            results.append(SectionResult(path=rel, tag="", op="delete"))
            continue

        assert after is not None
        persisted = bom + restore_line_endings(after, ending)
        if mv is not None:
            dest_path = _resolve_under_cwd(cwd, mv.dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(persisted, encoding="utf-8")
            if dest_path.resolve() != abs_path.resolve():
                abs_path.unlink()
            store.relocate(rel, mv.dest)
            new_tag = store.record(mv.dest, after)
            touched.extend([abs_path, dest_path])
            results.append(
                SectionResult(path=rel, tag=new_tag, op="move", move_dest=mv.dest)
            )
        else:
            abs_path.write_text(persisted, encoding="utf-8")
            new_tag = store.record(rel, after)
            touched.append(abs_path)
            results.append(SectionResult(path=rel, tag=new_tag, op="update"))

    return ApplyResult(sections=results, touched=touched)


def format_apply_success(result: ApplyResult) -> str:
    """Human/model-facing success summary with fresh tags for follow-up edits."""
    lines = ["ok: hashline applied"]
    for sec in result.sections:
        if sec.op == "delete":
            lines.append(f"  deleted {sec.path}")
        elif sec.op == "move":
            lines.append(f"  moved {sec.path} -> {sec.move_dest}#{sec.tag}")
        else:
            lines.append(f"  updated [{sec.path}#{sec.tag}]")
    lines.append("Re-ground the next edit on these tags (or a fresh read).")
    return "\n".join(lines)
