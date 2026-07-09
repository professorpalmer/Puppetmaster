"""Savings-gated tool-output offload for model-facing agentic results.

Large tool results are spilled to a durable blob under
``{state_dir}/tool_offload/`` and replaced in the conversation with a
head/tail preview plus a path pointer. Offload only happens when the
savings gate says the replacement is worth it (token floor + margin).

Policy (never raises from the hot path):

* Kill switch: ``PUPPETMASTER_TOOL_OFFLOAD=0`` disables spill entirely.
* Floor: original must be at least ``PUPPETMASTER_OFFLOAD_MIN_TOKENS``
  (default 3000, chars//4) before offload is considered.
* Margin: replacement must be at most ``PUPPETMASTER_OFFLOAD_MARGIN``
  (default 0.9) of the original char count.
* Below threshold / gate fail: return the original text unless it exceeds
  the hard cap (``HARD_CAP_CHARS``, default 48000), in which case apply
  last-resort ``_truncate``-style soft truncation.
* Write failure: fall back to the same hard-cap truncate; never break the
  tool loop.

Measured savings (chars->tokens via chars//4) are appended as numbers-only
JSONL under ``{state_dir}/tool_output_savings.jsonl`` so
``python -m puppetmaster savings`` can report a "tool-output offload" line
without inventing USD.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

CHARS_PER_TOKEN = 4
MIN_TOOL_RESULT_TOKENS = 3000
SAVINGS_MARGIN = 0.9
DEFAULT_HEAD_CHARS = 2000
DEFAULT_TAIL_CHARS = 2000
HARD_CAP_CHARS = 48000
OFFLOAD_SUBDIR = "tool_offload"
SAVINGS_JSONL = "tool_output_savings.jsonl"
ENTRY_KIND = "tool_output_offload"

PathLike = Union[str, Path]


def estimate_tokens(char_count: int) -> int:
    """Deterministic token estimate from character count (chars//4)."""
    try:
        return max(0, int(char_count) // CHARS_PER_TOKEN)
    except (TypeError, ValueError):
        return 0


def tokens_avoided(original_chars: int, compact_chars: int) -> int:
    return max(0, estimate_tokens(original_chars) - estimate_tokens(compact_chars))


def offload_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_TOOL_OFFLOAD", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _min_tokens() -> int:
    return max(0, _env_int("PUPPETMASTER_OFFLOAD_MIN_TOKENS", MIN_TOOL_RESULT_TOKENS))


def _margin() -> float:
    value = _env_float("PUPPETMASTER_OFFLOAD_MARGIN", SAVINGS_MARGIN)
    if value <= 0:
        return SAVINGS_MARGIN
    return value


def _head_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_OFFLOAD_HEAD_CHARS", DEFAULT_HEAD_CHARS))


def _tail_chars() -> int:
    return max(0, _env_int("PUPPETMASTER_OFFLOAD_TAIL_CHARS", DEFAULT_TAIL_CHARS))


def _hard_cap() -> int:
    return max(0, _env_int("PUPPETMASTER_OFFLOAD_HARD_CAP_CHARS", HARD_CAP_CHARS))


def _safe_chars(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except Exception:
        return 0


def gate_decision(original_chars: int, replacement_chars: int) -> Dict[str, Any]:
    """Return offload decision with reason and estimated tokens saved."""
    original = _safe_chars(original_chars)
    replacement = _safe_chars(replacement_chars)
    min_tokens = _min_tokens()
    margin = _margin()

    original_tokens = estimate_tokens(original)
    if original_tokens < min_tokens:
        return {
            "offload": False,
            "reason": "below floor ({0} tokens)".format(min_tokens),
            "estimated_tokens_saved": 0,
        }

    if replacement > int(original * margin):
        return {
            "offload": False,
            "reason": "replacement above margin ({0})".format(margin),
            "estimated_tokens_saved": 0,
        }

    saved = tokens_avoided(original, replacement)
    return {
        "offload": True,
        "reason": "passed gate",
        "estimated_tokens_saved": saved,
    }


def should_offload(original_chars: int, replacement_chars: int) -> bool:
    """True when offload would provably save tokens under the configured policy."""
    return bool(gate_decision(original_chars, replacement_chars)["offload"])


def head_tail_preview(
    text: str,
    head: int = DEFAULT_HEAD_CHARS,
    tail: int = DEFAULT_TAIL_CHARS,
) -> str:
    """Keep the first ``head`` and last ``tail`` chars with an omission marker."""
    if text is None:
        return ""
    body = str(text)
    head_n = max(0, int(head))
    tail_n = max(0, int(tail))
    if len(body) <= head_n + tail_n:
        return body
    omitted = len(body) - head_n - tail_n
    return (
        body[:head_n]
        + "\n... [omitted {0:,} characters] ...\n".format(omitted)
        + body[-tail_n:]
    )


def soft_truncate(text: str, limit: int) -> str:
    """Last-resort head truncate with a trailing note (never raises)."""
    body = "" if text is None else str(text)
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = HARD_CAP_CHARS
    if len(body) <= cap:
        return body
    return body[:cap] + "\n... (truncated, {0} more chars)".format(len(body) - cap)


def _sanitize_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip())[:80]
    return cleaned.strip("-._") or fallback


def _build_preview_message(
    preview: str,
    *,
    original_chars: int,
    file_path: str,
    tool_name: str = "",
) -> str:
    size_kb = original_chars / 1024.0
    if size_kb >= 1024:
        size_str = "{0:.1f} MB".format(size_kb / 1024.0)
    else:
        size_str = "{0:.1f} KB".format(size_kb)
    tool_bit = " from {0}".format(tool_name) if tool_name else ""
    lines = [
        "[tool output offloaded]",
        "This tool result{0} was too large ({1:,} characters, {2}).".format(
            tool_bit, original_chars, size_str
        ),
        "Full output saved to: {0}".format(file_path),
        "Use read_file with start_line and limit to read specific sections.",
        "",
        "Preview (head and tail):",
        preview,
    ]
    return "\n".join(lines)


def _maybe_hard_cap(text: str, meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    cap = _hard_cap()
    if cap <= 0 or len(text) <= cap:
        meta.setdefault("offloaded", False)
        meta.setdefault("tokens_saved", 0)
        return text, meta
    truncated = soft_truncate(text, cap)
    meta = dict(meta)
    meta.update(
        {
            "offloaded": False,
            "hard_capped": True,
            "original_chars": len(text),
            "compact_chars": len(truncated),
            "tokens_saved": tokens_avoided(len(text), len(truncated)),
            "reason": "hard cap ({0} chars)".format(cap),
        }
    )
    return truncated, meta


def record_offload_savings(
    state_dir: PathLike,
    *,
    original_chars: int,
    compact_chars: int,
    tokens_saved: int,
    job_id: str = "",
    task_id: str = "",
    tool_name: str = "",
    tool_call_id: str = "",
    path: str = "",
    reason: str = "offload",
) -> None:
    """Append one numbers-only savings record. Never raises."""
    saved = int(tokens_saved) if tokens_saved else tokens_avoided(original_chars, compact_chars)
    if saved <= 0:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": ENTRY_KIND,
        "job_id": job_id or "",
        "task_id": task_id or "",
        "tool_name": tool_name or "",
        "tool_call_id": tool_call_id or "",
        "original_chars": int(original_chars),
        "compact_chars": int(compact_chars),
        "tokens_saved": saved,
        "reason": reason or "offload",
        "path": path or "",
    }
    try:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / SAVINGS_JSONL
        with target.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass


def load_offload_savings(
    state_dirs: Optional[list] = None,
    *,
    since: Optional[datetime] = None,
) -> list:
    """Load savings records from one or more state dirs. Never raises."""
    dirs = state_dirs or []
    out: list = []
    seen_paths: set = set()
    for raw in dirs:
        try:
            path = Path(raw) / SAVINGS_JSONL
            key = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            continue
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if since is not None:
                        ts = _parse_ts(rec.get("ts"))
                        if ts is not None and ts < since:
                            continue
                    out.append(rec)
        except OSError:
            continue
    return out


def _parse_ts(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def aggregate_offload_savings(records: list) -> Dict[str, Any]:
    """Aggregate measured tool-output offload savings (tokens only, no USD)."""
    tokens = 0
    chars = 0
    count = 0
    for rec in records:
        kind = rec.get("kind")
        if kind is not None and kind not in (ENTRY_KIND, "offload", "compact"):
            continue
        saved = int(rec.get("tokens_saved") or 0)
        if saved <= 0:
            orig = int(rec.get("original_chars") or 0)
            compact = int(rec.get("compact_chars") or 0)
            saved = tokens_avoided(orig, compact)
        if saved <= 0:
            continue
        tokens += saved
        chars += max(
            0,
            int(rec.get("original_chars") or 0) - int(rec.get("compact_chars") or 0),
        )
        count += 1
    return {
        "offloads": count,
        "tokens_saved": tokens,
        "chars_saved": chars,
    }


def offload_tool_output(
    text: str,
    *,
    state_dir: Optional[PathLike] = None,
    job_id: str = "",
    task_id: str = "",
    tool_name: str = "",
    tool_call_id: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(model_facing_text, meta)``. Never raises.

    When below threshold, kill-switched, missing state_dir, or write fails,
    returns the original (or hard-capped) text with ``offloaded=False``.
    """
    body = "" if text is None else str(text)
    meta: Dict[str, Any] = {
        "offloaded": False,
        "tokens_saved": 0,
        "original_chars": len(body),
        "compact_chars": len(body),
        "reason": "",
        "path": "",
    }

    try:
        if not offload_enabled():
            meta["reason"] = "kill switch"
            return _maybe_hard_cap(body, meta)

        if not state_dir:
            meta["reason"] = "no state_dir"
            return _maybe_hard_cap(body, meta)

        head_n = _head_chars()
        tail_n = _tail_chars()
        preview = head_tail_preview(body, head=head_n, tail=tail_n)

        # Estimate replacement size with a placeholder path so the gate sees
        # approximately what the model will receive after a successful write.
        blob_id = _sanitize_id(tool_call_id, "") or _sanitize_id(
            "{0}-{1}".format(tool_name or "tool", uuid.uuid4().hex[:12]),
            "offload",
        )
        root = Path(state_dir)
        estimated_path = (root / OFFLOAD_SUBDIR / "{0}.txt".format(blob_id)).as_posix()
        estimated = _build_preview_message(
            preview,
            original_chars=len(body),
            file_path=estimated_path,
            tool_name=tool_name,
        )
        decision = gate_decision(len(body), len(estimated))
        meta["reason"] = str(decision.get("reason") or "")
        if not decision.get("offload"):
            return _maybe_hard_cap(body, meta)

        try:
            offload_dir = root / OFFLOAD_SUBDIR
            offload_dir.mkdir(parents=True, exist_ok=True)
            # Avoid clobbering an existing blob for a reused tool_call_id.
            target = offload_dir / "{0}.txt".format(blob_id)
            if target.exists():
                blob_id = "{0}-{1}".format(blob_id, uuid.uuid4().hex[:8])
                target = offload_dir / "{0}.txt".format(blob_id)
            target.write_text(body, encoding="utf-8")
            file_path = str(target.resolve())
        except OSError as exc:
            meta["reason"] = "write failed: {0}".format(type(exc).__name__)
            return _maybe_hard_cap(body, meta)

        model_text = _build_preview_message(
            preview,
            original_chars=len(body),
            file_path=file_path,
            tool_name=tool_name,
        )
        saved = tokens_avoided(len(body), len(model_text))
        meta.update(
            {
                "offloaded": True,
                "tokens_saved": saved,
                "original_chars": len(body),
                "compact_chars": len(model_text),
                "path": file_path,
                "reason": "passed gate",
                "ts": time.time(),
            }
        )
        record_offload_savings(
            root,
            original_chars=len(body),
            compact_chars=len(model_text),
            tokens_saved=saved,
            job_id=job_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id or blob_id,
            path=file_path,
            reason="tool-output offload",
        )
        return model_text, meta
    except Exception as exc:
        meta["reason"] = "error: {0}".format(type(exc).__name__)
        try:
            return _maybe_hard_cap(body, meta)
        except Exception:
            return soft_truncate(body, HARD_CAP_CHARS), meta
