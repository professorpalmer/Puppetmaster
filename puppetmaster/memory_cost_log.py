"""Local, append-only log of promoted-memory injection overhead.

Every dispatch that injects ``retrieved_memory`` into worker specs carries extra
prompt tokens the savings ledger otherwise ignores. One numbers-only record per
injection keeps the receipt honest about memory overhead as spend, not savings.

Opt out with ``PUPPETMASTER_MEMORY_COST_LOG=0``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from puppetmaster.fs_permissions import append_private_text

CHARS_PER_TOKEN = 4
ENTRY_KIND = "memory_injection"


def memory_cost_log_path() -> Path:
    override = os.environ.get("PUPPETMASTER_MEMORY_COST_LOG_PATH")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("PUPPETMASTER_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "memory_cost.jsonl"


def memory_cost_log_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_MEMORY_COST_LOG", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def estimate_tokens(chars: int) -> int:
    return max(0, int(chars) // CHARS_PER_TOKEN)


def estimate_injection_block_chars(memory: list[dict[str, Any]]) -> int:
    """Mirror the rendered injection block size (header + distilled bullets)."""
    if not memory:
        return 0
    from puppetmaster.adapters._prompts import _distill_memory_lines

    distilled = _distill_memory_lines(memory)
    if not distilled:
        return 0
    lines = [
        "",
        "Relevant promoted Puppetmaster memory (distilled facts/decisions):",
        *distilled,
        "",
        "Use this as retrieved context, but verify claims before relying on them.",
    ]
    return len("\n".join(lines))


def estimate_cost_usd(tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    try:
        from puppetmaster.model_registry import load_registry
        from puppetmaster.savings import resolve_counterfactual_model

        reference = resolve_counterfactual_model(load_registry())
        if reference is None:
            return 0.0
        return round(reference.estimate_cost_usd(tokens, 0), 6)
    except Exception:
        return 0.0


def record_memory_injection(
    *,
    job_id: str,
    record_count: int,
    memory: list[dict[str, Any]],
    role: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    """Append one numbers-only injection record. Never raises; never blocks."""
    if not memory_cost_log_enabled() or record_count <= 0 or not memory:
        return
    chars = estimate_injection_block_chars(memory)
    tokens = estimate_tokens(chars)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": ENTRY_KIND,
        "job_id": job_id,
        "task_id": task_id,
        "role": role,
        "record_count": int(record_count),
        "token_count": tokens,
        "estimated_cost_usd": estimate_cost_usd(tokens),
    }
    try:
        append_private_text(memory_cost_log_path(), json.dumps(rec) + "\n")
    except OSError:
        pass


def load_memory_cost(since: Optional[datetime] = None) -> list[dict]:
    path = memory_cost_log_path()
    if not path.is_file():
        return []
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if since is not None:
                    ts = _parse_ts(rec.get("ts"))
                    if ts is not None and ts < since:
                        continue
                out.append(rec)
    except OSError:
        return []
    return out


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def aggregate(records: list[dict]) -> dict:
    injections = [rec for rec in records if rec.get("kind") == ENTRY_KIND]
    tokens = sum(int(rec.get("token_count") or 0) for rec in injections)
    cost = round(sum(float(rec.get("estimated_cost_usd") or 0.0) for rec in injections), 6)
    records_injected = sum(int(rec.get("record_count") or 0) for rec in injections)
    return {
        "injections": len(injections),
        "records_injected": records_injected,
        "token_count": tokens,
        "estimated_cost_usd": cost,
    }
