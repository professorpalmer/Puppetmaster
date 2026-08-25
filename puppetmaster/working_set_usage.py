"""Local, numbers-only log of portable working-set warm-skips.

A skip means an analysis worker did not run because a fingerprint that
binds scoped source bytes *and* the task instruction already had labeled
artifacts. That is durable-state reuse, not provider KV-cache portability.

Privacy: counts only. No instruction text, claims, or file paths.

Opt out with ``PUPPETMASTER_WORKING_SET_USAGE=0``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from puppetmaster.fs_permissions import append_private_text

# ESTIMATE: tokens one skipped analysis worker would have burned (prompt +
# output). Conservative and labeled as an estimate, never a routing dollar.
DEFAULT_SKIP_BASELINE_TOKENS = 4000


def usage_log_path() -> Path:
    override = os.environ.get("PUPPETMASTER_WORKING_SET_USAGE_LOG")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("PUPPETMASTER_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "working_set_usage.jsonl"


def usage_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_WORKING_SET_USAGE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def record_reuse(*, artifact_count: int, caller: str = "runtime") -> None:
    """Append one numbers-only skip record. Never raises; never blocks."""
    if not usage_enabled():
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "reuse_skip",
        "caller": caller,
        "artifacts": max(0, int(artifact_count)),
    }
    try:
        path = usage_log_path()
        append_private_text(path, json.dumps(rec) + "\n")
    except OSError:
        pass


def load_usage(since: Optional[datetime] = None) -> list:
    path = usage_log_path()
    if not path.is_file():
        return []
    out = []
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


def _parse_ts(value):
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def aggregate(
    records: list,
    *,
    skip_baseline_tokens: Optional[int] = None,
) -> dict:
    """Roll skip records into measured counts + a labeled token estimate."""
    if skip_baseline_tokens is None:
        skip_baseline_tokens = int(
            os.environ.get(
                "PUPPETMASTER_WORKING_SET_SKIP_TOKENS",
                DEFAULT_SKIP_BASELINE_TOKENS,
            )
        )
    skips = len(records)
    artifacts_reused = 0
    for rec in records:
        try:
            artifacts_reused += int(rec.get("artifacts") or 0)
        except (TypeError, ValueError):
            continue
    avoided_tokens_est = skips * skip_baseline_tokens
    return {
        "skips": skips,
        "artifacts_reused": artifacts_reused,
        "avoided_tokens_est": avoided_tokens_est,
        "skip_baseline_tokens": skip_baseline_tokens,
    }
