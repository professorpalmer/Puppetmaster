"""Local, numbers-only counter of **$0 follow-up reads** — the third value
lever: re-reading a job's durable results (artifacts, stitched summary, live
feed) costs zero model tokens, because the work was already done and persisted.
Each such read is a swarm re-run that *didn't* happen.

Scope is deliberately narrow to keep the number honest:

* **User-facing result reads only.** We record at the CLI command / MCP tool
  entry points a human (or their agent) actually invokes — ``show``,
  ``artifacts``, ``partial_summary``, ``feed``/``live_artifacts``. We do **not**
  instrument the store's internal read methods (the orchestrator reads
  artifacts constantly; counting those would inflate the number into garbage).
* **Once per invocation.** A long-poll ``feed --follow`` records a single read,
  not one per poll iteration.
* **Numbers only, local only.** Records the read *kind* and caller, never job
  content. Nothing is emitted over the network.

Opt out with ``PUPPETMASTER_READS_USAGE=0``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The reads that mean "re-reading results I already paid to produce". Operational
# reads (status/logs/last/cost/audit/savings) are intentionally excluded so the
# headline stays "follow-up *result* reads served at $0".
RESULT_READ_KINDS = frozenset({"show", "artifacts", "partial_summary", "feed"})


def reads_log_path() -> Path:
    override = os.environ.get("PUPPETMASTER_READS_LOG")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("PUPPETMASTER_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "reads.jsonl"


def reads_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_READS_USAGE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def record_read(kind: str, *, caller: str = "cli") -> None:
    """Append one numbers-only read record. Never raises; never blocks."""
    if not reads_enabled():
        return
    if kind not in RESULT_READ_KINDS:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "caller": caller,
    }
    try:
        path = reads_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # measurement must never break a real read


def load_reads(since: Optional[datetime] = None) -> list[dict]:
    path = reads_log_path()
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
    by_kind: dict[str, int] = {}
    for r in records:
        k = r.get("kind", "?")
        by_kind[k] = by_kind.get(k, 0) + 1
    return {"reads": len(records), "by_kind": by_kind}
