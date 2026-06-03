"""Local, append-only log of CodeGraph queries — the measurement substrate for
the *exploration* half of Puppetmaster's value (cheap graph lookups instead of
agents crawling directories token-by-token).

Why this lives here and not in the per-job ``SwarmStore``: the biggest source
of CodeGraph savings isn't swarm jobs — it's *every* agent interaction routed
through the ``puppetmaster_codegraph_*`` MCP tools (e.g. a global "graph-first
search" rule). Those calls never touch a job's store, but they all funnel
through Puppetmaster's codegraph wrapper. So the usage log is a single global
file that captures graph queries from **any** platform's agent (Cursor, Claude
Code, Codex, in-swarm workers) — the shared substrate, not one adapter.

Privacy posture (deliberate): **numbers only.** We record command, result size,
token estimate, and latency — never the query text or any code. That keeps the
log safe to summarize and keeps it consistent with the rule that any future org
rollup ships numbers, not content. Local-only; nothing is emitted anywhere.

Opt out with ``PUPPETMASTER_CODEGRAPH_USAGE=0``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Commands whose results displace an agent reading/grepping files. Only these
# count toward exploration savings; status/init/repair are housekeeping.
EXPLORATION_COMMANDS = frozenset({"context", "search", "affected", "files"})

# Rough chars->tokens heuristic. Deliberately crude and documented as such — the
# point is an order-of-magnitude estimate, not billing-grade accounting.
CHARS_PER_TOKEN = 4

# ESTIMATE knob: how many tokens a cold, graph-less exploration would have
# burned to locate the same context (reading/grepping files until the target is
# found). Conservative default; override with PUPPETMASTER_EXPLORATION_BASELINE_TOKENS.
DEFAULT_EXPLORATION_BASELINE_TOKENS = 8000

# ESTIMATE knob: input price used to turn avoided tokens into dollars. Defaults
# to a mid-tier model input price; override with PUPPETMASTER_EXPLORATION_PRICE_PER_MTOK.
DEFAULT_INPUT_PRICE_PER_MTOK = 1.0


def usage_log_path() -> Path:
    override = os.environ.get("PUPPETMASTER_CODEGRAPH_USAGE_LOG")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("PUPPETMASTER_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".puppetmaster"
    return root / "codegraph_usage.jsonl"


def usage_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_CODEGRAPH_USAGE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def estimate_tokens(chars: int) -> int:
    return max(0, int(chars) // CHARS_PER_TOKEN)


def record_query(
    *,
    command: str,
    cwd: Optional[object],
    result_chars: int,
    latency_ms: float,
    ok: bool,
    caller: str = "mcp",
    query_chars: int = 0,
) -> None:
    """Append one numbers-only usage record. Never raises; never blocks."""
    if not usage_enabled():
        return
    if command not in EXPLORATION_COMMANDS:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "caller": caller,
        "cwd": str(cwd or ""),
        "query_chars": int(query_chars),  # size of the ask, not its content
        "result_chars": int(result_chars),
        "context_tokens": estimate_tokens(result_chars),
        "latency_ms": round(float(latency_ms), 1),
        "ok": bool(ok),
    }
    try:
        path = usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # measurement must never break a real query


def load_usage(since: Optional[datetime] = None) -> list[dict]:
    path = usage_log_path()
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


def aggregate(
    records: list[dict],
    *,
    exploration_baseline_tokens: Optional[int] = None,
    input_price_per_mtok: Optional[float] = None,
) -> dict:
    """Roll usage records into measured counts + clearly-labeled estimates."""
    if exploration_baseline_tokens is None:
        exploration_baseline_tokens = int(
            os.environ.get(
                "PUPPETMASTER_EXPLORATION_BASELINE_TOKENS",
                DEFAULT_EXPLORATION_BASELINE_TOKENS,
            )
        )
    if input_price_per_mtok is None:
        input_price_per_mtok = float(
            os.environ.get(
                "PUPPETMASTER_EXPLORATION_PRICE_PER_MTOK",
                DEFAULT_INPUT_PRICE_PER_MTOK,
            )
        )

    successful = [r for r in records if r.get("ok")]
    queries = len(successful)
    by_command: dict[str, int] = {}
    for r in successful:
        by_command[r.get("command", "?")] = by_command.get(r.get("command", "?"), 0) + 1
    context_tokens_fed = sum(int(r.get("context_tokens") or 0) for r in successful)

    # ESTIMATE: avoided exploration = (what a graph-less crawl would have read)
    # minus (the focused context we actually fed).
    avoided_exploration_tokens = queries * exploration_baseline_tokens
    net_tokens_saved_est = max(0, avoided_exploration_tokens - context_tokens_fed)
    dollars_saved_est = round(net_tokens_saved_est / 1_000_000.0 * input_price_per_mtok, 4)

    return {
        # --- MEASURED ---
        "queries": queries,
        "by_command": by_command,
        "context_tokens_fed": context_tokens_fed,
        # --- ESTIMATE (stated baseline) ---
        "exploration_baseline_tokens": exploration_baseline_tokens,
        "input_price_per_mtok": input_price_per_mtok,
        "avoided_exploration_tokens_est": avoided_exploration_tokens,
        "net_tokens_saved_est": net_tokens_saved_est,
        "dollars_saved_est": dollars_saved_est,
    }
