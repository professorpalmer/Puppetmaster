"""Token-consumption capture and rollup.

The router's pre-flight ``estimated_cost_usd`` numbers answer "which model is
proportionally cheaper", but they are *routing estimates*, not measured
consumption — reading them as absolute volume undercounts real usage by orders
of magnitude. And the dominant Cursor-SDK runtime is plan-billed, so marginal
cost is $0 and a dollars-only ledger reports nothing at all.

The honest fix is to record *token counts per run* even when the dollar cost is
zero: measured when the SDK hands us a usage object, and a clearly-labeled
char/4 approximation otherwise. ``token_usage`` builds the per-run record;
``aggregate_token_usage`` rolls a job's records into measured-vs-estimated
totals so ``cost`` can stop pretending the only numbers are 19 pre-flight
estimates.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from puppetmaster.models import Artifact

# Rough bytes-per-token for the char/4 fallback. Deliberately conservative and
# labeled as an estimate wherever it's surfaced — never presented as measured.
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return max(0, len(text) // _CHARS_PER_TOKEN)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def usage_from_sdk(sdk_usage: Any) -> Optional[dict[str, int]]:
    """Normalize a Cursor/Claude SDK usage object into in/out token counts.

    Returns ``None`` when no usable token counts are present, so the caller can
    fall back to an approximation.
    """
    if not isinstance(sdk_usage, dict):
        return None
    # Accept the common key spellings across SDKs without inventing data.
    tokens_in = (
        _coerce_int(sdk_usage.get("inputTokens"))
        or _coerce_int(sdk_usage.get("input_tokens"))
        or _coerce_int(sdk_usage.get("promptTokens"))
        or _coerce_int(sdk_usage.get("prompt_tokens"))
    )
    tokens_out = (
        _coerce_int(sdk_usage.get("outputTokens"))
        or _coerce_int(sdk_usage.get("output_tokens"))
        or _coerce_int(sdk_usage.get("completionTokens"))
        or _coerce_int(sdk_usage.get("completion_tokens"))
    )
    if tokens_in is None and tokens_out is None:
        return None
    result = {"tokens_in": tokens_in or 0, "tokens_out": tokens_out or 0}
    # Cursor's turn-ended usage also splits out cache read/write tokens. They're
    # priced differently from fresh input, so preserve them for the cost axis
    # instead of folding them into tokens_in (which would lie about pricing).
    cache_read = _coerce_int(sdk_usage.get("cacheReadTokens")) or _coerce_int(
        sdk_usage.get("cache_read_tokens")
    )
    cache_write = _coerce_int(sdk_usage.get("cacheWriteTokens")) or _coerce_int(
        sdk_usage.get("cache_write_tokens")
    )
    if cache_read is not None:
        result["cache_read_tokens"] = cache_read
    if cache_write is not None:
        result["cache_write_tokens"] = cache_write
    return result


def token_usage(
    *,
    sdk_usage: Any = None,
    prompt_text: Optional[str] = None,
    output_text: Optional[str] = None,
) -> dict[str, Any]:
    """Build a per-run token-usage record.

    Prefers measured SDK counts; otherwise approximates from prompt/output
    length and flags ``tokens_estimated=True`` so nothing is ever mistaken for
    a measured number.
    """
    measured = usage_from_sdk(sdk_usage)
    if measured is not None:
        record = {
            "tokens_in": measured["tokens_in"],
            "tokens_out": measured["tokens_out"],
            "tokens_estimated": False,
        }
        for cache_key in ("cache_read_tokens", "cache_write_tokens"):
            if cache_key in measured:
                record[cache_key] = measured[cache_key]
        return record
    return {
        "tokens_in": _approx_tokens(prompt_text),
        "tokens_out": _approx_tokens(output_text),
        "tokens_estimated": True,
    }


def usage_record_score(payload: dict) -> tuple:
    """Rank usage-bearing artifacts so a failed first attempt loses to the
    successful fallback run for the same task.

    Order: non-failed result, reported real cost, measured (not estimated)
    tokens, then token volume as a tiebreak.
    """
    result = str(payload.get("result") or "").lower()
    failed = result in ("failed", "blocked", "error", "cancelled")
    try:
        real_cost = float(payload.get("real_cost_usd") or 0.0)
    except (TypeError, ValueError):
        real_cost = 0.0
    tin = int(payload.get("tokens_in") or 0)
    tout = int(payload.get("tokens_out") or 0)
    estimated = bool(payload.get("tokens_estimated"))
    return (
        0 if failed else 1,
        1 if real_cost > 0 else 0,
        0 if estimated else 1,
        tin + tout,
    )


def select_usage_records(artifacts: Iterable[Artifact]) -> dict:
    """task_id -> best usage-bearing artifact payload fields.

    When a task retries after a failed first adapter (cursor -> agentic), both
    attempts stamp tokens. Prefer the successful / measured / higher-volume
    record so totals and pricing follow the run that actually did the work.
    Untasked artifacts (no task_id) each contribute once under a unique key.
    """
    records: dict = {}
    scores: dict = {}
    untasked = 0
    for artifact in artifacts:
        payload = getattr(artifact, "payload", None) or {}
        if "tokens_in" not in payload and "tokens_out" not in payload:
            continue
        task_id = getattr(artifact, "task_id", None)
        if not task_id:
            untasked += 1
            task_id = f"__untasked_{untasked}"
        score = usage_record_score(payload)
        prev = scores.get(task_id)
        if prev is not None and score <= prev:
            continue
        scores[task_id] = score
        records[task_id] = {
            "tokens_in": int(payload.get("tokens_in") or 0),
            "tokens_out": int(payload.get("tokens_out") or 0),
            "tokens_cached": int(payload.get("tokens_cached") or 0),
            "real_cost_usd": payload.get("real_cost_usd"),
            "tokens_estimated": bool(payload.get("tokens_estimated")),
            "model": payload.get("model"),
        }
    return records


def aggregate_token_usage(artifacts: Iterable[Artifact]) -> dict[str, Any]:
    """Roll per-run token records (stored on artifact payloads) into a job total.

    Splits measured from estimated so the surfaced number is honest. A task
    contributes once, preferring the successful fallback run over a failed
    first attempt that also stamped tokens.
    """
    measured_in = measured_out = 0
    estimated_in = estimated_out = 0
    measured_runs = estimated_runs = 0

    for record in select_usage_records(artifacts).values():
        tin = record["tokens_in"]
        tout = record["tokens_out"]
        if record["tokens_estimated"]:
            estimated_in += tin
            estimated_out += tout
            estimated_runs += 1
        else:
            measured_in += tin
            measured_out += tout
            measured_runs += 1

    return {
        "measured_runs": measured_runs,
        "measured_tokens_in": measured_in,
        "measured_tokens_out": measured_out,
        "estimated_runs": estimated_runs,
        "estimated_tokens_in": estimated_in,
        "estimated_tokens_out": estimated_out,
        "total_tokens": measured_in + measured_out + estimated_in + estimated_out,
    }
