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


def aggregate_token_usage(artifacts: Iterable[Artifact]) -> dict[str, Any]:
    """Roll per-run token records (stored on artifact payloads) into a job total.

    Splits measured from estimated so the surfaced number is honest. A run
    contributes once: any artifact carrying ``tokens_in``/``tokens_out`` counts.
    """
    measured_in = measured_out = 0
    estimated_in = estimated_out = 0
    measured_runs = estimated_runs = 0
    seen_tasks: set = set()

    for artifact in artifacts:
        payload = getattr(artifact, "payload", None) or {}
        if "tokens_in" not in payload and "tokens_out" not in payload:
            continue
        task_id = getattr(artifact, "task_id", None)
        if task_id and task_id in seen_tasks:
            continue
        if task_id:
            seen_tasks.add(task_id)
        tin = int(payload.get("tokens_in") or 0)
        tout = int(payload.get("tokens_out") or 0)
        if payload.get("tokens_estimated"):
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
