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
from puppetmaster.validation import validation_status_of

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
    def first_count(*keys: str) -> Optional[int]:
        for key in keys:
            value = _coerce_int(sdk_usage.get(key))
            if value is not None:
                return value
        return None

    tokens_in = first_count("inputTokens", "input_tokens", "promptTokens", "prompt_tokens")
    tokens_out = first_count("outputTokens", "output_tokens", "completionTokens", "completion_tokens")
    if tokens_in is None and tokens_out is None:
        return None
    result = {"tokens_in": tokens_in or 0, "tokens_out": tokens_out or 0}
    # Cursor's turn-ended usage and Anthropic's (Claude Code) usage split out
    # cache read/write tokens. They're priced differently from fresh input, so
    # preserve them for the cost axis instead of folding them into tokens_in
    # (which would lie about pricing).
    cache_read = first_count("cacheReadTokens", "cache_read_tokens", "cache_read_input_tokens")
    cache_write = first_count("cacheWriteTokens", "cache_write_tokens", "cache_creation_input_tokens")
    if cache_read is not None:
        result["cache_read_tokens"] = cache_read
    if cache_write is not None:
        result["cache_write_tokens"] = cache_write
    return result


def selected_token_usage(usage, previous=None):
    """Bounded presence facts; a missing turn cannot become a measured zero."""
    aliases = {
        'tokens_in': ('inputTokens', 'input_tokens', 'promptTokens', 'prompt_tokens'),
        'tokens_out': ('outputTokens', 'output_tokens', 'completionTokens', 'completion_tokens'),
        'cache_read_tokens': ('cacheReadTokens', 'cache_read_tokens', 'cache_read_input_tokens'),
        'cache_write_tokens': ('cacheWriteTokens', 'cache_write_tokens', 'cache_creation_input_tokens'),
    }
    usage = usage if isinstance(usage, dict) else {}
    estimated = usage.get('tokens_estimated', False)
    result = {'version': 1}
    for field, keys in aliases.items():
        value = next((usage[k] for k in keys if k in usage), None)
        if type(estimated) is not bool or type(value) is not int:
            value = None
        elif not 0 <= value <= 2**53 - 1:
            value = -1  # Bounded invalid count preserves the numeric-limit outcome.
        if previous is not None:
            prior = previous['selected_facts'].get(field)
            if prior is not None and value is not None:
                value = -1 if prior < 0 or value < 0 else prior + value
            else:
                value = None
            if value is not None and value > 2**53 - 1:
                value = -1
        result[field] = value
    return {'selected_facts': result, 'tokens_estimated': estimated is True or
            (previous is not None and previous['tokens_estimated'])}


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
            **selected_token_usage(sdk_usage),
        }
        for cache_key in ("cache_read_tokens", "cache_write_tokens"):
            if cache_key in measured:
                record[cache_key] = measured[cache_key]
        return record
    return {
        "tokens_in": _approx_tokens(prompt_text),
        "tokens_out": _approx_tokens(output_text),
        "tokens_estimated": True,
        "selected_facts": {"version": 1, "tokens_in": _approx_tokens(prompt_text), "tokens_out": _approx_tokens(output_text)},
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
    Artifacts whose canonical validation status is ``stale`` or ``superseded``
    are excluded so a reset generation cannot be priced from withdrawn output.
    Untasked artifacts (no task_id) each contribute once under a unique key.
    """
    records: dict = {}
    scores: dict = {}
    untasked = 0
    for artifact in artifacts:
        status = validation_status_of(artifact)
        if status in {"stale", "superseded"}:
            continue
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
            # Codex and Antigravity name the input-inclusive cache hit
            # cached_input_tokens; unread, every cache hit priced as fresh input.
            "tokens_cached": int(payload.get("tokens_cached") or payload.get("cached_input_tokens") or 0),
            "real_cost_usd": payload.get("real_cost_usd"),
            "tokens_estimated": bool(payload.get("tokens_estimated")),
            "model": payload.get("model"),
            "selected_facts": {key: payload["selected_facts"].get(key) for key in
                               ("tokens_in", "tokens_out", "cache_read_tokens", "cache_write_tokens")}
            if (isinstance(payload.get("selected_facts"), dict)
                and type(payload["selected_facts"].get("version")) is int
                and payload["selected_facts"]["version"] == 1
                and type(payload.get("tokens_estimated")) is bool) else {},
        }
        # An explicit provider cost (including zero) does not inherit the
        # legacy token-normalizer's missing-field ambiguity.
        if "real_cost_usd" in payload:
            records[task_id]["selected_facts"]["real_cost_usd"] = payload["real_cost_usd"]
        # Presence distinguishes SDK split (exclusive input) from legacy
        # tokens_cached (a subset of input). Do not synthesize absent keys.
        for key in ("cache_read_tokens", "cache_write_tokens"):
            if key in payload:
                records[task_id][key] = int(payload.get(key) or 0)
        records[task_id].update(_billable_presence(payload))
    return records


def _billable_presence(payload: dict) -> dict:
    """Which billable counters are unknown or inconsistent.

    The int fields above fold NULL to 0 for volume rollups; pricing must not.
    Input and output are always billable. A cache counter is billable when the
    adapter reported the key (absent means the adapter has no cache split).
    """
    def known(key: str) -> bool:
        value = payload.get(key)
        return type(value) is int and value >= 0

    unknown = [key for key in ("tokens_in", "tokens_out") if not known(key)]
    for key in ("tokens_cached", "cached_input_tokens", "cache_read_tokens", "cache_write_tokens"):
        if key in payload and not known(key):
            unknown.append(key)
    invalid = []
    split = "cache_read_tokens" in payload or "cache_write_tokens" in payload
    cached = payload.get("tokens_cached", payload.get("cached_input_tokens"))
    if not split and known("tokens_in") and type(cached) is int and cached > payload["tokens_in"]:
        # Legacy cached input is a subset of input.
        invalid.append("cached_exceeds_input")
    return {"usage_unknown": unknown, "usage_invalid": invalid}


def aggregate_token_usage(artifacts: Iterable[Artifact]) -> dict[str, Any]:
    """Roll per-run token records (stored on artifact payloads) into a job total.

    Splits measured from estimated so the surfaced number is honest. A task
    contributes once, preferring the successful fallback run over a failed
    first attempt that also stamped tokens. total_tokens is input + output +
    split cache reads + split cache writes across measured and estimated runs.
    Input/output fields retain their original counts; legacy tokens_cached is
    inclusive of input and is never added again.
    """
    measured_in = measured_out = 0
    estimated_in = estimated_out = 0
    measured_runs = estimated_runs = 0
    measured_read = measured_write = estimated_read = estimated_write = 0

    for record in select_usage_records(artifacts).values():
        tin = record["tokens_in"]
        tout = record["tokens_out"]
        if record["tokens_estimated"]:
            estimated_read += record.get("cache_read_tokens", 0)
            estimated_write += record.get("cache_write_tokens", 0)
            estimated_in += tin
            estimated_out += tout
            estimated_runs += 1
        else:
            measured_read += record.get("cache_read_tokens", 0)
            measured_write += record.get("cache_write_tokens", 0)
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
        "measured_cache_read_tokens": measured_read,
        "measured_cache_write_tokens": measured_write,
        "estimated_cache_read_tokens": estimated_read,
        "estimated_cache_write_tokens": estimated_write,
        "total_tokens": (
            measured_in + measured_out + estimated_in + estimated_out
            + measured_read + measured_write + estimated_read + estimated_write
        ),
    }
