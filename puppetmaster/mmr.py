"""Maximal marginal relevance (MMR) reranking for memory retrieval diversity.

Ports the OMP algorithm from ``packages/mnemopi/src/core/mmr.ts``: Jaccard
word-set similarity plus a greedy selection loop scoring
``lambda * relevance - (1 - lambda) * max_similarity_to_selected``. Pure,
deterministic, never raises.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

_DEFAULT_LAMBDA = 0.7


def memory_mmr_enabled() -> bool:
    return os.environ.get("PUPPETMASTER_MEMORY_MMR", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def memory_mmr_lambda() -> float:
    raw = os.environ.get("PUPPETMASTER_MEMORY_MMR_LAMBDA")
    if raw is None:
        return _DEFAULT_LAMBDA
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_LAMBDA
    return max(0.0, min(1.0, value))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = {word for word in text_a.lower().split() if word}
    words_b = {word for word in text_b.lower().split() if word}
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    if union == 0:
        return 0.0
    return intersection / union


def _default_similarity(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        left_text = str(left.get("statement") or "")
    else:
        left_text = str(left)
    if isinstance(right, dict):
        right_text = str(right.get("statement") or "")
    else:
        right_text = str(right)
    return jaccard_similarity(left_text, right_text)


def mmr_rerank(
    scored: list[tuple[Any, float]],
    lambda_param: float = 0.7,
    top_k: int = 10,
    similarity_fn: Optional[Callable[[Any, Any], float]] = None,
) -> list[Any]:
    """Rerank ``(record, score)`` pairs for diversity; ties keep original order."""
    limit = max(0, int(top_k))
    if limit <= 0:
        return []
    if not scored:
        return []
    if len(scored) <= 1:
        return [scored[0][0]]

    sim = similarity_fn or _default_similarity
    ordered = list(scored)
    ordered.sort(key=lambda item: item[1], reverse=True)

    selected: list[Any] = [ordered[0][0]]
    remaining = ordered[1:]

    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = float("-inf")
        for idx, (candidate, relevance) in enumerate(remaining):
            max_similarity = 0.0
            for picked in selected:
                similarity = sim(candidate, picked)
                if similarity > max_similarity:
                    max_similarity = similarity
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(remaining.pop(best_idx)[0])

    if len(selected) < limit:
        for candidate, _relevance in remaining:
            selected.append(candidate)
            if len(selected) >= limit:
                break

    return selected


def finalize_memory_retrieval(
    scored: list[tuple[float, float, str, dict[str, Any]]],
    terms: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Apply Wave 10 ordering, optional MMR diversity, and the empty-query guard."""
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    eligible = [
        (memory, score)
        for score, _confidence, _created_at_key, memory in scored
        if score > 0 or not terms
    ]
    if not eligible:
        return []
    if not memory_mmr_enabled():
        return [memory for memory, _score in eligible[:limit]]

    pool_size = min(len(eligible), max(limit, 3 * limit))
    pool = eligible[:pool_size]

    def _statement_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        return jaccard_similarity(
            str(left.get("statement") or ""),
            str(right.get("statement") or ""),
        )

    return mmr_rerank(
        pool,
        lambda_param=memory_mmr_lambda(),
        top_k=limit,
        similarity_fn=_statement_similarity,
    )
