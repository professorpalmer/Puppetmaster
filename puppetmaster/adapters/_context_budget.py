"""Live context compression for long agentic runs.

A long implement run accumulates big tool-result messages (file reads, search
dumps, terminal output). Left unchecked the conversation eventually exceeds the
model's context window and the provider returns a 413 / context-length error
mid-run -- turning a healthy worker into a hard failure. This mirrors Hermes'
``context_compressor`` at a fraction of the size: when the estimated prompt is
nearing the budget, the *oldest* large tool outputs are elided (their content
replaced with a short marker) while the message structure -- and every
assistant/tool ``tool_call_id`` pairing -- is preserved, so the provider never
sees an orphaned tool result.

Estimation is the standard ~4-chars-per-token heuristic; it does not need to be
exact, only monotonic, since it just decides *when* to shed old context.
"""
from __future__ import annotations

_ELIDED_MARKER = "[older tool output elided to fit the context window]"


def estimate_message_tokens(messages: list) -> int:
    """A cheap, monotonic token estimate for a message list (~4 chars/token,
    plus per-message and tool-call-argument overhead)."""
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            chars += len(str(function.get("arguments") or ""))
    return chars // 4 + 4 * len(messages)


def compress_history(
    messages: list,
    *,
    budget_tokens: int,
    keep_recent: int = 6,
    min_elide_chars: int = 400,
) -> "tuple[list, bool]":
    """Elide the oldest large tool outputs until the estimate fits ``budget_tokens``.

    Preserves the system prompt (index 0) and the most recent ``keep_recent``
    messages verbatim, and only rewrites ``role == "tool"`` message *content* --
    never assistant turns or tool_call ids -- so the wire stays structurally
    valid. Returns ``(messages, changed)``; ``changed`` is True when at least one
    message was elided. Mutates ``messages`` in place (and returns it).
    """
    if budget_tokens <= 0 or estimate_message_tokens(messages) <= budget_tokens:
        return messages, False

    changed = False
    cutoff = max(1, len(messages) - keep_recent)
    for index in range(1, cutoff):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if len(content) > min_elide_chars and content != _ELIDED_MARKER:
            messages[index] = {**message, "content": _ELIDED_MARKER}
            changed = True
            if estimate_message_tokens(messages) <= budget_tokens:
                break
    return messages, changed
