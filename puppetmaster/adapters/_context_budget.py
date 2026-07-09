"""Live context compression for long agentic runs.

A long implement run accumulates big tool-result messages (file reads, search
dumps, terminal output). Left unchecked the conversation eventually exceeds the
model's context window and the provider returns a 413 / context-length error
mid-run -- turning a healthy worker into a hard failure. This mirrors Hermes'
``context_compressor`` at a fraction of the size: when the estimated prompt is
nearing the budget *or* the turn count crosses a threshold, the *oldest* tool
outputs are collapsed to one-line stubs while the message structure -- and every
assistant/tool ``tool_call_id`` pairing -- is preserved, so the provider never
sees an orphaned tool result.

The static system/prefix block (everything assembled before the per-task
instruction seam) is never mutated -- that prefix must stay byte-stable for
provider prompt caches (v1.13.0 invariant).

Estimation is the standard ~4-chars-per-token heuristic; it does not need to be
exact, only monotonic, since it just decides *when* to shed old context.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

# Kill switch: ``PUPPETMASTER_HISTORY_COMPACT=0`` disables all compaction.
_HISTORY_COMPACT_ENV = "PUPPETMASTER_HISTORY_COMPACT"

# Turn-count trigger: after this many loop turns, older tool results are stubbed
# even if the token estimate is still under budget.
DEFAULT_COMPACT_AFTER_TURNS = 12

# Preserve the most recent K messages verbatim (tool/user/assistant turns).
DEFAULT_KEEP_RECENT = 4

# Budget-path only: skip stubbing tiny tool payloads until they exceed this.
DEFAULT_MIN_ELIDE_CHARS = 400

# One-line stubs for compacted tool results. The legacy elided marker is still
# recognized so already-compressed transcripts are not rewritten again.
_COMPACTED_STUB_OK = "[compacted tool result: ok]"
_ELIDED_MARKER = "[older tool output elided to fit the context window]"
_COMPACTED_PREFIX = "[compacted tool result:"

_EXIT_CODE_RE = re.compile(r"^exit=(-?\d+)", re.MULTILINE)


def history_compact_enabled() -> bool:
    """Return False when the env kill switch disables history compaction."""
    return os.environ.get(_HISTORY_COMPACT_ENV, "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


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


def static_prefix_end(messages: list) -> int:
    """Index of the first mutable (non-prefix) message.

    Leading ``role=system`` messages are the static/cache-stable prefix from
    ``split_prompt_messages`` and must never be rewritten. Returns 0 when there
    is no system prefix (legacy single-user assembly).
    """
    index = 0
    while index < len(messages) and (messages[index].get("role") or "") == "system":
        index += 1
    return index


def _already_compacted(content: str) -> bool:
    if not content:
        return False
    if content == _ELIDED_MARKER or content == _COMPACTED_STUB_OK:
        return True
    return content.startswith(_COMPACTED_PREFIX)


def _tool_name_for_call(
    messages: list,
    tool_call_id: Optional[str],
    before_index: int,
) -> Optional[str]:
    """Resolve the tool name for a tool-result by scanning prior assistant calls."""
    if not tool_call_id:
        return None
    for index in range(before_index - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") != tool_call_id:
                continue
            function = call.get("function") or {}
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
            return None
    return None


def _stub_for_tool_message(
    message: Dict[str, Any],
    tool_name: Optional[str] = None,
) -> str:
    """Build a one-line stub, optionally including tool name and exit code."""
    content = message.get("content") or ""
    exit_match = _EXIT_CODE_RE.search(content) if isinstance(content, str) else None
    exit_code = exit_match.group(1) if exit_match else None
    if tool_name and exit_code is not None:
        return f"[compacted tool result: {tool_name} exit={exit_code}]"
    if tool_name:
        return f"[compacted tool result: {tool_name}]"
    if exit_code is not None:
        return f"[compacted tool result: exit={exit_code}]"
    return _COMPACTED_STUB_OK


def compress_history(
    messages: list,
    *,
    budget_tokens: int,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    min_elide_chars: int = DEFAULT_MIN_ELIDE_CHARS,
    turn_count: int = 0,
    compact_after_turns: int = DEFAULT_COMPACT_AFTER_TURNS,
) -> "Tuple[list, bool]":
    """Collapse older tool outputs to one-line stubs when triggered.

    Triggers (either is enough):

    * estimated tokens exceed ``budget_tokens`` (chars//4 heuristic)
    * ``turn_count >= compact_after_turns`` (default 12)

    Preserves the static system prefix and the most recent ``keep_recent``
    messages verbatim, and only rewrites ``role == "tool"`` message *content* --
    never assistant turns or tool_call ids -- so the wire stays structurally
    valid. Disabled entirely when ``PUPPETMASTER_HISTORY_COMPACT=0``.

    Returns ``(messages, changed)``; ``changed`` is True when at least one
    message was stubbed. Mutates ``messages`` in place (and returns it).
    """
    if not history_compact_enabled():
        return messages, False

    over_budget = (
        budget_tokens > 0 and estimate_message_tokens(messages) > budget_tokens
    )
    over_turns = (
        compact_after_turns > 0 and turn_count >= compact_after_turns
    )
    if not over_budget and not over_turns:
        return messages, False

    changed = False
    prefix_end = static_prefix_end(messages)
    # Never touch the prefix; keep the trailing window verbatim.
    cutoff = max(prefix_end, len(messages) - max(0, keep_recent))
    for index in range(prefix_end, cutoff):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str) or _already_compacted(content):
            continue
        # Budget-only path keeps the size floor; turn-count path stubs everything.
        if not over_turns and len(content) <= min_elide_chars:
            continue
        tool_name = _tool_name_for_call(
            messages, message.get("tool_call_id"), index
        )
        stub = _stub_for_tool_message(message, tool_name)
        messages[index] = {**message, "content": stub}
        changed = True
        if over_budget and not over_turns and estimate_message_tokens(messages) <= budget_tokens:
            break
    return messages, changed
