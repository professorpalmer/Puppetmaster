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

Before stubbing, compaction promotes a single mutable working-facts note
(offload pointers, terminal exit stubs, bounded findings excerpt) immediately
after the static system prefix so later turns retain high-value continuity.

The static system/prefix block (everything assembled before the per-task
instruction seam) is never mutated -- that prefix must stay byte-stable for
provider prompt caches (v1.13.0 invariant).

Estimation is the standard ~4-chars-per-token heuristic; it does not need to be
exact, only monotonic, since it just decides *when* to shed old context.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from puppetmaster.redaction import redact_secrets
from puppetmaster.tool_offload import is_offload_stub

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

# Mutable pin placed after the static system prefix when compaction fires.
# Exactly one of these may exist; later compact passes update it in place.
_PROMOTION_MARKER = "[puppetmaster working facts]"
_MAX_PROMOTION_CHARS = 1600
_MAX_OFFLOAD_POINTERS = 6
_MAX_TERMINAL_FACTS = 8
_MAX_FINDINGS_EXCERPT_CHARS = 400

_EXIT_CODE_RE = re.compile(r"^exit=(-?\d+)", re.MULTILINE)
_COMPACTED_EXIT_RE = re.compile(
    r"^\[compacted tool result:\s*(?:(?P<name>[^\]]+?)\s+)?"
    r"exit=(?P<code>-?\d+)\]\s*$"
)
_OFFLOAD_PATH_RE = re.compile(r"(?m)^Full output saved to:\s*(.+?)\s*$")
_TOOL_OFFLOAD_DIR = "tool_offload"


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


def is_promotion_note(message: Optional[Dict[str, Any]]) -> bool:
    """True when ``message`` is the mutable system-owned working-facts pin.

    Only ``role=system`` notes are mutable. A ``role=user`` message that
    happens to begin with the marker text must never be identified or
    overwritten — collision with user task text is a hard correctness bug.
    """
    if not isinstance(message, dict):
        return False
    if (message.get("role") or "") != "system":
        return False
    content = message.get("content")
    return isinstance(content, str) and content.startswith(_PROMOTION_MARKER)


def static_prefix_end(messages: list) -> int:
    """Index of the first mutable (non-prefix) message.

    Leading ``role=system`` messages are the static/cache-stable prefix from
    ``split_prompt_messages`` and must never be rewritten. A working-facts
    promotion note is mutable even when ``role=system``, so it ends the
    static block. Returns 0 when there is no system prefix (legacy
    single-user assembly).
    """
    index = 0
    while index < len(messages):
        message = messages[index]
        if (message.get("role") or "") != "system":
            break
        if is_promotion_note(message):
            break
        index += 1
    return index


def _already_compacted(content: str) -> bool:
    if not content:
        return False
    if content == _ELIDED_MARKER or content == _COMPACTED_STUB_OK:
        return True
    if content.startswith(_COMPACTED_PREFIX):
        return True
    # Offload stubs already carry a durable path pointer + head/tail preview.
    # Rewriting them to a one-line compacted stub would destroy retrievability.
    return is_offload_stub(content)


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


def _confined_offload_pointer(raw_path: str) -> Optional[str]:
    """Return a normalized offload pointer, or None when not confined."""
    path = (raw_path or "").strip().strip("\"'")
    if not path or ".." in path.replace("\\", "/").split("/"):
        return None
    normalized = path.replace("\\", "/")
    if _TOOL_OFFLOAD_DIR not in normalized.split("/"):
        return None
    return path


def _parse_tool_arguments(raw: object) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _finding_excerpt_lines(args: Dict[str, Any]) -> List[str]:
    """Pull short claim/summary lines from a submit_findings payload."""
    items = args.get("artifacts")
    if items is None:
        if any(key in args for key in ("claim", "risk", "decision", "finding", "summary")):
            items = [args]
        else:
            items = []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    lines: List[str] = []
    budget = _MAX_FINDINGS_EXCERPT_CHARS
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "finding").strip() or "finding"
        text = (
            item.get("claim")
            or item.get("summary")
            or item.get("finding")
            or item.get("risk")
            or item.get("decision")
            or ""
        )
        text = " ".join(str(text).split())
        if not text:
            continue
        line = f"- {kind}: {text}"
        if len(line) > budget:
            line = line[: max(0, budget - 1)].rstrip() + "…"
        if not line or budget <= 0:
            break
        lines.append(line)
        budget -= len(line) + 1
        if budget <= 0:
            break
    return lines


def _collect_working_facts(messages: list) -> Dict[str, List[str]]:
    """Gather high-value continuity facts from the live transcript."""
    offload_pointers: List[str] = []
    seen_pointers = set()
    terminal_facts: List[str] = []
    seen_terminals = set()
    findings_lines: List[str] = []

    for index, message in enumerate(messages):
        if is_promotion_note(message):
            continue
        role = message.get("role") or ""
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name") or ""
                if name != "submit_findings":
                    continue
                findings_lines.extend(
                    _finding_excerpt_lines(_parse_tool_arguments(function.get("arguments")))
                )
            continue
        if role != "tool":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str) or not content:
            continue

        if is_offload_stub(content):
            for match in _OFFLOAD_PATH_RE.finditer(content):
                pointer = _confined_offload_pointer(match.group(1))
                if pointer is None or pointer in seen_pointers:
                    continue
                seen_pointers.add(pointer)
                offload_pointers.append(pointer)
                if len(offload_pointers) >= _MAX_OFFLOAD_POINTERS:
                    break

        if len(terminal_facts) < _MAX_TERMINAL_FACTS:
            tool_name = _tool_name_for_call(
                messages, message.get("tool_call_id"), index
            )
            compacted = _COMPACTED_EXIT_RE.match(content.strip())
            if compacted:
                code = compacted.group("code")
                name = (compacted.group("name") or tool_name or "").strip()
                if name and name.startswith("exit="):
                    name = ""
                fact = f"{name} exit={code}".strip() if name else f"exit={code}"
            else:
                exit_match = _EXIT_CODE_RE.search(content)
                if exit_match:
                    # Prefer terminal-shaped results; skip huge non-terminal
                    # dumps that merely mention exit= somewhere unrelated.
                    if tool_name not in (
                        None,
                        "run_terminal",
                        "run_command",
                        "shell",
                    ) and not content.lstrip().startswith("exit="):
                        fact = ""
                    else:
                        code = exit_match.group(1)
                        fact = (
                            f"{tool_name} exit={code}"
                            if tool_name
                            else f"exit={code}"
                        )
                else:
                    fact = ""
            if fact and fact not in seen_terminals:
                seen_terminals.add(fact)
                terminal_facts.append(fact)
    if len(findings_lines) > 1:
        # Re-bound concatenated findings to the excerpt budget.
        joined: List[str] = []
        used = 0
        for line in findings_lines:
            cost = len(line) + (1 if joined else 0)
            if used + cost > _MAX_FINDINGS_EXCERPT_CHARS:
                break
            joined.append(line)
            used += cost
        findings_lines = joined

    return {
        "offload_pointers": offload_pointers[:_MAX_OFFLOAD_POINTERS],
        "terminal_facts": terminal_facts[:_MAX_TERMINAL_FACTS],
        "findings_lines": findings_lines,
    }


def _render_promotion_note(facts: Dict[str, List[str]]) -> Optional[str]:
    """Render a bounded, redacted working-facts note, or None when empty."""
    sections: List[str] = []
    pointers = facts.get("offload_pointers") or []
    terminals = facts.get("terminal_facts") or []
    findings = facts.get("findings_lines") or []
    if pointers:
        sections.append(
            "Offload pointers:\n" + "\n".join(f"- {path}" for path in pointers)
        )
    if terminals:
        sections.append(
            "Terminal results:\n" + "\n".join(f"- {fact}" for fact in terminals)
        )
    if findings:
        sections.append("Submitted findings (excerpt):\n" + "\n".join(findings))
    if not sections:
        return None
    body = _PROMOTION_MARKER + "\n" + "\n".join(sections)
    redacted = redact_secrets(body) or body
    if len(redacted) > _MAX_PROMOTION_CHARS:
        redacted = redacted[: _MAX_PROMOTION_CHARS - 1].rstrip() + "…"
    return redacted


def _find_promotion_note_index(messages: list, prefix_end: int) -> Optional[int]:
    """Locate the single mutable promotion note after the static prefix."""
    if prefix_end < len(messages) and is_promotion_note(messages[prefix_end]):
        return prefix_end
    for index in range(prefix_end, len(messages)):
        if is_promotion_note(messages[index]):
            return index
    return None


def _promote_working_facts(messages: list, prefix_end: int) -> bool:
    """Append or update exactly one working-facts note after the static prefix.

    Returns True when the message list changed. Never mutates
    ``messages[:prefix_end]``. Only rewrites an existing *system-owned*
    promotion note; user messages are never identified by marker text.
    """
    facts = _collect_working_facts(messages)
    note = _render_promotion_note(facts)
    if note is None:
        return False

    existing = _find_promotion_note_index(messages, prefix_end)
    if existing is not None:
        current = messages[existing]
        # Defensive: never treat a non-system message as the mutable pin.
        if (current.get("role") or "") != "system" or not is_promotion_note(current):
            existing = None
        else:
            if current.get("content") == note:
                return False
            messages[existing] = {
                **current,
                "role": "system",
                "content": note,
            }
            return True

    messages.insert(
        prefix_end,
        {"role": "system", "content": note},
    )
    return True


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

    When triggered, first appends/updates a single mutable working-facts note
    after the static system prefix (offload pointers, terminal exit stubs,
    bounded submit_findings excerpt), then stubs older ``role=tool`` bodies.
    Preserves the static system prefix and the most recent ``keep_recent``
    messages verbatim, and only rewrites tool message *content* -- never
    assistant turns or tool_call ids -- so the wire stays structurally valid.
    Disabled entirely when ``PUPPETMASTER_HISTORY_COMPACT=0``.

    Returns ``(messages, changed)``; ``changed`` is True when the promotion
    note or at least one tool stub changed. Mutates ``messages`` in place
    (and returns it).
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
    if _promote_working_facts(messages, prefix_end):
        changed = True
    # Recompute after a possible insert; static bytes stay at [:prefix_end].
    mutable_start = static_prefix_end(messages)
    if mutable_start < len(messages) and is_promotion_note(messages[mutable_start]):
        mutable_start += 1
    # Never touch the prefix or promotion note; keep the trailing window verbatim.
    cutoff = max(mutable_start, len(messages) - max(0, keep_recent))
    for index in range(mutable_start, cutoff):
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
