"""History compaction for long agentic workers.

Covers turn-count + budget triggers, static-prefix invariance, kill switch,
and recent-K preservation. Deterministic -- no LLM-as-judge.
"""
from __future__ import annotations

import hashlib
import os
import unittest
from typing import Any, Dict, List
from unittest import mock

from puppetmaster.adapters._context_budget import (
    DEFAULT_COMPACT_AFTER_TURNS,
    DEFAULT_KEEP_RECENT,
    compress_history,
    estimate_message_tokens,
    history_compact_enabled,
    static_prefix_end,
)
from puppetmaster.adapters._prompts import TASK_INSTRUCTION_HEADER, split_prompt_messages


def _assistant_tool_call(call_id: str, name: str = "read_file") -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _synthetic_transcript(
    *,
    turns: int,
    payload_chars: int = 5000,
    with_system_prefix: bool = True,
) -> List[Dict[str, Any]]:
    """Build a static-first message list with N assistant/tool turn pairs."""
    static = (
        "You are an implement worker.\n\n"
        "Repo census: 12 files.\n"
    )
    task_user = f"{TASK_INSTRUCTION_HEADER}\nImplement history compaction.\n"
    # Mirror agentic assembly: system prefix + user task suffix.
    if with_system_prefix:
        system_prefix, user_suffix = split_prompt_messages(static + task_user)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": user_suffix or task_user},
        ]
    else:
        messages = [{"role": "user", "content": static + task_user}]

    big = "x" * payload_chars
    for i in range(turns):
        call_id = f"c{i}"
        tool_name = "run_terminal" if i % 3 == 0 else "read_file"
        messages.append(_assistant_tool_call(call_id, tool_name))
        if tool_name == "run_terminal":
            body = f"exit=0\n{big}"
        else:
            body = big
        messages.append(_tool_result(call_id, body))
    return messages


class HistoryCompactionEnabledTests(unittest.TestCase):
    def test_kill_switch_disables_compaction(self) -> None:
        messages = _synthetic_transcript(turns=DEFAULT_COMPACT_AFTER_TURNS + 2)
        prefix_before = [dict(m) for m in messages[: static_prefix_end(messages)]]
        with mock.patch.dict(os.environ, {"PUPPETMASTER_HISTORY_COMPACT": "0"}):
            self.assertFalse(history_compact_enabled())
            out, changed = compress_history(
                messages,
                budget_tokens=1,  # would otherwise force budget compaction
                turn_count=DEFAULT_COMPACT_AFTER_TURNS + 2,
                keep_recent=DEFAULT_KEEP_RECENT,
            )
        self.assertFalse(changed)
        # Every tool payload still full-size.
        for message in out:
            if message.get("role") == "tool":
                self.assertNotIn("compacted", message.get("content") or "")
                self.assertGreater(len(message.get("content") or ""), 100)
        self.assertEqual(out[: len(prefix_before)], prefix_before)

    def test_enabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PUPPETMASTER_HISTORY_COMPACT", None)
            self.assertTrue(history_compact_enabled())


class HistoryCompactionBehaviorTests(unittest.TestCase):
    def test_turn_count_stubs_older_tool_results(self) -> None:
        keep = 4
        turns = DEFAULT_COMPACT_AFTER_TURNS + 2
        messages = _synthetic_transcript(turns=turns, payload_chars=8000)
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,  # under budget -- turn trigger only
            keep_recent=keep,
            turn_count=turns,
            compact_after_turns=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        # Recent K messages (keep_recent window) stay full.
        for message in out[-keep:]:
            if message.get("role") == "tool":
                self.assertNotIn("compacted", message.get("content") or "")
                self.assertGreater(len(message.get("content") or ""), 1000)
        # Older tool results are one-line stubs.
        prefix_end = static_prefix_end(out)
        cutoff = len(out) - keep
        stubbed = 0
        for message in out[prefix_end:cutoff]:
            if message.get("role") != "tool":
                continue
            content = message.get("content") or ""
            self.assertTrue(content.startswith("[compacted tool result:"))
            self.assertLess(len(content), 120)
            stubbed += 1
        self.assertGreater(stubbed, 0)

    def test_recent_k_turns_remain_full(self) -> None:
        keep = 4
        messages = _synthetic_transcript(turns=14, payload_chars=6000)
        recent_before = [dict(m) for m in messages[-keep:]]
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=keep,
            turn_count=14,
        )
        self.assertTrue(changed)
        self.assertEqual(out[-keep:], recent_before)

    def test_static_prefix_bytes_unchanged(self) -> None:
        messages = _synthetic_transcript(turns=14, payload_chars=6000)
        prefix_end = static_prefix_end(messages)
        self.assertGreater(prefix_end, 0)
        prefix_bytes = [
            (m.get("role"), m.get("content")) for m in messages[:prefix_end]
        ]
        digest_before = hashlib.sha256(repr(prefix_bytes).encode("utf-8")).hexdigest()

        out, changed = compress_history(
            messages,
            budget_tokens=1,
            keep_recent=DEFAULT_KEEP_RECENT,
            turn_count=14,
        )
        self.assertTrue(changed)
        prefix_after = [
            (m.get("role"), m.get("content")) for m in out[:prefix_end]
        ]
        digest_after = hashlib.sha256(repr(prefix_after).encode("utf-8")).hexdigest()
        self.assertEqual(digest_before, digest_after)
        self.assertEqual(prefix_bytes, prefix_after)
        # System content still sits before the task header seam.
        system = out[0]
        self.assertEqual(system.get("role"), "system")
        self.assertNotIn(TASK_INSTRUCTION_HEADER, system.get("content") or "")

    def test_budget_trigger_without_turn_threshold(self) -> None:
        messages = _synthetic_transcript(turns=6, payload_chars=5000)
        before = estimate_message_tokens(messages)
        out, changed = compress_history(
            messages,
            budget_tokens=max(1, before // 3),
            keep_recent=2,
            turn_count=1,  # below DEFAULT_COMPACT_AFTER_TURNS
            compact_after_turns=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        self.assertLess(estimate_message_tokens(out), before)
        self.assertEqual(out[-1]["content"], "x" * 5000)

    def test_stub_includes_tool_name_and_exit(self) -> None:
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\ndo the thing"},
            _assistant_tool_call("c0", "run_terminal"),
            _tool_result("c0", "exit=0\n" + ("y" * 2000)),
            _assistant_tool_call("c1", "read_file"),
            _tool_result("c1", "z" * 2000),
            _assistant_tool_call("c2", "read_file"),
            _tool_result("c2", "keep-me-recent"),
        ]
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        self.assertIn("run_terminal", out[3]["content"])
        self.assertIn("exit=0", out[3]["content"])
        self.assertIn("read_file", out[5]["content"])
        self.assertEqual(out[-1]["content"], "keep-me-recent")

    def test_noop_under_budget_and_under_turn_threshold(self) -> None:
        messages = _synthetic_transcript(turns=3, payload_chars=200)
        snapshot = [dict(m) for m in messages]
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=DEFAULT_KEEP_RECENT,
            turn_count=2,
            compact_after_turns=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertFalse(changed)
        self.assertEqual(out, snapshot)


if __name__ == "__main__":
    unittest.main()
