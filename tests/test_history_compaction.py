"""History compaction for long agentic workers.

Covers turn-count + budget triggers, static-prefix invariance, kill switch,
and recent-K preservation. Deterministic -- no LLM-as-judge.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import hashlib
import json
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
    is_promotion_note,
    static_prefix_end,
)
from puppetmaster.adapters._prompts import TASK_INSTRUCTION_HEADER, split_prompt_messages
from puppetmaster.tool_offload import OFFLOAD_MARKER

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

def _assistant_submit_findings(call_id: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "submit_findings",
                    "arguments": json.dumps({"artifacts": artifacts}),
                },
            }
        ],
    }

def _tool_result(call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}

def _tool_content(messages: List[Dict[str, Any]], call_id: str) -> str:
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return message.get("content") or ""
    raise AssertionError(f"missing tool result for {call_id}")

def _promotion_notes(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in messages if is_promotion_note(m)]

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
        c0 = _tool_content(out, "c0")
        c1 = _tool_content(out, "c1")
        self.assertIn("run_terminal", c0)
        self.assertIn("exit=0", c0)
        self.assertIn("read_file", c1)
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

    def test_offload_stub_preserved_through_compaction(self) -> None:
        """Offload stubs keep their durable path pointer when history compacts."""
        pointer = "/tmp/state/tool_offload/call-big.txt"
        offload_stub = (
            f"{OFFLOAD_MARKER}\n"
            "This tool result from read_file was too large (64,000 characters, 62.5 KB).\n"
            f"Full output saved to: {pointer}\n"
            "Use read_offload with start_line and limit to read specific sections.\n"
            "\n"
            "Preview (head and tail):\n"
            + ("H" * 200)
            + "\n... [omitted 60,000 characters] ...\n"
            + ("T" * 200)
        )
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\ndo the thing"},
            _assistant_tool_call("c0", "read_file"),
            _tool_result("c0", offload_stub),
            _assistant_tool_call("c1", "run_terminal"),
            _tool_result("c1", "exit=0\n" + ("y" * 3000)),
            _assistant_tool_call("c2", "read_file"),
            _tool_result("c2", "z" * 3000),
            _assistant_tool_call("c3", "read_file"),
            _tool_result("c3", "keep-me-recent"),
        ]
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        # Offload stub must be left intact (pointer + marker), not rewritten.
        c0 = _tool_content(out, "c0")
        self.assertEqual(c0, offload_stub)
        self.assertIn(pointer, c0)
        self.assertTrue(c0.startswith(OFFLOAD_MARKER))
        # Ordinary older tool results still compact.
        self.assertTrue(_tool_content(out, "c1").startswith("[compacted tool result:"))
        self.assertTrue(_tool_content(out, "c2").startswith("[compacted tool result:"))
        self.assertEqual(out[-1]["content"], "keep-me-recent")

class HistoryPromotionTests(unittest.TestCase):
    def test_promotion_preserves_static_prefix_bytes(self) -> None:
        messages = _synthetic_transcript(turns=14, payload_chars=6000)
        prefix_end = static_prefix_end(messages)
        prefix_before = [
            (m.get("role"), m.get("content")) for m in messages[:prefix_end]
        ]
        digest_before = hashlib.sha256(repr(prefix_before).encode("utf-8")).hexdigest()

        out, changed = compress_history(
            messages,
            budget_tokens=1,
            keep_recent=DEFAULT_KEEP_RECENT,
            turn_count=14,
        )
        self.assertTrue(changed)
        # Static prefix index and bytes are unchanged even after note insert.
        self.assertEqual(static_prefix_end(out), prefix_end)
        prefix_after = [(m.get("role"), m.get("content")) for m in out[:prefix_end]]
        self.assertEqual(prefix_before, prefix_after)
        self.assertEqual(
            digest_before,
            hashlib.sha256(repr(prefix_after).encode("utf-8")).hexdigest(),
        )
        notes = _promotion_notes(out)
        self.assertEqual(len(notes), 1)
        self.assertEqual(out[prefix_end], notes[0])

    def test_promotion_note_idempotent_across_repeated_compaction(self) -> None:
        pointer = "/var/pm/tool_offload/blob-a.txt"
        offload_stub = (
            f"{OFFLOAD_MARKER}\n"
            f"Full output saved to: {pointer}\n"
            "Preview (head and tail):\n"
            + ("H" * 100)
        )
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\ndo the thing"},
            _assistant_tool_call("c0", "read_file"),
            _tool_result("c0", offload_stub),
            _assistant_tool_call("c1", "run_terminal"),
            _tool_result("c1", "exit=1\n" + ("y" * 3000)),
            _assistant_tool_call("c2", "read_file"),
            _tool_result("c2", "z" * 3000),
            _assistant_tool_call("c3", "read_file"),
            _tool_result("c3", "keep-me-recent"),
        ]
        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        first_notes = _promotion_notes(out)
        self.assertEqual(len(first_notes), 1)
        first_content = first_notes[0]["content"]

        out2, changed2 = compress_history(
            out,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS + 1,
        )
        notes = _promotion_notes(out2)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["content"], first_content)
        # Second pass may be a no-op once stubs + note are already settled.
        if changed2:
            self.assertEqual(len(_promotion_notes(out2)), 1)

    def test_promotion_keeps_offload_pointer_and_terminal_exit(self) -> None:
        pointer = "C:/state/tool_offload/call-big.txt"
        escape = "C:/etc/passwd"
        offload_stub = (
            f"{OFFLOAD_MARKER}\n"
            f"Full output saved to: {pointer}\n"
            "Preview:\nhead"
        )
        decoy = (
            f"{OFFLOAD_MARKER}\n"
            f"Full output saved to: {escape}\n"
            "Preview:\nbad"
        )
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\ndo the thing"},
            _assistant_tool_call("c0", "read_file"),
            _tool_result("c0", offload_stub),
            _assistant_tool_call("c_bad", "read_file"),
            _tool_result("c_bad", decoy),
            _assistant_tool_call("c1", "run_terminal"),
            _tool_result("c1", "exit=2\n" + ("y" * 2500)),
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
        notes = _promotion_notes(out)
        self.assertEqual(len(notes), 1)
        note = notes[0]["content"]
        self.assertIn(pointer, note)
        self.assertNotIn(escape, note)
        self.assertIn("exit=2", note)
        self.assertIn("run_terminal", note)
        self.assertTrue(note.startswith("[puppetmaster working facts]"))

    def test_promotion_includes_bounded_submit_findings_excerpt(self) -> None:
        long_claim = "important continuity claim " + ("x" * 2000)
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\nanalyze"},
            _assistant_submit_findings(
                "sf0",
                [{"type": "finding", "claim": long_claim, "evidence": ["a.py:1"]}],
            ),
            _tool_result("sf0", "Recorded 1 artifact(s). Analysis complete."),
            _assistant_tool_call("c0", "run_terminal"),
            _tool_result("c0", "exit=0\n" + ("y" * 3000)),
            _assistant_tool_call("c1", "read_file"),
            _tool_result("c1", "z" * 3000),
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
        notes = _promotion_notes(out)
        self.assertEqual(len(notes), 1)
        note = notes[0]["content"]
        self.assertIn("Submitted findings", note)
        self.assertIn("important continuity claim", note)
        self.assertLess(len(note), 1600)
        self.assertNotIn(long_claim, note)

    def test_promotion_note_size_bound(self) -> None:
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": f"{TASK_INSTRUCTION_HEADER}\ndo the thing"},
        ]
        for i in range(10):
            pointer = f"/tmp/state/tool_offload/blob-{i}.txt"
            stub = (
                f"{OFFLOAD_MARKER}\n"
                f"Full output saved to: {pointer}\n"
                "Preview:\n"
                + ("P" * 50)
            )
            messages.append(_assistant_tool_call(f"o{i}", "read_file"))
            messages.append(_tool_result(f"o{i}", stub))
            messages.append(_assistant_tool_call(f"t{i}", "run_terminal"))
            messages.append(_tool_result(f"t{i}", f"exit={i}\n" + ("y" * 2000)))
        messages.append(_assistant_tool_call("recent", "read_file"))
        messages.append(_tool_result("recent", "keep-me-recent"))

        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        notes = _promotion_notes(out)
        self.assertEqual(len(notes), 1)
        self.assertLessEqual(len(notes[0]["content"]), 1600)
        # Cap keeps the note concise even when many facts are available.
        self.assertLessEqual(notes[0]["content"].count("tool_offload"), 6)

    def test_user_marker_collision_does_not_overwrite_user_message(self) -> None:
        """A user message that begins with the promotion marker must stay intact.

        Compaction must insert a fresh system-owned note instead of rewriting
        the colliding user turn in place.
        """
        colliding_user = (
            "[puppetmaster working facts]\n"
            "IMPORTANT USER TASK DO NOT DESTROY — ship the fix"
        )
        pointer = "/var/pm/tool_offload/blob-collision.txt"
        offload_stub = (
            f"{OFFLOAD_MARKER}\n"
            f"Full output saved to: {pointer}\n"
            "Preview:\nhead"
        )
        messages = [
            {"role": "system", "content": "static prefix"},
            {"role": "user", "content": colliding_user},
            _assistant_tool_call("c0", "read_file"),
            _tool_result("c0", offload_stub),
            _assistant_tool_call("c1", "run_terminal"),
            _tool_result("c1", "exit=0\n" + ("y" * 3000)),
            _assistant_tool_call("c2", "read_file"),
            _tool_result("c2", "z" * 3000),
            _assistant_tool_call("c3", "read_file"),
            _tool_result("c3", "keep-me-recent"),
        ]
        self.assertFalse(is_promotion_note(messages[1]))

        out, changed = compress_history(
            messages,
            budget_tokens=10_000_000,
            keep_recent=2,
            turn_count=DEFAULT_COMPACT_AFTER_TURNS,
        )
        self.assertTrue(changed)
        user_messages = [m for m in out if m.get("role") == "user"]
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], colliding_user)
        self.assertIn("IMPORTANT USER TASK DO NOT DESTROY", user_messages[0]["content"])

        notes = _promotion_notes(out)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].get("role"), "system")
        self.assertIn(pointer, notes[0]["content"])
        # User collision is not the mutable pin; system note sits after prefix.
        self.assertEqual(out[static_prefix_end(out)], notes[0])
        self.assertIsNot(out[static_prefix_end(out)], user_messages[0])

if __name__ == "__main__":
    unittest.main()
