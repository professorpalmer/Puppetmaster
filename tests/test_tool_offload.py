"""Focused tests for savings-gated tool-output offload."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.tool_offload import (
    HARD_CAP_CHARS,
    estimate_tokens,
    gate_decision,
    head_tail_preview,
    offload_tool_output,
    should_offload,
)


class ToolOffloadTests(unittest.TestCase):
    def test_below_threshold_returns_full_text_no_file(self) -> None:
        # Well under the 3000-token floor (~12k chars).
        text = "x" * 2000
        with TemporaryDirectory() as tmp:
            model_text, meta = offload_tool_output(
                text, state_dir=tmp, tool_name="read_file", tool_call_id="tc1"
            )
            self.assertFalse(meta["offloaded"])
            self.assertEqual(model_text, text)
            self.assertEqual(list(Path(tmp).joinpath("tool_offload").glob("*.txt")), [])

    def test_above_threshold_spills_head_tail_and_records_savings(self) -> None:
        # >3000 tokens and head+tail preview is well under the 0.9 margin.
        text = ("HEAD-" + ("body" * 4000) + "-TAIL") * 2  # ~64k chars
        self.assertGreater(estimate_tokens(len(text)), 3000)
        with TemporaryDirectory() as tmp:
            model_text, meta = offload_tool_output(
                text,
                state_dir=tmp,
                job_id="job1",
                task_id="task1",
                tool_name="run_terminal",
                tool_call_id="call-big",
            )
            self.assertTrue(meta["offloaded"], meta)
            self.assertGreater(meta["tokens_saved"], 0)
            self.assertIn("[tool output offloaded]", model_text)
            self.assertIn("omitted", model_text)
            self.assertNotEqual(model_text, text)
            self.assertLess(len(model_text), len(text) * 0.9)
            blobs = list(Path(tmp).joinpath("tool_offload").glob("*.txt"))
            self.assertEqual(len(blobs), 1)
            self.assertEqual(blobs[0].read_text(encoding="utf-8"), text)
            ledger = Path(tmp) / "tool_output_savings.jsonl"
            self.assertTrue(ledger.is_file())
            self.assertIn("tool_output_offload", ledger.read_text(encoding="utf-8"))

    def test_kill_switch_never_offloads(self) -> None:
        text = "y" * 80_000
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PUPPETMASTER_TOOL_OFFLOAD": "0"}):
                model_text, meta = offload_tool_output(
                    text, state_dir=tmp, tool_name="web_fetch", tool_call_id="tc-kill"
                )
            self.assertFalse(meta["offloaded"])
            self.assertEqual(list(Path(tmp).joinpath("tool_offload").glob("*.txt")), [])
            # Kill switch still allows hard-cap soft truncate when oversized.
            self.assertLessEqual(len(model_text), HARD_CAP_CHARS + 80)
            self.assertIn("truncated", model_text)

    def test_gate_margin_blocks_when_replacement_too_large(self) -> None:
        original = 20_000  # 5000 tokens — above floor
        # Replacement at 95% of original fails the 0.9 margin.
        decision = gate_decision(original, int(original * 0.95))
        self.assertFalse(decision["offload"])
        self.assertIn("margin", decision["reason"])
        self.assertFalse(should_offload(original, int(original * 0.95)))
        # A compact replacement passes.
        self.assertTrue(should_offload(original, 4_000))

    def test_head_tail_preview_keeps_ends(self) -> None:
        text = "A" * 1000 + "MIDDLE" + "Z" * 1000
        preview = head_tail_preview(text, head=100, tail=100)
        self.assertTrue(preview.startswith("A" * 100))
        self.assertTrue(preview.endswith("Z" * 100))
        self.assertIn("omitted", preview)
        self.assertNotIn("MIDDLE", preview)


if __name__ == "__main__":
    unittest.main()
