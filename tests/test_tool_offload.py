"""Focused tests for savings-gated tool-output offload."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.tool_offload import (
    HARD_CAP_CHARS,
    OFFLOAD_MARKER,
    confine_offload_path,
    estimate_tokens,
    gate_decision,
    head_tail_preview,
    is_offload_stub,
    is_self_created_offload_blob,
    mark_offload_blob,
    offload_tool_output,
    read_offload_blob,
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

    def test_offload_stub_preserves_durable_path_pointer(self) -> None:
        text = ("HEAD-" + ("body" * 4000) + "-TAIL") * 2
        with TemporaryDirectory() as tmp:
            model_text, meta = offload_tool_output(
                text, state_dir=tmp, tool_name="read_file", tool_call_id="ptr1"
            )
            self.assertTrue(meta["offloaded"], meta)
            self.assertTrue(is_offload_stub(model_text))
            self.assertTrue(model_text.startswith(OFFLOAD_MARKER))
            self.assertIn("Full output saved to:", model_text)
            self.assertIn(meta["path"], model_text)
            self.assertIn("read_offload", model_text)
            # Pointer must resolve back to the spilled blob.
            blob = confine_offload_path(meta["path"], state_dir=tmp)
            self.assertEqual(blob.read_text(encoding="utf-8"), text)

    def test_read_offload_allows_self_blob(self) -> None:
        body = "line-one\nline-two\nline-three\nline-four\n"
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "tool_offload" / "self-blob.txt"
            target.parent.mkdir(parents=True)
            target.write_text(body, encoding="utf-8")
            mark_offload_blob(target)
            self.assertTrue(is_self_created_offload_blob(target))
            full = read_offload_blob(str(target.resolve()), state_dir=tmp)
            self.assertEqual(full, "line-one\nline-two\nline-three\nline-four")
            # Relative blob name under tool_offload also works.
            sliced = read_offload_blob(
                "self-blob.txt", state_dir=tmp, start_line=2, limit=2
            )
            self.assertEqual(sliced, "line-two\nline-three")

    def test_read_offload_refuses_traversal_and_foreign_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            offload_dir = Path(tmp) / "tool_offload"
            offload_dir.mkdir(parents=True)
            ok = offload_dir / "ok.txt"
            ok.write_text("safe\n", encoding="utf-8")
            mark_offload_blob(ok)

            # Path traversal relative to the offload root.
            escaped = read_offload_blob("../secrets.txt", state_dir=tmp)
            self.assertTrue(escaped.startswith("error:"), escaped)
            self.assertIn("escape", escaped.lower())

            # Absolute foreign path outside this state_dir's tool_offload.
            with TemporaryDirectory() as foreign:
                outsider = Path(foreign) / "passwd.txt"
                outsider.write_text("nope\n", encoding="utf-8")
                refused = read_offload_blob(str(outsider.resolve()), state_dir=tmp)
                self.assertTrue(refused.startswith("error:"), refused)
                self.assertIn("outside", refused.lower())

            # Sibling file under state_dir but not under tool_offload/.
            sibling = Path(tmp) / "tool_output_savings.jsonl"
            sibling.write_text("{}\n", encoding="utf-8")
            sibling_refused = read_offload_blob(str(sibling.resolve()), state_dir=tmp)
            self.assertTrue(sibling_refused.startswith("error:"), sibling_refused)

    def test_read_offload_refuses_planted_file_without_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            planted = Path(tmp) / "tool_offload" / "planted.txt"
            planted.parent.mkdir(parents=True)
            planted.write_text("secret-planted-body\n", encoding="utf-8")
            self.assertFalse(is_self_created_offload_blob(planted))
            refused = read_offload_blob("planted.txt", state_dir=tmp)
            self.assertTrue(refused.startswith("error:"), refused)
            self.assertIn("self-created", refused.lower())
            self.assertNotIn("secret-planted-body", refused)

    def test_offload_tool_output_marks_blob_readable(self) -> None:
        text = ("HEAD-" + ("body" * 4000) + "-TAIL") * 2
        with TemporaryDirectory() as tmp:
            model_text, meta = offload_tool_output(
                text, state_dir=tmp, tool_name="run_terminal", tool_call_id="marked"
            )
            self.assertTrue(meta["offloaded"], meta)
            blob = Path(meta["path"])
            self.assertTrue(is_self_created_offload_blob(blob))
            # Normal blob round-trip via read_offload (relative + absolute).
            via_abs = read_offload_blob(str(blob), state_dir=tmp, start_line=1, limit=1)
            self.assertTrue(via_abs.startswith("HEAD-"), via_abs)
            via_rel = read_offload_blob(blob.name, state_dir=tmp, start_line=1, limit=1)
            self.assertEqual(via_abs, via_rel)
            self.assertIn(meta["path"], model_text)

    def test_read_offload_large_blob_is_byte_bounded_without_full_load(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "tool_offload" / "huge.txt"
            target.parent.mkdir(parents=True)
            # ~5 MiB single-line prefix + a short second line we actually want.
            huge_prefix = "x" * 5_000_000
            target.write_text(huge_prefix + "\nwanted-line\n" + ("z" * 1000) + "\n", encoding="utf-8")
            mark_offload_blob(target)
            with patch.dict(os.environ, {"PUPPETMASTER_OFFLOAD_READ_MAX_CHARS": "64"}):
                sliced = read_offload_blob(
                    "huge.txt", state_dir=tmp, start_line=2, limit=1
                )
            self.assertEqual(sliced, "wanted-line")
            # Unbounded (no limit) must still clamp — never return megabytes.
            with patch.dict(os.environ, {"PUPPETMASTER_OFFLOAD_READ_MAX_CHARS": "128"}):
                capped = read_offload_blob("huge.txt", state_dir=tmp, start_line=1)
            self.assertLessEqual(len(capped), 128 + 80)
            self.assertIn("truncated", capped.lower())
            self.assertNotIn("wanted-line", capped)

if __name__ == "__main__":
    unittest.main()
