"""Tests for tool-batch segmentation and parallel execution."""
import json
import os
import unittest
from unittest.mock import Mock, patch

from puppetmaster.tool_batch import (
    is_parallel_enabled,
    plan_tool_batch_segments,
)


class ToolBatchSegmentationTests(unittest.TestCase):
    """Unit tests for plan_tool_batch_segments."""

    def setUp(self):
        # Ensure parallel execution is enabled for tests
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "1"

    def tearDown(self):
        # Clean up environment
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_PARALLEL", None)

    def _make_tool_call(self, name: str, args: dict, call_id: str = "test-id"):
        """Helper to create a tool call dict."""
        return {
            "id": call_id,
            "name": name,
            "arguments": args,
        }

    def test_all_parallel_safe_single_segment(self):
        """All parallel-safe tools should produce a single parallel segment."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "b.py"}, "2"),
            self._make_tool_call("search_code", {"query": "foo"}, "3"),
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "parallel")
        self.assertEqual(len(segments[0][1]), 3)

    def test_all_barriers_sequential_segment(self):
        """All barrier tools should produce a single sequential segment."""
        calls = [
            self._make_tool_call("write_file", {"path": "a.py", "content": "x"}, "1"),
            self._make_tool_call("edit_file", {"path": "b.py", "old_string": "y", "new_string": "z"}, "2"),
            self._make_tool_call("run_terminal", {"command": "pytest"}, "3"),
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "sequential")
        self.assertEqual(len(segments[0][1]), 3)

    def test_apply_hashline_is_mutating_barrier(self):
        """apply_hashline must not join a parallel read segment (like edit_file)."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "b.py"}, "2"),
            self._make_tool_call("apply_hashline", {"patch": "[a.py#ABCD]\nDEL 1\n"}, "3"),
            self._make_tool_call("search_code", {"query": "foo"}, "4"),
            self._make_tool_call("list_dir", {"path": "."}, "5"),
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][0], "parallel")
        self.assertEqual(len(segments[0][1]), 2)
        self.assertEqual(segments[1][0], "sequential")
        self.assertEqual(segments[1][1][0]["name"], "apply_hashline")
        self.assertEqual(segments[2][0], "parallel")
        self.assertEqual(len(segments[2][1]), 2)

    def test_mixed_batch_splits_correctly(self):
        """Mixed parallel-safe and barrier tools should split into segments."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "b.py"}, "2"),
            self._make_tool_call("write_file", {"path": "c.py", "content": "x"}, "3"),
            self._make_tool_call("search_code", {"query": "foo"}, "4"),
            self._make_tool_call("list_dir", {"path": "."}, "5"),
        ]
        segments = plan_tool_batch_segments(calls)
        # Expected: [parallel(read, read), sequential(write), parallel(search, list)]
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][0], "parallel")
        self.assertEqual(len(segments[0][1]), 2)
        self.assertEqual(segments[1][0], "sequential")
        self.assertEqual(len(segments[1][1]), 1)
        self.assertEqual(segments[2][0], "parallel")
        self.assertEqual(len(segments[2][1]), 2)

    def test_single_parallel_call_demoted_to_sequential(self):
        """A parallel segment with only one call should be demoted to sequential."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "sequential")
        self.assertEqual(len(segments[0][1]), 1)

    def test_path_overlap_closes_parallel_segment(self):
        """Path-scoped tools with overlapping paths should close the parallel segment."""
        calls = [
            self._make_tool_call("read_file", {"path": "src/a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "src/b.py"}, "2"),
            self._make_tool_call("read_file", {"path": "src/a.py"}, "3"),  # overlaps with call 1
            self._make_tool_call("list_dir", {"path": "."}, "4"),
        ]
        segments = plan_tool_batch_segments(calls)
        # Expected: [parallel(read src/a, read src/b), parallel(read src/a, list)]
        # The third read_file overlaps with the first, so it closes the segment
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0], "parallel")
        self.assertEqual(len(segments[0][1]), 2)
        self.assertEqual(segments[1][0], "parallel")
        self.assertEqual(len(segments[1][1]), 2)

    def test_unparseable_arguments_treated_as_barrier(self):
        """Tool calls with unparseable arguments should be treated as barriers."""
        class BadToolCall:
            name = "read_file"
            arguments = "not valid json"
        
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            BadToolCall(),
            self._make_tool_call("search_code", {"query": "foo"}, "2"),
        ]
        segments = plan_tool_batch_segments(calls)
        # Expected: [sequential(read), sequential(bad), sequential(search)]
        # All become sequential because single parallel calls are demoted
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "sequential")
        self.assertEqual(len(segments[0][1]), 3)

    def test_never_parallel_tools_are_barriers(self):
        """Tools in _NEVER_PARALLEL_TOOLS should always be barriers."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("clarify", {"question": "what?"}, "2"),
            self._make_tool_call("search_code", {"query": "foo"}, "3"),
        ]
        segments = plan_tool_batch_segments(calls)
        # Expected: [sequential(read), sequential(clarify), sequential(search)]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "sequential")
        self.assertEqual(len(segments[0][1]), 3)

    def test_kill_switch_disables_parallelization(self):
        """PUPPETMASTER_TOOL_BATCH_PARALLEL=0 should force all sequential."""
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "0"
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "b.py"}, "2"),
            self._make_tool_call("search_code", {"query": "foo"}, "3"),
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "sequential")
        self.assertEqual(len(segments[0][1]), 3)

    def test_is_parallel_enabled_respects_env_var(self):
        """is_parallel_enabled should respect PUPPETMASTER_TOOL_BATCH_PARALLEL."""
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "1"
        self.assertTrue(is_parallel_enabled())
        
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "0"
        self.assertFalse(is_parallel_enabled())
        
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "false"
        self.assertFalse(is_parallel_enabled())
        
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "off"
        self.assertFalse(is_parallel_enabled())

    def test_empty_batch_returns_empty_segments(self):
        """An empty tool call batch should return empty segments."""
        segments = plan_tool_batch_segments([])
        self.assertEqual(len(segments), 0)

    def test_preserves_original_call_order(self):
        """Segments should preserve the original tool call order."""
        calls = [
            self._make_tool_call("read_file", {"path": "a.py"}, "1"),
            self._make_tool_call("read_file", {"path": "b.py"}, "2"),
            self._make_tool_call("write_file", {"path": "c.py", "content": "x"}, "3"),
            self._make_tool_call("search_code", {"query": "foo"}, "4"),
        ]
        segments = plan_tool_batch_segments(calls)
        # Flatten segments to check order
        flattened = []
        for _kind, segment_calls in segments:
            flattened.extend(segment_calls)
        
        self.assertEqual(len(flattened), 4)
        self.assertEqual(flattened[0]["id"], "1")
        self.assertEqual(flattened[1]["id"], "2")
        self.assertEqual(flattened[2]["id"], "3")
        self.assertEqual(flattened[3]["id"], "4")


if __name__ == "__main__":
    unittest.main()
