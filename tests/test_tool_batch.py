"""Tests for tool-batch segmentation and parallel execution."""
import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import os
import unittest
from unittest.mock import MagicMock, patch

from puppetmaster.tool_batch import (
    DEFAULT_TOOL_BATCH_MAX_WORKERS,
    is_parallel_enabled,
    parallel_executor_max_workers,
    parallel_worker_cap,
    plan_tool_batch_segments,
)

class ToolBatchSegmentationTests(unittest.TestCase):
    """Unit tests for plan_tool_batch_segments."""

    def setUp(self):
        # Ensure parallel execution is enabled for tests
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "1"
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", None)

    def tearDown(self):
        # Clean up environment
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_PARALLEL", None)
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", None)

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

    def test_large_safe_batch_stays_one_parallel_segment(self):
        """Worker cap bounds the executor, not segmentation — one parallel run."""
        calls = [
            self._make_tool_call("read_file", {"path": f"f{i}.py"}, str(i))
            for i in range(20)
        ]
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "4"
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "parallel")
        self.assertEqual(len(segments[0][1]), 20)

class ToolBatchWorkerCapTests(unittest.TestCase):
    """Unit tests for the Wave 5 parallel worker cap."""

    def setUp(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "1"
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", None)

    def tearDown(self):
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_PARALLEL", None)
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", None)

    def test_default_cap(self):
        self.assertEqual(parallel_worker_cap(), DEFAULT_TOOL_BATCH_MAX_WORKERS)

    def test_cap_parsing_valid_override(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "4"
        self.assertEqual(parallel_worker_cap(), 4)

    def test_cap_parsing_invalid_falls_back(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "not-a-number"
        self.assertEqual(parallel_worker_cap(), DEFAULT_TOOL_BATCH_MAX_WORKERS)

    def test_cap_parsing_clamps_low_and_high(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "0"
        self.assertEqual(parallel_worker_cap(), 1)
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "-3"
        self.assertEqual(parallel_worker_cap(), 1)
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "999"
        self.assertEqual(parallel_worker_cap(), 64)

    def test_executor_max_workers_bounded_by_cap(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "4"
        self.assertEqual(parallel_executor_max_workers(20), 4)
        self.assertEqual(parallel_executor_max_workers(3), 3)
        self.assertEqual(parallel_executor_max_workers(0), 1)

    def test_kill_switch_still_forces_sequential_segments(self):
        """PUPPETMASTER_TOOL_BATCH_PARALLEL=0 remains the disable path."""
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "0"
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "8"
        calls = [
            {"id": "1", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "2", "name": "read_file", "arguments": {"path": "b.py"}},
            {"id": "3", "name": "search_code", "arguments": {"query": "foo"}},
        ]
        segments = plan_tool_batch_segments(calls)
        self.assertEqual(segments, [("sequential", calls)])
        self.assertFalse(is_parallel_enabled())

class ToolBatchExecutorBoundTests(unittest.TestCase):
    """Agentic parallel segment uses the capped ThreadPoolExecutor size."""

    def setUp(self):
        os.environ["PUPPETMASTER_TOOL_BATCH_PARALLEL"] = "1"
        os.environ["PUPPETMASTER_TOOL_BATCH_MAX_WORKERS"] = "3"

    def tearDown(self):
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_PARALLEL", None)
        os.environ.pop("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", None)

    def test_parallel_executor_uses_capped_max_workers(self):
        from puppetmaster.adapters.agentic import AgenticAdapter

        adapter = AgenticAdapter()
        calls = [
            {"id": str(i), "name": "list_dir", "arguments": {"path": "."}}
            for i in range(10)
        ]
        captured = {}

        class _FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        def _fake_executor(*, max_workers):
            captured["max_workers"] = max_workers
            pool = MagicMock()

            def _submit(fn, call):
                return _FakeFuture(fn(call))

            pool.submit.side_effect = _submit
            pool.__enter__.return_value = pool
            pool.__exit__.return_value = False
            return pool

        with patch(
            "puppetmaster.adapters.agentic.concurrent.futures.ThreadPoolExecutor",
            side_effect=_fake_executor,
        ), patch(
            "puppetmaster.adapters.agentic.concurrent.futures.as_completed",
            side_effect=lambda futures: list(futures),
        ), patch.object(
            adapter, "_execute_tool", return_value="ok"
        ):
            results = adapter._execute_tool_segment_parallel(
                calls, cwd=MagicMock(), implement=False, task=MagicMock()
            )

        self.assertEqual(captured["max_workers"], 3)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(output == "ok" for _call, output in results))

if __name__ == "__main__":
    unittest.main()
