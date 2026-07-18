"""Tool-batch segmentation for parallel execution of parallel-safe tool calls.

Adapted from Hermes agent's tool_dispatch_helpers pattern. When a model emits a
batch mixing parallel-safe reads with barrier (side-effecting) tools, this
planner splits the batch into ordered segments — maximal contiguous runs of
parallel-safe calls, separated by sequential barrier calls — so the agentic
loop can run the safe segments concurrently while preserving emission order and
side-effect boundaries.

Parallel segments are also *bounded*: the executor never spins more threads than
``PUPPETMASTER_TOOL_BATCH_MAX_WORKERS`` (default 8). Segmentation / barrier
correctness is independent of that cap — large safe batches stay one parallel
segment, they just run with a fixed worker pool.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tools that must never run concurrently (interactive / user-facing).
_NEVER_PARALLEL_TOOLS = frozenset({
    "clarify",
    "submit_findings",
    "submit_report",
    "update_plan",
})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset({
    "read_file",
    "read_offload",
    "list_dir",
    "search_code",
    "graph_search",
    "graph_context",
    "web_fetch",
})

# File tools can run concurrently when they target independent paths.
# Only read_file is truly parallel-safe; write/edit/apply_hashline are barriers
# (apply_hashline is mutating and not listed in _PARALLEL_SAFE_TOOLS).
_PATH_SCOPED_TOOLS = frozenset({"read_file"})

# Conservative default — enough for a typical multi-read turn without
# unbounded threading on a 50-file batch. Env-overridable; hard-capped.
DEFAULT_TOOL_BATCH_MAX_WORKERS = 8
_ABSOLUTE_MAX_TOOL_BATCH_WORKERS = 64


def is_parallel_enabled() -> bool:
    """Check if tool-batch parallelization is enabled via environment variable.
    
    Returns False when PUPPETMASTER_TOOL_BATCH_PARALLEL is explicitly set to 0/false/off.
    """
    val = os.environ.get("PUPPETMASTER_TOOL_BATCH_PARALLEL", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def parallel_worker_cap() -> int:
    """Return the env-overridable cap on concurrent tool-batch workers.

    ``PUPPETMASTER_TOOL_BATCH_MAX_WORKERS`` defaults to
    :data:`DEFAULT_TOOL_BATCH_MAX_WORKERS`. Invalid / empty values fall back to
    the default; values are clamped to ``[1, 64]``.
    """
    raw = os.environ.get("PUPPETMASTER_TOOL_BATCH_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_TOOL_BATCH_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TOOL_BATCH_MAX_WORKERS
    return max(1, min(value, _ABSOLUTE_MAX_TOOL_BATCH_WORKERS))


def parallel_executor_max_workers(call_count: int) -> int:
    """ThreadPoolExecutor size for a parallel segment of ``call_count`` calls.

    Always at least 1 and never larger than the configured cap or the call
    count (no idle threads for tiny segments).
    """
    n = max(0, int(call_count))
    if n <= 0:
        return 1
    return max(1, min(n, parallel_worker_cap()))


def plan_tool_batch_segments(tool_calls: list) -> list[tuple[str, list]]:
    """Split a tool-call batch into ordered (kind, calls) segments.
    
    Returns a list of (kind, calls) tuples where kind is either "parallel"
    (a maximal contiguous run of parallel-safe calls) or "sequential" (one or
    more barrier calls that must run in-order). Segments preserve the model's
    original call order exactly — a later call never crosses an earlier barrier
    — so tool-result ordering and side-effect boundaries are identical to
    fully-sequential execution.
    
    Per-call safety rules:
    * _NEVER_PARALLEL_TOOLS (interactive tools) → barrier.
    * Unparseable / non-dict arguments → barrier.
    * Path-scoped tools (read_file/write_file/edit_file) join a parallel run
      only when their target path does not overlap another path already reserved
      in the same run; an overlap closes the run so the conflicting call starts
      a NEW run after the first completes.
    * Anything not in _PARALLEL_SAFE_TOOLS → barrier.
    
    Parallel runs shorter than two calls are demoted to sequential (no
    concurrency win), and adjacent sequential segments are merged.
    """
    if not is_parallel_enabled():
        # Kill switch: return everything as one sequential segment
        return [("sequential", list(tool_calls))]
    
    segments: list[list] = []
    current: list = []
    reserved_paths: list[Path] = []

    def _close_parallel() -> None:
        nonlocal current, reserved_paths
        if current:
            segments.append(["parallel", current])
            current = []
            reserved_paths = []

    def _add_sequential(tc) -> None:
        _close_parallel()
        if segments and segments[-1][0] == "sequential":
            segments[-1][1].append(tc)
        else:
            segments.append(["sequential", [tc]])

    for tool_call in tool_calls:
        tool_name = _extract_tool_name(tool_call)
        
        if tool_name in _NEVER_PARALLEL_TOOLS:
            _add_sequential(tool_call)
            continue

        try:
            function_args = _extract_tool_arguments(tool_call)
        except Exception:
            logger.debug(
                "Could not parse args for %s — treating as sequential barrier",
                tool_name,
            )
            _add_sequential(tool_call)
            continue
        
        if not isinstance(function_args, dict):
            logger.debug(
                "Non-dict args for %s (%s) — treating as sequential barrier",
                tool_name,
                type(function_args).__name__,
            )
            _add_sequential(tool_call)
            continue

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                _add_sequential(tool_call)
                continue
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                # Same-subtree conflict inside this run: close it so this
                # call starts a fresh run AFTER the conflicting one lands.
                _close_parallel()
            reserved_paths.append(scoped_path)
            current.append(tool_call)
            continue

        if tool_name in _PARALLEL_SAFE_TOOLS:
            current.append(tool_call)
            continue

        _add_sequential(tool_call)

    _close_parallel()

    # Normalize: demote single-call parallel segments to sequential, merge adjacent sequential
    normalized: list[list] = []
    for kind, calls in segments:
        if kind == "parallel" and len(calls) < 2:
            kind = "sequential"
        if normalized and normalized[-1][0] == "sequential" and kind == "sequential":
            normalized[-1][1].extend(calls)
        else:
            normalized.append([kind, calls])
    
    return [(kind, calls) for kind, calls in normalized]


def _extract_tool_name(tool_call: Any) -> str:
    """Extract tool name from a tool call object.
    
    Handles both dict-like and object-like tool call structures.
    """
    if isinstance(tool_call, dict):
        return str(tool_call.get("name", ""))
    if hasattr(tool_call, "function"):
        func = tool_call.function
        if isinstance(func, dict):
            return str(func.get("name", ""))
        if hasattr(func, "name"):
            return str(func.name)
    if hasattr(tool_call, "name"):
        return str(tool_call.name)
    return ""


def _extract_tool_arguments(tool_call: Any) -> dict:
    """Extract and parse tool arguments from a tool call object."""
    args_raw = None
    
    if isinstance(tool_call, dict):
        args_raw = tool_call.get("arguments")
    elif hasattr(tool_call, "function"):
        func = tool_call.function
        if isinstance(func, dict):
            args_raw = func.get("arguments")
        elif hasattr(func, "arguments"):
            args_raw = func.arguments
    elif hasattr(tool_call, "arguments"):
        args_raw = tool_call.arguments
    
    if args_raw is None:
        return {}
    
    if isinstance(args_raw, dict):
        return args_raw
    
    if isinstance(args_raw, str):
        return json.loads(args_raw)
    
    return {}


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Optional[Path]:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]
