from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Hermetic tests: the orchestrator's first-run plan-catalog auto-discovery
# shells out to the Cursor SDK (node) when CURSOR_API_KEY is set. The suite
# must never make that network/subprocess call implicitly — tests that
# exercise the discovery helper inject their own catalog fetcher instead.
os.environ.setdefault("PUPPETMASTER_AUTODISCOVER", "0")

from puppetmaster.adapters import (
    ClaudeCodeAdapter,
    CodexAdapter,
    CursorAdapter,
    OpenAIAdapter,
    UnconfiguredProviderAdapter,
    build_claude_code_command,
    build_codex_exec_command,
    classify_claude_code_failure,
    classify_codex_failure,
    classify_cursor_failure,
    classify_openai_failure,
    last_codex_agent_message,
    parse_codex_events,
)
from puppetmaster.codegraph import (
    codegraph_affected,
    codegraph_context,
    codegraph_files_listing,
    codegraph_initialized,
    codegraph_query,
    codegraph_ready,
    enrich_prompt_with_codegraph,
    run_codegraph_cli,
)
from bench.codegraph_ab import (
    build_report,
    load_prompt,
    measure_enrichment,
    render_markdown,
)
from bench.three_way import (
    ArtifactFacts,
    ConfigCost,
    RepoFacts,
    load_artifact_facts,
    model_costs,
    render_markdown as render_three_way_markdown,
    scan_repo,
)
from puppetmaster.config import load_config
from puppetmaster.cli import (
    artifact_feed,
    artifact_feed_since,
    cursor_prompt,
    main as cli_main,
)
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.mcp_server import ASYNC_PROCESSES, call_tool, handle_message
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus, seconds_from_now
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.state import resolve_state_dir
from puppetmaster.stitcher import Stitcher
from puppetmaster.store import SwarmStore
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec
from typing import Optional


class _ContextManagerFakeProcess:
    """Drop-in fake for ``subprocess.Popen`` that supports ``with`` blocks.

    Patching ``subprocess.Popen`` affects ``subprocess.run`` too (the
    latter uses ``with Popen(...) as p:`` internally), so any fake we
    return has to behave like a context manager.
    """

    def __init__(self, pid: int = 1000, returncode: int = 0) -> None:
        self.pid = pid
        self.returncode = returncode
        self.args: list[str] = []
        self.stdout = None
        self.stderr = None
        self.stdin = None

    def __enter__(self) -> "_ContextManagerFakeProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.returncode

    def communicate(
        self, input: Optional[str] = None, timeout: Optional[float] = None
    ) -> tuple[str, str]:
        return ("", "")

    def kill(self) -> None:
        return None

    def terminate(self) -> None:
        return None


def _find_indexer_launches(popen_mock) -> list[list[str]]:
    """Return the calls to ``subprocess.Popen`` that launched our background
    indexer runner. Other paths (notably ``git rev-parse`` from state
    resolution) also go through ``Popen`` when patched globally, so we
    can't just count total calls.
    """
    launches: list[list[str]] = []
    for call in popen_mock.call_args_list:
        argv = call.args[0] if call.args else []
        if not isinstance(argv, (list, tuple)):
            continue
        if any("codegraph_index_runner" in str(part) for part in argv):
            launches.append(list(argv))
    return launches


def _wait_for_spawned_or_kill(test, spawned, *, timeout: float) -> None:
    """Wait for spawned subprocesses to finish, killing on timeout.

    The pattern this replaces — ``process.wait(timeout=15)`` in a bare
    loop — has two problems on a developer machine: (a) the inline
    local-demo swarm has crept past 15 s of wall time as the codebase
    grew, making the test deterministically fail; (b) when wait raises
    ``TimeoutExpired``, the spawned process is never terminated, so
    every failed run leaks a long-lived ``python -m puppetmaster run``
    child that holds open SQLite handles and orphans CodeGraph
    indexers. Both manifest as "the whole suite hangs" because later
    tests block on the same resources the orphans hold.

    Using a per-process try/finally with ``kill()`` + a second wait
    forces deterministic cleanup. The 60 s default gives ~3.5×
    headroom over the observed 17 s steady state and a clear failure
    message if the swarm has genuinely regressed beyond that.
    """
    import subprocess

    for process in spawned:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            test.fail(
                f"spawned puppetmaster swarm did not finish within {timeout}s "
                f"(pid={process.pid}, args={process.args!r}); killed to prevent "
                "leaking a long-lived child process"
            )


class PuppetmasterTests(unittest.TestCase):
    """Hermetic suite: pins ``PUPPETMASTER_MODELS_PATH`` to an empty
    location for every test so the developer's real
    ``~/.puppetmaster/models.json`` registry can't leak in and trip
    auto-routing (which would route default workers to claude-code,
    yielding BLOCKED artifacts and the wrong counts/roles).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._models_path_tmp = tempfile.mkdtemp(prefix="pm-models-isolation-")
        cls._prev_models_env = os.environ.get("PUPPETMASTER_MODELS_PATH")
        os.environ["PUPPETMASTER_MODELS_PATH"] = str(
            Path(cls._models_path_tmp) / "no-such-models.json"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._prev_models_env is None:
            os.environ.pop("PUPPETMASTER_MODELS_PATH", None)
        else:
            os.environ["PUPPETMASTER_MODELS_PATH"] = cls._prev_models_env
        shutil.rmtree(cls._models_path_tmp, ignore_errors=True)

    def test_mcp_lists_puppetmaster_agent_tools(self) -> None:
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = {tool["name"] for tool in response["result"]["tools"]}

        self.assertIn("puppetmaster_doctor", tool_names)
        self.assertIn("puppetmaster_cursor_review", tool_names)
        self.assertIn("puppetmaster_start_cursor_review", tool_names)
        self.assertIn("puppetmaster_claude_implement", tool_names)
        self.assertIn("puppetmaster_start_claude_implement", tool_names)
        self.assertIn("puppetmaster_start_swarm", tool_names)
        self.assertIn("puppetmaster_start_cursor_swarm", tool_names)
        self.assertIn("puppetmaster_status", tool_names)
        self.assertIn("puppetmaster_live_artifacts", tool_names)
        self.assertIn("puppetmaster_partial_summary", tool_names)

    def test_default_state_dir_stays_outside_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            resolved = resolve_state_dir(cwd=workspace)

            self.assertNotEqual(resolved, workspace / ".puppetmaster")
            self.assertIn("puppetmaster", str(resolved))

    def test_state_dir_env_override_can_be_workspace_relative(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": ".pm-env"}):
                resolved = resolve_state_dir(cwd=workspace)

            self.assertEqual(resolved, workspace / ".pm-env")

    def test_mcp_tool_call_wraps_cli_result(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="job_123\n",
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            with patch("puppetmaster.mcp_server.subprocess.run", return_value=completed) as run:
                result = call_tool(
                    "puppetmaster_last_job",
                    {"cwd": tmp, "state_dir": ".pm-test"},
                )

        called_args = run.call_args.args[0]
        self.assertEqual(called_args[-1], "last")
        self.assertIn("job_123", result["content"][0]["text"])
        self.assertFalse(result["isError"])

    def test_mcp_start_tool_returns_job_id_without_waiting_for_completion(self) -> None:
        with TemporaryDirectory() as tmp:
            before_process_count = len(ASYNC_PROCESSES)
            result = call_tool(
                "puppetmaster_start_swarm",
                {
                    "cwd": tmp,
                    "state_dir": ".pm-test",
                    "goal": "async mcp smoke",
                    "roles": ["explore"],
                    "allow_local_demo": True,
                    "worker_mode": "inline",
                },
            )
            payload = json.loads(result["content"][0]["text"])

            self.assertIn("job_", payload["job_id"])
            self.assertIn("pid", payload)
            self.assertFalse(result["isError"])

            spawned = ASYNC_PROCESSES[before_process_count:]
            _wait_for_spawned_or_kill(self, spawned, timeout=60)

            status = call_tool(
                "puppetmaster_status",
                {
                    "cwd": tmp,
                    "state_dir": ".pm-test",
                    "job_id": payload["job_id"],
                },
            )
            status_body = json.loads(status["content"][0]["text"])
            self.assertFalse(status["isError"], status_body)
            status_payload = json.loads(status_body["stdout"])

            self.assertEqual(status_payload["job"]["status"], "complete")
            self.assertGreaterEqual(
                status_payload["artifact_count"],
                2,
                msg=(
                    "spawned swarm should produce at least the base artifact pair "
                    "(finding + verification); the exact count is intentionally not "
                    "asserted because routing and memory artifacts have been added "
                    "since this test was written"
                ),
            )

    def test_mcp_custom_roles_fail_without_real_adapter_or_config(self) -> None:
        with TemporaryDirectory() as tmp:
            result = call_tool(
                "puppetmaster_start_swarm",
                {
                    "cwd": tmp,
                    "state_dir": ".pm-test",
                    "goal": "must not silently use demo workers",
                    "roles": ["pipeline-mapper", "decision-explainer"],
                },
            )
            payload = json.loads(result["content"][0]["text"])

            self.assertTrue(result["isError"])
            self.assertIn("demo local adapter", payload["error"])
            self.assertIn("puppetmaster_start_cursor_swarm", payload["fix"])

    def test_mcp_adapter_generates_config_for_custom_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            before_process_count = len(ASYNC_PROCESSES)
            result = call_tool(
                "puppetmaster_start_swarm",
                {
                    "cwd": tmp,
                    "state_dir": ".pm-test",
                    "goal": "explicit adapter mcp smoke",
                    "roles": ["pipeline-mapper"],
                    "adapter": "local",
                    "worker_mode": "inline",
                    "auto_route": False,
                },
            )
            payload = json.loads(result["content"][0]["text"])
            spawned = ASYNC_PROCESSES[before_process_count:]
            _wait_for_spawned_or_kill(self, spawned, timeout=60)

            status = call_tool(
                "puppetmaster_status",
                {
                    "cwd": tmp,
                    "state_dir": ".pm-test",
                    "job_id": payload["job_id"],
                },
            )
            status_body = json.loads(status["content"][0]["text"])
            status_payload = json.loads(status_body["stdout"])

            self.assertFalse(result["isError"])
            self.assertEqual(status_payload["job"]["status"], "complete")
            self.assertEqual(status_payload["tasks"][0]["adapter"], "local")

    def test_run_creates_artifacts_summary_and_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run("prove the swarm contract")

            artifacts = store.list_artifacts(result.job.id)
            memory = store.list_memory()

            self.assertGreaterEqual(len(artifacts), 7)
            self.assertTrue(result.summary_path.exists())
            self.assertIn("Final synthesis used structured JSON artifacts only", result.summary)
            self.assertGreaterEqual(len(memory), 4)
            self.assertTrue(all(artifact.sha256 for artifact in artifacts))

    def test_live_artifact_feed_and_partial_summary_are_available(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run(
                "make live artifacts inspectable",
                roles=["explore"],
                worker_mode="inline",
            )

            feed = artifact_feed(store, result.job.id)
            partial = Stitcher(store).preview(result.job.id)

            self.assertEqual(len(feed), 2)
            self.assertIn("artifact", feed[0])
            self.assertIn("# Puppetmaster Live Summary", partial)
            self.assertIn("make live artifacts inspectable", partial)

    def _store_for_backend(self, backend: str, root: Path):
        if backend == "file":
            return SwarmStore(root)
        return SQLiteSwarmStore(root)

    def test_event_cursor_and_since_read_are_consistent_across_backends(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()

                self.assertEqual(store.event_cursor("job-x"), 0)
                self.assertEqual(store.read_events_since("job-x", since=0), [])

                store.emit("job-x", "first", {"n": 1})
                store.emit("job-x", "second", {"n": 2})
                store.emit("job-x", "third", {"n": 3})

                self.assertEqual(store.event_cursor("job-x"), 3)

                first_batch = store.read_events_since("job-x", since=0)
                self.assertEqual([e["event"] for e in first_batch], ["first", "second", "third"])
                self.assertEqual([e["id"] for e in first_batch], [1, 2, 3])

                resumed = store.read_events_since("job-x", since=2)
                self.assertEqual([e["event"] for e in resumed], ["third"])
                self.assertEqual([e["id"] for e in resumed], [3])

                self.assertEqual(store.read_events_since("job-x", since=3), [])

    def test_file_reader_skips_torn_concurrent_append_line(self) -> None:
        """A malformed/torn line (Windows non-atomic append) is skipped, not fatal.

        POSIX O_APPEND is atomic; Windows appends are not, so two workers
        writing at once can interleave a partial or null-padded line into the
        JSONL stream. The reader must survive it and still return the
        well-formed events around it.
        """
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.emit("job-x", "first", {"n": 1})
            store.emit("job-x", "second", {"n": 2})

            stream = store.stream_dir / "job-x.jsonl"
            with stream.open("a", encoding="utf-8") as handle:
                handle.write('{"event": "torn", "payl\x00\x00')  # truncated + null pad
                handle.write("\n")
                handle.write("\n")  # stray blank line
            store.emit("job-x", "third", {"n": 3})

            events = [e["event"] for e in store.read_events("job-x")]
            self.assertEqual(events, ["first", "second", "third"])

    def test_wait_for_events_returns_immediately_when_events_present(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()
                store.emit("job-x", "ready", {"hint": "go"})

                start = time.monotonic()
                events = store.wait_for_events(
                    "job-x", since=0, timeout_seconds=5.0, poll_interval=0.05
                )
                elapsed = time.monotonic() - start

                self.assertEqual([e["event"] for e in events], ["ready"])
                self.assertLess(elapsed, 0.2)

    def test_wait_for_events_times_out_when_nothing_new(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()
                store.emit("job-x", "already-seen", {})

                start = time.monotonic()
                events = store.wait_for_events(
                    "job-x", since=1, timeout_seconds=0.2, poll_interval=0.05
                )
                elapsed = time.monotonic() - start

                self.assertEqual(events, [])
                self.assertGreaterEqual(elapsed, 0.15)
                self.assertLess(elapsed, 1.0)

    def test_wait_for_events_wakes_up_when_event_is_emitted_concurrently(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()

                def emit_later() -> None:
                    time.sleep(0.05)
                    store.emit("job-x", "arrived", {"v": 1})

                emitter = threading.Thread(target=emit_later)
                emitter.start()
                try:
                    start = time.monotonic()
                    events = store.wait_for_events(
                        "job-x", since=0, timeout_seconds=2.0, poll_interval=0.02
                    )
                    elapsed = time.monotonic() - start
                finally:
                    emitter.join()

                self.assertEqual([e["event"] for e in events], ["arrived"])
                self.assertLess(elapsed, 1.0)

    def test_artifact_feed_since_resumes_with_cursor(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run(
                "exercise the feed cursor",
                roles=["explore"],
                worker_mode="inline",
            )

            first_pass, cursor = artifact_feed_since(store, result.job.id, since=0)
            self.assertGreater(len(first_pass), 0)
            self.assertGreater(cursor, 0)

            second_pass, second_cursor = artifact_feed_since(
                store, result.job.id, since=cursor
            )
            self.assertEqual(second_pass, [])
            self.assertEqual(second_cursor, cursor)

    def test_mcp_follow_returns_existing_artifacts_immediately(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".pm-state"
            store = SQLiteSwarmStore(state_dir)
            result = Orchestrator(store).run(
                "feed the follow tool",
                roles=["explore"],
                worker_mode="inline",
            )

            response = call_tool(
                "puppetmaster_live_artifacts_follow",
                {
                    "job_id": result.job.id,
                    "state_dir": str(state_dir),
                    "since_cursor": 0,
                    "timeout_seconds": 1.0,
                },
            )

            self.assertFalse(response["isError"])
            body = json.loads(response["content"][0]["text"])
            self.assertGreater(body["item_count"], 0)
            self.assertFalse(body["timed_out"])
            self.assertGreater(body["next_cursor"], 0)
            self.assertEqual(body["job_id"], result.job.id)

    def test_mcp_follow_times_out_cleanly_when_no_new_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".pm-state"
            store = SQLiteSwarmStore(state_dir)
            result = Orchestrator(store).run(
                "feed the follow tool",
                roles=["explore"],
                worker_mode="inline",
            )
            current = store.event_cursor(result.job.id)

            start = time.monotonic()
            response = call_tool(
                "puppetmaster_live_artifacts_follow",
                {
                    "job_id": result.job.id,
                    "state_dir": str(state_dir),
                    "since_cursor": current,
                    "timeout_seconds": 0.2,
                    "poll_interval_seconds": 0.05,
                },
            )
            elapsed = time.monotonic() - start

            self.assertFalse(response["isError"])
            body = json.loads(response["content"][0]["text"])
            self.assertEqual(body["item_count"], 0)
            self.assertTrue(body["timed_out"])
            self.assertEqual(body["next_cursor"], current)
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 1.5)

    def test_mcp_live_artifacts_and_partial_summary_wrap_cli(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="[]\n",
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            arguments = {"job_id": "job_123", "state_dir": str(Path(tmp) / ".pm-test")}
            with patch("puppetmaster.mcp_server.subprocess.run", return_value=completed) as run:
                artifacts = call_tool(
                    "puppetmaster_live_artifacts",
                    {**arguments, "limit": 2},
                )
                summary = call_tool(
                    "puppetmaster_partial_summary",
                    arguments,
                )

        calls = [call.args[0] for call in run.call_args_list]
        self.assertIn("feed", calls[0])
        self.assertIn("--json", calls[0])
        self.assertIn("--limit", calls[0])
        self.assertEqual(calls[1][-2:], ["job_123", "--partial"])
        self.assertFalse(artifacts["isError"])
        self.assertFalse(summary["isError"])

    def test_custom_workers_still_emit_structured_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run("map architecture", roles=["explore", "test"])
            artifact_types = {artifact.type for artifact in store.list_artifacts(result.job.id)}

            self.assertEqual(len(store.list_tasks(result.job.id)), 2)
            self.assertIn(ArtifactType.FINDING, artifact_types)
            self.assertIn(ArtifactType.VERIFICATION, artifact_types)

    def test_locks_are_exclusive_until_released(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")

            self.assertTrue(store.acquire_lock("repo:path.py", "worker-a"))
            self.assertFalse(store.acquire_lock("repo:path.py", "worker-b"))
            store.release_lock("repo:path.py")
            self.assertTrue(store.acquire_lock("repo:path.py", "worker-b"))

    def test_task_leases_block_other_workers_until_recovered(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("recover abandoned work")
            task = Task(job_id=job.id, role="coder", instruction="write the patch")
            store.save_task(task)

            claimed = store.claim_task(task.id, "worker-a", lease_seconds=60)
            blocked = store.claim_task(task.id, "worker-b")
            store.save_task(replace(claimed, lease_expires_at=seconds_from_now(-1)))
            recovered = store.recover_stale_tasks(job.id)
            reclaimed = store.claim_task(task.id, "worker-b")

            self.assertIsNotNone(claimed)
            self.assertIsNone(blocked)
            self.assertEqual([task.id for task in recovered], [task.id])
            self.assertIsNotNone(reclaimed)
            self.assertEqual(reclaimed.lease_owner, "worker-b")
            self.assertEqual(reclaimed.attempts, 2)

    def test_max_attempts_marks_poison_task_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("stop poison retries")
            task = Task(
                job_id=job.id,
                role="coder",
                instruction="will not converge",
                attempts=store.max_task_attempts,
            )
            store.save_task(task)

            claimed = store.claim_task(task.id, "worker-a")
            failed = store.get_task_by_id(task.id)

            self.assertIsNone(claimed)
            self.assertEqual(failed.status, TaskStatus.FAILED)

    def test_status_snapshot_reports_counts_and_stale_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("inspect runtime state")
            task = Task(job_id=job.id, role="reviewer", instruction="look for risks")
            store.save_task(task)
            claimed = store.claim_task(task.id, "worker-a", lease_seconds=60)
            store.save_task(replace(claimed, lease_expires_at=seconds_from_now(-1)))

            snapshot = store.status_snapshot(job.id)

            self.assertIsNotNone(claimed)
            self.assertEqual(snapshot["task_counts"][str(TaskStatus.RUNNING)], 1)
            self.assertEqual(snapshot["stale_task_ids"], [task.id])

    def test_stitching_is_deterministic_for_existing_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run("make stitching replayable", roles=["explore"])

            first = result.summary
            second = Stitcher(store).stitch(result.job.id)

            self.assertEqual(first, second)

    def test_retrieved_memory_matches_goal_terms(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run("make stitching replayable", roles=["explore"])

            matches = store.retrieve_memory("independent workers", limit=3)

            self.assertEqual(store.latest_job().id, result.job.id)
            self.assertTrue(matches)

    def test_memory_retrieval_supports_scope_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            Orchestrator(store).run("make workers independent", roles=["explore"])

            scoped = store.retrieve_memory("workers", scope="swarm.findings")
            missing = store.retrieve_memory("workers", scope="swarm.decisions")

            self.assertTrue(scoped)
            self.assertFalse(any(memory["scope"] == "swarm.findings" for memory in missing))

    def test_subprocess_swarm_emits_worker_process_events(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run("prove workers are separate processes", roles=["explore"])
            events = store.read_events(result.job.id)
            event_names = {event["event"] for event in events}

            self.assertIn("task.claimed", event_names)
            self.assertIn("worker.completed_task", event_names)
            self.assertTrue(
                any(
                    artifact.created_by.startswith("worker-explore-")
                    for artifact in store.list_artifacts(result.job.id)
                )
            )

    def test_inline_worker_mode_avoids_subprocess_cold_start(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run(
                "prove fast inline orchestration",
                roles=["explore"],
                worker_mode="inline",
            )

            artifacts = store.list_artifacts(result.job.id)

            self.assertTrue(artifacts)
            self.assertTrue(all(artifact.created_by == "worker-explore-inline" for artifact in artifacts))
            self.assertEqual(store.latest_job().status, JobStatus.COMPLETE)

    def test_daemon_worker_mode_uses_warm_worker(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result_holder = {}

            def run_job() -> None:
                try:
                    result_holder["result"] = Orchestrator(store).run(
                        "prove warm daemon orchestration",
                        roles=["explore"],
                        worker_mode="daemon",
                    )
                except Exception as exc:
                    result_holder["error"] = exc

            thread = threading.Thread(target=run_job)
            thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and store.latest_job() is None:
                time.sleep(0.01)

            processed = WorkerDaemon(
                store,
                roles=["explore"],
                worker_id="daemon-test",
            ).run(max_tasks=1, max_idle_seconds=2)
            thread.join(timeout=3)

            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", result_holder)
            self.assertEqual(processed, 1)
            result = result_holder["result"]
            self.assertEqual(result.job.status, JobStatus.COMPLETE)
            self.assertTrue(
                all(
                    artifact.created_by == "daemon-test-explore"
                    for artifact in store.list_artifacts(result.job.id)
                )
            )

    def test_crash_recovery_demo_reclaims_abandoned_task(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            result = Orchestrator(store).run_crash_recovery_demo(
                "recover from a dead worker",
                crash_role="implement",
            )
            events = store.read_events(result.job.id)
            event_names = [event["event"] for event in events]
            implement_tasks = [
                task for task in store.list_tasks(result.job.id) if task.role == "implement"
            ]

            self.assertEqual(result.recovered_tasks, 1)
            self.assertIn("worker.crashed_after_claim", event_names)
            self.assertIn("task.recovered", event_names)
            self.assertEqual(implement_tasks[0].status, TaskStatus.COMPLETE)
            self.assertEqual(implement_tasks[0].attempts, 2)

    def test_worker_failure_marks_job_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")

            with self.assertRaises(RuntimeError):
                Orchestrator(store).run(
                    "fail closed",
                    specs=[
                        WorkerSpec(
                            role="broken",
                            instruction="use unsupported adapter",
                            adapter="missing-adapter",
                        )
                    ],
                )

            self.assertEqual(store.latest_job().status, JobStatus.FAILED)

    def test_dependency_graph_blocks_until_upstream_completes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            specs = [
                WorkerSpec(role="explore", instruction="find facts"),
                WorkerSpec(
                    role="architect",
                    instruction="choose design",
                    depends_on_roles=["explore"],
                ),
            ]
            result = Orchestrator(store).run("respect dependencies", specs=specs)
            tasks = {task.role: task for task in store.list_tasks(result.job.id)}
            events = store.read_events(result.job.id)

            self.assertEqual(tasks["explore"].status, TaskStatus.COMPLETE)
            self.assertEqual(tasks["architect"].status, TaskStatus.COMPLETE)
            self.assertEqual(tasks["architect"].depends_on, [tasks["explore"].id])
            self.assertIn("task.unblocked", {event["event"] for event in events})

    def test_shell_adapter_executes_command_and_records_verification(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            specs = [
                WorkerSpec(
                    role="runtime-check",
                    instruction="check python runtime",
                    adapter="shell",
                    payload={
                        "command": [sys.executable, "--version"],
                        "timeout_seconds": 10,
                    },
                )
            ]
            result = Orchestrator(store).run("run a shell verification", specs=specs)
            artifacts = store.list_artifacts(result.job.id)

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].type, ArtifactType.VERIFICATION)
            self.assertEqual(artifacts[0].payload["result"], "passed")
            self.assertIn("Python", artifacts[0].payload["stdout"] + artifacts[0].payload["stderr"])

    def test_artifact_schema_rejects_missing_required_payload(self) -> None:
        artifact = Artifact(
            job_id="job",
            task_id="task",
            type=ArtifactType.DECISION,
            created_by="worker",
            confidence=0.9,
            evidence=["unit-test"],
            payload={"decision": "do it"},
        )

        with self.assertRaises(ValueError):
            artifact.validate()

    def test_config_loads_enterprise_worker_specs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "puppetmaster.json"
            path.write_text(
                """
{
  "lease_seconds": 7,
  "workers": [
    {
      "role": "runtime-check",
      "instruction": "verify runtime",
      "adapter": "shell",
      "payload": {"command": ["python", "--version"]},
      "depends_on": []
    }
  ]
}
""",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.lease_seconds, 7)
            self.assertEqual(config.workers[0].adapter, "shell")

    def test_config_accepts_cursor_worker_specs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "puppetmaster.json"
            path.write_text(
                """
{
  "workers": [
    {
      "role": "agent",
      "instruction": "ask Cursor to inspect the repo",
      "adapter": "cursor",
      "payload": {"prompt": "Summarize this repository", "model": "default"}
    }
  ]
}
""",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.workers[0].adapter, "cursor")
            self.assertEqual(config.workers[0].payload["model"], "default")

    def test_cursor_prompt_modes_are_composable(self) -> None:
        prompt = cursor_prompt("Inspect repo", review=True, plan=True, dry_run=True)

        self.assertIn("Review mode", prompt)
        self.assertIn("Plan mode", prompt)
        self.assertIn("Dry-run constraint", prompt)

    def test_cursor_failure_classification_is_actionable(self) -> None:
        self.assertEqual(classify_cursor_failure("CURSOR_API_KEY is required"), "missing_api_key")
        self.assertEqual(classify_cursor_failure("model invalid"), "model_unavailable")
        self.assertEqual(classify_cursor_failure("operation timed out"), "timeout")

    def test_cursor_adapter_parses_sdk_result_into_artifacts(self) -> None:
        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        sdk_result = {
            "status": "finished",
            "result": json.dumps(
                {
                    "artifacts": [
                        {
                            "type": "finding",
                            "claim": "Streaming filters probable starters before waiver competition.",
                            "evidence": ["dugout/services/bot_transactions.py:123"],
                            "confidence": 0.87,
                        },
                        {
                            "type": "risk",
                            "risk": "Bots may skip valid pitcher stream candidates.",
                            "mitigation": "Add a regression test for probable starters outside the FA pool.",
                            "evidence": ["tests/test_bot_waivers.py"],
                            "confidence": 0.82,
                        },
                    ]
                }
            ),
        }
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps(sdk_result),
            stderr="",
        )

        with patch("puppetmaster.adapters.subprocess.run", return_value=completed) as run:
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        cursor_input = json.loads(run.call_args.kwargs["env"]["PUPPETMASTER_CURSOR_INPUT"])
        artifact_types = [artifact.type for artifact in artifacts]
        self.assertIn(ArtifactType.VERIFICATION, artifact_types)
        self.assertIn(ArtifactType.FINDING, artifact_types)
        self.assertIn(ArtifactType.RISK, artifact_types)
        self.assertEqual(artifacts[0].payload["result"], "passed")
        self.assertIn("Puppetmaster artifact contract", cursor_input["prompt"])

    def test_codegraph_helper_returns_none_when_cli_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch("puppetmaster.codegraph.shutil.which", return_value=None):
                self.assertFalse(codegraph_ready(tmp))
                self.assertIsNone(codegraph_context("map auth", tmp))

    def test_codegraph_helper_requires_initialized_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch("puppetmaster.codegraph.shutil.which", return_value="/usr/local/bin/codegraph"):
                self.assertFalse(codegraph_initialized(tmp))
                self.assertFalse(codegraph_ready(tmp))
                self.assertIsNone(codegraph_context("map auth", tmp))

    def test_codegraph_helper_returns_context_when_command_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            completed = subprocess.CompletedProcess(
                args=["codegraph"],
                returncode=0,
                stdout="### Entry points\n- streaming.py:42\n",
                stderr="",
            )
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                context = codegraph_context("map auth", tmp)

            command = run.call_args.args[0]
            self.assertIn("streaming.py:42", context)
            self.assertEqual(command[:2], ["codegraph", "context"])

    def test_enrich_prompt_returns_unchanged_when_disabled(self) -> None:
        prompt, used = enrich_prompt_with_codegraph(
            "Inspect repo",
            task_description="map auth",
            cwd="/tmp",
            disabled=True,
        )

        self.assertEqual(prompt, "Inspect repo")
        self.assertFalse(used)

    def test_run_codegraph_cli_reports_missing_cli(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch("puppetmaster.codegraph.shutil.which", return_value=None):
                payload = run_codegraph_cli(["status"], tmp)

            self.assertFalse(payload["ok"])
            self.assertIn("codegraph CLI not on PATH", payload["error"])

    def test_run_codegraph_cli_reports_uninitialized_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ):
                payload = run_codegraph_cli(["query", "Foo"], tmp)

            self.assertFalse(payload["ok"])
            self.assertIn("not initialized", payload["error"])

    def test_run_codegraph_cli_returns_structured_success(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout='{"results": []}',
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                payload = run_codegraph_cli(["query", "Foo", "--json"], tmp)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["returncode"], 0)
            self.assertEqual(payload["stdout"], '{"results": []}')
            self.assertEqual(run.call_args.args[0][:3], ["codegraph", "query", "Foo"])

    def test_codegraph_query_builds_expected_args(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout="[]",
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                codegraph_query("UserService", tmp, kind="class", limit=5)

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["codegraph", "query"])
            self.assertIn("UserService", command)
            self.assertIn("--kind", command)
            self.assertIn("class", command)
            self.assertIn("--limit", command)
            self.assertIn("5", command)
            self.assertIn("--json", command)

    def test_codegraph_query_rejects_blank_search(self) -> None:
        payload = codegraph_query("   ", "/tmp")
        self.assertFalse(payload["ok"])
        self.assertIn("search term is required", payload["error"])

    def test_codegraph_affected_requires_files(self) -> None:
        payload = codegraph_affected([], "/tmp")
        self.assertFalse(payload["ok"])
        self.assertIn("at least one changed file path", payload["error"])

    def test_codegraph_affected_passes_files_and_options(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout='{"affected": ["tests/test_auth.py"]}',
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                payload = codegraph_affected(
                    ["src/auth.py", "src/api.py"],
                    tmp,
                    depth=4,
                    filter_pattern="tests/**/*.py",
                )

            command = run.call_args.args[0]
            self.assertTrue(payload["ok"])
            self.assertIn("affected", command)
            self.assertIn("src/auth.py", command)
            self.assertIn("src/api.py", command)
            self.assertIn("--depth", command)
            self.assertIn("4", command)
            self.assertIn("--filter", command)
            self.assertIn("tests/**/*.py", command)

    def test_codegraph_files_listing_serializes_options(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout="[]",
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                codegraph_files_listing(
                    tmp,
                    path="src",
                    fmt="tree",
                    filter_pattern="*.ts",
                    max_depth=3,
                )

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["codegraph", "files"])
            self.assertIn("src", command)
            self.assertIn("--format", command)
            self.assertIn("tree", command)
            self.assertIn("--max-depth", command)
            self.assertIn("3", command)
            self.assertIn("--json", command)

    def test_mcp_exposes_codegraph_proxy_tools(self) -> None:
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = {tool["name"] for tool in response["result"]["tools"]}

        self.assertIn("puppetmaster_codegraph_search", tool_names)
        self.assertIn("puppetmaster_codegraph_context", tool_names)
        self.assertIn("puppetmaster_codegraph_affected", tool_names)
        self.assertIn("puppetmaster_codegraph_files", tool_names)
        self.assertIn("puppetmaster_codegraph_status", tool_names)
        self.assertIn("puppetmaster_codegraph_init", tool_names)

    def test_mcp_codegraph_search_returns_payload_when_cli_succeeds(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout='{"results": [{"name": "Foo"}]}',
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ):
                result = call_tool(
                    "puppetmaster_codegraph_search",
                    {"cwd": tmp, "query": "Foo"},
                )

        self.assertFalse(result["isError"])
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["ok"])
        self.assertIn("Foo", body["stdout"])

    def test_mcp_codegraph_search_reports_missing_cli_as_error(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch("puppetmaster.codegraph.shutil.which", return_value=None):
                result = call_tool(
                    "puppetmaster_codegraph_search",
                    {"cwd": tmp, "query": "Foo"},
                )

        self.assertTrue(result["isError"])
        body = json.loads(result["content"][0]["text"])
        self.assertFalse(body["ok"])
        self.assertIn("codegraph CLI not on PATH", body["error"])

    def test_mcp_codegraph_affected_requires_files(self) -> None:
        with TemporaryDirectory() as tmp:
            result = call_tool(
                "puppetmaster_codegraph_affected",
                {"cwd": tmp, "files": []},
            )

        self.assertTrue(result["isError"])
        body = json.loads(result["content"][0]["text"])
        self.assertIn("non-empty array", body["error"])

    def test_mcp_codegraph_status_does_not_require_init(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout="Backend: native\nNodes: 0",
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ) as run:
                result = call_tool(
                    "puppetmaster_codegraph_status",
                    {"cwd": tmp},
                )

        self.assertFalse(result["isError"])
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["ok"])
        self.assertIn("Backend: native", body["stdout"])
        self.assertEqual(run.call_args.args[0][:2], ["codegraph", "status"])

    def test_bench_load_prompt_reads_file_when_arg_starts_with_at(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text("the goal", encoding="utf-8")

            self.assertEqual(load_prompt(f"@{path}"), "the goal")
            self.assertEqual(load_prompt("inline prompt"), "inline prompt")

    def test_bench_measure_enrichment_reports_zero_when_codegraph_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch("puppetmaster.codegraph.shutil.which", return_value=None):
                measurement = measure_enrichment("explore auth", tmp)

            self.assertEqual(measurement.raw_prompt_chars, len("explore auth"))
            self.assertEqual(measurement.injected_context_chars, 0)
            self.assertFalse(measurement.codegraph_available)
            self.assertEqual(measurement.injection_ratio, 0.0)

    def test_bench_measure_enrichment_captures_injected_context(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codegraph"],
            returncode=0,
            stdout="### Entry points\n- streaming.py:42\n" + "x" * 200,
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=completed,
            ):
                measurement = measure_enrichment("map streaming", tmp)

            self.assertGreater(measurement.injected_context_chars, 0)
            self.assertTrue(measurement.codegraph_available)
            self.assertTrue(measurement.codegraph_initialized)
            self.assertGreater(measurement.injection_ratio, 0.0)

    def test_three_way_scan_repo_picks_up_python_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "src" / "lib.py").write_text("x = 1\n" * 50, encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "ignored.py").write_text("x", encoding="utf-8")

            facts = scan_repo(root)

            self.assertEqual(facts.file_count, 2)
            self.assertGreater(facts.source_bytes, 0)
            self.assertIn("python", facts.languages)

    def test_three_way_load_artifact_facts_falls_back_to_defaults_when_missing(
        self,
    ) -> None:
        facts = load_artifact_facts(None)
        self.assertEqual(facts.sample_count, 0)
        self.assertGreater(facts.avg_artifact_bytes, 0)
        self.assertGreater(facts.avg_worker_stdout_bytes, 0)

    def test_three_way_load_artifact_facts_reads_real_sqlite_store(self) -> None:
        with TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            store = SQLiteSwarmStore(state)
            result = Orchestrator(store).run(
                "exercise the bench loader",
                roles=["explore"],
                worker_mode="inline",
            )
            self.assertTrue(result.summary_path.exists())

            facts = load_artifact_facts(state)
            self.assertGreater(facts.sample_count, 0)
            self.assertGreater(facts.avg_artifact_bytes, 0)
            self.assertIn("state.sqlite3", facts.source)

    def test_three_way_model_costs_assigns_zero_resume_to_config_c(self) -> None:
        repo = RepoFacts(file_count=10, source_bytes=100_000, languages={"python": 100_000})
        artifacts = ArtifactFacts(
            source="test",
            sample_count=10,
            avg_artifact_bytes=800,
            avg_worker_stdout_bytes=2_500,
            avg_artifacts_per_worker=3.0,
        )
        configs = model_costs(
            repo=repo, artifacts=artifacts, workers=4, codegraph_context_bytes=4_000
        )
        self.assertEqual([c.label for c in configs[:3]], [
            "A. Agent only",
            "B. CodeGraph alone",
            "C. Puppetmaster + CodeGraph",
        ])
        a_cfg, b_cfg, c_cfg = configs
        self.assertEqual(c_cfg.resume_bytes, 0)
        self.assertEqual(a_cfg.resume_bytes, a_cfg.fresh_task_bytes)
        self.assertEqual(b_cfg.resume_bytes, b_cfg.fresh_task_bytes)

    def test_three_way_session_cost_amortizes_for_puppetmaster(self) -> None:
        repo = RepoFacts(file_count=10, source_bytes=100_000, languages={"python": 100_000})
        artifacts = ArtifactFacts(
            source="test",
            sample_count=10,
            avg_artifact_bytes=800,
            avg_worker_stdout_bytes=2_500,
            avg_artifacts_per_worker=3.0,
        )
        a_cfg, b_cfg, c_cfg = model_costs(
            repo=repo, artifacts=artifacts, workers=4, codegraph_context_bytes=4_000
        )

        for follow_ups in (0, 1):
            self.assertGreaterEqual(b_cfg.session_bytes(follow_ups), 0)

        large_k = 25
        c_at_k = c_cfg.session_bytes(large_k)
        b_at_k = b_cfg.session_bytes(large_k)
        a_at_k = a_cfg.session_bytes(large_k)

        self.assertEqual(c_at_k, c_cfg.fresh_task_bytes)
        self.assertLess(c_at_k, b_at_k)
        self.assertLess(c_at_k, a_at_k)

    def test_three_way_render_markdown_includes_session_table(self) -> None:
        repo = RepoFacts(file_count=10, source_bytes=100_000, languages={"python": 100_000})
        artifacts = ArtifactFacts(
            source="test",
            sample_count=10,
            avg_artifact_bytes=800,
            avg_worker_stdout_bytes=2_500,
            avg_artifacts_per_worker=3.0,
        )
        configs = model_costs(
            repo=repo, artifacts=artifacts, workers=4, codegraph_context_bytes=4_000
        )
        report = {
            "ran_at": "2026-05-13T17:00:00+00:00",
            "command": "python -m bench.three_way --cwd .",
            "repo_path": "/tmp/repo",
            "workers": 4,
            "discovery_scan_ratio": 0.10,
            "repo": {
                "file_count": repo.file_count,
                "source_bytes": repo.source_bytes,
                "languages": repo.languages,
            },
            "artifacts": {
                "source": artifacts.source,
                "sample_count": artifacts.sample_count,
                "avg_artifact_bytes": artifacts.avg_artifact_bytes,
                "avg_worker_stdout_bytes": artifacts.avg_worker_stdout_bytes,
                "avg_artifacts_per_worker": artifacts.avg_artifacts_per_worker,
            },
            "codegraph_available": True,
            "codegraph_initialized": True,
            "codegraph_context_bytes": 4_000,
            "codegraph_query_seconds": 0.15,
            "configs": [
                {
                    "label": c.label,
                    "fresh_task_bytes": c.fresh_task_bytes,
                    "resume_bytes": c.resume_bytes,
                }
                for c in configs
            ],
        }
        markdown = render_three_way_markdown(report, price_per_million=3.0)

        self.assertIn("Three-way cost comparison", markdown)
        self.assertIn("Fresh task cost", markdown)
        self.assertIn("Session cost", markdown)
        self.assertIn("durable resume", markdown.lower())
        self.assertIn("K=25", markdown)

    def test_bench_render_markdown_includes_enrichment_section(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch("puppetmaster.codegraph.shutil.which", return_value=None):
                measurement = measure_enrichment("a tiny prompt", tmp)
            report = build_report(
                cwd=tmp,
                prompt="a tiny prompt",
                model="default",
                command="python -m bench.codegraph_ab --dry-run",
                enrichment=measurement,
            )
            markdown = render_markdown(report)

        self.assertIn("CodeGraph prompt enrichment", markdown)
        self.assertIn("injection ratio", markdown)
        self.assertIn("python -m bench.codegraph_ab --dry-run", markdown)
        self.assertNotIn("Live Cursor SDK A/B", markdown)

    def test_mcp_codegraph_init_dispatches_indexer_in_background(self) -> None:
        """`index=true` runs init synchronously and forks the indexer
        instead of blocking the MCP transport on `codegraph init --index`.
        """
        completed = subprocess.CompletedProcess(
            args=["codegraph", "init"],
            returncode=0,
            stdout="initialized",
            stderr="",
        )
        fake_proc = _ContextManagerFakeProcess(pid=99999)

        with TemporaryDirectory() as tmp, TemporaryDirectory() as lock_dir:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = lock_dir
            try:
                with patch(
                    "puppetmaster.codegraph.shutil.which",
                    return_value="/usr/local/bin/codegraph",
                ), patch(
                    "puppetmaster.codegraph.subprocess.run",
                    return_value=completed,
                ) as run, patch(
                    "puppetmaster.mcp_server.subprocess.Popen",
                    return_value=fake_proc,
                ) as popen:
                    result = call_tool(
                        "puppetmaster_codegraph_init",
                        {"cwd": tmp, "index": True},
                    )
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        # The synchronous init step should NOT have used --index. Other
        # subprocess.run calls (e.g. `git rev-parse` from state resolution)
        # also flow through this patch, so find the codegraph call by name.
        codegraph_calls = [
            call.args[0]
            for call in run.call_args_list
            if call.args and isinstance(call.args[0], list)
            and call.args[0] and call.args[0][0] == "codegraph"
        ]
        self.assertEqual(len(codegraph_calls), 1)
        self.assertEqual(codegraph_calls[0][:2], ["codegraph", "init"])
        self.assertNotIn("--index", codegraph_calls[0])
        # The async indexer should have been spawned via Popen via our launcher.
        indexer_launches = _find_indexer_launches(popen)
        self.assertEqual(len(indexer_launches), 1)
        # The response should expose run metadata for polling.
        self.assertIn("index_run", payload)
        self.assertIn("run_id", payload["index_run"])
        self.assertIn("stdout_path", payload["index_run"])

    def test_mcp_codegraph_index_returns_immediately_with_run_id(self) -> None:
        """`puppetmaster_codegraph_index` is the dedicated background-only tool."""
        fake_proc = _ContextManagerFakeProcess(pid=12345)

        with TemporaryDirectory() as tmp, TemporaryDirectory() as lock_dir:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = lock_dir
            try:
                with patch(
                    "puppetmaster.codegraph.shutil.which",
                    return_value="/usr/local/bin/codegraph",
                ), patch(
                    "puppetmaster.mcp_server.subprocess.Popen",
                    return_value=fake_proc,
                ) as popen:
                    result = call_tool("puppetmaster_codegraph_index", {"cwd": tmp})
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["pid"], 12345)
        self.assertIn("run_id", payload)
        indexer_launches = _find_indexer_launches(popen)
        self.assertEqual(len(indexer_launches), 1)

    def test_mcp_codegraph_index_fails_fast_when_lock_is_held(self) -> None:
        """A second indexer call against the SAME repo sees `lock busy` immediately."""
        from puppetmaster.codegraph import acquire_codegraph_lock

        with TemporaryDirectory() as tmp, TemporaryDirectory() as lock_dir:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = lock_dir
            try:
                # Hold the per-repo lock for `tmp` so the MCP call
                # (which keys its lock on the same repo) sees it busy.
                held = acquire_codegraph_lock(repo_root=tmp)
                try:
                    with patch(
                        "puppetmaster.codegraph.shutil.which",
                        return_value="/usr/local/bin/codegraph",
                    ):
                        result = call_tool(
                            "puppetmaster_codegraph_index", {"cwd": tmp}
                        )
                finally:
                    held.release()
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

        self.assertTrue(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertFalse(payload["ok"])
        self.assertIn("Another CodeGraph indexer", payload["error"])
        self.assertEqual(payload["holder_pid"], os.getpid())

    def test_mcp_codegraph_status_surfaces_native_sqlite_breakage(self) -> None:
        broken = subprocess.CompletedProcess(
            args=["codegraph", "status"],
            returncode=0,
            stdout=(
                "CodeGraph index ok.\n"
                "Warning: better-sqlite3 native module failed to load "
                "(NODE_MODULE_VERSION mismatch); falling back to WASM driver."
            ),
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch(
                "puppetmaster.codegraph.shutil.which",
                return_value="/usr/local/bin/codegraph",
            ), patch(
                "puppetmaster.codegraph.subprocess.run",
                return_value=broken,
            ):
                result = call_tool("puppetmaster_codegraph_status", {"cwd": tmp})

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload.get("native_sqlite_broken"))
        # New hint points users at the dedicated repair command and surfaces the
        # Cursor-vs-shell Node ABI trap so they don't waste time on the wrong fix.
        self.assertIn("puppetmaster repair-codegraph", payload["hint"])
        self.assertIn("Cursor", payload["hint"])

    def test_codegraph_native_sqlite_broken_detects_known_markers(self) -> None:
        from puppetmaster.codegraph import codegraph_native_sqlite_broken

        self.assertTrue(codegraph_native_sqlite_broken("backend: wasm"))
        self.assertTrue(
            codegraph_native_sqlite_broken(
                "better-sqlite3 was compiled against a different Node ABI"
            )
        )
        self.assertFalse(codegraph_native_sqlite_broken("Backend: native; nodes: 12345"))
        self.assertFalse(codegraph_native_sqlite_broken(""))

    def test_repair_codegraph_finds_cursor_node_from_known_path(self) -> None:
        """find_cursor_node walks the per-platform candidate list."""
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            fake_node = Path(tmp) / "node"
            fake_node.write_text("#!/bin/sh\necho v22.22.0\n", encoding="utf-8")
            fake_node.chmod(0o755)
            # Override the macOS candidate list so the test is hermetic.
            with patch.object(
                codegraph_repair,
                "_CURSOR_NODE_CANDIDATES_MAC",
                (str(fake_node),),
            ), patch.object(codegraph_repair.sys, "platform", "darwin"):
                resolved = codegraph_repair.find_cursor_node()
            self.assertIsNotNone(resolved)
            self.assertEqual(str(resolved), str(fake_node))

    def test_repair_codegraph_explicit_override_wins(self) -> None:
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            fake = Path(tmp) / "node"
            fake.write_text("ok", encoding="utf-8")
            self.assertEqual(
                codegraph_repair.find_cursor_node(str(fake)),
                fake,
            )
            self.assertIsNone(
                codegraph_repair.find_cursor_node(str(Path(tmp) / "missing"))
            )

    def test_repair_codegraph_returns_failure_without_cursor_node(self) -> None:
        """Without a discoverable Cursor Node we surface a clear next-step list."""
        from puppetmaster import codegraph_repair

        with patch.object(codegraph_repair, "find_cursor_node", return_value=None):
            result = codegraph_repair.repair_codegraph_sqlite(verify=False)

        self.assertFalse(result.ok)
        self.assertIn("Cursor", result.message)
        self.assertTrue(result.next_steps)

    def test_repair_codegraph_runs_npm_rebuild_with_cursor_node_in_path(self) -> None:
        """Happy path: rebuild is invoked with Cursor's Node ahead of $PATH."""
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            cursor_node = Path(tmp) / "node"
            cursor_node.write_text("#!/bin/sh\necho v22.22.0\n", encoding="utf-8")
            cursor_node.chmod(0o755)
            install_dir = Path(tmp) / "codegraph"
            install_dir.mkdir()
            (install_dir / "dist").mkdir()
            (install_dir / "dist" / "bin").mkdir()
            (install_dir / "dist" / "bin" / "codegraph.js").write_text(
                "// stub", encoding="utf-8"
            )

            recorded: dict = {}

            def fake_run(cmd, **kwargs):  # noqa: ANN001
                recorded.setdefault("cmds", []).append((cmd, kwargs))
                if cmd[0].endswith("node") and len(cmd) > 1 and cmd[1] == "--version":
                    return subprocess.CompletedProcess(cmd, 0, "v22.22.0\n", "")
                if cmd[0].endswith("node") and len(cmd) >= 3 and cmd[2] == "status":
                    return subprocess.CompletedProcess(
                        cmd, 0, "Backend: native\nNodes: 1234\n", ""
                    )
                if Path(cmd[0]).name.startswith("npm"):
                    return subprocess.CompletedProcess(cmd, 0, "rebuilt!\n", "")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with patch.object(
                codegraph_repair.shutil,
                "which",
                return_value="/usr/local/bin/npm",
            ), patch.object(codegraph_repair.subprocess, "run", side_effect=fake_run):
                result = codegraph_repair.repair_codegraph_sqlite(
                    cursor_node=str(cursor_node),
                    codegraph_install=str(install_dir),
                    verify=True,
                    verify_cwd=tmp,
                )

        self.assertTrue(result.ok, msg=result.message)
        self.assertEqual(result.verify_backend, "native")
        self.assertEqual(result.cursor_node_version, "v22.22.0")
        npm_calls = [c for c, _ in recorded["cmds"] if c[0].endswith("npm")]
        self.assertEqual(len(npm_calls), 1)
        self.assertEqual(npm_calls[0][1:], ["rebuild", "better-sqlite3"])
        # Cursor's Node directory must be ahead of inherited PATH so the rebuild
        # picks up the right runtime, not whichever Node is on the user's shell.
        rebuild_kwargs = next(
            kw for c, kw in recorded["cmds"] if c[0].endswith("npm")
        )
        env_path = rebuild_kwargs["env"]["PATH"]
        self.assertTrue(env_path.startswith(str(cursor_node.parent)))

    def test_repair_codegraph_reports_rebuild_failure(self) -> None:
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            cursor_node = Path(tmp) / "node"
            cursor_node.write_text("ok", encoding="utf-8")
            install_dir = Path(tmp) / "codegraph"
            install_dir.mkdir()

            def fake_run(cmd, **kwargs):  # noqa: ANN001
                if cmd[0].endswith("node") and len(cmd) > 1 and cmd[1] == "--version":
                    return subprocess.CompletedProcess(cmd, 0, "v22.22.0\n", "")
                return subprocess.CompletedProcess(
                    cmd, 1, "", "node-gyp: command not found\n"
                )

            with patch.object(
                codegraph_repair.shutil,
                "which",
                return_value="/usr/local/bin/npm",
            ), patch.object(codegraph_repair.subprocess, "run", side_effect=fake_run):
                result = codegraph_repair.repair_codegraph_sqlite(
                    cursor_node=str(cursor_node),
                    codegraph_install=str(install_dir),
                    verify=False,
                )

        self.assertFalse(result.ok)
        self.assertIn("non-zero", result.message)
        self.assertIn("node-gyp", result.rebuild_stderr)

    def test_mcp_repair_codegraph_invokes_repair_module(self) -> None:
        """The MCP tool wires straight into repair_codegraph_sqlite."""
        from puppetmaster.codegraph_repair import RepairResult

        sentinel = RepairResult(
            ok=True,
            message="rebuilt",
            cursor_node_path="/Applications/Cursor.app/.../node",
            cursor_node_version="v22.22.0",
            codegraph_install_path="/usr/local/lib/node_modules/@colbymchenry/codegraph",
            rebuild_stdout="rebuilt!\n",
            rebuild_stderr="",
            verify_backend="native",
            next_steps=["Restart MCP."],
        )
        with TemporaryDirectory() as tmp, patch(
            "puppetmaster.mcp_server.repair_codegraph_sqlite",
            return_value=sentinel,
        ) as mcp_repair:
            result = call_tool(
                "puppetmaster_repair_codegraph",
                {"cwd": tmp, "verify": False},
            )

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["verify_backend"], "native")
        self.assertEqual(payload["command"], "npm rebuild better-sqlite3")
        self.assertEqual(mcp_repair.call_count, 1)
        kwargs = mcp_repair.call_args.kwargs
        # The MCP tool defaults verify_cwd to the cwd argument so verification
        # runs against the user's actual workspace, not the MCP server's cwd.
        self.assertTrue(kwargs["verify_cwd"])
        self.assertFalse(kwargs["verify"])

    def test_cli_repair_codegraph_returns_zero_on_success(self) -> None:
        """`python -m puppetmaster repair-codegraph` exits 0 when the rebuild works."""
        from puppetmaster import cli as cli_module
        from puppetmaster.codegraph_repair import RepairResult

        sentinel = RepairResult(
            ok=True,
            message="rebuilt",
            cursor_node_path="/Applications/Cursor.app/.../node",
            cursor_node_version="v22.22.0",
            codegraph_install_path="/usr/local/lib/node_modules/@colbymchenry/codegraph",
            rebuild_stdout="",
            rebuild_stderr="",
            verify_backend="native",
            next_steps=["Restart Puppetmaster MCP."],
        )
        with patch.object(
            cli_module,
            "repair_codegraph_sqlite",
            return_value=sentinel,
        ) as repair:
            rc = cli_module.main(["repair-codegraph", "--no-verify"])
        self.assertEqual(rc, 0)
        self.assertEqual(repair.call_count, 1)

    def test_mcp_registry_round_trip(self) -> None:
        """register -> list -> heartbeat -> deregister flows correctly."""
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                path = mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/tmp/test-workspace",
                    version="0.5.2-test",
                )
                self.assertTrue(path.exists())

                entries = mcp_registry.list_entries()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].pid, os.getpid())
                self.assertEqual(entries[0].workspace, "/tmp/test-workspace")
                self.assertTrue(entries[0].is_alive())
                self.assertFalse(entries[0].is_stale())

                before = entries[0].last_heartbeat
                time.sleep(0.01)
                self.assertTrue(mcp_registry.heartbeat(path))
                refreshed = mcp_registry.list_entries()[0]
                self.assertGreater(refreshed.last_heartbeat, before)

                mcp_registry.deregister(path)
                self.assertFalse(path.exists())
                self.assertEqual(mcp_registry.list_entries(), [])
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_registry_prune_dead_removes_dead_pids(self) -> None:
        """Tracking files whose PIDs are no longer alive get cleaned up."""
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                # Live registration (our own PID).
                live_path = mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/live",
                )
                # Synthetic dead registration — PID 1 is init/launchd and
                # *is* alive, so pick an obviously-impossible PID. 2**31 - 1
                # is well outside any kernel's PID range on Linux/macOS.
                dead_path = Path(tmp) / "2147483647.json"
                dead_path.write_text(
                    json.dumps(
                        {
                            "pid": 2147483647,
                            "workspace": "/dead",
                            "started_at": time.time() - 3600,
                            "last_heartbeat": time.time() - 3600,
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )

                cleaned = mcp_registry.prune_dead()
                self.assertEqual(len(cleaned), 1)
                self.assertEqual(cleaned[0].pid, 2147483647)
                self.assertFalse(dead_path.exists())
                self.assertTrue(live_path.exists())
                mcp_registry.deregister(live_path)
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_registry_stale_detection(self) -> None:
        """An alive entry with an old heartbeat is reported as stale."""
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                path = Path(tmp) / f"{os.getpid()}.json"
                path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "workspace": "/stale",
                            "started_at": time.time() - 3600,
                            "last_heartbeat": time.time() - 3600,
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )

                entries = mcp_registry.list_entries()
                self.assertEqual(len(entries), 1)
                self.assertTrue(entries[0].is_alive())
                self.assertTrue(entries[0].is_stale())

                summary = mcp_registry.summarize(entries)
                self.assertEqual(summary["count"], 1)
                self.assertEqual(summary["alive"], 1)
                self.assertEqual(summary["stale"], 1)
                self.assertEqual(summary["dead"], 0)
                mcp_registry.deregister(path)
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_registry_kill_stale_never_signals_self(self) -> None:
        """kill_stale must refuse to SIGTERM the current process."""
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                path = Path(tmp) / f"{os.getpid()}.json"
                path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "workspace": "/self",
                            "started_at": time.time() - 3600,
                            "last_heartbeat": time.time() - 3600,
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )

                killed = mcp_registry.kill_stale(self_pid=os.getpid())
                self.assertEqual(killed, [])
                self.assertTrue(path.exists())  # we did not kill ourselves
                mcp_registry.deregister(path)
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_status_tool_reports_self(self) -> None:
        """puppetmaster_mcp_status returns the running server in its snapshot."""
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/test-workspace",
                )
                result = call_tool("puppetmaster_mcp_status", {})
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["self_pid"], os.getpid())
        self.assertGreaterEqual(payload["count"], 1)
        pids = [row["pid"] for row in payload["servers"]]
        self.assertIn(os.getpid(), pids)

    def test_mcp_cleanup_tool_prunes_dead_files(self) -> None:
        """puppetmaster_mcp_cleanup prunes dead tracking files end-to-end."""
        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                dead_path = Path(tmp) / "2147483647.json"
                dead_path.write_text(
                    json.dumps(
                        {
                            "pid": 2147483647,
                            "workspace": "/dead",
                            "started_at": time.time() - 3600,
                            "last_heartbeat": time.time() - 3600,
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )

                result = call_tool("puppetmaster_mcp_cleanup", {})
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["pruned"]), 1)
        self.assertEqual(payload["pruned"][0]["pid"], 2147483647)
        self.assertEqual(payload["killed"], [])
        self.assertFalse(dead_path.exists())

    def test_cli_mcp_list_outputs_json_when_requested(self) -> None:
        """`python -m puppetmaster mcp list --json` prints a parseable snapshot."""
        import io
        from puppetmaster import cli as cli_module
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                mcp_registry.register(pid=os.getpid(), workspace="/listed")
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    rc = cli_module.main(["mcp", "list", "--json"])
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        self.assertEqual(rc, 0)
        snapshot = json.loads(buf.getvalue())
        self.assertGreaterEqual(snapshot["count"], 1)

    def test_tool_call_keepalive_skips_fast_handlers(self) -> None:
        """Handlers that finish before the start_after grace period emit no notifications.

        This is the desired default: short reads/writes pay zero
        protocol-level cost and only long-running calls keep the pipe warm.
        """
        from puppetmaster.mcp_server import _ToolCallKeepalive

        emitted: list[dict] = []

        def fake_emit(payload):
            emitted.append(payload)
            return True

        keepalive = _ToolCallKeepalive(
            tool_name="puppetmaster_fast",
            request_id=42,
            start_after_seconds=1.0,
            interval_seconds=1.0,
            emitter=fake_emit,
        )
        keepalive.start()
        time.sleep(0.05)
        keepalive.stop(wait=True)
        self.assertEqual(emitted, [])
        self.assertEqual(keepalive.emitted_count, 0)

    def test_tool_call_keepalive_emits_for_slow_handlers(self) -> None:
        """Tools that exceed start_after produce at least one well-formed notification."""
        from puppetmaster.mcp_server import _ToolCallKeepalive

        emitted: list[dict] = []

        def fake_emit(payload):
            emitted.append(payload)
            return True

        keepalive = _ToolCallKeepalive(
            tool_name="puppetmaster_codegraph_index",
            request_id="req-7",
            start_after_seconds=0.05,
            interval_seconds=0.05,
            emitter=fake_emit,
        )
        keepalive.start()
        time.sleep(0.2)
        keepalive.stop(wait=True)
        self.assertGreaterEqual(len(emitted), 1)
        first = emitted[0]
        self.assertEqual(first["jsonrpc"], "2.0")
        self.assertEqual(first["method"], "notifications/message")
        # Notifications must NEVER carry an `id` field — that's how the
        # client distinguishes them from responses.
        self.assertNotIn("id", first)
        params = first["params"]
        self.assertEqual(params["level"], "debug")
        self.assertEqual(params["logger"], "puppetmaster")
        data = params["data"]
        self.assertEqual(data["kind"], "tool_call_progress")
        self.assertEqual(data["tool"], "puppetmaster_codegraph_index")
        self.assertEqual(data["request_id"], "req-7")
        self.assertGreaterEqual(data["elapsed_seconds"], 0.05)

    def test_tool_call_keepalive_stops_when_pipe_is_broken(self) -> None:
        """A failing emitter (BrokenPipeError surrogate) shuts the loop down."""
        from puppetmaster.mcp_server import _ToolCallKeepalive

        call_count = {"value": 0}

        def failing_emit(payload):
            call_count["value"] += 1
            return False  # mimics BrokenPipeError swallowed by _emit_notification

        keepalive = _ToolCallKeepalive(
            tool_name="puppetmaster_slow",
            request_id=99,
            start_after_seconds=0.02,
            interval_seconds=0.02,
            emitter=failing_emit,
        )
        keepalive.start()
        time.sleep(0.2)
        keepalive.stop(wait=True)
        # The keepalive should have attempted at most a single emit before
        # bailing out on the broken pipe — we never spin in a tight failure
        # loop that would spam stderr or hold CPU.
        self.assertLessEqual(call_count["value"], 1)

    def test_tool_call_keepalive_disabled_via_env(self) -> None:
        """PUPPETMASTER_MCP_KEEPALIVE_DISABLED short-circuits the wiring in _process_message_safely."""
        from puppetmaster.mcp_server import _keepalive_disabled

        os.environ["PUPPETMASTER_MCP_KEEPALIVE_DISABLED"] = "1"
        try:
            self.assertTrue(_keepalive_disabled())
        finally:
            del os.environ["PUPPETMASTER_MCP_KEEPALIVE_DISABLED"]
        self.assertFalse(_keepalive_disabled())

    def test_process_message_safely_emits_keepalives_under_slow_handler(self) -> None:
        """End-to-end: a slow tools/call yields at least one keepalive
        notification on stdout before the response, proving the
        notification bytes flow through the same lock the response uses.
        """
        import io

        from puppetmaster import mcp_server

        slow_started = threading.Event()
        allow_finish = threading.Event()

        def slow_handler(message):
            slow_started.set()
            # Hold the handler open long enough for the keepalive to fire.
            allow_finish.wait(timeout=2.0)
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {"ok": True},
            }

        fake_stdout = io.StringIO()
        with patch.object(mcp_server, "handle_message", side_effect=slow_handler), patch.object(
            mcp_server, "_DEFAULT_KEEPALIVE_AFTER_SECONDS", 0.05
        ), patch.object(
            mcp_server, "_DEFAULT_KEEPALIVE_INTERVAL_SECONDS", 0.05
        ), patch.object(
            mcp_server.sys, "stdout", fake_stdout
        ):
            worker = threading.Thread(
                target=mcp_server._process_message_safely,
                args=({"method": "tools/call", "id": 1, "params": {"name": "puppetmaster_status"}},),
                daemon=True,
            )
            worker.start()
            self.assertTrue(slow_started.wait(timeout=1.0))
            time.sleep(0.2)
            allow_finish.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())

        frames = [json.loads(line) for line in fake_stdout.getvalue().splitlines() if line.strip()]
        # We should see at least one notification (no id) followed by the
        # response (has id). Order is enforced because both write under
        # _STDOUT_LOCK and the handler emits its frame last.
        notifications = [frame for frame in frames if "id" not in frame]
        responses = [frame for frame in frames if "id" in frame]
        self.assertGreaterEqual(len(notifications), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(notifications[0]["method"], "notifications/message")
        self.assertEqual(responses[0]["id"], 1)

    def test_resolve_codegraph_invocation_prefers_cursor_node(self) -> None:
        """When Cursor Node + codegraph.js are both discoverable, prefer that pair."""
        from puppetmaster import codegraph as codegraph_mod
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            node = Path(tmp) / "Cursor.app/Contents/Resources/app/resources/helpers/node"
            node.parent.mkdir(parents=True)
            node.write_text("ok", encoding="utf-8")
            install = Path(tmp) / "codegraph"
            (install / "dist" / "bin").mkdir(parents=True)
            (install / "dist" / "bin" / "codegraph.js").write_text("// stub", encoding="utf-8")

            with patch.object(codegraph_repair, "find_cursor_node", return_value=node), patch.object(
                codegraph_repair, "find_codegraph_install", return_value=install
            ):
                argv = codegraph_mod.resolve_codegraph_invocation()
            self.assertEqual(len(argv), 2)
            self.assertEqual(argv[0], str(node))
            self.assertEqual(argv[1], str(install / "dist" / "bin" / "codegraph.js"))

    def test_resolve_codegraph_invocation_falls_back_to_shim(self) -> None:
        """Without a Cursor install, we fall back to the codegraph shim on PATH."""
        from puppetmaster import codegraph as codegraph_mod
        from puppetmaster import codegraph_repair

        with patch.object(codegraph_repair, "find_cursor_node", return_value=None), patch.object(
            codegraph_repair, "find_codegraph_install", return_value=None
        ):
            argv = codegraph_mod.resolve_codegraph_invocation()
        self.assertEqual(argv, [codegraph_mod.CODEGRAPH_COMMAND])

    def test_resolve_codegraph_invocation_honors_env_override(self) -> None:
        """Explicit env vars short-circuit auto-detection (escape hatch for weird installs)."""
        from puppetmaster import codegraph as codegraph_mod

        with TemporaryDirectory() as tmp:
            node = Path(tmp) / "alt-node"
            js = Path(tmp) / "alt-codegraph.js"
            node.write_text("ok", encoding="utf-8")
            js.write_text("// stub", encoding="utf-8")
            os.environ["PUPPETMASTER_CODEGRAPH_NODE"] = str(node)
            os.environ["PUPPETMASTER_CODEGRAPH_JS"] = str(js)
            try:
                argv = codegraph_mod.resolve_codegraph_invocation()
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_NODE"]
                del os.environ["PUPPETMASTER_CODEGRAPH_JS"]
        self.assertEqual(argv, [str(node), str(js)])

    def test_input_staleness_watcher_triggers_when_idle(self) -> None:
        """No inbound messages for `stale_after_seconds` -> shutdown callback fires."""
        from puppetmaster import mcp_server

        triggered = threading.Event()
        try:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time() - 3600
                mcp_server._ACTIVE_TOOL_CALLS = 0
            watcher = mcp_server._InputStalenessWatcher(
                stale_after_seconds=0.05,
                check_interval_seconds=0.02,
                on_shutdown=triggered.set,
            )
            watcher.start()
            self.assertTrue(triggered.wait(timeout=1.0))
            self.assertTrue(watcher.triggered)
        finally:
            watcher.stop()
            # Reset module state so other tests see a fresh server.
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time()
                mcp_server._ACTIVE_TOOL_CALLS = 0
            mcp_server._SHUTDOWN_REQUESTED.clear()

    def test_input_staleness_watcher_holds_off_during_active_call(self) -> None:
        """Even if input is stale, an in-flight tool call defers shutdown."""
        from puppetmaster import mcp_server

        triggered = threading.Event()
        try:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time() - 3600
                mcp_server._ACTIVE_TOOL_CALLS = 2  # something is in flight
            watcher = mcp_server._InputStalenessWatcher(
                stale_after_seconds=0.05,
                check_interval_seconds=0.02,
                on_shutdown=triggered.set,
            )
            watcher.start()
            self.assertFalse(triggered.wait(timeout=0.3))
            self.assertFalse(watcher.triggered)
        finally:
            watcher.stop()
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time()
                mcp_server._ACTIVE_TOOL_CALLS = 0
            mcp_server._SHUTDOWN_REQUESTED.clear()

    def test_input_staleness_watcher_resets_when_message_arrives(self) -> None:
        """A new inbound message bumps the timestamp; watcher then ignores the staleness."""
        from puppetmaster import mcp_server

        triggered = threading.Event()
        try:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time() - 3600
                mcp_server._ACTIVE_TOOL_CALLS = 0
            watcher = mcp_server._InputStalenessWatcher(
                stale_after_seconds=0.2,
                check_interval_seconds=0.02,
                on_shutdown=triggered.set,
            )
            # Simulate Cursor sending us something, resetting the (otherwise stale)
            # timestamp BEFORE the watcher starts so its first check can never race
            # the mark and observe the stale value. The guarantee under test — a
            # fresh inbound message keeps the watcher from firing — is unchanged.
            mcp_server._mark_inbound_message()
            watcher.start()
            # Heartbeat refresh should keep us alive for at least one check cycle.
            self.assertFalse(triggered.wait(timeout=0.1))
        finally:
            watcher.stop()
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time()
                mcp_server._ACTIVE_TOOL_CALLS = 0
            mcp_server._SHUTDOWN_REQUESTED.clear()

    def test_input_staleness_can_be_disabled_via_env(self) -> None:
        from puppetmaster.mcp_server import _input_staleness_disabled

        os.environ["PUPPETMASTER_MCP_INPUT_STALE_DISABLED"] = "1"
        try:
            self.assertTrue(_input_staleness_disabled())
        finally:
            del os.environ["PUPPETMASTER_MCP_INPUT_STALE_DISABLED"]
        self.assertFalse(_input_staleness_disabled())

    def test_tool_call_counter_tracks_inflight_calls(self) -> None:
        """The dispatcher must increment/decrement the active-call counter."""
        from puppetmaster import mcp_server

        with mcp_server._INPUT_STATE_LOCK:
            mcp_server._ACTIVE_TOOL_CALLS = 0
        try:
            with patch.object(mcp_server, "handle_message", return_value=None):
                mcp_server._process_message_safely(
                    {"method": "tools/call", "id": 1, "params": {"name": "puppetmaster_doctor"}}
                )
            _, active_after = mcp_server._input_state_snapshot()
            self.assertEqual(active_after, 0)
        finally:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._ACTIVE_TOOL_CALLS = 0

    def test_doctor_codegraph_check_runs_under_cursor_node_invocation(self) -> None:
        """When Cursor Node is the runtime, the ok-message says so so users know which Node it verified against."""
        from puppetmaster import diagnostics

        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch.object(diagnostics, "codegraph_available", return_value=True), patch.object(
                diagnostics, "codegraph_initialized", return_value=True
            ), patch.object(
                diagnostics,
                "codegraph_status_command",
                return_value={"stdout": "Backend: native", "stderr": ""},
            ), patch.object(
                diagnostics, "codegraph_native_sqlite_broken", return_value=False
            ), patch.object(
                diagnostics,
                "resolve_codegraph_invocation",
                return_value=[
                    "/Applications/Cursor.app/Contents/Resources/app/resources/helpers/node",
                    "/opt/homebrew/lib/node_modules/@colbymchenry/codegraph/dist/bin/codegraph.js",
                ],
            ):
                check = diagnostics._codegraph_check(Path(tmp))
        self.assertEqual(check.status, "ok")
        self.assertIn("Cursor's bundled Node", check.detail)

    def test_doctor_flags_orphan_mcp_servers(self) -> None:
        """`puppetmaster doctor` warns when dead tracking files exist."""
        from puppetmaster.diagnostics import run_doctor

        with TemporaryDirectory() as tmp_state, TemporaryDirectory() as registry_dir:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = registry_dir
            try:
                dead_path = Path(registry_dir) / "2147483647.json"
                dead_path.write_text(
                    json.dumps(
                        {
                            "pid": 2147483647,
                            "workspace": "/dead",
                            "started_at": time.time() - 3600,
                            "last_heartbeat": time.time() - 3600,
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )
                checks = run_doctor(Path(tmp_state), Path(tmp_state))
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        mcp_check = next(c for c in checks if c.name == "mcp-servers")
        self.assertEqual(mcp_check.status, "warn")
        self.assertIn("puppetmaster mcp cleanup", mcp_check.detail)

    def test_mcp_main_loop_dispatches_messages_concurrently(self) -> None:
        """Slow tool calls must not block fast ones — the original 'Not
        connected' bug came from a single-threaded `for line in sys.stdin`.
        We submit one slow + one fast request via the same dispatcher and
        assert the fast response is written first.
        """
        import io
        import threading

        from puppetmaster import mcp_server

        order: list[str] = []
        order_lock = threading.Lock()
        slow_started = threading.Event()
        fast_arrived = threading.Event()

        original_handle = mcp_server.handle_message

        def _fake_handle(message):  # noqa: ANN001
            method = message.get("method") or ""
            params = (message.get("params") or {}).get("name")
            if params == "slow":
                slow_started.set()
                fast_arrived.wait(timeout=2.0)
                with order_lock:
                    order.append("slow")
                return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"ok": "slow"}}
            if params == "fast":
                with order_lock:
                    order.append("fast")
                fast_arrived.set()
                return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"ok": "fast"}}
            return original_handle(message)

        captured = io.StringIO()
        capture_lock = threading.Lock()

        def _drain(message):
            try:
                response = _fake_handle(message)
            except Exception as exc:  # pragma: no cover - defensive
                response = {"jsonrpc": "2.0", "id": message.get("id"), "error": str(exc)}
            if response is None:
                return
            with capture_lock:
                captured.write(json.dumps(response) + "\n")

        slow_msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "slow"}}
        fast_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "fast"}}

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            slow_future = pool.submit(_drain, slow_msg)
            self.assertTrue(slow_started.wait(timeout=2.0))
            fast_future = pool.submit(_drain, fast_msg)
            fast_future.result(timeout=2.0)
            slow_future.result(timeout=2.0)

        with order_lock:
            self.assertEqual(order, ["fast", "slow"],
                             "fast response must complete while slow handler is still blocked")
        output_lines = [line for line in captured.getvalue().splitlines() if line]
        first_response = json.loads(output_lines[0])
        self.assertEqual(first_response["result"]["ok"], "fast")

    def test_cursor_adapter_injects_codegraph_context_when_available(self) -> None:
        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="map pitcher streaming logic",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "/tmp/codegraph-repo"},
        )
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": ""}),
            stderr="",
        )

        with patch(
            "puppetmaster.adapters.enrich_prompt_with_codegraph",
            return_value=(
                "Inspect repo\n\nShared CodeGraph context for this task:\n```\nstreaming.py:42\n```\n",
                True,
            ),
        ), patch(
            "puppetmaster.adapters.subprocess.run",
            return_value=completed,
        ) as run:
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        cursor_input = json.loads(run.call_args.kwargs["env"]["PUPPETMASTER_CURSOR_INPUT"])
        self.assertIn("Shared CodeGraph context for this task", cursor_input["prompt"])
        self.assertIn("context:codegraph", artifacts[0].evidence)

    def test_cursor_adapter_skips_codegraph_when_unavailable(self) -> None:
        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "/tmp/no-codegraph"},
        )
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": ""}),
            stderr="",
        )

        with patch(
            "puppetmaster.adapters.enrich_prompt_with_codegraph",
            return_value=("Inspect repo", False),
        ), patch(
            "puppetmaster.adapters.subprocess.run",
            return_value=completed,
        ):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        self.assertNotIn("context:codegraph", artifacts[0].evidence)

    def test_cursor_adapter_degrades_empty_success(self) -> None:
        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": ""}),
            stderr="",
        )

        with patch("puppetmaster.adapters.subprocess.run", return_value=completed):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        self.assertEqual(artifacts[0].type, ArtifactType.VERIFICATION)
        self.assertEqual(artifacts[0].payload["result"], "degraded")
        self.assertEqual(artifacts[0].payload["failure"], "empty_or_unstructured_cursor_result")
        self.assertEqual(artifacts[1].type, ArtifactType.RISK)
        self.assertIn("without structured Puppetmaster findings", artifacts[1].payload["risk"])

    def test_capture_subprocess_stdout_inlines_short_text_without_sidecar(self) -> None:
        """No spool when total fits in head+tail; capture dict is still emitted."""
        from puppetmaster.adapters import capture_subprocess_stdout

        task = Task(
            job_id="job_short",
            role="pipeline-mapper",
            instruction="x",
            adapter="cursor",
            payload={},
        )
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": tmp}):
                capture = capture_subprocess_stdout(
                    text="hello world",
                    task=task,
                    sidecar_name="cursor_stdout",
                )
            self.assertFalse(capture["stdout_truncated"])
            self.assertEqual(capture["stdout_total_chars"], 11)
            self.assertEqual(capture["stdout_head_excerpt"], "hello world")
            self.assertEqual(capture["stdout_tail_excerpt"], "")
            self.assertNotIn("stdout_sidecar_path", capture)
            self.assertFalse(
                (Path(tmp) / "jobs" / "job_short" / "tasks").exists()
            )

    def test_capture_subprocess_stdout_spools_full_text_when_truncated(self) -> None:
        """Long stdout: head + tail inline AND full text preserved at sidecar path."""
        from puppetmaster.adapters import capture_subprocess_stdout

        task = Task(
            job_id="job_long",
            role="pipeline-mapper",
            instruction="x",
            adapter="cursor",
            payload={},
        )
        # 30KB > head(1k) + tail(8k); middle would have been silently dropped
        # under the pre-fix adapter behavior.
        long_text = "MIDDLE-MARKER-XYZ".join(["A" * 10000, "B" * 10000, "C" * 10000])
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": tmp}):
                capture = capture_subprocess_stdout(
                    text=long_text,
                    task=task,
                    sidecar_name="cursor_stdout",
                )
            self.assertTrue(capture["stdout_truncated"])
            self.assertEqual(capture["stdout_total_chars"], len(long_text))
            self.assertEqual(len(capture["stdout_head_excerpt"]), 1000)
            self.assertEqual(len(capture["stdout_tail_excerpt"]), 8000)
            sidecar = Path(capture["stdout_sidecar_path"])
            self.assertTrue(sidecar.exists())
            self.assertEqual(
                sidecar,
                Path(tmp) / "jobs" / "job_long" / "tasks" / task.id / "cursor_stdout.log",
            )
            # The whole payload must survive — including the middle bytes
            # that the inline head+tail cannot fit.
            spooled = sidecar.read_text(encoding="utf-8")
            self.assertEqual(spooled, long_text)
            self.assertIn("MIDDLE-MARKER-XYZ", spooled)

    def test_capture_subprocess_stdout_skips_sidecar_without_state_dir(self) -> None:
        """When PUPPETMASTER_STATE_DIR is unset (e.g. direct unit-test invocation),
        sidecar spooling is skipped gracefully but the new truncation markers are
        still emitted so callers can detect the drop.
        """
        from puppetmaster.adapters import capture_subprocess_stdout

        task = Task(
            job_id="job_nostate",
            role="pipeline-mapper",
            instruction="x",
            adapter="cursor",
            payload={},
        )
        # Strip the env var entirely so the helper returns None for state dir.
        env = {k: v for k, v in os.environ.items() if k != "PUPPETMASTER_STATE_DIR"}
        with patch.dict(os.environ, env, clear=True):
            capture = capture_subprocess_stdout(
                text="A" * 20000,
                task=task,
                sidecar_name="cursor_stdout",
            )
        self.assertTrue(capture["stdout_truncated"])
        self.assertIsNone(capture["stdout_sidecar_path"])
        self.assertEqual(capture["stdout_total_chars"], 20000)

    def test_cursor_adapter_emits_stdout_capture_in_verification(self) -> None:
        """Regression: every CursorAdapter verification artifact must carry the
        new stdout_capture metadata so consumers can recover the full output.
        """
        task = Task(
            job_id="job_cap_v",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        long_stdout = json.dumps(
            {
                "status": "finished",
                "result": json.dumps(
                    {
                        "artifacts": [
                            {
                                "type": "finding",
                                "claim": "noise",
                                "evidence": ["x"],
                                "confidence": 0.8,
                            }
                        ]
                    }
                )
                + "\n"
                + ("PADDING " * 2000),  # ~16KB padding
            }
        )
        completed = subprocess.CompletedProcess(
            args=["node"], returncode=0, stdout=long_stdout, stderr=""
        )
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": tmp}):
                with patch(
                    "puppetmaster.adapters.subprocess.run", return_value=completed
                ):
                    artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

            verification = artifacts[0]
            self.assertEqual(verification.type, ArtifactType.VERIFICATION)
            cap = verification.payload.get("stdout_capture")
            self.assertIsNotNone(cap, "verification payload missing stdout_capture")
            self.assertEqual(cap["stdout_total_chars"], len(long_stdout))
            self.assertTrue(cap["stdout_truncated"])
            # When truncated, sidecar should exist on disk with full content
            self.assertIsNotNone(cap["stdout_sidecar_path"])
            self.assertTrue(Path(cap["stdout_sidecar_path"]).exists())

    def test_cursor_adapter_degraded_risk_has_stdout_capture(self) -> None:
        """Regression for the original bug: when Cursor returns no structured
        artifacts and stdout exceeds head+tail, the middle bytes must survive
        via the sidecar referenced from BOTH the verification AND the degraded
        risk artifact.
        """
        task = Task(
            job_id="job_cap_d",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="cursor",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        # Long markdown-y response with NO {"artifacts": []} envelope.
        long_result = (
            "Here is a long markdown answer.\n\n"
            + "MIDDLE-MARKER-SHOULD-SURVIVE\n"
            + ("PADDING line\n" * 2000)
        )
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": long_result}),
            stderr="",
        )
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": tmp}):
                with patch(
                    "puppetmaster.adapters.subprocess.run", return_value=completed
                ):
                    artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

            verification, risk = artifacts[0], artifacts[1]
            self.assertEqual(verification.payload["result"], "degraded")
            self.assertEqual(risk.type, ArtifactType.RISK)

            v_cap = verification.payload.get("stdout_capture")
            r_cap = risk.payload.get("stdout_capture")
            self.assertIsNotNone(v_cap)
            self.assertIsNotNone(r_cap)
            # Both artifacts should reference a sidecar (path may differ if
            # the verification used the raw stdout vs the degraded risk using
            # the parsed result_text). What matters: the middle marker is
            # recoverable from at least one of them.
            recovered = ""
            for cap in (v_cap, r_cap):
                p = cap.get("stdout_sidecar_path")
                if p and Path(p).exists():
                    recovered += Path(p).read_text(encoding="utf-8")
            self.assertIn(
                "MIDDLE-MARKER-SHOULD-SURVIVE",
                recovered,
                "middle of long stdout was silently dropped — the bug is back",
            )

    def test_claude_code_command_uses_full_edit_permission_mode(self) -> None:
        command = build_claude_code_command(
            prompt="Implement the change",
            executable="claude",
            model="sonnet",
            permission_mode="acceptEdits",
            allowed_tools=["Read", "Edit", "Bash"],
        )

        self.assertEqual(command[:2], ["claude", "--print"])
        self.assertIn("--permission-mode", command)
        self.assertIn("acceptEdits", command)
        self.assertIn("--allowedTools", command)
        self.assertIn("Read,Edit,Bash", command)

    def test_claude_code_missing_cli_returns_blocked_artifact(self) -> None:
        task = Task(
            job_id="job",
            role="claude-code",
            instruction="run claude",
            adapter="claude-code",
            payload={"executable": "definitely-not-claude-code"},
        )

        artifact = ClaudeCodeAdapter().run(task, "goal", "worker")[0]

        self.assertEqual(artifact.payload["result"], "blocked")
        self.assertEqual(artifact.payload["failure"], "missing_cli")

    def test_claude_code_failure_classification_is_actionable(self) -> None:
        self.assertEqual(classify_claude_code_failure("please login first"), "not_authenticated")
        self.assertEqual(classify_claude_code_failure("Credit balance is too low"), "billing_or_quota")
        self.assertEqual(classify_claude_code_failure("permission denied"), "permission_denied")
        self.assertEqual(classify_claude_code_failure("model invalid"), "model_unavailable")

    def test_claude_code_adapter_defaults_to_opus_4_8(self) -> None:
        """With no model pinned (and no router stamp), the claude-code adapter
        must default to claude-opus-4-8 rather than the CLI's own default."""
        from puppetmaster.adapters import DEFAULT_CLAUDE_CODE_MODEL

        self.assertEqual(DEFAULT_CLAUDE_CODE_MODEL, "claude-opus-4-8")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
print('{"result":"ok"}')
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)

            captured: dict = {}

            def fake_build(**kwargs):
                captured.update(kwargs)
                return [sys.executable, str(fake_claude)]

            # Default case: no model in payload.
            task = Task(
                job_id="job",
                role="claude-code",
                instruction="implement a change",
                adapter="claude-code",
                payload={"executable": [sys.executable, str(fake_claude)], "cwd": str(repo), "timeout_seconds": 10},
            )
            with patch(
                "puppetmaster.adapters.build_claude_code_command", side_effect=fake_build
            ):
                ClaudeCodeAdapter().run(task, "goal", "worker")
            self.assertEqual(captured["model"], "claude-opus-4-8")

            # Explicit model still wins over the default.
            captured.clear()
            task_pinned = Task(
                job_id="job",
                role="claude-code",
                instruction="implement a change",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                    "model": "claude-haiku-4-5",
                },
            )
            with patch(
                "puppetmaster.adapters.build_claude_code_command", side_effect=fake_build
            ):
                ClaudeCodeAdapter().run(task_pinned, "goal", "worker")
            self.assertEqual(captured["model"], "claude-haiku-4-5")

    def test_claude_code_adapter_injects_codegraph_context_when_available(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
print('{"result":"ok"}')
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)

            task = Task(
                job_id="job",
                role="claude-code",
                instruction="map the auth flow",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                    "prompt": "Inspect the repo and propose a small fix.",
                },
            )

            with patch(
                "puppetmaster.adapters.enrich_prompt_with_codegraph",
                return_value=(
                    "Inspect the repo and propose a small fix.\n\n"
                    "Shared CodeGraph context for this task:\n```\nauth.py:42\n```\n",
                    True,
                ),
            ) as enrich:
                artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")

            self.assertEqual(enrich.call_count, 1)
            self.assertIn("context:codegraph", artifacts[0].evidence)

    def test_claude_code_adapter_skips_codegraph_when_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
print('{"result":"ok"}')
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)

            task = Task(
                job_id="job",
                role="claude-code",
                instruction="ship a tiny change",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                    "prompt": "Make the change.",
                },
            )

            with patch(
                "puppetmaster.adapters.enrich_prompt_with_codegraph",
                return_value=("Make the change.", False),
            ):
                artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")

            self.assertNotIn("context:codegraph", artifacts[0].evidence)

    def test_claude_code_adapter_captures_tracked_git_diff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            target = repo / "sample.txt"
            target.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True, capture_output=True)
            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
from pathlib import Path
Path("sample.txt").write_text("after\\n", encoding="utf-8")
print('{"result":"ok"}')
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)
            task = Task(
                job_id="job",
                role="claude-code",
                instruction="edit the file",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                },
            )

            artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")
            patch_artifacts = [artifact for artifact in artifacts if artifact.type == ArtifactType.PATCH]

            self.assertEqual(artifacts[0].payload["result"], "passed")
            self.assertEqual(patch_artifacts[0].payload["files"], ["sample.txt"])
            self.assertIn("-before", patch_artifacts[0].payload["unified_diff"])
            self.assertIn("+after", patch_artifacts[0].payload["unified_diff"])

    def test_claude_code_blocks_dirty_worktree_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            target = repo / "sample.txt"
            target.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Puppetmaster Tests",
                    "-c",
                    "user.email=tests@example.com",
                    "commit",
                    "-m",
                    "init",
                ],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            target.write_text("dirty\n", encoding="utf-8")
            task = Task(
                job_id="job",
                role="claude-code",
                instruction="edit the file",
                adapter="claude-code",
                payload={
                    "executable": sys.executable,
                    "extra_args": ["-c", "print('should not run')"],
                    "cwd": str(repo),
                },
            )

            artifact = ClaudeCodeAdapter().run(task, "goal", "worker")[0]

            self.assertEqual(artifact.payload["result"], "blocked")
            self.assertEqual(artifact.payload["failure"], "dirty_worktree")

    def test_unconfigured_provider_adapter_returns_blocked_artifact(self) -> None:
        """`UnconfiguredProviderAdapter` is the class operators wire in for
        adapter slots that don't have a concrete implementation yet. It must
        return a structured `result="blocked"` verification artifact so jobs
        referencing the stub fail loudly but cleanly. Previously this test
        covered the `codex` adapter slot; v0.7.0 promoted codex to a real
        adapter, so the test now exercises the stub class directly.
        """
        adapter = UnconfiguredProviderAdapter("future-provider", "Future Provider")
        task = Task(
            id="t-stub",
            job_id="job-stub",
            role="future-review",
            adapter="future-provider",
            instruction="Try to use the unconfigured provider.",
            payload={},
        )
        artifacts = adapter.run(task, "goal", "worker")

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.VERIFICATION)
        self.assertEqual(artifact.payload["adapter"], "future-provider")
        self.assertEqual(artifact.payload["result"], "blocked")
        self.assertIn("provider stub", artifact.payload["message"])

    def test_codex_adapter_missing_cli_returns_blocked(self) -> None:
        """When the `codex` binary isn't on PATH and no executable override
        resolves, the adapter must return a blocked verification with a
        failure code of `missing_cli` rather than crashing or shelling out."""
        task = Task(
            id="t-codex-1",
            job_id="job-codex-1",
            role="codex-review",
            adapter="codex",
            instruction="Review the repo.",
            payload={
                "executable": "/nonexistent/path/to/codex-cli-binary",
                "cwd": str(Path.cwd()),
            },
        )
        artifacts = CodexAdapter().run(task, "goal", "worker")

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.VERIFICATION)
        self.assertEqual(artifact.payload["adapter"], "codex")
        self.assertEqual(artifact.payload["result"], "blocked")
        self.assertEqual(artifact.payload["failure"], "missing_cli")

    def test_build_codex_exec_command_emits_expected_flags(self) -> None:
        """The Codex command builder must produce the non-interactive
        flag soup that v0.7.0 ships by default: `exec --json`,
        `approval_policy="never"`, `--sandbox`, `--ephemeral`,
        `--skip-git-repo-check`, `-C cwd`, `-m model`, and the prompt as
        the final positional argument. Asserting on the exact command
        shape protects against accidental regressions."""
        cmd = build_codex_exec_command(
            executable=["codex"],
            prompt="hello world",
            model="gpt-5.4-mini",
            cwd=Path("/tmp/codex-test-cwd"),
            sandbox="workspace-write",
            approval_policy="never",
            ephemeral=True,
            skip_git_repo_check=True,
        )
        self.assertEqual(cmd[0], "codex")
        self.assertEqual(cmd[1], "exec")
        self.assertIn("--json", cmd)
        self.assertIn("-c", cmd)
        self.assertIn('approval_policy="never"', cmd)
        self.assertIn("--sandbox", cmd)
        self.assertIn("workspace-write", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertIn("-C", cmd)
        # The builder normalizes the cwd via Path; compare against the same
        # normalization so the assertion holds on Windows (\tmp\...) too.
        self.assertIn(str(Path("/tmp/codex-test-cwd")), cmd)
        self.assertIn("-m", cmd)
        self.assertIn("gpt-5.4-mini", cmd)
        self.assertEqual(cmd[-1], "hello world")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)

    def test_build_codex_exec_command_with_danger_bypass(self) -> None:
        cmd = build_codex_exec_command(
            executable=["codex"],
            prompt="audit",
            model=None,
            sandbox="workspace-write",
            dangerously_bypass=True,
        )
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)

    def test_parse_codex_events_skips_banners_and_invalid_json(self) -> None:
        """Codex CLI mixes a few non-JSON banner lines into its --json
        stream when stdin isn't a TTY (e.g. "Reading additional input
        from stdin...") and the websocket layer occasionally emits ERROR
        lines. The parser must skip those gracefully and return only the
        valid JSONL events."""
        sample = "\n".join(
            [
                "Reading additional input from stdin...",
                '{"type":"thread.started","thread_id":"abc"}',
                "2026-05-28T15:04:11Z ERROR codex_api: warning text",
                '{"type":"turn.started"}',
                "not-json {",
                '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}',
                '{"type":"turn.completed","usage":{"input_tokens":42,"output_tokens":7,"cached_input_tokens":0,"reasoning_output_tokens":3}}',
                "",
            ]
        )
        events = parse_codex_events(sample)
        types = [ev.get("type") for ev in events]
        self.assertEqual(
            types,
            ["thread.started", "turn.started", "item.completed", "turn.completed"],
        )
        last = last_codex_agent_message(events)
        self.assertEqual(last, "hi")

    def test_last_codex_agent_message_returns_final_agent_message(self) -> None:
        """When Codex emits multiple item.completed events (tool calls,
        reasoning summaries, then the final reply), only the LAST
        item.completed of type=agent_message should be returned."""
        events = [
            {"type": "item.completed", "item": {"type": "command_execution", "text": "ran ls"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "interim"}},
            {"type": "item.completed", "item": {"type": "command_execution", "text": "ran cat"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
        self.assertEqual(last_codex_agent_message(events), "FINAL")

    def test_last_codex_agent_message_returns_empty_when_none_present(self) -> None:
        events = [
            {"type": "thread.started", "thread_id": "x"},
            {"type": "turn.started"},
            {"type": "turn.failed", "error": {"message": "401"}},
        ]
        self.assertEqual(last_codex_agent_message(events), "")

    def test_classify_codex_failure_known_signals(self) -> None:
        self.assertEqual(
            classify_codex_failure("401 Unauthorized: Missing bearer"),
            "not_authenticated",
        )
        self.assertEqual(classify_codex_failure("Not logged in"), "not_authenticated")
        self.assertEqual(classify_codex_failure("rate limit exceeded"), "rate_limit")
        self.assertEqual(classify_codex_failure("credit balance is too low"), "billing_or_quota")
        self.assertEqual(
            classify_codex_failure("the requested model is not found"),
            "model_unavailable",
        )
        self.assertEqual(classify_codex_failure("request timed out"), "timeout")
        self.assertEqual(classify_codex_failure("DNS resolution failed"), "network_error")
        self.assertEqual(classify_codex_failure("approval was denied by user"), "approval_denied")
        self.assertEqual(classify_codex_failure("sandbox: write blocked"), "sandbox_denied")
        self.assertEqual(classify_codex_failure("completely unrelated text"), "unknown")

    def test_diagnostics_list_provider_neutral_adapters(self) -> None:
        rows = adapter_status(Path.cwd())
        names = {row["name"] for row in rows}

        self.assertIn("cursor", names)
        self.assertIn("claude-code", names)
        self.assertIn("codex", names)

    def test_cursor_sdk_detected_in_puppetmaster_package_dir(self) -> None:
        """`puppetmaster adapters` from an unrelated workspace must still see
        the bundled @cursor/sdk install — the adapter resolves the SDK from
        the Puppetmaster package's own node_modules at runtime, not from cwd."""
        from puppetmaster import diagnostics

        with TemporaryDirectory() as tmp:
            unrelated_repo = Path(tmp) / "ff-data-engineering"
            unrelated_repo.mkdir()
            fake_package_root = Path(tmp) / "Puppetmaster"
            (fake_package_root / "node_modules" / "@cursor" / "sdk").mkdir(parents=True)
            fake_diagnostics_file = fake_package_root / "puppetmaster" / "diagnostics.py"
            fake_diagnostics_file.parent.mkdir()
            fake_diagnostics_file.write_text("# stub", encoding="utf-8")
            with patch.object(diagnostics, "__file__", str(fake_diagnostics_file)):
                self.assertTrue(diagnostics._cursor_sdk_installed(unrelated_repo))
                location = diagnostics._find_cursor_sdk_install(unrelated_repo)
            self.assertIsNotNone(location)
            # macOS resolves /var -> /private/var; compare resolved paths.
            expected = (fake_package_root / "node_modules" / "@cursor" / "sdk").resolve()
            self.assertEqual(location.resolve(), expected)

    def test_cursor_sdk_detection_honors_workspace_install(self) -> None:
        """Local repo node_modules install still counts (precedence over package dir)."""
        from puppetmaster import diagnostics

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            (workspace / "node_modules" / "@cursor" / "sdk").mkdir(parents=True)
            fake_pkg_root = Path(tmp) / "nowhere"  # no SDK here
            fake_diagnostics_file = fake_pkg_root / "puppetmaster" / "diagnostics.py"
            fake_diagnostics_file.parent.mkdir(parents=True)
            with patch.object(diagnostics, "__file__", str(fake_diagnostics_file)):
                self.assertTrue(diagnostics._cursor_sdk_installed(workspace))

    def test_cursor_sdk_detection_returns_false_when_neither_exists(self) -> None:
        from puppetmaster import diagnostics

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            fake_pkg_root = Path(tmp) / "nowhere"
            fake_diagnostics_file = fake_pkg_root / "puppetmaster" / "diagnostics.py"
            fake_diagnostics_file.parent.mkdir(parents=True)
            old_home = os.environ.pop("PUPPETMASTER_HOME", None)
            try:
                with patch.object(diagnostics, "__file__", str(fake_diagnostics_file)):
                    self.assertFalse(diagnostics._cursor_sdk_installed(workspace))
            finally:
                if old_home is not None:
                    os.environ["PUPPETMASTER_HOME"] = old_home

    def test_find_state_dir_for_job_locates_owning_project(self) -> None:
        """`puppetmaster show <job_id>` must work from any cwd by auto-discovering
        which project state dir owns the job — no PUPPETMASTER_STATE_DIR export required."""
        from puppetmaster import state as state_module

        with TemporaryDirectory() as tmp:
            projects_root = Path(tmp) / "projects"
            project_a = projects_root / "ff-data-engineering-cfbfad67d9fc"
            project_b = projects_root / "ff-ios-589b71a4121f"
            (project_a / "jobs" / "job_476cbf98144f").mkdir(parents=True)
            (project_b / "jobs" / "job_83a3481f7ae8").mkdir(parents=True)

            with patch.object(state_module, "app_state_root", return_value=Path(tmp)):
                self.assertEqual(
                    state_module.find_state_dir_for_job("job_476cbf98144f"),
                    project_a,
                )
                self.assertEqual(
                    state_module.find_state_dir_for_job("job_83a3481f7ae8"),
                    project_b,
                )
                self.assertIsNone(
                    state_module.find_state_dir_for_job("job_does_not_exist")
                )

    def test_find_state_dir_for_job_returns_none_when_root_missing(self) -> None:
        from puppetmaster import state as state_module

        with TemporaryDirectory() as tmp:
            with patch.object(state_module, "app_state_root", return_value=Path(tmp)):
                self.assertIsNone(state_module.find_state_dir_for_job("job_abc"))

    def test_list_project_state_dirs_handles_missing_root(self) -> None:
        from puppetmaster import state as state_module

        with TemporaryDirectory() as tmp:
            with patch.object(state_module, "app_state_root", return_value=Path(tmp)):
                self.assertEqual(state_module.list_project_state_dirs(), [])

    def test_codegraph_lock_path_is_per_repo(self) -> None:
        """Different repos get different lock files so they can index in parallel."""
        from puppetmaster import codegraph as codegraph_mod

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = tmp
            try:
                repo_a = Path(tmp) / "ff-data-engineering"
                repo_b = Path(tmp) / "ff-ios"
                repo_a.mkdir()
                repo_b.mkdir()
                lock_a = codegraph_mod.codegraph_lock_path(repo_a)
                lock_b = codegraph_mod.codegraph_lock_path(repo_b)
                self.assertNotEqual(lock_a, lock_b)
                self.assertIn("ff-data-engineering", lock_a.name)
                self.assertIn("ff-ios", lock_b.name)
                # And legacy callers still get the global lock.
                legacy = codegraph_mod.codegraph_lock_path()
                self.assertEqual(legacy.name, "codegraph-indexer.lock")
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

    def test_codegraph_lock_allows_different_repos_in_parallel(self) -> None:
        """Two repos can hold their per-repo locks at the same time."""
        from puppetmaster import codegraph as codegraph_mod

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = tmp
            try:
                repo_a = Path(tmp) / "repo-a"
                repo_b = Path(tmp) / "repo-b"
                repo_a.mkdir()
                repo_b.mkdir()
                lock_a = codegraph_mod.acquire_codegraph_lock(repo_root=repo_a)
                try:
                    # This MUST succeed: a different repo holds an
                    # unrelated lock.
                    lock_b = codegraph_mod.acquire_codegraph_lock(
                        repo_root=repo_b
                    )
                    lock_b.release()
                finally:
                    lock_a.release()
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

    def test_codegraph_lock_busy_when_same_repo(self) -> None:
        """Second acquire on the SAME repo's lock raises CodegraphLockBusy."""
        from puppetmaster import codegraph as codegraph_mod

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = tmp
            try:
                repo = Path(tmp) / "shared-repo"
                repo.mkdir()
                first = codegraph_mod.acquire_codegraph_lock(repo_root=repo)
                try:
                    with self.assertRaises(codegraph_mod.CodegraphLockBusy):
                        codegraph_mod.acquire_codegraph_lock(repo_root=repo)
                finally:
                    first.release()
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

    def test_codegraph_lock_stale_pid_auto_clear(self) -> None:
        """When the lock file records a dead PID, the next acquire takes over."""
        from puppetmaster import codegraph as codegraph_mod

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"] = tmp
            try:
                repo = Path(tmp) / "stale-repo"
                repo.mkdir()
                lock_path = codegraph_mod.codegraph_lock_path(repo)
                lock_path.write_text("99999999\n", encoding="utf-8")
                # The PID 99999999 is essentially guaranteed dead.
                # Acquire should still work because the file was never
                # actually flock'd by anything alive.
                acquired = codegraph_mod.acquire_codegraph_lock(repo_root=repo)
                try:
                    self.assertIsNotNone(acquired._fd)
                    self.assertEqual(
                        lock_path.read_text(encoding="utf-8").strip(),
                        str(os.getpid()),
                    )
                finally:
                    acquired.release()
            finally:
                del os.environ["PUPPETMASTER_CODEGRAPH_LOCK_DIR"]

    def test_pid_is_alive_helper(self) -> None:
        from puppetmaster.codegraph import _pid_is_alive

        self.assertTrue(_pid_is_alive(os.getpid()))
        self.assertFalse(_pid_is_alive(99999999))
        self.assertFalse(_pid_is_alive(0))
        self.assertFalse(_pid_is_alive(-1))

    def test_idle_keepalive_emits_when_no_tool_call_active(self) -> None:
        """The idle keepalive fires periodic notifications while nothing is running."""
        from puppetmaster import mcp_server

        emitted: list[dict] = []

        def fake_emit(notification):
            emitted.append(notification)
            return True

        with mcp_server._INPUT_STATE_LOCK:
            mcp_server._ACTIVE_TOOL_CALLS = 0
        mcp_server._SHUTDOWN_REQUESTED.clear()
        keepalive = mcp_server._IdleKeepalive(
            interval_seconds=0.05,
            emitter=fake_emit,
        )
        keepalive.start()
        try:
            # Poll up to a generous deadline instead of a fixed sleep: at a
            # 0.05s interval two emissions take ~0.1s, but a loaded CI runner
            # can starve the timer thread, so wait (don't assume) for >= 2.
            deadline = time.time() + 5.0
            while len(emitted) < 2 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            keepalive.stop()
        self.assertGreaterEqual(len(emitted), 2)
        for notification in emitted:
            self.assertEqual(notification["method"], "notifications/message")
            self.assertEqual(notification["params"]["data"]["kind"], "idle_keepalive")
            self.assertNotIn("id", notification)

    def test_idle_keepalive_suppressed_during_tool_call(self) -> None:
        """An in-flight tool call shouldn't get extra keepalive traffic from the idle pinger."""
        from puppetmaster import mcp_server

        emitted: list[dict] = []

        def fake_emit(notification):
            emitted.append(notification)
            return True

        with mcp_server._INPUT_STATE_LOCK:
            mcp_server._ACTIVE_TOOL_CALLS = 1
        mcp_server._SHUTDOWN_REQUESTED.clear()
        keepalive = mcp_server._IdleKeepalive(
            interval_seconds=0.05,
            emitter=fake_emit,
        )
        keepalive.start()
        try:
            time.sleep(0.25)
        finally:
            keepalive.stop()
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._ACTIVE_TOOL_CALLS = 0
        self.assertEqual(emitted, [])

    def test_idle_keepalive_stops_on_broken_pipe(self) -> None:
        """A failed write should terminate the keepalive thread cleanly."""
        from puppetmaster import mcp_server

        emitted: list[dict] = []
        calls = {"n": 0}

        def fake_emit(notification):
            calls["n"] += 1
            emitted.append(notification)
            return False  # simulate pipe down

        with mcp_server._INPUT_STATE_LOCK:
            mcp_server._ACTIVE_TOOL_CALLS = 0
        mcp_server._SHUTDOWN_REQUESTED.clear()
        keepalive = mcp_server._IdleKeepalive(
            interval_seconds=0.05,
            emitter=fake_emit,
        )
        keepalive.start()
        try:
            time.sleep(0.25)
        finally:
            keepalive.stop()
        # Exactly one emit attempt (we returned False on the first one).
        self.assertEqual(calls["n"], 1)

    def test_parallel_doctor_calls_do_not_kill_mcp_server(self) -> None:
        """Regression: 30 parallel doctor MCP calls used to silently kill the
        server (exit 0 via stdin EOF) because subprocess.run children
        inherited the parent's fd 0 and somehow caused the parent's stdin
        reader to receive a phantom EOF. Every subprocess in the server's
        code path now passes stdin=DEVNULL. This test spawns the real
        server over stdio, sends 30 parallel doctor calls, and asserts the
        server stays alive and every call returns a response."""
        import subprocess
        import threading

        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.Popen(
            [sys.executable, "-m", "puppetmaster.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(repo_root),
            env={
                **os.environ,
                "PUPPETMASTER_MCP_INPUT_STALE_DISABLED": "1",
                "PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED": "1",
            },
            bufsize=0,
        )
        responses: dict = {}
        reader_done = threading.Event()

        def reader():
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if "id" in msg and "method" not in msg:
                    responses[msg["id"]] = msg
            reader_done.set()

        threading.Thread(target=reader, daemon=True).start()

        write_lock = threading.Lock()

        def send(rid):
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "method": "tools/call",
                    "params": {"name": "puppetmaster_doctor", "arguments": {}},
                }
            ) + "\n"
            with write_lock:
                assert proc.stdin is not None
                proc.stdin.write(payload.encode("utf-8"))
                proc.stdin.flush()

        try:
            # Handshake.
            send(0)
            deadline = time.time() + 15
            while time.time() < deadline and 0 not in responses:
                time.sleep(0.05)
            self.assertIn(0, responses, "handshake never returned")

            # Hammer with 30 parallel doctor calls.
            senders = []
            for i in range(30):
                t = threading.Thread(target=send, args=(1000 + i,))
                t.start()
                senders.append(t)
            for t in senders:
                t.join()

            deadline = time.time() + 60
            missing = set(range(1000, 1030))
            while time.time() < deadline and missing:
                missing = {r for r in range(1000, 1030) if r not in responses}
                if missing:
                    time.sleep(0.2)

            self.assertEqual(
                set(), missing, f"missing responses for ids: {sorted(missing)}"
            )
            self.assertIsNone(
                proc.poll(),
                f"server unexpectedly died with exit={proc.poll()}",
            )
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def test_idle_keepalive_can_be_disabled_via_env(self) -> None:
        from puppetmaster.mcp_server import _idle_keepalive_disabled

        os.environ["PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED"] = "1"
        try:
            self.assertTrue(_idle_keepalive_disabled())
        finally:
            del os.environ["PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED"]
        self.assertFalse(_idle_keepalive_disabled())

    def test_starter_config_is_loadable(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "puppetmaster.json"
            path.write_text(starter_config(), encoding="utf-8")

            config = load_config(path)

            self.assertGreaterEqual(len(config.workers), 2)

    def test_sqlite_backend_runs_enterprise_workflow(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            specs = [
                WorkerSpec(role="explore", instruction="find facts"),
                WorkerSpec(
                    role="runtime-check",
                    instruction="check python runtime",
                    adapter="shell",
                    payload={
                        "command": [sys.executable, "--version"],
                        "timeout_seconds": 10,
                    },
                    depends_on_roles=["explore"],
                ),
            ]
            result = Orchestrator(store).run("run sqlite backed workflow", specs=specs)
            tasks = store.list_tasks(result.job.id)
            events = store.read_events(result.job.id)

            self.assertEqual(store.get_job(result.job.id).status.value, "complete")
            self.assertEqual(len(tasks), 2)
            self.assertEqual(len(store.list_artifacts(result.job.id)), 3)
            self.assertIn("task.unblocked", {event["event"] for event in events})

    def test_sqlite_reopens_existing_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / ".puppetmaster"
            store = SQLiteSwarmStore(root)
            result = Orchestrator(store).run("persist across process restart", roles=["explore"])

            reopened = SQLiteSwarmStore(root)

            self.assertEqual(reopened.get_job(result.job.id).status, JobStatus.COMPLETE)
            self.assertEqual(len(reopened.list_artifacts(result.job.id)), 2)

    def test_sqlite_schema_status_and_doctor_are_available(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteSwarmStore(root / ".puppetmaster")
            store.init()

            status = store.schema_status()
            checks = {check.name: check for check in run_doctor(root, root / ".puppetmaster")}

            self.assertEqual(status["schema_version"], "1")
            self.assertEqual(status["expected_schema_version"], "1")
            self.assertEqual(checks["sqlite-state"].status, "ok")

    def test_cli_last_and_clean_support_daily_run_management(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = str(Path(tmp) / ".puppetmaster")

            run_code = cli_main([
                "--state-dir",
                state_dir,
                "--backend",
                "file",
                "run",
                "daily driver check",
                "--workers",
                "explore",
            ])
            clean_code = cli_main([
                "--state-dir",
                state_dir,
                "--backend",
                "file",
                "clean",
                "--completed",
            ])

            self.assertEqual(run_code, 0)
            self.assertEqual(clean_code, 0)

    def test_cli_approve_and_reject_accept_job_targets(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = str(Path(tmp) / ".puppetmaster")
            cli_main([
                "--state-dir",
                state_dir,
                "--backend",
                "file",
                "run",
                "approval gate check",
            ])
            store = SwarmStore(Path(state_dir))
            job_id = store.latest_job().id

            approve_code = cli_main(["--state-dir", state_dir, "--backend", "file", "approve", job_id])
            reject_code = cli_main([
                "--state-dir",
                state_dir,
                "--backend",
                "file",
                "reject",
                job_id,
                "--reason",
                "unit test",
            ])
            events = [event["event"] for event in store.read_events(job_id)]

            self.assertEqual(approve_code, 0)
            self.assertEqual(reject_code, 0)
            self.assertIn("artifact.approved", events)
            self.assertIn("artifact.rejected", events)

    def test_cli_missing_config_returns_setup_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main([
                    "--state-dir",
                    str(Path(tmp) / ".puppetmaster"),
                    "run",
                    "missing config",
                    "--config",
                    str(Path(tmp) / "missing.json"),
                ])

            self.assertEqual(code, 1)
            self.assertIn("missing.json", stderr.getvalue())


class ModelRouterTests(unittest.TestCase):
    """Tests for the user-owned LLM model registry and routing engine.

    These exercise the pure-function pieces (classifier, policy engine,
    cost estimation) and the orchestrator integration that stamps the
    chosen model + adapter into the task and persists a ROUTING
    artifact for the audit trail.
    """

    def _three_tier_registry(self):
        from puppetmaster.model_registry import ModelSpec

        return [
            ModelSpec(
                id="cheap-model",
                adapter="claude-code",
                adapter_model_name="cheap-v1",
                capability_score=40,
                input_per_mtok_usd=0.10,
                output_per_mtok_usd=0.50,
                tags=["cheap", "fast"],
            ),
            ModelSpec(
                id="mid-model",
                adapter="claude-code",
                adapter_model_name="mid-v1",
                capability_score=80,
                input_per_mtok_usd=3.0,
                output_per_mtok_usd=15.0,
                tags=["balanced"],
            ),
            ModelSpec(
                id="frontier-model",
                adapter="claude-code",
                adapter_model_name="frontier-v1",
                capability_score=95,
                input_per_mtok_usd=15.0,
                output_per_mtok_usd=75.0,
                tags=["frontier", "reasoning"],
            ),
        ]

    def test_classifier_assigns_higher_score_to_harder_roles(self) -> None:
        from puppetmaster.router import TaskSignals, classify_capability_needed

        easy = classify_capability_needed(
            TaskSignals(instruction="format this file", role="verify-runtime")
        )
        hard = classify_capability_needed(
            TaskSignals(
                instruction="security audit of authentication across every endpoint",
                role="audit",
            )
        )
        self.assertLess(easy, hard)
        self.assertGreaterEqual(hard, 90)
        self.assertLessEqual(easy, 35)

    def test_classifier_honors_explicit_min_capability_override(self) -> None:
        from puppetmaster.router import TaskSignals, classify_capability_needed

        # Even a "verify-runtime" role gets escalated when the user pins it.
        signal = TaskSignals(
            instruction="ignored", role="verify-runtime", explicit_min_capability=80
        )
        self.assertEqual(classify_capability_needed(signal), 80)

    def test_balanced_policy_picks_cheapest_sufficient_model(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        # Need ~70 (implement role). Cheapest model that clears the bar is mid-model.
        signal = TaskSignals(instruction="add a feature", role="implement")
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "mid-model")
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertEqual(rejected_ids, {"cheap-model", "frontier-model"})

    def test_balanced_policy_escalates_to_frontier_for_audits(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="security audit across every module",
            role="audit",
        )
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "frontier-model")

    def test_cheap_policy_always_picks_lowest_cost(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="security audit across every module",
            role="audit",
        )
        decision = route_task(signal, self._three_tier_registry(), policy="cheap")
        # Even though the task needs high capability, cheap policy ignores fit.
        self.assertEqual(decision.model.id, "cheap-model")

    def test_quality_policy_always_picks_highest_capability(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="trivial task", role="verify-runtime")
        decision = route_task(signal, self._three_tier_registry(), policy="quality")
        self.assertEqual(decision.model.id, "frontier-model")

    def test_required_tag_filter_excludes_models_lacking_tag(self) -> None:
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        signal = TaskSignals(
            instruction="cheap fast task",
            role="explore",
            required_tags=["cheap"],
        )
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "cheap-model")

        impossible = TaskSignals(
            instruction="task", role="explore", required_tags=["nonexistent"]
        )
        with self.assertRaises(NoEligibleModelError):
            route_task(impossible, self._three_tier_registry(), policy="balanced")

    def test_max_cost_budget_rejects_pricier_models(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="security audit across every module",
            role="audit",
            explicit_max_cost_usd=0.01,
        )
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        # Frontier is over budget; mid and cheap remain.
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("frontier-model", rejected_ids)
        # No sufficient model in budget → falls back to highest-capability under cap.
        self.assertIn("budget", "".join(r for _, r in decision.rejected if r))

    def test_routing_decision_records_rejection_reasons(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="add a feature", role="implement")
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        rejected_map = {spec.id: why for spec, why in decision.rejected}
        self.assertIn("capability_score", rejected_map["cheap-model"])
        self.assertIn("pricier", rejected_map["frontier-model"])

    def test_balanced_equal_cost_rejection_is_not_labeled_pricier(self) -> None:
        """Two equal-priced models that both clear the bar must not be
        rejected with a 'pricier' reason — the tie-break is capability
        right-sizing, not cost. Regression for the opus-4-7 vs opus-4-8
        (both $5/$25) case."""
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import TaskSignals, route_task

        registry = [
            ModelSpec(
                id="tier-a",
                adapter="claude-code",
                adapter_model_name="a",
                capability_score=98,
                input_per_mtok_usd=5.0,
                output_per_mtok_usd=25.0,
            ),
            ModelSpec(
                id="tier-b",
                adapter="claude-code",
                adapter_model_name="b",
                capability_score=99,
                input_per_mtok_usd=5.0,
                output_per_mtok_usd=25.0,
            ),
        ]
        signal = TaskSignals(instruction="hard task", role="audit", explicit_min_capability=95)
        decision = route_task(signal, registry, policy="balanced")
        # Lower-capability equal-cost model is right-sized and wins.
        self.assertEqual(decision.model.id, "tier-a")
        rejected_map = {spec.id: why for spec, why in decision.rejected}
        self.assertIn("tier-b", rejected_map)
        self.assertNotIn("pricier", rejected_map["tier-b"])
        self.assertIn("same estimated cost", rejected_map["tier-b"])

    def test_artifact_payload_carries_audit_fields(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="add a feature", role="implement")
        decision = route_task(signal, self._three_tier_registry(), policy="balanced")
        payload = decision.to_artifact_payload()
        for key in (
            "model_id",
            "adapter",
            "adapter_model_name",
            "policy",
            "capability_needed",
            "capability_score",
            "estimated_cost_usd",
            "reason",
            "rejected",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["policy"], "balanced")
        self.assertIsInstance(payload["rejected"], list)

    def test_registry_round_trips_through_disk(self) -> None:
        from puppetmaster.model_registry import (
            load_registry,
            save_registry,
            starter_registry,
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            specs = starter_registry()
            save_registry(specs, path)
            loaded = load_registry(path)
            self.assertEqual(len(loaded), len(specs))
            self.assertEqual({s.id for s in loaded}, {s.id for s in specs})

    def test_registry_environment_override(self) -> None:
        from puppetmaster.model_registry import default_registry_path

        with TemporaryDirectory() as tmp:
            override = str(Path(tmp) / "custom.json")
            with patch.dict(os.environ, {"PUPPETMASTER_MODELS_PATH": override}):
                self.assertEqual(default_registry_path(), Path(override))

    def test_cost_estimate_scales_linearly_with_tokens(self) -> None:
        spec = self._three_tier_registry()[0]
        small = spec.estimate_cost_usd(1_000, 1_000)
        big = spec.estimate_cost_usd(10_000, 10_000)
        self.assertAlmostEqual(big, small * 10, places=6)

    def test_models_init_writes_starter_registry(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            code = cli_main(["models", "init", "--registry-path", str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text())
            self.assertIn("models", payload)
            self.assertGreater(len(payload["models"]), 0)

    def test_models_init_refuses_overwrite_without_force(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text("{}")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(["models", "init", "--registry-path", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("already exists", stderr.getvalue())

    def test_cli_route_command_emits_json(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            cli_main(["models", "init", "--registry-path", str(path)])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "route",
                        "Audit the auth subsystem for security flaws",
                        "--role",
                        "audit",
                        "--registry-path",
                        str(path),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            data = json.loads(stdout.getvalue())
            self.assertIn("model_id", data)
            self.assertIn("rejected", data)
            self.assertEqual(data["policy"], "balanced")

    def test_orchestrator_auto_routes_and_emits_routing_artifact(self) -> None:
        """End-to-end check: an auto_route spec gets adapter swapped
        and a ROUTING artifact persisted with the chosen model id."""
        from puppetmaster.model_registry import save_registry
        from puppetmaster.models import ArtifactType
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(self._three_tier_registry(), registry_path)
            state_dir = Path(tmp) / ".puppetmaster"

            from puppetmaster.orchestrator import Orchestrator
            from puppetmaster.store_factory import create_store

            store = create_store("file", state_dir)
            store.init()

            orchestrator = Orchestrator(store)
            spec = WorkerSpec(
                role="audit",
                instruction="security audit across every endpoint and module",
                adapter="local",  # router must override this
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                    "routing_policy": "balanced",
                },
                depends_on_roles=[],
            )

            # We don't want to actually run an adapter — just inspect that
            # _create_tasks stamps the chosen model into the task and
            # persists a ROUTING artifact. Drive _create_tasks directly.
            job = store.create_job("router integration check")
            tasks = orchestrator._create_tasks(job, [spec])
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.adapter, "claude-code")
            self.assertEqual(task.payload.get("model"), "frontier-v1")
            self.assertEqual(task.payload.get("router_model_id"), "frontier-model")

            artifacts = store.list_artifacts(job.id)
            routing = [a for a in artifacts if a.type == ArtifactType.ROUTING]
            self.assertEqual(len(routing), 1)
            self.assertEqual(routing[0].task_id, task.id)
            self.assertEqual(routing[0].payload["model_id"], "frontier-model")
            self.assertEqual(routing[0].payload["adapter"], "claude-code")
            self.assertIn("rejected", routing[0].payload)

    def test_vision_signal_detection_finds_image_references(self) -> None:
        from puppetmaster.router import has_detailed_vision_signal, has_vision_signal

        self.assertTrue(has_vision_signal("describe the screenshot"))
        self.assertTrue(has_vision_signal("read this diagram"))
        self.assertTrue(has_vision_signal("look at the image"))
        self.assertFalse(has_vision_signal("read this file"))
        self.assertFalse(has_vision_signal("refactor the database layer"))

        self.assertTrue(
            has_detailed_vision_signal(
                "OCR every detail of the diagram"
            )
        )
        self.assertTrue(
            has_detailed_vision_signal("read this screenshot")
        )
        self.assertFalse(has_detailed_vision_signal("describe the screenshot"))

    def test_router_auto_requires_vision_tag_for_image_tasks(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        # cheap-model has no vision tag → filtered out automatically.
        registry = self._three_tier_registry()
        from dataclasses import replace as dc_replace

        registry = [
            dc_replace(registry[0], tags=registry[0].tags),  # no vision
            dc_replace(registry[1], tags=registry[1].tags + ["vision"]),
            dc_replace(registry[2], tags=registry[2].tags + ["vision"]),
        ]
        signal = TaskSignals(
            instruction="describe this screenshot for me",
            role="explore",
        )
        decision = route_task(signal, registry, policy="balanced")
        self.assertNotEqual(decision.model.id, "cheap-model")
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("cheap-model", rejected_ids)
        cheap_reason = next(why for spec, why in decision.rejected if spec.id == "cheap-model")
        self.assertIn("vision", cheap_reason)

    def test_router_routes_detailed_vision_to_detailed_vision_tagged_model(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="OCR every detail of the diagram and extract every element in the image",
            role="explore",
        )
        decision = route_task(signal, starter_registry(), policy="balanced")
        # Must land on a detailed-vision-tagged model. Under balanced policy the
        # cheapest qualifying one wins (so the OpenAI tier may beat the Claude
        # tier here on price). Both options carry detailed-vision.
        self.assertIn("detailed-vision", decision.model.tags)
        # Non-detailed-vision models must show up in rejected with the tag reason.
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("cursor/composer-2-5", rejected_ids)
        self.assertIn("cursor/gpt-5-5", rejected_ids)
        # And under quality policy, the frontier flagship opus-4-8 (cap 99)
        # now wins over opus-4-7 (98) and gpt-5.5 (96).
        quality_decision = route_task(
            signal, starter_registry(), policy="quality"
        )
        self.assertEqual(quality_decision.model.id, "claude-code/opus-4-8")

    def test_starter_registry_encodes_four_tiers(self) -> None:
        from puppetmaster.model_registry import starter_registry

        specs = starter_registry()
        ids = {s.id for s in specs}
        self.assertIn("cursor/composer-2-5", ids)
        self.assertIn("cursor/gpt-5-5", ids)
        self.assertIn("claude-code/opus-4-6", ids)
        self.assertIn("claude-code/opus-4-7", ids)
        # Capability scores are monotone across tiers.
        by_id = {s.id: s for s in specs}
        self.assertLess(
            by_id["cursor/composer-2-5"].capability_score,
            by_id["cursor/gpt-5-5"].capability_score,
        )
        self.assertLess(
            by_id["cursor/gpt-5-5"].capability_score,
            by_id["claude-code/opus-4-6"].capability_score,
        )
        self.assertLess(
            by_id["claude-code/opus-4-6"].capability_score,
            by_id["claude-code/opus-4-7"].capability_score,
        )
        # Opus 4.8 is the new frontier flagship — strictly above 4.7 and the
        # single highest-capability model in the starter registry.
        self.assertIn("claude-code/opus-4-8", ids)
        self.assertLess(
            by_id["claude-code/opus-4-7"].capability_score,
            by_id["claude-code/opus-4-8"].capability_score,
        )
        self.assertEqual(
            by_id["claude-code/opus-4-8"].capability_score,
            max(s.capability_score for s in specs),
        )
        # Same per-token price as 4.7 (it strictly dominates) with a far
        # larger context window.
        self.assertEqual(
            by_id["claude-code/opus-4-8"].input_per_mtok_usd,
            by_id["claude-code/opus-4-7"].input_per_mtok_usd,
        )
        self.assertGreater(
            by_id["claude-code/opus-4-8"].context_window,
            by_id["claude-code/opus-4-7"].context_window,
        )
        # Vision tagging matches the user-stated preferences.
        self.assertNotIn("vision", by_id["cursor/composer-2-5"].tags)
        self.assertIn("vision", by_id["cursor/gpt-5-5"].tags)
        self.assertIn("vision", by_id["claude-code/opus-4-6"].tags)
        self.assertIn("detailed-vision", by_id["claude-code/opus-4-7"].tags)
        self.assertIn("detailed-vision", by_id["claude-code/opus-4-8"].tags)
        self.assertNotIn(
            "detailed-vision", by_id["claude-code/opus-4-6"].tags
        )

    def test_starter_registry_routes_hardest_task_to_opus_4_8(self) -> None:
        """The absolute-hardest tasks must route to the frontier flagship
        (Opus 4.8), not saturate one notch below it on the older 4.7."""
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction=(
                "Perform an exhaustive security audit across every module and "
                "exploit any authentication bypass you can find."
            ),
            role="security-review",
        )
        decision = route_task(signal, starter_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "claude-code/opus-4-8")
        # 4.7 should be in the rejected set (sufficient-but-not-chosen), proving
        # the flagship was preferred for the hardest tier.
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("claude-code/opus-4-7", rejected_ids)

    def test_starter_registry_routes_easy_task_to_composer(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.router import TaskSignals, route_task

        decision = route_task(
            TaskSignals(instruction="format these files", role="verify-runtime"),
            starter_registry(),
            policy="balanced",
        )
        self.assertEqual(decision.model.id, "cursor/composer-2-5")

    def test_balanced_tie_break_picks_lower_capability_when_costs_equal(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import TaskSignals, route_task

        # Two free models, both sufficient for the task. Tie-break: the
        # smaller (right-sized) wins instead of wasting the bigger one.
        registry = [
            ModelSpec(
                id="small-free",
                adapter="cursor",
                adapter_model_name="small",
                capability_score=60,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                tags=["cursor"],
            ),
            ModelSpec(
                id="big-free",
                adapter="cursor",
                adapter_model_name="big",
                capability_score=90,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                tags=["cursor"],
            ),
        ]
        decision = route_task(
            TaskSignals(instruction="map the auth module", role="explore"),
            registry,
            policy="balanced",
        )
        self.assertEqual(decision.model.id, "small-free")

    def test_default_workers_opt_into_auto_routing(self) -> None:
        from puppetmaster.workers import DEFAULT_WORKERS

        for spec in DEFAULT_WORKERS:
            self.assertTrue(
                spec.payload.get("auto_route"),
                f"DEFAULT_WORKERS[{spec.role}] should auto-route by default",
            )

    def test_orchestrator_silently_passes_through_when_registry_missing(self) -> None:
        """If the user hasn't run `models init`, auto-routing must be
        a no-op — no exception, no orphan ROUTING artifacts."""
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            missing_registry = Path(tmp) / "does-not-exist.json"
            from puppetmaster.orchestrator import Orchestrator
            from puppetmaster.store_factory import create_store

            store = create_store("file", state_dir)
            store.init()
            orchestrator = Orchestrator(store)

            spec = WorkerSpec(
                role="explore",
                instruction="map repo",
                adapter="local",
                payload={
                    "auto_route": True,
                    "registry_path": str(missing_registry),
                },
            )
            job = store.create_job("empty registry check")
            tasks = orchestrator._create_tasks(job, [spec])
            self.assertEqual(tasks[0].adapter, "local")

            from puppetmaster.models import ArtifactType

            artifacts = store.list_artifacts(job.id)
            self.assertFalse(
                any(a.type == ArtifactType.ROUTING for a in artifacts)
            )

    def test_cost_command_sums_routing_artifacts_for_job(self) -> None:
        from puppetmaster.model_registry import save_registry
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(self._three_tier_registry(), registry_path)
            state_dir = Path(tmp) / ".puppetmaster"
            from puppetmaster.orchestrator import Orchestrator
            from puppetmaster.store_factory import create_store

            store = create_store("file", state_dir)
            store.init()
            orchestrator = Orchestrator(store)
            spec = WorkerSpec(
                role="audit",
                instruction="security audit across every endpoint",
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                },
            )
            job = store.create_job("cost check")
            orchestrator._create_tasks(job, [spec])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--backend",
                        "file",
                        "cost",
                        job.id,
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            data = json.loads(stdout.getvalue())
            self.assertEqual(data["job_id"], job.id)
            self.assertGreater(data["total_estimated_cost_usd"], 0.0)
            self.assertEqual(len(data["tasks"]), 1)
            self.assertEqual(data["tasks"][0]["role"], "audit")

    def test_orchestrator_passes_through_specs_without_auto_route(self) -> None:
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            from puppetmaster.orchestrator import Orchestrator
            from puppetmaster.store_factory import create_store

            store = create_store("file", state_dir)
            store.init()
            orchestrator = Orchestrator(store)

            spec = WorkerSpec(
                role="explore",
                instruction="map the repo",
                adapter="local",
                payload={"model": "user-chosen-model"},
            )
            job = store.create_job("no auto-route")
            tasks = orchestrator._create_tasks(job, [spec])
            self.assertEqual(tasks[0].adapter, "local")
            self.assertEqual(tasks[0].payload["model"], "user-chosen-model")

            # No routing artifact when auto_route is off.
            from puppetmaster.models import ArtifactType

            artifacts = store.list_artifacts(job.id)
            self.assertFalse(
                any(a.type == ArtifactType.ROUTING for a in artifacts)
            )

    def test_mcp_swarm_config_writer_enables_auto_route_by_default(self) -> None:
        """Regression: MCP start_cursor_swarm / start_swarm must stamp auto_route on
        every generated worker payload, otherwise the orchestrator's auto-routing path
        is silently bypassed for the very entry points users hit from Cursor.
        """
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {"goal": "regression check", "cwd": tmp, "state_dir": str(Path(tmp) / "state")}
            config_path = write_generated_swarm_config(args, ["explore", "audit"], "cursor")
            cfg = json.loads(Path(config_path).read_text())

            self.assertEqual(len(cfg["workers"]), 2)
            for worker in cfg["workers"]:
                payload = worker["payload"]
                self.assertTrue(
                    payload.get("auto_route"),
                    f"auto_route should default to True for MCP swarm role={worker['role']}, "
                    f"got payload={payload}",
                )

    def test_mcp_swarm_config_writer_respects_pinned_model(self) -> None:
        """When the MCP caller pins a model, auto_route should default to off so the
        user's pin is honored.
        """
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {
                "goal": "pin check",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "model": "opus-4-7",
            }
            config_path = write_generated_swarm_config(args, ["audit"], "cursor")
            cfg = json.loads(Path(config_path).read_text())

            payload = cfg["workers"][0]["payload"]
            self.assertEqual(payload["model"], "opus-4-7")
            self.assertNotIn("auto_route", payload)

    def test_mcp_swarm_config_writer_force_route_with_pinned_model(self) -> None:
        """Caller can force auto_route=True even with a pinned model; routing knobs
        (routing_policy, max_cost_usd, min_capability, required_tags) propagate.
        """
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {
                "goal": "force route",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "model": "opus-4-7",
                "auto_route": True,
                "routing_policy": "quality",
                "max_cost_usd": 1.5,
                "min_capability": 90,
                "required_tags": ["vision"],
            }
            config_path = write_generated_swarm_config(args, ["audit"], "cursor")
            cfg = json.loads(Path(config_path).read_text())

            payload = cfg["workers"][0]["payload"]
            self.assertTrue(payload.get("auto_route"))
            self.assertEqual(payload.get("routing_policy"), "quality")
            self.assertEqual(payload.get("max_cost_usd"), 1.5)
            self.assertEqual(payload.get("min_capability"), 90)
            self.assertEqual(payload.get("required_tags"), ["vision"])

    def test_mcp_swarm_config_writer_explicit_opt_out(self) -> None:
        """auto_route=False from MCP must keep generated payloads routing-free."""
        from puppetmaster.mcp_server import write_generated_swarm_config

        with TemporaryDirectory() as tmp:
            args = {
                "goal": "opt out",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "auto_route": False,
            }
            config_path = write_generated_swarm_config(args, ["explore"], "cursor")
            cfg = json.loads(Path(config_path).read_text())

            payload = cfg["workers"][0]["payload"]
            self.assertNotIn("auto_route", payload)


class _FakeUrlopenResponse:
    """Stand-in for a urlopen() context manager returning a fixed body + status."""

    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self._status = status

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self._status

    def read(self) -> bytes:
        return self._body


class OpenAIAdapterTests(unittest.TestCase):
    """Covers the OpenAI adapter end-to-end: happy path, error paths, env handling.

    The adapter calls urllib.request.urlopen against the Chat Completions API.
    We mock that call surface so the tests are hermetic (no network, no API key
    required).
    """

    def _task(self, **payload_overrides: object) -> "Task":
        from puppetmaster.models import Task

        payload = {
            "prompt": "Inspect the auth module and surface findings.",
            "cwd": ".",
            "model": "gpt-5.4-mini",
            "timeout_seconds": 30,
            "disable_codegraph": True,
        }
        payload.update(payload_overrides)
        return Task(
            job_id="job-openai",
            role="openai-explore",
            instruction="Inspect the auth module and surface findings.",
            adapter="openai",
            payload=payload,
        )

    def test_openai_adapter_parses_chat_completion_into_artifacts(self) -> None:
        from puppetmaster.models import ArtifactType

        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "artifacts": [
                                    {
                                        "type": "finding",
                                        "claim": "auth.login swallows DB exceptions.",
                                        "evidence": ["app/auth.py:42"],
                                        "confidence": 0.86,
                                    },
                                    {
                                        "type": "risk",
                                        "risk": "Silent failure on DB outage.",
                                        "mitigation": "Re-raise after logging.",
                                        "evidence": ["app/auth.py:42"],
                                        "confidence": 0.82,
                                    },
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 420, "completion_tokens": 180, "total_tokens": 600},
        }
        fake_response = _FakeUrlopenResponse(json.dumps(response_body))

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            artifacts = OpenAIAdapter().run(self._task(), "goal", "worker-openai")

        # Request built correctly
        request = urlopen.call_args[0][0]
        self.assertEqual(request.method or "POST", "POST")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertIn("Puppetmaster artifact contract", body["messages"][0]["content"])
        self.assertEqual(request.headers["Authorization"], "Bearer sk-test")

        # Artifacts parsed
        types = [a.type for a in artifacts]
        self.assertIn(ArtifactType.VERIFICATION, types)
        self.assertIn(ArtifactType.FINDING, types)
        self.assertIn(ArtifactType.RISK, types)
        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["result"], "passed")
        self.assertEqual(verification.payload["tokens_in"], 420)
        self.assertEqual(verification.payload["tokens_out"], 180)
        self.assertEqual(verification.payload["tokens_total"], 600)
        self.assertEqual(verification.payload["finish_reason"], "stop")
        self.assertEqual(verification.payload["model"], "gpt-5.4-mini")

    def test_openai_adapter_missing_api_key_fails_fast_without_http(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "puppetmaster.adapters.urllib.request.urlopen"
        ) as urlopen:
            artifacts = OpenAIAdapter().run(self._task(), "goal", "worker-openai")

        urlopen.assert_not_called()
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].payload["failure"], "missing_api_key")
        self.assertEqual(artifacts[0].payload["result"], "failed")

    def test_openai_adapter_accepts_api_key_from_payload(self) -> None:
        """payload['openai_api_key'] should override missing env var."""
        fake_response = _FakeUrlopenResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"artifacts":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )
        )
        with patch.dict(os.environ, {}, clear=True), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            artifacts = OpenAIAdapter().run(
                self._task(openai_api_key="sk-payload-override"),
                "goal",
                "worker-openai",
            )
        request = urlopen.call_args[0][0]
        self.assertEqual(request.headers["Authorization"], "Bearer sk-payload-override")
        # No artifacts -> degraded (success but no structured findings)
        self.assertEqual(artifacts[0].payload["result"], "degraded")

    def test_openai_adapter_http_401_maps_to_missing_api_key(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"invalid api key"}}'),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-bogus"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", side_effect=error
        ):
            artifacts = OpenAIAdapter().run(self._task(), "goal", "worker-openai")

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].payload["failure"], "missing_api_key")
        self.assertEqual(artifacts[0].payload["returncode"], 401)

    def test_openai_adapter_http_429_maps_to_rate_limit(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"rate limit reached"),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", side_effect=error
        ):
            artifacts = OpenAIAdapter().run(self._task(), "goal", "worker-openai")

        self.assertEqual(artifacts[0].payload["failure"], "rate_limit")
        self.assertEqual(artifacts[0].payload["returncode"], 429)

    def test_openai_adapter_timeout_surfaces_as_failure(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            artifacts = OpenAIAdapter().run(self._task(), "goal", "worker-openai")

        self.assertEqual(artifacts[0].payload["failure"], "timeout")
        self.assertEqual(artifacts[0].payload["result"], "failed")

    def test_openai_adapter_passes_base_url_organization_and_optional_knobs(self) -> None:
        fake_response = _FakeUrlopenResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"artifacts":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            OpenAIAdapter().run(
                self._task(
                    openai_base_url="https://api.example.invalid/v1",
                    openai_organization="org-puppetmaster",
                    max_output_tokens=2048,
                    temperature=0.4,
                ),
                "goal",
                "worker-openai",
            )

        request = urlopen.call_args[0][0]
        self.assertEqual(
            request.full_url, "https://api.example.invalid/v1/chat/completions"
        )
        # urllib lowercases custom header keys
        header_keys = {k.lower(): v for k, v in request.headers.items()}
        self.assertEqual(header_keys.get("openai-organization"), "org-puppetmaster")
        body = json.loads(request.data.decode("utf-8"))
        # GPT-5+ family uses `max_completion_tokens`; we default to that name.
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["temperature"], 0.4)

    def test_openai_adapter_legacy_max_tokens_opt_in(self) -> None:
        """Some OpenAI-compatible providers still want the old `max_tokens` key.
        legacy_max_tokens=True forces the legacy parameter name.
        """
        fake_response = _FakeUrlopenResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"artifacts":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            OpenAIAdapter().run(
                self._task(max_output_tokens=512, legacy_max_tokens=True),
                "goal",
                "worker-openai",
            )
        body = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 512)
        self.assertNotIn("max_completion_tokens", body)

    def test_openai_adapter_reasoning_effort_passthrough(self) -> None:
        fake_response = _FakeUrlopenResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"artifacts":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            OpenAIAdapter().run(
                self._task(reasoning_effort="high"),
                "goal",
                "worker-openai",
            )
        body = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(body["reasoning_effort"], "high")

    def test_classify_openai_failure_covers_known_buckets(self) -> None:
        self.assertEqual(classify_openai_failure("", 401), "missing_api_key")
        self.assertEqual(classify_openai_failure("", 403), "forbidden")
        self.assertEqual(classify_openai_failure("", 404), "model_unavailable")
        self.assertEqual(classify_openai_failure("", 429), "rate_limit")
        self.assertEqual(classify_openai_failure("", 503), "openai_server_error")
        self.assertEqual(
            classify_openai_failure("Rate limit exceeded", None), "rate_limit"
        )
        self.assertEqual(
            classify_openai_failure("model not found", None), "model_unavailable"
        )
        self.assertEqual(
            classify_openai_failure("maximum context length", None),
            "context_length_exceeded",
        )
        self.assertEqual(classify_openai_failure("weird error", None), "unknown")

    def test_openai_starter_registry_includes_gpt_5_family(self) -> None:
        from puppetmaster.model_registry import starter_registry

        registry = {spec.id: spec for spec in starter_registry()}
        for expected in (
            "openai/gpt-5-5",
            "openai/gpt-5-4",
            "openai/gpt-5-4-mini",
            "openai/gpt-5-4-nano",
        ):
            self.assertIn(expected, registry, f"missing {expected}")

        gpt55 = registry["openai/gpt-5-5"]
        self.assertEqual(gpt55.adapter, "openai")
        self.assertEqual(gpt55.adapter_model_name, "gpt-5.5")
        self.assertEqual(gpt55.input_per_mtok_usd, 5.0)
        self.assertEqual(gpt55.output_per_mtok_usd, 30.0)
        self.assertGreater(gpt55.capability_score, 90)
        self.assertIn("vision", gpt55.tags)

        nano = registry["openai/gpt-5-4-nano"]
        self.assertEqual(nano.adapter_model_name, "gpt-5.4-nano")
        self.assertIn("cheap", nano.tags)
        self.assertLess(nano.capability_score, 60)


class InstallerTests(unittest.TestCase):
    """Tests for :mod:`puppetmaster.installers`.

    These tests intentionally avoid touching the real ``~/.codex/config.toml``
    or ``~/.cursor/mcp.json`` by either (a) writing to a tempdir and
    pointing the Cursor installer at that file, or (b) stubbing the
    ``codex`` CLI with a small shell script that records its argv to
    a file. The handshake test is exercised directly against the real
    MCP server using the in-tree Python — that's the same code path
    real users hit, and it runs fast (sub-second).
    """

    def test_handshake_returns_tool_count_for_working_server(self):
        from puppetmaster.installers import handshake_mcp_server

        result = handshake_mcp_server(timeout_seconds=15.0)
        self.assertTrue(result.ok, msg=f"handshake failed: {result.error}")
        self.assertGreater(
            result.tool_count,
            10,
            msg=f"expected MCP server to advertise plenty of tools, got {result.tool_count}",
        )

    def test_handshake_reports_failure_for_missing_python(self):
        from puppetmaster.installers import handshake_mcp_server

        result = handshake_mcp_server(
            python_executable="/nonexistent/python-that-cannot-possibly-exist",
            timeout_seconds=2.0,
        )
        self.assertFalse(result.ok)
        self.assertIn("not resolvable", result.error)

    def test_install_cursor_writes_entry_into_empty_mcp_json(self):
        from puppetmaster.installers import install_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            result = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")
            data = json.loads(target.read_text("utf-8"))
            self.assertIn("puppetmaster", data["mcpServers"])
            self.assertEqual(
                data["mcpServers"]["puppetmaster"]["command"],
                sys.executable,
            )
            self.assertEqual(
                data["mcpServers"]["puppetmaster"]["args"],
                ["-m", "puppetmaster.mcp_server"],
            )

    def test_install_cursor_preserves_other_servers_and_existing_env(self):
        from puppetmaster.installers import install_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            prior = {
                "mcpServers": {
                    "navdata": {"url": "https://example.com/sse"},
                    "puppetmaster": {
                        "command": "python",
                        "args": ["-m", "puppetmaster.mcp_server"],
                        "env": {
                            "CURSOR_API_KEY": "secret-do-not-touch",
                            "PYTHONPATH": "/some/where",
                        },
                    },
                }
            }
            target.write_text(json.dumps(prior, indent=2), encoding="utf-8")
            result = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")
            data = json.loads(target.read_text("utf-8"))
            self.assertEqual(
                data["mcpServers"]["navdata"]["url"],
                "https://example.com/sse",
                msg="install must not touch unrelated MCP servers",
            )
            self.assertEqual(
                data["mcpServers"]["puppetmaster"]["env"]["CURSOR_API_KEY"],
                "secret-do-not-touch",
                msg="install must preserve existing env keys to avoid wiping API keys",
            )
            self.assertEqual(
                data["mcpServers"]["puppetmaster"]["command"], sys.executable
            )

    def test_install_cursor_idempotent_when_entry_already_matches(self):
        from puppetmaster.installers import install_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            first = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(first.status, "installed")
            mtime_after_first = target.stat().st_mtime_ns
            second = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(second.status, "unchanged")
            self.assertEqual(
                target.stat().st_mtime_ns,
                mtime_after_first,
                msg="unchanged install must not rewrite the file",
            )

    def test_install_cursor_force_rewrites_even_when_match(self):
        from puppetmaster.installers import install_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            time.sleep(0.01)
            result = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                force=True,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")

    def test_install_cursor_dry_run_does_not_write(self):
        from puppetmaster.installers import install_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            result = install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                dry_run=True,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "would_install")
            self.assertFalse(target.exists())

    def test_install_codex_reports_error_when_cli_missing(self):
        from puppetmaster.installers import install_codex_mcp

        result = install_codex_mcp(
            codex_executable="/nonexistent/codex-that-cannot-exist",
            skip_handshake=True,
        )
        self.assertEqual(result.status, "error")
        joined = " ".join(result.messages).lower()
        self.assertIn("not found", joined)

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_invokes_mcp_add_via_stub(self):
        """Stub the `codex` CLI with a shell script and verify argv shape.

        The stub records every invocation to a log file and returns
        ``rc=1`` for ``mcp get`` (so the installer sees "no existing
        entry") and ``rc=0`` for ``mcp add``. After install we read the
        log and assert the installer called ``codex mcp add puppetmaster
        -- <sys.executable> -m puppetmaster.mcp_server``.
        """
        from puppetmaster.installers import install_codex_mcp

        with TemporaryDirectory() as tmp:
            stub_log = Path(tmp) / "codex_calls.log"
            stub = Path(tmp) / "codex"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                'if [ "$1" = "mcp" ] && [ "$2" = "get" ]; then\n'
                '  echo "No MCP server named \\"puppetmaster\\" found." >&2\n'
                "  exit 1\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            result = install_codex_mcp(
                codex_executable=str(stub),
                skip_handshake=True,
            )
            self.assertEqual(
                result.status,
                "installed",
                msg=f"expected install, got {result.status}: {result.messages}",
            )
            log = stub_log.read_text("utf-8")
            self.assertIn("mcp get puppetmaster", log)
            self.assertIn(
                f"mcp add puppetmaster -- {sys.executable} -m puppetmaster.mcp_server",
                log,
                msg=f"unexpected codex args. full log:\n{log}",
            )

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_idempotent_when_entry_already_matches(self):
        """When `codex mcp get` reports an existing matching entry, skip."""
        from puppetmaster.installers import install_codex_mcp

        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "codex"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                'if [ "$1" = "mcp" ] && [ "$2" = "get" ]; then\n'
                '  echo "name: puppetmaster"\n'
                f'  echo "command: {sys.executable}"\n'
                '  echo "args: -m puppetmaster.mcp_server"\n'
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            result = install_codex_mcp(
                codex_executable=str(stub),
                skip_handshake=True,
            )
            self.assertEqual(result.status, "unchanged")


class InstallRulesTests(unittest.TestCase):
    """Tests for :mod:`puppetmaster.rules`.

    These tests exercise the merge-block protocol (the only nontrivial
    part of the rule installer) without touching the user's real
    ``~/.cursor``, ``~/.codex``, or ``~/.claude`` directories by
    targeting a tempdir cwd and explicitly listing targets.
    """

    def test_render_cursor_mdc_has_frontmatter_and_body(self):
        from puppetmaster.rules import render_cursor_mdc

        content = render_cursor_mdc()
        self.assertTrue(content.startswith("---\n"))
        self.assertIn("alwaysApply: true", content)
        self.assertIn("# Puppetmaster orchestration", content)
        self.assertIn("puppetmaster_route_task", content)

    def test_render_agents_block_is_marker_wrapped(self):
        from puppetmaster.rules import BEGIN_MARKER, END_MARKER, render_agents_block

        block = render_agents_block()
        self.assertTrue(block.startswith(BEGIN_MARKER))
        self.assertTrue(block.rstrip().endswith(END_MARKER))
        self.assertIn("# Puppetmaster orchestration", block)

    def test_merge_into_empty_creates_block(self):
        from puppetmaster.rules import merge_block_into_text, render_agents_block

        merged, action = merge_block_into_text("", render_agents_block())
        self.assertEqual(action, "created")
        self.assertEqual(merged, render_agents_block())

    def test_merge_preserves_existing_user_content(self):
        from puppetmaster.rules import (
            BEGIN_MARKER,
            END_MARKER,
            merge_block_into_text,
            render_agents_block,
        )

        existing = (
            "# Project conventions\n\n"
            "- Use TypeScript strict mode\n"
            "- Run tests before committing\n"
        )
        merged, action = merge_block_into_text(existing, render_agents_block())
        self.assertEqual(action, "created")
        self.assertIn("# Project conventions", merged)
        self.assertIn("Use TypeScript strict mode", merged)
        self.assertIn(BEGIN_MARKER, merged)
        self.assertIn(END_MARKER, merged)
        self.assertTrue(
            merged.index("# Project conventions") < merged.index(BEGIN_MARKER),
            msg="user content must come before the puppetmaster block when appended",
        )

    def test_merge_replaces_existing_block_only(self):
        """A re-run should rewrite the puppetmaster block in place, leaving
        surrounding user content untouched and at byte-identical positions."""
        from puppetmaster.rules import (
            BEGIN_MARKER,
            END_MARKER,
            merge_block_into_text,
            render_agents_block,
        )

        stale_block = (
            f"{BEGIN_MARKER}\n"
            "stale content from a previous version\n"
            f"{END_MARKER}\n"
        )
        before = "# header user content\n\n" + stale_block + "\n# footer user content\n"
        merged, action = merge_block_into_text(before, render_agents_block())
        self.assertEqual(action, "replaced")
        self.assertIn("# header user content", merged)
        self.assertIn("# footer user content", merged)
        self.assertNotIn("stale content from a previous version", merged)
        self.assertIn("puppetmaster_route_task", merged)

    def test_merge_is_idempotent_when_block_matches(self):
        from puppetmaster.rules import merge_block_into_text, render_agents_block

        full_block = render_agents_block()
        with_existing = "# header\n\n" + full_block
        merged, action = merge_block_into_text(with_existing, full_block)
        self.assertEqual(action, "unchanged")
        self.assertEqual(merged, with_existing)

    def test_install_rules_writes_cursor_and_agents_in_tempdir(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            result = install_rules(cwd=cwd, targets=["cursor", "agents"])
            self.assertEqual(result.overall_status, "installed")
            self.assertTrue((cwd / ".cursor" / "rules" / "puppetmaster.mdc").is_file())
            self.assertTrue((cwd / "AGENTS.md").is_file())
            mdc = (cwd / ".cursor" / "rules" / "puppetmaster.mdc").read_text("utf-8")
            self.assertIn("alwaysApply: true", mdc)
            agents = (cwd / "AGENTS.md").read_text("utf-8")
            self.assertIn("puppetmaster:rules:begin", agents)

    def test_install_rules_dry_run_writes_nothing(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_rules(cwd=cwd, targets=["cursor", "agents"], dry_run=True)
            self.assertEqual(result.overall_status, "would_install")
            self.assertFalse((cwd / ".cursor" / "rules" / "puppetmaster.mdc").exists())
            self.assertFalse((cwd / "AGENTS.md").exists())

    def test_install_rules_idempotent_on_rerun(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            install_rules(cwd=cwd, targets=["cursor", "agents"])
            agents_mtime = (cwd / "AGENTS.md").stat().st_mtime_ns
            mdc_mtime = (cwd / ".cursor" / "rules" / "puppetmaster.mdc").stat().st_mtime_ns
            time.sleep(0.01)
            result = install_rules(cwd=cwd, targets=["cursor", "agents"])
            self.assertEqual(result.overall_status, "unchanged")
            self.assertEqual(
                (cwd / "AGENTS.md").stat().st_mtime_ns,
                agents_mtime,
                msg="unchanged run must not rewrite AGENTS.md",
            )
            self.assertEqual(
                (cwd / ".cursor" / "rules" / "puppetmaster.mdc").stat().st_mtime_ns,
                mdc_mtime,
                msg="unchanged run must not rewrite .mdc",
            )

    def test_install_rules_preserves_unrelated_user_content_on_rerun(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            install_rules(cwd=cwd, targets=["agents"])
            agents_path = cwd / "AGENTS.md"
            existing = agents_path.read_text("utf-8")
            agents_path.write_text(
                "# User wrote this AFTER puppetmaster installed\n\n"
                + existing
                + "\n# Trailing user content too\n",
                encoding="utf-8",
            )
            result = install_rules(cwd=cwd, targets=["agents"], force=True)
            self.assertIn(result.overall_status, {"installed", "unchanged"})
            final = agents_path.read_text("utf-8")
            self.assertIn("# User wrote this AFTER puppetmaster installed", final)
            self.assertIn("# Trailing user content too", final)
            self.assertIn("puppetmaster:rules:begin", final)
            self.assertEqual(
                final.count("puppetmaster:rules:begin"),
                1,
                msg="force-rerun must not duplicate the puppetmaster block",
            )

    def test_doctor_agent_rules_check_warns_when_mcp_present_but_no_rules(self):
        """Doctor should nudge the user when MCP is wired but rules are missing."""
        from puppetmaster.diagnostics import _agent_rules_check

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".cursor").mkdir()
            (cwd / ".cursor" / "mcp.json").write_text(
                '{"mcpServers": {"puppetmaster": {"command": "python"}}}',
                encoding="utf-8",
            )
            check = _agent_rules_check(cwd)
            self.assertEqual(check.status, "warn")
            self.assertIn("install-rules", check.detail)

    def test_doctor_agent_rules_check_ok_when_rule_present(self):
        from puppetmaster.diagnostics import _agent_rules_check

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".cursor" / "rules").mkdir(parents=True)
            (cwd / ".cursor" / "rules" / "puppetmaster.mdc").write_text(
                "---\nalwaysApply: true\n---\n",
                encoding="utf-8",
            )
            check = _agent_rules_check(cwd)
            self.assertEqual(check.status, "ok")


class ModelBillingFieldTests(unittest.TestCase):
    def test_billing_defaults_to_unknown_and_validates(self) -> None:
        from puppetmaster.model_registry import ModelSpec

        spec = ModelSpec(id="m", adapter="cursor", adapter_model_name="x")
        self.assertEqual(spec.billing, "unknown")
        self.assertFalse(spec.is_plan_billed)

        plan = ModelSpec(id="m", adapter="cursor", adapter_model_name="x", billing="plan")
        self.assertTrue(plan.is_plan_billed)

        with self.assertRaises(ValueError):
            ModelSpec(id="m", adapter="cursor", adapter_model_name="x", billing="free")

    def test_billing_survives_json_round_trip_and_drops_default(self) -> None:
        from puppetmaster.model_registry import (
            ModelSpec,
            load_registry,
            save_registry,
        )

        specs = [
            ModelSpec(id="a", adapter="cursor", adapter_model_name="x", billing="plan"),
            ModelSpec(id="b", adapter="openai", adapter_model_name="y", billing="api"),
            ModelSpec(id="c", adapter="claude-code", adapter_model_name="z"),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(specs, path)
            raw = path.read_text(encoding="utf-8")
            # default "unknown" should be dropped from the serialized form.
            self.assertNotIn('"billing": "unknown"', raw)
            self.assertIn('"billing": "plan"', raw)

            loaded = {s.id: s for s in load_registry(path)}
            self.assertEqual(loaded["a"].billing, "plan")
            self.assertEqual(loaded["b"].billing, "api")
            self.assertEqual(loaded["c"].billing, "unknown")

    def test_starter_registry_tags_cursor_as_plan_billed(self) -> None:
        from puppetmaster.model_registry import starter_registry

        specs = {s.id: s for s in starter_registry()}
        cursor_specs = [s for s in specs.values() if s.adapter == "cursor"]
        self.assertTrue(cursor_specs)
        self.assertTrue(all(s.billing == "plan" for s in cursor_specs))
        openai_specs = [s for s in specs.values() if s.adapter == "openai"]
        self.assertTrue(all(s.billing == "api" for s in openai_specs))


class BillingAwareRoutingTests(unittest.TestCase):
    def _mixed_registry(self):
        from puppetmaster.model_registry import ModelSpec

        # A plan-billed model and an api-billed model at the SAME capability and
        # the same cost. prefer_plan_billed should break the tie toward plan.
        return [
            ModelSpec(
                id="plan-mid",
                adapter="cursor",
                adapter_model_name="plan-v1",
                capability_score=80,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                billing="plan",
                tags=["balanced"],
            ),
            ModelSpec(
                id="api-mid",
                adapter="openai",
                adapter_model_name="api-v1",
                capability_score=80,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                billing="api",
                tags=["balanced"],
            ),
        ]

    def test_prefer_plan_billed_breaks_ties_toward_plan(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="implement a feature", role="implement", explicit_min_capability=78
        )
        decision = route_task(signal, self._mixed_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "plan-mid")
        self.assertEqual(decision.to_artifact_payload()["billing"], "plan")

    def test_prefer_plan_billed_off_falls_back_to_capability_tiebreak(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="implement a feature",
            role="implement",
            explicit_min_capability=78,
            prefer_plan_billed=False,
        )
        # With no plan preference and equal cost+capability, the tie falls to the
        # first by capability ordering; both are valid, but plan must NOT be forced.
        decision = route_task(signal, self._mixed_registry(), policy="balanced")
        self.assertIn(decision.model.id, {"plan-mid", "api-mid"})

    def test_allow_api_billing_false_blocks_api_models(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        # Registry has ONLY api-billed models; plan-only must refuse rather than
        # silently spend out-of-pocket.
        api_only = [
            ModelSpec(
                id="api-frontier",
                adapter="openai",
                adapter_model_name="api-v1",
                capability_score=96,
                input_per_mtok_usd=5.0,
                output_per_mtok_usd=30.0,
                billing="api",
            )
        ]
        signal = TaskSignals(
            instruction="hard audit",
            role="audit",
            allow_api_billing=False,
        )
        with self.assertRaises(NoEligibleModelError):
            route_task(signal, api_only, policy="balanced")

    def test_allow_api_billing_false_keeps_plan_models(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="implement", role="implement", explicit_min_capability=78, allow_api_billing=False
        )
        decision = route_task(signal, self._mixed_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "plan-mid")
        rejected = {spec.id: reason for spec, reason in decision.rejected}
        self.assertIn("api-mid", rejected)
        self.assertIn("api billing disabled", rejected["api-mid"])


class PlatformBillingDetectionTests(unittest.TestCase):
    def test_cursor_billing_keyed_is_plan(self) -> None:
        from puppetmaster.platform_billing import detect_cursor_billing

        s = detect_cursor_billing(env={"CURSOR_API_KEY": "k"})
        self.assertEqual(s.billing, "plan")
        self.assertTrue(s.healthy)

        s2 = detect_cursor_billing(env={})
        self.assertEqual(s2.billing, "unknown")
        self.assertFalse(s2.healthy)

    def test_claude_billing_api_key_vs_oauth_vs_none(self) -> None:
        from puppetmaster.platform_billing import detect_claude_billing

        import json as _json

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            # api key wins.
            s = detect_claude_billing(env={"ANTHROPIC_API_KEY": "k"}, home=home)
            self.assertEqual(s.billing, "api")
            self.assertTrue(s.healthy)
            # A real oauthAccount (uuid/email) -> plan, with seat/org in detail.
            (home / ".claude.json").write_text(
                _json.dumps(
                    {
                        "oauthAccount": {
                            "accountUuid": "abc-123",
                            "emailAddress": "me@example.com",
                            "seatTier": "max",
                            "organizationName": "Acme",
                        }
                    }
                ),
                encoding="utf-8",
            )
            s2 = detect_claude_billing(env={}, home=home)
            self.assertEqual(s2.billing, "plan")
            self.assertTrue(s2.healthy)
            self.assertIn("seat_tier:max", s2.evidence)
            self.assertIn("Acme", s2.detail)

        # A config-only ~/.claude.json (no oauthAccount) is NOT proof of auth —
        # it survives a logout, so it must read as unauthenticated.
        with TemporaryDirectory() as tmp_cfg:
            home = Path(tmp_cfg)
            (home / ".claude.json").write_text("{}", encoding="utf-8")
            s_cfg = detect_claude_billing(env={}, home=home)
            self.assertEqual(s_cfg.billing, "unknown")
            self.assertFalse(s_cfg.healthy)

        # Credentials-file fallback -> plan.
        with TemporaryDirectory() as tmp_cred:
            home = Path(tmp_cred)
            (home / ".claude").mkdir()
            (home / ".claude" / ".credentials.json").write_text(
                '{"claudeAiOauth": {"accessToken": "x"}}', encoding="utf-8"
            )
            s_cred = detect_claude_billing(env={}, home=home)
            self.assertEqual(s_cred.billing, "plan")

        with TemporaryDirectory() as tmp2:
            s3 = detect_claude_billing(env={}, home=Path(tmp2))
            self.assertEqual(s3.billing, "unknown")
            self.assertFalse(s3.healthy)

    def test_codex_billing_reads_auth_file_first(self) -> None:
        import json as _json

        from puppetmaster.platform_billing import detect_codex_billing

        # auth_mode=apikey -> api, no subprocess.
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".codex").mkdir()
            (home / ".codex" / "auth.json").write_text(
                _json.dumps({"OPENAI_API_KEY": "sk-x", "auth_mode": "apikey"}),
                encoding="utf-8",
            )

            def _boom(cmd):
                raise AssertionError("subprocess must not run when auth.json present")

            s = detect_codex_billing(run=_boom, home=home)
            self.assertEqual(s.billing, "api")
            self.assertIn("codex_auth:apikey", s.evidence)

        # auth_mode=chatgpt (or tokens block) -> plan.
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".codex").mkdir()
            (home / ".codex" / "auth.json").write_text(
                _json.dumps({"auth_mode": "chatgpt", "tokens": {"access": "x"}}),
                encoding="utf-8",
            )
            s = detect_codex_billing(run=lambda cmd: (1, "", ""), home=home)
            self.assertEqual(s.billing, "plan")
            self.assertIn("codex_auth:chatgpt", s.evidence)

    def test_codex_billing_falls_back_to_login_status(self) -> None:
        from puppetmaster.platform_billing import detect_codex_billing

        # No auth.json -> fall back to parsing `codex login status`.
        with TemporaryDirectory() as tmp:
            home = Path(tmp)  # empty: no ~/.codex/auth.json
            api = detect_codex_billing(
                run=lambda cmd: (0, "Logged in using an API key - sk-***", ""), home=home
            )
            self.assertEqual(api.billing, "api")
            chatgpt = detect_codex_billing(
                run=lambda cmd: (0, "Logged in using ChatGPT", ""), home=home
            )
            self.assertEqual(chatgpt.billing, "plan")
            missing = detect_codex_billing(
                run=lambda cmd: (127, "", "command not found"), home=home
            )
            self.assertEqual(missing.billing, "unknown")
            self.assertFalse(missing.healthy)
            out = detect_codex_billing(run=lambda cmd: (1, "Not logged in", ""), home=home)
            self.assertEqual(out.billing, "unknown")
            self.assertFalse(out.healthy)

    def test_detect_adapter_billing_dispatch_and_unknown(self) -> None:
        from puppetmaster.platform_billing import detect_adapter_billing

        s = detect_adapter_billing("cursor", env={"CURSOR_API_KEY": "k"})
        self.assertEqual(s.billing, "plan")
        # unknown adapter is benign pass-through.
        u = detect_adapter_billing("mystery-adapter")
        self.assertEqual(u.adapter, "mystery-adapter")
        self.assertTrue(u.healthy)
        self.assertEqual(u.billing, "unknown")


class CursorDiscoveryTests(unittest.TestCase):
    def _ok_run(self, models):
        import json as _json

        def run(command, env):
            return (0, _json.dumps({"ok": True, "models": models}), "")

        return run

    def test_fetch_catalog_success(self) -> None:
        from puppetmaster.cursor_discovery import fetch_cursor_catalog

        run = self._ok_run([{"id": "gpt-5.5", "displayName": "GPT 5.5"}, {"id": "bad"}])
        catalog = fetch_cursor_catalog(env={"CURSOR_API_KEY": "k"}, run=run)
        ids = {m["id"] for m in catalog}
        self.assertEqual(ids, {"gpt-5.5", "bad"})

    def test_fetch_catalog_requires_key(self) -> None:
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            fetch_cursor_catalog,
        )

        with self.assertRaises(CursorDiscoveryError):
            fetch_cursor_catalog(env={}, run=self._ok_run([]))

    def test_fetch_catalog_handles_failures(self) -> None:
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            fetch_cursor_catalog,
        )

        with self.assertRaises(CursorDiscoveryError):
            fetch_cursor_catalog(
                env={"CURSOR_API_KEY": "k"}, run=lambda c, e: (1, "", "boom")
            )
        with self.assertRaises(CursorDiscoveryError):
            fetch_cursor_catalog(
                env={"CURSOR_API_KEY": "k"}, run=lambda c, e: (0, "not json", "")
            )
        with self.assertRaises(CursorDiscoveryError):
            fetch_cursor_catalog(
                env={"CURSOR_API_KEY": "k"}, run=lambda c, e: (0, '{"ok": false}', "")
            )

    def test_catalog_to_specs_overlays_and_seeds(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(
                id="cursor/gpt-5-5",
                adapter="cursor",
                adapter_model_name="gpt-5.5",
                capability_score=92,
                tags=["cursor", "frontier"],
            )
        ]
        catalog = [
            {"id": "gpt-5.5", "displayName": "GPT 5.5"},
            {"id": "new-model", "displayName": "New Model"},
        ]
        specs = {s.adapter_model_name: s for s in catalog_to_specs(catalog, existing)}
        # overlay keeps capability + id, forces plan billing, adds discovered tag.
        self.assertEqual(specs["gpt-5.5"].capability_score, 92)
        self.assertEqual(specs["gpt-5.5"].billing, "plan")
        self.assertIn("discovered", specs["gpt-5.5"].tags)
        # unknown model gets a seeded plan-billed spec.
        self.assertEqual(specs["new-model"].billing, "plan")
        self.assertEqual(specs["new-model"].id, "cursor/new-model")

    def test_catalog_inherits_capability_cross_adapter(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        # A native claude-code frontier entry; the same model is exposed by the
        # Cursor plan. The discovered cursor spec should inherit cap 99.
        existing = [
            ModelSpec(
                id="claude-code/opus-4-8",
                adapter="claude-code",
                adapter_model_name="claude-opus-4-8",
                capability_score=99,
                tags=["frontier", "long-context"],
            )
        ]
        catalog = [{"id": "claude-opus-4-8", "displayName": "Claude Opus 4.8"}]
        spec = catalog_to_specs(catalog, existing)[0]
        self.assertEqual(spec.capability_score, 99)
        self.assertEqual(spec.billing, "plan")
        self.assertEqual(spec.adapter, "cursor")
        self.assertIn("frontier", spec.tags)

    def test_merge_drops_stale_and_preserves_non_cursor(self) -> None:
        from puppetmaster.cursor_discovery import merge_catalog_into_registry
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(id="cursor/old", adapter="cursor", adapter_model_name="old-v1"),
            ModelSpec(id="claude/x", adapter="claude-code", adapter_model_name="x"),
        ]
        catalog = [{"id": "fresh-v1", "displayName": "Fresh"}]
        merged, report = merge_catalog_into_registry(existing, catalog)
        ids = {s.id for s in merged}
        self.assertIn("claude/x", ids)  # non-cursor preserved
        self.assertNotIn("cursor/old", ids)  # stale cursor dropped
        self.assertIn("old-v1", report["dropped_stale_cursor_models"])
        self.assertIn("fresh-v1", report["added"])

    def test_model_in_catalog(self) -> None:
        from puppetmaster.cursor_discovery import model_in_catalog

        catalog = [{"id": "a"}, {"id": "b"}]
        self.assertTrue(model_in_catalog("a", catalog))
        self.assertFalse(model_in_catalog("zzz", catalog))


class PreflightTests(unittest.TestCase):
    def test_ready_for_plan_billed_adapter(self) -> None:
        from puppetmaster.preflight import preflight_check

        r = preflight_check("cursor", env={"CURSOR_API_KEY": "k"})
        self.assertTrue(r.ok)
        self.assertEqual(r.billing, "plan")

    def test_blocks_unauthenticated_adapter(self) -> None:
        from puppetmaster.preflight import preflight_check

        r = preflight_check("cursor", env={})
        self.assertFalse(r.ok)
        self.assertIn("not ready", r.reason)

    def test_blocks_api_when_disallowed(self) -> None:
        from puppetmaster.preflight import preflight_check

        r = preflight_check(
            "openai", allow_api_billing=False, env={"OPENAI_API_KEY": "k"}
        )
        self.assertFalse(r.ok)
        self.assertIn("api billing disabled", r.reason)

    def test_cursor_model_not_in_catalog_blocks(self) -> None:
        from puppetmaster.preflight import preflight_check

        r = preflight_check(
            "cursor",
            "ghost-model",
            env={"CURSOR_API_KEY": "k"},
            catalog_fetcher=lambda: [{"id": "real-model"}],
        )
        self.assertFalse(r.ok)
        self.assertIn("not in the Cursor plan catalog", r.reason)

    def test_cursor_catalog_unavailable_degrades_to_ok(self) -> None:
        from puppetmaster.cursor_discovery import CursorDiscoveryError
        from puppetmaster.preflight import preflight_check

        def boom():
            raise CursorDiscoveryError("offline")

        r = preflight_check(
            "cursor", "any", env={"CURSOR_API_KEY": "k"}, catalog_fetcher=boom
        )
        self.assertTrue(r.ok)
        self.assertIn("catalog unverified", r.reason)


class StitcherAlertTests(unittest.TestCase):
    def _verification(self, *, failure=None, result="passed"):
        from puppetmaster.models import Artifact, ArtifactType

        payload = {"check": "did the thing", "result": result, "adapter": "claude-code"}
        if failure:
            payload["failure"] = failure
        return Artifact(
            job_id="j",
            task_id="t",
            type=ArtifactType.VERIFICATION,
            created_by="w",
            payload=payload,
            confidence=0.55,
            evidence=["adapter:claude-code"],
        )

    def test_billing_failure_surfaces_as_alert(self) -> None:
        from puppetmaster.stitcher import Stitcher

        summary = Stitcher(None)._render_summary(
            "T", "goal", [self._verification(failure="billing_or_quota", result="failed")], []
        )
        self.assertIn("## Alerts (action required)", summary)
        self.assertIn("billing_or_quota", summary)
        self.assertIn("claude-code", summary)
        # alert appears before the verification section.
        self.assertLess(summary.index("Alerts"), summary.index("## Verification"))

    def test_clean_run_has_no_alert_section(self) -> None:
        from puppetmaster.stitcher import Stitcher

        summary = Stitcher(None)._render_summary(
            "T", "goal", [self._verification(result="passed")], []
        )
        self.assertNotIn("## Alerts", summary)


class DashboardTests(unittest.TestCase):
    def _seed_store(self, tmp):
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        store = SwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        job = store.create_job("dashboard demo goal")
        task = Task(
            job_id=job.id,
            role="implement",
            instruction="do the thing",
            adapter="cursor",
            status=TaskStatus.COMPLETE,
            payload={"model": "gpt-5.5"},
            attempts=2,
        )
        store.save_task(task)
        store.save_artifact(
            Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                created_by="worker-implement", payload={"claim": "found the bug"},
                confidence=0.95, evidence=["foo.py"],
            )
        )
        store.save_artifact(
            Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router",
                payload={
                    "model_id": "cursor/gpt-5-5", "adapter": "cursor", "policy": "balanced",
                    "estimated_cost_usd": 0.0,
                },
                confidence=1.0, evidence=["policy:balanced"],
            )
        )
        store.save_artifact(
            Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router-fallback",
                payload={
                    "model_id": "cursor/gpt-5-5", "adapter": "cursor", "policy": "balanced",
                    "reason": "policy=balanced: cheapest sufficient model", "estimated_cost_usd": 0.0,
                },
                confidence=0.9, evidence=["fallback:from=claude-code"],
            )
        )
        store.save_artifact(
            Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                created_by="worker-implement-old",
                payload={"check": "x", "result": "failed", "failure": "billing_or_quota", "adapter": "claude-code"},
                confidence=0.55, evidence=["adapter:claude-code"],
            )
        )
        return store, job

    def test_build_job_snapshot(self) -> None:
        from puppetmaster.dashboard import build_job_snapshot

        with TemporaryDirectory() as tmp:
            store, job = self._seed_store(tmp)
            snap = build_job_snapshot(store, job.id)

            self.assertEqual(snap["job"]["id"], job.id)
            self.assertEqual(snap["job"]["goal"], "dashboard demo goal")
            self.assertEqual(len(snap["tasks"]), 1)
            self.assertEqual(snap["tasks"][0]["model"], "gpt-5.5")
            self.assertEqual(snap["tasks"][0]["attempts"], 2)
            self.assertEqual(snap["counts"]["finding"], 1)
            self.assertEqual(snap["artifacts"]["finding"][0]["statement"], "found the bug")
            # the router-fallback routing artifact is surfaced as a reroute
            self.assertEqual(len(snap["reroutes"]), 1)
            self.assertIn("balanced", snap["reroutes"][0]["reason"])
            # the billing failure surfaces as an alert
            self.assertTrue(any("billing_or_quota" in a for a in snap["alerts"]))

    def test_dashboard_http_serves_index_and_job(self) -> None:
        import threading
        import urllib.request

        from puppetmaster.dashboard import serve

        with TemporaryDirectory() as tmp:
            store, job = self._seed_store(tmp)
            httpd = serve(
                Path(tmp) / ".puppetmaster",
                backend="file",
                host="127.0.0.1",
                port=0,
                open_browser=False,
                serve_forever=False,
            )
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
                    self.assertEqual(r.status, 200)
                    self.assertIn(b"Puppetmaster", r.read())
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/job?id={job.id}"
                ) as r:
                    data = json.loads(r.read())
                    self.assertEqual(data["job"]["id"], job.id)
                    self.assertEqual(len(data["tasks"]), 1)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/jobs") as r:
                    jobs = json.loads(r.read())
                    self.assertTrue(any(j["id"] == job.id for j in jobs))
            finally:
                httpd.shutdown()

    def test_dashboard_rejects_traversal_job_id(self) -> None:
        """/api/job must reject ids that aren't plain job ids before they reach
        the store path join (no `..` / absolute-path traversal)."""
        import threading
        import urllib.error
        import urllib.parse
        import urllib.request

        from puppetmaster.dashboard import serve

        with TemporaryDirectory() as tmp:
            self._seed_store(tmp)
            httpd = serve(
                Path(tmp) / ".puppetmaster", backend="file", host="127.0.0.1",
                port=0, open_browser=False, serve_forever=False,
            )
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                evil = urllib.parse.quote("../../../../etc/passwd", safe="")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job?id={evil}")
                    self.fail("expected HTTP 400 for traversal id")
                except urllib.error.HTTPError as exc:
                    self.assertEqual(exc.code, 400)
                    self.assertIn(b"invalid id", exc.read())
            finally:
                httpd.shutdown()

    def test_cost_rollup_tolerates_non_numeric_cost(self) -> None:
        """A ROUTING artifact with a non-numeric estimated_cost_usd must not
        500 the snapshot (validate only requires model_id/adapter/policy)."""
        from puppetmaster.dashboard import build_job_snapshot
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("cost robustness")
            store.save_artifact(
                Artifact(
                    job_id=job.id, task_id="t", type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={"model_id": "m", "adapter": "cursor", "policy": "balanced",
                             "estimated_cost_usd": "not-a-number"},
                    confidence=0.9, evidence=["x"],
                )
            )
            snap = build_job_snapshot(store, job.id)
            self.assertEqual(snap["cost"]["total_estimated_cost_usd"], 0.0)


class EnsurePlanCatalogTests(unittest.TestCase):
    """First-run guarantee: auto-routed work always has a plan-billed frontier
    to land on, so it never falls off to a per-token / depleted account."""

    def _status(self, healthy=True, billing="plan"):
        from puppetmaster.platform_billing import BillingStatus

        return BillingStatus(
            adapter="cursor", billing=billing, healthy=healthy, detail="t", evidence=[]
        )

    def _registry_path(self, tmp):
        from puppetmaster.model_registry import save_registry, starter_registry

        path = Path(tmp) / "models.json"
        save_registry(starter_registry(), path)
        return path

    def test_skips_when_plan_frontier_already_present(self) -> None:
        from puppetmaster.cursor_discovery import ensure_cursor_plan_catalog
        from puppetmaster.model_registry import ModelSpec, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [ModelSpec(id="cursor/claude-opus-4-8", adapter="cursor",
                           adapter_model_name="claude-opus-4-8", capability_score=99,
                           input_per_mtok_usd=0.0, output_per_mtok_usd=0.0,
                           context_window=200000, billing="plan")],
                path,
            )
            called = {"fetch": False}

            def fetch():
                called["fetch"] = True
                return []

            report = ensure_cursor_plan_catalog(
                path, billing_detector=lambda: self._status(), catalog_fetcher=fetch
            )
            self.assertEqual(report["action"], "skip")
            self.assertEqual(report["reason"], "plan_frontier_present")
            self.assertFalse(called["fetch"])  # no network when frontier exists

    def test_skips_when_cursor_unauthenticated(self) -> None:
        from puppetmaster.cursor_discovery import ensure_cursor_plan_catalog

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            report = ensure_cursor_plan_catalog(
                path,
                billing_detector=lambda: self._status(healthy=False, billing="unknown"),
                catalog_fetcher=lambda: [{"id": "claude-opus-4-8"}],
            )
            self.assertEqual(report["action"], "skip")
            self.assertEqual(report["reason"], "cursor_unauthenticated")

    def test_discovers_when_authenticated_and_no_frontier(self) -> None:
        from puppetmaster.cursor_discovery import (
            ensure_cursor_plan_catalog,
            has_plan_frontier,
        )
        from puppetmaster.model_registry import load_registry, read_discovery_meta

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            self.assertFalse(has_plan_frontier(load_registry(path)))  # starter is thin
            report = ensure_cursor_plan_catalog(
                path,
                billing_detector=lambda: self._status(),
                catalog_fetcher=lambda: [
                    {"id": "claude-opus-4-8", "displayName": "Opus 4.8"},
                    {"id": "gpt-5.5", "displayName": "GPT-5.5"},
                ],
            )
            self.assertEqual(report["action"], "discovered")
            merged = load_registry(path)
            # The plan now carries a frontier model (cap inherited from claude-code/opus-4-8).
            self.assertTrue(has_plan_frontier(merged))
            self.assertTrue(
                any(s.adapter == "cursor" and s.adapter_model_name == "claude-opus-4-8"
                    and s.billing == "plan" for s in merged)
            )
            self.assertTrue(read_discovery_meta(path).get("cursor"))

    def test_does_not_reenumerate_after_prior_discovery(self) -> None:
        from puppetmaster.cursor_discovery import ensure_cursor_plan_catalog
        from puppetmaster.model_registry import write_discovery_meta

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            write_discovery_meta("cursor", 3, path)  # we've tried before
            called = {"fetch": False}

            def fetch():
                called["fetch"] = True
                return [{"id": "claude-opus-4-8"}]

            report = ensure_cursor_plan_catalog(
                path, billing_detector=lambda: self._status(), catalog_fetcher=fetch
            )
            self.assertEqual(report["action"], "skip")
            self.assertEqual(report["reason"], "already_discovered")
            self.assertFalse(called["fetch"])

    def test_discovery_failure_degrades_to_unavailable(self) -> None:
        from puppetmaster.cursor_discovery import ensure_cursor_plan_catalog

        def boom():
            raise RuntimeError("cursor sdk offline")

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            report = ensure_cursor_plan_catalog(
                path, billing_detector=lambda: self._status(), catalog_fetcher=boom
            )
            self.assertEqual(report["action"], "unavailable")
            self.assertIn("offline", report["error"])


class AwaitJobTests(unittest.TestCase):
    def test_await_returns_when_already_terminal(self) -> None:
        from puppetmaster.cli import await_job_state
        from puppetmaster.models import JobStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("do work")
            store.update_job_status(job.id, JobStatus.COMPLETE)
            state = await_job_state(store, job.id, timeout_seconds=2.0)
            self.assertTrue(state["terminal"])
            self.assertFalse(state["timed_out"])
            self.assertEqual(state["status"], "complete")

    def test_await_times_out_for_running_job(self) -> None:
        from puppetmaster.cli import await_job_state
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("never finishes")
            state = await_job_state(store, job.id, timeout_seconds=0.3, poll_interval_seconds=0.05)
            self.assertFalse(state["terminal"])
            self.assertTrue(state["timed_out"])


class PreflightDispatchGateTests(unittest.TestCase):
    def _task(self, adapter="cursor", payload=None):
        from puppetmaster.models import Task

        return Task(
            job_id="j",
            role="implement",
            instruction="do work",
            adapter=adapter,
            payload=payload or {},
        )

    def test_preflight_skipped_for_local_adapter(self) -> None:
        from puppetmaster.workers import LocalWorker

        worker = LocalWorker("implement")
        self.assertIsNone(worker._preflight(self._task(adapter="local")))
        self.assertIsNone(worker._preflight(self._task(adapter="shell")))

    def test_preflight_skipped_when_opted_out(self) -> None:
        from puppetmaster.workers import LocalWorker

        worker = LocalWorker("implement")
        task = self._task(adapter="cursor", payload={"skip_preflight": True})
        self.assertIsNone(worker._preflight(task))

    def test_preflight_blocks_unhealthy_adapter(self) -> None:
        from unittest.mock import patch

        from puppetmaster.preflight import PreflightResult
        from puppetmaster.workers import LocalWorker

        blocked = PreflightResult(
            ok=False,
            adapter="cursor",
            model="default",
            billing="unknown",
            reason="adapter not ready: CURSOR_API_KEY is not set",
            evidence=["cursor_api_key:missing", "preflight:unhealthy"],
        )
        worker = LocalWorker("implement")
        with patch("puppetmaster.preflight.preflight_check", return_value=blocked):
            artifact = worker._preflight(self._task(adapter="cursor"))
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.payload["result"], "blocked")
        self.assertEqual(artifact.payload["failure"], "preflight_blocked")

    def test_run_returns_blocked_artifact_and_failed_status(self) -> None:
        from unittest.mock import patch

        from puppetmaster.models import TaskStatus
        from puppetmaster.preflight import PreflightResult
        from puppetmaster.workers import LocalWorker

        blocked = PreflightResult(
            ok=False,
            adapter="claude-code",
            model=None,
            billing="api",
            reason="api billing disabled",
            evidence=["preflight:api_blocked"],
        )
        worker = LocalWorker("implement")
        with patch("puppetmaster.preflight.preflight_check", return_value=blocked):
            run, artifacts = worker.run(self._task(adapter="claude-code"), "goal")
        self.assertEqual(run.status, TaskStatus.FAILED)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].payload["failure"], "preflight_blocked")

    def test_run_dispatches_when_preflight_passes(self) -> None:
        from unittest.mock import patch

        from puppetmaster.models import ArtifactType, TaskStatus
        from puppetmaster.preflight import PreflightResult
        from puppetmaster.workers import LocalWorker

        ok = PreflightResult(
            ok=True, adapter="cursor", model="default", billing="plan", reason="ready"
        )
        sentinel = [
            __import__("puppetmaster.models", fromlist=["Artifact"]).Artifact(
                job_id="j",
                task_id="t",
                type=ArtifactType.FINDING,
                created_by="w",
                payload={"claim": "did it"},
                confidence=0.9,
                evidence=["adapter:cursor"],
            )
        ]

        class _FakeAdapter:
            def run(self, task, goal, worker_id):
                return sentinel

        worker = LocalWorker("implement")
        with patch("puppetmaster.preflight.preflight_check", return_value=ok), patch(
            "puppetmaster.workers.get_adapter", return_value=_FakeAdapter()
        ):
            run, artifacts = worker.run(self._task(adapter="cursor"), "goal")
        self.assertEqual(run.status, TaskStatus.COMPLETE)
        self.assertIs(artifacts, sentinel)


class TelemetryTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        from puppetmaster.telemetry import telemetry_enabled

        self.assertFalse(telemetry_enabled({}))

    def test_enable_and_override_logic(self) -> None:
        from puppetmaster.telemetry import telemetry_enabled

        self.assertTrue(telemetry_enabled({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318"}))
        self.assertTrue(telemetry_enabled({"OTEL_TRACES_EXPORTER": "console"}))
        # explicit off wins even with an endpoint set.
        self.assertFalse(
            telemetry_enabled(
                {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4318", "PUPPETMASTER_OTEL_ENABLED": "false"}
            )
        )
        # explicit on wins with nothing else.
        self.assertTrue(telemetry_enabled({"PUPPETMASTER_OTEL_ENABLED": "true"}))

    def test_record_job_trace_noop_when_disabled(self) -> None:
        from puppetmaster.models import Job
        from puppetmaster.telemetry import record_job_trace

        self.assertFalse(record_job_trace(Job(goal="g"), [], [], env={}))

    def test_build_job_trace_shape(self) -> None:
        from puppetmaster.models import (
            Artifact,
            ArtifactType,
            Job,
            Task,
            TaskStatus,
        )
        from puppetmaster.telemetry import build_job_trace

        job = Job(goal="audit the thing")
        t_ok = Task(
            job_id=job.id, role="explore", instruction="x", adapter="cursor",
            status=TaskStatus.COMPLETE,
        )
        t_bad = Task(
            job_id=job.id, role="implement", instruction="y", adapter="claude-code",
            status=TaskStatus.FAILED,
        )
        routing = Artifact(
            job_id=job.id, task_id=t_ok.id, type=ArtifactType.ROUTING, created_by="router",
            payload={"model_id": "cursor/claude-opus-4-8", "adapter": "cursor",
                     "policy": "balanced", "estimated_cost_usd": 0.0},
            confidence=0.9, evidence=["role:explore"],
        )
        failed = Artifact(
            job_id=job.id, task_id=t_bad.id, type=ArtifactType.VERIFICATION, created_by="w",
            payload={"check": "preflight", "result": "blocked", "failure": "preflight_blocked"},
            confidence=0.95, evidence=["adapter:claude-code"],
        )
        trace = build_job_trace(job, [t_ok, t_bad], [routing, failed])
        self.assertEqual(trace.job_id, job.id)
        self.assertEqual(len(trace.tasks), 2)
        spans = {t.task_id: t for t in trace.tasks}
        self.assertEqual(spans[t_ok.id].model, "cursor/claude-opus-4-8")
        self.assertEqual(spans[t_bad.id].failure, "preflight_blocked")
        # attributes render without the OTel SDK.
        self.assertIn("puppetmaster.job.id", trace.attributes())
        self.assertIn("gen_ai.system", spans[t_ok.id].attributes())


class WorkerRuntimeFailureStatusTests(unittest.TestCase):
    """The fix for the defect: a recoverable adapter failure (or a preflight
    block) must record the task FAILED, not COMPLETE."""

    def _store_job_task(self, tmp, adapter="claude-code"):
        from puppetmaster.models import Task, TaskStatus
        from puppetmaster.store import SwarmStore

        store = SwarmStore(Path(tmp) / ".puppetmaster")
        job = store.create_job("ship it")
        store.update_job_status(job.id, __import__("puppetmaster.models", fromlist=["JobStatus"]).JobStatus.RUNNING)
        task = Task(job_id=job.id, role="implement", instruction="do", adapter=adapter, status=TaskStatus.QUEUED)
        store.save_task(task)
        return store, job, task

    def test_recoverable_artifact_marks_task_failed(self) -> None:
        from unittest.mock import patch

        from puppetmaster.models import AgentRun, Artifact, ArtifactType, TaskStatus
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            store, job, task = self._store_job_task(tmp)

            class _FakeWorker:
                def __init__(self, role, worker_id=None):
                    self.role = role

                def run(self, t, goal):
                    run = AgentRun(job_id=t.job_id, task_id=t.id, role=t.role, worker_id="w", status=TaskStatus.COMPLETE)
                    art = Artifact(
                        job_id=t.job_id, task_id=t.id, type=ArtifactType.VERIFICATION,
                        created_by="w", payload={"check": "x", "result": "blocked", "failure": "billing_or_quota"},
                        confidence=0.5, evidence=["adapter:claude-code"],
                    )
                    return run, [art]

            runtime = WorkerRuntime(store=store, job_id=job.id, role="implement", worker_id="w")
            with patch("puppetmaster.worker_runtime.LocalWorker", _FakeWorker):
                runtime.run_once()
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.FAILED)

    def test_clean_run_marks_task_complete(self) -> None:
        from unittest.mock import patch

        from puppetmaster.models import AgentRun, Artifact, ArtifactType, TaskStatus
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            store, job, task = self._store_job_task(tmp, adapter="local")

            class _FakeWorker:
                def __init__(self, role, worker_id=None):
                    self.role = role

                def run(self, t, goal):
                    run = AgentRun(job_id=t.job_id, task_id=t.id, role=t.role, worker_id="w", status=TaskStatus.COMPLETE)
                    art = Artifact(
                        job_id=t.job_id, task_id=t.id, type=ArtifactType.FINDING,
                        created_by="w", payload={"claim": "did it"}, confidence=0.9, evidence=["adapter:local"],
                    )
                    return run, [art]

            runtime = WorkerRuntime(store=store, job_id=job.id, role="implement", worker_id="w")
            with patch("puppetmaster.worker_runtime.LocalWorker", _FakeWorker):
                runtime.run_once()
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)


class AutoFallbackTests(unittest.TestCase):
    def _setup(self, tmp):
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        store = SwarmStore(Path(tmp) / ".puppetmaster")
        job = store.create_job("fix the bug")
        task = Task(
            job_id=job.id, role="implement", instruction="implement the fix",
            adapter="claude-code", status=TaskStatus.FAILED,
            payload={"auto_route": True, "model": "claude-opus-4-8"},
        )
        store.save_task(task)
        store.save_artifact(Artifact(
            job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION, created_by="w",
            payload={"check": "x", "result": "blocked", "failure": "billing_or_quota"},
            confidence=0.5, evidence=["adapter:claude-code"],
        ))
        return store, job, task

    def test_reroutes_to_funded_alternate_adapter(self) -> None:
        from unittest.mock import patch

        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.models import TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus

        registry = [
            ModelSpec(id="claude-code/opus-4-8", adapter="claude-code", adapter_model_name="claude-opus-4-8", capability_score=99, input_per_mtok_usd=5, output_per_mtok_usd=25, billing="unknown"),
            ModelSpec(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5", capability_score=90, billing="plan", tags=["cursor"]),
        ]

        def _billing(adapter, **kw):
            if adapter == "cursor":
                return BillingStatus(adapter="cursor", billing="plan", healthy=True, detail="ok", evidence=[])
            return BillingStatus(adapter=adapter, billing="unknown", healthy=False, detail="no", evidence=[])

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_billing):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.adapter, "cursor")
            self.assertEqual(updated.payload["fallback_attempts"], 1)
            self.assertEqual(updated.payload["fallback_from_adapter"], "claude-code")

    def test_no_funded_alternate_means_no_reroute(self) -> None:
        from unittest.mock import patch

        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus

        registry = [
            ModelSpec(id="claude-code/opus-4-8", adapter="claude-code", adapter_model_name="claude-opus-4-8", capability_score=99, billing="unknown"),
        ]

        def _billing(adapter, **kw):
            return BillingStatus(adapter=adapter, billing="unknown", healthy=False, detail="no", evidence=[])

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_billing):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 0)

    def test_hard_failure_vs_recoverable_classification(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("g")
            hard = Task(job_id=job.id, role="a", instruction="x", adapter="missing", status=TaskStatus.FAILED)
            soft = Task(job_id=job.id, role="b", instruction="y", adapter="claude-code", status=TaskStatus.FAILED)
            store.save_task(hard)
            store.save_task(soft)
            store.save_artifact(Artifact(
                job_id=job.id, task_id=soft.id, type=ArtifactType.VERIFICATION, created_by="w",
                payload={"check": "x", "result": "blocked", "failure": "billing_or_quota"},
                confidence=0.5, evidence=["a"],
            ))
            orch = Orchestrator(store)
            allowed = {hard.id, soft.id}
            self.assertTrue(orch._has_hard_failure(job, allowed))  # hard task is non-recoverable
            self.assertTrue(orch._should_fail_closed(job, allowed))

            # With only the soft (recoverable) task, neither should fail closed.
            store.update_task_status(hard, TaskStatus.COMPLETE)
            self.assertFalse(orch._has_hard_failure(job, allowed))
            self.assertFalse(orch._should_fail_closed(job, allowed))


class LiveProbeTests(unittest.TestCase):
    def test_classify_billing_and_auth_and_ok(self) -> None:
        from puppetmaster.preflight import classify_live_probe

        self.assertEqual(classify_live_probe("claude-code", 1, "Credit balance is too low"), "billing_or_quota")
        # Codex's own classifier returns a richer auth class; just confirm a
        # failure is detected (not a clean pass) for a not-logged-in probe.
        self.assertIsNotNone(classify_live_probe("codex", 0, "You are not logged in"))
        # An adapter with no specific classifier falls back to the auth marker.
        self.assertEqual(classify_live_probe("cursor", 0, "unauthorized"), "auth")
        self.assertIsNone(classify_live_probe("openai", 0, "ok"))

    def test_live_probe_with_injected_prober(self) -> None:
        from puppetmaster.preflight import live_probe

        bad = live_probe("claude-code", "claude-opus-4-8", prober=lambda a, m: (1, "", "credit balance is too low"))
        self.assertFalse(bad.ok)
        self.assertIn("live_probe:billing_or_quota", bad.evidence)

        good = live_probe("claude-code", "claude-opus-4-8", prober=lambda a, m: (0, "ok", ""))
        self.assertTrue(good.ok)

    def test_live_probe_unrunnable_does_not_block(self) -> None:
        """A probe that can't reach a verdict (missing CLI / timeout) must not
        block a dispatch the static checks already cleared."""
        from puppetmaster.preflight import live_probe

        missing_cli = live_probe(
            "claude-code", "claude-opus-4-8",
            prober=lambda a, m: (127, "", "command not found"),
        )
        self.assertTrue(missing_cli.ok)
        self.assertIn("live_probe:skipped_unverified", missing_cli.evidence)

        timed_out = live_probe(
            "codex", "gpt-5.5",
            prober=lambda a, m: (124, "", "live probe timed out"),
        )
        self.assertTrue(timed_out.ok)
        self.assertIn("live_probe:skipped_unverified", timed_out.evidence)

    def test_live_probe_cursor_uses_catalog(self) -> None:
        from puppetmaster.preflight import live_probe

        catalog = [{"id": "gpt-5.5"}]
        # Catalog ok + a real generation that succeeds -> ok (inject the
        # generation prober so the test stays hermetic / never shells to node).
        ok = live_probe(
            "cursor", "gpt-5.5",
            catalog_fetcher=lambda: catalog,
            prober=lambda a, m: (0, "ok", ""),
        )
        self.assertTrue(ok.ok)
        # Model not in the plan catalog -> blocks before any generation.
        missing = live_probe(
            "cursor", "nope",
            catalog_fetcher=lambda: catalog,
            prober=lambda a, m: (0, "ok", ""),
        )
        self.assertFalse(missing.ok)
        self.assertIn("live_probe:model_not_in_catalog", missing.evidence)

    def test_live_probe_cursor_blocks_on_plan_exhaustion(self) -> None:
        """A Cursor plan can enumerate the catalog yet be rate-limited / out of
        monthly allowance — the generation probe must catch that the static
        catalog ping can't."""
        from puppetmaster.preflight import live_probe

        catalog = [{"id": "gpt-5.5"}]
        rate_limited = live_probe(
            "cursor", "gpt-5.5",
            catalog_fetcher=lambda: catalog,
            prober=lambda a, m: (1, "", "rate limit exceeded; usage limit reached"),
        )
        self.assertFalse(rate_limited.ok)
        self.assertIn("live_probe:billing_or_quota", rate_limited.evidence)

    def test_live_probe_cursor_generation_unrunnable_does_not_block(self) -> None:
        """If the generation probe itself can't run (node missing), don't block a
        plan the catalog already validated — degrade to unverified."""
        from puppetmaster.preflight import live_probe

        catalog = [{"id": "gpt-5.5"}]
        result = live_probe(
            "cursor", "gpt-5.5",
            catalog_fetcher=lambda: catalog,
            prober=lambda a, m: (127, "", "node not found"),
        )
        self.assertTrue(result.ok)
        self.assertIn("live_probe:skipped_unverified", result.evidence)

    def test_preflight_check_live_blocks_on_billing(self) -> None:
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.preflight import preflight_check

        healthy = BillingStatus(adapter="claude-code", billing="plan", healthy=True, detail="oauth", evidence=["claude_oauth:present"])
        result = preflight_check(
            "claude-code", "claude-opus-4-8",
            live=True,
            prober=lambda a, m: (1, "", "credit balance is too low"),
            billing_status=healthy,
        )
        self.assertFalse(result.ok)
        self.assertIn("live_probe:billing_or_quota", result.evidence)

    def test_probe_openai_uses_max_completion_tokens(self) -> None:
        """Regression: GPT-5+ rejects the legacy ``max_tokens`` parameter with a
        400 on a *funded* account; the probe must send ``max_completion_tokens``
        so it doesn't falsely block a working key."""
        import io
        import json as _json
        import urllib.request
        from puppetmaster import preflight

        sent_bodies = []

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            sent_bodies.append(_json.loads(request.data.decode("utf-8")))
            return _Resp(b'{"choices":[{"message":{"content":"ok"}}]}')

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            rc, out, err = preflight._probe_openai(None, {"OPENAI_API_KEY": "k"})
        finally:
            urllib.request.urlopen = original

        self.assertEqual(rc, 0)
        self.assertEqual(len(sent_bodies), 1)
        self.assertIn("max_completion_tokens", sent_bodies[0])
        self.assertNotIn("max_tokens", sent_bodies[0])

    def test_probe_openai_falls_back_to_legacy_max_tokens(self) -> None:
        """OpenAI-compatible endpoints that predate the rename reject
        ``max_completion_tokens``; the probe then retries with ``max_tokens``."""
        import io
        import json as _json
        import urllib.error
        import urllib.request
        from puppetmaster import preflight

        sent_params = []

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            body = _json.loads(request.data.decode("utf-8"))
            if "max_completion_tokens" in body:
                sent_params.append("max_completion_tokens")
                err_body = io.BytesIO(
                    b'{"error":{"code":"unsupported_parameter",'
                    b'"message":"max_completion_tokens not supported"}}'
                )
                raise urllib.error.HTTPError(
                    "url", 400, "Bad Request", {}, err_body
                )
            sent_params.append("max_tokens")
            return _Resp(b'{"choices":[]}')

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            rc, out, err = preflight._probe_openai(None, {"OPENAI_API_KEY": "k"})
        finally:
            urllib.request.urlopen = original

        self.assertEqual(rc, 0)
        self.assertEqual(sent_params, ["max_completion_tokens", "max_tokens"])


class SubscriptionPlanCatalogTests(unittest.TestCase):
    """Curated catalogs + first-run auto-merge for the CLI agent loops
    (Claude Code OAuth, Codex/ChatGPT) that can't self-enumerate models."""

    def _status(self, adapter, healthy=True, billing="plan"):
        from puppetmaster.platform_billing import BillingStatus

        return BillingStatus(
            adapter=adapter, billing=billing, healthy=healthy, detail="t", evidence=[]
        )

    def _registry_path(self, tmp):
        from puppetmaster.model_registry import save_registry, starter_registry

        path = Path(tmp) / "models.json"
        save_registry(starter_registry(), path)
        return path

    def test_plan_merge_zeroes_price_preserves_id_and_capability(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.static_catalog import merge_curated_into_registry

        merged, report = merge_curated_into_registry(
            "claude-code", "plan", starter_registry()
        )
        self.assertEqual(report["source"], "claude")
        opus = next(s for s in merged if s.adapter_model_name == "claude-opus-4-8")
        self.assertEqual(opus.billing, "plan")
        self.assertEqual(opus.input_per_mtok_usd, 0.0)
        self.assertEqual(opus.output_per_mtok_usd, 0.0)
        # Existing id + (possibly user-tuned) capability are preserved.
        self.assertEqual(opus.id, "claude-code/opus-4-8")
        self.assertEqual(opus.capability_score, 99)
        self.assertIn("plan-billed", opus.tags)
        # A curated model not yet in the starter registry is added.
        self.assertIn("claude-sonnet-4-5", report["added"])

    def test_api_merge_keeps_reference_prices(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.static_catalog import merge_curated_into_registry

        merged, _ = merge_curated_into_registry(
            "claude-code", "api", starter_registry()
        )
        opus = next(s for s in merged if s.adapter_model_name == "claude-opus-4-8")
        self.assertEqual(opus.billing, "api")
        self.assertEqual(opus.input_per_mtok_usd, 5.0)
        self.assertEqual(opus.output_per_mtok_usd, 25.0)
        self.assertNotIn("plan-billed", opus.tags)

    def test_discovers_when_claude_subscription_and_no_frontier(self) -> None:
        from puppetmaster.cursor_discovery import has_plan_frontier
        from puppetmaster.model_registry import load_registry, read_discovery_meta
        from puppetmaster.static_catalog import ensure_subscription_plan_catalog

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            self.assertFalse(has_plan_frontier(load_registry(path)))

            def detector(adapter):
                return self._status(adapter, healthy=(adapter == "claude-code"),
                                    billing="plan" if adapter == "claude-code" else "unknown")

            report = ensure_subscription_plan_catalog(path, billing_detector=detector)
            self.assertEqual(report["action"], "discovered")
            self.assertEqual(report["adapter"], "claude-code")
            merged = load_registry(path)
            self.assertTrue(has_plan_frontier(merged))
            self.assertTrue(read_discovery_meta(path).get("claude"))

    def test_skips_when_plan_frontier_present(self) -> None:
        from puppetmaster.model_registry import ModelSpec, save_registry
        from puppetmaster.static_catalog import ensure_subscription_plan_catalog

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [ModelSpec(id="cursor/x", adapter="cursor", adapter_model_name="x",
                           capability_score=90, billing="plan")],
                path,
            )
            called = {"n": 0}

            def detector(adapter):
                called["n"] += 1
                return self._status(adapter)

            report = ensure_subscription_plan_catalog(path, billing_detector=detector)
            self.assertEqual(report["action"], "skip")
            self.assertEqual(report["reason"], "plan_frontier_present")
            self.assertEqual(called["n"], 0)  # no billing probe when frontier exists

    def test_skips_when_no_subscription_adapter(self) -> None:
        from puppetmaster.static_catalog import ensure_subscription_plan_catalog

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            report = ensure_subscription_plan_catalog(
                path,
                billing_detector=lambda a: self._status(a, healthy=True, billing="api"),
            )
            self.assertEqual(report["action"], "skip")
            self.assertEqual(report["reason"], "no_subscription_adapter")

    def test_idempotent_when_source_already_discovered(self) -> None:
        from puppetmaster.model_registry import write_discovery_meta
        from puppetmaster.static_catalog import ensure_subscription_plan_catalog

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            write_discovery_meta("claude", 5, path)
            # claude already discovered -> skipped; codex not authed -> overall skip.
            report = ensure_subscription_plan_catalog(
                path,
                billing_detector=lambda a: self._status(
                    a, healthy=(a == "claude-code"),
                    billing="plan" if a == "claude-code" else "unknown",
                ),
            )
            self.assertEqual(report["action"], "skip")

    def test_unavailable_on_detector_exception(self) -> None:
        from puppetmaster.static_catalog import ensure_subscription_plan_catalog

        def boom(adapter):
            raise RuntimeError("billing probe crashed")

        with TemporaryDirectory() as tmp:
            path = self._registry_path(tmp)
            report = ensure_subscription_plan_catalog(path, billing_detector=boom)
            self.assertEqual(report["action"], "unavailable")
            self.assertIn("billing probe crashed", report["error"])


class ApiDiscoveryTests(unittest.TestCase):
    def test_fetch_openai_models_parses_data(self) -> None:
        import json as _json

        from puppetmaster.api_discovery import fetch_openai_models

        body = _json.dumps({"data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4-mini"}, {"bad": 1}]})
        catalog = fetch_openai_models(env={"OPENAI_API_KEY": "k"}, getter=lambda u, h: (200, body))
        self.assertEqual([m["id"] for m in catalog], ["gpt-5.5", "gpt-5.4-mini"])

    def test_fetch_requires_key(self) -> None:
        from puppetmaster.api_discovery import ApiDiscoveryError, fetch_anthropic_models

        with self.assertRaises(ApiDiscoveryError):
            fetch_anthropic_models(env={})

    def test_catalog_to_specs_inherits_and_seeds(self) -> None:
        from puppetmaster.api_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(id="openai/gpt-5-5", adapter="openai", adapter_model_name="gpt-5.5", capability_score=96, input_per_mtok_usd=5, output_per_mtok_usd=30, billing="api"),
        ]
        catalog = [{"id": "gpt-5.5"}, {"id": "gpt-7-future"}]
        specs = catalog_to_specs("openai", "api", catalog, existing)
        by_name = {s.adapter_model_name: s for s in specs}
        self.assertEqual(by_name["gpt-5.5"].capability_score, 96)  # overlay
        self.assertEqual(by_name["gpt-7-future"].capability_score, 60)  # seed
        self.assertEqual(by_name["gpt-7-future"].billing, "api")
        self.assertIn("discovered", by_name["gpt-5.5"].tags)

    def test_merge_adds_without_dropping(self) -> None:
        from puppetmaster.api_discovery import merge_api_catalog_into_registry
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(id="openai/legacy", adapter="openai", adapter_model_name="gpt-old", capability_score=40, billing="api"),
            ModelSpec(id="cursor/x", adapter="cursor", adapter_model_name="composer-2.5", capability_score=55, billing="plan"),
        ]
        merged, report = merge_api_catalog_into_registry("openai", "api", existing, [{"id": "gpt-5.5"}])
        names = {(s.adapter, s.adapter_model_name) for s in merged}
        self.assertIn(("openai", "gpt-old"), names)  # not dropped
        self.assertIn(("openai", "gpt-5.5"), names)  # added
        self.assertIn(("cursor", "composer-2.5"), names)  # preserved
        self.assertIn("gpt-5.5", report["added"])


class CatalogStalenessTests(unittest.TestCase):
    def test_write_read_and_staleness(self) -> None:
        from datetime import datetime, timezone

        from puppetmaster.model_registry import (
            catalog_staleness_days,
            read_discovery_meta,
            write_discovery_meta,
        )

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            write_discovery_meta("cursor", 7, registry_path, now_iso="2026-05-01T00:00:00Z")
            meta = read_discovery_meta(registry_path)
            self.assertEqual(meta["cursor"]["count"], 7)
            now = datetime(2026, 6, 10, tzinfo=timezone.utc)  # ~40 days later
            age = catalog_staleness_days(meta, "cursor", now=now)
            self.assertGreater(age, 30)
            self.assertIsNone(catalog_staleness_days(meta, "openai"))


class TelemetryContextAndMetricsTests(unittest.TestCase):
    def test_traceparent_roundtrip(self) -> None:
        import re

        from puppetmaster.telemetry import new_traceparent, parse_traceparent

        tp = new_traceparent()
        self.assertRegex(tp, r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$")
        parsed = parse_traceparent(tp)
        self.assertIsNotNone(parsed)
        self.assertIsNone(parse_traceparent("garbage"))
        self.assertIsNone(parse_traceparent(None))

    def test_live_telemetry_enabled_logic(self) -> None:
        from puppetmaster.telemetry import live_telemetry_enabled

        self.assertFalse(live_telemetry_enabled({"PUPPETMASTER_OTEL_LIVE": "true"}))  # tracing off
        self.assertTrue(
            live_telemetry_enabled(
                {"PUPPETMASTER_OTEL_ENABLED": "true", "PUPPETMASTER_OTEL_LIVE": "true"}
            )
        )
        self.assertFalse(live_telemetry_enabled({"PUPPETMASTER_OTEL_ENABLED": "true"}))

    def test_build_job_metrics(self) -> None:
        from puppetmaster.models import Job, Task, TaskStatus
        from puppetmaster.telemetry import build_job_metrics, build_job_trace

        job = Job(goal="g")
        tasks = [
            Task(job_id=job.id, role="a", instruction="x", adapter="cursor", status=TaskStatus.COMPLETE),
            Task(job_id=job.id, role="b", instruction="y", adapter="claude-code", status=TaskStatus.FAILED),
        ]
        metrics = build_job_metrics(build_job_trace(job, tasks, []))
        self.assertEqual(metrics["jobs"], 1)
        self.assertEqual(metrics["tasks"], 2)
        self.assertIn(str(TaskStatus.COMPLETE), metrics["tasks_by_status"])

    def test_build_task_span(self) -> None:
        from puppetmaster.models import Job, Task, TaskStatus
        from puppetmaster.telemetry import build_task_span

        job = Job(goal="g")
        task = Task(
            job_id=job.id, role="implement", instruction="x", adapter="cursor",
            status=TaskStatus.COMPLETE, payload={"router_model_id": "cursor/gpt-5-5"},
        )
        span = build_task_span(task, [])
        self.assertEqual(span.model, "cursor/gpt-5-5")
        self.assertEqual(span.role, "implement")

    def test_record_job_trace_with_traceparent_noop_when_disabled(self) -> None:
        from puppetmaster.models import Job
        from puppetmaster.telemetry import record_job_trace

        self.assertFalse(
            record_job_trace(Job(goal="g"), [], [], env={}, traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        )


if __name__ == "__main__":
    unittest.main()

