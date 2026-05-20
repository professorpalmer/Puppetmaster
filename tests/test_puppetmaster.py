from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import threading
import time
import unittest
import sys
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import (
    ClaudeCodeAdapter,
    CursorAdapter,
    build_claude_code_command,
    classify_claude_code_failure,
    classify_cursor_failure,
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


class PuppetmasterTests(unittest.TestCase):
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
            for process in spawned:
                process.wait(timeout=15)

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
            self.assertEqual(status_payload["artifact_count"], 2)

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
                },
            )
            payload = json.loads(result["content"][0]["text"])
            spawned = ASYNC_PROCESSES[before_process_count:]
            for process in spawned:
                process.wait(timeout=15)

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
            watcher.start()
            mcp_server._mark_inbound_message()  # simulate Cursor sending us something
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
                    "executable": str(fake_claude),
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
                    "executable": str(fake_claude),
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
                    "executable": str(fake_claude),
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

    def test_provider_stub_returns_blocked_verification_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            specs = [
                WorkerSpec(
                    role="codex-review",
                    instruction="Ask Codex to review the repo.",
                    adapter="codex",
                )
            ]
            result = Orchestrator(store).run("exercise provider stub", specs=specs)
            artifact = store.list_artifacts(result.job.id)[0]

            self.assertEqual(artifact.type, ArtifactType.VERIFICATION)
            self.assertEqual(artifact.payload["adapter"], "codex")
            self.assertEqual(artifact.payload["result"], "blocked")

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
            time.sleep(0.25)
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


if __name__ == "__main__":
    unittest.main()

