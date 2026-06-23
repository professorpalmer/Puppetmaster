from __future__ import annotations

import argparse
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
from unittest.mock import MagicMock, patch

# Hermetic tests: the orchestrator's first-run plan-catalog auto-discovery
# shells out to the Cursor SDK (node) when CURSOR_API_KEY is set. The suite
# must never make that network/subprocess call implicitly — tests that
# exercise the discovery helper inject their own catalog fetcher instead.
os.environ.setdefault("PUPPETMASTER_AUTODISCOVER", "0")

from puppetmaster.adapters import (
    ClaudeCodeAdapter,
    CodexAdapter,
    CursorAdapter,
    HermesAdapter,
    OpenAIAdapter,
    StreamedProcess,
    UnconfiguredProviderAdapter,
    build_claude_code_command,
    build_codex_exec_command,
    build_hermes_chat_command,
    classify_claude_code_failure,
    classify_codex_failure,
    classify_cursor_failure,
    classify_hermes_failure,
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
from puppetmaster.hermes_spawn_tree import emit_spawn_tree
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

    def setUp(self) -> None:
        # CodeGraph's Cursor-Node invocation is memoized process-wide for perf
        # (codegraph._CURSOR_INVOCATION_CACHE). On a dev host that actually has
        # Cursor + a global codegraph install, the first test to trigger real
        # resolution caches a concrete invocation, which then bypasses the
        # shutil.which / subprocess mocks the codegraph helper tests rely on.
        # Reset the memo before each test so resolution is recomputed under
        # whatever each test mocks. (CI hosts have neither Cursor nor a global
        # shim, so this is a no-op there, but it makes the suite deterministic
        # everywhere.)
        from puppetmaster import codegraph as _codegraph_mod

        _codegraph_mod.reset_cursor_codegraph_invocation_cache()
        self.addCleanup(_codegraph_mod.reset_cursor_codegraph_invocation_cache)

    def test_mcp_lists_puppetmaster_agent_tools(self) -> None:
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = {tool["name"] for tool in response["result"]["tools"]}

        self.assertIn("puppetmaster_doctor", tool_names)
        self.assertIn("puppetmaster_cursor_review", tool_names)
        self.assertIn("puppetmaster_start_cursor_review", tool_names)
        self.assertIn("puppetmaster_claude_implement", tool_names)
        self.assertIn("puppetmaster_start_claude_implement", tool_names)
        self.assertIn("puppetmaster_codex", tool_names)
        self.assertIn("puppetmaster_start_codex", tool_names)
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
            # The launcher pid is now labeled honestly and monitoring is pointed
            # at job_id rather than the misleading supervisor pid (C2).
            self.assertEqual(payload["launcher_pid"], payload["pid"])
            self.assertEqual(payload["monitor_with"]["job_id"], payload["job_id"])
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

    def test_mcp_status_compact_arg_maps_to_cli_flag(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_run_cli(command, args):
            captured["command"] = command
            return {"content": [{"type": "text", "text": "{}"}], "isError": False}

        with patch.object(mcp_server, "run_cli", side_effect=fake_run_cli):
            result = mcp_server.run_status({"job_id": "job_x", "compact": True})

        self.assertFalse(result["isError"])
        self.assertEqual(captured["command"], ["status", "job_x", "--compact"])

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

    def test_hermes_spawn_tree_emitter_writes_snapshot_and_index(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SwarmStore(root / ".puppetmaster")
            store.init()
            job = store.create_job("show completed swarm in Hermes history")
            tasks = [
                Task(
                    job_id=job.id,
                    role="implement",
                    instruction="write the patch",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                ),
                Task(
                    job_id=job.id,
                    role="review",
                    instruction="verify the patch",
                    adapter="cursor",
                    status=TaskStatus.COMPLETE,
                ),
            ]
            store.save_tasks(tasks)
            artifacts = [
                Artifact(
                    job_id=job.id,
                    task_id=tasks[0].id,
                    type=ArtifactType.FINDING,
                    created_by="worker-implement",
                    payload={"claim": "implemented Hermes replay snapshot support"},
                    confidence=0.95,
                    evidence=["puppetmaster/hermes_spawn_tree.py"],
                ),
                Artifact(
                    job_id=job.id,
                    task_id=tasks[0].id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker-implement",
                    payload={"check": "tests", "result": "completed", "tokens_in": 123, "tokens_out": 45},
                    confidence=0.9,
                    evidence=["pytest"],
                ),
                Artifact(
                    job_id=job.id,
                    task_id=tasks[0].id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "cursor/gpt-5-5",
                        "model": "gpt-5.5",
                        "adapter": "cursor",
                        "policy": "balanced",
                        "estimated_cost_usd": 0.0,
                    },
                    confidence=0.9,
                    evidence=["policy:balanced"],
                ),
                Artifact(
                    job_id=job.id,
                    task_id=tasks[1].id,
                    type=ArtifactType.FINDING,
                    created_by="worker-review",
                    payload={"claim": "verified the snapshot schema"},
                    confidence=0.95,
                    evidence=["tests/test_puppetmaster.py"],
                ),
                Artifact(
                    job_id=job.id,
                    task_id=tasks[1].id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker-review",
                    payload={"check": "schema", "result": "completed"},
                    confidence=0.9,
                    evidence=["pytest"],
                ),
            ]
            store.save_artifacts(artifacts)
            completed = store.update_job_status(job.id, JobStatus.COMPLETE)

            snapshot_path = emit_spawn_tree(
                store,
                completed,
                store.list_artifacts(job.id),
                [],
                env={
                    "HERMES_HOME": str(root / "hermes-home"),
                    "HERMES_SESSION_ID": "session:one",
                },
            )

            self.assertIsNotNone(snapshot_path)
            assert snapshot_path is not None
            session_dir = root / "hermes-home" / "spawn-trees" / "session_one"
            self.assertEqual(snapshot_path.parent, session_dir)
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(snapshot.keys()),
                ["session_id", "started_at", "finished_at", "label", "subagents"],
            )
            self.assertEqual(snapshot["session_id"], "session:one")
            self.assertEqual(snapshot["label"], "show completed swarm in Hermes history")
            self.assertEqual(len(snapshot["subagents"]), len(tasks))
            entries_by_id = {e["subagent_id"]: e for e in snapshot["subagents"]}
            self.assertEqual(set(entries_by_id), {t.id for t in tasks})
            self.assertEqual(entries_by_id[tasks[0].id]["model"], "gpt-5.5")
            self.assertEqual(entries_by_id[tasks[0].id]["status"], "completed")

            index_lines = (session_dir / "_index.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(index_lines), 1)
            index_entry = json.loads(index_lines[0])
            self.assertEqual(
                list(index_entry.keys()),
                ["path", "session_id", "started_at", "finished_at", "label", "count"],
            )
            self.assertEqual(index_entry["path"], str(snapshot_path.resolve()))
            self.assertEqual(index_entry["count"], len(tasks))

    def test_hermes_spawn_tree_emitter_honors_opt_out(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SwarmStore(root / ".puppetmaster")
            store.init()
            job = store.create_job("do not write Hermes history")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="write nothing",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(task)
            artifact = Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.VERIFICATION,
                created_by="worker-implement",
                payload={"check": "opt out", "result": "completed"},
                confidence=0.9,
                evidence=["env"],
            )
            store.save_artifact(artifact)
            completed = store.update_job_status(job.id, JobStatus.COMPLETE)
            hermes_home = root / "hermes-opt-out"

            emitted = emit_spawn_tree(
                store,
                completed,
                store.list_artifacts(job.id),
                [],
                env={
                    "HERMES_HOME": str(hermes_home),
                    "PUPPETMASTER_HERMES_SPAWN_TREE": "0",
                },
            )

            self.assertIsNone(emitted)
            self.assertFalse((hermes_home / "spawn-trees").exists())

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

    def test_batch_store_methods_across_backends(self) -> None:
        from puppetmaster.models import (
            Artifact,
            ArtifactType,
            MemoryRecord,
            Task,
            TaskStatus,
        )

        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()

                job_a = store.create_job("job a")
                job_b = store.create_job("job b")
                task_a = Task(
                    job_id=job_a.id,
                    role="implement",
                    instruction="a",
                    adapter="cursor",
                    status=TaskStatus.QUEUED,
                )
                task_b = Task(
                    job_id=job_b.id,
                    role="review",
                    instruction="b",
                    adapter="cursor",
                    status=TaskStatus.QUEUED,
                )
                store.save_tasks([task_a, task_b])
                self.assertEqual(len(store.list_tasks(job_a.id)), 1)
                self.assertEqual(len(store.list_tasks(job_b.id)), 1)

                artifact = Artifact(
                    job_id=job_a.id,
                    task_id=task_a.id,
                    type=ArtifactType.FINDING,
                    created_by="w",
                    payload={"claim": "x"},
                    confidence=0.9,
                    evidence=["e"],
                )
                store.save_artifacts([artifact])
                self.assertEqual(store.get_artifact_job_id(artifact.id), job_a.id)
                self.assertIsNone(store.get_artifact_job_id("missing"))

                batch_tasks = store.list_tasks_for_jobs([job_a.id, job_b.id])
                self.assertEqual({task.id for task in batch_tasks}, {task_a.id, task_b.id})
                batch_artifacts = store.list_artifacts_for_jobs([job_a.id, job_b.id])
                self.assertEqual(len(batch_artifacts), 1)

                memory = MemoryRecord(
                    scope="swarm.findings",
                    statement="found it",
                    evidence=["e"],
                    source_artifacts=[artifact.id],
                    confidence=0.85,
                )
                store.promote_memories([memory])
                self.assertEqual(len(store.list_memory()), 1)

    def test_retrieve_memory_multi_term_parity_across_backends(self) -> None:
        from puppetmaster.models import MemoryRecord

        memory = MemoryRecord(
            scope="swarm.findings",
            statement="independent workers coordinate via the store",
            evidence=["e"],
            source_artifacts=[],
            confidence=0.9,
        )
        results: dict[str, set[str]] = {}
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()
                store.promote_memories([memory])

                partial = store.retrieve_memory("workers nonexistent", limit=10)
                partial_ids = {item["id"] for item in partial}
                self.assertTrue(partial_ids)
                self.assertEqual(partial_ids, {memory.id})
                results[backend] = partial_ids

                none_match = store.retrieve_memory("nonexistent zzzqqq", limit=10)
                self.assertEqual(none_match, [])

        self.assertEqual(results["file"], results["sqlite"])

    def test_save_artifacts_emits_events_in_same_transaction(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus

        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("artifact events")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="x",
                adapter="cursor",
                status=TaskStatus.QUEUED,
            )
            store.save_task(task)
            artifacts = [
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by="w",
                    payload={"claim": f"c{i}"},
                    confidence=0.8 + i * 0.05,
                    evidence=["e"],
                )
                for i in range(3)
            ]
            store.save_artifacts(artifacts)

            events = store.read_events_since(job.id, since=0)
            saved = [e for e in events if e["event"] == "artifact.saved"]
            self.assertEqual(len(saved), len(artifacts))
            saved_by_id = {e["payload"]["artifact_id"]: e for e in saved}
            self.assertEqual(set(saved_by_id.keys()), {artifact.id for artifact in artifacts})
            for artifact in artifacts:
                event = saved_by_id[artifact.id]
                stored = store.get_artifacts_by_ids(job.id, [artifact.id])[artifact.id]
                self.assertEqual(event["payload"]["task_id"], artifact.task_id)
                self.assertEqual(event["payload"]["type"], str(artifact.type))
                self.assertEqual(event["payload"]["confidence"], artifact.confidence)
                self.assertEqual(event["payload"]["sha256"], stored.sha256)

    def test_batch_list_methods_dedupe_duplicate_job_ids(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus

        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                store = self._store_for_backend(backend, Path(tmp) / ".puppetmaster")
                store.init()
                job = store.create_job("dedupe jobs")
                task = Task(
                    job_id=job.id,
                    role="implement",
                    instruction="x",
                    adapter="cursor",
                    status=TaskStatus.QUEUED,
                )
                store.save_task(task)
                artifact = Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by="w",
                    payload={"claim": "x"},
                    confidence=0.9,
                    evidence=["e"],
                )
                store.save_artifact(artifact)

                dup_tasks = store.list_tasks_for_jobs([job.id, job.id])
                dup_artifacts = store.list_artifacts_for_jobs([job.id, job.id])
                self.assertEqual(len(dup_tasks), 1)
                self.assertEqual(len(dup_artifacts), 1)
                self.assertEqual({t.id for t in dup_tasks}, {task.id})
                self.assertEqual({a.id for a in dup_artifacts}, {artifact.id})

    def test_sqlite_chunked_in_lists_for_artifacts(self) -> None:
        from unittest.mock import patch

        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster import sqlite_store

        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("chunked in")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="x",
                adapter="cursor",
                status=TaskStatus.QUEUED,
            )
            store.save_task(task)
            artifacts = [
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by="w",
                    payload={"claim": f"c{i}"},
                    confidence=0.9,
                    evidence=["e"],
                )
                for i in range(5)
            ]
            store.save_artifacts(artifacts)
            artifact_ids = [artifact.id for artifact in artifacts]
            extra_ids = ["missing-a", "missing-b"]
            query_ids = artifact_ids + extra_ids

            with patch.object(sqlite_store, "_SQLITE_IN_CHUNK", 2):
                by_id = store.get_artifacts_by_ids(job.id, query_ids)
            self.assertEqual(set(by_id.keys()), set(artifact_ids))

            jobs = [job.id, job.id, "missing-job"]
            with patch.object(sqlite_store, "_SQLITE_IN_CHUNK", 2):
                by_type = store.list_artifacts_by_type("finding", job_ids=jobs)
            self.assertEqual({a.id for a in by_type}, set(artifact_ids))

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
            task = Task(
                job_id=job.id,
                role="reviewer",
                instruction="look for risks",
                payload={"prompt": "review the long prompt body"},
            )
            store.save_task(task)
            claimed = store.claim_task(task.id, "worker-a", lease_seconds=60)
            store.save_task(replace(claimed, lease_expires_at=seconds_from_now(-1)))

            snapshot = store.status_snapshot(job.id)

            self.assertIsNotNone(claimed)
            self.assertEqual(snapshot["job"]["goal"], "inspect runtime state")
            self.assertEqual(snapshot["tasks"][0]["instruction"], "look for risks")
            self.assertEqual(
                snapshot["tasks"][0]["payload"]["prompt"],
                "review the long prompt body",
            )
            self.assertEqual(snapshot["task_counts"][str(TaskStatus.RUNNING)], 1)
            self.assertEqual(snapshot["stale_task_ids"], [task.id])

    def test_status_snapshot_compact_omits_prompt_bodies(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("inspect runtime state")
            task = Task(
                job_id=job.id,
                role="reviewer",
                instruction="look for risks",
                payload={"prompt": "review the long prompt body", "model": "m"},
            )
            store.save_task(task)

            snapshot = store.status_snapshot(job.id, compact=True)
            task_snapshot = snapshot["tasks"][0]

            self.assertNotIn("goal", snapshot["job"])
            self.assertEqual(snapshot["job"]["goal_ref"]["chars"], len("inspect runtime state"))
            self.assertEqual(len(snapshot["job"]["goal_ref"]["sha256"]), 64)
            self.assertNotIn("instruction", task_snapshot)
            self.assertEqual(
                task_snapshot["instruction_ref"]["chars"],
                len("look for risks"),
            )
            self.assertEqual(len(task_snapshot["instruction_ref"]["sha256"]), 64)
            self.assertNotIn("prompt", task_snapshot["payload"])
            self.assertEqual(
                task_snapshot["payload"]["prompt_ref"]["chars"],
                len("review the long prompt body"),
            )
            self.assertEqual(len(task_snapshot["payload"]["prompt_ref"]["sha256"]), 64)
            self.assertEqual(task_snapshot["payload"]["model"], "m")

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

    def test_prompt_with_memory_dedupes_and_caps_statements(self) -> None:
        """Injected promoted memory is deduped and each statement size-capped."""
        from puppetmaster.adapters import prompt_with_memory, _MEMORY_STATEMENT_MAX_CHARS
        from puppetmaster.models import Task

        long_statement = "decided to " + ("x" * 600)
        retrieved = [
            {"scope": "swarm.decisions", "statement": "use sqlite store"},
            {"scope": "swarm.decisions", "statement": "use sqlite store"},  # dup
            {"scope": "swarm.decisions", "statement": long_statement},
        ]
        task = Task(job_id="j", role="explore", instruction="x", payload={"retrieved_memory": retrieved})
        result = prompt_with_memory("base prompt", task)

        self.assertEqual(result.count("use sqlite store"), 1)  # deduped
        self.assertIn("…", result)  # long statement truncated
        for line in result.splitlines():
            if line.startswith("- ["):
                # bullet body never exceeds the cap (plus the scope prefix/ellipsis).
                self.assertLessEqual(len(line), _MEMORY_STATEMENT_MAX_CHARS + 40)

    def test_prompt_with_memory_noop_without_memory(self) -> None:
        from puppetmaster.adapters import prompt_with_memory
        from puppetmaster.models import Task

        task = Task(job_id="j", role="explore", instruction="x", payload={})
        self.assertEqual(prompt_with_memory("base prompt", task), "base prompt")

    def test_memory_retrieval_supports_scope_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            Orchestrator(store).run("make workers independent", roles=["explore"])

            scoped = store.retrieve_memory("workers", scope="swarm.findings")
            missing = store.retrieve_memory("workers", scope="swarm.decisions")

            self.assertTrue(scoped)
            self.assertFalse(any(memory["scope"] == "swarm.findings" for memory in missing))

    def test_fresh_by_default_skips_memory_for_evaluative_roles(self) -> None:
        from puppetmaster.models import MemoryRecord

        goal = "audit fresh memory injection"
        memory = MemoryRecord(
            scope="swarm.verification",
            statement="prior audit claimed everything is clean",
            evidence=["e"],
            source_artifacts=["artifact_x"],
            confidence=0.9,
        )
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.promote_memory(memory)
            orch = Orchestrator(store)

            eval_spec = WorkerSpec(role="audit", instruction="verify the delta")
            explore_spec = WorkerSpec(role="explore", instruction="map the repo")
            routed = orch._with_retrieved_memory([eval_spec, explore_spec], goal)

            self.assertNotIn("retrieved_memory", routed[0].payload)
            self.assertIn("retrieved_memory", routed[1].payload)
            self.assertTrue(routed[1].payload["retrieved_memory"])

    def test_disable_memory_payload_overrides_role_defaults(self) -> None:
        from puppetmaster.models import MemoryRecord

        goal = "shared context for workers"
        memory = MemoryRecord(
            scope="swarm.findings",
            statement="shared context for workers",
            evidence=["e"],
            source_artifacts=["artifact_y"],
            confidence=0.85,
        )
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.promote_memory(memory)
            orch = Orchestrator(store)

            forced_fresh = WorkerSpec(
                role="explore",
                instruction="explore without memory",
                payload={"disable_memory": True},
            )
            forced_memory = WorkerSpec(
                role="audit",
                instruction="audit with inherited memory",
                payload={"disable_memory": False},
            )
            routed = orch._with_retrieved_memory([forced_fresh, forced_memory], goal)

            self.assertNotIn("retrieved_memory", routed[0].payload)
            self.assertIn("retrieved_memory", routed[1].payload)

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

    def test_start_implement_uses_locked_platform(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True, "job_id": "j1"}

        with patch(
            "puppetmaster.platform_lock.enabled_adapters", return_value={"cursor"}
        ), patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            result = mcp_server.start_implement({"goal": "ship the audit", "cwd": "."})

        self.assertIn("--implement", captured["command"])
        self.assertEqual(captured["command"][0], "cursor")
        self.assertEqual(result["implement_adapter"], "cursor")

    def test_start_implement_falls_back_to_claude_when_cursor_disabled(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch(
            "puppetmaster.platform_lock.enabled_adapters", return_value={"claude-code"}
        ), patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            result = mcp_server.start_implement({"goal": "ship it", "cwd": "."})

        self.assertEqual(captured["command"][0], "claude")
        self.assertEqual(result["implement_adapter"], "claude-code")

    def test_start_implement_rejects_disabled_requested_adapter(self) -> None:
        from puppetmaster import mcp_server

        with patch(
            "puppetmaster.platform_lock.enabled_adapters", return_value={"cursor"}
        ):
            result = mcp_server.start_implement(
                {"goal": "ship it", "cwd": ".", "adapter": "claude-code"}
            )
        self.assertTrue(result.get("isError"))

    def test_client_is_codex_from_handshake_info(self) -> None:
        from puppetmaster import mcp_server

        self.assertTrue(
            mcp_server._client_is_codex({"name": "codex-mcp-client", "title": "Codex"})
        )
        self.assertTrue(mcp_server._client_is_codex({"name": "codex", "version": "0.134.0"}))
        self.assertFalse(mcp_server._client_is_codex({"name": "cursor-vscode"}))
        self.assertFalse(mcp_server._client_is_codex({"name": "claude-code"}))
        self.assertFalse(mcp_server._client_is_codex({}))

    def test_host_enforces_tool_timeout_prefers_handshake_then_env(self) -> None:
        from puppetmaster import mcp_server

        original = dict(mcp_server._CLIENT_INFO)
        try:
            # Known Codex client wins regardless of env.
            mcp_server._CLIENT_INFO = {"name": "codex-mcp-client", "title": "Codex"}
            self.assertTrue(mcp_server._host_enforces_tool_timeout({}))
            # Known non-Codex client is never misread from stray CODEX_ env vars.
            mcp_server._CLIENT_INFO = {"name": "cursor-vscode"}
            self.assertFalse(mcp_server._host_enforces_tool_timeout({"CODEX_CI": "1"}))
            # No handshake captured yet -> fall back to env markers.
            mcp_server._CLIENT_INFO = {}
            self.assertTrue(mcp_server._host_enforces_tool_timeout({"CODEX_THREAD_ID": "x"}))
            self.assertFalse(mcp_server._host_enforces_tool_timeout({"CURSOR_AGENT": "1"}))
            self.assertFalse(mcp_server._host_enforces_tool_timeout({}))
        finally:
            mcp_server._CLIENT_INFO = original

    def test_initialize_captures_client_info(self) -> None:
        from puppetmaster import mcp_server

        original = dict(mcp_server._CLIENT_INFO)
        try:
            mcp_server._CLIENT_INFO = {}
            mcp_server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"clientInfo": {"name": "codex-mcp-client", "title": "Codex"}},
                }
            )
            self.assertEqual(mcp_server._CLIENT_INFO.get("name"), "codex-mcp-client")
            self.assertTrue(mcp_server._host_enforces_tool_timeout({}))
        finally:
            mcp_server._CLIENT_INFO = original

    def test_should_autodetach_respects_arg_and_kill_switch(self) -> None:
        from puppetmaster import mcp_server

        # Explicit per-call override wins over everything.
        self.assertTrue(mcp_server._should_autodetach_worker({"autodetach": True}))
        with patch.dict(os.environ, {}, clear=False):
            os.environ["CODEX_CI"] = "1"
            self.assertFalse(mcp_server._should_autodetach_worker({"autodetach": False}))
        # On a Codex host, default is to auto-detach...
        with patch.object(mcp_server, "_host_enforces_tool_timeout", return_value=True):
            self.assertTrue(mcp_server._should_autodetach_worker({}))
            # ...unless the operator opts out via the kill switch.
            with patch.dict(os.environ, {"PUPPETMASTER_MCP_SYNC_AUTODETACH": "0"}):
                self.assertFalse(mcp_server._should_autodetach_worker({}))
        # Non-hard-timeout host keeps inline blocking.
        with patch.object(mcp_server, "_host_enforces_tool_timeout", return_value=False):
            self.assertFalse(mcp_server._should_autodetach_worker({}))

    def test_run_worker_cli_blocks_inline_off_hard_timeout_host(self) -> None:
        from puppetmaster import mcp_server

        with patch.object(mcp_server, "_should_autodetach_worker", return_value=False), patch.object(
            mcp_server, "run_cli", return_value={"ran": "inline"}
        ) as run_cli, patch.object(mcp_server, "start_cli") as start_cli:
            result = mcp_server.run_worker_cli(["cursor", "goal"], {"goal": "g"})

        run_cli.assert_called_once()
        start_cli.assert_not_called()
        self.assertEqual(result, {"ran": "inline"})

    def test_run_worker_cli_autodetaches_to_job_on_hard_timeout_host(self) -> None:
        from puppetmaster import mcp_server

        start_body = {"job_id": "job_abc123", "run_id": "mcp_1", "next_steps": ["old"]}
        start_result = {
            "content": [{"type": "text", "text": json.dumps(start_body)}],
            "isError": False,
        }
        with patch.object(mcp_server, "_should_autodetach_worker", return_value=True), patch.object(
            mcp_server, "start_cli", return_value=start_result
        ) as start_cli, patch.object(mcp_server, "run_cli") as run_cli:
            result = mcp_server.run_worker_cli(["cursor", "goal"], {"goal": "g"})

        start_cli.assert_called_once()
        run_cli.assert_not_called()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["autodetached"])
        self.assertEqual(body["job_id"], "job_abc123")
        self.assertTrue(any("job_abc123" in step for step in body["next_steps"]))
        self.assertFalse(result["isError"])

    def test_run_codex_autodetaches_under_codex_host(self) -> None:
        from puppetmaster import mcp_server

        start_body = {"job_id": "job_zzz", "run_id": "mcp_2"}
        start_result = {
            "content": [{"type": "text", "text": json.dumps(start_body)}],
            "isError": False,
        }
        with patch.object(mcp_server, "_worktree_preflight", return_value=None), patch.object(
            mcp_server, "_should_autodetach_worker", return_value=True
        ), patch.object(mcp_server, "start_cli", return_value=start_result), patch.object(
            mcp_server, "run_cli"
        ) as run_cli:
            result = mcp_server.run_codex({"goal": "fix it", "cwd": ".", "sandbox": "read-only"})

        run_cli.assert_not_called()
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["autodetached"])

    def test_cursor_failure_classification_is_actionable(self) -> None:
        self.assertEqual(classify_cursor_failure("CURSOR_API_KEY is required"), "missing_api_key")
        self.assertEqual(classify_cursor_failure("model invalid"), "model_unavailable")
        self.assertEqual(
            classify_cursor_failure("forbidden-model: fable-5 is not on your plan"),
            "model_unavailable",
        )
        self.assertEqual(
            classify_cursor_failure("unknown model fable-5 rejected by Cursor SDK"),
            "model_unavailable",
        )
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

    def test_cursor_implement_emits_patch_when_tree_changes(self) -> None:
        task = Task(
            job_id="job",
            role="cursor",
            instruction="add a helper",
            adapter="cursor",
            payload={
                "prompt": "Add a helper",
                "cwd": ".",
                "mode": "implement",
                "disable_codegraph": True,
            },
        )
        completed = StreamedProcess(
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": "done"}),
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        before = {"sha": "base123", "changed_files": [], "untracked_files": [], "diff": ""}
        after = {
            "sha": "base123",
            "changed_files": ["helper.py"],
            "untracked_files": [],
            "diff": "diff --git a/helper.py b/helper.py\n+def helper():\n+    return 1\n",
        }
        with patch("puppetmaster.adapters.git_snapshot", side_effect=[before, after]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
        ) as run:
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        cursor_input = json.loads(run.call_args.kwargs["env"]["PUPPETMASTER_CURSOR_INPUT"])
        self.assertIn("Implement mode", cursor_input["prompt"])
        types = [a.type for a in artifacts]
        self.assertIn(ArtifactType.VERIFICATION, types)
        self.assertIn(ArtifactType.PATCH, types)
        patch_artifact = next(a for a in artifacts if a.type == ArtifactType.PATCH)
        self.assertEqual(patch_artifact.payload["status"], "applied")
        self.assertIn("helper.py", patch_artifact.payload["files"])
        self.assertIn("diff --git", patch_artifact.payload["unified_diff"])
        self.assertEqual(artifacts[0].payload["result"], "passed")

    def test_cursor_implement_prompt_demands_a_final_report(self) -> None:
        """Field report (v0.9.40 CI fix): the implement worker's diagnosis only
        existed as prose the pipeline threw away. The prompt must now ask for a
        closing report so there is always something to persist."""
        from puppetmaster.adapters import CursorAdapter

        prompt = CursorAdapter._implement_prompt("Fix the failing tests")
        self.assertIn("Reporting contract", prompt)
        self.assertIn("what you ran to verify", prompt)

    def test_cursor_implement_wraps_prose_report_as_finding(self) -> None:
        """A successful implement run whose final message is prose must keep
        that report as a FINDING artifact — not drop it and let the stitched
        summary read 'Findings: None' for a perfectly good run."""
        task = Task(
            job_id="job",
            role="cursor",
            instruction="fix CI",
            adapter="cursor",
            payload={
                "prompt": "Fix CI",
                "cwd": ".",
                "mode": "implement",
                "disable_codegraph": True,
            },
        )
        report = (
            "## CI fix report\n"
            "Root cause: npm --prefix got backslashes on Windows.\n"
            "Changed puppetmaster/installers.py to use as_posix(); ran pytest."
        )
        completed = StreamedProcess(
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": report}),
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
        ):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        findings = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].payload["claim"], "CI fix report")
        self.assertIn("as_posix()", findings[0].payload["report"])
        self.assertIn("report:final-message", findings[0].evidence)
        self.assertEqual(artifacts[0].payload["result"], "passed")

    def test_cursor_implement_parses_structured_report_into_typed_artifacts(self) -> None:
        """If the implement worker does return the JSON artifact contract, it
        parses into typed artifacts exactly like analyze mode."""
        task = Task(
            job_id="job",
            role="cursor",
            instruction="fix CI",
            adapter="cursor",
            payload={
                "prompt": "Fix CI",
                "cwd": ".",
                "mode": "implement",
                "disable_codegraph": True,
            },
        )
        structured = json.dumps(
            {
                "artifacts": [
                    {
                        "type": "finding",
                        "claim": "Windows paths broke npm --prefix",
                        "evidence": ["puppetmaster/installers.py"],
                        "confidence": 0.9,
                    },
                    {
                        "type": "decision",
                        "decision": "Use as_posix() for npm arguments",
                        "why": "npm chokes on backslashes",
                        "evidence": ["puppetmaster/installers.py"],
                        "confidence": 0.9,
                    },
                ]
            }
        )
        completed = StreamedProcess(
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": structured}),
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
        ):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        types = [a.type for a in artifacts]
        self.assertIn(ArtifactType.FINDING, types)
        self.assertIn(ArtifactType.DECISION, types)

    def test_cursor_implement_failed_run_emits_no_report_finding(self) -> None:
        """A failed implement run must not dress its output up as a report."""
        task = Task(
            job_id="job",
            role="cursor",
            instruction="fix CI",
            adapter="cursor",
            payload={
                "prompt": "Fix CI",
                "cwd": ".",
                "mode": "implement",
                "disable_codegraph": True,
            },
        )
        completed = StreamedProcess(
            returncode=1,
            stdout="boom",
            stderr="agent crashed",
            timed_out=False,
            live_log_path=None,
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
        ):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        self.assertNotIn(ArtifactType.FINDING, [a.type for a in artifacts])
        self.assertEqual(artifacts[0].payload["result"], "failed")

    def test_with_report_contract_skips_structured_prompts(self) -> None:
        """Prompts that already demand the JSON artifact contract (swarm
        review/plan roles) must not get a conflicting prose-report request."""
        from puppetmaster.adapters import _IMPLEMENT_REPORT_CONTRACT, with_report_contract

        structured = "Review the repo.\n\nPuppetmaster artifact contract:\nReturn only JSON."
        self.assertEqual(with_report_contract(structured), structured)

        plain = "Fix the failing tests"
        wrapped = with_report_contract(plain)
        self.assertIn(_IMPLEMENT_REPORT_CONTRACT, wrapped)
        self.assertEqual(with_report_contract(wrapped), wrapped)

    def test_cursor_implement_blocks_on_dirty_tree(self) -> None:
        task = Task(
            job_id="job",
            role="cursor",
            instruction="add a helper",
            adapter="cursor",
            payload={"prompt": "Add a helper", "cwd": ".", "mode": "implement"},
        )
        dirty = {
            "sha": "base123",
            "changed_files": ["already_dirty.py"],
            "untracked_files": [],
            "diff": "x",
        }
        with patch("puppetmaster.adapters.git_snapshot", return_value=dirty), patch(
            "puppetmaster.adapters.subprocess.run"
        ) as run:
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        run.assert_not_called()  # never spawns the agent on a dirty tree
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].payload["result"], "blocked")
        self.assertEqual(artifacts[0].payload["failure"], "dirty_worktree")

    def test_cursor_implement_allow_dirty_runs_agent(self) -> None:
        task = Task(
            job_id="job",
            role="cursor",
            instruction="add a helper",
            adapter="cursor",
            payload={
                "prompt": "Add a helper",
                "cwd": ".",
                "mode": "implement",
                "allow_dirty": True,
                "disable_codegraph": True,
            },
        )
        completed = StreamedProcess(
            returncode=0,
            stdout=json.dumps({"status": "finished", "result": "ok"}),
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        before_dirty = {
            "sha": "s",
            "changed_files": ["pre.py"],
            "untracked_files": [],
            "diff": "diff --git a/pre.py b/pre.py\n+already dirty\n",
        }
        after_dirty = {
            "sha": "s",
            "changed_files": ["pre.py", "helper.py"],
            "untracked_files": [],
            "worker_changed_files": ["helper.py"],
            "worker_untracked_files": [],
            "worker_diff": "diff --git a/helper.py b/helper.py\n+def helper():\n+    return 1\n",
            "diff": (
                "diff --git a/pre.py b/pre.py\n+already dirty\n"
                "diff --git a/helper.py b/helper.py\n+def helper():\n+    return 1\n"
            ),
        }
        with patch("puppetmaster.adapters.git_snapshot", side_effect=[before_dirty, after_dirty]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
        ) as run:
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

        run.assert_called_once()  # dirty guard bypassed
        self.assertEqual(artifacts[0].payload["result"], "passed")
        self.assertTrue(artifacts[0].payload["baseline_diff_present"])
        self.assertTrue(artifacts[0].payload["worker_diff_present"])
        patch_artifact = next(a for a in artifacts if a.type == ArtifactType.PATCH)
        self.assertEqual(patch_artifact.payload["files"], ["helper.py"])
        self.assertTrue(patch_artifact.payload["baseline_diff_present"])
        self.assertTrue(patch_artifact.payload["worker_diff_present"])
        self.assertNotIn("patch_artifact_emitted", patch_artifact.payload)
        self.assertIn("helper.py", patch_artifact.payload["unified_diff"])
        self.assertNotIn("pre.py", patch_artifact.payload["unified_diff"])

    def test_dirty_baseline_without_worker_diff_emits_no_patch(self) -> None:
        from puppetmaster.adapters import _should_emit_patch_artifact

        before_dirty = {
            "sha": "s",
            "changed_files": ["pre.py"],
            "untracked_files": [],
            "diff": "diff --git a/pre.py b/pre.py\n+already dirty\n",
        }
        after_unattributed = {
            "sha": "s",
            "changed_files": ["pre.py", "helper.py"],
            "untracked_files": [],
            "worker_diff": "",
            "diff": (
                "diff --git a/pre.py b/pre.py\n+already dirty\n"
                "diff --git a/helper.py b/helper.py\n+def helper():\n+    return 1\n"
            ),
        }

        self.assertFalse(_should_emit_patch_artifact(before_dirty, after_unattributed))

    def test_patch_gate_requires_diff_text_without_worker_diff(self) -> None:
        from puppetmaster.adapters import _should_emit_patch_artifact

        clean_before = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        after_without_diff_text = {
            "sha": "s",
            "changed_files": ["helper.py"],
            "untracked_files": [],
            "diff": "",
        }

        self.assertFalse(
            _should_emit_patch_artifact(clean_before, after_without_diff_text)
        )

    def test_cursor_implement_dirty_tree_snapshot_failure_emits_no_patch(self) -> None:
        from puppetmaster import adapters

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "pre.py").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "pre.py"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            (repo / "pre.py").write_text("user dirty\n", encoding="utf-8")
            before_tree = adapters.git_worktree_tree(repo)
            task = Task(
                job_id="job",
                role="cursor",
                instruction="add a helper",
                adapter="cursor",
                payload={
                    "prompt": "Add a helper",
                    "cwd": str(repo),
                    "mode": "implement",
                    "allow_dirty": True,
                    "disable_codegraph": True,
                },
            )
            completed = StreamedProcess(
                returncode=0,
                stdout=json.dumps({"status": "finished", "result": "ok"}),
                stderr="",
                timed_out=False,
                live_log_path=None,
            )

            def fake_run(*args: object, **kwargs: object) -> StreamedProcess:
                (repo / "helper.py").write_text("pm change\n", encoding="utf-8")
                return completed

            with patch(
                "puppetmaster.adapters.git_worktree_tree",
                side_effect=[before_tree, ""],
            ), patch("puppetmaster.adapters.run_streamed_subprocess", side_effect=fake_run):
                artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

            self.assertEqual(artifacts[0].payload["result"], "passed")
            self.assertNotIn(ArtifactType.PATCH, [artifact.type for artifact in artifacts])

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

    def test_codegraph_prompt_section_advertises_self_serve_cli(self) -> None:
        from puppetmaster.codegraph import codegraph_prompt_section

        section = codegraph_prompt_section("auth.py:42 -> login()")
        self.assertIn("auth.py:42", section)
        # Workers run sandboxed (no live MCP tools), so the injected snapshot
        # must tell them they can refresh the graph view themselves via the
        # ABI-safe CLI instead of only relying on the frozen context.
        self.assertIn("python -m puppetmaster codegraph", section)
        self.assertIn("affected", section)

    def test_run_codegraph_cli_reports_missing_cli(self) -> None:
        from puppetmaster import codegraph as codegraph_module

        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with patch("puppetmaster.codegraph.shutil.which", return_value=None), patch.object(
                codegraph_module, "_cursor_codegraph_invocation", return_value=None
            ):
                payload = run_codegraph_cli(["status"], tmp)

            self.assertFalse(payload["ok"])
            # With no Node at all (npx also absent), the only floor we can't
            # cross — the hint names Node as the prerequisite.
            self.assertIn("Node.js", payload["error"])
            self.assertIn("npm install -g @colbymchenry/codegraph", payload["error"])

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
        self.assertIn("Node.js", body["error"])

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
        """Without any discoverable Node we surface a clear next-step list."""
        from puppetmaster import codegraph_repair

        # Patch the generalized resolver the repair path actually calls.
        with patch.object(codegraph_repair, "find_runtime_node", return_value=None):
            result = codegraph_repair.repair_codegraph_sqlite(verify=False)

        self.assertFalse(result.ok)
        self.assertIn("Node", result.message)
        self.assertTrue(result.next_steps)

    def test_find_runtime_node_falls_back_to_path_node(self) -> None:
        """On a non-Cursor host, find_runtime_node returns `node` from PATH.

        This is the fix for harnesses other than Cursor (claude-code, codex,
        openai, hermes): no Cursor.app, no env override, but `node` on PATH
        must still resolve so repair-codegraph can run.
        """
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            path_node = Path(tmp) / "node"
            path_node.write_text("#!/bin/sh\necho v22.0.0\n", encoding="utf-8")
            path_node.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if k != "PUPPETMASTER_CODEGRAPH_NODE"}
            with patch.dict(os.environ, env, clear=True), patch.object(
                codegraph_repair, "_CURSOR_NODE_CANDIDATES_MAC", ()
            ), patch.object(
                codegraph_repair, "_CURSOR_NODE_CANDIDATES_LINUX", ()
            ), patch.object(
                codegraph_repair, "_CURSOR_NODE_CANDIDATES_WIN", ()
            ), patch.object(
                codegraph_repair.shutil, "which", return_value=str(path_node)
            ):
                resolved = codegraph_repair.find_runtime_node()
            self.assertEqual(str(resolved), str(path_node))

    def test_find_runtime_node_honors_env_override(self) -> None:
        """PUPPETMASTER_CODEGRAPH_NODE wins over auto-detection."""
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            env_node = Path(tmp) / "envnode"
            env_node.write_text("ok", encoding="utf-8")
            with patch.dict(
                os.environ, {"PUPPETMASTER_CODEGRAPH_NODE": str(env_node)}
            ):
                resolved = codegraph_repair.find_runtime_node()
            self.assertEqual(str(resolved), str(env_node))

    def test_find_cursor_node_alias_points_at_runtime_node(self) -> None:
        """The back-compat alias must resolve to the generalized function."""
        from puppetmaster import codegraph_repair

        self.assertIs(
            codegraph_repair.find_cursor_node, codegraph_repair.find_runtime_node
        )

    def test_find_codegraph_install_from_shim_when_npm_misses(self) -> None:
        """When `npm root -g` points at the wrong prefix, follow the shim.

        Reproduces the cross-prefix trap: the package is installed under one
        Node's prefix (e.g. Homebrew) while the PATH npm leads elsewhere (e.g.
        a pyenv/Hermes npm). `npm root -g` misses the package, but the
        `codegraph` shim symlinks straight into it.
        """
        from puppetmaster import codegraph_repair

        with TemporaryDirectory() as tmp:
            # Real install lives under a "homebrew" prefix.
            pkg = Path(tmp) / "homebrew" / "lib" / "node_modules" / "@colbymchenry" / "codegraph"
            (pkg / "dist" / "bin").mkdir(parents=True)
            real_js = pkg / "dist" / "bin" / "codegraph.js"
            real_js.write_text("// stub", encoding="utf-8")
            # `codegraph` shim symlinks to the real JS entry point.
            shim = Path(tmp) / "bin" / "codegraph"
            shim.parent.mkdir(parents=True)
            shim.symlink_to(real_js)
            # `npm root -g` returns a DIFFERENT prefix that lacks the package.
            empty_root = Path(tmp) / "other" / "node_modules"
            empty_root.mkdir(parents=True)

            def fake_run(cmd, **kwargs):  # noqa: ANN001
                return subprocess.CompletedProcess(cmd, 0, str(empty_root) + "\n", "")

            def fake_which(name):  # noqa: ANN001
                if name == "npm":
                    return "/usr/bin/npm"
                if name == "codegraph":
                    return str(shim)
                return None

            with patch.object(
                codegraph_repair.subprocess, "run", side_effect=fake_run
            ), patch.object(
                codegraph_repair.shutil, "which", side_effect=fake_which
            ):
                resolved = codegraph_repair.find_codegraph_install()
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved, pkg.resolve())

    def test_find_codegraph_install_survives_garbage_npm_root(self) -> None:
        """A non-path `npm root -g` result must not raise ENAMETOOLONG.

        Regression: a misconfigured npm (or, as surfaced in CI, a test that
        globally mocks subprocess.run to return a giant JSON blob) made
        find_codegraph_install feed a 50KB string into Path(...).is_dir(),
        raising `OSError: File name too long`. The function must treat any
        result that doesn't look like a single path as "not found" and fall
        through to the shim resolver (here: no shim -> None), never crash.
        """
        from puppetmaster import codegraph_repair

        giant = '{"status": "finished", "result": "' + ("PADDING line\\n" * 4000) + '"}'

        def fake_run(cmd, **kwargs):  # noqa: ANN001
            return subprocess.CompletedProcess(cmd, 0, giant, "")

        def fake_which(name):  # noqa: ANN001
            if name == "npm":
                return "/usr/bin/npm"
            return None  # no codegraph shim

        with patch.object(
            codegraph_repair.subprocess, "run", side_effect=fake_run
        ), patch.object(
            codegraph_repair.shutil, "which", side_effect=fake_which
        ):
            # Must not raise; resolves to None because nothing valid was found.
            resolved = codegraph_repair.find_codegraph_install()
        self.assertIsNone(resolved)

    def test_looks_like_single_path_rejects_multiline_and_huge(self) -> None:
        from puppetmaster import codegraph_repair as cr

        self.assertTrue(cr._looks_like_single_path("/opt/homebrew/lib/node_modules"))
        self.assertFalse(cr._looks_like_single_path(""))
        self.assertFalse(cr._looks_like_single_path("line1\nline2"))
        self.assertFalse(cr._looks_like_single_path("has\x00null"))
        self.assertFalse(cr._looks_like_single_path("x" * 5000))

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

    def _reset_codegraph_autoheal(self) -> None:
        from puppetmaster import codegraph as codegraph_module

        codegraph_module.reset_codegraph_autoheal_state()

    def test_codegraph_native_broken_detects_hard_module_load_failure(self) -> None:
        """The hard `NODE_MODULE_VERSION` load error is detected, not just WASM."""
        from puppetmaster.codegraph import codegraph_native_sqlite_broken

        stderr = (
            "Error: The module '/x/better_sqlite3.node' was compiled against a "
            "different Node.js version using NODE_MODULE_VERSION 127. This version "
            "of Node.js requires NODE_MODULE_VERSION 131."
        )
        self.assertTrue(codegraph_native_sqlite_broken(stderr))

    def test_run_codegraph_cli_autoheals_and_retries_on_abi_error(self) -> None:
        """A better-sqlite3 ABI failure triggers a one-shot rebuild + retry."""
        from puppetmaster import codegraph as codegraph_module
        from puppetmaster.codegraph_repair import RepairResult

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        broken = {
            "ok": False,
            "command": "codegraph status",
            "cwd": "/repo",
            "returncode": 1,
            "stdout": "",
            "stderr": "better_sqlite3.node was compiled against a different Node.js version",
        }
        healed = {
            "ok": True,
            "command": "codegraph status",
            "cwd": "/repo",
            "returncode": 0,
            "stdout": "Backend: native; nodes: 42",
            "stderr": "",
        }
        once = MagicMock(side_effect=[broken, healed])
        repair = MagicMock(return_value=RepairResult(ok=True, message="rebuilt"))

        with patch.object(codegraph_module, "codegraph_available", return_value=True), \
            patch.object(codegraph_module, "codegraph_initialized", return_value=True), \
            patch.object(codegraph_module, "_run_codegraph_once", once), \
            patch("puppetmaster.codegraph_repair.repair_codegraph_sqlite", repair):
            result = codegraph_module.run_codegraph_cli(["status"], "/repo")

        self.assertTrue(result["ok"])
        self.assertEqual(once.call_count, 2)
        self.assertEqual(repair.call_count, 1)
        self.assertEqual(result["autoheal"], {"ok": True, "message": "rebuilt"})

    def test_run_codegraph_cli_no_autoheal_on_clean_failure(self) -> None:
        """A normal non-ABI failure must NOT trigger an expensive rebuild."""
        from puppetmaster import codegraph as codegraph_module

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        failure = {
            "ok": False,
            "command": "codegraph search foo",
            "cwd": "/repo",
            "returncode": 2,
            "stdout": "",
            "stderr": "no results found",
        }
        once = MagicMock(return_value=failure)
        repair = MagicMock()

        with patch.object(codegraph_module, "codegraph_available", return_value=True), \
            patch.object(codegraph_module, "codegraph_initialized", return_value=True), \
            patch.object(codegraph_module, "_run_codegraph_once", once), \
            patch("puppetmaster.codegraph_repair.repair_codegraph_sqlite", repair):
            result = codegraph_module.run_codegraph_cli(["search", "foo"], "/repo")

        self.assertFalse(result["ok"])
        self.assertEqual(once.call_count, 1)
        repair.assert_not_called()
        self.assertNotIn("autoheal", result)

    def test_run_codegraph_cli_autoheal_disabled_by_env(self) -> None:
        """PUPPETMASTER_CODEGRAPH_AUTOHEAL=0 disables the rebuild entirely."""
        from puppetmaster import codegraph as codegraph_module

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        broken = {
            "ok": False,
            "command": "codegraph status",
            "cwd": "/repo",
            "returncode": 1,
            "stdout": "",
            "stderr": "better_sqlite3 NODE_MODULE_VERSION mismatch",
        }
        once = MagicMock(return_value=broken)
        repair = MagicMock()

        with patch.dict(os.environ, {"PUPPETMASTER_CODEGRAPH_AUTOHEAL": "0"}), \
            patch.object(codegraph_module, "codegraph_available", return_value=True), \
            patch.object(codegraph_module, "codegraph_initialized", return_value=True), \
            patch.object(codegraph_module, "_run_codegraph_once", once), \
            patch("puppetmaster.codegraph_repair.repair_codegraph_sqlite", repair):
            result = codegraph_module.run_codegraph_cli(["status"], "/repo")

        self.assertFalse(result["ok"])
        self.assertEqual(once.call_count, 1)
        repair.assert_not_called()

    def test_run_codegraph_cli_no_retry_when_repair_fails(self) -> None:
        """A failed rebuild must NOT trigger a blind retry of the command."""
        from puppetmaster import codegraph as codegraph_module
        from puppetmaster.codegraph_repair import RepairResult

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        broken = {
            "ok": False,
            "command": "codegraph status",
            "cwd": "/repo",
            "returncode": 1,
            "stdout": "",
            "stderr": "better_sqlite3.node was compiled against a different Node.js version",
        }
        once = MagicMock(return_value=broken)
        repair = MagicMock(return_value=RepairResult(ok=False, message="rebuild failed"))

        with patch.object(codegraph_module, "codegraph_available", return_value=True), \
            patch.object(codegraph_module, "codegraph_initialized", return_value=True), \
            patch.object(codegraph_module, "_run_codegraph_once", once), \
            patch("puppetmaster.codegraph_repair.repair_codegraph_sqlite", repair):
            result = codegraph_module.run_codegraph_cli(["status"], "/repo")

        self.assertFalse(result["ok"])
        self.assertEqual(once.call_count, 1)
        self.assertEqual(repair.call_count, 1)
        self.assertEqual(result["autoheal"], {"ok": False, "message": "rebuild failed"})

    def test_codegraph_should_autoheal_uses_strict_abi_predicate(self) -> None:
        """Only the hard Node-ABI load signature should trip auto-heal."""
        from puppetmaster import codegraph as codegraph_module

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        # A package-name mention alone is not enough (avoids false rebuilds).
        self.assertFalse(
            codegraph_module._codegraph_should_autoheal(
                {"ok": False, "stderr": "could not load better-sqlite3 plugin"}
            )
        )
        self.assertFalse(
            codegraph_module._codegraph_should_autoheal(
                {"ok": False, "stderr": "no results found"}
            )
        )
        self.assertTrue(
            codegraph_module._codegraph_should_autoheal(
                {"ok": False, "stderr": "Error: ... NODE_MODULE_VERSION 127 ..."}
            )
        )

    def test_codegraph_autoheal_skipped_when_env_install_pinned(self) -> None:
        """A user-pinned custom install must not be auto-rebuilt globally."""
        from puppetmaster import codegraph as codegraph_module

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        with TemporaryDirectory() as tmp:
            node = Path(tmp) / "node"
            js = Path(tmp) / "codegraph.js"
            node.write_text("#!/bin/sh\n")
            js.write_text("// codegraph\n")
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_CODEGRAPH_NODE": str(node),
                    "PUPPETMASTER_CODEGRAPH_JS": str(js),
                },
            ):
                self.assertFalse(
                    codegraph_module._codegraph_should_autoheal(
                        {"ok": False, "stderr": "NODE_MODULE_VERSION mismatch"}
                    )
                )

    def test_codegraph_autoheal_claim_is_one_at_a_time_with_cooldown(self) -> None:
        """Only one caller claims an attempt; a failed attempt is gated by cooldown."""
        from puppetmaster import codegraph as codegraph_module

        self.addCleanup(self._reset_codegraph_autoheal)
        self._reset_codegraph_autoheal()

        self.assertTrue(codegraph_module._claim_codegraph_autoheal())
        # Still in-progress -> nobody else can claim.
        self.assertFalse(codegraph_module._claim_codegraph_autoheal())
        # Mark the attempt done-but-failed; cooldown should still block reclaim.
        with codegraph_module._AUTOHEAL_LOCK:
            codegraph_module._AUTOHEAL_STATE["in_progress"] = False
        self.assertFalse(codegraph_module._claim_codegraph_autoheal())
        # After cooldown elapses, a fresh attempt is allowed (not wedged forever).
        with codegraph_module._AUTOHEAL_LOCK:
            codegraph_module._AUTOHEAL_STATE["last_attempt_at"] = (
                time.monotonic() - codegraph_module._AUTOHEAL_COOLDOWN_SECONDS - 1
            )
        self.assertTrue(codegraph_module._claim_codegraph_autoheal())

    def test_cli_codegraph_passthrough_defaults_to_no_timeout(self) -> None:
        """Passthrough must not impose the short context timeout on long ops."""
        from puppetmaster import cli as cli_module

        captured = {}

        def fake_run(cli_args, cwd, *, require_initialized=True, timeout_seconds=None, **kwargs):
            captured["timeout_seconds"] = timeout_seconds
            return {"ok": True, "stdout": "", "stderr": "", "returncode": 0}

        with patch("puppetmaster.codegraph.run_codegraph_cli", side_effect=fake_run):
            rc = cli_module.main(["codegraph", "index"])

        self.assertEqual(rc, 0)
        self.assertIsNone(captured["timeout_seconds"])

    def test_codegraph_available_via_cursor_node_without_shim(self) -> None:
        """Available when Cursor-Node invocation resolves even if shim is off PATH."""
        from puppetmaster import codegraph as codegraph_module

        with patch.object(codegraph_module.shutil, "which", return_value=None), \
            patch.object(
                codegraph_module,
                "_cursor_codegraph_invocation",
                return_value=["/cursor/node", "/install/codegraph.js"],
            ):
            self.assertTrue(codegraph_module.codegraph_available())

    def test_cli_codegraph_passthrough_routes_through_run_codegraph_cli(self) -> None:
        """`puppetmaster codegraph <args>` delegates to the ABI-safe runner."""
        from puppetmaster import cli as cli_module

        captured = {}

        def fake_run(cli_args, cwd, *, require_initialized=True, **kwargs):
            captured["cli_args"] = cli_args
            captured["cwd"] = cwd
            captured["require_initialized"] = require_initialized
            return {"ok": True, "stdout": "ok\n", "stderr": "", "returncode": 0}

        with patch("puppetmaster.codegraph.run_codegraph_cli", side_effect=fake_run):
            rc = cli_module.main(["codegraph", "--cwd", "/repo", "search", "router"])

        self.assertEqual(rc, 0)
        self.assertEqual(captured["cli_args"], ["search", "router"])
        self.assertEqual(captured["cwd"], "/repo")
        self.assertTrue(captured["require_initialized"])

    def test_cli_codegraph_passthrough_status_skips_init_requirement(self) -> None:
        """`status` runs even before the workspace is initialized."""
        from puppetmaster import cli as cli_module

        captured = {}

        def fake_run(cli_args, cwd, *, require_initialized=True, **kwargs):
            captured["require_initialized"] = require_initialized
            return {"ok": True, "stdout": "Backend: native\n", "stderr": "", "returncode": 0}

        with patch("puppetmaster.codegraph.run_codegraph_cli", side_effect=fake_run):
            rc = cli_module.main(["codegraph", "status"])

        self.assertEqual(rc, 0)
        self.assertFalse(captured["require_initialized"])

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
                    parent_pid=12345,
                    parent_process="datagrip",
                )
                self.assertTrue(path.exists())

                entries = mcp_registry.list_entries()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].pid, os.getpid())
                self.assertEqual(entries[0].workspace, "/tmp/test-workspace")
                self.assertEqual(entries[0].parent_pid, 12345)
                self.assertEqual(entries[0].parent_process, "datagrip")
                self.assertTrue(entries[0].is_alive())
                self.assertFalse(entries[0].is_stale())

                payload = mcp_registry.summarize(entries)["servers"][0]
                self.assertEqual(payload["parent_pid"], 12345)
                self.assertEqual(payload["parent_process"], "datagrip")

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

    @unittest.skipIf(os.name == "nt", "parent-process detection uses POSIX ps")
    def test_mcp_registry_detects_parent_process_identity(self) -> None:
        from puppetmaster import mcp_registry

        completed = subprocess.CompletedProcess(
            ["ps", "-p", "4242", "-o", "comm="],
            0,
            "/Applications/DataGrip.app/Contents/MacOS/datagrip\n",
            "",
        )
        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                with patch.object(mcp_registry.os, "getppid", return_value=4242), patch.object(
                    mcp_registry.subprocess, "run", return_value=completed
                ) as run:
                    mcp_registry.register(pid=os.getpid(), workspace="/auto-parent")

                entry = mcp_registry.list_entries()[0]
                self.assertEqual(entry.parent_pid, 4242)
                self.assertEqual(entry.parent_process, "datagrip")
                run.assert_called_once()
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_registry_keeps_old_entries_without_parent_identity(self) -> None:
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                path = Path(tmp) / f"{os.getpid()}.json"
                path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "workspace": "/old",
                            "started_at": time.time(),
                            "last_heartbeat": time.time(),
                            "transport": "stdio",
                        }
                    ),
                    encoding="utf-8",
                )

                entry = mcp_registry.list_entries()[0]
                self.assertIsNone(entry.parent_pid)
                self.assertIsNone(entry.parent_process)
                payload = mcp_registry.summarize([entry])["servers"][0]
                self.assertIsNone(payload["parent_pid"])
                self.assertIsNone(payload["parent_process"])
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

    def test_mcp_registry_heartbeat_reregister_preserves_parent_identity(self) -> None:
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                path = mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/self-heal",
                    version="test-version",
                    parent_pid=12345,
                    parent_process="datagrip",
                )
                thread = mcp_registry.HeartbeatThread(path, interval_seconds=0.5)
                path.unlink()

                with patch.object(mcp_registry.os, "getppid", return_value=99999):
                    thread._reregister()

                entry = mcp_registry.list_entries()[0]
                self.assertEqual(entry.workspace, "/self-heal")
                self.assertEqual(entry.version, "test-version")
                self.assertEqual(entry.parent_pid, 12345)
                self.assertEqual(entry.parent_process, "datagrip")
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
                mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/listed",
                    parent_pid=12345,
                    parent_process="datagrip",
                )
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    rc = cli_module.main(["mcp", "list", "--json"])
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        self.assertEqual(rc, 0)
        snapshot = json.loads(buf.getvalue())
        self.assertGreaterEqual(snapshot["count"], 1)
        server = next(row for row in snapshot["servers"] if row["pid"] == os.getpid())
        self.assertEqual(server["parent_pid"], 12345)
        self.assertEqual(server["parent_process"], "datagrip")

    def test_cli_mcp_list_outputs_parent_process_columns(self) -> None:
        import io
        from puppetmaster import cli as cli_module
        from puppetmaster import mcp_registry

        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"] = tmp
            try:
                mcp_registry.register(
                    pid=os.getpid(),
                    workspace="/listed",
                    parent_pid=12345,
                    parent_process="datagrip",
                )
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    rc = cli_module.main(["mcp", "list"])
            finally:
                del os.environ["PUPPETMASTER_MCP_REGISTRY_DIR"]

        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("PPID", output)
        self.assertIn("PARENT", output)
        self.assertIn("12345", output)
        self.assertIn("datagrip", output)

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

    def test_max_block_seconds_default_and_disable(self) -> None:
        """The block cap defaults on, honors override, and disables at 0."""
        from puppetmaster import mcp_server

        os.environ.pop("PUPPETMASTER_MCP_MAX_BLOCK_SECONDS", None)
        self.assertEqual(
            mcp_server._resolve_max_block_seconds(),
            mcp_server._DEFAULT_MAX_BLOCK_SECONDS,
        )
        try:
            os.environ["PUPPETMASTER_MCP_MAX_BLOCK_SECONDS"] = "12.5"
            self.assertEqual(mcp_server._resolve_max_block_seconds(), 12.5)
            os.environ["PUPPETMASTER_MCP_MAX_BLOCK_SECONDS"] = "0"
            self.assertEqual(mcp_server._resolve_max_block_seconds(), 0.0)
            os.environ["PUPPETMASTER_MCP_MAX_BLOCK_SECONDS"] = "garbage"
            self.assertEqual(
                mcp_server._resolve_max_block_seconds(),
                mcp_server._DEFAULT_MAX_BLOCK_SECONDS,
            )
        finally:
            os.environ.pop("PUPPETMASTER_MCP_MAX_BLOCK_SECONDS", None)

    def test_capped_block_seconds_clamps_only_when_exceeded(self) -> None:
        """A requested block over the ceiling is clamped and flagged; under it passes through."""
        from puppetmaster import mcp_server

        try:
            os.environ["PUPPETMASTER_MCP_MAX_BLOCK_SECONDS"] = "45"
            self.assertEqual(mcp_server._capped_block_seconds(300.0), (45.0, True))
            self.assertEqual(mcp_server._capped_block_seconds(10.0), (10.0, False))
            self.assertEqual(mcp_server._capped_block_seconds(45.0), (45.0, False))
            os.environ["PUPPETMASTER_MCP_MAX_BLOCK_SECONDS"] = "0"
            self.assertEqual(mcp_server._capped_block_seconds(99999.0), (99999.0, False))
        finally:
            os.environ.pop("PUPPETMASTER_MCP_MAX_BLOCK_SECONDS", None)

    def test_feed_follow_caps_block_and_stamps_capped(self) -> None:
        """run_feed_follow clamps an oversized timeout under the Codex ceiling."""
        from puppetmaster import mcp_server

        observed: dict = {}

        class _FakeStore:
            def wait_for_events(self, job_id, since, timeout_seconds, poll_interval):
                observed["timeout_seconds"] = timeout_seconds
                return None

        with patch.dict(os.environ, {"PUPPETMASTER_MCP_MAX_BLOCK_SECONDS": "45"}), patch.object(
            mcp_server, "create_store", return_value=_FakeStore()
        ), patch.object(
            mcp_server, "mcp_state_dir", return_value=Path("/tmp/pm-state")
        ), patch(
            "puppetmaster.cli.artifact_feed_since", return_value=([], 0)
        ):
            result = mcp_server.run_feed_follow(
                {"job_id": "job_abc", "timeout_seconds": 300}
            )

        # The blocking wait must never see the oversized request.
        self.assertEqual(observed["timeout_seconds"], 45.0)
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(body["capped"])
        self.assertEqual(body["requested_timeout_seconds"], 300.0)
        self.assertEqual(body["effective_timeout_seconds"], 45.0)
        self.assertTrue(body["timed_out"])

    def test_extract_progress_token(self) -> None:
        """progressToken is pulled from params._meta only when present."""
        from puppetmaster.mcp_server import _extract_progress_token

        self.assertEqual(
            _extract_progress_token({"name": "t", "_meta": {"progressToken": "tok-1"}}),
            "tok-1",
        )
        self.assertEqual(
            _extract_progress_token({"name": "t", "_meta": {"progressToken": 7}}), 7
        )
        self.assertIsNone(_extract_progress_token({"name": "t"}))
        self.assertIsNone(_extract_progress_token({"name": "t", "_meta": {}}))
        self.assertIsNone(_extract_progress_token("not-a-dict"))

    def test_keepalive_emits_progress_only_with_token(self) -> None:
        """A progressToken yields notifications/progress frames; absent it, only logs."""
        from puppetmaster.mcp_server import _ToolCallKeepalive

        with_token: list[dict] = []
        keepalive = _ToolCallKeepalive(
            tool_name="puppetmaster_await_job",
            request_id="req-1",
            progress_token="tok-9",
            start_after_seconds=0.02,
            interval_seconds=0.02,
            emitter=lambda payload: (with_token.append(payload) or True),
        )
        keepalive.start()
        # Poll instead of a fixed sleep: on a loaded CI runner the keepalive
        # thread can be starved past a hardcoded window (observed flake on
        # GitHub macOS runners), but it always emits eventually.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            methods = {frame["method"] for frame in with_token}
            if {"notifications/message", "notifications/progress"} <= methods:
                break
            time.sleep(0.01)
        keepalive.stop(wait=True)

        methods = {frame["method"] for frame in with_token}
        self.assertIn("notifications/message", methods)
        self.assertIn("notifications/progress", methods)
        progress_frames = [
            f for f in with_token if f["method"] == "notifications/progress"
        ]
        first = progress_frames[0]
        self.assertNotIn("id", first)
        self.assertEqual(first["params"]["progressToken"], "tok-9")
        # Per MCP, the progress value must strictly increase across frames.
        values = [f["params"]["progress"] for f in progress_frames]
        self.assertEqual(values, sorted(values))
        if len(values) >= 2:
            self.assertLess(values[0], values[-1])

        without_token: list[dict] = []
        bare = _ToolCallKeepalive(
            tool_name="puppetmaster_await_job",
            request_id="req-2",
            start_after_seconds=0.02,
            interval_seconds=0.02,
            emitter=lambda payload: (without_token.append(payload) or True),
        )
        bare.start()
        deadline = time.time() + 5.0
        while time.time() < deadline and not without_token:
            time.sleep(0.01)
        bare.stop(wait=True)
        self.assertTrue(without_token)
        self.assertTrue(
            all(f["method"] == "notifications/message" for f in without_token)
        )

    def test_ensure_codex_timeouts_inserts_when_absent(self) -> None:
        """Timeout keys are added under the puppetmaster table when missing."""
        from puppetmaster import installers

        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[mcp_servers.puppetmaster]\n"
                'command = "python"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n',
                encoding="utf-8",
            )
            messages = installers._ensure_codex_timeouts(config)
            text = config.read_text(encoding="utf-8")

        self.assertIn(
            f"tool_timeout_sec = {installers.CODEX_TOOL_TIMEOUT_SEC}", text
        )
        self.assertIn(
            f"startup_timeout_sec = {installers.CODEX_STARTUP_TIMEOUT_SEC}", text
        )
        # The original keys must survive the insert.
        self.assertIn('command = "python"', text)
        self.assertTrue(any("set Codex timeouts" in m for m in messages))

    def test_ensure_codex_timeouts_preserves_user_values(self) -> None:
        """An existing tool_timeout_sec is never clobbered."""
        from puppetmaster import installers

        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[mcp_servers.puppetmaster]\n"
                'command = "python"\n'
                "tool_timeout_sec = 999\n",
                encoding="utf-8",
            )
            installers._ensure_codex_timeouts(config)
            text = config.read_text(encoding="utf-8")

        self.assertIn("tool_timeout_sec = 999", text)
        self.assertNotIn(f"tool_timeout_sec = {installers.CODEX_TOOL_TIMEOUT_SEC}", text)
        # startup was absent, so it should have been added.
        self.assertIn(
            f"startup_timeout_sec = {installers.CODEX_STARTUP_TIMEOUT_SEC}", text
        )

    def test_ensure_codex_timeouts_skips_when_table_missing(self) -> None:
        """No puppetmaster table => leave the file untouched, warn cleanly."""
        from puppetmaster import installers

        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            original = "[mcp_servers.other]\ncommand = \"x\"\n"
            config.write_text(original, encoding="utf-8")
            messages = installers._ensure_codex_timeouts(config)
            self.assertEqual(config.read_text(encoding="utf-8"), original)
        self.assertTrue(any("skipped Codex timeout tuning" in m for m in messages))

    def test_resolve_codegraph_invocation_prefers_cursor_node(self) -> None:
        """When Cursor Node + codegraph.js are both discoverable, prefer that pair."""
        from puppetmaster import codegraph as codegraph_mod
        from puppetmaster import codegraph_repair

        codegraph_mod.reset_cursor_codegraph_invocation_cache()
        self.addCleanup(codegraph_mod.reset_cursor_codegraph_invocation_cache)

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

    def test_resolve_codegraph_invocation_falls_back_to_bare_command(self) -> None:
        """With no Cursor install, no shim, and npx disabled, fall through to the
        bare ``codegraph`` command (whose failure surfaces the install hint)."""
        from puppetmaster import codegraph as codegraph_mod
        from puppetmaster import codegraph_repair

        codegraph_mod.reset_cursor_codegraph_invocation_cache()
        self.addCleanup(codegraph_mod.reset_cursor_codegraph_invocation_cache)

        with patch.object(codegraph_repair, "find_cursor_node", return_value=None), patch.object(
            codegraph_repair, "find_codegraph_install", return_value=None
        ), patch("puppetmaster.codegraph.shutil.which", return_value=None):
            argv = codegraph_mod.resolve_codegraph_invocation()
        self.assertEqual(argv, [codegraph_mod.CODEGRAPH_COMMAND])

    def test_resolve_codegraph_invocation_uses_npx_when_only_node_present(self) -> None:
        """No shim, no Cursor — but Node/npx present: resolve to the universal
        npx fallback so CodeGraph is available with zero manual install."""
        from puppetmaster import codegraph as codegraph_mod
        from puppetmaster import codegraph_repair

        codegraph_mod.reset_cursor_codegraph_invocation_cache()
        self.addCleanup(codegraph_mod.reset_cursor_codegraph_invocation_cache)

        def fake_which(cmd):
            return "/usr/local/bin/npx" if cmd == "npx" else None

        with patch.object(codegraph_repair, "find_cursor_node", return_value=None), patch.object(
            codegraph_repair, "find_codegraph_install", return_value=None
        ), patch("puppetmaster.codegraph.shutil.which", side_effect=fake_which), patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("PUPPETMASTER_CODEGRAPH_NO_NPX", None)
            argv = codegraph_mod.resolve_codegraph_invocation()
            self.assertEqual(argv, ["/usr/local/bin/npx", "-y", codegraph_mod.CODEGRAPH_PACKAGE])
            # The cheap readiness probe deliberately ignores the npx leg: npx
            # "availability" means Node exists, not that CodeGraph is warm (the
            # first run pays a cold download). Only explicit commands take it.
            self.assertFalse(codegraph_mod.codegraph_available())

    def test_npx_fallback_disabled_by_env(self) -> None:
        """PUPPETMASTER_CODEGRAPH_NO_NPX=1 removes the npx leg entirely."""
        from puppetmaster import codegraph as codegraph_mod

        def fake_which(cmd):
            return "/usr/local/bin/npx" if cmd == "npx" else None

        with patch("puppetmaster.codegraph.shutil.which", side_effect=fake_which), patch.object(
            codegraph_mod, "_cursor_codegraph_invocation", return_value=None
        ), patch.dict(os.environ, {"PUPPETMASTER_CODEGRAPH_NO_NPX": "1"}):
            self.assertIsNone(codegraph_mod._npx_codegraph_invocation())
            self.assertFalse(codegraph_mod.codegraph_available())
            # With npx disabled and no shim/Cursor, invocation falls through to
            # the bare command, whose "not found" surfaces the install hint.
            self.assertEqual(
                codegraph_mod.resolve_codegraph_invocation(),
                [codegraph_mod.CODEGRAPH_COMMAND],
            )

    def test_codegraph_available_is_cheap_and_never_provisions(self) -> None:
        """Regression: ``codegraph_available()`` must stay a cheap probe.

        It once counted the npx leg, so on a plain Node host (npx present, no
        shim/Cursor) it returned True — sending ``doctor`` and per-worker
        adapters into a synchronous ``npm install -g`` that blocked the MCP
        server (Codex "handshake never returned"). The probe must report
        not-ready without shelling out at all; the npx fallback belongs only to
        explicit ``run_codegraph_cli`` commands.
        """
        from puppetmaster import codegraph as codegraph_mod

        def fake_which(cmd):
            return "/usr/local/bin/npx" if cmd == "npx" else None

        with patch.object(
            codegraph_mod, "_cursor_codegraph_invocation", return_value=None
        ), patch("puppetmaster.codegraph.shutil.which", side_effect=fake_which), patch(
            "puppetmaster.codegraph.subprocess.run"
        ) as run_mock:
            self.assertFalse(codegraph_mod.codegraph_available())
            run_mock.assert_not_called()

    def test_ensure_provisioned_global_installs_then_uses_shim(self) -> None:
        """On first use with only Node present, a one-time global install lands
        the fast shim on PATH so future calls skip npx entirely."""
        from puppetmaster import codegraph as codegraph_mod

        codegraph_mod.reset_codegraph_provisioning_state()
        self.addCleanup(codegraph_mod.reset_codegraph_provisioning_state)

        which_calls = {"n": 0}

        def fake_which(cmd):
            if cmd == "npx":
                return "/usr/local/bin/npx"
            if cmd == "npm":
                return "/usr/local/bin/npm"
            if cmd == codegraph_mod.CODEGRAPH_COMMAND:
                # Shim appears only after the global install has run.
                return "/usr/local/bin/codegraph" if which_calls["n"] else None
            return None

        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["/usr/local/bin/npm", "install", "-g"]:
                which_calls["n"] = 1  # install succeeded → shim now on PATH
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(codegraph_mod, "_cursor_codegraph_invocation", return_value=None), patch(
            "puppetmaster.codegraph.shutil.which", side_effect=fake_which
        ), patch("puppetmaster.codegraph.subprocess.run", side_effect=fake_run) as run_mock, patch.dict(
            os.environ, {}, clear=False
        ):
            for var in ("PUPPETMASTER_CODEGRAPH_NO_NPX", "PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL",
                        "PUPPETMASTER_CODEGRAPH_NODE", "PUPPETMASTER_CODEGRAPH_JS"):
                os.environ.pop(var, None)
            self.assertTrue(codegraph_mod.ensure_codegraph_provisioned())
            installed = any(
                call.args[0][:3] == ["/usr/local/bin/npm", "install", "-g"]
                for call in run_mock.call_args_list
            )
            self.assertTrue(installed)

    def test_ensure_provisioned_returns_false_without_node(self) -> None:
        """No Node anywhere → provisioning can't bootstrap a Node CLI."""
        from puppetmaster import codegraph as codegraph_mod

        codegraph_mod.reset_codegraph_provisioning_state()
        self.addCleanup(codegraph_mod.reset_codegraph_provisioning_state)

        with patch("puppetmaster.codegraph.shutil.which", return_value=None), patch.object(
            codegraph_mod, "_cursor_codegraph_invocation", return_value=None
        ), patch.dict(os.environ, {}, clear=False):
            for var in ("PUPPETMASTER_CODEGRAPH_NODE", "PUPPETMASTER_CODEGRAPH_JS"):
                os.environ.pop(var, None)
            self.assertFalse(codegraph_mod.ensure_codegraph_provisioned())

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

        # Pin off the orthogonal parent-death reap: on loaded macOS CI runners a
        # test process can transiently reparent to launchd (getppid()==1), which
        # would fire this watcher independently of the idle-staleness path under
        # test. Parent-death has its own dedicated test elsewhere.
        no_orphan = patch.object(
            mcp_server._InputStalenessWatcher, "_parent_is_dead", return_value=False
        )
        no_orphan.start()
        self.addCleanup(no_orphan.stop)

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

        # Pin off parent-death (see test_input_staleness_watcher_triggers_when_idle):
        # an orphan reparent on CI must not be mistaken for the idle path here.
        no_orphan = patch.object(
            mcp_server._InputStalenessWatcher, "_parent_is_dead", return_value=False
        )
        no_orphan.start()
        self.addCleanup(no_orphan.stop)

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

        # Pin off parent-death (see test_input_staleness_watcher_triggers_when_idle):
        # an orphan reparent is the only other shutdown trigger, and we want this
        # test to isolate the heartbeat-reset guarantee, not parent liveness.
        no_orphan = patch.object(
            mcp_server._InputStalenessWatcher, "_parent_is_dead", return_value=False
        )
        no_orphan.start()
        self.addCleanup(no_orphan.stop)

        triggered = threading.Event()
        try:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT = time.time() - 3600
                mcp_server._ACTIVE_TOOL_CALLS = 0
            # Stale window (5s) is two orders of magnitude wider than the 0.1s
            # assertion wait, so even heavy CI scheduling jitter cannot let the
            # timer fire inside the window — killing the prior macOS flake while
            # preserving the guarantee: a fresh inbound message defers shutdown.
            watcher = mcp_server._InputStalenessWatcher(
                stale_after_seconds=5.0,
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

    def test_reaper_drops_exited_launchers_keeps_running(self) -> None:
        """`_reap_async_processes` waitpid-reaps exited launchers and keeps live ones."""
        from puppetmaster import mcp_server

        class _FakeLauncher:
            def __init__(self, exited: bool) -> None:
                self._exited = exited
                self.poll_calls = 0

            def poll(self):
                self.poll_calls += 1
                return 0 if self._exited else None

        done_a, done_b, running = _FakeLauncher(True), _FakeLauncher(True), _FakeLauncher(False)
        with mcp_server._ASYNC_PROCESSES_LOCK:
            saved = list(mcp_server.ASYNC_PROCESSES)
            mcp_server.ASYNC_PROCESSES[:] = [done_a, running, done_b]
        try:
            reaped = mcp_server._reap_async_processes()
            self.assertEqual(reaped, 2)
            self.assertEqual(mcp_server.ASYNC_PROCESSES, [running])
            self.assertGreaterEqual(running.poll_calls, 1)
        finally:
            with mcp_server._ASYNC_PROCESSES_LOCK:
                mcp_server.ASYNC_PROCESSES[:] = saved

    def test_track_async_process_sweeps_in_same_critical_section(self) -> None:
        """Appending a launcher also reaps already-exited ones, so bursts can't pile up."""
        from puppetmaster import mcp_server

        class _FakeLauncher:
            def __init__(self, exited: bool) -> None:
                self._exited = exited

            def poll(self):
                return 0 if self._exited else None

        stale, fresh = _FakeLauncher(True), _FakeLauncher(False)
        with mcp_server._ASYNC_PROCESSES_LOCK:
            saved = list(mcp_server.ASYNC_PROCESSES)
            mcp_server.ASYNC_PROCESSES[:] = [stale]
        try:
            mcp_server._track_async_process(fresh)
            self.assertEqual(mcp_server.ASYNC_PROCESSES, [fresh])
        finally:
            with mcp_server._ASYNC_PROCESSES_LOCK:
                mcp_server.ASYNC_PROCESSES[:] = saved

    def test_reaper_thread_reaps_on_interval_and_final_sweep(self) -> None:
        """The daemon reaper clears exited launchers on its tick and on stop()."""
        from puppetmaster import mcp_server

        class _FakeLauncher:
            def __init__(self) -> None:
                self.alive = True

            def poll(self):
                return None if self.alive else 0

        launcher = _FakeLauncher()
        with mcp_server._ASYNC_PROCESSES_LOCK:
            saved = list(mcp_server.ASYNC_PROCESSES)
            mcp_server.ASYNC_PROCESSES[:] = [launcher]
        reaper = mcp_server._AsyncProcessReaper(interval_seconds=0.02)
        try:
            reaper.start()
            time.sleep(0.1)
            # Still running -> not reaped yet.
            self.assertIn(launcher, mcp_server.ASYNC_PROCESSES)
            launcher.alive = False
            deadline = time.time() + 1.0
            while launcher in mcp_server.ASYNC_PROCESSES and time.time() < deadline:
                time.sleep(0.02)
            self.assertNotIn(launcher, mcp_server.ASYNC_PROCESSES)
        finally:
            reaper.stop()
            with mcp_server._ASYNC_PROCESSES_LOCK:
                mcp_server.ASYNC_PROCESSES[:] = saved

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

    def test_claude_code_adapter_streams_and_surfaces_live_log_on_timeout(self) -> None:
        """Claude Code runs through the streamed runner (live log + heartbeat),
        and a timeout reports ``failed`` + ``timeout`` while still exposing the
        live log path. The flat blocking ``subprocess.run`` path is gone."""
        streamed = StreamedProcess(
            returncode=None,
            stdout="partial",
            stderr="",
            timed_out=True,
            live_log_path="/tmp/claude_implement_live.log",
        )
        task = Task(
            job_id="job-claude-timeout",
            role="claude-code",
            instruction="ship a tiny change",
            adapter="claude-code",
            payload={"cwd": str(Path.cwd()), "allow_dirty": True, "disable_codegraph": True},
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/claude"), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch("puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
        ) as streamed_run, patch("puppetmaster.adapters.subprocess.run") as blocking_run:
            artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")

        streamed_run.assert_called_once()
        blocking_run.assert_not_called()
        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "failed")
        self.assertEqual(verification.payload["failure"], "timeout")
        self.assertEqual(verification.payload["live_log"], "/tmp/claude_implement_live.log")

    def test_claude_code_failure_classification_is_actionable(self) -> None:
        self.assertEqual(classify_claude_code_failure("please login first"), "not_authenticated")
        self.assertEqual(classify_claude_code_failure("Credit balance is too low"), "billing_or_quota")
        self.assertEqual(classify_claude_code_failure("permission denied"), "permission_denied")
        self.assertEqual(classify_claude_code_failure("model invalid"), "model_unavailable")
        self.assertEqual(
            classify_claude_code_failure(
                '{"type":"error","error":{"type":"not_found_error",'
                '"message":"model: claude-fable-5"}}'
            ),
            "model_unavailable",
        )
        self.assertEqual(
            classify_claude_code_failure(
                '{"type":"error","error":{"type":"permission_error",'
                '"message":"model claude-fable-5 is not permitted"}}'
            ),
            "model_unavailable",
        )

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

    def test_claude_code_run_persists_final_report_as_finding(self) -> None:
        """Claude's final message (the json envelope's `result`) is the worker's
        report; it must land as a FINDING instead of dying in a stdout tail."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
print('{"result":"Fixed the auth bug in auth.py; ran pytest, all green."}')
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)

            task = Task(
                job_id="job",
                role="claude-code",
                instruction="fix the auth bug",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                    "prompt": "Fix the auth bug.",
                    "disable_codegraph": True,
                },
            )
            artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")

        findings = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertEqual(len(findings), 1)
        self.assertIn("Fixed the auth bug", findings[0].payload["claim"])
        self.assertIn("adapter:claude-code", findings[0].evidence)
        self.assertIn("report:final-message", findings[0].evidence)

    def test_claude_code_prompt_carries_report_contract(self) -> None:
        """Claude workers get the reporting contract appended unless the prompt
        already demands the structured JSON artifact contract."""
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
                instruction="fix it",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                    "prompt": "Fix it.",
                },
            )
            with patch(
                "puppetmaster.adapters.enrich_prompt_with_codegraph",
                return_value=("Fix it.", False),
            ) as enrich:
                ClaudeCodeAdapter().run(task, "goal", "worker")

            self.assertIn("Reporting contract", enrich.call_args[0][0])

    def test_claude_code_adapter_captures_tracked_git_diff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            target = repo / "sample.txt"
            target.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True, capture_output=True)
            # Commit so the tree is clean (HEAD exists); the agent's edit is then
            # the only change. Staged-but-uncommitted content now counts as dirty
            # (it's captured by `git diff HEAD`), which is the corrected gating.
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=test@example.com",
                    "-c",
                    "user.name=Test",
                    "commit",
                    "-m",
                    "seed",
                ],
                cwd=repo,
                check=True,
                capture_output=True,
            )
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

    def test_claude_code_adapter_records_measured_tokens(self) -> None:
        # Token metering must be universal, not Cursor-only: when Claude Code's
        # JSON stdout carries a usage object, the verification artifact records
        # it as *measured* (tokens_estimated=False) so rollup/cost tell the truth.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=t@e.co", "-c", "user.name=T", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            fake_claude = root / "fake_claude.py"
            fake_claude.write_text(
                """#!/usr/bin/env python3
import json
print(json.dumps({"result": "ok", "usage": {"input_tokens": 321, "output_tokens": 77}}))
""",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)
            task = Task(
                job_id="job",
                role="claude-code",
                instruction="noop",
                adapter="claude-code",
                payload={
                    "executable": [sys.executable, str(fake_claude)],
                    "cwd": str(repo),
                    "timeout_seconds": 10,
                },
            )
            artifacts = ClaudeCodeAdapter().run(task, "goal", "worker")
            verification = artifacts[0].payload
            self.assertEqual(verification["tokens_in"], 321)
            self.assertEqual(verification["tokens_out"], 77)
            self.assertFalse(verification["tokens_estimated"])

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

    def test_git_snapshot_captures_staged_and_untracked_changes(self) -> None:
        from puppetmaster.adapters import git_snapshot

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            # Stage an edit to a tracked file (plain `git diff` would miss this).
            (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
            # Create a brand-new untracked file (plain `git diff` never shows it).
            (repo / "new_file.txt").write_text("hello\n", encoding="utf-8")

            snap = git_snapshot(repo)

            self.assertTrue(snap["is_worktree"])
            self.assertIn("tracked.txt", snap["changed_files"])  # staged change caught
            self.assertIn("new_file.txt", snap["untracked_files"])
            # Both the staged edit and the untracked file appear in the diff.
            self.assertIn("two", snap["diff"])
            self.assertIn("new_file.txt", snap["diff"])
            self.assertIn("+hello", snap["diff"])

    def test_git_snapshot_worker_diff_excludes_preexisting_dirty_files(self) -> None:
        from puppetmaster.adapters import git_snapshot

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "pre.py").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "pre.py"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )

            (repo / "pre.py").write_text("user dirty\n", encoding="utf-8")
            before = git_snapshot(repo)
            (repo / "helper.py").write_text("pm change\n", encoding="utf-8")

            after = git_snapshot(repo, base_tree=before["tree"])

            self.assertIn("pre.py", after["diff"])
            self.assertIn("helper.py", after["worker_changed_files"])
            self.assertIn("helper.py", after["worker_untracked_files"])
            self.assertIn("helper.py", after["worker_diff"])
            self.assertNotIn("pre.py", after["worker_diff"])

    def test_git_snapshot_worker_diff_fails_closed_without_after_tree(self) -> None:
        from puppetmaster.adapters import _should_emit_patch_artifact, git_snapshot

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "pre.py").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "pre.py"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )

            (repo / "pre.py").write_text("user dirty\n", encoding="utf-8")
            before = git_snapshot(repo)
            (repo / "helper.py").write_text("pm change\n", encoding="utf-8")

            with patch("puppetmaster.adapters.git_worktree_tree", return_value=""):
                after = git_snapshot(repo, base_tree=before["tree"])

            self.assertEqual(after["worker_changed_files"], [])
            self.assertEqual(after["worker_diff"], "")
            self.assertFalse(_should_emit_patch_artifact(before, after))

    def test_git_snapshot_worker_tree_preserves_real_index(self) -> None:
        from puppetmaster.adapters import git_snapshot

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            target = repo / "tracked.py"
            target.write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )

            target.write_text("staged\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True, capture_output=True)
            target.write_text("worktree\n", encoding="utf-8")
            (repo / "new.py").write_text("untracked\n", encoding="utf-8")

            cached_before = subprocess.run(
                ["git", "diff", "--cached", "--binary"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            worktree_before = subprocess.run(
                ["git", "diff", "--binary"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout

            git_snapshot(repo)

            cached_after = subprocess.run(
                ["git", "diff", "--cached", "--binary"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            worktree_after = subprocess.run(
                ["git", "diff", "--binary"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(cached_after, cached_before)
            self.assertEqual(worktree_after, worktree_before)

    def test_git_snapshot_worker_diff_covers_whole_repo_from_subdir(self) -> None:
        from puppetmaster.adapters import git_snapshot

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "pkg").mkdir()
            (repo / "pkg" / "seed.py").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-m", "seed"],
                cwd=repo,
                check=True,
                capture_output=True,
            )

            subdir = repo / "pkg"
            before = git_snapshot(subdir)
            (repo / "outside.py").write_text("pm change\n", encoding="utf-8")

            after = git_snapshot(subdir, base_tree=before["tree"])

            self.assertIn("outside.py", after["worker_changed_files"])
            self.assertIn("outside.py", after["worker_untracked_files"])
            self.assertIn("outside.py", after["worker_diff"])

    def test_git_snapshot_flags_non_worktree(self) -> None:
        from puppetmaster.adapters import git_snapshot

        with TemporaryDirectory() as tmp:
            snap = git_snapshot(Path(tmp))
            self.assertFalse(snap["is_worktree"])

    def test_implement_blocks_outside_worktree(self) -> None:
        with TemporaryDirectory() as tmp:
            task = Task(
                job_id="job",
                role="cursor",
                instruction="add a helper",
                adapter="cursor",
                payload={
                    "prompt": "Add a helper",
                    "cwd": tmp,  # not a git work tree
                    "mode": "implement",
                    "disable_codegraph": True,
                },
            )
            # git_snapshot runs real git and reports not-a-worktree, so the
            # guard returns a blocked artifact before the agent is ever spawned.
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].payload["result"], "blocked")
            self.assertEqual(artifacts[0].payload["failure"], "not_a_worktree")

    def test_build_patch_payload_redacts_and_marks_truncation(self) -> None:
        from puppetmaster.adapters import build_patch_payload

        secret = "sk-" + "a" * 40
        # Secret near the tail so it survives truncation (the inline excerpt
        # keeps the last 20k chars) — proving redaction runs on the full diff.
        big_diff = "diff --git a/f b/f\n" + ("+x\n" * 30000) + "+" + secret + "\n"
        before = {"sha": "base", "changed_files": ["f"], "untracked_files": [], "diff": ""}
        after = {"sha": "uncommitted", "changed_files": ["f"], "untracked_files": [], "diff": big_diff}
        task = Task(job_id="job", role="cursor", instruction="x", adapter="cursor", payload={})

        payload = build_patch_payload(
            task=task,
            before=before,
            after=after,
            status="applied",
            change="changed",
            sidecar_name="t",
        )

        self.assertNotIn(secret, payload["unified_diff"])
        self.assertIn("sk-<redacted>", payload["unified_diff"])
        self.assertTrue(payload["diff_truncated"])
        self.assertLessEqual(len(payload["unified_diff"]), 20000)
        self.assertGreater(payload["diff_total_chars"], 20000)

    def test_start_implement_supports_codex_adapter(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch(
            "puppetmaster.platform_lock.enabled_adapters", return_value={"codex"}
        ), patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            result = mcp_server.start_implement({"goal": "ship it", "cwd": "."})

        self.assertEqual(captured["command"][0], "codex")
        self.assertEqual(result["implement_adapter"], "codex")

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

    def test_codex_adapter_streams_and_surfaces_live_log(self) -> None:
        """Codex must run through the streamed runner (live sidecar log +
        heartbeat) — not a flat, silent blocking ``subprocess.run`` — so a long
        run is visibly alive instead of looking stalled. The verification
        payload exposes the live log path, and the legacy blocking path is
        never used."""
        agent_text = json.dumps(
            {"artifacts": [{"type": "finding", "claim": "ok", "evidence": ["a.py:1"], "confidence": 0.9}]}
        )
        events_stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "th_test"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {"type": "item.completed", "item": {"type": "agent_message", "text": agent_text}}
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cached_input_tokens": 0,
                            "reasoning_output_tokens": 0,
                        },
                    }
                ),
            ]
        )
        streamed = StreamedProcess(
            returncode=0,
            stdout=events_stdout,
            stderr="",
            timed_out=False,
            live_log_path="/tmp/codex_exec_live.log",
        )
        task = Task(
            id="t-codex-stream",
            job_id="job-codex-stream",
            role="codex-review",
            adapter="codex",
            instruction="Review the repo.",
            payload={"cwd": str(Path.cwd()), "sandbox": "read-only", "disable_codegraph": True},
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
        ), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
        ) as streamed_run, patch(
            "puppetmaster.adapters.subprocess.run"
        ) as blocking_run:
            artifacts = CodexAdapter().run(task, "goal", "worker")

        streamed_run.assert_called_once()
        self.assertEqual(streamed_run.call_args.kwargs["sidecar_name"], "codex_exec")
        blocking_run.assert_not_called()
        verification = artifacts[0]
        self.assertEqual(verification.payload["adapter"], "codex")
        self.assertEqual(verification.payload["result"], "passed")
        self.assertEqual(verification.payload["live_log"], "/tmp/codex_exec_live.log")

    def test_codex_write_capable_prose_is_report_not_degraded(self) -> None:
        """A workspace-write Codex run that reports in prose did its job — the
        report becomes a FINDING and the run passes. Only read-only runs (whose
        whole contract is structured findings) stay degraded on prose."""
        events_stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "th_test"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Patched cli.py to handle empty args; pytest passes.",
                        },
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
            ]
        )
        streamed = StreamedProcess(
            returncode=0,
            stdout=events_stdout,
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        task = Task(
            id="t-codex-report",
            job_id="job-codex-report",
            role="codex-implement",
            adapter="codex",
            instruction="Fix the CLI crash.",
            payload={"cwd": str(Path.cwd()), "sandbox": "workspace-write", "disable_codegraph": True},
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
        ), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
        ):
            artifacts = CodexAdapter().run(task, "goal", "worker")

        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "passed")
        findings = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertEqual(len(findings), 1)
        self.assertIn("Patched cli.py", findings[0].payload["claim"])
        self.assertNotIn(ArtifactType.RISK, [a.type for a in artifacts])

    def test_codex_read_only_prose_still_degrades(self) -> None:
        """Read-only Codex (review-style) keeps strict semantics: prose without
        structured artifacts is degraded, with the RISK marker preserved."""
        events_stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Looks fine to me."},
                    }
                ),
            ]
        )
        streamed = StreamedProcess(
            returncode=0,
            stdout=events_stdout,
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        task = Task(
            id="t-codex-ro",
            job_id="job-codex-ro",
            role="codex-review",
            adapter="codex",
            instruction="Review the repo.",
            payload={"cwd": str(Path.cwd()), "sandbox": "read-only", "disable_codegraph": True},
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
        ), patch("puppetmaster.adapters.run_streamed_subprocess", return_value=streamed):
            artifacts = CodexAdapter().run(task, "goal", "worker")

        self.assertEqual(artifacts[0].payload["result"], "degraded")
        risks = [a for a in artifacts if a.type == ArtifactType.RISK]
        self.assertEqual(len(risks), 1)
        self.assertNotIn(ArtifactType.FINDING, [a.type for a in artifacts])

    def test_generated_swarm_codex_read_only_allows_dirty_diff_review(self) -> None:
        """A generated MCP analysis swarm may route to Codex. Its read-only
        payload must keep Codex out of the full-edit dirty-worktree guard so the
        worker can review the caller's existing dirty diff.
        """
        from puppetmaster.mcp_server import write_generated_swarm_config

        events_stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": json.dumps(
                                {
                                    "artifacts": [
                                        {
                                            "type": "finding",
                                            "claim": "dirty diff reviewed",
                                            "evidence": ["diff"],
                                            "confidence": 0.9,
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                )
            ]
        )
        streamed = StreamedProcess(
            returncode=0,
            stdout=events_stdout,
            stderr="",
            timed_out=False,
            live_log_path=None,
        )
        with TemporaryDirectory() as tmp:
            config_path = write_generated_swarm_config(
                {"goal": "review dirty diff", "cwd": tmp, "state_dir": str(Path(tmp) / "state")},
                ["audit"],
                "cursor",
            )
            payload = json.loads(Path(config_path).read_text())["workers"][0]["payload"]
            task = Task(
                id="t-generated-codex-ro",
                job_id="job-generated-codex-ro",
                role="audit",
                adapter="codex",
                instruction="Review the dirty diff.",
                payload={**payload, "model": "gpt-5.4-mini", "disable_codegraph": True},
            )
            dirty = {
                "sha": "s",
                "changed_files": ["puppetmaster/mcp_server.py"],
                "untracked_files": [],
                "diff": "diff --git a/puppetmaster/mcp_server.py b/puppetmaster/mcp_server.py",
            }
            with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
                "puppetmaster.adapters.git_snapshot", side_effect=[dirty, dirty]
            ), patch("puppetmaster.adapters.worktree_guard") as guard, patch(
                "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
            ) as run:
                artifacts = CodexAdapter().run(task, "goal", "worker")

        guard.assert_not_called()
        command = run.call_args.kwargs["command"]
        self.assertIn("--sandbox", command)
        self.assertIn("read-only", command)
        self.assertEqual(artifacts[0].payload["result"], "passed")
        self.assertTrue(any(a.type == ArtifactType.FINDING for a in artifacts))

    def test_codex_adapter_timeout_surfaces_failed_with_live_log(self) -> None:
        """A timed-out Codex run reports ``failed`` + ``timeout`` and still
        carries the live log path so the operator can see how far it got."""
        streamed = StreamedProcess(
            returncode=None,
            stdout="partial output",
            stderr="",
            timed_out=True,
            live_log_path="/tmp/codex_exec_live.log",
        )
        task = Task(
            id="t-codex-timeout",
            job_id="job-codex-timeout",
            role="codex-review",
            adapter="codex",
            instruction="Review the repo.",
            payload={"cwd": str(Path.cwd()), "sandbox": "read-only", "disable_codegraph": True},
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
        ), patch("puppetmaster.adapters.run_streamed_subprocess", return_value=streamed):
            artifacts = CodexAdapter().run(task, "goal", "worker")

        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "failed")
        self.assertEqual(verification.payload["failure"], "timeout")
        self.assertEqual(verification.payload["live_log"], "/tmp/codex_exec_live.log")

    def test_codex_adapter_bypassed_read_only_uses_worktree_guard(self) -> None:
        from puppetmaster.adapters import verification_artifact

        task = Task(
            id="t-codex-bypass",
            job_id="job-codex-bypass",
            role="codex-review",
            adapter="codex",
            instruction="Review the repo.",
            payload={
                "cwd": str(Path.cwd()),
                "sandbox": "read-only",
                "dangerously_bypass_approvals_and_sandbox": True,
                "disable_codegraph": True,
            },
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        blocked = [
            verification_artifact(
                task=task,
                worker_id="worker",
                adapter="codex",
                check="guard",
                result="blocked",
                confidence=0.9,
                evidence=["guarded"],
                payload={"failure": "guarded"},
            )
        ]
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/codex"), patch(
            "puppetmaster.adapters.git_snapshot", return_value=clean
        ), patch(
            "puppetmaster.adapters.worktree_guard", return_value=blocked
        ) as guard, patch(
            "puppetmaster.adapters.run_streamed_subprocess"
        ) as streamed:
            artifacts = CodexAdapter().run(task, "goal", "worker")

        guard.assert_called_once()
        streamed.assert_not_called()
        self.assertIs(artifacts, blocked)

    def test_codex_and_claude_workers_get_codegraph_cli_on_pythonpath(self) -> None:
        """Every agentic harness must put Puppetmaster's source root on the
        worker's PYTHONPATH so the injected `python -m puppetmaster codegraph`
        instructions resolve to this tree instead of a stale pip install.

        Regression guard: Codex and Claude previously omitted
        inject_worker_cli_env (only Cursor had it), so on a host with an older
        pip puppetmaster their workers hit "unknown command" on codegraph and
        silently fell back to grep."""
        from puppetmaster.adapters import ClaudeCodeAdapter, CodexAdapter
        from puppetmaster.codegraph import puppetmaster_source_root

        root = puppetmaster_source_root()
        streamed = StreamedProcess(
            returncode=0, stdout="{}", stderr="", timed_out=False, live_log_path=None
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}

        def captured_env(adapter_cls, adapter_name, role):
            task = Task(
                id=f"t-{adapter_name}-env",
                job_id=f"job-{adapter_name}-env",
                role=role,
                adapter=adapter_name,
                instruction="Do work.",
                payload={
                    "cwd": str(Path.cwd()),
                    "sandbox": "read-only",
                    "disable_codegraph": True,
                },
            )
            with patch(
                "puppetmaster.adapters.resolve_command",
                return_value=f"/usr/bin/{adapter_name}",
            ), patch(
                "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
            ), patch(
                "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
            ) as run:
                adapter_cls().run(task, "goal", "worker")
            return run.call_args.kwargs["env"]

        codex_env = captured_env(CodexAdapter, "codex", "codex-review")
        self.assertTrue(codex_env["PYTHONPATH"].startswith(root))

        claude_env = captured_env(ClaudeCodeAdapter, "claude-code", "implement")
        self.assertTrue(claude_env["PYTHONPATH"].startswith(root))

    def test_run_streamed_subprocess_closes_stdin_so_readers_see_eof(self) -> None:
        """The streamed runner must close the child's stdin (DEVNULL). A CLI
        that reads stdin would otherwise block forever on a non-interactive
        worker — the silent "stall" we are eliminating. With stdin closed, a
        reader sees EOF immediately and exits cleanly within the timeout."""
        from puppetmaster.adapters import run_streamed_subprocess

        task = Task(
            id="t-stdin",
            job_id="job-stdin",
            role="codex-review",
            adapter="codex",
            instruction="x",
            payload={},
        )
        result = run_streamed_subprocess(
            command=[
                sys.executable,
                "-c",
                "import sys; data = sys.stdin.read(); sys.stdout.write('EOF' if data == '' else 'BLOCKED')",
            ],
            env=None,
            task=task,
            sidecar_name="stdin_probe",
            timeout_seconds=10,
        )
        self.assertFalse(result.timed_out)
        self.assertEqual(result.returncode, 0)
        self.assertIn("EOF", result.stdout)

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
        self.assertEqual(
            classify_codex_failure('{"error":{"code":"model_not_found"}}'),
            "model_unavailable",
        )
        self.assertEqual(classify_codex_failure("request timed out"), "timeout")
        self.assertEqual(classify_codex_failure("DNS resolution failed"), "network_error")
        self.assertEqual(classify_codex_failure("approval was denied by user"), "approval_denied")
        self.assertEqual(classify_codex_failure("sandbox: write blocked"), "sandbox_denied")
        self.assertEqual(classify_codex_failure("completely unrelated text"), "unknown")

    def test_build_hermes_chat_command_implement_flags(self) -> None:
        command = build_hermes_chat_command(
            prompt="ship it",
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            max_turns=42,
            toolsets="coding",
            yolo=True,
        )
        self.assertEqual(command[0], "hermes")
        self.assertIn("chat", command)
        self.assertIn("-q", command)
        self.assertIn("ship it", command)
        self.assertIn("-Q", command)
        self.assertIn("--source", command)
        self.assertEqual(command[command.index("--source") + 1], "tool")
        self.assertIn("--cli", command)
        self.assertIn("--yolo", command)
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "anthropic/claude-sonnet-4")
        self.assertIn("--provider", command)
        self.assertEqual(command[command.index("--provider") + 1], "anthropic")
        self.assertIn("--max-turns", command)
        self.assertEqual(command[command.index("--max-turns") + 1], "42")
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "coding")
        self.assertIn("--ignore-rules", command)

    def test_build_hermes_chat_command_analyze_omits_yolo(self) -> None:
        command = build_hermes_chat_command(
            prompt="review",
            toolsets="web,search",
            yolo=False,
        )
        self.assertNotIn("--yolo", command)
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "web,search")

    def test_build_hermes_chat_command_isolation_default_and_override(self) -> None:
        default_command = build_hermes_chat_command(prompt="x")
        self.assertIn("--ignore-rules", default_command)
        self.assertNotIn("--safe-mode", default_command)

        opted_out = build_hermes_chat_command(prompt="x", ignore_rules=False)
        self.assertNotIn("--ignore-rules", opted_out)

    def test_hermes_adapter_implement_uses_isolated_session(self) -> None:
        streamed = StreamedProcess(
            returncode=0,
            stdout="done",
            stderr="",
            timed_out=False,
            live_log_path="/tmp/hermes_implement_live.log",
        )
        task = Task(
            id="t-hermes-session",
            job_id="job-hermes-session",
            role="hermes-implement",
            adapter="hermes",
            instruction="Fix the bug.",
            payload={
                "cwd": str(Path.cwd()),
                "implement": True,
                "disable_codegraph": True,
            },
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
        ), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
        ) as streamed_run:
            HermesAdapter().run(task, "goal", "worker")

        streamed_run.assert_called_once()
        self.assertTrue(streamed_run.call_args.kwargs["start_new_session"])
        command = streamed_run.call_args.kwargs["command"]
        self.assertEqual(command[0], "/usr/bin/hermes")
        self.assertIn("chat", command)
        self.assertIn("--yolo", command)
        self.assertIn("--cli", command)
        self.assertIn("-Q", command)
        self.assertIn("--ignore-rules", command)
        self.assertEqual(command[command.index("--source") + 1], "tool")

    def test_hermes_implement_nonzero_exit_with_diff_is_success(self) -> None:
        streamed = StreamedProcess(
            returncode=1,
            stdout="Applied fix despite provider flake.",
            stderr="provider teardown warning",
            timed_out=False,
            live_log_path=None,
        )
        task = Task(
            id="t-hermes-diff",
            job_id="job-hermes-diff",
            role="hermes-implement",
            adapter="hermes",
            instruction="Fix the bug.",
            payload={
                "cwd": str(Path.cwd()),
                "implement": True,
                "disable_codegraph": True,
            },
        )
        clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
        dirty = {
            "sha": "s2",
            "changed_files": ["cli.py"],
            "untracked_files": [],
            "diff": "diff --git a/cli.py b/cli.py\n+fix",
            "worker_diff": "diff --git a/cli.py b/cli.py\n+fix",
        }
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch(
            "puppetmaster.adapters.git_snapshot", side_effect=[clean, dirty]
        ), patch(
            "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
        ):
            artifacts = HermesAdapter().run(task, "goal", "worker")

        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "passed")
        self.assertIsNone(verification.payload["failure"])
        self.assertTrue(verification.payload["has_work"])
        patches = [a for a in artifacts if a.type == ArtifactType.PATCH]
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].payload["status"], "applied")

    def test_hermes_reasoning_effort_injects_isolated_config(self) -> None:
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML (hermes extra) not installed")
        from puppetmaster.adapters import hermes_reasoning_effort_env

        with TemporaryDirectory() as tmp:
            real_home = Path(tmp) / "hermes_home"
            real_home.mkdir()
            (real_home / "config.yaml").write_text(
                "model:\n  default: claude-sonnet-4-5\n"
                "mcp_servers:\n  puppetmaster:\n    command: pm\n",
                encoding="utf-8",
            )
            (real_home / "auth.json").write_text('{"active_provider": "anthropic"}', encoding="utf-8")
            base_env = {"HERMES_HOME": str(real_home), "PATH": os.environ.get("PATH", "")}

            with hermes_reasoning_effort_env(base_env, "high") as run_env:
                effort_home = Path(run_env["HERMES_HOME"])
                # A fresh ephemeral home, never the user's real one.
                self.assertNotEqual(effort_home, real_home)
                # auth.json is preserved verbatim via symlink (credentials intact).
                self.assertTrue((effort_home / "auth.json").is_symlink())
                self.assertEqual(
                    (effort_home / "auth.json").read_text(), '{"active_provider": "anthropic"}'
                )
                # config.yaml is a real rewritten file carrying the effort, with
                # the user's MCP servers preserved.
                self.assertFalse((effort_home / "config.yaml").is_symlink())
                import yaml as _yaml

                cfg = _yaml.safe_load((effort_home / "config.yaml").read_text())
                self.assertEqual(cfg["agent"]["reasoning_effort"], "high")
                self.assertIn("puppetmaster", cfg["mcp_servers"])
                self.assertEqual(cfg["model"]["default"], "claude-sonnet-4-5")
            # Cleaned up on exit; the real home is untouched.
            self.assertFalse(effort_home.exists())
            self.assertTrue((real_home / "config.yaml").is_file())

    def test_hermes_reasoning_effort_env_sessions_are_hermetic(self) -> None:
        """An effort-run must not write sessions into the user's real ~/.hermes."""
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML (hermes extra) not installed")
        from puppetmaster.adapters import hermes_reasoning_effort_env

        with TemporaryDirectory() as tmp:
            real_home = Path(tmp) / "hermes_home"
            (real_home / "sessions").mkdir(parents=True)
            (real_home / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
            base_env = {"HERMES_HOME": str(real_home), "PATH": os.environ.get("PATH", "")}

            with hermes_reasoning_effort_env(base_env, "high") as run_env:
                effort_home = Path(run_env["HERMES_HOME"])
                effort_sessions = effort_home / "sessions"
                # sessions/ is a real throwaway dir, NOT a symlink to the real one.
                self.assertTrue(effort_sessions.is_dir())
                self.assertFalse(effort_sessions.is_symlink())
                # A session a worker writes lands in the temp home, not the real one.
                (effort_sessions / "abc.json").write_text("{}", encoding="utf-8")

            # The user's real session store stays empty after the run.
            self.assertEqual(list((real_home / "sessions").iterdir()), [])

    def test_hermes_reasoning_effort_env_passthrough_when_invalid_or_empty(self) -> None:
        from puppetmaster.adapters import hermes_reasoning_effort_env

        base_env = {"HERMES_HOME": "/nonexistent", "X": "1"}
        for bad in (None, "", "turbo", "none"):
            with hermes_reasoning_effort_env(base_env, bad) as run_env:
                # Unknown/empty effort must never fail the worker or relocate home.
                self.assertIs(run_env, base_env)

    def test_hermes_implement_reasoning_effort_reaches_subprocess(self) -> None:
        try:
            import yaml as _yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML (hermes extra) not installed")

        streamed = StreamedProcess(
            returncode=0, stdout="done", stderr="", timed_out=False, live_log_path=None
        )
        captured: dict = {}

        def fake_run(**kwargs):
            env = kwargs["env"]
            home = env.get("HERMES_HOME")
            captured["home"] = home
            if home and (Path(home) / "config.yaml").is_file():
                import yaml as _y

                cfg = _y.safe_load((Path(home) / "config.yaml").read_text())
                captured["effort"] = (cfg.get("agent") or {}).get("reasoning_effort")
            return streamed

        with TemporaryDirectory() as tmp:
            real_home = Path(tmp) / "hermes_home"
            real_home.mkdir()
            (real_home / "config.yaml").write_text("model:\n  default: gpt-5.5\n", encoding="utf-8")
            (real_home / "auth.json").write_text("{}", encoding="utf-8")
            task = Task(
                id="t-hermes-effort",
                job_id="job-hermes-effort",
                role="hermes-implement",
                adapter="hermes",
                instruction="Fix the bug.",
                payload={
                    "cwd": str(Path.cwd()),
                    "implement": True,
                    "disable_codegraph": True,
                    "reasoning_effort": "xhigh",
                },
            )
            clean = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
            with patch.dict(os.environ, {"HERMES_HOME": str(real_home)}), patch(
                "puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"
            ), patch("puppetmaster.adapters.worktree_guard", return_value=None), patch(
                "puppetmaster.adapters.git_snapshot", side_effect=[clean, clean]
            ), patch(
                "puppetmaster.adapters.run_streamed_subprocess", side_effect=fake_run
            ):
                HermesAdapter().run(task, "goal", "worker")

        self.assertEqual(captured.get("effort"), "xhigh")
        self.assertNotEqual(captured.get("home"), str(real_home))

    def test_classify_hermes_failure_known_signals(self) -> None:
        self.assertEqual(
            classify_hermes_failure("hermes: command not found"),
            "missing_cli",
        )
        self.assertEqual(
            classify_hermes_failure("No provider credentials configured; run hermes login"),
            "not_authenticated",
        )
        self.assertEqual(
            classify_hermes_failure("Provider verification failed for anthropic"),
            "not_authenticated",
        )
        self.assertEqual(
            classify_hermes_failure("maximum context length exceeded"),
            "context_length_exceeded",
        )
        self.assertEqual(
            classify_hermes_failure("401 Unauthorized from provider"),
            "not_authenticated",
        )
        self.assertEqual(
            classify_hermes_failure(
                "API call failed after 3 retries: HTTP 404: model: claude-sonnet-4-20250514"
            ),
            "model_unavailable",
        )
        self.assertEqual(classify_hermes_failure("completely unrelated text"), "unknown")

    def test_hermes_is_lockable_platform(self) -> None:
        from puppetmaster import platform_lock

        self.assertIn("hermes", platform_lock.KNOWN_ADAPTERS)

    def test_mcp_hermes_command_builds_implement_invocation(self) -> None:
        from puppetmaster.mcp_server import (
            _IMPLEMENT_ADAPTER_PRIORITY,
            _implement_command,
            hermes_command,
        )

        self.assertIn("hermes", _IMPLEMENT_ADAPTER_PRIORITY)
        command = hermes_command(
            {
                "goal": "ship it",
                "cwd": "/repo",
                "model": "gpt-5",
                "provider": "openai-api",
                "max_turns": 8,
                "use_hermes_rules": True,
            },
            implement=True,
        )
        self.assertEqual(command[0], "hermes")
        self.assertIn("ship it", command)
        self.assertEqual(command[command.index("--mode") + 1], "implement")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5")
        self.assertEqual(command[command.index("--provider") + 1], "openai-api")
        self.assertEqual(command[command.index("--max-turns") + 1], "8")
        self.assertIn("--use-hermes-rules", command)
        # The implement dispatcher routes the hermes adapter to this builder.
        self.assertEqual(
            _implement_command({"goal": "x", "cwd": "/r"}, "hermes")[0], "hermes"
        )

    def test_hermes_curated_catalog_stamps_provider(self) -> None:
        from puppetmaster.static_catalog import curated_to_specs

        specs = {s.adapter_model_name: s for s in curated_to_specs("hermes", "api", [])}
        self.assertIn("gemini-2.5-flash", specs)
        self.assertIn("claude-sonnet-4-5", specs)
        self.assertIn("gpt-5", specs)
        self.assertEqual(specs["gemini-2.5-flash"].payload_defaults["provider"], "gemini")
        self.assertEqual(specs["claude-sonnet-4-5"].payload_defaults["provider"], "anthropic")
        self.assertEqual(specs["gpt-5"].payload_defaults["provider"], "openai-api")
        for spec in specs.values():
            self.assertEqual(spec.adapter, "hermes")
            self.assertEqual(spec.billing, "api")

    def test_routing_stamps_hermes_provider_into_payload(self) -> None:
        from puppetmaster.orchestrator import merge_routing_payload
        from puppetmaster.router import TaskSignals, route_task
        from puppetmaster.static_catalog import curated_to_specs

        registry = curated_to_specs("hermes", "api", [])
        signals = TaskSignals(
            role="hermes-implement",
            instruction="Fix a typo in the docs",
            allowed_adapters={"hermes"},
        )
        decision = route_task(signals, registry, policy="balanced")
        self.assertEqual(decision.model.adapter, "hermes")
        payload = merge_routing_payload({"cwd": "/repo", "mode": "implement"}, decision)
        self.assertEqual(payload["model"], decision.model.adapter_model_name)
        self.assertEqual(payload["provider"], decision.model.payload_defaults["provider"])

    def test_diagnostics_list_provider_neutral_adapters(self) -> None:
        rows = adapter_status(Path.cwd())
        names = {row["name"] for row in rows}

        self.assertIn("cursor", names)
        self.assertIn("claude-code", names)
        self.assertIn("codex", names)

    def test_codex_configured_does_not_regress_for_openai_key_only(self) -> None:
        """Availability must not be gated on billing context: an
        OPENAI_API_KEY-only setup (no healthy Codex auth) stays configured,
        while a setup with neither key nor healthy auth is unconfigured.
        """
        from puppetmaster import diagnostics

        def codex_row() -> dict:
            return next(r for r in adapter_status(Path.cwd()) if r["name"] == "codex")

        with patch.object(diagnostics, "_codex_cli_installed", return_value=True), patch(
            "puppetmaster.platform_billing.detect_codex_billing"
        ) as billing:
            billing.return_value = MagicMock(healthy=False)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-present"}, clear=False):
                self.assertTrue(codex_row()["configured"])
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                self.assertFalse(codex_row()["configured"])

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

    def test_cursor_sdk_detected_inside_package_dir_node_modules(self) -> None:
        """Field report (Zane): a valid install at
        site-packages/puppetmaster/node_modules/@cursor/sdk — Node's FIRST
        resolution hop from cursor_sdk_runner.mjs — was invisible to the probe,
        which only checked one fixed level up. Runtime worked; diagnostics said
        'SDK not found' on a working machine. The probe must mirror Node's full
        upward node_modules walk."""
        from puppetmaster import diagnostics

        with TemporaryDirectory() as tmp:
            unrelated_repo = Path(tmp) / "some-workspace"
            unrelated_repo.mkdir()
            site_packages = Path(tmp) / "site-packages"
            package_dir = site_packages / "puppetmaster"
            sdk = package_dir / "node_modules" / "@cursor" / "sdk"
            sdk.mkdir(parents=True)
            fake_diagnostics_file = package_dir / "diagnostics.py"
            fake_diagnostics_file.write_text("# stub", encoding="utf-8")
            with patch.object(diagnostics, "__file__", str(fake_diagnostics_file)):
                self.assertTrue(diagnostics._cursor_sdk_installed(unrelated_repo))
                location = diagnostics._find_cursor_sdk_install(unrelated_repo)
            self.assertIsNotNone(location)
            self.assertEqual(location.resolve(), sdk.resolve())

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

    def test_stray_prints_never_reach_protocol_stdout(self) -> None:
        """fd 1 is reserved for JSON-RPC frames. A stray print() during a
        tool call must be diverted to stderr instead of corrupting the
        protocol stream — Codex's rmcp client treats the first non-frame
        byte as a fatal transport error and never reconnects, so one loose
        line means 'Transport closed' for the whole session."""
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        bootstrap = (
            "import sys\n"
            "from puppetmaster import mcp_server\n"
            "_orig = mcp_server.handle_message\n"
            "def noisy(message):\n"
            "    print('STRAY-DIAGNOSTIC-LINE')\n"
            "    return _orig(message)\n"
            "mcp_server.handle_message = noisy\n"
            "sys.exit(mcp_server.main())\n"
        )
        frames = "".join(
            json.dumps(message) + "\n"
            for message in (
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
        )
        with TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-c", bootstrap],
                input=frames.encode("utf-8"),
                capture_output=True,
                cwd=str(repo_root),
                env={
                    **os.environ,
                    "PUPPETMASTER_MCP_REGISTRY_DIR": tmp,
                    "PUPPETMASTER_MCP_INPUT_STALE_DISABLED": "1",
                    "PUPPETMASTER_MCP_IDLE_KEEPALIVE_DISABLED": "1",
                },
                timeout=60,
            )

        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        stdout_lines = [line for line in stdout_text.splitlines() if line.strip()]
        self.assertTrue(stdout_lines, "server produced no protocol frames")
        for line in stdout_lines:
            json.loads(line)  # every stdout line must be a parseable frame
        response_ids = {
            frame.get("id")
            for frame in map(json.loads, stdout_lines)
            if "id" in frame
        }
        self.assertLessEqual({1, 2}, response_ids, "handshake responses missing")
        self.assertNotIn("STRAY-DIAGNOSTIC-LINE", stdout_text)
        self.assertIn(
            "STRAY-DIAGNOSTIC-LINE",
            proc.stderr.decode("utf-8", errors="replace"),
            "stray output should be diverted to stderr, not dropped",
        )

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

    def test_cli_logs_collapses_heartbeats_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = str(Path(tmp) / ".puppetmaster")
            store = SwarmStore(Path(state_dir))
            job = store.create_job("heartbeat collapse check")
            for _ in range(5):
                store.emit(job.id, "run.heartbeat", {"run_id": "r1"})
            store.emit(job.id, "task.lease_renewed", {"task_id": "t1"})
            store.emit(job.id, "verification.recorded", {"confidence": 0.9})

            def _run_logs(extra: list[str]) -> tuple[str, str]:
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = cli_main(
                        ["--state-dir", state_dir, "--backend", "file", "logs", job.id] + extra
                    )
                self.assertEqual(code, 0)
                return out.getvalue(), err.getvalue()

            default_out, default_err = _run_logs([])
            self.assertIn("verification.recorded", default_out)
            self.assertNotIn("run.heartbeat", default_out)
            self.assertIn("collapsed 6 heartbeat event(s)", default_err)
            self.assertIn("run.heartbeat=5", default_err)

            all_out, _ = _run_logs(["--all"])
            self.assertIn("run.heartbeat", all_out)
            self.assertIn("task.lease_renewed", all_out)

            filtered_out, _ = _run_logs(["--event-type", "heartbeat"])
            self.assertIn("run.heartbeat", filtered_out)
            self.assertNotIn("verification.recorded", filtered_out)

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

    def test_classifier_ceiling_tracks_fable_5_frontier(self) -> None:
        from puppetmaster.router import TaskSignals, classify_capability_needed

        signal = TaskSignals(
            instruction=(
                "security audit across every module with detailed vision on "
                "every screenshot and cross-repo architecture review"
            ),
            role="security-review",
            payload_size_chars=25_000,
            explicit_min_capability=100,
        )
        self.assertEqual(classify_capability_needed(signal), 100)

    def test_starter_registry_includes_fable_5_entries(self) -> None:
        from puppetmaster.model_registry import starter_registry

        registry = {spec.id: spec for spec in starter_registry()}
        self.assertIn("cursor/fable-5", registry)
        self.assertIn("claude-code/fable-5", registry)

        cursor_fable = registry["cursor/fable-5"]
        self.assertEqual(cursor_fable.adapter, "cursor")
        self.assertEqual(cursor_fable.adapter_model_name, "fable-5")
        self.assertEqual(cursor_fable.capability_score, 100)
        self.assertEqual(cursor_fable.billing, "plan")
        self.assertEqual(cursor_fable.input_per_mtok_usd, 0.0)
        self.assertIn("mythos-class", cursor_fable.tags)

        claude_fable = registry["claude-code/fable-5"]
        self.assertEqual(claude_fable.adapter, "claude-code")
        self.assertEqual(claude_fable.adapter_model_name, "claude-fable-5")
        self.assertEqual(claude_fable.capability_score, 100)
        self.assertEqual(claude_fable.input_per_mtok_usd, 10.0)
        self.assertEqual(claude_fable.output_per_mtok_usd, 50.0)
        self.assertEqual(claude_fable.context_window, 1_000_000)
        self.assertEqual(claude_fable.billing, "unknown")
        self.assertIn("2026-06-22", claude_fable.notes)

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

    def test_payload_defaults_round_trip_and_drop_empty_defaults(self) -> None:
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        specs = [
            ModelSpec(
                id="openai/high",
                adapter="openai",
                adapter_model_name="gpt-5.5",
                payload_defaults={"reasoning_effort": "high"},
            ),
            ModelSpec(
                id="openai/plain",
                adapter="openai",
                adapter_model_name="gpt-5.4",
            ),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(specs, path)
            raw = path.read_text(encoding="utf-8")
            self.assertIn('"payload_defaults"', raw)
            self.assertEqual(raw.count('"payload_defaults"'), 1)
            loaded = {spec.id: spec for spec in load_registry(path)}
            self.assertEqual(
                loaded["openai/high"].payload_defaults,
                {"reasoning_effort": "high"},
            )
            self.assertEqual(loaded["openai/plain"].payload_defaults, {})

    def test_routing_payload_merge_applies_defaults_below_task_payload(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.orchestrator import merge_routing_payload
        from puppetmaster.router import RoutingDecision

        spec = ModelSpec(
            id="openai/high",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            payload_defaults={"reasoning_effort": "high", "temperature": 0},
        )
        decision = RoutingDecision(
            model=spec,
            policy="balanced",
            capability_needed=75,
            estimated_tokens_in=1000,
            estimated_tokens_out=1000,
            estimated_cost_usd=0.01,
            reason="test",
        )
        merged = merge_routing_payload(
            {"reasoning_effort": "low", "prompt": "keep me"},
            decision,
            {"attempt": 1},
        )
        self.assertEqual(merged["reasoning_effort"], "low")
        self.assertEqual(merged["temperature"], 0)
        self.assertEqual(merged["model"], "gpt-5.5")
        self.assertEqual(merged["router_model_id"], "openai/high")
        self.assertEqual(merged["attempt"], 1)

    def test_output_token_multiplier_scales_cost_and_rejects_non_positive(self) -> None:
        from puppetmaster.model_registry import ModelSpec

        base = ModelSpec(
            id="base",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=10.0,
        )
        high = ModelSpec(
            id="high",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=10.0,
            output_token_multiplier=3.0,
        )
        self.assertAlmostEqual(high.estimate_cost_usd(1_000, 2_000), 0.061)
        self.assertGreater(high.estimate_cost_usd(1_000, 2_000), base.estimate_cost_usd(1_000, 2_000))
        with self.assertRaises(ValueError):
            ModelSpec(
                id="bad",
                adapter="openai",
                adapter_model_name="gpt-5.5",
                output_token_multiplier=0,
            )

    def test_models_setup_wizard_adds_openai_effort_variant(self) -> None:
        from puppetmaster.cli import ModelRegistryWizard
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        base = ModelSpec(
            id="openai/gpt-5-5",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            capability_score=96,
            tags=["openai", "reasoning"],
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([base], path)
            stdin = io.StringIO("1\n1\nhigh\n\n\n\n\nq\n\n")
            stdout = io.StringIO()
            code = ModelRegistryWizard(path, stdin, stdout).run()
            self.assertEqual(code, 0)
            loaded = {spec.id: spec for spec in load_registry(path)}
            self.assertIn("openai/gpt-5-5-high", loaded)
            variant = loaded["openai/gpt-5-5-high"]
            self.assertEqual(variant.payload_defaults, {"reasoning_effort": "high"})
            self.assertIn("effort:high", variant.tags)

    def test_models_setup_wizard_refuses_cursor_effort_variant(self) -> None:
        from puppetmaster.cli import ModelRegistryWizard
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        base = ModelSpec(
            id="cursor/gpt-5-5",
            adapter="cursor",
            adapter_model_name="gpt-5.5",
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([base], path)
            stdin = io.StringIO("1\n1\nq\n")
            stdout = io.StringIO()
            code = ModelRegistryWizard(path, stdin, stdout).run()
            self.assertEqual(code, 0)
            self.assertIn("does not expose an effort knob", stdout.getvalue())
            loaded = load_registry(path)
            self.assertEqual(len(loaded), 1)

    def test_models_set_applies_codex_effort_payload_defaults(self) -> None:
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [
                    ModelSpec(
                        id="codex/gpt-5-5",
                        adapter="codex",
                        adapter_model_name="gpt-5.5",
                    )
                ],
                path,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "models",
                        "set",
                        "--registry-path",
                        str(path),
                        "codex/gpt-5-5",
                        "effort=high",
                    ]
                )
            self.assertEqual(code, 0)
            printed = json.loads(stdout.getvalue())
            self.assertEqual(
                printed["payload_defaults"],
                {"extra_args": ["-c", "model_reasoning_effort=high"]},
            )
            loaded = {spec.id: spec for spec in load_registry(path)}
            self.assertEqual(
                loaded["codex/gpt-5-5"].payload_defaults,
                {"extra_args": ["-c", "model_reasoning_effort=high"]},
            )

    def test_models_set_applies_hermes_effort_payload_defaults(self) -> None:
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [
                    ModelSpec(
                        id="hermes/gpt-5-5",
                        adapter="hermes",
                        adapter_model_name="gpt-5.5",
                        payload_defaults={"provider": "openai-api"},
                    )
                ],
                path,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "models",
                        "set",
                        "--registry-path",
                        str(path),
                        "hermes/gpt-5-5",
                        "effort=high",
                    ]
                )
            self.assertEqual(code, 0)
            loaded = {spec.id: spec for spec in load_registry(path)}
            # Effort is stamped without clobbering the existing provider default.
            self.assertEqual(
                loaded["hermes/gpt-5-5"].payload_defaults,
                {"provider": "openai-api", "reasoning_effort": "high"},
            )
            self.assertIn("effort:high", loaded["hermes/gpt-5-5"].tags)

    def test_model_payload_defaults_for_effort_hermes_rejects_unknown(self) -> None:
        from puppetmaster.cli import model_payload_defaults_for_effort

        self.assertEqual(
            model_payload_defaults_for_effort("hermes", "xhigh"),
            {"reasoning_effort": "xhigh"},
        )
        with self.assertRaises(ValueError):
            model_payload_defaults_for_effort("hermes", "none")

    def test_models_setup_wizard_adds_hermes_effort_variant(self) -> None:
        from puppetmaster.cli import ModelRegistryWizard
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        base = ModelSpec(
            id="hermes/gpt-5-5",
            adapter="hermes",
            adapter_model_name="gpt-5.5",
            tags=["hermes", "openai"],
            payload_defaults={"provider": "openai-api"},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([base], path)
            # menu 1 (effort variant) -> base #1 -> effort high -> accept suggested
            # id -> capability 97 -> default multiplier -> confirm add -> q -> save.
            stdin = io.StringIO("1\n1\nhigh\n\n97\n\ny\nq\ny\n")
            stdout = io.StringIO()
            code = ModelRegistryWizard(path, stdin, stdout).run()
            self.assertEqual(code, 0)
            variants = {s.id: s for s in load_registry(path)}
            variant = variants.get("hermes/gpt-5-5-high")
            self.assertIsNotNone(variant, stdout.getvalue())
            self.assertEqual(
                variant.payload_defaults,
                {"provider": "openai-api", "reasoning_effort": "high"},
            )
            self.assertIn("effort:high", variant.tags)

    def test_models_setup_wizard_exits_on_closed_stdin(self) -> None:
        from puppetmaster.cli import ModelRegistryWizard
        from puppetmaster.model_registry import ModelSpec, save_registry

        base = ModelSpec(id="openai/gpt-5-5", adapter="openai", adapter_model_name="gpt-5.5")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([base], path)
            stdin = io.StringIO("")  # immediate EOF: must exit, never loop
            stdout = io.StringIO()
            code = ModelRegistryWizard(path, stdin, stdout).run()
            self.assertEqual(code, 0)
            self.assertIn("Input closed", stdout.getvalue())

    def test_models_setup_wizard_refuses_duplicate_variant_id(self) -> None:
        from puppetmaster.cli import ModelRegistryWizard
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        base = ModelSpec(id="openai/gpt-5-5", adapter="openai", adapter_model_name="gpt-5.5")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry([base], path)
            # Accept the suggested id but override it to collide with the base entry.
            stdin = io.StringIO("1\n1\nhigh\nopenai/gpt-5-5\nq\n\n")
            stdout = io.StringIO()
            code = ModelRegistryWizard(path, stdin, stdout).run()
            self.assertEqual(code, 0)
            self.assertIn("already exists", stdout.getvalue())
            self.assertEqual(len(load_registry(path)), 1)

    def test_models_set_effort_merges_defaults_and_swaps_tag(self) -> None:
        from puppetmaster.model_registry import ModelSpec, load_registry, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [
                    ModelSpec(
                        id="openai/gpt-5-5",
                        adapter="openai",
                        adapter_model_name="gpt-5.5",
                        tags=["openai", "effort:low"],
                        payload_defaults={"temperature": 0.2, "reasoning_effort": "low"},
                    )
                ],
                path,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "models",
                        "set",
                        "--registry-path",
                        str(path),
                        "openai/gpt-5-5",
                        "effort=high",
                    ]
                )
            self.assertEqual(code, 0)
            loaded = {spec.id: spec for spec in load_registry(path)}
            updated = loaded["openai/gpt-5-5"]
            self.assertEqual(
                updated.payload_defaults,
                {"temperature": 0.2, "reasoning_effort": "high"},
            )
            self.assertIn("effort:high", updated.tags)
            self.assertNotIn("effort:low", updated.tags)

    def test_model_spec_rejects_non_dict_payload_defaults(self) -> None:
        from puppetmaster.model_registry import ModelSpec

        with self.assertRaises(ValueError):
            ModelSpec(
                id="bad",
                adapter="openai",
                adapter_model_name="gpt-5.5",
                payload_defaults=["not", "a", "dict"],
            )

    def test_models_set_unknown_id_exits_nonzero(self) -> None:
        from puppetmaster.model_registry import ModelSpec, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            save_registry(
                [ModelSpec(id="known", adapter="openai", adapter_model_name="gpt-5.5")],
                path,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "models",
                        "set",
                        "--registry-path",
                        str(path),
                        "missing",
                        "effort=high",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("unknown model id", stderr.getvalue())

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

    def test_initial_route_drops_adapter_with_missing_cli(self) -> None:
        """The router must not first-pick a model whose CLI isn't installed —
        even when billing reads healthy — and must record why it skipped it."""
        from unittest.mock import patch

        from puppetmaster.model_registry import ModelSpec, save_registry
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store_factory import create_store
        from puppetmaster.workers import WorkerSpec

        registry = [
            ModelSpec(id="claude-code/opus", adapter="claude-code", adapter_model_name="claude-opus-4-8", capability_score=99, billing="plan"),
            ModelSpec(id="cursor/gpt", adapter="cursor", adapter_model_name="gpt-5.5", capability_score=95, billing="plan", tags=["cursor"]),
        ]

        def _healthy(adapter, **kw):
            return BillingStatus(adapter=adapter, billing="plan", healthy=True, detail="ok", evidence=[])

        with TemporaryDirectory() as tmp:
            rp = Path(tmp) / "models.json"
            save_registry(registry, rp)
            store = create_store("file", Path(tmp) / ".puppetmaster")
            store.init()
            orch = Orchestrator(store)
            job = store.create_job("router cli check")
            spec = WorkerSpec(
                role="implement",
                instruction="implement the thing",
                adapter="local",
                payload={"auto_route": True, "registry_path": str(rp), "routing_policy": "balanced"},
                depends_on_roles=[],
            )
            with patch("puppetmaster.platform_billing.detect_adapter_billing_cached", side_effect=_healthy), \
                 patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True), \
                 patch("puppetmaster.preflight.adapter_cli_present", side_effect=lambda a, **kw: a != "claude-code"):
                tasks = orch._create_tasks(job, [spec])
            self.assertEqual(tasks[0].adapter, "cursor")
            events = store.read_events(job.id)
            missing = [e for e in events if e.get("event") == "router.adapter_cli_missing"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["payload"]["adapters"], ["claude-code"])

    def test_initial_route_never_fails_closed_when_no_cli_installed(self) -> None:
        """If dropping CLI-missing adapters would empty the registry (e.g. a host
        with no CLIs), keep it intact so dispatch/fallback surface the precise
        error instead of the router silently routing nothing."""
        from unittest.mock import patch

        from puppetmaster.model_registry import save_registry
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store_factory import create_store
        from puppetmaster.workers import WorkerSpec

        def _healthy(adapter, **kw):
            return BillingStatus(adapter=adapter, billing="plan", healthy=True, detail="ok", evidence=[])

        with TemporaryDirectory() as tmp:
            rp = Path(tmp) / "models.json"
            save_registry(self._three_tier_registry(), rp)  # all claude-code
            store = create_store("file", Path(tmp) / ".puppetmaster")
            store.init()
            orch = Orchestrator(store)
            job = store.create_job("router never-closed check")
            spec = WorkerSpec(
                role="implement",
                instruction="implement the thing",
                adapter="local",
                payload={"auto_route": True, "registry_path": str(rp), "routing_policy": "balanced"},
                depends_on_roles=[],
            )
            with patch("puppetmaster.platform_billing.detect_adapter_billing_cached", side_effect=_healthy), \
                 patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True), \
                 patch("puppetmaster.preflight.adapter_cli_present", return_value=False):
                tasks = orch._create_tasks(job, [spec])
            self.assertEqual(tasks[0].adapter, "claude-code")
            events = store.read_events(job.id)
            self.assertEqual(
                [e for e in events if e.get("event") == "router.adapter_cli_missing"],
                [],
            )

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
        # And under quality policy, the plan-billed cursor/fable-5 (cap 100)
        # wins over claude-code/fable-5, opus-4-8 (99), and gpt-5.5 (96).
        quality_decision = route_task(
            signal, starter_registry(), policy="quality"
        )
        self.assertEqual(quality_decision.model.id, "cursor/fable-5")

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
        # Fable 5 is the frontier flagship — strictly above Opus 4.8 and the
        # single highest-capability model in the starter registry.
        self.assertIn("claude-code/opus-4-8", ids)
        self.assertIn("cursor/fable-5", ids)
        self.assertIn("claude-code/fable-5", ids)
        self.assertLess(
            by_id["claude-code/opus-4-7"].capability_score,
            by_id["claude-code/opus-4-8"].capability_score,
        )
        self.assertLess(
            by_id["claude-code/opus-4-8"].capability_score,
            by_id["claude-code/fable-5"].capability_score,
        )
        self.assertEqual(
            by_id["claude-code/fable-5"].capability_score,
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

    def test_starter_registry_routes_hardest_task_to_fable_5(self) -> None:
        """The absolute-hardest tasks must route to the frontier flagship
        (Fable 5), not saturate one notch below it on Opus 4.8."""
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
        self.assertEqual(decision.model.id, "cursor/fable-5")
        # Opus 4.8 should be in the rejected set (sufficient-but-not-chosen),
        # proving the flagship was preferred for the hardest tier.
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("claude-code/opus-4-8", rejected_ids)

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
            from puppetmaster.models import ArtifactType

            routing = [
                a
                for a in store.list_artifacts(job.id)
                if a.type == ArtifactType.ROUTING
            ]
            self.assertEqual(len(routing), 1)
            self.assertGreater(routing[0].payload["nominal_cost_usd"], 0.0)
            self.assertEqual(
                data["total_estimated_cost_usd"],
                routing[0].payload["estimated_cost_usd"],
            )
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

    def test_mcp_swarm_config_writer_marks_generated_workers_read_only(self) -> None:
        """Generated MCP swarms are analysis runs. If routing later selects an
        edit-capable adapter, the payload must keep it on the adapter's
        read-only path so dirty diffs can be reviewed without tripping the
        full-edit clean-tree guard.
        """
        from puppetmaster.mcp_server import write_generated_swarm_config
        from puppetmaster.workers import WorkerSpec, swarm_mode

        with TemporaryDirectory() as tmp:
            args = {"goal": "review dirty diff", "cwd": tmp, "state_dir": str(Path(tmp) / "state")}
            config_path = write_generated_swarm_config(args, ["audit"], "cursor")
            cfg = json.loads(Path(config_path).read_text())

            payload = cfg["workers"][0]["payload"]
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["sandbox"], "read-only")
            self.assertFalse(payload["dangerously_bypass_approvals_and_sandbox"])
            self.assertEqual(
                swarm_mode(
                    [
                        WorkerSpec(
                            role="audit",
                            instruction="review",
                            adapter="codex",
                            payload=payload,
                        )
                    ]
                ),
                "analysis",
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

    def test_mcp_swarm_config_defaults_to_fresh_memory(self) -> None:
        from puppetmaster.mcp_server import write_generated_swarm_config
        from puppetmaster.models import MemoryRecord
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.config import load_config

        goal = "swarm fresh memory default"
        memory = MemoryRecord(
            scope="swarm.findings",
            statement="prior swarm conclusion",
            evidence=["e"],
            source_artifacts=["artifact_z"],
            confidence=0.9,
        )
        with TemporaryDirectory() as tmp:
            args = {"goal": goal, "cwd": tmp, "state_dir": str(Path(tmp) / "state")}
            config_path = write_generated_swarm_config(args, ["explore", "audit"], "cursor")
            cfg = json.loads(Path(config_path).read_text())
            for worker in cfg["workers"]:
                self.assertTrue(worker["payload"].get("disable_memory"))

            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.promote_memory(memory)
            specs = load_config(config_path).workers
            routed = Orchestrator(store)._with_retrieved_memory(specs, goal)
            for spec in routed:
                self.assertNotIn(
                    "retrieved_memory",
                    spec.payload,
                    f"swarm role={spec.role} should be fresh by default",
                )

    def test_mcp_swarm_config_disable_memory_false_restores_injection(self) -> None:
        from puppetmaster.mcp_server import write_generated_swarm_config
        from puppetmaster.models import MemoryRecord
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.config import load_config

        goal = "swarm memory opt-in"
        memory = MemoryRecord(
            scope="swarm.findings",
            statement="shared swarm context",
            evidence=["e"],
            source_artifacts=["artifact_w"],
            confidence=0.85,
        )
        with TemporaryDirectory() as tmp:
            args = {
                "goal": goal,
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "disable_memory": False,
            }
            config_path = write_generated_swarm_config(args, ["explore"], "cursor")
            cfg = json.loads(Path(config_path).read_text())
            self.assertFalse(cfg["workers"][0]["payload"]["disable_memory"])

            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.promote_memory(memory)
            specs = load_config(config_path).workers
            routed = Orchestrator(store)._with_retrieved_memory(specs, goal)
            self.assertIn("retrieved_memory", routed[0].payload)

    def test_start_swarm_passes_disable_memory_flag_by_default(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            mcp_server.start_swarm(
                {
                    "goal": "fresh swarm",
                    "cwd": ".",
                    "roles": ["explore"],
                    "allow_local_demo": True,
                }
            )

        self.assertIn("--disable-memory", captured["command"])

    def test_start_swarm_disable_memory_false_passes_enable_memory(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            mcp_server.start_swarm(
                {
                    "goal": "memory swarm",
                    "cwd": ".",
                    "roles": ["explore"],
                    "allow_local_demo": True,
                    "disable_memory": False,
                }
            )

        self.assertIn("--enable-memory", captured["command"])
        self.assertNotIn("--disable-memory", captured["command"])

    def test_codex_schema_requires_goal(self) -> None:
        from puppetmaster.mcp_server import codex_schema

        schema = codex_schema()
        self.assertIn("goal", schema["required"])
        self.assertIn("sandbox", schema["properties"])
        self.assertIn("executable", schema["properties"])

    def test_run_codex_builds_codex_cli_command(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_run_worker_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch.object(mcp_server, "run_worker_cli", side_effect=fake_run_worker_cli):
            mcp_server.run_codex(
                {
                    "goal": "ship codex worker",
                    "cwd": "/tmp/repo",
                    "model": "gpt-5.4-mini",
                    "sandbox": "read-only",
                    "timeout_seconds": 120,
                    "allow_dirty": True,
                    "executable": "/opt/codex",
                    "disable_memory": True,
                }
            )

        command = captured["command"]
        self.assertEqual(command[0], "codex")
        self.assertIn("ship codex worker", command)
        self.assertIn("--cwd", command)
        self.assertIn("/tmp/repo", command)
        self.assertIn("--model", command)
        self.assertIn("gpt-5.4-mini", command)
        self.assertIn("--sandbox", command)
        self.assertIn("read-only", command)
        self.assertIn("--timeout-seconds", command)
        self.assertIn("120", command)
        self.assertIn("--allow-dirty", command)
        self.assertIn("--executable", command)
        self.assertIn("/opt/codex", command)
        self.assertIn("--disable-memory", command)

    def test_single_adapter_mcp_commands_forward_disable_memory(self) -> None:
        from puppetmaster import mcp_server

        claude = mcp_server.claude_command({"goal": "fresh claude", "cwd": ".", "disable_memory": True})
        openai = mcp_server.openai_command({"goal": "fresh openai", "cwd": ".", "disable_memory": True})
        codex = mcp_server.codex_command({"goal": "fresh codex", "cwd": ".", "disable_memory": True})

        self.assertIn("--disable-memory", claude)
        self.assertIn("--disable-memory", openai)
        self.assertIn("--disable-memory", codex)

    def test_single_adapter_mcp_commands_do_not_disable_memory_by_default(self) -> None:
        from puppetmaster import mcp_server

        claude = mcp_server.claude_command({"goal": "default claude", "cwd": "."})
        openai = mcp_server.openai_command({"goal": "default openai", "cwd": "."})
        codex = mcp_server.codex_command({"goal": "default codex", "cwd": "."})

        self.assertNotIn("--disable-memory", claude)
        self.assertNotIn("--disable-memory", openai)
        self.assertNotIn("--disable-memory", codex)

    def test_single_adapter_mcp_run_functions_forward_disable_memory(self) -> None:
        from puppetmaster import mcp_server

        captured = []

        def fake_run_worker_cli(command, args):
            captured.append(command)
            return {"ok": True}

        with patch.object(mcp_server, "run_worker_cli", side_effect=fake_run_worker_cli):
            mcp_server.run_claude({"goal": "fresh claude", "cwd": ".", "disable_memory": True})
            mcp_server.run_openai({"goal": "fresh openai", "cwd": ".", "disable_memory": True})
            mcp_server.run_codex({"goal": "fresh codex", "cwd": ".", "disable_memory": True})

        self.assertEqual([command[0] for command in captured], ["claude", "openai", "codex"])
        for command in captured:
            self.assertIn("--disable-memory", command)

    def test_start_codex_builds_codex_cli_command(self) -> None:
        from puppetmaster import mcp_server

        captured = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch.object(mcp_server, "start_cli", side_effect=fake_start_cli):
            mcp_server.start_codex({"goal": "async codex", "cwd": "."})

        self.assertEqual(captured["command"][0], "codex")
        self.assertIn("async codex", captured["command"])


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
                    # Custom (untrusted) host requires the explicit opt-in so the
                    # adapter is willing to send the API key there.
                    openai_allow_untrusted_base_url=True,
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

    def test_openai_adapter_refuses_untrusted_base_url(self) -> None:
        """The adapter must not send OPENAI_API_KEY to an arbitrary host.

        Without the explicit opt-in, a caller-supplied non-allowlisted
        base_url is refused before any network call (no key exfiltration).
        """
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=False), patch(
            "puppetmaster.adapters.urllib.request.urlopen"
        ) as urlopen:
            artifacts = OpenAIAdapter().run(
                self._task(openai_base_url="https://evil.example.com/v1"),
                "goal",
                "worker-openai",
            )

        urlopen.assert_not_called()
        self.assertEqual(artifacts[0].payload["failure"], "untrusted_base_url")
        self.assertEqual(artifacts[0].payload["result"], "failed")

    def test_openai_adapter_allows_allowlisted_host_via_env(self) -> None:
        """PUPPETMASTER_OPENAI_ALLOWED_HOSTS extends the trusted host set."""
        fake_response = _FakeUrlopenResponse(
            json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}})
        )
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk",
                "PUPPETMASTER_OPENAI_ALLOWED_HOSTS": "proxy.internal",
            },
            clear=False,
        ), patch(
            "puppetmaster.adapters.urllib.request.urlopen", return_value=fake_response
        ) as urlopen:
            OpenAIAdapter().run(
                self._task(openai_base_url="https://proxy.internal/v1"),
                "goal",
                "worker-openai",
            )
        urlopen.assert_called_once()

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
            classify_openai_failure('{"error":{"code":"model_not_found"}}', None),
            "model_unavailable",
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
            stub = Path(tmp) / "codex-stub"
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
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n'
                "startup_timeout_sec = 30\n"
                "tool_timeout_sec = 300\n",
                encoding="utf-8",
            )
            stub = Path(tmp) / "codex-stub"
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
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False):
                result = install_codex_mcp(
                    codex_executable=str(stub),
                    skip_handshake=True,
                )
            self.assertEqual(result.status, "unchanged")

    def test_resolve_mcp_env_precedence_map_and_redaction(self):
        from puppetmaster.installers import McpEnvRequest, resolve_mcp_env
        from puppetmaster.redaction import clear_registered_secrets, redact_secrets

        with TemporaryDirectory() as tmp:
            clear_registered_secrets()
            env_file = Path(tmp) / "env.zsh"
            secret = "sk-" + "x" * 24
            env_file.write_text(
                f"export FOO=file\nSHARED=file\nOPENAI_API_KEY={secret}\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            try:
                resolved = resolve_mcp_env(
                    McpEnvRequest(
                        direct=("FOO=direct",),
                        inherit=("SHARED",),
                        env_files=(env_file,),
                        map_env=("CODEX_HOME=MY_CODEX_API_HOME",),
                    ),
                    existing_env={"FOO": "existing"},
                    source_env={
                        "SHARED": "inherited",
                        "MY_CODEX_API_HOME": "/tmp/api-codex",
                    },
                )
                self.assertTrue(resolved.ok, msg=resolved.errors)
                self.assertEqual(resolved.env["FOO"], "existing")
                self.assertEqual(resolved.env["SHARED"], "inherited")
                self.assertEqual(resolved.env["CODEX_HOME"], "/tmp/api-codex")
                self.assertNotIn(secret, redact_secrets(f"leaked {secret}") or "")
            finally:
                clear_registered_secrets()

    def test_env_file_permission_warning_and_missing_file_error(self):
        from puppetmaster.installers import McpEnvRequest, resolve_mcp_env

        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "env.zsh"
            env_file.write_text("export CODEX_HOME=/tmp/codex-api\n", encoding="utf-8")
            env_file.chmod(0o644)
            resolved = resolve_mcp_env(McpEnvRequest(env_files=(env_file,)))
            self.assertTrue(resolved.ok, msg=resolved.errors)
            self.assertEqual(resolved.env["CODEX_HOME"], "/tmp/codex-api")
            self.assertTrue(any("group/world readable" in m for m in resolved.messages))

            missing = resolve_mcp_env(McpEnvRequest(env_files=(Path(tmp) / "missing.env",)))
            self.assertFalse(missing.ok)
            self.assertTrue(any("env file not found" in e for e in missing.errors))

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_idempotent_entry_still_sets_missing_timeouts(self):
        from puppetmaster.installers import (
            CODEX_STARTUP_TIMEOUT_SEC,
            CODEX_TOOL_TIMEOUT_SEC,
            install_codex_mcp,
        )

        with TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            config = codex_home / "config.toml"
            config.write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n',
                encoding="utf-8",
            )
            stub = Path(tmp) / "codex-stub"
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
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False):
                result = install_codex_mcp(codex_executable=str(stub), skip_handshake=True)

            self.assertEqual(result.status, "installed", msg=result.messages)
            text = config.read_text("utf-8")
            self.assertIn(f"startup_timeout_sec = {CODEX_STARTUP_TIMEOUT_SEC}", text)
            self.assertIn(f"tool_timeout_sec = {CODEX_TOOL_TIMEOUT_SEC}", text)

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_preserves_existing_env_unless_force_env(self):
        from puppetmaster.installers import install_codex_mcp

        with TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n\n'
                "[mcp_servers.puppetmaster.env]\n"
                'FOO = "old"\n',
                encoding="utf-8",
            )
            stub_log = Path(tmp) / "codex_calls.log"
            stub = Path(tmp) / "codex-stub"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                'if [ "$1" = "mcp" ] && [ "$2" = "add" ] && [ "$3" = "--help" ]; then\n'
                '  echo "--env <KEY=VALUE>"\n'
                "  exit 0\n"
                "fi\n"
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
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False):
                result = install_codex_mcp(
                    codex_executable=str(stub),
                    env=("FOO=new", "BAR=1"),
                    skip_handshake=True,
                )
                self.assertEqual(result.status, "installed", msg=result.messages)
                log = stub_log.read_text("utf-8")
                self.assertIn("--env FOO=old", log)
                self.assertIn("--env BAR=1", log)
                self.assertNotIn("--env FOO=new", log)

                stub_log.write_text("", encoding="utf-8")
                forced = install_codex_mcp(
                    codex_executable=str(stub),
                    env=("FOO=new",),
                    force_env=True,
                    skip_handshake=True,
                )
                self.assertEqual(forced.status, "installed", msg=forced.messages)
                forced_log = stub_log.read_text("utf-8")
                self.assertIn("--env FOO=new", forced_log)

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_env_file_uses_wrapper_without_raw_secret_in_outputs(self):
        from puppetmaster.installers import install_codex_mcp

        with TemporaryDirectory() as tmp:
            secret = "sk-" + "z" * 24
            env_file = Path(tmp) / "env.zsh"
            env_file.write_text(
                f"export OPENAI_API_KEY={secret}\nexport CODEX_HOME={Path(tmp) / 'api-home'}\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n\n'
                "[mcp_servers.puppetmaster.env]\n"
                'LEGACY = "keep-me"\n',
                encoding="utf-8",
            )
            stub_log = Path(tmp) / "codex_calls.log"
            stub = Path(tmp) / "codex-stub"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                'if [ "$1" = "mcp" ] && [ "$2" = "add" ] && [ "$3" = "--help" ]; then\n'
                '  echo "--env <KEY=VALUE>"\n'
                "  exit 0\n"
                "fi\n"
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
            wrapper = Path(tmp) / "codex-mcp-wrapper.py"
            managed_env = Path(tmp) / "codex-mcp.env.json"
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False), \
                    patch("puppetmaster.installers._CODEX_WRAPPER_PATH", wrapper), \
                    patch("puppetmaster.installers._CODEX_MANAGED_ENV_PATH", managed_env):
                result = install_codex_mcp(
                    codex_executable=str(stub),
                    env_files=(env_file,),
                    skip_handshake=True,
                )
            self.assertEqual(result.status, "installed", msg=result.messages)
            self.assertTrue(wrapper.exists())
            self.assertTrue(managed_env.exists())
            self.assertEqual(managed_env.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(secret, wrapper.read_text("utf-8"))
            log = stub_log.read_text("utf-8")
            self.assertNotIn("--env", log)
            self.assertNotIn(secret, log)
            self.assertNotIn(secret, "\n".join(result.messages))
            managed = json.loads(managed_env.read_text("utf-8"))
            self.assertEqual(managed["LEGACY"], "keep-me")
            self.assertEqual(managed["OPENAI_API_KEY"], secret)

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_codex_without_env_support_uses_wrapper_not_env_flags(self):
        from puppetmaster.installers import install_codex_mcp

        with TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n\n'
                "[mcp_servers.puppetmaster.env]\n"
                'FOO = "old"\n',
                encoding="utf-8",
            )
            stub_log = Path(tmp) / "codex_calls.log"
            stub = Path(tmp) / "codex-stub"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                'if [ "$1" = "mcp" ] && [ "$2" = "add" ] && [ "$3" = "--help" ]; then\n'
                '  echo "Usage: codex mcp add <NAME> -- <COMMAND>"\n'
                "  exit 0\n"
                "fi\n"
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
            wrapper = Path(tmp) / "codex-mcp-wrapper.py"
            managed_env = Path(tmp) / "codex-mcp.env.json"
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False), \
                    patch("puppetmaster.installers._CODEX_WRAPPER_PATH", wrapper), \
                    patch("puppetmaster.installers._CODEX_MANAGED_ENV_PATH", managed_env):
                result = install_codex_mcp(codex_executable=str(stub), skip_handshake=True)
            self.assertEqual(result.status, "installed", msg=result.messages)
            log = stub_log.read_text("utf-8")
            self.assertNotIn("--env", log)
            self.assertIn(f"mcp add puppetmaster -- {sys.executable} {wrapper}", log)
            self.assertTrue(managed_env.exists())

    def test_install_claude_reports_error_when_cli_missing(self):
        from puppetmaster.installers import install_claude_mcp

        result = install_claude_mcp(
            claude_executable="/nonexistent/claude-that-cannot-exist",
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
    def test_install_claude_invokes_mcp_add_user_scope_via_stub(self):
        """Stub the `claude` CLI and verify the user-scope `mcp add` argv."""
        from puppetmaster.installers import install_claude_mcp

        with TemporaryDirectory() as tmp:
            stub_log = Path(tmp) / "claude_calls.log"
            stub = Path(tmp) / "claude"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                'if [ "$1" = "mcp" ] && [ "$2" = "get" ]; then\n'
                '  echo "No MCP server found with name: puppetmaster" >&2\n'
                "  exit 1\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            result = install_claude_mcp(
                claude_executable=str(stub),
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
                f"mcp add --scope user puppetmaster -- {sys.executable} -m puppetmaster.mcp_server",
                log,
                msg=f"unexpected claude args. full log:\n{log}",
            )

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_install_claude_idempotent_when_entry_already_matches(self):
        """When `claude mcp get` reports a matching entry, skip the rewrite."""
        from puppetmaster.installers import install_claude_mcp

        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "claude"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                'if [ "$1" = "mcp" ] && [ "$2" = "get" ]; then\n'
                '  echo "puppetmaster:"\n'
                '  echo "  Type: stdio"\n'
                f'  echo "  Command: {sys.executable}"\n'
                '  echo "  Args: -m puppetmaster.mcp_server"\n'
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            result = install_claude_mcp(
                claude_executable=str(stub),
                skip_handshake=True,
            )
            self.assertEqual(result.status, "unchanged")

    @unittest.skipIf(
        sys.platform == "win32",
        "bash CLI stub is POSIX-only scaffolding; installer logic is OS-agnostic "
        "and covered on Linux + macOS",
    )
    def test_uninstall_claude_removes_user_scope_entry(self):
        from puppetmaster.installers import uninstall_claude_mcp

        with TemporaryDirectory() as tmp:
            stub_log = Path(tmp) / "claude_calls.log"
            stub = Path(tmp) / "claude"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$@" >> {stub_log}\n'
                "exit 0\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            result = uninstall_claude_mcp(claude_executable=str(stub))
            self.assertEqual(result.status, "removed")
            self.assertIn(
                "mcp remove --scope user puppetmaster",
                stub_log.read_text("utf-8"),
            )

    def test_uninstall_claude_unchanged_when_cli_missing(self):
        from puppetmaster.installers import uninstall_claude_mcp

        result = uninstall_claude_mcp(
            claude_executable="/nonexistent/claude-that-cannot-exist",
        )
        self.assertEqual(result.status, "unchanged")

    def test_resolve_claude_command_multiword_and_missing(self):
        from puppetmaster.installers import resolve_claude_command

        # Multi-word commands resolve their head and keep the tail.
        resolved = resolve_claude_command(f"{sys.executable} -m something")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[1:], ["-m", "something"])
        self.assertIsNone(resolve_claude_command("/nonexistent/claude-nope"))


class InstallHermesMcpTests(unittest.TestCase):
    """Tests for :func:`install_hermes_mcp` / :func:`uninstall_hermes_mcp`.

    Hermes owns ``~/.hermes/config.yaml``; the installer edits it directly
    (idempotent, preserving every other key) rather than shelling out to the
    interactive ``hermes mcp add``. These tests target a tempdir config so the
    user's real ``~/.hermes`` is never touched.
    """

    def setUp(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed; install puppetmaster-ai[hermes]")

    def _load(self, path):
        import yaml

        return yaml.safe_load(path.read_text("utf-8"))

    def test_install_hermes_writes_entry_into_empty_config(self):
        from puppetmaster.installers import install_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            result = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")
            data = self._load(target)
            entry = data["mcp_servers"]["puppetmaster"]
            self.assertEqual(entry["command"], sys.executable)
            self.assertEqual(entry["args"], ["-m", "puppetmaster.mcp_server"])

    def test_install_hermes_preserves_other_servers_and_keys(self):
        from puppetmaster.installers import install_hermes_mcp
        import yaml

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            prior = {
                "model": {"default": "nous/hermes", "provider": "nous"},
                "mcp_servers": {
                    "navdata": {"url": "https://example.com/sse"},
                    "puppetmaster": {
                        "command": "python",
                        "args": ["-m", "puppetmaster.mcp_server"],
                        "env": {"PUPPETMASTER_STATE_DIR": "/keep/me"},
                        "tools": {"include": ["puppetmaster_doctor"]},
                    },
                },
            }
            target.write_text(yaml.safe_dump(prior, sort_keys=False), encoding="utf-8")
            result = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")
            data = self._load(target)
            self.assertEqual(data["model"]["default"], "nous/hermes")
            self.assertEqual(
                data["mcp_servers"]["navdata"]["url"],
                "https://example.com/sse",
                msg="install must not touch unrelated MCP servers",
            )
            entry = data["mcp_servers"]["puppetmaster"]
            self.assertEqual(entry["command"], sys.executable)
            self.assertEqual(
                entry["env"]["PUPPETMASTER_STATE_DIR"],
                "/keep/me",
                msg="install must preserve user-set env keys",
            )
            self.assertEqual(
                entry["tools"]["include"],
                ["puppetmaster_doctor"],
                msg="install must preserve a user's tool-include filter",
            )

    def test_install_hermes_idempotent_when_entry_already_matches(self):
        from puppetmaster.installers import install_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            first = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(first.status, "installed")
            mtime_after_first = target.stat().st_mtime_ns
            second = install_hermes_mcp(
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

    def test_install_hermes_force_rewrites_even_when_match(self):
        from puppetmaster.installers import install_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            result = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                force=True,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "installed")

    def test_install_hermes_dry_run_does_not_write(self):
        from puppetmaster.installers import install_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            result = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                dry_run=True,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "would_install")
            self.assertFalse(target.exists())

    def test_install_hermes_dispatch_writes_persistent_soul_rule(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hermes_home = root / "hermes"
            state_dir = root / "state"
            config = hermes_home / "config.yaml"
            soul = hermes_home / "SOUL.md"

            with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}, clear=False):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = cli_main([
                        "--state-dir",
                        str(state_dir),
                        "install-hermes-mcp",
                        "--path",
                        str(config),
                        "--skip-handshake",
                    ])
                self.assertEqual(rc, 0)
                self.assertTrue(soul.is_file())
                self.assertIn("puppetmaster:rules:begin", soul.read_text("utf-8"))
                self.assertIn("[install-hermes-rule] installed", stdout.getvalue())

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = cli_main([
                        "--state-dir",
                        str(state_dir),
                        "install-hermes-mcp",
                        "--path",
                        str(config),
                        "--skip-handshake",
                    ])
                self.assertEqual(rc, 0)
                self.assertIn("[install-hermes-rule] unchanged", stdout.getvalue())
                self.assertEqual(
                    soul.read_text("utf-8").count("puppetmaster:rules:begin"),
                    1,
                    msg="re-run must not duplicate the Hermes SOUL.md rule block",
                )

            with TemporaryDirectory() as dry_tmp:
                dry_root = Path(dry_tmp)
                dry_home = dry_root / "hermes"
                dry_state = dry_root / "state"
                dry_config = dry_home / "config.yaml"
                with patch.dict(os.environ, {"HERMES_HOME": str(dry_home)}, clear=False):
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        rc = cli_main([
                            "--state-dir",
                            str(dry_state),
                            "install-hermes-mcp",
                            "--path",
                            str(dry_config),
                            "--skip-handshake",
                            "--dry-run",
                        ])
                    self.assertEqual(rc, 0)
                    self.assertIn("[install-hermes-rule] would_install", stdout.getvalue())
                    self.assertFalse((dry_home / "SOUL.md").exists())

    def test_install_hermes_reports_error_on_invalid_yaml(self):
        from puppetmaster.installers import install_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            target.write_text("model: [unclosed\n", encoding="utf-8")
            result = install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            self.assertEqual(result.status, "error")

    def test_build_hermes_entry_drops_stale_http_transport(self):
        from puppetmaster.installers import build_hermes_mcp_entry

        entry = build_hermes_mcp_entry(
            sys.executable,
            prior={"url": "https://old", "headers": {"x": "y"}, "env": {"A": "B"}},
        )
        self.assertNotIn("url", entry)
        self.assertNotIn("headers", entry)
        self.assertEqual(entry["command"], sys.executable)
        self.assertEqual(entry["env"], {"A": "B"})

    def test_hermes_config_path_honors_hermes_home(self):
        from puppetmaster.installers import hermes_config_path

        with TemporaryDirectory() as tmp:
            path = hermes_config_path({"HERMES_HOME": tmp})
            self.assertEqual(path, Path(tmp) / "config.yaml")

    def test_uninstall_hermes_removes_entry_and_is_idempotent(self):
        from puppetmaster.installers import install_hermes_mcp, uninstall_hermes_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            install_hermes_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            removed = uninstall_hermes_mcp(target_path=target)
            self.assertEqual(removed.status, "removed")
            self.assertNotIn("puppetmaster", self._load(target).get("mcp_servers", {}))
            again = uninstall_hermes_mcp(target_path=target)
            self.assertEqual(again.status, "unchanged")

    def test_uninstall_hermes_unchanged_when_config_missing(self):
        from puppetmaster.installers import uninstall_hermes_mcp

        with TemporaryDirectory() as tmp:
            result = uninstall_hermes_mcp(target_path=Path(tmp) / "config.yaml")
            self.assertEqual(result.status, "unchanged")


class InstallHermesSkillTests(unittest.TestCase):
    """Tests for :func:`install_hermes_skill` — shipping the bundled Puppetmaster
    skill into Hermes' skills dir so a fresh `pip install` has procedural
    knowledge, not just the per-turn hook nudge."""

    def test_bundled_skill_is_packaged(self):
        from puppetmaster.installers import bundled_skill_dir

        src = bundled_skill_dir()
        self.assertIsNotNone(src, "the puppetmaster skill must ship in the package")
        self.assertTrue((src / "SKILL.md").is_file())
        body = (src / "SKILL.md").read_text(encoding="utf-8")
        # The skill must teach the edit verb — the whole point of v0.9.73+.
        self.assertIn("puppetmaster_edit", body)
        self.assertIn("name: puppetmaster", body)

    def test_install_skill_into_empty_dir(self):
        from puppetmaster.installers import install_hermes_skill

        with TemporaryDirectory() as tmp:
            skills = Path(tmp)
            out = install_hermes_skill(skills_dir=skills)
            self.assertEqual(out.status, "installed")
            landed = skills / "autonomous-ai-agents" / "puppetmaster" / "SKILL.md"
            self.assertTrue(landed.is_file())

    def test_install_skill_idempotent(self):
        from puppetmaster.installers import install_hermes_skill

        with TemporaryDirectory() as tmp:
            skills = Path(tmp)
            install_hermes_skill(skills_dir=skills)
            again = install_hermes_skill(skills_dir=skills)
            self.assertEqual(again.status, "unchanged")

    def test_install_skill_does_not_clobber_customized_without_force(self):
        from puppetmaster.installers import install_hermes_skill

        with TemporaryDirectory() as tmp:
            skills = Path(tmp)
            target = skills / "autonomous-ai-agents" / "puppetmaster"
            target.mkdir(parents=True)
            (target / "SKILL.md").write_text("# my customized skill\n", encoding="utf-8")
            out = install_hermes_skill(skills_dir=skills)
            self.assertEqual(out.status, "skipped")
            # The user's content is preserved.
            self.assertIn(
                "customized", (target / "SKILL.md").read_text(encoding="utf-8")
            )
            # ...until they opt in with force.
            forced = install_hermes_skill(skills_dir=skills, force=True)
            self.assertEqual(forced.status, "updated")
            self.assertIn(
                "name: puppetmaster",
                (target / "SKILL.md").read_text(encoding="utf-8"),
            )

    def test_install_skill_dry_run_writes_nothing(self):
        from puppetmaster.installers import install_hermes_skill

        with TemporaryDirectory() as tmp:
            skills = Path(tmp)
            out = install_hermes_skill(skills_dir=skills, dry_run=True)
            self.assertEqual(out.status, "would_install")
            self.assertFalse(
                (skills / "autonomous-ai-agents" / "puppetmaster").exists()
            )


class InstallHermesPluginTests(unittest.TestCase):
    """Tests for :func:`install_hermes_plugin` — shipping the bundled
    puppetmaster-learn plugin into Hermes' plugins dir so the auto-/learn
    flywheel is present after a fresh `pip install`."""

    def test_bundled_plugin_is_packaged(self):
        from puppetmaster.installers import bundled_plugin_dir

        src = bundled_plugin_dir()
        self.assertIsNotNone(src, "the puppetmaster-learn plugin must ship in the package")
        self.assertTrue((src / "plugin.yaml").is_file())
        self.assertTrue((src / "__init__.py").is_file())

    def test_install_plugin_into_empty_dir(self):
        from puppetmaster.installers import install_hermes_plugin

        with TemporaryDirectory() as tmp:
            plugins = Path(tmp)
            out = install_hermes_plugin(plugins_dir=plugins)
            self.assertEqual(out.status, "installed")
            landed = plugins / "puppetmaster-learn"
            self.assertTrue((landed / "plugin.yaml").is_file())
            self.assertTrue((landed / "__init__.py").is_file())

    def test_install_plugin_honors_hermes_home(self):
        from puppetmaster.installers import install_hermes_plugin

        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "fakehermes"
            out = install_hermes_plugin(env={"HERMES_HOME": str(home)})
            self.assertEqual(out.status, "installed")
            self.assertTrue((home / "plugins" / "puppetmaster-learn" / "plugin.yaml").is_file())

    def test_install_plugin_idempotent(self):
        from puppetmaster.installers import install_hermes_plugin

        with TemporaryDirectory() as tmp:
            plugins = Path(tmp)
            install_hermes_plugin(plugins_dir=plugins)
            again = install_hermes_plugin(plugins_dir=plugins)
            self.assertEqual(again.status, "unchanged")

    def test_install_plugin_dry_run_writes_nothing(self):
        from puppetmaster.installers import install_hermes_plugin

        with TemporaryDirectory() as tmp:
            plugins = Path(tmp)
            out = install_hermes_plugin(plugins_dir=plugins, dry_run=True)
            self.assertEqual(out.status, "would_install")
            self.assertFalse((plugins / "puppetmaster-learn").exists())


class PuppetmasterLearnPluginTests(unittest.TestCase):
    """Tests for the bundled plugin's pure helpers. The plugin is not importable
    as a normal package (its dir name has a hyphen and must not import
    puppetmaster), so we load it by file path via importlib."""

    @staticmethod
    def _load_plugin_module():
        import importlib.util

        from puppetmaster.installers import bundled_plugin_dir

        src = bundled_plugin_dir()
        assert src is not None, "bundled plugin must be packaged"
        spec = importlib.util.spec_from_file_location(
            "puppetmaster_learn_under_test", src / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_build_skill_candidate_shape(self):
        mod = self._load_plugin_module()
        job = {
            "id": "job-abc123",
            "goal": "Fix the flaky retry backoff in the network client",
            "summary": "The backoff used a fixed delay; switched to exponential with jitter.",
            "cwd": "/tmp/project",
        }
        candidate = mod.build_skill_candidate(job)

        self.assertIn("slug", candidate)
        self.assertTrue(candidate["slug"])
        self.assertLessEqual(len(candidate["slug"]), 60)
        self.assertNotIn(" ", candidate["slug"])

        skill_md = candidate["skill_md"]
        # Provenance line names the source job.
        self.assertIn("from job job-abc123", skill_md)
        # Frontmatter carries a name.
        self.assertIn("name:", skill_md)
        self.assertTrue(skill_md.startswith("---\n"))

        meta = candidate["meta"]
        self.assertEqual(meta["job_id"], "job-abc123")
        self.assertEqual(meta["source"], "puppetmaster-auto-learn")
        self.assertIn("created_iso", meta)

    def test_write_candidate_writes_files_and_is_idempotent(self):
        mod = self._load_plugin_module()
        job = {
            "id": "job-xyz789",
            "goal": "Add structured logging to the ingest pipeline",
            "summary": "Introduced a JSON formatter and request-id propagation.",
            "cwd": "/tmp/project",
        }
        candidate = mod.build_skill_candidate(job)

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = mod._write_candidate(
                candidate["slug"], candidate["skill_md"], candidate["meta"], home=home
            )
            self.assertIsNotNone(path)
            written = Path(path)
            self.assertTrue((written / "SKILL.md").is_file())
            self.assertTrue((written / "candidate.json").is_file())
            recorded = json.loads((written / "candidate.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["job_id"], "job-xyz789")

            # Second call for the same job_id is a no-op (idempotent).
            again = mod._write_candidate(
                candidate["slug"], candidate["skill_md"], candidate["meta"], home=home
            )
            self.assertIsNone(again)
            dirs = [p for p in (home / "skills-candidates").iterdir() if p.is_dir()]
            self.assertEqual(len(dirs), 1)

    def test_normalize_detail_reads_full_status_snapshot(self):
        """`status <job>` (full JSON, not --compact) carries goal/status/tasks.

        Regression guard: `status --compact` strips prompt bodies including the
        goal, which silently produced empty "Untitled" candidates.
        """
        from datetime import datetime, timezone

        mod = self._load_plugin_module()
        # Durability requires a *recent* completion (30-min window), so generate
        # the timestamp dynamically — a hardcoded one silently goes stale within
        # the hour and turns this into a clock-dependent time-bomb.
        payload = {
            "job": {
                "goal": "Build the auto-/learn flywheel",
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            "tasks": [{"id": "task_1"}, {"id": "task_2"}],
        }
        detail = mod._normalize_detail(payload)
        self.assertEqual(detail["goal"], "Build the auto-/learn flywheel")
        self.assertEqual(detail["status"], "complete")
        self.assertEqual(detail["task_count"], 2)
        # A completed swarm with tasks is durable enough for a candidate.
        self.assertTrue(mod._is_durable(detail))

    def test_goal_from_summary_parses_goal_line(self):
        mod = self._load_plugin_module()
        summary = (
            "# Puppetmaster Stitched Summary\n\n"
            "Goal: Wire the Hermes SOUL.md reflexive-routing rule\n\n"
            "## Findings\n- did the thing\n"
        )
        self.assertEqual(
            mod._goal_from_summary(summary),
            "Wire the Hermes SOUL.md reflexive-routing rule",
        )
        self.assertEqual(mod._goal_from_summary(""), "")
        self.assertEqual(mod._goal_from_summary("no goal here"), "")


class HermesRegistryCredentialTests(unittest.TestCase):
    """Credential-aware filtering of the curated Hermes catalog.

    The router must never seed a Hermes model whose provider has no usable
    credential, or it will confidently route a worker to a guaranteed runtime
    failure. These tests isolate ``$HOME`` (so the developer's real
    ``~/.hermes`` never leaks in) and the process environment.
    """

    def _isolated(self, tmp, env):
        """Patch ``Path.home`` to a tempdir and replace os.environ with ``env``."""
        from pathlib import Path as _P

        return (
            patch("puppetmaster.adapters.Path.home", return_value=_P(tmp)),
            patch.dict(os.environ, env, clear=True),
        )

    def test_available_providers_from_process_env(self):
        from puppetmaster.adapters import available_hermes_providers

        with TemporaryDirectory() as tmp:
            home_patch, env_patch = self._isolated(
                tmp, {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}
            )
            with home_patch, env_patch:
                self.assertEqual(
                    available_hermes_providers(), {"gemini", "anthropic"}
                )

    def test_google_key_satisfies_gemini_only(self):
        from puppetmaster.adapters import available_hermes_providers

        with TemporaryDirectory() as tmp:
            home_patch, env_patch = self._isolated(tmp, {"GOOGLE_API_KEY": "g"})
            with home_patch, env_patch:
                self.assertEqual(available_hermes_providers(), {"gemini"})

    def test_available_providers_from_env_file(self):
        from puppetmaster.adapters import available_hermes_providers

        with TemporaryDirectory() as tmp:
            hermes_dir = Path(tmp) / ".hermes"
            hermes_dir.mkdir()
            (hermes_dir / ".env").write_text("OPENAI_API_KEY=sk-x\n", encoding="utf-8")
            home_patch, env_patch = self._isolated(tmp, {})
            with home_patch, env_patch:
                self.assertEqual(available_hermes_providers(), {"openai-api"})

    def test_oauth_providers_counted(self):
        from puppetmaster.adapters import available_hermes_providers

        with TemporaryDirectory() as tmp:
            hermes_dir = Path(tmp) / ".hermes"
            hermes_dir.mkdir()
            (hermes_dir / "auth.json").write_text(
                json.dumps({"providers": {"nous": {"token": "x"}}}), encoding="utf-8"
            )
            home_patch, env_patch = self._isolated(tmp, {})
            with home_patch, env_patch:
                self.assertEqual(available_hermes_providers(), {"nous"})

    def test_no_credentials_returns_empty(self):
        from puppetmaster.adapters import available_hermes_providers

        with TemporaryDirectory() as tmp:
            home_patch, env_patch = self._isolated(tmp, {})
            with home_patch, env_patch:
                self.assertEqual(available_hermes_providers(), set())

    def test_curated_to_specs_filters_by_allowed_providers(self):
        from puppetmaster.static_catalog import curated_to_specs

        specs = {
            s.adapter_model_name: s
            for s in curated_to_specs(
                "hermes", "api", [], allowed_providers={"gemini"}
            )
        }
        self.assertIn("gemini-2.5-flash", specs)
        self.assertNotIn("gpt-5", specs)
        self.assertNotIn("claude-sonnet-4-5", specs)

    def test_curated_to_specs_none_keeps_all(self):
        from puppetmaster.static_catalog import curated_to_specs

        specs = curated_to_specs("hermes", "api", [], allowed_providers=None)
        names = {s.adapter_model_name for s in specs}
        self.assertIn("gpt-5", names)
        self.assertIn("gemini-2.5-flash", names)
        self.assertIn("claude-sonnet-4-5", names)

    def test_merge_reports_skipped_models(self):
        from puppetmaster.static_catalog import merge_curated_into_registry

        merged, report = merge_curated_into_registry(
            "hermes", "api", [], allowed_providers={"gemini"}
        )
        seeded = {s.adapter_model_name for s in merged if s.adapter == "hermes"}
        self.assertIn("gemini-2.5-flash", seeded)
        self.assertNotIn("gpt-5", seeded)
        skipped_models = {s["model"] for s in report["skipped"]}
        self.assertIn("gpt-5", skipped_models)
        self.assertIn("claude-sonnet-4-5", skipped_models)
        providers = {s["provider"] for s in report["skipped"]}
        self.assertEqual(providers, {"openai-api", "anthropic"})

    def test_merge_no_filter_reports_empty_skipped(self):
        from puppetmaster.static_catalog import merge_curated_into_registry

        _, report = merge_curated_into_registry("hermes", "api", [])
        self.assertEqual(report["skipped"], [])


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
        self.assertIn("Use Puppetmaster to", content)
        self.assertIn("PM this", content)

    def test_render_agents_block_is_marker_wrapped(self):
        from puppetmaster.rules import BEGIN_MARKER, END_MARKER, render_agents_block

        block = render_agents_block()
        self.assertTrue(block.startswith(BEGIN_MARKER))
        self.assertTrue(block.rstrip().endswith(END_MARKER))
        self.assertIn("# Puppetmaster orchestration", block)
        self.assertIn("Delegate-first gate", block)

    def test_rules_mandate_codegraph_first_exploration(self):
        """The managed rules must push CodeGraph as hard as delegation:
        graph every directory touched, explore the graph not the tree, and
        use partial graphs + narrow native search for unsupported languages
        instead of re-crawling covered code."""
        from puppetmaster.rules import RULE_BODY, render_cursor_mdc

        for content in (RULE_BODY, render_cursor_mdc()):
            flattened = " ".join(content.split())
            self.assertIn("CodeGraph-first exploration (must obey)", flattened)
            self.assertIn("graph every directory you interact with", flattened)
            self.assertIn("puppetmaster_codegraph_init", flattened)
            self.assertIn("puppetmaster_codegraph_status", flattened)
            self.assertIn("Partial coverage is still coverage", flattened)
            self.assertIn("never re-crawl directories the graph already covers", flattened)

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

    def test_install_rules_skips_cursor_when_platform_lock_excludes_it(self):
        """A claude-code-only user (git repo cwd) must not get a .cursor rule.

        Auto-detection treats any git repo as "cursor present", so without the
        platform-lock filter setup would write .cursor/rules/puppetmaster.mdc on a
        machine where the user never routes to Cursor. ``agents`` (cross-tool)
        still lands.
        """
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            result = install_rules(cwd=cwd, enabled_adapters={"claude-code"})
            self.assertEqual(result.overall_status, "installed")
            self.assertFalse(
                (cwd / ".cursor" / "rules" / "puppetmaster.mdc").exists(),
                msg="cursor disabled by the lock must not write a .cursor rule",
            )
            self.assertTrue((cwd / "AGENTS.md").is_file())
            self.assertEqual(
                {o.target for o in result.outcomes}, {"agents"}
            )

    def test_install_rules_includes_cursor_when_lock_enables_it(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            result = install_rules(cwd=cwd, enabled_adapters={"cursor"})
            self.assertTrue((cwd / ".cursor" / "rules" / "puppetmaster.mdc").is_file())
            self.assertIn("cursor", {o.target for o in result.outcomes})

    def test_install_rules_unfiltered_default_still_writes_cursor_in_git_repo(self):
        """enabled_adapters=None (standalone install-rules) preserves behavior."""
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            result = install_rules(cwd=cwd)
            self.assertTrue((cwd / ".cursor" / "rules" / "puppetmaster.mdc").is_file())

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

    def test_install_rules_writes_hermes_soul_block(self):
        """Explicit hermes_global target writes a managed block into SOUL.md."""
        from puppetmaster.rules import install_rules, hermes_soul_path

        with TemporaryDirectory() as home_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                soul = hermes_soul_path()
                result = install_rules(targets=["hermes_global"])
                self.assertEqual(result.overall_status, "installed")
                self.assertEqual(soul, Path(home_tmp) / "SOUL.md")
                self.assertTrue(soul.is_file())
                text = soul.read_text(encoding="utf-8")
                self.assertIn("puppetmaster:rules:begin", text)
                self.assertIn("Puppetmaster orchestration", text)

    def test_install_rules_hermes_preserves_existing_soul_content(self):
        """The managed block must merge into a populated SOUL.md, not clobber it."""
        from puppetmaster.rules import install_rules, hermes_soul_path

        with TemporaryDirectory() as home_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                soul = hermes_soul_path()
                soul.write_text("# Persona\nload-bearing user content\n", encoding="utf-8")
                install_rules(targets=["hermes_global"])
                text = soul.read_text(encoding="utf-8")
                self.assertIn("# Persona", text)
                self.assertIn("load-bearing user content", text)
                self.assertIn("puppetmaster:rules:begin", text)

    def test_install_rules_hermes_idempotent_on_rerun(self):
        from puppetmaster.rules import install_rules, hermes_soul_path

        with TemporaryDirectory() as home_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                soul = hermes_soul_path()
                install_rules(targets=["hermes_global"])
                result = install_rules(targets=["hermes_global"])
                self.assertEqual(result.outcomes[0].status, "unchanged")
                self.assertEqual(
                    soul.read_text(encoding="utf-8").count("puppetmaster:rules:begin"),
                    1,
                    msg="re-run must not duplicate the Hermes block",
                )

    def test_install_rules_auto_detects_hermes_with_global(self):
        """--global picks up hermes_global when ~/.hermes exists (HERMES_HOME).

        ``enabled_adapters`` is pinned to ``{"hermes"}`` so the run stays
        hermetic — it must not touch the host's real ~/.codex / ~/.claude.
        """
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as home_tmp, TemporaryDirectory() as work_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                result = install_rules(
                    cwd=Path(work_tmp),
                    install_global=True,
                    enabled_adapters={"hermes"},
                )
                targets = {o.target for o in result.outcomes}
                self.assertIn("hermes_global", targets)
                self.assertNotIn("codex_global", targets)
                self.assertNotIn("claude_global", targets)

    def test_install_rules_skips_hermes_when_platform_lock_excludes_it(self):
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as home_tmp, TemporaryDirectory() as work_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                result = install_rules(
                    cwd=Path(work_tmp),
                    install_global=True,
                    enabled_adapters={"cursor"},
                )
                targets = {o.target for o in result.outcomes}
                self.assertNotIn("hermes_global", targets)

    def test_uninstall_rules_strips_hermes_block_preserving_content(self):
        from puppetmaster.rules import install_rules, uninstall_rules, hermes_soul_path

        with TemporaryDirectory() as home_tmp:
            with patch.dict(os.environ, {"HERMES_HOME": home_tmp}):
                soul = hermes_soul_path()
                soul.write_text("# Persona\nkeep me\n", encoding="utf-8")
                install_rules(targets=["hermes_global"])
                result = uninstall_rules(targets=["hermes_global"])
                self.assertEqual(result.outcomes[0].status, "removed")
                text = soul.read_text(encoding="utf-8")
                self.assertIn("# Persona", text)
                self.assertIn("keep me", text)
                self.assertNotIn("puppetmaster:rules:begin", text)

    def test_doctor_agent_rules_check_warns_when_mcp_present_but_no_rules(self):
        """Doctor should nudge the user when MCP is wired but rules are missing."""
        from puppetmaster import diagnostics
        from puppetmaster.diagnostics import _agent_rules_check

        # Pin a clean home: the check also inspects ~/.codex / ~/.claude, so a
        # real global-rules install (now a first-class option) must not make this
        # non-hermetic.
        with TemporaryDirectory() as tmp, TemporaryDirectory() as home_tmp:
            cwd = Path(tmp)
            (cwd / ".cursor").mkdir()
            (cwd / ".cursor" / "mcp.json").write_text(
                '{"mcpServers": {"puppetmaster": {"command": "python"}}}',
                encoding="utf-8",
            )
            with patch.object(diagnostics.Path, "home", return_value=Path(home_tmp)):
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


class MarginalCostRoutingTests(unittest.TestCase):
    def _plan_priced_registry(self):
        from puppetmaster.model_registry import ModelSpec

        return [
            ModelSpec(
                id="claude-code/opus",
                adapter="claude-code",
                adapter_model_name="opus",
                capability_score=82,
                input_per_mtok_usd=5.0,
                output_per_mtok_usd=25.0,
                billing="plan",
            ),
            ModelSpec(
                id="cursor/composer",
                adapter="cursor",
                adapter_model_name="composer",
                capability_score=82,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                billing="plan",
            ),
        ]

    def test_plan_billed_priced_model_competes_as_zero_peer(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="implement a feature", role="implement")
        decision = route_task(signal, self._plan_priced_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "claude-code/opus")
        for _, reason in decision.rejected:
            self.assertNotIn("pricier", reason)

    def test_explicit_max_cost_allows_plan_billed_priced_model(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(
            instruction="implement a feature",
            role="implement",
            explicit_max_cost_usd=0.0005,
        )
        decision = route_task(signal, self._plan_priced_registry(), policy="balanced")
        self.assertEqual(decision.model.id, "claude-code/opus")

    def test_routing_decision_records_marginal_and_nominal_costs(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signal = TaskSignals(instruction="implement a feature", role="implement")
        decision = route_task(signal, self._plan_priced_registry(), policy="balanced")
        self.assertEqual(decision.estimated_cost_usd, 0.0)
        self.assertGreater(decision.nominal_cost_usd, 0.0)
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["estimated_cost_usd"], 0.0)
        self.assertGreater(payload["nominal_cost_usd"], 0.0)
        self.assertIn("baseline_nominal_cost_usd", payload)


class RegistryReconciliationTests(unittest.TestCase):
    def test_reconcile_upgrades_unknown_billing(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.platform_billing import BillingStatus, reconcile_registry

        specs = [
            ModelSpec(
                id="claude-code/opus",
                adapter="claude-code",
                adapter_model_name="opus",
                billing="unknown",
            )
        ]

        def detect(adapter: str) -> BillingStatus:
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=True,
                detail="oauth",
                evidence=[],
            )

        result = reconcile_registry(specs, detect=detect)
        self.assertEqual(result.specs[0].billing, "plan")
        self.assertEqual(
            result.upgraded,
            [{"model_id": "claude-code/opus", "from": "unknown", "to": "plan"}],
        )
        self.assertEqual(result.dropped, [])

    def test_reconcile_filters_unhealthy_when_healthy_survivors_exist(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.platform_billing import BillingStatus, reconcile_registry

        specs = [
            ModelSpec(
                id="cursor/composer",
                adapter="cursor",
                adapter_model_name="composer",
                billing="plan",
            ),
            ModelSpec(
                id="claude-code/opus",
                adapter="claude-code",
                adapter_model_name="opus",
                billing="unknown",
            ),
        ]

        def detect(adapter: str) -> BillingStatus:
            if adapter == "cursor":
                return BillingStatus(
                    adapter=adapter,
                    billing="unknown",
                    healthy=False,
                    detail="CURSOR_API_KEY is not set",
                    evidence=[],
                )
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=True,
                detail="oauth",
                evidence=[],
            )

        result = reconcile_registry(specs, detect=detect)
        self.assertEqual([s.id for s in result.specs], ["claude-code/opus"])
        self.assertEqual(result.specs[0].billing, "plan")
        self.assertEqual(len(result.dropped), 1)

    def test_reconcile_drops_unhealthy_adapter(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.platform_billing import BillingStatus, reconcile_registry

        specs = [
            ModelSpec(
                id="cursor/composer",
                adapter="cursor",
                adapter_model_name="composer",
                billing="plan",
            )
        ]

        def detect(adapter: str) -> BillingStatus:
            return BillingStatus(
                adapter=adapter,
                billing="unknown",
                healthy=False,
                detail="CURSOR_API_KEY is not set",
                evidence=[],
            )

        result = reconcile_registry(specs, detect=detect)
        self.assertEqual(len(result.specs), 1)
        self.assertEqual(result.specs[0].id, "cursor/composer")
        self.assertEqual(len(result.dropped), 1)
        self.assertIn("no usable credentials", result.dropped[0]["reason"])

    def test_reconcile_detection_exception_keeps_spec(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.platform_billing import reconcile_registry

        specs = [
            ModelSpec(
                id="cursor/composer",
                adapter="cursor",
                adapter_model_name="composer",
                billing="plan",
            )
        ]

        def detect(adapter: str):
            raise RuntimeError("probe failed")

        result = reconcile_registry(specs, detect=detect)
        self.assertEqual(len(result.specs), 1)
        self.assertEqual(result.specs[0].id, "cursor/composer")
        self.assertEqual(result.dropped, [])

    def test_reconcile_all_dropped_returns_upgraded_unfiltered(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.platform_billing import BillingStatus, reconcile_registry

        specs = [
            ModelSpec(
                id="cursor/a",
                adapter="cursor",
                adapter_model_name="a",
                billing="unknown",
            ),
            ModelSpec(
                id="cursor/b",
                adapter="cursor",
                adapter_model_name="b",
                billing="unknown",
            ),
        ]

        def detect(adapter: str) -> BillingStatus:
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=False,
                detail="no key",
                evidence=[],
            )

        result = reconcile_registry(specs, detect=detect)
        self.assertEqual(len(result.specs), 2)
        self.assertTrue(all(s.billing == "plan" for s in result.specs))
        self.assertEqual(len(result.dropped), 2)


class BillingCacheTests(unittest.TestCase):
    def test_detect_adapter_billing_cached_respects_ttl(self) -> None:
        from puppetmaster.platform_billing import (
            BillingStatus,
            clear_billing_cache,
            detect_adapter_billing_cached,
        )

        calls = []

        def fake_detect(adapter: str, **kwargs) -> BillingStatus:
            calls.append(adapter)
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=True,
                detail="ok",
                evidence=[],
            )

        clear_billing_cache()
        with patch(
            "puppetmaster.platform_billing.detect_adapter_billing",
            side_effect=fake_detect,
        ):
            detect_adapter_billing_cached("cursor", ttl_seconds=60)
            detect_adapter_billing_cached("cursor", ttl_seconds=60)
        self.assertEqual(calls, ["cursor"])

        clear_billing_cache()
        with patch(
            "puppetmaster.platform_billing.detect_adapter_billing",
            side_effect=fake_detect,
        ):
            detect_adapter_billing_cached("cursor", ttl_seconds=60)
        self.assertEqual(calls, ["cursor", "cursor"])

    def test_detect_adapter_billing_cached_ttl_zero_bypasses(self) -> None:
        from puppetmaster.platform_billing import (
            BillingStatus,
            clear_billing_cache,
            detect_adapter_billing_cached,
        )

        calls = []

        def fake_detect(adapter: str, **kwargs) -> BillingStatus:
            calls.append(adapter)
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=True,
                detail="ok",
                evidence=[],
            )

        clear_billing_cache()
        with patch(
            "puppetmaster.platform_billing.detect_adapter_billing",
            side_effect=fake_detect,
        ):
            detect_adapter_billing_cached("cursor", ttl_seconds=0)
            detect_adapter_billing_cached("cursor", ttl_seconds=0)
        self.assertEqual(calls, ["cursor", "cursor"])


class AutoRoutingReconciliationTests(unittest.TestCase):
    def test_apply_auto_routing_reconciles_registry_and_audits_drops(self) -> None:
        from puppetmaster.model_registry import ModelSpec, save_registry
        from puppetmaster.models import ArtifactType
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store_factory import create_store
        from puppetmaster.workers import WorkerSpec

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(
                [
                    ModelSpec(
                        id="cursor/composer",
                        adapter="cursor",
                        adapter_model_name="composer",
                        capability_score=82,
                        billing="plan",
                    ),
                    ModelSpec(
                        id="claude-code/opus",
                        adapter="claude-code",
                        adapter_model_name="opus",
                        capability_score=82,
                        input_per_mtok_usd=5.0,
                        output_per_mtok_usd=25.0,
                        billing="unknown",
                    ),
                ],
                registry_path,
            )
            state_dir = Path(tmp) / ".puppetmaster"
            store = create_store("file", state_dir)
            store.init()
            orchestrator = Orchestrator(store)
            job = store.create_job("reconcile routing")

            def detect(adapter: str) -> BillingStatus:
                if adapter == "cursor":
                    return BillingStatus(
                        adapter=adapter,
                        billing="unknown",
                        healthy=False,
                        detail="CURSOR_API_KEY is not set",
                        evidence=[],
                    )
                return BillingStatus(
                    adapter=adapter,
                    billing="plan",
                    healthy=True,
                    detail="oauth",
                    evidence=[],
                )

            spec = WorkerSpec(
                role="implement",
                instruction="implement a feature",
                adapter="local",
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                    "routing_policy": "balanced",
                },
                depends_on_roles=[],
            )

            with patch(
                "puppetmaster.platform_billing.detect_adapter_billing_cached",
                side_effect=detect,
            ):
                tasks = orchestrator._create_tasks(job, [spec])

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].adapter, "claude-code")
            self.assertEqual(tasks[0].payload.get("router_model_id"), "claude-code/opus")

            events = store.read_events(job.id)
            reconciled = [
                e for e in events if e.get("event") == "router.registry_reconciled"
            ]
            self.assertEqual(len(reconciled), 1)
            self.assertEqual(len(reconciled[0]["payload"]["dropped"]), 1)

            routing = [
                a
                for a in store.list_artifacts(job.id)
                if a.type == ArtifactType.ROUTING
            ]
            self.assertEqual(len(routing), 1)
            rejected_ids = {entry["id"] for entry in routing[0].payload["rejected"]}
            self.assertIn("cursor/composer", rejected_ids)


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

    def test_claude_billing_bedrock_env_profile(self) -> None:
        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            s = detect_claude_billing(
                env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "myprofile"},
                home=home,
            )
            self.assertEqual(s.billing, "api")
            self.assertTrue(s.healthy)
            self.assertIn("claude_bedrock:enabled", s.evidence)
            self.assertIn("aws_credentials:profile", s.evidence)

    def test_claude_billing_bedrock_env_bearer_token(self) -> None:
        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            s = detect_claude_billing(
                env={
                    "CLAUDE_CODE_USE_BEDROCK": "true",
                    "AWS_BEARER_TOKEN_BEDROCK": "tok",
                },
                home=home,
            )
            self.assertTrue(s.healthy)
            self.assertIn("claude_bedrock:enabled", s.evidence)
            self.assertIn("aws_credentials:bearer_token", s.evidence)

    def test_claude_billing_bedrock_settings_json_aws_file(self) -> None:
        import json as _json

        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".claude").mkdir()
            (home / ".claude" / "settings.json").write_text(
                _json.dumps({"env": {"CLAUDE_CODE_USE_BEDROCK": "1"}}),
                encoding="utf-8",
            )
            (home / ".aws").mkdir()
            (home / ".aws" / "credentials").write_text(
                "[default]\naws_access_key_id = x\n", encoding="utf-8"
            )
            s = detect_claude_billing(env={}, home=home)
            self.assertTrue(s.healthy)
            self.assertIn("claude_bedrock:enabled", s.evidence)
            self.assertIn("aws_credentials:config_file", s.evidence)

    def test_claude_billing_bedrock_no_credentials(self) -> None:
        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            s = detect_claude_billing(
                env={"CLAUDE_CODE_USE_BEDROCK": "1"},
                home=home,
            )
            self.assertEqual(s.billing, "api")
            self.assertFalse(s.healthy)
            self.assertIn("claude_bedrock:enabled", s.evidence)
            self.assertIn("aws_credentials:missing", s.evidence)

    def test_claude_billing_bedrock_disabled_regression(self) -> None:
        import json as _json

        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            # unset -> existing oauth/api-key behavior.
            (home / ".claude.json").write_text(
                _json.dumps(
                    {
                        "oauthAccount": {
                            "accountUuid": "abc",
                            "seatTier": "pro",
                        }
                    }
                ),
                encoding="utf-8",
            )
            s_unset = detect_claude_billing(env={}, home=home)
            self.assertEqual(s_unset.billing, "plan")
            self.assertTrue(s_unset.healthy)

            s_zero = detect_claude_billing(
                env={"CLAUDE_CODE_USE_BEDROCK": "0"},
                home=home,
            )
            self.assertEqual(s_zero.billing, "plan")
            self.assertTrue(s_zero.healthy)

            s_api = detect_claude_billing(
                env={"ANTHROPIC_API_KEY": "k"},
                home=Path(tmp),
            )
            self.assertEqual(s_api.billing, "api")
            self.assertTrue(s_api.healthy)

    def test_claude_billing_bedrock_wins_over_api_key(self) -> None:
        from puppetmaster.platform_billing import detect_claude_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            s = detect_claude_billing(
                env={
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "ANTHROPIC_API_KEY": "k",
                    "AWS_PROFILE": "bedrock",
                },
                home=home,
            )
            self.assertEqual(s.billing, "api")
            self.assertTrue(s.healthy)
            self.assertIn("claude_bedrock:enabled", s.evidence)
            self.assertIn("aws_credentials:profile", s.evidence)
            self.assertNotIn("anthropic_api_key:set", s.evidence)

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

            s = detect_codex_billing(run=_boom, env={}, home=home)
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
            s = detect_codex_billing(run=lambda cmd: (1, "", ""), env={}, home=home)
            self.assertEqual(s.billing, "plan")
            self.assertIn("codex_auth:chatgpt", s.evidence)

    def test_codex_billing_uses_codex_home_auth_before_default_home(self) -> None:
        import json as _json

        from puppetmaster.platform_billing import detect_codex_billing

        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            codex_home = Path(tmp) / "api-codex-home"
            (home / ".codex").mkdir(parents=True)
            codex_home.mkdir()
            (home / ".codex" / "auth.json").write_text(
                _json.dumps({"auth_mode": "chatgpt", "tokens": {"access": "x"}}),
                encoding="utf-8",
            )
            (codex_home / "auth.json").write_text(
                _json.dumps({"OPENAI_API_KEY": "sk-x", "auth_mode": "apikey"}),
                encoding="utf-8",
            )

            api = detect_codex_billing(
                run=lambda cmd: (_ for _ in ()).throw(
                    AssertionError("subprocess must not run when auth.json exists")
                ),
                env={"CODEX_HOME": str(codex_home)},
                home=home,
            )
            self.assertEqual(api.billing, "api")
            self.assertIn("codex_auth_path:$CODEX_HOME/auth.json", api.evidence)
            self.assertIn("auth_context:process", api.evidence)

            plan = detect_codex_billing(run=lambda cmd: (1, "", ""), env={}, home=home)
            self.assertEqual(plan.billing, "plan")
            self.assertIn("codex_auth_path:~/.codex/auth.json", plan.evidence)

    def test_codex_billing_falls_back_to_login_status(self) -> None:
        from puppetmaster.platform_billing import detect_codex_billing

        # No auth.json -> fall back to parsing `codex login status`.
        with TemporaryDirectory() as tmp:
            home = Path(tmp)  # empty: no ~/.codex/auth.json
            api = detect_codex_billing(
                run=lambda cmd: (0, "Logged in using an API key - sk-***", ""),
                env={},
                home=home,
            )
            self.assertEqual(api.billing, "api")
            chatgpt = detect_codex_billing(
                run=lambda cmd: (0, "Logged in using ChatGPT", ""),
                env={},
                home=home,
            )
            self.assertEqual(chatgpt.billing, "plan")
            missing = detect_codex_billing(
                run=lambda cmd: (127, "", "command not found"),
                env={},
                home=home,
            )
            self.assertEqual(missing.billing, "unknown")
            self.assertFalse(missing.healthy)
            out = detect_codex_billing(
                run=lambda cmd: (1, "Not logged in", ""),
                env={},
                home=home,
            )
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

    def test_catalog_inherits_fable_5_frontier_kin(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(
                id="claude-code/fable-5",
                adapter="claude-code",
                adapter_model_name="claude-fable-5",
                capability_score=100,
                context_window=1_000_000,
                tags=["frontier", "mythos-class", "long-context"],
            )
        ]
        catalog = [{"id": "fable-5", "displayName": "Claude Fable 5"}]
        spec = catalog_to_specs(catalog, existing)[0]
        self.assertEqual(spec.capability_score, 100)
        self.assertEqual(spec.adapter_model_name, "fable-5")
        self.assertEqual(spec.billing, "plan")
        self.assertIn("mythos-class", spec.tags)

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


class AdapterCliPresenceTests(unittest.TestCase):
    """`adapter_cli_present` closes the gap billing detection can't see: an
    adapter that reads billing-healthy off a stale auth file but whose CLI
    binary is gone."""

    def test_cli_less_adapters_are_always_present(self) -> None:
        from puppetmaster.preflight import adapter_cli_executable, adapter_cli_present

        # cursor (bundled SDK runner) and openai (HTTP) have no CLI to install.
        for adapter in ("cursor", "openai"):
            self.assertIsNone(adapter_cli_executable(adapter))
            self.assertTrue(
                adapter_cli_present(adapter, resolver=lambda _name: None)
            )

    def test_claude_and_codex_gate_on_resolvable_binary(self) -> None:
        from puppetmaster.preflight import adapter_cli_executable, adapter_cli_present

        self.assertEqual(adapter_cli_executable("claude-code"), "claude")
        self.assertEqual(adapter_cli_executable("codex"), "codex")

        present = lambda name: f"/usr/local/bin/{name}"
        absent = lambda _name: None
        for adapter in ("claude-code", "codex"):
            self.assertTrue(adapter_cli_present(adapter, resolver=present))
            self.assertFalse(adapter_cli_present(adapter, resolver=absent))

    def test_executable_honors_env_override(self) -> None:
        from puppetmaster.preflight import adapter_cli_executable

        self.assertEqual(
            adapter_cli_executable(
                "claude-code", env={"CLAUDE_CODE_COMMAND": "/opt/claude"}
            ),
            "/opt/claude",
        )
        self.assertEqual(
            adapter_cli_executable("codex", env={"CODEX_COMMAND": "codex-next"}),
            "codex-next",
        )


class BedrockModelResolutionTests(unittest.TestCase):
    """Claude Code on Bedrock rejects short Cursor-style model names; resolution
    must forward a real Bedrock id or omit --model with a precise note rather
    than letting Bedrock reject e.g. `claude-opus-4-8`."""

    def test_is_bedrock_model_id_recognizes_real_ids(self) -> None:
        from puppetmaster.adapters import is_bedrock_model_id

        for good in (
            "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "eu.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-1-20250805-v1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-1-20250805-v1:0",
        ):
            self.assertTrue(is_bedrock_model_id(good), good)
        for bad in ("claude-opus-4-8", "claude-opus-4-6", "gpt-5.5", "", None):
            self.assertFalse(is_bedrock_model_id(bad), repr(bad))

    def test_off_bedrock_returns_requested_model_unchanged(self) -> None:
        from puppetmaster.adapters import DEFAULT_CLAUDE_CODE_MODEL, resolve_claude_code_model

        with TemporaryDirectory() as tmp:
            model, note = resolve_claude_code_model(
                {"model": "claude-opus-4-8"}, env={}, home=Path(tmp)
            )
            self.assertEqual(model, "claude-opus-4-8")
            self.assertIsNone(note)
            # No payload model -> the adapter default, still unchanged.
            model2, note2 = resolve_claude_code_model({}, env={}, home=Path(tmp))
            self.assertEqual(model2, DEFAULT_CLAUDE_CODE_MODEL)
            self.assertIsNone(note2)

    def test_bedrock_prefers_explicit_override(self) -> None:
        from puppetmaster.adapters import resolve_claude_code_model

        env = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "ANTHROPIC_MODEL": "us.anthropic.claude-opus-4-1-20250805-v1:0",
        }
        with TemporaryDirectory() as tmp:
            model, note = resolve_claude_code_model(
                {"model": "claude-opus-4-8"}, env=env, home=Path(tmp)
            )
            self.assertEqual(model, "us.anthropic.claude-opus-4-1-20250805-v1:0")
            self.assertIsNone(note)
            # payload.bedrock_model wins even over a short requested model.
            model2, note2 = resolve_claude_code_model(
                {"model": "claude-opus-4-8", "bedrock_model": "anthropic.claude-x-v1:0"},
                env={"CLAUDE_CODE_USE_BEDROCK": "1"},
                home=Path(tmp),
            )
            self.assertEqual(model2, "anthropic.claude-x-v1:0")
            self.assertIsNone(note2)

    def test_bedrock_passes_through_already_bedrock_shaped_request(self) -> None:
        from puppetmaster.adapters import resolve_claude_code_model

        with TemporaryDirectory() as tmp:
            model, note = resolve_claude_code_model(
                {"model": "us.anthropic.claude-opus-4-1-20250805-v1:0"},
                env={"CLAUDE_CODE_USE_BEDROCK": "1"},
                home=Path(tmp),
            )
            self.assertEqual(model, "us.anthropic.claude-opus-4-1-20250805-v1:0")
            self.assertIsNone(note)

    def test_bedrock_short_name_omits_model_with_actionable_note(self) -> None:
        from puppetmaster.adapters import resolve_claude_code_model

        with TemporaryDirectory() as tmp:
            model, note = resolve_claude_code_model(
                {"model": "claude-opus-4-8"},
                env={"CLAUDE_CODE_USE_BEDROCK": "1"},
                home=Path(tmp),
            )
            self.assertIsNone(model)  # --model omitted, not forwarded to Bedrock
            self.assertIsNotNone(note)
            self.assertIn("ANTHROPIC_MODEL", note)
            self.assertIn("Bedrock", note)


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

    def test_list_all_projects_snapshot_aggregates_across_projects(self) -> None:
        """--all-projects aggregates jobs from every project state dir and
        labels each row with the digest-stripped project slug."""
        from puppetmaster.dashboard import list_all_projects_snapshot
        from puppetmaster.store_factory import create_store

        with TemporaryDirectory() as tmp:
            projects_root = Path(tmp) / "projects"
            dir_a = projects_root / "alpha-0123456789ab"
            dir_b = projects_root / "beta-ba9876543210"
            store_a = create_store("sqlite", dir_a)
            store_a.init()
            job_a = store_a.create_job("alpha goal")
            store_b = create_store("sqlite", dir_b)
            store_b.init()
            job_b = store_b.create_job("beta goal")

            with patch(
                "puppetmaster.state.list_project_state_dirs",
                return_value=[dir_a, dir_b],
            ):
                rows = list_all_projects_snapshot()

            by_id = {row["id"]: row for row in rows}
            self.assertIn(job_a.id, by_id)
            self.assertIn(job_b.id, by_id)
            self.assertEqual(by_id[job_a.id]["project"], "alpha")
            self.assertEqual(by_id[job_b.id]["project"], "beta")
            required = {"id", "goal", "status", "created_at", "completed_at", "project"}
            for row in rows:
                self.assertTrue(required <= set(row))

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

    def test_extract_metadata_falls_back_to_payload_token_counts(self) -> None:
        """Adapters without a Claude-style JSON stdout envelope (cursor/codex/
        openai) still stamp token_usage() counts top-level on the payload —
        the chips must surface those, flagged when estimated."""
        from puppetmaster.dashboard import _extract_metadata

        meta = _extract_metadata(
            {
                "model": "composer-2.5",
                "tokens_in": 2646,
                "tokens_out": 8866,
                "tokens_estimated": True,
                "stdout_capture": {"stdout_head_excerpt": "not json {"},
            }
        )
        self.assertEqual(meta["tokens_in"], 2646)
        self.assertEqual(meta["tokens_out"], 8866)
        self.assertTrue(meta["tokens_estimated"])

        envelope = json.dumps(
            {"usage": {"input_tokens": 10, "output_tokens": 20}, "num_turns": 3}
        )
        measured = _extract_metadata(
            {
                "tokens_in": 999,
                "tokens_estimated": True,
                "stdout_capture": {"stdout_head_excerpt": envelope},
            }
        )
        self.assertEqual(measured["tokens_in"], 10)
        self.assertEqual(measured["tokens_out"], 20)
        self.assertNotIn("tokens_estimated", measured)

    def test_build_job_snapshot_includes_task_activity_and_progress(self) -> None:
        """Each task row carries its instruction plus an artifact-backed
        activity timeline, and PATCH diffs attach to exactly the task that
        produced them (never repeated across cards)."""
        from puppetmaster.dashboard import build_job_snapshot
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus

        with TemporaryDirectory() as tmp:
            store, job = self._seed_store(tmp)
            patched_task = store.list_tasks(job.id)[0]
            other_task = Task(
                job_id=job.id, role="review", instruction="look it over",
                adapter="cursor", status=TaskStatus.RUNNING, payload={}, attempts=1,
            )
            store.save_task(other_task)
            store.save_artifact(
                Artifact(
                    job_id=job.id, task_id=patched_task.id, type=ArtifactType.PATCH,
                    created_by="worker-implement",
                    payload={
                        "change": "add helper",
                        "files": ["helper.py"],
                        "unified_diff": "diff --git a/helper.py b/helper.py\n+x\n",
                        "diff_truncated": False,
                        "diff_total_chars": 40,
                    },
                    confidence=1.0, evidence=["worktree"],
                )
            )

            snap = build_job_snapshot(store, job.id)

            self.assertEqual(snap["progress"], {"complete": 1, "running": 1})
            by_id = {row["id"]: row for row in snap["tasks"]}
            patched_row = by_id[patched_task.id]
            self.assertEqual(patched_row["instruction"], "do the thing")
            activity_types = [item["type"] for item in patched_row["activity"]]
            self.assertIn("routing", activity_types)
            self.assertIn("patch", activity_types)
            patch_item = next(
                item for item in patched_row["activity"] if item["type"] == "patch"
            )
            self.assertEqual(patch_item["diff"]["files"], ["helper.py"])
            routing_item = next(
                item for item in patched_row["activity"] if item["type"] == "routing"
            )
            self.assertIn("Routed to", routing_item["text"])
            other_types = [item["type"] for item in by_id[other_task.id]["activity"]]
            self.assertNotIn("patch", other_types)

    def test_renderer_js_neutralizes_xss_and_preserves_digits(self) -> None:
        """Execute the actual client-side renderer under node: script tags and
        javascript: links must come out inert, and ordinary digits must survive
        the code-span stash (regression: bare numeric placeholders once turned
        every number in prose into `undefined`)."""
        import shutil

        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        from puppetmaster.dashboard import RENDERER_JS

        harness = RENDERER_JS + r"""
const assert = require("assert");
assert.ok(md("We found 3 issues across 12 files").includes("3 issues across 12 files"));
assert.ok(md("Run `pytest` — 537 passed").includes("537 passed"));
assert.ok(md("Run `pytest` — 537 passed").includes("<code>pytest</code>"));
const xss = md("<script>alert(1)</script> [x](javascript:alert(2))");
assert.ok(!xss.includes("<script>"));
assert.ok(!xss.includes("javascript:"));
assert.ok(xss.includes("&lt;script&gt;"));
// A forged sentinel in artifact text must not dereference the stash.
assert.ok(!md("\uE000 0 \uE000").includes("undefined"));
// Loose lists (blank lines between items) stay one list, so numbering
// continues instead of every item restarting at 1.
assert.ok(md("1. first\n\n2. second").includes("<ol><li>first</li><li>second</li></ol>"));
console.log("renderer-ok");
"""
        completed = subprocess.run(
            [node, "-e", harness], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("renderer-ok", completed.stdout)


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
        # Thin starter: drop the seeded plan-billed frontier so discovery tests
        # still exercise the no-frontier first-run path.
        thin = [s for s in starter_registry() if s.id != "cursor/fable-5"]
        save_registry(thin, path)
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

    def test_fallback_skips_adapter_with_missing_cli(self) -> None:
        """Regression (Rishi): a stale auth file keeps claude-code/codex reading
        billing-healthy after the CLI is uninstalled. Fallback must NOT cascade
        into a binary that isn't there — it must leave the task FAILED instead."""
        from unittest.mock import patch

        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore

        registry = [
            ModelSpec(
                id="claude-code/opus-4-8",
                adapter="claude-code",
                adapter_model_name="claude-opus-4-8",
                capability_score=99,
                input_per_mtok_usd=5,
                output_per_mtok_usd=25,
                billing="unknown",
            ),
        ]

        def _healthy(adapter, **kw):
            # Billing reads healthy off the stale ~/.claude.json — the trap.
            return BillingStatus(
                adapter=adapter, billing="plan", healthy=True, detail="oauth", evidence=[]
            )

        def _make(tmp):
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("fix the bug")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the fix",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={"auto_route": True, "model": "x", "router_model_id": "cursor/x"},
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="w",
                    payload={"check": "x", "result": "blocked", "failure": "billing_or_quota", "adapter": "cursor"},
                    confidence=0.5,
                    evidence=["adapter:cursor"],
                )
            )
            return store, job, task

        # CLI uninstalled -> no reroute, task stays FAILED (no doomed cascade).
        with TemporaryDirectory() as tmp:
            store, job, task = _make(tmp)
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_healthy), \
                 patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True), \
                 patch("puppetmaster.preflight.adapter_cli_present", return_value=False):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 0)
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.FAILED)

        # Same setup, CLI installed -> the fallback proceeds normally.
        with TemporaryDirectory() as tmp:
            store, job, task = _make(tmp)
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_healthy), \
                 patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True), \
                 patch("puppetmaster.preflight.adapter_cli_present", return_value=True):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 1)
            self.assertEqual(store.get_task_by_id(task.id).adapter, "claude-code")

    def test_model_unavailable_on_fable_5_reroutes_to_opus_4_8(self) -> None:
        from unittest.mock import patch

        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore

        registry = [
            ModelSpec(
                id="claude-code/fable-5",
                adapter="claude-code",
                adapter_model_name="claude-fable-5",
                capability_score=100,
                input_per_mtok_usd=10.0,
                output_per_mtok_usd=50.0,
                billing="unknown",
            ),
            ModelSpec(
                id="claude-code/opus-4-8",
                adapter="claude-code",
                adapter_model_name="claude-opus-4-8",
                capability_score=99,
                input_per_mtok_usd=5.0,
                output_per_mtok_usd=25.0,
                billing="unknown",
            ),
            ModelSpec(
                id="cursor/opus-4-8",
                adapter="cursor",
                adapter_model_name="claude-opus-4-8",
                capability_score=99,
                billing="plan",
                tags=["cursor"],
            ),
        ]

        def _billing(adapter, **kw):
            if adapter == "cursor":
                return BillingStatus(adapter="cursor", billing="plan", healthy=True, detail="ok", evidence=[])
            return BillingStatus(adapter=adapter, billing="unknown", healthy=False, detail="no", evidence=[])

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("frontier task")
            task = Task(
                job_id=job.id,
                role="audit",
                instruction="security audit across every module",
                adapter="claude-code",
                status=TaskStatus.FAILED,
                payload={
                    "auto_route": True,
                    "model": "claude-fable-5",
                    "router_model_id": "claude-code/fable-5",
                },
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="w",
                    payload={
                        "check": "x",
                        "result": "blocked",
                        "failure": "model_unavailable",
                        "adapter": "claude-code",
                    },
                    confidence=0.5,
                    evidence=["adapter:claude-code"],
                )
            )
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_billing):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.adapter, "cursor")
            self.assertEqual(updated.payload["model"], "claude-opus-4-8")
            self.assertEqual(updated.payload["fallback_from_adapter"], "claude-code")
            routing = [
                a for a in store.list_artifacts(job.id)
                if a.type == ArtifactType.ROUTING and a.created_by == "router-fallback"
            ]
            self.assertEqual(len(routing), 1)
            self.assertEqual(routing[0].payload["fallback_reason"], "model_unavailable")
            self.assertEqual(routing[0].payload["model_id"], "cursor/opus-4-8")

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


class CommandProbeResolutionTests(unittest.TestCase):
    """`_resolve_probe_command` must launch Windows .cmd shims correctly.

    Regression: ``doctor`` probed bare ``npm``, which on Windows raised
    FileNotFoundError (WinError 2) because subprocess(shell=False) can't run a
    .cmd shim — surfacing a misleading ``error`` row even when npm was installed.
    """

    def test_resolves_posix_executable_to_full_path(self) -> None:
        from unittest.mock import patch

        from puppetmaster.diagnostics import _resolve_probe_command

        with patch("puppetmaster.diagnostics.os.name", "posix"), patch(
            "puppetmaster.diagnostics.shutil.which", return_value="/usr/bin/npm"
        ):
            self.assertEqual(
                _resolve_probe_command(["npm", "--version"]),
                ["/usr/bin/npm", "--version"],
            )

    def test_windows_cmd_shim_routed_through_command_processor(self) -> None:
        from unittest.mock import patch

        from puppetmaster.diagnostics import _resolve_probe_command

        cmd_path = r"C:\Program Files\nodejs\npm.cmd"
        with patch("puppetmaster.diagnostics.os.name", "nt"), patch(
            "puppetmaster.diagnostics.shutil.which", return_value=cmd_path
        ), patch.dict("os.environ", {"COMSPEC": r"C:\Windows\System32\cmd.exe"}):
            self.assertEqual(
                _resolve_probe_command(["npm", "--version"]),
                [r"C:\Windows\System32\cmd.exe", "/c", cmd_path, "--version"],
            )

    def test_returns_none_when_executable_absent(self) -> None:
        from unittest.mock import patch

        from puppetmaster.diagnostics import _resolve_probe_command

        with patch("puppetmaster.diagnostics.shutil.which", return_value=None):
            self.assertIsNone(_resolve_probe_command(["npm", "--version"]))


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
        thin = [s for s in starter_registry() if s.id != "cursor/fable-5"]
        save_registry(thin, path)
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


class PlatformLockTests(unittest.TestCase):
    """The platform lock restricts which adapters Puppetmaster may use,
    enforced at routing, auto-discovery, and auto-fallback."""

    def _path(self, tmp: str) -> Path:
        return Path(tmp) / "models.json"

    def test_default_is_unrestricted(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pl.ONLY_ENV, None)
            p = self._path(tmp)
            self.assertEqual(pl.enabled_adapters(p), set(pl.KNOWN_ADAPTERS))
            self.assertFalse(pl.is_restricted(p))
            self.assertIsNone(pl.active_allowlist(p))

    def test_only_enable_disable_reset_roundtrip(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pl.ONLY_ENV, None)
            p = self._path(tmp)
            pl.set_enabled({"cursor"}, p)
            self.assertEqual(pl.enabled_adapters(p), {"cursor"})
            self.assertTrue(pl.is_restricted(p))
            self.assertEqual(pl.active_allowlist(p), frozenset({"cursor"}))

            pl.enable({"codex"}, p)
            self.assertEqual(pl.enabled_adapters(p), {"cursor", "codex"})

            pl.disable({"cursor"}, p)
            self.assertEqual(pl.enabled_adapters(p), {"codex"})

            pl.reset(p)
            self.assertEqual(pl.enabled_adapters(p), set(pl.KNOWN_ADAPTERS))
            self.assertFalse(pl.is_restricted(p))

    def test_env_override_wins_over_file(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp:
            p = self._path(tmp)
            pl.set_enabled({"codex"}, p)  # file says codex only
            with patch.dict(os.environ, {pl.ONLY_ENV: "cursor,openai"}):
                self.assertEqual(pl.enabled_adapters(p), {"cursor", "openai"})
                self.assertEqual(pl.active_allowlist(p), frozenset({"cursor", "openai"}))

    def test_unknown_adapter_never_blocked(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pl.ONLY_ENV, None)
            p = self._path(tmp)
            pl.set_enabled({"cursor"}, p)
            # shell / internal adapters are not platform-billed → always allowed.
            self.assertTrue(pl.is_adapter_enabled("shell", p))
            self.assertFalse(pl.is_adapter_enabled("codex", p))
            self.assertTrue(pl.is_adapter_enabled("cursor", p))

    def test_route_task_rejects_disabled_adapter(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.router import TaskSignals, route_task

        reg = starter_registry()
        sig = TaskSignals(
            instruction="security audit of auth",
            role="audit",
            allowed_adapters=frozenset({"cursor"}),
        )
        decision = route_task(sig, reg, policy="balanced")
        self.assertEqual(decision.model.adapter, "cursor")

    def test_route_task_raises_when_no_enabled_adapter_in_registry(self) -> None:
        from puppetmaster.model_registry import starter_registry
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        reg = starter_registry()
        sig = TaskSignals(instruction="x", allowed_adapters=frozenset({"nonexistent"}))
        with self.assertRaises(NoEligibleModelError):
            route_task(sig, reg)

    def test_signals_from_worker_spec_inherits_lock(self) -> None:
        from puppetmaster import platform_lock as pl
        from puppetmaster.router import signals_from_worker_spec
        from puppetmaster.workers import WorkerSpec

        with patch.dict(os.environ, {pl.ONLY_ENV: "cursor"}):
            spec = WorkerSpec(role="audit", instruction="audit", payload={"auto_route": True})
            sig = signals_from_worker_spec(spec)
            self.assertEqual(sig.allowed_adapters, frozenset({"cursor"}))

    def test_signals_payload_override_wins(self) -> None:
        from puppetmaster import platform_lock as pl
        from puppetmaster.router import signals_from_worker_spec
        from puppetmaster.workers import WorkerSpec

        with patch.dict(os.environ, {pl.ONLY_ENV: "cursor"}):
            spec = WorkerSpec(
                role="audit",
                instruction="audit",
                payload={"allowed_adapters": ["codex", "openai"]},
            )
            sig = signals_from_worker_spec(spec)
            self.assertEqual(sig.allowed_adapters, frozenset({"codex", "openai"}))

    def test_cli_platform_only_and_status_json(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pl.ONLY_ENV, None)
            p = self._path(tmp)
            rc = cli_main(["platform", "only", "cursor", "--registry-path", str(p)])
            self.assertEqual(rc, 0)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    ["platform", "status", "--registry-path", str(p), "--json"]
                )
            self.assertEqual(rc, 0)
            data = json.loads(buf.getvalue())
            self.assertEqual(data["enabled"], ["cursor"])
            self.assertTrue(data["restricted"])
            self.assertIn("claude-code", data["disabled"])

    def test_cli_platform_rejects_unknown_adapter(self) -> None:
        with TemporaryDirectory() as tmp:
            p = self._path(tmp)
            rc = cli_main(["platform", "only", "bogus", "--registry-path", str(p)])
            self.assertEqual(rc, 1)

    def test_fallback_skips_disabled_adapter(self) -> None:
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pl.ONLY_ENV, None)
            p = self._path(tmp)
            pl.set_enabled({"cursor"}, p)
            # When everything but cursor is disabled, a cursor failure has no
            # enabled platform to fall back onto.
            self.assertFalse(pl.is_adapter_enabled("codex", p))
            self.assertFalse(pl.is_adapter_enabled("openai", p))
            self.assertTrue(pl.is_adapter_enabled("cursor", p))


class AutoEscalationTests(unittest.TestCase):
    """Confidence-based mid-run escalation: a COMPLETE task whose verification
    confidence is below threshold gets re-dispatched one capability tier up."""

    def _setup(self, tmp, *, confidence, model_id, payload_extra=None):
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        store = SwarmStore(Path(tmp) / ".puppetmaster")
        job = store.create_job("do the work")
        payload = {"auto_route": True, "router_model_id": model_id, "model": "x"}
        payload.update(payload_extra or {})
        task = Task(
            job_id=job.id, role="implement", instruction="implement the thing",
            adapter="cursor", status=TaskStatus.COMPLETE, payload=payload,
        )
        store.save_task(task)
        store.save_artifact(Artifact(
            job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
            created_by="w", payload={"check": "self", "result": "done"},
            confidence=confidence, evidence=["adapter:cursor"],
        ))
        return store, job, task

    def _registry(self):
        from puppetmaster.model_registry import ModelSpec

        return [
            ModelSpec(id="cursor/composer-2-5", adapter="cursor", adapter_model_name="composer-2.5", capability_score=55, billing="plan", tags=["cursor"]),
            ModelSpec(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5", capability_score=90, billing="plan", tags=["cursor"]),
        ]

    def _plan_billing(self, adapter, **kw):
        from puppetmaster.platform_billing import BillingStatus

        return BillingStatus(adapter=adapter, billing="plan", healthy=True, detail="ok", evidence=[])

    def _run_reroute(self, store, job):
        from unittest.mock import patch

        from puppetmaster.orchestrator import Orchestrator

        orch = Orchestrator(store)
        with patch("puppetmaster.model_registry.load_registry", return_value=self._registry()), \
             patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=self._plan_billing), \
             patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True):
            return orch._reroute_low_confidence(job)

    def test_escalates_low_confidence_to_stronger_model(self) -> None:
        from puppetmaster.models import TaskStatus

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(
                tmp, confidence=0.5, model_id="cursor/composer-2-5",
                payload_extra={"min_confidence": 0.8},
            )
            rerouted = self._run_reroute(store, job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.payload["router_model_id"], "cursor/gpt-5-5")
            self.assertEqual(updated.payload["escalation_attempts"], 1)
            self.assertEqual(updated.payload["escalated_from_model"], "cursor/composer-2-5")
            # A router-escalation ROUTING artifact records the why.
            arts = [a for a in store.list_artifacts(job.id) if a.created_by == "router-escalation"]
            self.assertEqual(len(arts), 1)
            self.assertEqual(arts[0].payload["escalated_from_model"], "cursor/composer-2-5")

    def test_no_escalation_when_confidence_meets_threshold(self) -> None:
        from puppetmaster.models import TaskStatus

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(
                tmp, confidence=0.9, model_id="cursor/composer-2-5",
                payload_extra={"min_confidence": 0.8},
            )
            self.assertEqual(self._run_reroute(store, job), 0)
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)

    def test_no_escalation_when_already_top_tier(self) -> None:
        with TemporaryDirectory() as tmp:
            # Current model is already the strongest in the registry.
            store, job, task = self._setup(
                tmp, confidence=0.4, model_id="cursor/gpt-5-5",
                payload_extra={"min_confidence": 0.9},
            )
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_disabled_by_default(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PUPPETMASTER_ESCALATE_CONFIDENCE", None)
            # Low confidence but no threshold configured anywhere -> no-op.
            store, job, task = self._setup(
                tmp, confidence=0.2, model_id="cursor/composer-2-5",
            )
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_env_threshold_enables_escalation(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"PUPPETMASTER_ESCALATE_CONFIDENCE": "0.7"}
        ):
            store, job, task = self._setup(
                tmp, confidence=0.5, model_id="cursor/composer-2-5",
            )
            self.assertEqual(self._run_reroute(store, job), 1)

    def test_bounded_by_max_attempts(self) -> None:
        from puppetmaster.orchestrator import _MAX_ESCALATION_ATTEMPTS

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(
                tmp, confidence=0.3, model_id="cursor/composer-2-5",
                payload_extra={"min_confidence": 0.9, "escalation_attempts": _MAX_ESCALATION_ATTEMPTS},
            )
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_escalation_threshold_parsing(self) -> None:
        from puppetmaster.models import Task
        from puppetmaster.orchestrator import Orchestrator

        mk = lambda p: Task(job_id="j", role="implement", instruction="i", payload=p)
        self.assertEqual(Orchestrator._escalation_threshold(mk({"min_confidence": 0.6})), 0.6)
        self.assertIsNone(Orchestrator._escalation_threshold(mk({})))
        self.assertIsNone(Orchestrator._escalation_threshold(mk({"min_confidence": "nope"})))
        self.assertIsNone(Orchestrator._escalation_threshold(mk({"min_confidence": 1.5})))
        with patch.dict(os.environ, {"PUPPETMASTER_ESCALATE_CONFIDENCE": "0.55"}):
            self.assertEqual(Orchestrator._escalation_threshold(mk({})), 0.55)

    def test_latest_verification_confidence(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("g")
            orch = Orchestrator(store)
            self.assertIsNone(orch._latest_verification_confidence(job, "task_x"))
            store.save_artifact(Artifact(
                job_id=job.id, task_id="task_x", type=ArtifactType.VERIFICATION,
                created_by="w", payload={"check": "a", "result": "r"}, confidence=0.4,
                evidence=["e"], created_at="2026-01-01T00:00:00Z",
            ))
            store.save_artifact(Artifact(
                job_id=job.id, task_id="task_x", type=ArtifactType.VERIFICATION,
                created_by="w", payload={"check": "b", "result": "r"}, confidence=0.85,
                evidence=["e"], created_at="2026-01-02T00:00:00Z",
            ))
            self.assertEqual(orch._latest_verification_confidence(job, "task_x"), 0.85)


class RoutingAuditTests(unittest.TestCase):
    """The read-only self-audit recommender: aggregation + conservative
    suggestions + the store collector."""

    def _rec(self, model_id, *, conf=None, needed=40, cost=0.001,
             escalated=False, escalated_from=None, fell_back=False,
             adapter="cursor"):
        from puppetmaster.audit import TaskAuditRecord

        return TaskAuditRecord(
            model_id=model_id, adapter=adapter, capability_needed=needed,
            est_cost_usd=cost, confidence=conf, escalated=escalated,
            escalated_from=escalated_from, fell_back=fell_back,
        )

    def test_under_delivering_model_gets_lower_score_suggested(self) -> None:
        from puppetmaster.audit import build_audit_report

        # weak/55 is the initial pick on 6 tasks; 4 escalate away to strong/80.
        records = []
        for _ in range(4):
            records.append(self._rec("strong/80", conf=0.95, escalated=True,
                                      escalated_from="weak/55"))
        for _ in range(2):
            records.append(self._rec("weak/55", conf=0.5))
        report = build_audit_report(
            records, {"weak/55": 55, "strong/80": 80}, min_sample=5
        )
        weak = next(m for m in report.models if m.model_id == "weak/55")
        self.assertEqual(weak.selections, 6)  # 2 retained + 4 escalated away
        self.assertAlmostEqual(weak.escalated_away_rate, 4 / 6, places=2)
        self.assertIn("under-provisioned", weak.flags)
        self.assertIsNotNone(weak.suggested_score)
        self.assertLess(weak.suggested_score, 55)
        sug = report.suggestions
        self.assertEqual(len(sug), 1)
        self.assertEqual(sug[0]["model_id"], "weak/55")

    def test_well_calibrated_model_gets_no_suggestion(self) -> None:
        from puppetmaster.audit import build_audit_report

        records = [self._rec("ok/60", conf=0.8, needed=55) for _ in range(8)]
        report = build_audit_report(records, {"ok/60": 60})
        ok = next(m for m in report.models if m.model_id == "ok/60")
        self.assertEqual(ok.flags, [])
        self.assertIsNone(ok.suggested_score)
        self.assertEqual(report.suggestions, [])

    def test_small_sample_is_not_acted_on(self) -> None:
        from puppetmaster.audit import build_audit_report

        # 100% escalated away, but only 2 picks — below MIN_SAMPLE.
        records = [
            self._rec("strong/80", conf=0.9, escalated=True, escalated_from="weak/55")
            for _ in range(2)
        ]
        report = build_audit_report(records, {"weak/55": 55, "strong/80": 80})
        weak = next(m for m in report.models if m.model_id == "weak/55")
        self.assertEqual(weak.escalated_away, 2)
        self.assertIsNone(weak.suggested_score)
        self.assertEqual(report.suggestions, [])

    def test_over_used_is_flagged_but_not_auto_adjusted(self) -> None:
        from puppetmaster.audit import build_audit_report

        # strong/90 confidently doing low-need (need 20) work.
        records = [self._rec("strong/90", conf=0.95, needed=20) for _ in range(8)]
        report = build_audit_report(records, {"strong/90": 90})
        strong = next(m for m in report.models if m.model_id == "strong/90")
        self.assertIn("possibly-over-used", strong.flags)
        self.assertIsNone(strong.suggested_score)  # no counterfactual -> no number
        self.assertEqual(report.suggestions, [])

    def _rec_with_tokens(self, model_id, *, est_in, est_out, act_in, act_out,
                         measured, cost=0.001):
        from puppetmaster.audit import TaskAuditRecord

        return TaskAuditRecord(
            model_id=model_id, adapter="cursor", capability_needed=40,
            est_cost_usd=cost, confidence=0.9, escalated=False,
            escalated_from=None, fell_back=False,
            est_tokens_in=est_in, est_tokens_out=est_out,
            actual_tokens_in=act_in, actual_tokens_out=act_out,
            actual_tokens_measured=measured,
        )

    def test_token_drift_reconciled_per_model_and_jobwide(self) -> None:
        from puppetmaster.audit import build_audit_report

        # Estimated 1000 tokens/task, actually burned 1500 — router under-estimated.
        records = [
            self._rec_with_tokens("m/60", est_in=600, est_out=400,
                                  act_in=900, act_out=600, measured=True)
            for _ in range(4)
        ]
        report = build_audit_report(records, {"m/60": 60})
        m = next(x for x in report.models if x.model_id == "m/60")
        self.assertEqual(m.runs_with_actuals, 4)
        self.assertEqual(m.measured_runs, 4)
        self.assertEqual(m.est_tokens, 4000)
        self.assertEqual(m.actual_tokens, 6000)
        self.assertAlmostEqual(m.token_drift_ratio, 1.5, places=3)
        # Job-wide rollup mirrors the per-model figures.
        self.assertEqual(report.tasks_with_actuals, 4)
        self.assertAlmostEqual(report.token_drift_ratio, 1.5, places=3)

    def test_cost_drift_uses_injected_price_fn(self) -> None:
        from puppetmaster.audit import build_audit_report

        records = [
            self._rec_with_tokens("metered/70", est_in=1000, est_out=0,
                                  act_in=2000, act_out=0, measured=True, cost=0.001),
            # A task with no actuals must not anchor the cost-drift denominator,
            # even though its est_cost still counts toward headline est spend.
            self._rec("metered/70", conf=0.9, cost=0.005),
        ]
        # $1/Mtok input -> 2000 tokens actual = $0.002 actual vs $0.001 estimated.
        report = build_audit_report(
            records, {"metered/70": 70},
            actual_cost_fn=lambda mid, tin, tout: (tin / 1_000_000.0) * 1.0,
        )
        m = next(x for x in report.models if x.model_id == "metered/70")
        self.assertAlmostEqual(m.actual_spend_usd, 0.002, places=6)
        self.assertAlmostEqual(m.cost_drift_ratio, 2.0, places=3)
        self.assertAlmostEqual(report.total_actual_spend_usd, 0.002, places=6)
        # Denominator is the reconciled task's est ($0.001), NOT all-tasks est.
        self.assertAlmostEqual(report.total_est_spend_reconciled_usd, 0.001, places=6)
        self.assertAlmostEqual(report.cost_drift_ratio, 2.0, places=3)

    def test_records_without_actuals_excluded_from_drift(self) -> None:
        from puppetmaster.audit import build_audit_report

        # No usage reported (actual_tokens_measured stays None) -> not reconciled,
        # and drift is left unknown rather than faked as zero.
        records = [self._rec("m/60", conf=0.9) for _ in range(3)]
        report = build_audit_report(records, {"m/60": 60})
        m = next(x for x in report.models if x.model_id == "m/60")
        self.assertEqual(m.runs_with_actuals, 0)
        self.assertIsNone(m.token_drift_ratio)
        self.assertEqual(report.tasks_with_actuals, 0)
        self.assertIsNone(report.token_drift_ratio)

    def test_approximated_actuals_counted_but_flagged_unmeasured(self) -> None:
        from puppetmaster.audit import build_audit_report

        records = [
            self._rec_with_tokens("m/60", est_in=500, est_out=500,
                                  act_in=400, act_out=400, measured=False)
        ]
        report = build_audit_report(records, {"m/60": 60})
        m = next(x for x in report.models if x.model_id == "m/60")
        self.assertEqual(m.runs_with_actuals, 1)
        self.assertEqual(m.measured_runs, 0)  # char/4 approximation, not measured
        self.assertAlmostEqual(m.token_drift_ratio, 0.8, places=3)

    def test_collect_records_captures_token_usage(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("recon goal")
            task = Task(
                job_id=job.id, role="implement", instruction="do it",
                adapter="cursor", status=TaskStatus.COMPLETE,
                payload={
                    "router_model_id": "m/60",
                    "router_capability_needed": 60,
                    "router_estimated_cost_usd": 0.001,
                },
            )
            store.save_task(task)
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router",
                payload={"model_id": "m/60", "adapter": "cursor",
                         "policy": "balanced", "capability_needed": 60,
                         "estimated_cost_usd": 0.001,
                         "estimated_tokens_in": 800, "estimated_tokens_out": 200},
                confidence=0.9, evidence=["role:implement"],
            ))
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                created_by="w",
                payload={"check": "x", "result": "passed",
                         "tokens_in": 1200, "tokens_out": 300,
                         "tokens_estimated": False},
                confidence=0.92, evidence=["e"],
            ))

            records, jobs = collect_records(store)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r.est_tokens_total, 1000)
            self.assertEqual(r.actual_tokens_total, 1500)
            self.assertTrue(r.actual_tokens_measured)
            self.assertTrue(r.has_actuals)

            report = build_audit_report(records, {"m/60": 60})
            self.assertEqual(report.total_est_tokens, 1000)
            self.assertEqual(report.total_actual_tokens, 1500)
            self.assertAlmostEqual(report.token_drift_ratio, 1.5, places=3)

    def test_collect_records_from_store(self) -> None:
        from puppetmaster.audit import build_audit_report, collect_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("audit goal")
            task = Task(
                job_id=job.id, role="implement", instruction="do it",
                adapter="cursor", status=TaskStatus.COMPLETE,
                payload={
                    "router_model_id": "strong/80",
                    "router_capability_needed": 70,
                    "router_estimated_cost_usd": 0.01,
                },
            )
            store.save_task(task)
            # initial pick of weak/55, escalation to strong/80, final verification.
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router",
                payload={"model_id": "weak/55", "adapter": "cursor",
                         "policy": "balanced", "capability_needed": 50,
                         "estimated_cost_usd": 0.002},
                confidence=0.9, evidence=["role:implement"],
            ))
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router-escalation",
                payload={"model_id": "strong/80", "adapter": "cursor",
                         "policy": "escalating", "escalated_from_model": "weak/55",
                         "escalated_from_confidence": 0.4, "confidence_threshold": 0.7},
                confidence=0.9, evidence=["escalate"],
            ))
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                created_by="w", payload={"check": "x", "result": "passed"},
                confidence=0.92, evidence=["e"],
            ))

            records, jobs = collect_records(store)
            self.assertEqual(jobs, 1)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r.model_id, "strong/80")  # final model
            self.assertTrue(r.escalated)
            self.assertEqual(r.escalated_from, "weak/55")
            self.assertEqual(r.confidence, 0.92)

            report = build_audit_report(records, {"weak/55": 55, "strong/80": 80})
            weak = next(m for m in report.models if m.model_id == "weak/55")
            self.assertEqual(weak.escalated_away, 1)

    def test_cli_apply_writes_lowered_score(self) -> None:
        import argparse

        from puppetmaster.cli import _run_audit_command
        from puppetmaster.model_registry import (
            ModelSpec,
            load_registry,
            save_registry,
        )
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(
                [
                    ModelSpec(id="weak/55", adapter="cursor",
                              adapter_model_name="weak", capability_score=55),
                    ModelSpec(id="strong/80", adapter="cursor",
                              adapter_model_name="strong", capability_score=80),
                ],
                registry_path,
            )
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("g")
            # 6 tasks initially picked weak/55; 4 escalate to strong/80.
            for i in range(6):
                escalated = i < 4
                final = "strong/80" if escalated else "weak/55"
                task = Task(
                    job_id=job.id, role="implement", instruction="x",
                    adapter="cursor", status=TaskStatus.COMPLETE,
                    payload={"router_model_id": final,
                             "router_capability_needed": 50,
                             "router_estimated_cost_usd": 0.001},
                )
                store.save_task(task)
                store.save_artifact(Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={"model_id": "weak/55", "adapter": "cursor",
                             "policy": "balanced", "capability_needed": 50},
                    confidence=0.9, evidence=["r"],
                ))
                if escalated:
                    store.save_artifact(Artifact(
                        job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                        created_by="router-escalation",
                        payload={"model_id": "strong/80", "adapter": "cursor",
                                 "policy": "escalating",
                                 "escalated_from_model": "weak/55"},
                        confidence=0.9, evidence=["e"],
                    ))
                store.save_artifact(Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                    created_by="w", payload={"check": "c", "result": "passed"},
                    confidence=0.5 if not escalated else 0.95, evidence=["v"],
                ))

            args = argparse.Namespace(
                registry_path=str(registry_path), window=None,
                apply=True, json=False,
            )
            rc = _run_audit_command(args, store)
            self.assertEqual(rc, 0)
            after = {s.id: s.capability_score for s in load_registry(registry_path)}
            self.assertLess(after["weak/55"], 55)  # lowered
            self.assertEqual(after["strong/80"], 80)  # untouched


class CodegraphUsageTests(unittest.TestCase):
    """The local, numbers-only codegraph usage log + its aggregation."""

    def _with_log(self, tmp):
        import os
        os.environ["PUPPETMASTER_CODEGRAPH_USAGE_LOG"] = str(Path(tmp) / "usage.jsonl")
        os.environ.pop("PUPPETMASTER_CODEGRAPH_USAGE", None)

    def tearDown(self) -> None:
        import os
        os.environ.pop("PUPPETMASTER_CODEGRAPH_USAGE_LOG", None)
        os.environ.pop("PUPPETMASTER_CODEGRAPH_USAGE", None)

    def test_record_is_numbers_only_and_aggregates(self) -> None:
        from puppetmaster import codegraph_usage as cu

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            cu.record_query(command="context", cwd="/repo", result_chars=4000,
                            latency_ms=120.0, ok=True, caller="mcp", query_chars=30)
            cu.record_query(command="search", cwd="/repo", result_chars=800,
                            latency_ms=40.0, ok=True, caller="swarm", query_chars=12)
            recs = cu.load_usage()
            self.assertEqual(len(recs), 2)
            # cwd is a privacy-preserving hash, not the raw path.
            import hashlib
            expected = hashlib.sha256(str(Path("/repo").resolve()).encode("utf-8")).hexdigest()[:12]
            self.assertEqual(recs[0]["cwd"], expected)
            # numbers-only: never stores the query text itself.
            self.assertNotIn("query", recs[0])
            self.assertEqual(recs[0]["context_tokens"], 1000)  # 4000 // 4
            agg = cu.aggregate(recs, exploration_baseline_tokens=8000,
                               input_price_per_mtok=1.0)
            self.assertEqual(agg["queries"], 2)
            self.assertEqual(agg["context_tokens_fed"], 1200)  # 1000 + 200
            self.assertEqual(agg["avoided_exploration_tokens_est"], 16000)
            self.assertEqual(agg["net_tokens_saved_est"], 14800)

    def test_non_exploration_commands_are_ignored(self) -> None:
        from puppetmaster import codegraph_usage as cu

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            cu.record_query(command="status", cwd="/r", result_chars=10,
                            latency_ms=5.0, ok=True)
            cu.record_query(command="init", cwd="/r", result_chars=10,
                            latency_ms=5.0, ok=True)
            self.assertEqual(cu.load_usage(), [])

    def test_disabled_writes_nothing(self) -> None:
        import os
        from puppetmaster import codegraph_usage as cu

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            os.environ["PUPPETMASTER_CODEGRAPH_USAGE"] = "0"
            cu.record_query(command="context", cwd="/r", result_chars=4000,
                            latency_ms=1.0, ok=True)
            self.assertEqual(cu.load_usage(), [])


class ReadsLogTests(unittest.TestCase):
    """The $0 follow-up reads counter — user-facing result reads only."""

    def _with_log(self, tmp):
        import os
        os.environ["PUPPETMASTER_READS_LOG"] = str(Path(tmp) / "reads.jsonl")
        os.environ.pop("PUPPETMASTER_READS_USAGE", None)

    def tearDown(self) -> None:
        import os
        os.environ.pop("PUPPETMASTER_READS_LOG", None)
        os.environ.pop("PUPPETMASTER_READS_USAGE", None)

    def test_records_result_reads_and_aggregates(self) -> None:
        from puppetmaster import reads_log as rl

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            rl.record_read("show", caller="cli")
            rl.record_read("artifacts", caller="mcp")
            rl.record_read("partial_summary", caller="cli")
            agg = rl.aggregate(rl.load_reads())
            self.assertEqual(agg["reads"], 3)
            self.assertEqual(agg["by_kind"]["show"], 1)
            # numbers-only: no job content stored.
            self.assertNotIn("job_id", rl.load_reads()[0])

    def test_operational_reads_are_ignored(self) -> None:
        from puppetmaster import reads_log as rl

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            rl.record_read("status", caller="cli")   # operational, not a result read
            rl.record_read("savings", caller="cli")  # must never self-count
            self.assertEqual(rl.load_reads(), [])

    def test_disabled_writes_nothing(self) -> None:
        import os
        from puppetmaster import reads_log as rl

        with TemporaryDirectory() as tmp:
            self._with_log(tmp)
            os.environ["PUPPETMASTER_READS_USAGE"] = "0"
            rl.record_read("show", caller="cli")
            self.assertEqual(rl.load_reads(), [])


class EnsureCursorSdkTests(unittest.TestCase):
    """Bootstrap @cursor/sdk for pip/pipx installs (wheels can't ship node_modules)."""

    def test_unchanged_when_sdk_already_resolvable(self) -> None:
        from puppetmaster.installers import ensure_cursor_sdk

        with patch(
            "puppetmaster.diagnostics._find_cursor_sdk_install",
            return_value=Path("/fake/site-packages/node_modules/@cursor/sdk"),
        ):
            result = ensure_cursor_sdk(Path("/tmp"))
        self.assertEqual(result.status, "unchanged")
        self.assertIn("@cursor/sdk", result.detail)

    def test_skipped_when_npm_missing(self) -> None:
        from puppetmaster.installers import ensure_cursor_sdk

        with patch("puppetmaster.diagnostics._find_cursor_sdk_install", return_value=None), \
                patch("puppetmaster.installers.shutil.which", return_value=None):
            result = ensure_cursor_sdk(Path("/tmp"))
        self.assertEqual(result.status, "skipped")
        self.assertIn("npm not on PATH", result.detail)

    def test_installs_via_npm_into_package_root(self) -> None:
        from types import SimpleNamespace

        from puppetmaster.installers import ensure_cursor_sdk

        sdk_path = Path("/fake/site-packages/node_modules/@cursor/sdk")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="added 13 packages", stderr="")

        with patch(
            "puppetmaster.diagnostics._find_cursor_sdk_install",
            side_effect=[None, sdk_path],
        ), patch("puppetmaster.installers.subprocess.run", side_effect=fake_run):
            result = ensure_cursor_sdk(
                Path("/tmp"),
                package_root=Path("/fake/site-packages"),
                npm_executable="/usr/local/bin/npm",
            )
        self.assertEqual(result.status, "installed")
        self.assertEqual(result.location, str(sdk_path))
        self.assertEqual(
            calls[0],
            ["/usr/local/bin/npm", "install", "@cursor/sdk", "--prefix", "/fake/site-packages"],
        )

    def test_error_when_npm_fails(self) -> None:
        from types import SimpleNamespace

        from puppetmaster.installers import ensure_cursor_sdk

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="npm ERR! network timeout")

        with patch("puppetmaster.diagnostics._find_cursor_sdk_install", return_value=None), \
                patch("puppetmaster.installers.subprocess.run", side_effect=fake_run):
            result = ensure_cursor_sdk(Path("/tmp"), npm_executable="/usr/local/bin/npm")
        self.assertEqual(result.status, "error")
        self.assertIn("npm ERR! network timeout", result.detail)

    def test_billing_detail_no_longer_overclaims_sdk_auth(self) -> None:
        """Step 1 used to print 'Cursor SDK authenticated' while only checking
        an env var — the same run's step 2 then said 'not detected'."""
        from puppetmaster.platform_billing import detect_cursor_billing

        status = detect_cursor_billing(env={"CURSOR_API_KEY": "key_x"})
        self.assertTrue(status.healthy)
        self.assertNotIn("SDK authenticated", status.detail)
        self.assertIn("CURSOR_API_KEY is set", status.detail)


class SetupPlatformStepTests(unittest.TestCase):
    """The `setup` wizard's platform-lock step (non-interactive paths)."""

    def setUp(self) -> None:
        import os
        from puppetmaster import platform_lock as pl

        self._env_before = {
            "PUPPETMASTER_MODELS_PATH": os.environ.get("PUPPETMASTER_MODELS_PATH"),
            pl.ONLY_ENV: os.environ.get(pl.ONLY_ENV),
        }

    def _args(self, **kw):
        from types import SimpleNamespace
        base = {"platforms": None, "skip_platforms": False}
        base.update(kw)
        return SimpleNamespace(**base)

    def _isolated(self, tmp):
        import os
        from puppetmaster import platform_lock as pl
        os.environ["PUPPETMASTER_MODELS_PATH"] = str(Path(tmp) / "models.json")
        os.environ.pop(pl.ONLY_ENV, None)

    def tearDown(self) -> None:
        import os

        for key, value in self._env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_explicit_platforms_sets_lock(self) -> None:
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl
        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            rc = _setup_platform_step(self._args(platforms="cursor"))
            self.assertEqual(rc, 0)
            self.assertEqual(pl.enabled_adapters(), {"cursor"})
            self.assertTrue(pl.is_restricted())

    def test_unknown_platform_returns_error_and_writes_nothing(self) -> None:
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl
        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            rc = _setup_platform_step(self._args(platforms="banana"))
            self.assertEqual(rc, 1)
            self.assertFalse(pl.is_restricted())

    def test_skip_leaves_lock_unchanged(self) -> None:
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl
        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            rc = _setup_platform_step(self._args(skip_platforms=True))
            self.assertEqual(rc, 0)
            self.assertFalse(pl.is_restricted())

    def test_cursor_detected_when_npm_can_bootstrap_missing_sdk(self) -> None:
        """Field report: a pipx-installed Cursor user read '(not detected on
        this machine)' because wheels can't ship node_modules — npm
        availability must count, since install-cursor-mcp bootstraps the SDK."""
        from puppetmaster.cli import _detected_platforms

        with patch("puppetmaster.diagnostics._cursor_sdk_installed", return_value=False), \
                patch("shutil.which", return_value="/usr/local/bin/npm"), \
                patch.dict(os.environ, {"CURSOR_API_KEY": "key_x"}):
            detected = _detected_platforms(Path("/tmp"))
        self.assertTrue(detected["cursor"])

    def test_cursor_not_detected_without_sdk_or_npm(self) -> None:
        from puppetmaster.cli import _detected_platforms

        with patch("puppetmaster.diagnostics._cursor_sdk_installed", return_value=False), \
                patch("shutil.which", return_value=None), \
                patch.dict(os.environ, {"CURSOR_API_KEY": "key_x"}):
            detected = _detected_platforms(Path("/tmp"))
        self.assertFalse(detected["cursor"])

    def test_first_run_noninteractive_locks_to_detected_platforms(self) -> None:
        """Field report: a Claude-Code-only user ended up with cursor enabled
        because setup defaulted to all-on instead of detecting."""
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            detected = {"cursor": False, "claude-code": True, "codex": True, "openai": False}
            with patch("puppetmaster.cli._detected_platforms", return_value=detected):
                rc = _setup_platform_step(self._args())
            self.assertEqual(rc, 0)
            self.assertEqual(pl.enabled_adapters(), {"claude-code", "codex"})

    def test_noninteractive_respects_existing_lock(self) -> None:
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            pl.set_enabled({"cursor"})
            detected = {"cursor": False, "claude-code": True, "codex": False, "openai": False}
            with patch("puppetmaster.cli._detected_platforms", return_value=detected):
                rc = _setup_platform_step(self._args())
            self.assertEqual(rc, 0)
            self.assertEqual(pl.enabled_adapters(), {"cursor"})

    def test_nothing_detected_leaves_default_unrestricted(self) -> None:
        from puppetmaster.cli import _setup_platform_step
        from puppetmaster import platform_lock as pl

        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            detected = {a: False for a in pl.KNOWN_ADAPTERS}
            with patch("puppetmaster.cli._detected_platforms", return_value=detected):
                rc = _setup_platform_step(self._args())
            self.assertEqual(rc, 0)
            self.assertFalse(pl.is_restricted())

    def test_state_display_flags_undetected_platforms(self) -> None:
        from puppetmaster.cli import _setup_platform_step

        with TemporaryDirectory() as tmp:
            self._isolated(tmp)
            detected = {"cursor": False, "claude-code": True, "codex": True, "openai": False}
            buf = io.StringIO()
            with patch("puppetmaster.cli._detected_platforms", return_value=detected):
                with contextlib.redirect_stdout(buf):
                    rc = _setup_platform_step(self._args(platforms="cursor"))
            self.assertEqual(rc, 0)
            output = buf.getvalue()
            self.assertIn("not detected on this machine", output)
            self.assertIn("enabled anyway (explicit --platforms)", output)


class RoutingBaselineSnapshotTests(unittest.TestCase):
    """The router stamps a decision-time savings baseline onto every decision."""

    def _registry(self):
        from puppetmaster.model_registry import ModelSpec
        return [
            ModelSpec(id="cheap/40", adapter="cursor", adapter_model_name="c",
                      capability_score=40, input_per_mtok_usd=0.1, output_per_mtok_usd=0.2),
            ModelSpec(id="frontier/95", adapter="cursor", adapter_model_name="f",
                      capability_score=95, input_per_mtok_usd=5.0, output_per_mtok_usd=25.0),
        ]

    def test_decision_records_frontier_baseline(self) -> None:
        from puppetmaster.router import route_task, TaskSignals

        d = route_task(
            TaskSignals(role="explore", instruction="tiny lookup"),
            self._registry(), policy="balanced",
        )
        p = d.to_artifact_payload()
        self.assertEqual(p["baseline_model_id"], "frontier/95")
        self.assertGreater(p["baseline_cost_usd"], 0.0)
        # cheap pick should cost no more than the frontier baseline.
        self.assertLessEqual(p["estimated_cost_usd"], p["baseline_cost_usd"])

    def test_baseline_respects_required_tags(self) -> None:
        # Regression: the baseline must come from the SAME eligible set the pick
        # is drawn from. A required-tag constraint that excludes the strongest
        # model must also exclude it from the baseline, or savings are bogus.
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import route_task, TaskSignals

        registry = [
            ModelSpec(id="api/cheap", adapter="openai", adapter_model_name="c",
                      capability_score=50, input_per_mtok_usd=0.15,
                      output_per_mtok_usd=0.9, billing="api", tags=["openai", "cheap"]),
            ModelSpec(id="api/frontier", adapter="openai", adapter_model_name="f",
                      capability_score=90, input_per_mtok_usd=5.0,
                      output_per_mtok_usd=30.0, billing="api", tags=["openai"]),
            # Strongest overall, but a DIFFERENT platform/tag — must not leak in.
            ModelSpec(id="plan/top", adapter="claude-code", adapter_model_name="t",
                      capability_score=99, input_per_mtok_usd=0.0,
                      output_per_mtok_usd=0.0, billing="plan", tags=["plan"]),
        ]
        d = route_task(
            TaskSignals(role="explore", instruction="tiny lookup",
                        required_tags=["openai"]),
            registry, policy="balanced",
        )
        p = d.to_artifact_payload()
        # baseline is the strongest *openai* model, NOT the plan/top giant.
        self.assertEqual(p["baseline_model_id"], "api/frontier")
        self.assertGreater(p["baseline_cost_usd"], 0.0)

    def test_baseline_respects_billing_gate_no_overclaim(self) -> None:
        # Regression: with API billing forbidden, the baseline can't be an
        # API model the run could never have used — that would overclaim savings.
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.router import route_task, TaskSignals

        registry = [
            ModelSpec(id="plan/only", adapter="claude-code", adapter_model_name="p",
                      capability_score=60, input_per_mtok_usd=0.0,
                      output_per_mtok_usd=0.0, billing="plan", tags=["plan"]),
            ModelSpec(id="api/giant", adapter="openai", adapter_model_name="g",
                      capability_score=99, input_per_mtok_usd=5.0,
                      output_per_mtok_usd=30.0, billing="api", tags=["openai"]),
        ]
        d = route_task(
            TaskSignals(role="explore", instruction="tiny lookup",
                        allow_api_billing=False),
            registry, policy="balanced",
        )
        p = d.to_artifact_payload()
        # baseline must be the plan model, and savings must be zero (no overclaim).
        self.assertEqual(p["baseline_model_id"], "plan/only")
        self.assertEqual(p["baseline_cost_usd"], 0.0)


class SavingsLedgerTests(unittest.TestCase):
    """Policy-aware savings aggregation with the two honesty rules."""

    def _rec(self, policy, chosen, baseline, has_baseline=True,
             picked="cheap/40", baseline_id="frontier/95"):
        from puppetmaster.savings import RoutingRecord
        return RoutingRecord(policy=policy, chosen_cost_usd=chosen,
                             baseline_cost_usd=baseline, has_baseline=has_baseline,
                             picked_model_id=picked, baseline_model_id=baseline_id)

    def test_only_cost_optimizing_policies_count_as_savings(self) -> None:
        from puppetmaster.savings import summarize_routing

        recs = [
            self._rec("balanced", 0.0, 0.04),       # plan win
            self._rec("balanced", 0.01, 0.04),      # cheaper API
            self._rec("quality", 0.16, 0.16),       # deliberate spend, NOT savings
        ]
        s = summarize_routing(recs)
        self.assertEqual(s.cost_optimizing_tasks, 2)
        self.assertAlmostEqual(s.saved_usd, 0.07, places=4)   # 0.04 + 0.03
        self.assertEqual(s.deliberate_tasks, 1)
        self.assertAlmostEqual(s.deliberate_spend_usd, 0.16, places=4)
        self.assertEqual(s.plan_routed_tasks, 1)

    def test_tasks_without_baseline_are_excluded_from_dollars(self) -> None:
        from puppetmaster.savings import summarize_routing

        recs = [
            self._rec("balanced", 0.01, 0.0, has_baseline=False),  # pre-snapshot
            self._rec("balanced", 0.0, 0.04, has_baseline=True),
        ]
        s = summarize_routing(recs)
        self.assertEqual(s.tasks_without_baseline, 1)
        self.assertEqual(s.cost_optimizing_tasks, 1)
        self.assertAlmostEqual(s.saved_usd, 0.04, places=4)

    def test_build_metrics_rates_and_none_for_empty_denominator(self) -> None:
        from puppetmaster.savings import build_metrics, SelfHeal

        recs = [
            # right-sized: ran something other than the strongest (even at $0).
            self._rec("balanced", 0.0, 0.0, has_baseline=False,
                      picked="cheap/40", baseline_id="frontier/95"),
            # not right-sized: ran the strongest model itself.
            self._rec("balanced", 0.05, 0.05, has_baseline=True,
                      picked="frontier/95", baseline_id="frontier/95"),
            self._rec("quality", 0.10, 0.10, has_baseline=True),   # excluded (policy)
        ]
        heal = SelfHeal(fallbacks=1, escalations=2)
        m = build_metrics(
            recs, heal,
            codegraph={"context_tokens_fed": 8000},
            reads={"reads": 6},
            jobs=2,
        )
        # 1 of 2 cost-optimizing tasks ran below the strongest model (by identity,
        # independent of dollars — works for plan-billed $0 models).
        self.assertAlmostEqual(m["capability_match_rate"], 0.5, places=3)
        self.assertAlmostEqual(m["escalation_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(m["fallback_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(m["reuse_reads_per_job"], 3.0, places=3)
        self.assertAlmostEqual(m["context_tokens_per_job"], 4000.0, places=1)
        self.assertEqual(m["sample"]["cost_optimizing_judgeable"], 2)

        empty = build_metrics([], SelfHeal(), {"context_tokens_fed": 0}, {"reads": 0}, 0)
        self.assertIsNone(empty["capability_match_rate"])
        self.assertIsNone(empty["escalation_rate"])
        self.assertIsNone(empty["reuse_reads_per_job"])
        self.assertIsNone(empty["context_tokens_per_job"])

    def test_collect_counts_routing_and_self_heal(self) -> None:
        from puppetmaster.savings import collect_routing_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("g")
            task = Task(job_id=job.id, role="implement", instruction="x",
                        adapter="cursor", status=TaskStatus.COMPLETE,
                        payload={"router_model_id": "cheap/40"})
            store.save_task(task)
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router",
                payload={"model_id": "cheap/40", "adapter": "cursor",
                         "policy": "balanced", "estimated_cost_usd": 0.0,
                         "baseline_cost_usd": 0.05},
                confidence=0.9, evidence=["r"]))
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router-fallback",
                payload={"model_id": "cheap/40", "adapter": "cursor",
                         "policy": "balanced"},
                confidence=0.9, evidence=["f"]))
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                created_by="router-escalation",
                payload={"model_id": "frontier/95", "adapter": "cursor",
                         "policy": "escalating"},
                confidence=0.9, evidence=["e"]))

            records, jobs, heal = collect_routing_records([store])
            self.assertEqual(jobs, 1)
            self.assertEqual(len(records), 1)  # only the 'router' artifact
            self.assertTrue(records[0].has_baseline)
            self.assertEqual(heal.fallbacks, 1)
            self.assertEqual(heal.escalations, 1)

    def test_counterfactual_prices_tokens_against_reference(self) -> None:
        from puppetmaster.savings import (
            RoutingRecord,
            compute_counterfactual,
            resolve_counterfactual_model,
        )
        from puppetmaster.model_registry import ModelSpec

        registry = [
            ModelSpec(id="plan/top", adapter="cursor", adapter_model_name="t",
                      capability_score=90, input_per_mtok_usd=0.0,
                      output_per_mtok_usd=0.0, billing="plan"),
            ModelSpec(id="api/frontier", adapter="openai", adapter_model_name="f",
                      capability_score=88, input_per_mtok_usd=5.0,
                      output_per_mtok_usd=30.0, billing="api"),
        ]
        # Default reference = highest-capability *priced* model, not the $0 giant.
        ref = resolve_counterfactual_model(registry)
        self.assertEqual(ref.id, "api/frontier")

        recs = [
            RoutingRecord(policy="balanced", chosen_cost_usd=0.0,
                          baseline_cost_usd=0.0, has_baseline=False,
                          tokens_in=1_000_000, tokens_out=1_000_000),
        ]
        cf = compute_counterfactual(recs, ref)
        self.assertTrue(cf.reference_priced)
        # 1M in @ $5 + 1M out @ $30 = $35 naive; actual $0 (plan) -> avoided $35.
        self.assertAlmostEqual(cf.naive_cost_usd, 35.0, places=4)
        self.assertAlmostEqual(cf.actual_cost_usd, 0.0, places=4)
        self.assertAlmostEqual(cf.avoided_usd, 35.0, places=4)
        self.assertEqual(cf.tasks, 1)

    def test_counterfactual_unpriced_reference_is_zero(self) -> None:
        from puppetmaster.savings import RoutingRecord, compute_counterfactual
        from puppetmaster.model_registry import ModelSpec

        plan_only = ModelSpec(id="plan/only", adapter="cursor", adapter_model_name="p",
                              capability_score=80, input_per_mtok_usd=0.0,
                              output_per_mtok_usd=0.0, billing="plan")
        recs = [RoutingRecord(policy="balanced", chosen_cost_usd=0.0,
                              baseline_cost_usd=0.0, has_baseline=False,
                              tokens_in=500, tokens_out=500)]
        cf = compute_counterfactual(recs, plan_only)
        self.assertFalse(cf.reference_priced)
        self.assertEqual(cf.naive_cost_usd, 0.0)
        self.assertEqual(cf.avoided_usd, 0.0)

    def test_counterfactual_env_override_selects_model(self) -> None:
        import os
        from puppetmaster.savings import resolve_counterfactual_model, COUNTERFACTUAL_MODEL_ENV
        from puppetmaster.model_registry import ModelSpec

        registry = [
            ModelSpec(id="api/cheap", adapter="openai", adapter_model_name="c",
                      capability_score=50, input_per_mtok_usd=0.1,
                      output_per_mtok_usd=0.2, billing="api"),
            ModelSpec(id="api/frontier", adapter="openai", adapter_model_name="f",
                      capability_score=99, input_per_mtok_usd=5.0,
                      output_per_mtok_usd=30.0, billing="api"),
        ]
        os.environ[COUNTERFACTUAL_MODEL_ENV] = "api/cheap"
        try:
            ref = resolve_counterfactual_model(registry)
            self.assertEqual(ref.id, "api/cheap")
        finally:
            os.environ.pop(COUNTERFACTUAL_MODEL_ENV, None)

    def test_collect_dedups_router_artifacts_by_task_id(self) -> None:
        from puppetmaster.savings import collect_routing_records
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("g")
            task = Task(job_id=job.id, role="implement", instruction="x",
                        adapter="cursor", status=TaskStatus.COMPLETE)
            store.save_task(task)
            # Two 'router' artifacts for the SAME task (e.g. a re-dispatch) must
            # be counted once, not twice — no inflated savings.
            for _ in range(2):
                store.save_artifact(Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={"model_id": "cheap/40", "adapter": "cursor",
                             "policy": "balanced", "estimated_cost_usd": 0.0,
                             "baseline_cost_usd": 0.05},
                    confidence=0.9, evidence=["r"]))
            records, _, _ = collect_routing_records([store])
            self.assertEqual(len(records), 1)


class PuppetmasterFrictionFixTests(unittest.TestCase):
    """Coverage for the JAD-migration friction-log fixes (stalled reaper, mode
    banner, show fallback, finalize, wait, codegraph flag hoist, routing flags)."""

    def _store(self, tmp: str):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    # --- #1: swarm mode classification ---------------------------------
    def test_swarm_mode_analysis_vs_edit(self) -> None:
        from puppetmaster.workers import DEFAULT_WORKERS, spec_edits_files, swarm_mode

        # The default analysis swarm (all local) is read-only despite the
        # "implement" role name.
        self.assertEqual(swarm_mode(DEFAULT_WORKERS), "analysis")
        self.assertFalse(any(spec_edits_files(s) for s in DEFAULT_WORKERS))

        edit_spec = WorkerSpec(
            role="cursor", instruction="x", adapter="cursor",
            payload={"mode": "implement"},
        )
        self.assertTrue(spec_edits_files(edit_spec))
        self.assertEqual(swarm_mode([edit_spec]), "edit")
        self.assertTrue(
            spec_edits_files(WorkerSpec(role="c", instruction="x", adapter="claude-code"))
        )
        self.assertFalse(
            spec_edits_files(
                WorkerSpec(
                    role="audit",
                    instruction="x",
                    adapter="claude-code",
                    payload={"read_only": True},
                )
            )
        )
        self.assertFalse(
            spec_edits_files(
                WorkerSpec(
                    role="audit",
                    instruction="x",
                    adapter="codex",
                    payload={"no_edit": True},
                )
            )
        )
        self.assertFalse(
            spec_edits_files(
                WorkerSpec(
                    role="dry-run",
                    instruction="x",
                    adapter="codex",
                    payload={"dry_run": True},
                )
            )
        )
        self.assertFalse(
            spec_edits_files(
                WorkerSpec(
                    role="dry-run-implement",
                    instruction="x",
                    adapter="codex",
                    payload={"mode": "implement", "dry_run": True},
                )
            )
        )
        self.assertTrue(
            spec_edits_files(
                WorkerSpec(
                    role="impl",
                    instruction="x",
                    adapter="cursor",
                    payload={"mode": "implement", "review": True},
                )
            )
        )

    # --- edit verb (Tier 2: lightweight single in-place edit) -----------
    def test_build_edit_payload_defaults_are_cheap_inplace(self) -> None:
        from puppetmaster.workers import build_edit_payload

        p = build_edit_payload(
            instruction="fix the bug", cwd="/repo", adapter="hermes"
        )
        # In-place: dirty tree allowed, no isolated worktree guard.
        self.assertTrue(p["allow_dirty"])
        self.assertTrue(p["allow_non_worktree"])
        # Actually edits (→ swarm_mode == edit).
        self.assertEqual(p["mode"], "implement")
        # Cheap auto-routing by default, pinned to the chosen adapter only.
        self.assertTrue(p["auto_route"])
        self.assertEqual(p["routing_policy"], "cheap")
        self.assertEqual(p["allowed_adapters"], ["hermes"])
        # CodeGraph stays on (no disable flag).
        self.assertNotIn("disable_codegraph", p)

    def test_build_edit_payload_pinned_model_disables_routing(self) -> None:
        from puppetmaster.workers import build_edit_payload

        p = build_edit_payload(
            instruction="x", cwd="/repo", adapter="hermes", model="gpt-5-nano"
        )
        # An explicit model pin wins — never auto-route around it.
        self.assertEqual(p["model"], "gpt-5-nano")
        self.assertNotIn("auto_route", p)
        self.assertNotIn("routing_policy", p)

    def test_build_edit_payload_can_opt_out_of_routing_and_codegraph(self) -> None:
        from puppetmaster.workers import build_edit_payload

        p = build_edit_payload(
            instruction="x", cwd="/repo", adapter="codex",
            auto_route=False, disable_codegraph=True,
        )
        self.assertNotIn("auto_route", p)
        self.assertTrue(p["disable_codegraph"])

    def test_build_edit_spec_is_edit_mode(self) -> None:
        from puppetmaster.workers import build_edit_spec, spec_edits_files, swarm_mode

        spec = build_edit_spec(
            instruction="fix it", adapter="hermes", cwd="/repo"
        )
        self.assertEqual(spec.adapter, "hermes")
        self.assertTrue(spec_edits_files(spec))
        self.assertEqual(swarm_mode([spec]), "edit")

    def test_pick_implement_adapter_priority_and_lock(self) -> None:
        from puppetmaster.workers import (
            NoImplementAdapterError,
            pick_implement_adapter,
        )

        # Priority order honored when several enabled.
        self.assertEqual(
            pick_implement_adapter({"hermes", "cursor", "codex"}), "cursor"
        )
        # Falls through to the only enabled one.
        self.assertEqual(pick_implement_adapter({"hermes"}), "hermes")
        # Explicit request honored when enabled.
        self.assertEqual(
            pick_implement_adapter({"hermes", "codex"}, "codex"), "codex"
        )
        # Requested-but-disabled raises with context.
        with self.assertRaises(NoImplementAdapterError) as ctx:
            pick_implement_adapter({"hermes"}, "cursor")
        self.assertEqual(ctx.exception.requested, "cursor")
        # Non-implement adapter raises.
        with self.assertRaises(NoImplementAdapterError):
            pick_implement_adapter({"openai"}, "openai")
        # Nothing enabled raises.
        with self.assertRaises(NoImplementAdapterError):
            pick_implement_adapter(set())

    def test_edit_verb_e2e_through_orchestrator(self) -> None:
        """End-to-end: an edit spec runs through the real Orchestrator with a
        stub edit-capable adapter that makes a real on-disk change, completes,
        classifies the run as ``edit``, satisfies the auto-attached require_diff
        gate, and persists the worker's PATCH artifact. Network-free (the adapter
        is faked) — asserts the wiring spec → run → gate → edit mode → artifact,
        not the LLM (the live LLM path is verified separately via the CLI E2E).
        """
        import subprocess as _sp

        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.workers import build_edit_spec

        with TemporaryDirectory() as tmp:
            # A real git repo so the require_diff gate (auto-attached for
            # mode=implement) can observe an actual change.
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "x.py").write_text("bug\n", encoding="utf-8")
            for cmd in (
                ["git", "init", "-q"],
                ["git", "config", "user.email", "t@t.co"],
                ["git", "config", "user.name", "t"],
                ["git", "add", "-A"],
                ["git", "commit", "-qm", "init"],
            ):
                _sp.run(cmd, cwd=repo, check=True, capture_output=True)

            store = self._store(tmp)
            spec = build_edit_spec(
                instruction="fix the bug", adapter="hermes", cwd=str(repo),
                auto_route=False,  # wiring test: don't depend on a model registry
            )

            class _FakeEditAdapter:
                def run(self, task, goal, worker_id):
                    # Make the real edit the require_diff gate expects.
                    (repo / "x.py").write_text("fixed\n", encoding="utf-8")
                    return [
                        Artifact(
                            job_id=task.job_id,
                            task_id=task.id,
                            type=ArtifactType.PATCH,
                            created_by=worker_id,
                            payload={
                                "change": "rewrite x.py: bug -> fixed",
                                "diff": "--- a/x.py\n+++ b/x.py\n@@\n-bug\n+fixed\n",
                                "files": ["x.py"],
                                "worker_diff_present": True,
                            },
                            confidence=0.9,
                            evidence=["adapter:hermes"],
                        )
                    ]

            with patch(
                "puppetmaster.workers.get_adapter", return_value=_FakeEditAdapter()
            ):
                result = Orchestrator(store).run(
                    "fix the bug", specs=[spec], lease_seconds=5,
                    worker_mode="inline",
                )

            self.assertEqual(result.mode, "edit")
            self.assertEqual(result.job.status, JobStatus.COMPLETE)
            self.assertTrue(
                any(a.type == ArtifactType.PATCH for a in result.artifacts),
                "edit run should persist a PATCH artifact",
            )
            # The require_diff gate is auto-attached for mode=implement and must
            # have observed the real change.
            self.assertEqual((repo / "x.py").read_text(encoding="utf-8"), "fixed\n")

    def test_mcp_edit_command_builder_maps_args(self) -> None:
        from puppetmaster import mcp_server

        cmd = mcp_server.edit_command(
            {
                "instruction": "fix the bug",
                "cwd": "/repo",
                "adapter": "hermes",
                "routing_policy": "cheap",
                "disable_codegraph": True,
            }
        )
        self.assertEqual(cmd[0], "edit")
        self.assertEqual(cmd[1], "fix the bug")
        self.assertIn("--adapter", cmd)
        self.assertIn("hermes", cmd)
        self.assertIn("--routing-policy", cmd)
        self.assertIn("--disable-codegraph", cmd)

    def test_mcp_edit_command_no_auto_route_flag(self) -> None:
        from puppetmaster import mcp_server

        cmd = mcp_server.edit_command(
            {"instruction": "x", "cwd": "/r", "auto_route": False}
        )
        self.assertIn("--no-auto-route", cmd)

    def test_mcp_run_edit_rejects_disabled_adapter(self) -> None:
        from puppetmaster import mcp_server

        with patch(
            "puppetmaster.platform_lock.enabled_adapters", return_value={"hermes"}
        ):
            result = mcp_server.run_edit(
                {"instruction": "x", "cwd": ".", "adapter": "cursor"}
            )
        self.assertTrue(result.get("isError"))

    def test_mcp_edit_tool_registered(self) -> None:
        from puppetmaster import mcp_server

        resp = mcp_server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("puppetmaster_edit", names)

    # --- #2 / #4: stalled-job reaper -----------------------------------
    def test_reaper_marks_dead_orchestrator_job_stalled(self) -> None:
        from puppetmaster.liveness import reap_stalled_jobs

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            # Record a heartbeat for a pid that is certainly not alive.
            store.write_json(
                store.job_dir(job.id) / "orchestrator.json",
                {
                    "pid": 999999999,
                    "host": __import__("socket").gethostname(),
                    "started_at": seconds_from_now(-30),
                    "heartbeat_at": seconds_from_now(-30),
                },
            )
            reaped = reap_stalled_jobs(store)
            self.assertEqual(len(reaped), 1)
            self.assertEqual(reaped[0]["reason"], "orchestrator_pid_gone")
            self.assertEqual(store.get_job(job.id).status, JobStatus.STALLED)

    def test_reaper_leaves_live_lease_job_running(self) -> None:
        from puppetmaster.liveness import reap_stalled_jobs

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            task = Task(
                job_id=job.id, role="r", instruction="i",
                status=TaskStatus.RUNNING,
                lease_owner="w",
                lease_expires_at=seconds_from_now(120),
            )
            store.save_task(task)
            reaped = reap_stalled_jobs(store)
            self.assertEqual(reaped, [])
            self.assertEqual(store.get_job(job.id).status, JobStatus.RUNNING)

    def test_reaper_requeues_stale_tasks(self) -> None:
        from puppetmaster.liveness import reap_stalled_jobs

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            # Fresh heartbeat from THIS process => orchestrator looks alive, so
            # the job is not stalled, but the dead worker's task is requeued.
            store.write_json(
                store.job_dir(job.id) / "orchestrator.json",
                {
                    "pid": os.getpid(),
                    "host": __import__("socket").gethostname(),
                    "started_at": seconds_from_now(-5),
                    "heartbeat_at": seconds_from_now(-1),
                },
            )
            stale = Task(
                job_id=job.id, role="r", instruction="i",
                status=TaskStatus.RUNNING,
                lease_owner="dead-worker",
                lease_expires_at=seconds_from_now(-30),
            )
            store.save_task(stale)
            reap_stalled_jobs(store)
            self.assertEqual(store.get_job(job.id).status, JobStatus.RUNNING)
            self.assertEqual(
                store.get_task_by_id(stale.id).status, TaskStatus.QUEUED
            )

    # --- #3: show fallback + finalize ----------------------------------
    def test_show_falls_back_to_preview_when_not_finalized(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            rc = cli_main(
                ["--state-dir", str(store.root), "--backend", "sqlite", "show", job.id]
            )
            self.assertEqual(rc, 0)  # no crash, even with no stitched.md

    def test_finalize_stitches_and_completes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            rc = cli_main(
                ["--state-dir", str(store.root), "--backend", "sqlite", "finalize", job.id]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(store.get_job(job.id).status, JobStatus.COMPLETE)
            self.assertTrue(
                (store.job_dir(job.id) / "summaries" / "stitched.md").is_file()
            )

    def test_finalize_cli_run_reports_diff_source(self) -> None:
        from puppetmaster.cli import finalize_cli_run
        from puppetmaster.orchestrator import RunResult

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            artifact = Artifact(
                job_id=job.id,
                task_id="t1",
                type=ArtifactType.VERIFICATION,
                created_by="w1",
                confidence=0.9,
                evidence=["adapter:cursor"],
                payload={
                    "check": "run",
                    "result": "passed",
                    "baseline_diff_present": True,
                    "worker_diff_present": False,
                },
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                finalize_cli_run(
                    RunResult(
                        job=job,
                        artifacts=[artifact],
                        summary="",
                        summary_path=Path(tmp) / "summary.md",
                        mode="edit",
                    )
                )

        outcome = stderr.getvalue()
        self.assertIn("baseline_diff_present=True", outcome)
        self.assertIn("worker_diff_present=False", outcome)
        self.assertIn("patch_artifact_emitted=False", outcome)
        self.assertIn("commit_present=False", outcome)

    # --- #6: codegraph global-flag hoisting ----------------------------
    def test_hoist_global_codegraph_flags(self) -> None:
        from puppetmaster.cli import _hoist_global_codegraph_flags

        cwd, timeout, remaining = _hoist_global_codegraph_flags(
            ["init", "--cwd", "/repo", "--timeout", "30"]
        )
        self.assertEqual(cwd, "/repo")
        self.assertEqual(timeout, 30)
        self.assertEqual(remaining, ["init"])

        cwd, _, remaining = _hoist_global_codegraph_flags(["search", "--cwd=/r", "foo"])
        self.assertEqual(cwd, "/r")
        self.assertEqual(remaining, ["search", "foo"])

        # Flags after a literal `--` are forwarded to codegraph untouched.
        cwd, _, remaining = _hoist_global_codegraph_flags(["--", "--cwd", "x"])
        self.assertIsNone(cwd)
        self.assertEqual(remaining, ["--", "--cwd", "x"])

    # --- #7: direct-adapter routing flags ------------------------------
    def test_routing_payload_from_args(self) -> None:
        from argparse import Namespace

        from puppetmaster.cli import routing_payload_from_args

        off = Namespace(auto_route=False)
        self.assertEqual(routing_payload_from_args(off, adapter="cursor"), {})

        on = Namespace(
            auto_route=True, routing_policy="cheap", max_cost_usd=0.5, min_capability=70
        )
        payload = routing_payload_from_args(on, adapter="cursor")
        self.assertTrue(payload["auto_route"])
        self.assertEqual(payload["allowed_adapters"], ["cursor"])
        self.assertEqual(payload["routing_policy"], "cheap")
        self.assertEqual(payload["max_cost_usd"], 0.5)
        self.assertEqual(payload["min_capability"], 70)

    def test_direct_single_adapter_cli_sets_disable_memory_payload(self) -> None:
        captured = []

        def fake_run(self, goal, **kwargs):  # noqa: ANN001
            captured.append((goal, kwargs["specs"]))
            return object()

        with TemporaryDirectory() as tmp, patch(
            "puppetmaster.cli.Orchestrator.run", autospec=True, side_effect=fake_run
        ), patch("puppetmaster.cli.finalize_cli_run", return_value=0):
            state_dir = str(Path(tmp) / "state")
            cases = [
                ("claude-code", ["claude", "fresh claude", "--cwd", tmp, "--disable-memory"]),
                (
                    "openai",
                    ["openai", "fresh openai", "--cwd", tmp, "--disable-codegraph", "--disable-memory"],
                ),
                (
                    "codex",
                    ["codex", "fresh codex", "--cwd", tmp, "--disable-codegraph", "--disable-memory"],
                ),
            ]
            for adapter, command in cases:
                with self.subTest(adapter=adapter):
                    rc = cli_main(["--state-dir", state_dir, *command])
                    self.assertEqual(rc, 0)

        self.assertEqual([specs[0].adapter for _, specs in captured], ["claude-code", "openai", "codex"])
        self.assertEqual(
            [specs[0].payload.get("disable_memory") for _, specs in captured],
            [True, True, True],
        )

    def test_direct_codex_read_only_sandbox_sets_generic_no_edit_payload(self) -> None:
        captured = []

        def fake_run(self, goal, **kwargs):  # noqa: ANN001
            captured.append((goal, kwargs["specs"]))
            return object()

        with TemporaryDirectory() as tmp, patch(
            "puppetmaster.cli.Orchestrator.run", autospec=True, side_effect=fake_run
        ), patch("puppetmaster.cli.finalize_cli_run", return_value=0):
            cases = [
                ("read-only", False, True),
                ("workspace-write", False, False),
                ("read-only", True, False),
            ]
            for sandbox, bypass, expected_read_only in cases:
                with self.subTest(sandbox=sandbox, bypass=bypass):
                    command = [
                        "--state-dir",
                        str(Path(tmp) / "state"),
                        "codex",
                        f"{sandbox} review",
                        "--cwd",
                        tmp,
                        "--disable-codegraph",
                        "--sandbox",
                        sandbox,
                    ]
                    if bypass:
                        command.append("--dangerously-bypass-approvals-and-sandbox")
                    rc = cli_main(command)
                    self.assertEqual(rc, 0)

        for (_, specs), (_, _, expected_read_only) in zip(captured, cases):
            payload = specs[0].payload
            self.assertEqual(payload.get("read_only", False), expected_read_only)

    # --- #10: wait exit codes ------------------------------------------
    def test_wait_returns_nonzero_for_stalled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.STALLED)
            rc = cli_main(
                ["--state-dir", str(store.root), "--backend", "sqlite", "wait", job.id]
            )
            self.assertEqual(rc, 1)

    def test_wait_returns_zero_for_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.COMPLETE)
            rc = cli_main(
                ["--state-dir", str(store.root), "--backend", "sqlite", "wait", job.id]
            )
            self.assertEqual(rc, 0)

    def test_await_treats_stalled_as_terminal(self) -> None:
        from puppetmaster.cli import await_job_state

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.STALLED)
            state = await_job_state(store, job.id, timeout_seconds=1.0)
            self.assertTrue(state["terminal"])
            self.assertEqual(state["status"], "stalled")

    def test_await_returns_nonzero_for_stalled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.STALLED)
            rc = cli_main(
                ["--state-dir", str(store.root), "--backend", "sqlite", "await", job.id]
            )
            self.assertEqual(rc, 1)

    def test_mcp_await_marks_stalled_as_error(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.STALLED)
            result = call_tool(
                "puppetmaster_await_job",
                {
                    "cwd": tmp,
                    "state_dir": str(store.root),
                    "backend": "sqlite",
                    "job_id": job.id,
                    "timeout_seconds": 1,
                },
            )
            payload = json.loads(result["content"][0]["text"])
            self.assertTrue(result["isError"])
            self.assertEqual(payload["status"], "stalled")


class PuppetmasterLoudFailureTests(unittest.TestCase):
    """Tier-1 reliability: a run that did zero work must never look like success.

    Covers the blocked->FAILED veto (#1) and the built-in run-quality verdict
    (#10) that lets the parent stop eyeballing artifact composition.
    """

    def _blocked_artifact(self):
        from puppetmaster.adapters import verification_artifact
        from puppetmaster.models import Task

        task = Task(job_id="job_x", role="cursor", instruction="do the thing")
        return verification_artifact(
            task=task,
            worker_id="w1",
            adapter="cursor",
            check=task.instruction,
            result="blocked",
            confidence=0.8,
            evidence=["status:dirty-repo"],
            payload={"failure": "dirty_worktree"},
        )

    def _finding_artifact(self):
        from puppetmaster.models import Artifact, ArtifactType

        return Artifact(
            job_id="job_x",
            task_id="task_x",
            type=ArtifactType.FINDING,
            created_by="w1",
            confidence=0.9,
            evidence=["e"],
            payload={"finding": "something real", "detail": "x"},
        )

    def test_blocked_verdict_vetoes_complete(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        blocked = self._blocked_artifact()
        self.assertEqual(WorkerRuntime._blocked_verdict([blocked]), "dirty_worktree")
        self.assertIsNone(WorkerRuntime._blocked_verdict([self._finding_artifact()]))

    def test_quality_blocked_is_untrustworthy(self) -> None:
        from puppetmaster.quality import assess_run_quality

        verdict = assess_run_quality([self._blocked_artifact()])
        self.assertEqual(verdict["quality"], "blocked")
        self.assertFalse(verdict["trustworthy"])
        self.assertIn("dirty_worktree", verdict["blocking_failures"])

    def test_quality_empty_when_no_artifacts(self) -> None:
        from puppetmaster.quality import assess_run_quality

        self.assertEqual(assess_run_quality([])["quality"], "empty")

    def test_quality_degraded_when_only_verification(self) -> None:
        from puppetmaster.adapters import verification_artifact
        from puppetmaster.models import Task
        from puppetmaster.quality import assess_run_quality

        task = Task(job_id="job_x", role="cursor", instruction="x")
        passed = verification_artifact(
            task=task, worker_id="w1", adapter="cursor", check="x",
            result="passed", confidence=0.9, evidence=["e"], payload={},
        )
        self.assertEqual(assess_run_quality([passed])["quality"], "degraded")

    def test_quality_ok_with_substantive_artifact(self) -> None:
        from puppetmaster.quality import assess_run_quality

        verdict = assess_run_quality([self._finding_artifact()])
        self.assertEqual(verdict["quality"], "ok")
        self.assertTrue(verdict["trustworthy"])

    def test_warn_run_quality_treats_running_empty_job_as_in_progress(self) -> None:
        """A still-running job with no artifacts yet must not be warned as low-confidence."""
        from puppetmaster.cli import _warn_run_quality
        from puppetmaster.models import JobStatus

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("implement in flight")
            store.update_job_status(job.id, JobStatus.RUNNING)

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _warn_run_quality(store, job.id)
            message = err.getvalue()
            self.assertIn("in progress", message)
            self.assertNotIn("low-confidence", message)

    def test_warn_run_quality_warns_when_finished_job_is_empty(self) -> None:
        """A terminal job that produced nothing still gets the low-confidence warning."""
        from puppetmaster.cli import _warn_run_quality
        from puppetmaster.models import JobStatus

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("finished but empty")
            store.update_job_status(job.id, JobStatus.COMPLETE)

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _warn_run_quality(store, job.id)
            message = err.getvalue()
            self.assertIn("low-confidence", message)
            self.assertIn("empty", message)

    def test_worker_timeout_emits_blocked_artifact(self) -> None:
        """A killed-on-timeout worker records a blocked artifact so the run can't look done."""
        from puppetmaster.models import Task
        from puppetmaster.quality import assess_run_quality

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("long playwright run")
            task = Task(job_id=job.id, role="implement", instruction="e2e")
            store.save_task(task)

            Orchestrator(store)._emit_timeout_artifact(job, [task], timeout_seconds=600)

            artifacts = store.list_artifacts(job.id)
            self.assertTrue(artifacts)
            verdict = assess_run_quality(artifacts)
            self.assertEqual(verdict["quality"], "blocked")
            self.assertIn("worker_timeout", verdict["blocking_failures"])

    def test_worker_wait_extends_while_progressing(self) -> None:
        """A4: a worker still emitting events past base timeout is extended, not killed."""
        import subprocess as sp
        from puppetmaster.models import Task

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("long verify")
            task = Task(job_id=job.id, role="cursor", instruction="e2e")

            class ProgressingProc:
                """Times out a few times while emitting events, then exits clean."""
                def __init__(self) -> None:
                    self.calls = 0
                    self.returncode = 0

                def wait(self, timeout=None):
                    self.calls += 1
                    store.emit(job.id, "run.heartbeat", {"n": self.calls})
                    if self.calls <= 3:
                        raise sp.TimeoutExpired(cmd="x", timeout=timeout)
                    return 0

                def terminate(self):
                    raise AssertionError("must not kill a progressing worker")

            orch = Orchestrator(store)
            with patch.object(Orchestrator, "_worker_wait_timeout", staticmethod(lambda tasks: 0)), \
                 patch.object(Orchestrator, "_worker_hard_cap", staticmethod(lambda tasks, base: 9999)):
                orch._wait_for_worker(ProgressingProc(), job, [task])
            events = [e["event"] for e in store.read_events(job.id)]
            self.assertIn("worker.timeout_extended", events)
            self.assertNotIn("worker.timed_out", events)

    def test_worker_wait_kills_when_no_progress(self) -> None:
        """A4: a quiet (wedged) worker past base timeout is still killed loudly."""
        import subprocess as sp
        from puppetmaster.models import Task

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("wedged")
            task = Task(job_id=job.id, role="cursor", instruction="e2e")

            class WedgedProc:
                """Always times out and emits nothing — no demonstrable progress."""
                def __init__(self) -> None:
                    self.returncode = -15
                    self.killed = False

                def wait(self, timeout=None):
                    if self.killed:
                        return -15
                    raise sp.TimeoutExpired(cmd="x", timeout=timeout)

                def terminate(self):
                    self.killed = True

                def kill(self):
                    self.killed = True

            orch = Orchestrator(store)
            with patch.object(Orchestrator, "_worker_wait_timeout", staticmethod(lambda tasks: 0)), \
                 patch.object(Orchestrator, "_worker_hard_cap", staticmethod(lambda tasks, base: 0)):
                with self.assertRaises(RuntimeError):
                    orch._wait_for_worker(WedgedProc(), job, [task])
            events = [e["event"] for e in store.read_events(job.id)]
            self.assertIn("worker.timed_out", events)


class PuppetmasterGateTests(unittest.TestCase):
    """Non-bypassable completion gates (#2 commit, #11 drift ratchet)."""

    def _store(self, tmp: str):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    def _git_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for args in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "t@t.t"],
            ["git", "config", "user.name", "t"],
        ):
            subprocess.run(args, cwd=str(path), check=True, capture_output=True)
        (path / "seed.txt").write_text("seed\n")
        subprocess.run(["git", "add", "-A"], cwd=str(path), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "seed"], cwd=str(path), check=True, capture_output=True
        )

    def _task(self, **payload):
        from puppetmaster.models import Task

        return Task(job_id="job_g", role="cursor", instruction="x", payload=payload)

    def test_gate_specs_resolved_from_convenience_flags(self) -> None:
        from puppetmaster.gates import task_gate_specs

        task = self._task(require_diff=True, commit={"author": "a <a@a>"})
        kinds = {s["kind"] for s in task_gate_specs(task)}
        self.assertEqual(kinds, {"require_diff", "committed"})
        self.assertEqual(task_gate_specs(self._task()), [])

    def test_implement_mode_defaults_require_diff_invariant(self) -> None:
        """A3: implement tasks auto-require a diff so a no-op 'complete' fails loudly."""
        from puppetmaster.gates import task_gate_specs

        implement_kinds = {s["kind"] for s in task_gate_specs(self._task(mode="implement"))}
        self.assertIn("require_diff", implement_kinds)
        # Explicit opt-outs disable the auto-invariant for legitimate no-op tasks.
        self.assertEqual(task_gate_specs(self._task(mode="implement", require_diff=False)), [])
        self.assertEqual(task_gate_specs(self._task(mode="implement", allow_empty_diff=True)), [])
        # Non-implement tasks remain ungated by default.
        self.assertEqual(task_gate_specs(self._task(mode="review")), [])

    def test_implement_no_diff_fails_loudly(self) -> None:
        """A3 end-to-end: an implement task that changed nothing FAILS its gate."""
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            task = self._task(mode="implement", cwd=str(repo))
            result = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertFalse(result.passed)
            self.assertIn("require_diff", result.failed_reason)

    def test_require_diff_fails_on_empty_tree(self) -> None:
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            task = self._task(require_diff=True, cwd=str(repo))
            result = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertFalse(result.passed)
            self.assertIn("no diff", result.failed_reason)

    def test_require_diff_passes_with_changes(self) -> None:
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            (repo / "new.txt").write_text("change\n")
            task = self._task(require_diff=True, cwd=str(repo))
            result = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertTrue(result.passed)

    def test_ratchet_establishes_then_enforces(self) -> None:
        from puppetmaster.gates import evaluate_task_gates

        # Emit the metric JSON via an argv list (shell=False) so the oracle is
        # cross-platform — a shell `echo '...'` leaves the quotes intact under
        # cmd.exe and breaks the JSON parse on Windows.
        def oracle(value: int) -> list:
            return [
                sys.executable,
                "-c",
                f"import json;print(json.dumps({{'metrics': {{'m': {value}}}}}))",
            ]

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            task = self._task(gates=[{"kind": "ratchet", "command": oracle(5), "metric": "m"}], cwd=str(repo))
            # First run establishes baseline 5 and passes.
            self.assertTrue(evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo).passed)

            # A regression to 7 fails.
            bad = self._task(gates=[{"kind": "ratchet", "command": oracle(7), "metric": "m"}], cwd=str(repo))
            evald = evaluate_task_gates(bad, [], store, worker_id="w1", cwd=repo)
            self.assertFalse(evald.passed)
            self.assertIn("regressed", evald.failed_reason)

            # Shrinking to 3 passes and tightens the baseline.
            good = self._task(gates=[{"kind": "ratchet", "command": oracle(3), "metric": "m"}], cwd=str(repo))
            self.assertTrue(evaluate_task_gates(good, [], store, worker_id="w1", cwd=repo).passed)
            # Baseline now 3: returning to 5 must fail.
            again = self._task(gates=[{"kind": "ratchet", "command": oracle(5), "metric": "m"}], cwd=str(repo))
            self.assertFalse(evaluate_task_gates(again, [], store, worker_id="w1", cwd=repo).passed)

    def test_command_gate_exit_code(self) -> None:
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            ok = self._task(gates=[{"kind": "command", "command": "true"}], cwd=str(repo))
            self.assertTrue(evaluate_task_gates(ok, [], store, worker_id="w1", cwd=repo).passed)
            bad = self._task(gates=[{"kind": "command", "command": "false"}], cwd=str(repo))
            self.assertFalse(evaluate_task_gates(bad, [], store, worker_id="w1", cwd=repo).passed)

    def test_committed_gate_fails_when_dirty_then_autocommits(self) -> None:
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            (repo / "work.txt").write_text("uncommitted\n")
            # Without auto, a dirty tree fails the commit post-condition.
            strict = self._task(gates=[{"kind": "committed"}], cwd=str(repo))
            self.assertFalse(evaluate_task_gates(strict, [], store, worker_id="w1", cwd=repo).passed)
            # With auto, the runtime commits the work and the gate passes.
            auto = self._task(
                gates=[{"kind": "committed", "auto": True, "message": "feat: x"}], cwd=str(repo)
            )
            self.assertTrue(evaluate_task_gates(auto, [], store, worker_id="w1", cwd=repo).passed)

    def test_write_scope_gate_blocks_out_of_scope_writes(self) -> None:
        """B3/C1: a task that writes outside its declared scope FAILS the gate."""
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            (repo / "src").mkdir()
            (repo / "src" / "in_scope.py").write_text("ok\n")
            (repo / "stray.py").write_text("oops\n")

            task = self._task(write_scope=["src/**"], cwd=str(repo))
            result = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertFalse(result.passed)
            self.assertIn("outside declared scope", result.failed_reason)

            # Remove the stray write -> the gate passes.
            (repo / "stray.py").unlink()
            ok = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertTrue(ok.passed)

    def test_predict_write_conflicts_flags_overlapping_scopes(self) -> None:
        """B3/C1: overlapping declared scopes are predicted before dispatch."""
        from puppetmaster.conflicts import predict_write_conflicts, scopes_overlap

        self.assertTrue(scopes_overlap(["src/api/**"], ["src/api/routes.py"]))
        self.assertFalse(scopes_overlap(["src/api/**"], ["src/ui/**"]))

        conflicts = predict_write_conflicts([
            ("t1", ["src/api/**"]),
            ("t2", ["src/api/routes.py"]),
            ("t3", ["docs/**"]),
            ("t4", []),  # undeclared -> skipped
        ])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["tasks"], ["t1", "t2"])

    def test_affected_specs_rules_template_and_command(self) -> None:
        """B2: changed files map to specs via declarative rules and via a command."""
        from puppetmaster.affected import affected_specs

        mapping = {"rules": [
            {"match": "src/*", "specs": ["tests/{stem}_test.py"]},
            {"match": "src/api/*", "specs": ["tests/api_smoke.py"]},
        ]}
        specs = affected_specs(["src/foo.py", "src/api/routes.py"], mapping)
        self.assertIn("tests/foo_test.py", specs)
        self.assertIn("tests/routes_test.py", specs)
        self.assertIn("tests/api_smoke.py", specs)
        # Stable + de-duplicated, and an empty changed set yields nothing.
        self.assertEqual(len(specs), len(set(specs)))
        self.assertEqual(affected_specs([], mapping), [])
        # Command strategy: receives changed files on stdin, prints specs.
        self.assertEqual(affected_specs(["a.py", "b.py"], {"command": "cat"}), ["a.py", "b.py"])
        # A mapping with neither rules nor command is a usage error.
        with self.assertRaises(ValueError):
            affected_specs(["a.py"], {})

    def test_affected_cli_inline_rule(self) -> None:
        """B2 CLI: inline --rule shorthand resolves affected specs."""
        from puppetmaster.cli import _run_affected_command
        from types import SimpleNamespace

        args = SimpleNamespace(
            config=None, rule=["src/*=>tests/{stem}_test.py"],
            changed=["src/foo.py"], git_range=None, cwd=".", json=True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _run_affected_command(args)
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertIn("tests/foo_test.py", out["affected_specs"])

    def test_worktree_ports_are_deterministic_and_distinct(self) -> None:
        """B1: each worktree gets a stable, non-colliding port block."""
        from puppetmaster.ports import worktree_port_base, worktree_port_env, apply_worktree_ports

        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            base_a = worktree_port_base(a)
            base_b = worktree_port_base(b)
            # Deterministic: same path -> same base.
            self.assertEqual(base_a, worktree_port_base(a))
            # Distinct worktrees almost surely get distinct blocks.
            self.assertNotEqual(base_a, base_b)
            # Range stays below Linux's ephemeral floor (32768) for cross-OS safety.
            self.assertTrue(10000 <= base_a < 32768)

            env = worktree_port_env(a)
            self.assertEqual(env["PORT"], str(base_a))
            self.assertEqual(env["PUPPETMASTER_PORT_BASE"], str(base_a))
            self.assertEqual(env["PUPPETMASTER_PORT_0"], str(base_a))

            # apply_worktree_ports respects an already-pinned PORT unless overridden.
            pinned = {"PORT": "3000"}
            apply_worktree_ports(pinned, a)
            self.assertEqual(pinned["PORT"], "3000")
            self.assertEqual(pinned["PUPPETMASTER_PORT_BASE"], str(base_a))
            apply_worktree_ports(pinned, a, override_port=True)
            self.assertEqual(pinned["PORT"], str(base_a))

    def test_reserve_port_skips_busy_ports(self) -> None:
        """B1 bulletproof path: reserve_port bumps past a live listener (EADDRINUSE)."""
        import socket
        from puppetmaster.ports import reserve_port, worktree_port_base, _port_is_free

        with TemporaryDirectory() as wt:
            hint = worktree_port_base(wt)
            # Occupy the hinted port with a real listener.
            busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                busy.bind(("127.0.0.1", hint))
                busy.listen(1)
                reserved = reserve_port(wt)
                # Must not hand back the occupied port, and must be bindable.
                self.assertNotEqual(reserved, hint)
                self.assertTrue(_port_is_free(reserved))
            except OSError:
                self.skipTest("could not bind the hinted port in this environment")
            finally:
                busy.close()

    def test_committed_gate_excludes_generated_artifacts(self) -> None:
        """C2: configured generated artifacts are kept out of the auto-commit."""
        from puppetmaster.gates import evaluate_task_gates

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            (repo / "real.txt").write_text("real change\n")
            (repo / "parity-scoreboard.json").write_text("{\"generated\": true}\n")
            task = self._task(
                gates=[{
                    "kind": "committed", "auto": True, "message": "feat: real",
                    "exclude": ["parity-scoreboard.json"],
                }],
                cwd=str(repo),
            )
            evald = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            self.assertTrue(evald.passed)

            committed = subprocess.run(
                ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
                cwd=str(repo), capture_output=True, text=True, check=True,
            ).stdout
            self.assertIn("real.txt", committed)
            self.assertNotIn("parity-scoreboard.json", committed)
            # And it's gitignored so it won't resurface in future diffs.
            self.assertIn("parity-scoreboard.json", (repo / ".gitignore").read_text())

    def test_gate_failure_marks_run_untrustworthy(self) -> None:
        from puppetmaster.gates import evaluate_task_gates
        from puppetmaster.quality import assess_run_quality

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = Path(tmp) / "repo"
            self._git_repo(repo)
            task = self._task(require_diff=True, cwd=str(repo))
            evald = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)
            verdict = assess_run_quality(evald.artifacts)
            self.assertEqual(verdict["quality"], "blocked")
            self.assertFalse(verdict["trustworthy"])


class ReviewGateTests(unittest.TestCase):
    """The ``review`` gate: a strictly-stronger model judges the diff before a
    task may COMPLETE. The live judge call is injected so the suite never hits a
    model, and the env flag is controlled so default behavior stays unchanged."""

    def _store(self, tmp):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    def _git_repo(self, path, *, with_change):
        path.mkdir(parents=True, exist_ok=True)
        for args in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "t@t.t"],
            ["git", "config", "user.name", "t"],
        ):
            subprocess.run(args, cwd=str(path), check=True, capture_output=True)
        (path / "seed.txt").write_text("seed\n")
        subprocess.run(["git", "add", "-A"], cwd=str(path), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "seed"], cwd=str(path), check=True, capture_output=True
        )
        if with_change:
            (path / "feature.py").write_text("def f():\n    return 1\n")

    def _task(self, **payload):
        from puppetmaster.models import Task

        return Task(job_id="job_r", role="cursor", instruction="add feature f", payload=payload)

    def _registry(self):
        from puppetmaster.model_registry import ModelSpec

        return [
            ModelSpec(id="cursor/composer-2-5", adapter="cursor", adapter_model_name="composer-2.5", capability_score=55, billing="plan", tags=["cursor"]),
            ModelSpec(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5", capability_score=90, billing="plan", tags=["cursor"]),
        ]

    # ----- pure helpers -----

    def test_build_review_prompt_contains_intent_diff_and_contract(self) -> None:
        from puppetmaster.gates import _REVIEW_VERDICT_MARKER, build_review_prompt

        prompt = build_review_prompt(self._task(), "THE_DIFF_BODY", "THE_RUBRIC")
        self.assertIn("add feature f", prompt)
        self.assertIn("THE_DIFF_BODY", prompt)
        self.assertIn("THE_RUBRIC", prompt)
        self.assertIn(_REVIEW_VERDICT_MARKER, prompt)

    def test_parse_review_verdict_extracts_last_marker(self) -> None:
        from puppetmaster.gates import _REVIEW_VERDICT_MARKER, parse_review_verdict

        text = (
            f"first {_REVIEW_VERDICT_MARKER} {{\"pass\": true, \"severity\": \"none\"}}\n"
            f"final {_REVIEW_VERDICT_MARKER} "
            '{"pass": false, "severity": "major", "reasons": ["broken"]}'
        )
        verdict = parse_review_verdict(text)
        self.assertEqual(verdict["pass"], False)
        self.assertEqual(verdict["reasons"], ["broken"])

    def test_parse_review_verdict_none_on_missing_or_malformed(self) -> None:
        from puppetmaster.gates import _REVIEW_VERDICT_MARKER, parse_review_verdict

        self.assertIsNone(parse_review_verdict("no marker here"))
        self.assertIsNone(parse_review_verdict(f"{_REVIEW_VERDICT_MARKER} {{not json"))

    def test_resolve_judge_picks_cheapest_strictly_stronger(self) -> None:
        from puppetmaster.gates import resolve_judge_model

        with patch("puppetmaster.model_registry.load_registry", return_value=self._registry()), \
             patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True):
            judge = resolve_judge_model(
                self._task(router_model_id="cursor/composer-2-5"), {}
            )
        self.assertEqual(judge.id, "cursor/gpt-5-5")

    def test_resolve_judge_peer_when_implementer_top_tier(self) -> None:
        from puppetmaster.gates import resolve_judge_model

        with patch("puppetmaster.model_registry.load_registry", return_value=self._registry()), \
             patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True):
            judge = resolve_judge_model(
                self._task(router_model_id="cursor/gpt-5-5"), {}
            )
        # No strictly-stronger model exists; the strongest available (the peer
        # top tier) is used rather than skipping review entirely.
        self.assertEqual(judge.id, "cursor/gpt-5-5")

    def test_resolve_judge_none_without_registry(self) -> None:
        from puppetmaster.gates import resolve_judge_model

        with patch("puppetmaster.model_registry.load_registry", return_value=[]):
            self.assertIsNone(resolve_judge_model(self._task(), {}))

    def test_default_judge_disabled_without_env(self) -> None:
        from unittest.mock import Mock

        from puppetmaster.gates import default_judge_review

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PUPPETMASTER_REVIEW_GATE", None)
            verdict = default_judge_review(
                prompt="p", judge=Mock(id="cursor/gpt-5-5", adapter="cursor"),
                cwd=Path("."), timeout=10, task=self._task(),
            )
        self.assertFalse(verdict.available)
        self.assertTrue(verdict.passed)  # unavailable judge is a no-op, not a block

    def test_default_judge_fail_closed_on_unparseable(self) -> None:
        from unittest.mock import Mock

        from puppetmaster.gates import default_judge_review
        from puppetmaster.models import Artifact, ArtifactType

        judge = Mock(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5")
        noise = Artifact(
            job_id="job_r", task_id="t", type=ArtifactType.VERIFICATION,
            created_by="judge", payload={"check": "c", "result": "passed", "stdout": "no verdict here"},
            confidence=0.9, evidence=["adapter:cursor"],
        )
        fake_adapter = Mock()
        fake_adapter.run.return_value = [noise]
        with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}), \
             patch("puppetmaster.adapters.get_adapter", return_value=fake_adapter):
            verdict = default_judge_review(
                prompt="p", judge=judge, cwd=Path("."), timeout=10, task=self._task(),
            )
        self.assertTrue(verdict.available)
        self.assertFalse(verdict.passed)  # a judge that ran but gave no verdict → reject

    def test_default_judge_reads_verdict_from_stdout(self) -> None:
        from unittest.mock import Mock

        from puppetmaster.gates import _REVIEW_VERDICT_MARKER, default_judge_review
        from puppetmaster.models import Artifact, ArtifactType

        judge = Mock(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5")
        out = Artifact(
            job_id="job_r", task_id="t", type=ArtifactType.VERIFICATION,
            created_by="judge",
            payload={"check": "c", "result": "passed", "stdout": f"{_REVIEW_VERDICT_MARKER} {{\"pass\": true, \"severity\": \"none\"}}"},
            confidence=0.9, evidence=["adapter:cursor"],
        )
        fake_adapter = Mock()
        fake_adapter.run.return_value = [out]
        with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}), \
             patch("puppetmaster.adapters.get_adapter", return_value=fake_adapter):
            verdict = default_judge_review(
                prompt="p", judge=judge, cwd=Path("."), timeout=10, task=self._task(),
            )
        self.assertTrue(verdict.available)
        self.assertTrue(verdict.passed)

    # ----- gate behavior (injected judge) -----

    def _eval_review(self, tmp, *, with_change, judge_verdict, spec_extra=None):
        from unittest.mock import Mock

        import puppetmaster.gates as gates
        from puppetmaster.gates import ReviewVerdict, evaluate_task_gates

        store = self._store(tmp)
        repo = Path(tmp) / "repo"
        self._git_repo(repo, with_change=with_change)
        payload = {"review": True, "cwd": str(repo)}
        payload.update(spec_extra or {})
        task = self._task(**payload)
        fake_model = Mock(id="cursor/gpt-5-5")
        with patch.object(gates, "resolve_judge_model", return_value=fake_model), \
             patch.object(gates, "_REVIEW_JUDGE", return_value=judge_verdict):
            return evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)

    def test_gate_passes_when_judge_approves(self) -> None:
        from puppetmaster.gates import ReviewVerdict

        with TemporaryDirectory() as tmp:
            evald = self._eval_review(
                tmp, with_change=True,
                judge_verdict=ReviewVerdict(available=True, passed=True, severity="none"),
            )
            self.assertTrue(evald.passed)

    def test_gate_fails_when_judge_rejects(self) -> None:
        from puppetmaster.gates import ReviewVerdict

        with TemporaryDirectory() as tmp:
            evald = self._eval_review(
                tmp, with_change=True,
                judge_verdict=ReviewVerdict(
                    available=True, passed=False, severity="major", reasons=["incorrect logic"]
                ),
            )
            self.assertFalse(evald.passed)
            self.assertIn("rejected", evald.failed_reason)
            gate_art = [a for a in evald.artifacts if (a.payload or {}).get("kind") == "review"]
            self.assertEqual(len(gate_art), 1)
            self.assertFalse(gate_art[0].payload["passed"])

    def test_gate_failure_is_blocked_verdict(self) -> None:
        from puppetmaster.gates import ReviewVerdict
        from puppetmaster.quality import assess_run_quality

        with TemporaryDirectory() as tmp:
            evald = self._eval_review(
                tmp, with_change=True,
                judge_verdict=ReviewVerdict(available=True, passed=False, reasons=["bad"]),
            )
            verdict = assess_run_quality(evald.artifacts)
            self.assertEqual(verdict["quality"], "blocked")
            self.assertFalse(verdict["trustworthy"])

    def test_gate_skips_when_no_diff(self) -> None:
        from puppetmaster.gates import ReviewVerdict

        with TemporaryDirectory() as tmp:
            # Judge would reject, but there's no diff so the gate never consults it.
            evald = self._eval_review(
                tmp, with_change=False,
                judge_verdict=ReviewVerdict(available=True, passed=False),
            )
            self.assertTrue(evald.passed)

    def test_gate_skips_when_judge_unavailable(self) -> None:
        from puppetmaster.gates import ReviewVerdict

        with TemporaryDirectory() as tmp:
            evald = self._eval_review(
                tmp, with_change=True,
                judge_verdict=ReviewVerdict(available=False, passed=True),
            )
            self.assertTrue(evald.passed)

    def test_deterministic_sampling(self) -> None:
        from puppetmaster.gates import _review_sampled

        self.assertTrue(_review_sampled("any-task", 1.0))
        self.assertFalse(_review_sampled("any-task", 0.0))
        # Stable across calls for the same id.
        self.assertEqual(_review_sampled("t-123", 0.5), _review_sampled("t-123", 0.5))

    # ----- auto-attach wiring -----

    def test_review_auto_attached_for_implement_when_env_set(self) -> None:
        from puppetmaster.gates import task_gate_specs

        with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}):
            kinds = [s["kind"] for s in task_gate_specs(self._task(mode="implement"))]
            self.assertIn("review", kinds)
            # Ordered last so the cheap gates run before the expensive judge.
            self.assertEqual(kinds[-1], "review")
            # A single task opts out even with the global flag on.
            self.assertNotIn(
                "review",
                [s["kind"] for s in task_gate_specs(self._task(mode="implement", review=False))],
            )

    def test_review_not_attached_by_default(self) -> None:
        from puppetmaster.gates import task_gate_specs

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PUPPETMASTER_REVIEW_GATE", None)
            kinds = [s["kind"] for s in task_gate_specs(self._task(mode="implement"))]
            self.assertNotIn("review", kinds)
            # Explicit per-task opt-in still works without the global flag.
            self.assertIn(
                "review",
                [s["kind"] for s in task_gate_specs(self._task(mode="implement", review=True))],
            )


class ReviewEscalationTests(unittest.TestCase):
    """Objective escalation: a task a review gate rejected is re-routed one
    capability tier up (vs. the self-reported-confidence escalation)."""

    def _registry(self):
        from puppetmaster.model_registry import ModelSpec

        return [
            ModelSpec(id="cursor/composer-2-5", adapter="cursor", adapter_model_name="composer-2.5", capability_score=55, billing="plan", tags=["cursor"]),
            ModelSpec(id="cursor/gpt-5-5", adapter="cursor", adapter_model_name="gpt-5.5", capability_score=90, billing="plan", tags=["cursor"]),
        ]

    def _plan_billing(self, adapter, **kw):
        from puppetmaster.platform_billing import BillingStatus

        return BillingStatus(adapter=adapter, billing="plan", healthy=True, detail="ok", evidence=[])

    def _setup(self, tmp, *, model_id, passed=False, payload_extra=None):
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.store import SwarmStore

        store = SwarmStore(Path(tmp) / ".puppetmaster")
        job = store.create_job("do the work")
        payload = {"auto_route": True, "router_model_id": model_id, "model": "x", "mode": "implement"}
        payload.update(payload_extra or {})
        task = Task(
            job_id=job.id, role="implement", instruction="implement the thing",
            adapter="cursor", status=TaskStatus.FAILED, payload=payload,
        )
        store.save_task(task)
        store.save_artifact(Artifact(
            job_id=job.id, task_id=task.id, type=ArtifactType.GATE,
            created_by="w", payload={"gate": "review", "kind": "review", "passed": passed, "reason": "judge rejected"},
            confidence=0.9, evidence=["gate:review", "failed" if not passed else "passed"],
        ))
        return store, job, task

    def _run_reroute(self, store, job):
        from puppetmaster.orchestrator import Orchestrator

        orch = Orchestrator(store)
        with patch("puppetmaster.model_registry.load_registry", return_value=self._registry()), \
             patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=self._plan_billing), \
             patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True):
            return orch._reroute_failed_review(job)

    def test_failed_review_task_ids(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.orchestrator import Orchestrator

        def gate(task_id, passed, at):
            return Artifact(
                job_id="j", task_id=task_id, type=ArtifactType.GATE, created_by="w",
                payload={"kind": "review", "passed": passed}, confidence=0.9,
                evidence=["gate:review"], created_at=at,
            )

        arts = [
            gate("A", False, "2026-01-01T00:00:00Z"),
            gate("B", True, "2026-01-01T00:00:00Z"),
            # C: an older rejection superseded by a newer pass → not failed.
            gate("C", False, "2026-01-01T00:00:00Z"),
            gate("C", True, "2026-01-02T00:00:00Z"),
        ]
        self.assertEqual(Orchestrator._failed_review_task_ids(arts), {"A"})

    def test_reroute_escalates_rejected_diff(self) -> None:
        from puppetmaster.models import TaskStatus

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp, model_id="cursor/composer-2-5")
            self.assertEqual(self._run_reroute(store, job), 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.payload["router_model_id"], "cursor/gpt-5-5")
            self.assertEqual(updated.payload["review_escalation_attempts"], 1)
            arts = [a for a in store.list_artifacts(job.id) if a.created_by == "router-review-escalation"]
            self.assertEqual(len(arts), 1)

    def test_no_reroute_when_review_passed(self) -> None:
        from puppetmaster.models import TaskStatus

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp, model_id="cursor/composer-2-5", passed=True)
            # A passed review (and a FAILED status for some other reason) is not a
            # review-escalation trigger.
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_no_reroute_when_already_top_tier(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp, model_id="cursor/gpt-5-5")
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_reroute_bounded_by_max_attempts(self) -> None:
        from puppetmaster.orchestrator import _MAX_ESCALATION_ATTEMPTS

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(
                tmp, model_id="cursor/composer-2-5",
                payload_extra={"review_escalation_attempts": _MAX_ESCALATION_ATTEMPTS},
            )
            self.assertEqual(self._run_reroute(store, job), 0)

    def test_review_rejection_is_reachable_not_fail_closed(self) -> None:
        """Regression guard (redteam): a review rejection must NOT make
        _run_workers raise before the escalation sweep runs — it's pending
        re-route, like a recoverable failure — but a task with no budget left
        is terminal and the job must fail closed."""
        from dataclasses import replace

        from puppetmaster.orchestrator import Orchestrator, _MAX_ESCALATION_ATTEMPTS

        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp, model_id="cursor/composer-2-5")
            orch = Orchestrator(store)
            self.assertFalse(orch._should_fail_closed(job, {task.id}))
            store.save_task(
                replace(
                    task,
                    payload={**task.payload, "review_escalation_attempts": _MAX_ESCALATION_ATTEMPTS},
                )
            )
            self.assertTrue(orch._should_fail_closed(job, {task.id}))


class PuppetmasterSalvageAndLivenessTests(unittest.TestCase):
    """#3 salvage structured content from raw stdout; #9 loud liveness."""

    def _store(self, tmp: str):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    def test_analyze_salvages_findings_from_stdout_on_failure(self) -> None:
        from puppetmaster.models import Task

        task = Task(
            job_id="job", role="reviewer", instruction="review",
            adapter="cursor", payload={"prompt": "review", "cwd": "."},
        )
        # A non-zero run whose structured findings nonetheless sit in stdout —
        # previously lost; now salvaged before declaring degraded/failed.
        completed = subprocess.CompletedProcess(
            args=["node"],
            returncode=2,
            stdout=json.dumps([{"type": "finding", "claim": "recovered finding", "evidence": ["e"]}]),
            stderr="",
        )
        with patch("puppetmaster.adapters.subprocess.run", return_value=completed):
            artifacts = CursorAdapter().run(task, "goal", "worker-cursor")
        claims = [a.payload.get("claim") for a in artifacts if str(a.type) == "finding"]
        self.assertIn("recovered finding", claims)

    def test_worker_cli_env_prepends_source_root(self) -> None:
        from puppetmaster.codegraph import inject_worker_cli_env, puppetmaster_source_root

        root = puppetmaster_source_root()
        env = inject_worker_cli_env({})
        self.assertTrue(env["PYTHONPATH"].startswith(root))
        # An existing PYTHONPATH is preserved after the injected root.
        env2 = inject_worker_cli_env({"PYTHONPATH": "/existing"})
        self.assertTrue(env2["PYTHONPATH"].startswith(root))
        self.assertIn("/existing", env2["PYTHONPATH"])

    def test_scrub_foreign_interpreter_env_drops_python_path(self) -> None:
        """A foreign Python worker (Hermes) must not inherit the parent's
        PYTHONPATH/PYTHONHOME, or it imports Puppetmaster's site-packages and
        crashes on a cross-interpreter version clash."""
        from puppetmaster.codegraph import scrub_foreign_interpreter_env

        env = scrub_foreign_interpreter_env(
            {"PYTHONPATH": "/pyenv/3.9/site-packages", "PYTHONHOME": "/pyenv/3.9", "PATH": "/usr/bin"}
        )
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        # Unrelated vars are left intact.
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_hermes_adapter_scrubs_foreign_python_env_not_inject(self) -> None:
        """The Hermes worker env must be built with the foreign-interpreter
        scrub, never inject_worker_cli_env — the latter leaks the parent's
        PYTHONPATH into the foreign Hermes Python and crashes the worker."""
        import inspect
        from puppetmaster import adapters

        for fn in (adapters.HermesAdapter._run_implement, adapters.HermesAdapter._run_analyze):
            src = inspect.getsource(fn)
            self.assertIn("scrub_foreign_interpreter_env", src, fn.__name__)
            self.assertNotIn("inject_worker_cli_env", src, fn.__name__)

    def test_liveness_summary_flags_dead_pid(self) -> None:
        from puppetmaster.liveness import liveness_summary

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            store.update_job_status(job.id, JobStatus.RUNNING)
            store.write_json(
                store.job_dir(job.id) / "orchestrator.json",
                {
                    "pid": 999999999,
                    "host": socket.gethostname(),
                    "started_at": seconds_from_now(-30),
                    "heartbeat_at": seconds_from_now(-30),
                },
            )
            summary = liveness_summary(store, store.get_job(job.id))
            self.assertFalse(summary["pid_alive"])
            self.assertIn("dead", summary["verdict"])


class PuppetmasterLifecycleTests(unittest.TestCase):
    """Durable-state GC (#8) and effort rollup (#7)."""

    def _store(self, tmp: str):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    def test_effort_tag_roundtrip(self) -> None:
        from puppetmaster.lifecycle import job_effort_id, tag_job_effort

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("goal")
            self.assertIsNone(job_effort_id(store, job.id))
            tag_job_effort(store, job.id, "migration-x")
            self.assertEqual(job_effort_id(store, job.id), "migration-x")

    def test_gc_only_reaps_old_terminal_jobs(self) -> None:
        from puppetmaster.lifecycle import gc_terminal_jobs

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            running = store.create_job("running")
            store.update_job_status(running.id, JobStatus.RUNNING)
            done = store.create_job("done")
            store.update_job_status(done.id, JobStatus.COMPLETE)

            # Dry-run with a 0-day window reaps the terminal job, not the running one.
            reaped = gc_terminal_jobs(store, older_than_days=0, force=False)
            ids = {r["job_id"] for r in reaped}
            self.assertIn(done.id, ids)
            self.assertNotIn(running.id, ids)
            self.assertFalse(reaped[0]["deleted"])
            self.assertIsNotNone(store.get_job(done.id))  # dry-run didn't delete

            # A 7-day window leaves the just-created terminal job alone.
            self.assertEqual(gc_terminal_jobs(store, older_than_days=7, force=False), [])

            # Force actually deletes.
            gc_terminal_jobs(store, older_than_days=0, force=True)
            self.assertNotIn(done.id, {j.id for j in store.list_jobs()})

    def test_delete_job_refuses_unsafe_ids(self) -> None:
        """delete_job must never rglob outside its own jobs tree (D1 data-loss guard)."""
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("real job")
            sentinel = store.jobs_dir / "DO_NOT_DELETE.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            for unsafe in ["", "   ", ".", "..", "../..", "/", "../../etc"]:
                with self.assertRaises(ValueError):
                    store.delete_job(unsafe)
            # The jobs tree and the real job survive every refusal.
            self.assertTrue(sentinel.exists())
            self.assertIsNotNone(store.get_job(job.id))
            # A legitimate id still deletes.
            store.delete_job(job.id)
            self.assertNotIn(job.id, {j.id for j in store.list_jobs()})

    def test_status_snapshot_surfaces_outcome_signals(self) -> None:
        """A2+F2: status carries a quality verdict + diff/commit presence."""
        from puppetmaster.models import Artifact, ArtifactType

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            job = store.create_job("implement something")
            # No artifacts yet -> empty/untrustworthy, no diff, no commit.
            snap = store.status_snapshot(job.id)
            self.assertIn("outcome", snap)
            self.assertEqual(snap["outcome"]["quality"], "empty")
            self.assertFalse(snap["outcome"]["diff_present"])
            self.assertFalse(snap["outcome"]["baseline_diff_present"])
            self.assertFalse(snap["outcome"]["worker_diff_present"])
            self.assertFalse(snap["outcome"]["patch_artifact_emitted"])
            self.assertFalse(snap["outcome"]["commit_present"])

            store.save_artifact(Artifact(
                job_id=job.id, task_id="t1", type=ArtifactType.VERIFICATION, created_by="w1",
                confidence=0.9, evidence=["e"],
                payload={
                    "check": "run",
                    "result": "passed",
                    "baseline_diff_present": True,
                    "worker_diff_present": False,
                },
            ))
            snap = store.status_snapshot(job.id)
            self.assertFalse(snap["outcome"]["diff_present"])
            self.assertTrue(snap["outcome"]["baseline_diff_present"])
            self.assertFalse(snap["outcome"]["worker_diff_present"])
            self.assertFalse(snap["outcome"]["patch_artifact_emitted"])

            store.save_artifact(Artifact(
                job_id=job.id, task_id="t1", type=ArtifactType.PATCH, created_by="w1",
                confidence=0.9, evidence=["e"],
                payload={
                    "change": "edit",
                    "files": ["a.py"],
                    "baseline_diff_present": True,
                    "worker_diff_present": True,
                    "patch_artifact_emitted": True,
                },
            ))
            store.save_artifact(Artifact(
                job_id=job.id, task_id="t1", type=ArtifactType.GATE, created_by="w1",
                confidence=0.95, evidence=["gate:committed", "passed"],
                payload={"gate": "committed", "kind": "committed", "passed": True},
            ))
            snap = store.status_snapshot(job.id)
            self.assertTrue(snap["outcome"]["diff_present"])
            self.assertTrue(snap["outcome"]["baseline_diff_present"])
            self.assertTrue(snap["outcome"]["worker_diff_present"])
            self.assertTrue(snap["outcome"]["patch_artifact_emitted"])
            self.assertTrue(snap["outcome"]["commit_present"])
            self.assertEqual(snap["outcome"]["artifact_count"], 3)

    def test_gc_all_projects_force_skips_active_worktree(self) -> None:
        """`gc --force --all-projects` must not delete the active project's state (D1)."""
        from puppetmaster.cli import _run_gc_command
        from types import SimpleNamespace

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            done = store.create_job("active worktree run")
            store.update_job_status(done.id, JobStatus.COMPLETE)

            args = SimpleNamespace(
                all_projects=True, force=True, older_than_days=0, json=True, backend="file"
            )
            with patch("puppetmaster.cli._gc_target_stores", return_value=[store]):
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                    _run_gc_command(args, store)
            # The active project's terminal job is preserved, not deleted.
            self.assertIsNotNone(store.get_job(done.id))

    def test_rollup_filters_by_effort(self) -> None:
        from puppetmaster.lifecycle import rollup_stores, tag_job_effort

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            a = store.create_job("a")
            tag_job_effort(store, a.id, "EFF")
            b = store.create_job("b")  # untagged

            scoped = rollup_stores([store], effort_id="EFF")
            self.assertEqual(scoped["jobs"], 1)

            everything = rollup_stores([store], effort_id=None)
            self.assertEqual(everything["jobs"], 2)
            self.assertIn("EFF", everything["efforts_seen"])


class PuppetmasterUsageTests(unittest.TestCase):
    """Token-consumption capture & rollup (#5 metering, #6 estimate labeling)."""

    def test_usage_from_sdk_accepts_key_spellings(self) -> None:
        from puppetmaster.usage import usage_from_sdk

        self.assertEqual(
            usage_from_sdk({"inputTokens": 10, "outputTokens": 5}),
            {"tokens_in": 10, "tokens_out": 5},
        )
        self.assertEqual(
            usage_from_sdk({"prompt_tokens": 7, "completion_tokens": 3}),
            {"tokens_in": 7, "tokens_out": 3},
        )
        self.assertIsNone(usage_from_sdk(None))
        self.assertIsNone(usage_from_sdk({"unrelated": 1}))

    def test_token_usage_measured_vs_estimated(self) -> None:
        from puppetmaster.usage import token_usage

        measured = token_usage(sdk_usage={"inputTokens": 100, "outputTokens": 40})
        self.assertFalse(measured["tokens_estimated"])
        self.assertEqual(measured["tokens_in"], 100)

        estimated = token_usage(prompt_text="a" * 40, output_text="b" * 20)
        self.assertTrue(estimated["tokens_estimated"])
        self.assertEqual(estimated["tokens_in"], 10)
        self.assertEqual(estimated["tokens_out"], 5)

    def test_sdk_usage_from_stdout_extraction(self) -> None:
        from puppetmaster.adapters import sdk_usage_from_stdout

        stdout = json.dumps({"status": "finished", "result": "x", "usage": {"inputTokens": 9}})
        self.assertEqual(sdk_usage_from_stdout(stdout), {"inputTokens": 9})
        self.assertIsNone(sdk_usage_from_stdout("not json"))

    def test_cursor_turn_ended_usage_is_measured_with_cache(self) -> None:
        """The real Cursor `turn-ended` usage shape yields measured (not
        estimated) counts and preserves cache read/write tokens for the cost axis."""
        from puppetmaster.usage import token_usage, usage_from_sdk

        # Exactly the shape the streaming runner now accumulates and emits.
        sdk_usage = {
            "inputTokens": 48000,
            "outputTokens": 1200,
            "cacheReadTokens": 30000,
            "cacheWriteTokens": 500,
        }
        self.assertEqual(
            usage_from_sdk(sdk_usage),
            {
                "tokens_in": 48000,
                "tokens_out": 1200,
                "cache_read_tokens": 30000,
                "cache_write_tokens": 500,
            },
        )
        record = token_usage(sdk_usage=sdk_usage)
        self.assertFalse(record["tokens_estimated"])
        self.assertEqual(record["tokens_in"], 48000)
        self.assertEqual(record["tokens_out"], 1200)
        self.assertEqual(record["cache_read_tokens"], 30000)
        self.assertEqual(record["cache_write_tokens"], 500)

    def test_cursor_runner_null_usage_falls_back_to_estimate(self) -> None:
        """When the runtime reports no usage (runner emits usage: null), the
        record must be a clearly-labeled estimate, never a fake-measured zero."""
        from puppetmaster.adapters import sdk_usage_from_stdout
        from puppetmaster.usage import token_usage

        stdout = json.dumps({"status": "finished", "result": "hello", "usage": None})
        record = token_usage(
            sdk_usage=sdk_usage_from_stdout(stdout),
            prompt_text="x" * 80,
            output_text="hello",
        )
        self.assertTrue(record["tokens_estimated"])
        self.assertNotIn("cache_read_tokens", record)

    def test_aggregate_splits_measured_and_estimated(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.usage import aggregate_token_usage

        def va(task_id, tin, tout, estimated):
            return Artifact(
                job_id="j", task_id=task_id, type=ArtifactType.VERIFICATION,
                created_by="w", confidence=0.9, evidence=["e"],
                payload={
                    "check": "c", "result": "passed",
                    "tokens_in": tin, "tokens_out": tout, "tokens_estimated": estimated,
                },
            )

        roll = aggregate_token_usage(
            [va("t1", 100, 40, False), va("t2", 20, 10, True), va("t1", 999, 999, False)]
        )
        # t1 counted once (dedup), measured; t2 estimated.
        self.assertEqual(roll["measured_runs"], 1)
        self.assertEqual(roll["measured_tokens_in"], 100)
        self.assertEqual(roll["estimated_runs"], 1)
        self.assertEqual(roll["estimated_tokens_in"], 20)
        self.assertEqual(roll["total_tokens"], 100 + 40 + 20 + 10)


class PuppetmasterGateReplayCliTests(unittest.TestCase):
    """`puppetmaster gate` replays the runtime's completion gates outside a
    worker run, so a parent agent or CI can enforce the same post-conditions."""

    def _store(self, tmp: str):
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        return store

    def _git_repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@e.co", "-c", "user.name=T", "commit", "-m", "seed"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        return repo

    def _args(self, store, repo: Path, **overrides):
        from types import SimpleNamespace

        defaults = dict(
            cwd=str(repo),
            require_diff=False,
            gate_command=None,
            ratchet_command=None,
            metric=None,
            committed=False,
            gates_json=None,
            json=True,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_require_diff_fails_on_clean_tree_but_committed_passes(self) -> None:
        from puppetmaster.cli import _run_gate_command

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = self._git_repo(Path(tmp))
            # Clean tree: require_diff must fail (exit 1)...
            self.assertEqual(
                _run_gate_command(self._args(store, repo, require_diff=True), store), 1
            )
            # ...while committed passes (nothing uncommitted) (exit 0).
            self.assertEqual(
                _run_gate_command(self._args(store, repo, committed=True), store), 0
            )

    def test_ratchet_establishes_then_enforces_monotonic(self) -> None:
        from puppetmaster.cli import _run_gate_command

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = self._git_repo(Path(tmp))
            metric_file = repo / "metric.txt"
            metric_file.write_text("5", encoding="utf-8")
            # A tiny oracle that prints {"metrics": {"violations": N}} from a file.
            oracle = (
                f'{sys.executable} -c "import json,pathlib;'
                f"print(json.dumps({{'metrics': {{'violations': "
                f"int(pathlib.Path('metric.txt').read_text())}}}}))\""
            )
            args = self._args(store, repo, ratchet_command=oracle, metric="violations")
            # First run establishes baseline=5 → pass.
            self.assertEqual(_run_gate_command(args, store), 0)
            # Regress to 9 → ratchet loosened → fail.
            metric_file.write_text("9", encoding="utf-8")
            self.assertEqual(_run_gate_command(args, store), 1)
            # Tighten to 2 → pass and move baseline down.
            metric_file.write_text("2", encoding="utf-8")
            self.assertEqual(_run_gate_command(args, store), 0)

    def test_no_gates_specified_is_a_usage_error(self) -> None:
        from puppetmaster.cli import _run_gate_command

        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            repo = self._git_repo(Path(tmp))
            self.assertEqual(_run_gate_command(self._args(store, repo), store), 2)


class PuppetmasterMcpVerbTests(unittest.TestCase):
    """gc / rollup / gate are first-class MCP verbs that shell to the CLI."""

    def test_gc_rollup_gate_build_expected_cli_commands(self) -> None:
        from unittest.mock import patch

        import puppetmaster.mcp_server as mcp

        captured: list = []

        def fake_run_cli(command, args):
            captured.append(command)
            return {"content": [], "isError": False}

        with patch.object(mcp, "run_cli", side_effect=fake_run_cli):
            mcp.run_gc({"older_than_days": 3, "all_projects": True, "force": True})
            mcp.run_rollup({"effort_id": "mig-1", "all_projects": True})
            mcp.run_gate(
                {
                    "gate_cwd": "/tmp/x",
                    "require_diff": True,
                    "command": "pytest -q",
                    "committed": True,
                    "gates": [{"kind": "ratchet", "command": "c", "metric": "m"}],
                }
            )

        gc_cmd, rollup_cmd, gate_cmd = captured
        self.assertEqual(
            gc_cmd, ["gc", "--json", "--older-than-days", "3", "--all-projects", "--force"]
        )
        self.assertEqual(rollup_cmd, ["rollup", "--json", "--effort", "mig-1", "--all-projects"])
        self.assertEqual(gate_cmd[:2], ["gate", "--json"])
        self.assertIn("--require-diff", gate_cmd)
        self.assertIn("--committed", gate_cmd)
        self.assertIn("--cwd", gate_cmd)
        self.assertIn("--command", gate_cmd)
        self.assertIn("--gates-json", gate_cmd)

    def test_gc_rollup_gate_registered_as_tools(self) -> None:
        import puppetmaster.mcp_server as mcp

        names = {tool.name for tool in mcp.tools()}
        self.assertIn("puppetmaster_gc", names)
        self.assertIn("puppetmaster_rollup", names)
        self.assertIn("puppetmaster_gate", names)

    def test_dashboard_tool_reuses_running_server(self) -> None:
        import puppetmaster.mcp_server as mcp

        self.assertIn("puppetmaster_dashboard", {tool.name for tool in mcp.tools()})
        with patch.object(mcp, "_dashboard_alive", return_value=True), patch.object(
            mcp, "_spawn_dashboard_server"
        ) as popen:
            result = mcp.call_tool(
                "puppetmaster_dashboard", {"cwd": "/tmp", "job_id": "job_abc"}
            )
        body = json.loads(result["content"][0]["text"])
        popen.assert_not_called()
        self.assertTrue(body["already_running"])
        self.assertFalse(body["started"])
        self.assertEqual(body["url"], "http://127.0.0.1:8787/?job=job_abc")

    def test_dashboard_tool_starts_server_when_absent(self) -> None:
        import puppetmaster.mcp_server as mcp

        spawned = MagicMock()
        spawned.pid = 4321
        spawned.poll.return_value = None
        # alive checks: initial probe (absent), readiness loop, post-loop verify
        with patch.object(
            mcp, "_dashboard_alive", side_effect=[False, True, True]
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp", "port": 9000})
        body = json.loads(result["content"][0]["text"])
        command = popen.call_args.args[0]
        self.assertIn("dashboard", command)
        self.assertIn("--no-open", command)
        self.assertIn("9000", command)
        self.assertTrue(body["started"])
        self.assertEqual(body["pid"], 4321)
        self.assertEqual(body["url"], "http://127.0.0.1:9000/")

    def test_dashboard_tool_forwards_all_projects_flag(self) -> None:
        import puppetmaster.mcp_server as mcp

        schema = mcp.dashboard_schema()
        self.assertIn("all_projects", schema["properties"])

        spawned = MagicMock()
        spawned.pid = 5555
        spawned.poll.return_value = None
        with patch.object(
            mcp, "_dashboard_alive", side_effect=[False, True, True]
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            result = mcp.call_tool(
                "puppetmaster_dashboard", {"cwd": "/tmp", "all_projects": True}
            )
        body = json.loads(result["content"][0]["text"])
        command = popen.call_args.args[0]
        self.assertIn("--all-projects", command)
        self.assertTrue(body["all_projects"])

    def test_dashboard_tool_omits_all_projects_by_default(self) -> None:
        import puppetmaster.mcp_server as mcp

        spawned = MagicMock()
        spawned.pid = 5556
        spawned.poll.return_value = None
        with patch.object(
            mcp, "_dashboard_alive", side_effect=[False, True, True]
        ), patch.object(mcp, "_spawn_dashboard_server", return_value=spawned) as popen:
            mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        command = popen.call_args.args[0]
        self.assertNotIn("--all-projects", command)

    def test_dashboard_tool_reports_failed_start(self) -> None:
        import puppetmaster.mcp_server as mcp

        dead = MagicMock()
        dead.pid = 4321
        dead.poll.return_value = 1
        with patch.object(mcp, "_dashboard_alive", return_value=False), patch.object(
            mcp, "_spawn_dashboard_server", return_value=dead
        ):
            result = mcp.call_tool("puppetmaster_dashboard", {"cwd": "/tmp"})
        self.assertTrue(result["isError"])
        body = json.loads(result["content"][0]["text"])
        self.assertEqual(body["error"], "dashboard failed to start")

    def test_installed_rules_teach_the_dashboard_verb(self) -> None:
        from puppetmaster.rules import RULE_BODY

        self.assertIn("puppetmaster_dashboard", RULE_BODY)


class InvocationGateTests(unittest.TestCase):
    """Tests for the classifier-gated auto-invocation decision."""

    def test_should_delegate_audit_suggests_review_verb(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("audit the auth module for security issues across the repo")
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.suggested_verb, "puppetmaster_start_cursor_review")
        self.assertGreaterEqual(d.capability_score, 60)

    def test_should_not_delegate_trivial_typo(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("fix a typo in the README")
        self.assertFalse(d.should_delegate)
        self.assertIn("trivial", d.matched_signals)

    def test_explicit_trigger_forces_delegation(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("use puppetmaster to add a comment")
        self.assertTrue(d.should_delegate)
        self.assertIn("explicit-trigger", d.matched_signals)

    def test_explicit_inline_opt_out_wins(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("refactor everything across the repo, but do it inline")
        self.assertFalse(d.should_delegate)
        self.assertIn("explicit-inline", d.matched_signals)

    def test_kill_switch_disables_delegation(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("audit everything", env={"PUPPETMASTER_AUTO_INVOKE_DISABLED": "1"})
        self.assertFalse(d.should_delegate)
        self.assertIn("kill-switch", d.matched_signals)

    def test_hard_scope_overrides_just_under_threshold(self):
        from puppetmaster.invocation_gate import should_delegate

        # A high threshold forces the score under the bar; broad scope still wins.
        d = should_delegate("rename every usage of getCwd across the codebase", threshold=98)
        self.assertTrue(d.should_delegate)
        self.assertIn("hard-scope", d.matched_signals)

    def test_role_inference_maps_to_verbs(self):
        from puppetmaster.invocation_gate import infer_role_and_verb

        self.assertEqual(infer_role_and_verb("design the caching layer")[1], "puppetmaster_start_cursor_plan")
        self.assertEqual(infer_role_and_verb("where is ClientError defined")[1], "puppetmaster_codegraph_search")

    def test_codegraph_lookups_delegate_regardless_of_score(self):
        """A structural "where is X / who calls Y / what implements Z" lookup must
        delegate to CodeGraph even when its capability score is low. Unlike an
        edit (where a small change *might* be faster inline, so the score gate
        guards the round-trip), a lookup is cheap and strictly beats inline grep —
        there's no penalty to protect against, so score must not hold it back.
        """
        from puppetmaster.invocation_gate import should_delegate

        for prompt in (
            "where is ClientError defined",
            "who calls the retry helper",
            "what implements the SwarmStore interface",
            "find all usages of deprecated_fn",
            "trace how the auth token flows through the request",
        ):
            d = should_delegate(prompt)
            self.assertTrue(d.should_delegate, prompt)
            self.assertEqual(d.suggested_verb, "puppetmaster_codegraph_search", prompt)

    def test_codegraph_lookup_directive_points_at_codegraph(self):
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("who calls the retry helper")
        directive = d.directive()
        self.assertIn("puppetmaster_codegraph_search", directive)
        self.assertIn("CodeGraph", directive)
        self.assertNotIn("fan it out to a swarm", directive)

    def test_conceptual_question_stays_inline_not_a_lookup(self):
        """A conceptual "what is X" / "what does the flag do" question is NOT a
        structural lookup and must stay inline — only the where/who/what-implements
        family delegates to CodeGraph."""
        from puppetmaster.invocation_gate import should_delegate

        for prompt in ("what is a SwarmStore", "what does the --cwd flag do"):
            d = should_delegate(prompt)
            self.assertFalse(d.should_delegate, prompt)

    def test_lookup_explicit_inline_optout_wins_over_codegraph(self):
        """The explicit-inline opt-out must beat the always-delegate lookup rule —
        "just answer me / no puppetmaster" keeps even a lookup inline."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("where is ClientError defined — no puppetmaster, just answer")
        self.assertFalse(d.should_delegate)
        self.assertIn("explicit-inline", d.matched_signals)

    def test_feature_implementation_routes_to_single_implement_worker(self):
        """Plain, focused feature work must delegate to a SINGLE worker, not the
        read-only swarm default. With the scope-aware refinement these focused
        intents (no broad-scope signal) land on the lightweight ``edit`` verb —
        still one coherent worker, never a fan-out swarm. Broad-scope work keeps
        ``start_implement`` (see test_broad_scope_implementation_uses_implement).
        """
        from puppetmaster.invocation_gate import should_delegate

        for prompt in (
            "add a CSV export endpoint to the billing service",
            "create the webhook handler for Stripe events",
            "wire up retries with backoff in the API client",
        ):
            d = should_delegate(prompt)
            self.assertTrue(d.should_delegate, prompt)
            self.assertEqual(d.suggested_verb, "puppetmaster_edit", prompt)

    def test_broad_scope_implementation_uses_implement_worker(self):
        """A coupled, multi-file change (broad-scope signal) keeps the heavier
        ``start_implement`` verb, where an isolated worktree + one PATCH is the
        right shape — not the in-place ``edit`` verb."""
        from puppetmaster.invocation_gate import should_delegate

        for prompt in (
            "refactor the auth module across the whole codebase",
            "migrate every caller of the old client to the new api",
        ):
            d = should_delegate(prompt)
            self.assertTrue(d.should_delegate, prompt)
            self.assertEqual(d.suggested_verb, "puppetmaster_start_implement", prompt)

    def test_implement_directive_steers_to_clean_worktree_not_swarm(self):
        """The injected directive for a broad-scope implement task must point at
        a single clean-worktree worker and must NOT tell the host to fan out a
        swarm."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("refactor the auth module across the whole codebase")
        directive = d.directive()
        self.assertIn("clean worktree", directive)
        self.assertIn("not a", directive.lower())
        self.assertNotIn("fan it out to a swarm", directive)

    def test_focused_edit_directive_steers_to_edit_verb(self):
        """A focused edit's directive must point at the lightweight ``edit`` verb
        (in-place, cheap, inline diff) and not a fan-out swarm."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("add a CSV export endpoint to the billing service")
        self.assertEqual(d.suggested_verb, "puppetmaster_edit")
        directive = d.directive()
        self.assertIn("puppetmaster_edit", directive)
        self.assertIn("single", directive.lower())
        self.assertNotIn("fan it out to a swarm", directive)

    def test_review_directive_still_uses_swarm_framing(self):
        """Read-only analysis keeps the swarm framing — swarms are correct there."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("audit the auth module for security issues across the repo")
        self.assertEqual(d.suggested_verb, "puppetmaster_start_cursor_review")
        self.assertIn("fan it out to a swarm", d.directive())

    def test_trivial_add_comment_stays_inline_despite_broadened_implement(self):
        """Broadening implement detection to include 'add' must not start
        delegating routine one-liners; the trivial carve-out wins."""
        from puppetmaster.invocation_gate import should_delegate

        for prompt in ("add a comment to the parser", "fix a typo in the docstring"):
            d = should_delegate(prompt)
            self.assertFalse(d.should_delegate, prompt)
            self.assertIn("trivial", d.matched_signals, prompt)

    def test_last_mile_work_routes_to_edit_not_implement(self):
        """Work that builds on uncommitted changes must route to the in-place
        ``edit`` verb — an isolated-worktree implement job branched off HEAD
        can't see uncommitted modules and would clobber or rebuild them."""
        from puppetmaster.invocation_gate import should_delegate

        for prompt in (
            "finish the module I just wrote and add a test suite",
            "build on my uncommitted changes to wire the new adapter in",
            "implement the last mile on top of the code I just added",
            "wrap up this WIP and create the README caveats",
        ):
            d = should_delegate(prompt)
            self.assertTrue(d.should_delegate, prompt)
            self.assertEqual(d.suggested_verb, "puppetmaster_edit", prompt)
            self.assertIn("last-mile", d.matched_signals, prompt)

    def test_last_mile_overrides_broad_scope_implement(self):
        """The dirty-tree dependency is the binding constraint: even with a
        broad-scope signal (which normally keeps ``start_implement``), a prompt
        that builds on uncommitted work routes to the in-place ``edit`` verb —
        ``start_implement``'s worktree off HEAD would miss that work entirely."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "refactor every caller to build on the uncommitted client I just wrote"
        )
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.suggested_verb, "puppetmaster_edit")
        self.assertIn("last-mile", d.matched_signals)

    def test_trivial_wins_over_last_mile(self):
        """A trivial edit that happens to reference just-written code stays
        inline — the trivial carve-out is evaluated before the last-mile
        redirect, so a one-liner never spins up a worker."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("fix a typo in the file I just wrote")
        self.assertFalse(d.should_delegate)
        self.assertIn("trivial", d.matched_signals)

    def test_last_mile_directive_calls_out_uncommitted_work(self):
        """The injected directive for last-mile work must explain that ``edit``
        sees the live tree (uncommitted changes) where ``start_implement`` would
        not — that rationale is what stops the host falling back to inline."""
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("finish the module I just wrote and add tests")
        directive = d.directive()
        self.assertIn("puppetmaster_edit", directive)
        self.assertIn("uncommitted", directive.lower())
        self.assertNotIn("fan it out to a swarm", directive)


class HookRunnerTests(unittest.TestCase):
    """Tests for host hook payload → response translation."""

    # When a Puppetmaster MCP server is alive, steering is in effect. Most hook
    # tests assert that behavior, so assume tools-available unless a test says
    # otherwise. (The "no server alive -> no-op" contract has its own tests.)
    _TOOLS_ON = {"PUPPETMASTER_HOOK_ASSUME_TOOLS": "1"}
    _TOOLS_OFF = {"PUPPETMASTER_HOOK_ASSUME_TOOLS": "0"}

    def test_user_prompt_injects_directive_when_delegating(self):
        from puppetmaster.hook_runner import handle_hook

        r = handle_hook(
            {"prompt": "refactor the auth module across all files"},
            host="cursor", event="beforeSubmitPrompt", env=self._TOOLS_ON,
        )
        self.assertEqual(r.action, "allow")
        self.assertIn("Puppetmaster", r.context)
        out = r.to_host_json("cursor")
        self.assertEqual(out["permission"], "allow")
        self.assertIn("additionalContext", out)

    def test_pre_tool_allows_native_task_fanout(self):
        """Native Task/Agent must NOT be hard-denied: it wedged turns when the
        swarm tools weren't connected, and it over-reaches onto native subagents
        Puppetmaster does not replace (browser-use, ci-investigator, …).
        Steering toward a swarm stays at the non-blocking prompt layer."""
        from puppetmaster.hook_runner import handle_hook

        for host, event in (("claude", "PreToolUse"), ("cursor", "pre-tool")):
            for tool in ("Task", "Agent"):
                r = handle_hook(
                    {"tool_name": tool}, host=host, event=event, env=self._TOOLS_ON
                )
                self.assertEqual(r.action, "allow", f"{host}/{tool}")

    def test_hook_is_noop_when_no_mcp_server_available(self):
        """The core fix: with no Puppetmaster MCP server alive there is nothing
        to redirect to, so the hook must allow every native tool and inject no
        directive — never wedge a session toward an unreachable verb."""
        from puppetmaster.hook_runner import handle_hook

        # A recursive shell search would normally be redirected...
        redirected = handle_hook(
            {"tool_name": "shell", "command": "rg -r TODO ./src"},
            host="cursor", event="pre-tool", env=self._TOOLS_ON,
        )
        self.assertEqual(redirected.action, "deny")
        # ...but not when no server is available.
        for payload, event in (
            ({"tool_name": "shell", "command": "rg -r TODO ./src"}, "pre-tool"),
            ({"tool_name": "Task"}, "pre-tool"),
            ({"prompt": "audit the whole repo for security risks"}, "beforeSubmitPrompt"),
        ):
            r = handle_hook(payload, host="cursor", event=event, env=self._TOOLS_OFF)
            self.assertEqual(r.action, "allow")
            self.assertEqual(r.context, "")

    def test_deny_redirect_names_cli_fallback(self):
        """A redirected recursive search must not be a dead-end if the MCP tool
        isn't reachable — the deny reason names the CLI passthrough."""
        from puppetmaster.hook_runner import handle_hook

        claude = handle_hook(
            {"tool_name": "shell", "command": "grep -R foo ."},
            host="claude", event="PreToolUse", env=self._TOOLS_ON,
        )
        self.assertEqual(claude.action, "deny")
        self.assertIn("puppetmaster_codegraph_search", claude.reason)
        self.assertIn("python -m puppetmaster codegraph", claude.reason)

    def test_prompt_directive_suggests_host_portable_verb(self):
        from puppetmaster.hook_runner import handle_hook

        prompt = {"prompt": "review the whole repo for security risks"}
        claude = handle_hook(
            prompt, host="claude", event="UserPromptSubmit", env=self._TOOLS_ON
        )
        self.assertEqual(claude.action, "allow")
        self.assertTrue(claude.decision.should_delegate)
        self.assertNotIn("cursor", claude.decision.suggested_verb)
        self.assertNotIn("cursor", claude.context)

        cursor = handle_hook(
            prompt, host="cursor", event="beforeSubmitPrompt", env=self._TOOLS_ON
        )
        self.assertEqual(cursor.decision.suggested_verb, "puppetmaster_start_cursor_review")

    def test_verb_for_host_translation_table(self):
        from puppetmaster.hook_runner import verb_for_host

        self.assertEqual(
            verb_for_host("puppetmaster_start_cursor_swarm", "claude"),
            "puppetmaster_start_swarm",
        )
        self.assertEqual(
            verb_for_host("puppetmaster_start_cursor_implement", "codex"),
            "puppetmaster_start_implement",
        )
        # Generic verbs pass through untouched on every host.
        self.assertEqual(
            verb_for_host("puppetmaster_codegraph_search", "claude"),
            "puppetmaster_codegraph_search",
        )
        self.assertEqual(
            verb_for_host("puppetmaster_start_cursor_swarm", "cursor"),
            "puppetmaster_start_cursor_swarm",
        )
        # Hermes is not Cursor → gets the platform-portable verb, never a
        # cursor-specific one it can't invoke.
        self.assertEqual(
            verb_for_host("puppetmaster_start_cursor_implement", "hermes"),
            "puppetmaster_start_implement",
        )

    def test_hermes_pre_llm_call_injects_context_for_single_implement(self):
        """Hermes' pre_llm_call carries user_message under `extra`; a single,
        focused edit intent must inject a delegate directive as {"context": ...}
        and steer to the lightweight in-place ``edit`` verb (not a cursor verb,
        not a fan-out swarm)."""
        from puppetmaster.hook_runner import handle_hook

        payload = {
            "hook_event_name": "pre_llm_call",
            "session_id": "s1",
            "cwd": "/repo",
            "extra": {
                "user_message": "add a --verbose flag to the savings command and wire it through",
                "is_first_turn": True,
            },
        }
        r = handle_hook(payload, host="hermes", event="pre_llm_call", env=self._TOOLS_ON)
        self.assertEqual(r.action, "allow")
        self.assertTrue(r.decision.should_delegate)
        self.assertEqual(r.decision.suggested_verb, "puppetmaster_edit")
        out = r.to_host_json("hermes")
        self.assertIn("context", out)
        self.assertIn("Puppetmaster", out["context"])
        self.assertIn("puppetmaster_edit", out["context"])
        self.assertNotIn("cursor", out["context"])

    def test_hermes_pre_llm_call_noop_for_trivial_edit(self):
        """A trivial edit stays inline: Hermes must receive an empty {} (which
        its shell-hook bridge treats as a silent no-op), never a directive."""
        from puppetmaster.hook_runner import handle_hook

        payload = {
            "hook_event_name": "pre_llm_call",
            "extra": {"user_message": "fix a typo in the README header"},
        }
        r = handle_hook(payload, host="hermes", event="pre_llm_call", env=self._TOOLS_ON)
        self.assertEqual(r.action, "allow")
        self.assertFalse(r.decision.should_delegate)
        self.assertEqual(r.to_host_json("hermes"), {})

    def test_hermes_pre_tool_call_blocks_broad_search(self):
        """A genuinely recursive shell search on Hermes is deny-redirected using
        Hermes' canonical block shape {"action": "block", "message": ...}.

        Matches Hermes' real wire shape from agent/shell_hooks._serialize_payload:
        top-level `tool_name` + `tool_input` (the tool's args dict).
        """
        from puppetmaster.hook_runner import handle_hook

        payload = {
            "hook_event_name": "pre_tool_call",
            "tool_name": "terminal",
            "tool_input": {"command": "grep -R TODO ."},
            "session_id": "s1",
            "cwd": "/repo",
        }
        r = handle_hook(payload, host="hermes", event="pre_tool_call", env=self._TOOLS_ON)
        self.assertEqual(r.action, "deny")
        out = r.to_host_json("hermes")
        self.assertEqual(out["action"], "block")
        self.assertIn("puppetmaster_codegraph_search", out["message"])

    def test_hermes_host_is_noop_when_no_mcp_server(self):
        """No alive Puppetmaster MCP server → the Hermes hook is a pure no-op,
        never injecting a directive for a verb the agent can't call."""
        from puppetmaster.hook_runner import handle_hook

        payload = {
            "hook_event_name": "pre_llm_call",
            "extra": {"user_message": "refactor the auth module across all files"},
        }
        r = handle_hook(payload, host="hermes", event="pre_llm_call", env=self._TOOLS_OFF)
        self.assertEqual(r.action, "allow")
        self.assertEqual(r.to_host_json("hermes"), {})

    def test_pre_tool_allows_native_grep_glob(self):
        # Field fix: native search tools are read-only inspection in disguise as
        # often as not, and their scope isn't visible in the hook payload — so we
        # never hard-deny them (that wedged legitimate work).
        from puppetmaster.hook_runner import handle_hook

        for tool in ("Grep", "Glob", "codebase_search"):
            r = handle_hook({"tool_name": tool}, host="cursor", event="pre-tool")
            self.assertEqual(r.action, "allow", tool)

    def test_pre_tool_allows_readonly_shell_inspection(self):
        from puppetmaster.hook_runner import classify_tool

        for cmd in ("git log", "git log --oneline -20", "git show HEAD",
                    "git diff HEAD~1", "ls ~/.cursor", "cat foo.py | head -40",
                    "grep TODO app.py", "rg pattern src/app.ts"):
            redirect, _ = classify_tool("shell", cmd)
            self.assertFalse(redirect, cmd)

    def test_pre_tool_still_redirects_recursive_shell_search(self):
        from puppetmaster.hook_runner import classify_tool

        for cmd in ("rg -r TODO ./src", "grep -R foo .", "find . -name '*.py'"):
            redirect, verb = classify_tool("shell", cmd)
            self.assertTrue(redirect, cmd)
            self.assertTrue(verb.startswith("puppetmaster_"))

    def test_pre_tool_allows_puppetmaster_tools(self):
        from puppetmaster.hook_runner import handle_hook

        r = handle_hook(
            {"tool_name": "mcp__puppetmaster__codegraph_search"},
            host="cursor", event="pre-tool",
        )
        self.assertEqual(r.action, "allow")

    def test_pre_tool_redirects_broad_shell_search(self):
        from puppetmaster.hook_runner import classify_tool

        redirect, verb = classify_tool("shell", "rg -r 'TODO' ./src")
        self.assertTrue(redirect)
        self.assertTrue(verb.startswith("puppetmaster_"))

    def test_kill_switch_allows_everything(self):
        from puppetmaster.hook_runner import handle_hook

        r = handle_hook(
            {"tool_name": "Grep"}, host="cursor", event="pre-tool",
            env={"PUPPETMASTER_AUTO_INVOKE_DISABLED": "1"},
        )
        self.assertEqual(r.action, "allow")

    def test_run_reads_stdin_and_emits_json(self):
        from puppetmaster.hook_runner import run

        stdin = io.StringIO(json.dumps({"prompt": "audit the whole repo for races"}))
        stdout = io.StringIO()
        rc = run(
            ["--host", "cursor", "--event", "user-prompt"],
            stdin=stdin, stdout=stdout,
            env={"PUPPETMASTER_HOOK_ASSUME_TOOLS": "1"},
        )
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("additionalContext", payload)

    def test_run_fails_open_on_garbage_stdin(self):
        from puppetmaster.hook_runner import run

        stdout = io.StringIO()
        rc = run(["--host", "cursor", "--event", "user-prompt"], stdin=io.StringIO("not json"), stdout=stdout)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue())["permission"], "allow")


class ProviderProxyTests(unittest.TestCase):
    """Tests for the OpenAI-compatible enforcement proxy (pure logic)."""

    def test_extract_user_prompt_from_messages(self):
        from puppetmaster.provider_proxy import extract_user_prompt

        body = {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "audit the repo"}]}
        self.assertEqual(extract_user_prompt(body), "audit the repo")

    def test_extract_user_prompt_structured_content(self):
        from puppetmaster.provider_proxy import extract_user_prompt

        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]}]}
        self.assertEqual(extract_user_prompt(body), "hi\nthere")

    def test_transform_injects_directive_when_delegating(self):
        from puppetmaster.provider_proxy import transform_chat_request

        body = {"messages": [{"role": "user", "content": "refactor the database layer across all modules"}]}
        new_body, decision = transform_chat_request(body)
        self.assertTrue(decision.should_delegate)
        self.assertEqual(new_body["messages"][0]["role"], "system")
        self.assertIn("Puppetmaster", new_body["messages"][0]["content"])

    def test_transform_passthrough_when_inline(self):
        from puppetmaster.provider_proxy import transform_chat_request

        body = {"messages": [{"role": "user", "content": "fix a typo"}]}
        new_body, decision = transform_chat_request(body)
        self.assertFalse(decision.should_delegate)
        self.assertEqual(new_body["messages"], body["messages"])

    def test_upstream_allowlist(self):
        from puppetmaster.provider_proxy import is_upstream_allowed

        self.assertTrue(is_upstream_allowed("https://api.openai.com/v1"))
        self.assertTrue(is_upstream_allowed("http://127.0.0.1:9999"))
        self.assertFalse(is_upstream_allowed("https://evil.example.com"))
        self.assertFalse(is_upstream_allowed("http://api.openai.com"))  # not https

    def test_advice_response_shape(self):
        from puppetmaster.invocation_gate import should_delegate
        from puppetmaster.provider_proxy import build_advice_response

        decision = should_delegate("audit the repo")
        resp = build_advice_response(decision)
        self.assertEqual(resp["object"], "chat.completion")
        self.assertIn("Puppetmaster", resp["choices"][0]["message"]["content"])


class HookInstallerTests(unittest.TestCase):
    """Tests for the deterministic hook installer (idempotent + non-destructive)."""

    def test_install_writes_cursor_and_claude(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(cwd=cwd)
            self.assertEqual(result.overall_status, "installed")
            cursor = json.loads((cwd / ".cursor" / "hooks.json").read_text())
            self.assertIn("beforeSubmitPrompt", cursor["hooks"])
            self.assertIn("invocation-gate", json.dumps(cursor))
            claude = json.loads((cwd / ".claude" / "settings.json").read_text())
            self.assertIn("UserPromptSubmit", claude["hooks"])
            self.assertIn("PreToolUse", claude["hooks"])

    def test_install_is_idempotent(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            install_hooks(cwd=cwd)
            again = install_hooks(cwd=cwd)
            self.assertEqual(again.overall_status, "unchanged")

    def test_install_filters_to_enabled_cursor_only(self):
        """A cursor-only lock must not write .claude/settings.json hooks."""
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(cwd=cwd, enabled_adapters={"cursor"})
            self.assertEqual(result.overall_status, "installed")
            self.assertTrue((cwd / ".cursor" / "hooks.json").exists())
            self.assertFalse((cwd / ".claude" / "settings.json").exists())
            self.assertEqual({o.target for o in result.outcomes}, {"cursor"})

    def test_install_filters_to_enabled_claude_only(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(cwd=cwd, enabled_adapters={"claude-code"})
            self.assertTrue((cwd / ".claude" / "settings.json").exists())
            self.assertFalse((cwd / ".cursor" / "hooks.json").exists())
            self.assertEqual({o.target for o in result.outcomes}, {"claude"})

    def test_install_with_no_enabled_platforms_writes_nothing(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(cwd=cwd, enabled_adapters=set())
            self.assertEqual(result.overall_status, "unchanged")
            self.assertFalse((cwd / ".cursor" / "hooks.json").exists())
            self.assertFalse((cwd / ".claude" / "settings.json").exists())

    def test_explicit_targets_override_enabled_filter(self):
        """Explicit targets win over the lock filter (explicit intent)."""
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(
                cwd=cwd, targets=["claude"], enabled_adapters={"cursor"}
            )
            self.assertTrue((cwd / ".claude" / "settings.json").exists())
            self.assertFalse((cwd / ".cursor" / "hooks.json").exists())

    def test_install_preserves_user_hooks(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            cursor_path = cwd / ".cursor" / "hooks.json"
            cursor_path.parent.mkdir(parents=True)
            cursor_path.write_text(json.dumps({
                "version": 1,
                "hooks": {"beforeSubmitPrompt": [{"command": "my-own-hook"}]},
            }))
            install_hooks(cwd=cwd, targets=["cursor"])
            data = json.loads(cursor_path.read_text())
            commands = json.dumps(data["hooks"]["beforeSubmitPrompt"])
            self.assertIn("my-own-hook", commands)
            self.assertIn("invocation-gate", commands)

    def test_dry_run_writes_nothing(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            result = install_hooks(cwd=cwd, dry_run=True)
            self.assertEqual(result.overall_status, "would_install")
            self.assertFalse((cwd / ".cursor" / "hooks.json").exists())

    def test_global_scope_writes_under_home_not_cwd(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as proj, TemporaryDirectory() as home_tmp:
            cwd, home = Path(proj), Path(home_tmp)
            result = install_hooks(cwd=cwd, scope="global", home=home)
            self.assertEqual(result.overall_status, "installed")
            # global lands under ~ , never under the workspace
            self.assertTrue((home / ".cursor" / "hooks.json").exists())
            self.assertTrue((home / ".claude" / "settings.json").exists())
            self.assertFalse((cwd / ".cursor" / "hooks.json").exists())

    def test_global_scope_is_idempotent_and_labeled(self):
        from puppetmaster.hook_installers import install_hooks

        with TemporaryDirectory() as home_tmp:
            home = Path(home_tmp)
            install_hooks(scope="global", home=home)
            again = install_hooks(scope="global", home=home)
            self.assertEqual(again.overall_status, "unchanged")
            self.assertTrue(any("~/.cursor/hooks.json" in o.reason for o in again.outcomes))

    def test_unknown_scope_is_an_error(self):
        from puppetmaster.hook_installers import install_hooks

        result = install_hooks(scope="bogus")
        self.assertEqual(result.overall_status, "error")

    def test_gate_command_uses_forward_slashes_for_windows_python(self):
        """Windows python paths must survive the host's POSIX-shell hook exec.

        Field report (Claude Code on Windows): the backslashed path written
        by setup lost every backslash by the time the hook ran — Git Bash
        treats unquoted backslashes as escapes. Forward slashes are valid
        Windows separators and pass through every layer unmangled.
        """
        from puppetmaster.hook_installers import _gate_command

        command = _gate_command(
            "claude", "pre-tool", python=r"C:\Users\pawel\AppData\Local\Programs\Python\python.exe"
        )
        self.assertNotIn("\\", command)
        self.assertTrue(
            command.startswith("C:/Users/pawel/AppData/Local/Programs/Python/python.exe ")
        )

    def test_gate_command_quotes_paths_with_spaces(self):
        from puppetmaster.hook_installers import _gate_command

        command = _gate_command(
            "claude", "user-prompt", python=r"C:\Program Files\Python312\python.exe"
        )
        self.assertTrue(command.startswith('"C:/Program Files/Python312/python.exe" -m'))

    def test_gate_command_leaves_posix_paths_untouched(self):
        from puppetmaster.hook_installers import _gate_command

        command = _gate_command("cursor", "user-prompt", python="/usr/local/bin/python3")
        self.assertTrue(
            command.startswith("/usr/local/bin/python3 -m puppetmaster invocation-gate")
        )


class HermesHookInstallerTests(unittest.TestCase):
    """Tests for the Hermes YAML hooks installer (idempotent + non-destructive)."""

    def setUp(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed; install puppetmaster-ai[hermes]")

    def _load(self, path):
        import yaml
        return yaml.safe_load(path.read_text("utf-8"))

    def test_install_writes_pre_llm_and_pre_tool_hooks(self):
        from puppetmaster.hook_installers import install_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            outcome = install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            self.assertEqual(outcome.status, "installed")
            cfg = self._load(target)
            hooks = cfg["hooks"]
            self.assertIn("pre_llm_call", hooks)
            self.assertIn("pre_tool_call", hooks)
            cmd = hooks["pre_llm_call"][0]["command"]
            self.assertIn("puppetmaster invocation-gate", cmd)
            self.assertIn("--host hermes", cmd)
            self.assertIn("--event user-prompt", cmd)
            self.assertIn("--event pre-tool", hooks["pre_tool_call"][0]["command"])

    def test_install_is_idempotent(self):
        from puppetmaster.hook_installers import install_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            again = install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            self.assertEqual(again.status, "unchanged")

    def test_install_preserves_other_config_and_user_hooks(self):
        """Must merge into an existing config.yaml: keep mcp_servers, other
        sections, and any user-authored hook entries untouched."""
        import yaml
        from puppetmaster.hook_installers import install_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            target.write_text(yaml.safe_dump({
                "mcp_servers": {"puppetmaster": {"command": "python", "args": ["-m", "puppetmaster.mcp_server"]}},
                "model": "anthropic/claude-opus-4-8",
                "hooks": {
                    "pre_tool_call": [
                        {"matcher": "terminal", "command": "~/.hermes/agent-hooks/block-rm.sh"}
                    ]
                },
            }), encoding="utf-8")
            outcome = install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            self.assertEqual(outcome.status, "installed")
            cfg = self._load(target)
            # Untouched config preserved.
            self.assertEqual(cfg["model"], "anthropic/claude-opus-4-8")
            self.assertIn("puppetmaster", cfg["mcp_servers"])
            # User's pre_tool_call hook survives alongside ours.
            cmds = [e.get("command", "") for e in cfg["hooks"]["pre_tool_call"]]
            self.assertIn("~/.hermes/agent-hooks/block-rm.sh", cmds)
            self.assertTrue(any("invocation-gate" in c for c in cmds))
            self.assertIn("pre_llm_call", cfg["hooks"])

    def test_install_into_default_empty_hooks_block(self):
        """Hermes ships `hooks: {}` by default — installer must populate it."""
        import yaml
        from puppetmaster.hook_installers import install_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            target.write_text(yaml.safe_dump({"hooks": {}, "model": "x"}), encoding="utf-8")
            outcome = install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            self.assertEqual(outcome.status, "installed")
            cfg = self._load(target)
            self.assertIn("pre_llm_call", cfg["hooks"])

    def test_uninstall_removes_only_our_hooks(self):
        import yaml
        from puppetmaster.hook_installers import install_hermes_hooks, uninstall_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            target.write_text(yaml.safe_dump({
                "hooks": {"pre_tool_call": [{"matcher": "terminal", "command": "~/keep.sh"}]},
            }), encoding="utf-8")
            install_hermes_hooks(target_path=target, python="/usr/bin/python3")
            outcome = uninstall_hermes_hooks(target_path=target)
            self.assertEqual(outcome.status, "removed")
            cfg = self._load(target)
            # Ours gone, user's kept, pre_llm_call (ours-only) dropped.
            self.assertNotIn("pre_llm_call", cfg["hooks"])
            cmds = [e.get("command", "") for e in cfg["hooks"]["pre_tool_call"]]
            self.assertEqual(cmds, ["~/keep.sh"])

    def test_uninstall_is_idempotent_and_noop_without_config(self):
        from puppetmaster.hook_installers import uninstall_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            outcome = uninstall_hermes_hooks(target_path=target)
            self.assertEqual(outcome.status, "unchanged")

    def test_dry_run_writes_nothing(self):
        from puppetmaster.hook_installers import install_hermes_hooks

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            outcome = install_hermes_hooks(target_path=target, dry_run=True, python="/usr/bin/python3")
            self.assertEqual(outcome.status, "would_install")
            self.assertFalse(target.exists())


class HostGatedIdleReapTests(unittest.TestCase):
    """Idle reaping only fires on hosts that transparently respawn (Cursor).

    Field reports: Codex "Transport closed" and Claude Code "Connection
    status failed" after an idle gap — both were the staleness watcher
    reaping a server its host would never respawn.
    """

    def test_cursor_markers_enable_idle_reap(self):
        from puppetmaster.mcp_server import _host_transparently_respawns

        self.assertTrue(_host_transparently_respawns({"CURSOR_TRACE_ID": "abc"}))
        self.assertTrue(_host_transparently_respawns({"CURSOR_AGENT": "1", "PATH": "/usr/bin"}))

    def test_claude_and_codex_markers_disable_idle_reap(self):
        from puppetmaster.mcp_server import _host_transparently_respawns

        # Claude/Codex markers win even when Cursor markers are also present
        # (e.g. env inherited through nested tooling) — bias toward not reaping.
        self.assertFalse(_host_transparently_respawns({"CLAUDECODE": "1", "CURSOR_TRACE_ID": "x"}))
        self.assertFalse(_host_transparently_respawns({"CLAUDE_CODE_ENTRYPOINT": "cli"}))
        self.assertFalse(_host_transparently_respawns({"CODEX_SANDBOX": "1", "CURSOR_AGENT": "1"}))

    def test_unknown_host_never_idle_reaps(self):
        from puppetmaster.mcp_server import _host_transparently_respawns

        self.assertFalse(_host_transparently_respawns({"PATH": "/usr/bin"}))
        self.assertFalse(_host_transparently_respawns({}))

    def test_watcher_skips_idle_reap_when_disabled_for_host(self):
        from puppetmaster import mcp_server

        with mcp_server._INPUT_STATE_LOCK:
            prior = (mcp_server._LAST_INBOUND_MESSAGE_AT, mcp_server._ACTIVE_TOOL_CALLS)
            mcp_server._LAST_INBOUND_MESSAGE_AT = time.time() - 99999
            mcp_server._ACTIVE_TOOL_CALLS = 0
        try:
            stale_args = dict(
                stale_after_seconds=1.0,
                check_interval_seconds=1.0,
                on_shutdown=lambda: None,
            )
            gated = mcp_server._InputStalenessWatcher(reap_enabled=False, **stale_args)
            open_reap = mcp_server._InputStalenessWatcher(reap_enabled=True, **stale_args)
            self.assertFalse(gated._should_reap())
            self.assertTrue(open_reap._should_reap())
        finally:
            with mcp_server._INPUT_STATE_LOCK:
                mcp_server._LAST_INBOUND_MESSAGE_AT, mcp_server._ACTIVE_TOOL_CALLS = prior

    def test_parent_death_reap_is_host_gated(self):
        """Regression: a reparented-to-init (``getppid()==1``) server on a
        non-respawn host (Codex/Claude) must NOT self-reap. An exiting
        intermediate launcher reparents us to init while the real owner is
        alive and still holds the pipe; reaping there closes stdin and the
        host sees a false "Transport closed". stdin EOF is the authoritative
        owner-death signal on those hosts. Reap stays on for respawn hosts."""
        from puppetmaster import mcp_server

        with patch.object(
            mcp_server._InputStalenessWatcher, "_parent_is_dead", staticmethod(lambda: True)
        ):
            codex_like = mcp_server._InputStalenessWatcher(
                stale_after_seconds=99999.0,
                check_interval_seconds=1.0,
                on_shutdown=lambda: None,
                reap_enabled=False,
            )
            cursor_like = mcp_server._InputStalenessWatcher(
                stale_after_seconds=99999.0,
                check_interval_seconds=1.0,
                on_shutdown=lambda: None,
                reap_enabled=True,
            )
            self.assertFalse(codex_like._should_reap())
            self.assertTrue(cursor_like._should_reap())

    def test_codex_handshake_suppresses_reap_despite_respawn_env(self):
        """Regression: Codex scrubs CODEX_* from our env, so a Codex session
        launched from a Cursor terminal leaks CURSOR_* and freezes
        ``reap_enabled=True`` at startup. Once the handshake identifies a Codex
        client, parent-death must NOT reap — otherwise the server self-kills
        and Codex reports "Transport closed"."""
        from puppetmaster import mcp_server

        # Env-derived gate said "respawn host" (looks like Cursor).
        watcher = mcp_server._InputStalenessWatcher(
            stale_after_seconds=99999.0,
            check_interval_seconds=1.0,
            on_shutdown=lambda: None,
            reap_enabled=True,
        )
        prior_client = dict(mcp_server._CLIENT_INFO)
        try:
            mcp_server._CLIENT_INFO.clear()
            mcp_server._CLIENT_INFO.update({"name": "codex-mcp-client", "title": "Codex"})
            with patch.object(
                mcp_server._InputStalenessWatcher, "_parent_is_dead", staticmethod(lambda: True)
            ):
                self.assertFalse(watcher._should_reap())
        finally:
            mcp_server._CLIENT_INFO.clear()
            mcp_server._CLIENT_INFO.update(prior_client)

    def test_forced_flag_reenables_idle_reap(self):
        from puppetmaster.mcp_server import _input_staleness_forced

        os.environ["PUPPETMASTER_MCP_INPUT_STALE_FORCED"] = "1"
        try:
            self.assertTrue(_input_staleness_forced())
        finally:
            del os.environ["PUPPETMASTER_MCP_INPUT_STALE_FORCED"]
        self.assertFalse(_input_staleness_forced())


class WorktreePreflightTests(unittest.TestCase):
    """Full-edit MCP verbs refuse non-git cwds at the verb, not after spawn."""

    def test_non_worktree_cwd_fails_fast_with_remediation(self):
        from puppetmaster.mcp_server import _worktree_preflight

        with TemporaryDirectory() as tmp:
            result = _worktree_preflight({"cwd": tmp})
            self.assertIsNotNone(result)
            self.assertTrue(result.get("isError"))
            text = result["content"][0]["text"]
            self.assertIn("not_a_worktree", text)
            self.assertIn("git init", text)
            self.assertIn("allow_non_worktree", text)

    def test_allow_non_worktree_skips_preflight(self):
        from puppetmaster.mcp_server import _worktree_preflight

        with TemporaryDirectory() as tmp:
            self.assertIsNone(_worktree_preflight({"cwd": tmp, "allow_non_worktree": True}))

    def test_git_repo_cwd_passes_preflight(self):
        from puppetmaster.mcp_server import _worktree_preflight

        with TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True, capture_output=True)
            self.assertIsNone(_worktree_preflight({"cwd": tmp}))

    def test_codex_read_only_sandbox_is_exempt(self):
        from puppetmaster.mcp_server import _codex_is_write_capable

        self.assertFalse(_codex_is_write_capable({"sandbox": "read-only"}))
        self.assertTrue(_codex_is_write_capable({}))  # default workspace-write
        self.assertTrue(
            _codex_is_write_capable(
                {"sandbox": "read-only", "dangerously_bypass_approvals_and_sandbox": True}
            )
        )


class InvocationGateCliTests(unittest.TestCase):
    """The should-delegate / install-hooks CLI surface."""

    def test_should_delegate_cli_json(self):
        from puppetmaster.cli import main as cli_main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["should-delegate", "audit the whole repo for security holes", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["should_delegate"])

    def test_install_hooks_cli(self):
        from puppetmaster.cli import main as cli_main

        with TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                os.chdir(tmp)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = cli_main(["install-hooks", "--target", "cursor"])
                self.assertEqual(rc, 0)
                self.assertTrue((Path(tmp) / ".cursor" / "hooks.json").exists())
            finally:
                os.chdir(cwd)

    def test_install_hooks_cli_global_targets_home(self):
        from puppetmaster.cli import main as cli_main
        from puppetmaster import hook_installers

        with TemporaryDirectory() as proj, TemporaryDirectory() as home_tmp:
            home = Path(home_tmp)
            cwd = Path.cwd()
            try:
                os.chdir(proj)
                buf = io.StringIO()
                with patch.object(hook_installers.Path, "home", return_value=home), \
                        contextlib.redirect_stdout(buf):
                    rc = cli_main(["install-hooks", "--global", "--target", "claude"])
                self.assertEqual(rc, 0)
                self.assertIn("scope=global", buf.getvalue())
                self.assertTrue((home / ".claude" / "settings.json").exists())
                self.assertFalse((Path(proj) / ".claude" / "settings.json").exists())
            finally:
                os.chdir(cwd)


class SetupHooksStepTests(unittest.TestCase):
    """The `setup` wizard installs auto-invocation hooks as step 7."""

    def _args(self, **over):
        base = dict(
            skip_doctor=True, skip_models=True, skip_platforms=True,
            skip_rules=True, skip_hooks=False, global_rules=False,
            global_hooks=False, force=False, state_dir=None, platforms=None,
        )
        base.update(over)
        return MagicMock(**base)

    def test_setup_installs_hooks(self):
        import puppetmaster.cli as cli

        class _Res:
            status = "installed"
            messages: list = []

        with TemporaryDirectory() as tmp:
            cwd0 = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(cli, "install_cursor_mcp", return_value=_Res()), \
                        patch.object(cli, "install_codex_mcp", return_value=_Res()), \
                        patch.object(cli, "install_claude_mcp", return_value=_Res()), \
                        patch.object(cli, "resolve_claude_command", return_value="claude"), \
                        patch("puppetmaster.platform_lock.enabled_adapters",
                              return_value={"cursor", "claude-code"}):
                    rc = cli._run_setup(self._args())
                self.assertEqual(rc, 0)
                self.assertTrue((Path(tmp) / ".cursor" / "hooks.json").exists())
                self.assertTrue((Path(tmp) / ".claude" / "settings.json").exists())
            finally:
                os.chdir(cwd0)

    def test_setup_skip_hooks_writes_no_hooks(self):
        import puppetmaster.cli as cli

        class _Res:
            status = "installed"
            messages: list = []

        with TemporaryDirectory() as tmp:
            cwd0 = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(cli, "install_cursor_mcp", return_value=_Res()), \
                        patch.object(cli, "install_codex_mcp", return_value=_Res()), \
                        patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor"}):
                    rc = cli._run_setup(self._args(skip_hooks=True))
                self.assertEqual(rc, 0)
                self.assertFalse((Path(tmp) / ".cursor" / "hooks.json").exists())
            finally:
                os.chdir(cwd0)

    def test_setup_global_hooks_writes_under_home(self):
        import puppetmaster.cli as cli
        from puppetmaster import hook_installers

        class _Res:
            status = "installed"
            messages: list = []

        with TemporaryDirectory() as tmp, TemporaryDirectory() as home_tmp:
            cwd0 = Path.cwd()
            home = Path(home_tmp)
            try:
                os.chdir(tmp)
                with patch.object(cli, "install_cursor_mcp", return_value=_Res()), \
                        patch.object(cli, "install_codex_mcp", return_value=_Res()), \
                        patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor"}), \
                        patch.object(hook_installers.Path, "home", return_value=home):
                    rc = cli._run_setup(self._args(global_hooks=True))
                self.assertEqual(rc, 0)
                self.assertTrue((home / ".cursor" / "hooks.json").exists())
                self.assertFalse((Path(tmp) / ".cursor" / "hooks.json").exists())
            finally:
                os.chdir(cwd0)

    def test_setup_installs_hermes_hooks_when_hermes_enabled(self):
        """The wizard's Hermes step must wire Hermes' native shell hooks too —
        parity with the Cursor/Claude hooks in the final step. Regression guard:
        early versions only installed the Hermes MCP server and silently skipped
        its auto-invocation hooks, so `setup` left Hermes a non-delegating host.
        """
        import puppetmaster.cli as cli

        class _Res:
            status = "installed"
            messages: list = []

        captured = {"hermes_hooks": 0}

        def fake_hermes_hooks(**kwargs):
            captured["hermes_hooks"] += 1
            return MagicMock(status="installed", reason="wrote hermes hooks")

        with TemporaryDirectory() as tmp:
            cwd0 = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(cli, "install_cursor_mcp", return_value=_Res()), \
                        patch.object(cli, "install_codex_mcp", return_value=_Res()), \
                        patch.object(cli, "install_hermes_mcp", return_value=_Res()), \
                        patch.object(cli, "install_hermes_hooks", side_effect=fake_hermes_hooks), \
                        patch.object(cli, "_seed_hermes_registry"), \
                        patch("shutil.which", return_value="/usr/local/bin/hermes"), \
                        patch("puppetmaster.platform_lock.enabled_adapters",
                              return_value={"hermes"}):
                    rc = cli._run_setup(self._args())
                self.assertEqual(rc, 0)
                self.assertEqual(
                    captured["hermes_hooks"], 1,
                    "setup must install Hermes hooks when hermes is enabled",
                )
            finally:
                os.chdir(cwd0)

    def test_setup_skip_hooks_skips_hermes_hooks_too(self):
        """--skip-hooks must suppress the Hermes hooks too, not just Cursor/Claude."""
        import puppetmaster.cli as cli

        class _Res:
            status = "installed"
            messages: list = []

        captured = {"hermes_hooks": 0}

        with TemporaryDirectory() as tmp:
            cwd0 = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(cli, "install_hermes_mcp", return_value=_Res()), \
                        patch.object(cli, "install_hermes_hooks",
                                     side_effect=lambda **k: captured.__setitem__(
                                         "hermes_hooks", captured["hermes_hooks"] + 1)), \
                        patch.object(cli, "_seed_hermes_registry"), \
                        patch("shutil.which", return_value="/usr/local/bin/hermes"), \
                        patch("puppetmaster.platform_lock.enabled_adapters",
                              return_value={"hermes"}):
                    rc = cli._run_setup(self._args(skip_hooks=True))
                self.assertEqual(rc, 0)
                self.assertEqual(captured["hermes_hooks"], 0)
            finally:
                os.chdir(cwd0)


class UninstallTests(unittest.TestCase):
    """Tests for ``puppetmaster uninstall`` and its removal helpers."""

    def test_strip_block_preserves_surrounding_agents_content(self):
        from puppetmaster.rules import (
            BEGIN_MARKER,
            END_MARKER,
            render_agents_block,
            strip_block_from_text,
        )

        user_header = "# Project conventions\n\n- Keep tests green\n\n"
        user_footer = "\n# Trailing notes\n"
        existing = user_header + render_agents_block() + user_footer
        stripped, action = strip_block_from_text(existing)
        self.assertEqual(action, "removed")
        self.assertEqual(stripped, user_header + user_footer)
        self.assertNotIn(BEGIN_MARKER, stripped)
        self.assertNotIn(END_MARKER, stripped)

    def test_uninstall_cursor_mcp_preserves_other_servers(self):
        from puppetmaster.installers import install_cursor_mcp, uninstall_cursor_mcp

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            target.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "navdata": {"url": "https://example.com/sse"},
                            "puppetmaster": {
                                "command": sys.executable,
                                "args": ["-m", "puppetmaster.mcp_server"],
                            },
                        }
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            result = uninstall_cursor_mcp(target_path=target)
            self.assertEqual(result.status, "removed")
            data = json.loads(target.read_text("utf-8"))
            self.assertNotIn("puppetmaster", data["mcpServers"])
            self.assertEqual(data["mcpServers"]["navdata"]["url"], "https://example.com/sse")

    def test_remove_codex_puppetmaster_table(self):
        from puppetmaster.installers import _remove_codex_puppetmaster_table

        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[mcp_servers.other]\ncommand = \"echo\"\n\n"
                "[mcp_servers.puppetmaster]\n"
                "command = \"python\"\n"
                "args = [\"-m\", \"puppetmaster.mcp_server\"]\n"
                "tool_timeout_sec = 300\n\n"
                "[features]\nenabled = true\n",
                encoding="utf-8",
            )
            removed, messages = _remove_codex_puppetmaster_table(config)
            self.assertTrue(removed)
            text = config.read_text("utf-8")
            self.assertNotIn("[mcp_servers.puppetmaster]", text)
            self.assertIn("[mcp_servers.other]", text)
            self.assertIn("[features]", text)
            again, _ = _remove_codex_puppetmaster_table(config)
            self.assertFalse(again)

    def test_uninstall_codex_mcp_removes_managed_wrapper_files(self):
        from puppetmaster.installers import uninstall_codex_mcp

        with TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "codex"
            codex_home.mkdir()
            config = codex_home / "config.toml"
            config.write_text(
                "[mcp_servers.puppetmaster]\n"
                f'command = "{sys.executable}"\n'
                'args = ["-m", "puppetmaster.mcp_server"]\n',
                encoding="utf-8",
            )
            wrapper = Path(tmp) / "codex-mcp-wrapper.py"
            managed_env = Path(tmp) / "codex-mcp.env.json"
            wrapper.write_text("# managed\n", encoding="utf-8")
            managed_env.write_text("{}\n", encoding="utf-8")
            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False), \
                    patch("puppetmaster.installers._CODEX_WRAPPER_PATH", wrapper), \
                    patch("puppetmaster.installers._CODEX_MANAGED_ENV_PATH", managed_env):
                result = uninstall_codex_mcp(codex_executable="/nonexistent/codex")
            self.assertEqual(result.status, "removed", msg=result.messages)
            self.assertFalse(wrapper.exists())
            self.assertFalse(managed_env.exists())
            self.assertNotIn("[mcp_servers.puppetmaster]", config.read_text("utf-8"))

    def test_uninstall_hooks_preserves_foreign_hooks(self):
        from puppetmaster.hook_installers import install_hooks, uninstall_hooks

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            foreign = {
                "version": 1,
                "hooks": {
                    "beforeSubmitPrompt": [{"command": "echo user-hook"}],
                },
            }
            hooks_path = cwd / ".cursor" / "hooks.json"
            hooks_path.parent.mkdir(parents=True)
            hooks_path.write_text(json.dumps(foreign, indent=2) + "\n", encoding="utf-8")
            install_hooks(cwd=cwd, targets=["cursor"], scope="project")
            result = uninstall_hooks(cwd=cwd, targets=["cursor"], scopes=["project"])
            self.assertIn(result.overall_status, {"removed", "unchanged"})
            data = json.loads(hooks_path.read_text("utf-8"))
            self.assertEqual(data["hooks"]["beforeSubmitPrompt"], [{"command": "echo user-hook"}])

    def test_uninstall_is_idempotent(self):
        from puppetmaster.installers import install_cursor_mcp, uninstall_cursor_mcp
        from puppetmaster.rules import install_rules, uninstall_rules

        with TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            target = cwd / ".cursor" / "mcp.json"
            install_rules(cwd=cwd, targets=["cursor", "agents"])
            install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            uninstall_rules(cwd=cwd)
            uninstall_cursor_mcp(target_path=target)
            rules_again = uninstall_rules(cwd=cwd)
            mcp_again = uninstall_cursor_mcp(target_path=target)
            self.assertEqual(rules_again.overall_status, "unchanged")
            self.assertEqual(mcp_again.status, "unchanged")

    def test_uninstall_dry_run_writes_nothing(self):
        from puppetmaster import cli
        from puppetmaster.installers import install_cursor_mcp
        from puppetmaster.rules import install_rules

        with TemporaryDirectory() as tmp, TemporaryDirectory() as home_tmp:
            cwd = Path(tmp)
            (cwd / ".git").mkdir()
            target = cwd / ".cursor" / "mcp.json"
            install_rules(cwd=cwd, targets=["cursor", "agents"])
            install_cursor_mcp(
                target_path=target,
                python_executable=sys.executable,
                skip_handshake=True,
            )
            agents_before = (cwd / "AGENTS.md").read_text("utf-8")
            mcp_before = target.read_text("utf-8")
            cwd0 = os.getcwd()
            try:
                os.chdir(cwd)
                with patch.object(cli.Path, "home", return_value=Path(home_tmp)):
                    rc = cli._run_uninstall(
                        argparse.Namespace(
                            cwd=str(cwd),
                            dry_run=True,
                            purge_state=False,
                            yes=True,
                        )
                    )
                self.assertEqual(rc, 0)
            finally:
                os.chdir(cwd0)
            self.assertEqual((cwd / "AGENTS.md").read_text("utf-8"), agents_before)
            self.assertEqual(target.read_text("utf-8"), mcp_before)
            self.assertTrue((cwd / ".cursor" / "rules" / "puppetmaster.mdc").is_file())


class AuditFixTests(unittest.TestCase):
    def test_sqlite_concurrent_claim_has_exactly_one_winner(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("race claim")
            task = Task(job_id=job.id, role="coder", instruction="work")
            store.save_task(task)

            barrier = threading.Barrier(2)
            results: list[Optional[Task]] = []
            lock = threading.Lock()

            def claim(worker_id: str) -> None:
                barrier.wait()
                claimed = store.claim_task(task.id, worker_id, lease_seconds=60)
                with lock:
                    results.append(claimed)

            threads = [
                threading.Thread(target=claim, args=("worker-a",)),
                threading.Thread(target=claim, args=("worker-b",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            winners = [claimed for claimed in results if claimed is not None]
            self.assertEqual(len(winners), 1)
            self.assertEqual(store.get_task_by_id(task.id).lease_owner, winners[0].lease_owner)

    def test_lease_fenced_terminal_write_conflict(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("fence terminal writes")
            task = Task(job_id=job.id, role="coder", instruction="work")
            store.save_task(task)
            claimed = store.claim_task(task.id, "worker-a", lease_seconds=60)
            self.assertIsNotNone(claimed)

            conflict = store.update_task_status(
                claimed, TaskStatus.COMPLETE, worker_id="worker-b"
            )
            self.assertEqual(conflict.status, TaskStatus.RUNNING)
            self.assertEqual(conflict.lease_owner, "worker-a")

            completed = store.update_task_status(
                claimed, TaskStatus.COMPLETE, worker_id="worker-a"
            )
            self.assertEqual(completed.status, TaskStatus.COMPLETE)

    def test_sqlite_lease_fenced_terminal_write_conflict(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("sqlite fence")
            task = Task(job_id=job.id, role="coder", instruction="work")
            store.save_task(task)
            claimed = store.claim_task(task.id, "worker-a", lease_seconds=60)
            self.assertIsNotNone(claimed)

            conflict = store.update_task_status(
                claimed, TaskStatus.COMPLETE, worker_id="worker-b"
            )
            self.assertEqual(conflict.status, TaskStatus.RUNNING)

    def test_acquire_lock_breaks_stale_task_lock(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            lock_path = store.locks_dir / "task_task123.lock"
            lock_path.write_text(
                json.dumps({"owner": "dead-worker", "at": time.time() - 120}),
                encoding="utf-8",
            )
            self.assertTrue(store.acquire_lock("task:task123", "worker-b", ttl_seconds=30))

    def test_gate_command_parses_string_without_shell(self) -> None:
        from puppetmaster import gates

        command = "touch gate-marker.txt"
        with patch("puppetmaster.gates.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                ["touch", "gate-marker.txt"], 0, "", ""
            )
            gates._run(command, Path(tempfile.gettempdir()), timeout=5)
        argv = run_mock.call_args.args[0]
        if os.name == "nt":
            # Windows: raw string handed to CreateProcess (shlex would eat
            # C:\path backslashes); still no cmd.exe involved.
            self.assertEqual(argv, command)
        else:
            self.assertEqual(argv, ["touch", "gate-marker.txt"])
        self.assertFalse(run_mock.call_args.kwargs.get("shell"))

    def test_gate_engine_error_fails_closed_for_gated_tasks(self) -> None:
        from puppetmaster.gates import GateEvaluation
        from puppetmaster.worker_runtime import WorkerRuntime

        store = MagicMock()
        runtime = WorkerRuntime(
            store=store,
            job_id="job_x",
            role="coder",
            worker_id="worker-a",
        )
        task = Task(
            job_id="job_x",
            role="coder",
            instruction="do work",
            payload={"gates": [{"kind": "require_diff"}]},
        )
        with patch(
            "puppetmaster.gates.evaluate_task_gates",
            side_effect=RuntimeError("boom"),
        ):
            evaluation = runtime._evaluate_gates(task, [])
        self.assertIsInstance(evaluation, GateEvaluation)
        self.assertFalse(evaluation.passed)
        self.assertIn("gate_engine_error", evaluation.failed_reason or "")

    def test_gate_engine_error_passes_through_ungated_tasks(self) -> None:
        from puppetmaster.gates import GateEvaluation
        from puppetmaster.worker_runtime import WorkerRuntime

        store = MagicMock()
        runtime = WorkerRuntime(
            store=store,
            job_id="job_x",
            role="coder",
            worker_id="worker-a",
        )
        task = Task(job_id="job_x", role="coder", instruction="do work")
        with patch(
            "puppetmaster.gates.evaluate_task_gates",
            side_effect=RuntimeError("boom"),
        ):
            evaluation = runtime._evaluate_gates(task, [])
        self.assertTrue(evaluation.passed)

    def test_heartbeat_lease_loss_sets_abort_signal(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        store = MagicMock()
        store.heartbeat_run.side_effect = lambda run: run
        store.renew_task_lease.return_value = None
        runtime = WorkerRuntime(
            store=store,
            job_id="job_x",
            role="coder",
            worker_id="worker-a",
            poll_seconds=0.01,
            heartbeat_seconds=0.01,
        )
        stop = threading.Event()
        run = MagicMock()
        runtime._heartbeat_until_stopped(run, "task_x", stop)
        self.assertTrue(stop.is_set())
        self.assertTrue(runtime._lease_lost.is_set())

    def test_heartbeat_loop_uses_heartbeat_interval_not_poll_interval(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        store = MagicMock()
        store.heartbeat_run.side_effect = lambda run: run
        store.renew_task_lease.return_value = MagicMock()
        runtime = WorkerRuntime(
            store=store,
            job_id="job_x",
            role="coder",
            worker_id="worker-a",
            lease_seconds=10,
            poll_seconds=0.01,
            heartbeat_seconds=0.25,
        )
        stop = MagicMock()
        stop.wait.side_effect = [False, True]

        runtime._heartbeat_until_stopped(MagicMock(), "task_x", stop)

        stop.wait.assert_any_call(0.25)
        self.assertEqual(store.heartbeat_run.call_count, 1)
        self.assertEqual(store.renew_task_lease.call_count, 1)

    def test_heartbeat_interval_keeps_margin_for_short_leases(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        runtime = WorkerRuntime(
            store=MagicMock(),
            job_id="job_x",
            role="coder",
            worker_id="worker-a",
            lease_seconds=2,
            heartbeat_seconds=2.0,
        )

        self.assertAlmostEqual(runtime._heartbeat_interval(), 2 / 3)

    def test_inline_workers_scope_state_dir_env_to_store_root(self) -> None:
        from puppetmaster.orchestrator import Orchestrator

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("inline state dir")
            task = Task(job_id=job.id, role="codex", instruction="x")
            store.save_task(task)
            seen = []

            def complete_task(runtime):
                seen.append(os.environ.get("PUPPETMASTER_STATE_DIR"))
                stored = store.get_task_by_id(task.id)
                store.update_task_status(stored, TaskStatus.COMPLETE)
                return 1

            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": "parent-state"}):
                with patch(
                    "puppetmaster.orchestrator.WorkerRuntime.run_until_idle",
                    autospec=True,
                    side_effect=complete_task,
                ):
                    Orchestrator(store)._run_inline_workers(job, [task])
                self.assertEqual(os.environ["PUPPETMASTER_STATE_DIR"], "parent-state")

            self.assertEqual(seen, [str(store.root)])

    def test_redact_secrets_scrubs_argument_supplied_keys(self) -> None:
        from puppetmaster.redaction import (
            clear_registered_secrets,
            redact_secrets,
            register_secret_value,
        )

        clear_registered_secrets()
        secret = "sk-argument-supplied-test-key-1234567890"
        register_secret_value(secret)
        redacted = redact_secrets(f"Authorization: Bearer {secret}") or ""
        self.assertNotIn(secret, redacted)
        self.assertIn("<secret:redacted>", redacted)
        clear_registered_secrets()

    def test_redact_secrets_catches_common_token_shapes(self) -> None:
        from puppetmaster.redaction import redact_secrets

        # Fake fixtures assembled at runtime so secret scanners (e.g. GitHub
        # push protection) don't flag the literal source as a leaked token.
        fake_slack = "xoxb-" + "1234567890-1234567890-" + "abcdefghijklmnopqrst"
        fake_github = "ghp_" + "1234567890abcdefghijklmnopqrstuvwxyz"
        samples = {
            fake_github: "ghp_<redacted>",
            "AKIAIOSFODNN7EXAMPLE": "AKIA<redacted>",
            fake_slack: "xoxb-<redacted>",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature": "eyJ<redacted>",
        }
        for raw, needle in samples.items():
            redacted = redact_secrets(raw) or ""
            self.assertIn(needle, redacted, msg=raw)
            self.assertNotIn(raw, redacted, msg=raw)

    def test_sqlite_stalled_job_sets_completed_at(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("stall")
            updated = store.update_job_status(job.id, JobStatus.STALLED)
            self.assertIsNotNone(updated.completed_at)

    def test_sqlite_save_run_emits_event_in_same_transaction(self) -> None:
        from puppetmaster.models import AgentRun

        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("atomic run event")
            task = Task(job_id=job.id, role="coder", instruction="work")
            store.save_task(task)
            run = AgentRun(job_id=job.id, task_id=task.id, role="coder", worker_id="w")
            store.save_run(run)
            events = store.read_events(job.id)
            self.assertTrue(any(event["event"] == "run.saved" for event in events))

    def test_file_claim_compare_and_swap_blocks_double_winner(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("file cas")
            task = Task(job_id=job.id, role="coder", instruction="work")
            store.save_task(task)

            first = store.claim_task(task.id, "worker-a", lease_seconds=60)
            second = store.claim_task(task.id, "worker-b", lease_seconds=60)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(store.get_task_by_id(task.id).lease_owner, "worker-a")


class SecurityHardeningTests(unittest.TestCase):
    """Privacy/security defaults from the cross-cutting hardening pass."""

    def test_run_streamed_subprocess_redacts_live_log_not_stdout(self) -> None:
        from puppetmaster.adapters import run_streamed_subprocess

        secret = "super-secret-openai-key-xyz123456"
        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_STATE_DIR"] = tmp
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False):
                task = Task(
                    id="t-redact-live",
                    job_id="job-redact-live",
                    role="codex-review",
                    adapter="codex",
                    instruction="x",
                    payload={},
                )
                result = run_streamed_subprocess(
                    command=[
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('OPENAI_API_KEY', ''))",
                    ],
                    env=os.environ.copy(),
                    task=task,
                    sidecar_name="secret_probe",
                    timeout_seconds=10,
                )
                self.assertIn(secret, result.stdout)
                self.assertIsNotNone(result.live_log_path)
                live_text = Path(result.live_log_path).read_text(encoding="utf-8")
                self.assertNotIn(secret, live_text)
                self.assertIn("redacted", live_text.lower())
            os.environ.pop("PUPPETMASTER_STATE_DIR", None)

    def test_shell_adapter_redacts_verification_stdout_stderr(self) -> None:
        from puppetmaster.adapters import ShellAdapter

        secret = "sk-leaked-shell-secret-key-abc12345"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=False), patch(
            "puppetmaster.adapters.subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=f"before {secret} after",
                stderr=f"err {secret}",
            ),
        ):
            artifacts = ShellAdapter().run(
                Task(
                    id="t-shell",
                    job_id="job-shell",
                    role="verify",
                    adapter="shell",
                    instruction="run",
                    payload={"command": ["echo", "hi"]},
                ),
                "goal",
                "worker-shell",
            )

        payload = artifacts[0].payload
        self.assertNotIn(secret, payload["stdout"])
        self.assertNotIn(secret, payload["stderr"])

    def test_task_payload_api_keys_scrubbed_on_persist(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                root = Path(tmp) / ".puppetmaster"
                store = SwarmStore(root) if backend == "file" else SQLiteSwarmStore(root)
                store.init()
                job = store.create_job("scrub secrets")
                task = Task(
                    id="task-secret",
                    job_id=job.id,
                    role="openai-review",
                    adapter="openai",
                    instruction="review",
                    payload={
                        "openai_api_key": "sk-live-in-memory",
                        "custom_token": "tok-live",
                        "model": "gpt-5.4-mini",
                    },
                )
                store.save_task(task)
                loaded = store.get_task_by_id(task.id)
                self.assertEqual(loaded.payload["openai_api_key"], "<redacted>")
                self.assertEqual(loaded.payload["custom_token"], "<redacted>")
                self.assertEqual(loaded.payload["model"], "gpt-5.4-mini")
                self.assertEqual(task.payload["openai_api_key"], "sk-live-in-memory")

    def test_codegraph_usage_stores_hashed_cwd_not_raw_path(self) -> None:
        import hashlib

        from puppetmaster import codegraph_usage as cu

        raw = "/secret/acme/proprietary-repo"
        expected = hashlib.sha256(str(Path(raw).resolve()).encode("utf-8")).hexdigest()[:12]
        with TemporaryDirectory() as tmp:
            os.environ["PUPPETMASTER_CODEGRAPH_USAGE_LOG"] = str(Path(tmp) / "usage.jsonl")
            cu.record_query(
                command="search",
                cwd=raw,
                result_chars=100,
                latency_ms=1.0,
                ok=True,
            )
            rec = cu.load_usage()[0]
            self.assertEqual(rec["cwd"], expected)
            self.assertNotIn("proprietary", rec["cwd"])
        os.environ.pop("PUPPETMASTER_CODEGRAPH_USAGE_LOG", None)

    def test_fetch_openai_models_refuses_untrusted_base_url(self) -> None:
        from puppetmaster.api_discovery import ApiDiscoveryError, fetch_openai_models

        with self.assertRaises(ApiDiscoveryError) as ctx:
            fetch_openai_models(
                env={
                    "OPENAI_API_KEY": "sk-test",
                    "OPENAI_BASE_URL": "https://evil.example.com/v1",
                },
                getter=lambda u, h: (200, '{"data":[]}'),
            )
        self.assertIn("untrusted host", str(ctx.exception).lower())

    def test_probe_openai_refuses_untrusted_base_url(self) -> None:
        from puppetmaster.preflight import _probe_openai

        rc, _out, err = _probe_openai(
            "gpt-5.4-mini",
            {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "https://evil.example.com/v1",
            },
        )
        self.assertEqual(rc, 1)
        self.assertIn("untrusted host", err.lower())

    @unittest.skipUnless(os.name != "nt", "POSIX-only permission check")
    def test_sensitive_state_dirs_created_owner_only(self) -> None:
        from puppetmaster.fs_permissions import supports_posix_modes
        from puppetmaster.state import ensure_state_dir

        if not supports_posix_modes():
            self.skipTest("POSIX modes unavailable")
        with TemporaryDirectory() as tmp:
            state_dir = ensure_state_dir(Path(tmp) / "nested" / "state")
            mode = state_dir.stat().st_mode & 0o777
            self.assertEqual(mode, 0o700)


class CodegraphFreshnessTests(unittest.TestCase):
    """Index-freshness detection, surfacing, and background self-heal."""

    def setUp(self) -> None:
        from puppetmaster import codegraph as cg

        cg.reset_codegraph_autosync_state()
        self.addCleanup(cg.reset_codegraph_autosync_state)

    def _make_indexed_repo(self, tmp: str, *, db_mtime: float) -> Path:
        root = Path(tmp)
        (root / ".codegraph").mkdir(exist_ok=True)
        db = root / ".codegraph" / "codegraph.db"
        db.write_text("index", encoding="utf-8")
        os.utime(db, (db_mtime, db_mtime))
        return root

    def test_freshness_uninitialized_when_no_codegraph_dir(self) -> None:
        from puppetmaster.codegraph import codegraph_freshness

        with TemporaryDirectory() as tmp:
            verdict = codegraph_freshness(tmp)
            self.assertEqual(verdict.state, "uninitialized")
            self.assertFalse(verdict.is_stale)
            self.assertIsNone(verdict.warning_text())

    def test_freshness_no_index_when_db_missing(self) -> None:
        from puppetmaster.codegraph import codegraph_freshness

        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            verdict = codegraph_freshness(tmp)
            self.assertEqual(verdict.state, "no_index")

    def test_freshness_fresh_when_sources_older_than_index_fs(self) -> None:
        from puppetmaster.codegraph import codegraph_freshness

        with TemporaryDirectory() as tmp:
            root = self._make_indexed_repo(tmp, db_mtime=10_000.0)
            src = root / "module.py"
            src.write_text("x = 1\n", encoding="utf-8")
            os.utime(src, (9_000.0, 9_000.0))  # older than the index
            # Force the filesystem fallback (no git work tree resolution).
            with patch("puppetmaster.codegraph._run_git", return_value=None):
                verdict = codegraph_freshness(tmp)
            self.assertEqual(verdict.state, "fresh")
            self.assertEqual(verdict.changed_count, 0)

    def test_freshness_stale_when_source_newer_than_index_fs(self) -> None:
        from puppetmaster.codegraph import codegraph_freshness

        with TemporaryDirectory() as tmp:
            root = self._make_indexed_repo(tmp, db_mtime=10_000.0)
            src = root / "module.py"
            src.write_text("x = 1\n", encoding="utf-8")
            os.utime(src, (20_000.0, 20_000.0))  # newer than the index
            with patch("puppetmaster.codegraph._run_git", return_value=None):
                verdict = codegraph_freshness(tmp)
            self.assertEqual(verdict.state, "stale")
            self.assertGreaterEqual(verdict.changed_count, 1)
            self.assertIn("module.py", verdict.changed_sample)
            warning = verdict.warning_text()
            self.assertIsNotNone(warning)
            self.assertIn("STALE", warning)
            self.assertIn("codegraph sync", warning)

    def test_freshness_unknown_when_scan_budget_exhausted(self) -> None:
        from puppetmaster import codegraph as cg

        with TemporaryDirectory() as tmp:
            root = self._make_indexed_repo(tmp, db_mtime=10_000.0)
            src = root / "old.py"
            src.write_text("x = 1\n", encoding="utf-8")
            os.utime(src, (9_000.0, 9_000.0))  # older — no drift to find
            with patch("puppetmaster.codegraph._run_git", return_value=None), patch.object(
                cg, "_FRESHNESS_MAX_FILES", 0
            ):
                verdict = cg.codegraph_freshness(tmp)
            # Budget blown before confirming freshness → never a false "fresh".
            self.assertEqual(verdict.state, "unknown")

    def test_freshness_stale_via_git_dirty_file(self) -> None:
        import shutil as _shutil

        from puppetmaster.codegraph import codegraph_freshness

        if _shutil.which("git") is None:
            self.skipTest("git not available")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@t",
            }
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True, env=env)
            tracked = root / "tracked.py"
            tracked.write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=tmp, check=True, env=env)
            subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp, check=True, env=env)
            # Index built in the past; then a dirty edit lands after it.
            self._make_indexed_repo(tmp, db_mtime=1_000.0)
            tracked.write_text("x = 2\n", encoding="utf-8")
            os.utime(tracked, (2_000_000_000.0, 2_000_000_000.0))
            verdict = codegraph_freshness(tmp)
            self.assertEqual(verdict.state, "stale")
            self.assertIn("tracked.py", verdict.changed_sample)

    def test_autosync_returns_none_when_disabled(self) -> None:
        from puppetmaster.codegraph import maybe_autosync_codegraph

        with TemporaryDirectory() as tmp:
            self._make_indexed_repo(tmp, db_mtime=10_000.0)
            with patch.dict(os.environ, {"PUPPETMASTER_CODEGRAPH_AUTOSYNC": "0"}):
                self.assertIsNone(maybe_autosync_codegraph(tmp))

    def test_autosync_skips_when_not_stale(self) -> None:
        from puppetmaster.codegraph import CodegraphFreshness, maybe_autosync_codegraph

        with TemporaryDirectory() as tmp:
            self._make_indexed_repo(tmp, db_mtime=10_000.0)
            fresh = CodegraphFreshness(state="fresh")
            with patch.dict(
                os.environ, {"PUPPETMASTER_CODEGRAPH_AUTOSYNC": "1"}
            ), patch("puppetmaster.codegraph._spawn_codegraph_autosync") as spawn:
                self.assertIsNone(maybe_autosync_codegraph(tmp, fresh))
            spawn.assert_not_called()

    def test_autosync_spawns_once_then_dedupes_within_cooldown(self) -> None:
        from puppetmaster.codegraph import CodegraphFreshness, maybe_autosync_codegraph

        with TemporaryDirectory() as tmp:
            self._make_indexed_repo(tmp, db_mtime=10_000.0)
            stale = CodegraphFreshness(state="stale", changed_count=3)
            with patch.dict(
                os.environ, {"PUPPETMASTER_CODEGRAPH_AUTOSYNC": "1"}
            ), patch(
                "puppetmaster.codegraph._spawn_codegraph_autosync",
                return_value={"ok": True, "action": "spawned"},
            ) as spawn:
                first = maybe_autosync_codegraph(tmp, stale)
                second = maybe_autosync_codegraph(tmp, stale)
            self.assertEqual(first["action"], "spawned")
            self.assertEqual(second["action"], "skipped")
            self.assertEqual(spawn.call_count, 1)

    def test_enrich_prompt_appends_stale_warning_and_triggers_autosync(self) -> None:
        from puppetmaster.codegraph import CodegraphFreshness, enrich_prompt_with_codegraph

        stale = CodegraphFreshness(state="stale", changed_count=2)
        with patch(
            "puppetmaster.codegraph.codegraph_context",
            return_value="auth.py:1 -> login()",
        ), patch(
            "puppetmaster.codegraph.codegraph_freshness", return_value=stale
        ), patch(
            "puppetmaster.codegraph.maybe_autosync_codegraph"
        ) as autosync:
            prompt, used = enrich_prompt_with_codegraph(
                "Inspect repo", task_description="map auth", cwd="/repo"
            )
        self.assertTrue(used)
        self.assertIn("STALE", prompt)
        autosync.assert_called_once()

    def test_mcp_status_attaches_freshness_block(self) -> None:
        import json

        from puppetmaster import mcp_server
        from puppetmaster.codegraph import CodegraphFreshness

        stale = CodegraphFreshness(state="stale", changed_count=4)
        with patch(
            "puppetmaster.mcp_server.codegraph_status_command",
            return_value={"ok": True, "command": "codegraph status", "stdout": "ok"},
        ), patch(
            "puppetmaster.mcp_server.codegraph_freshness", return_value=stale
        ), patch(
            "puppetmaster.mcp_server.maybe_autosync_codegraph"
        ) as autosync:
            result = mcp_server.run_codegraph_status({"cwd": "/repo"})
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload.get("index_stale"))
        self.assertEqual(payload["index_freshness"]["state"], "stale")
        self.assertIn("STALE", payload.get("hint", ""))
        autosync.assert_called_once()

    def test_doctor_codegraph_check_warns_when_index_stale(self) -> None:
        from puppetmaster import diagnostics
        from puppetmaster.codegraph import CodegraphFreshness

        stale = CodegraphFreshness(state="stale", changed_count=5)
        with patch(
            "puppetmaster.diagnostics.codegraph_available", return_value=True
        ), patch(
            "puppetmaster.diagnostics.codegraph_initialized", return_value=True
        ), patch(
            "puppetmaster.diagnostics.codegraph_status_command",
            return_value={"ok": True, "stdout": "Backend: native", "stderr": ""},
        ), patch(
            "puppetmaster.diagnostics.codegraph_native_sqlite_broken",
            return_value=False,
        ), patch(
            "puppetmaster.diagnostics.codegraph_freshness", return_value=stale
        ):
            check = diagnostics._codegraph_check(Path("/repo"))
        self.assertEqual(check.status, "warn")
        self.assertIn("STALE", check.detail)


class JsonPrefixDecodeTests(unittest.TestCase):
    """Trailing-brace trailers must not falsely degrade a good analyze run.

    A worker can emit valid JSON followed by a brace-bearing trailer (a
    Hermes -Q session/cost footer, a courtesy "Done {ok}", or a second JSON
    blob). The greedy first-opener-to-last-closer slice over-reaches into the
    trailer and json.loads raises, so the run was falsely marked
    empty_or_unstructured. The raw_decode fallback recovers the real payload
    while genuine prose still yields nothing.
    """

    def setUp(self) -> None:
        from puppetmaster.adapters import parse_cursor_artifact_payload

        self.parse = parse_cursor_artifact_payload
        self.good = {"findings": [{"type": "finding", "claim": "x"}]}
        self.good_json = json.dumps(self.good)

    def test_clean_json_still_parses(self) -> None:
        self.assertEqual(self.parse(self.good_json), self.good)

    def test_trailing_session_cost_footer(self) -> None:
        text = f"{self.good_json}\n[session abc | 1,240 tokens | $0.003]"
        self.assertEqual(self.parse(text), self.good)

    def test_trailing_courtesy_brace_line(self) -> None:
        text = f"{self.good_json}\n\nDone {{ok}}"
        self.assertEqual(self.parse(text), self.good)

    def test_trailing_second_json_object(self) -> None:
        text = f'{self.good_json}\n\n{{"session":"abc"}}'
        self.assertEqual(self.parse(text), self.good)

    def test_trailing_prose_without_brace_already_worked(self) -> None:
        text = f"{self.good_json}\n\nLet me know!"
        self.assertEqual(self.parse(text), self.good)

    def test_fenced_json_with_trailing_footer(self) -> None:
        text = f"```json\n{self.good_json}\n```\n[session abc | 12 tokens]"
        self.assertEqual(self.parse(text), self.good)

    def test_leading_whitespace_and_prose_then_json(self) -> None:
        text = f"   Here is the report:\n\n{self.good_json}\n[session]"
        self.assertEqual(self.parse(text), self.good)

    def test_crlf_line_endings(self) -> None:
        text = f"{self.good_json}\r\n\r\n[session abc | 5 tokens]"
        self.assertEqual(self.parse(text), self.good)

    def test_first_brace_is_false_opener_then_real_json(self) -> None:
        text = f"Use {{x}} then:\n{self.good_json}\n[session]"
        self.assertEqual(self.parse(text), self.good)

    def test_genuine_prose_still_returns_none(self) -> None:
        # The guard against over-correction: real unstructured prose must NOT
        # be salvaged, so genuine degrades aren't masked.
        self.assertIsNone(self.parse("The entrypoint is main() in cli.py."))

    def test_prose_with_unparseable_braces_returns_none(self) -> None:
        self.assertIsNone(self.parse("Wrap it in {curly} or [square] brackets."))

    def test_hermes_analyze_recovers_artifacts_through_trailer(self) -> None:
        from puppetmaster.adapters import cursor_result_artifacts

        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="hermes",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        text = f"{self.good_json}\n[session abc | 1,240 tokens | $0.003]"
        artifacts = cursor_result_artifacts(task, "worker-hermes", text)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].type, ArtifactType.FINDING)

    def test_prose_yields_zero_artifacts(self) -> None:
        from puppetmaster.adapters import cursor_result_artifacts

        task = Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="hermes",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )
        artifacts = cursor_result_artifacts(
            task, "worker-hermes", "The entrypoint is main()."
        )
        self.assertEqual(artifacts, [])


class AdapterProvenanceTests(unittest.TestCase):
    """Artifacts must record which adapter actually produced them.

    The shared parser previously hardcoded ``adapter:cursor-sdk`` as the
    evidence fallback, so a Hermes/Codex/OpenAI worker that emitted a
    finding/risk/decision without its own evidence was mislabeled as Cursor.
    """

    def _task(self) -> Task:
        return Task(
            job_id="job",
            role="pipeline-mapper",
            instruction="inspect repo",
            adapter="hermes",
            payload={"prompt": "Inspect repo", "cwd": "."},
        )

    def test_item_evidence_default_uses_calling_adapter(self) -> None:
        from puppetmaster.adapters import cursor_artifact_from_item

        artifact = cursor_artifact_from_item(
            self._task(), "w", {"type": "risk", "risk": "x"}, adapter="hermes"
        )
        self.assertIsNotNone(artifact)
        self.assertIn("adapter:hermes", artifact.evidence)
        self.assertNotIn("adapter:cursor-sdk", artifact.evidence)

    def test_item_evidence_default_back_compat_cursor(self) -> None:
        from puppetmaster.adapters import cursor_artifact_from_item

        artifact = cursor_artifact_from_item(
            self._task(), "w", {"type": "finding", "claim": "x"}
        )
        self.assertIn("adapter:cursor-sdk", artifact.evidence)

    def test_explicit_evidence_is_preserved(self) -> None:
        from puppetmaster.adapters import cursor_artifact_from_item

        artifact = cursor_artifact_from_item(
            self._task(),
            "w",
            {"type": "finding", "claim": "x", "evidence": ["file:foo.py"]},
            adapter="hermes",
        )
        self.assertEqual(artifact.evidence, ["file:foo.py"])

    def test_result_artifacts_thread_adapter_label(self) -> None:
        from puppetmaster.adapters import cursor_result_artifacts

        text = json.dumps({"risks": [{"type": "risk", "risk": "boom"}]})
        artifacts = cursor_result_artifacts(self._task(), "w", text, adapter="hermes")
        self.assertEqual(len(artifacts), 1)
        self.assertIn("adapter:hermes", artifacts[0].evidence)

    def test_degraded_artifact_labels_adapter(self) -> None:
        from puppetmaster.adapters import cursor_degraded_artifact

        with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": "/tmp"}):
            artifact = cursor_degraded_artifact(
                self._task(), "w", "blah", adapter="hermes"
            )
        self.assertIn("adapter:hermes", artifact.evidence)
        self.assertIn("hermes", artifact.payload["risk"])

    def test_decision_why_default_is_vendor_neutral(self) -> None:
        from puppetmaster.adapters import cursor_artifact_from_item

        artifact = cursor_artifact_from_item(
            self._task(), "w", {"type": "decision", "decision": "do x"}, adapter="hermes"
        )
        self.assertNotIn("Cursor", artifact.payload["why"])


class DirtyWorktreePathsNoteTests(unittest.TestCase):
    """The clean-tree block should name the offending paths, not just say 'dirty'."""

    def test_note_lists_changed_and_untracked(self) -> None:
        from puppetmaster.adapters import dirty_worktree_paths_note

        note = dirty_worktree_paths_note(["a.py"], ["__pycache__/x.pyc"])
        self.assertIn("a.py (modified)", note)
        self.assertIn("__pycache__/x.pyc (untracked)", note)

    def test_note_empty_when_clean(self) -> None:
        from puppetmaster.adapters import dirty_worktree_paths_note

        self.assertEqual(dirty_worktree_paths_note([], []), "")

    def test_note_truncates_long_lists(self) -> None:
        from puppetmaster.adapters import dirty_worktree_paths_note

        note = dirty_worktree_paths_note([f"f{i}.py" for i in range(15)], [], limit=10)
        self.assertIn("(+5 more)", note)

    def test_guard_message_includes_paths(self) -> None:
        from puppetmaster.adapters import HermesAdapter, git_snapshot

        task = Task(
            job_id="job",
            role="impl",
            instruction="build it",
            adapter="hermes",
            payload={"cwd": ".", "mode": "implement"},
        )
        dirty = {
            "changed_files": ["a.py"],
            "untracked_files": ["junk/__pycache__/x.pyc"],
            "sha": "abc",
        }
        with patch(
            "puppetmaster.adapters.enrich_prompt_with_codegraph",
            return_value=("p", False),
        ), patch(
            "puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"
        ), patch(
            "puppetmaster.adapters.git_snapshot", return_value=dirty
        ), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch("puppetmaster.adapters.snapshot_has_diff", return_value=False):
            artifacts = HermesAdapter()._run_implement(task, "goal", "w")
        msg = artifacts[0].payload["message"]
        self.assertEqual(artifacts[0].payload["failure"], "dirty_worktree")
        self.assertIn("a.py (modified)", msg)
        self.assertIn("junk/__pycache__/x.pyc (untracked)", msg)


class McpServerCodeStalenessTests(unittest.TestCase):
    """A long-lived MCP server on pre-upgrade code must be detectable."""

    def _entry(self, version, **kw):
        from puppetmaster.mcp_registry import McpServerEntry

        defaults = dict(
            pid=os.getpid(),  # alive
            workspace="/repo",
            started_at=time.time(),
            last_heartbeat=time.time(),
            version=version,
        )
        defaults.update(kw)
        return McpServerEntry(**defaults)

    def test_summarize_flags_version_mismatch_as_code_stale(self) -> None:
        from puppetmaster.mcp_registry import summarize

        snap = summarize([self._entry("0.9.60")], installed="0.9.61")
        self.assertEqual(snap["code_stale"], 1)
        self.assertEqual(snap["installed_version"], "0.9.61")
        self.assertTrue(snap["servers"][0]["code_stale"])

    def test_summarize_matching_version_is_not_code_stale(self) -> None:
        from puppetmaster.mcp_registry import summarize

        snap = summarize([self._entry("0.9.61")], installed="0.9.61")
        self.assertEqual(snap["code_stale"], 0)
        self.assertFalse(snap["servers"][0]["code_stale"])

    def test_dead_server_is_never_code_stale(self) -> None:
        from puppetmaster.mcp_registry import summarize

        # PID 0 is never alive; a dead/old server isn't actionable.
        snap = summarize([self._entry("0.9.60", pid=0)], installed="0.9.61")
        self.assertEqual(snap["code_stale"], 0)

    def test_missing_version_is_not_code_stale(self) -> None:
        from puppetmaster.mcp_registry import summarize

        snap = summarize([self._entry(None)], installed="0.9.61")
        self.assertEqual(snap["code_stale"], 0)

    def test_status_hint_surfaced_when_code_stale(self) -> None:
        from puppetmaster import mcp_server
        from puppetmaster.mcp_registry import summarize

        stale = summarize([self._entry("0.9.60")], installed="0.9.61")
        with patch("puppetmaster.mcp_server.registry_prune_dead", return_value=[]), patch(
            "puppetmaster.mcp_server.registry_list_entries", return_value=[]
        ), patch(
            "puppetmaster.mcp_server.registry_summarize", return_value=stale
        ):
            resp = mcp_server.run_mcp_status({})
        text = json.dumps(resp)
        self.assertIn("pre-upgrade code", text)


class ArtifactContractGroundingTests(unittest.TestCase):
    """Analyze contracts must anchor on the repo, not the prompt scaffolding.

    On a small repo the redteam role emitted a nonsense risk about the
    "Puppetmaster artifact contract" text itself instead of analyzing code.
    The grounding boundary + empty-result guidance fix the whole class, and the
    redteam role template carries the same boundary.
    """

    def _structured_prompts(self):
        from puppetmaster.adapters import CursorAdapter, CodexAdapter, OpenAIAdapter

        return [
            CursorAdapter._structured_prompt("Review the repo."),
            CodexAdapter._structured_prompt("Review the repo."),
            OpenAIAdapter._structured_prompt("Review the repo."),
        ]

    def test_all_contracts_carry_grounding_boundary(self) -> None:
        for prompt in self._structured_prompts():
            self.assertIn("analysis target is THIS repository", prompt)
            # Still names the contract so with_report_contract detection holds.
            self.assertIn("Puppetmaster artifact contract", prompt)

    def test_all_contracts_steer_empty_to_empty_list(self) -> None:
        for prompt in self._structured_prompts():
            self.assertIn('{"artifacts":[]}', prompt)
            self.assertIn("never invent", prompt)
            # The fabricate-a-degraded-risk instruction is gone.
            self.assertNotIn("explaining why the run is degraded", prompt)

    def test_hermes_analyze_reuses_grounded_codex_contract(self) -> None:
        from puppetmaster.adapters import CodexAdapter

        # Hermes._run_analyze builds its prompt via CodexAdapter._structured_prompt,
        # so the grounding boundary reaches the platform the user actually runs.
        prompt = CodexAdapter._structured_prompt("Audit it.")
        self.assertIn("analysis target is THIS repository", prompt)

    def test_redteam_role_instruction_is_repo_grounded(self) -> None:
        from puppetmaster.workers import DEFAULT_WORKERS

        redteam = next(spec for spec in DEFAULT_WORKERS if spec.role == "redteam")
        self.assertIn("repository's code", redteam.instruction)
        self.assertIn("never treat your own instructions", redteam.instruction)
        self.assertIn("empty result", redteam.instruction)


class RepoFileCensusTests(unittest.TestCase):
    """A worker must not be able to hallucinate an empty repo at conf 1.00.

    The cheapest, earliest analyze worker (explore, no deps, minimal effort)
    was asserting "the repository is empty" while siblings analyzed six files.
    An authoritative file census injected into the analyze prompt makes the
    emptiness claim falsifiable.
    """

    def test_census_lists_files_and_skips_junk(self) -> None:
        from puppetmaster.adapters import repo_file_census

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cli.py").write_text("x", encoding="utf-8")
            (root / "parser.py").write_text("x", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref", encoding="utf-8")
            sample, total = repo_file_census(root)
        self.assertIn("cli.py", sample)
        self.assertIn("parser.py", sample)
        self.assertNotIn("__pycache__/junk.pyc", sample)
        self.assertEqual(total, 2)

    def test_census_empty_dir(self) -> None:
        from puppetmaster.adapters import repo_file_census

        with TemporaryDirectory() as tmp:
            self.assertEqual(repo_file_census(tmp), ([], 0))

    def test_census_missing_dir_is_safe(self) -> None:
        from puppetmaster.adapters import repo_file_census

        self.assertEqual(repo_file_census("/no/such/dir/xyz"), ([], 0))
        self.assertEqual(repo_file_census(None), ([], 0))

    def test_census_truncates_with_overflow(self) -> None:
        from puppetmaster.adapters import repo_file_census

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(120):
                (root / f"f{i:03d}.py").write_text("x", encoding="utf-8")
            sample, total = repo_file_census(root, limit=100)
        self.assertEqual(total, 120)
        self.assertEqual(len(sample), 100)

    def test_with_census_states_not_empty_when_files_exist(self) -> None:
        from puppetmaster.adapters import with_repo_census

        with TemporaryDirectory() as tmp:
            (Path(tmp) / "main.py").write_text("x", encoding="utf-8")
            grounded = with_repo_census("PROMPT", tmp)
        self.assertIn("PROMPT", grounded)
        self.assertIn("main.py", grounded)
        self.assertIn("NOT empty", grounded)
        self.assertIn("tooling failure", grounded)

    def test_with_census_soft_note_when_unenumerable(self) -> None:
        from puppetmaster.adapters import with_repo_census

        with TemporaryDirectory() as tmp:
            grounded = with_repo_census("PROMPT", tmp)
        # We never assert emptiness ourselves on a miss.
        self.assertIn("none enumerated", grounded)
        self.assertIn("Do not assert the repository is empty", grounded)

    def test_analyze_paths_inject_census(self) -> None:
        """Every read-only analyze adapter grounds its prompt with the census."""
        from puppetmaster import adapters

        task = Task(
            job_id="job",
            role="explore",
            instruction="Map the problem.",
            adapter="hermes",
            payload={"prompt": "Map it", "cwd": "."},
        )
        seen = {}

        def _capture(prompt, cwd):
            seen["called"] = True
            return prompt + "\n[CENSUS]"

        # The census is injected before the worker subprocess; make the
        # subprocess raise so we assert grounding happened without simulating a
        # full Hermes run. Hermes analyze is the platform under migration.
        with patch.object(adapters, "with_repo_census", side_effect=_capture), patch(
            "puppetmaster.adapters.enrich_prompt_with_codegraph",
            return_value=("P", False),
        ), patch(
            "puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"
        ), patch(
            "puppetmaster.adapters.run_streamed_subprocess",
            side_effect=RuntimeError("stop-after-census"),
        ):
            with self.assertRaises(RuntimeError):
                adapters.HermesAdapter()._run_analyze(task, "goal", "w")
        self.assertTrue(seen.get("called"))


class HermesAnalyzeRetryTests(unittest.TestCase):
    """A clean Hermes analyze run that returns prose the parser can't structure
    (the minimal-effort flicker) gets one stricter JSON-only reprompt before it
    is accepted as degraded."""

    def _prose(self) -> "StreamedProcess":
        return StreamedProcess(
            returncode=0,
            stdout="The repository looks healthy; I found nothing notable.",
            stderr="",
            timed_out=False,
            live_log_path=None,
        )

    def _json(self) -> "StreamedProcess":
        body = json.dumps(
            {"artifacts": [{"type": "finding", "claim": "real bug", "evidence": ["cli.py:5"], "confidence": 0.9}]}
        )
        return StreamedProcess(
            returncode=0, stdout=body, stderr="", timed_out=False, live_log_path=None
        )

    def _task(self, **payload) -> Task:
        base = {"cwd": str(Path.cwd()), "disable_codegraph": True}
        base.update(payload)
        return Task(
            id="t-retry",
            job_id="job-retry",
            role="explore",
            adapter="hermes",
            instruction="Map the repo.",
            payload=base,
        )

    def test_retry_recovers_structured_output(self) -> None:
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), patch(
            "puppetmaster.adapters.run_streamed_subprocess",
            side_effect=[self._prose(), self._json()],
        ) as run:
            artifacts = HermesAdapter().run(self._task(), "goal", "worker")
        self.assertEqual(run.call_count, 2)
        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "passed")
        self.assertIn("retry:recovered", verification.evidence)
        self.assertTrue(any(a.type == ArtifactType.FINDING for a in artifacts))

    def test_retry_exhausted_stays_degraded(self) -> None:
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), patch(
            "puppetmaster.adapters.run_streamed_subprocess",
            side_effect=[self._prose(), self._prose()],
        ) as run:
            artifacts = HermesAdapter().run(self._task(), "goal", "worker")
        self.assertEqual(run.call_count, 2)
        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "degraded")
        self.assertIn("retry:exhausted", verification.evidence)

    def test_retry_can_be_disabled(self) -> None:
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), patch(
            "puppetmaster.adapters.run_streamed_subprocess",
            side_effect=[self._prose()],
        ) as run:
            artifacts = HermesAdapter().run(self._task(analyze_retry=False), "goal", "worker")
        self.assertEqual(run.call_count, 1)
        verification = artifacts[0]
        self.assertEqual(verification.payload["result"], "degraded")
        self.assertNotIn("retry:recovered", verification.evidence)
        self.assertNotIn("retry:exhausted", verification.evidence)


class StitcherDedupTests(unittest.TestCase):
    """N workers finding the same thing collapses to one bullet, not N near-dupes.

    Uses the real lexically-diverse paraphrases from the live stress test that
    a pure high token gate missed but that all cite the same files.
    """

    def _finding(self, claim: str, conf: float, by: str, evidence=None) -> Artifact:
        return Artifact(
            job_id="j",
            task_id="t",
            type=ArtifactType.FINDING,
            created_by=by,
            confidence=conf,
            evidence=evidence if evidence is not None else ["adapter:hermes"],
            payload={"claim": claim},
        )

    def test_paraphrased_findings_sharing_a_locus_collapse(self) -> None:
        from puppetmaster.stitcher import Stitcher

        ev = ["arithmetic.py:5-8", "adapter:hermes"]
        arts = [
            self._finding(
                "The 'multiply' function is implemented using repeated addition, "
                "which can be less efficient for large numbers",
                0.90,
                "w1",
                ev,
            ),
            self._finding(
                "Multiplication is implemented inefficiently using repeated addition",
                1.00,
                "w2",
                ev,
            ),
            self._finding(
                "The multiplication operation is implemented using repeated addition, "
                "which may be less efficient than Python's native",
                0.90,
                "w3",
                ev,
            ),
        ]
        bullets = Stitcher._bullet_payloads(arts, "claim", dedupe=True)
        self.assertEqual(len(bullets), 1)
        self.assertIn("reported by 3 workers", bullets[0])
        self.assertIn("confidence=1.00", bullets[0])

    def test_distinct_bugs_at_same_file_stay_separate(self) -> None:
        """The collision guard: two unrelated bugs both citing cli.py must not
        merge just because they share a locus."""
        from puppetmaster.stitcher import Stitcher

        arts = [
            self._finding("KeyError on unsupported operators", 0.9, "w1", ["cli.py:5", "cli.py:9"]),
            self._finding("No division support in the CLI", 0.95, "w2", ["cli.py:5"]),
        ]
        bullets = Stitcher._bullet_payloads(arts, "claim", dedupe=True)
        self.assertEqual(len(bullets), 2)
        self.assertNotIn("reported by", " ".join(bullets))

    def test_strong_wording_overlap_merges_without_locus(self) -> None:
        from puppetmaster.stitcher import Stitcher

        arts = [
            self._finding("parse() crashes on malformed input", 0.9, "w1"),
            self._finding("parse() crashes on malformed input strings", 0.9, "w2"),
        ]
        bullets = Stitcher._bullet_payloads(arts, "claim", dedupe=True)
        self.assertEqual(len(bullets), 1)

    def test_unrelated_findings_without_locus_stay_separate(self) -> None:
        from puppetmaster.stitcher import Stitcher

        arts = [
            self._finding("no division support in the calculator", 0.9, "w1"),
            self._finding("the parser rejects negative numbers", 0.9, "w2"),
        ]
        bullets = Stitcher._bullet_payloads(arts, "claim", dedupe=True)
        self.assertEqual(len(bullets), 2)

    def test_dedupe_off_keeps_all(self) -> None:
        from puppetmaster.stitcher import Stitcher

        arts = [
            self._finding("same claim", 0.9, "w1"),
            self._finding("same claim", 0.9, "w2"),
        ]
        bullets = Stitcher._bullet_payloads(arts, "claim")
        self.assertEqual(len(bullets), 2)

    def test_evidence_loci_extraction(self) -> None:
        from puppetmaster.stitcher import Stitcher

        art = self._finding(
            "x", 0.9, "w1", ["arithmetic.py:5-8", "cli.py", "adapter:hermes", "context:codegraph"]
        )
        loci = Stitcher._evidence_loci(art)
        self.assertIn("arithmetic.py", loci)
        self.assertIn("cli.py", loci)
        self.assertNotIn("adapter", loci)
        self.assertNotIn("hermes", loci)

    def test_claims_similar_signature(self) -> None:
        from puppetmaster.stitcher import Stitcher

        same = frozenset({"arithmetic.py"})
        # Low lexical overlap but shared locus -> merge.
        self.assertTrue(
            Stitcher._claims_similar(
                "multiplication is implemented inefficiently using repeated addition",
                "the multiply function uses repeated addition for large numbers",
                same,
                same,
            )
        )
        # Zero overlap, shared locus -> still separate (collision guard).
        self.assertFalse(
            Stitcher._claims_similar(
                "keyerror on unsupported operators",
                "no division support in the cli",
                same,
                same,
            )
        )


class McpServerUpdateNudgeTests(unittest.TestCase):
    """A long-lived stdio MCP server can't hot-reload; surface newer on-disk
    code as a one-line nudge in every tool response instead of a silent gap."""

    def setUp(self) -> None:
        from puppetmaster import mcp_server

        self.mcp = mcp_server
        mcp_server.reset_server_update_cache()
        self.addCleanup(mcp_server.reset_server_update_cache)

    def test_note_when_disk_is_newer(self) -> None:
        with patch.object(self.mcp, "_SERVER_RUNNING_VERSION", "0.9.63"), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.64"
        ):
            note = self.mcp.server_update_note(now=1000.0)
        self.assertIsNotNone(note)
        self.assertIn("0.9.63", note)
        self.assertIn("0.9.64", note)
        self.assertIn("Restart", note)

    def test_no_note_when_versions_match(self) -> None:
        with patch.object(self.mcp, "_SERVER_RUNNING_VERSION", "0.9.64"), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.64"
        ):
            self.assertIsNone(self.mcp.server_update_note(now=1000.0))

    def test_no_note_on_downgrade(self) -> None:
        # A stale env where disk is *older* shouldn't nag — that's not an upgrade.
        with patch.object(self.mcp, "_SERVER_RUNNING_VERSION", "0.9.64"), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.63"
        ):
            self.assertIsNone(self.mcp.server_update_note(now=1000.0))

    def test_no_note_when_disk_version_unknown(self) -> None:
        with patch.object(self.mcp, "_SERVER_RUNNING_VERSION", "0.9.64"), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value=None
        ):
            self.assertIsNone(self.mcp.server_update_note(now=1000.0))

    def test_note_is_cached_within_ttl(self) -> None:
        with patch.object(self.mcp, "_SERVER_RUNNING_VERSION", "0.9.63"), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.64"
        ) as disk:
            first = self.mcp.server_update_note(now=1000.0)
            second = self.mcp.server_update_note(now=1005.0)  # within 30s TTL
        self.assertEqual(first, second)
        self.assertEqual(disk.call_count, 1)

    def test_call_tool_annotates_stale_response(self) -> None:
        from types import SimpleNamespace

        fake = {"x": SimpleNamespace(handler=lambda args: {"ok": True})}
        with patch.object(self.mcp, "_tool_registry", return_value=fake), patch.object(
            self.mcp, "_SERVER_RUNNING_VERSION", "0.9.63"
        ), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.64"
        ):
            result = self.mcp.call_tool("x", {})
        self.assertTrue(result["ok"])
        self.assertIn("server_update_available", result)
        self.assertIn("0.9.64", result["server_update_available"])

    def test_call_tool_clean_when_current(self) -> None:
        from types import SimpleNamespace

        fake = {"x": SimpleNamespace(handler=lambda args: {"ok": True})}
        with patch.object(self.mcp, "_tool_registry", return_value=fake), patch.object(
            self.mcp, "_SERVER_RUNNING_VERSION", "0.9.64"
        ), patch.object(
            self.mcp, "installed_puppetmaster_version", return_value="0.9.64"
        ):
            result = self.mcp.call_tool("x", {})
        self.assertNotIn("server_update_available", result)


class HermesToolSessionPruneTests(unittest.TestCase):
    """Puppetmaster prunes throwaway Hermes worker sessions (--source tool) after
    each worker run, so they don't pile up in Hermes' session store / desktop
    panel. It does this via Hermes' own ``sessions prune`` CLI (cascade-correct,
    only deletes ENDED sessions so a sibling worker is never disturbed)."""

    def test_cleanup_enabled_by_default(self) -> None:
        from puppetmaster.adapters import _hermes_session_cleanup_enabled

        self.assertTrue(_hermes_session_cleanup_enabled({}))

    def test_cleanup_opt_out(self) -> None:
        from puppetmaster.adapters import _hermes_session_cleanup_enabled

        for off in ("0", "false", "no", "off", "OFF"):
            self.assertFalse(
                _hermes_session_cleanup_enabled({"PUPPETMASTER_HERMES_PRUNE_SESSIONS": off}),
                off,
            )

    def test_prune_builds_correct_cli_command(self) -> None:
        import puppetmaster.adapters as adapters
        from types import SimpleNamespace

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), \
                patch("puppetmaster.adapters.subprocess.run", side_effect=fake_run):
            adapters.prune_hermes_tool_sessions("hermes")

        cmd = captured["cmd"]
        self.assertEqual(cmd[:1], ["/usr/bin/hermes"])
        self.assertIn("sessions", cmd)
        self.assertIn("prune", cmd)
        self.assertIn("--source", cmd)
        self.assertIn("tool", cmd)
        self.assertIn("--older-than", cmd)
        self.assertIn("0", cmd)
        self.assertIn("--yes", cmd)

    def test_prune_skipped_when_disabled(self) -> None:
        import puppetmaster.adapters as adapters

        with patch("puppetmaster.adapters.subprocess.run") as run:
            adapters.prune_hermes_tool_sessions(
                "hermes", env={"PUPPETMASTER_HERMES_PRUNE_SESSIONS": "0"}
            )
        run.assert_not_called()

    def test_prune_only_targets_tool_source_never_user_sessions(self) -> None:
        """A guard so we can never delete a real user source (cli / tui / telegram)
        even if a task payload sets a weird source."""
        import puppetmaster.adapters as adapters

        with patch("puppetmaster.adapters.subprocess.run") as run:
            adapters.prune_hermes_tool_sessions("hermes", source="cli")
            adapters.prune_hermes_tool_sessions("hermes", source="telegram")
        run.assert_not_called()

    def test_prune_never_raises(self) -> None:
        import puppetmaster.adapters as adapters

        # Missing CLI -> resolve_command returns None -> clean no-op.
        with patch("puppetmaster.adapters.resolve_command", return_value=None):
            adapters.prune_hermes_tool_sessions("nope")  # must not raise
        # subprocess blowing up -> swallowed.
        with patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/hermes"), \
                patch("puppetmaster.adapters.subprocess.run", side_effect=OSError("boom")):
            adapters.prune_hermes_tool_sessions("hermes")  # must not raise

    def test_adapter_run_invokes_prune_in_finally(self) -> None:
        """The prune fires from HermesAdapter.run() even when the worker path
        raises — session hygiene must not depend on a happy path."""
        from puppetmaster.adapters import HermesAdapter

        task = Task(
            id="t", job_id="j", role="explore", adapter="hermes",
            instruction="x", payload={"cwd": ".", "source": "tool"},
        )
        with patch.object(HermesAdapter, "_run_analyze", side_effect=RuntimeError("worker died")), \
                patch("puppetmaster.adapters.prune_hermes_tool_sessions") as prune:
            with self.assertRaises(RuntimeError):
                HermesAdapter().run(task, "goal", "worker")
        prune.assert_called_once()


if __name__ == "__main__":
    unittest.main()
