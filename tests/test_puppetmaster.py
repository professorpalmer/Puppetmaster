from __future__ import annotations

import contextlib
import io
import json
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
    build_claude_code_command,
    classify_claude_code_failure,
    classify_cursor_failure,
)
from puppetmaster.config import load_config
from puppetmaster.cli import cursor_prompt, main as cli_main
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.mcp_server import ASYNC_PROCESSES, call_tool, handle_message
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus, seconds_from_now
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.stitcher import Stitcher
from puppetmaster.store import SwarmStore
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec


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
        self.assertIn("puppetmaster_status", tool_names)

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
                    "worker_mode": "inline",
                },
            )
            payload = json.loads(result["content"][0]["text"])

            self.assertIn("job_", payload["job_id"])
            self.assertIn("pid", payload)
            self.assertFalse(result["isError"])

            spawned = ASYNC_PROCESSES[before_process_count:]
            try:
                deadline = time.monotonic() + 5
                status_payload = None
                while time.monotonic() < deadline:
                    status = call_tool(
                        "puppetmaster_status",
                        {
                            "cwd": tmp,
                            "state_dir": ".pm-test",
                            "job_id": payload["job_id"],
                        },
                    )
                    status_body = json.loads(status["content"][0]["text"])
                    if status["isError"] or not status_body.get("stdout"):
                        time.sleep(0.05)
                        continue
                    try:
                        status_payload = json.loads(status_body["stdout"])
                    except json.JSONDecodeError:
                        time.sleep(0.05)
                        continue
                    if status_payload["job"]["status"] == "complete":
                        break
                    time.sleep(0.05)
            finally:
                for process in spawned:
                    process.wait(timeout=5)

            self.assertEqual(status_payload["job"]["status"], "complete")
            self.assertEqual(status_payload["artifact_count"], 2)

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
            checks = {check.name: check for check in run_doctor(root)}

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

