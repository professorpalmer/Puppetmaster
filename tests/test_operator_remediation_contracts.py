from __future__ import annotations

import os
import re
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from puppetmaster.cli.helpers import artifact_feed_since
from puppetmaster.cli.commands_jobs import read_job_state
from puppetmaster.cli import main as cli_main
from puppetmaster.cost import price_job
from puppetmaster.delivery import delivery_verdict
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.adapters._streaming import run_streamed_subprocess
from puppetmaster.adapters.registry import adapter_runtime_capabilities
from puppetmaster.mcp_server import goal_schema, mcp_state_dir
from puppetmaster import mcp_server
from puppetmaster.model_registry import starter_registry
from puppetmaster.receipt import build_job_receipt
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import LaunchConflictError, SwarmStore
from puppetmaster.swarm_launch import build_analysis_swarm_specs
from puppetmaster.workers import WorkerSpec


class LaunchIdentityContractTests(unittest.TestCase):
    def test_file_and_sqlite_launch_keys_are_idempotent_and_fail_closed(self) -> None:
        for store_type in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_type.__name__), tempfile.TemporaryDirectory() as tmp:
                store = store_type(Path(tmp))
                store.init()
                first = store.create_job("same goal", launch_key="retry-1")
                second = store.create_job("  same   goal ", launch_key="retry-1")
                self.assertEqual(first.id, second.id)
                with self.assertRaises(LaunchConflictError):
                    store.create_job("different goal", launch_key="retry-1")

    def test_concurrent_retries_converge_on_one_job_in_both_stores(self) -> None:
        for store_type in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_type.__name__), tempfile.TemporaryDirectory() as tmp:
                store = store_type(Path(tmp))
                # Exercise the launch-key transaction, not the independent
                # first-open WAL initialization contract.
                store.init()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = list(
                        pool.map(
                            lambda _: store.create_job("concurrent", launch_key="same-key"),
                            range(4),
                        )
                    )
                self.assertEqual({job.id for job in jobs}, {jobs[0].id})

    def test_launch_fingerprint_covers_the_normalized_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            first = store.create_job(
                "goal",
                launch_key="request-key",
                launch_fingerprint=store.launch_fingerprint(
                    "goal", request={"adapter": "codex", "timeout": 600}
                ),
            )
            same = store.create_job(
                "goal",
                launch_key="request-key",
                launch_fingerprint=store.launch_fingerprint(
                    "goal", request={"timeout": 600, "adapter": "codex"}
                ),
            )
            self.assertEqual(first.id, same.id)
            with self.assertRaises(LaunchConflictError):
                store.create_job(
                    "goal",
                    launch_key="request-key",
                    launch_fingerprint=store.launch_fingerprint(
                        "goal", request={"adapter": "claude-code", "timeout": 600}
                    ),
                )

    def test_orchestrator_retry_returns_existing_job_without_redispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            orchestrator = Orchestrator(store)
            specs = [WorkerSpec(role="analysis", instruction="inspect", adapter="local")]
            with patch.object(orchestrator, "_run_workers") as run_workers:
                first = orchestrator.run("goal", specs=specs, launch_key="retry-key")
                second = orchestrator.run("goal", specs=specs, launch_key="retry-key")
            self.assertEqual(first.job.id, second.job.id)
            self.assertEqual(run_workers.call_count, 1)
            self.assertEqual(len(store.list_tasks(first.job.id)), 1)

    def test_max_output_bytes_is_propagated_to_worker_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            orchestrator = Orchestrator(store)
            specs = [WorkerSpec(role="analysis", instruction="inspect", adapter="local")]
            with patch.dict(os.environ, {"PUPPETMASTER_MAX_OUTPUT_BYTES": "1234"}), patch.object(
                orchestrator, "_run_workers"
            ):
                result = orchestrator.run("goal", specs=specs)
            self.assertEqual(store.list_tasks(result.job.id)[0].payload["max_output_bytes"], 1234)


class OperatorContractTests(unittest.TestCase):
    def test_bundled_operator_skill_frontmatter_and_links_are_valid(self) -> None:
        skill = (
            Path(__file__).resolve().parents[1]
            / "puppetmaster"
            / "skills"
            / "puppetmaster"
            / "SKILL.md"
        )
        text = skill.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        for required in ("name:", "description:", "compatibility:", "license:"):
            self.assertIn(required, frontmatter)
        for unsupported in ("version:", "author:", "platforms:"):
            self.assertNotIn(unsupported, frontmatter)
        links = re.findall(r"\]\((references/[^)]+\.md)\)", text)
        self.assertEqual(
            set(links),
            {"references/monitoring-state-machine.md", "references/recovery.md"},
        )
        for link in links:
            self.assertTrue((skill.parent / link).is_file(), link)

        from puppetmaster.installers import install_hermes_skill

        with tempfile.TemporaryDirectory() as tmp:
            outcome = install_hermes_skill(skills_dir=Path(tmp))
            self.assertEqual(outcome.status, "installed")
            installed = Path(outcome.target)
            for link in links:
                self.assertTrue((installed / link).is_file(), link)

    def test_cost_drift_uses_tokens_and_nominal_cost_for_plan_models(self) -> None:
        task_id = "task_cost"
        artifacts = [
            Artifact(
                job_id="job_cost",
                task_id=task_id,
                type=ArtifactType.ROUTING,
                created_by="router",
                confidence=1.0,
                evidence=["route"],
                payload={
                    "model_id": "codex/gpt-5-6-luna",
                    "adapter": "codex",
                    "policy": "balanced",
                    "estimated_tokens_in": 100,
                    "estimated_tokens_out": 100,
                    "estimated_cost_usd": 0.0,
                    "nominal_cost_usd": 0.01,
                },
            ),
            Artifact(
                job_id="job_cost",
                task_id=task_id,
                type=ArtifactType.VERIFICATION,
                created_by="worker",
                confidence=1.0,
                evidence=["usage"],
                payload={
                    "model": "gpt-5.6-luna",
                    "check": "usage",
                    "result": "passed",
                    "tokens_in": 1000,
                    "tokens_out": 1000,
                    "tokens_estimated": False,
                },
            ),
        ]
        priced = price_job(artifacts, starter_registry())
        self.assertEqual(priced.route_estimated_tokens, 200)
        self.assertEqual(priced.measured_usage_tokens, 2000)
        self.assertEqual(priced.token_estimate_drift_ratio, 10.0)
        self.assertGreater(priced.nominal_usage_cost_usd, 0.0)
        self.assertIsNotNone(priced.nominal_cost_drift_ratio)

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            job = store.create_job("cost")
            task = Task(job_id=job.id, id=task_id, role="cost", instruction="measure")
            store.save_task(task)
            for artifact in artifacts:
                store.save_artifact(
                    Artifact(
                        job_id=job.id,
                        task_id=artifact.task_id,
                        type=artifact.type,
                        created_by=artifact.created_by,
                        confidence=artifact.confidence,
                        evidence=artifact.evidence,
                        payload=artifact.payload,
                    )
                )
            receipt = build_job_receipt(store, job.id)
            self.assertEqual(receipt["estimate_drift"]["token_ratio"], 10.0)

    def test_streamed_output_budget_is_a_real_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = Task(job_id="job_budget", role="test", instruction="emit")
            completed = run_streamed_subprocess(
                command=[
                    sys.executable,
                    "-c",
                    "import sys,time; print('x'*4096); sys.stdout.flush(); time.sleep(5)",
                ],
                env=os.environ.copy(),
                task=task,
                sidecar_name="budget",
                timeout_seconds=10,
                cwd=tmp,
                max_output_bytes=128,
            )
            self.assertTrue(completed.output_limit_hit)
            self.assertIn("max_output_bytes exceeded", completed.stderr)

    def test_runtime_capabilities_do_not_claim_unimplemented_limits(self) -> None:
        self.assertNotIn(
            "output_bytes", adapter_runtime_capabilities("local")["enforced_runtime_limits"]
        )
        self.assertIn(
            "output_bytes", adapter_runtime_capabilities("codex")["enforced_runtime_limits"]
        )
        schema = goal_schema("goal")
        self.assertIn("max_output_bytes", schema["properties"])
        self.assertNotIn("max_total_tokens", schema["properties"])
        self.assertNotIn("no_progress_timeout_seconds", schema["properties"])

    def test_job_ref_state_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            owner = Path(tmp) / "owner"
            with patch(
                "puppetmaster.state.find_state_dir_for_job", return_value=owner
            ):
                with self.assertRaisesRegex(ValueError, "state_id"):
                    mcp_state_dir(
                        {
                            "cwd": tmp,
                            "job_ref": {"job_id": "job_x", "state_id": "wrong"},
                        }
                    )

    def test_cli_await_rejects_terminal_empty_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            job = store.create_job("empty")
            store.update_job_status(job.id, JobStatus.COMPLETE)
            rc = cli_main(
                [
                    "--state-dir",
                    str(store.root),
                    "--backend",
                    "sqlite",
                    "await",
                    job.id,
                    "--timeout-seconds",
                    "0.1",
                ]
            )
            self.assertEqual(rc, 1)

    def test_run_rejects_ambiguous_goal_transports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            goal_file = Path(tmp) / "goal.txt"
            goal_file.write_text("file goal", encoding="utf-8")
            rc = cli_main(
                [
                    "--state-dir",
                    str(Path(tmp) / "state"),
                    "run",
                    "positional goal",
                    "--goal-file",
                    str(goal_file),
                ]
            )
            self.assertEqual(rc, 1)

    def test_read_job_state_reports_the_post_reaper_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            job = store.create_job("liveness")
            store.update_job_status(job.id, JobStatus.RUNNING)
            with patch(
                "puppetmaster.cli.commands_jobs._reap_quietly",
                side_effect=lambda target: target.update_job_status(
                    job.id, JobStatus.STALLED
                ),
            ):
                state = read_job_state(store, job.id)
            self.assertTrue(state["terminal"])
            self.assertEqual(state["status"], "stalled")

    def test_progress_separates_liveness_from_substantive_artifacts(self) -> None:
        for store_type in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_type.__name__), tempfile.TemporaryDirectory() as tmp:
                store = store_type(Path(tmp))
                store.init()
                job = store.create_job("progress")
                store.emit(job.id, "run.heartbeat", {"worker": "test"})
                progress = store.status_snapshot(job.id, compact=True)["progress"]
                self.assertIsNotNone(progress["last_liveness_at"])
                self.assertIsNone(progress["last_substantive_artifact_at"])

    def test_analysis_worker_diff_fails_the_orchestration_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            job = store.create_job("analysis")
            task = Task(job_id=job.id, role="analysis", instruction="inspect")
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker",
                    confidence=1.0,
                    evidence=["snapshot"],
                    payload={
                        "check": "snapshot",
                        "result": "passed",
                        "baseline_diff_present": True,
                        "worker_diff_present": True,
                    },
                )
            )
            orchestrator = Orchestrator(store)
            orchestrator._enforce_analysis_no_worker_diff(job, [task])
            blocked = [
                artifact
                for artifact in store.list_artifacts(job.id)
                if (artifact.payload or {}).get("failure") == "analysis_worker_diff"
            ]
            self.assertEqual(len(blocked), 1)
            self.assertEqual(orchestrator._final_job_status(job), JobStatus.FAILED)

    def test_omitted_roles_make_one_assignment_and_structured_scope_survives(self) -> None:
        specs = build_analysis_swarm_specs(
            "goal",
            [],
            adapter="local",
            cwd="/repo",
        )
        self.assertEqual([spec.role for spec in specs], ["analysis"])
        structured = build_analysis_swarm_specs(
            "goal",
            [{
                "name": "mapper",
                "instruction": "Map only the CLI surface.",
                "source_scope": ["puppetmaster/cli"],
                "negative_scope": ["docs"],
            }],
            adapter="local",
            cwd="/repo",
        )
        self.assertEqual(structured[0].payload["source_scope"], ["puppetmaster/cli"])
        self.assertEqual(structured[0].payload["negative_scope"], ["docs"])

    def test_duplicate_legacy_roles_warn_before_mcp_handoff(self) -> None:
        response = {
            "content": [{"type": "text", "text": '{"job_id":"job_x"}'}],
            "isError": False,
        }
        with patch.object(
            mcp_server, "_platform_lock_preflight", return_value=None
        ), patch.object(
            mcp_server, "write_generated_swarm_config", return_value=Path("config.json")
        ), patch.object(mcp_server, "start_cli", return_value=response):
            result = mcp_server.start_swarm(
                {
                    "goal": "same goal",
                    "roles": ["explore", "review"],
                    "adapter": "local",
                }
            )
        body = json.loads(result["content"][0]["text"])
        self.assertEqual(body["warnings"][0]["kind"], "duplicate_legacy_roles")
        self.assertEqual(body["warnings"][0]["fan_out_multiplier"], 2)

    def test_platform_neutral_start_uses_one_generated_assignment(self) -> None:
        response = {
            "content": [{"type": "text", "text": '{"job_id":"job_x"}'}],
            "isError": False,
        }
        with patch.object(
            mcp_server, "_enabled_swarm_adapters", return_value=["codex", "local"]
        ), patch.object(
            mcp_server, "_platform_lock_preflight", return_value=None
        ), patch.object(
            mcp_server, "write_generated_swarm_config", return_value=Path("config.json")
        ) as write_config, patch.object(
            mcp_server, "start_cli", return_value=response
        ):
            mcp_server.start_swarm({"goal": "one assignment"})
        self.assertEqual(write_config.call_args.args[1], [])
        self.assertEqual(write_config.call_args.args[2], "codex")

    def test_filtered_refs_advance_cursor_without_payload_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp))
            store.init()
            job = store.create_job("feed")
            task = Task(job_id=job.id, role="analysis", instruction="inspect")
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by="test",
                    payload={"claim": "claim", "large": "x" * 500},
                    confidence=0.9,
                    evidence=["test:file.py:1"],
                )
            )
            items, cursor = artifact_feed_since(
                store,
                job.id,
                refs=True,
                include_types=["finding"],
                max_bytes=4096,
            )
            self.assertGreater(cursor, 0)
            self.assertEqual(len(items), 1)
            self.assertNotIn("large", items[0]["artifact"])

            bounded, bounded_cursor = artifact_feed_since(
                store,
                job.id,
                refs=False,
                max_bytes=256,
            )
            self.assertEqual(bounded_cursor, cursor)
            self.assertLessEqual(
                len(json.dumps(bounded, separators=(",", ":")).encode("utf-8")),
                256,
            )
            self.assertTrue(bounded[0]["artifact"]["payload_omitted"])

    def test_cancelled_is_terminal_but_not_successful(self) -> None:
        result = delivery_verdict("cancelled")
        self.assertEqual(result["verdict"], "blocked")
        self.assertFalse(result["successful"])
        unknown_quality = delivery_verdict("complete")
        self.assertEqual(unknown_quality["verdict"], "blocked")
        self.assertFalse(unknown_quality["successful"])


if __name__ == "__main__":
    unittest.main()
