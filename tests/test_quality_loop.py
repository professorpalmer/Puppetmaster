"""Opt-in cleanup and bounded same-adapter review repair."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.quality_loop import (
    maybe_requeue_review_repair,
    review_loop_enabled,
    run_cleanup_pass,
)
from puppetmaster.sqlite_store import SQLiteSwarmStore


class QualityLoopTests(unittest.TestCase):
    def test_cleanup_skips_when_disabled_or_no_paths(self) -> None:
        task = Task(job_id="j", role="implement", instruction="x", payload={})
        arts = run_cleanup_pass(task, [])
        self.assertEqual(arts, [])
        task = Task(
            job_id="j",
            role="implement",
            instruction="x",
            payload={"cleanup": True},
        )
        arts = run_cleanup_pass(task, [])
        self.assertEqual(arts[-1].payload["kind"], "cleanup")
        self.assertEqual(arts[-1].payload["note"], "no edited paths")

    def test_cleanup_failure_does_not_raise(self) -> None:
        task = Task(
            job_id="j",
            role="implement",
            instruction="x",
            payload={"cleanup": True, "cwd": "."},
        )
        patch = Artifact(
            job_id="j",
            task_id=task.id,
            type=ArtifactType.PATCH,
            created_by="w",
            confidence=0.9,
            evidence=["test:patch"],
            payload={"change": "edit", "files": ["src/a.py"], "changed_files": ["src/a.py"]},
        )

        def boom(_command, **_kwargs):
            raise RuntimeError("ruff missing")

        arts = run_cleanup_pass(task, [patch], runner=boom)
        self.assertEqual(arts[-1].payload["result"] if False else arts[-1].payload.get("kind"), "cleanup")
        self.assertEqual(arts[-1].payload.get("note"), "cleanup failed; implement kept")

    def test_review_repair_requeues_same_adapter_with_reasons(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("repair")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="ship it",
                adapter="agentic",
                status=TaskStatus.FAILED,
                payload={"review_loop": True, "review_loop_limit": 2, "model": "cheap"},
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GATE,
                    created_by="reviewer",
                    confidence=0.9,
                    evidence=["test:review"],
                    payload={
                        "gate": "review",
                        "kind": "review",
                        "passed": False,
                        "reason": "missing test",
                    },
                )
            )
            repaired = maybe_requeue_review_repair(store, job.id)
            self.assertEqual(len(repaired), 1)
            self.assertEqual(repaired[0].status, TaskStatus.QUEUED)
            self.assertEqual(repaired[0].adapter, "agentic")
            self.assertTrue(repaired[0].payload["allow_dirty"])
            self.assertIn("missing test", repaired[0].instruction)
            self.assertTrue(review_loop_enabled(repaired[0].payload))

    def test_review_loop_does_not_multiply_with_model_escalation(self) -> None:
        from puppetmaster.orchestrator import Orchestrator

        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("no multiply")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="x",
                adapter="agentic",
                status=TaskStatus.FAILED,
                payload={
                    "review_loop": True,
                    "router_model_id": "agentic/x",
                    "review_escalation_attempts": 0,
                },
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GATE,
                    created_by="reviewer",
                    confidence=0.9,
                    evidence=["test:review"],
                    payload={
                        "gate": "review",
                        "kind": "review",
                        "passed": False,
                        "reason": "nits",
                    },
                )
            )
            orch = Orchestrator(store)
            self.assertEqual(orch._reroute_failed_review(job), 0)
            self.assertIn(task.id, orch._review_pending_reroute_ids(job))


class ReviewRepairResumeTests(unittest.TestCase):
    """A review repair continues the rejected attempt's own provider session."""

    SESSION = "11111111-2222-3333-4444-555555555555"
    THREAD = "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"

    def setUp(self) -> None:
        from unittest.mock import patch

        homes = TemporaryDirectory()
        self.addCleanup(homes.cleanup)
        self.homes = Path(homes.name)
        env = patch.dict(
            os.environ,
            {"CLAUDE_CONFIG_DIR": str(self.homes / "claude"), "CODEX_HOME": str(self.homes / "codex")},
        )
        env.start()
        self.addCleanup(env.stop)
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = SQLiteSwarmStore(Path(tmp.name) / ".puppetmaster")
        self.store.init()

    def _rejected(self, adapter: str, receipt: dict, payload: dict = None) -> Task:
        job = self.store.create_job("repair")
        task = Task(
            job_id=job.id,
            role="implement",
            instruction="ship it",
            adapter=adapter,
            status=TaskStatus.FAILED,
            payload={"review_loop": True, **(payload or {})},
        )
        self.store.save_task(task)
        if receipt:
            self.store.save_artifact(
                Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                    created_by="worker", confidence=0.9, evidence=[f"adapter:{adapter}"],
                    payload={"adapter": adapter, "check": "ship it", "result": "passed", **receipt},
                )
            )
        self.store.save_artifact(
            Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.GATE, created_by="reviewer",
                confidence=0.9, evidence=["test:review"],
                payload={"gate": "review", "kind": "review", "passed": False, "reason": "missing docstring"},
            )
        )
        return task

    def _claude_session_on_disk(self) -> None:
        project = self.homes / "claude" / "projects" / "-repo"
        project.mkdir(parents=True)
        (project / f"{self.SESSION}.jsonl").write_text("{}\n", encoding="utf-8")

    def test_claude_repair_resumes_the_rejected_attempts_session(self) -> None:
        self._claude_session_on_disk()
        task = self._rejected("claude-code", {"session_id": self.SESSION})
        repaired = maybe_requeue_review_repair(self.store, task.job_id)
        resume = repaired[0].payload["resume"]
        self.assertEqual(resume["status"], "resolved")
        self.assertEqual((resume["adapter"], resume["session_id"]), ("claude-code", self.SESSION))
        self.assertEqual(resume["from_task_id"], task.id)
        self.assertIn("missing docstring", repaired[0].instruction)
        events = [e for e in self.store.read_events(task.job_id) if e.get("event") == "quality.review_repair"]
        self.assertTrue(events[-1]["payload"]["resumed"])

    def test_switch_off_runs_fresh_and_drops_a_stale_record(self) -> None:
        self._claude_session_on_disk()
        stale = {"status": "resolved", "adapter": "claude-code", "session_id": "old-session"}
        task = self._rejected(
            "claude-code", {"session_id": self.SESSION}, {"review_repair_resume": False, "resume": stale}
        )
        repaired = maybe_requeue_review_repair(self.store, task.job_id)
        self.assertNotIn("resume", repaired[0].payload)

    def test_missing_session_records_why_it_runs_fresh(self) -> None:
        (self.homes / "claude" / "projects").mkdir(parents=True)
        task = self._rejected("claude-code", {"session_id": self.SESSION})
        resume = maybe_requeue_review_repair(self.store, task.job_id)[0].payload["resume"]
        self.assertEqual(resume["status"], "unavailable")
        self.assertIn("not in the local session store", resume["reason"])

    def test_ephemeral_codex_attempt_records_why_it_runs_fresh(self) -> None:
        task = self._rejected("codex", {"thread_id": self.THREAD, "ephemeral": True})
        resume = maybe_requeue_review_repair(self.store, task.job_id)[0].payload["resume"]
        self.assertEqual(resume["status"], "unavailable")
        self.assertIn("ephemeral", resume["reason"])

    def test_no_session_receipt_runs_fresh_as_before(self) -> None:
        task = self._rejected("claude-code", {})
        self.assertNotIn("resume", maybe_requeue_review_repair(self.store, task.job_id)[0].payload)

    def test_codex_review_loop_tasks_default_to_persisted_threads(self) -> None:
        from unittest.mock import patch

        from puppetmaster.adapters import CodexAdapter

        task = Task(job_id="j", role="implement", instruction="x", adapter="codex",
                    payload={"review_loop": True, "cwd": "."})
        with patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)), \
                patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p):
            prepared = CodexAdapter()._prepare_cli_invocation(task, "goal", "w", Path("."), "/usr/bin/codex")
        self.assertNotIn("--ephemeral", prepared.command)
        plain = Task(job_id="j", role="implement", instruction="x", adapter="codex", payload={"cwd": "."})
        with patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)), \
                patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p):
            prepared = CodexAdapter()._prepare_cli_invocation(plain, "goal", "w", Path("."), "/usr/bin/codex")
        self.assertIn("--ephemeral", prepared.command)
