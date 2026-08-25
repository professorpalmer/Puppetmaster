"""Portable working-set cache v1 (index sidecar + instruction-aware reuse)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest

from puppetmaster.adapters._prompts import build_implement_prompt, with_prewalk_plan
from puppetmaster.job_brief import build_job_brief, write_job_brief
from puppetmaster.models import (
    AgentRun,
    Artifact,
    ArtifactType,
    JobStatus,
    Task,
    TaskStatus,
)
from puppetmaster.prewalk import IMPLEMENT_ROLE, PREWALK_PLAN_SECTION_HEADER
from puppetmaster.store import SwarmStore
from puppetmaster.validation import validation_status_of
from puppetmaster.working_set import (
    ARTIFACT_INDEX_FILENAME,
    WORKING_SET_BRIEF_LINE,
    maybe_reuse_artifacts,
    read_artifact_index,
    rebuild_artifact_index,
    reuse_fingerprint,
    stamp_fresh_validation,
    working_set_brief_line,
    write_artifact_index,
)


INSTRUCTION = "Find the retry bug in src/a.py"


def _git_init_with_file(root: Path, rel: str, content: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", rel], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _analysis_payload(cwd: Path, *, extra: Optional[dict] = None) -> dict:
    payload: dict[str, Any] = {
        "cwd": str(cwd),
        "source_scope": ["src/a.py"],
        "read_only": True,
    }
    if extra:
        payload.update(extra)
    return payload


def _finding(
    job_id: str,
    task_id: str,
    claim: str,
    *,
    details: Optional[str] = None,
    validation: Optional[dict[str, Any]] = None,
) -> Artifact:
    payload: dict[str, Any] = {"claim": claim}
    if details is not None:
        payload["details"] = details
    if validation is not None:
        payload["validation"] = validation
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="test",
        payload=payload,
        confidence=0.9,
        evidence=["src/a.py"],
    )


class ArtifactIndexTests(unittest.TestCase):
    def test_write_job_brief_creates_empty_index_sidecar(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print(1)\n", encoding="utf-8")
            job_dir = root / "jobs" / "job-1"
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                return_value="",
            ):
                written = write_job_brief(job_dir, "ship it", root)
            self.assertIsNotNone(written)
            index_path = job_dir / ARTIFACT_INDEX_FILENAME
            self.assertTrue(index_path.is_file())
            self.assertEqual(read_artifact_index(job_dir), [])
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(raw.get("artifacts"), [])

    def test_compact_refs_omit_full_finding_bodies(self) -> None:
        with TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            huge = "FINDING_BODY_" + ("x" * 4000)
            artifact = _finding(
                "job",
                "task",
                "short claim",
                details=huge,
            )
            path = write_artifact_index(job_dir, [artifact])
            self.assertIsNotNone(path)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(huge, text)
            self.assertNotIn("FINDING_BODY_", text)
            refs = read_artifact_index(job_dir)
            self.assertEqual(len(refs), 1)
            self.assertNotIn("details", refs[0])
            self.assertIn("id", refs[0])
            self.assertIn("claim", refs[0])

    def test_job_brief_line_stable_when_index_grows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "lib.py").write_text("def f(): pass\n", encoding="utf-8")
            job_dir = Path(tmp) / "job-state"
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                return_value="lib.py:1 -> f()",
            ):
                before = build_job_brief("map the library", root)
            write_artifact_index(job_dir, [])
            grown = [
                _finding("j", "t1", "first"),
                _finding("j", "t2", "second"),
            ]
            write_artifact_index(job_dir, grown)
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                return_value="lib.py:1 -> f()",
            ):
                after = build_job_brief("map the library", root)
            self.assertEqual(before, after)
            self.assertIn(WORKING_SET_BRIEF_LINE, before)
            self.assertEqual(before.count(working_set_brief_line()), 1)
            self.assertIn(ARTIFACT_INDEX_FILENAME, before)
            self.assertNotIn("first", before)
            self.assertNotIn("second", before)
            self.assertEqual(len(read_artifact_index(job_dir)), 2)

    def test_working_set_kill_switch_skips_index_and_brief_line(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.py").write_text("x=1\n", encoding="utf-8")
            job_dir = root / "jobs" / "job-1"
            with mock.patch.dict(os.environ, {"PUPPETMASTER_WORKING_SET": "0"}):
                with mock.patch(
                    "puppetmaster.codegraph.codegraph_context",
                    return_value="",
                ):
                    brief = build_job_brief("goal", root)
                    path = write_job_brief(job_dir, "goal", root)
            self.assertIsNotNone(path)
            self.assertNotIn(ARTIFACT_INDEX_FILENAME, brief)
            self.assertNotIn(WORKING_SET_BRIEF_LINE, brief)
            self.assertFalse((job_dir / ARTIFACT_INDEX_FILENAME).exists())

    def test_write_job_brief_does_not_wipe_existing_index(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print(1)\n", encoding="utf-8")
            job_dir = root / "jobs" / "job-1"
            job_dir.mkdir(parents=True)
            index_path = job_dir / ARTIFACT_INDEX_FILENAME
            index_path.write_text(
                json.dumps({"artifacts": [{"id": "keep-me"}]}) + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                return_value="",
            ):
                write_job_brief(job_dir, "goal", root)
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifacts"][0]["id"], "keep-me")


class ReuseFingerprintTests(unittest.TestCase):
    def test_same_scope_different_instruction_does_not_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            payload = _analysis_payload(root)
            task_a = Task(
                job_id="job",
                role="explore",
                instruction=INSTRUCTION,
                payload=payload,
            )
            task_b = Task(
                job_id="job",
                role="explore",
                instruction="A different question about the same file",
                payload=payload,
            )
            fp_a = reuse_fingerprint(task_a)
            fp_b = reuse_fingerprint(task_b)
            self.assertIsNotNone(fp_a)
            self.assertIsNotNone(fp_b)
            self.assertNotEqual(fp_a, fp_b)

    def test_same_instruction_different_prompt_does_not_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            base = _analysis_payload(root)
            task_a = Task(
                job_id="job",
                role="explore",
                instruction=INSTRUCTION,
                payload=dict(base, prompt=INSTRUCTION),
            )
            task_b = Task(
                job_id="job",
                role="explore",
                instruction=INSTRUCTION,
                payload=dict(base, prompt="A different assembled prompt"),
            )
            fp_a = reuse_fingerprint(task_a)
            fp_b = reuse_fingerprint(task_b)
            self.assertIsNotNone(fp_a)
            self.assertIsNotNone(fp_b)
            self.assertNotEqual(fp_a, fp_b)

    def test_missing_cwd_or_scope_fails_closed(self) -> None:
        task = Task(
            job_id="job",
            role="explore",
            instruction=INSTRUCTION,
            payload={"read_only": True},
        )
        self.assertIsNone(reuse_fingerprint(task))


class MaybeReuseTests(unittest.TestCase):
    def _seed_labeled_finding(self, store, root: Path, instruction: str):
        job = store.create_job(instruction)
        store.update_job_status(job.id, JobStatus.RUNNING)
        source_task = Task(
            job_id=job.id,
            role="explore",
            instruction=instruction,
            status=TaskStatus.COMPLETE,
            payload=_analysis_payload(root),
        )
        store.save_task(source_task)
        finding = _finding(job.id, source_task.id, "retry is missing")
        stamped = stamp_fresh_validation(source_task, [finding])
        self.assertTrue(stamped[0].payload.get("validation"))
        store.save_artifact(stamped[0])
        return job, source_task, stamped[0]

    def test_maybe_reuse_returns_reused_clones(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, source_task, source = self._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.QUEUED,
                payload=_analysis_payload(root),
            )
            store.save_task(incoming)
            clones = maybe_reuse_artifacts(store, incoming)
            self.assertEqual(len(clones), 1)
            self.assertEqual(validation_status_of(clones[0]), "reused")
            self.assertNotEqual(clones[0].id, source.id)
            self.assertEqual(clones[0].task_id, incoming.id)
            self.assertEqual(
                clones[0].payload["validation"]["source_artifact_ids"],
                [source.id],
            )

    def test_persist_reused_does_not_enqueue_follow_ups(self) -> None:
        from puppetmaster.working_set import persist_reused_artifacts

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            _job, _source_task, _source = self._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=_job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.QUEUED,
                payload=_analysis_payload(root),
            )
            store.save_task(incoming)
            clones = maybe_reuse_artifacts(store, incoming)
            with mock.patch.object(
                store, "maybe_enqueue_follow_ups_from_artifact"
            ) as enqueue:
                persist_reused_artifacts(
                    store, incoming, clones, worker_id="test"
                )
            enqueue.assert_not_called()

    def test_different_instruction_does_not_skip(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _source_task, _source = self._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=job.id,
                role="explore",
                instruction="What does this file export?",
                status=TaskStatus.QUEUED,
                payload=_analysis_payload(root),
            )
            self.assertEqual(maybe_reuse_artifacts(store, incoming), [])

    def test_unlabeled_artifacts_do_not_skip(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job(INSTRUCTION)
            source_task = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.COMPLETE,
                payload=_analysis_payload(root),
            )
            store.save_task(source_task)
            store.save_artifact(
                _finding(job.id, source_task.id, "unlabeled legacy")
            )
            incoming = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                payload=_analysis_payload(root),
            )
            self.assertEqual(maybe_reuse_artifacts(store, incoming), [])

    def test_reuse_kill_switch_and_payload_opt_out(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _source_task, _source = self._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                payload=_analysis_payload(root),
            )
            with mock.patch.dict(os.environ, {"PUPPETMASTER_WORKING_SET_REUSE": "0"}):
                self.assertEqual(maybe_reuse_artifacts(store, incoming), [])
            opted = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                payload=_analysis_payload(root, extra={"skip_working_set_reuse": True}),
            )
            self.assertEqual(maybe_reuse_artifacts(store, opted), [])
            with mock.patch.dict(os.environ, {"PUPPETMASTER_WORKING_SET": "0"}):
                self.assertEqual(maybe_reuse_artifacts(store, incoming), [])

    def test_prewalk_implement_never_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _source_task, _source = self._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction=INSTRUCTION,
                payload=_analysis_payload(
                    root,
                    extra={"prewalk": True, "mode": "implement", "read_only": True},
                ),
            )
            self.assertEqual(maybe_reuse_artifacts(store, incoming), [])


class PrewalkInjectionUnchangedTests(unittest.TestCase):
    def test_prewalk_still_injects_plan_bodies(self) -> None:
        artifacts = [
            {
                "type": "decision",
                "payload": {
                    "decision": "Add retry helper in client.py",
                    "why": "Centralize backoff",
                    "plan": ["Create retry_with_backoff", "Wire callers"],
                },
            }
        ]
        task = Task(
            job_id="job-prewalk-1",
            role=IMPLEMENT_ROLE,
            instruction="Apply the upstream plan",
            payload={
                "prewalk": True,
                "prewalk_artifacts": artifacts,
                "mode": "implement",
                "prompt": "Apply the plan",
            },
        )
        assembled = build_implement_prompt("Apply the plan")
        result = with_prewalk_plan(assembled, task)
        self.assertIn("Decision: Add retry helper in client.py", result)
        self.assertIn("Why: Centralize backoff", result)
        self.assertIn("Create retry_with_backoff", result)
        self.assertIn(PREWALK_PLAN_SECTION_HEADER, result)


class WorkerRuntimeReuseTests(unittest.TestCase):
    def _store_job(self, tmp: str):
        store = SwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        job = store.create_job(INSTRUCTION)
        store.update_job_status(job.id, JobStatus.RUNNING)
        return store, job

    def test_warm_skip_does_not_call_adapter(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store, job = self._store_job(tmp)
            source_task = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.COMPLETE,
                payload=_analysis_payload(root),
            )
            store.save_task(source_task)
            finding = _finding(job.id, source_task.id, "retry is missing")
            store.save_artifact(stamp_fresh_validation(source_task, [finding])[0])

            queued = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.QUEUED,
                adapter="local",
                payload=_analysis_payload(root),
            )
            store.save_task(queued)

            class _BoomWorker:
                def __init__(self, role, worker_id=None):
                    raise AssertionError("adapter should not run on warm skip")

            runtime = WorkerRuntime(
                store=store, job_id=job.id, role="explore", worker_id="w"
            )
            with mock.patch(
                "puppetmaster.worker_runtime.LocalWorker", _BoomWorker
            ):
                self.assertTrue(runtime.run_once())
            updated = store.get_task_by_id(queued.id)
            self.assertEqual(updated.status, TaskStatus.COMPLETE)
            reused = [
                art
                for art in store.list_artifacts(job.id)
                if validation_status_of(art) == "reused"
            ]
            self.assertTrue(reused)
            self.assertEqual(reused[0].task_id, queued.id)

    def test_different_instruction_calls_adapter(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store, job = self._store_job(tmp)
            source_task = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.COMPLETE,
                payload=_analysis_payload(root),
            )
            store.save_task(source_task)
            finding = _finding(job.id, source_task.id, "retry is missing")
            store.save_artifact(stamp_fresh_validation(source_task, [finding])[0])

            queued = Task(
                job_id=job.id,
                role="explore",
                instruction="Describe the public API",
                status=TaskStatus.QUEUED,
                adapter="local",
                payload=_analysis_payload(root),
            )
            store.save_task(queued)
            called = {"n": 0}

            class _FakeWorker:
                def __init__(self, role, worker_id=None):
                    self.role = role
                    self.worker_id = worker_id or "w"

                def run(self, t, goal):
                    called["n"] += 1
                    run = AgentRun(
                        job_id=t.job_id,
                        task_id=t.id,
                        role=t.role,
                        worker_id=self.worker_id,
                        status=TaskStatus.COMPLETE,
                    )
                    art = _finding(t.job_id, t.id, "adapter ran")
                    return run, [art]

            runtime = WorkerRuntime(
                store=store, job_id=job.id, role="explore", worker_id="w"
            )
            with mock.patch(
                "puppetmaster.worker_runtime.LocalWorker", _FakeWorker
            ):
                self.assertTrue(runtime.run_once())
            self.assertEqual(called["n"], 1)
            self.assertEqual(store.get_task_by_id(queued.id).status, TaskStatus.COMPLETE)

    def test_real_run_stamps_fresh_and_rebuilds_index(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store, job = self._store_job(tmp)
            queued = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.QUEUED,
                adapter="local",
                payload=_analysis_payload(root),
            )
            store.save_task(queued)

            class _FakeWorker:
                def __init__(self, role, worker_id=None):
                    self.worker_id = worker_id or "w"

                def run(self, t, goal):
                    run = AgentRun(
                        job_id=t.job_id,
                        task_id=t.id,
                        role=t.role,
                        worker_id=self.worker_id,
                        status=TaskStatus.COMPLETE,
                    )
                    return run, [_finding(t.job_id, t.id, "fresh from adapter")]

            runtime = WorkerRuntime(
                store=store, job_id=job.id, role="explore", worker_id="w"
            )
            with mock.patch(
                "puppetmaster.worker_runtime.LocalWorker", _FakeWorker
            ):
                runtime.run_once()
            artifacts = [
                art
                for art in store.list_artifacts(job.id)
                if art.type == ArtifactType.FINDING
            ]
            self.assertTrue(artifacts)
            self.assertEqual(validation_status_of(artifacts[0]), "fresh")
            self.assertEqual(
                artifacts[0].payload["validation"]["fingerprint"],
                reuse_fingerprint(queued),
            )
            refs = read_artifact_index(store.job_dir(job.id))
            self.assertTrue(refs)
            encoded = json.dumps(refs)
            self.assertNotIn("fresh from adapter" * 2, encoded)

    def test_rebuild_from_store_list(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, source_task, source = MaybeReuseTests()._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            path = rebuild_artifact_index(store.job_dir(job.id), store, job.id)
            self.assertIsNotNone(path)
            refs = read_artifact_index(store.job_dir(job.id))
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["id"], source.id)


class WorkingSetUsageTests(unittest.TestCase):
    def test_persist_records_numbers_only_skip(self) -> None:
        from puppetmaster.working_set import persist_reused_artifacts
        from puppetmaster import working_set_usage as wsu

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "ws.jsonl"
            root = Path(tmp) / "repo"
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            helper = MaybeReuseTests()
            job, _source_task, _source = helper._seed_labeled_finding(
                store, root, INSTRUCTION
            )
            incoming = Task(
                job_id=job.id,
                role="explore",
                instruction=INSTRUCTION,
                status=TaskStatus.QUEUED,
                payload=_analysis_payload(root),
            )
            store.save_task(incoming)
            clones = maybe_reuse_artifacts(store, incoming)
            env = {
                "PUPPETMASTER_WORKING_SET_USAGE_LOG": str(log_path),
                "PUPPETMASTER_WORKING_SET_USAGE": "1",
            }
            with mock.patch.dict(os.environ, env):
                persisted = persist_reused_artifacts(
                    store, incoming, clones, worker_id="test"
                )
                records = wsu.load_usage()
            self.assertEqual(len(persisted), 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["kind"], "reuse_skip")
            self.assertEqual(records[0]["artifacts"], 1)
            self.assertNotIn("instruction", records[0])
            rolled = wsu.aggregate(records, skip_baseline_tokens=4000)
            self.assertEqual(rolled["skips"], 1)
            self.assertEqual(rolled["avoided_tokens_est"], 4000)


class HermesIsolationTests(unittest.TestCase):
    def test_hermes_chat_command_does_not_resume_sessions(self) -> None:
        from puppetmaster.adapters.hermes import build_hermes_chat_command

        command = build_hermes_chat_command(
            prompt="do the work",
            model="deepseek/deepseek-v4-flash",
            provider="openrouter",
        )
        joined = " ".join(command)
        self.assertNotIn("--resume", command)
        self.assertNotIn("resume", joined.lower().split())
        self.assertIn("chat", command)
        self.assertIn("-q", command)


if __name__ == "__main__":
    unittest.main()
