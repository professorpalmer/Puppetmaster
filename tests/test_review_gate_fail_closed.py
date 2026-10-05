"""Focused RED tests for requested, attributable review gates.

These tests intentionally exercise only the public gate evaluation seam and
the judge resolver.  They never call a live model.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from puppetmaster.gates import ReviewVerdict, evaluate_task_gates
from puppetmaster.model_registry import ModelSpec, registry_digest, save_registry
from puppetmaster.models import Task
from puppetmaster.sqlite_store import SQLiteSwarmStore


class RequestedReviewFailClosedTests(TestCase):

    def test_string_false_cannot_pass_a_review_gate(self) -> None:
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(repo)
            judge = Mock(id="cursor/gpt-5-6", adapter="cursor", adapter_model_name="gpt-5.6")
            marker = gates._REVIEW_VERDICT_MARKER
            artifacts = [
                gates.Artifact(
                    job_id=task.job_id, task_id=task.id, type=gates.ArtifactType.VERIFICATION,
                    created_by="judge", confidence=1.0, evidence=[],
                    payload={"stdout": marker + ' {"pass":"false","reasons":[]}'},
                )
            ]
            with patch("puppetmaster.adapters.get_adapter", return_value=Mock(run=Mock(return_value=artifacts))):
                with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}, clear=False):
                    verdict = gates.default_judge_review(
                        prompt="review", judge=judge, cwd=repo, timeout=1, task=task
                    )
            self.assertFalse(verdict.passed)
    def _store(self, root: str) -> SQLiteSwarmStore:
        store = SQLiteSwarmStore(Path(root) / ".puppetmaster")
        store.init()
        return store

    def _repo(self, root: str) -> Path:
        repo = Path(root) / "repo"
        repo.mkdir(parents=True)
        for argv in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "review@test.invalid"],
            ["git", "config", "user.name", "Review Test"],
        ):
            subprocess.run(argv, cwd=repo, check=True, capture_output=True)
        (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "seed"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        (repo / "feature.py").write_text("def feature():\n    return 1\n", encoding="utf-8")
        return repo

    def _task(self, repo: Path, *, review=True, router_model_id="cursor/gpt-5-5") -> Task:
        return Task(
            job_id="job-review",
            id="task-review",
            role="implement",
            instruction="add the feature",
            payload={
                "cwd": str(repo),
                "review": review,
                "router_model_id": router_model_id,
            },
        )

    def _raw_review_task(self, repo: Path, review_gate: dict) -> Task:
        return Task(
            job_id="job-review",
            id="task-review",
            role="implement",
            instruction="add the feature",
            payload={
                "cwd": str(repo),
                "router_model_id": "cursor/gpt-5-5",
                "gates": [{"kind": "review", **review_gate}],
            },
        )

    @staticmethod
    def _review_payload(evaluation) -> dict:
        matches = [
            artifact.payload
            for artifact in evaluation.artifacts
            if (artifact.payload or {}).get("kind") == "review"
        ]
        assert len(matches) == 1
        return matches[0]

    def test_explicit_review_fails_when_live_review_flag_is_missing(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(repo, review=True)
            judge = Mock(
                id="cursor/gpt-5-6",
                adapter="cursor",
                adapter_model_name="gpt-5.6",
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates, "_REVIEW_JUDGE", gates.default_judge_review
            ), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PUPPETMASTER_REVIEW_GATE", None)
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertFalse(payload["passed"])
            self.assertIn("PUPPETMASTER_REVIEW_GATE", payload["reason"])
            self.assertIn("disabled", payload["reason"].lower())

    def test_explicit_review_fails_when_no_adequate_judge_resolves(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(repo, review=True)

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=None), patch.object(
                gates, "_REVIEW_JUDGE"
            ) as review_call:
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            review_call.assert_not_called()
            payload = self._review_payload(evaluation)
            self.assertFalse(payload["passed"])
            self.assertIsNone(payload["judge"])
            self.assertIn("no adequate judge", payload["reason"].lower())

    def test_raw_review_gate_cannot_make_its_implicit_request_optional(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._raw_review_task(repo, {"required": False})

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=None):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertTrue(payload["review_requested"])
            self.assertTrue(payload["review_required"])
            self.assertIn("no adequate judge", payload["reason"].lower())

    def test_requested_raw_review_cannot_disable_required_when_judge_is_missing(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._raw_review_task(
                repo,
                {"requested": True, "required": False},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=None):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertTrue(payload["review_requested"])
            self.assertTrue(payload["review_required"])
            self.assertIn("no adequate judge", payload["reason"].lower())

    def test_requested_raw_review_cannot_disable_required_when_judge_is_unavailable(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._raw_review_task(
                repo,
                {"requested": True, "required": False},
            )
            judge = Mock(id="cursor/gpt-5-6", adapter="cursor")
            unavailable = ReviewVerdict(
                available=False,
                passed=True,
                reasons=["provider offline"],
                detail={"availability_reason": "provider offline"},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates,
                "_REVIEW_JUDGE",
                return_value=unavailable,
            ):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertTrue(payload["review_requested"])
            self.assertTrue(payload["review_required"])
            self.assertIn("provider offline", payload["reason"])

    def test_requested_raw_review_cannot_disable_required_when_live_review_is_disabled(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._raw_review_task(
                repo,
                {"requested": True, "required": False},
            )
            judge = Mock(
                id="cursor/gpt-5-6",
                adapter="cursor",
                adapter_model_name="gpt-5.6",
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates,
                "_REVIEW_JUDGE",
                gates.default_judge_review,
            ), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PUPPETMASTER_REVIEW_GATE", None)
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertTrue(payload["review_requested"])
            self.assertTrue(payload["review_required"])
            self.assertIn("PUPPETMASTER_REVIEW_GATE", payload["reason"])

    def test_raw_nonrequested_review_remains_explicitly_optional(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._raw_review_task(
                repo,
                {"requested": False, "required": False},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=None):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertTrue(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertFalse(payload["review_requested"])
            self.assertFalse(payload["review_required"])
            self.assertEqual(payload["review_status"], "skipped")
            self.assertIn("optional review skipped", payload["reason"].lower())

    def test_explicit_review_flag_cannot_be_weakened_by_optional_gate_entry(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(repo, review=True)
            task = Task(
                job_id=task.job_id,
                id=task.id,
                role=task.role,
                instruction=task.instruction,
                payload={
                    **task.payload,
                    "gates": [
                        {
                            "kind": "review",
                            "required": False,
                            "requested": False,
                        }
                    ],
                },
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=None):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertTrue(payload["review_required"])
            self.assertTrue(payload["review_requested"])
            self.assertIn("no adequate judge", payload["reason"].lower())

    def test_explicit_review_fails_with_precise_unavailable_judge_reason(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(repo, review=True)
            judge = Mock(id="cursor/gpt-5-6")
            unavailable = ReviewVerdict(
                available=False,
                passed=True,
                reasons=["provider offline"],
                detail={"availability_reason": "provider offline"},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates, "_REVIEW_JUDGE", return_value=unavailable
            ):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["judge"], "cursor/gpt-5-6")
            self.assertIn("provider offline", payload["reason"])
            self.assertEqual(payload["availability_reason"], "provider offline")

    def test_successful_review_is_bound_to_judge_evaluator_and_full_diff(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(
                repo,
                review={
                    "evaluator_revision": "review-gate-v1",
                    "independence": "different_worker",
                },
            )
            judge = Mock(id="cursor/gpt-5-6", adapter="cursor")
            collected_diff = gates._collect_diff([], repo)
            expected_fingerprint = "sha256:" + hashlib.sha256(
                collected_diff.encode("utf-8")
            ).hexdigest()
            approved = ReviewVerdict(
                available=True,
                passed=True,
                severity="none",
                detail={"judge_identity": "review-worker-42"},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates, "_REVIEW_JUDGE", return_value=approved
            ):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertTrue(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertEqual(payload["judge_identity"], "review-worker-42")
            self.assertNotEqual(payload["judge_identity"], "implementer-worker")
            self.assertEqual(payload["judge_model"], "cursor/gpt-5-6")
            self.assertEqual(payload["judge_adapter"], "cursor")
            self.assertEqual(payload["evaluator_revision"], "review-gate-v1")
            self.assertEqual(payload["reviewed_artifact_fingerprint"], expected_fingerprint)

    def test_different_worker_constraint_blocks_same_worker_identity(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = self._task(
                repo,
                review={"independence": "different_worker"},
            )
            judge = Mock(id="cursor/gpt-5-6", adapter="cursor")
            approved_by_implementer = ReviewVerdict(
                available=True,
                passed=True,
                detail={"judge_identity": "implementer-worker"},
            )

            # Act
            with patch.object(gates, "resolve_judge_model", return_value=judge), patch.object(
                gates, "_REVIEW_JUDGE", return_value=approved_by_implementer
            ):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertFalse(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertEqual(payload["review_status"], "independence_failed")
            self.assertIn("implementer worker", payload["reason"].lower())

    def test_global_policy_review_remains_explicitly_optional(self) -> None:
        # Arrange
        import puppetmaster.gates as gates

        with TemporaryDirectory() as root:
            repo = self._repo(root)
            task = Task(
                job_id="job-review",
                id="task-review",
                role="implement",
                instruction="add the feature",
                payload={"cwd": str(repo), "mode": "implement"},
            )

            # Act
            with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}), patch.object(
                gates, "resolve_judge_model", return_value=None
            ):
                evaluation = evaluate_task_gates(
                    task,
                    [],
                    self._store(root),
                    worker_id="implementer-worker",
                    cwd=repo,
                )

            # Assert
            self.assertTrue(evaluation.passed)
            payload = self._review_payload(evaluation)
            self.assertFalse(payload["review_required"])
            self.assertFalse(payload["review_requested"])
            self.assertEqual(payload["review_status"], "skipped")


class IndependentJudgeSelectionTests(TestCase):
    @staticmethod
    def _model(model_id: str, capability: int, family: str) -> ModelSpec:
        adapter, adapter_model_name = model_id.split("/", 1)
        return ModelSpec(
            id=model_id,
            adapter=adapter,
            adapter_model_name=adapter_model_name,
            capability_score=capability,
            tags=[f"family:{family}"],
            billing="plan",
        )

    def test_different_model_family_constraint_selects_independent_judge(self) -> None:
        # Arrange
        from puppetmaster.gates import resolve_judge_model

        with TemporaryDirectory() as root:
            registry = [
                self._model("cursor/gpt-5-5", 90, "openai"),
                self._model("cursor/gpt-5-6", 95, "openai"),
                self._model("claude-code/opus-4-1", 95, "anthropic"),
            ]
            registry_path = Path(root) / "models.json"
            save_registry(registry, registry_path)
            task = Task(
                job_id="job-review",
                role="implement",
                instruction="high-risk edit",
                payload={
                    "router_model_id": "cursor/gpt-5-5",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(registry),
                },
            )

            # Act
            with patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                judge = resolve_judge_model(
                    task, {"independence": "different_model_family"}
                )

            # Assert
            self.assertIsNotNone(judge)
            self.assertEqual(judge.id, "claude-code/opus-4-1")

    def test_different_model_family_constraint_never_falls_back_to_same_family(self) -> None:
        # Arrange
        from puppetmaster.gates import resolve_judge_model

        with TemporaryDirectory() as root:
            registry = [
                self._model("cursor/gpt-5-5", 90, "openai"),
                self._model("cursor/gpt-5-6", 95, "openai"),
            ]
            registry_path = Path(root) / "models.json"
            save_registry(registry, registry_path)
            task = Task(
                job_id="job-review",
                role="implement",
                instruction="high-risk edit",
                payload={
                    "router_model_id": "cursor/gpt-5-5",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(registry),
                },
            )

            # Act
            with patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                judge = resolve_judge_model(
                    task, {"independence": "different_model_family"}
                )

            # Assert
            self.assertIsNone(judge)


class DispatchableJudgeSelectionTests(TestCase):
    """A review judge must be a model that can actually run on this host."""

    @staticmethod
    def _model(model_id: str, capability: int) -> ModelSpec:
        adapter, adapter_model_name = model_id.split("/", 1)
        return ModelSpec(
            id=model_id,
            adapter=adapter,
            adapter_model_name=adapter_model_name,
            capability_score=capability,
            billing="plan",
        )

    def _judge(self, cli_present) -> object:
        from types import SimpleNamespace

        from puppetmaster.gates import resolve_judge_model

        with TemporaryDirectory() as root:
            registry = [
                self._model("claude-code/haiku-4-5", 55),
                self._model("antigravity/gemini-3-5-flash", 78),
                self._model("claude-code/claude-sonnet-4-5", 82),
            ]
            registry_path = Path(root) / "models.json"
            save_registry(registry, registry_path)
            task = Task(
                job_id="job-review",
                role="implement",
                instruction="edit",
                payload={
                    "router_model_id": "claude-code/haiku-4-5",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(registry),
                },
            )
            healthy = SimpleNamespace(healthy=True, billing="plan")
            with patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True), \
                    patch("puppetmaster.platform_billing.detect_adapter_billing", return_value=healthy), \
                    patch("puppetmaster.preflight.adapter_cli_present", side_effect=cli_present):
                return resolve_judge_model(task, {})

    def test_judge_skips_a_cheaper_model_whose_cli_is_missing(self) -> None:
        judge = self._judge(lambda adapter: adapter != "antigravity")
        self.assertEqual(judge.id, "claude-code/claude-sonnet-4-5")

    def test_nothing_dispatchable_keeps_the_previous_choice(self) -> None:
        judge = self._judge(lambda adapter: False)
        self.assertEqual(judge.id, "antigravity/gemini-3-5-flash")


class JudgeIsolationTests(TestCase):
    """The judge runs read-only and independent of the implementer's payload."""

    def _judge_spec(self) -> ModelSpec:
        return ModelSpec(
            id="claude-code/claude-sonnet-4-5",
            adapter="claude-code",
            adapter_model_name="claude-sonnet-4-5",
            capability_score=82,
            billing="plan",
            payload_defaults={"extra_args": ["--effort", "medium"]},
        )

    def test_judge_payload_drops_implementer_write_and_resume_state(self) -> None:
        from puppetmaster import gates

        captured = {}

        class FakeAdapter:
            def run(self, task, goal, worker_id):
                captured["payload"] = dict(task.payload)
                return []

        task = Task(
            job_id="j", role="implement", adapter="claude-code", instruction="edit",
            payload={
                "cwd": "/repo", "mode": "implement", "permission_mode": "acceptEdits",
                "allow_dirty": True, "review_loop": True,
                "resume": {"status": "resolved", "adapter": "claude-code", "session_id": "s"},
                "registry_path": "/reg/models.json", "registry_digest": "sha256:abc",
            },
        )
        with patch.dict(os.environ, {"PUPPETMASTER_REVIEW_GATE": "1"}), \
                patch("puppetmaster.adapters.get_adapter", return_value=FakeAdapter()):
            gates.default_judge_review(prompt="p", judge=self._judge_spec(), cwd=Path("/repo"), timeout=30, task=task)

        payload = captured["payload"]
        for leaked in ("resume", "allow_dirty", "permission_mode", "review_loop"):
            self.assertNotIn(leaked, payload)
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["sandbox"], "read-only")
        self.assertEqual(payload["model"], "claude-sonnet-4-5")
        self.assertEqual(payload["extra_args"], ["--effort", "medium"])
        self.assertEqual((payload["registry_path"], payload["registry_digest"]), ("/reg/models.json", "sha256:abc"))

    def test_verdict_is_read_from_json_wrapped_stdout(self) -> None:
        import json as _json

        from puppetmaster.gates import _verdict_from_artifacts
        from puppetmaster.models import Artifact, ArtifactType

        result = 'Looks fine.\nPUPPETMASTER_REVIEW_VERDICT {"pass": true, "severity": "none", "reasons": []}\nVERDICT: PASS - ok'
        artifact = Artifact(
            job_id="j", task_id="t", type=ArtifactType.VERIFICATION, created_by="judge",
            confidence=0.9, evidence=["adapter:claude-code"],
            payload={"check": "review", "result": "passed", "stdout": _json.dumps({"type": "result", "result": result})},
        )
        verdict = _verdict_from_artifacts([artifact])
        self.assertIsNotNone(verdict)
        self.assertIs(verdict["pass"], True)


class CumulativeReviewDiffTests(TestCase):
    """A repair is reviewed against the task's original base, not just its own attempt."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    def test_repair_after_committed_work_reviews_the_whole_task(self) -> None:
        import puppetmaster.gates as gates
        from puppetmaster.gates import ReviewVerdict, evaluate_task_gates
        from puppetmaster.models import Artifact, ArtifactType

        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            self._git(repo, "init", "-q")
            self._git(repo, "config", "user.email", "t@t")
            self._git(repo, "config", "user.name", "t")
            (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-qm", "base")
            base = self._git(repo, "rev-parse", "HEAD")

            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            job = store.create_job("cumulative review")
            task = Task(job_id=job.id, role="implement", instruction="edit", adapter="claude-code",
                        payload={"review": True, "cwd": str(repo), "mode": "implement", "allow_dirty": True})
            store.save_task(task)
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION, created_by="worker",
                confidence=0.9, evidence=["adapter:claude-code"],
                payload={"adapter": "claude-code", "check": "edit", "result": "passed", "base_sha": base},
            ))
            (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
            self._git(repo, "commit", "-qam", "attempt 1 committed its work")
            (repo / "new_test.py").write_text("assert True\n", encoding="utf-8")
            self._git(repo, "add", "new_test.py")
            self._git(repo, "commit", "-qm", "repair committed the rest")

            seen = {}

            def judge(**kwargs):
                seen["prompt"] = kwargs["prompt"]
                return ReviewVerdict(available=True, passed=True, severity="none", reasons=[], detail={})

            with patch.object(gates, "resolve_judge_model", return_value=Mock(id="claude-code/claude-sonnet-4-5", adapter="claude-code")), \
                    patch.object(gates, "_REVIEW_JUDGE", side_effect=judge):
                evaluation = evaluate_task_gates(task, [], store, worker_id="w1", cwd=repo)

            review = next(result for result in evaluation.results if result.kind == "review")
            self.assertTrue(review.passed, review.reason)
            self.assertIn("-x = 1", seen["prompt"])
            self.assertIn("+x = 2", seen["prompt"])
            self.assertIn("new_test.py", seen["prompt"])


class ReviewScopeAndJudgeDefaultsTests(TestCase):
    """Review only the task's own commits; a judge never inherits write permissions."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    def _repo(self, tmp: str) -> Path:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        self._git(repo, "init", "-q")
        self._git(repo, "config", "user.email", "t@t")
        self._git(repo, "config", "user.name", "t")
        (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "base")
        return repo

    def _verification(self, job_id: str, task_id: str, base: str, head: str, at: str):
        from puppetmaster.models import Artifact, ArtifactType

        return Artifact(
            job_id=job_id, task_id=task_id, type=ArtifactType.VERIFICATION, created_by="worker",
            confidence=0.9, evidence=["adapter:claude-code"],
            payload={"adapter": "claude-code", "check": "edit", "result": "passed", "base_sha": base, "head_sha": head},
            created_at=at,
        )

    def test_foreign_commit_between_attempts_disables_the_cumulative_range(self) -> None:
        from puppetmaster.gates import _task_base_sha

        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.ensure_schema()
            job = store.create_job("scope")
            task = Task(job_id=job.id, role="implement", instruction="edit", adapter="claude-code", payload={})
            store.save_task(task)
            contiguous = [
                self._verification(job.id, task.id, "aaa", "bbb", "2026-10-04T01:00:00+00:00"),
                self._verification(job.id, task.id, "bbb", "ccc", "2026-10-04T01:05:00+00:00"),
            ]
            self.assertEqual(_task_base_sha(store, task, contiguous), "aaa")
            gap = [
                self._verification(job.id, task.id, "aaa", "bbb", "2026-10-04T01:00:00+00:00"),
                self._verification(job.id, task.id, "fff", "ggg", "2026-10-04T01:05:00+00:00"),
            ]
            self.assertIsNone(_task_base_sha(store, task, gap))

    def test_base_that_is_not_an_ancestor_is_not_diffed(self) -> None:
        from puppetmaster.gates import _cumulative_diff

        with TemporaryDirectory() as tmp:
            repo = self._repo(tmp)
            (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
            self._git(repo, "commit", "-qam", "dropped later")
            dropped = self._git(repo, "rev-parse", "HEAD")
            self._git(repo, "reset", "-q", "--hard", "HEAD~1")
            (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-qm", "new line of history")
            self.assertEqual(_cumulative_diff(repo, dropped), "")

    def test_judge_keeps_same_adapter_executable_and_drops_write_defaults(self) -> None:
        from puppetmaster.gates import _judge_payload

        task = Task(job_id="j", role="implement", adapter="claude-code", instruction="edit",
                    payload={"executable": "/opt/claude-beta/claude", "permission_mode": "acceptEdits"})
        judge = ModelSpec(id="claude-code/claude-sonnet-4-5", adapter="claude-code",
                          adapter_model_name="claude-sonnet-4-5", capability_score=82, billing="plan",
                          payload_defaults={"permission_mode": "bypassPermissions", "allow_dirty": True,
                                            "extra_args": ["--effort", "medium"]})
        payload = _judge_payload(task, judge, prompt="p", cwd=Path("/repo"), timeout=30)
        self.assertEqual(payload["executable"], "/opt/claude-beta/claude")
        self.assertNotIn("permission_mode", payload)
        self.assertNotIn("allow_dirty", payload)
        self.assertEqual(payload["extra_args"], ["--effort", "medium"])
        other = ModelSpec(id="codex/gpt-5-6-luna", adapter="codex", adapter_model_name="gpt-5.6-luna",
                          capability_score=91, billing="plan")
        self.assertNotIn("executable", _judge_payload(task, other, prompt="p", cwd=Path("/repo"), timeout=30))
