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
