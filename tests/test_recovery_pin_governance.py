"""Focused governance tests for routing recovery, pins, and escalation.

The suite is intentionally black-box at the orchestration seams: it proves the
authority and provenance a persisted task carries without depending on a live
model, provider credential, or CLI installation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest import TestCase
from unittest.mock import patch

from puppetmaster.model_registry import ModelSpec, registry_digest, save_registry
from puppetmaster.models import AgentRun, Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.orchestrator import Orchestrator, _payload_has_explicit_model_pin
from puppetmaster.platform_billing import BillingStatus
from puppetmaster.store import SwarmStore
from puppetmaster.workers import WorkerSpec


class RecoveryPinGovernanceTests(TestCase):
    @staticmethod
    def _store_job(root: str, goal: str = "govern routed work"):
        store = SwarmStore(Path(root) / ".puppetmaster")
        job = store.create_job(goal)
        return store, job

    @staticmethod
    def _model(
        model_id: str,
        capability: int,
        *,
        adapter: Optional[str] = None,
        enabled: bool = True,
        retired: bool = False,
        role_scores: Optional[dict[str, int]] = None,
    ) -> ModelSpec:
        resolved_adapter, wire_name = model_id.split("/", 1)
        resolved_adapter = adapter or resolved_adapter
        cards = {}
        for role, score in (role_scores or {}).items():
            cards[role] = {
                "capability": score,
                "sample_count": 8,
                "scale": "puppetmaster-capability-0-100",
                "scale_version": "v1",
                "last_calibrated": date.today().isoformat(),
                "provenance": {"source": "focused-test", "version": "v1"},
            }
        return ModelSpec(
            id=model_id,
            adapter=resolved_adapter,
            adapter_model_name=wire_name,
            capability_score=capability,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=1.0,
            billing="plan",
            enabled=enabled,
            retired=retired,
            retirement_reason="superseded" if retired else "",
            retirement_authority="user:test" if retired else "",
            role_scorecards=cards,
        )

    @staticmethod
    def _healthy(adapter: str, **_kwargs) -> BillingStatus:
        return BillingStatus(
            adapter=adapter,
            billing="plan",
            healthy=True,
            detail="focused test",
            evidence=[],
        )

    @staticmethod
    def _failure(job_id: str, task_id: str, failure: str, created_at: str) -> Artifact:
        return Artifact(
            job_id=job_id,
            task_id=task_id,
            type=ArtifactType.VERIFICATION,
            created_by="worker",
            payload={"check": "worker", "result": "blocked", "failure": failure},
            confidence=0.2,
            evidence=[failure],
            created_at=created_at,
        )

    def test_newest_failure_controls_recoverability_even_when_older_failure_was_soft(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            store, job = self._store_job(root)
            old_soft_new_hard = Task(
                job_id=job.id,
                role="implement",
                instruction="first task",
                status=TaskStatus.FAILED,
            )
            old_hard_new_soft = Task(
                job_id=job.id,
                role="implement",
                instruction="second task",
                status=TaskStatus.FAILED,
            )
            artifacts = [
                self._failure(
                    job.id,
                    old_soft_new_hard.id,
                    "billing_or_quota",
                    "2026-08-22T00:00:00Z",
                ),
                self._failure(
                    job.id,
                    old_soft_new_hard.id,
                    "adapter_exception",
                    "2026-08-22T00:00:01Z",
                ),
                self._failure(
                    job.id,
                    old_hard_new_soft.id,
                    "adapter_exception",
                    "2026-08-22T00:00:00Z",
                ),
                self._failure(
                    job.id,
                    old_hard_new_soft.id,
                    "model_unavailable",
                    "2026-08-22T00:00:01Z",
                ),
            ]

            # Act
            recoverable = Orchestrator(store)._recoverable_failure_by_task(
                job, artifacts=artifacts
            )

            # Assert
            self.assertNotIn(old_soft_new_hard.id, recoverable)
            self.assertEqual(recoverable[old_hard_new_soft.id], "model_unavailable")

    def test_auto_route_wins_and_clears_stale_explicit_pin_provenance(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            models = [
                self._model("cursor/low", 40),
                self._model("cursor/selected", 80),
            ]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            spec = WorkerSpec(
                role="implement",
                instruction="implement a cross-module migration",
                adapter="cursor",
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                    "model": "stale-wire-name",
                    "router_model_id": "cursor/stale",
                    "pinned_model": "cursor/stale",
                    "pinned_adapter_model_name": "stale-wire-name",
                    "min_capability": 70,
                },
            )

            # Act
            with patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                task = Orchestrator(store)._create_tasks(job, [spec])[0]

            # Assert
            self.assertTrue(task.payload["auto_route"])
            self.assertEqual(task.payload["router_model_id"], "cursor/selected")
            self.assertEqual(task.payload["model"], "selected")
            self.assertNotIn("pinned_model", task.payload)
            self.assertNotIn("pinned_adapter_model_name", task.payload)
            self.assertFalse(_payload_has_explicit_model_pin(task.payload))
            self.assertEqual(Path(task.payload["registry_path"]), registry_path)
            self.assertEqual(task.payload["registry_digest"], registry_digest(models))

    def test_initial_route_and_bound_digest_describe_the_same_registry_snapshot(self) -> None:
        # Arrange
        from puppetmaster.routing_authority import load_bound_registry

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            routed_snapshot = [self._model("cursor/selected", 90)]
            replacement_snapshot = [self._model("cursor/replacement", 30)]
            save_registry(routed_snapshot, registry_path)
            store, job = self._store_job(root)
            spec = WorkerSpec(
                role="implement",
                instruction="implement a cross-module migration",
                adapter="cursor",
                payload={
                    "auto_route": True,
                    "registry_path": str(registry_path),
                    "min_capability": 80,
                },
            )
            loads = 0

            def changing_registry(_path):
                nonlocal loads
                loads += 1
                if loads == 1:
                    return routed_snapshot
                save_registry(replacement_snapshot, registry_path)
                return replacement_snapshot

            # Act
            with patch(
                "puppetmaster.model_registry.load_registry",
                side_effect=changing_registry,
            ), patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                task = Orchestrator(store)._create_tasks(job, [spec])[0]

            # Assert
            _path, bound_registry, bound_digest = load_bound_registry(task.payload)
            self.assertEqual(task.payload["registry_digest"], bound_digest)
            self.assertIn(
                task.payload["router_model_id"],
                {model.id for model in bound_registry},
                "selected model must exist in the exact registry epoch bound to the task",
            )

    def test_explicit_pin_is_resolved_and_bound_to_live_registry_at_creation(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            models = [self._model("cursor/current", 80)]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            spec = WorkerSpec(
                role="implement",
                instruction="use the requested model",
                adapter="cursor",
                payload={
                    "model": "current",
                    "registry_path": str(registry_path),
                },
            )

            # Act
            with patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                task = Orchestrator(store)._create_tasks(job, [spec])[0]

            # Assert
            self.assertEqual(task.payload["pinned_model"], "cursor/current")
            self.assertEqual(task.payload["router_model_id"], "cursor/current")
            self.assertEqual(task.payload["model"], "current")
            self.assertEqual(Path(task.payload["registry_path"]), registry_path)
            self.assertEqual(task.payload["registry_digest"], registry_digest(models))

    def test_retired_explicit_pin_is_rejected_before_task_is_persisted(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            retired = self._model(
                "cursor/retired", 90, enabled=False, retired=True
            )
            save_registry([retired], registry_path)
            store, job = self._store_job(root)
            spec = WorkerSpec(
                role="implement",
                instruction="must not invoke retired authority",
                adapter="cursor",
                payload={
                    "model": "retired",
                    "registry_path": str(registry_path),
                },
            )

            # Act / Assert
            with self.assertRaisesRegex(RuntimeError, "(?i)model|pin|retir|disabled"):
                Orchestrator(store)._create_tasks(job, [spec])
            self.assertEqual(store.list_tasks(job.id), [])

    def test_pin_retired_after_creation_is_blocked_before_adapter_invocation(self) -> None:
        # Arrange
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            enabled = self._model("cursor/current", 80)
            save_registry([enabled], registry_path)
            store, job = self._store_job(root)
            spec = WorkerSpec(
                role="implement",
                instruction="run only while authority remains live",
                adapter="cursor",
                payload={"model": "current", "registry_path": str(registry_path)},
            )
            with patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                task = Orchestrator(store)._create_tasks(job, [spec])[0]
            save_registry(
                [replace(
                    enabled,
                    enabled=False,
                    retired=True,
                    retirement_reason="superseded",
                    retirement_authority="user:test",
                )],
                registry_path,
            )
            runtime = WorkerRuntime(
                store=store,
                job_id=job.id,
                role=task.role,
                worker_id="dispatch-guard",
            )
            completed_run = AgentRun(
                job_id=job.id,
                task_id=task.id,
                role=task.role,
                worker_id="dispatch-guard",
                status=TaskStatus.COMPLETE,
            )

            # Act
            with patch("puppetmaster.worker_runtime.LocalWorker") as worker:
                worker.return_value.run.return_value = (completed_run, [])
                self.assertTrue(runtime.run_once())

            # Assert
            worker.assert_not_called()
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.FAILED)
            failures = [
                event
                for event in store.read_events(job.id)
                if event["event"] == "worker.failed_task"
            ]
            self.assertTrue(failures)
            detail = str(failures[-1]["payload"]).lower()
            self.assertTrue("retir" in detail or "registry" in detail)

    def test_enabled_legacy_model_only_pin_is_canonicalized_before_dispatch(self) -> None:
        # Arrange
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            models = [self._model("cursor/current", 80)]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="dispatch a legacy persisted pin",
                adapter="cursor",
                status=TaskStatus.QUEUED,
                payload={"model": "current", "auto_route": False},
            )
            store.save_task(task)
            runtime = WorkerRuntime(
                store=store,
                job_id=job.id,
                role=task.role,
                worker_id="legacy-dispatch-guard",
            )
            completed_run = AgentRun(
                job_id=job.id,
                task_id=task.id,
                role=task.role,
                worker_id="legacy-dispatch-guard",
                status=TaskStatus.COMPLETE,
            )

            # Act
            with patch.dict(
                "os.environ", {"PUPPETMASTER_MODELS_PATH": str(registry_path)}
            ), patch("puppetmaster.worker_runtime.LocalWorker") as worker:
                worker.return_value.run.return_value = (completed_run, [])
                self.assertTrue(runtime.run_once())

            # Assert
            worker.assert_called_once()
            dispatched = worker.return_value.run.call_args.args[0]
            self.assertEqual(dispatched.payload["pinned_model"], "cursor/current")
            self.assertEqual(dispatched.payload["router_model_id"], "cursor/current")
            self.assertEqual(Path(dispatched.payload["registry_path"]), registry_path)
            self.assertEqual(
                dispatched.payload["registry_digest"], registry_digest(models)
            )
            persisted = store.get_task_by_id(task.id)
            self.assertEqual(persisted.status, TaskStatus.COMPLETE)
            self.assertEqual(persisted.payload["pinned_model"], "cursor/current")
            self.assertEqual(
                persisted.payload["registry_digest"], registry_digest(models)
            )

    def test_retired_legacy_model_only_pin_is_blocked_before_dispatch(self) -> None:
        # Arrange
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "models.json"
            retired = self._model(
                "cursor/retired", 90, enabled=False, retired=True
            )
            save_registry([retired], registry_path)
            store, job = self._store_job(root)
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="never dispatch retired legacy authority",
                adapter="cursor",
                status=TaskStatus.QUEUED,
                payload={"model": "retired", "auto_route": False},
            )
            store.save_task(task)
            runtime = WorkerRuntime(
                store=store,
                job_id=job.id,
                role=task.role,
                worker_id="legacy-dispatch-guard",
            )
            completed_run = AgentRun(
                job_id=job.id,
                task_id=task.id,
                role=task.role,
                worker_id="legacy-dispatch-guard",
                status=TaskStatus.COMPLETE,
            )

            # Act
            with patch.dict(
                "os.environ", {"PUPPETMASTER_MODELS_PATH": str(registry_path)}
            ), patch("puppetmaster.worker_runtime.LocalWorker") as worker:
                worker.return_value.run.return_value = (completed_run, [])
                self.assertTrue(runtime.run_once())

            # Assert
            worker.assert_not_called()
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.FAILED)
            failures = [
                event
                for event in store.read_events(job.id)
                if event["event"] == "worker.failed_task"
            ]
            self.assertTrue(failures)
            detail = str(failures[-1]["payload"]).lower()
            self.assertTrue("retir" in detail or "registry" in detail)

    def test_divergent_stamped_and_executable_pin_identities_block_dispatch(self) -> None:
        # Arrange
        from puppetmaster.worker_runtime import WorkerRuntime

        mismatches = (
            {
                "model": "different-wire",
                "router_model_id": "cursor/current",
                "pinned_model": "cursor/current",
                "pinned_adapter_model_name": "current",
            },
            {
                "model": "current",
                "router_model_id": "cursor/different-id",
                "pinned_model": "cursor/current",
                "pinned_adapter_model_name": "current",
            },
        )
        for mismatch in mismatches:
            with self.subTest(mismatch=mismatch), TemporaryDirectory() as root:
                registry_path = Path(root) / "models.json"
                models = [self._model("cursor/current", 80)]
                save_registry(models, registry_path)
                store, job = self._store_job(root)
                task = Task(
                    job_id=job.id,
                    role="implement",
                    instruction="reject contradictory persisted pin identity",
                    adapter="cursor",
                    status=TaskStatus.QUEUED,
                    payload={
                        **mismatch,
                        "auto_route": False,
                        "registry_path": str(registry_path),
                        "registry_digest": registry_digest(models),
                    },
                )
                store.save_task(task)
                runtime = WorkerRuntime(
                    store=store,
                    job_id=job.id,
                    role=task.role,
                    worker_id="identity-dispatch-guard",
                )
                completed_run = AgentRun(
                    job_id=job.id,
                    task_id=task.id,
                    role=task.role,
                    worker_id="identity-dispatch-guard",
                    status=TaskStatus.COMPLETE,
                )

                # Act
                with patch("puppetmaster.worker_runtime.LocalWorker") as worker:
                    worker.return_value.run.return_value = (completed_run, [])
                    self.assertTrue(runtime.run_once())

                # Assert
                worker.assert_not_called()
                self.assertEqual(
                    store.get_task_by_id(task.id).status, TaskStatus.FAILED
                )
                failures = [
                    event
                    for event in store.read_events(job.id)
                    if event["event"] == "worker.failed_task"
                ]
                self.assertTrue(failures)
                self.assertIn(
                    "registry_authority_invalid", str(failures[-1]["payload"])
                )

    def test_fallback_uses_task_registry_authority_and_records_source_model(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "override-models.json"
            models = [
                self._model("cursor/failed", 70),
                self._model("claude-code/recovery", 90),
            ]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="recover the task",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={
                    "auto_route": True,
                    "model": "failed",
                    "router_model_id": "cursor/failed",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(models),
                },
            )
            store.save_task(task)
            store.save_artifact(
                self._failure(
                    job.id,
                    task.id,
                    "billing_or_quota",
                    "2026-08-22T00:00:00Z",
                )
            )

            # Act
            with patch(
                "puppetmaster.model_registry.default_registry_path",
                side_effect=AssertionError("fallback consulted ambient registry"),
            ), patch(
                "puppetmaster.platform_billing.detect_adapter_billing_cached",
                side_effect=self._healthy,
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ), patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = Orchestrator(store)._reroute_recoverable_failures(job)

            # Assert
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.payload["router_model_id"], "claude-code/recovery")
            self.assertEqual(updated.payload["fallback_from_model"], "cursor/failed")
            self.assertEqual(Path(updated.payload["registry_path"]), registry_path)
            self.assertEqual(updated.payload["registry_digest"], registry_digest(models))
            artifacts = [
                artifact
                for artifact in store.list_artifacts(job.id)
                if artifact.created_by == "router-fallback"
            ]
            self.assertEqual(artifacts[-1].payload["fallback_from_model"], "cursor/failed")

    def test_confidence_escalation_uses_role_effective_scores_and_task_registry(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "override-models.json"
            models = [
                self._model("cursor/current", 90, role_scores={"implement": 50}),
                self._model("cursor/role-stronger", 80, role_scores={"implement": 70}),
            ]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the change",
                adapter="cursor",
                status=TaskStatus.COMPLETE,
                payload={
                    "auto_route": True,
                    "model": "current",
                    "router_model_id": "cursor/current",
                    "min_confidence": 0.8,
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(models),
                },
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker",
                    payload={"check": "worker", "result": "done"},
                    confidence=0.4,
                    evidence=["low-confidence"],
                )
            )

            # Act
            with patch(
                "puppetmaster.model_registry.default_registry_path",
                side_effect=AssertionError("escalation consulted ambient registry"),
            ), patch(
                "puppetmaster.platform_billing.detect_adapter_billing",
                side_effect=self._healthy,
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ), patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = Orchestrator(store)._reroute_low_confidence(job)

            # Assert
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.payload["router_model_id"], "cursor/role-stronger")
            self.assertEqual(updated.payload["escalated_from_model"], "cursor/current")
            self.assertEqual(Path(updated.payload["registry_path"]), registry_path)
            self.assertEqual(updated.payload["registry_digest"], registry_digest(models))
            artifact = next(
                item
                for item in store.list_artifacts(job.id)
                if item.created_by == "router-escalation"
            )
            self.assertEqual(artifact.payload["escalated_from_model"], "cursor/current")
            self.assertEqual(artifact.payload["effective_capability_score"], 70)

    def test_review_escalation_uses_role_effective_scores_and_task_registry(self) -> None:
        # Arrange
        with TemporaryDirectory() as root:
            registry_path = Path(root) / "override-models.json"
            models = [
                self._model("cursor/current", 90, role_scores={"implement": 50}),
                self._model("cursor/role-stronger", 80, role_scores={"implement": 70}),
            ]
            save_registry(models, registry_path)
            store, job = self._store_job(root)
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the change",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={
                    "auto_route": True,
                    "model": "current",
                    "router_model_id": "cursor/current",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(models),
                },
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GATE,
                    created_by="review-worker",
                    payload={"gate": "review", "kind": "review", "passed": False},
                    confidence=1.0,
                    evidence=["review:rejected"],
                )
            )

            # Act
            with patch(
                "puppetmaster.model_registry.default_registry_path",
                side_effect=AssertionError("review escalation consulted ambient registry"),
            ), patch(
                "puppetmaster.platform_billing.detect_adapter_billing",
                side_effect=self._healthy,
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ), patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = Orchestrator(store)._reroute_failed_review(job)

            # Assert
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.payload["router_model_id"], "cursor/role-stronger")
            self.assertEqual(
                updated.payload["review_escalated_from_model"], "cursor/current"
            )
            self.assertEqual(Path(updated.payload["registry_path"]), registry_path)
            self.assertEqual(updated.payload["registry_digest"], registry_digest(models))
            artifact = next(
                item
                for item in store.list_artifacts(job.id)
                if item.created_by == "router-review-escalation"
            )
            self.assertEqual(
                artifact.payload["review_escalated_from_model"], "cursor/current"
            )
            self.assertEqual(artifact.payload["effective_capability_score"], 70)

    def test_judge_selection_uses_task_registry_and_role_effective_capability(self) -> None:
        # Arrange
        from puppetmaster.gates import resolve_judge_model

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "override-models.json"
            models = [
                self._model("cursor/current", 90, role_scores={"implement": 50}),
                self._model("claude-code/reviewer", 80, role_scores={"review": 70}),
            ]
            save_registry(models, registry_path)
            task = Task(
                job_id="job-review",
                role="implement",
                instruction="implement the change",
                adapter="cursor",
                payload={
                    "router_model_id": "cursor/current",
                    "registry_path": str(registry_path),
                    "registry_digest": registry_digest(models),
                },
            )

            # Act
            with patch(
                "puppetmaster.model_registry.default_registry_path",
                side_effect=AssertionError("judge consulted ambient registry"),
            ), patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ):
                judge = resolve_judge_model(task, {})

            # Assert
            self.assertIsNotNone(judge)
            self.assertEqual(judge.id, "claude-code/reviewer")

    def test_judge_selection_rejects_registry_authority_drift(self) -> None:
        # Arrange
        from puppetmaster.gates import resolve_judge_model

        with TemporaryDirectory() as root:
            registry_path = Path(root) / "override-models.json"
            models = [
                self._model("cursor/current", 70),
                self._model("claude-code/reviewer", 90),
            ]
            save_registry(models, registry_path)
            task = Task(
                job_id="job-review",
                role="implement",
                instruction="implement the change",
                adapter="cursor",
                payload={
                    "router_model_id": "cursor/current",
                    "registry_path": str(registry_path),
                    "registry_digest": "0" * 64,
                },
            )

            # Act / Assert
            with patch(
                "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "(?i)registry.*drift|digest"):
                resolve_judge_model(task, {})

    def test_all_secondary_routes_fail_closed_on_registry_authority_drift(self) -> None:
        # Arrange
        from puppetmaster.routing_authority import RegistryAuthorityError

        cases = (
            ("fallback", TaskStatus.FAILED, "_reroute_recoverable_failures"),
            ("confidence", TaskStatus.COMPLETE, "_reroute_low_confidence"),
            ("review", TaskStatus.FAILED, "_reroute_failed_review"),
        )
        for kind, status, method_name in cases:
            with self.subTest(kind=kind), TemporaryDirectory() as root:
                registry_path = Path(root) / "override-models.json"
                models = [
                    self._model("cursor/current", 70),
                    self._model("claude-code/recovery", 90),
                ]
                save_registry(models, registry_path)
                store, job = self._store_job(root)
                payload = {
                    "auto_route": True,
                    "model": "current",
                    "router_model_id": "cursor/current",
                    "registry_path": str(registry_path),
                    "registry_digest": "0" * 64,
                }
                if kind == "confidence":
                    payload["min_confidence"] = 0.8
                task = Task(
                    job_id=job.id,
                    role="implement",
                    instruction="continue only under bound authority",
                    adapter="cursor",
                    status=status,
                    payload=payload,
                )
                store.save_task(task)
                if kind == "fallback":
                    store.save_artifact(
                        self._failure(
                            job.id,
                            task.id,
                            "billing_or_quota",
                            "2026-08-22T00:00:00Z",
                        )
                    )
                elif kind == "confidence":
                    store.save_artifact(
                        Artifact(
                            job_id=job.id,
                            task_id=task.id,
                            type=ArtifactType.VERIFICATION,
                            created_by="worker",
                            payload={"check": "worker", "result": "done"},
                            confidence=0.4,
                            evidence=["low-confidence"],
                        )
                    )
                else:
                    store.save_artifact(
                        Artifact(
                            job_id=job.id,
                            task_id=task.id,
                            type=ArtifactType.GATE,
                            created_by="review-worker",
                            payload={
                                "gate": "review",
                                "kind": "review",
                                "passed": False,
                            },
                            confidence=1.0,
                            evidence=["review:rejected"],
                        )
                    )

                # Act / Assert
                with patch(
                    "puppetmaster.platform_billing.detect_adapter_billing_cached",
                    side_effect=self._healthy,
                ), patch(
                    "puppetmaster.platform_billing.detect_adapter_billing",
                    side_effect=self._healthy,
                ), patch(
                    "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
                ), patch(
                    "puppetmaster.preflight.adapter_cli_present", return_value=True
                ), self.assertRaisesRegex(
                    RegistryAuthorityError, "(?i)registry.*drift|digest"
                ):
                    getattr(Orchestrator(store), method_name)(job)


if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()
