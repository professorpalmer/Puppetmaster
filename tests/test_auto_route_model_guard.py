from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from puppetmaster.models import Job
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.router import NoEligibleModelError
from puppetmaster.store import SwarmStore
from puppetmaster.workers import WorkerSpec


class AutoRouteModelGuardTests(unittest.TestCase):
    def _orchestrator(self, root: str) -> tuple[Orchestrator, Job]:
        store = SwarmStore(Path(root) / ".puppetmaster")
        job = store.create_job("auto-route guard")
        return Orchestrator(store), job

    def test_empty_registry_rejects_model_backed_auto_route_before_task_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator, job = self._orchestrator(tmp)
            spec = WorkerSpec(
                role="explore",
                instruction="audit the routing path",
                adapter="agentic",
                payload={"auto_route": True, "model": ""},
            )
            registry_path = Path(tmp) / "models.json"
            with mock.patch(
                "puppetmaster.model_registry.default_registry_path",
                return_value=registry_path,
            ), mock.patch(
                "puppetmaster.model_registry.load_registry",
                return_value=[],
            ):
                with self.assertRaises(NoEligibleModelError) as ctx:
                    orchestrator._apply_auto_routing(job, [spec])

            message = str(ctx.exception)
            self.assertIn("role='explore'", message)
            self.assertIn("adapter='agentic'", message)
            self.assertIn("model registry is empty", message)
            self.assertEqual(orchestrator.store.list_tasks(job.id), [])
            self.assertEqual(
                orchestrator.store.read_events(job.id)[-1]["event"],
                "router.registry_empty",
            )

    def test_no_eligible_model_rejects_model_backed_auto_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator, job = self._orchestrator(tmp)
            spec = WorkerSpec(
                role="audit",
                instruction="audit the routing path",
                adapter="agentic",
                payload={"auto_route": True},
            )
            registry_path = Path(tmp) / "models.json"
            registry_entry = SimpleNamespace(adapter="agentic")
            reconciliation = SimpleNamespace(
                specs=[registry_entry],
                upgraded=[],
                dropped=[],
            )
            with mock.patch(
                "puppetmaster.model_registry.default_registry_path",
                return_value=registry_path,
            ), mock.patch(
                "puppetmaster.model_registry.load_registry",
                return_value=[registry_entry],
            ), mock.patch(
                "puppetmaster.platform_billing.reconcile_registry",
                return_value=reconciliation,
            ), mock.patch(
                "puppetmaster.preflight.adapter_cli_present",
                return_value=True,
            ), mock.patch(
                "puppetmaster.router.route_task",
                side_effect=NoEligibleModelError("allowed_model_ids is empty"),
            ):
                with self.assertRaises(NoEligibleModelError) as ctx:
                    orchestrator._apply_auto_routing(job, [spec])

            message = str(ctx.exception)
            self.assertIn("role='audit'", message)
            self.assertIn("adapter='agentic'", message)
            self.assertIn("no eligible model in registry", message)
            self.assertIn("Review the routing constraints", message)
            self.assertEqual(orchestrator.store.list_tasks(job.id), [])
            self.assertEqual(
                orchestrator.store.read_events(job.id)[-1]["event"],
                "router.no_eligible_model",
            )

    def test_local_auto_route_can_still_pass_through_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator, job = self._orchestrator(tmp)
            spec = WorkerSpec(
                role="explore",
                instruction="run the local fixture",
                adapter="local",
                payload={"auto_route": True},
            )
            with mock.patch(
                "puppetmaster.model_registry.load_registry",
                return_value=[],
            ):
                routed, decisions = orchestrator._apply_auto_routing(job, [spec])

            self.assertEqual(routed, [spec])
            self.assertEqual(decisions, [])
            self.assertEqual(orchestrator.store.list_tasks(job.id), [])

    def test_explicit_model_pin_can_still_pass_through_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator, job = self._orchestrator(tmp)
            spec = WorkerSpec(
                role="explore",
                instruction="run the pinned worker",
                adapter="agentic",
                payload={"auto_route": True, "model": "grok-4.5"},
            )
            with mock.patch(
                "puppetmaster.model_registry.load_registry",
                return_value=[],
            ):
                routed, decisions = orchestrator._apply_auto_routing(job, [spec])

            self.assertEqual(routed, [spec])
            self.assertEqual(decisions, [])
            self.assertEqual(routed[0].payload["model"], "grok-4.5")


if __name__ == "__main__":
    unittest.main()
