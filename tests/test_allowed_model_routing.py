"""Focused coverage for allowed-model routing + run_status_error reroute."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def _grok_composer_registry(*, include_fable: bool = False):
    from puppetmaster.model_registry import ModelSpec

    specs = [
        ModelSpec(
            id="cursor/composer-2-5",
            adapter="cursor",
            adapter_model_name="composer-2.5",
            capability_score=55,
            input_per_mtok_usd=0.5,
            output_per_mtok_usd=2.5,
            billing="plan",
            tags=["tools", "cursor", "cheap", "fast", "code"],
            enabled=True,
        ),
        ModelSpec(
            id="cursor/grok-4-5",
            adapter="cursor",
            adapter_model_name="grok-4.5",
            capability_score=97,
            input_per_mtok_usd=2.0,
            output_per_mtok_usd=10.0,
            billing="plan",
            tags=["tools", "cursor", "workhorse", "code", "reasoning"],
            enabled=True,
        ),
    ]
    if include_fable:
        specs.append(
            ModelSpec(
                id="cursor/claude-fable-5",
                adapter="cursor",
                adapter_model_name="claude-fable-5",
                capability_score=100,
                input_per_mtok_usd=10.0,
                output_per_mtok_usd=50.0,
                billing="plan",
                tags=["tools", "cursor", "frontier", "code"],
                enabled=True,
            )
        )
    return specs


class GrokComposerEnabledRegistryTests(unittest.TestCase):
    def test_quality_picks_grok_cheap_picks_composer(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        registry = _grok_composer_registry()
        instruction = "Implement a durable multi-file routing fix with tests"
        quality = route_task(
            TaskSignals(instruction=instruction, role="implement"),
            registry,
            policy="quality",
        )
        cheap = route_task(
            TaskSignals(instruction=instruction, role="implement"),
            registry,
            policy="cheap",
        )
        self.assertEqual(quality.model.id, "cursor/grok-4-5")
        self.assertEqual(quality.model.adapter_model_name, "grok-4.5")
        self.assertEqual(cheap.model.id, "cursor/composer-2-5")
        self.assertEqual(cheap.model.adapter_model_name, "composer-2.5")


class AllowedModelFilterTests(unittest.TestCase):
    def test_allowed_model_ids_accept_adapter_names_and_registry_ids(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        registry = _grok_composer_registry(include_fable=True)
        decision = route_task(
            TaskSignals(
                instruction="hard audit across every module",
                role="audit",
                allowed_model_ids=frozenset({"grok-4.5", "composer-2.5"}),
            ),
            registry,
            policy="quality",
        )
        self.assertEqual(decision.model.id, "cursor/grok-4-5")
        self.assertEqual(decision.allowed_model_ids, ["composer-2.5", "grok-4.5"])
        rejected_ids = {spec.id for spec, _ in decision.rejected}
        self.assertIn("cursor/claude-fable-5", rejected_ids)

    def test_disallowed_candidates_never_selected(self) -> None:
        from dataclasses import replace

        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        registry = _grok_composer_registry(include_fable=True)
        disabled_fable_registry = [
            replace(spec, enabled=False)
            if spec.id == "cursor/claude-fable-5"
            else spec
            for spec in registry
        ]
        with self.assertRaises(NoEligibleModelError) as ctx:
            route_task(
                TaskSignals(
                    instruction="fix a typo",
                    role="explore",
                    allowed_model_ids=frozenset({"claude-fable-5"}),
                ),
                disabled_fable_registry,
                policy="quality",
            )
        self.assertIn("allowed_model_ids", str(ctx.exception))

        # With Fable enabled elsewhere but not allowlisted, quality must still
        # stay inside Grok/Composer.
        decision = route_task(
            TaskSignals(
                instruction="security audit across every endpoint",
                role="audit",
                allowed_model_ids=frozenset({"cursor/grok-4-5", "cursor/composer-2-5"}),
            ),
            registry,
            policy="quality",
        )
        self.assertEqual(decision.model.id, "cursor/grok-4-5")
        self.assertNotEqual(decision.model.id, "cursor/claude-fable-5")

    def test_signals_from_worker_spec_reads_allowed_model_ids(self) -> None:
        from puppetmaster.router import signals_from_worker_spec
        from puppetmaster.workers import WorkerSpec

        spec = WorkerSpec(
            role="review",
            instruction="review the change",
            adapter="cursor",
            payload={
                "auto_route": True,
                "allowed_models": ["grok-4.5", "composer-2.5"],
            },
        )
        signals = signals_from_worker_spec(spec)
        self.assertEqual(signals.allowed_model_ids, frozenset({"grok-4.5", "composer-2.5"}))

    def test_explicit_empty_allowlist_fails_closed(self) -> None:
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        registry = _grok_composer_registry(include_fable=True)
        with self.assertRaises(NoEligibleModelError) as ctx:
            route_task(
                TaskSignals(
                    instruction="implement anything",
                    role="implement",
                    allowed_model_ids=frozenset(),
                ),
                registry,
                policy="quality",
            )
        self.assertIn("explicitly empty", str(ctx.exception))

    def test_unset_allowlist_remains_unrestricted(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        registry = _grok_composer_registry(include_fable=True)
        decision = route_task(
            TaskSignals(instruction="implement a fix", role="implement"),
            registry,
            policy="quality",
        )
        self.assertEqual(decision.model.id, "cursor/claude-fable-5")
        self.assertIsNone(decision.allowed_model_ids)


class GlobalAllowlistSnapshotTests(unittest.TestCase):
    def test_global_allowlist_snapshotted_and_task_override_wins(self) -> None:
        from puppetmaster.orchestrator import merge_routing_payload
        from puppetmaster.router import TaskSignals, route_task

        registry = _grok_composer_registry(include_fable=True)
        global_saved = {"allowed_model_ids": ["claude-fable-5"]}
        with mock.patch("puppetmaster.router._load_routing_overrides", return_value=global_saved):
            global_decision = route_task(
                TaskSignals(instruction="audit everything", role="audit"),
                registry,
                policy="quality",
            )
        self.assertEqual(global_decision.model.id, "cursor/claude-fable-5")
        self.assertEqual(global_decision.allowed_model_ids, ["claude-fable-5"])
        merged = merge_routing_payload({"auto_route": True}, global_decision)
        self.assertEqual(merged["allowed_model_ids"], ["claude-fable-5"])

        with mock.patch("puppetmaster.router._load_routing_overrides", return_value=global_saved):
            override_decision = route_task(
                TaskSignals(
                    instruction="cheap fix",
                    role="implement",
                    allowed_model_ids=frozenset({"composer-2.5"}),
                ),
                registry,
                policy="cheap",
            )
        self.assertEqual(override_decision.model.id, "cursor/composer-2-5")
        self.assertEqual(override_decision.allowed_model_ids, ["composer-2.5"])


class McpAllowedModelPropagationTests(unittest.TestCase):
    def test_implement_edit_agentic_commands_emit_allowed_models_flags(self) -> None:
        from puppetmaster.mcp_server import (
            agentic_command,
            claude_command,
            codex_command,
            cursor_command,
            edit_command,
            hermes_command,
        )

        args = {
            "goal": "implement the routing fix",
            "instruction": "fix the typo",
            "cwd": ".",
            "auto_route": True,
            "allowed_model_ids": ["grok-4.5", "composer-2.5"],
        }
        self.assertIn(
            ["--allowed-models", "composer-2.5"],
            _pairwise(cursor_command(args, implement=True)),
        )
        self.assertIn(
            ["--allowed-models", "grok-4.5"],
            _pairwise(claude_command(args)),
        )
        self.assertIn(
            ["--allowed-models", "grok-4.5"],
            _pairwise(codex_command(args)),
        )
        self.assertIn(
            ["--allowed-models", "grok-4.5"],
            _pairwise(hermes_command(args, implement=True)),
        )
        self.assertIn(
            ["--allowed-models", "grok-4.5"],
            _pairwise(agentic_command(args, implement=True)),
        )
        self.assertIn(
            ["--allowed-models", "composer-2.5"],
            _pairwise(edit_command(args)),
        )


def _pairwise(command: list[str]) -> list[list[str]]:
    pairs: list[list[str]] = []
    index = 0
    while index + 1 < len(command):
        pairs.append([command[index], command[index + 1]])
        index += 1
    return pairs


class RunStatusErrorRerouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from puppetmaster.platform_billing import clear_billing_cache

        clear_billing_cache()

    def tearDown(self) -> None:
        from puppetmaster.platform_billing import clear_billing_cache

        clear_billing_cache()

    def test_status_error_with_model_forbidden_classifies_unavailable(self) -> None:
        from puppetmaster.failure import classify_cursor_failure

        self.assertEqual(
            classify_cursor_failure(
                "status: error — model claude-fable-5 is not permitted on this plan"
            ),
            "model_unavailable",
        )
        self.assertEqual(
            classify_cursor_failure(
                "status:error Forbidden model claude-fable-5 unavailable"
            ),
            "model_unavailable",
        )
        self.assertEqual(
            classify_cursor_failure("status: error — agent run failed"),
            "run_status_error",
        )

    def test_run_status_error_reroutes_auto_routed_same_adapter(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore
        from puppetmaster.workers import RECOVERABLE_FAILURES, SAME_ADAPTER_MODEL_REROUTE

        self.assertIn("run_status_error", RECOVERABLE_FAILURES)
        self.assertIn("run_status_error", SAME_ADAPTER_MODEL_REROUTE)

        registry = _grok_composer_registry(include_fable=True)

        def _billing(adapter, **_kwargs):
            if adapter == "cursor":
                return BillingStatus(
                    adapter="cursor",
                    billing="plan",
                    healthy=True,
                    detail="ok",
                    evidence=[],
                )
            return BillingStatus(
                adapter=adapter,
                billing="unknown",
                healthy=False,
                detail="no",
                evidence=[],
            )

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("auto-routed cursor swarm")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the durable routing fix",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={
                    "auto_route": True,
                    "model": "claude-fable-5",
                    "router_model_id": "cursor/claude-fable-5",
                    "allowed_model_ids": ["grok-4.5", "composer-2.5"],
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
                        "result": "failed",
                        "failure": "run_status_error",
                        "adapter": "cursor",
                    },
                    confidence=0.5,
                    evidence=["adapter:cursor"],
                )
            )
            orch = Orchestrator(store)
            with mock.patch(
                "puppetmaster.model_registry.load_registry", return_value=registry
            ), mock.patch(
                "puppetmaster.platform_billing.detect_adapter_billing_cached",
                side_effect=_billing,
            ), mock.patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.adapter, "cursor")
            self.assertIn(updated.payload.get("model"), {"grok-4.5", "composer-2.5"})
            self.assertNotEqual(updated.payload.get("model"), "claude-fable-5")
            self.assertIn(
                "cursor/claude-fable-5", updated.payload.get("tried_models") or []
            )

    def test_explicit_pin_does_not_reroute_on_run_status_error(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore

        registry = _grok_composer_registry(include_fable=True)

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("pinned cursor run")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the fix",
                adapter="cursor",
                status=TaskStatus.FAILED,
                # Explicit pin: model set, auto_route absent/false.
                payload={"model": "claude-fable-5"},
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
                        "result": "failed",
                        "failure": "run_status_error",
                        "adapter": "cursor",
                    },
                    confidence=0.5,
                    evidence=["adapter:cursor"],
                )
            )
            orch = Orchestrator(store)
            with mock.patch(
                "puppetmaster.model_registry.load_registry", return_value=registry
            ), mock.patch(
                "puppetmaster.platform_billing.detect_adapter_billing_cached",
                return_value=BillingStatus(
                    adapter="cursor",
                    billing="plan",
                    healthy=True,
                    detail="ok",
                    evidence=[],
                ),
            ), mock.patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 0)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.FAILED)
            self.assertEqual(updated.payload.get("model"), "claude-fable-5")

    def test_run_status_error_allows_only_one_same_adapter_alternate(self) -> None:
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore

        registry = _grok_composer_registry(include_fable=True)

        def _billing(adapter, **_kwargs):
            return BillingStatus(
                adapter=adapter,
                billing="plan",
                healthy=True,
                detail="ok",
                evidence=[],
            )

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("run-status-error exhaustion")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement the fix",
                adapter="cursor",
                status=TaskStatus.FAILED,
                payload={
                    "auto_route": True,
                    "model": "claude-fable-5",
                    "router_model_id": "cursor/claude-fable-5",
                    "allowed_model_ids": ["grok-4.5", "composer-2.5"],
                    "run_status_error_same_adapter_reroutes": 1,
                    "tried_models": [
                        "cursor/claude-fable-5",
                        "cursor/grok-4-5",
                    ],
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
                        "result": "failed",
                        "failure": "run_status_error",
                        "adapter": "cursor",
                    },
                    confidence=0.5,
                    evidence=["adapter:cursor"],
                )
            )
            orch = Orchestrator(store)
            with mock.patch(
                "puppetmaster.model_registry.load_registry", return_value=registry
            ), mock.patch(
                "puppetmaster.platform_billing.detect_adapter_billing_cached",
                side_effect=_billing,
            ), mock.patch(
                "puppetmaster.preflight.adapter_cli_present", return_value=True
            ):
                rerouted = orch._reroute_recoverable_failures(job)
            self.assertEqual(rerouted, 0)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.FAILED)


class DiscoveryPreservesDisabledOverlayTests(unittest.TestCase):
    def test_cursor_catalog_refresh_keeps_disabled_overlay(self) -> None:
        from puppetmaster.cursor_discovery import catalog_to_specs, merge_catalog_into_registry
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(
                id="cursor/claude-fable-5",
                adapter="cursor",
                adapter_model_name="claude-fable-5",
                capability_score=100,
                billing="plan",
                enabled=False,
            ),
            ModelSpec(
                id="cursor/grok-4-5",
                adapter="cursor",
                adapter_model_name="grok-4-5",  # slug form in overlay
                capability_score=97,
                billing="plan",
                enabled=False,
            ),
        ]
        catalog = [
            {"id": "claude-fable-5", "displayName": "Fable 5"},
            {"id": "grok-4.5", "displayName": "Grok 4.5"},
            {"id": "composer-2.5", "displayName": "Composer 2.5"},
        ]
        specs = catalog_to_specs(catalog, existing)
        by_name = {spec.adapter_model_name: spec for spec in specs}
        self.assertFalse(by_name["claude-fable-5"].enabled)
        self.assertFalse(by_name["grok-4.5"].enabled)
        self.assertTrue(by_name["composer-2.5"].enabled)

        merged, _report = merge_catalog_into_registry(existing, catalog)
        merged_by_name = {spec.adapter_model_name: spec for spec in merged}
        self.assertFalse(merged_by_name["claude-fable-5"].enabled)
        self.assertFalse(merged_by_name["grok-4.5"].enabled)


if __name__ == "__main__":
    unittest.main()
