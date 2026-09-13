"""Generated auto-routed swarms must recover across a launch-lane adapter lock.

Issue #195: swarm_launch pins allowed_adapters to the launch adapter so the
initial route cannot silently hop (v1.20.6). Orchestrator fallback then
rebuilds signals from that same pin and rejects a funded agentic/openai
identity after billing_or_quota. The public path is a generated Codex swarm
task, not a hand-built payload that omits allowed_adapters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.model_registry import ModelSpec, registry_digest, save_registry
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.platform_billing import BillingStatus, _BILLING_CACHE
from puppetmaster.store import SwarmStore
from puppetmaster.swarm_launch import build_analysis_swarm_specs


def _persist_registry(root, models):
    path = Path(root) / "models.json"
    save_registry(models, path)
    return {
        "registry_path": str(path),
        "registry_digest": registry_digest(models),
    }


def _codex_openai_registry():
    return [
        ModelSpec(
            id="codex/gpt-5-5",
            adapter="codex",
            adapter_model_name="gpt-5.5",
            capability_score=90,
            billing="plan",
            tags=["codex"],
        ),
        ModelSpec(
            id="agentic/openai/gpt-5-6-sol",
            adapter="agentic",
            adapter_model_name="gpt-5.6-sol",
            capability_score=92,
            billing="api",
            tags=["agentic"],
            payload_defaults={"provider": "openai"},
        ),
    ]


def _billing(adapter, **_kwargs):
    if adapter == "agentic":
        return BillingStatus(
            adapter="agentic",
            billing="api",
            healthy=True,
            detail="openai ready",
            evidence=["openai"],
        )
    return BillingStatus(
        adapter=adapter,
        billing="unknown",
        healthy=False,
        detail="no",
        evidence=[],
    )


class GeneratedSwarmFallbackTests(TestCase):
    def setUp(self) -> None:
        _BILLING_CACHE.clear()

    def tearDown(self) -> None:
        _BILLING_CACHE.clear()

    def _failed_generated_task(self, tmp, *, adapter="codex", extra_payload=None):
        registry = _codex_openai_registry()
        if adapter == "cursor":
            registry = [
                ModelSpec(
                    id="cursor/grok-4-6",
                    adapter="cursor",
                    adapter_model_name="grok-4.6",
                    capability_score=90,
                    billing="plan",
                    tags=["cursor"],
                ),
                registry[1],
            ]
        specs = build_analysis_swarm_specs(
            "audit the repo",
            ["explore"],
            adapter=adapter,
            cwd=str(tmp),
            allowed_model_ids=[spec.id for spec in registry],
        )
        spec = specs[0]
        payload = dict(spec.payload)
        payload.update(_persist_registry(tmp, registry))
        if extra_payload:
            payload.update(extra_payload)
        store = SwarmStore(Path(tmp) / ".puppetmaster")
        job = store.create_job("generated swarm fallback")
        task = Task(
            job_id=job.id,
            role=spec.role,
            instruction=spec.instruction,
            adapter=spec.adapter,
            status=TaskStatus.FAILED,
            payload=payload,
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
                    "failure": "billing_or_quota",
                    "adapter": adapter,
                },
                confidence=0.5,
                evidence=["adapter:%s" % adapter],
            )
        )
        return store, job, task, spec

    def _reroute(self, store, job):
        orch = Orchestrator(store)
        with patch(
            "puppetmaster.platform_billing.detect_adapter_billing",
            side_effect=_billing,
        ), patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=True,
        ), patch(
            "puppetmaster.preflight.adapter_cli_present",
            return_value=True,
        ), patch(
            "puppetmaster.providers.available_providers",
            return_value={"openai"},
        ):
            return orch._reroute_recoverable_failures(job)

    def test_generated_codex_swarm_falls_back_to_funded_openai(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task, spec = self._failed_generated_task(tmp)
            self.assertEqual(spec.payload.get("allowed_adapters"), ["codex"])
            self.assertEqual(spec.payload.get("adapter_lock"), "lane")
            self.assertTrue(spec.payload.get("auto_route"))
            rerouted = self._reroute(store, job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.QUEUED)
            self.assertEqual(updated.adapter, "agentic")
            self.assertEqual(
                updated.payload.get("router_model_id"),
                "agentic/openai/gpt-5-6-sol",
            )
            self.assertEqual(updated.payload.get("billing"), "api")
            self.assertEqual(updated.payload.get("allowed_adapters"), ["agentic"])
            self.assertEqual(updated.payload.get("adapter_lock"), "lane")
            fallbacks = [
                artifact
                for artifact in store.list_artifacts(job.id)
                if artifact.created_by == "router-fallback"
                and artifact.type == ArtifactType.ROUTING
            ]
            self.assertEqual(len(fallbacks), 1)
            body = fallbacks[0].payload
            self.assertEqual(body.get("model_id"), "agentic/openai/gpt-5-6-sol")
            self.assertEqual(body.get("adapter"), "agentic")
            self.assertEqual(body.get("billing"), "api")
            self.assertEqual(body.get("fallback_reason"), "billing_or_quota")
            self.assertEqual(body.get("fallback_from_adapter"), "codex")

    def test_hard_adapter_pin_still_blocks_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task, spec = self._failed_generated_task(
                tmp,
                extra_payload={"adapter_lock": "hard"},
            )
            self.assertEqual(spec.payload.get("allowed_adapters"), ["codex"])
            rerouted = self._reroute(store, job)
            self.assertEqual(rerouted, 0)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.status, TaskStatus.FAILED)
            self.assertEqual(updated.adapter, "codex")
            self.assertFalse(
                any(
                    artifact.created_by == "router-fallback"
                    for artifact in store.list_artifacts(job.id)
                )
            )

    def test_generated_cursor_lane_also_recovers_to_openai(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task, spec = self._failed_generated_task(tmp, adapter="cursor")
            self.assertEqual(spec.payload.get("allowed_adapters"), ["cursor"])
            rerouted = self._reroute(store, job)
            self.assertEqual(rerouted, 1)
            updated = store.get_task_by_id(task.id)
            self.assertEqual(updated.adapter, "agentic")
            self.assertEqual(
                updated.payload.get("router_model_id"),
                "agentic/openai/gpt-5-6-sol",
            )
