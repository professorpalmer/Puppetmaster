"""Unit tests for OMP-style plan-then-cheap prewalk (no subprocess)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster.adapters._prompts import (
    build_implement_prompt,
    with_prewalk_plan,
)
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.prewalk import (
    IMPLEMENT_ROLE,
    PLAN_ROLE,
    PREWALK_PLAN_SECTION_HEADER,
    VERIFY_ROLE,
    build_prewalk_specs,
    format_plan_artifacts_for_injection,
    inject_plan_into_prompt,
)
from puppetmaster.savings import collect_routing_records, summarize_routing
from puppetmaster.store import SwarmStore
from puppetmaster.workers import (
    ANALYSIS_NO_EDIT_PAYLOAD,
    spec_edits_files,
    spec_explicitly_no_edit,
    swarm_mode,
)

class BuildPrewalkSpecsTests(unittest.TestCase):
    def test_plan_before_implement_with_depends_on(self) -> None:
        specs = build_prewalk_specs("Add retries to the client", cwd="/repo")
        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].role, PLAN_ROLE)
        self.assertEqual(specs[1].role, IMPLEMENT_ROLE)
        self.assertEqual(specs[2].role, VERIFY_ROLE)
        self.assertEqual(specs[1].depends_on_roles, [PLAN_ROLE])
        self.assertEqual(specs[2].depends_on_roles, [IMPLEMENT_ROLE])
        self.assertEqual(specs[0].depends_on_roles, [])

    def test_plan_is_read_only_implement_is_edit(self) -> None:
        specs = build_prewalk_specs("Wire the flag", cwd="/repo")
        plan, implement, verify = specs
        self.assertTrue(spec_explicitly_no_edit(plan))
        self.assertFalse(spec_edits_files(plan))
        for key, value in ANALYSIS_NO_EDIT_PAYLOAD.items():
            self.assertEqual(plan.payload.get(key), value)
        self.assertFalse(spec_explicitly_no_edit(implement))
        self.assertTrue(spec_edits_files(implement))
        self.assertEqual(implement.payload.get("mode"), "implement")
        self.assertNotEqual(implement.payload.get("read_only"), True)
        self.assertTrue(spec_explicitly_no_edit(verify))
        self.assertFalse(spec_edits_files(verify))
        self.assertEqual(swarm_mode(specs), "edit")

    def test_routing_policies_quality_then_cheap(self) -> None:
        specs = build_prewalk_specs("Refactor paginate", cwd="/repo")
        plan, implement, verify = specs
        self.assertTrue(plan.payload.get("auto_route"))
        self.assertEqual(plan.payload.get("routing_policy"), "quality")
        self.assertTrue(implement.payload.get("auto_route"))
        self.assertEqual(implement.payload.get("routing_policy"), "cheap")
        self.assertTrue(implement.payload.get("prewalk"))
        self.assertTrue(verify.payload.get("prewalk"))
        self.assertEqual(verify.payload.get("prewalk_role"), VERIFY_ROLE)

    def test_implement_instruction_requires_applying_upstream_plan(self) -> None:
        specs = build_prewalk_specs("Ship the feature", cwd="/repo")
        implement = specs[1]
        self.assertIn("upstream plan", implement.instruction.lower())
        self.assertIn("apply", implement.instruction.lower())
        self.assertIn(PREWALK_PLAN_SECTION_HEADER, implement.payload["prompt"])
        self.assertIn("Ship the feature", implement.payload["prompt"])

    def test_pinned_model_disables_auto_route_fields(self) -> None:
        specs = build_prewalk_specs(
            "x",
            cwd="/repo",
            plan_model="claude-opus",
            implement_model="gpt-5-nano",
            implement_adapter="hermes",
        )
        plan, implement, _verify = specs
        self.assertEqual(plan.payload.get("model"), "claude-opus")
        self.assertNotIn("auto_route", plan.payload)
        self.assertEqual(implement.payload.get("model"), "gpt-5-nano")
        self.assertNotIn("auto_route", implement.payload)

    def test_implement_adapter_pin_constrains_routing(self) -> None:
        specs = build_prewalk_specs(
            "x", cwd="/repo", implement_adapter="claude-code"
        )
        implement = specs[1]
        self.assertEqual(implement.adapter, "claude-code")
        self.assertEqual(implement.payload.get("allowed_adapters"), ["claude-code"])

    def test_rejects_empty_goal(self) -> None:
        with self.assertRaises(ValueError):
            build_prewalk_specs("  ", cwd="/repo")

class FormatPlanInjectionTests(unittest.TestCase):
    def test_formats_decision_artifact_objects(self) -> None:
        artifact = Artifact(
            job_id="j",
            task_id="t",
            type=ArtifactType.DECISION,
            created_by="plan",
            payload={
                "decision": "Add retry helper in client.py",
                "why": "Centralize backoff",
                "plan": [
                    "Create retry_with_backoff in client.py",
                    {"step": "Wire callers", "files": ["api.py", "cli.py"]},
                ],
                "files": ["client.py"],
            },
            confidence=0.9,
            evidence=["goal:retries"],
        )
        text = format_plan_artifacts_for_injection([artifact])
        self.assertIn("Decision: Add retry helper in client.py", text)
        self.assertIn("Why: Centralize backoff", text)
        self.assertIn("Plan steps:", text)
        self.assertIn("1. Create retry_with_backoff in client.py", text)
        self.assertIn("2. Wire callers [api.py, cli.py]", text)
        self.assertIn("Files: client.py", text)

    def test_formats_plan_typed_dict_and_steps_key(self) -> None:
        artifacts = [
            {
                "type": "plan",
                "payload": {
                    "steps": ["Touch a.py", "Test it"],
                    "notes": "Keep the change small",
                },
            }
        ]
        text = format_plan_artifacts_for_injection(artifacts)
        self.assertIn("1. Touch a.py", text)
        self.assertIn("2. Test it", text)
        self.assertIn("Notes: Keep the change small", text)

    def test_skips_non_plan_artifacts(self) -> None:
        artifacts = [
            {
                "type": "finding",
                "payload": {"claim": "irrelevant"},
            },
            {
                "type": "decision",
                "payload": {
                    "decision": "Do the thing",
                    "why": "Because",
                },
            },
        ]
        text = format_plan_artifacts_for_injection(artifacts)
        self.assertIn("Decision: Do the thing", text)
        self.assertNotIn("irrelevant", text)

    def test_inject_plan_into_prompt_prepends_section(self) -> None:
        artifacts = [
            {
                "type": "decision",
                "payload": {"decision": "Ship it", "why": "Ready"},
            }
        ]
        prompt = inject_plan_into_prompt("Implement the goal", artifacts)
        self.assertTrue(prompt.startswith(PREWALK_PLAN_SECTION_HEADER))
        self.assertIn("Decision: Ship it", prompt)
        self.assertTrue(prompt.rstrip().endswith("Implement the goal"))

    def test_inject_plan_replaces_placeholder_section(self) -> None:
        stub = (
            f"Goal:\ndo the thing\n\n"
            f"{PREWALK_PLAN_SECTION_HEADER}\n"
            "(Read this job's upstream plan/decision artifacts from the plan "
            "worker and apply them exactly. The plan worker completed before you "
            "were unblocked.)"
        )
        artifacts = [
            {
                "type": "decision",
                "payload": {"decision": "Touch client.py", "why": "Centralize"},
            }
        ]
        prompt = inject_plan_into_prompt(stub, artifacts)
        self.assertIn("Decision: Touch client.py", prompt)
        self.assertIn("Goal:\ndo the thing", prompt)
        self.assertNotIn("(Read this job's upstream plan", prompt)
        self.assertEqual(prompt.count(PREWALK_PLAN_SECTION_HEADER), 1)

    def test_inject_plan_noop_without_usable_artifacts(self) -> None:
        prompt = inject_plan_into_prompt("Keep me", [{"type": "risk", "payload": {"risk": "x", "mitigation": "y"}}])
        self.assertEqual(prompt, "Keep me")

class WithPrewalkPlanTests(unittest.TestCase):
    def test_with_prewalk_plan_injects_decision_from_payload_artifacts(self) -> None:
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
        specs = build_prewalk_specs("Add retries", cwd="/repo")
        implement_prompt = specs[1].payload["prompt"]
        task = Task(
            job_id="job-prewalk-1",
            role=IMPLEMENT_ROLE,
            instruction=specs[1].instruction,
            payload={
                "prewalk": True,
                "prewalk_artifacts": artifacts,
                "mode": "implement",
                "prompt": implement_prompt,
            },
        )
        assembled = build_implement_prompt(implement_prompt)
        result = with_prewalk_plan(assembled, task)
        self.assertIn("Decision: Add retry helper in client.py", result)
        self.assertIn("Why: Centralize backoff", result)
        self.assertIn("Create retry_with_backoff", result)
        self.assertNotIn("(Read this job's upstream plan", result)

    def test_with_prewalk_plan_loads_artifacts_from_store(self) -> None:
        artifacts = [
            Artifact(
                job_id="job-store",
                task_id="plan-1",
                type=ArtifactType.DECISION,
                created_by="plan",
                payload={
                    "decision": "Edit api.py only",
                    "why": "Smallest change",
                },
                confidence=0.9,
                evidence=["goal"],
            )
        ]
        task = Task(
            job_id="job-store",
            role=IMPLEMENT_ROLE,
            instruction="Apply the plan",
            payload={"prewalk": True, "mode": "implement"},
        )
        base = build_implement_prompt(
            f"Apply the plan\n\n{PREWALK_PLAN_SECTION_HEADER}\n(placeholder stub)"
        )
        with mock.patch(
            "puppetmaster.adapters._prompts._load_job_artifacts_for_task",
            return_value=artifacts,
        ):
            result = with_prewalk_plan(base, task)
        self.assertIn("Decision: Edit api.py only", result)
        self.assertNotIn("(placeholder stub)", result)

    def test_with_prewalk_plan_noop_without_prewalk_flag(self) -> None:
        task = Task(
            job_id="j",
            role="implement",
            instruction="x",
            payload={
                "prewalk_artifacts": [
                    {"type": "decision", "payload": {"decision": "Should not appear"}}
                ]
            },
        )
        prompt = "plain implement prompt"
        self.assertEqual(with_prewalk_plan(prompt, task), prompt)

class PrewalkMcpCommandTests(unittest.TestCase):
    def test_prewalk_command_builds_cli_argv(self) -> None:
        from puppetmaster.mcp_server import prewalk_command

        command = prewalk_command(
            {
                "goal": "Add a flag",
                "cwd": "/repo",
                "adapter": "hermes",
                "plan_adapter": "cursor",
                "verify_adapter": "local",
                "verify_model": "composer-2",
                "timeout_seconds": 600,
                "verify_timeout_seconds": 300,
                "allow_dirty": True,
            }
        )
        self.assertEqual(command[0], "prewalk")
        self.assertIn("Add a flag", command)
        self.assertIn("--adapter", command)
        self.assertIn("hermes", command)
        self.assertIn("--plan-adapter", command)
        self.assertIn("cursor", command)
        self.assertIn("--verify-adapter", command)
        self.assertIn("local", command)
        self.assertIn("--verify-model", command)
        self.assertIn("composer-2", command)
        self.assertIn("--allow-dirty", command)
        self.assertIn("--timeout-seconds", command)
        self.assertIn("600", command)
        self.assertIn("--verify-timeout-seconds", command)
        self.assertIn("300", command)

    def test_build_prewalk_passes_verify_settings(self) -> None:
        specs = build_prewalk_specs(
            "x",
            cwd="/repo",
            verify_adapter="cursor",
            verify_model="composer-2",
            verify_timeout_seconds=321,
        )
        verify = specs[2]
        self.assertEqual(verify.adapter, "cursor")
        self.assertEqual(verify.payload.get("model"), "composer-2")
        self.assertEqual(verify.payload.get("timeout_seconds"), 321)

class PrewalkSavingsLedgerTests(unittest.TestCase):
    """Prewalk jobs stamp one ROUTING artifact per leg; savings must attribute
    quality plan spend and cheap implement savings without double-counting."""

    def test_prewalk_routing_legs_attribute_without_double_count(self) -> None:
        specs = build_prewalk_specs("Harden the retry path", cwd="/repo")
        self.assertEqual(len(specs), 3)
        plan_spec, implement_spec, _verify_spec = specs
        self.assertEqual(plan_spec.payload.get("routing_policy"), "quality")
        self.assertEqual(implement_spec.payload.get("routing_policy"), "cheap")

        # Distinct per-leg costs: plan deliberate spend must not offset or
        # inflate implement savings (and vice versa).
        plan_chosen = 0.12
        implement_chosen = 0.02
        implement_baseline = 0.10

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("prewalk: Harden the retry path")
            plan_task = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction=plan_spec.instruction,
                adapter=plan_spec.adapter,
                status=TaskStatus.COMPLETE,
                payload=dict(plan_spec.payload),
            )
            implement_task = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction=implement_spec.instruction,
                adapter=implement_spec.adapter,
                status=TaskStatus.COMPLETE,
                payload=dict(implement_spec.payload),
            )
            store.save_task(plan_task)
            store.save_task(implement_task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=plan_task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "frontier/95",
                        "adapter": "cursor",
                        "policy": plan_spec.payload["routing_policy"],
                        "estimated_cost_usd": plan_chosen,
                        "baseline_cost_usd": plan_chosen,
                        "baseline_model_id": "frontier/95",
                    },
                    confidence=0.9,
                    evidence=["prewalk-plan"],
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=implement_task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "cheap/40",
                        "adapter": "cursor",
                        "policy": implement_spec.payload["routing_policy"],
                        "estimated_cost_usd": implement_chosen,
                        "baseline_cost_usd": implement_baseline,
                        "baseline_model_id": "frontier/95",
                    },
                    confidence=0.9,
                    evidence=["prewalk-implement"],
                )
            )
            # Re-dispatch noise on implement must not double-count savings.
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=implement_task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "cheap/40",
                        "adapter": "cursor",
                        "policy": "cheap",
                        "estimated_cost_usd": implement_chosen,
                        "baseline_cost_usd": implement_baseline,
                        "baseline_model_id": "frontier/95",
                    },
                    confidence=0.9,
                    evidence=["prewalk-implement-redispatch"],
                )
            )

            records, jobs, _heal = collect_routing_records([store])
            self.assertEqual(jobs, 1)
            self.assertEqual(len(records), 2)
            policies = sorted(r.policy for r in records)
            self.assertEqual(policies, ["cheap", "quality"])

            summary = summarize_routing(records)
            self.assertEqual(summary.tasks_total, 2)
            self.assertEqual(summary.deliberate_tasks, 1)
            self.assertAlmostEqual(
                summary.deliberate_spend_usd, plan_chosen, places=6
            )
            self.assertEqual(summary.cost_optimizing_tasks, 1)
            self.assertAlmostEqual(
                summary.saved_usd, implement_baseline - implement_chosen, places=6
            )
            self.assertAlmostEqual(summary.baseline_usd, implement_baseline, places=6)
            self.assertAlmostEqual(summary.chosen_usd, implement_chosen, places=6)
            # Plan spend is reported separately — never subtracted from savings.
            self.assertNotAlmostEqual(
                summary.saved_usd,
                (implement_baseline - implement_chosen) - plan_chosen,
                places=6,
            )

if __name__ == "__main__":
    unittest.main()
