"""Operator worker-effort profile: default and enforced effort for unpinned workers."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from test_flow_runtime import Base, agent, graph, judge
from puppetmaster import flow
from puppetmaster.adapters.codex import build_codex_exec_command
from puppetmaster.flow import JobNodeExecutor
from puppetmaster.model_registry import ModelSpec, save_registry
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.store_factory import create_store
from puppetmaster.swarm_reasoning import (
    WorkerEffortError,
    WorkerEffortProfile,
    apply_swarm_reasoning,
    operator_effort_profile,
)

HIGH = {"PUPPETMASTER_WORKER_EFFORT": "high"}
ENFORCE_HIGH = {"PUPPETMASTER_WORKER_EFFORT": "high", "PUPPETMASTER_WORKER_EFFORT_POLICY": "enforce"}


class ProfileTests(unittest.TestCase):
    def test_parsing(self) -> None:
        self.assertEqual(operator_effort_profile({}), WorkerEffortProfile())
        self.assertEqual(operator_effort_profile({"PUPPETMASTER_WORKER_EFFORT": " High "}),
                         WorkerEffortProfile("high", False))
        self.assertEqual(operator_effort_profile(ENFORCE_HIGH), WorkerEffortProfile("high", True))

    def test_invalid_or_ambiguous_values_fail_clearly(self) -> None:
        for env, needle in (({"PUPPETMASTER_WORKER_EFFORT": "max"}, "must be one of"),
                            ({"PUPPETMASTER_WORKER_EFFORT": "high,low"}, "must be one of"),
                            ({"PUPPETMASTER_WORKER_EFFORT": "high",
                              "PUPPETMASTER_WORKER_EFFORT_POLICY": "strict"}, "POLICY must be one of"),
                            ({"PUPPETMASTER_WORKER_EFFORT_POLICY": "enforce"}, "needs PUPPETMASTER_WORKER_EFFORT")):
            with self.subTest(env=env), self.assertRaisesRegex(WorkerEffortError, needle):
                operator_effort_profile(env)

    def test_precedence_and_provenance(self) -> None:
        cases = (
            (WorkerEffortProfile(), {}, "medium", None, "swarm_default"),
            (WorkerEffortProfile("high"), {}, "high", None, "operator_default"),
            (WorkerEffortProfile("high"), {"reasoning_effort": "low"}, "low", "low", "caller"),
            (WorkerEffortProfile("high", True), {}, "high", None, "operator_enforced"),
            (WorkerEffortProfile("high", True), {"reasoning_effort": "high"}, "high", "high", "operator_enforced"),
        )
        for profile, caller, effort, requested, source in cases:
            with self.subTest(profile=profile, caller=caller):
                merged = apply_swarm_reasoning({}, caller, adapter="codex", profile=profile)
                self.assertEqual((merged["reasoning_effort"], merged["requested_reasoning_effort"],
                                  merged["reasoning_effort_source"]), (effort, requested, source))
                self.assertEqual(merged["extra_args"], ["-c", f"model_reasoning_effort={effort}"])

    def test_enforced_profile_rejects_a_conflicting_pin(self) -> None:
        with self.assertRaisesRegex(WorkerEffortError, "requested low"):
            apply_swarm_reasoning({}, {"reasoning_effort": "low"}, adapter="codex",
                                  profile=WorkerEffortProfile("high", True))

    def test_a_restamped_payload_keeps_its_source(self) -> None:
        stamped = apply_swarm_reasoning({}, {}, adapter="codex", profile=WorkerEffortProfile("high"))
        again = apply_swarm_reasoning(dict(stamped), stamped, adapter="codex", profile=WorkerEffortProfile("high"))
        self.assertEqual((again["reasoning_effort"], again["reasoning_effort_source"]), ("high", "operator_default"))
        cleared = apply_swarm_reasoning(dict(stamped), stamped, adapter="codex", profile=WorkerEffortProfile())
        self.assertEqual((cleared["reasoning_effort"], cleared["reasoning_effort_source"]), ("medium", "swarm_default"))


class StockTaskTests(Base):
    """Flow agent, judge and resumed visits through the real task creation."""

    def setUp(self) -> None:
        super().setUp()
        self.registry = self.root / "models.json"
        save_registry([ModelSpec(id="codex/sol", adapter="codex", adapter_model_name="sol",
                                 capability_score=100, billing="plan")], self.registry)
        self.store = create_store("file", self.root / ".puppetmaster")
        self.store.init()
        g = graph([agent("build"), judge("check")],
                  defaults={"adapter": "codex", "model": "sol",
                            "payload": {"registry_path": str(self.registry)}})
        self.run = flow.new_run(self.state, g, "goal", cwd=str(self.work))
        self.executor = JobNodeExecutor(self.state)

    def tasks(self, env: dict) -> dict:
        nodes = {node["id"]: node for node in self.run.graph["nodes"]}
        self.run.sessions["build"] = {"job_id": "job-prior", "task_id": "task-prior"}
        specs = [self.executor._spec("build", nodes["build"], self.run, "", 1),
                 self.executor._spec("build", nodes["build"], self.run, "fix it", 2),
                 self.executor._spec("check", nodes["check"], self.run, "built", 1)]
        self.assertNotIn("reasoning_effort", specs[0].payload)
        self.assertIn("resume_from", specs[1].payload)
        with patch.dict(os.environ, env):
            job = self.store.create_job("effort profile")
            created = Orchestrator(self.store)._create_tasks(job, specs)
        return {name: task.payload for name, task in zip(("cold", "resumed", "judge"), created)}

    def test_no_setting_keeps_medium(self) -> None:
        for name, payload in self.tasks({}).items():
            with self.subTest(task=name):
                self.assertEqual(payload["reasoning_effort"], "medium")
                self.assertEqual(payload["reasoning_effort_source"], "swarm_default")

    def test_operator_default_reaches_every_worker_and_the_cli(self) -> None:
        for name, payload in self.tasks(HIGH).items():
            with self.subTest(task=name):
                self.assertEqual(payload["reasoning_effort"], "high")
                self.assertIsNone(payload["requested_reasoning_effort"])
                self.assertEqual(payload["reasoning_effort_source"], "operator_default")
                command = build_codex_exec_command(executable=["codex"], model=payload["model"],
                                                   extra_args=payload["extra_args"])
                self.assertIn("model_reasoning_effort=high", command)
                self.assertNotIn("model_reasoning_effort=medium", command)

    def test_node_effort_is_a_caller_pin_over_the_default(self) -> None:
        self.run.graph["nodes"][1]["effort"] = "xhigh"
        payloads = self.tasks(HIGH)
        self.assertEqual((payloads["judge"]["reasoning_effort"], payloads["judge"]["reasoning_effort_source"]),
                         ("xhigh", "caller"))
        self.assertEqual(payloads["cold"]["reasoning_effort"], "high")

    def test_enforced_profile_refuses_a_conflicting_node_before_launch(self) -> None:
        self.run.graph["nodes"][1]["effort"] = "low"
        with self.assertRaisesRegex(WorkerEffortError, "requested low"):
            self.tasks(ENFORCE_HIGH)
        with patch.dict(os.environ, ENFORCE_HIGH):
            problems = flow.validate_graph(graph([agent("a", effort="low"), agent("b", escalate=True)],
                                                 defaults={"adapter": "codex", "lanes": {"judge": "high"}}))
        self.assertTrue(any("node 'a' effort low conflicts" in p for p in problems))
        self.assertTrue(any("node 'b' escalate conflicts" in p for p in problems))
        self.assertFalse(any("defaults" in p for p in problems))

    def test_invalid_setting_refuses_task_creation(self) -> None:
        with self.assertRaisesRegex(WorkerEffortError, "must be one of"):
            self.tasks({"PUPPETMASTER_WORKER_EFFORT": "maximum"})


if __name__ == "__main__":
    unittest.main()
