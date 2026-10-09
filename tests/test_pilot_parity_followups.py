"""Adapter parity follow-ups that the pilot applied outside the worker scopes."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.swarm_reasoning import WorkerEffortError, apply_swarm_reasoning
from puppetmaster.worker_resume import session_on_disk, task_resume_record
from puppetmaster.workers import WorkerSpec

_ENFORCE_HIGH = {"PUPPETMASTER_WORKER_EFFORT": "high", "PUPPETMASTER_WORKER_EFFORT_POLICY": "enforce"}


class OutputStylePromptTests(unittest.TestCase):
    def test_directive_reaches_a_payload_prompt(self):
        # Adapters send payload.prompt in place of instruction (F11).
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = Orchestrator(SQLiteSwarmStore(Path(tmp) / "state"))
            spec = WorkerSpec(role="explore", instruction="look", adapter="claude-code",
                              payload={"prompt": "the real prompt", "output_style": "ste"})
            styled = orchestrator._with_output_style([spec])[0]
        self.assertEqual(styled.payload["output_style"], "ste")
        self.assertTrue(styled.payload["prompt"].endswith("\n\nthe real prompt"))
        self.assertNotEqual(styled.payload["prompt"], "the real prompt")
        self.assertTrue(styled.instruction.endswith("\n\nlook"))


class FxPinPassThroughTests(unittest.TestCase):
    def test_fx_pin_is_not_bound_to_the_registry(self):
        # fx resolves FX_MODEL itself, and no catalog lists fx models.
        spec = WorkerSpec(role="build", instruction="x", adapter="fx",
                          payload={"model": "anthropic/claude-opus-5-5"})
        bound = Orchestrator._bind_explicit_model_pins([spec])[0]
        self.assertEqual(bound.payload["model"], "anthropic/claude-opus-5-5")


class AntigravityEffortChannelTests(unittest.TestCase):
    def _apply(self, model: str) -> dict:
        payload = {"model": model}
        return apply_swarm_reasoning(payload, payload, adapter="antigravity")

    def test_slug_effort_is_the_recorded_effort(self):
        result = self._apply("gemini-3.7-flash-high")
        self.assertEqual(result["reasoning_effort"], "high")
        self.assertEqual(result["reasoning_effort_source"], "model_slug")

    def test_model_without_effort_flag_records_unsupported(self):
        result = self._apply("gemini-2.5-flash")
        self.assertNotIn("reasoning_effort", result)
        self.assertEqual(result["reasoning_effort_source"], "adapter_unsupported")

    def test_flag_model_gets_the_run_effort(self):
        result = self._apply("gemini-3.8-flash")
        self.assertEqual(result["reasoning_effort"], "medium")
        self.assertEqual(result["reasoning_effort_source"], "swarm_default")

    def test_enforce_refuses_a_conflicting_slug_and_a_flagless_model(self):
        with patch.dict(os.environ, _ENFORCE_HIGH):
            self.assertEqual(self._apply("gemini-3.7-flash-high")["reasoning_effort"], "high")
            for model in ("gemini-3.7-flash-low", "gemini-2.5-flash"):
                with self.subTest(model=model), self.assertRaises(WorkerEffortError):
                    self._apply(model)


class RiskArtifactRecoveryTests(unittest.TestCase):
    def test_auth_risk_does_not_hide_a_recoverable_failure(self):
        # Agentic writes a RISK artifact with failure auth_failed:401 after the
        # not_authenticated VERIFICATION; the task must still reroute (F53).
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / "state")
            job = store.create_job("auth")
            orchestrator = Orchestrator(store)
            artifacts = [
                Artifact(job_id=job.id, task_id="t1", type=ArtifactType.VERIFICATION,
                         created_by="w", confidence=1.0, evidence=[], created_at="2026-10-09T00:00:01+00:00",
                         payload={"failure": "not_authenticated"}),
                Artifact(job_id=job.id, task_id="t1", type=ArtifactType.RISK,
                         created_by="w", confidence=0.95, evidence=[], created_at="2026-10-09T00:00:02+00:00",
                         payload={"failure": "auth_failed:401"}),
            ]
            mapped = orchestrator._recoverable_failure_by_task(job, artifacts)
        self.assertEqual(mapped, {"t1": "not_authenticated"})


class FxResumeTests(unittest.TestCase):
    def test_fx_resolves_an_explicit_session(self):
        record = task_resume_record({"resume_session_id": "sess-abc123"}, "fx")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["adapter"], "fx")
        self.assertEqual(record["session_id"], "sess-abc123")

    def test_fx_session_store_is_unknown_not_missing(self):
        self.assertIsNone(session_on_disk("fx", "sess-abc123"))


if __name__ == "__main__":
    unittest.main()
