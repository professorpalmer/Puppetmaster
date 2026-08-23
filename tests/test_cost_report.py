"""Focused coverage for the reusable ``build_cost_report`` extract."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import contextlib
import inspect
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.cli import main as cli_main
from puppetmaster.model_registry import ModelSpec, save_registry
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.store_factory import create_store


def _registry():
    return [
        ModelSpec(
            id="mid-model",
            adapter="claude-code",
            adapter_model_name="mid-v1",
            capability_score=80,
            input_per_mtok_usd=3.0,
            output_per_mtok_usd=15.0,
            billing="api",
        ),
        ModelSpec(
            id="flagship-model",
            adapter="claude-code",
            adapter_model_name="flagship-v1",
            capability_score=99,
            input_per_mtok_usd=15.0,
            output_per_mtok_usd=75.0,
            billing="api",
        ),
    ]


def _usage_verification(job_id: str) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id="task-1",
        type=ArtifactType.VERIFICATION,
        created_by="worker-1",
        confidence=0.9,
        evidence=["adapter:test"],
        payload={
            "adapter": "test",
            "check": "do the thing",
            "result": "passed",
            "model": "mid-v1",
            "tokens_in": 1_000_000,
            "tokens_out": 1_000_000,
            "tokens_estimated": False,
        },
    )


class BuildCostReportTests(unittest.TestCase):
    def test_build_cost_report_matches_mari_signature(self) -> None:
        from puppetmaster.cost import build_cost_report

        params = inspect.signature(build_cost_report).parameters
        self.assertEqual(list(params)[:2], ["store", "job_id"])
        self.assertIn("registry", params)

    def test_cli_json_matches_build_cost_report(self) -> None:
        from puppetmaster.cost import build_cost_report

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(_registry(), registry_path)
            state_dir = Path(tmp) / ".puppetmaster"
            store = create_store("file", state_dir)
            store.init()
            job = store.create_job("cost extract")
            store.save_artifacts([_usage_verification(job.id)])
            report = build_cost_report(store, job.id, registry=_registry())
            self.assertEqual(report["job_id"], job.id)
            self.assertIn("actual_cost", report)
            self.assertIn("counterfactual", report)
            self.assertIn("estimate_drift", report)
            self.assertAlmostEqual(
                report["actual_cost"]["total_marginal_cost_usd"], 18.0, places=6
            )

            prior = os.environ.get("PUPPETMASTER_MODELS_PATH")
            os.environ["PUPPETMASTER_MODELS_PATH"] = str(registry_path)
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli_main(
                        [
                            "--state-dir",
                            str(state_dir),
                            "--backend",
                            "file",
                            "cost",
                            job.id,
                            "--json",
                        ]
                    )
            finally:
                if prior is None:
                    os.environ.pop("PUPPETMASTER_MODELS_PATH", None)
                else:
                    os.environ["PUPPETMASTER_MODELS_PATH"] = prior
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_cli_and_mcp_call_build_cost_report(self) -> None:
        from puppetmaster import mcp_server
        from puppetmaster.cli.commands_gate import _run_cost_command

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry(_registry(), registry_path)
            state_dir = Path(tmp) / ".puppetmaster"
            store = create_store("file", state_dir)
            store.init()
            job = store.create_job("cost wiring")
            store.save_artifacts([_usage_verification(job.id)])
            sentinel = {
                "job_id": job.id,
                "cost_basis": "preflight_routing_estimate",
                "total_estimated_cost_usd": 0.0,
                "actual_cost": {"total_marginal_cost_usd": 18.0},
            }

            class Args:
                pass
            Args.json = True
            Args.job_id = job.id
            Args.registry_path = str(registry_path)

            with patch(
                "puppetmaster.cost.build_cost_report", return_value=sentinel
            ) as mocked:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = _run_cost_command(Args(), store)
                self.assertEqual(rc, 0)
                mocked.assert_called_once()
                self.assertEqual(json.loads(out.getvalue()), sentinel)

            with patch(
                "puppetmaster.cost.build_cost_report", return_value=sentinel
            ) as mocked:
                result = mcp_server.run_job_cost(
                    {
                        "job_id": job.id,
                        "state_dir": str(state_dir),
                        "backend": "file",
                        "registry_path": str(registry_path),
                        "cwd": tmp,
                    }
                )
                mocked.assert_called_once()
                self.assertFalse(result.get("isError"))
                body = json.loads(result["content"][0]["text"])
                self.assertEqual(json.loads(body["stdout"]), sentinel)


if __name__ == "__main__":
    unittest.main()
