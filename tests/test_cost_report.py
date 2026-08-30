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
import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import patch

from puppetmaster import mcp_server
from puppetmaster.cli import main as cli_main
from puppetmaster.cli.commands_gate import _run_cost_command
from puppetmaster.cli.commands_jobs import _run_finalize_command
from puppetmaster.cost import (
    build_cost_report,
    build_current_registry_cost_report,
    final_routing_artifacts,
    price_job,
    routing_estimate_rows,
    valid_terminal_cost_receipt,
)
from puppetmaster.model_registry import ModelSpec, save_registry
from puppetmaster.models import (
    Artifact,
    ArtifactType,
    JobStatus,
    Task,
    TaskStatus,
    job_from_dict,
    to_jsonable,
)
from puppetmaster.store_factory import create_store
from puppetmaster.usage import aggregate_token_usage, select_usage_records
from puppetmaster.validation import validation_status_of

_RECOVERABLE_STATUSES = (
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.STITCHING,
    JobStatus.STALLED,
)
_COST_FINAL_NON_COMPLETE = (JobStatus.FAILED, JobStatus.CANCELLED)
_BACKENDS = ("file", "sqlite")


def _spec(
    model_id: str,
    adapter_model_name: str,
    score: int,
    inp: float,
    out: float,
    *,
    adapter: str = "claude-code",
    billing: str = "api",
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        adapter=adapter,
        adapter_model_name=adapter_model_name,
        capability_score=score,
        input_per_mtok_usd=inp,
        output_per_mtok_usd=out,
        billing=billing,
    )


def _registry():
    return [
        _spec("mid-model", "mid-v1", 80, 3.0, 15.0),
        _spec("flagship-model", "flagship-v1", 99, 15.0, 75.0),
    ]


def _flagship_only():
    return [spec for spec in _registry() if spec.id == "flagship-model"]


def _plan_registry():
    return [
        _spec(
            "plan/cursor",
            "default",
            80,
            5.0,
            25.0,
            adapter="cursor",
            billing="plan",
        ),
        *_flagship_only(),
    ]


def _cheaper_mid():
    return [_spec("mid-model", "mid-v1", 80, 0.01, 0.01)]


def _routing(
    job_id: str,
    task_id: str,
    model_id: str,
    *,
    billing: Optional[str] = None,
    created_by: str = "router",
    estimated_cost_usd: Optional[float] = None,
    validation: Optional[dict] = None,
) -> Artifact:
    payload = {"model_id": model_id, "adapter": "claude-code", "policy": "balanced"}
    if billing is not None:
        payload["billing"] = billing
    if estimated_cost_usd is not None:
        payload["estimated_cost_usd"] = estimated_cost_usd
    if validation is not None:
        payload["validation"] = validation
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.ROUTING,
        created_by=created_by,
        confidence=0.9,
        evidence=["role:audit"],
        payload=payload,
    )


def _usage(
    job_id: str,
    task_id: str = "t1",
    *,
    model: str = "mid-v1",
    tokens_in: int = 1_000_000,
    tokens_out: int = 1_000_000,
    real_cost_usd: Optional[float] = None,
    validation: Optional[dict] = None,
) -> Artifact:
    payload = {
        "adapter": "test",
        "check": "do the thing",
        "result": "passed",
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_estimated": False,
    }
    if real_cost_usd is not None:
        payload["real_cost_usd"] = real_cost_usd
    if validation is not None:
        payload["validation"] = validation
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.VERIFICATION,
        created_by="worker-1",
        confidence=0.9,
        evidence=["adapter:test"],
        payload=payload,
    )


@contextlib.contextmanager
def _models_path(path: Path):
    prior = os.environ.get("PUPPETMASTER_MODELS_PATH")
    os.environ["PUPPETMASTER_MODELS_PATH"] = str(path)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("PUPPETMASTER_MODELS_PATH", None)
        else:
            os.environ["PUPPETMASTER_MODELS_PATH"] = prior


def _make_store(backend: str, tmp: str):
    state_dir = Path(tmp) / f".puppetmaster-{backend}"
    store = create_store(backend, state_dir)
    store.init()
    return store


@contextlib.contextmanager
def _harness(backend: str = "file", registry=None):
    specs = list(_registry() if registry is None else registry)
    with TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "models.json"
        save_registry(specs, registry_path)
        yield _make_store(backend, tmp), registry_path


def _job_with_usage(store, goal: str, **usage_kwargs):
    job = store.create_job(goal)
    store.save_artifacts([_usage(job.id, **usage_kwargs)])
    return job


def _write_receipt(store, job_id: str, status: JobStatus, receipt):
    store.save_job(
        replace(store.get_job(job_id), status=status, cost_receipt=receipt)
    )


def _assert_immune_to_registry_churn(
    test, store, job_id: str, registry_path: Path, first: dict, registries
) -> None:
    for registry in registries:
        save_registry(registry, registry_path)
        test.assertEqual(build_cost_report(store, job_id, registry=registry), first)
    mutated = build_cost_report(store, job_id, registry=registries[-1])
    mutated["actual_cost"]["total_marginal_cost_usd"] = 99.0
    test.assertEqual(build_cost_report(store, job_id, registry=registries[-1]), first)


def _stamp(store, job_id: str, status: JobStatus, registry_path: Path):
    with _models_path(registry_path):
        return store.update_job_status(job_id, status)


def _finalize_production(store, job_id: str, registry_path: Path):
    """Production finalize: stitch, then COMPLETE only when the job is open."""

    class Args:
        pass

    Args.job_id = job_id
    with _models_path(registry_path), contextlib.redirect_stdout(io.StringIO()):
        return _run_finalize_command(Args(), store)


def _seed_priced_job(store, registry_path: Path, registry):
    save_registry(registry, registry_path)
    job = store.create_job("stable cost")
    task = Task(
        job_id=job.id,
        role="audit",
        instruction="price me",
        status=TaskStatus.COMPLETE,
    )
    store.save_task(task)
    store.save_artifacts([_usage(job.id, task.id)])
    _stamp(store, job.id, JobStatus.COMPLETE, registry_path)
    return job, task


def _cost_args(job_id: str, registry_path: Path, *, as_json: bool = False):
    class Args:
        pass

    Args.json = as_json
    Args.job_id = job_id
    Args.registry_path = str(registry_path)
    return Args()


def _run_cost_text(store, job_id: str, registry_path: Path, *, as_json: bool = False):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = _run_cost_command(_cost_args(job_id, registry_path, as_json=as_json), store)
    return rc, out.getvalue()


def _job_payload(**overrides):
    data = {
        "id": "job_1",
        "goal": "g",
        "status": "queued",
        "created_at": "2026-08-30T00:00:00+00:00",
    }
    data.update(overrides)
    return data


def _complete_legacy(store, job_id: str):
    store.save_job(replace(store.get_job(job_id), status=JobStatus.COMPLETE))


def _assert_source_total(test, report, source: str, total):
    test.assertEqual(report["pricing_source"], source)
    if total is None:
        test.assertIsNone(report["actual_cost"]["total_marginal_cost_usd"])
        return
    test.assertAlmostEqual(
        report["actual_cost"]["total_marginal_cost_usd"], total, places=6
    )


class BuildCostReportTests(unittest.TestCase):
    def test_build_cost_report_matches_mari_signature(self) -> None:
        params = inspect.signature(build_cost_report).parameters
        self.assertEqual(list(params)[:2], ["store", "job_id"])
        self.assertIn("registry", params)

    def test_cli_json_matches_build_cost_report(self) -> None:
        with _harness() as (store, registry_path):
            job = _job_with_usage(store, "cost extract")
            report = build_cost_report(store, job.id, registry=_registry())
            self.assertEqual(report["job_id"], job.id)
            self.assertIn("actual_cost", report)
            self.assertIn("counterfactual", report)
            self.assertIn("estimate_drift", report)
            self.assertAlmostEqual(
                report["actual_cost"]["total_marginal_cost_usd"], 18.0, places=6
            )
            with _models_path(registry_path):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli_main(
                        [
                            "--state-dir",
                            str(registry_path.parent / ".puppetmaster-file"),
                            "--backend",
                            "file",
                            "cost",
                            job.id,
                            "--json",
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_cli_and_mcp_call_build_cost_report(self) -> None:
        with _harness() as (store, registry_path):
            job = _job_with_usage(store, "cost wiring")
            sentinel = {
                "job_id": job.id,
                "cost_basis": "preflight_routing_estimate",
                "total_estimated_cost_usd": 0.0,
                "actual_cost": {"total_marginal_cost_usd": 18.0},
            }
            with patch(
                "puppetmaster.cost.build_cost_report", return_value=sentinel
            ) as mocked:
                rc, text = _run_cost_text(
                    store, job.id, registry_path, as_json=True
                )
                self.assertEqual(rc, 0)
                mocked.assert_called_once()
                self.assertEqual(json.loads(text), sentinel)

            with patch(
                "puppetmaster.cost.build_cost_report", return_value=sentinel
            ) as mocked:
                result = mcp_server.run_job_cost(
                    {
                        "job_id": job.id,
                        "state_dir": str(registry_path.parent / ".puppetmaster-file"),
                        "backend": "file",
                        "registry_path": str(registry_path),
                        "cwd": str(registry_path.parent),
                    }
                )
                mocked.assert_called_once()
                self.assertFalse(result.get("isError"))
                body = json.loads(result["content"][0]["text"])
                self.assertEqual(json.loads(body["stdout"]), sentinel)


class JobCostReceiptRoundTripTests(unittest.TestCase):
    def test_job_from_dict_cost_receipt(self) -> None:
        job = job_from_dict(
            _job_payload(
                status="complete",
                cost_receipt={"job_id": "job_1", "pricing_source": "terminal_receipt"},
            )
        )
        self.assertEqual(job.status, JobStatus.COMPLETE)
        self.assertEqual(job.cost_receipt["pricing_source"], "terminal_receipt")
        self.assertEqual(job_from_dict(to_jsonable(job)).cost_receipt, job.cost_receipt)
        self.assertIsNone(job_from_dict(_job_payload()).cost_receipt)
        self.assertIsNone(
            job_from_dict(_job_payload(id="job_2", cost_receipt="not-a-dict")).cost_receipt
        )


class TerminalReceiptLifecycleTests(unittest.TestCase):
    def _assert_stable_across_registry_churn(self, backend: str) -> None:
        with _harness(backend) as (store, registry_path):
            job, _task = _seed_priced_job(store, registry_path, _registry())
            first = build_cost_report(store, job.id, registry=_registry())
            _assert_source_total(self, first, "terminal_receipt", 18.0)
            added = list(_registry()) + [_spec("new-model", "new-v1", 50, 1.0, 1.0)]
            _assert_immune_to_registry_churn(
                self,
                store,
                job.id,
                registry_path,
                first,
                (added, _cheaper_mid(), _flagship_only()),
            )
            self.assertEqual(store.get_job(job.id).status, JobStatus.COMPLETE)

    def test_terminal_receipt_stable_across_registry_churn(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                self._assert_stable_across_registry_churn(backend)

    def test_running_and_stalled_reprice_without_stamping(self) -> None:
        cases = (
            ("still running", None, False),
            ("recoverable stall", JobStatus.STALLED, True),
        )
        for backend in _BACKENDS:
            for goal, status, check_receipt in cases:
                with self.subTest(backend=backend, goal=goal):
                    with _harness(backend) as (store, registry_path):
                        job = _job_with_usage(store, goal)
                        if status is not None:
                            _stamp(store, job.id, status, registry_path)
                            if check_receipt:
                                self.assertIsNone(store.get_job(job.id).cost_receipt)
                        live = build_cost_report(store, job.id, registry=_registry())
                        _assert_source_total(self, live, "current_registry", 18.0)
                        ghost = build_cost_report(store, job.id, registry=_flagship_only())
                        _assert_source_total(self, ghost, "current_registry", None)
                        self.assertEqual(ghost["actual_cost"]["unpriced_tasks"], 1)

    def test_failed_and_cancelled_stamp_once(self) -> None:
        for backend in _BACKENDS:
            for status in _COST_FINAL_NON_COMPLETE:
                with self.subTest(backend=backend, status=str(status)):
                    with _harness(backend) as (store, registry_path):
                        job = _job_with_usage(store, f"cost-final {status}")
                        _stamp(store, job.id, status, registry_path)
                        report = build_cost_report(store, job.id, registry=_flagship_only())
                        _assert_source_total(self, report, "terminal_receipt", 18.0)

    def test_recoverable_transitions_clear_receipt(self) -> None:
        for backend in _BACKENDS:
            for status in _RECOVERABLE_STATUSES:
                with self.subTest(backend=backend, status=str(status)):
                    with _harness(backend) as (store, registry_path):
                        job, _task = _seed_priced_job(store, registry_path, _registry())
                        self.assertIsNotNone(store.get_job(job.id).cost_receipt)
                        store.update_job_status(job.id, status)
                        loaded = store.get_job(job.id)
                        self.assertEqual(loaded.status, status)
                        self.assertIsNone(loaded.cost_receipt)
                        report = build_cost_report(store, job.id, registry=_registry())
                        _assert_source_total(self, report, "current_registry", 18.0)

    def test_complete_without_usage_does_not_stamp_zero(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                with _harness(backend) as (store, registry_path):
                    job = store.create_job("flush later")
                    completed = _stamp(store, job.id, JobStatus.COMPLETE, registry_path)
                    self.assertEqual(completed.status, JobStatus.COMPLETE)
                    self.assertIsNone(completed.cost_receipt)
                    store.save_artifacts(
                        [_usage(job.id, model="ghost-model", tokens_in=500, tokens_out=500, real_cost_usd=0.42)]
                    )
                    report = build_cost_report(store, job.id, registry=_registry())
                    _assert_source_total(self, report, "legacy_artifacts", 0.42)
                    self.assertIsNone(store.get_job(job.id).cost_receipt)
                    again = _stamp(store, job.id, JobStatus.COMPLETE, registry_path)
                    self.assertTrue(valid_terminal_cost_receipt(again.cost_receipt, job.id))
                    frozen = build_cost_report(store, job.id, registry=_flagship_only())
                    _assert_source_total(self, frozen, "terminal_receipt", 0.42)

    def test_repeated_final_transition_is_first_writer_wins(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                with _harness(backend) as (store, registry_path):
                    job, _task = _seed_priced_job(store, registry_path, _registry())
                    first = build_cost_report(store, job.id)
                    save_registry(_cheaper_mid(), registry_path)
                    _stamp(store, job.id, JobStatus.COMPLETE, registry_path)
                    self.assertEqual(
                        build_cost_report(store, job.id, registry=_cheaper_mid()), first
                    )

    def test_worker_completion_does_not_stamp_receipt(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                with _harness(backend) as (store, registry_path):
                    job = _job_with_usage(store, "worker cannot complete")
                    with _models_path(registry_path):
                        refused = store.update_job_status(
                            job.id, JobStatus.COMPLETE, actor="worker-1"
                        )
                    self.assertEqual(refused.status, JobStatus.QUEUED)
                    self.assertIsNone(store.get_job(job.id).cost_receipt)

    def test_stamp_failure_does_not_block_completion(self) -> None:
        with _harness("sqlite") as (store, _registry_path):
            job = store.create_job("stamp boom")
            with patch(
                "puppetmaster.cost.build_current_registry_cost_report",
                side_effect=RuntimeError("registry exploded"),
            ):
                completed = store.update_job_status(job.id, JobStatus.COMPLETE)
            self.assertEqual(completed.status, JobStatus.COMPLETE)
            self.assertIsNone(completed.cost_receipt)

    def _assert_reset_reopens_job(self, store, job_id: str) -> None:
        reopened = store.get_job(job_id)
        self.assertEqual(reopened.status, JobStatus.RUNNING)
        self.assertIsNone(reopened.completed_at)
        self.assertIsNone(reopened.cost_receipt)

    def _reset_clears_and_refreezes(self, backend: str) -> None:
        with _harness(backend) as (store, registry_path):
            job, task = _seed_priced_job(store, registry_path, _registry())
            first = build_cost_report(store, job.id)
            self.assertEqual(first["pricing_source"], "terminal_receipt")
            store.reset_subgraph(job.id, [task.id], include_descendants=False)
            self._assert_reset_reopens_job(store, job.id)
            after_reset = build_cost_report(store, job.id, registry=_flagship_only())
            self.assertEqual(after_reset["pricing_source"], "current_registry")
            self.assertEqual(after_reset["token_usage"]["total_tokens"], 0)

            store.save_artifacts([_usage(job.id, task.id, tokens_in=1_000, tokens_out=1_000)])
            save_registry(_cheaper_mid(), registry_path)
            self.assertEqual(store.get_job(job.id).status, JobStatus.RUNNING)
            self.assertEqual(_finalize_production(store, job.id, registry_path), 0)
            self.assertEqual(store.get_job(job.id).status, JobStatus.COMPLETE)
            refrozen = build_cost_report(store, job.id, registry=_registry())
            self.assertEqual(refrozen["pricing_source"], "terminal_receipt")
            self.assertEqual(refrozen["actual_cost"]["tasks"][0]["tokens_in"], 1_000)
            self.assertEqual(refrozen["actual_cost"]["tasks"][0]["tokens_out"], 1_000)
            self.assertAlmostEqual(
                refrozen["actual_cost"]["total_marginal_cost_usd"], 0.00002, places=6
            )
            self.assertNotEqual(refrozen["actual_cost"]["total_marginal_cost_usd"], 18.0)

    def test_reset_clears_and_refreezes(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                self._reset_clears_and_refreezes(backend)

    def test_reset_supersedes_plan_route_and_refreezes_new_generation(self) -> None:
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                with _harness(backend, registry=_plan_registry()) as (
                    store,
                    registry_path,
                ):
                    job = store.create_job("plan then metered")
                    task = Task(
                        job_id=job.id,
                        role="audit",
                        instruction="price me",
                        status=TaskStatus.COMPLETE,
                    )
                    store.save_task(task)
                    store.save_artifacts(
                        [
                            _routing(
                                job.id,
                                task.id,
                                "plan/cursor",
                                billing="plan",
                                estimated_cost_usd=1.5,
                            ),
                            _usage(job.id, task.id, model="plan/cursor"),
                        ]
                    )
                    _stamp(store, job.id, JobStatus.COMPLETE, registry_path)
                    first = build_cost_report(store, job.id, registry=_plan_registry())
                    _assert_source_total(self, first, "terminal_receipt", 0.0)
                    self.assertEqual(first["actual_cost"]["tasks"][0]["billing"], "plan")

                    store.reset_subgraph(job.id, [task.id], include_descendants=False)
                    self._assert_reset_reopens_job(store, job.id)
                    routes = [
                        artifact
                        for artifact in store.list_artifacts(job.id)
                        if artifact.type == ArtifactType.ROUTING
                    ]
                    self.assertTrue(routes)
                    self.assertTrue(
                        all(
                            validation_status_of(artifact) == "superseded"
                            for artifact in routes
                        )
                    )
                    after_reset = build_cost_report(
                        store, job.id, registry=_plan_registry()
                    )
                    self.assertEqual(after_reset["pricing_source"], "current_registry")
                    self.assertEqual(after_reset["token_usage"]["total_tokens"], 0)
                    self.assertEqual(after_reset["total_estimated_cost_usd"], 0.0)
                    self.assertEqual(final_routing_artifacts(store.list_artifacts(job.id)), {})

                    store.save_artifacts(
                        [
                            _routing(
                                job.id,
                                task.id,
                                "flagship-model",
                                billing="api",
                                estimated_cost_usd=9.0,
                                validation={"status": "fresh", "generation": 1},
                            ),
                            _usage(
                                job.id,
                                task.id,
                                model="flagship-v1",
                                tokens_in=1_000,
                                tokens_out=1_000,
                            ),
                        ]
                    )
                    self.assertEqual(store.get_job(job.id).status, JobStatus.RUNNING)
                    self.assertEqual(_finalize_production(store, job.id, registry_path), 0)
                    frozen = build_cost_report(store, job.id, registry=_cheaper_mid())
                    self.assertEqual(frozen["pricing_source"], "terminal_receipt")
                    priced = frozen["actual_cost"]["tasks"][0]
                    self.assertEqual(priced["tokens_in"], 1_000)
                    self.assertEqual(priced["tokens_out"], 1_000)
                    self.assertEqual(priced["model_id"], "flagship-model")
                    self.assertNotEqual(priced["billing"], "plan")
                    self.assertAlmostEqual(
                        frozen["actual_cost"]["total_marginal_cost_usd"], 0.09, places=6
                    )
                    self.assertAlmostEqual(frozen["total_estimated_cost_usd"], 9.0, places=6)

    def test_sqlite_receipt_clear_rolls_back_with_reset(self) -> None:
        with _harness("sqlite") as (store, registry_path):
            job, task = _seed_priced_job(store, registry_path, _registry())
            self.assertIsNotNone(store.get_job(job.id).cost_receipt)
            real_session = store._session

            @contextlib.contextmanager
            def boom_after_writes():
                with real_session() as connection:
                    yield connection
                    raise sqlite3.OperationalError("simulated busy/crash")

            with patch.object(store, "_session", boom_after_writes):
                with self.assertRaises(sqlite3.OperationalError):
                    store.reset_subgraph(job.id, [task.id], include_descendants=False)
            self.assertIsNotNone(store.get_job(job.id).cost_receipt)
            self.assertEqual(store.get_job(job.id).status, JobStatus.COMPLETE)


class UnpricedAndIdentityTests(unittest.TestCase):
    def test_selected_cost_identity(self) -> None:
        mixed = [
            _usage("job_x", "priced", tokens_out=0),
            _usage("job_x", "ghost", model="ghost-v1", tokens_in=500_000, tokens_out=0),
        ]
        plan = [_usage("job_x", model="plan/cursor")]
        reported = [
            _usage(
                "job_x",
                model="ghost-model",
                tokens_in=500,
                tokens_out=500,
                real_cost_usd=0.05,
            )
        ]
        with self.subTest(case="mixed_unpriced"):
            report = build_current_registry_cost_report("job_x", mixed, _registry())
            actual = report["actual_cost"]
            self.assertEqual(actual["priced_tasks"], 1)
            self.assertEqual(actual["unpriced_tasks"], 1)
            self.assertIsNone(actual["total_marginal_cost_usd"])
            self.assertIsNone(actual["measured_cost_usd"])
            self.assertAlmostEqual(actual["priced_subtotal_usd"], 3.0, places=6)
            self.assertAlmostEqual(
                actual["by_model"]["mid-model"]["marginal_cost_usd"], 3.0, places=6
            )
            self.assertIsNone(actual["by_model"]["ghost-v1"]["marginal_cost_usd"])
            ghost_task = next(t for t in actual["tasks"] if t["task_id"] == "ghost")
            self.assertFalse(ghost_task["priced"])
            self.assertIsNone(ghost_task["marginal_cost_usd"])
            cf = report["counterfactual"]
            self.assertFalse(cf["actual_priced"])
            self.assertIsNone(cf["actual_cost_usd"])
            self.assertIsNone(cf["avoided_usd"])

        with self.subTest(case="plan_zero"):
            report = build_current_registry_cost_report("job_x", plan, _plan_registry())
            actual = report["actual_cost"]
            self.assertEqual(actual["priced_tasks"], 1)
            self.assertEqual(actual["unpriced_tasks"], 0)
            self.assertEqual(actual["total_marginal_cost_usd"], 0.0)
            self.assertEqual(actual["priced_subtotal_usd"], 0.0)
            self.assertTrue(actual["tasks"][0]["priced"])
            cf = report["counterfactual"]
            self.assertTrue(cf["actual_priced"])
            self.assertEqual(cf["actual_cost_usd"], 0.0)
            self.assertAlmostEqual(cf["avoided_usd"], 90.0, places=6)

        with self.subTest(case="provider_reported"):
            report = build_current_registry_cost_report("job_x", reported, _registry())
            actual = report["actual_cost"]
            self.assertEqual(actual["unpriced_tasks"], 0)
            self.assertAlmostEqual(actual["total_marginal_cost_usd"], 0.05, places=6)
            self.assertEqual(actual["tasks"][0]["billing"], "reported")

    def test_final_route_absent_id_does_not_fall_through(self) -> None:
        artifacts = [
            _routing("job_x", "t1", "missing-routed-id"),
            _usage("job_x"),
        ]
        cost = price_job(artifacts, _registry())
        self.assertEqual(cost.unpriced_tasks, 1)
        self.assertEqual(cost.priced_tasks, 0)
        self.assertNotIn("mid-model", cost.by_model)
        self.assertIn("missing-routed-id", cost.by_model)

    def test_select_usage_excludes_stale_and_superseded(self) -> None:
        artifacts = [
            _usage("job_x", validation={"status": "superseded", "generation": 0}),
            _usage("job_x", tokens_in=10, tokens_out=10, validation={"status": "stale"}),
            _usage(
                "job_x",
                tokens_in=1_000,
                tokens_out=1_000,
                validation={"status": "fresh", "generation": 1},
            ),
        ]
        selected = select_usage_records(artifacts)
        self.assertEqual(selected["t1"]["tokens_in"], 1_000)
        self.assertEqual(selected["t1"]["tokens_out"], 1_000)
        roll = aggregate_token_usage(artifacts)
        self.assertEqual(roll["measured_tokens_in"], 1_000)
        self.assertEqual(roll["total_tokens"], 2_000)
        cost = price_job(artifacts, _registry())
        self.assertEqual(len(cost.tasks), 1)
        self.assertEqual(cost.tasks[0].tokens_in, 1_000)
        self.assertAlmostEqual(cost.total_marginal_cost_usd, 0.018, places=6)

    def test_withdrawn_routing_is_excluded_before_selection(self) -> None:
        artifacts = [
            _routing(
                "job_x",
                "t1",
                "plan/cursor",
                billing="plan",
                created_by="router-escalation",
                estimated_cost_usd=5.0,
                validation={"status": "superseded", "generation": 0},
            ),
            _routing(
                "job_x",
                "t1",
                "mid-model",
                estimated_cost_usd=2.0,
                validation={"status": "stale"},
            ),
            _routing(
                "job_x",
                "t1",
                "flagship-model",
                billing="api",
                estimated_cost_usd=1.0,
                validation={"status": "fresh", "generation": 1},
            ),
            _usage("job_x", model="flagship-v1", tokens_in=1_000, tokens_out=1_000),
        ]
        finals = final_routing_artifacts(artifacts)
        self.assertEqual(list(finals), ["t1"])
        self.assertEqual((finals["t1"].payload or {}).get("model_id"), "flagship-model")
        rows, _by_model, total = routing_estimate_rows(artifacts)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_id"], "flagship-model")
        self.assertAlmostEqual(total, 1.0, places=6)
        cost = price_job(artifacts, _registry())
        self.assertEqual(cost.priced_tasks, 1)
        self.assertEqual(cost.tasks[0].model_id, "flagship-model")
        self.assertAlmostEqual(cost.total_marginal_cost_usd, 0.09, places=6)
        unlabeled = [
            _routing("job_x", "t2", "mid-model", estimated_cost_usd=3.0),
            _usage("job_x", "t2"),
        ]
        self.assertIn("t2", final_routing_artifacts(unlabeled))
        _rows, _models, unlabeled_total = routing_estimate_rows(unlabeled)
        self.assertAlmostEqual(unlabeled_total, 3.0, places=6)


class LegacyArtifactCostTests(unittest.TestCase):
    def test_legacy_completed_job_stable_across_registry_churn(self) -> None:
        with _harness(registry=_flagship_only()) as (store, registry_path):
            job = _job_with_usage(store, "pre-upgrade complete")
            _complete_legacy(store, job.id)
            self.assertIsNone(store.get_job(job.id).cost_receipt)

            first = build_cost_report(store, job.id, registry=_flagship_only())
            _assert_source_total(self, first, "legacy_artifacts", None)
            self.assertIsNone(first["counterfactual"])
            self.assertEqual(first["actual_cost"]["unpriced_tasks"], 1)
            self.assertIsNone(store.get_job(job.id).cost_receipt)
            _assert_immune_to_registry_churn(
                self,
                store,
                job.id,
                registry_path,
                first,
                (_registry(), _cheaper_mid(), _flagship_only()),
            )
            self.assertIsNone(store.get_job(job.id).cost_receipt)
            self.assertIsNone(
                build_cost_report(store, job.id, registry=_registry())[
                    "actual_cost"
                ]["total_marginal_cost_usd"]
            )

    def test_legacy_known_artifact_prices(self) -> None:
        cases = (
            (
                "legacy plan",
                lambda job_id: [
                    _routing(job_id, "t1", "plan/cursor", billing="plan"),
                    _usage(job_id, model="plan/cursor"),
                ],
                [],
                0.0,
                "plan",
            ),
            (
                "legacy reported",
                lambda job_id: [
                    _usage(
                        job_id,
                        model="ghost-model",
                        tokens_in=500,
                        tokens_out=500,
                        real_cost_usd=0.05,
                    )
                ],
                _cheaper_mid(),
                0.05,
                "reported",
            ),
        )
        for goal, artifacts_for, registry, total, billing in cases:
            with self.subTest(goal=goal):
                with _harness() as (store, _registry_path):
                    job = store.create_job(goal)
                    store.save_artifacts(artifacts_for(job.id))
                    _complete_legacy(store, job.id)
                    report = build_cost_report(store, job.id, registry=registry)
                    _assert_source_total(self, report, "legacy_artifacts", total)
                    self.assertEqual(report["actual_cost"]["tasks"][0]["billing"], billing)
                    if billing == "plan":
                        self.assertTrue(report["actual_cost"]["tasks"][0]["priced"])
                    self.assertIsNone(report["counterfactual"])


class ReceiptValidationTests(unittest.TestCase):
    def test_malformed_receipt_falls_back_to_legacy(self) -> None:
        cases = (
            {},
            {"job_id": "job_1", "pricing_source": "terminal_receipt"},
            {
                "job_id": "other",
                "pricing_source": "terminal_receipt",
                "actual_cost": {},
                "token_usage": {},
                "tasks": [],
            },
            {
                "job_id": "job_1",
                "pricing_source": "current_registry",
                "actual_cost": {},
                "token_usage": {},
                "tasks": [],
            },
        )
        for blob in cases:
            with self.subTest(blob=blob):
                with _harness() as (store, _registry_path):
                    job = _job_with_usage(
                        store,
                        "bad receipt",
                        model="ghost-model",
                        tokens_in=10,
                        tokens_out=0,
                        real_cost_usd=0.01,
                    )
                    _write_receipt(
                        store,
                        job.id,
                        JobStatus.COMPLETE,
                        dict(blob, job_id=blob.get("job_id", job.id)),
                    )
                    self.assertIsInstance(store.get_job(job.id).cost_receipt, dict)
                    report = build_cost_report(store, job.id, registry=_registry())
                    _assert_source_total(self, report, "legacy_artifacts", 0.01)

    def test_valid_receipt_shape_is_additive(self) -> None:
        good = {
            "job_id": "job_1",
            "pricing_source": "terminal_receipt",
            "actual_cost": {},
            "token_usage": {},
            "tasks": [],
        }
        self.assertTrue(valid_terminal_cost_receipt(good, "job_1"))
        self.assertFalse(valid_terminal_cost_receipt(good, "job_2"))
        self.assertFalse(valid_terminal_cost_receipt({}, "job_1"))
        self.assertFalse(valid_terminal_cost_receipt(None, "job_1"))

    def test_stalled_leftover_receipt_is_not_returned(self) -> None:
        with _harness() as (store, _registry_path):
            job = _job_with_usage(store, "stalled leftover")
            leftover = {
                "job_id": job.id,
                "pricing_source": "terminal_receipt",
                "actual_cost": {"total_marginal_cost_usd": 18.0},
                "token_usage": {},
                "tasks": [],
            }
            _write_receipt(store, job.id, JobStatus.STALLED, leftover)
            report = build_cost_report(store, job.id, registry=_flagship_only())
            _assert_source_total(self, report, "current_registry", None)


class CostCliHumanTests(unittest.TestCase):
    def test_human_output_unknown_versus_plan_zero(self) -> None:
        with _harness(registry=_flagship_only()) as (store, registry_path):
            job = _job_with_usage(store, "unpriced cli", tokens_out=0)
            rc, text = _run_cost_text(store, job.id, registry_path)
            self.assertEqual(rc, 0)
            self.assertIn("cost unavailable", text)
            self.assertNotIn("$0.000000", text)
            self.assertNotIn("avoided $", text)
            self.assertNotIn("→ avoided", text)

        with _harness(registry=_plan_registry()) as (store, registry_path):
            job = _job_with_usage(store, "plan zero cli", model="plan/cursor")
            rc, text = _run_cost_text(store, job.id, registry_path)
            self.assertEqual(rc, 0)
            self.assertIn("actual measured spend = $0.000000", text)
            self.assertNotIn("cost unavailable", text)
            self.assertIn("avoided $", text)

            mixed = build_current_registry_cost_report(
                job.id,
                [
                    _usage(job.id, "priced", tokens_out=0),
                    _usage(job.id, "ghost", model="ghost-v1", tokens_in=10, tokens_out=0),
                ],
                _registry(),
            )
            store.save_job(replace(store.get_job(job.id), cost_receipt=None))
            with patch("puppetmaster.cost.build_cost_report", return_value=mixed):
                rc, mixed_text = _run_cost_text(store, job.id, registry_path)
            self.assertEqual(rc, 0)
            self.assertIn("cost unavailable", mixed_text)
            self.assertIn("unavailable", mixed_text)
            self.assertNotIn("→ avoided", mixed_text)
            mid_at = mixed_text.find("mid-model")
            ghost_at = mixed_text.find("ghost-v1")
            self.assertNotEqual(mid_at, -1)
            self.assertNotEqual(ghost_at, -1)
            self.assertLess(mid_at, ghost_at)

    def test_human_output_priced_subtotal_zero_with_unpriced(self) -> None:
        mixed = build_current_registry_cost_report(
            "job_x",
            [
                _usage("job_x", "plan", model="plan/cursor"),
                _usage("job_x", "ghost", model="ghost-v1", tokens_in=10, tokens_out=0),
            ],
            _plan_registry(),
        )
        actual = mixed["actual_cost"]
        self.assertEqual(actual["priced_tasks"], 1)
        self.assertEqual(actual["unpriced_tasks"], 1)
        self.assertEqual(actual["priced_subtotal_usd"], 0.0)
        self.assertIsNone(actual["total_marginal_cost_usd"])
        self.assertIsNone(actual["by_model"]["ghost-v1"]["marginal_cost_usd"])
        self.assertEqual(actual["by_model"]["plan/cursor"]["marginal_cost_usd"], 0.0)

        with _harness(registry=_plan_registry()) as (store, registry_path):
            job = store.create_job("mixed plan zero")
            with patch("puppetmaster.cost.build_cost_report", return_value=mixed):
                rc, text = _run_cost_text(store, job.id, registry_path)
            self.assertEqual(rc, 0)
            self.assertIn("cost unavailable", text)
            self.assertIn(
                "priced subtotal (excludes unpriced tasks): $0.000000", text
            )
            self.assertNotIn("→ avoided", text)


if __name__ == "__main__":
    unittest.main()
