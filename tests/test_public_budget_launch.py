"""Public launches must not create jobs that cannot pass first admission."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from puppetmaster.budget import BudgetAdmissionError, BudgetPolicy
from puppetmaster.cli._dispatch import _main
from puppetmaster.invocation import execution_scope, invocation
from puppetmaster.models import Task
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.workers import WorkerSpec

import unittest


def _spec(adapter, **payload):
    return WorkerSpec(role=adapter, instruction="test", adapter=adapter, payload=payload)


def _stop(job):
    raise InterruptedError(job.id)


def _launch_until_tasks(store, adapter, policy, **payload):
    """Create the job and its tasks, then stop before workers."""
    original = Orchestrator._create_tasks
    box = {}

    def create_then_stop(self, job, specs):
        tasks = original(self, job, specs)
        box["job_id"] = job.id
        box["payload"] = dict(tasks[0].payload)
        raise InterruptedError(job.id)

    with patch.object(Orchestrator, "_create_tasks", create_then_stop):
        try:
            Orchestrator(store).run(
                "test", specs=[_spec(adapter, **payload)], budget_policy=policy)
        except InterruptedError:
            pass
        else:
            raise AssertionError("launch did not stop after creating tasks")
    return box["job_id"], box["payload"]


def _first_admit(store, job_id, adapter, payload):
    task = Task(job_id=job_id, role=adapter, instruction="test",
                adapter=adapter, payload=dict(payload))
    store.save_task(task)
    with execution_scope(store, SimpleNamespace(id="run"), task):
        with invocation():
            return store.budget_snapshot(job_id)


class PublicBudgetLaunchTests(unittest.TestCase):
    def test_token_cap_fails_before_job_creation(self):
        for store_type in (SwarmStore, SQLiteSwarmStore):
            for adapter in ("agentic", "codex"):
                with self.subTest(store=store_type.__name__, adapter=adapter), \
                        TemporaryDirectory() as root:
                    store = store_type(Path(root))
                    with self.assertRaisesRegex(
                            ValueError,
                            r"budget_max_tokens_in.*%s.*allowance" % adapter):
                        Orchestrator(store).run(
                            "test",
                            specs=[_spec(adapter, timeout_seconds=900)],
                            budget_policy=BudgetPolicy(max_tokens_in=60000),
                            on_job_created=_stop)
                    self.assertEqual(store.list_jobs(), [])

    def test_cli_token_and_usd_caps_fail_closed(self):
        original = Orchestrator.run

        def stop_after_create(instance, *args, **kwargs):
            kwargs["on_job_created"] = _stop
            return original(instance, *args, **kwargs)

        for verb, extra in (
                ("agentic", ["--mode", "analyze", "--budget-max-tokens-in", "60000"]),
                ("codex", ["--budget-max-tokens-in", "60000"]),
                ("agentic", ["--mode", "analyze", "--budget-max-usd", "0.25"]),
                ("codex", ["--budget-max-usd", "0.25"])):
            with self.subTest(verb=verb, extra=extra[-2:]), TemporaryDirectory() as root, \
                    patch.object(Orchestrator, "run", autospec=True,
                                 side_effect=stop_after_create):
                with self.assertRaisesRegex(ValueError, r"budget_max_"):
                    _main(["--state-dir", root, verb, "test", *extra])
                self.assertEqual(SwarmStore(Path(root)).list_jobs(), [])

    def test_elapsed_cap_reaches_first_invocation(self):
        for store_type in (SwarmStore, SQLiteSwarmStore):
            for adapter in ("agentic", "codex"):
                with self.subTest(store=store_type.__name__, adapter=adapter), \
                        TemporaryDirectory() as root:
                    store = store_type(Path(root))
                    policy = BudgetPolicy(max_elapsed_seconds=300)
                    job_id, payload = _launch_until_tasks(
                        store, adapter, policy, timeout_seconds=900)
                    self.assertEqual(store.get_job(job_id).budget_policy, policy)
                    self.assertEqual(payload["budget_allowance"]["elapsed_seconds"], 300)
                    snap = _first_admit(store, job_id, adapter, payload)
                    self.assertEqual(len(snap["reservations"]), 1)
                    self.assertEqual(
                        snap["reservations"][0]["allowance"]["elapsed_seconds"], 300)

    def test_attempts_only_still_admits_without_allowance(self):
        with TemporaryDirectory() as root:
            store = SwarmStore(Path(root))
            job_id, payload = _launch_until_tasks(
                store, "agentic", BudgetPolicy(max_attempts=2), timeout_seconds=900)
            self.assertNotIn("budget_allowance", payload)
            _first_admit(store, job_id, "agentic", payload)

    def test_explicit_token_allowance_reaches_first_invocation(self):
        with TemporaryDirectory() as root:
            store = SwarmStore(Path(root))
            job_id, payload = _launch_until_tasks(
                store, "agentic", BudgetPolicy(max_tokens_in=60000),
                timeout_seconds=900, budget_allowance={"tokens_in": 1000})
            self.assertEqual(payload["budget_allowance"]["tokens_in"], 1000)
            snap = _first_admit(store, job_id, "agentic", payload)
            self.assertEqual(snap["reservations"][0]["allowance"]["tokens_in"], 1000)

    def test_pending_reconciliation_still_blocks(self):
        with TemporaryDirectory() as root:
            store = SwarmStore(Path(root))
            policy = BudgetPolicy(max_elapsed_seconds=300, max_attempts=2)
            job_id, payload = _launch_until_tasks(
                store, "codex", policy, timeout_seconds=120)
            _first_admit(store, job_id, "codex", payload)
            self.assertEqual(
                store.budget_snapshot(job_id)["reservations"][0]["state"],
                "pending_reconciliation")
            task = Task(job_id=job_id, role="codex", instruction="retry",
                        adapter="codex", payload=payload)
            store.save_task(task)
            # The second attempt waits for the first to settle; it never does
            # here, so the wait ends and admission still fails closed.
            with self.assertRaisesRegex(BudgetAdmissionError, "waited"), \
                    patch("puppetmaster.invocation.DEFAULT_BUDGET_WAIT_SECONDS", 0.2), \
                    patch("puppetmaster.invocation.BUDGET_WAIT_POLL_SECONDS", 0.05), \
                    execution_scope(store, SimpleNamespace(id="run-2"), task):
                with invocation():
                    self.fail("pending unknown liability admitted a second invocation")

    def test_cli_elapsed_reaches_orchestrator(self):
        captured = {}

        def capture(_instance, *_args, **kwargs):
            captured["policy"] = kwargs["budget_policy"]
            captured["specs"] = kwargs["specs"]
            raise InterruptedError("captured")

        for verb in ("agentic", "codex"):
            command = ["--state-dir", "unused", verb, "test",
                       "--budget-max-elapsed-seconds", "300"]
            if verb == "agentic":
                command = ["--state-dir", "unused", verb, "test", "--mode", "analyze",
                           "--budget-max-elapsed-seconds", "300"]
            with self.subTest(verb=verb), TemporaryDirectory() as root, \
                    patch.object(Orchestrator, "run", autospec=True, side_effect=capture):
                command[1] = root
                with self.assertRaises(InterruptedError):
                    _main(command)
                self.assertEqual(captured["policy"].max_elapsed_seconds, 300)
                self.assertEqual(captured["specs"][0].payload["timeout_seconds"], 900)


if __name__ == "__main__":
    unittest.main()
