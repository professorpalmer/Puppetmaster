"""Under a job cap, a worker waits for unsettled siblings instead of failing.

Codex benchmark: four parallel 180 s workers under a 240 s elapsed cap. Three
failed admission at once because the first had not reported yet. Option 2
(chosen 2026-10-08): an attempt that is blocked only by unsettled earlier
attempts waits and tries again. An exhausted cap, or an unknown that can never
settle, still fails at once.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import threading
from dataclasses import replace
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from puppetmaster.budget import (BudgetAdmissionError, BudgetLiability, BudgetPolicy,
                                 BudgetUnsettled, check_admission)
from puppetmaster.cancellation import JobCancelled
from puppetmaster.invocation import execution_scope, invocation
from puppetmaster.models import JobStatus, Task
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore

from tests.test_public_budget_launch import _launch_until_tasks


def _record(state, elapsed):
    liability = {"billing": "plan", "plan_marginal_usd": 0, "cost_state": "known",
                 "elapsed_seconds": elapsed}
    return {"state": state, "liability": None, "allowance": liability}


class AdmissionKindsTests(unittest.TestCase):
    def test_only_an_unsettled_earlier_attempt_can_wait(self):
        policy = BudgetPolicy(max_elapsed_seconds=720)
        with self.assertRaises(BudgetUnsettled):
            check_admission(policy, [_record("pending_reconciliation", 180), _record("reserved", 180)])
        # The new attempt has no bounded allowance: no wait can help.
        with self.assertRaises(BudgetAdmissionError) as unbounded:
            check_admission(policy, [_record("reserved", None)])
        self.assertNotIsInstance(unbounded.exception, BudgetUnsettled)
        self.assertIn("can never be known", str(unbounded.exception))
        # Exhausted stays exhausted.
        with self.assertRaises(BudgetAdmissionError) as exhausted:
            check_admission(policy, [_record("settled", 600), _record("reserved", 180)])
        self.assertNotIsInstance(exhausted.exception, BudgetUnsettled)
        self.assertIn("exhausted", str(exhausted.exception))


class WaitForSiblingsTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("puppetmaster.invocation.BUDGET_WAIT_POLL_SECONDS", 0.05)
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, store):
        job_id, payload = _launch_until_tasks(store, "codex", BudgetPolicy(max_elapsed_seconds=720),
                                              timeout_seconds=180)
        # The helper stops the launch, which ends the job; a live job runs.
        store.save_job(replace(store.get_job(job_id), status=JobStatus.RUNNING))
        return job_id, payload

    def claimed(self, store, job_id, payload, name):
        task = Task(job_id=job_id, role="codex", instruction=name, adapter="codex", payload=dict(payload))
        store.save_task(task)
        return store.claim_task(task.id, f"worker-{name}", lease_seconds=60)

    def start_first(self, store, job_id, payload):
        first = self.claimed(store, job_id, payload, "first")
        with execution_scope(store, SimpleNamespace(id="run-1"), first):
            with invocation():
                pass
        reservation = store.budget_snapshot(job_id)["reservations"][0]
        self.assertEqual(reservation["state"], "pending_reconciliation")
        return reservation

    def second_in_thread(self, store, task, outcome):
        def run():
            try:
                with execution_scope(store, SimpleNamespace(id="run-2"), task):
                    with invocation():
                        outcome["admitted_at"] = time.monotonic()
            except BaseException as exc:  # recorded for the assertions
                outcome["error"] = exc
        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def test_a_queued_attempt_runs_when_its_sibling_settles(self):
        for store_type in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_type.__name__), TemporaryDirectory() as root:
                store = store_type(Path(root))
                job_id, payload = self.launch(store)
                first = self.start_first(store, job_id, payload)
                second = self.claimed(store, job_id, payload, "second")
                outcome = {}
                thread = self.second_in_thread(store, second, outcome)
                time.sleep(0.5)
                self.assertNotIn("admitted_at", outcome)
                self.assertNotIn("error", outcome)
                settled_at = time.monotonic()
                store.reconcile_reservation(
                    job_id, first["attempt"]["attempt_id"], reconciliation_id="final",
                    liability=BudgetLiability(billing="plan", plan_marginal_usd=0,
                                              cost_state="known", elapsed_seconds=150),
                    final=True, evidence="test settlement")
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertNotIn("error", outcome)
                self.assertGreaterEqual(outcome["admitted_at"], settled_at)
                self.assertEqual(len(store.budget_snapshot(job_id)["reservations"]), 2)
                waiting = [e for e in store.read_events(job_id) if e.get("event") == "budget_admission.waiting"]
                self.assertEqual(len(waiting), 1)

    def test_a_cut_ends_the_wait(self):
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(Path(root))
            job_id, payload = self.launch(store)
            self.start_first(store, job_id, payload)
            second = self.claimed(store, job_id, payload, "second")
            outcome = {}
            thread = self.second_in_thread(store, second, outcome)
            time.sleep(0.3)
            # A busy store can refuse one cut; production retries each second
            # (_cut_jobs), and a retry finishes a half-done cut.
            for _ in range(50):
                try:
                    store.cut_task(job_id, second.id)
                    break
                except Exception:
                    time.sleep(0.05)
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(outcome.get("error"), JobCancelled)
            self.assertNotIn("admitted_at", outcome)
            self.assertEqual(len(store.budget_snapshot(job_id)["reservations"]), 1)

    def test_an_exhausted_cap_fails_without_a_wait(self):
        with TemporaryDirectory() as root:
            store = SwarmStore(Path(root))
            job_id, payload = _launch_until_tasks(store, "codex", BudgetPolicy(max_elapsed_seconds=300),
                                                  timeout_seconds=180)
            first = self.start_first(store, job_id, payload)
            store.reconcile_reservation(
                job_id, first["attempt"]["attempt_id"], reconciliation_id="final",
                liability=BudgetLiability(billing="plan", plan_marginal_usd=0,
                                          cost_state="known", elapsed_seconds=200),
                final=True, evidence="test settlement")
            second = self.claimed(store, job_id, payload, "second")
            started = time.monotonic()
            with self.assertRaisesRegex(BudgetAdmissionError, "exhausted"):
                with execution_scope(store, SimpleNamespace(id="run-2"), second):
                    with invocation():
                        self.fail("an exhausted cap admitted an attempt")
            self.assertLess(time.monotonic() - started, 1)


if __name__ == "__main__":
    unittest.main()
