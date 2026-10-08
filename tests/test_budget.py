"""Stage-one reservation contract; no provider calls or global dispatch changes."""
import json
import os
import sqlite3
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401

from puppetmaster.attempts import ExecutionAttempt, UsageObservation
from puppetmaster.budget import BudgetPolicy, BudgetLiability, BudgetAdmissionError, BudgetConflictError
from puppetmaster.models import Task, TaskStatus, job_from_dict, to_jsonable
from puppetmaster.store import SwarmStore, LaunchConflictError
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError


def api(amount=1, **kwargs):
    return BudgetLiability(billing="api", cost_state="known", api_usd=amount, **kwargs)


class BudgetContract:
    store_type = SwarmStore

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.store_type(self.root)
        self.job = self.store.create_job("budget", budget_policy=BudgetPolicy(max_usd=2))
        self.task = Task(job_id=self.job.id, role="test", instruction="budget")
        self.store.save_task(self.task)
        self.attempt = ExecutionAttempt(self.job.id, self.task.id, "run", "invocation",
                                        "2026-09-06T00:00:00+00:00", "codex")

    def reserve(self, amount=1, attempt=None):
        return self.store.reserve_dispatch(attempt or self.attempt, api(amount))

    def adopt(self):
        return self.store.adopt_dispatch(self.job.id, self.attempt.attempt_id, adoption_id="owner")

    def reconcile(self, liability, key="final", final=True):
        return self.store.reconcile_reservation(
            self.job.id, self.attempt.attempt_id, reconciliation_id=key,
            liability=liability, final=final, evidence="authoritative cumulative SDK snapshot")

    def test_reservation_fences_later_attempt_identity(self):
        self.reserve()
        self.adopt()
        for field in ("run_id", "task_id", "adapter", "model", "provider", "started_at"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(BudgetConflictError, "invocation identity conflict"):
                    self.store.record_attempt(replace(self.attempt, **{field: "different"}))
        self.assertEqual(self.store.list_attempts(self.job.id), [])
        self.assertTrue(self.store.record_attempt(self.attempt))
        self.assertFalse(self.store.record_attempt(self.attempt))
        self.assertEqual(self.reconcile(api())["state"], "settled")

    def test_recorded_attempt_fences_reservation_identity(self):
        self.store.record_attempt(self.attempt)
        for field in ("run_id", "task_id", "adapter", "model", "provider", "started_at"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(BudgetConflictError, "invocation identity conflict"):
                    self.reserve(attempt=replace(self.attempt, **{field: "different"}))
        with self.assertRaisesRegex(BudgetConflictError, "already recorded"):
            self.reserve()
        self.assertEqual(self.store.budget_snapshot(self.job.id)["reservations"], [])

    def test_reopen_legacy_identity_and_provider(self):
        self.attempt = replace(self.attempt, provider="openai")
        self.reserve()
        self.store = self.store_type(self.root)
        with self.assertRaisesRegex(BudgetConflictError, "invocation identity conflict"):
            self.store.record_attempt(replace(self.attempt, provider=None))
        self.assertTrue(self.store.record_attempt(self.attempt))
        self.assertFalse(self.store.record_attempt(self.attempt))
        # Old reservation JSON has no provider; missing and explicit None agree.
        other = replace(self.attempt, attempt_id="legacy", provider=None)
        record = self.reserve(attempt=other)
        del record["attempt"]["provider"]
        with self.store._budget_scope(self.job.id):
            self.store._save_budget_record(record)
        self.assertEqual(self.reserve(attempt=other), record)
        self.assertTrue(self.store.record_attempt(other))

    def test_released_identity_cannot_be_recorded(self):
        self.reserve()
        self.store.release_undispatched(self.job.id, self.attempt.attempt_id,
                                       non_dispatch_proof="preflight refused")
        with self.assertRaisesRegex(BudgetConflictError, "contradicts non-dispatch"):
            self.store.record_attempt(self.attempt)
        self.assertEqual(self.store.list_attempts(self.job.id), [])

    def test_old_cross_ledger_conflict_blocks_transitions_and_usage(self):
        self.reserve()
        self.adopt()
        self.reconcile(api(.5), key="partial", final=False)
        before = self.store.budget_snapshot(self.job.id)
        conflicting = replace(self.attempt, run_id="different", model="other")
        # Reproduce persisted data written before the cross-ledger guard existed.
        if isinstance(self.store, SQLiteSwarmStore):
            with self.store._session() as db:
                db.execute("INSERT INTO execution_attempts VALUES (?, ?, ?, ?)",
                           (self.job.id, conflicting.attempt_id, conflicting.task_id,
                            json.dumps(asdict(conflicting))))
        else:
            self.store.write_json(self.store._ledger_dir(self.job.id) / "attempts" /
                                  (self.store._ledger_key(conflicting.attempt_id) + ".json"), conflicting)
        self.store = self.store_type(self.root)
        observation = UsageObservation(self.job.id, self.attempt.attempt_id,
                                       "sdk", "sdk", self.attempt.started_at)
        for operation in (self.reserve, self.adopt,
                          lambda: self.reconcile(api()),
                          lambda: self.reconcile(api(.5), key="partial", final=False),
                          lambda: self.store.record_attempt(conflicting),
                          lambda: self.store.record_usage_observation(observation)):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(BudgetConflictError, "invocation identity conflict"):
                    operation()
        self.assertEqual(self.store.budget_snapshot(self.job.id), before)
        self.assertEqual(self.store.list_usage_observations(self.job.id), [])

    def test_policy_roundtrip_and_legacy(self):
        self.assertEqual(self.store.get_job(self.job.id).budget_policy, BudgetPolicy(max_usd=2))
        raw = to_jsonable(self.job)
        self.assertEqual(job_from_dict(raw), self.job)
        del raw["budget_policy"]
        self.assertIsNone(job_from_dict(raw).budget_policy)
        legacy = self.store.create_job("legacy")
        attempt = replace(self.attempt, job_id=legacy.id)
        self.store.reserve_dispatch(attempt, BudgetLiability())
        snapshot = self.store.budget_snapshot(legacy.id)
        self.assertIsNone(snapshot["policy"])
        self.assertIsNone(snapshot["totals"]["marginal_usd"]["total"])
        self.assertEqual(self.store.list_attempts(legacy.id), [])

    def test_launch_replay_preserves_policy(self):
        job = self.store.create_job("same", launch_key="key", budget_policy=BudgetPolicy(max_usd=1))
        self.assertEqual(self.store.create_job("same", launch_key="key", budget_policy=job.budget_policy), job)
        with self.assertRaises(LaunchConflictError):
            self.store.create_job("same", launch_key="key", budget_policy=BudgetPolicy(max_usd=2))

    def test_admission_duplicate_conflict_and_terminal_replay(self):
        record = self.reserve(2)
        self.assertEqual(self.reserve(2), record)
        with self.assertRaises(BudgetConflictError):
            self.reserve(1)
        with self.assertRaises(BudgetConflictError):
            self.reserve(2, replace(self.attempt, run_id="other"))
        with self.assertRaises(BudgetAdmissionError):
            self.reserve(1, replace(self.attempt, attempt_id="second"))
        adopted = self.adopt()
        self.assertEqual(adopted, self.adopt())
        with self.assertRaises(BudgetConflictError):
            self.store.adopt_dispatch(self.job.id, self.attempt.attempt_id, adoption_id="other")
        settled = self.reconcile(api(1))
        self.assertEqual(settled["state"], "settled")
        self.assertEqual(self.reconcile(api(1)), settled)
        self.assertEqual(self.reserve(2), settled)
        with self.assertRaises(BudgetConflictError):
            self.reconcile(api(2))
        with self.assertRaises(BudgetConflictError):
            self.reconcile(api(1), key="different")
        self.reserve(1, replace(self.attempt, attempt_id="second"))

    def test_crash_pending_reopen_and_reset(self):
        self.reserve(2)
        self.adopt()
        reopened = self.store_type(self.root)
        snap = reopened.budget_snapshot(self.job.id)
        self.assertEqual(snap["reservations"][0]["state"], "dispatching")
        self.assertEqual(snap["totals"]["marginal_usd"]["total"], 2)
        self.reconcile(BudgetLiability(), key="crashed", final=False)
        snap = self.store.budget_snapshot(self.job.id)
        self.assertIsNone(snap["totals"]["marginal_usd"]["total"])
        with self.assertRaises(BudgetAdmissionError):
            self.reserve(0, replace(self.attempt, attempt_id="second"))
        self.store.save_task(replace(self.task, status=TaskStatus.FAILED, attempts=8))
        self.store.reset_subgraph(self.job.id, [self.task.id])
        self.assertEqual(reopened.budget_snapshot(self.job.id), snap)
        self.assertEqual(self.reconcile(api(1))["state"], "settled")

    def test_release_requires_non_dispatch_and_proof_replay(self):
        self.reserve(2)
        with self.assertRaises(ValueError):
            self.store.release_undispatched(self.job.id, self.attempt.attempt_id, non_dispatch_proof="")
        result = self.store.release_undispatched(self.job.id, self.attempt.attempt_id,
                                               non_dispatch_proof="preflight refused")
        self.assertEqual(result, self.store.release_undispatched(
            self.job.id, self.attempt.attempt_id, non_dispatch_proof="preflight refused"))
        with self.assertRaises(BudgetConflictError):
            self.adopt()
        self.assertEqual(self.store.budget_snapshot(self.job.id)["totals"]["attempts"], 0)
        other = replace(self.attempt, attempt_id="second")
        self.reserve(2, other)
        self.store.adopt_dispatch(self.job.id, other.attempt_id, adoption_id="owner")
        with self.assertRaises(BudgetConflictError):
            self.store.release_undispatched(self.job.id, other.attempt_id, non_dispatch_proof="timeout")

    def test_partial_unknown_and_no_observation_double_count(self):
        self.reserve()
        self.adopt()
        self.store.record_attempt(self.attempt)
        for key in ("stdout", "return"):
            self.store.record_usage_observation(UsageObservation(
                self.job.id, self.attempt.attempt_id, key, "sdk", self.attempt.started_at,
                cost_state="measured", cost_usd=1, cost_basis="api"))
        partial = replace(api(.5), cost_state="partial")
        self.assertEqual(self.reconcile(partial, key="partial")["state"], "pending_reconciliation")
        snap = self.store.budget_snapshot(self.job.id)["totals"]["marginal_usd"]
        self.assertEqual(snap, {"total": None, "known_subtotal": .5, "state": "partial"})
        self.reconcile(api(1))
        self.assertEqual(self.store.budget_snapshot(self.job.id)["totals"]["marginal_usd"]["total"], 1)

    def test_plan_api_separation(self):
        plan = BudgetLiability(billing="plan", cost_state="known", plan_marginal_usd=0,
                               api_equivalent_usd=100)
        self.store.reserve_dispatch(self.attempt, plan)
        self.adopt()
        self.reconcile(plan)
        self.reserve(2, replace(self.attempt, attempt_id="api"))
        totals = self.store.budget_snapshot(self.job.id)["totals"]
        self.assertEqual(totals["marginal_usd"]["total"], 2)
        self.assertEqual(totals["api_usd"]["total"], 2)
        self.assertEqual(totals["plan_marginal_usd"]["total"], 0)
        self.assertEqual(totals["api_equivalent_usd"]["total"], 100)

    def test_independent_limits_and_unknown(self):
        for policy, allowance in [
            (BudgetPolicy(max_tokens_in=1), api(tokens_in=2)),
            (BudgetPolicy(max_tokens_out=1), api(tokens_out=2)),
            (BudgetPolicy(max_elapsed_seconds=1), api(elapsed_seconds=2)),
            (BudgetPolicy(max_tokens_out=1), api()),
            (BudgetPolicy(max_attempts=0), api()),
            (BudgetPolicy(max_usd=1), BudgetLiability()),
        ]:
            with self.subTest(policy=policy):
                self.store.save_job(replace(self.job, budget_policy=policy))
                with self.assertRaises(BudgetAdmissionError):
                    self.store.reserve_dispatch(self.attempt, allowance)
        self.assertEqual(self.store.budget_snapshot(self.job.id)["reservations"], [])

    def test_unknown_final_actual_overrun_and_decimal_admission(self):
        self.reserve(.1)
        self.reserve(.2, replace(self.attempt, attempt_id="second"))
        self.store.save_job(replace(self.job, budget_policy=BudgetPolicy(max_usd=.3)))
        self.reserve(0, replace(self.attempt, attempt_id="third"))
        self.adopt()
        self.assertEqual(self.reconcile(BudgetLiability(), key="unknown")["state"],
                         "pending_reconciliation")
        self.assertEqual(self.reconcile(api(5))["state"], "settled")
        with self.assertRaises(BudgetAdmissionError):
            self.reserve(0, replace(self.attempt, attempt_id="fourth"))

    def test_reserved_crash_retains_allowance_and_usage_contradicts_release(self):
        self.reserve(2)
        self.assertEqual(self.store_type(self.root).budget_snapshot(self.job.id)[
            "reservations"][0]["state"], "reserved")
        self.store.record_attempt(self.attempt)
        with self.assertRaises(BudgetConflictError):
            self.store.release_undispatched(self.job.id, self.attempt.attempt_id,
                                           non_dispatch_proof="incorrect claim")
        historical = replace(self.attempt, attempt_id="historical")
        self.store.record_attempt(historical)
        with self.assertRaises(BudgetConflictError):
            self.reserve(0, historical)

    def test_delete_and_events_unchanged(self):
        before = self.store.read_events_since(self.job.id, 0)
        self.reserve()
        self.adopt()
        self.reconcile(api())
        self.assertEqual(self.store.read_events_since(self.job.id, 0), before)
        self.store.delete_job(self.job.id)
        reopened = self.store_type(self.root)
        reopened.init()
        self.assertEqual(reopened._budget_records(self.job.id), [])


class AdmissionMessageTests(unittest.TestCase):
    """The refusal names its case: an unsettled earlier attempt, or a cap that is spent."""

    @staticmethod
    def record(state, liability=None, elapsed=240.0):
        return {"state": state, "liability": liability,
                "allowance": asdict(BudgetLiability(elapsed_seconds=elapsed))}

    def test_an_unsettled_earlier_attempt_is_indeterminate_not_exhausted(self):
        from puppetmaster.budget import check_admission

        pending = self.record("pending_reconciliation", asdict(BudgetLiability()))
        with self.assertRaisesRegex(BudgetAdmissionError,
                                    "max_elapsed_seconds: indeterminate: an earlier attempt"):
            check_admission(BudgetPolicy(max_elapsed_seconds=1800), [pending, self.record("reserved")])

    def test_a_spent_cap_says_exhausted_with_the_numbers(self):
        from puppetmaster.budget import check_admission

        with self.assertRaisesRegex(BudgetAdmissionError,
                                    "max_elapsed_seconds: exhausted: .* total to 480, above the cap of 320"):
            check_admission(BudgetPolicy(max_elapsed_seconds=320),
                            [self.record("dispatching"), self.record("reserved")])
        check_admission(BudgetPolicy(max_elapsed_seconds=480),
                        [self.record("dispatching"), self.record("reserved")])


class FileBudgetTests(BudgetContract, unittest.TestCase):
    def test_contention(self):
        key = f"budget:{self.job.id}"
        self.store.acquire_lock(key, "holder")
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.reserve()
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.store.record_attempt(self.attempt)
        finally:
            self.store.release_lock(key, owner="holder")
        self.reserve()


class SQLiteBudgetTests(BudgetContract, unittest.TestCase):
    store_type = SQLiteSwarmStore

    def test_atomic_admission_independent_stores(self):
        def reserve(index):
            store = SQLiteSwarmStore(self.root)
            store.attach()
            try:
                store.reserve_dispatch(replace(self.attempt, attempt_id=str(index)), api(2))
                return True
            except BudgetAdmissionError:
                return False
        with ThreadPoolExecutor(max_workers=6) as pool:
            self.assertEqual(sum(pool.map(reserve, range(12))), 1)
        self.assertEqual(self.store.budget_snapshot(self.job.id)["totals"]["marginal_usd"]["total"], 2)

    def test_reserve_races_conflicting_attempt(self):
        barrier = Barrier(2)

        def write(reserving):
            store = SQLiteSwarmStore(self.root)
            store.attach()
            barrier.wait(timeout=5)
            try:
                if reserving:
                    store.reserve_dispatch(self.attempt, api())
                else:
                    store.record_attempt(replace(self.attempt, run_id="different"))
                return True
            except BudgetConflictError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(write, (True, False))), 1)
        self.assertNotEqual(bool(self.store.list_attempts(self.job.id)),
                            bool(self.store._budget_records(self.job.id)))

    def test_concurrent_duplicate(self):
        def reserve(_):
            store = SQLiteSwarmStore(self.root)
            store.attach()
            return store.reserve_dispatch(self.attempt, api(2))
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(reserve, range(8)))
        self.assertTrue(all(record == records[0] for record in records))
        self.assertEqual(len(self.store.budget_snapshot(self.job.id)["reservations"]), 1)

    def test_enclosing_rollback(self):
        with self.assertRaises(RuntimeError):
            with self.store._completion_scope(self.job.id):
                self.reserve()
                self.store.record_attempt(self.attempt)
                raise RuntimeError("rollback")
        self.assertEqual(self.store.budget_snapshot(self.job.id)["reservations"], [])

        self.assertEqual(self.store.list_attempts(self.job.id), [])

    def test_v3_migration_and_foreign_keys(self):
        self.store.record_attempt(self.attempt)
        with closing(sqlite3.connect(self.store.db_path)) as db, db:
            db.execute("DROP TABLE budget_reservations")
            db.execute("UPDATE metadata SET value = '3' WHERE key = 'schema_version'")
        with self.assertRaises(SqliteSchemaError):
            SQLiteSwarmStore(self.root).attach()
        migrated = SQLiteSwarmStore(self.root)
        migrated.init()
        self.assertEqual(migrated.schema_status()["schema_version"], "7")
        self.assertEqual(migrated.list_attempts(self.job.id), [self.attempt])
        self.assertEqual(migrated.budget_snapshot(self.job.id)["reservations"], [])
        new_attempt = replace(self.attempt, attempt_id="post-migration")
        migrated.reserve_dispatch(new_attempt, api())
        migrated.adopt_dispatch(self.job.id, new_attempt.attempt_id, adoption_id="owner")
        with self.assertRaisesRegex(BudgetConflictError, "invocation identity conflict"):
            migrated.record_attempt(replace(new_attempt, run_id="other", model="other"))
        self.assertTrue(migrated.record_attempt(new_attempt))
        with migrated._session() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO budget_reservations VALUES('absent','id','reserved','{}')")

    def test_migration_rollback(self):
        with closing(sqlite3.connect(self.store.db_path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("DROP TABLE budget_reservations")
            db.execute("UPDATE metadata SET value = '3' WHERE key = 'schema_version'")
            db.commit()
            def deny_index(action, name, *unused):
                return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_INDEX and
                        name == "idx_budget_job_state" else sqlite3.SQLITE_OK)
            db.set_authorizer(deny_index)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    with db:
                        self.store._migrate_schema(db)
            finally:
                # Python 3.9 cannot disable the authorizer with None.
                db.set_authorizer(lambda *unused: sqlite3.SQLITE_OK)
            self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0], "3")
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='budget_reservations'").fetchone())


class ValidationTests(unittest.TestCase):
    def test_policy_and_liability_validation(self):
        self.assertTrue(all(v is None for v in asdict(BudgetPolicy()).values()))
        for kwargs in ({"max_usd": -1}, {"max_usd": float("nan")},
                       {"max_attempts": True}, {"max_tokens_in": 1.2}):
            with self.assertRaises(ValueError):
                BudgetPolicy(**kwargs)
        for kwargs in ({"billing": "api", "api_usd": 0},
                       {"cost_state": "known"}, {"billing": "plan", "api_usd": 1},
                       {"tokens_in": True}, {"elapsed_seconds": float("inf")}):
            with self.assertRaises(ValueError):
                BudgetLiability(**kwargs)


if __name__ == "__main__":
    unittest.main()
