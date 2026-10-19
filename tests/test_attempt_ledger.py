"""Immutable invocation ledger contract shared by both persistence backends."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401

from puppetmaster.attempts import (
    ExecutionAttempt, UsageObservation, LedgerConflictError, canonical_record,
)
from puppetmaster.identity import StoreMetadataDrift
from puppetmaster.models import AgentRun, Task, TaskStatus
from puppetmaster.orchestrator import _ATTACH_RESPAWN_LIMIT
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError
from puppetmaster.store import SwarmStore


class LedgerFixture:
    store_type = SwarmStore

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "state"
        self.store = self.store_type(self.root)
        self.store.init()
        self.job = self.store.create_job("ledger contract")
        self.task = Task(job_id=self.job.id, role="implement", instruction="test")
        self.store.save_task(self.task)
        self.run = AgentRun(job_id=self.job.id, task_id=self.task.id,
                            role="implement", worker_id="same-worker")
        self.attempt = ExecutionAttempt.from_run(self.run, adapter="codex")


class LedgerContract(LedgerFixture):
    def observation(self, **kwargs):
        return UsageObservation(job_id=self.job.id, attempt_id=self.attempt.attempt_id,
                                observation_id="source-event-1", source="codex",
                                observed_at=self.run.started_at, **kwargs)

    def test_attempt_identity_retry_reset_and_reopen(self):
        self.assertTrue(self.store.record_attempt(self.attempt))
        self.assertFalse(self.store.record_attempt(self.attempt))
        original_usage = self.observation(returncode=-9, timed_out=True)
        self.store.record_usage_observation(original_usage)
        failed = replace(self.run, status=TaskStatus.FAILED)
        self.store.save_run(failed)
        self.store.save_task(replace(self.task, status=TaskStatus.FAILED, attempts=7))
        self.store.reset_subgraph(self.job.id, [self.task.id])
        retry = AgentRun(job_id=self.job.id, task_id=self.task.id,
                         role=self.run.role, worker_id=self.run.worker_id)
        second = ExecutionAttempt.from_run(retry, adapter="codex")
        self.assertNotEqual(second.attempt_id, self.attempt.attempt_id)
        self.store.record_attempt(second)
        # A second invocation inside one AgentRun also gets its own identity.
        third = ExecutionAttempt.from_run(retry, adapter="codex", invocation_id="call-2")
        self.store.record_attempt(third)
        reopened = self.store_type(self.root)
        self.assertEqual(set(reopened.list_attempts(self.job.id)),
                         {self.attempt, second, third})
        self.assertEqual(len(reopened.list_attempts(self.job.id, task_id=self.task.id)), 3)
        self.assertEqual(reopened.list_attempts(self.job.id, task_id="other"), [])
        self.assertEqual(self.store.get_task_by_id(self.task.id).attempts, 0)
        self.assertEqual(reopened.list_usage_observations(self.job.id), [original_usage])

    def test_unknown_measured_zero_partial_and_duplicate(self):
        self.store.record_attempt(self.attempt)
        unknown = self.observation()
        zero = replace(unknown, observation_id="zero", usage_state="measured",
                       tokens_in=0, tokens_out=0, cost_state="measured", cost_usd=0,
                       cost_basis="plan_marginal")
        partial = replace(unknown, observation_id="partial", usage_state="measured",
                          tokens_in=42, cost_basis="api")
        for obs in (unknown, zero, partial):
            self.assertTrue(self.store.record_usage_observation(obs))
            self.assertFalse(self.store.record_usage_observation(obs))
        reopened = self.store_type(self.root)
        records = {r.observation_id: r for r in reopened.list_usage_observations(self.job.id)}
        self.assertIsNone(records[unknown.observation_id].tokens_in)
        self.assertIsNone(records[unknown.observation_id].cost_usd)
        self.assertEqual(records["zero"].tokens_in, 0)
        self.assertEqual(records["zero"].cost_usd, 0)
        self.assertIsNone(records["partial"].tokens_out)
        self.assertIsNone(records["partial"].cost_usd)
        self.assertEqual(reopened.list_usage_observations(self.job.id, attempt_id="missing"), [])
        with self.assertRaises(LedgerConflictError):
            self.store.record_usage_observation(replace(zero, tokens_in=1))
        with self.assertRaises(LedgerConflictError):
            self.store.record_attempt(replace(self.attempt, model="different"))

    def test_observation_key_scoped_to_attempt(self):
        self.store.record_attempt(self.attempt)
        other = replace(self.attempt, attempt_id="second-call")
        self.store.record_attempt(other)
        observation = self.observation()
        self.store.record_usage_observation(observation)
        self.store.record_usage_observation(replace(observation, attempt_id=other.attempt_id))
        self.assertEqual(len(self.store.list_usage_observations(self.job.id)), 2)
        with self.assertRaises(ValueError):
            self.store.record_usage_observation(replace(observation, attempt_id="absent"))

    def test_legacy_and_reuse_do_not_create_attempts(self):
        self.store.save_run(self.run)
        self.assertEqual(self.store.list_attempts(self.job.id), [])
        self.assertEqual(self.store.list_usage_observations(self.job.id), [])

    def test_no_events_and_job_deletion(self):
        before = self.store.read_events_since(self.job.id, 0)
        self.store.record_attempt(self.attempt)
        self.store.record_usage_observation(self.observation())
        self.assertEqual(self.store.read_events_since(self.job.id, 0), before)
        self.store.delete_job(self.job.id)
        self.assertEqual(self.store.list_attempts(self.job.id), [])
        self.assertEqual(self.store.list_usage_observations(self.job.id), [])

    def test_opaque_observation_ids_are_not_paths(self):
        self.store.record_attempt(self.attempt)
        observation = replace(self.observation(), observation_id="../../arbitrary/path")
        self.store.record_usage_observation(observation)
        self.assertEqual(self.store.list_usage_observations(self.job.id), [observation])


class FileLedgerTests(LedgerContract, unittest.TestCase):
    def test_contention_does_not_overwrite(self):
        key = self.store._ledger_key(self.attempt.attempt_id)
        lock = f"consumption:{self.job.id}:{key}"
        self.assertTrue(self.store.acquire_lock(lock, "holder"))
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.store.record_attempt(self.attempt)
        finally:
            self.store.release_lock(lock, owner="holder")
        self.assertTrue(self.store.record_attempt(self.attempt))


class SQLiteLedgerTests(LedgerContract, unittest.TestCase):
    store_type = SQLiteSwarmStore

    def test_concurrent_replay(self):
        def write(_):
            # Each writer is a worker. Windows can move ChangeTime alone under
            # live writers. Attach then raises StoreMetadataDrift, and the
            # supervisor respawns the worker.
            for respawn in range(_ATTACH_RESPAWN_LIMIT + 1):
                store = SQLiteSwarmStore(self.root)
                try:
                    store.attach()
                except StoreMetadataDrift:
                    if respawn == _ATTACH_RESPAWN_LIMIT:
                        raise
                    continue
                return store.record_attempt(self.attempt)
        with ThreadPoolExecutor(max_workers=4) as pool:
            inserted = list(pool.map(write, range(8)))
        self.assertEqual(sum(inserted), 1)

    def test_enclosing_transaction_rollback(self):
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.store._completion_scope(self.job.id):
                self.store.record_attempt(self.attempt)
                self.store.record_usage_observation(self.observation())
                raise RuntimeError("rollback")
        self.assertEqual(self.store.list_attempts(self.job.id), [])
        self.assertEqual(self.store.list_usage_observations(self.job.id), [])

    def test_migration_failure_rolls_back_tables_and_version(self):
        with closing(sqlite3.connect(Path(self.tmp.name) / "migration.sqlite3")) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO metadata VALUES('schema_version', '2')")
            db.commit()
            def deny_second_table(action, name, *unused):
                if action == sqlite3.SQLITE_CREATE_TABLE and name == "usage_observations":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(deny_second_table)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    with db:
                        self.store._migrate_schema(db)
            finally:
                # Python 3.9 cannot disable the authorizer with None.
                db.set_authorizer(lambda *unused: sqlite3.SQLITE_OK)
            self.assertEqual(db.execute("SELECT value FROM metadata").fetchone()[0], "2")
            self.assertIsNone(db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'execution_attempts'"
            ).fetchone())

    def test_v2_migration_does_not_invent_history(self):
        self.store.save_run(self.run)
        with closing(sqlite3.connect(self.store.db_path)) as db, db:
            db.execute("DROP TABLE usage_observations")
            db.execute("DROP TABLE execution_attempts")
            db.execute("UPDATE metadata SET value = '2' WHERE key = 'schema_version'")
        with self.assertRaises(SqliteSchemaError):
            SQLiteSwarmStore(self.root).attach()
        migrated = SQLiteSwarmStore(self.root)
        migrated.init()
        self.assertEqual(migrated.schema_status()["schema_version"], "7")
        self.assertEqual(migrated.get_job(self.job.id).goal, self.job.goal)
        self.assertEqual(migrated.get_task_by_id(self.task.id), self.task)
        self.assertEqual(migrated.list_attempts(self.job.id), [])
        self.assertEqual(migrated.list_usage_observations(self.job.id), [])
        migrated.record_attempt(self.attempt)
        SQLiteSwarmStore(self.root).init()
        self.assertEqual(migrated.list_attempts(self.job.id), [self.attempt])


class RecordValidationTests(unittest.TestCase):
    def test_legacy_observation_does_not_imply_process_success(self):
        legacy = {"job_id": "job", "attempt_id": "run", "observation_id": "event",
                  "source": "codex", "observed_at": "stamp"}
        observation = UsageObservation(**legacy)
        self.assertIsNone(observation.returncode)
        self.assertIsNone(observation.timed_out)

    def test_validation_and_canonicalization(self):
        obs = UsageObservation("job", "run", "event", "sdk", "stamp")
        self.assertIsNone(json.loads(canonical_record(obs))["tokens_in"])
        with self.assertRaises(FrozenInstanceError):
            obs.tokens_in = 4
        for changes in ({"tokens_in": 0}, {"usage_state": "measured"},
                        {"usage_state": "measured", "tokens_in": True},
                        {"usage_state": "measured", "tokens_in": -1},
                        {"cost_state": "measured", "cost_usd": float("nan")},
                        {"cost_state": "measured", "cost_usd": 0},
                        {"cost_usd": 0}, {"observation_id": ""},
                        {"returncode": True}, {"returncode": 1.5},
                        {"timed_out": 1}, {"timed_out": "false"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(obs, **changes)
        integer = replace(obs, cost_state="measured", cost_usd=0, cost_basis="api")
        self.assertEqual(canonical_record(integer), canonical_record(replace(integer, cost_usd=0.0)))


if __name__ == "__main__":
    unittest.main()
