"""Release-gate regressions from job_2d881f4b456c and Marionette's byte probes."""
import ast
import hashlib
import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from readonly_fixtures import replacement_blocked

from puppetmaster.attempts import ExecutionAttempt, UsageObservation
from puppetmaster.contracts import EffectReceipt, ContractConflict, immutable_digest
from puppetmaster.identity import StoreIdentityError
from puppetmaster.models import AgentRun, Artifact, ArtifactType, JobRef, JobStatus, Task, TaskStatus, to_jsonable
from puppetmaster.projections import connection
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.store_contracts import task_binding


@contextmanager
def measured_reads(*, deny_bodies=False):
    """Measure text crossing SQLite's Python boundary, not serialized output."""
    metrics = dict(bytes=0, rows=0, connections=0, writes=[], reads=[])
    connect = sqlite3.connect

    class Cursor(sqlite3.Cursor):
        def account(self, row):
            if row is not None:
                metrics['rows'] += 1
                metrics['bytes'] += sum(len(v.encode()) if isinstance(v, str) else len(v)
                                        if isinstance(v, bytes) else 0 for v in row)
            return row

        def fetchone(self):
            return self.account(super().fetchone())

        def fetchall(self):
            return [self.account(row) for row in super().fetchall()]

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            return self.cursor(factory=Cursor).execute(sql, parameters)

    def authorize(action, table, column, *rest):
        if action == sqlite3.SQLITE_READ:
            metrics['reads'].append((table, column))
            if deny_bodies and (table == 'completions' or (column == 'data' and table in {
                    'jobs', 'tasks', 'artifacts', 'runs', 'execution_attempts', 'usage_observations'})):
                return sqlite3.SQLITE_DENY
        if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
            metrics['writes'].append((table, column))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def traced(*args, **kwargs):
        kwargs['factory'] = Connection
        c = connect(*args, **kwargs)
        c.set_authorizer(authorize)
        c.set_trace_callback(lambda sql: metrics['writes'].append(sql)
                             if sql.upper().startswith(('BEGIN IMMEDIATE', 'BEGIN EXCLUSIVE')) else None)
        metrics['connections'] += 1
        return c

    from puppetmaster.readonly import ReadConnection

    class TracedReader(ReadConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_authorizer(authorize)
            self.set_trace_callback(lambda sql: metrics['writes'].append(sql)
                if sql.upper().startswith(('BEGIN IMMEDIATE', 'BEGIN EXCLUSIVE')) else None)
            metrics['connections'] += 1

        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            # Account actual helper-to-Python values, with the same body-denying
            # SQLite authorizer now running in the isolated reader process.
            rows = result.fetchall()
            for row in rows:
                metrics['rows'] += 1
                metrics['bytes'] += sum(len(v.encode()) if isinstance(v, str) else len(v)
                    if isinstance(v, bytes) else 0 for v in row)
            result.rows = iter(rows)
            return result

    with ExitStack() as stack:
        stack.enter_context(patch('sqlite3.connect', side_effect=traced))
        stack.enter_context(patch('puppetmaster.readonly.ReadConnection', TracedReader))
        if deny_bodies:
            stack.enter_context(patch.object(SwarmStore, 'read_json', side_effect=AssertionError('file body read')))
        yield metrics


class ReleaseBlockerTests(unittest.TestCase):
    def stores(self):
        for backend in (SwarmStore, SQLiteSwarmStore):
            with TemporaryDirectory() as tmp:
                store = backend(Path(tmp) / 'state')
                job = store.create_job('target')
                yield store, job, store.job_ref(job.id)

    def task_run(self, store, job):
        task = Task(job_id=job.id, role='explore', instruction='private')
        store.save_task(task)
        task = store.claim_task(task.id, 'worker')
        run = AgentRun(job.id, task.id, task.role, 'worker', status=TaskStatus.COMPLETE, completed_at='now')
        return task, run

    def test_unscoped_reads_and_accessor_reject_replacement(self):
        for store, job, ref in self.stores():
            store.root.rename(store.root.parent / 'old')
            successor = type(store)(store.root)
            successor.create_job('replacement')
            for read in (store.list_job_summaries, store.read_job_summary_changes,
                         lambda: store.incarnation, lambda: store.get_completion_receipt(ref, 'r')):
                with self.assertRaises(StoreIdentityError):
                    read()

    def test_completion_one_mib_never_reads_intent_or_reserves_writer(self):
        for store, job, ref in self.stores():
            task, run = self.task_run(store, job)
            with patch.object(store, 'reconcile_completions'):
                receipt = store.submit_completion(task, run, [], {'payload': 'x' * 1048576}, job_ref=ref)
            self.assertEqual(receipt.outcome, 'pending_publication')
            with ExitStack() as stack:
                for method in ('read_json', '_get_completion', '_completion_records', '_completion_intent_scope'):
                    stack.enter_context(patch.object(store, method, side_effect=AssertionError('intent body accessed')))
                if store.backend_name == 'sqlite':
                    stack.enter_context(patch.object(store, '_writer_scope', side_effect=AssertionError('writer reserved')))
                with measured_reads(deny_bodies=True) as metrics:
                    found = store.get_completion_receipt(ref, run.id)
            self.assertEqual(found, receipt)
            self.assertEqual(metrics['connections'], 1)
            self.assertFalse(metrics['writes'])
            self.assertLess(metrics['bytes'], 1024)
            store.reconcile_completions(job.id)
            with measured_reads(deny_bodies=True):
                self.assertEqual(store.get_completion_receipt(ref, run.id).outcome, 'published')

    def test_receipt_read_validation_and_use_share_snapshot(self):
        from puppetmaster import identity
        for store, job, ref in self.stores():
            task, run = self.task_run(store, job)
            record = {'task': to_jsonable(task), 'run': to_jsonable(run), 'intent_digest': 'a' * 64,
                      'publication': 'published', 'payload': 'x' * 1048576}
            store._save_completion(job.id, record)
            original = identity.read_identity
            swapped = []
            protected = []
            def replace_store():
                store.root.rename(store.root.parent / 'old')
                type(store)(store.root).init()

            def swap_after_read(c, backend):
                value = original(c, backend)
                if not swapped:
                    swapped.append(True)
                    protected.append(replacement_blocked(replace_store))
                return value

            with patch('puppetmaster.identity.read_identity', side_effect=swap_after_read):
                self.assertEqual(store.get_completion_receipt(ref, run.id).intent_digest, 'a' * 64)
            self.assertEqual(protected, [sys.platform == 'win32'])
            if protected == [True]:
                replace_store()
            with self.assertRaises(StoreIdentityError):
                store.get_completion_receipt(ref, run.id)

    def test_legacy_refs_read_through_python_cli_and_mcp_without_certification(self):
        from puppetmaster.mcp_server import run_await_job, run_feed_follow, job_schema
        for store, job, ref in self.stores():
            legacy = JobRef(job.id, ref.state_id)
            selected = type(store)(store.root).bind_job_ref(legacy, legacy_read=True)
            self.assertEqual(selected.list_job_summaries(job_ref=legacy).items[0].job_ref, legacy)
            with self.assertRaisesRegex(StoreIdentityError, 'explicitly rebind'):
                selected.bind_job_ref(legacy)
            options = dict(job_ref=legacy.as_dict(), state_dir=str(store.root), backend=store.backend_name,
                           timeout_seconds=0.001)
            for read in (run_await_job, run_feed_follow):
                result = read(options)
                body = json.loads(result['content'][0]['text'])
                self.assertEqual(body['job_ref'], legacy.as_dict())
            result = subprocess.run([sys.executable, '-m', 'puppetmaster', '--state-dir', str(store.root),
                '--backend', store.backend_name, '--job-ref', json.dumps(legacy.as_dict()),
                'await', job.id, '--json', '--timeout-seconds', '0.001'], capture_output=True, text=True)
            self.assertIn(result.returncode, (0,1), result.stderr)
            self.assertEqual(json.loads(result.stdout)['job_ref'], legacy.as_dict())
            task, run = self.task_run(store, job)
            with self.assertRaises(StoreIdentityError):
                selected.request_cancellation(legacy, 'request', [task_binding(task)])
        # The schema does not restrict the version: a legacy ref passes to JobRef.
        self.assertNotIn('properties', job_schema()['properties']['job_ref'])

    def test_legacy_effect_cancel_and_completion_replay_preserve_persisted_identity(self):
        for store, job, ref in self.stores():
            task, run = self.task_run(store, job)
            legacy = JobRef(job.id, ref.state_id)
            binding = task_binding(task)
            intent = EffectReceipt(legacy, 'effect', 'd' * 64, binding, run.id, 'attempt', 1,
                                   'not_dispatched', 'reconcile_first')
            prior = replace(intent, outcome='succeeded', revision=3, evidence_refs=('provider:ok',))
            # Exact v1.24 wire bytes: no version/incarnation/null injected.
            raw = to_jsonable(prior)
            self.assertEqual(raw['job_ref'], {'job_id': job.id, 'state_id': ref.state_id})
            from puppetmaster.contracts import CancellationReceipt
            cancellation = CancellationReceipt(legacy, 'cancel', (binding,), 'requested', 1)
            with connection(store) as c:
                c.execute('INSERT INTO contract_receipts VALUES(?,?,?,?)', ('effect',job.id,'effect',json.dumps(raw)))
                c.execute('INSERT INTO contract_receipts VALUES(?,?,?,?)', ('cancel',job.id,'cancel',json.dumps(to_jsonable(cancellation))))
            with patch.object(store, 'list_attempts', side_effect=AssertionError('replay enumerated attempts')):
                self.assertEqual(store.execute_effect(replace(intent, job_ref=ref), lambda: self.fail('redispatched')), prior)
            self.assertEqual(store.request_cancellation(ref, 'cancel', [binding]), cancellation)
            with self.assertRaises(ContractConflict):
                store.record_effect(replace(intent, job_ref=ref, request_digest='other'))
            submission = dict(task_id=task.id, job_id=job.id, lease_id=task.lease_id, worker_id=run.worker_id,
                run_id=run.id, started_at=run.started_at, completed_at=run.completed_at, artifacts=[],
                event_payload={'receipt': to_jsonable(cancellation)})
            digest = hashlib.sha256(json.dumps(submission, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            record = dict(task=to_jsonable(task), run=to_jsonable(run), artifacts=[], event_payload=submission['event_payload'],
                          event_cursor=0, done=True, intent_digest=digest, publication='published')
            store._save_completion(job.id, record)
            receipt = store.submit_completion(task, run, [], {'receipt': cancellation}, job_ref=ref)
            self.assertEqual(receipt.intent_digest, digest)
            self.assertEqual(receipt.outcome, 'published')
            with connection(store) as c:
                persisted = json.loads(c.execute("SELECT data FROM contract_receipts WHERE kind='effect'").fetchone()[0])
            self.assertEqual(persisted, raw)

    def test_oversized_legacy_binding_and_scope_never_cross_python_boundary(self):
        for store, job, ref in self.stores():
            task = Task(job_id=job.id, role='explore', instruction='private', lease_owner='x' * 300000)
            store.save_task(task)
            with measured_reads(deny_bodies=True) as metrics:
                page = store.list_task_refs(ref, max_bytes=1024)
            self.assertEqual(page.outcome, 'unavailable')
            self.assertFalse(page.items)
            self.assertLess(metrics['bytes'], 3000)
            with connection(store) as c:
                scope = json.dumps(dict(origin='z' * 300000, project_id=None, session_id=None))
                c.execute("UPDATE projection_current SET scope=? WHERE kind='job'", (scope,))
                c.execute("UPDATE projection_changes SET scope=? WHERE kind='job'", (scope,))
            for read in (store.list_job_summaries, store.read_job_summary_changes):
                with measured_reads(deny_bodies=True) as metrics:
                    page = read(max_bytes=1024)
                self.assertEqual(page.outcome, 'unavailable')
                self.assertLess(metrics['bytes'], 6000)

    def test_oversized_identity_is_unavailable_not_truncated(self):
        for store, job, ref in self.stores():
            with connection(store) as c:
                c.execute("INSERT INTO projection_current(kind,job_id,id,revision,stamp) VALUES('task',?,?,0,'legacy_unknown')",
                          (job.id, 'i' * 300000))
            with measured_reads(deny_bodies=True) as metrics:
                page = store.list_task_refs(ref)
            self.assertEqual(page.outcome, 'unavailable')
            self.assertFalse(page.items)
            self.assertLess(metrics['bytes'], 2000)

    def test_snapshot_membership_survives_every_page_write_and_filter_transition(self):
        for store, job, ref in self.stores():
            jobs = [job] + [store.create_job('j%d' % i, origin='host') for i in range(5)]
            tasks, artifacts = [], []
            for i in range(6):
                task = Task(job_id=job.id, role='explore', instruction='private', id='task_%02d' % i)
                store.save_task(task)
                tasks.append(task)
                artifact = Artifact(job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                                    payload={'claim': str(i)}, created_by='worker', confidence=1,
                                    evidence=['test'], id='artifact_%02d' % i)
                store.save_artifact(artifact)
                artifacts.append(artifact)
            for read in (lambda **kw: store.list_job_summaries(**kw),
                         lambda **kw: store.list_task_refs(ref, **kw),
                         lambda **kw: store.list_artifact_refs(ref, **kw)):
                expected = read().items
                page = read(limit=1)
                actual = list(page.items)
                counter = 0
                while page.next_cursor:
                    # Unrelated insert plus mutations both behind and ahead of keyset.
                    store.create_job('later %d' % counter)
                    store.save_job(replace(jobs[-1], status=JobStatus.COMPLETE, origin='other'))
                    store.save_task(replace(tasks[-1], status=TaskStatus.FAILED))
                    store.save_artifact(replace(artifacts[-1], execution_status='failed'))
                    page = read(limit=1, cursor=page.next_cursor)
                    self.assertNotIn(page.outcome, ('cursor_expired','unavailable'))
                    actual.extend(page.items)
                    counter += 1
                    self.assertLess(counter, 100)
                self.assertEqual(actual, list(expected))
            # Frozen filtered membership must retain a row that leaves the filter.
            for item in jobs:
                store.save_job(replace(item, origin='host', status=JobStatus.RUNNING))
            expected = store.list_job_summaries(origin='host', status='running').items
            page = store.list_job_summaries(origin='host', status='running', limit=1)
            actual = list(page.items)
            while page.next_cursor:
                for item in jobs:
                    store.save_job(replace(item, origin='other', status=JobStatus.COMPLETE))
                incoming = store.create_job('new match', origin='host')
                store.save_job(replace(incoming, status=JobStatus.RUNNING))
                page = store.list_job_summaries(origin='host', status='running', limit=1, cursor=page.next_cursor)
                self.assertNotEqual(page.outcome, 'cursor_expired')
                actual.extend(page.items)
            self.assertEqual(actual, list(expected))

    def test_snapshot_retention_expires_and_history_unrelated_deletion_does_not(self):
        for store, job, ref in self.stores():
            other = store.create_job('other')
            for i in range(4):
                store.record_attempt(ExecutionAttempt(job.id,'task','run',str(i),'now','codex'))
                store.save_task(Task(job_id=job.id, role='explore', instruction='x', id='task%d' % i))
            store.record_attempt(ExecutionAttempt(other.id,'task','run','other','now','codex'))
            first = store.list_attempt_refs(ref, limit=1)
            store.delete_job(other.id)
            collected = list(first.items)
            page = first
            while page.next_cursor:
                store.record_attempt(ExecutionAttempt(job.id,'task','later','new'+str(len(collected)),'now','codex'))
                page = store.list_attempt_refs(ref, limit=1, cursor=page.next_cursor)
                self.assertNotEqual(page.outcome, 'cursor_expired')
                collected.extend(page.items)
            self.assertEqual(len(collected), 4)
            page = store.list_task_refs(ref, limit=1)
            with connection(store) as c:
                c.execute("DELETE FROM projection_versions WHERE kind='task'")
            self.assertEqual(store.list_task_refs(ref, cursor=page.next_cursor).outcome, 'cursor_expired')

    def test_observation_identity_uses_exact_scoped_attempt_outside_loaded_page(self):
        for store, job, ref in self.stores():
            other = store.create_job('other')
            for i in range(8):
                store.record_attempt(ExecutionAttempt(job.id,'task','run%d' % i,'attempt%d' % i,'now','codex'))
            store.record_attempt(ExecutionAttempt(other.id,'wrong-task','wrong-run','attempt7','now','codex'))
            self.assertEqual(store.list_attempt_refs(ref, limit=1).items[0].facts['attempt_id'], 'attempt0')
            for i in (6,7):
                store.record_usage_observation(UsageObservation(job.id,'attempt%d' % i,'exit','process','now',returncode=i))
            # A legacy missing relationship must not borrow a cross-job match.
            with connection(store) as c:
                c.execute("INSERT INTO historical_refs(kind,job_id,id,facts) VALUES('observation',?,?,?)",
                          (job.id,'missing',json.dumps(dict(attempt_id='missing',observation_id='missing'))))
            with patch.object(store, 'list_attempts', side_effect=AssertionError('enumerated attempts')), measured_reads(deny_bodies=True) as metrics:
                observed = store.list_usage_observation_refs(ref)
                outcomes = store.list_process_outcome_refs(ref)
            self.assertFalse(metrics['writes'])
            for page in (observed, outcomes):
                for item in page.items:
                    self.assertEqual(item.facts['job_id'], job.id)
                    if item.facts['attempt_id'] == 'missing':
                        self.assertEqual(item.facts['identity_state'], 'unavailable')
                        self.assertIsNone(item.facts['run_id'])
                    else:
                        self.assertEqual(item.facts['identity_state'], 'available')
                        self.assertEqual(item.facts['task_id'], 'task')
                        self.assertEqual(item.facts['run_id'], 'run' + item.facts['attempt_id'][-1])
            self.assertLess(metrics['rows'], 35)

    def test_display_and_economics_do_not_infer_legacy_payloads(self):
        for store, job, ref in self.stores():
            store.save_job(replace(job, goal='secret' * 200000, label='owner=session:fake ' * 20000))
            with measured_reads(deny_bodies=True) as metrics:
                page = store.list_job_summaries()
            wire = json.dumps(to_jsonable(page))
            self.assertEqual(page.items[0].goal_preview, ('secret' * 200000)[:512])
            self.assertTrue(page.items[0].goal_preview_truncated)
            self.assertNotIn('session:fake', wire)
            self.assertIsNone(page.items[0].origin)
            self.assertLess(metrics['bytes'], 4096)
            counts = store.historical_evidence_counts(ref)
            self.assertFalse(counts.complete_invocation_history)
            self.assertEqual(counts.coverage, 'unknown')

    def test_all_history_pages_keep_membership_under_interleaved_updates(self):
        for store, job, ref in self.stores():
            runs = []
            for i in range(5):
                run = AgentRun(job.id, 'task', 'explore', 'worker', id='run%d' % i)
                runs.append(run)
                store.save_run(run)
                store.record_attempt(ExecutionAttempt(job.id, 'task', run.id, 'attempt%d' % i, 'now', 'codex'))
                store.record_usage_observation(UsageObservation(job.id, 'attempt%d' % i, 'exit', 'process', 'now', returncode=i))
            for read in (store.list_attempt_refs, store.list_run_refs,
                         store.list_usage_observation_refs, store.list_process_outcome_refs):
                expected = read(ref).items
                page = read(ref, limit=1)
                sequences = [i.sequence for i in page.items]
                while page.next_cursor:
                    suffix = '%s_%s' % (read.__name__, len(sequences))
                    store.save_run(replace(runs[-1], completed_at=suffix))
                    store.save_run(AgentRun(job.id, 'task', 'explore', 'worker', id=suffix))
                    store.record_attempt(ExecutionAttempt(job.id,'task',suffix,suffix,'now','codex'))
                    store.record_usage_observation(UsageObservation(job.id,suffix,'exit','process','now',returncode=0))
                    page = read(ref, limit=1, cursor=page.next_cursor)
                    self.assertNotIn(page.outcome, ('unavailable', 'cursor_expired'))
                    sequences.extend(i.sequence for i in page.items)
                self.assertEqual(sequences, [i.sequence for i in expected])

    def test_completion_projection_migrates_legacy_and_file_pending_is_unavailable(self):
        for store, job, ref in self.stores():
            task, run = self.task_run(store, job)
            record = dict(task=to_jsonable(task), run=to_jsonable(run), artifacts=[], done=True,
                          intent_digest='f' * 64, publication='published', payload='x' * 1048576)
            store._save_completion(job.id, record)
            with connection(store) as c:
                c.execute('DELETE FROM completion_receipts')
                c.execute("DELETE FROM projection_meta WHERE key='completion_version'")
            with measured_reads(deny_bodies=True):
                self.assertEqual(store.get_completion_receipt(ref, run.id).outcome, 'unavailable')
            reopened = type(store)(store.root)
            reopened.init()
            with measured_reads(deny_bodies=True) as metrics:
                receipt = reopened.get_completion_receipt(ref, run.id)
            self.assertEqual(receipt.intent_digest, 'f' * 64)
            self.assertLess(metrics['bytes'], 1024)
            with connection(store) as c:
                c.execute("INSERT INTO projection_pending VALUES('interrupted')")
            with measured_reads(deny_bodies=True):
                self.assertEqual(store.get_completion_receipt(ref, run.id).outcome, 'unavailable')

    def test_oversized_attempt_identity_is_explicitly_unavailable(self):
        for store, job, ref in self.stores():
            store.record_attempt(ExecutionAttempt(job.id,'task','r' * 300000,'a','now','codex'))
            store.record_usage_observation(UsageObservation(job.id,'a','exit','process','now',returncode=0))
            with measured_reads(deny_bodies=True) as metrics:
                attempts = store.list_attempt_refs(ref)
                observed = store.list_usage_observation_refs(ref)
            self.assertEqual(attempts.outcome, 'unavailable')
            self.assertEqual(observed.items[0].facts['identity_state'], 'unavailable')
            self.assertIsNone(observed.items[0].facts['run_id'])
            self.assertIsNone(observed.items[0].facts['task_id'])
            self.assertLess(metrics['bytes'], 2000)

    def test_legacy_effect_transition_preserves_original_ref(self):
        for store, job, ref in self.stores():
            task, run = self.task_run(store, job)
            legacy = JobRef(job.id, ref.state_id)
            prior = EffectReceipt(legacy,'effect','d' * 64,task_binding(task),run.id,'a',2,
                                  'in_flight','reconcile_first',('dispatch:a',))
            with connection(store) as c:
                c.execute('INSERT INTO contract_receipts VALUES(?,?,?,?)',
                          ('effect',job.id,'effect',json.dumps(to_jsonable(prior))))
            updated = store.advance_effect(ref,'effect',expected_revision=2,outcome='succeeded',evidence_refs=('ok',))
            self.assertEqual(updated.job_ref, legacy)
            self.assertEqual(store.advance_effect(ref,'effect',expected_revision=2,outcome='succeeded',evidence_refs=('ok',)), updated)
            self.assertEqual(store.get_effect_receipt(legacy,'effect'), updated)

    def test_change_retention_is_explicit_expiry(self):
        for store, job, ref in self.stores():
            for i in range(5):
                store.create_job('job%d' % i)
            page = store.read_job_summary_changes(limit=1)
            with connection(store) as c:
                c.execute('DELETE FROM projection_changes WHERE revision=(SELECT MIN(revision) FROM projection_changes)')
            self.assertEqual(store.read_job_summary_changes(cursor=page.next_cursor,limit=1).outcome, 'cursor_expired')

    def test_snapshot_key_lookup_does_not_sort_unrelated_entities(self):
        for store, job, ref in self.stores():
            with connection(store) as c:
                c.executemany("""INSERT INTO projection_current(kind,job_id,id,revision,stamp)
                    VALUES('job',?,?,0,'legacy_unknown')""",
                    [('unrelated_%05d' % i, 'unrelated_%05d' % i) for i in range(4000)])
            steps = []

            @contextmanager
            def measured(*args, **kwargs):
                with connection(*args, **kwargs) as c:
                    c.set_progress_handler(lambda: steps.append(1) or 0, 100)
                    yield c

            with patch('puppetmaster.projections.connection', measured):
                page = store.list_job_summaries(origin='host', max_scan=2)
            self.assertEqual(page.scanned, 2)
            self.assertEqual(page.outcome, 'partial')
            self.assertFalse(page.items)
            self.assertLess(len(steps), 30, 'snapshot query sorted unrelated entity keys')

    def test_snapshot_birth_cutoff_does_not_scan_all_later_insertions(self):
        for store, job, ref in self.stores():
            with connection(store) as c:
                for key in ('a', 'z'):
                    c.execute("INSERT INTO projection_current(kind,job_id,id,revision,stamp) VALUES('job',?,?,0,'legacy_unknown')", (key,key))
            first = store.list_job_summaries(limit=1, max_scan=1)
            self.assertEqual(first.items[0].id, 'a')
            store.create_job('advance revision')
            with connection(store) as c:
                c.executemany("INSERT INTO projection_current(kind,job_id,id,revision,stamp) VALUES('job',?,?,0,'legacy_unknown')",
                              [('b%05d' % i, 'b%05d' % i) for i in range(4000)])
            steps = []

            @contextmanager
            def measured(*args, **kwargs):
                with connection(*args, **kwargs) as c:
                    c.set_progress_handler(lambda: steps.append(1) or 0, 100)
                    yield c

            with patch('puppetmaster.projections.connection', measured):
                page = store.list_job_summaries(cursor=first.next_cursor, max_scan=2)
            self.assertEqual(page.scanned, 2)
            self.assertEqual(page.outcome, 'partial')
            self.assertFalse(page.items)
            self.assertNotEqual(page.next_cursor, first.next_cursor)
            self.assertLess(len(steps), 30, 'birth cutoff scanned all later insertions')

    def test_cli_graph_legacy_selection_keeps_reference_binding(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from puppetmaster.cli._dispatch import main
        for store, job, ref in self.stores():
            legacy = JobRef(job.id, ref.state_id)
            with patch('puppetmaster.cli._dispatch._resolve_store_for_job',
                       side_effect=AssertionError('bound reference was resolved again without identity')):
                output = StringIO()
                with redirect_stdout(output):
                    result = main(['--state-dir', str(store.root), '--backend', store.backend_name,
                                   '--job-ref', json.dumps(legacy.as_dict()), 'graph', job.id])
            self.assertEqual(result, 0)
            self.assertIsInstance(json.loads(output.getvalue()), dict)

    def test_python39_grammar(self):
        for path in Path('puppetmaster').rglob('*.py'):
            ast.parse(path.read_text(), feature_version=(3,9))


if __name__ == '__main__':
    unittest.main()
