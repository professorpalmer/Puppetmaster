from contextlib import closing
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from puppetmaster.attempts import ExecutionAttempt, UsageObservation
from puppetmaster.identity import StoreIdentityError
from puppetmaster.models import JobRef, Task, AgentRun, to_jsonable
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.store_contracts import task_binding, _cancel
from puppetmaster.state import state_identity, resolve_job_state


class IncarnationHistoryTests(unittest.TestCase):
    def stores(self):
        for cls in (SwarmStore, SQLiteSwarmStore):
            with TemporaryDirectory() as tmp:
                store = cls(Path(tmp) / 'state')
                job = store.create_job('private body ' * 10000)
                yield store, job

    def test_wire_and_legacy_receipt(self):
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            self.assertEqual(ref.version, 2)
            self.assertEqual(JobRef(**ref.as_dict()), ref)
            legacy = JobRef(job.id, state_identity(store.root))
            self.assertEqual(legacy.as_dict(), dict(job_id=job.id, state_id=ref.state_id))
            self.assertEqual(JobRef(**to_jsonable(legacy)), legacy)
            store.validate_job_ref(legacy)
            with self.assertRaisesRegex(StoreIdentityError, 'rebind'):
                store.request_cancellation(legacy, 'stop', [])
            raw = dict(job_ref=legacy.as_dict(), request_id='old', bindings=[], outcome='already_terminal', revision=1)
            self.assertEqual(to_jsonable(_cancel(raw)), dict(raw, cleanup='unknown'))
            for bad in (dict(version=2), dict(version=1, incarnation=ref.incarnation), dict(version=3)):
                with self.assertRaises(ValueError):
                    JobRef(job.id, ref.state_id, **bad)

    def test_reopen_copy_and_readonly(self):
        for store, job in self.stores():
            identity = store.incarnation
            other = type(store)(store.root)
            self.assertEqual(other.job_ref(job.id), store.job_ref(job.id))
            if isinstance(other, SQLiteSwarmStore):
                with patch.object(other, 'ensure_schema', side_effect=AssertionError('write')):
                    other.attach()
            self.assertEqual(other.incarnation, identity)
            copied = store.root.parent / 'copy'
            shutil.copytree(store.root, copied)
            clone = type(store)(copied)
            clone.init()
            self.assertEqual(clone.incarnation, identity)
            self.assertNotEqual(clone.job_ref(job.id).state_id, store.job_ref(job.id).state_id)

    def test_replacement_before_selection_and_operation(self):
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            task = Task(job_id=job.id, role='explore', instruction='secret')
            store.save_task(task)
            store.validate_job_ref(ref)
            root = store.root
            root.rename(root.parent / 'old')
            replacement = type(store)(root)
            replacement.init()
            replacement.save_job(job)
            if isinstance(replacement, SQLiteSwarmStore):
                with replacement._session() as c:
                    c.execute("INSERT INTO jobs(id,data) VALUES(?,?)", (job.id,json.dumps(to_jsonable(job))))
            replacement.save_task(task)
            self.assertNotEqual(replacement.incarnation, ref.incarnation)
            with self.assertRaises(StoreIdentityError):
                resolve_job_state(job_ref=ref, state_dir=root)
            with self.assertRaises(StoreIdentityError):
                store.request_cancellation(ref, 'stop', [task_binding(task)])
            with self.assertRaises(StoreIdentityError):
                replacement.request_cancellation(ref, 'stop', [task_binding(task)])
            self.assertIsNone(replacement.get_cancellation_receipt(replacement.job_ref(job.id), 'stop'))

    def test_missing_corrupt_and_v5_migration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteSwarmStore(root)
            job = store.create_job('job')
            identity = store.incarnation
            from contextlib import closing
            with closing(sqlite3.connect(store.db_path)) as c, c:
                c.execute("UPDATE metadata SET value='5' WHERE key='schema_version'")
            migrated = SQLiteSwarmStore(root)
            migrated.init()
            self.assertEqual(migrated.incarnation, identity)
            from contextlib import closing
            with closing(sqlite3.connect(store.db_path)) as c, c:
                c.execute("DELETE FROM metadata WHERE key='incarnation'")
            with self.assertRaises(StoreIdentityError):
                SQLiteSwarmStore(root).init()
            with self.assertRaises(StoreIdentityError):
                SQLiteSwarmStore(root).job_ref(job.id)
            from contextlib import closing
            with closing(sqlite3.connect(store.db_path)) as c, c:
                c.execute("UPDATE metadata SET value='5' WHERE key='schema_version'")
            SQLiteSwarmStore(root).init()
            self.assertNotEqual(SQLiteSwarmStore(root).incarnation, identity)
            from contextlib import closing
            with closing(sqlite3.connect(store.db_path)) as c, c:
                c.execute("UPDATE metadata SET value='broken' WHERE key='incarnation'")
            with self.assertRaises(StoreIdentityError):
                SQLiteSwarmStore(root).init()

    def test_concurrent_bootstrap(self):
        for cls in (SwarmStore, SQLiteSwarmStore):
            with TemporaryDirectory() as tmp:
                def bootstrap(_):
                    store = cls(tmp)
                    store.init()
                    return store
                with ThreadPoolExecutor(max_workers=4) as pool:
                    stores = list(pool.map(bootstrap, range(8)))
                # Bootstrap is the race under test. Read identities once it has
                # settled: a readonly identity read waits for an idle source and
                # gives up after 5s of sibling activity, which a stalled Windows
                # runner exceeded mid-bootstrap (ReadUnavailable: active reader).
                self.assertEqual(len({store.incarnation for store in stores}), 1)

    def test_historical_pages_bounds_inserts_filters_and_no_hydration(self):
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            for index in range(8):
                store.record_attempt(ExecutionAttempt(job.id,'t','r',str(index),'now','codex'))
            with patch.object(store, 'list_attempts', side_effect=AssertionError('ledger hydrated')), \
                 patch.object(store, 'read_json', side_effect=AssertionError('body hydrated')):
                first = store.list_attempt_refs(ref, limit=2)
                self.assertEqual(first.captured_count, 8)
                self.assertEqual(first.scanned, 3)
                self.assertFalse(first.complete_invocation_history)
            store.record_attempt(ExecutionAttempt(job.id,'t','r','new','now','codex'))
            items = list(first.items)
            token = first.next_cursor
            while token:
                page = store.list_attempt_refs(ref, cursor=token, limit=2)
                items.extend(page.items)
                self.assertEqual(page.captured_count, 8)
                token = page.next_cursor
            self.assertEqual(len(items), 8)
            self.assertEqual(len({item.sequence for item in items}), 8)
            with self.assertRaises(ValueError):
                store.list_run_refs(ref, cursor=first.next_cursor)
            bounded = store.list_attempt_refs(ref, max_scan=1)
            self.assertEqual(bounded.scanned, 1)
            byte_bounded = store.list_attempt_refs(ref, max_bytes=1500)
            self.assertLessEqual(len(json.dumps(to_jsonable(byte_bounded)).encode()), 1500)
            for options in ({'limit':201},{'max_scan':1001},{'max_bytes':262145},{'limit':True}):
                with self.assertRaises(ValueError):
                    store.list_attempt_refs(ref, **options)
            store.root.rename(store.root.parent / 'old')
            replacement = type(store)(store.root)
            replacement.init()
            replacement.save_job(job)
            if isinstance(replacement, SQLiteSwarmStore):
                with replacement._session() as c:
                    c.execute("INSERT INTO jobs(id,data) VALUES(?,?)", (job.id,json.dumps(to_jsonable(job))))
            with self.assertRaises(ValueError):
                replacement.list_attempt_refs(replacement.job_ref(job.id), cursor=first.next_cursor)

    def test_outcomes_preserve_zero_unknown_and_cost_bases(self):
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            store.record_attempt(ExecutionAttempt(job.id,'t','run','a','now','codex'))
            store.record_attempt(ExecutionAttempt(job.id,'t','run2','b','now','codex'))
            for index, basis in enumerate(('api','plan_marginal','api_equivalent')):
                store.record_usage_observation(UsageObservation(job.id,'a',str(index),'test','now',
                    cost_state='estimated' if basis == 'api_equivalent' else 'measured',cost_usd=0,cost_basis=basis,returncode=index,timed_out=False))
            store.record_usage_observation(UsageObservation(job.id,'b','unknown','test','now'))
            page = store.list_process_outcome_refs(ref)
            self.assertEqual(page.captured_count, 3)
            self.assertEqual({x.facts['cost_basis'] for x in page.items}, {'api','plan_marginal','api_equivalent'})
            self.assertTrue(all(x.facts['cost_usd'] == 0 and x.facts['tokens_in'] is None for x in page.items))
            self.assertEqual(store.list_usage_observation_refs(ref).captured_count, 4)
            store.record_usage_observation(UsageObservation(job.id,'a','conflicting-cost','test','now',
                cost_state='measured',cost_usd=1,cost_basis='api'))
            observed = store.list_usage_observation_refs(ref)
            self.assertEqual(observed.captured_count, 5)
            self.assertEqual({x.facts['cost_usd'] for x in observed.items if x.facts['cost_basis'] == 'api'}, {0,1})
            from puppetmaster.consumption import build_attempt_consumption_report
            report = build_attempt_consumption_report(store, job.id)
            self.assertIsNone(report.totals.api_cost_usd.total)
            self.assertEqual(report.totals.api_cost_usd.conflicting_attempts, 1)
            self.assertFalse(report.complete_invocation_history)
            self.assertEqual(store.list_run_refs(ref).coverage, 'unknown')
            run = AgentRun(job.id,'t','explore','worker')
            store.save_run(run)
            self.assertEqual(store.list_run_refs(ref).captured_count, 1)
            store.save_run(replace(run, completed_at='later'))
            self.assertEqual(store.list_run_refs(ref).captured_count, 1)

    def test_replacement_between_precheck_and_transaction(self):
        for store, job in self.stores():
            task = Task(job_id=job.id, role='explore', instruction='private')
            store.save_task(task)
            ref = store.job_ref(job.id)
            original = store.validate_job_ref
            replaced = []

            def validating(value, **kwargs):
                result = original(value, **kwargs)
                if kwargs.get('connection') is None and not replaced:
                    store.root.rename(store.root.parent / 'old')
                    successor = type(store)(store.root)
                    successor.init()
                    if isinstance(successor, SQLiteSwarmStore):
                        with successor._session() as c:
                            c.execute("INSERT INTO jobs(id,data) VALUES(?,?)", (job.id,json.dumps(to_jsonable(job))))
                    else:
                        successor.save_job(job)
                    successor.save_task(task)
                    replaced.append(successor)
                return result

            with patch.object(store, 'validate_job_ref', side_effect=validating):
                with self.assertRaises(StoreIdentityError):
                    store.request_cancellation(ref, 'race', [task_binding(task)])
            successor = replaced[0]
            self.assertIsNone(successor.get_cancellation_receipt(successor.job_ref(job.id), 'race'))

    def test_crash_after_bootstrap_reopens_same_identity(self):
        with TemporaryDirectory() as tmp:
            code = ("import os,sys; from puppetmaster.sqlite_store import SQLiteSwarmStore; "
                    "s=SQLiteSwarmStore(sys.argv[1]); s.init(); print(s.incarnation,flush=True); os._exit(0)")
            result = subprocess.run([sys.executable, '-c', code, tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            store = SQLiteSwarmStore(tmp)
            store.attach()
            self.assertEqual(store.incarnation, result.stdout.strip())

    def test_history_deletion_reset_reopen_and_file_repair(self):
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            task = Task(job_id=job.id, role='explore', instruction='private')
            store.save_task(task)
            for index, adapter in enumerate(('codex', 'local')):
                attempt = ExecutionAttempt(job.id,task.id,'run'+str(index),'a'+str(index),'now',adapter)
                store.record_attempt(attempt)
                store.record_usage_observation(UsageObservation(job.id,attempt.attempt_id,'exit','process','now',returncode=1-index))
            first = store.list_attempt_refs(ref, limit=1)
            store.reset_subgraph(job.id, [task.id])
            reopened = type(store)(store.root)
            next_page = reopened.list_attempt_refs(ref, cursor=first.next_cursor)
            self.assertEqual(len(next_page.items), 1)
            self.assertEqual(reopened.historical_evidence_counts(ref).captured_process_outcomes, 2)
            if store.backend_name == 'file':
                from puppetmaster.projections import connection
                with connection(store) as c:
                    c.execute("INSERT INTO projection_pending VALUES('crashed-history-write')")
                self.assertEqual(store.list_attempt_refs(ref).outcome, 'unavailable')
                self.assertIsNone(store.historical_evidence_counts(ref).captured_attempts)
                identity = store.incarnation
                store.repair_metadata_index()
                self.assertEqual(store.incarnation, identity)
                self.assertEqual(store.historical_evidence_counts(ref).captured_attempts, 2)
                self.assertEqual(store.list_attempt_refs(ref, cursor=first.next_cursor).outcome, 'cursor_expired')
            store.delete_job(job.id)
            self.assertEqual(store.list_attempt_refs(ref).outcome, 'unavailable')
            self.assertIsNone(store.historical_evidence_counts(ref).captured_attempts)

    def test_metadata_queries_do_not_read_sources_or_scan_other_jobs(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            job = store.create_job('target')
            other = store.create_job('other')
            with store._writer_scope() as c:
                for index in range(1500):
                    attempt = ExecutionAttempt(other.id,'t','r',str(index),'now','codex')
                    c.execute("INSERT INTO execution_attempts VALUES(?,?,?,?)", (other.id,str(index),'t',json.dumps(to_jsonable(attempt))))
                value = dict(to_jsonable(ExecutionAttempt(job.id,'t','r','target','now','codex')),
                             instruction='secret'*200000, stdout='secret'*200000)
                c.execute("INSERT INTO execution_attempts VALUES(?,?,?,?)", (job.id,'target','t',json.dumps(value)))
            from puppetmaster.projections import connection
            steps = []

            @contextmanager
            def guarded(*args, **kwargs):
                with connection(*args, **kwargs) as c:
                    c.set_authorizer(lambda action, table, *rest: sqlite3.SQLITE_DENY
                        if action == sqlite3.SQLITE_READ and table in {'jobs','tasks','runs','execution_attempts','usage_observations','artifacts'}
                        else sqlite3.SQLITE_OK)
                    c.set_progress_handler(lambda: steps.append(1) or 0, 100)
                    yield c

            ref = store.job_ref(job.id)
            with patch('puppetmaster.projections.connection', guarded):
                page = store.list_attempt_refs(ref, max_scan=1)
                counts = store.historical_evidence_counts(ref)
            self.assertEqual(page.items[0].facts['attempt_id'], 'target')
            self.assertEqual(counts.captured_attempts, 1)
            self.assertLess(len(steps), 30, "query scanned unrelated history")
            self.assertNotIn('secret', json.dumps(to_jsonable(page)))

    def test_historical_lock_is_unavailable_and_retry_preserves_cursor(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            job = store.create_job('job')
            for index in range(2):
                store.record_attempt(ExecutionAttempt(job.id,'t','r',str(index),'now','codex'))
            ref = store.job_ref(job.id)
            token = store.list_attempt_refs(ref, limit=1).next_cursor
            from puppetmaster.projections import connection

            @contextmanager
            def no_wait(*args, **kwargs):
                with connection(*args, **kwargs) as c:
                    c.execute('PRAGMA busy_timeout=0')
                    yield c

            blocker = store.connect()
            try:
                blocker.execute('PRAGMA locking_mode=EXCLUSIVE')
                blocker.execute('BEGIN EXCLUSIVE')
                with patch('puppetmaster.projections.connection', no_wait):
                    page = store.list_attempt_refs(ref, cursor=token)
                    self.assertEqual(page.outcome, 'unavailable')
                    self.assertEqual(page.next_cursor, token)
                    self.assertEqual(store.historical_evidence_counts(ref).outcome, 'unavailable')
            finally:
                blocker.close()
            self.assertEqual(len(store.list_attempt_refs(ref, cursor=token).items), 1)

    def test_typescript_wire_fixtures(self):
        import shutil
        if not shutil.which('tsc'):
            self.skipTest('TypeScript compiler unavailable')
        declarations = Path('clients/typescript/puppetmaster.ts').read_text().split('export interface LegacyJobRef',1)[1]
        declarations = 'export interface LegacyJobRef' + declarations
        fixtures = []
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            store.record_attempt(ExecutionAttempt(job.id,'t','r','a','now','codex'))
            store.record_usage_observation(UsageObservation(job.id,'a','exit','process','now',returncode=0,timed_out=False))
            page = store.list_process_outcome_refs(ref)
            self.assertIs(page.items[0].facts['timed_out'], False)
            fixtures.append(('HistoricalPage', page))
            fixtures.append(('HistoricalPage', store.list_process_outcome_refs(JobRef(job.id, ref.state_id))))
            from puppetmaster.contracts import CompletionReceipt
            fixtures.append(('CompletionReceipt', CompletionReceipt(ref, 'run', None, 'unavailable')))
            fixtures.append(('MetadataPage', store.list_job_summaries()))
            fixtures.append(('HistoricalCounts', store.historical_evidence_counts(ref)))
            fixtures.append(('JobRef', ref))
            fixtures.append(('JobRef', JobRef(job.id, ref.state_id)))
        for index, (name, value) in enumerate(fixtures):
            declarations += '\nconst value%d: %s = %s;' % (index,name,json.dumps(to_jsonable(value)))
        declarations += '\n// @ts-expect-error v2 requires incarnation\nconst bad: JobRef = {job_id:"j",state_id:"s",version:2};\n'
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'wire.ts'
            source.write_text(declarations)
            result = subprocess.run(['tsc','--noEmit','--strict','--skipLibCheck',str(source)], capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_cli_and_mcp_keep_incarnation_across_subprocess_boundary(self):
        from puppetmaster.mcp_server import run_cli, job_schema
        from puppetmaster.cli._parser import build_parser
        for store, job in self.stores():
            ref = store.job_ref(job.id)
            command = [sys.executable, '-m', 'puppetmaster', '--state-dir', str(store.root),
                       '--backend', store.backend_name, '--job-ref', json.dumps(ref.as_dict()),
                       'await', job.id, '--json', '--timeout-seconds', '0.001']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertIn(result.returncode, (0,1), result.stderr)
            self.assertEqual(json.loads(result.stdout)['job_ref'], ref.as_dict())
            with patch('puppetmaster.mcp_server.subprocess.run', return_value=subprocess.CompletedProcess([],0,'{}','')) as run:
                run_cli(['status', job.id], dict(state_dir=str(store.root), job_ref=ref.as_dict()))
            args = run.call_args.args[0]
            parsed = build_parser().parse_args(args[3:])
            self.assertEqual(json.loads(parsed.job_ref), ref.as_dict())
        schema = job_schema()['properties']['job_ref']
        self.assertEqual(schema['required'], ['job_id', 'state_id'])
        # The schema is loose; JobRef parses the shape at the boundary.
        from puppetmaster.models import JobRef
        with self.assertRaises(ValueError):
            JobRef(**{**ref.as_dict(), 'version': 2, 'incarnation': None})
        with self.assertRaises(ValueError):
            JobRef(job_id='j', state_id='s', version=1, incarnation=str(__import__('uuid').uuid4()))

    def test_child_attach_refuses_replaced_launch_identity(self):
        from puppetmaster.identity import prepare_launch
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'state'
            incarnation = prepare_launch(root, 'sqlite')
            command = [sys.executable, '-m', 'puppetmaster', '--state-dir', str(root),
                       '--store-incarnation', incarnation, 'state']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            root.rename(Path(tmp)/'old')
            replacement = SQLiteSwarmStore(root)
            replacement.init()
            identity = replacement.incarnation
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('store replaced', result.stderr)
            self.assertEqual(replacement.incarnation, identity)

    def test_concurrent_prepare_launch_ensures_current_schema_once(self):
        from puppetmaster.identity import prepare_launch

        original = SQLiteSwarmStore._execute_schema_script
        schema_ensures = []

        def counted(connection, script):
            schema_ensures.append(root)
            return original(connection, script)

        with TemporaryDirectory() as tmp, patch.object(
            SQLiteSwarmStore, '_execute_schema_script', new=staticmethod(counted)
        ):
            root = Path(tmp) / 'state'
            with ThreadPoolExecutor(max_workers=8) as pool:
                incarnations = list(pool.map(lambda _: prepare_launch(root, 'sqlite'), range(16)))

        self.assertEqual(len(set(incarnations)), 1)
        self.assertEqual(schema_ensures, [root])

    def test_completion_requires_explicit_incarnation(self):
        for store, job in self.stores():
            task = Task(job_id=job.id, role='explore', instruction='private')
            store.save_task(task)
            run = AgentRun(job.id, task.id, task.role, 'worker')
            with self.assertRaises(StoreIdentityError):
                store.submit_completion(task, run, [], {})
            legacy = JobRef(job.id, state_identity(store.root))
            with self.assertRaises(StoreIdentityError):
                store.submit_completion(task, run, [], {}, job_ref=legacy)

    def test_file_upgrade_missing_identity_and_history_backfill(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            job = store.create_job('legacy')
            store.record_attempt(ExecutionAttempt(job.id,'t','r','a','now','codex'))
            identity = store.incarnation
            with closing(sqlite3.connect(store.root/'metadata.sqlite3')) as c, c:
                c.execute("DELETE FROM historical_refs")
                c.execute("DELETE FROM projection_meta WHERE key='history_version'")
            self.assertEqual(SwarmStore(tmp).list_attempt_refs(store.job_ref(job.id)).outcome, 'unavailable')
            reopened = SwarmStore(tmp)
            reopened.init()
            self.assertEqual(reopened.incarnation, identity)
            self.assertEqual(reopened.historical_evidence_counts(reopened.job_ref(job.id)).captured_attempts, 1)
            with closing(sqlite3.connect(store.root/'metadata.sqlite3')) as c, c:
                c.execute("DELETE FROM projection_meta WHERE key='incarnation'")
            with self.assertRaises(StoreIdentityError):
                SwarmStore(tmp).init()

    def test_python39_syntax(self):
        for path in Path('puppetmaster').rglob('*.py'):
            ast.parse(path.read_text(), feature_version=(3,9))


if __name__ == '__main__':
    unittest.main()
