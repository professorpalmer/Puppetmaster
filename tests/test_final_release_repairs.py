from contextlib import closing
"""Final release regressions: real producer payloads and read-only receipts."""
import json
import sqlite3
import subprocess
import sys
import time
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401

from readonly_fixtures import damaged_sidecars

from puppetmaster import readonly
from puppetmaster.adapters import CodexAdapter, StreamedProcess
from puppetmaster.adapters.agentic import AgenticAdapter
from puppetmaster.contracts import CancellationReceipt, EffectReceipt, TaskBinding
from puppetmaster.identity import StoreIdentityError
from puppetmaster.models import ArtifactType, Task
from puppetmaster.projections import connection
from puppetmaster.providers import AssistantTurn
from puppetmaster.selected_economics import freeze
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.store_contracts import _save


class FinalReleaseRepairs(unittest.TestCase):
    def test_receipts_never_mutate_delete_or_wal_source(self):
        for cls in (SwarmStore, SQLiteSwarmStore):
            for mode in ('delete', 'wal'):
                with self.subTest(backend=cls.backend_name, mode=mode), TemporaryDirectory() as tmp:
                    store = cls(Path(tmp) / 'state')
                    job = store.create_job('receipts')
                    ref = store.job_ref(job.id)
                    binding = TaskBinding('task', None, None, None)
                    cancel = CancellationReceipt(ref, 'cancel', (binding,), 'stale_binding', 1)
                    effect = EffectReceipt(ref, 'effect', 'digest', binding, 'run', 'attempt', 1,
                                           'not_dispatched', 'safe')
                    with connection(store) as c:
                        _save(c, 'cancel', ref, 'cancel', cancel)
                        _save(c, 'effect', ref, 'effect', effect)
                    path = store.root / ('state.sqlite3' if cls is SQLiteSwarmStore else 'metadata.sqlite3')
                    with closing(sqlite3.connect(path)) as c:
                        c.execute('PRAGMA journal_mode=' + mode)
                    path.chmod(0o644)
                    def snapshot():
                        return {p.name: (p.stat().st_mode, p.stat().st_size, p.stat().st_mtime_ns,
                                         p.stat().st_ctime_ns, p.read_bytes()) for p in store.root.iterdir() if p.is_file()}
                    before = snapshot()
                    statements = []
                    original = readonly.ReadConnection.execute
                    def trace(c, sql, parameters=()):
                        statements.append(sql)
                        return original(c, sql, parameters)
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(readonly.ReadConnection, 'execute', trace))
                        chmod = stack.enter_context(patch('os.chmod', side_effect=AssertionError('chmod')))
                        if cls is SQLiteSwarmStore:
                            stack.enter_context(patch.object(store, '_session', side_effect=AssertionError('writable session')))
                        self.assertEqual(store.get_effect_receipt(ref, 'effect'), effect)
                        self.assertEqual(store.get_cancellation_receipt(ref, 'cancel'), cancel)
                        self.assertIsNone(store.get_effect_receipt(ref, 'absent'))
                        self.assertIsNone(store.get_cancellation_receipt(ref, 'absent'))
                        chmod.assert_not_called()
                    self.assertTrue(statements)
                    self.assertTrue(all(sql.lstrip().upper().startswith(('SELECT', 'PRAGMA TABLE_INFO')) for sql in statements), statements)
                    self.assertEqual(snapshot(), before)
                    store.root.rename(Path(tmp) / 'old')
                    cls(store.root).init()
                    for getter in (store.get_effect_receipt, store.get_cancellation_receipt):
                        with self.assertRaises(StoreIdentityError):
                            getter(ref, 'absent')

    def test_helper_confirms_live_reader_and_attach_joins_it(self):
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            holder = sqlite3.connect(supervisor.db_path)
            try:
                holder.execute('BEGIN')
                holder.execute('SELECT * FROM metadata').fetchall()
                with damaged_sidecars(holder, supervisor.db_path):
                    verdicts = []
                    real_receive = readonly.ReadConnection._receive

                    def receive(conn):
                        try:
                            return real_receive(conn)
                        except sqlite3.OperationalError as err:
                            verdicts.append(getattr(err, 'sqlite_errorcode', None))
                            raise
                    # The helper re-asks until the attach window closes; 2s
                    # leaves a loaded Windows runner room to start it.
                    with self.assertRaises(readonly.ReadUnavailable) as caught, \
                            patch.object(readonly.ReadConnection, '_receive', receive):
                        readonly.connect(supervisor, timeout=2, attach_binding=True)
                    # Both lock APIs confirm a live conflicting reader: the
                    # missing sidecars still prevent a read, but the helper
                    # classifies it SQLITE_BUSY. Attach binding may still end in
                    # its own bounded ReadTimeout when the window closes mid-retry
                    # (test_accepted_retry_timeout_preserves_error_only_for_ordinary).
                    self.assertIn(5, verdicts)
                    if not isinstance(caught.exception, readonly.ReadTimeout):
                        self.assertEqual(getattr(caught.exception, 'sqlite_errorcode', None), 5)
            finally:
                holder.close()
            holder = sqlite3.connect(supervisor.db_path)
            holder.execute('BEGIN')
            holder.execute('SELECT * FROM metadata').fetchall()
            clock = [0.0]
            released = []
            def sleep(delay):
                clock[0] += max(delay, .3)
                if clock[0] > 1.2 and not released:
                    released.append(True)
                    holder.close()
            worker = SQLiteSwarmStore(tmp)
            with closing(holder), patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)), \
                    patch.object(worker, 'ensure_schema', side_effect=AssertionError('worker DDL')):
                worker.attach()
            # A live reader means live sidecars: attach joins the WAL instead
            # of waiting for the reader to leave.
            self.assertFalse(released)
            self.assertLess(clock[0], 5)
            self.assertEqual(worker.incarnation, supervisor.incarnation)

    def test_attach_unavailable_budget_and_identity_fence(self):
        for error, expected in ((readonly.ReadUnavailable('live sidecars'), readonly.ReadUnavailable),
                                (readonly.ReadUnavailable('source missing'), readonly.ReadUnavailable)):
            with TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                clock = [0.0]
                def sleep(delay):
                    clock[0] += delay
                with patch.object(readonly, 'ReadConnection', side_effect=error) as opens, \
                        patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)):
                    with self.assertRaises(expected):
                        readonly.connect(store, attach_binding=True, timeout=5)
                self.assertLessEqual(clock[0], 1.01)
                if 'missing' in str(error):
                    self.assertEqual(opens.call_count, 1)
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / 'state')
            store.ensure_schema()
            def replace(_):
                store.root.rename(Path(tmp) / 'old')
                SQLiteSwarmStore(store.root).ensure_schema()
            busy = sqlite3.OperationalError('database is locked')
            original = readonly.ReadConnection
            calls = [0]
            def open_once(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 1:
                    raise busy
                return original(*args, **kwargs)
            with patch.object(readonly, 'ReadConnection', side_effect=open_once), patch.object(readonly, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=replace)):
                with self.assertRaises(StoreIdentityError):
                    readonly.connect(store, attach_binding=True)

    def test_actual_adapter_payloads_preserve_selected_presence(self):
        cases = [({}, None, None, False), ({'input_tokens': 0, 'output_tokens': 0}, 0, 0, False),
                 ({'input_tokens': 123, 'output_tokens': 45}, 123, 45, False),
                 ({'input_tokens': 123}, 123, None, False),
                 ({'input_tokens': 12, 'output_tokens': 4, 'tokens_estimated': True}, 12, 4, True)]
        for name in ('codex', 'agentic'):
            for raw, tin, tout, estimated in cases:
                with self.subTest(adapter=name, raw=raw), TemporaryDirectory() as tmp:
                    task = Task(job_id='job', id='task', role='explore', instruction='inspect', adapter=name,
                                payload={'cwd': tmp, 'sandbox': 'read-only', 'disable_codegraph': True,
                                         'provider': 'openai', 'model': 'test-model'})
                    if name == 'codex':
                        events = '\n'.join(json.dumps(event) for event in [
                            {'type': 'turn.completed', 'usage': raw},
                            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"artifacts":[]}'}}])
                        clean = {'sha': 's', 'changed_files': [], 'untracked_files': [], 'diff': ''}
                        with patch('puppetmaster.adapters.resolve_command', return_value='/usr/bin/codex'), \
                                patch('puppetmaster.adapters.git_snapshot', return_value=clean), \
                                patch('puppetmaster.adapters.run_streamed_subprocess', return_value=StreamedProcess(returncode=0, stdout=events, stderr='')):
                            artifacts = CodexAdapter().run(task, 'goal', 'worker')
                    else:
                        turn = AssistantTurn(text='', usage={'prompt_tokens': raw.get('input_tokens', 0),
                            'completion_tokens': raw.get('output_tokens', 0)}, accounting_usage=raw,
                            tool_calls=[{'id':'submit', 'name':'submit_findings', 'arguments':{'artifacts':[]}}])
                        with patch.object(AgenticAdapter, '_provider_call', return_value=turn):
                            artifacts = AgenticAdapter().run(task, 'goal', 'worker')
                    artifact = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
                    payload = artifact.payload
                    self.assertIs(payload['tokens_estimated'], estimated)
                    self.assertEqual(payload['selected_facts']['version'], 1)
                    self.assertEqual(payload['selected_facts']['tokens_in'], tin)
                    self.assertEqual(payload['selected_facts']['tokens_out'], tout)
                    self.assertLess(len(json.dumps(payload['selected_facts'])), 512)
                    receipt = {'actual_cost': {'tasks': [{'task_id': task.id, 'priced': True,
                                                        'billing': 'api', 'marginal_cost_usd': .5}]}}
                    totals = freeze(receipt, [artifact])['totals']
                    self.assertEqual(totals['tokens_in']['total'], tin)
                    self.assertEqual(totals['tokens_out']['total'], tout)
                    self.assertEqual(totals['tokens_in']['state'], 'unknown' if tin is None else 'estimated' if estimated else 'measured')
                    self.assertEqual(totals['api_equivalent_cost_usd']['total'], .5 if tin is not None and tout is not None else None)

    def test_agentic_multiturn_presence_and_forced_submit(self):
        for partial in (False, True):
            with self.subTest(partial=partial), TemporaryDirectory() as tmp:
                task = Task(job_id='job', id='task', role='explore', instruction='inspect', adapter='agentic',
                            payload={'cwd': tmp, 'disable_codegraph': True, 'provider': 'openai',
                                     'model': 'test-model', 'max_turns': 1})
                turns = [
                    AssistantTurn(text='', usage={'prompt_tokens': 3, 'completion_tokens': 2},
                        accounting_usage={'prompt_tokens': 3, 'completion_tokens': 2},
                        tool_calls=[{'id': 'read', 'name': 'list_files', 'arguments': {}}]),
                    AssistantTurn(text='', usage={'prompt_tokens': 5, 'completion_tokens': 0},
                        accounting_usage={'prompt_tokens': 5, **({} if partial else {'completion_tokens': 0}),
                                          'tokens_estimated': True},
                        tool_calls=[{'id': 'submit', 'name': 'submit_findings', 'arguments': {'artifacts': []}}]),
                ]
                with patch.object(AgenticAdapter, '_provider_call', side_effect=turns):
                    artifacts = AgenticAdapter().run(task, 'goal', 'worker')
                payload = next(a.payload for a in artifacts if a.type == ArtifactType.VERIFICATION)
                self.assertTrue(payload['submit_forced_max_turns'])
                self.assertTrue(payload['tokens_estimated'])
                self.assertEqual(payload['selected_facts']['tokens_in'], 8)
                self.assertEqual(payload['selected_facts']['tokens_out'], None if partial else 2)


if __name__ == '__main__':
    unittest.main()
