"""Budget admission at real WorkerRuntime and provider invocation boundaries."""
import sys
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from test_invocation_accounting import RuntimeAccountingContract, Adapter, CliAdapter, ProviderAdapter, verification
from puppetmaster.adapters import StreamedProcess
from puppetmaster.budget import BudgetPolicy, BudgetLiability, BudgetAdmissionError
from puppetmaster.invocation import execution_scope, invocation
from puppetmaster.models import TaskStatus, new_id
from puppetmaster.providers import AssistantTurn, ProviderError
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


class BudgetRuntimeContract:
    store_type = SwarmStore
    setUp = RuntimeAccountingContract.setUp
    run_adapter = RuntimeAccountingContract.run_adapter
    records = RuntimeAccountingContract.records

    def configure(self, policy, **payload):
        self.store.save_job(replace(self.job, budget_policy=policy))
        self.task = replace(self.task, payload={**self.task.payload, **payload})
        self.store.save_task(self.task)

    def snapshot(self):
        return self.store_type(self.root).budget_snapshot(self.job.id)

    def test_stream_terminal_controls_budget_settlement(self):
        from puppetmaster import providers
        protocols = [
            (providers._openai_chat_stream,
             [{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0}}], '[DONE]'),
            (providers._openai_chat_stream,
             [{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'stop'}]}],
             '[DONE]'),
            (providers._openai_chat_stream,
             [{'choices': [{'delta': {'content': 'partial'}}],
               'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0}}], '[DONE]'),
            (providers._anthropic_chat_stream,
             [{'type': 'message_start', 'message': {'usage': {
                 'input_tokens': 0, 'output_tokens': 0, 'cost': 0}}},
              {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'partial'}}],
             {'type': 'message_stop'}),
            (providers._openai_responses_chat_stream,
             [{'type': 'response.output_text.delta', 'delta': 'partial'}],
             {'type': 'response.completed', 'response': {'status': 'completed',
                 'usage': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}}}),
        ]
        for parser, chunks, terminal in protocols:
            for terminated in (False, True):
                with self.subTest(parser=parser.__name__, terminated=terminated):
                    # Each case has its own durable job and budget.
                    self.job = self.store.create_job('stream budget')
                    self.task = replace(self.task, job_id=self.job.id, id=new_id("task"))
                    self.configure(BudgetPolicy(max_tokens_in=10), billing='api',
                                   budget_allowance={'tokens_in': 10})
                    events = chunks + ([terminal] if terminated else [])
                    wire = ''.join('data: ' + (e if isinstance(e, str) else json.dumps(e)) + '\n\n'
                                   for e in events).encode()
                    with mock.patch.object(providers, '_open_stream', return_value=io.BytesIO(wire)):
                        turn = parser(base_url='https://test', api_key=None, model='test',
                                      messages=[], tools=None, extra={}, headers={}, timeout=1, on_delta=None)
                    self.assertEqual(turn.accounting_complete, terminated)
                    with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
                        with mock.patch('puppetmaster.adapters.agentic.provider_chat', return_value=turn), \
                                mock.patch('puppetmaster.adapters.agentic.get_provider_circuit_breaker'), \
                                mock.patch('puppetmaster.rate_limit_state.admit_or_raise'):
                            ProviderAdapter()._provider_call(provider='openai', model='test',
                                messages=[], tools=None, extra={}, timeout=1, max_retries=0)
                        state = self.snapshot()['reservations'][0]['state']
                        if not terminated or not turn.accounting_usage:
                            self.assertEqual(state, 'pending_reconciliation')
                            # An unsettled attempt never settles here: the next one
                            # waits for it, then still fails closed.
                            with self.assertRaisesRegex(BudgetAdmissionError, "waited"), \
                                    mock.patch('puppetmaster.invocation.DEFAULT_BUDGET_WAIT_SECONDS', 0.2), \
                                    mock.patch('puppetmaster.invocation.BUDGET_WAIT_POLL_SECONDS', 0.05):
                                with invocation():
                                    self.fail('truncated stream reopened capped admission')
                        else:
                            self.assertEqual(state, 'settled')

    def test_boundary_guard_preserves_legacy_and_resets_after_exception(self):
        from puppetmaster.invocation import check_external_dispatch
        lost = mock.Mock(return_value=True)
        with execution_scope(self.store, SimpleNamespace(id='legacy'), self.task, lease_lost=lost):
            with invocation():
                check_external_dispatch()
        lost.assert_not_called()
        self.configure(BudgetPolicy(max_attempts=2))
        lost.return_value = False
        with execution_scope(self.store, SimpleNamespace(id='budgeted'), self.task, lease_lost=lost):
            with self.assertRaises(BudgetAdmissionError), invocation():
                lost.return_value = True
                check_external_dispatch()
            check_external_dispatch()
        check_external_dispatch()

    def test_lease_loss_in_provider_preparation_blocks_network(self):
        from puppetmaster import providers
        self.configure(BudgetPolicy(max_attempts=2))
        original = providers._apply_openai_explicit_cache
        prepared = []
        def prepare(*args, **kwargs):
            result = original(*args, **kwargs)
            prepared.append(True)
            self.runtime._lease_lost.set()
            return result
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
                mock.patch('puppetmaster.adapters.agentic.get_provider_circuit_breaker'), \
                mock.patch('puppetmaster.rate_limit_state.admit_or_raise'), \
                mock.patch.object(providers, '_apply_openai_explicit_cache', side_effect=prepare), \
                mock.patch('urllib.request.urlopen') as network:
            self.run_adapter(ProviderAdapter())
        self.assertTrue(prepared)
        network.assert_not_called()
        self.assertNotEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_lease_loss_in_http_request_construction_blocks_all_transports(self):
        from puppetmaster import providers, bedrock
        transports = [providers._post_json, providers._open_stream,
                      bedrock._post_bedrock, bedrock._open_bedrock_event_stream,
                      bedrock._get_bedrock]
        import urllib.request
        original = urllib.request.Request
        for transport in transports:
            with self.subTest(transport=transport.__name__):
                self.runtime._lease_lost.clear()
                self.job = self.store.create_job('transport lease')
                self.task = replace(self.task, job_id=self.job.id, id=new_id('task'))
                self.configure(BudgetPolicy(max_attempts=2))
                def prepare(*args, **kwargs):
                    request = original(*args, **kwargs)
                    self.runtime._lease_lost.set()
                    return request
                with execution_scope(self.store, SimpleNamespace(id='run'), self.task,
                                     lease_lost=self.runtime._lease_lost.is_set):
                    with self.assertRaises(BudgetAdmissionError), invocation():
                        with mock.patch('urllib.request.Request', side_effect=prepare), \
                                mock.patch('urllib.request.urlopen') as network:
                            kwargs = {} if transport is bedrock._get_bedrock else {'body': {}}
                            transport('https://test', headers={}, timeout=1, **kwargs)
                    network.assert_not_called()
                self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_lease_loss_in_cli_environment_preparation_blocks_spawn(self):
        from puppetmaster.adapters import _streaming
        self.configure(BudgetPolicy(max_attempts=2))
        def lose(*args, **kwargs):
            self.runtime._lease_lost.set()
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task,
                             lease_lost=self.runtime._lease_lost.is_set):
            with self.assertRaises(BudgetAdmissionError), invocation():
                with mock.patch.object(_streaming, 'stamp_worker_env', side_effect=lose), \
                        mock.patch.object(_streaming.subprocess, 'Popen') as spawn:
                    _streaming.run_streamed_subprocess(command=['test'], env={}, task=self.task,
                        sidecar_name='lease', timeout_seconds=1)
            spawn.assert_not_called()
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_bedrock_stream_requires_message_stop(self):
        from puppetmaster.bedrock import _assistant_turn_from_converse_stream
        for stopped in (False, True):
            events = [{'contentBlockDelta': {'delta': {'text': 'partial'}}},
                      {'metadata': {'usage': {'inputTokens': 0, 'outputTokens': 0}}}]
            if stopped:
                events.append({'messageStop': {'stopReason': 'end_turn'}})
            turn = _assistant_turn_from_converse_stream(iter(events))
            self.assertEqual(turn.accounting_complete, stopped)
            self.assertEqual(turn.text, 'partial')

    def test_lease_loss_during_persistence_prevents_external_call(self):
        for operation in ('adopt_dispatch', 'reconcile_reservation', 'record_attempt'):
            with self.subTest(operation=operation):
                self.runtime._lease_lost.clear()
                self.job = self.store.create_job('lease race')
                self.task = replace(self.task, job_id=self.job.id, id=new_id("task"))
                self.runtime.job_id = self.job.id
                self.configure(BudgetPolicy(max_attempts=2))
                original = getattr(self.store, operation)
                def lose(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.runtime._lease_lost.set()
                    return result
                call = mock.Mock(return_value=[])
                with mock.patch.object(self.store, operation, side_effect=lose):
                    self.run_adapter(Adapter(call))
                call.assert_not_called()
                self.assertNotEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)
                self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_codex_terminal_usage_settles_but_missing_usage_stays_unknown(self):
        for billing, usage, settled in (
                ('api', {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}, True),
                ('plan', {'input_tokens': 0, 'output_tokens': 0}, True),
                ('plan', {'input_tokens': 0}, False),
                ('plan', {}, False), ('api', None, False)):
            with self.subTest(billing=billing, usage=usage):
                self.job = self.store.create_job('codex terminal')
                self.task = replace(self.task, job_id=self.job.id, id=new_id("task"))
                self.runtime.job_id = self.job.id
                self.configure(BudgetPolicy(max_tokens_in=10), billing=billing,
                               budget_allowance={'tokens_in': 10})
                event = {'type': 'turn.completed'}
                if usage is not None:
                    event['usage'] = usage
                self.run_adapter(CliAdapter(lambda task: StreamedProcess(
                    returncode=0, stdout=json.dumps(event), stderr='')))
                self.assertEqual(self.snapshot()['reservations'][0]['state'],
                                 'settled' if settled else 'pending_reconciliation')

    def test_cap_blocks_before_adapter_and_marks_failure(self):
        self.configure(BudgetPolicy(max_attempts=0))
        call = mock.Mock()
        self.run_adapter(Adapter(call))
        call.assert_not_called()
        self.assertEqual(self.records(), ([], []))
        self.assertEqual(self.snapshot()['reservations'], [])
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.FAILED)

    def test_every_indeterminate_unit_fails_closed(self):
        for cap in ('max_usd', 'max_tokens_in', 'max_tokens_out', 'max_elapsed_seconds'):
            with self.subTest(cap=cap):
                self.store.reset_subgraph(self.job.id, [self.task.id])
                self.configure(BudgetPolicy(**{cap: 10}))
                call = mock.Mock()
                self.run_adapter(Adapter(call))
                call.assert_not_called()

    def test_reset_reopen_cannot_erase_attempt_cap(self):
        self.configure(BudgetPolicy(max_attempts=1))
        self.run_adapter(Adapter(lambda task: [verification(task)]))
        self.store.reset_subgraph(self.job.id, [self.task.id])
        call = mock.Mock()
        self.run_adapter(Adapter(call))
        call.assert_not_called()
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_reuse_does_not_reserve(self):
        self.configure(BudgetPolicy(max_attempts=0))
        RuntimeAccountingContract.test_successful_persisted_reuse_creates_no_attempt(self)
        self.assertEqual(self.snapshot()['reservations'], [])

    def test_authoritative_zero_and_idempotent_callback(self):
        self.configure(BudgetPolicy(max_usd=1, max_tokens_in=10, max_tokens_out=10,
                                    max_elapsed_seconds=30), billing='api', budget_allowance={
            'billing': 'api', 'cost_state': 'known', 'api_usd': 1,
            'tokens_in': 10, 'tokens_out': 10, 'elapsed_seconds': 30})
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
            with invocation() as capture:
                self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')
                capture.observe({'cost_usd': 0, 'tokens_in': 0, 'tokens_out': 0},
                                cost_basis='api', final=True)
            capture.reconcile(True)
        snap = self.snapshot()
        self.assertEqual(snap['reservations'][0]['state'], 'settled')
        self.assertEqual(snap['totals']['marginal_usd']['total'], 0)
        self.assertEqual(len(snap['reservations'][0]['reconciliations']), 2)

    def test_settled_consumption_blocks_next_allowance(self):
        self.configure(BudgetPolicy(max_usd=1), billing='api', budget_allowance={
            'billing': 'api', 'cost_state': 'known', 'api_usd': 0.75})
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
            with invocation() as capture:
                capture.observe({'cost_usd': 0.5}, cost_basis='api', final=True)
            with self.assertRaises(BudgetAdmissionError):
                with invocation():
                    self.fail('remaining budget cannot cover allowance')
        self.assertEqual(self.snapshot()['totals']['marginal_usd']['total'], 0.5)
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)

    def test_estimated_price_does_not_settle_api_charge(self):
        self.configure(BudgetPolicy(max_attempts=1), billing='api')
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
            with invocation() as capture:
                capture.observe({'cost_usd': 0.1, 'cost_state': 'estimated'},
                                cost_basis='api', final=True)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')
        self.assertIsNone(self.snapshot()['totals']['marginal_usd']['total'])

    def test_conflicting_callback_cannot_settle_stale_snapshot(self):
        self.configure(BudgetPolicy(max_attempts=1), billing='api')
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
            with invocation() as capture:
                capture.observe({'cost_usd': 0}, cost_basis='api', final=True)
                capture.observe({'cost_usd': 1}, cost_basis='api', final=True)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_partial_and_exception_remain_pending(self):
        self.configure(BudgetPolicy(max_attempts=3), billing='api')
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
            with invocation() as capture:
                capture.observe({'tokens_in': 4}, final=True)
            with self.assertRaises(TimeoutError):
                with invocation():
                    raise TimeoutError('unknown remote outcome')
        self.assertTrue(all(r['state'] == 'pending_reconciliation' for r in self.snapshot()['reservations']))

    def test_lease_loss_before_dispatch_releases_only_unadopted(self):
        self.configure(BudgetPolicy(max_attempts=1))
        lost = mock.Mock(side_effect=[False, True])
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task, lease_lost=lost):
            with self.assertRaises(BudgetAdmissionError):
                with invocation():
                    self.fail('dispatched')
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'released')
        self.assertEqual(self.records(), ([], []))

    def test_lease_loss_after_dispatch_preserves_pending(self):
        self.configure(BudgetPolicy(max_attempts=1), billing='plan')
        lost = mock.Mock(return_value=False)
        with execution_scope(self.store, SimpleNamespace(id='run'), self.task, lease_lost=lost):
            with invocation() as capture:
                lost.return_value = True
                capture.observe({'tokens_in': 0, 'tokens_out': 0}, final=True)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_adoption_failure_never_calls_external_adapter(self):
        self.configure(BudgetPolicy(max_attempts=1))
        call = mock.Mock()
        with mock.patch.object(self.store, 'adopt_dispatch', side_effect=OSError('disk')):
            self.run_adapter(Adapter(call))
        call.assert_not_called()
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'reserved')

    def test_settlement_failure_retains_durable_fence_and_result(self):
        self.configure(BudgetPolicy(max_attempts=2))
        original = self.store.reconcile_reservation
        def reconcile(*args, **kwargs):
            if kwargs['reconciliation_id'] == 'dispatch:outcome':
                raise OSError('disk')
            return original(*args, **kwargs)
        with mock.patch.object(self.store, 'reconcile_reservation', side_effect=reconcile):
            self.run_adapter(Adapter(lambda task: [verification(task)]))
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)
        record = self.snapshot()['reservations'][0]
        self.assertEqual(record['state'], 'pending_reconciliation')
        self.store.reconcile_reservation(self.job.id, record['attempt']['attempt_id'],
            reconciliation_id='recovery', liability=BudgetLiability(billing='api',
            cost_state='known', api_usd=0), final=True, evidence='authoritative recovered billing')
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'settled')

    def test_provider_retry_is_separately_admitted(self):
        self.configure(BudgetPolicy(max_attempts=1))
        with mock.patch('puppetmaster.adapters.agentic.provider_chat',
                        side_effect=ProviderError('retry', status=500, reason='server_error')) as call, \
                mock.patch('puppetmaster.adapters.agentic.time.sleep'), \
                mock.patch('puppetmaster.adapters.agentic.provider_key_pool', return_value=['test']):
            self.run_adapter(ProviderAdapter())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)

    def test_provider_retry_success_has_two_identities(self):
        self.configure(BudgetPolicy(max_attempts=2))
        error = ProviderError('retry', status=500, reason='http_status:500')
        turn = AssistantTurn(text='ok', tool_calls=[], usage={'prompt_tokens': 2})
        with mock.patch('puppetmaster.adapters.agentic.provider_chat', side_effect=[error, turn]) as call, \
                mock.patch('puppetmaster.adapters.agentic.time.sleep'), \
                mock.patch('puppetmaster.adapters.agentic.get_provider_circuit_breaker'), \
                mock.patch('puppetmaster.rate_limit_state.admit_or_raise'):
            self.run_adapter(ProviderAdapter())
        self.assertEqual(call.call_count, 2)
        self.assertEqual(len({r['attempt']['attempt_id'] for r in self.snapshot()['reservations']}), 2)

    def test_failover_preserves_both_reservations(self):
        self.configure(BudgetPolicy(max_attempts=2))
        RuntimeAccountingContract.test_failed_failover_retains_each_provider(self)
        self.assertEqual(self.snapshot()['totals']['attempts'], 2)

    def test_cli_timeout_cannot_settle_final_looking_stdout(self):
        self.configure(BudgetPolicy(max_attempts=1), billing='api')
        result = StreamedProcess(returncode=1, timed_out=True, stderr='', stdout=
            '{"type":"result","total_cost_usd":0,"usage":{"input_tokens":0,"output_tokens":0}}')
        self.run_adapter(CliAdapter(lambda task: result))
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')

    def test_crash_without_finally_keeps_pending_after_reopen(self):
        self.configure(BudgetPolicy(max_attempts=1))
        script = """
import os, sys
from types import SimpleNamespace
from puppetmaster.store import SwarmStore
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.invocation import execution_scope, invocation
store = (SQLiteSwarmStore if sys.argv[1] == 'sqlite' else SwarmStore)(sys.argv[2])
task = store.get_task_by_id(sys.argv[3])
with execution_scope(store, SimpleNamespace(id='crashed-run'), task):
    with invocation():
        os._exit(23)
"""
        result = subprocess.run([sys.executable, '-c', script,
            'sqlite' if self.store_type is SQLiteSwarmStore else 'file', str(self.root), self.task.id],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(self.snapshot()['reservations'][0]['state'], 'pending_reconciliation')
        call = mock.Mock()
        self.run_adapter(Adapter(call))
        call.assert_not_called()

    def test_concurrent_invocations_cannot_both_take_last_attempt(self):
        self.configure(BudgetPolicy(max_attempts=1))
        barrier = Barrier(2)
        dispatched = []
        def call(index):
            store = self.store_type(self.root)
            barrier.wait(timeout=5)
            try:
                with execution_scope(store, SimpleNamespace(id=f'run-{index}'), self.task):
                    with invocation():
                        dispatched.append(index)
                        return True
            except (BudgetAdmissionError, RuntimeError):
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            admitted = list(pool.map(call, [1, 2]))
        self.assertEqual(sum(admitted), 1)
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)

    def test_lease_loss_exception_does_not_overwrite_successor(self):
        self.configure(BudgetPolicy(max_attempts=1))
        def call(task):
            self.store.save_task(replace(task, status=TaskStatus.COMPLETE))
            self.runtime._lease_lost.set()
            raise RuntimeError('obsolete worker')
        self.run_adapter(Adapter(call))
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)


class FileRuntimeBudgetTests(BudgetRuntimeContract, unittest.TestCase):
    def test_contender_between_reservation_and_adoption(self):
        self.configure(BudgetPolicy(max_attempts=1))
        reserved, contender_tried, winner_done = Event(), Event(), Event()
        winner = self.store_type(self.root)
        contender = self.store_type(self.root)
        reserve = winner.reserve_dispatch
        acquire = contender.acquire_lock
        records = contender._budget_records

        def pause_after_reservation(*args, **kwargs):
            result = reserve(*args, **kwargs)
            reserved.set()
            self.assertTrue(contender_tried.wait(5))
            return result

        def signal_acquisition(*args, **kwargs):
            result = acquire(*args, **kwargs)
            contender_tried.set()
            return result

        def hold_contender_lock(*args, **kwargs):
            self.assertTrue(winner_done.wait(5))
            return records(*args, **kwargs)

        def call(store, index):
            try:
                with execution_scope(store, SimpleNamespace(id=f'run-{index}'), self.task):
                    with invocation():
                        return True
            except (BudgetAdmissionError, RuntimeError) as exc:
                return str(exc)
            finally:
                if store is winner:
                    winner_done.set()

        with mock.patch.object(winner, 'reserve_dispatch', side_effect=pause_after_reservation), \
                mock.patch.object(contender, 'acquire_lock', side_effect=signal_acquisition), \
                mock.patch.object(contender, '_budget_records', side_effect=hold_contender_lock), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(call, winner, 1)
            self.assertTrue(reserved.wait(5))
            second = pool.submit(call, contender, 2)
            outcomes = [first.result(timeout=5), second.result(timeout=5)]
        self.assertIs(outcomes[0], True, outcomes)
        self.assertIsNot(outcomes[1], True, outcomes)
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)


class SQLiteRuntimeBudgetTests(BudgetRuntimeContract, unittest.TestCase):
    store_type = SQLiteSwarmStore

    def test_transient_writer_lock_admits_final_attempt_once(self):
        self.configure(BudgetPolicy(max_attempts=1))
        blocker = self.store.connect()
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN IMMEDIATE")
        self.store.busy_timeout_ms = 0
        dispatched = []
        with mock.patch.object(self.store, '_sleep_lock_backoff',
                               side_effect=lambda attempt: blocker.rollback()) as retry:
            with execution_scope(self.store, SimpleNamespace(id='run'), self.task):
                with invocation():
                    dispatched.append('first')
                with self.assertRaises(BudgetAdmissionError):
                    with invocation():
                        dispatched.append('second')
        retry.assert_called_once_with(0)
        self.assertEqual(dispatched, ['first'])
        self.assertEqual(self.snapshot()['totals']['attempts'], 1)
