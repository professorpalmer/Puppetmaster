"""Issue #136: selection-independent launch billing and honest valuation."""
import os
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401
import puppetmaster.preflight  # Import before detector patches bind module references.
from puppetmaster.model_registry import ModelSpec, save_registry, stamp_model_billing
from puppetmaster.platform_billing import BillingStatus, detect_codex_billing
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.workers import WorkerSpec
from puppetmaster.store import SwarmStore
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.models import Artifact, ArtifactType, JobStatus
from puppetmaster.usage import token_usage
from puppetmaster.invocation import execution_scope, invocation
from types import SimpleNamespace


class Issue136Tests(unittest.TestCase):
    def test_missing_home_does_not_block_custom_command_or_probe_ambient_auth(self):
        from puppetmaster.platform_billing import auth_context
        with patch.object(Path, 'home', side_effect=RuntimeError('Could not determine home directory')):
            context = auth_context(env={})
            self.assertIsNone(context.home)
            command = [r'C:\Program Files\node.exe', r'C:\tools\codex.js']
            with patch('puppetmaster.platform_billing._default_runner',
                       return_value=(0, 'Logged in using API key SECRET', '')) as run:
                status = detect_codex_billing(context=context, codex_command=command)
                self.assertEqual(status.billing, 'api')
                self.assertNotIn('SECRET', str(status))
                self.assertEqual(run.call_args.args[0], command + ['login', 'status'])
            with patch('puppetmaster.platform_billing._read_codex_auth',
                       side_effect=AssertionError('no ambient auth without home')):
                status = detect_codex_billing(context=context, run=lambda _: (127, '', ''))
                self.assertEqual(status.billing, 'unknown')
                self.assertFalse(status.healthy)

    def test_explicit_auth_home_works_without_process_home(self):
        with TemporaryDirectory() as tmp, patch.object(Path, 'home', side_effect=RuntimeError('no home')):
            home = Path(tmp)
            (home / 'auth.json').write_text('{"auth_mode":"apikey"}')
            status = detect_codex_billing(env={'CODEX_HOME': tmp}, run=lambda _: self.fail('unneeded probe'))
            self.assertEqual(status.billing, 'api')
            for env, given_home in (({}, Path('.')), ({'CODEX_HOME': '~/.codex'}, None),
                                    ({'HOME': '.', 'USERPROFILE': '.'}, None)):
                with patch('puppetmaster.platform_billing._read_codex_auth',
                           side_effect=AssertionError('relative or ambient auth path')):
                    status = detect_codex_billing(env=env, home=given_home, run=lambda _: (127, '', ''))
                    self.assertEqual(status.billing, 'unknown')
                    self.assertFalse(status.healthy)

    def test_direct_and_routed_persist_and_reopen(self):
        for backend in (SwarmStore, SQLiteSwarmStore):
            for billing in ('plan', 'api', 'unknown'):
                for routed in (False, True):
                    with self.subTest(backend=backend.__name__, billing=billing, routed=routed), TemporaryDirectory() as tmp:
                        model = ModelSpec(id='codex/gpt-5-5', adapter='codex', adapter_model_name='gpt-5-5', billing='unknown', input_per_mtok_usd=5, output_per_mtok_usd=30, capability_score=99)
                        registry = Path(tmp) / 'models.json'
                        save_registry([model], registry)
                        store = backend(Path(tmp) / 'state')
                        store.init()
                        job = store.create_job('inspect')
                        status = BillingStatus('codex', billing, True, 'SECRET', ['SECRET' * 2000])
                        with patch('puppetmaster.platform_billing.detect_adapter_billing', return_value=status), patch('puppetmaster.platform_billing.detect_adapter_billing_cached', return_value=status), patch('puppetmaster.preflight.adapter_cli_present', return_value=True), patch('puppetmaster.platform_lock.is_adapter_enabled', return_value=True):
                            task = Orchestrator(store)._create_tasks(job, [WorkerSpec(role='explore', instruction='inspect', adapter='codex', payload={'model':model.id, 'auto_route':routed, 'registry_path':str(registry), 'min_capability':1})])[0]
                        reopened = backend(Path(tmp) / 'state')
                        payload = reopened.get_task_by_id(task.id).payload
                        self.assertEqual(payload['billing'], billing)
                        self.assertEqual(payload['registry_billing'], 'unknown')
                        self.assertNotIn('SECRET', str(payload))
                        provenance = reopened.list_artifacts(job.id)[-1].payload
                        self.assertEqual(provenance['billing'], billing)
                        self.assertEqual(provenance['registry_billing'], 'unknown')
                        with execution_scope(reopened, SimpleNamespace(id='run'), task):
                            with invocation(adapter='codex', model='gpt-5-5') as call:
                                call.observe({'input_tokens':1000000, 'output_tokens':0}, final=True)
                        observations = reopened.list_usage_observations(job.id)
                        self.assertTrue(observations)
                        reopened.save_artifact(Artifact(job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION, created_by='test', confidence=1, evidence=['fixture'], payload={'check':'usage', 'result':'ok', 'model':model.adapter_model_name, **token_usage(sdk_usage={'input_tokens':1000000, 'output_tokens':0})}))
                        with patch('puppetmaster.cost.load_registry', return_value=[model]):
                            reopened.update_job_status(job.id, JobStatus.COMPLETE)
                        receipt = backend(Path(tmp) / 'state').get_job(job.id).cost_receipt
                        actual = receipt['actual_cost']
                        self.assertEqual(actual['total_marginal_cost_usd'], None if billing == 'unknown' else 0 if billing == 'plan' else 5)
                        self.assertEqual(actual['tasks'][0]['api_equivalent_cost_usd'], 5)
                        selected = reopened.get_selected_economics(reopened.job_ref(job.id))
                        self.assertIsNone(selected.totals.api_cost_usd.total)
                        if billing == 'plan':
                            self.assertEqual(selected.totals.plan_marginal_cost_usd.total, 0)
                        else:
                            self.assertEqual(selected.totals.api_equivalent_cost_usd.total, 5)
                            self.assertEqual(selected.totals.api_equivalent_cost_usd.state, 'estimated')

    def test_executable_defaults_bill_effective_launch_in_both_stores(self):
        def auth_status(command, **kwargs):
            self.assertEqual(command[1:], ['login', 'status'])
            return (0, 'Logged in using API key' if command[0] == 'api-codex'
                    else 'Logged in using ChatGPT', '')

        for backend in (SwarmStore, SQLiteSwarmStore):
            for routed in (False, True):
                with self.subTest(backend=backend.__name__, routed=routed), TemporaryDirectory() as tmp:
                    model = ModelSpec(
                        id='codex/default-executable', adapter='codex',
                        adapter_model_name='test', billing='unknown',
                        payload_defaults={'executable': ['api-codex']},
                        input_per_mtok_usd=5, output_per_mtok_usd=30,
                        capability_score=99,
                    )
                    (Path(tmp) / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'access_token': 'SECRET'}}))
                    registry = Path(tmp) / 'models.json'
                    save_registry([model], registry)
                    store = backend(Path(tmp) / 'state')
                    store.init()
                    job = store.create_job('inspect')
                    with patch.dict(os.environ, {'CODEX_HOME': tmp, 'HOME': tmp, 'USERPROFILE': tmp}, clear=True), patch(
                        'puppetmaster.platform_billing._default_runner', side_effect=auth_status
                    ) as run, patch(
                        'puppetmaster.platform_billing.detect_adapter_billing_cached',
                        return_value=BillingStatus('codex', 'plan', True, 'fixture'),
                    ), patch('puppetmaster.preflight.adapter_cli_present', return_value=True), patch(
                        'puppetmaster.platform_lock.is_adapter_enabled', return_value=True
                    ):
                        task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                            role='explore', instruction='inspect', adapter='codex',
                            payload={'model': model.id, 'auto_route': routed,
                                     'registry_path': str(registry), 'min_capability': 1},
                        )])[0]
                    reopened = backend(Path(tmp) / 'state')
                    payload = reopened.get_task_by_id(task.id).payload
                    self.assertEqual(payload['executable'], ['api-codex'])
                    self.assertEqual(payload['billing'], 'api')
                    self.assertEqual(payload['registry_billing'], 'unknown')
                    self.assertEqual(payload['billing_source'], 'launch_auth')
                    self.assertEqual(payload['billing_evidence'], ['launch_auth:api'])
                    self.assertTrue(run.call_args_list)
                    self.assertTrue(all(call.args[0] == ['api-codex', 'login', 'status']
                                        for call in run.call_args_list))
                    artifacts = reopened.list_artifacts(job.id)
                    route = artifacts[-1]
                    self.assertEqual(route.payload['billing'], 'api')
                    self.assertEqual(route.payload['registry_billing'], 'unknown')
                    usage = Artifact(
                        job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                        created_by='test', confidence=1, evidence=['fixture'],
                        payload={'check': 'usage', 'result': 'ok',
                                 'tokens_in': 1000000, 'tokens_out': 0},
                    )
                    reopened.save_artifact(usage)
                    with patch("puppetmaster.cost.load_registry", return_value=[model]):
                        reopened.update_job_status(job.id, JobStatus.COMPLETE)
                    receipt = backend(Path(tmp) / "state").get_job(job.id).cost_receipt
                    self.assertEqual(receipt["actual_cost"]["total_marginal_cost_usd"], 5)

    def test_launch_defaults_do_not_supply_explicit_billing_authority(self):
        model = ModelSpec(
            id='codex/test', adapter='codex', adapter_model_name='test', billing='unknown',
            payload_defaults={'model': 'codex/test', 'billing': 'plan',
                              'billing_source': 'explicit', 'executable': ['api-codex']},
        )
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {'CODEX_HOME': tmp}, clear=True), patch(
            'puppetmaster.platform_billing._default_runner',
            return_value=(0, 'Logged in using API key', ''),
        ) as run:
            result = stamp_model_billing({}, model)
            self.assertEqual(result['billing'], 'api')
            self.assertEqual(result['billing_source'], 'launch_auth')
            explicit = stamp_model_billing({'model': model.id, 'billing': 'plan'}, model)
            self.assertEqual(explicit['billing'], 'plan')
            self.assertEqual(explicit['billing_source'], 'explicit')
            self.assertEqual(run.call_count, 1)
            changed = stamp_model_billing(
                {'model': 'codex/old', 'billing': 'plan', 'billing_source': 'explicit',
                 'billing_evidence': ['stale'], 'executable': ['caller-codex']}, model,
            )
            self.assertEqual(changed['billing'], 'api')
            self.assertEqual(changed['registry_billing'], 'unknown')
            self.assertEqual(changed['billing_evidence'], ['launch_auth:api'])
            self.assertEqual(changed['executable'], ['caller-codex'])
            self.assertEqual(run.call_args.args[0], ['caller-codex', 'login', 'status'])

    def test_identity_and_transport(self):
        model = ModelSpec(id='codex/test', adapter='codex', adapter_model_name='test', billing='unknown')
        with patch('puppetmaster.platform_billing.detect_adapter_billing', return_value=BillingStatus('codex','plan',True,'secret')):
            detected = stamp_model_billing({'model':model.id}, model)
            self.assertEqual(detected['billing_source'], 'launch_auth')
            for override in ('api','plan'):
                self.assertEqual(stamp_model_billing({'model':model.id,'billing':override},model)['billing'],override)
            for adapter in ('openai','agentic'):
                other = replace(model, id=adapter+'/test', adapter=adapter)
                self.assertEqual(stamp_model_billing(detected,other,previous_adapter='codex')['billing'],'unknown')
        with patch('puppetmaster.platform_billing.detect_adapter_billing', side_effect=OSError('secret')):
            self.assertEqual(stamp_model_billing(detected,model)['billing'],'unknown')

    def test_codex_auth_seams(self):
        import json
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            for mode, expected in (('chatgpt','plan'),('apikey','api')):
                (home/'auth.json').write_text(json.dumps({'auth_mode':mode}),encoding='utf-8')
                result = detect_codex_billing(env={'CODEX_HOME':tmp},home=home,run=lambda cmd: self.fail('unexpected CLI'))
                self.assertEqual(result.billing,expected)
            (home/'auth.json').unlink()
            for output, expected in (('Logged in using ChatGPT','plan'),('Logged in using API key','api'),('unavailable','unknown'),('Logged in','unknown')):
                self.assertEqual(detect_codex_billing(env={'CODEX_HOME':tmp},home=home,run=lambda cmd:(0,output,'')).billing,expected)

    def test_fallback_uses_supplied_context(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp, patch('puppetmaster.platform_billing.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='Logged in using API key', stderr='')) as run:
            env = {'CODEX_HOME': tmp, 'CODEX_COMMAND': 'codex-test'}
            self.assertEqual(detect_codex_billing(env=env, home=Path(tmp)).billing, 'api')
            self.assertEqual(run.call_args.args[0], ['codex-test', 'login', 'status'])
            self.assertEqual(run.call_args.kwargs['env'], env)
            self.assertEqual(run.call_args.kwargs['encoding'], 'utf-8')

    def test_raw_openai_payload_key(self):
        model = ModelSpec(id='openai/test', adapter='openai', adapter_model_name='test', billing='unknown')
        with patch.dict(os.environ, {}, clear=True):
            result = stamp_model_billing({'model':model.id, 'openai_api_key':'fixture'},model)
        self.assertEqual(result['billing'], 'api')
        self.assertEqual(result['registry_billing'], 'unknown')

    def test_unknown_reported_cost_is_not_attributable(self):
        from puppetmaster.cost import price_job_from_artifacts, price_job
        route = Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
                         created_by='router', confidence=1, evidence=['fixture'],
                         payload={'model_id':'codex/test','billing':'unknown'})
        usage = Artifact(job_id='j', task_id='t', type=ArtifactType.VERIFICATION,
                         created_by='test', confidence=1, evidence=['fixture'],
                         payload={'real_cost_usd':5, 'tokens_in':1000000, 'tokens_out':0})
        for result in (price_job([route,usage],[]),price_job_from_artifacts([route,usage])):
            self.assertFalse(result.tasks[0].priced)
            self.assertEqual(result.measured_cost_usd,0)

    def test_explicit_windows_command_vector_is_preserved(self):
        model = ModelSpec(id='codex/test', adapter='codex', adapter_model_name='test', billing='unknown')
        command = [r'C:\Program Files\node.exe', r'C:\tools\codex.js']
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {'CODEX_HOME':tmp}, clear=True), patch('puppetmaster.platform_billing._default_runner', return_value=(0,'Logged in using API key','')) as run:
            result = stamp_model_billing({'model':model.id,'executable':command},model)
        self.assertEqual(result['billing'],'api')
        self.assertEqual(run.call_args.args[0], command + ['login','status'])
        self.assertEqual(command, [r'C:\Program Files\node.exe', r'C:\tools\codex.js'])

    def test_absent_registry_missing_billing_cost_reopen(self):
        from puppetmaster.cost import build_cost_report, price_job, price_job_from_artifacts
        for backend in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(backend=backend.__name__), TemporaryDirectory() as tmp:
                store = backend(Path(tmp) / 'state')
                job = store.create_job('legacy absent registry')
                usage = Artifact(job_id=job.id, task_id='t', type=ArtifactType.VERIFICATION,
                    created_by='test', confidence=1, evidence=['fixture'],
                    payload={'check':'usage', 'result':'ok', 'model':'removed-model',
                             'real_cost_usd':5, **token_usage(sdk_usage={'input_tokens':10, 'output_tokens':2})})
                store.save_artifact(usage)
                with patch('puppetmaster.cost.load_registry', return_value=[]):
                    store.update_job_status(job.id, JobStatus.COMPLETE)
                reopened = backend(Path(tmp) / 'state')
                report = build_cost_report(reopened, job.id, [])
                self.assertIsNone(report['actual_cost']['total_marginal_cost_usd'])
                self.assertIsNone(report['actual_cost']['measured_cost_usd'])
                self.assertEqual(report['actual_cost']['tasks'][0]['billing'], 'unknown')
                selected = reopened.get_selected_economics(reopened.job_ref(job.id))
                for metric in (selected.totals.api_cost_usd, selected.totals.plan_marginal_cost_usd,
                               selected.totals.api_equivalent_cost_usd):
                    self.assertIsNone(metric.total)
                    self.assertEqual(metric.state, 'unknown')
                for priced in (price_job([usage], []), price_job_from_artifacts([usage])):
                    self.assertFalse(priced.tasks[0].priced)
                    self.assertEqual(priced.measured_cost_usd, 0)

    def test_custom_command_conflicting_auth_failure_and_redaction(self):
        cases = [(0, 'Logged in using API key SECRET', 'api'),
                 (0, 'Logged in using ChatGPT SECRET', 'plan'),
                 (1, 'Logged in using API key SECRET', 'unknown'),
                 (127, 'SECRET', 'unknown'), (124, 'SECRET', 'unknown'),
                 (0, 'SECRET', 'unknown'), (0, 'API key unavailable SECRET', 'unknown')]
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'auth.json').write_text('{"auth_mode":"chatgpt"}')
            for code, output, expected in cases:
                with self.subTest(code=code, expected=expected), patch.dict(os.environ, {'CODEX_HOME':tmp}, clear=True):
                    status = detect_codex_billing(codex_command='custom-codex',
                        run=lambda cmd: (code, output, 'SECRET'))
                    self.assertEqual(status.billing, expected)
                    self.assertNotIn('SECRET', str(status))
                    model = ModelSpec(id='codex/test', adapter='codex', adapter_model_name='test',
                        billing='unknown', payload_defaults={'executable':['custom-codex']})
                    with patch('puppetmaster.platform_billing._default_runner', return_value=(code,output,'SECRET')):
                        result = stamp_model_billing({'model':model.id}, model)
                        self.assertEqual(result['billing'], expected)
                        self.assertNotIn('SECRET', str(result))
                        explicit = stamp_model_billing({'model':model.id, 'billing':'plan'}, model)
                        self.assertEqual(explicit['billing_source'], 'explicit')
                        self.assertEqual(explicit['billing'], 'plan')

    def test_custom_probe_subprocess_failure_is_unknown_and_bounded(self):
        import subprocess
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'auth.json').write_text('{"auth_mode":"chatgpt"}')
            for error in (OSError('SECRET'), subprocess.TimeoutExpired('SECRET', 15, output='SECRET')):
                with self.subTest(error=type(error).__name__), patch(
                    'puppetmaster.platform_billing.subprocess.run', side_effect=error
                ) as run:
                    result = detect_codex_billing(codex_command=['custom-codex'], env={'CODEX_HOME':tmp})
                    self.assertEqual(result.billing, 'unknown')
                    self.assertNotIn('SECRET', str(result))
                    self.assertEqual(run.call_args.kwargs['timeout'], 15)
            result = detect_codex_billing(codex_command=['custom-codex'], env={'CODEX_HOME':tmp},
                run=lambda cmd: (0, 'x' * 8192 + 'Logged in using ChatGPT SECRET', ''))
            self.assertEqual(result.billing, 'unknown')
            self.assertNotIn('SECRET', str(result))


class CodexLoginStatusTests(unittest.TestCase):
    """Only Codex saying so means logged out; a failed check is unverified."""

    def status(self, result):
        # A custom command owns its credential context, so only the probe decides.
        return detect_codex_billing(codex_command="codex-test", run=lambda _: result)

    def test_failed_timed_out_or_unrecognized_checks_are_unverified(self):
        for result, cause in (((1, "", "boom"), "exit 1"), ((124, "", "timed out"), "timed out"),
                              ((0, "status: ok", ""), "unrecognized output")):
            with self.subTest(cause=cause):
                status = self.status(result)
                self.assertFalse(status.healthy)
                self.assertIn("codex_login:unverified", status.evidence)
                self.assertIn(f"codex_login_status:{cause}", status.evidence)
                self.assertNotIn("not logged in", status.detail)

    def test_explicit_logout_and_login_are_unchanged(self):
        logged_out = self.status((1, "Not logged in", ""))
        self.assertIn("codex_login:none", logged_out.evidence)
        self.assertFalse(logged_out.healthy)
        plan = self.status((0, "Logged in using ChatGPT", ""))
        self.assertTrue(plan.healthy)
        self.assertEqual(plan.billing, "plan")
