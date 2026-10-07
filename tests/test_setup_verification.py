import contextlib
from dataclasses import FrozenInstanceError, replace
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import venv

from puppetmaster import setup_verification as verification
from puppetmaster.attempts import ExecutionAttempt, UsageObservation
from puppetmaster.cli._parser import build_parser
from puppetmaster.cli.commands_install import _run_setup
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus

MODEL = 'codex/test-model'
FIXTURE = Path('/fixture/verification.txt')
NONCE = 'a' * 64
PIN = 'codex/gpt-5-6-sol'
WIRE = 'gpt-5.6-sol'


def require_process_discovery(test):
    try:
        verification._owned_posix_pids('test-preflight', 1)
    except (OSError, subprocess.SubprocessError) as exc:
        test.skipTest(f'Host denies process discovery: {type(exc).__name__}')


class ProbeStore:
    def __init__(self):
        self.task = Task('job', 'explore', 'read fixture', id='task', adapter='codex',
                         status=TaskStatus.COMPLETE, attempts=1, completed_at='now', payload={
                             'model': 'test-model', 'pinned_model': MODEL,
                             'pinned_adapter_model_name': 'test-model', 'router_model_id': MODEL,
                             'auto_route': False, 'allowed_model_ids': [MODEL]})
        self.jobs = [SimpleNamespace(id='job')]
        self.tasks = [self.task]
        self.delivery = {'verdict': 'delivered', 'stale_task_ids': [], 'incomplete_tasks': False}
        self.attempts = [ExecutionAttempt('job', 'task', 'run', 'attempt', 'now', 'codex', 'test-model')]
        self.usage = [UsageObservation('job', 'attempt', 'obs', 'codex', 'now', 'measured', 10, 20)]
        self.artifacts = [Artifact('job', 'task', ArtifactType.FINDING, 'worker',
                                  {'claim': f'FIRST_RUN_PROOF nonce={NONCE} sum=46'}, .9, [str(FIXTURE)]),
                          Artifact('job', 'task', ArtifactType.VERIFICATION, 'worker',
                                   {'check': 'read', 'result': 'passed', 'returncode': 0,
                                    'model': 'test-model', 'sandbox': 'read-only',
                                    'approval_policy': 'never'}, .9, ['adapter:codex'])]

    def list_jobs(self): return self.jobs
    def list_tasks(self, job): return self.tasks
    def list_attempts(self, job): return self.attempts
    def list_usage_observations(self, job, attempt_id): return self.usage
    def list_artifacts(self, job): return self.artifacts
    def status_snapshot(self, job, compact): return {'delivery': self.delivery}


def registry_wire_store():
    """A probe store whose pin has a dashed registry id and dotted wire name."""
    store = ProbeStore()
    task = replace(store.task, payload={
        'model': WIRE, 'pinned_model': PIN,
        'pinned_adapter_model_name': WIRE, 'router_model_id': PIN,
        'auto_route': False, 'allowed_model_ids': [PIN]})
    store.task = task
    store.tasks = [task]
    store.attempts = [replace(store.attempts[0], model=WIRE)]
    store.artifacts = [
        store.artifacts[0],
        replace(store.artifacts[1], payload={**store.artifacts[1].payload, 'model': WIRE}),
    ]
    return store


class RegistryWireIdentityTests(unittest.TestCase):
    """The pinned wire identity is the persisted pin, not the id string."""

    def test_dashed_registry_id_validates_against_its_dotted_wire_name(self):
        self.assertEqual(verification._validate(registry_wire_store(), PIN, NONCE, FIXTURE), '')

    def test_another_models_wire_name_is_still_rejected(self):
        store = registry_wire_store()
        store.attempts = [replace(store.attempts[0], model='gpt-5.5')]
        self.assertEqual(
            verification._validate(store, PIN, NONCE, FIXTURE),
            'Execution ledger does not prove one fresh attempt on the requested model.')


class SuccessPredicateTests(unittest.TestCase):
    def test_success(self):
        self.assertEqual(verification._validate(ProbeStore(), MODEL, NONCE, FIXTURE), '')

    def test_failures(self):
        mutations = {
            'empty jobs': lambda s: s.jobs.clear(),
            'extra jobs': lambda s: s.jobs.append(s.jobs[0]),
            'empty tasks': lambda s: s.tasks.clear(),
            'extra tasks': lambda s: s.tasks.append(s.tasks[0]),
            'retry': lambda s: s.tasks.__setitem__(0, replace(s.task, attempts=2)),
            'incomplete': lambda s: s.tasks.__setitem__(0, replace(s.task, status=TaskStatus.RUNNING)),
            'no completion time': lambda s: s.tasks.__setitem__(0, replace(s.task, completed_at=None)),
            'adapter': lambda s: s.tasks.__setitem__(0, replace(s.task, adapter='openai')),
            'no attempts': lambda s: s.attempts.clear(),
            'two attempts': lambda s: s.attempts.append(s.attempts[0]),
            'attempt model': lambda s: s.attempts.__setitem__(0, replace(s.attempts[0], model='other')),
            'no usage': lambda s: s.usage.clear(),
            'estimated': lambda s: s.usage.__setitem__(0, replace(s.usage[0], usage_state='estimated')),
            'zero usage': lambda s: s.usage.__setitem__(0, replace(s.usage[0], tokens_out=0)),
            'empty artifacts': lambda s: s.artifacts.clear(),
            'wrong nonce': lambda s: s.artifacts[0].payload.update(claim='17 + 29 = 46'),
            'nonce prefix': lambda s: s.artifacts[0].payload.update(claim='17 + 29 = 46; ' + NONCE + 'f'),
            'nonce only': lambda s: s.artifacts[0].payload.update(claim=NONCE),
            'wrong evidence': lambda s: s.artifacts[0].evidence.__setitem__(0, '/other'),
            'bad confidence': lambda s: s.artifacts.__setitem__(0, replace(s.artifacts[0], confidence=2)),
            'stale delivery': lambda s: s.delivery.update(stale_task_ids=['task']),
        }
        for state in ('pending', 'blocked', 'empty', 'degraded', 'stale'):
            mutations[state] = lambda s, state=state: s.delivery.update(verdict=state)
        for field in ('model', 'pinned_model', 'pinned_adapter_model_name', 'router_model_id',
                      'auto_route', 'allowed_model_ids'):
            mutations['identity ' + field] = lambda s, field=field: s.task.payload.update({field: 'wrong'})
        for status in ('stale', 'reused', 'superseded'):
            mutations['artifact ' + status] = lambda s, status=status: s.artifacts[0].payload.update(validation={'status': status})
        for field, value in [('returncode', 1), ('result', 'degraded'), ('model', 'other'),
                             ('sandbox', 'workspace-write'), ('approval_policy', 'on-request'), ('turn_failed', True)]:
            mutations['execution ' + field] = lambda s, field=field, value=value: s.artifacts[1].payload.update({field: value})
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                store = ProbeStore()
                mutate(store)
                self.assertTrue(verification._validate(store, MODEL, NONCE, FIXTURE))

    def test_canonical_proof_rejects_ambiguous_and_unrelated_claims(self):
        proof = f'FIRST_RUN_PROOF nonce={NONCE} sum=46'
        claims = [
            f'FIRST_RUN_PROOF sum={NONCE} nonce=46',
            f'FIRST_RUN_PROOF token={NONCE} sum=46',
            f'FIRST_RUN_PROOF nonce={NONCE} sum=47',
            f'FIRST_RUN_PROOF nonce=_{NONCE} sum=46',
            f'FIRST_RUN_PROOF nonce={NONCE}_ sum=46',
            f'FIRST_RUN_PROOF nonce=x{NONCE} sum=46',
            f'FIRST_RUN_PROOF nonce={NONCE}x sum=46',
            f'17 29 46; nonce={NONCE}; everything passed',
            'The fixture is correct', proof + '\n' + proof,
            proof + ' but sum=47',
        ]
        for claim in claims:
            with self.subTest(claim=claim):
                store = ProbeStore()
                store.artifacts[0].payload['claim'] = claim
                self.assertTrue(verification._validate(store, MODEL, NONCE, FIXTURE))
        for duplicate in (True, False):
            store = ProbeStore()
            if duplicate:
                store.artifacts.append(store.artifacts[0])
            else:
                store.artifacts[0].evidence.clear()
            self.assertTrue(verification._validate(store, MODEL, NONCE, FIXTURE))


class SetupBranchTests(unittest.TestCase):
    def test_parser_and_branch_do_not_install(self):
        args = build_parser().parse_args(['setup', '--verify-first-run', MODEL])
        for passed in (False, True):
            with patch.object(verification, 'verify_first_run', return_value=verification.FirstRunResult(passed, 'reason')) as probe, \
                    patch('puppetmaster.cli.commands_install._setup_platform_step', side_effect=AssertionError('installer')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(_run_setup(args), 0 if passed else 1)
                probe.assert_called_once_with(verification.FirstRunRequest(MODEL))
                self.assertIn('first_run_verified:', output.getvalue())
                self.assertIn('restart', output.getvalue())
        self.assertIsNone(build_parser().parse_args(['setup']).verify_first_run)

    def test_ordinary_setup_never_calls_probe(self):
        args = build_parser().parse_args(['setup'])
        with patch.object(verification, 'verify_first_run', side_effect=AssertionError('implicit live call')) as probe, \
                patch('puppetmaster.cli.commands_install.run_doctor', side_effect=RuntimeError('static setup reached')):
            with self.assertRaisesRegex(RuntimeError, 'static setup reached'), contextlib.redirect_stdout(io.StringIO()):
                _run_setup(args)
            probe.assert_not_called()

    def test_invalid_model_and_frozen_types(self):
        for model in ('gpt-6-astra', 'openai/test', 'codex/', 'codex/a b', 'codex/a/b', None, 5):
            with patch.object(verification.subprocess, 'Popen', side_effect=AssertionError('launch')):
                self.assertFalse(verification.verify_first_run(verification.FirstRunRequest(model)).passed)
        with self.assertRaises(FrozenInstanceError):
            verification.FirstRunRequest(MODEL).model = 'other'
        with self.assertRaises(FrozenInstanceError):
            verification.FirstRunResult(True, '').passed = False


@unittest.skipUnless(os.name == 'posix', 'POSIX process ownership')
class ProcessTests(unittest.TestCase):
    def test_isolation_and_cleanup_for_each_outcome(self):
        for outcome in ('pass', 'failed rc', 'no state', 'timeout', 'inspect timeout', 'inspect failed', 'bad result'):
            seen = []
            def run(command, cwd, env, deadline):
                seen.append((command, cwd, env))
                state = Path(env['PUPPETMASTER_STATE_DIR'])
                self.assertNotEqual(str(state), '/existing-state')
                self.assertNotIn('PUPPETMASTER_LAUNCH_KEY', env)
                self.assertNotIn('PYTHONPATH', env)
                self.assertEqual(env['PUPPETMASTER_WORKING_SET'], '0')
                self.assertEqual(env['PUPPETMASTER_JOB_BRIEF'], '0')
                self.assertEqual(env['PUPPETMASTER_WORKING_SET_REUSE'], '0')
                if len(seen) == 1:
                    self.assertEqual(command[:3], [sys.executable, '-m', 'puppetmaster'])
                    config = json.loads(Path(command[command.index('--config') + 1]).read_text())
                    worker = config['workers'][0]
                    fixture = Path(cwd) / 'verification.txt'
                    nonce = fixture.read_text().split('nonce=')[1].strip()
                    self.assertEqual(len(nonce), 64)
                    self.assertNotIn(nonce, json.dumps(config))
                    self.assertNotEqual(state.parent, fixture.parent)
                    self.assertFalse((fixture.parent / '.git').exists())
                    payload = worker['payload']
                    self.assertEqual(payload['model'], MODEL)
                    self.assertEqual(payload['allowed_model_ids'], [MODEL])
                    self.assertFalse(payload['auto_route'])
                    self.assertFalse(payload['dangerously_bypass_approvals_and_sandbox'])
                    self.assertEqual(payload['sandbox'], 'read-only')
                    self.assertEqual(payload['approval_policy'], 'never')
                    for key in ('disable_memory', 'disable_codegraph', 'skip_working_set_reuse'):
                        self.assertTrue(payload[key])
                    if outcome == 'timeout': raise subprocess.TimeoutExpired(command, 120)
                    if outcome == 'failed rc': return 2
                    if outcome != 'no state':
                        state.mkdir()
                        (state / 'state.sqlite3').touch()
                else:
                    if outcome == 'inspect timeout': raise subprocess.TimeoutExpired(command, 120)
                    if outcome == 'inspect failed': return 1
                    Path(command[-1]).write_text(json.dumps({'reason': 'invalid finding' if outcome == 'bad result' else ''}))
                return 0
            with self.subTest(outcome=outcome), patch.object(verification, '_run_owned', side_effect=run), \
                    patch.dict(os.environ, {'PUPPETMASTER_LAUNCH_KEY': 'inherited', 'PYTHONPATH': '/checkout',
                                            'PUPPETMASTER_STATE_DIR': '/existing-state'}):
                result = verification.verify_first_run(verification.FirstRunRequest(MODEL))
                self.assertEqual(result.passed, outcome == 'pass', result)
                for _, cwd, env in seen:
                    self.assertFalse(Path(cwd).exists())
                    self.assertFalse(Path(env['PUPPETMASTER_STATE_DIR']).parent.exists())

    def test_actual_outer_timeout_kills_descendants(self):
        require_process_discovery(self)
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / 'survived'
            grandchild = 'import time; from pathlib import Path; time.sleep(1.5); Path(%r).touch()' % str(marker)
            child = 'import subprocess,sys,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,"-c",%r]); time.sleep(30)' % grandchild
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                verification._run_owned([sys.executable, '-c', child], root, dict(os.environ), started + 3.2)
            self.assertLess(time.monotonic() - started, 3.5)
            time.sleep(.5)
            self.assertFalse(marker.exists())

    def test_escaped_session_descendant_after_exit_and_timeout(self):
        require_process_discovery(self)
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as root:
                pidfile = Path(root) / 'pid'
                marker = Path(root) / 'survived'
                grandchild = (
                    'import os,time; from pathlib import Path; '
                    f'Path({str(pidfile)!r}).write_text(str(os.getpid())); '
                    f'time.sleep(1); Path({str(marker)!r}).touch(); time.sleep(30)')
                child = (
                    'import subprocess,sys,time; from pathlib import Path; '
                    f'subprocess.Popen([sys.executable,"-c",{grandchild!r}], start_new_session=True); '
                    f'p=Path({str(pidfile)!r})\n'
                    'while not p.exists(): time.sleep(.005)\n'
                    + ('time.sleep(30)' if timeout else ''))
                started = time.monotonic()
                try:
                    if timeout:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            verification._run_owned([sys.executable, '-c', child], root,
                                                    dict(os.environ), started + 3.4)
                    else:
                        self.assertEqual(verification._run_owned([sys.executable, '-c', child], root,
                                         dict(os.environ), started + 5), 0)
                    self.assertTrue(pidfile.exists(), 'grandchild must actually have launched')
                    self.assertLess(time.monotonic() - started, 3.5)
                    time.sleep(1.1)
                    self.assertFalse(marker.exists(), 'escaped-session grandchild survived cleanup')
                finally:
                    if pidfile.exists():
                        try:
                            os.kill(int(pidfile.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass


class OwnershipUtilityTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_sigstop_permission_error_preserves_cleanup_and_timeout(self):
        self._check_group_permission_error(signal.SIGSTOP)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_sigkill_permission_error_preserves_cleanup_and_timeout(self):
        self._check_group_permission_error(signal.SIGKILL)

    def _check_group_permission_error(self, denied_signal):
        import errno
        from puppetmaster import win_process
        from unittest.mock import Mock

        def signal_group(pid, sig):
            if sig == denied_signal:
                raise PermissionError(errno.EPERM, 'Operation not permitted')

        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                process = Mock(pid=4321)
                primary = subprocess.TimeoutExpired(['probe'], 1)
                process.wait.side_effect = [primary if timeout else 0, 0, 0]
                with patch.object(win_process.os, 'name', 'posix'), \
                        patch.object(win_process.os, 'killpg', side_effect=signal_group) as group, \
                        patch.object(win_process.os, 'kill') as kill, \
                        patch.object(verification, '_owned_posix_pids', return_value=[]), \
                        patch.object(win_process, '_owned_posix_pids', side_effect=[[5678], [], []]) as discover, \
                        patch.object(verification.subprocess, 'Popen', return_value=process):
                    if timeout:
                        with self.assertRaises(subprocess.TimeoutExpired) as raised:
                            verification._run_owned(['probe'], '.', {}, time.monotonic() + 10)
                        self.assertIs(raised.exception, primary)
                    else:
                        self.assertEqual(verification._run_owned(['probe'], '.', {}, time.monotonic() + 10), 0)
                    win_process.stop_owned_process(process, 'owner', time.monotonic() + 1)
                kill.assert_called_once_with(5678, signal.SIGKILL)
                self.assertEqual(discover.call_count, 3)
                group.assert_any_call(4321, signal.SIGSTOP)
                group.assert_any_call(4321, signal.SIGKILL)
                self.assertEqual(process.kill.call_count, 2)
                self.assertEqual(process.wait.call_count, 3)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_cleanup_discovery_error_preserves_original_timeout(self):
        self._check_cleanup_failure('discovery')

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_one_denied_descendant_does_not_stop_remaining_kills(self):
        self._check_cleanup_failure('descendant')

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_group_oserror_does_not_stop_remaining_layers(self):
        self._check_cleanup_failure('group')

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_leader_kill_error_still_attempts_bounded_wait(self):
        self._check_cleanup_failure('leader')

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_cleanup_wait_error_preserves_original_timeout(self):
        self._check_cleanup_failure('wait')

    def _check_cleanup_failure(self, boundary):
        import errno
        from puppetmaster import win_process
        from unittest.mock import Mock, call

        for error in (PermissionError(errno.EPERM, 'denied'),
                      OSError(errno.EIO, 'I/O error')):
            with self.subTest(error=type(error).__name__):
                primary = subprocess.TimeoutExpired(['probe'], 7)
                process = Mock(pid=4321)
                process.wait.side_effect = [primary, error if boundary == 'wait' else 0]
                if boundary == 'leader':
                    process.kill.side_effect = error
                discovery = error if boundary == 'discovery' else [[5678, 5679, 5680], []]
                with patch.object(win_process.os, 'name', 'posix'), \
                        patch.object(verification, '_owned_posix_pids', return_value=[]), \
                        patch.object(verification.subprocess, 'Popen', return_value=process), \
                        patch.object(win_process, '_owned_posix_pids', side_effect=discovery), \
                        patch.object(win_process.os, 'killpg',
                                     side_effect=error if boundary == 'group' else None) as group, \
                        patch.object(win_process.os, 'kill',
                                     side_effect=[None, error, None] if boundary == 'descendant' else None) as kill:
                    with self.assertRaises(subprocess.TimeoutExpired) as raised:
                        verification._run_owned(['probe'], '.', {}, time.monotonic() + 10)
                self.assertIs(raised.exception, primary)
                self.assertEqual(group.call_args_list, [call(4321, signal.SIGSTOP),
                                                       call(4321, signal.SIGKILL)])
                if boundary != 'discovery':
                    self.assertEqual(kill.call_args_list, [call(pid, signal.SIGKILL)
                                                          for pid in (5678, 5679, 5680)])
                process.kill.assert_called_once()
                self.assertEqual(process.wait.call_count, 2)
                self.assertGreater(process.wait.call_args.kwargs['timeout'], 0)
                self.assertLessEqual(process.wait.call_args.kwargs['timeout'], 3 + 1e-9)  # one-ulp float rounding of (t + 3) - t

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_exhausted_cleanup_deadline_still_kills_group_and_leader(self):
        from puppetmaster import win_process
        from unittest.mock import Mock
        process = Mock(pid=4321)
        with patch.object(win_process.os, 'name', 'posix'), \
                patch.object(win_process.os, 'killpg') as group, \
                patch.object(win_process, '_owned_posix_pids') as discover:
            win_process.stop_owned_process(process, 'owner', time.monotonic() - 1)
        discover.assert_not_called()
        group.assert_any_call(4321, signal.SIGKILL)
        process.kill.assert_called_once()
        process.wait.assert_not_called()

    def test_process_discovery_uses_exact_inherited_marker(self):
        from puppetmaster import win_process
        output = ('10 cmd PUPPETMASTER_PROCESS_OWNER=abc\n'
                  '11 cmd PUPPETMASTER_PROCESS_OWNER=abcd\n'
                  '12 cmd X_PUPPETMASTER_PROCESS_OWNER=abc\n')
        with patch.object(win_process.subprocess, 'run', return_value=SimpleNamespace(stdout=output)) as run:
            self.assertEqual(win_process._owned_posix_pids('abc', .2), [10])
        self.assertEqual(run.call_args.kwargs['timeout'], .2)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_cleanup_reaps_leader_even_if_discovery_fails(self):
        from puppetmaster import win_process
        from unittest.mock import Mock
        process = Mock(pid=4321)
        with patch.object(win_process.os, 'name', 'posix'), \
                patch.object(win_process.os, 'killpg') as group, \
                patch.object(win_process, '_owned_posix_pids', side_effect=PermissionError):
            win_process.stop_owned_process(process, 'owner', time.monotonic() + 1)
        group.assert_any_call(4321, signal.SIGKILL)
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "requires POSIX process-group signals")
    def test_cleanup_after_leader_exit_and_repeated_cleanup(self):
        from puppetmaster import win_process
        from unittest.mock import Mock
        process = Mock(pid=4321)
        with patch.object(win_process.os, 'name', 'posix'), \
                patch.object(win_process.os, 'killpg', side_effect=ProcessLookupError), \
                patch.object(win_process.os, 'kill') as kill, \
                patch.object(win_process, '_owned_posix_pids', side_effect=[[5678], [], []]):
            for _ in range(2):
                win_process.stop_owned_process(process, 'owner', time.monotonic() + 1)
        kill.assert_called_once_with(5678, signal.SIGKILL)

    def test_deadline_and_discovery_failure_prevent_launch(self):
        with patch.object(verification.subprocess, 'Popen') as spawn:
            with self.assertRaises(subprocess.TimeoutExpired):
                verification._run_owned(['probe'], '.', {}, time.monotonic())
            spawn.assert_not_called()
        with patch.object(verification.os, 'name', 'posix'), \
                patch.object(verification, '_owned_posix_pids', side_effect=PermissionError), \
                patch.object(verification.subprocess, 'Popen') as spawn:
            with self.assertRaises(PermissionError):
                verification._run_owned(['probe'], '.', {}, time.monotonic() + 120)
            spawn.assert_not_called()


class WindowsOwnershipTests(unittest.TestCase):
    def test_hidden_creation_and_bounded_cleanup(self):
        process = SimpleNamespace(pid=4321, wait=lambda timeout: 0)
        with patch.object(verification.os, 'name', 'nt'), \
                patch.object(verification.subprocess, 'Popen', return_value=process) as spawn, \
                patch.object(verification, 'stop_owned_process') as stop, \
                patch('puppetmaster.win_process.WindowsJob'):
            deadline = time.monotonic() + 120
            self.assertEqual(verification._run_owned(['probe'], '.', {}, deadline), 0)
        options = spawn.call_args.kwargs
        self.assertEqual(options['creationflags'], 0x08000004)
        self.assertNotIn('start_new_session', options)
        self.assertEqual(stop.call_args.args[:2], (process, options['env']['PUPPETMASTER_PROCESS_OWNER']))
        self.assertLessEqual(stop.call_args.args[2], deadline)

    def test_windows_cleanup_failures_preserve_timeout_and_fallbacks(self):
        from puppetmaster import win_process
        from unittest.mock import Mock
        for boundary in ('job', 'leader', 'wait'):
            with self.subTest(boundary=boundary):
                process = Mock(pid=4321)
                primary = subprocess.TimeoutExpired(['probe'], 7)
                error = PermissionError('denied')
                process.wait.side_effect = [primary, error if boundary == 'wait' else 0]
                if boundary == 'leader':
                    process.kill.side_effect = error
                with patch.object(win_process.os, 'name', 'nt'), \
                        patch.object(verification.subprocess, 'Popen', return_value=process), \
                        patch.object(win_process, 'WindowsJob') as job_type, \
                        patch.object(win_process, '_toolhelp_kill_process_tree',
                                     side_effect=error if boundary == 'toolhelp' else None) as tree, \
                        patch.object(win_process, '_taskkill_process_tree',
                                     side_effect=error if boundary == 'taskkill' else None) as taskkill:
                    job_type.return_value.terminate.side_effect = error if boundary == 'job' else None
                    with self.assertRaises(subprocess.TimeoutExpired) as raised:
                        verification._run_owned(['probe'], '.', {}, time.monotonic() + 10)
                self.assertIs(raised.exception, primary)
                tree.assert_not_called()
                taskkill.assert_not_called()
                self.assertGreaterEqual(job_type.return_value.terminate.call_count, 1)
                process.kill.assert_called_once()
                self.assertEqual(process.wait.call_count, 2)
                self.assertLessEqual(process.wait.call_args.kwargs['timeout'], 3 + 1e-9)  # one-ulp float rounding of (t + 3) - t

    def test_toolhelp_denied_descendant_continues_to_remaining_pids(self):
        from puppetmaster import win_process
        from unittest.mock import call
        with patch.object(win_process, '_toolhelp_tree_pids', return_value=[30, 20, 10]), \
                patch.object(win_process, '_terminate_pid',
                             side_effect=[PermissionError('denied'), True, True]) as terminate:
            self.assertTrue(win_process._toolhelp_kill_process_tree(10, deadline=time.monotonic() + 1))
        self.assertEqual(terminate.call_args_list, [call(30), call(20), call(10)])

    def test_windows_cleanup_after_leader_exit_is_idempotent(self):
        from puppetmaster import win_process
        from unittest.mock import Mock
        process = Mock(pid=4321)
        process.wait.return_value = 0
        with patch.object(win_process.os, 'name', 'nt'), \
                patch.object(win_process, '_toolhelp_kill_process_tree', return_value=True) as tree, \
                patch.object(win_process, '_taskkill_process_tree', return_value=True) as taskkill:
            for _ in range(2):
                win_process.stop_owned_process(process, 'owner', time.monotonic() + .5)
        tree.assert_not_called()
        taskkill.assert_not_called()
        self.assertEqual(process._puppetmaster_job.terminate.call_count, 2)


@unittest.skipUnless(os.name == 'posix', 'POSIX first-run verification')
class InstalledModuleTests(unittest.TestCase):
    def test_installed_module_runs_real_runtime_with_fake_codex(self):
        require_process_discovery(self)
        # A dependency-free venv installed layout proves imports do not fall
        # back to the checkout. Only the external Codex executable is simulated.
        from puppetmaster.model_registry import ModelSpec, save_registry
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            venv.EnvBuilder(with_pip=False, symlinks=True).create(root / 'venv')
            python = root / 'venv/bin/python'
            site = subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
            shutil.copytree(Path(verification.__file__).parent, Path(site) / 'puppetmaster', ignore=shutil.ignore_patterns('__pycache__'))
            registry = root / 'models.json'
            save_registry([ModelSpec(id=MODEL, adapter='codex', adapter_model_name='test-model')], registry)
            # Billing preflight reads Codex auth independently of CODEX_COMMAND.
            # Keep the simulated login independent of the host's CLI and account.
            codex_home = root / 'codex-home'
            codex_home.mkdir()
            (codex_home / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt'}))
            fake = root / 'fake_codex.py'
            fake.write_text('''import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['login', 'status']:
    print('Logged in using ChatGPT'); raise SystemExit(0)
assert args[0] == 'exec', args
assert args[args.index('-m') + 1] == 'test-model', args
assert args[args.index('--sandbox') + 1] == 'read-only'
assert 'approval_policy="never"' in args
assert '--dangerously-bypass-approvals-and-sandbox' not in args
prompt = sys.stdin.read()
fixture = Path.cwd() / 'verification.txt'
nonce = fixture.read_text().split('nonce=')[1].strip()
assert nonce not in prompt
finding = {'type': 'finding', 'claim': 'FIRST_RUN_PROOF nonce=' + nonce + ' sum=46', 'evidence': [str(fixture)], 'confidence': .9}
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'artifacts':[finding]})}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':50}}))
''')
            env = {**os.environ, 'CODEX_HOME': str(codex_home),
                   'CODEX_COMMAND': f'{python} {fake}', 'PUPPETMASTER_MODELS_PATH': str(registry),
                   'PUPPETMASTER_ONLY_ADAPTERS': 'codex', 'PUPPETMASTER_LAUNCH_KEY': 'must-be-removed',
                   'PUPPETMASTER_STATE_DIR': str(root / 'unused-state'), 'PYTHONPATH': '/does-not-exist'}
            env.pop('PUPPETMASTER_WORKER', None)
            env.pop('PUPPETMASTER_ALLOW_NESTED', None)
            result = subprocess.run([str(python), '-m', 'puppetmaster', 'setup', '--verify-first-run', MODEL],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('first_run_verified: pass', result.stdout)
            self.assertFalse((root / 'unused-state').exists())


if __name__ == '__main__':
    unittest.main()
