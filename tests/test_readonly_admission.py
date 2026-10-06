import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import readonly
from puppetmaster.sqlite_store import SQLiteSwarmStore

sys.path.insert(0, str(Path(__file__).parent))
from readonly_fixtures import close_new_cleanup_owners  # noqa: E402


def admitted_process(root, barrier, active, peak):
    from puppetmaster.readonly_admission import ReaderAdmission
    class ObservedAdmission(ReaderAdmission):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with active.get_lock():
                active.value += 1
                peak.value = max(peak.value, active.value)

        def release(self):
            if self.fd is not None or self.local_lock is not None:
                with active.get_lock():
                    active.value -= 1
            super().release()
    store = SQLiteSwarmStore(root)
    barrier.wait(timeout=30)
    ready = readonly._Transport.ready
    prepared = []
    def prepare(transport, deadline):
        ready(transport, deadline)
        if not prepared:
            prepared.append(True)
            assert active.value == 0
            barrier.wait(timeout=30)
    with patch.object(readonly, 'ReaderAdmission', ObservedAdmission), \
            patch.object(readonly._Transport, 'ready', prepare):
        store.attach()
        assert store._attached
        with readonly.connect(store, attach_binding=True) as connection:
            assert connection.execute('SELECT 42').fetchone()[0] == 42


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        close_new_cleanup_owners(self)

    def test_gc_only_retires_and_slot_lookup_never_closes_other_database(self):
        import gc
        import weakref
        with TemporaryDirectory() as root, TemporaryDirectory() as other:
            store = SQLiteSwarmStore(root)
            peer = SQLiteSwarmStore(other)
            store.ensure_schema()
            peer.ensure_schema()
            with readonly.connect(store, reuse=True) as connection:
                transport = connection.transport
                token = connection._token
                slot_ref = weakref.ref(connection._slot)
            del connection
            with patch.object(transport, 'close', side_effect=AssertionError('GC performed I/O')):
                del store._readonly_slot
                gc.collect()
                self.assertIsNone(slot_ref())
                self.assertTrue(readonly._cleanup.owners[token].retired)
                with readonly.connect(peer) as connection:
                    self.assertEqual(connection.execute('SELECT 42').fetchone()[0], 42)
            readonly._cleanup.maintain(limit=len(readonly._cleanup.owners))
            self.assertTrue(transport.closed)
            self.assertNotIn(token, readonly._cleanup.owners)

    def test_cleanup_capacity_retains_and_rotates_failed_owners(self):
        from unittest.mock import Mock
        registry = readonly.CleanupRegistry()
        transports = [Mock(closed=True) for _ in range(10)]
        tokens = [registry.register(t, (i,)) for i, t in enumerate(transports)]
        for token, transport in zip(tokens, transports):
            transport.close.side_effect = OSError('persistent teardown')
            registry.retire(token)
        with self.assertLogs('puppetmaster.readonly_cleanup', level='WARNING'):
            registry.maintain()
        self.assertEqual([t.close.call_count for t in transports], [1] * 8 + [0] * 2)
        self.assertEqual(len(registry.owners), 10)
        for transport in transports:
            transport.close.side_effect = None
        registry.shutdown()
        self.assertEqual(registry.owners, {})

    def test_gc_cleanup_inherited_owners_are_inert(self):
        import os
        from unittest.mock import Mock
        registry = readonly.CleanupRegistry()
        transport = Mock()
        token = registry.register(transport, (1, 2))
        with patch.object(readonly.os, 'getpid', return_value=os.getpid() + 1):
            registry.retire(token)
            registry.close(token)
        transport.close.assert_not_called()
        self.assertFalse(registry.owners[token].retired)

    def test_connection_retries_failed_windows_release(self):
        import os
        import threading
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission

        for reuse in (False, True):
            for mode in (('close', 'abort', 'control_error') if reuse else ('close', 'abort')):
                abort = mode != 'close'
                with self.subTest(reuse=reuse, mode=mode), TemporaryDirectory() as root:
                    store = SQLiteSwarmStore(root)
                    store.ensure_schema()
                    connection = readonly.connect(store, reuse=reuse)
                    connection._permit.release()
                    native = admission._WindowsFileLock.__new__(admission._WindowsFileLock)
                    native.pid, native.handle = os.getpid(), 123
                    failure = OSError(6, 'CloseHandle failed')
                    native.api = Mock()
                    native.api.close.side_effect = [failure, None]
                    permit = admission.ReaderAdmission.__new__(admission.ReaderAdmission)
                    permit.pid, permit.fd = os.getpid(), None
                    permit.windows_lock = native
                    local = permit.local_lock = threading.Lock()
                    local.acquire()
                    connection._permit = permit
                    target = connection._slot if reuse else connection.transport
                    slot = connection._slot
                    method = 'discard' if reuse else 'close'
                    with patch.object(target, method, wraps=getattr(target, method)) as teardown, \
                            patch.object(connection, '_control', wraps=connection._control) as control, \
                            patch.object(connection, '_receive', wraps=connection._receive,
                                side_effect=OSError('release acknowledgement failed')
                                if mode == 'control_error' else None):
                        with self.assertRaises(OSError) as caught:
                            (connection._abort if mode == 'abort' else connection.close)()
                        self.assertIs(caught.exception, failure)
                        self.assertIs(caught.exception.admission_owner, permit)
                        self.assertEqual(native.handle, 123)
                        self.assertTrue(local.locked())
                        self.assertFalse(connection.transport.busy.locked())
                        if reuse:
                            self.assertFalse(slot.busy.locked())
                        connection.close()
                        self.assertIsNone(native.handle)
                        self.assertFalse(local.locked())
                        connection.close()
                        self.assertEqual(native.api.close.call_count, 2)
                        self.assertEqual(teardown.call_count, int(abort or not reuse))
                        self.assertEqual(control.call_count, int(reuse and mode != 'abort'))

    def test_queued_expiry_closes_ready_helper_without_opening_source(self):
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            with readonly.connect(store):
                for options in ({}, {'reuse': True}, {'attach_binding': True}, {'launch_binding': True}):
                    transports = []
                    original = readonly._Transport
                    def transport(path, deadline=None):
                        result = original(path)
                        transports.append(result)
                        return result
                    with self.subTest(options=options), patch.object(readonly, '_Transport', transport):
                        with self.assertRaises(readonly.ReadTimeout):
                            readonly.connect(store, timeout=.05, **options)
                    self.assertEqual(len(transports), 1)
                    readonly._cleanup.close(transports[0].token)
                    self.assertTrue(transports[0].closed)
                    self.assertIsNotNone(transports[0].process.poll())

    def test_sessions_are_serial_per_path_and_other_paths_progress(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as root, TemporaryDirectory() as other:
            store = SQLiteSwarmStore(root)
            peer = SQLiteSwarmStore(other)
            store.ensure_schema()
            peer.ensure_schema()
            queued = threading.Barrier(9)
            original = readonly.ReaderAdmission
            def admission(*args, **kwargs):
                queued.wait(5)
                return original(*args, **kwargs)
            active = [0, 0]
            mutex = threading.Lock()
            def read():
                with readonly.connect(store) as connection:
                    with mutex:
                        active[0] += 1
                        active[1] = max(active)
                    try:
                        self.assertEqual(connection.execute('SELECT 42').fetchone()[0], 42)
                    finally:
                        with mutex:
                            active[0] -= 1
            with readonly.connect(store) as owner, ThreadPoolExecutor(max_workers=8) as pool:
                with patch.object(readonly, 'ReaderAdmission', admission):
                    futures = [pool.submit(read) for _ in range(32)]
                    queued.wait(5)
                with readonly.connect(peer) as independent:
                    self.assertEqual(independent.execute('SELECT 1').fetchone()[0], 1)
                owner.close()
                for future in futures:
                    future.result(timeout=10)
            self.assertEqual(active, [0, 1])

    def test_failures_and_release_ack_keep_permit_until_safe(self):
        from puppetmaster.readonly_admission import ReaderAdmission
        import time
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            for reuse in (False, True):
                for failure in (OSError('spawn failed'), readonly.ReadTimeout('first response'),
                                readonly.ReadUnavailable('EOF')):
                    target = '_Transport' if isinstance(failure, OSError) else None
                    context = (patch.object(readonly, target, side_effect=failure) if target else
                               patch.object(readonly.ReadConnection, '_receive', side_effect=failure))
                    with context, self.assertRaises(type(failure)):
                        readonly.connect(store, reuse=reuse)
                    with readonly.connect(store, reuse=reuse) as connection:
                        self.assertEqual(connection.execute('SELECT 1').fetchone()[0], 1)
            connection = readonly.connect(store, reuse=True)
            control = connection._control
            def release(*args):
                with self.assertRaises(readonly.ReadTimeout):
                    ReaderAdmission(store.db_path, time.monotonic())
                return control(*args)
            with patch.object(connection, '_control', release):
                connection.close()
            permit = ReaderAdmission(store.db_path, time.monotonic() + 1)
            permit.release()
            connection = readonly.connect(store)
            transport_close = connection.transport.close
            def reap(deadline=None):
                with self.assertRaises(readonly.ReadTimeout):
                    ReaderAdmission(store.db_path, time.monotonic())
                transport_close()
            with patch.object(connection.transport, 'close', reap):
                connection._abort()
            permit = ReaderAdmission(store.db_path, time.monotonic() + 1)
            permit.release()

    def test_process_death_releases_kernel_permit(self):
        import subprocess
        import sys
        import time
        from puppetmaster.readonly_admission import ReaderAdmission, _coordination_path
        with TemporaryDirectory() as root:
            # The child must contend on this process's (test-isolated) lock file.
            process = subprocess.Popen([sys.executable, '-c',
                'import sys,time; from pathlib import Path; from puppetmaster import readonly_admission as a; '
                'a._coordination_path=lambda path, selected=None: Path(sys.argv[2]); '
                'p=a.ReaderAdmission(sys.argv[1],time.monotonic()+5); print("ready",flush=True); '
                'sys.stdin.read()', root, str(_coordination_path(root))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                with self.assertRaises(readonly.ReadTimeout):
                    ReaderAdmission(root, time.monotonic() + .02)
                process.kill()
                process.wait(timeout=5)
                permit = ReaderAdmission(root, time.monotonic() + 1)
                permit.release()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()

    @unittest.skipUnless(hasattr(__import__('os'), 'fork'), 'requires fork')
    def test_fork_child_cannot_unlock_parent(self):
        import os
        import time
        from puppetmaster.readonly_admission import ReaderAdmission
        with TemporaryDirectory() as root:
            permit = ReaderAdmission(root, time.monotonic() + 1)
            child = os.fork()
            if child == 0:
                try:
                    permit.release()
                    try:
                        ReaderAdmission(root, time.monotonic() + .02)
                    except readonly.ReadTimeout:
                        os._exit(0)
                    os._exit(1)
                except BaseException:
                    os._exit(2)
            try:
                self.assertEqual(os.waitpid(child, 0)[1], 0)
                with self.assertRaises(readonly.ReadTimeout):
                    ReaderAdmission(root, time.monotonic() + .02)
            finally:
                permit.release()
            ReaderAdmission(root, time.monotonic() + 1).release()

    def test_unsafe_coordination_files_rejected(self):
        import os
        from pathlib import Path
        import time
        from puppetmaster.readonly_admission import ReaderAdmission, _coordination_path
        if os.name == 'nt':
            self.skipTest('POSIX coordination path')
        with TemporaryDirectory() as root:
            db = Path(root) / 'db'
            db.touch()
            target = _coordination_path(db)
            unsafe = Path(root) / 'unsafe'
            unsafe.write_text('')
            try:
                target.symlink_to(unsafe)
            except OSError:
                self.skipTest('symlink creation unavailable')
            try:
                with self.assertRaises(OSError):
                    ReaderAdmission(Path(root) / 'db', time.monotonic() + 1)
            finally:
                target.unlink()

    def test_admission_does_not_reset_deadline_or_ordinary_window(self):
        from types import SimpleNamespace
        import sqlite3
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            for reuse, window, budget in ((True, .1, 3), (False, 1.0, 3), (True, .1, .45), (False, 1.0, .45)):
                clock = [0.0]
                admission = readonly.ReaderAdmission
                timer = SimpleNamespace(monotonic=lambda: clock[0],
                                        sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
                def delayed(*args, **kwargs):
                    permit = admission(*args, **kwargs)
                    clock[0] = min(args[1], clock[0] + .4)
                    return permit
                with patch.object(readonly, 'time', timer), \
                        patch.object(readonly, 'ReaderAdmission', delayed), \
                        patch.object(readonly.ReadConnection, '_receive',
                                     side_effect=sqlite3.OperationalError('database is locked')):
                    with self.assertRaises(sqlite3.OperationalError):
                        readonly.connect(store, timeout=budget, reuse=reuse)
                expected = min(budget, .4 + window)
                self.assertLessEqual(clock[0], expected)
                self.assertGreaterEqual(clock[0], expected - .05)
            clock = [0.0]
            constructor = readonly.ReadConnection
            def delayed_constructor(*args, **kwargs):
                clock[0] += .75
                return constructor(*args, **kwargs)
            with patch.object(readonly, 'time', timer), \
                    patch.object(readonly, 'ReadConnection', delayed_constructor), \
                    patch.object(readonly, '_Transport') as spawn:
                with self.assertRaises(readonly.ReadTimeout):
                    readonly.connect(store, timeout=.5)
                spawn.assert_not_called()

    def test_attach_queue_uses_aggregate_deadline(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            clock = [0.0]
            admission = readonly.ReaderAdmission
            def delayed(path, deadline, **kwargs):
                self.assertEqual(deadline, 25)
                permit = admission(path, deadline, **kwargs)
                clock[0] = 6
                return permit
            timer = SimpleNamespace(monotonic=lambda: clock[0])
            with patch.object(readonly, 'time', timer), \
                    patch.object(readonly, 'ReaderAdmission', delayed):
                with readonly.connect(store, timeout=5, attach_binding=True, attach_deadline=25) as connection:
                    self.assertLessEqual(connection.timeout, 5)
                    self.assertEqual(connection.execute('SELECT 1').fetchone()[0], 1)

    def test_spawned_processes_share_one_admitted_session(self):
        import multiprocessing
        with TemporaryDirectory() as root:
            SQLiteSwarmStore(root).ensure_schema()
            ctx = multiprocessing.get_context('spawn')
            barrier = ctx.Barrier(32)
            active = ctx.Value('i', 0)
            peak = ctx.Value('i', 0)
            processes = [ctx.Process(target=admitted_process, args=(root, barrier, active, peak))
                         for _ in range(32)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=60)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(active.value, 0)
                self.assertEqual(peak.value, 1)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    process.close()

    def test_real_receive_eof_and_timeout_reap_before_next_admission(self):
        import queue
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            original = readonly.ReadConnection._receive
            for eof in (True, False):
                connections = []
                def receive(connection):
                    connections.append(connection)
                    options = {'return_value': ''} if eof else {'side_effect': queue.Empty}
                    with patch.object(connection.responses, 'get', **options):
                        return original(connection)
                with patch.object(readonly.ReadConnection, '_receive', receive):
                    with self.assertRaises(readonly.ReadUnavailable):
                        readonly.connect(store)
                self.assertIsNotNone(connections[0].process.poll())
                self.assertFalse(connections[0].transport.reader.is_alive())
                with readonly.connect(store) as connection:
                    self.assertEqual(connection.execute('SELECT 1').fetchone()[0], 1)

    def test_teardown_faults_retain_ownership_until_retry(self):
        from contextlib import ExitStack
        for reuse in (False, True):
            for failure in ('terminate', 'wait'):
                for action in ('close', 'abort'):
                    with self.subTest(reuse=reuse, failure=failure, action=action), TemporaryDirectory() as root:
                        store = SQLiteSwarmStore(root)
                        store.ensure_schema()
                        connection = readonly.connect(store, reuse=reuse)
                        # A configured callback forces cached close through teardown.
                        if reuse and action == 'close':
                            connection.set_trace_callback(lambda sql: None)
                        transport = connection.transport
                        finalizer = connection._slot.finalizer if reuse else None
                        try:
                            with ExitStack() as faults:
                                faults.enter_context(patch.object(connection.process, failure,
                                    side_effect=OSError('injected teardown failure')))
                                if failure == 'wait':
                                    # Keep the real helper alive while reaping is unavailable.
                                    faults.enter_context(patch.object(connection.process, 'terminate',
                                                                    return_value=None))
                                with self.assertRaisesRegex(OSError, 'injected'):
                                    (connection.close if action == 'close' else connection._abort)()
                                self.assertFalse(connection.closed)
                                self.assertFalse(transport.closed)
                                self.assertIsNone(connection.process.poll())
                                self.assertFalse(transport.busy.locked())
                                self.assertIsNone(connection._slot)
                                self.assertIs(readonly._cleanup.owners[connection._token].transport, transport)
                                if reuse:
                                    self.assertFalse(finalizer.alive)
                                with self.assertRaises(readonly.ReadTimeout):
                                    readonly.connect(store, timeout=.02)
                            connection.close()
                            self.assertIsNotNone(connection.process.poll())
                            self.assertTrue(transport.closed)
                            self.assertFalse(transport.busy.locked())
                            if reuse:
                                self.assertIsNone(connection._slot)
                                self.assertFalse(finalizer.alive)
                            connection.close()
                            with readonly.connect(store, reuse=reuse) as later:
                                self.assertEqual(later.execute('SELECT 1').fetchone()[0], 1)
                        finally:
                            connection.close()

    def test_cold_thread_start_failure_retains_transport_without_admission(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse), TemporaryDirectory() as root:
                store = SQLiteSwarmStore(root)
                store.ensure_schema()
                before = set(readonly._cleanup.owners)
                with patch.object(readonly.threading.Thread, 'start', side_effect=RuntimeError('start failed')), \
                        patch.object(readonly.ReaderProcess, 'terminate', side_effect=OSError('terminate failed')):
                    with self.assertRaisesRegex(OSError, 'terminate failed'):
                        readonly.connect(store, reuse=reuse)
                tokens = set(readonly._cleanup.owners) - before
                self.assertEqual(len(tokens), 1)
                token = tokens.pop()
                owner = readonly._cleanup.owners[token]
                self.assertTrue(owner.retired)
                self.assertIsNone(owner.permit)
                with readonly.connect(store) as later:
                    self.assertEqual(later.execute('SELECT 1').fetchone()[0], 1)
                owner.transport.reader.start()
                readonly._cleanup.close(token)
                self.assertTrue(owner.transport.closed)

    def test_constructor_teardown_failure_retains_transport_and_admission(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse), TemporaryDirectory() as root:
                store = SQLiteSwarmStore(root)
                store.ensure_schema()
                before = set(readonly._cleanup.owners)
                with patch.object(readonly.ReadConnection, '_receive',
                                  side_effect=readonly.ReadUnavailable('injected response failure')), \
                        patch.object(readonly.ReaderProcess, 'terminate', side_effect=OSError('terminate failed')):
                    with self.assertRaisesRegex(readonly.ReadUnavailable,
                                                'injected response failure') as caught:
                        readonly.connect(store, reuse=reuse)
                self.assertIsInstance(caught.exception.__cause__, OSError)
                self.assertEqual(str(caught.exception.__cause__), 'terminate failed')
                tokens = set(readonly._cleanup.owners) - before
                self.assertEqual(len(tokens), 1)
                token = tokens.pop()
                owner = readonly._cleanup.owners[token]
                self.assertIsNotNone(owner.permit)
                try:
                    with patch.object(owner.transport.process, 'terminate', side_effect=OSError('still unavailable')):
                        with self.assertRaises(readonly.ReadTimeout):
                            readonly.connect(store, timeout=.05)
                    readonly._cleanup.close(token)
                    with readonly.connect(store) as later:
                        self.assertEqual(later.execute('SELECT 1').fetchone()[0], 1)
                finally:
                    readonly._cleanup.close(token)

    def test_failed_release_ack_teardown_retries_only_on_later_close(self):
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            connection = readonly.connect(store, reuse=True)
            try:
                with patch.object(connection, '_receive', side_effect=OSError('lost ack')), \
                        patch.object(connection.process, 'terminate',
                                     side_effect=OSError('injected terminate')) as terminate:
                    with self.assertRaisesRegex(OSError, 'injected terminate'):
                        connection.close()
                    self.assertEqual(terminate.call_count, 1)
                    self.assertIsNone(connection._slot)
                    with self.assertRaises(readonly.ReadTimeout):
                        readonly.connect(store, timeout=.02)
                with patch.object(connection, '_control', side_effect=AssertionError('must retry teardown')):
                    connection.close()
                self.assertIsNotNone(connection.process.poll())
                with readonly.connect(store, reuse=True) as later:
                    self.assertEqual(later.execute('SELECT 1').fetchone()[0], 1)
            finally:
                connection.close()

    def test_aliases_share_database_identity(self):
        import os
        import time
        from pathlib import Path
        from puppetmaster.readonly_admission import ReaderAdmission
        with TemporaryDirectory() as root:
            source = Path(root) / 'Database'
            source.touch()
            hardlink = Path(root) / 'linked'
            os.link(source, hardlink)
            aliases = [hardlink]
            case_alias = source.with_name('DATABASE')
            if case_alias.exists() and os.path.samefile(source, case_alias):
                aliases.append(case_alias)
            permit = ReaderAdmission(source, time.monotonic() + 1)
            try:
                for alias in aliases:
                    with self.subTest(alias=alias), self.assertRaises(readonly.ReadTimeout):
                        ReaderAdmission(alias, time.monotonic() + .02)
            finally:
                permit.release()
            for alias in aliases:
                ReaderAdmission(alias, time.monotonic() + 1).release()

    def test_outer_readmission_obeys_existing_ordinary_window(self):
        import io
        import json
        import queue
        import threading
        import weakref
        from types import SimpleNamespace
        for reuse, window in ((True, .1), (False, 1.0)):
            with TemporaryDirectory() as root:
                store = SQLiteSwarmStore(root)
                store.ensure_schema()
                cleanup = readonly.CleanupRegistry()
                now = [0.0]
                deadlines = []
                class Transport:
                    def __init__(self, path, deadline=None):
                        self.closed = False
                        self.calls = 0
                        self.busy = threading.Lock()
                        self.process = SimpleNamespace(stdin=io.StringIO())
                        self.responses = SimpleNamespace(get=self.get)
                        self.token = cleanup.register(self, readonly.selection(store)[1])

                    def ready(self, deadline):
                        pass

                    def get(self, timeout):
                        self.calls += 1
                        if self.calls == 1:
                            return json.dumps(dict(session_closed=True, kind='unavailable',
                                error='unable to open database: live sidecars'))
                        now[0] += timeout
                        raise queue.Empty()

                    def close(self, deadline=None):
                        self.closed = True

                original = readonly.ReaderAdmission
                def admission(path, deadline, **kwargs):
                    deadlines.append(deadline)
                    return original(path, deadline, **kwargs)
                timer = SimpleNamespace(monotonic=lambda: now[0],
                    sleep=lambda delay: now.__setitem__(0, now[0] + delay))
                with patch.object(readonly, '_cleanup', cleanup), \
                        patch.object(readonly, '_reuse_slots', weakref.WeakValueDictionary()), \
                        patch.object(readonly, '_reuse_lock', threading.Lock()), \
                        patch.object(readonly, '_Transport', Transport), \
                        patch.object(readonly, 'time', timer), \
                        patch.object(readonly, 'ReaderAdmission', admission):
                    with self.assertRaises(readonly.ReadTimeout):
                        readonly.connect(store, reuse=reuse)
                self.assertEqual(deadlines, [5, window])
                self.assertEqual(now[0], window)
                cleanup.shutdown()

    @unittest.skipUnless(hasattr(__import__('os'), 'fork'), 'requires fork')
    def test_fork_callbacks_and_inherited_locked_registry_are_bounded(self):
        import subprocess
        import sys
        # A separate interpreter controls callback registration order and bounds
        # failures without leaving deadlocked children in the test runner.
        script = r'''
import os, sys, time, threading
from pathlib import Path
from tempfile import TemporaryDirectory
box = {}
def earlier():
    import puppetmaster.readonly_admission as admission
    box['child'] = admission.ReaderAdmission(box['other'], time.monotonic()+.2)
os.register_at_fork(after_in_child=earlier)
from puppetmaster import readonly
import puppetmaster.readonly_admission as admission
def later():
    admission._after_fork()
    try:
        admission.ReaderAdmission(box['other'], time.monotonic()+.02)
    except readonly.ReadTimeout:
        box['safe'] = True
os.register_at_fork(after_in_child=later)
with TemporaryDirectory() as root:
    box['other'] = Path(root)/'other'
    box['other'].touch()
    parent = admission.ReaderAdmission(root, time.monotonic()+1)
    ready, done = threading.Event(), threading.Event()
    def hold():
        with admission._registry_lock:
            ready.set()
            done.wait(3)
    thread = threading.Thread(target=hold)
    thread.start()
    assert ready.wait(1)
    pid = os.fork()
    if pid == 0:
        try:
            assert box['safe']
            parent.release()
            try:
                admission.ReaderAdmission(root, time.monotonic()+.02)
            except readonly.ReadTimeout:
                box['child'].release()
                os._exit(0)
            os._exit(2)
        except BaseException:
            os._exit(3)
    done.set()
    thread.join(1)
    end = time.monotonic()+3
    while time.monotonic()<end:
        child, status = os.waitpid(pid, os.WNOHANG)
        if child:
            assert status == 0, status
            break
        time.sleep(.01)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        raise AssertionError('fork callback deadlock')
    parent.release()
'''
        subprocess.run([sys.executable, '-c', script], check=True, timeout=8)

    @unittest.skipUnless(hasattr(__import__('os'), 'fork'), 'requires fork')
    def test_fork_at_close_boundary_does_not_inherit_unregistered_lock(self):
        import subprocess
        import sys
        script = r"""
import os, time, threading
from tempfile import TemporaryDirectory
from unittest.mock import patch
from puppetmaster import readonly_admission as admission
with TemporaryDirectory() as root:
    permit = admission.ReaderAdmission(root, time.monotonic()+1)
    boundary, forking = threading.Event(), threading.Event()
    real_close = os.close
    original_pid = os.getpid()
    closing_fd = permit.fd
    def close(fd):
        if fd == closing_fd and os.getpid() == original_pid:
            boundary.set()
            assert forking.wait(2)
        real_close(fd)
    os.register_at_fork(before=forking.set)
    ready_r, ready_w = os.pipe()
    done_r, done_w = os.pipe()
    with patch.object(admission.os, 'close', close):
        thread = threading.Thread(target=permit.release)
        thread.start()
        assert boundary.wait(2)
        child = os.fork()
        if child == 0:
            os.write(ready_w, b'1')
            os.read(done_r, 1)
            os._exit(0)
    try:
        thread.join(2)
        assert not thread.is_alive()
        assert os.read(ready_r, 1) == b'1'
        admission.ReaderAdmission(root, time.monotonic()+.2).release()
    finally:
        os.write(done_w, b'1')
        assert os.waitpid(child, 0)[1] == 0
        for fd in (ready_r, ready_w, done_r, done_w):
            os.close(fd)
"""
        subprocess.run([sys.executable, '-c', script], check=True, timeout=8)

    def test_active_connection_parent_death_allows_later_reader(self):
        import subprocess
        import sys
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            process = subprocess.Popen([sys.executable, '-c',
                'import sys; from puppetmaster.sqlite_store import SQLiteSwarmStore; '
                'from puppetmaster import readonly; '
                'c=readonly.connect(SQLiteSwarmStore(sys.argv[1])); '
                'print(c.execute("SELECT 42").fetchone()[0],flush=True); sys.stdin.read()', root],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), '42')
                process.kill()
                process.wait(timeout=5)
                with readonly.connect(store, attach_binding=True) as later:
                    self.assertEqual(later.execute('SELECT 1').fetchone()[0], 1)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()

    def test_windows_file_lock_contract(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock
        from pathlib import Path
        from puppetmaster import readonly_admission as admission
        for deadline in (0, .003, .025):
            for busy in (False, True):
                with self.subTest(deadline=deadline, busy=busy), TemporaryDirectory() as root:
                    now = [0.0]
                    timer = SimpleNamespace(monotonic=lambda: now[0],
                        sleep=lambda delay: now.__setitem__(0, now[0]+delay))
                    api = Mock()
                    api.open.return_value = 123
                    api.lock.return_value = not busy
                    with patch.object(admission, '_windows_api', return_value=api), \
                            patch.object(admission, '_windows_directory', return_value=Path(root)), \
                            patch.dict('os.environ', {'LOCALAPPDATA': '/wrong', 'HOME': '/wrong', 'TMP': '/wrong'}):
                        if busy:
                            with self.assertRaises(readonly.ReadTimeout):
                                admission._WindowsFileLock((4, 5), deadline, timer)
                            self.assertAlmostEqual(now[0], deadline)
                        else:
                            permit = admission._WindowsFileLock((4, 5), deadline, timer)
                            thread = threading.Thread(target=permit.release)
                            thread.start()
                            thread.join(1)
                            self.assertIsNone(permit.handle)
                            permit.release()
                        api.close.assert_called_once_with(123)
                        target = api.open.call_args[0][0]
                        self.assertEqual(target.parent, Path(root)/'PuppetmasterReaders')
                        api.validate.assert_called_once_with(123)

    def test_windows_failures_retain_native_error_and_failed_close_handle(self):
        import time
        from pathlib import Path
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        for stage in ('open', 'validate', 'lock', 'close'):
            with self.subTest(stage=stage), TemporaryDirectory() as root:
                api = Mock()
                api.open.return_value = 123
                getattr(api, stage).side_effect = OSError(1234, 'native failure')
                with patch.object(admission, '_windows_api', return_value=api), \
                        patch.object(admission, '_windows_directory', return_value=Path(root)):
                    if stage == 'close':
                        permit = admission._WindowsFileLock((1, 2), time.monotonic()+1, time)
                        with self.assertRaisesRegex(OSError, '1234'):
                            permit.release()
                        self.assertEqual(permit.handle, 123)
                        api.close.side_effect = None
                        permit.release()
                    else:
                        with self.assertRaisesRegex(OSError, '1234'):
                            admission._WindowsFileLock((1, 2), time.monotonic()+1, time)
                        self.assertEqual(api.close.call_count, int(stage != 'open'))

    def test_windows_kernel_flags_validation_and_native_failures(self):
        import ctypes
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        kernel = Mock()
        kernel.CreateFileW.return_value = 123
        def info(handle, pointer):
            pointer._obj.attributes = 0x80
            pointer._obj.links = 1
            return True
        kernel.GetFileInformationByHandle.side_effect = info
        native = lambda code: OSError(code, 'native Windows error')
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
                patch.object(ctypes, 'get_last_error', return_value=5, create=True), \
                patch.object(ctypes, 'WinError', side_effect=native, create=True):
            api = admission._windows_api()
            self.assertEqual(api.open('stable.lock'), 123)
            args = kernel.CreateFileW.call_args[0]
            self.assertEqual(args[2:6], (3, None, 4, 0x00200080))
            api.validate(123)
            kernel.SetHandleInformation.assert_called_once_with(123, 1, 0)
            self.assertTrue(api.lock(123))
            self.assertEqual(kernel.LockFileEx.call_args[0][1:5], (3, 0, 1, 0))
            for attributes, links in ((0x400, 1), (0x10, 1), (0x80, 2)):
                def unsafe(handle, pointer):
                    pointer._obj.attributes = attributes
                    pointer._obj.links = links
                    return True
                kernel.GetFileInformationByHandle.side_effect = unsafe
                with self.assertRaisesRegex(OSError, 'unsafe'):
                    api.validate(123)
            kernel.LockFileEx.return_value = False
            with self.assertRaisesRegex(OSError, '5'):
                api.lock(123)
            with patch.object(ctypes, 'get_last_error', return_value=33):
                self.assertFalse(api.lock(123))
            kernel.CreateFileW.return_value = ctypes.c_void_p(-1).value
            with self.assertRaisesRegex(OSError, '5'):
                api.open('stable.lock')
            kernel.CloseHandle.return_value = False
            with self.assertRaisesRegex(OSError, '5'):
                api.close(123)

    def test_windows_acquiring_thread_exit_does_not_release_and_stale_file_reopens(self):
        import threading
        import time
        from pathlib import Path
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        with TemporaryDirectory() as root:
            api = Mock()
            api.open.return_value = 123
            owners = []
            with patch.object(admission, '_windows_api', return_value=api), \
                    patch.object(admission, '_windows_directory', return_value=Path(root)):
                thread = threading.Thread(target=lambda: owners.append(
                    admission._WindowsFileLock((1, 2), time.monotonic()+1, time)))
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(owners), 1)
                api.close.assert_not_called()
                owners.pop().release()
                # No abandoned-owner marker exists: only the kernel lock state
                # matters, even if a previous process left the file on disk.
                target = api.open.call_args[0][0]
                target.touch()
                admission._WindowsFileLock((1, 2), time.monotonic()+1, time).release()
                self.assertEqual(api.lock.call_count, 2)
                self.assertEqual(api.close.call_count, 2)

    def test_windows_known_folder_is_independent_of_environment(self):
        import ctypes
        from pathlib import Path
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        shell, ole = Mock(), Mock()
        def known_folder(folder, flags, token, result):
            result._obj.value = '/os-user-data'
            return 0
        shell.SHGetKnownFolderPath.side_effect = known_folder
        with patch.object(ctypes, 'WinDLL', side_effect=[shell, ole, shell, ole], create=True):
            for value in ('/first', '/second'):
                with patch.dict('os.environ', dict(LOCALAPPDATA=value, HOME=value, TMP=value)):
                    self.assertEqual(admission._windows_directory(), Path('/os-user-data'))
        self.assertEqual(ole.CoTaskMemFree.call_count, 2)

    def test_windows_constructor_cleanup_failure_retains_owner(self):
        import time
        from pathlib import Path
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        with TemporaryDirectory() as root:
            api = Mock()
            api.open.return_value = 123
            api.validate.side_effect = OSError(5, 'validation')
            api.close.side_effect = OSError(6, 'close')
            with patch.object(admission, '_windows_api', return_value=api), \
                    patch.object(admission, '_windows_directory', return_value=Path(root)):
                with self.assertRaisesRegex(OSError, '6') as caught:
                    admission._WindowsFileLock((1, 2), time.monotonic()+1, time)
            owner = caught.exception.admission_owner
            self.assertEqual(owner.handle, 123)
            self.assertEqual(caught.exception.__context__.errno, 5)
            api.close.side_effect = None
            owner.release()

    def test_windows_admission_prevents_recursive_bypass(self):
        from unittest.mock import Mock
        from puppetmaster import readonly_admission as admission
        import time
        lock = Mock()
        with patch.object(admission.os, 'name', 'nt'), \
                patch.object(admission, '_WindowsFileLock', return_value=lock) as create:
            first = admission.ReaderAdmission(None, time.monotonic()+1, selected=(9, 10))
            try:
                with self.assertRaises(readonly.ReadTimeout):
                    admission.ReaderAdmission(None, time.monotonic(), selected=(9, 10))
                create.assert_called_once()
            finally:
                first.release()
            admission.ReaderAdmission(None, time.monotonic()+1, selected=(9, 10)).release()
            self.assertEqual(lock.release.call_count, 2)

    def test_selected_identity_requires_no_extra_database_open(self):
        import os
        import time
        from pathlib import Path
        from puppetmaster import readonly_admission as admission
        with TemporaryDirectory() as root:
            path = Path(root) / 'db'
            path.touch()
            selected = path.stat().st_dev, path.stat().st_ino
            real_open = os.open
            def checked(target, *args, **kwargs):
                self.assertNotEqual(Path(target), path)
                return real_open(target, *args, **kwargs)
            with patch.object(admission.os, 'open', checked):
                admission.ReaderAdmission(path, time.monotonic()+1, selected=selected).release()

    def test_hardlinked_stores_ready_before_waiting_for_source(self):
        import os
        from pathlib import Path
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(Path(root) / 'source')
            store.ensure_schema()
            alias_root = Path(root) / 'alias'
            alias_root.mkdir()
            os.link(store.db_path, alias_root / 'state.sqlite3')
            alias = SQLiteSwarmStore(alias_root)
            with readonly.connect(store), \
                    patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn:
                with self.assertRaises(readonly.ReadTimeout):
                    readonly.connect(alias, timeout=.02)
                self.assertEqual(spawn.call_count, 1)
            with readonly.connect(alias) as connection:
                self.assertEqual(connection.execute('SELECT 1').fetchone()[0], 1)
