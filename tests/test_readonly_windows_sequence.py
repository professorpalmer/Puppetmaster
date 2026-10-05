"""Native metadata interleavings, without requiring a Windows test host."""
from tests.readonly_fixtures import ProtocolTransport
import ctypes
import io
import json
import queue
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import ANY, patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from puppetmaster import readonly, readonly_worker as worker
from puppetmaster.sqlite_store import SQLiteSwarmStore, _is_sqlite_lock_error


class WindowsSequenceTests(unittest.TestCase):
    def native_stamp(self, times, sizes):
        epoch = 116444736000000000
        values = iter(times)
        def query(handle, kind, pointer, size):
            write, change = next(values)
            pointer._obj.write = epoch + write
            pointer._obj.change = epoch + change
            return True
        kernel = SimpleNamespace(GetFileInformationByHandleEx=query)
        stats = [SimpleNamespace(st_dev=1, st_ino=2, st_size=size,
                                 st_mtime_ns=write * 100) for size, write in sizes]
        with patch.object(worker.os, 'name', 'nt'), \
                patch.dict(sys.modules, msvcrt=SimpleNamespace(get_osfhandle=lambda fd: fd)), \
                patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
                patch.object(worker.os, 'fstat', side_effect=stats) as stat:
            result = worker.source_stamp(fd=123)
        return result, stat.call_count

    def test_size_and_changetime_cannot_come_from_different_writes(self):
        # A WAL checkpoint grows the main file between BasicInfo and fstat;
        # LastWriteTime is published later, with the writer's close/flush.
        result, calls = self.native_stamp(
            [(10, 20), (10, 21), (10, 21), (10, 21)],
            [(200, 10), (200, 10)])
        descriptor, _ = self.native_stamp([(10, 21)] * 2, [(200, 10)])
        self.assertEqual(result, descriptor)
        self.assertEqual(result, (1, 2, 200, 1000, 2100))
        # The old mixed sample looked like ctime-only drift at descriptor bind.
        self.assertFalse(worker.same_store_write((1, 2, 200, 1000, 2000), descriptor))
        self.assertEqual(calls, 2)

    def test_unchanged_native_metadata_takes_one_sample(self):
        result, calls = self.native_stamp([(10, 20)] * 2, [(100, 10)])
        self.assertEqual(result, (1, 2, 100, 1000, 2000))
        self.assertEqual(calls, 1)

    def test_native_metadata_retry_is_bounded(self):
        with self.assertRaisesRegex(OSError, 'metadata did not stabilize'):
            self.native_stamp([(10, 20), (10, 21)] * 8, [(100, 10)] * 8)

    def test_numeric_lock_codes_are_authoritative(self):
        for code in (5, 6, 261, 262, 517):
            error = readonly.ReadUnavailable('unable to open database: active reader; sidecars may be missing')
            error.sqlite_errorcode = code
            self.assertTrue(_is_sqlite_lock_error(error))
        error = sqlite3.OperationalError('database is locked')
        error.sqlite_errorcode = 11
        self.assertFalse(_is_sqlite_lock_error(error))
        self.assertFalse(_is_sqlite_lock_error(readonly.ReadUnavailable('reader timed out')))

    def test_attach_does_not_rebind_after_unproven_source_change(self):
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            store = SQLiteSwarmStore(tmp)
            with patch.object(store, '_connect_readonly', side_effect=readonly.ReadUnavailable(
                    'unable to open database: source changed')) as connect, \
                    patch.object(store, '_sleep_lock_backoff') as backoff:
                with self.assertRaisesRegex(readonly.ReadUnavailable, 'source changed'):
                    store.attach()
            connect.assert_called_once_with(attach_binding=True, attach_deadline=ANY)
            backoff.assert_not_called()
            self.assertFalse(store._attached)

    def test_zero_busy_budget_observes_helper_lock_after_slow_start(self):
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            store = SQLiteSwarmStore(tmp)
            store.busy_timeout_ms = 0
            clock = [0.0]
            waits = []
            class Transport(ProtocolTransport):
                def __init__(self, path, deadline=None):
                    super().__init__(path)
                    import threading
                    self.busy = threading.Lock()
                    self.process = SimpleNamespace(stdin=__import__('io').StringIO())
                    self.responses = SimpleNamespace(get=self.get)
                    self.closed = False
                    clock[0] += .2  # Windows startup exceeds the former 100 ms.
                def get(self, timeout):
                    waits.append(timeout)
                    if timeout < .2:
                        raise queue.Empty()
                    return json.dumps(dict(session_closed=True, kind='unavailable', code=5,
                        error='unable to open database: active reader; sidecars may be missing'))
                def close(self, deadline=None):
                    self.closed = True
            with patch.object(readonly, '_Transport', Transport), \
                    patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0])), \
                    patch.object(store, '_sleep_lock_backoff') as backoff:
                with self.assertRaises(readonly.ReadUnavailable) as caught:
                    store.attach()
            self.assertTrue(_is_sqlite_lock_error(caught.exception))
            self.assertEqual(caught.exception.sqlite_errorcode, 5)
            self.assertEqual(len(waits), 5)
            for timeout in waits:
                self.assertAlmostEqual(timeout, 4.8)
            self.assertEqual(store.lock_error_count, 5)
            self.assertEqual([call.args for call in backoff.call_args_list],
                             [(0,), (1,), (2,), (3,)])
            self.assertFalse(store._attached)

    def test_attach_nonresponding_helper_gets_one_startup_allowance(self):
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            store = SQLiteSwarmStore(tmp)
            store.busy_timeout_ms = 0
            waits = []
            transports = []
            class Transport(ProtocolTransport):
                def __init__(self, path, deadline=None):
                    super().__init__(path)
                    import threading
                    self.busy = threading.Lock()
                    self.process = SimpleNamespace(stdin=__import__('io').StringIO())
                    self.responses = SimpleNamespace(get=self.get)
                    self.closed = False
                    transports.append(self)
                def get(self, timeout):
                    waits.append(timeout)
                    raise queue.Empty()
                def close(self, deadline=None):
                    self.closed = True
            with patch.object(readonly, '_Transport', Transport), \
                    patch.object(store, '_sleep_lock_backoff') as backoff:
                with self.assertRaisesRegex(readonly.ReadUnavailable, 'reader timed out'):
                    store.attach()
            self.assertEqual(len(waits), 1)
            self.assertGreater(waits[0], 4.9)
            self.assertLessEqual(waits[0], 5.0)
            self.assertEqual(len(transports), 1)
            self.assertTrue(transports[0].closed)
            backoff.assert_not_called()
            self.assertEqual(store.lock_error_count, 0)
            self.assertFalse(store._attached)

    def test_windows_attach_retries_only_after_active_wal_proof(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            replies = iter([
                json.dumps(dict(session_closed=True, kind='unavailable', code=5,
                    wal_snapshot=True,
                    error='unable to open database: active reader; sidecars may be missing')),
                json.dumps(dict(journal='wal')),
            ])

            class Permit:
                def release(self):
                    pass

            class Transport:
                def __init__(self, path, deadline=None):
                    self.closed = False
                    self.busy = threading.Lock()
                    self.process = SimpleNamespace(stdin=io.StringIO())
                    self.responses = SimpleNamespace(get=lambda timeout: next(replies))
                    self.token = readonly._cleanup.register(self, readonly.selection(store)[1])

                def ready(self, deadline):
                    pass

                def close(self, deadline=None):
                    self.closed = True

            with patch.object(readonly, '_Transport', Transport), \
                    patch.object(readonly, 'ReaderAdmission', return_value=Permit()), \
                    patch.object(readonly.os, 'name', 'nt'), \
                    patch.object(readonly, 'source_stamp', return_value=(1, 2, 3, 4, 5)), \
                    patch.object(readonly, '_source_stamp', return_value=(
                        (1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
                        None, None, None, None)):
                connection = readonly.ReadConnection(
                    store, 1, attach_binding=True, attach_deadline=time.monotonic() + 1)
            requests = [json.loads(line) for line in connection.process.stdin.getvalue().splitlines()]
            self.assertEqual([request['wal_snapshot'] for request in requests], [False, True])
            connection.close()

    def assert_guard_open_denial_retried(self, attach_budget=None, **binding):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            # Start the clock after fixture setup: a slow ensure_schema on a
            # loaded Windows runner used to spend the whole budget before the
            # retry under test ran.
            if attach_budget is not None:
                binding['attach_deadline'] = time.monotonic() + attach_budget
            replies = iter([
                json.dumps(dict(
                    session_closed=True,
                    kind='unavailable',
                    error='unable to open database: source changed or unavailable: '
                          '[WinError 5] Access is denied.',
                    source_open_contention=True,
                )),
                json.dumps(dict(journal='delete')),
            ])

            class Permit:
                def release(self):
                    pass

            class Transport:
                def __init__(self, path, deadline=None):
                    self.closed = False
                    self.busy = threading.Lock()
                    self.process = SimpleNamespace(stdin=io.StringIO())
                    self.responses = SimpleNamespace(get=lambda timeout: next(replies))
                    self.token = readonly._cleanup.register(self, readonly.selection(store)[1])

                def ready(self, deadline):
                    pass

                def close(self, deadline=None):
                    self.closed = True

            with patch.object(readonly, '_Transport', Transport), \
                    patch.object(readonly, 'ReaderAdmission', return_value=Permit()), \
                    patch.object(readonly, 'source_stamp', return_value=(1, 2, 3, 4, 5)), \
                    patch.object(readonly, '_source_stamp', return_value=(
                        (1, 2, 3, 4, 5), None, None, None, None, None)):
                connection = readonly.ReadConnection(store, 1, **binding)
            requests = connection.process.stdin.getvalue().splitlines()
            self.assertEqual(len(requests), 2)
            connection.close()

    def test_windows_attach_retries_transient_guard_open_denial(self):
        self.assert_guard_open_denial_retried(attach_binding=True, attach_budget=2)

    def test_windows_launch_retries_transient_guard_open_denial(self):
        self.assert_guard_open_denial_retried(launch_binding=True)


if __name__ == '__main__':
    unittest.main()
