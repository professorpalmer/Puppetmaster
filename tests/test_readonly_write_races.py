"""Only proven pre-snapshot writes retry, in the same bounded helper."""
from tests.readonly_fixtures import ProtocolTransport
import io
import json
import queue
import sqlite3
import sys
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from puppetmaster import identity, readonly, readonly_worker as worker, sqlite_store
from puppetmaster.sqlite_store import SQLiteSwarmStore


class ReadonlyWriteRaceTests(unittest.TestCase):
    def test_attach_validation_batches_checks_under_exclusive_lock(self):
        with TemporaryDirectory() as tmp:
            supervisor = SQLiteSwarmStore(tmp)
            supervisor.ensure_schema()
            store = SQLiteSwarmStore(tmp)
            execute = readonly.ReadConnection.execute
            statements = []

            def observed(connection, sql, parameters=()):
                statements.append(sql)
                return execute(connection, sql, parameters)

            with patch.object(readonly.ReadConnection, 'execute', observed):
                store.attach()
            self.assertTrue(store._attached)
            self.assertEqual(store._incarnation, supervisor._incarnation)
            self.assertEqual(len(statements), 3)
            for table in ('completions', 'execution_attempts', 'usage_observations', 'budget_reservations'):
                with self.subTest(table=table):
                    with closing(sqlite3.connect(store.db_path)) as database:
                        database.execute(f'ALTER TABLE {table} RENAME TO hidden_table')
                        database.commit()
                    try:
                        with self.assertRaises(sqlite_store.SqliteSchemaError):
                            SQLiteSwarmStore(tmp).attach()
                    finally:
                        with closing(sqlite3.connect(store.db_path)) as database:
                            database.execute(f'ALTER TABLE hidden_table RENAME TO {table}')
                            database.commit()

    def test_writer_between_stat_and_descriptor_is_classified(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.sqlite3'
            with closing(sqlite3.connect(path)) as c, c:
                c.execute('CREATE TABLE sample(value)')
            original = worker.source_stamp
            written = []
            responses = []
            def race(path_arg=None, *, fd=None):
                if fd is not None and not written:
                    written.append(True)
                    with closing(sqlite3.connect(path)) as c, c:
                        c.execute('INSERT INTO sample VALUES(1)')
                return original(path_arg, fd=fd)
            with patch.object(worker, 'source_stamp', side_effect=race), \
                    patch.object(worker, 'emit', side_effect=responses.append):
                worker.main(path)
            self.assertEqual(len(responses), 1)
            self.assertTrue(responses[0]['same_store_write'])
            self.assertEqual(responses[0]['kind'], 'unavailable')
            responses.clear()
            with patch.object(worker, 'emit', side_effect=responses.append), \
                    patch.object(worker.sys, 'stdin', io.StringIO(
                        '{"sql":"SELECT value FROM sample","parameters":[]}\n')):
                worker.main(path)
            self.assertEqual(responses[-1]['rows'], [(1,)])

    def test_proven_writes_bind_with_one_helper_for_each_read_mode(self):
        for options, proof in (({}, 'same_store_write'), ({'reuse': True}, 'same_store_write'),
                               ({'attach_binding': True}, 'same_store_write'),
                               ({'attach_binding': True}, 'launch_topology_change'),
                               ({'launch_binding': True}, 'same_store_write')):
            with self.subTest(options=options, proof=proof), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                original = readonly.ReadConnection._receive
                retries = []
                # Startup, release, and the fixture write must not consume the
                # reuse mode's 100 ms budget before its synthetic error arrives.
                # Advance retry backoff deterministically; deadline/ABA behavior
                # is exercised separately below.
                clock = [0.0]
                def sleep(delay):
                    clock[0] += delay
                count = 1 if options.get('reuse') else 2
                def race(c):
                    response = original(c)
                    if not c._opened and len(retries) < count:
                        # Park this successful test session before injecting a
                        # failure at the helper's real failed-open boundary.
                        c._opened = True
                        c._control('release', True)
                        c._opened = False
                        retries.append(c.transport)
                        store.create_job('concurrent write')
                        error = dict(session_closed=True, kind='unavailable', error='unable to open database: source changed',
                                     **{proof: True})
                        with patch.object(c.responses, 'get', return_value=json.dumps(error)):
                            return original(c)
                    return response
                with patch.object(readonly.ReadConnection, '_receive', race), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)):
                    with readonly.connect(store, **options) as c:
                        self.assertEqual(identity.read_identity(c, 'sqlite'), store._incarnation)
                        self.assertEqual(c.execute('SELECT count(*) FROM jobs').fetchone()[0], count)
                        self.assertEqual(len(retries), count)
                        self.assertTrue(all(t is c.transport for t in retries))
                    self.assertEqual(spawn.call_count, 1)
                if getattr(store, '_readonly_transport', None):
                    store._readonly_transport.close()

    def test_write_retry_deadline_and_aba_reap_the_only_helper(self):
        for aba in (False, True):
            with self.subTest(aba=aba), TemporaryDirectory() as tmp:
                store = SQLiteSwarmStore(tmp)
                store.ensure_schema()
                original = readonly.ReadConnection._receive
                clock = [0.0]
                opened = []
                def race(c):
                    response = original(c)
                    if not c._opened:
                        opened.append(c)
                        c._opened = True
                        c._control('release', True)
                        c._opened = False
                        clock[0] += .1
                        error = dict(session_closed=True, kind='unavailable', error='unable to open database: source changed',
                                     same_store_write=True)
                        with patch.object(c.responses, 'get', return_value=json.dumps(error)):
                            return original(c)
                    return response
                def sleep(delay):
                    clock[0] += delay
                    if aba:
                        moved = store.db_path.with_suffix('.old')
                        store.db_path.rename(moved)
                        moved.rename(store.db_path)
                with patch.object(readonly.ReadConnection, '_receive', race), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)):
                    with self.assertRaises(identity.StoreIdentityError if aba else readonly.ReadUnavailable):
                        readonly.connect(store, attach_binding=True, timeout=.25)
                    self.assertEqual(spawn.call_count, 1)
                self.assertTrue(opened)
                for c in opened:
                    if not c.closed:
                        readonly._cleanup.close(c._token)
                        c.close()
                self.assertTrue(all(c.closed and c.process.poll() is not None for c in opened))
                self.assertLessEqual(clock[0], .35)

    def test_identity_metadata_and_sidecar_changes_are_not_write_proofs(self):
        before = (1, 2, 100, 200, 300)
        for after in (None, (1, 3, 100, 201, 301), (1, 2, 100, 200, 301)):
            self.assertFalse(worker.same_store_write(before, after))
        self.assertTrue(worker.same_store_write(before, (1, 2, 100, 201, 301)))
        # Windows may publish size before the last writer closes its handle.
        grown = (1, 2, 101, 200, 301)
        self.assertTrue(worker.same_store_write(before, grown))
        readonly._metadata_fence(before, grown)
        with self.assertRaises(identity.StoreIdentityError):
            readonly._metadata_fence(before, (1, 2, 100, 200, 301))

    def test_mixed_writes_and_locks_share_attach_deadline(self):
        for code in (5, 6):  # SQLITE_BUSY, SQLITE_LOCKED (constants absent on Python 3.9).
            with self.subTest(code=code), TemporaryDirectory() as tmp:
                supervisor = SQLiteSwarmStore(tmp)
                supervisor.ensure_schema()
                store = SQLiteSwarmStore(tmp)
                store.busy_timeout_ms = 100
                clock = [0.0]
                receive = readonly.ReadConnection._receive
                responses = {}

                def race(c):
                    # The clock is mocked; the real helper only has to answer.
                    real_get = c.responses.get
                    with patch.object(c.responses, 'get', lambda timeout=None: real_get(timeout=30)):
                        response = receive(c)
                    if not c._opened:
                        c._opened = True
                        c._control('release', True)
                        c._opened = False
                        count = responses.get(c, 0)
                        responses[c] = count + 1
                        # Deliver the numeric lock before the deadline; exhaustion
                        # before a response is covered by the silent-query test.
                        clock[0] += min(.08, c.timeout, max(0, .5 - clock[0] - .000001))
                        error = (dict(session_closed=True, kind='unavailable', error='unable to open database: source changed',
                                      same_store_write=True) if count < 2 else
                                 dict(session_closed=True, kind='unavailable', error='opaque contention', code=code))
                        with patch.object(c.responses, 'get', return_value=json.dumps(error)):
                            try:
                                return receive(c)
                            except sqlite3.OperationalError as exc:
                                if count >= 2:
                                    self.assertEqual(exc.sqlite_errorcode, code)
                                    self.assertTrue(readonly._locked(exc))
                                    self.assertTrue(sqlite_store._is_sqlite_lock_error(exc))
                                raise
                    return response

                def sleep(delay):
                    clock[0] += delay

                timer = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
                with patch.object(readonly.ReadConnection, '_receive', race), \
                        patch.object(readonly, 'time', timer), \
                        patch.object(sqlite_store, 'time', timer), \
                        patch.object(sqlite_store.random, 'uniform', return_value=0), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn:
                    with self.assertRaises(sqlite3.OperationalError):
                        store.attach()
                self.assertLessEqual(clock[0], .501)
                self.assertEqual(spawn.call_count, 1)
                self.assertEqual(store.lock_error_count, 1)
                self.assertFalse(store._attached)
                for c in responses:
                    if not c.closed:
                        readonly._cleanup.close(c._token)
                        c.close()
                self.assertTrue(all(c.closed and c.process.poll() is not None for c in responses))

    def test_attach_startup_uses_shared_deadline(self):
        for delay, prior_busy in ((6.0, False), (24.0, False), (None, False), (None, True)):
            with self.subTest(delay=delay, prior_busy=prior_busy), TemporaryDirectory() as tmp:
                SQLiteSwarmStore(tmp).ensure_schema()
                store = SQLiteSwarmStore(tmp)
                store.busy_timeout_ms = 5000
                clock = [0.0]
                waits = []
                transports = []
                before = {p.name: p.read_bytes() for p in store.root.iterdir() if p.is_file()}
                with closing(sqlite3.connect(store.db_path)) as database:
                    class Transport(ProtocolTransport):
                        def __init__(self, path, deadline=None):
                            super().__init__(path)
                            self.busy = threading.Lock()
                            self.process = SimpleNamespace(stdin=io.StringIO())
                            self.responses = SimpleNamespace(get=self.get)
                            self.closed = False
                            self.started = False
                            transports.append(self)

                        def get(self, timeout):
                            waits.append((clock[0], timeout))
                            if not self.started:
                                self.started = True
                                if prior_busy and len(transports) == 1:
                                    clock[0] += 6.0
                                    return json.dumps(dict(session_closed=True, kind='OperationalError',
                                        error='database is locked', code=5))
                                if delay is None or delay > timeout:
                                    clock[0] += timeout
                                    raise queue.Empty()
                                clock[0] += delay
                                return json.dumps(dict(journal='delete'))
                            request = json.loads(self.process.stdin.getvalue().splitlines()[-1])
                            if 'open' in request:
                                clock[0] += timeout
                                raise queue.Empty()
                            cursor = database.execute(request['sql'], request['parameters'])
                            return json.dumps(dict(rows=cursor.fetchall(),
                                names=[d[0] for d in cursor.description or ()]))

                        def close(self, deadline=None):
                            self.closed = True

                    def sleep(seconds):
                        clock[0] += seconds

                    timer = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
                    with patch.object(readonly, '_Transport', Transport), \
                            patch.object(readonly, 'time', timer), \
                            patch.object(sqlite_store, 'time', timer):
                        if delay is None:
                            with self.assertRaises(readonly.ReadTimeout):
                                store.attach()
                            self.assertEqual(clock[0], 11.05 if prior_busy else 25.0)
                        else:
                            store.attach()
                            self.assertTrue(store._attached)
                            self.assertEqual(clock[0], delay)
                            self.assertTrue(all(t <= 5 for _, t in waits[1:]))
                self.assertEqual(waits[0], (0.0, 25.0))
                self.assertTrue(all(start + t <= 25 for start, t in waits))
                self.assertEqual(len(transports), 1)
                self.assertTrue(all(t.closed and not t.busy.locked() for t in transports))
                self.assertEqual(store.lock_error_count, 0)
                self.assertEqual(store._attached, delay is not None)
                self.assertEqual(before,
                    {p.name: p.read_bytes() for p in store.root.iterdir() if p.is_file()})

    def test_attach_open_near_deadline_bounds_every_validation_response(self):
        # PRAGMA, batched schema validation, incarnation.
        for silent_query in range(3):
            for exhausted in (False, True):
                with self.subTest(query=silent_query, exhausted=exhausted), TemporaryDirectory() as tmp:
                    supervisor = SQLiteSwarmStore(tmp)
                    supervisor.ensure_schema()
                    store = SQLiteSwarmStore(tmp)
                    store.busy_timeout_ms = 5000
                    clock = [0.0]
                    transports = []
                    waits = []
                    with closing(sqlite3.connect(store.db_path)) as database:
                        class Transport(ProtocolTransport):
                            def __init__(self, path, deadline=None):
                                super().__init__(path)
                                self.busy = threading.Lock()
                                self.process = SimpleNamespace(stdin=io.StringIO())
                                self.responses = SimpleNamespace(get=self.get)
                                self.closed = False
                                self.opens = 0
                                self.queries = 0
                                transports.append(self)

                            def get(self, timeout):
                                waits.append((clock[0], timeout))
                                if self.opens < 4:
                                    self.opens += 1
                                    clock[0] += 4.99
                                    return json.dumps(dict(session_closed=True, kind='unavailable',
                                        error='unable to open database: source changed', same_store_write=True))
                                if self.opens == 4:
                                    self.opens += 1
                                    clock[0] = 25.0 if exhausted and silent_query == 0 else 24.0
                                    return json.dumps(dict(journal='delete'))
                                query = self.queries
                                self.queries += 1
                                if query == silent_query:
                                    clock[0] += timeout
                                    raise queue.Empty()
                                request = json.loads(self.process.stdin.getvalue().splitlines()[-1])
                                cursor = database.execute(request['sql'], request['parameters'])
                                result = dict(rows=cursor.fetchall(), names=[d[0] for d in cursor.description or ()])
                                if exhausted and query + 1 == silent_query:
                                    clock[0] = 25.0
                                return json.dumps(result)

                            def close(self, deadline=None):
                                self.closed = True

                        def sleep(delay):
                            clock[0] += delay

                        timer = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
                        with patch.object(readonly, '_Transport', Transport), \
                                patch.object(readonly, 'time', timer), \
                                patch.object(sqlite_store, 'time', timer):
                            with self.assertRaises(readonly.ReadTimeout):
                                store.attach()
                    self.assertLessEqual(clock[0], 25.0)
                    self.assertEqual(len(transports), 1)
                    self.assertTrue(transports[0].closed)
                    self.assertFalse(transports[0].busy.locked())
                    self.assertFalse(store._attached)
                    self.assertEqual(store.lock_error_count, 0)
                    self.assertTrue(all(start < 25 and timeout <= 25 - start for start, timeout in waits))

    def test_attach_proven_write_uses_bounded_aggregate_budget_in_one_helper(self):
        for outcome in ('success', 'exhausted', 'aba', 'replacement', 'unproven', 'silent'):
            with self.subTest(outcome=outcome), TemporaryDirectory() as tmp:
                supervisor = SQLiteSwarmStore(Path(tmp) / 'state')
                supervisor.ensure_schema()
                store = SQLiteSwarmStore(supervisor.root)
                store.busy_timeout_ms = 100
                receive = readonly.ReadConnection._receive
                clock = [0.0]
                failed = []
                snapshot = []

                def files():
                    return {p.name: (p.read_bytes(), worker.source_stamp(p))
                            for p in store.root.iterdir() if p.is_file()}

                def race(c):
                    if not c._opened and failed and outcome == 'silent':
                        self.assertLessEqual(c.timeout, .1)
                        with patch.object(c.responses, 'get', side_effect=queue.Empty):
                            return receive(c)
                    # The clock is mocked; the real helper only has to answer
                    # at all, not within the 100ms busy budget of wall time.
                    real_get = c.responses.get
                    with patch.object(c.responses, 'get', lambda timeout=None: real_get(timeout=30)):
                        response = receive(c)
                    if not c._opened and (not failed or outcome in ('exhausted', 'aba', 'replacement')):
                        c._opened = True
                        c._control('release', True)
                        c._opened = False
                        failed.append(c)
                        if len(failed) == 1 or outcome == 'exhausted':
                            supervisor.create_job('concurrent write')
                        snapshot[:] = [files()]
                        # The initial response arrived just before the busy
                        # deadline; handling it/backoff consumed the remainder.
                        clock[0] += .101
                        error = dict(session_closed=True, kind='unavailable', error='unable to open database: source changed',
                                     same_store_write=outcome != 'unproven')
                        with patch.object(c.responses, 'get', return_value=json.dumps(error)):
                            return receive(c)
                    return response

                def sleep(delay):
                    clock[0] += delay
                    if outcome in ('aba', 'replacement') and len(failed) == 2:
                        moved = store.db_path.with_suffix('.old')
                        store.db_path.rename(moved)
                        if outcome == 'aba':
                            moved.rename(store.db_path)
                        else:
                            store.db_path.write_bytes(moved.read_bytes())

                with patch.object(readonly.ReadConnection, '_receive', race), \
                        patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)), \
                        patch.object(sqlite_store, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        patch.object(store, '_sleep_lock_backoff') as backoff:
                    if outcome == 'success':
                        store.attach()
                        self.assertTrue(store._attached)
                        self.assertEqual(store._incarnation, supervisor._incarnation)
                    else:
                        error = (identity.StoreIdentityError if outcome in ('aba', 'replacement') else
                                 readonly.ReadTimeout if outcome == 'silent' else readonly.ReadUnavailable)
                        with self.assertRaises(error):
                            store.attach()
                        self.assertFalse(store._attached)
                    self.assertEqual(spawn.call_count, 1)
                    backoff.assert_not_called()
                self.assertEqual(store.lock_error_count, 0)
                for c in failed:
                    if not c.closed:
                        readonly._cleanup.close(c._token)
                        c.close()
                self.assertTrue(all(c.closed and c.process.poll() is not None for c in failed))
                self.assertEqual(len(failed), 5 if outcome == 'exhausted' else
                                 2 if outcome in ('aba', 'replacement') else 1)
                self.assertLessEqual(clock[0], .55)
                if outcome in ('success', 'exhausted', 'unproven', 'silent'):
                    self.assertEqual(files(), snapshot[0])


if __name__ == '__main__':
    unittest.main()
