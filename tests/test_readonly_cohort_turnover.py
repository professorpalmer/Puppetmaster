"""A WAL cohort that closes during a helper open is contention, not a new source.

Every store call opens and closes its own connection. The last close
checkpoints and deletes the -wal/-shm pair. When the helper takes its first
stamps while that pair exists and takes its lock after the pair is gone, the
stamps differ only in the sidecars. The main file is the same file with the
same bytes, so ordinary and reuse reads must retry, like "live sidecars".
"""
from contextlib import closing
import io
import json
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from puppetmaster import identity, readonly, readonly_worker as worker
from puppetmaster.sqlite_store import SQLiteSwarmStore


@unittest.skipIf(sys.platform == 'win32', 'POSIX quiet-source path')
class CohortTurnoverWorkerTests(unittest.TestCase):
    def test_closing_writer_between_stamps_and_lock_is_a_topology_change(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.sqlite3'
            with closing(sqlite3.connect(path)) as c, c:
                c.execute('PRAGMA journal_mode=WAL')
                c.execute('CREATE TABLE sample(value)')
                c.execute('INSERT INTO sample VALUES(1)')
            self.assertFalse(Path(str(path) + '-wal').exists())
            original = worker.stamps
            taken = []

            def closing_writer(path_arg):
                # The first stamps see a live cohort of a read-only writer.
                # The writer then closes: SQLite deletes the pair, and the
                # main file keeps its inode, size and times.
                if taken:
                    return original(path_arg)
                writer = sqlite3.connect(path)
                writer.execute('SELECT count(*) FROM sample').fetchone()
                result = original(path_arg)
                writer.close()
                taken.append(result)
                return result

            responses = []
            with patch.object(worker, 'stamps', side_effect=closing_writer), \
                    patch.object(worker, 'emit', side_effect=responses.append):
                worker.main(path)
            before = taken[0]
            self.assertIsNotNone(before[1])
            self.assertIsNotNone(before[2])
            self.assertEqual(original(path)[0], before[0])
            self.assertEqual(responses, [dict(
                kind='unavailable', error='unable to open database: source changed',
                launch_topology_change=True, same_store_write=False)])

            # The retry sees a quiet source and reads it.
            responses.clear()
            with patch.object(worker, 'emit', side_effect=responses.append), \
                    patch.object(worker.sys, 'stdin', io.StringIO(
                        '{"sql":"SELECT value FROM sample","parameters":[]}\n')):
                worker.main(path)
            self.assertEqual(responses[-1]['rows'], [(1,)])


class CohortTurnoverRetryTests(unittest.TestCase):
    def _bind_after_turnover(self, options, elapsed=0.0, **proof):
        """Bind once; the first open answers with the helper's turnover error.

        ``elapsed`` is the time the helper took for that first answer.
        """
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            original = readonly.ReadConnection._receive
            turned = []
            clock = [0.0]

            def sleep(delay):
                clock[0] += delay

            def race(c):
                response = original(c)
                if not c._opened and not turned:
                    # Park the real session, then answer this open with the
                    # error the helper emits for a closed cohort.
                    c._opened = True
                    c._control('release', True)
                    c._opened = False
                    turned.append(c.transport)
                    clock[0] += elapsed
                    error = dict(session_closed=True, kind='unavailable',
                                 error='unable to open database: source changed', **proof)
                    with patch.object(c.responses, 'get', return_value=json.dumps(error)):
                        return original(c)
                return response

            try:
                with patch.object(readonly.ReadConnection, '_receive', race), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        patch.object(readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)):
                    with readonly.connect(store, **options) as c:
                        self.assertEqual(identity.read_identity(c, 'sqlite'), store._incarnation)
                        self.assertEqual(turned, [c.transport])
                    self.assertEqual(spawn.call_count, 1)
            finally:
                if getattr(store, '_readonly_transport', None):
                    store._readonly_transport.close()

    def test_ordinary_and_reuse_reads_retry_a_cohort_turnover(self):
        for options in ({}, {'reuse': True}):
            with self.subTest(options=options):
                self._bind_after_turnover(options, launch_topology_change=True, same_store_write=False)

    def test_slow_turnover_answer_still_retries_in_the_reuse_window(self):
        # Under load the first answer can take 60 ms of the 100 ms reuse
        # window. The writer is gone, so the retry must not wait 50 ms.
        self._bind_after_turnover({'reuse': True}, elapsed=.06,
                                  launch_topology_change=True, same_store_write=False)

    def test_source_change_without_proof_still_raises(self):
        # A changed main file (new inode, metadata drift) carries neither
        # proof. It is not contention and must not retry.
        for options in ({}, {'reuse': True}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(readonly.ReadUnavailable, 'source changed'):
                    self._bind_after_turnover(options, same_store_write=False)


if __name__ == '__main__':
    unittest.main()
