"""Only a stable reused-helper invalidation earns one fresh open."""
import json
import sqlite3
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401

from readonly_fixtures import close_new_cleanup_owners, damaged_sidecars

from puppetmaster import readonly, state
from puppetmaster.identity import StoreIdentityError
from puppetmaster.sqlite_store import SQLiteSwarmStore


class HelperInvalidationTests(unittest.TestCase):
    def setUp(self):
        close_new_cleanup_owners(self)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteSwarmStore(Path(self.directory.name) / 'state')
        self.job = self.store.create_job('first')
        self.reader = state._OwnershipReader()
        self.reader.root = self.store.root
        self.reader._read_selection = readonly.selection(self.reader)
        with readonly.connect(self.reader, reuse=True) as c:
            self.assertEqual(c.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)
        self.transport = self.reader._readonly_transport
        self.addCleanup(lambda: self.reader._readonly_transport.close())

    def invalidate(self, *, mutate=None, repeat=False, on_retry=None):
        receive = readonly.ReadConnection._receive
        clock = [0.0]
        attempts = []

        def injected(c):
            response = receive(c)
            if not c._opened:
                attempts.append(c.transport)
                if len(attempts) == 2 and on_retry:
                    on_retry()
                if len(attempts) == 1 or repeat:
                    if mutate:
                        mutate()
                    clock[0] += .2  # Startup can consume the old 100 ms retry window.
                    with patch.object(c.responses, 'get', return_value=json.dumps(dict(
                            kind='unavailable', error='unable to open database: source changed'))):
                        return receive(c)
            return response

        return attempts, patch.object(readonly.ReadConnection, '_receive', injected), patch.object(
            readonly, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))

    def test_stable_reused_helper_gets_one_fresh_attempt_after_slow_invalidation(self):
        attempts, inject, clock = self.invalidate()
        with inject, clock:
            self.assertTrue(state.state_owns_job(self.store.root, self.job.id, _reader=self.reader))
        self.assertEqual(len(attempts), 2)
        self.assertIs(attempts[0], self.transport)
        self.assertIsNot(attempts[1], self.transport)
        self.assertTrue(self.transport.closed)
        self.assertIsNotNone(self.transport.process.poll())

    def test_committed_write_between_sessions_is_visible(self):
        second = self.store.create_job('second')
        attempts, inject, clock = self.invalidate()
        with inject, clock:
            self.assertTrue(state.state_owns_job(self.store.root, second.id, _reader=self.reader))
        self.assertEqual(len(attempts), 2)

    def test_second_invalidation_is_explicit_and_bounded(self):
        attempts, inject, clock = self.invalidate(repeat=True)
        with inject, clock, self.assertRaisesRegex(readonly.ReadUnavailable, 'source changed'):
            readonly.connect(self.reader, reuse=True)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(t.closed for t in attempts))

    def test_expired_caller_deadline_does_not_refresh(self):
        attempts, inject, clock = self.invalidate()
        with inject, clock, self.assertRaises(readonly.ReadUnavailable):
            readonly.connect(self.reader, reuse=True, timeout=.1)
        self.assertEqual(len(attempts), 1)

    def test_fresh_helper_source_change_is_not_retried(self):
        self.transport.close()
        attempts, inject, clock = self.invalidate(repeat=True)
        with inject, clock, self.assertRaises(readonly.ReadUnavailable):
            readonly.connect(self.reader, reuse=False)
        self.assertEqual(len(attempts), 1)

    def test_source_change_while_fresh_helper_opens_is_explicit(self):
        attempts, inject, clock = self.invalidate(on_retry=self.store.db_path.touch)
        with inject, clock, self.assertRaisesRegex(readonly.ReadUnavailable, 'source changed'):
            readonly.connect(self.reader, reuse=True)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(t.closed for t in attempts))

    def test_committed_change_during_open_is_not_retried(self):
        def change():
            # The held advisory lock prevents a writer: change only after the
            # helper exits, at the same boundary as a failed open's cleanup.
            self.transport.close()
            self.store.create_job('concurrent commit')
        attempts, inject, clock = self.invalidate(mutate=change)
        with inject, clock, self.assertRaises(readonly.ReadUnavailable):
            readonly.connect(self.reader, reuse=True)
        self.assertEqual(len(attempts), 1)

    def test_same_path_replacement_and_aba_are_not_retried(self):
        for aba in (False, True):
            with self.subTest(aba=aba):
                path = self.store.db_path
                old = path.with_suffix('.old')
                def replace():
                    self.transport.close()
                    path.rename(old)
                    if aba:
                        old.rename(path)
                    else:
                        path.write_bytes(old.read_bytes())
                attempts, inject, clock = self.invalidate(mutate=replace)
                with inject, clock, self.assertRaises((readonly.ReadUnavailable, StoreIdentityError)):
                    readonly.connect(self.reader, reuse=True)
                self.assertEqual(len(attempts), 1)
                if not aba:
                    path.unlink()
                    old.rename(path)
                self.reader._read_selection = readonly.selection(self.reader)
                with readonly.connect(self.reader, reuse=True):
                    pass
                self.transport = self.reader._readonly_transport

    def test_live_wal_is_read_and_missing_wal_stays_unavailable(self):
        with closing(sqlite3.connect(self.store.db_path)) as writer:
            writer.execute("INSERT INTO jobs(id,data) VALUES('wal_job','{}')")
            writer.commit()
            # A complete live WAL is read through its committed snapshot; a
            # cohort missing a sidecar stays unavailable and is never rebuilt.
            self.assertTrue(state.state_owns_job(self.store.root, 'wal_job', _reader=self.reader))
            with damaged_sidecars(writer, self.store.db_path):
                with self.assertRaises(readonly.ReadUnavailable):
                    state.state_owns_job(self.store.root, 'wal_job', _reader=self.reader)
        self.assertTrue(state.state_owns_job(self.store.root, 'wal_job', _reader=self.reader))


if __name__ == '__main__':
    unittest.main()
