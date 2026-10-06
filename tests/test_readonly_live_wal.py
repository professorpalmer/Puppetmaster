"""POSIX observer reads join a live WAL instead of blocking or refusing."""
import os
import sqlite3
import stat
import sys
import time
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from readonly_fixtures import damaged_sidecars, file_bytes

from puppetmaster import readonly
from puppetmaster.sqlite_store import SQLiteSwarmStore


def live_wal_writer(test, store):
    """A connection holding an open, uncheckpointed WAL for the whole test."""
    writer = sqlite3.connect(store.db_path, isolation_level=None)
    test.addCleanup(writer.close)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute(f'PRAGMA busy_timeout = {int(store.busy_timeout_ms)}')
    return writer


@unittest.skipIf(os.name == 'nt', 'POSIX WAL join')
class LiveWalObserverReadTests(unittest.TestCase):
    def test_reads_committed_rows_while_a_writer_holds_the_wal(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            writer = live_wal_writer(self, store)
            writer.execute("INSERT INTO metadata VALUES('wal_only','committed')")
            writer.execute('BEGIN IMMEDIATE')
            writer.execute('PRAGMA cache_size=1')
            writer.execute("INSERT INTO metadata VALUES('uncommitted',?)", ('x' * 1000000,))
            wal = Path(str(store.db_path) + '-wal')
            self.assertGreater(wal.stat().st_size, 32)
            main = file_bytes(store.db_path)
            # Only the WAL carries the committed row, so a successful read
            # proves the snapshot was joined, not taken from the main file.
            with closing(sqlite3.connect(store.db_path.as_uri() + '?immutable=1', uri=True)) as alone:
                self.assertIsNone(
                    alone.execute("SELECT value FROM metadata WHERE key='wal_only'").fetchone())
            with closing(readonly.connect(store)) as reader:
                self.assertEqual(reader.source_journal_mode, 'wal')
                self.assertEqual(
                    reader.execute("SELECT value FROM metadata WHERE key='wal_only'").fetchone()[0],
                    'committed')
                self.assertIsNone(
                    reader.execute("SELECT value FROM metadata WHERE key='uncommitted'").fetchone())
            writer.rollback()
            # The observer never writes the main database or the WAL itself.
            self.assertEqual(file_bytes(store.db_path), main)

    def test_a_writer_commits_while_an_observer_read_is_open(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            holder = live_wal_writer(self, store)
            holder.execute("INSERT INTO metadata VALUES('wal_only','committed')")
            with closing(readonly.connect(store)) as reader:
                self.assertEqual(
                    reader.execute("SELECT value FROM metadata WHERE key='wal_only'").fetchone()[0],
                    'committed')
                writer = sqlite3.connect(store.db_path, isolation_level=None, timeout=5)
                self.addCleanup(writer.close)
                writer.execute('PRAGMA busy_timeout = 5000')
                started = time.monotonic()
                writer.execute('BEGIN IMMEDIATE')
                writer.execute("INSERT INTO metadata VALUES('during_read','1')")
                writer.execute('COMMIT')
                # Uncontended: the observer holds no byte-range lock at all.
                self.assertLess(time.monotonic() - started, 2)
                # The reader's snapshot was pinned before that commit.
                self.assertIsNone(
                    reader.execute("SELECT value FROM metadata WHERE key='during_read'").fetchone())
            with closing(readonly.connect(store)) as fresh:
                self.assertEqual(
                    fresh.execute("SELECT value FROM metadata WHERE key='during_read'").fetchone()[0],
                    '1')

    def test_quiet_source_still_uses_the_immutable_descriptor_path(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            # Supervisor close checkpointed and removed the sidecars.
            self.assertFalse(Path(str(store.db_path) + '-wal').exists())
            before = {p.name: (file_bytes(p), p.stat().st_mtime_ns)
                      for p in store.root.iterdir() if p.is_file()}
            with closing(readonly.connect(store)) as reader:
                self.assertEqual(reader.execute('SELECT count(*) FROM jobs').fetchone()[0], 0)
            self.assertEqual({p.name for p in store.root.iterdir() if p.is_file()}, set(before))
            self.assertEqual({p.name: (file_bytes(p), p.stat().st_mtime_ns)
                              for p in store.root.iterdir() if p.is_file()}, before)

    def test_live_wal_without_shm_is_unavailable_and_creates_nothing(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.ensure_schema()
            writer = live_wal_writer(self, store)
            writer.execute("INSERT INTO metadata VALUES('wal_only','committed')")
            with damaged_sidecars(writer, store.db_path, ('-shm',)):
                present = sorted(p.name for p in store.root.iterdir())
                self.assertIn('state.sqlite3-wal', present)
                self.assertNotIn('state.sqlite3-shm', present)
                with self.assertRaises(readonly.ReadUnavailable):
                    readonly.connect(store, timeout=.2)
                self.assertEqual(sorted(p.name for p in store.root.iterdir()), present)


@unittest.skipIf(os.name == 'nt', 'POSIX WAL join')
class JoinableWalTests(unittest.TestCase):
    def cohort(self, tmp):
        from puppetmaster import readonly_worker as worker
        path = Path(tmp) / 'source.sqlite3'
        writer = sqlite3.connect(path, isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute('CREATE TABLE sample(value)')
        writer.execute("INSERT INTO sample VALUES('A')")
        return worker, path

    def test_only_a_complete_sound_cohort_is_joinable(self):
        with TemporaryDirectory() as tmp:
            worker, path = self.cohort(tmp)
            before = worker.stamps(path)
            self.assertTrue(worker.joinable_wal(path, before, b'\x02\x02'))
            # A delete-mode main header, a rollback journal, or either sidecar
            # missing is never joinable: a reader must not build one.
            self.assertFalse(worker.joinable_wal(path, before, b'\x01\x01'))
            self.assertFalse(worker.joinable_wal(path, before[:3] + [before[0]], b'\x02\x02'))
            for index in (1, 2):
                damaged = list(before)
                damaged[index] = None
                self.assertFalse(worker.joinable_wal(path, damaged, b'\x02\x02'))
            empty = list(before)
            empty[1] = before[1][:2] + (0,) + before[1][3:]
            self.assertFalse(worker.joinable_wal(path, empty, b'\x02\x02'))

    def test_a_damaged_wal_index_is_not_joinable(self):
        with TemporaryDirectory() as tmp:
            worker, path = self.cohort(tmp)
            before = worker.stamps(path)
            shm = Path(str(path) + '-shm')
            saved = file_bytes(shm)
            with open(shm, 'r+b') as index:
                index.write(b'\x00' * 100)
            try:
                self.assertIsNone(worker.wal_index_header(path))
                self.assertFalse(worker.joinable_wal(path, before, b'\x02\x02'))
            finally:
                with open(shm, 'r+b') as index:
                    index.write(saved[:100])


class WalAttestationTests(unittest.TestCase):
    def test_only_the_wal_cohort_widens_the_linux_descriptor_proof(self):
        from puppetmaster import readonly_worker as worker
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source'
            path.write_bytes(b'A')
            with path.open('rb') as source:
                st = os.fstat(source.fileno())
                main = (st.st_dev, st.st_ino, stat.S_IFREG)
                wal = (st.st_dev, st.st_ino + 1, stat.S_IFREG)
                shm = (st.st_dev, st.st_ino + 2, stat.S_IFREG)
                foreign = (st.st_dev, st.st_ino + 9, stat.S_IFREG)
                sidecars = (wal[:2], shm[:2])
                baseline = {source.fileno(): main}
                for opened, attested in (
                    ({101: main}, True),
                    ({101: main, 102: wal, 103: shm}, True),
                    ({101: wal, 102: shm}, False),  # Main never appeared.
                    ({101: main, 102: foreign}, False),
                ):
                    with self.subTest(opened=opened), \
                            patch.object(worker, 'linux_fd_snapshot', return_value={**baseline, **opened}):
                        if attested:
                            worker.attest_linux_database(baseline, source.fileno(), sidecars=sidecars)
                        else:
                            with self.assertRaisesRegex(OSError, 'attestation'):
                                worker.attest_linux_database(baseline, source.fileno(), sidecars=sidecars)
                # The immutable path still demands exactly one new descriptor.
                with patch.object(worker, 'linux_fd_snapshot',
                                  return_value={**baseline, 101: main, 102: wal}):
                    with self.assertRaisesRegex(OSError, 'attestation'):
                        worker.attest_linux_database(baseline, source.fileno())


if __name__ == '__main__':
    unittest.main()
