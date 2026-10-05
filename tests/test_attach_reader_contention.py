"""Attach contention must not turn lock retries into interpreter startup storms."""
import json
import sqlite3
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401

from puppetmaster import readonly
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.identity import StoreIdentityError


class NoAdmission:
    """Exercise the helper BUSY protocol independently of production admission."""
    def __init__(self, *args, **kwargs):
        pass

    def release(self):
        pass


class AttachReaderContentionTests(unittest.TestCase):
    def test_concurrent_attaches_do_not_wait_for_a_quiet_database(self):
        # Attach used the readonly helper, which needs zero live connections
        # and a checkpointed WAL. 32 attachers behind live writers starved
        # there. Attach now joins the WAL like the writer it is about to be.
        with TemporaryDirectory() as root:
            supervisor = SQLiteSwarmStore(root)
            supervisor.ensure_schema()
            count = 32
            barrier = threading.Barrier(count)
            def attach():
                store = SQLiteSwarmStore(root)
                barrier.wait(timeout=10)
                store.attach()
                return store._incarnation
            # A live writer with uncheckpointed WAL frames: the helper path
            # could not attach until it closed and checkpointed.
            writer = supervisor.connect()
            job = supervisor.create_job("live writer")
            writer.execute("SELECT 1 FROM metadata").fetchall()
            started = time.monotonic()
            try:
                with patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        ThreadPoolExecutor(max_workers=count) as pool:
                    futures = [pool.submit(attach) for _ in range(count)]
                    self.assertEqual([f.result(timeout=15) for f in futures],
                                     [supervisor._incarnation] * count)
                    self.assertEqual(spawn.call_count, 0)
            finally:
                writer.close()
            self.assertLess(time.monotonic() - started, 10)
            # The last close may checkpoint WAL frames into the file; content
            # is what attach must not change.
            self.assertEqual([j.id for j in SQLiteSwarmStore(root).list_jobs()], [job.id])

    @patch.object(readonly, 'ReaderAdmission', NoAdmission)
    def test_replacement_between_lock_retries_is_rejected(self):
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(Path(root) / 'state')
            store.ensure_schema()
            receive = readonly.ReadConnection._receive
            replaced = []
            def observed(connection):
                try:
                    return receive(connection)
                except sqlite3.OperationalError:
                    if not replaced:
                        held.close()
                        # Wait for the failed helper to release its source fd.
                        connection.process.stdin.write(json.dumps({'open': str(store.db_path)}) + '\n')
                        connection.process.stdin.flush()
                        receive(connection)
                        connection._control('release', True)
                        store.root.rename(Path(root) / 'old')
                        SQLiteSwarmStore(store.root).ensure_schema()
                        replaced.append(True)
                    raise
            with readonly.connect(store) as held, \
                    patch.object(readonly.ReadConnection, '_receive', observed), \
                    patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn:
                with self.assertRaises(StoreIdentityError):
                    readonly.connect(store, attach_binding=True)
                self.assertEqual(spawn.call_count, 1)
            self.assertTrue(replaced)

    @patch.object(readonly, 'ReaderAdmission', NoAdmission)
    def test_persistent_lock_is_bounded_and_helper_reaped(self):
        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            with readonly.connect(store):
                processes = []
                spawn = readonly._Transport
                def tracked(*args, **kwargs):
                    process = spawn(*args, **kwargs)
                    processes.append(process.process)
                    return process
                started = time.monotonic()
                with patch.object(readonly, '_Transport', tracked):
                    with self.assertRaisesRegex(sqlite3.OperationalError, 'database is locked|active reader|reader timed out'):
                        readonly.connect(store, timeout=.3, attach_binding=True)
                self.assertLess(time.monotonic() - started, 2)
                self.assertEqual(len(processes), 1)
                readonly._cleanup.maintain(limit=len(readonly._cleanup.owners))
                self.assertTrue(all(p.poll() is not None for p in processes))

    def test_replacement_while_queued_is_rejected(self):
        for aba in (False, True, 'directory'):
            with self.subTest(aba=aba), TemporaryDirectory() as root:
                store = SQLiteSwarmStore(Path(root) / 'state')
                store.ensure_schema()
                queued = threading.Event()
                proceed = threading.Event()
                admission = readonly.ReaderAdmission
                def observed(*args, **kwargs):
                    queued.set()
                    self.assertTrue(proceed.wait(5))
                    return admission(*args, **kwargs)
                with readonly.connect(store) as held, \
                        patch.object(readonly, 'ReaderAdmission', observed), \
                        patch.object(readonly, '_Transport', wraps=readonly._Transport) as spawn, \
                        ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(readonly.connect, store, attach_binding=True)
                    try:
                        self.assertTrue(queued.wait(5))
                        held.close()
                        if aba == 'directory':
                            moved = Path(root) / 'moved'
                            store.root.rename(moved)
                            moved.rename(store.root)
                        elif aba:
                            moved = store.root / 'moved'
                            store.db_path.rename(moved)
                            moved.rename(store.db_path)
                        else:
                            store.root.rename(Path(root) / 'old')
                            SQLiteSwarmStore(store.root).ensure_schema()
                    finally:
                        proceed.set()
                    with self.assertRaises(StoreIdentityError):
                        future.result(timeout=5)
                    self.assertEqual(spawn.call_count, 1)


class HelperSpawnGateTests(unittest.TestCase):
    def test_concurrent_helper_births_do_not_overlap(self):
        import subprocess

        with TemporaryDirectory() as root:
            store = SQLiteSwarmStore(root)
            store.ensure_schema()
            active = 0
            peak = 0
            mutex = threading.Lock()

            class SlowPopen(subprocess.Popen):
                def __init__(self, *args, **kwargs):
                    nonlocal active, peak
                    with mutex:
                        active += 1
                        peak = max(peak, active)
                    try:
                        time.sleep(0.05)
                        super().__init__(*args, **kwargs)
                    finally:
                        with mutex:
                            active -= 1

            transports = []

            def launch():
                transports.append(readonly._Transport(store.db_path))

            with patch.object(readonly, 'ReaderProcess', SlowPopen), \
                    ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    futures = [pool.submit(launch) for _ in range(2)]
                    for future in futures:
                        future.result(timeout=10)
                    self.assertEqual(peak, 1, peak)
                finally:
                    for transport in transports:
                        transport.close()


if __name__ == '__main__':
    unittest.main()


class FileBackendAttachBudgetTests(unittest.TestCase):
    def test_file_worker_attach_outlasts_an_ordinary_read_budget(self):
        # An open transaction in another process blocks readonly binding until it
        # closes. SQLite-backend attach waits out its full attach budget; the
        # file backend used the 5s ordinary read and failed worker startup
        # ("active reader; sidecars may be missing") under supervisor churn.
        import subprocess
        from puppetmaster.store import SwarmStore
        from puppetmaster.store_factory import create_worker_store
        with TemporaryDirectory() as root:
            supervisor = SwarmStore(root)
            supervisor.init()
            hold = 7
            holder = subprocess.Popen(
                [sys.executable, '-c',
                 'import sqlite3, sys, time\n'
                 'c = sqlite3.connect(sys.argv[1], isolation_level=None)\n'
                 'c.execute("BEGIN")\n'
                 'c.execute("SELECT count(*) FROM sqlite_master").fetchone()\n'
                 'print("held", flush=True)\n'
                 'time.sleep(float(sys.argv[2]))\n',
                 str(Path(root) / 'metadata.sqlite3'), str(hold)],
                stdout=subprocess.PIPE, text=True)
            self.addCleanup(holder.wait, 30)
            self.addCleanup(holder.kill)
            self.assertEqual(holder.stdout.readline().strip(), 'held')
            started = time.monotonic()
            worker = create_worker_store('file', root)
            self.assertGreater(time.monotonic() - started, hold - 2)
            self.assertEqual(worker.incarnation, supervisor.incarnation)
