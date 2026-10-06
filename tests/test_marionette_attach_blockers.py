from contextlib import closing
"""Marionette attach-proof shape and historical scope authority regressions."""
import hashlib
import json
import shutil
import sqlite3
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from readonly_fixtures import damaged_sidecars, file_bytes

from puppetmaster.identity import StoreIdentityError
from puppetmaster.projections import connection
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError
from puppetmaster.store_factory import create_store


def fingerprint(root):
    paths = [root] + sorted(root.rglob('*')) if root.exists() else []
    return {str(p.relative_to(root)): (
        hashlib.sha256(file_bytes(p)).hexdigest() if p.is_file() else None,
        p.stat().st_mode, p.stat().st_ino, p.stat().st_size,
        p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths}


def locked_fingerprint(root):
    import ast
    import subprocess
    script = "import sys; sys.path.insert(0, 'tests'); from test_marionette_attach_blockers import fingerprint; from pathlib import Path; print(repr(fingerprint(Path(sys.argv[1]))))"
    return ast.literal_eval(subprocess.check_output([sys.executable, '-c', script, str(root)], text=True))


class MarionetteBlockerTests(unittest.TestCase):
    def stores(self):
        for backend in ('sqlite', 'file'):
            with TemporaryDirectory() as tmp:
                store = create_store(backend, Path(tmp) / 'state', mode='ensure')
                job = store.create_job('owned', origin='host', project_id='repo', session_id='session')
                yield store, job

    def delete_mode(self, store):
        path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
        with closing(sqlite3.connect(path)) as c, c:
            self.assertEqual(c.execute('PRAGMA journal_mode=DELETE').fetchone()[0], 'delete')
        return path

    def test_rejected_attach_old_delete_database_preserves_proof(self):
        for version in ('1', '999'):
            with TemporaryDirectory() as tmp:
                root = Path(tmp) / 'state'
                root.mkdir(mode=0o755)
                db = root / 'state.sqlite3'
                with closing(sqlite3.connect(db)) as c, c:
                    c.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
                    c.execute('INSERT INTO metadata VALUES(\'schema_version\',?)', (version,))
                before = fingerprint(root)
                with self.assertRaises(SqliteSchemaError):
                    create_store('sqlite', root, mode='attach')
                self.assertEqual(fingerprint(root), before)
                probe = create_store('sqlite', root)
                self.assertEqual(probe.schema_status()['journal_mode'], 'delete')
                self.assertEqual(fingerprint(root), before)

    def test_unsupported_full_schema_attach_is_byte_invariant(self):
        for store, job in self.stores():
            if store.backend_name != 'sqlite':
                continue
            path = self.delete_mode(store)
            with closing(sqlite3.connect(path)) as c, c:
                c.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
            before = fingerprint(store.root)
            with self.assertRaises(SqliteSchemaError):
                create_store('sqlite', store.root, mode='attach')
            self.assertEqual(fingerprint(store.root), before)

    def test_unknown_prior_status_cannot_hide_from_filter(self):
        for store, job in self.stores():
            from puppetmaster.models import JobStatus
            checkpoint = store.list_job_summaries().revision
            store.save_job(replace(job, status=JobStatus.COMPLETE))
            with connection(store) as c:
                c.execute("UPDATE projection_changes SET previous_status=NULL,previous_membership=NULL WHERE revision>?",
                          (checkpoint,))
            page = store.read_job_summary_changes(after_revision=checkpoint, status='running', project_id='repo')
            self.assertEqual(page.outcome, 'unavailable')
            self.assertEqual(page.revision, checkpoint)
            self.assertFalse(page.items)

    def test_missing_and_recreated_roots_are_not_bootstrapped(self):
        for backend in ('sqlite', 'file'):
            with TemporaryDirectory() as tmp:
                root = Path(tmp) / 'missing'
                selected = create_store(backend, root)
                self.assertFalse(root.exists())
                self.assertEqual(selected.list_job_summaries().outcome, 'unavailable')
                with self.assertRaises((sqlite3.OperationalError, SqliteSchemaError)):
                    create_store(backend, root, mode='attach')
                self.assertFalse(root.exists())
                fresh = create_store(backend, root, mode='ensure')
                fresh.create_job('new')
                with self.assertRaises(StoreIdentityError):
                    selected.list_job_summaries()
                attached = create_store(backend, root, mode='attach')
                root.rename(root.parent / 'old')
                self.assertEqual(attached.list_job_summaries().outcome, 'unavailable')
                self.assertFalse(root.exists())
                create_store(backend, root, mode='ensure')
                with self.assertRaises(StoreIdentityError):
                    attached.list_job_summaries()

    def test_open_time_replacement_rejected(self):
        for store, job in self.stores():
            selected = create_store(store.backend_name, store.root)
            from puppetmaster.readonly import ReaderProcess
            original = ReaderProcess
            def swapped(*args, **kwargs):
                store.root.rename(store.root.parent / 'old')
                shutil.copytree(store.root.parent / 'old', store.root)
                return original(*args, **kwargs)
            with patch('puppetmaster.readonly.ReaderProcess', side_effect=swapped):
                with self.assertRaises(StoreIdentityError):
                    selected.list_job_summaries()

    def test_delete_mode_attach_and_metadata_preserve_bytes_stat_and_mode(self):
        for store, job in self.stores():
            path = self.delete_mode(store)
            store.root.chmod(0o755)
            before = fingerprint(store.root)
            attached = create_store(store.backend_name, store.root, mode='attach')
            self.assertEqual(len(attached.list_job_summaries().items), 1)
            self.assertEqual(len(attached.read_job_summary_changes().items), 1)
            self.assertEqual(fingerprint(store.root), before)
            with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as c, c:
                self.assertEqual(c.execute('PRAGMA journal_mode').fetchone()[0], 'delete')

    def test_v5_read_attachment_usable_without_identity_bootstrap(self):
        for store, job in self.stores():
            if store.backend_name != 'sqlite':
                continue
            path = self.delete_mode(store)
            with closing(sqlite3.connect(path)) as c, c:
                c.execute("UPDATE metadata SET value='5' WHERE key='schema_version'")
                c.execute("DELETE FROM metadata WHERE key='incarnation'")
            before = fingerprint(store.root)
            attached = create_store('sqlite', store.root, mode='attach')
            self.assertEqual(attached.get_job(job.id), job)
            self.assertEqual(fingerprint(store.root), before)

    def test_previous_authority_survives_unseen_loss_and_later_writes(self):
        for store, job in self.stores():
            checkpoint = store.list_job_summaries().revision
            store.save_job(replace(job, origin='foreign', project_id='other', session_id=None))
            store.create_job('unrelated')
            for filters in ({}, {'project_id': 'repo'}, {'origin': 'host'}):
                page = store.read_job_summary_changes(after_revision=checkpoint, **filters)
                item = next(i for i in page.items if i.id == job.id)
                self.assertEqual(item.previous_membership, 'present')
                self.assertEqual((item.previous_origin, item.previous_project_id, item.previous_session_id),
                                 ('host', 'repo', 'session'))
                if filters:
                    self.assertTrue(item.deleted)
                self.assertEqual(item.origin, 'foreign')

    def test_unprovable_loss_preserves_cursor_and_revision(self):
        for store, job in self.stores():
            checkpoint = store.list_job_summaries().revision
            store.save_job(replace(job, project_id='foreign'))
            for damage in ("previous_membership=NULL", "previous_scope=NULL", "previous_scope='[]'",
                           "previous_scope='" + 'x' * 10000 + "'"):
                with connection(store) as c:
                    c.execute("UPDATE projection_changes SET previous_membership='present',previous_scope=? WHERE revision>?",
                              (json.dumps(dict(origin='host',project_id='repo',session_id='session')), checkpoint))
                    c.execute('UPDATE projection_changes SET ' + damage + ' WHERE revision>?', (checkpoint,))
                page = store.read_job_summary_changes(after_revision=checkpoint, project_id='repo')
                self.assertEqual(page.outcome, 'unavailable')
                self.assertEqual(page.revision, checkpoint)
                self.assertIsNone(page.next_cursor)
                self.assertFalse(page.items)

    def test_retention_without_cursor_requires_refresh(self):
        for store, job in self.stores():
            checkpoint = store.list_job_summaries().revision
            store.save_job(replace(job, project_id='foreign'))
            with connection(store) as c:
                c.execute('DELETE FROM projection_changes WHERE revision>?', (checkpoint,))
            store.create_job('later')
            page = store.read_job_summary_changes(after_revision=checkpoint, project_id='repo')
            self.assertEqual(page.outcome, 'unavailable')
            self.assertEqual(page.reason, 'change_history_unavailable')
            self.assertEqual(page.revision, checkpoint)
            self.assertIsNone(page.next_cursor)

    def test_snapshot_delete_and_later_write_keeps_original_membership(self):
        for store, job in self.stores():
            for _ in range(4):
                store.create_job('owned', project_id='repo')
            expected = store.list_job_summaries(project_id='repo').items
            page = store.list_job_summaries(project_id='repo', limit=1)
            actual = list(page.items)
            for item in expected:
                store.delete_job(item.id)
            while page.next_cursor:
                store.create_job('later', project_id='repo')
                page = store.list_job_summaries(project_id='repo', limit=1, cursor=page.next_cursor)
                self.assertLessEqual(page.scanned, 2)
                actual.extend(page.items)
            self.assertEqual(tuple(actual), expected)

    def test_unprovable_continuation_does_not_advance(self):
        for store, job in self.stores():
            checkpoint = store.list_job_summaries().revision
            for _ in range(3):
                store.save_job(replace(job, project_id='foreign'))
            first = store.read_job_summary_changes(after_revision=checkpoint, limit=1)
            self.assertIsNotNone(first.next_cursor)
            with connection(store) as c:
                c.execute("UPDATE projection_changes SET previous_membership=NULL WHERE revision>?",
                          (first.items[0].revision,))
            page = store.read_job_summary_changes(after_revision=checkpoint, cursor=first.next_cursor, limit=1)
            self.assertEqual(page.outcome, 'unavailable')
            self.assertEqual(page.revision, checkpoint)
            self.assertEqual(page.next_cursor, first.next_cursor)
            self.assertFalse(page.items)

    def test_file_rejected_attach_preserves_existing_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.chmod(0o755)
            before = fingerprint(root)
            with self.assertRaises(sqlite3.OperationalError):
                create_store('file', root, mode='attach')
            self.assertEqual(fingerprint(root), before)

    def test_checkpointed_wal_attach_does_not_create_sidecars(self):
        for store, job in self.stores():
            if store.backend_name != 'sqlite':
                continue
            # Supervisor close has checkpointed the WAL. No sidecars remain.
            self.assertFalse((store.root / 'state.sqlite3-wal').exists())
            before = fingerprint(store.root)
            create_store('sqlite', store.root, mode='attach')
            self.assertEqual(fingerprint(store.root), before)

    def test_live_wal_joins_then_checkpointed_commits_preserve_source(self):
        from contextlib import closing
        for store, job in self.stores():
            path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
            with closing(sqlite3.connect(path)) as writer:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute('PRAGMA wal_autocheckpoint=0')
                table = 'metadata' if store.backend_name == 'sqlite' else 'projection_meta'
                writer.execute(f"INSERT INTO {table} VALUES('wal_only','committed')")
                writer.commit()
                self.assertGreater(Path(str(path) + '-wal').stat().st_size, 32)
                # The main file alone cannot see this committed row.
                with closing(sqlite3.connect(path.as_uri() + '?immutable=1', uri=True)) as main:
                    self.assertIsNone(main.execute(f"SELECT value FROM {table} WHERE key='wal_only'").fetchone())
                committed_wal_size = Path(str(path) + '-wal').stat().st_size
                writer.execute('PRAGMA cache_size=1')
                writer.execute(f"INSERT INTO {table} VALUES('uncommitted',?)", ('x' * 1000000,))
                self.assertGreater(Path(str(path) + '-wal').stat().st_size, committed_wal_size)
                before = locked_fingerprint(store.root)
                from puppetmaster.readonly import connect
                # This fixture never releases its writer. Attach and ordinary
                # observer reads both join its committed WAL snapshot instead
                # of waiting for a checkpoint; the uncommitted row stays
                # invisible and writers keep the source to themselves.
                with patch.object(SQLiteSwarmStore, 'busy_timeout_ms', 100):
                    attached = create_store(store.backend_name, store.root, mode='attach')
                    if store.backend_name == 'sqlite':
                        self.assertEqual(attached._incarnation, store._incarnation)
                    else:
                        self.assertEqual(attached.incarnation, store.incarnation)
                    page = store.list_job_summaries()
                self.assertEqual(page.outcome, 'complete')
                self.assertEqual([item.id for item in page.items], [job.id])
                # A WAL reader records read marks in the -shm index. Database
                # content -- the main file and the WAL -- stays byte-identical,
                # and no file is created, repaired, or removed.
                current = locked_fingerprint(store.root)
                self.assertEqual(set(current), set(before))
                content = [name for name in before if not name.endswith('-shm')]
                self.assertEqual({name: current[name] for name in content},
                                 {name: before[name] for name in content})
                writer.rollback()
            # After the writer closes and checkpoints, only committed facts are
            # readable. No full-store snapshot or sidecars are created.
            before = fingerprint(store.root)
            attached = create_store(store.backend_name, store.root, mode='attach')
            with closing(connect(attached)) as reader:
                self.assertEqual(reader.execute(f"SELECT value FROM {table} WHERE key='wal_only'").fetchone()[0], 'committed')
                self.assertIsNone(reader.execute(f"SELECT value FROM {table} WHERE key='uncommitted'").fetchone())
                process = reader.process
            self.assertIsNotNone(process.poll())
            self.assertEqual(len(attached.list_job_summaries().items), 1)
            self.assertEqual(fingerprint(store.root), before)

    def test_live_wal_missing_sidecar_is_unavailable_without_cursor_advance(self):
        from puppetmaster.readonly import ReadUnavailable
        from contextlib import closing
        for missing in (('-wal',), ('-shm',), ('-wal', '-shm')):
            for store, job in self.stores():
                store.create_job('second')
                first = store.list_job_summaries(limit=1)
                changes = store.read_job_summary_changes(limit=1)
                self.assertIsNotNone(first.next_cursor)
                self.assertIsNotNone(changes.next_cursor)
                path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
                with closing(sqlite3.connect(path)) as writer:
                    writer.execute('PRAGMA journal_mode=WAL')
                    writer.execute('BEGIN IMMEDIATE')
                    with damaged_sidecars(writer, path, missing):
                        # Damage is test setup only; the reader must not repair it.
                        before = locked_fingerprint(store.root)
                        with patch.object(SQLiteSwarmStore, 'busy_timeout_ms', 100), \
                                self.assertRaises((ReadUnavailable, SqliteSchemaError)):
                            create_store(store.backend_name, store.root, mode='attach')
                        page = store.list_job_summaries(limit=1, cursor=first.next_cursor)
                        self.assertEqual(page.outcome, 'unavailable')
                        self.assertEqual(page.next_cursor, first.next_cursor)
                        self.assertEqual(page.retry_after_ms, 100)
                        page = store.read_job_summary_changes(after_revision=0, limit=1, cursor=changes.next_cursor)
                        self.assertEqual(page.outcome, 'unavailable')
                        self.assertEqual(page.revision, 0)
                        self.assertEqual(page.next_cursor, changes.next_cursor)
                        self.assertFalse(page.items)
                        self.assertEqual(locked_fingerprint(store.root), before)

    def test_wal_disappears_during_reader_open_is_unavailable(self):
        from contextlib import closing
        for store, job in self.stores():
            path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
            with closing(sqlite3.connect(path)) as writer:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute('BEGIN IMMEDIATE')
                writer.commit()
                from puppetmaster.readonly import ReaderProcess
                damage = damaged_sidecars(writer, path, ('-shm',))
                entered = []
                def changing(*args, **kwargs):
                    if not entered:
                        damage.__enter__()
                        entered.append(True)
                    return ReaderProcess(*args, **kwargs)
                try:
                    with patch('puppetmaster.readonly.ReaderProcess', side_effect=changing):
                        page = store.read_job_summary_changes(after_revision=7)
                finally:
                    if entered:
                        damage.__exit__(None, None, None)
                self.assertEqual(page.outcome, 'unavailable')
                self.assertEqual(page.revision, 7)
                self.assertEqual(page.reason, 'read_snapshot_unavailable')
                self.assertEqual(page.retry_after_ms, 100)
                self.assertFalse(page.items)
