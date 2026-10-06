"""Weak discovery tolerates unavailable global stores but protects its target."""
import os
import json
import sqlite3
import sys
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from readonly_fixtures import damaged_sidecars

from puppetmaster import readonly, state
from puppetmaster.adapters._prompts import with_prewalk_plan
from puppetmaster.identity import StoreIdentityError
from puppetmaster.models import Task
from puppetmaster.sqlite_store import SQLiteSwarmStore


class OwnershipDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.owner = SQLiteSwarmStore(self.root / 'owner')
        self.job = self.owner.create_job('target')
        self.other = SQLiteSwarmStore(self.root / 'other')
        self.other.create_job('unrelated')
        registry = patch.object(state, 'list_project_state_dirs', return_value=[self.other.root, self.owner.root])
        registry.start()
        self.addCleanup(registry.stop)
        state._ownership_cache.clear()

    def resolve(self, **kwargs):
        return state.resolve_job_state(job_id=self.job.id, default_dir=self.owner.root, **kwargs)

    def test_unrelated_live_wal_is_not_an_owner(self):
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute("INSERT INTO jobs VALUES ('live', '{}')")
            writer.commit()
            self.assertEqual(self.resolve(), self.owner.root)
            self.assertEqual(state.find_state_dir_for_job(self.job.id), self.owner.root)

    def test_unrelated_source_change_does_not_abort_discovery(self):
        connect = readonly.connect
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            def changing(store, **kwargs):
                if store.root == self.other.root:
                    writer.execute("INSERT OR REPLACE INTO jobs VALUES ('changing', '{}')")
                    writer.commit()
                    raise readonly.ReadUnavailable('unable to open database: source changed')
                return connect(store, **kwargs)
            with patch.object(readonly, 'connect', side_effect=changing):
                self.assertEqual(self.resolve(), self.owner.root)

    def test_requested_live_wal_is_read_even_without_job_directory(self):
        # Observer reads join a live WAL's committed snapshot, so a store a job
        # is writing to is found by its rows, not refused as unavailable.
        with closing(sqlite3.connect(self.owner.db_path)) as writer:
            writer.execute("UPDATE jobs SET data=data")
            writer.commit()
            # Force a WAL frame even on SQLite builds optimizing no-op updates.
            writer.execute("INSERT INTO jobs VALUES ('another', '{}')")
            writer.commit()
            self.owner.job_dir(self.job.id).rename(self.root / 'hidden-job')
            self.assertEqual(self.resolve(), self.owner.root)
            self.assertEqual(state.find_state_dir_for_job(self.job.id), self.owner.root)

    def test_live_global_duplicate_is_ambiguous_in_weak_mode(self):
        # The live duplicate is readable now, so weak mode sees two owners and
        # refuses to guess instead of ignoring the unreadable one.
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute('INSERT INTO jobs VALUES (?, ?)', (self.job.id, '{}'))
            writer.commit()
            with self.assertRaisesRegex(ValueError, "ambiguous job_id"):
                self.resolve()

    def test_unknown_candidate_cannot_be_ignored(self):
        self.other.db_path.write_bytes(b'not a database')
        with self.assertRaises(sqlite3.DatabaseError):
            self.resolve()

    def test_strict_ref_filters_unrelated_and_checks_incarnation(self):
        ref = self.owner.job_ref(self.job.id).as_dict()
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute('INSERT INTO jobs VALUES (?, ?)', (self.job.id, '{}'))
            writer.commit()
            self.assertEqual(self.resolve(job_ref=ref), self.owner.root)
        ref['incarnation'] = '00000000-0000-0000-0000-000000000001'
        with self.assertRaises(StoreIdentityError):
            self.resolve(job_ref=ref)

    def test_strict_ref_and_explicit_store_read_a_live_target(self):
        ref = self.owner.job_ref(self.job.id).as_dict()
        with closing(sqlite3.connect(self.owner.db_path)) as writer:
            writer.execute("INSERT INTO jobs VALUES ('live', '{}')")
            writer.commit()
            for kwargs in ({'job_ref': ref}, {'state_dir': self.owner.root}):
                with self.subTest(kwargs=kwargs):
                    self.assertEqual(self.resolve(**kwargs), self.owner.root)

    def test_inline_plan_never_needs_global_discovery(self):
        task = Task(job_id=self.job.id, role='implement', instruction='work', payload={
            'prewalk': True, 'prewalk_artifacts': [
                {'type': 'decision', 'payload': {'decision': 'Use the supplied plan', 'why': 'known'}}]})
        with patch.dict(os.environ, {}, clear=True), patch.object(
                state, 'find_state_dir_for_job', side_effect=readonly.ReadUnavailable('source changed')) as discover:
            rendered = with_prewalk_plan('Do the task', task)
        self.assertIn('Use the supplied plan', rendered)
        discover.assert_not_called()

    def test_missing_global_wal_is_skipped_without_recreating_sidecars(self):
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute('INSERT INTO jobs VALUES (?, ?)', (self.job.id, '{}'))
            writer.commit()
            sidecars = [Path(str(self.other.db_path) + suffix) for suffix in ('-wal', '-shm')]
            with damaged_sidecars(writer, self.other.db_path):
                self.assertEqual(self.resolve(), self.owner.root)
                self.assertTrue(all(not p.exists() for p in sidecars))

    def test_changed_requested_source_is_not_rescued_by_membership(self):
        connect = readonly.connect
        changed = []
        def changing(store, **kwargs):
            if store.root == self.owner.root and not changed:
                changed.append(True)
                self.owner.create_job('concurrent commit')
                raise readonly.ReadUnavailable('unable to open database: source changed')
            return connect(store, **kwargs)
        with patch.object(readonly, 'connect', side_effect=changing):
            with self.assertRaisesRegex(readonly.ReadUnavailable, 'source changed'):
                self.resolve()

    def test_inline_artifacts_keep_explicit_store_for_freshness(self):
        from puppetmaster.validation import refresh_cited_freshness
        task = Task(job_id=self.job.id, role='implement', instruction='work', payload={
            'prewalk': True, 'cwd': str(self.root), 'prewalk_artifacts': [
                {'type': 'decision', 'payload': {'decision': 'Keep bound context', 'why': 'known'}}]})
        with patch.dict(os.environ, {state.STATE_DIR_ENV: str(self.owner.root)}), patch.object(
                state, 'find_state_dir_for_job', side_effect=AssertionError('global discovery')), patch(
                'puppetmaster.validation.refresh_cited_freshness', wraps=refresh_cited_freshness) as refresh:
            rendered = with_prewalk_plan('Do the task', task)
        self.assertIn('Keep bound context', rendered)
        self.assertTrue(refresh.called)
        self.assertEqual(refresh.call_args.kwargs['store'].root, self.owner.root)

    def test_unrelated_continuous_commits_allow_discovery(self):
        ready = threading.Event()
        stop = threading.Event()
        errors = []
        def write():
            try:
                with closing(sqlite3.connect(self.other.db_path)) as writer:
                    while not stop.is_set():
                        writer.execute("INSERT OR REPLACE INTO jobs VALUES ('changing', '{}')")
                        writer.commit()
                        ready.set()
                        stop.wait(.002)
            except Exception as exc:
                errors.append(exc)
                ready.set()
        thread = threading.Thread(target=write)
        thread.start()
        try:
            self.assertTrue(ready.wait(5))
            self.assertEqual(self.resolve(), self.owner.root)
        finally:
            stop.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_discovery_preserves_wal_and_shared_memory(self):
        for store in (self.other, self.owner):
            with self.subTest(store=store.root), closing(sqlite3.connect(store.db_path)) as writer:
                writer.execute("INSERT INTO jobs VALUES ('live', '{}')")
                writer.commit()
                def snapshot():
                    # A live-WAL reader may record read marks in -shm; the
                    # database and its WAL must not change.
                    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                            for p in store.root.glob('state.sqlite3*') if not p.name.endswith('-shm')}
                before = snapshot()
                self.assertEqual(self.resolve(), self.owner.root)
                self.assertEqual(snapshot(), before)

    def test_readable_duplicate_owners_remain_ambiguous(self):
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute('INSERT INTO jobs VALUES (?, ?)', (self.job.id, '{}'))
            writer.commit()
        for resolve in (self.resolve, lambda: state.find_state_dir_for_job(self.job.id)):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                resolve()

    def test_unavailable_caller_target_does_not_select_second_owner(self):
        with closing(sqlite3.connect(self.other.db_path)) as writer:
            writer.execute('INSERT INTO jobs VALUES (?, ?)', (self.job.id, '{}'))
            writer.commit()
        with closing(sqlite3.connect(self.owner.db_path)) as writer:
            writer.execute("INSERT INTO jobs VALUES ('live', '{}')")
            writer.commit()
            # A live caller target is readable now; with the job in a second
            # store too, resolution refuses to pick one.
            with self.assertRaisesRegex(ValueError, "ambiguous job_id"):
                self.resolve()

    def test_dashboard_fallback_with_continuously_changing_global_store(self):
        from puppetmaster import dashboard as dash, mcp_server as mcp
        ready = threading.Event()
        stop = threading.Event()
        errors = []
        commits = []
        def write():
            try:
                with closing(sqlite3.connect(self.other.db_path)) as writer:
                    while not stop.is_set():
                        writer.execute("INSERT OR REPLACE INTO jobs VALUES ('changing', ?)",
                                       (str(len(commits)),))
                        writer.commit()
                        commits.append(True)
                        ready.set()
                        stop.wait(.001)
            except Exception as exc:
                errors.append(exc)
                ready.set()
        thread = threading.Thread(target=write)
        thread.start()
        try:
            self.assertTrue(ready.wait(5))
            identity = {"pid": 4242, "state_dir_id": "abc", "service": "puppetmaster-dashboard"}
            with patch.object(dash, 'dashboard_serves', return_value=True), patch.object(
                    dash, 'read_dashboard_runfile', return_value=None), patch.object(
                    dash, 'dashboard_identity', return_value=identity), patch.object(
                    dash, 'write_dashboard_runfile') as runfile, patch.object(
                    mcp, '_spawn_dashboard_server') as spawn:
                for iteration in range(100):
                    with self.subTest(iteration=iteration):
                        result = mcp.call_tool('puppetmaster_dashboard',
                                               {'cwd': '/tmp', 'job_id': 'job_abc'})
                        body = json.loads(result['content'][0]['text'])
                        self.assertTrue(body['already_running'])
                        self.assertFalse(body['started'])
                        self.assertEqual(body['url'], 'http://127.0.0.1:8787/?job=job_abc')
                self.assertEqual(runfile.call_count, 100)
                self.assertEqual(Path(runfile.call_args.args[0]), state.resolve_state_dir(cwd=Path('/tmp')))
                spawn.assert_not_called()
        finally:
            stop.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertGreater(len(commits), 1)

    def test_sqlite_contention_skips_only_opportunistic_candidates(self):
        connect = readonly.connect
        for code in (None, 5, 6, 261, 262):
            for root in (self.owner.root, self.other.root):
                with self.subTest(code=code, root=root):
                    state._ownership_cache.clear()
                    error = sqlite3.OperationalError('database is locked' if code is None else 'busy')
                    if code is not None:
                        error.sqlite_errorcode = code
                    def locked(store, **kwargs):
                        if store.root == root:
                            raise error
                        return connect(store, **kwargs)
                    with patch.object(readonly, 'connect', side_effect=locked):
                        if root == self.owner.root:
                            with self.assertRaises(sqlite3.OperationalError):
                                self.resolve()
                        else:
                            self.assertEqual(self.resolve(), self.owner.root)

    def test_other_operational_errors_are_not_silently_skipped(self):
        with patch.object(readonly, 'connect', side_effect=sqlite3.OperationalError('no such table: jobs')):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'no such table'):
                self.resolve()
