"""Wave 3 + Slice 10: durable SQLite PRAGMAs, quick_check, and safe backup."""

from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster.diagnostics import run_doctor
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteBackupError

class SqliteConnectionPragmaTests(unittest.TestCase):
    def test_fresh_connections_apply_durable_pragma_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()

            first = store.connect()
            try:
                self._assert_connection_policy(first, store)
            finally:
                first.close()

            # A second handle must re-apply connection-local PRAGMAs even though
            # journal_mode already persisted on the database file.
            second = store.connect()
            try:
                self._assert_connection_policy(second, store)
            finally:
                second.close()

    def test_schema_status_and_doctor_expose_effective_pragmas(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / ".puppetmaster"
            store = SQLiteSwarmStore(state_dir)
            store.init()

            status = store.schema_status()
            self.assertEqual(status["journal_mode"].lower(), "wal")
            self.assertEqual(status["busy_timeout"], str(store.busy_timeout_ms))
            self.assertEqual(status["foreign_keys"], "1")
            self.assertEqual(
                status["synchronous"], str(store._synchronous_pragma_value())
            )
            self.assertEqual(
                status["expected_busy_timeout"], str(store.busy_timeout_ms)
            )
            self.assertEqual(status["expected_foreign_keys"], "1")
            self.assertEqual(
                status["expected_synchronous"],
                str(store._synchronous_pragma_value()),
            )
            self.assertEqual(status["integrity_status"], "ok")
            self.assertEqual(status["quick_check"].lower(), "ok")
            self.assertEqual(status["check_kind"], "quick_check")

            checks = {
                check.name: check for check in run_doctor(root, state_dir)
            }
            sqlite_check = checks["sqlite-state"]
            self.assertEqual(sqlite_check.status, "ok")
            self.assertIn("journal=wal", sqlite_check.detail)
            self.assertIn(f"busy_timeout={store.busy_timeout_ms}", sqlite_check.detail)
            self.assertIn("foreign_keys=1", sqlite_check.detail)
            self.assertIn(
                f"synchronous={store._synchronous_pragma_value()}",
                sqlite_check.detail,
            )
            self.assertIn("quick_check=ok", sqlite_check.detail)

    def _assert_connection_policy(
        self, connection, store: SQLiteSwarmStore
    ) -> None:
        journal = connection.execute("PRAGMA journal_mode").fetchone()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        synchronous = connection.execute("PRAGMA synchronous").fetchone()

        self.assertEqual(str(journal[0]).lower(), "wal")
        self.assertEqual(int(busy_timeout[0]), store.busy_timeout_ms)
        self.assertEqual(int(foreign_keys[0]), 1)
        self.assertEqual(int(synchronous[0]), store._synchronous_pragma_value())

class SqliteIntegrityAndBackupTests(unittest.TestCase):
    def test_integrity_status_ok_on_clean_db(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            report = store.integrity_status()
            self.assertEqual(report["integrity_status"], "ok")
            self.assertEqual(report["quick_check"].lower(), "ok")
            self.assertEqual(report["check_kind"], "quick_check")

    def test_integrity_status_warns_on_corrupt_db(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            db_path = state_dir / "state.sqlite3"
            db_path.write_bytes(b"this is not a sqlite database at all")
            store = SQLiteSwarmStore(state_dir)
            report = store.integrity_status()
            self.assertIn(report["integrity_status"], {"warn", "unavailable"})
            self.assertNotEqual(report["quick_check"].lower(), "ok")

            checks = {
                check.name: check
                for check in run_doctor(Path(tmp), state_dir)
            }
            sqlite_check = checks["sqlite-state"]
            self.assertEqual(sqlite_check.status, "warn")
            self.assertIn("quick_check=", sqlite_check.detail)

    def test_integrity_status_unavailable_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            report = store.integrity_status()
            self.assertEqual(report["integrity_status"], "unavailable")
            self.assertEqual(report["quick_check"], "missing")

    def test_integrity_status_unavailable_when_locked(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            with mock.patch.object(
                store,
                "_connect_readonly",
                side_effect=sqlite3.OperationalError("database is locked"),
            ):
                report = store.integrity_status()
            self.assertEqual(report["integrity_status"], "unavailable")
            self.assertIn("locked", report["quick_check"].lower())

            locked = {
                "integrity_status": "unavailable",
                "quick_check": "locked: database is locked",
                "check_kind": "quick_check",
            }
            with mock.patch.object(
                SQLiteSwarmStore, "integrity_status", return_value=locked
            ):
                checks = {
                    check.name: check
                    for check in run_doctor(Path(tmp), store.root)
                }
            self.assertEqual(checks["sqlite-state"].status, "warn")
            self.assertIn("unavailable", checks["sqlite-state"].detail)

    def test_backup_succeeds_under_default_confine(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            store.create_job("backup me")
            dest = store.backup_to("state-backup.sqlite3")
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.parent, (store.root / "backups").resolve())
            # Round-trip: backup opens as a real SQLite DB with our schema.
            connection = sqlite3.connect(dest)
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(str(row[0]), str(store.schema_version))
            finally:
                connection.close()

    def test_backup_refuses_existing_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            first = store.backup_to("once.sqlite3")
            with self.assertRaises(SqliteBackupError) as ctx:
                store.backup_to(first)
            self.assertIn("already exists", str(ctx.exception).lower())

    def test_backup_refuses_path_outside_confine(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteSwarmStore(root / ".puppetmaster")
            store.init()
            outside = root / "escape.sqlite3"
            with self.assertRaises(SqliteBackupError) as ctx:
                store.backup_to(outside)
            self.assertIn("escapes confine", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
