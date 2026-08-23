"""Hermetic proof of the celld paying slice: serial inbox, hibernate/alarm, sqlite file."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import contextlib
import io
import json
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.cell import (
    SQLITE_MAGIC,
    CellBusyError,
    CellRegistry,
    interned_poll,
    is_sqlite_file,
)
from puppetmaster.cli import main as cli_main
from puppetmaster.mcp_server import handle_message, tools


class CellInboxTests(unittest.TestCase):
    @contextlib.contextmanager
    def _cells(self):
        """Cell registry whose sqlite handles are closed before rmdir.

        Windows cannot unlink ``*.sqlite`` while a connection is open
        (WinError 32). Product ``CellRegistry.close()`` checkpoints WAL
        so TemporaryDirectory teardown succeeds on every platform.
        """
        with TemporaryDirectory() as tmp:
            registry = CellRegistry(tmp)
            try:
                yield registry
            finally:
                registry.close()

    def test_serial_inbox_two_enqueues_one_at_a_time(self) -> None:
        with self._cells() as registry:
            first = registry.enqueue("job-alpha", "ping", {"n": 1})
            second = registry.enqueue("job-alpha", "ping", {"n": 2})
            self.assertLess(first, second)
            status = registry.status("job-alpha")
            self.assertEqual(status["inbox_depth"], 2)
            seen: list[int] = []
            event_a = registry.process_one("job-alpha", handler=lambda ev: seen.append(ev["id"]))
            self.assertIsNotNone(event_a)
            self.assertEqual(event_a["id"], first)
            self.assertEqual(event_a["payload"]["n"], 1)
            mid = registry.status("job-alpha")
            self.assertEqual(mid["inbox_depth"], 1)
            self.assertFalse(mid["hibernating"])
            event_b = registry.process_one("job-alpha", handler=lambda ev: seen.append(ev["id"]))
            self.assertEqual(event_b["id"], second)
            self.assertEqual(seen, [first, second])
            done = registry.status("job-alpha")
            self.assertEqual(done["inbox_depth"], 0)
            self.assertEqual(event_a["id"] + 1, event_b["id"])

    def test_never_two_handlers_concurrently(self) -> None:
        with self._cells() as registry:
            registry.enqueue("job-mutex", "hold", {"n": 1})
            registry.enqueue("job-mutex", "next", {"n": 2})
            started = threading.Event()
            release = threading.Event()
            errors: list[BaseException] = []

            def slow(event):
                started.set()
                release.wait(timeout=2.0)

            def runner():
                try:
                    registry.process_one("job-mutex", owner="slow", handler=slow)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=runner)
            thread.start()
            self.assertTrue(started.wait(timeout=2.0))
            with self.assertRaises(CellBusyError):
                registry.process_one("job-mutex", owner="intruder")
            release.set()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            leftover = registry.status("job-mutex")
            self.assertEqual(leftover["inbox_depth"], 1)

    def test_hibernate_then_alarm_resume(self) -> None:
        with self._cells() as registry:
            registry.enqueue("job-sleep", "work", {"step": "once"})
            registry.process_one("job-sleep")
            hibernated = registry.status("job-sleep")
            self.assertTrue(hibernated["hibernating"])
            self.assertFalse(hibernated["live"])
            self.assertTrue(Path(hibernated["path"]).is_file())
            due = time.time() - 1
            registry.set_alarm("job-sleep", due)
            self.assertEqual(registry.status("job-sleep")["next_alarm"], f"{due:.6f}")
            woken = registry.tick()
            self.assertEqual(len(woken), 1)
            resumed = registry.status("job-sleep")
            self.assertFalse(resumed["hibernating"])
            self.assertTrue(resumed["live"])
            self.assertIsNone(resumed["next_alarm"])
            self.assertEqual(resumed["inbox_depth"], 1)
            alarm = registry.process_one("job-sleep")
            self.assertEqual(alarm["kind"], "alarm")
            registry.set_alarm("job-sleep", time.time() - 5)
            registry.hibernate("job-sleep")
            store = type("Store", (), {"root": registry.root})()
            polled = interned_poll(store)
            self.assertEqual(len(polled), 1)
            self.assertFalse(polled[0]["hibernating"])

    def test_inspectable_on_disk_sqlite(self) -> None:
        with self._cells() as registry:
            registry.enqueue("job-disk", "note", {"hello": "world"})
            status = registry.status("job-disk")
            path = Path(status["path"])
            self.assertTrue(path.is_file())
            self.assertTrue(is_sqlite_file(path))
            with path.open("rb") as handle:
                self.assertEqual(handle.read(16), SQLITE_MAGIC)
            # sqlite3.connect-as-context is a transaction manager: it does
            # not close the OS handle. Close explicitly so Windows can
            # later unlink the inspectable file (WinError 32 otherwise).
            connection = sqlite3.connect(str(path))
            try:
                kinds = [
                    row[0]
                    for row in connection.execute("SELECT kind FROM inbox ORDER BY id")
                ]
                self.assertEqual(kinds, ["note"])
                hibernating = connection.execute(
                    "SELECT value FROM cell_meta WHERE key = 'hibernating'"
                ).fetchone()[0]
                self.assertIn(hibernating, ("0", "1"))
            finally:
                connection.close()
            inspect = registry.inspect("job-disk")
            self.assertEqual(inspect["inbox_depth"], 1)
            self.assertEqual(inspect["inbox"][0]["kind"], "note")
            self.assertTrue(inspect["sqlite"])
            registry.close()
            path.unlink()
            self.assertFalse(path.exists())


class CellCliMcpTests(unittest.TestCase):
    def test_cli_cell_status_json(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = CellRegistry(tmp)
            try:
                registry.enqueue("job-cli", "ping", {"ok": True})
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli_main(["--state-dir", tmp, "cell-status", "job-cli", "--json"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["cell_id"], "job-cli")
                self.assertEqual(payload["inbox_depth"], 1)
                self.assertIn("hibernating", payload)
                self.assertIn("next_alarm", payload)
                self.assertTrue(Path(payload["path"]).is_file())
            finally:
                registry.close()

    def test_mcp_tool_is_additive(self) -> None:
        names = {tool.name for tool in tools()}
        self.assertIn("puppetmaster_cell_status", names)
        response = handle_message({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
        listed = {tool["name"] for tool in response["result"]["tools"]}
        self.assertIn("puppetmaster_cell_status", listed)
        self.assertIn("puppetmaster_status", listed)
        self.assertIn("puppetmaster_effort_index", listed)


if __name__ == "__main__":
    unittest.main()
