from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from puppetmaster import readonly_worker


class DescriptorOpenRetryTests(unittest.TestCase):
    """macOS SQLite now and then fails to open /dev/fd/N under load; the same open
    then succeeds at once. 1 in ~1600 reads at 16-way load surfaced as a bare
    'unavailable' page, which Marionette refused as a changed view."""

    def _flaky(self, failures, message='unable to open database file'):
        real = sqlite3.connect
        calls = []

        def connect(*args, **kwargs):
            calls.append(args)
            if len(calls) <= failures:
                raise sqlite3.OperationalError(message)
            return real(*args, **kwargs)
        return connect, calls

    def test_a_spurious_cantopen_is_retried(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / 'metadata.sqlite3'
            sqlite3.connect(db).close()
            connect, calls = self._flaky(2)
            with patch.object(readonly_worker.sqlite3, 'connect', connect):
                readonly_worker.connect_descriptor(db.as_uri() + '?mode=ro').close()
            self.assertEqual(len(calls), 3)

    def test_a_persistent_or_other_error_still_raises(self):
        connect, calls = self._flaky(99)
        with patch.object(readonly_worker.sqlite3, 'connect', connect):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'unable to open database file'):
                readonly_worker.connect_descriptor('file:///nowhere?mode=ro')
        self.assertEqual(len(calls), 5)
        connect, calls = self._flaky(1, 'disk I/O error')
        with patch.object(readonly_worker.sqlite3, 'connect', connect):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'disk I/O error'):
                readonly_worker.connect_descriptor('file:///nowhere?mode=ro')
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
