"""`--all-projects` aggregation must report unreadable projects, not erase them.

`list_all_projects_snapshot` walks every project state dir on the machine. Its
per-project body used to be wrapped in a bare ``except Exception: continue``, so
a single locked/corrupt ``state.sqlite3`` — or one job row a newer schema wrote
and this build cannot decode — removed that project's *entire* job list from the
board with no trace anywhere: no stderr line, no wire field, no log. The board
looked complete while showing less than everything. The defect is the silence.

These tests pin the fix from three angles:

* **reported, not dropped** — a healthy project's rows still come back while the
  broken one shows up in ``snapshot.errors`` with its slug and exception type.
  The corruption here is *real* (garbage bytes over the sqlite file, producing a
  genuine ``sqlite3.DatabaseError``), never a patched-in fake, because a mocked
  error proves the ``except`` clause is spelled right and nothing more.
* **the wire contract is unchanged** — ``/api/jobs`` is a bare JSON array and the
  page does ``for (const j of jobs)`` over the top level, so the failures ride on
  a ``list`` *subclass* attribute and must never reach the wire.
* **genuine bugs still crash** — ``AttributeError`` and friends are not "this
  project is unreadable", they are "this code is wrong", and must propagate.

A new file rather than an addition to tests/test_puppetmaster.py: that file is
huge and concurrently edited by other workstreams around the same lines.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.dashboard import (
    ProjectsSnapshot,
    list_all_projects_snapshot,
    project_slug,
    serve,
)
from puppetmaster.store_factory import create_store

# Deliberately not a valid SQLite header ("SQLite format 3\000"), and long
# enough that sqlite reads a full page before deciding: opening this raises
# sqlite3.DatabaseError("file is not a database").
_GARBAGE = b"this is not a database, it is a pile of bytes\n" * 128

HEALTHY_DIR = "alpha-0123456789ab"
CORRUPT_DIR = "beta-ba9876543210"


class _Fixture:
    """A temp `app_state_root()` holding one healthy and one corrupt project."""

    def __init__(self, test: unittest.TestCase) -> None:
        tmp = TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.projects = self.root / "projects"

        self.healthy_dir = self.projects / HEALTHY_DIR
        store = create_store("sqlite", self.healthy_dir)
        store.init()
        self.job_ids = [
            store.create_job("alpha goal one").id,
            store.create_job("alpha goal two").id,
        ]

        # Corrupt the second project for real. Built by hand rather than
        # init()-then-clobber so no live handle or -wal sidecar is left over to
        # change which error sqlite raises (Windows keeps a mandatory lock on
        # an open DB file).
        self.corrupt_dir = self.projects / CORRUPT_DIR
        self.corrupt_dir.mkdir(parents=True)
        (self.corrupt_dir / "state.sqlite3").write_bytes(_GARBAGE)

    def patch_state_root(self):
        """Point `list_project_state_dirs` at this fixture only.

        `app_state_root` is patched (not `list_project_state_dirs`) so the real
        directory walk runs — it is the code under test's actual input.
        """
        return patch("puppetmaster.state.app_state_root", return_value=self.root)


class ProjectSlugTests(unittest.TestCase):
    """`project_slug` replaced an inlined `re.sub(r"-[0-9a-f]{12}$", ...)`;
    output must be byte-identical to that strip."""

    def test_matches_the_previous_inline_strip(self) -> None:
        import re

        for name in (
            "alpha-0123456789ab",
            "no-digest",
            "beta-ba9876543210",
            # Boundary cases the 12-hex assumption must keep treating as
            # "no digest here": 11 and 13 hex chars, and non-hex.
            "gamma-0123456789a",
            "gamma-0123456789abc",
            "delta-0123456789zz",
            "0123456789ab",
        ):
            with self.subTest(name=name):
                previous = re.sub(r"-[0-9a-f]{12}$", "", name) or name
                self.assertEqual(project_slug(name), previous)

    def test_strips_digest_and_leaves_plain_names_alone(self) -> None:
        self.assertEqual(project_slug("alpha-0123456789ab"), "alpha")
        self.assertEqual(project_slug("no-digest"), "no-digest")


class UnreadableProjectIsReportedTests(unittest.TestCase):
    def test_corrupt_project_is_reported_not_dropped(self) -> None:
        fixture = _Fixture(self)
        with fixture.patch_state_root():
            snapshot = list_all_projects_snapshot()

        # The healthy project's jobs are still all there...
        by_id = {row["id"]: row for row in snapshot}
        for job_id in fixture.job_ids:
            self.assertIn(job_id, by_id)
            self.assertEqual(by_id[job_id]["project"], "alpha")

        # ...and the broken one is named out loud instead of vanishing.
        self.assertTrue(snapshot.partial)
        self.assertEqual([e["project"] for e in snapshot.errors], ["beta"])
        message = snapshot.errors[0]["error"]
        self.assertRegex(message, r"^\w*Error: .+")

        # Specifically the real sqlite failure, not some incidental OSError.
        # Compared against the exception the same corrupt file raises when read
        # directly, so this stays exact across sqlite/CPython wording changes.
        with self.assertRaises(sqlite3.Error) as raised:
            create_store("sqlite", fixture.corrupt_dir).list_jobs()
        expected = f"{type(raised.exception).__name__}: {raised.exception}"
        self.assertEqual(message, expected)

    def test_the_corruption_is_real_not_mocked(self) -> None:
        """Guards the fixture itself: if a future sqlite/store change stopped
        raising here, the test above would pass vacuously (zero errors, zero
        assertions about them) while proving nothing."""
        fixture = _Fixture(self)
        store = create_store("sqlite", fixture.corrupt_dir)
        with self.assertRaises(sqlite3.DatabaseError):
            store.list_jobs()

    def test_all_healthy_projects_report_no_errors(self) -> None:
        """The reporting path must stay quiet when nothing is broken."""
        fixture = _Fixture(self)
        (fixture.corrupt_dir / "state.sqlite3").unlink()
        create_store("sqlite", fixture.corrupt_dir).init()
        with fixture.patch_state_root():
            snapshot = list_all_projects_snapshot()
        self.assertEqual(snapshot.errors, [])
        self.assertFalse(snapshot.partial)


class WireContractTests(unittest.TestCase):
    """`/api/jobs` is a bare JSON array; the page iterates the top level and an
    existing test builds `{row["id"]: row for row in rows}`. Reshaping this into
    a dict would break both, so the errors must ride on an attribute."""

    def test_snapshot_is_a_list_that_serializes_as_a_bare_array(self) -> None:
        fixture = _Fixture(self)
        with fixture.patch_state_root():
            snapshot = list_all_projects_snapshot()

        self.assertIsInstance(snapshot, list)
        self.assertIsInstance(snapshot, ProjectsSnapshot)
        self.assertTrue(snapshot.errors, "fixture should have produced a failure")

        wire = json.loads(json.dumps(snapshot))
        self.assertIs(type(wire), list)
        self.assertEqual(len(wire), len(snapshot))
        for row in wire:
            self.assertIsInstance(row, dict)
            self.assertTrue(
                {"id", "goal", "status", "created_at", "completed_at", "project"}
                <= set(row)
            )
        # The failures are diagnostics, not payload: nothing leaks onto the wire.
        self.assertNotIn("errors", json.dumps(snapshot))

    def test_iterating_the_top_level_still_works(self) -> None:
        """Exactly what the page's `for (const j of jobs)` and
        tests/test_puppetmaster.py's `{row["id"]: row for row in rows}` do."""
        fixture = _Fixture(self)
        with fixture.patch_state_root():
            snapshot = list_all_projects_snapshot()
        self.assertEqual(
            sorted({row["id"]: row for row in snapshot}), sorted(fixture.job_ids)
        )


class GenuineBugsPropagateTests(unittest.TestCase):
    def test_attribute_error_is_not_swallowed(self) -> None:
        """AttributeError is a bug in this process, not an unreadable project.
        The old bare `except Exception` laundered it into "that project has no
        jobs" -- so a typo'd attribute silently emptied the whole board."""
        fixture = _Fixture(self)
        with fixture.patch_state_root(), patch(
            "puppetmaster.dashboard._job_snapshot_meta",
            side_effect=AttributeError("boom"),
        ):
            with self.assertRaises(AttributeError) as ctx:
                list_all_projects_snapshot()
        self.assertIn("boom", str(ctx.exception))

    def test_import_error_is_not_swallowed(self) -> None:
        fixture = _Fixture(self)
        with fixture.patch_state_root(), patch(
            "puppetmaster.dashboard._job_snapshot_meta",
            side_effect=ImportError("no such module"),
        ):
            with self.assertRaises(ImportError):
                list_all_projects_snapshot()


class SkipWarningReachesStderrTests(unittest.TestCase):
    """`log_message` is silenced in this handler and stderr is what the
    `--background` / MCP launch paths capture into dashboard.err.log, so stderr
    is the only channel from here that reaches a human. It must also dedupe:
    /api/jobs is polled every 1.5s."""

    def _make_handler(self, fixture):
        from puppetmaster.dashboard import make_handler

        return make_handler(
            lambda: create_store("sqlite", fixture.healthy_dir),
            all_projects=True,
            backend="sqlite",
            state_dir=fixture.healthy_dir,
        )

    def _poll(self, handler) -> list:
        """Run the handler's /api/jobs all-projects branch without a socket.

        BaseHTTPRequestHandler.__init__ *is* the request loop, so it cannot be
        instantiated without a live connection; drive do_GET on a bare instance
        with the response plumbing captured instead. The captured status is
        asserted so do_GET's own `except Exception -> 500` can never make a
        broken poll look like a successful one.
        """
        sent: list = []
        instance = handler.__new__(handler)
        instance.path = "/api/jobs"
        instance.client_address = ("127.0.0.1", 0)
        instance._send = lambda code, body, content_type: sent.append((code, body))
        instance.do_GET()
        self.assertEqual([code for code, _ in sent], [200])
        return json.loads(sent[0][1])

    def test_new_failures_warn_once_and_are_deduped(self) -> None:
        import io
        import sys

        fixture = _Fixture(self)
        handler = self._make_handler(fixture)

        buffer = io.StringIO()
        with fixture.patch_state_root(), patch.object(sys, "stderr", buffer):
            # Three polls -- what the page's 1.5s interval produces in ~5s.
            for _ in range(3):
                rows = self._poll(handler)
                self.assertEqual(len(rows), len(fixture.job_ids))

        lines = [ln for ln in buffer.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1, f"expected one deduped warning, got {lines}")
        self.assertRegex(lines[0], r"^dashboard: skipping project beta: \w*Error: .+")

    def test_skipped_projects_dict_tracks_latest_state(self) -> None:
        import io
        import sys

        fixture = _Fixture(self)
        handler = self._make_handler(fixture)

        buffer = io.StringIO()
        with fixture.patch_state_root(), patch.object(sys, "stderr", buffer):
            self._poll(handler)
            self.assertIn("beta", handler.skipped_projects)
            self.assertRegex(handler.skipped_projects["beta"], r"^\w*Error: .+")

            # Heal the project: the latest-state dict must drop it, so a
            # recovered project stops being reported as skipped.
            (fixture.corrupt_dir / "state.sqlite3").unlink()
            create_store("sqlite", fixture.corrupt_dir).init()
            self._poll(handler)
            self.assertEqual(handler.skipped_projects, {})


class ApiJobsStaysAJsonArrayTests(unittest.TestCase):
    """End to end over a real socket: a corrupt project in the walk must not
    change the shape (or the status code) of /api/jobs."""

    def test_api_jobs_returns_a_json_array_with_a_broken_project_present(self) -> None:
        fixture = _Fixture(self)
        with fixture.patch_state_root():
            httpd = serve(
                fixture.healthy_dir,
                backend="sqlite",
                host="127.0.0.1",
                port=0,
                open_browser=False,
                serve_forever=False,
                all_projects=True,
            )
            self.addCleanup(httpd.server_close)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            port = httpd.server_address[1]
            url = f"http://127.0.0.1:{port}/api/jobs"
            with urllib.request.urlopen(url) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read())

        self.assertIs(type(payload), list)
        self.assertEqual(
            sorted(row["id"] for row in payload), sorted(fixture.job_ids)
        )
        self.assertEqual({row["project"] for row in payload}, {"alpha"})


if __name__ == "__main__":
    unittest.main()
