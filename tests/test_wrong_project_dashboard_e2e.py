"""End-to-end proof that the reported wrong-project dashboard now announces itself.

THE INCIDENT THIS REPRODUCES
============================
A user ran ``puppetmaster dashboard`` for two repositories. One board showed
live work; the other showed almost nothing, and nothing on screen explained
why.

``state.default_state_dir()`` hashes ``_git_root(cwd) or cwd``. The session had
been opened at ``C:\\Projects\\puppetmaster`` — a plain wrapper folder that is
*not* a git repository; the real checkout sits one level down at
``C:\\Projects\\puppetmaster\\Puppetmaster``. So the ``or`` fallback fired, and
one logical project silently forked into two state dirs:

* ``puppetmaster-c3177e6032c4`` — 1 stale job. This is what the dashboard
  served, and it served it correctly by its own identity check.
* ``Puppetmaster-b92145e840c8`` — the real history, where every job had
  actually been written.

The board was right about *which state dir it was serving* and silent about
*that state dir being the wrong project*. That silence is the defect.

WHY THE FIXTURE IS SHAPED THIS WAY
==================================
Two properties of the fixture below are load-bearing rather than incidental:

1. The nested repository is created with a **real** ``git init``, not a stubbed
   ``.git`` directory and not a patched ``_git_root``. The whole incident is a
   disagreement between what ``git rev-parse --show-toplevel`` answers from the
   wrapper (nothing) and from the checkout (the checkout), so a fixture that
   fakes git away would prove something else. Everything here therefore runs
   only when a real ``git`` is on PATH.

2. The wrapper's state dir is seeded with exactly **one** job — not zero. The
   real bad dir held a single stale job, which is precisely why the dashboard's
   empty-state branch never rendered and why nothing looked obviously broken. A
   fixture with zero jobs would pass against a naive "warn when the dir is
   empty" rule and would therefore not prove the reported case at all.

WHAT IS PROVED
==============
Every surface a human or agent could have consulted at the moment of confusion
now names the busier dir, and none of them leaks an absolute path:

* the fork itself is real (two distinct hashed dirs for one project);
* ``puppetmaster doctor``'s ``state-identity`` row warns;
* the stderr line ``serve()`` prints beside "Reading durable state from:";
* the dashboard's ``/api/diagnostics`` endpoint, with ``/api/meta`` asserted
  alongside it so the identity contract that prevents duplicate dashboards is
  visibly undisturbed.

Two negative controls carry equal weight: run from the *correct* place, and run
with an explicit ``--state-dir``, everything must stay quiet. Without them the
suite could pass while warning about everything.

This file only observes. It adds no behaviour and modifies no source; the
detector (``puppetmaster.state_health``), the doctor row, the dashboard
endpoint and the MCP warning are pinned unit-by-unit in
tests/test_state_dir_identity.py and tests/test_dashboard_state_dir_warning.py.
What is new here is assembling all of them against one real git tree.
"""
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
import shutil
import subprocess
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple
from unittest.mock import patch

from puppetmaster import state
from puppetmaster.dashboard import serve
from puppetmaster.diagnostics import run_doctor
from puppetmaster.state_health import (
    diagnose_state_dir,
    short_warning,
    summarize_project_state_dir,
)

# Fixed mtimes keep the fixture deterministic; the detector reads them only to
# report "last activity", never to decide.
_JOB_EPOCH = 1_700_000_000.0

# One, not zero -- see "WHY THE FIXTURE IS SHAPED THIS WAY" above. This is also
# exactly ``diagnose_state_dir``'s ``near_empty_max`` default, so the fixture
# sits on the boundary the real incident sat on.
WRAPPER_JOBS = 1

# "Several", standing in for the real dir's history. Enough to clear the
# detector's materially-busier bar from a 1-job active dir.
INNER_JOBS = 6

# /api/meta is the identity contract every dashboard-reuse check reads, under a
# short socket timeout and a bounded read. An oversized body silently becomes
# None on the client, which reads as "not my dashboard" and makes the CLI and
# MCP spawn duplicate servers -- so its size is part of the contract.
META_MAX_BYTES = 4096


class Incident(NamedTuple):
    """The reported filesystem, resolved. See :func:`_incident`."""

    base: Path
    wrapper: Path
    inner: Path
    wrapper_state_dir: Path
    inner_state_dir: Path


def _make_jobs(state_dir: Path, count: int, *, offset: float = 0.0) -> None:
    """Fake job history: ``<state_dir>/jobs/<id>`` directories.

    Directories are the whole fixture because the detector counts them rather
    than opening the store -- it has to stay SQLite-free so that answering a
    diagnostic question never spins up the WAL writer.
    """
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        job = jobs / f"job_{state_dir.name[:8]}_{index:03d}"
        job.mkdir(exist_ok=True)
        stamp = _JOB_EPOCH + offset + index
        os.utime(job, (stamp, stamp))


@contextlib.contextmanager
def _chdir(target: Path):
    """Run the block with ``target`` as the *process* cwd, then restore it.

    Not cosmetic. ``dashboard._state_dir_diagnosis`` calls the detector with no
    ``cwd``, so ``/api/diagnostics`` diagnoses against ``Path.cwd()`` -- the
    serving process's cwd *is* the workspace whose jobs the user expects to
    see. Reproducing the incident over HTTP therefore means the serving process
    actually standing in the wrapper directory. This repo's own root is a git
    checkout, so without the chdir the detector's "cwd is a git root" guard
    would fire and the HTTP assertions would pass vacuously.
    """
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _incident(tmp: str):
    """Build the reported situation from scratch, with a real git repository.

    ``<tmp>/wrapper`` is an ordinary folder with no repository in it or above
    it; ``<tmp>/wrapper/Inner`` is a genuine ``git init`` checkout. That is the
    incident's precondition, and it is left real rather than mocked: the bug is
    a disagreement between what git answers from each directory.

    ``app_state_root`` is redirected under ``<tmp>`` so both derived state dirs
    live in the temp tree. Yields inside the patch on purpose --
    ``projects_root``, ``project_state_dir_for`` and ``default_state_dir`` all
    resolve that root at *call* time, so any expected value derived after the
    block exits would silently be computed against the real AppData root and
    would make the expectation wrong rather than the assertion.
    """
    base = Path(tmp).resolve()
    wrapper = base / "wrapper"
    inner = wrapper / "Inner"
    inner.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=inner, check=True, capture_output=True)
    with patch.object(state, "app_state_root", return_value=base / "app-state"):
        wrapper_state_dir = state.project_state_dir_for(wrapper)
        inner_state_dir = state.project_state_dir_for(inner)
        _make_jobs(wrapper_state_dir, WRAPPER_JOBS)
        _make_jobs(inner_state_dir, INNER_JOBS, offset=100.0)
        yield Incident(base, wrapper, inner, wrapper_state_dir, inner_state_dir)


@unittest.skipUnless(shutil.which("git"), "git required")
class WrongProjectDashboardEndToEndTests(unittest.TestCase):
    """One real git tree, every surface that could have broken the silence."""

    # -- shared assertions -------------------------------------------------

    def _assert_no_absolute_path(self, text: str, paths) -> None:
        """No path from the fixture tree appears in ``text``, raw or escaped.

        The escaped form matters: these strings travel through
        ``json.dumps``, which doubles a Windows separator, so a naive
        substring check for ``C:\\Users\\...`` would miss ``C:\\\\Users\\\\...``
        in a JSON body and quietly pass. On POSIX the two forms coincide and
        the second check is a harmless duplicate.

        Note what is *not* asserted for a JSON body: "contains no os.sep at
        all". ``short_warning``'s em-dash is emitted by ``json.dumps`` as the
        escape ``\\u2014``, whose backslash is ``os.sep`` on Windows -- so that
        stronger form would fail on an artefact of the server's own encoding.
        """
        for path in paths:
            raw = str(path)
            self.assertNotIn(raw, text, raw)
            escaped = json.dumps(raw)[1:-1]
            self.assertNotIn(escaped, text, escaped)

    def _state_identity_row(self, root: Path, state_dir: Path):
        """The ``state-identity`` row from a full ``run_doctor`` report.

        Through ``run_doctor`` rather than the private check, because the row
        only helps if it actually reaches the report a user runs -- and because
        the report must survive a folder that is not a repository at all.
        """
        checks = {c.name: c for c in run_doctor(root, state_dir)}
        self.assertIn("state-identity", checks)
        row = checks["state-identity"]
        # A diagnostic that crashes the report it lives in would be worse than
        # the silence it replaces.
        self.assertNotIn(row.status, {"fail", "error"}, row.detail)
        self.assertTrue(row.evidence, row.detail)
        return row

    # -- 1. the fork is real ----------------------------------------------

    def test_wrapper_and_nested_repo_fork_into_two_state_dirs(self) -> None:
        """The or-fallback fires: one project, two hashed dirs, no error.

        This is the whole root cause, stated as an equality. Both derived paths
        are computed inside the ``app_state_root`` patch, since
        ``project_state_dir_for`` resolves ``projects_root()`` at call time.
        """
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            from_wrapper = state.default_state_dir(cwd=fx.wrapper)
            from_inner = state.default_state_dir(cwd=fx.inner)
            formula_wrapper = state.project_state_dir_for(fx.wrapper)
            formula_inner = state.project_state_dir_for(fx.inner)
            projects_root = state.projects_root()
            wrapper_jobs = summarize_project_state_dir(fx.wrapper_state_dir).job_count
            inner_jobs = summarize_project_state_dir(fx.inner_state_dir).job_count

            # The fork.
            self.assertNotEqual(from_wrapper, from_inner)
            # Real git wins from inside the checkout: --show-toplevel and
            # Path.resolve() agree, so the formula the detector uses to map a
            # sibling checkout to its state dir matches what the checkout
            # itself would resolve to.
            self.assertEqual(from_inner, formula_inner)
            # ...and from the wrapper there is no repository to find, so cwd
            # itself gets hashed. If this ever fails, the machine's temp dir is
            # inside a git checkout and the fixture's precondition is void.
            self.assertEqual(
                from_wrapper,
                formula_wrapper,
                "the wrapper must not resolve to any enclosing git root",
            )
            # Both are project-scoped: one logical project, two siblings under
            # the same projects/ root. That is what makes it a fork rather than
            # two unrelated projects.
            self.assertEqual(from_wrapper.parent, projects_root)
            self.assertEqual(from_inner.parent, projects_root)
            # The reported asymmetry, in the counter the detector actually uses.
            self.assertEqual(wrapper_jobs, WRAPPER_JOBS)
            self.assertEqual(inner_jobs, INNER_JOBS)

    # -- 2. the doctor surface --------------------------------------------

    def test_doctor_warns_and_names_the_busy_dir_and_every_remedy(self) -> None:
        """``puppetmaster doctor`` stops reporting the fork as a benign optional."""
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            row = self._state_identity_row(fx.wrapper, fx.wrapper_state_dir)
            detail = row.detail

            self.assertEqual(row.status, "warn", detail)

            # Both job counts, each bound to the dir it belongs to. Asserting a
            # bare "6" would be vacuous -- the 12-hex digest in either basename
            # can contain that character by chance.
            self.assertIn(
                f"{fx.wrapper_state_dir.name} holds {WRAPPER_JOBS} job(s)", detail
            )
            self.assertIn(f"{fx.inner_state_dir.name} holds {INNER_JOBS}", detail)

            # Every escape hatch, because which one applies depends on what the
            # user was actually trying to do: move, pin, or look at everything.
            for needle in (
                "not a git repository",
                "git root",
                "--state-dir",
                "--all-projects",
                "puppetmaster projects",
            ):
                self.assertIn(needle, detail, needle)

            # Basenames only. A doctor report is pasted into issues verbatim.
            self._assert_no_absolute_path(
                detail,
                (
                    fx.base,
                    fx.wrapper,
                    fx.inner,
                    fx.wrapper_state_dir,
                    fx.inner_state_dir,
                ),
            )

    # -- 3. the launch surface --------------------------------------------

    def test_short_warning_is_one_safe_sentence_naming_the_nested_repo(self) -> None:
        """The one-line form ``serve()`` prints to stderr and MCP embeds.

        ``short_warning`` names the *workspace* basename of the nested repo
        rather than its full hashed state dir name, because it is capped at 140
        characters for a status line. The two identify the same directory: a
        state dir's basename is that workspace name plus a 12-hex digest, which
        the ``startswith`` assertion below pins rather than assumes.
        """
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            diagnosis = diagnose_state_dir(
                cwd=fx.wrapper, state_dir=fx.wrapper_state_dir
            )
            warning = short_warning(diagnosis)

            self.assertIsInstance(warning, str)
            self.assertTrue(warning)
            self.assertLessEqual(len(warning), 140)

            self.assertTrue(
                fx.inner_state_dir.name.startswith(f"{fx.inner.name}-"),
                f"{fx.inner_state_dir.name} does not name {fx.inner.name}",
            )
            self.assertIn(f"nested repo {fx.inner.name} has {INNER_JOBS}", warning)
            self.assertIn(
                f"{fx.wrapper_state_dir.name} has {WRAPPER_JOBS} job(s)", warning
            )

            # This line lands in dashboard.err.log, which users paste verbatim.
            self._assert_no_absolute_path(
                warning,
                (
                    fx.base,
                    fx.wrapper,
                    fx.inner,
                    fx.wrapper_state_dir,
                    fx.inner_state_dir,
                ),
            )
            # Here -- unlike a JSON body -- the stronger form does hold: the
            # sentence contains no path separator of any kind.
            self.assertNotIn(os.sep, warning)
            self.assertNotIn("/", warning)

    # -- 4. the HTTP surface ----------------------------------------------

    def _serve(self, state_dir: Path) -> int:
        """Bind a real dashboard on ``state_dir`` and return its port.

        ``port=0`` lets the OS pick, so a developer's own dashboard on 8787
        cannot collide with the test. ``addCleanup`` is LIFO, so
        ``server_close`` is registered FIRST in order to run LAST -- shutdown
        has to stop the serving thread before the socket is closed under it.
        """
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            httpd = serve(
                state_dir,
                backend="sqlite",
                host="127.0.0.1",
                port=0,
                open_browser=False,
                serve_forever=False,
            )
        self.addCleanup(httpd.server_close)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    @staticmethod
    def _get(port: int, path: str) -> dict:
        """One GET against the live server; the raw body is kept for scanning."""
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=30
        ) as resp:
            return {"code": resp.status, "raw": resp.read().decode("utf-8")}

    def test_live_dashboard_reports_the_fork_without_disturbing_its_identity(
        self,
    ) -> None:
        """A real bound server: /api/diagnostics tells, /api/meta is untouched.

        Both endpoints in one test on purpose. The diagnosis is the new thing;
        /api/meta is the thing it must not have cost us, and asserting them
        against the same running server is the only way to show that the
        filesystem walk behind one did not leak into the other.
        """
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            # The serving process must actually stand in the wrapper: the
            # endpoint diagnoses against Path.cwd(). See _chdir.
            with _chdir(fx.wrapper):
                port = self._serve(fx.wrapper_state_dir)
                diagnostics = self._get(port, "/api/diagnostics")
                meta = self._get(port, "/api/meta")

            self.assertEqual(diagnostics["code"], 200)
            diagnosis = json.loads(diagnostics["raw"])["diagnosis"]
            self.assertIsNotNone(diagnosis, diagnostics["raw"])
            self.assertTrue(diagnosis["suspect"])
            self.assertEqual(diagnosis["candidate"], fx.inner_state_dir.name)
            self.assertEqual(diagnosis["candidate_jobs"], INNER_JOBS)
            # The board is still honest about what it *is* serving -- it was
            # never wrong about that; it was only silent about it being the
            # wrong project.
            self.assertEqual(diagnosis["project"], fx.wrapper_state_dir.name)
            self.assertEqual(diagnosis["jobs"], WRAPPER_JOBS)
            self.assertTrue(diagnosis["message"])

            # The endpoint is unauthenticated and reachable off-loopback under
            # --mobile, so the body carries basenames and never a filesystem
            # layout.
            self._assert_no_absolute_path(
                diagnostics["raw"],
                (
                    fx.base,
                    fx.wrapper,
                    fx.inner,
                    fx.wrapper_state_dir,
                    fx.inner_state_dir,
                ),
            )

            # ...and the identity contract that guards against duplicate
            # dashboards still answers, small and unchanged.
            self.assertEqual(meta["code"], 200)
            meta_body = json.loads(meta["raw"])
            self.assertEqual(meta_body["service"], "puppetmaster-dashboard")
            self.assertEqual(meta_body["project"], fx.wrapper_state_dir.name)
            self.assertFalse(meta_body["all_projects"])
            self.assertLess(len(meta["raw"].encode("utf-8")), META_MAX_BYTES)

    # -- 5. negative control: run from the right place ---------------------

    def test_running_from_the_nested_repo_says_nothing_at_all(self) -> None:
        """The correct workspace on the identical filesystem must stay quiet.

        As important as the positive case: without it the suite could pass
        while warning about everything. The paired wrapper assertion at the end
        is what makes this non-vacuous -- the same tree, the same run, one
        warning and one silence, decided only by where you stand.
        """
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            row = self._state_identity_row(fx.inner, fx.inner_state_dir)
            diagnosis = diagnose_state_dir(
                cwd=fx.inner, state_dir=fx.inner_state_dir
            )
            warning = short_warning(diagnosis)
            control = self._state_identity_row(fx.wrapper, fx.wrapper_state_dir)

            self.assertNotEqual(row.status, "warn", row.detail)
            self.assertNotIn(row.status, {"warn", "fail", "error"}, row.detail)
            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertIsNone(warning)
            # ...and it is quiet for the *right* reason: cwd is the git root,
            # so its state dir is correct by definition and a nested checkout
            # would be a legitimately separate project.
            self.assertIn("verdict:workspace-is-git-root", diagnosis.evidence)
            self.assertIn("cwd_is_git_root:true", diagnosis.evidence)

            # Not vacuous: the very same fixture still warns from the wrapper.
            self.assertEqual(control.status, "warn", control.detail)

    # -- 6. negative control: an explicit state dir is a deliberate choice --

    def test_an_explicit_state_dir_is_never_second_guessed(self) -> None:
        """``--state-dir`` / ``PUPPETMASTER_STATE_DIR`` opts out by construction.

        Only the hashed ``projects/`` layout can fork, so a path the user named
        themselves is out of scope -- contradicting it would make the documented
        escape hatch from this very warning itself produce the warning.
        """
        with TemporaryDirectory() as tmp, _incident(tmp) as fx:
            explicit = fx.base / "explicit-state"
            _make_jobs(explicit, WRAPPER_JOBS)
            projects_root = state.projects_root()

            row = self._state_identity_row(fx.wrapper, explicit)
            diagnosis = diagnose_state_dir(cwd=fx.wrapper, state_dir=explicit)
            warning = short_warning(diagnosis)
            control = self._state_identity_row(fx.wrapper, fx.wrapper_state_dir)

            # The premise: this really is outside the hashed layout.
            self.assertNotEqual(explicit.parent, projects_root)

            self.assertNotEqual(row.status, "warn", row.detail)
            self.assertEqual(row.status, "optional", row.detail)
            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.candidates, ())
            self.assertIsNone(warning)

            # Not vacuous: same cwd, same disk, same run -- only the pin
            # differs, and unpinned still warns.
            self.assertEqual(control.status, "warn", control.detail)


if __name__ == "__main__":
    unittest.main()
