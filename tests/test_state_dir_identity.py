"""Detect-only identity checks for a forked / wrong-project state dir.

``default_state_dir`` hashes ``_git_root(cwd) or cwd``. A session opened one
level *above* the real checkout makes the ``or`` fallback fire and silently
splits one logical project into two state dirs. These tests pin the detector
that notices, and pin that the detector stays cheap: no git subprocess when a
state dir is supplied, and never a SQLite connection.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import contextlib
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import state
from puppetmaster.diagnostics import run_doctor
from puppetmaster.state_health import (
    VERDICTS,
    diagnose_state_dir,
    short_warning,
    summarize_project_state_dir,
)


_JOB_EPOCH = 1_700_000_000.0


def _make_git(path: Path) -> None:
    """Mark ``path`` as a checkout the way a real clone does."""
    (path / ".git").mkdir(parents=True, exist_ok=True)


def _make_jobs(state_dir: Path, count: int, *, offset: float = 0.0) -> Path:
    """Fake job history: ``<state_dir>/jobs/<id>`` dirs with fixed mtimes."""
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        job = jobs / f"job_{state_dir.name[:8]}_{index:03d}"
        job.mkdir(exist_ok=True)
        stamp = _JOB_EPOCH + offset + index
        os.utime(job, (stamp, stamp))
    return jobs


def _evidence_keys(diagnosis) -> set[str]:
    return {item.split(":", 1)[0] for item in diagnosis.evidence}


class StateDirIdentityTests(unittest.TestCase):
    """The detector's guard chain, cheapest guard first."""

    @contextlib.contextmanager
    def _app_state(self, tmp: Path):
        """Redirect the app state root at ``tmp/app-state`` for the block.

        Scoping rule for every test below: ``project_state_dir_for``,
        ``projects_root``, ``app_state_root``, ``list_project_state_dirs`` and
        ``default_state_dir`` resolve the root *at call time*. Any expected
        value derived from them must therefore be produced inside this block —
        call it here and bind it to a local, then assert on the local. Calling
        one of them after the block exits recomputes against the real AppData
        root and silently makes the *expectation* wrong, not the actual value.
        """
        with patch(
            "puppetmaster.state.app_state_root", return_value=tmp / "app-state"
        ):
            yield

    def _diagnose(self, **kwargs):
        """Every call also pins the closed verdict vocabulary."""
        diagnosis = diagnose_state_dir(**kwargs)
        self.assertIn(diagnosis.verdict, VERDICTS)
        return diagnosis

    def _fork_fixture(self, tmp: Path, *, active_jobs: int, child_jobs: int):
        """The reported incident: wrapper folder above the real checkout.

        ``tmp/puppetmaster`` is not a git repo, ``tmp/puppetmaster/Puppetmaster``
        is. Returns ``(wrapper, child, active_dir, child_dir)``.
        """
        wrapper = tmp / "puppetmaster"
        child = wrapper / "Puppetmaster"
        child.mkdir(parents=True)
        _make_git(child)
        active_dir = state.project_state_dir_for(wrapper)
        child_dir = state.project_state_dir_for(child)
        _make_jobs(active_dir, active_jobs)
        _make_jobs(child_dir, child_jobs, offset=100.0)
        return wrapper, child, active_dir, child_dir

    def test_reported_incident_is_suspect_with_one_candidate(self) -> None:
        """One stale job in the wrapper dir, real history in the nested repo."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, child, active_dir, child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=5
                )
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)
                # Bound here, not below: project_state_dir_for resolves
                # projects_root() at call time, so the expected path has to come
                # from the same app_state_root patch that produced the actual
                # one. Called after the block it would rebuild the same 12-hex
                # digest under the real AppData root and never match.
                expected_child_dir = state.project_state_dir_for(child)

            self.assertTrue(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "forked-project")
            self.assertEqual(diagnosis.active.job_count, 1)
            self.assertEqual(len(diagnosis.candidates), 1)
            candidate = diagnosis.candidates[0]
            self.assertEqual(candidate.state_dir, child_dir)
            self.assertEqual(candidate.state_dir, expected_child_dir)
            self.assertEqual(candidate.workspace, child)
            self.assertEqual(candidate.job_count, 5)
            self.assertLessEqual(
                {
                    "verdict",
                    "active",
                    "active_jobs",
                    "cwd",
                    "cwd_is_git_root",
                    "expected_for_cwd",
                    "scanned_children",
                    "candidates",
                    "candidate",
                },
                _evidence_keys(diagnosis),
            )
            self.assertIn("cwd_is_git_root:false", diagnosis.evidence)
            self.assertIn(f"candidate:{child_dir.name}=5", diagnosis.evidence)

    def test_populated_active_dir_is_not_suspect(self) -> None:
        """Anti-noise: a wrapper with real history of its own stays quiet."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper = root / "workspace"
                child = wrapper / "app"
                child.mkdir(parents=True)
                _make_git(child)
                active_dir = state.project_state_dir_for(wrapper)
                _make_jobs(active_dir, 4)
                _make_jobs(state.project_state_dir_for(child), 12, offset=100.0)
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)

            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "ok")

    def test_name_twin_beats_a_populated_active_dir(self) -> None:
        """A same-slug twin that is *far* busier still wins (guard 6)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, active_dir, child_dir = self._fork_fixture(
                    root, active_jobs=4, child_jobs=24
                )
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)

            self.assertTrue(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "name-twin")
            self.assertIn(f"name_twin:{child_dir.name}", diagnosis.evidence)

    def test_child_repo_without_history_is_not_a_candidate(self) -> None:
        """A fresh nested clone with no jobs proves nothing."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, active_dir, _child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=0
                )
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)

            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.candidates, ())
            self.assertEqual(diagnosis.verdict, "new")

    def test_cwd_that_is_a_git_root_is_never_suspect(self) -> None:
        """Guard 2: a busy submodule / vendored checkout is legitimately separate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                repo = root / "repo"
                vendored = repo / "vendor-checkout"
                vendored.mkdir(parents=True)
                _make_git(repo)
                _make_git(vendored)
                active_dir = state.project_state_dir_for(repo)
                _make_jobs(state.project_state_dir_for(vendored), 20)
                diagnosis = self._diagnose(cwd=repo, state_dir=active_dir)

            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "workspace-is-git-root")
            self.assertEqual(diagnosis.candidates, ())

    def test_explicit_state_dir_outside_projects_root_is_not_diagnosed(self) -> None:
        """Guard 1: an explicit --state-dir / env pin is a deliberate choice."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, _active, child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=24
                )
                explicit = root / "explicit-state"
                _make_jobs(explicit, 1)
                diagnosis = self._diagnose(cwd=wrapper, state_dir=explicit)

            self.assertGreater(summarize_project_state_dir(child_dir).job_count, 1)
            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "not-project-scoped")
            self.assertEqual(diagnosis.candidates, ())

    def test_state_dir_from_enclosing_git_root_is_not_diagnosed(self) -> None:
        """Guard 3: running inside a subdirectory of a repo is normal."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                repo = root / "repo"
                sub = repo / "packages"
                nested = sub / "nested"
                nested.mkdir(parents=True)
                _make_git(repo)
                _make_git(nested)
                # What the enclosing git root hashes to -- not what ``sub`` does.
                active_dir = state.project_state_dir_for(repo)
                _make_jobs(active_dir, 1)
                _make_jobs(state.project_state_dir_for(nested), 30, offset=100.0)
                diagnosis = self._diagnose(cwd=sub, state_dir=active_dir)
                # Same call-time resolution as everywhere else: bind what ``sub``
                # hashes to under the patch. Outside the block this would be
                # rebuilt against the real AppData root -- it happens to still
                # compare unequal here, so the mistake would sit latent.
                sub_dir = state.project_state_dir_for(sub)

            self.assertNotEqual(active_dir.name, sub_dir.name)
            self.assertFalse(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "not-cwd-derived")

    def test_two_populated_child_repos_are_ambiguous_best_first(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper = root / "workspaces"
                alpha = wrapper / "alpha"
                beta = wrapper / "beta"
                alpha.mkdir(parents=True)
                beta.mkdir(parents=True)
                _make_git(alpha)
                _make_git(beta)
                active_dir = state.project_state_dir_for(wrapper)
                active_dir.mkdir(parents=True, exist_ok=True)
                _make_jobs(state.project_state_dir_for(alpha), 4)
                _make_jobs(state.project_state_dir_for(beta), 9, offset=100.0)
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)

            self.assertTrue(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(diagnosis.verdict, "ambiguous")
            self.assertEqual([c.job_count for c in diagnosis.candidates], [9, 4])
            self.assertEqual(
                [c.workspace.name for c in diagnosis.candidates], ["beta", "alpha"]
            )

    def test_probe_with_explicit_state_dir_never_shells_out_to_git(self) -> None:
        """A supplied state dir means filesystem stats only."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, active_dir, _child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=5
                )
                with patch(
                    "puppetmaster.state.subprocess.run",
                    side_effect=AssertionError("unexpected git"),
                ):
                    diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)

            self.assertTrue(diagnosis.suspect, diagnosis.evidence)

    def test_probe_never_opens_the_sqlite_store(self) -> None:
        """Counting job dirs must not spin up the WAL writer."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, active_dir, _child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=5
                )
                (active_dir / "state.sqlite3").write_bytes(b"")
                with patch(
                    "sqlite3.connect",
                    side_effect=AssertionError("opened the sqlite store"),
                ):
                    diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)
                    summary = summarize_project_state_dir(active_dir)

            self.assertTrue(diagnosis.suspect, diagnosis.evidence)
            self.assertEqual(summary.job_count, 1)

    def test_summarize_missing_dir_yields_zeros(self) -> None:
        with TemporaryDirectory() as tmp:
            summary = summarize_project_state_dir(Path(tmp) / "nope")
            self.assertEqual(summary.job_count, 0)
            self.assertIsNone(summary.last_activity)

    def test_short_warning_has_no_absolute_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper, _child, active_dir, _child_dir = self._fork_fixture(
                    root, active_jobs=1, child_jobs=5
                )
                diagnosis = self._diagnose(cwd=wrapper, state_dir=active_dir)
                healthy = self._diagnose(cwd=wrapper, state_dir=root / "pinned")

            warning = short_warning(diagnosis)
            self.assertIsNotNone(warning)
            self.assertLessEqual(len(warning), 140)
            self.assertNotIn(str(root), warning)
            self.assertNotIn(os.sep, warning)
            self.assertIsNone(short_warning(healthy))

    @unittest.skipUnless(shutil.which("git"), "git is required for this test")
    def test_real_git_root_pivot_and_formula_agree(self) -> None:
        """Real ``git rev-parse`` output must agree with ``project_state_dir_for``.

        Load-bearing: the detector maps a sibling checkout to its state dir
        via the formula, never via git. If ``--show-toplevel`` and
        ``Path.resolve()`` disagreed, guard 3 would fire on a real repo.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outer = root / "puppetmaster"
            repo = outer / "Puppetmaster"
            sub = repo / "puppetmaster"
            sub.mkdir(parents=True)
            subprocess.run(
                ["git", "init", "-q"], cwd=repo, check=True, capture_output=True
            )
            with self._app_state(root):
                from_sub = state.default_state_dir(cwd=sub)
                from_repo = state.default_state_dir(cwd=repo)
                from_outer = state.default_state_dir(cwd=outer)
                formula = state.project_state_dir_for(repo)

            # The pivot fires from a subdirectory: both land on the repo root.
            self.assertEqual(from_sub, from_repo)
            # The wrapper folder is the fork: it hashes to something else.
            self.assertNotEqual(from_repo, from_outer)
            # git's --show-toplevel string and Path.resolve() agree.
            self.assertEqual(from_repo, formula)


class StateIdentityDoctorRowTests(unittest.TestCase):
    """``puppetmaster doctor`` surfaces the fork instead of a benign optional."""

    @contextlib.contextmanager
    def _app_state(self, tmp: Path):
        """Same call-time-resolution rule as ``StateDirIdentityTests._app_state``.

        ``run_doctor`` reaches ``project_state_dir_for``/``projects_root``
        through the detector, so both the rows and any path they are compared
        against must be produced inside this block.
        """
        with patch(
            "puppetmaster.state.app_state_root", return_value=tmp / "app-state"
        ):
            yield

    def _row(self, root: Path, state_dir: Path):
        checks = {c.name: c for c in run_doctor(root, state_dir)}
        self.assertIn("state-identity", checks)
        return checks["state-identity"]

    def test_doctor_rows_never_fail_and_warn_on_a_fork(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._app_state(root):
                wrapper = root / "puppetmaster"
                child = wrapper / "Puppetmaster"
                child.mkdir(parents=True)
                (child / ".git").mkdir()
                active_dir = state.project_state_dir_for(wrapper)
                child_dir = state.project_state_dir_for(child)
                _make_jobs(active_dir, 1)
                _make_jobs(child_dir, 5, offset=100.0)
                explicit = root / "explicit-state"
                explicit.mkdir()

                forked = self._row(wrapper, active_dir)
                pinned = self._row(wrapper, explicit)
                healthy = self._row(child, child_dir)

        self.assertEqual(forked.status, "warn")
        for needle in (
            child_dir.name,
            "5",
            "not a git repository",
            "--state-dir",
            "puppetmaster projects",
            "dashboard --all-projects",
        ):
            self.assertIn(needle, forked.detail)

        self.assertEqual(pinned.status, "optional")
        self.assertEqual(healthy.status, "ok")

        for row in (forked, pinned, healthy):
            self.assertNotIn(row.status, {"fail", "error"}, row.detail)
            self.assertTrue(row.evidence)


if __name__ == "__main__":
    unittest.main()
