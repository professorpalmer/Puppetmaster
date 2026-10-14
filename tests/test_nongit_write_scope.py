"""Write-scope enforcement in a worker directory that is not a Git work tree.

R109 (PM 1.43.2): a worker in a directory without Git wrote a test outside its
declared scope, and the gate reported PASS with no changed files. The Git
snapshot sees nothing outside Git, and the gate read that as "no writes".
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.adapters import ClaudeCodeAdapter
from puppetmaster.adapters._base import make_patch_artifact
from puppetmaster.adapters._git import git_snapshot, worktree_guard
from puppetmaster.adapters._prompts import build_cli_implement_prompt
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.local import ShellAdapter
from puppetmaster.gates import _gate_write_scope
from puppetmaster.models import Artifact, ArtifactType, Task

SCOPE = {"scope": ["src/allowed.py", "reports/*"]}


def _touch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run(root: Path, change) -> list[Artifact]:
    """Snapshot, let ``change`` act as the worker, snapshot again, emit the PATCH."""
    task = Task(job_id="j", role="build", instruction="i", payload={"cwd": str(root)})
    before = git_snapshot(root)
    time.sleep(0.01)
    change(root)
    after = git_snapshot(root, base_tree=None, base_manifest=before.get("manifest"))
    return [make_patch_artifact(task, "w", before, after, adapter="test", status="applied",
                                change="c", sidecar_name="t")]


class NonGitGateTests(unittest.TestCase):
    def test_unobserved_run_fails_instead_of_passing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "src" / "allowed.py", "x")
            _touch(root / "stray.txt", "outside")
            result = _gate_write_scope("scope", SCOPE, [], root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["observation"], "unavailable")

    def test_git_control_still_names_the_stray_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            _touch(root / "stray.txt", "outside")
            result = _gate_write_scope("scope", SCOPE, [], root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["out_of_scope"], ["stray.txt"])

    def test_addition_outside_scope_fails_by_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "src" / "allowed.py", "x")
            arts = _run(root, lambda r: (_touch(r / "src" / "allowed.py", "y"),
                                         _touch(r / "tests" / "test_other.py", "t")))
            result = _gate_write_scope("scope", SCOPE, arts, root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["out_of_scope"], ["tests/test_other.py"])

    def test_edit_delete_and_rename_outside_scope_fail_by_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("keep.py", "gone.py", "old.py"):
                _touch(root / name, "v1")

            def change(r):
                _touch(r / "keep.py", "v2 longer")
                (r / "gone.py").unlink()
                (r / "old.py").rename(r / "new.py")

            result = _gate_write_scope("scope", SCOPE, _run(root, change), root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["out_of_scope"], ["gone.py", "keep.py", "new.py", "old.py"])

    def test_in_scope_run_passes_with_its_changes_listed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "src" / "allowed.py", "x")
            _touch(root / "untouched.txt", "same")
            arts = _run(root, lambda r: (_touch(r / "src" / "allowed.py", "longer y"),
                                         _touch(r / "reports" / "r.md", "report")))
            result = _gate_write_scope("scope", SCOPE, arts, root)
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.detail["changed_files"], ["reports/r.md", "src/allowed.py"])

    def test_caches_and_the_live_state_dir_are_not_charged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "run-state"
            _touch(state / "state.sqlite3", "a")
            with patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": str(state)}):
                arts = _run(root, lambda r: (_touch(r / "run-state" / "state.sqlite3", "bb"),
                                             _touch(r / "src" / "__pycache__" / "a.pyc", "c"),
                                             _touch(r / "src" / "allowed.py", "x")))
            result = _gate_write_scope("scope", SCOPE, arts, root)
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.detail["changed_files"], ["src/allowed.py"])


class NonGitLaunchGuardTests(unittest.TestCase):
    def _task(self, root: Path, **payload) -> Task:
        return Task(job_id="j", role="build", instruction="i",
                    payload={"cwd": str(root), "allow_non_worktree": True, **payload})

    def test_scoped_task_refused_when_the_directory_cannot_be_observed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("puppetmaster.adapters._dir_manifest.MAX_FILES", 2):
                for i in range(4):
                    _touch(root / f"f{i}.txt", "x")
                before = git_snapshot(root)
            blocked = worktree_guard(self._task(root, write_scope=["a.py"]), "w", "codex", root, before)
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked[0].payload["failure"], "scope_unobservable")

    def test_unscoped_task_still_runs_without_git(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("puppetmaster.adapters._dir_manifest.MAX_FILES", 2):
                for i in range(4):
                    _touch(root / f"f{i}.txt", "x")
                before = git_snapshot(root)
            self.assertIsNone(worktree_guard(self._task(root), "w", "codex", root, before))

    def test_observable_scoped_task_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = git_snapshot(root)
            self.assertIsNone(worktree_guard(self._task(root, write_scope=["a.py"]), "w", "codex", root, before))


class NonGitAdapterRunTests(unittest.TestCase):
    """A real Claude Code adapter run: real snapshots, guard, and PATCH; only the CLI is fake."""

    RESULT = json.dumps({
        "type": "result", "subtype": "success", "is_error": False, "session_id": "s",
        "result": "Done.\nVERDICT: PASS - ok", "usage": {"input_tokens": 1, "output_tokens": 1},
    })

    def _run(self, root: Path, write) -> list[Artifact]:
        task = Task(job_id="j", role="builder", instruction="Build.", adapter="claude-code",
                    payload={"cwd": str(root), "allow_dirty": True, "allow_non_worktree": True,
                             "permission_mode": "acceptEdits", "output_format": "json",
                             "write_scope": ["tests/test_declared.py"]})

        def fake_cli(**kwargs):
            time.sleep(0.01)
            write(root)
            self.prompt = kwargs.get("stdin_data") or ""
            return StreamedProcess(returncode=0, stdout=self.RESULT, stderr="", timed_out=False)

        with TemporaryDirectory() as homes, patch.dict(os.environ, {
            "CLAUDE_CONFIG_DIR": str(Path(homes) / "claude"), "PUPPETMASTER_HOME": str(Path(homes) / "pm"),
        }), patch("puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"), patch(
            "puppetmaster.adapters.run_streamed_subprocess", side_effect=fake_cli,
        ), patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)):
            return ClaudeCodeAdapter().run(task, "goal", "worker")

    def test_r109_shape_fails_naming_the_undeclared_test(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "src" / "m.py", "x")
            arts = self._run(root, lambda r: _touch(r / "tests" / "test_other_name.py", "t"))
            result = _gate_write_scope("scope", {"scope": ["tests/test_declared.py"]}, arts, root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["out_of_scope"], ["tests/test_other_name.py"])
        self.assertIn("- tests/test_declared.py", self.prompt)

    def test_declared_write_passes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            arts = self._run(root, lambda r: _touch(r / "tests" / "test_declared.py", "t"))
            result = _gate_write_scope("scope", {"scope": ["tests/test_declared.py"]}, arts, root)
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.detail["changed_files"], ["tests/test_declared.py"])

    def test_no_change_run_passes_as_observed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            arts = self._run(root, lambda r: None)
            result = _gate_write_scope("scope", {"scope": ["tests/test_declared.py"]}, arts, root)
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(result.detail["changed_files"], [])


class ScopedShellNodeTests(unittest.TestCase):
    """A scoped shell node is a writer: its delta is observed too."""

    def _run(self, root: Path, code: str) -> list[Artifact]:
        task = Task(job_id="j", role="build", instruction="i", adapter="shell",
                    payload={"cwd": str(root), "command": [sys.executable, "-c", code],
                             "write_scope": ["x.py"]})
        return ShellAdapter().run(task, "goal", "w")

    def test_write_outside_scope_fails_by_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            arts = self._run(root, "open('x.py','w').write('1');open('other.txt','w').write('2')")
            result = _gate_write_scope("scope", {"scope": ["x.py"]}, arts, root)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["out_of_scope"], ["other.txt"])

    def test_unscoped_shell_check_takes_no_snapshot(self):
        task = Task(job_id="j", role="check", instruction="i", adapter="shell",
                    payload={"command": [sys.executable, "-c", "print(1)"]})
        with patch("puppetmaster.adapters.git_snapshot") as snap:
            arts = ShellAdapter().run(task, "goal", "w")
        snap.assert_not_called()
        self.assertEqual(len(arts), 1)


class ScopeInPromptTests(unittest.TestCase):
    def test_declared_paths_reach_the_worker(self):
        task = Task(job_id="j", role="build", instruction="Build it.",
                    payload={"write_scope": ["tests/test_t1.py", "reports/t1.md"]})
        prompt = build_cli_implement_prompt(task)
        self.assertIn("- tests/test_t1.py\n- reports/t1.md", prompt)
        self.assertLess(prompt.index("Write scope:"), prompt.index("Build it."))

    def test_unscoped_prompt_has_no_scope_section(self):
        prompt = build_cli_implement_prompt(Task(job_id="j", role="build", instruction="Build it."))
        self.assertNotIn("Write scope:", prompt)


if __name__ == "__main__":
    unittest.main()
