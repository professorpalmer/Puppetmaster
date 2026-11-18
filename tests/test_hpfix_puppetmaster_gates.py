"""Regression tests for write_scope attribution path matching.

Event paths are put in repo root form when they are recorded, and the gate
compares exact paths (bugs 56e0b53a-1 and 56e0b53a-4).
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.gates import _gate_write_scope
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.worker_attribution import (
    RepoFrame,
    WorkerReferences,
    claude_tool_use_references,
    codex_event_references,
    path_is_referenced,
    path_was_dirty_before,
)


def _patch(files, referenced, baseline):
    return Artifact(
        job_id="j", task_id="t", type=ArtifactType.PATCH, created_by="w",
        confidence=1.0, evidence=[],
        payload={
            "worker_delta_attributed": True,
            "files": files,
            "worker_referenced_paths": referenced,
            "baseline_dirty_paths": baseline,
        },
    )


def _gate(files, referenced, baseline):
    with TemporaryDirectory() as tmp:
        result = _gate_write_scope(
            "ws", {"scope": ["src/*"]}, [_patch(files, referenced, baseline)], Path(tmp)
        )
    detail = result.detail or {}
    return result.passed, detail.get("concurrent_changes"), detail.get("out_of_scope")


def _claude_edits(*paths):
    return json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": path}} for path in paths
    ]}})


class ReferencedPathMatchTests(unittest.TestCase):
    """Bug 56e0b53a-1: a root path does not name a file in a subdirectory."""

    def test_root_reference_does_not_claim_pre_dirty_subdir_file(self) -> None:
        passed, concurrent, out = _gate(["docs/README.md"], ["README.md"], ["docs/README.md"])
        self.assertTrue(passed)
        self.assertEqual(concurrent, ["docs/README.md"])
        self.assertIsNone(out)

    def test_subdir_reference_does_not_claim_pre_dirty_root_file(self) -> None:
        passed, concurrent, out = _gate(["README.md"], ["docs/README.md"], ["README.md"])
        self.assertTrue(passed)
        self.assertEqual(concurrent, ["README.md"])
        self.assertIsNone(out)

    def test_exact_reference_is_still_charged(self) -> None:
        passed, concurrent, out = _gate(
            ["docs/README.md"], ["docs/README.md"], ["docs/README.md"]
        )
        self.assertFalse(passed)
        self.assertEqual(out, ["docs/README.md"])


class BaselineDirtyMatchTests(unittest.TestCase):
    """Bug 56e0b53a-4: a dirty root file does not excuse a subdirectory file."""

    def test_dirty_root_file_does_not_excuse_new_subdir_file(self) -> None:
        passed, concurrent, out = _gate(["docs/README.md"], [], ["README.md"])
        self.assertFalse(passed)
        self.assertEqual(out, ["docs/README.md"])
        self.assertIsNone(concurrent)

    def test_dirty_subdir_file_does_not_excuse_new_root_file(self) -> None:
        passed, concurrent, out = _gate(["README.md"], [], ["docs/README.md"])
        self.assertFalse(passed)
        self.assertEqual(out, ["README.md"])
        self.assertIsNone(concurrent)

    def test_exact_and_directory_baseline_still_excuse(self) -> None:
        self.assertTrue(path_was_dirty_before("docs/README.md", ["docs/README.md"]))
        self.assertTrue(path_was_dirty_before("docs/README.md", ["docs/"]))
        self.assertFalse(path_was_dirty_before("pkg/x.py", ["x.py"]))
        passed, concurrent, _ = _gate(["docs/README.md"], [], ["docs/README.md"])
        self.assertTrue(passed)
        self.assertEqual(concurrent, ["docs/README.md"])


class SubdirectoryWorkerTests(unittest.TestCase):
    """A worker in a repo subdirectory names its files relative to its cwd."""

    def test_subdir_worker_edit_is_charged_to_the_worker(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / ".git").mkdir()
            (repo / "pkg").mkdir()
            cwd = repo / "pkg"
            refs = claude_tool_use_references(_claude_edits(str(cwd / "x.py"), "y.py"), cwd)
            self.assertEqual(refs.paths, ["pkg/x.py", "pkg/y.py"])
            self.assertTrue(path_is_referenced("pkg/x.py", refs))
            self.assertFalse(path_is_referenced("x.py", refs))
            passed, concurrent, out = _gate(["pkg/x.py"], refs.paths, ["pkg/x.py"])
        self.assertFalse(passed)
        self.assertEqual(out, ["pkg/x.py"])
        self.assertIsNone(concurrent)

    def test_codex_command_token_in_subdir(self) -> None:
        events = [{"type": "item.completed", "item": {
            "type": "command_execution", "command": "echo x >> ../logs/run.txt",
        }}]
        refs = codex_event_references(events, Path("/repo/pkg"), Path("/repo"))
        self.assertEqual(refs.paths, ["logs/run.txt"])

    def test_top_level_is_found_from_a_nested_cwd(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / ".git").write_text("gitdir: elsewhere\n")
            nested = repo / "a" / "b"
            nested.mkdir(parents=True)
            frame = RepoFrame(nested)
            self.assertEqual(frame.canonical("c.py"), "a/b/c.py")


class CanonicalPathTests(unittest.TestCase):
    """The repo root form of an event path."""

    def setUp(self) -> None:
        self.frame = RepoFrame(Path("/repo/pkg"), Path("/repo"))

    def test_relative_path_joins_the_cwd(self) -> None:
        self.assertEqual(self.frame.canonical("x.py"), "pkg/x.py")

    def test_absolute_path_below_the_top_level(self) -> None:
        self.assertEqual(self.frame.canonical("/repo/docs/README.md"), "docs/README.md")

    def test_dot_and_dotdot_segments(self) -> None:
        self.assertEqual(self.frame.canonical("./x.py"), "pkg/x.py")
        self.assertEqual(self.frame.canonical("./sub/../x.py"), "pkg/x.py")
        self.assertEqual(self.frame.canonical("../lib/y.py"), "lib/y.py")
        self.assertEqual(self.frame.canonical("sub/"), "pkg/sub")

    def test_path_outside_the_repository_stays_absolute(self) -> None:
        self.assertEqual(self.frame.canonical("/etc/hosts"), "/etc/hosts")
        self.assertEqual(self.frame.canonical("../../other/z.py"), "/other/z.py")
        self.assertEqual(self.frame.canonical("/repository/x.py"), "/repository/x.py")
        self.assertEqual(self.frame.canonical("/repo"), "")
        refs = WorkerReferences([self.frame.canonical("/etc/hosts")])
        self.assertFalse(path_is_referenced("etc/hosts", refs))
        self.assertFalse(path_is_referenced("hosts", refs))

    def test_windows_separators(self) -> None:
        frame = RepoFrame(Path("C:/repo/pkg"), Path("C:/repo"))
        self.assertEqual(frame.canonical("sub\\x.py"), "pkg/sub/x.py")
        self.assertEqual(frame.canonical("C:\\repo\\docs\\README.md"), "docs/README.md")
        self.assertEqual(frame.canonical("..\\lib\\y.py"), "lib/y.py")
        self.assertEqual(frame.canonical("D:\\repo\\x.py"), "D:/repo/x.py")

    def test_linked_folder_above_an_absent_file(self) -> None:
        with TemporaryDirectory() as tmp:
            real = Path(tmp).resolve() / "real"
            (real / "pkg").mkdir(parents=True)
            link = Path(tmp).resolve() / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("no symlinks on this host")
            frame = RepoFrame(real / "pkg", real)
            self.assertEqual(frame.canonical(str(link / "pkg" / "absent.py")), "pkg/absent.py")
            self.assertEqual(frame.canonical(str(link / "new" / "absent.py")), "new/absent.py")

    def test_without_git_the_cwd_is_the_top_level(self) -> None:
        with TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            self.assertEqual(RepoFrame(cwd).canonical("out/a.txt"), "out/a.txt")
            self.assertEqual(RepoFrame(cwd).canonical(str(cwd / "b.txt")), "b.txt")


class DistinctPathsNeverMatchTests(unittest.TestCase):

    def test_distinct_canonical_paths_never_match(self) -> None:
        rng = random.Random(305)
        parts = ["a", "b", "docs", "README.md", "x.py", "pkg", "src"]
        paths = sorted({
            "/".join(rng.choice(parts) for _ in range(rng.randint(1, 4)))
            for _ in range(400)
        })
        frame = RepoFrame(Path("/repo"), Path("/repo"))
        for left in paths:
            self.assertEqual(frame.canonical(left), left)
            refs = WorkerReferences([left])
            for right in paths:
                self.assertEqual(path_is_referenced(right, refs), left == right)
                below = right.startswith(left + "/")
                self.assertEqual(path_was_dirty_before(right, [left]), left == right or below)


if __name__ == "__main__":
    unittest.main()
