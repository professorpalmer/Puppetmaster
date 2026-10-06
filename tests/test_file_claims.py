"""Hermetic coverage for durable cooperative file claims."""
from __future__ import annotations

import os
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import unittest

from puppetmaster.file_claims import (
    FileClaimConflict,
    FileClaimRegistry,
    default_file_claim_db_path,
)


class FileClaimRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "repo"
        self.root.mkdir()
        self._git(self.root, "init")
        self._git(self.root, "config", "user.email", "tests@example.invalid")
        self._git(self.root, "config", "user.name", "Tests")
        (self.root / "tracked.txt").write_text("base", encoding="utf-8")
        self._git(self.root, "add", "tracked.txt")
        self._git(self.root, "commit", "-m", "base")
        self.now = 1_700_000_000.0
        self.registry = FileClaimRegistry(Path(self.tempdir.name) / "claims.sqlite3", clock=lambda: self.now)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _git(directory: Path, *args: str) -> str:
        command = ["git", "-C", str(directory)]
        if args and args[0] == "init":
            command.extend(["-c", "init.defaultBranch=main"])
        command.extend(args)
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()

    def test_conflict_renew_and_expiry_use_last_renewal(self) -> None:
        claim = self.registry.acquire(self.root, "tracked.txt", "worker-a", 10)
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire(self.root, "tracked.txt", "worker-b", 10)
        self.now += 9
        self.assertTrue(self.registry.renew(self.root, "tracked.txt", claim.claim_id))
        self.now += 9
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire(self.root, "tracked.txt", "worker-b", 10)
        self.now += 2
        replacement = self.registry.acquire(self.root, "tracked.txt", "worker-b", 10)
        self.assertNotEqual(claim.claim_id, replacement.claim_id)
        self.assertFalse(self.registry.release(self.root, "tracked.txt", claim.claim_id))
        self.assertEqual(["acquired", "renewed", "expired", "acquired"],
                         [record.event for record in self.registry.audit_records()])

    def test_forced_steal_invalidates_old_fencing_token_and_audits_owners(self) -> None:
        old = self.registry.acquire(self.root, "tracked.txt", "worker-a", 60)
        new = self.registry.acquire(self.root, "tracked.txt", "worker-b", 60, force=True)
        self.assertNotEqual(old.claim_id, new.claim_id)
        self.assertFalse(self.registry.renew(self.root, "tracked.txt", old.claim_id))
        self.assertFalse(self.registry.release(self.root, "tracked.txt", old.claim_id))
        stolen = [record for record in self.registry.audit_records() if record.event == "stolen"][-1]
        self.assertEqual(("stolen", "worker-a", "worker-b"),
                         (stolen.event, stolen.old_owner, stolen.new_owner))

    def test_directory_claims_overlap_only_at_path_boundaries(self) -> None:
        directory = self.registry.acquire(self.root, "src", "worker-a", 60)
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire(self.root, "src/file", "worker-b", 60)
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire(self.root, "src", "worker-b", 60)
        self.registry.acquire(self.root, "src2/file", "worker-b", 60)
        self.assertTrue(self.registry.release(self.root, "src", directory.claim_id))

    def test_redundant_requested_descendants_are_deduplicated(self) -> None:
        claims = self.registry.acquire_many(self.root, ["src/file", "src", "src/other"], "worker-a", 60)
        self.assertEqual(["src"], [claim.path for claim in claims])

    def test_force_steals_all_overlapping_claims_then_acquires_exact_paths(self) -> None:
        parent = self.registry.acquire(self.root, "src", "worker-a", 60)
        sibling = self.registry.acquire(self.root, "src2", "worker-c", 60)
        replacement = self.registry.acquire_many(
            self.root, ["src/file", "src2/file"], "worker-b", 60, force=True
        )
        self.assertEqual(["src/file", "src2/file"], [claim.path for claim in replacement])
        self.assertFalse(self.registry.renew(self.root, "src", parent.claim_id))
        self.assertFalse(self.registry.renew(self.root, "src2", sibling.claim_id))
        stolen = [record for record in self.registry.audit_records() if record.event == "stolen"]
        self.assertEqual({"worker-a", "worker-c"}, {record.old_owner for record in stolen})

    def test_non_git_workspace_uses_canonical_root(self) -> None:
        workspace = Path(self.tempdir.name) / "plain"
        workspace.mkdir()
        registry = FileClaimRegistry(Path(self.tempdir.name) / "plain.sqlite3")
        claim = registry.acquire(workspace, "file.txt", "worker", 60)
        self.assertEqual("file.txt", claim.path)
        self.assertEqual(str(workspace.resolve()), claim.repo_identity)

    def test_repo_subdirectory_resolves_git_common_dir_from_candidate(self) -> None:
        subdirectory = self.root / "nested"
        subdirectory.mkdir()
        self.assertEqual(self.registry.repository_identity(self.root),
                         self.registry.repository_identity(subdirectory))

    def test_root_and_git_internal_claims(self) -> None:
        root_claim = self.registry.acquire(self.root, ".", "worker-a", 60)
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire(self.root, "tracked.txt", "worker-b", 60)
        self.assertTrue(self.registry.release(self.root, ".", root_claim.claim_id))
        with self.assertRaises(ValueError):
            self.registry.acquire(self.root, ".git/config", "worker-a", 60)

    def test_separate_registry_instances_share_sqlite_serialization(self) -> None:
        first = self.registry.acquire(self.root, "tracked.txt", "worker-a", 60)
        other = FileClaimRegistry(self.registry.db_path, clock=lambda: self.now)
        with self.assertRaises(FileClaimConflict):
            other.acquire(self.root, "tracked.txt", "worker-b", 60)
        self.assertTrue(other.release(self.root, "tracked.txt", first.claim_id))

    def test_multi_path_conflict_does_not_leave_partial_claim(self) -> None:
        (self.root / "other.txt").write_text("other", encoding="utf-8")
        self.registry.acquire(self.root, "tracked.txt", "worker-a", 60)
        with self.assertRaises(FileClaimConflict):
            self.registry.acquire_many(self.root, ["other.txt", "tracked.txt"], "worker-b", 60)
        acquired = [record for record in self.registry.audit_records() if record.event == "acquired"]
        self.assertEqual(["tracked.txt"], [record.path for record in acquired])
        self.assertEqual("worker-b", self.registry.acquire(self.root, "other.txt", "worker-b", 60).owner)

    def test_sibling_worktrees_have_independent_conflict_scopes(self) -> None:
        worktree = Path(self.tempdir.name) / "linked-worktree"
        self._git(self.root, "worktree", "add", "-b", "linked", str(worktree))
        first = self.registry.acquire(self.root, "tracked.txt", "worker-a", 60)
        second = self.registry.acquire(worktree, "tracked.txt", "worker-b", 60)
        self.assertNotEqual(first.repo_identity, second.repo_identity)
        self.assertTrue(self.registry.release(worktree, "tracked.txt", second.claim_id))
        self.assertEqual(default_file_claim_db_path(self.root), default_file_claim_db_path(worktree))

    def test_symlink_escape_is_rejected(self) -> None:
        outside = Path(self.tempdir.name) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        alias = self.root / "escape.txt"
        try:
            alias.symlink_to(outside)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.registry.acquire(self.root, "escape.txt", "worker-a", 60)

    def test_validation_and_lazy_expiry(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.acquire(self.root, "tracked.txt", "", 60)
        with self.assertRaises(ValueError):
            self.registry.acquire(self.root, "tracked.txt", "worker", 0)
        self.registry.acquire(self.root, "tracked.txt", "worker", 1)
        self.now += 1
        # Expiry is lazy: the next acquisition reaps and audits the stale row.
        self.registry.acquire(self.root, "tracked.txt", "worker-b", 60)
        events = [record.event for record in self.registry.audit_records()]
        self.assertIn("expired", events)


def _competing_claim(db_path: str, repo: str, owner: str, result: object) -> None:
    registry = FileClaimRegistry(db_path)
    try:
        registry.acquire(repo, "race.txt", owner, 60)
    except FileClaimConflict:
        result.put(False)
    else:
        result.put(True)


class FileClaimProcessTests(unittest.TestCase):
    def test_concurrent_processes_exactly_one_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            subprocess.check_call(["git", "-C", str(root), "-c", "init.defaultBranch=main", "init"], stderr=subprocess.DEVNULL)
            db = str(Path(directory) / "claims.sqlite3")
            context = multiprocessing.get_context("spawn")
            results = context.Queue()
            workers = [context.Process(target=_competing_claim,
                                       args=(db, str(root), "worker-%d" % i, results)) for i in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(10)
            self.assertTrue(all(worker.exitcode == 0 for worker in workers))
            self.assertEqual([True], [value for value in (results.get(), results.get()) if value])


if __name__ == "__main__":
    unittest.main()
