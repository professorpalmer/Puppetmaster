from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from puppetmaster.edit_admission import EditAdmissionTimeout, edit_admission
from puppetmaster.file_claims import FileClaimRegistry
from puppetmaster.adapters.registry import ADAPTERS


@dataclass
class Task:
    job_id: str = "job-1"
    id: str = "task-1"
    adapter: str = "shell"
    payload: dict = None
    generation: int = 3
    lease_id: str = "lease-1"

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


class Store:
    def __init__(self):
        self.events = []

    def emit(self, job_id, event, payload):
        self.events.append((job_id, event, payload))


class EditAdmissionTests(unittest.TestCase):
    def test_declared_scope_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "claims.sqlite3"
            store = Store()
            task = Task(payload={"cwd": str(root), "write_scope": ["src/a", "src", "src/b"]})
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                with edit_admission(store, task, "worker") as owner:
                    self.assertEqual(("src",), tuple(c.path for c in owner.claims))
                    self.assertTrue(owner.check())
                    self.assertEqual(task.generation, owner.generation)
                self.assertEqual([], FileClaimRegistry(db).list_active(root))
            self.assertEqual("edit_admission.released", store.events[-1][1])

    def test_read_only_matrix_does_not_claim_and_unknown_is_conservative(self):
        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                for adapter, payload, expected in [
                    ("local", {}, 0),
                    ("openai", {}, 0),
                    ("shell", {"read_only": True}, 1),
                    ("codex", {"sandbox": "read-only", "dangerously_bypass_approvals_and_sandbox": True}, 1),
                    ("claude-code", {"permission_mode": "plan"}, 0),
                    ("claude-code", {"permission_mode": "acceptEdits"}, 1),
                    ("agy", {"mode": "plan"}, 0),
                    ("agy", {"mode": "accept-edits"}, 1),
                    ("future-adapter", {}, 1),
                    # Regression: the adapter Marionette pins for run_swarm was
                    # the one row missing here, so "read-only does not claim"
                    # was never asserted for the adapter every real analysis
                    # swarm uses. Read-only agentic workers must take 0 claims.
                    ("agentic", {"read_only": True, "no_edit": True, "dry_run": True}, 0),
                    ("agentic", {"read_only": True}, 0),
                    ("agentic", {"sandbox": "read-only"}, 0),
                    ("cursor", {"read_only": True}, 0),
                    ("hermes", {"no_edit": True}, 0),
                    ("agentic", {}, 1),
                    ("agentic", {"mode": "implement"}, 1),
                ]:
                    task = Task(adapter=adapter, payload={"cwd": str(root), **payload})
                    with edit_admission(Store(), task, "worker") as owner:
                        self.assertEqual(expected, len(owner.claims), adapter)
                # Keep this test coupled to the real registry: adding an
                # adapter cannot silently escape the admission matrix.
                for adapter in ADAPTERS:
                    task = Task(adapter=adapter, payload={"cwd": str(root)})
                    with edit_admission(Store(), task, "worker") as owner:
                        self.assertIn(len(owner.claims), (0, 1), adapter)

    def test_managed_claim_renews_past_ttl_and_conflict_wait_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            task = Task(payload={"cwd": str(root), "edit_claim_ttl_seconds": 0.1})
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                first = edit_admission(Store(), task, "one")
                try:
                    time.sleep(0.25)
                    self.assertTrue(first.check())
                    other = Task(id="task-2", payload={"cwd": str(root), "edit_admission_wait_seconds": 0.1})
                    with self.assertRaises(EditAdmissionTimeout):
                        edit_admission(Store(), other, "two")
                finally:
                    first.close()

    def test_empty_write_scope_claims_nothing(self):
        """An explicit [] means "writes nothing", not "the whole workspace"."""
        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                task = Task(payload={"cwd": str(root), "write_scope": []})
                with edit_admission(Store(), task, "worker") as owner:
                    self.assertEqual(0, len(owner.claims))
                # control: an ABSENT scope still means the whole workspace
                with edit_admission(Store(), Task(payload={"cwd": str(root)}), "worker") as owner:
                    self.assertEqual((".",), tuple(c.path for c in owner.claims))

    def test_timeout_names_the_holder_and_the_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                holder = edit_admission(Store(), Task(payload={"cwd": str(root)}), "holder")
                try:
                    other = Task(id="task-2", payload={
                        "cwd": str(root), "edit_admission_wait_seconds": 0.1})
                    with self.assertRaises(EditAdmissionTimeout) as ctx:
                        edit_admission(Store(), other, "waiter")
                    self.assertIn("holder", str(ctx.exception))
                    self.assertIn("waited", str(ctx.exception))
                finally:
                    holder.close()

    def test_admission_reports_how_long_it_queued(self):
        import threading

        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                with edit_admission(Store(), Task(payload={"cwd": str(root)}), "solo") as owner:
                    self.assertLess(owner.waited_seconds, 0.2)
                holder = edit_admission(Store(), Task(payload={"cwd": str(root)}), "holder")
                threading.Timer(0.3, holder.close).start()
                store = Store()
                waiter = Task(id="task-2", payload={"cwd": str(root), "edit_admission_wait_seconds": 5})
                with edit_admission(store, waiter, "waiter") as owner:
                    self.assertGreaterEqual(owner.waited_seconds, 0.25)
                acquired = [p for _, event, p in store.events if event == "edit_admission.acquired"]
                self.assertEqual(acquired[0]["waited_seconds"], owner.waited_seconds)

    def test_worker_receipts_carry_the_admission_wait(self):
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.worker_runtime import _with_admission_wait

        receipt = Artifact(job_id="j", task_id="t", type=ArtifactType.VERIFICATION, created_by="w",
                           confidence=0.9, evidence=["adapter:codex"], payload={"check": "c", "result": "passed"})
        finding = Artifact(job_id="j", task_id="t", type=ArtifactType.FINDING, created_by="w",
                           confidence=0.9, evidence=["a.py:1"], payload={"claim": "x"})
        stamped_receipt, stamped_finding = _with_admission_wait([receipt, finding], 12.5)
        self.assertEqual(stamped_receipt.payload["edit_admission_wait_seconds"], 12.5)
        self.assertEqual(stamped_receipt.payload["result"], "passed")
        self.assertIs(stamped_finding, finding)

    def test_default_wait_outlasts_the_adapter_wall_timeout(self):
        """A holder fences its claim for its WHOLE run, so the default wait has
        to cover that or a waiter fails while the holder is still working."""
        import puppetmaster.edit_admission as admission
        from puppetmaster.adapters.agentic import DEFAULT_IMPLEMENT_TIMEOUT_SECONDS
        self.assertGreaterEqual(
            admission.DEFAULT_ADMISSION_WAIT_SECONDS, DEFAULT_IMPLEMENT_TIMEOUT_SECONDS)

    def test_exception_releases_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root, db = Path(directory), Path(directory) / "claims.sqlite3"
            task = Task(payload={"cwd": str(root)})
            with patch("puppetmaster.edit_admission.default_file_claim_db_path", return_value=db):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    with edit_admission(Store(), task, "worker"):
                        raise RuntimeError("boom")
                self.assertEqual([], FileClaimRegistry(db).list_active(root))


if __name__ == "__main__":
    unittest.main()
