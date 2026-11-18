from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from puppetmaster import worker_resume
from puppetmaster.models import ArtifactType


def _receipt(created_at: str, **payload):
    return SimpleNamespace(
        task_id="t1", type=ArtifactType.VERIFICATION, created_at=created_at, payload=payload
    )


class _Store:
    def __init__(self, task, artifacts):
        self.task = task
        self.artifacts = artifacts

    def get_job(self, job_id):
        return object()

    def list_tasks(self, job_id):
        return [self.task]

    def list_artifacts(self, job_id):
        return self.artifacts


class ResolveSkipsReceiptWithoutSessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = self._tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = self._old
        self._tmp.cleanup()

    def _resolve(self, artifacts, adapter="claude-code"):
        task = SimpleNamespace(id="t1", role="builder", adapter=adapter, updated_at="2", created_at="1")
        return worker_resume._resolve(
            _Store(task, artifacts), {"resume_from": {"job_id": "j", "role": "builder"}}, adapter
        )

    def test_newer_receipt_without_session_does_not_hide_older_session(self):
        record = self._resolve([
            _receipt("1", adapter="claude-code", session_id="abc123"),
            _receipt("2", adapter="claude-code"),
        ])
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["session_id"], "abc123")
        self.assertEqual(record["from_task_id"], "t1")

    def test_newest_session_wins_when_several_have_one(self):
        record = self._resolve([
            _receipt("1", adapter="claude-code", session_id="old1"),
            _receipt("2", adapter="claude-code", session_id="new2"),
            _receipt("3", adapter="claude-code"),
        ])
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["session_id"], "new2")

    def test_no_receipt_with_session_stays_unavailable(self):
        record = self._resolve([_receipt("1", adapter="claude-code")])
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["reason"], "prior task 't1' recorded no session_id")

    def test_ephemeral_codex_receipt_with_thread_stays_unavailable(self):
        record = self._resolve(
            [
                _receipt("1", adapter="codex", thread_id="th1", ephemeral=True),
                _receipt("2", adapter="codex"),
            ],
            adapter="codex",
        )
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("ran ephemeral", record["reason"])


if __name__ == "__main__":
    unittest.main()
