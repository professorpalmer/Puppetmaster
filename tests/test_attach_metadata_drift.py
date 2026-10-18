"""Metadata-only drift during a worker attach is respawned, not fatal.

Before: a ctime-only change to the store inside the attach binding (macOS sets
com.apple.provenance the first time a new app lineage writes a file; chmod to
the same mode does the same) raised StoreIdentityError("store source metadata
changed during binding"). The worker exited 1 before claiming its task and a
2-worker claude-code swarm on macOS failed one second after launch.

The fence itself is unchanged: from stat alone the drift is indistinguishable
from the file being renamed away and back mid-binding, so a binding still never
completes across it. The drift is typed (StoreMetadataDrift), the worker exits
WORKER_ATTACH_FAILED_EXIT, and the supervisor respawns the role.
"""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401

from puppetmaster import readonly
from puppetmaster.identity import StoreIdentityError, StoreMetadataDrift
from puppetmaster.sqlite_store import SQLiteSwarmStore, SqliteSchemaError
from puppetmaster.worker_runtime import WORKER_ATTACH_FAILED_EXIT, _transient_attach_failure, main as worker_main


def _drift_on_first_admit(target):
    original = readonly.ReadConnection._admit
    state = {"drifted": 0}

    def admit(self, path, deadline):
        result = original(self, path, deadline)
        if not state["drifted"]:
            state["drifted"] += 1
            os.chmod(target, os.stat(target).st_mode & 0o7777)  # ctime only
        return result
    return admit, state


@unittest.skipIf(os.name == "nt", "ctime-only drift via chmod is a POSIX behaviour")
class AttachMetadataDriftTests(unittest.TestCase):
    def test_ctime_only_drift_is_typed_as_metadata_drift(self) -> None:
        for name in ("state.sqlite3", ""):
            with self.subTest(target=name or "root"), TemporaryDirectory() as root:
                SQLiteSwarmStore(root).ensure_schema()
                admit, _ = _drift_on_first_admit(Path(root) / name if name else Path(root))
                with patch.object(readonly.ReadConnection, "_admit", admit):
                    with self.assertRaises(StoreMetadataDrift):
                        SQLiteSwarmStore(root).attach()

    def test_worker_exits_tempfail_on_drift_then_a_fresh_worker_attaches(self) -> None:
        with TemporaryDirectory() as root:
            SQLiteSwarmStore(root).ensure_schema()
            admit, state = _drift_on_first_admit(Path(root) / "state.sqlite3")
            argv = ["--state-dir", root, "--backend", "sqlite", "--job-id", "job_none",
                    "--role", "explore", "--worker-id", "w-1"]
            # worker main exports PUPPETMASTER_STATE_DIR; keep it out of later tests.
            with patch.object(readonly.ReadConnection, "_admit", admit), patch.dict(os.environ):
                self.assertEqual(worker_main(argv), WORKER_ATTACH_FAILED_EXIT)
                self.assertEqual(state["drifted"], 1)
                # The respawned worker binds cleanly (the drift has passed).
                fresh = SQLiteSwarmStore(root)
                fresh.attach()
                self.assertTrue(fresh._attached)

    def test_replacement_and_missing_schema_are_not_respawned(self) -> None:
        self.assertFalse(_transient_attach_failure(
            StoreIdentityError("store removed or replaced since selection; explicitly reopen")))
        self.assertFalse(_transient_attach_failure(SqliteSchemaError("no schema")))
        self.assertTrue(_transient_attach_failure(StoreMetadataDrift("drift")))
        unstable = OSError("source metadata did not stabilize")
        unstable.source_open_contention = True
        self.assertTrue(_transient_attach_failure(unstable))
        self.assertFalse(_transient_attach_failure(OSError("other")))
        import sqlite3
        self.assertTrue(_transient_attach_failure(sqlite3.OperationalError("database is locked")))


if __name__ == "__main__":
    unittest.main()
