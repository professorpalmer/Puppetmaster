"""Process identity: a pid that now names a later process is not the recorded one."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import subprocess
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.interprocess_lock import InterProcessFileLock
from puppetmaster.proc_identity import own_identity, pid_reused, process_identity

SUPPORTED = sys.platform.startswith("linux") or sys.platform == "darwin" or os.name == "nt"


def sleeper(test: unittest.TestCase) -> subprocess.Popen:
    """A long-lived child that a background thread reaps the moment it dies."""
    kwargs = {"creationflags": 0x00000200} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
    reaper = threading.Thread(target=process.wait, daemon=True)
    reaper.start()

    def cleanup() -> None:
        if process.returncode is None:
            try:
                process.kill()
            except OSError:
                pass
        reaper.join(5)

    test.addCleanup(cleanup)
    return process


@unittest.skipUnless(SUPPORTED, "no process identity on this platform")
class ProcessIdentityTests(unittest.TestCase):
    def test_identity_is_stable_and_tells_processes_apart(self):
        mine = own_identity()
        self.assertIsNotNone(mine)
        self.assertEqual(process_identity(os.getpid()), mine)
        child = sleeper(self)
        theirs = process_identity(child.pid)
        self.assertIsNotNone(theirs)
        self.assertNotEqual(theirs, mine)
        self.assertEqual(process_identity(child.pid), theirs)
        self.assertTrue(pid_reused(child.pid, mine))
        self.assertFalse(pid_reused(child.pid, theirs))
        self.assertFalse(pid_reused(child.pid, None))

    def test_impossible_pids_have_no_identity(self):
        for pid in (0, -1, "12", None):
            self.assertIsNone(process_identity(pid))

    @unittest.skipIf(os.name == "nt", "an open handle keeps an exited Windows process queryable")
    def test_a_reaped_process_has_no_identity(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self.assertIsNone(process_identity(child.pid))


@unittest.skipUnless(SUPPORTED, "no process identity on this platform")
class LockOwnerIdentityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.lock = InterProcessFileLock.for_target(Path(self._tmp.name) / "thing", timeout=0)

    def plant(self, pid: int, proc) -> None:
        self.lock.path.write_text(json.dumps({"pid": pid, "created_at": time.time(), "token": "x",
                                              "proc": proc}), encoding="utf-8")

    def test_locks_record_their_owner(self):
        with self.lock:
            data = json.loads(self.lock.path.read_text(encoding="utf-8"))
        self.assertEqual(data["proc"], own_identity())

    def test_a_lock_whose_pid_now_names_another_process_is_recovered_at_once(self):
        child = sleeper(self)
        self.plant(child.pid, "an-owner-that-died")
        with self.lock:  # timeout 0: recovered on the first look, not the next poll
            pass

    def test_a_lock_held_by_the_recorded_process_is_kept(self):
        child = sleeper(self)
        self.plant(child.pid, process_identity(child.pid))
        with self.assertRaises(TimeoutError):
            self.lock.acquire()

    def test_a_lock_without_an_identity_trusts_a_live_pid(self):
        child = sleeper(self)
        self.plant(child.pid, None)
        with self.assertRaises(TimeoutError):
            self.lock.acquire()


if __name__ == "__main__":
    unittest.main()
