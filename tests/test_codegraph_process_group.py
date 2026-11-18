"""A CodeGraph command that times out must not leave a child process alive."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from puppetmaster import codegraph

# The shim starts a child that sleeps, then waits for it. This is the shape of the
# npm shim and the real Node binary.
SHIM = """
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)", sys.argv[1]])
child.wait()
"""


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A killed child that nobody waited for can stay as a zombie for a moment.
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(out) and not out.startswith("Z")


class CodegraphProcessGroupTests(unittest.TestCase):
    def test_timeout_ends_the_child_of_the_shim(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            with self.assertRaises(subprocess.TimeoutExpired):
                codegraph._run_cli(
                    [sys.executable, "-c", SHIM, str(pid_file)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
            self.assertTrue(pid_file.exists(), "the child did not start before the timeout")
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 10
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(_alive(pid), "the child of the shim is still alive after the timeout")

    def test_a_command_that_ends_in_time_returns_its_output(self):
        completed = codegraph._run_cli(
            [sys.executable, "-c", "print('ok')"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.strip(), "ok")

    def test_check_raises_on_a_failed_command(self):
        with self.assertRaises(subprocess.CalledProcessError):
            codegraph._run_cli([sys.executable, "-c", "raise SystemExit(3)"], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
