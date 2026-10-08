"""A helper killed past its teardown budget is reaped by the next sweep, not lost."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import logging
import subprocess
import threading
import time
import unittest

from puppetmaster.readonly import ReapDeferred, _Transport, _TransportState
from puppetmaster.readonly_cleanup import CleanupRegistry

# Ignores SIGTERM, as a helper that is slow to exit under load does.
_STUBBORN = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('r', flush=True); time.sleep(60)"


def _stubborn_transport() -> _Transport:
    transport = _Transport.__new__(_Transport)
    transport.state = _TransportState.OPEN
    transport.close_event = threading.Event()
    transport.reader_exited = threading.Event()
    transport.pid = os.getpid()
    transport.reader = None
    transport._reader_attempted = False
    transport.process = subprocess.Popen([sys.executable, "-c", _STUBBORN], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True)
    transport.process.stdout.readline()
    # SIGKILL can land before the post-kill wait(timeout=0) on a fast runner,
    # and then nothing is deferred. Time out the TERM wait and the KILL wait of
    # the first close. Later waits are real, so the next sweep reaps for real.
    real_wait, waits = transport.process.wait, []

    def wait(timeout=None):
        waits.append(timeout)
        if len(waits) <= 2:
            raise subprocess.TimeoutExpired("reader", timeout)
        return real_wait(timeout=timeout)

    transport.process.wait = wait
    return transport


@unittest.skipIf(os.name == "nt", "Windows terminate is TerminateProcess; nothing ignores it")
class ReapDeferredTests(unittest.TestCase):
    def test_a_spent_budget_defers_the_reap_and_the_next_sweep_reaps_it(self):
        registry = CleanupRegistry()
        transport = _stubborn_transport()
        self.addCleanup(transport.process.kill)
        token = registry.register(transport, ("store", 1))
        registry.retire(token)

        with self.assertRaises(ReapDeferred):
            transport.close(deadline=time.monotonic() + 0.05)
        registry.maintain(deadline=time.monotonic() + 0.05)
        self.assertTrue(transport.closed)
        self.assertIsNotNone(transport.process.poll())
        self.assertEqual(registry.owners, {})

    def test_a_deferred_reap_in_the_sweep_logs_at_debug_not_warning(self):
        registry = CleanupRegistry()
        transport = _stubborn_transport()
        self.addCleanup(transport.process.kill)
        token = registry.register(transport, ("store", 1))
        registry.retire(token)

        with self.assertLogs("puppetmaster.readonly_cleanup", level=logging.DEBUG) as logs:
            registry.maintain(deadline=time.monotonic() + 0.05)
        self.assertIn(token, registry.owners)
        self.assertEqual([r.levelno for r in logs.records], [logging.DEBUG])
        self.assertIn("reader teardown timed out", logs.output[0])

        registry.maintain(deadline=time.monotonic() + 0.05)
        self.assertEqual(registry.owners, {})
        self.assertIsNotNone(transport.process.poll())


if __name__ == "__main__":
    unittest.main()
