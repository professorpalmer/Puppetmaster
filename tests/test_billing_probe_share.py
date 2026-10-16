"""One ``codex login status`` probe serves every PM process within the TTL.

Codex R138: each worker process ran ``<codex> login status`` before launch.
The benchmark's Codex command is a wrapper that verifies itself on each
start, so the probe cost seconds on every node visit.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster import platform_billing


class SharedLoginProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {"CODEX_COMMAND": str(self.tmp / "codex-wrapper"), "HOME": str(self.tmp),
                    "CODEX_HOME": str(self.tmp / "codex-home")}
        patcher = patch.dict(os.environ, {"PUPPETMASTER_APP_STATE_ROOT": str(self.tmp / "state")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

    def probe(self, reply):
        def runner(command, env=None):
            self.calls.append(command)
            return reply
        with patch.object(platform_billing, "_default_runner", side_effect=runner):
            return platform_billing.detect_codex_billing(env=self.env)

    def test_a_second_process_reuses_a_healthy_probe(self):
        first = self.probe((0, "Logged in using ChatGPT", ""))
        second = self.probe((0, "Logged in using ChatGPT", ""))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((second.billing, second.healthy), ("plan", True))
        self.assertEqual(second.evidence, first.evidence)

    def test_a_failed_probe_is_never_shared(self):
        self.probe((1, "Not logged in", ""))
        status = self.probe((0, "Logged in using ChatGPT", ""))
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(status.healthy)

    def test_an_expired_probe_runs_again(self):
        self.probe((0, "Logged in using ChatGPT", ""))
        with patch.object(platform_billing.time, "time", return_value=platform_billing.time.time() + 301):
            self.probe((0, "Logged in using ChatGPT", ""))
        self.assertEqual(len(self.calls), 2)

    def test_ttl_zero_turns_sharing_off(self):
        with patch.dict(os.environ, {"PUPPETMASTER_BILLING_TTL_SECONDS": "0"}):
            self.probe((0, "Logged in using ChatGPT", ""))
            self.probe((0, "Logged in using ChatGPT", ""))
        self.assertEqual(len(self.calls), 2)

    def test_another_command_probes_for_itself(self):
        self.probe((0, "Logged in using ChatGPT", ""))
        self.env["CODEX_COMMAND"] = str(self.tmp / "other-wrapper")
        status = self.probe((0, "Logged in using an API key", ""))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(status.billing, "api")

    def test_the_record_holds_no_raw_output(self):
        self.probe((0, "Logged in using ChatGPT - person@example.com", ""))
        records = list((self.tmp / "state" / "billing-probes").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertNotIn("person@example.com", records[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
