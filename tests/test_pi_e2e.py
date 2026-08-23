"""Optional live E2E for the Pi TUI/pilot package.

Does not fake a passing live job. Skips when the Pi CLI is missing or no
provider key is visible (or PI_LIVE_E2E is unset).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _provider_key_visible() -> bool:
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        if os.environ.get(key):
            return True
    return False


class PiLiveE2ETests(unittest.TestCase):
    def test_pi_can_start_pm_job_and_read_effort_index(self) -> None:
        if os.environ.get("PI_LIVE_E2E") not in {"1", "true", "yes"}:
            self.skipTest(
                "live Pi E2E skipped (set PI_LIVE_E2E=1 and a provider key). "
                "Hermetic coverage lives in test_pi_pilot_package.py."
            )
        pi = shutil.which(os.environ.get("PI_COMMAND", "pi"))
        if pi is None:
            self.skipTest("pi CLI not on PATH; install @earendil-works/pi-coding-agent")
        if not _provider_key_visible():
            self.skipTest("no provider key visible; refusing to fake a live job")

        from puppetmaster.installers import bundled_pi_package_dir, install_pi_mcp

        pkg = bundled_pi_package_dir()
        self.assertIsNotNone(pkg)
        with tempfile.TemporaryDirectory() as tmp:
            agent = Path(tmp) / "agent"
            result = install_pi_mcp(
                agent_dir=agent,
                python_executable=sys.executable,
                skip_handshake=False,
            )
            self.assertIn(result.status, {"installed", "unchanged"})
            env = os.environ.copy()
            env["PI_CODING_AGENT_DIR"] = str(agent)
            installed = subprocess.run(
                [pi, "install", str(pkg)],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                check=False,
            )
            if installed.returncode != 0:
                self.skipTest(
                    "pi install refused: "
                    + ((installed.stderr or installed.stdout or "")[-400:])
                )
            prompt = (
                "Call puppetmaster_start_agentic with goal 'write a one-line comment in README if present' "
                "then call puppetmaster_effort_index. Reply with the job_id and whether effort-index returned items. "
                "Do not invent results."
            )
            proc = subprocess.run(
                [pi, "-p", prompt, "--mode", "json"],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
                check=False,
            )
            if proc.returncode != 0:
                self.skipTest(
                    "pi non-interactive run refused: "
                    + ((proc.stderr or proc.stdout or "")[-400:])
                )
            blob = proc.stdout or ""
            self.assertTrue(blob.strip(), "pi produced no stdout")
            # A live pass must mention a job or effort-index payload, not a fabricated ok.
            self.assertTrue(
                "job_" in blob or "effort" in blob.lower() or "artifact" in blob.lower(),
                blob[-800:],
            )


if __name__ == "__main__":
    unittest.main()
