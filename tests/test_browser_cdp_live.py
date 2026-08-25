"""Live Chrome attach/reuse for auth handoff. Skipped in CI.

Run: PM_BROWSER_LIVE=1 python -m unittest tests.test_browser_cdp_live
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import unittest
import urllib.request

from puppetmaster import browser_cdp as b


def _port_open(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/json/version" % int(port), timeout=2
        ) as r:
            json.loads(r.read().decode())
        return True
    except Exception:
        return False


@unittest.skipUnless(
    os.environ.get("PM_BROWSER_LIVE", "").strip() in ("1", "true", "yes"),
    "set PM_BROWSER_LIVE=1 to launch a real Chrome",
)
class BrowserCdpLiveWorkflowTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "PM_BROWSER_HEADED",
                "PM_BROWSER_USER_DATA_DIR",
                "PM_BROWSER_CDP_PORT",
                "PM_BROWSER_ATTACH_ONLY",
                "PM_BROWSER_ALLOW_LOCAL",
            )
        }
        self._janitor = b._JANITOR
        self.profile = tempfile.mkdtemp(prefix="pm-auth-wf-")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        os.environ["PM_BROWSER_USER_DATA_DIR"] = self.profile
        os.environ["PM_BROWSER_CDP_PORT"] = str(self.port)
        os.environ.pop("PM_BROWSER_ATTACH_ONLY", None)
        os.environ.pop("PM_BROWSER_ALLOW_LOCAL", None)
        b.set_janitor(False)
        b.reset_session(keep_profile=True)

    def tearDown(self):
        try:
            b.set_janitor(True)
            b.reset_session(keep_profile=True)
        except Exception:
            pass
        b.set_janitor(self._janitor)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.profile, ignore_errors=True)

    def test_headed_handoff_then_worker_attach(self):
        if not b._find_chrome():
            self.skipTest("Chrome/Chromium not found")
        out = b.auth_handoff("https://example.com")
        self.assertNotIn("Set-Cookie", out)
        self.assertNotIn("cookie=", out.lower())
        self.assertIn("Do not paste passwords or cookies", out)
        self.assertIn("Auth handoff", out)
        self.assertTrue(
            "example.com" in out.lower() or "Navigated" in out,
            out,
        )
        info = b.session_info()
        self.assertTrue(info["connected"], info)
        self.assertTrue(info["headed"], info)
        self.assertEqual(info["port"], self.port)
        self.assertTrue(info["owns_proc"], info)
        self.assertTrue(info["persistent"], info)
        self.assertTrue(_port_open(self.port))

        worker = b._Session()
        err = worker.ensure()
        self.assertIsNone(err, err)
        self.assertFalse(worker.owns_proc)
        self.assertEqual(worker.port, self.port)

        worker.shutdown()
        self.assertTrue(_port_open(self.port), "worker shutdown must not reap shared Chrome")
        self.assertTrue(os.path.isdir(self.profile))

        b.set_janitor(True)
        b.reset_session(keep_profile=True)
        self.assertFalse(_port_open(self.port), "janitor reset must close owned Chrome")
        self.assertTrue(os.path.isdir(self.profile), "durable profile must survive reset")


if __name__ == "__main__":
    unittest.main()
