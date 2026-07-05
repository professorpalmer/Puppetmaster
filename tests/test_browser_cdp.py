"""Hermetic tests for the stdlib CDP browser engine (no real Chrome/network).

Written as unittest.TestCase so `python -m unittest discover -s tests` (the CI
runner) collects them. A fake session replaces the module singleton so
navigate/snapshot/click/type/dispatch return deterministic strings, and unsafe
or internal URLs are asserted refused. No pytest fixtures (monkeypatch/tmp_path)
are used, since unittest discovery does not provide them.
"""
import base64
import os
import tempfile
import unittest

import puppetmaster.browser_cdp as b


class _FakeSession:
    def __init__(self):
        self.calls = []
        self._page = {"title": "Test Page", "href": "https://example.com/",
                      "ready": "complete", "text": "hello world"}

    def ensure(self):
        return None  # pretend Chrome is up

    def _cmd(self, method, params=None, timeout=30.0):
        self.calls.append((method, params))
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(b"PNGDATA").decode()}
        return {}

    def _eval(self, expr):
        if "querySelectorAll" in expr:  # the snapshot walker
            return '@e1 a "Learn more"'
        # click()/type_text() wrap a data-pm-ref querySelector in an IIFE
        # (checked AFTER the snapshot walker, whose JS also mentions the attr).
        if "data-pm-ref" in expr:
            return True
        if "document.readyState" in expr:
            return "complete"
        if "document.title" in expr:
            return self._page["title"]
        if "innerText" in expr or "body.innerText" in expr:
            return self._page["text"]
        if "location.href" in expr or "href" in expr:
            return self._page["href"]
        return None

    def shutdown(self):
        pass


class BrowserCdpTest(unittest.TestCase):
    def setUp(self):
        self._saved_session = getattr(b, "_SESSION", None)
        self.fake = _FakeSession()
        b._SESSION = self.fake
        self._saved_allow_local = os.environ.pop("PM_BROWSER_ALLOW_LOCAL", None)

    def tearDown(self):
        b._SESSION = self._saved_session
        if self._saved_allow_local is not None:
            os.environ["PM_BROWSER_ALLOW_LOCAL"] = self._saved_allow_local
        else:
            os.environ.pop("PM_BROWSER_ALLOW_LOCAL", None)

    def test_navigate_returns_title_and_url(self):
        out = b.navigate("https://example.com")
        self.assertIn("Navigated to https://example.com/", out)
        self.assertIn("Test Page", out)

    def test_navigate_refuses_internal_url(self):
        out = b.navigate("http://169.254.169.254/latest/meta-data/")
        self.assertIn("Refused to navigate", out)

    def test_navigate_refuses_localhost(self):
        out = b.navigate("http://localhost:8080/admin")
        self.assertIn("Refused to navigate", out)

    def test_snapshot_lists_refs(self):
        out = b.snapshot()
        self.assertIn("@e1", out)
        self.assertIn("Learn more", out)

    def test_click_and_type(self):
        self.assertIn("Clicked @e1", b.click("@e1"))
        self.assertIn("Typed into @e2", b.type_text("@e2", "hello"))

    def test_get_text(self):
        self.assertIn("hello world", b.get_text())

    def test_screenshot_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            out = b.screenshot(out_dir=td)
            self.assertIn("Saved screenshot to", out)
            self.assertIn(td, out)

    def test_dispatch_routes_known_and_unknown(self):
        self.assertIsNotNone(b.dispatch("browser_get_text", {}))
        self.assertIsNone(b.dispatch("not_a_browser_tool", {}))

    def test_chrome_not_found_message(self):
        saved_find = b._find_chrome
        try:
            b._find_chrome = lambda: None
            b._SESSION = b._Session()
            out = b.navigate("https://example.com")
            self.assertIn("Chrome/Chromium not found", out)
        finally:
            b._find_chrome = saved_find


if __name__ == "__main__":
    unittest.main()
