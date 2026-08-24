"""Hermetic tests for the stdlib CDP browser engine (no real Chrome/network).

Written as unittest.TestCase so `python -m unittest discover -s tests` (the CI
runner) collects them. A fake session replaces the module singleton so
navigate/snapshot/click/type/dispatch return deterministic strings, and unsafe
or internal URLs are asserted refused. No pytest fixtures (monkeypatch/tmp_path)
are used, since unittest discovery does not provide them.
"""
import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

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
        if "__pmNet" in expr or "JSON.stringify" in expr:
            return self._page.get("network", "[]")
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
        self.assertIsNotNone(b.dispatch("browser_network", {}))
        self.assertIsNone(b.dispatch("not_a_browser_tool", {}))

    def test_type_uses_native_value_setter(self):
        exprs = []
        orig = self.fake._eval

        def capture(expr):
            exprs.append(expr)
            return orig(expr)

        self.fake._eval = capture
        self.assertIn("Typed into @e2", b.type_text("@e2", "hello"))
        joined = "\n".join(exprs)
        self.assertIn("getOwnPropertyDescriptor", joined)
        self.assertIn("input", joined)
        self.assertIn("change", joined)

    def test_click_uses_real_mouse_events(self):
        exprs = []
        orig = self.fake._eval

        def capture(expr):
            exprs.append(expr)
            return orig(expr)

        self.fake._eval = capture
        self.assertIn("Clicked @e1", b.click("@e1"))
        joined = "\n".join(exprs)
        self.assertIn("MouseEvent", joined)
        self.assertIn("mousedown", joined)
        self.assertIn("mouseup", joined)

    def test_network_log_empty_and_captured(self):
        empty = b.network_log()
        self.assertIn("No captured network traffic", empty)
        self.fake._page["network"] = (
            '[{"url":"https://example.com/search","status":200,"body":"<JAD_ERROR>"}]'
        )
        captured = b.network_log()
        self.assertIn("https://example.com/search", captured)
        self.assertIn("<JAD_ERROR>", captured)
        self.assertIn("200", captured)

    def test_chrome_not_found_message(self):
        saved_find = b._find_chrome
        try:
            b._find_chrome = lambda: None
            b._SESSION = b._Session()
            out = b.navigate("https://example.com")
            self.assertIn("Chrome/Chromium not found", out)
        finally:
            b._find_chrome = saved_find


class _WsScriptedSock:
    """In-memory socket: HTTP 101 whose Accept matches the client's key."""

    def __init__(self, extra_frames=b"", accept=None, omit_accept=False):
        self.sent = bytearray()
        self._out = b""
        self._extra = extra_frames
        self._accept = accept
        self._omit_accept = omit_accept
        self._hs = False

    def sendall(self, data):
        self.sent.extend(data)
        if self._hs:
            return
        raw = bytes(data)
        if b"Sec-WebSocket-Key:" not in raw:
            return
        self._hs = True
        key = None
        for line in raw.split(b"\r\n"):
            if line.lower().startswith(b"sec-websocket-key:"):
                key = line.split(b":", 1)[1].strip().decode("ascii")
                break
        accept = self._accept if self._accept is not None else b._ws_accept_key(key)
        hdr = [
            b"HTTP/1.1 101 Switching Protocols",
            b"Upgrade: websocket",
            b"Connection: Upgrade",
        ]
        if not self._omit_accept:
            hdr.append(b"Sec-WebSocket-Accept: " + accept.encode("ascii"))
        self._out = b"\r\n".join(hdr) + b"\r\n\r\n" + self._extra

    def recv(self, n):
        if not self._out:
            return b""
        chunk, self._out = self._out[:n], self._out[n:]
        return chunk

    def settimeout(self, _t):
        pass

    def close(self):
        pass


def _unmasked_frame(opcode, payload=b""):
    header = bytearray([0x80 | (opcode & 0x0F)])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < (1 << 16):
        header.append(126)
        header += __import__("struct").pack(">H", n)
    else:
        header.append(127)
        header += __import__("struct").pack(">Q", n)
    return bytes(header) + payload


def _decode_masked_frames(blob):
    """Yield (opcode, payload) from client-masked frames in *blob*."""
    import struct
    i = 0
    out = []
    while i + 2 <= len(blob):
        b0, b1 = blob[i], blob[i + 1]
        i += 2
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", blob[i:i + 2])[0]
            i += 2
        elif length == 127:
            length = struct.unpack(">Q", blob[i:i + 8])[0]
            i += 8
        mask = blob[i:i + 4]
        i += 4
        data = bytes(blob[j] ^ mask[(j - i) % 4] for j in range(i, i + length))
        i += length
        out.append((opcode, data))
    return out


class WsRfc6455Test(unittest.TestCase):
    def _connect(self, sock):
        saved = __import__("socket").create_connection

        def fake_conn(_addr, timeout=None):
            return sock

        import socket
        socket.create_connection = fake_conn
        try:
            return b._WS("ws://127.0.0.1:9/devtools")
        finally:
            socket.create_connection = saved

    def test_handshake_accepts_matching_sec_websocket_accept(self):
        sock = _WsScriptedSock()
        ws = self._connect(sock)
        self.assertIsNotNone(ws)
        ws.close()

    def test_handshake_rejects_mismatched_sec_websocket_accept(self):
        sock = _WsScriptedSock(accept="AAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        with self.assertRaises(ConnectionError) as ctx:
            self._connect(sock)
        self.assertIn("accept mismatch", str(ctx.exception))

    def test_handshake_rejects_missing_sec_websocket_accept(self):
        sock = _WsScriptedSock(omit_accept=True)
        with self.assertRaises(ConnectionError) as ctx:
            self._connect(sock)
        self.assertIn("accept mismatch", str(ctx.exception))

    def test_ping_is_answered_with_matching_pong(self):
        payload = b"cdp-ping-payload"
        frames = _unmasked_frame(0x9, payload) + _unmasked_frame(0x1, b'{"id":1}')
        sock = _WsScriptedSock(extra_frames=frames)
        ws = self._connect(sock)
        text = ws.recv()
        self.assertEqual(text, '{"id":1}')
        # After the HTTP request, remaining sendall bytes are WS frames.
        hs_end = bytes(sock.sent).find(b"\r\n\r\n")
        self.assertNotEqual(hs_end, -1)
        rest = bytes(sock.sent)[hs_end + 4:]
        decoded = _decode_masked_frames(rest)
        pongs = [pl for op, pl in decoded if op == 0xA]
        self.assertEqual(pongs, [payload])

    def test_unsolicited_pong_is_ignored(self):
        frames = _unmasked_frame(0xA, b"unsolicited") + _unmasked_frame(0x1, b"ok")
        sock = _WsScriptedSock(extra_frames=frames)
        ws = self._connect(sock)
        self.assertEqual(ws.recv(), "ok")
        hs_end = bytes(sock.sent).find(b"\r\n\r\n")
        rest = bytes(sock.sent)[hs_end + 4:]
        self.assertEqual(_decode_masked_frames(rest), [])


if __name__ == "__main__":
    unittest.main()
