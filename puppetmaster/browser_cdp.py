"""Stdlib Chrome DevTools Protocol (CDP) browser engine.

A dependency-free browser backend for the ``agentic`` adapter's browser toolset,
so browser-capable swarms run on the standalone-keys stack (Marionette's identity)
rather than requiring the Hermes adapter + agent-browser CLI.

Design:
- Launch a local Chrome/Chromium with ``--headless=new --remote-debugging-port``
  and a throwaway profile; discover the page target's websocket via DevTools /json.
- Talk CDP over a MINIMAL, self-contained RFC6455 websocket client built on the
  stdlib ``socket`` module (no websockets/websocket-client dependency).
- Expose small agent-facing functions returning STRINGS (never raise), mirroring
  the Hermes browser tool surface: navigate / snapshot (accessibility-ish tree
  with @e1 refs) / click / type / scroll / back / get_text / screenshot.

Safety: navigation is refused for unsafe/internal URLs via is_safe_url unless
PM_BROWSER_ALLOW_LOCAL=1. Every public function is best-effort.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from typing import Any, Optional

_SNAPSHOT_LIMIT = 12000
_TEXT_LIMIT = 12000

_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
]


def _find_chrome() -> Optional[str]:
    env = os.environ.get("PM_BROWSER_CHROME", "").strip()
    if env and (os.path.exists(env) or shutil.which(env)):
        return env
    for c in _CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _WS:
    """Minimal RFC6455 client: text frames only, enough for CDP."""

    def __init__(self, url: str, timeout: float = 30.0):
        assert url.startswith("ws://")
        rest = url[len("ws://"):]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.host = host
        self.port = int(port or 80)
        self.path = "/" + path
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._buf = b""
        self._handshake()

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("ws handshake closed early")
            resp += chunk
        if b" 101 " not in resp.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"ws handshake failed: {resp[:120]!r}")

    def send(self, text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < (1 << 16):
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("ws closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self) -> str:
        while True:
            b0, b1 = self._recv_exact(2)
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            data = self._recv_exact(length)
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:
                raise ConnectionError("ws closed by peer")
            if opcode in (0x9, 0xA):
                continue
            return data.decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class _Session:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.profile_dir: Optional[str] = None
        self.ws: Optional[_WS] = None
        self._id = 0
        self._lock = threading.Lock()

    def ensure(self) -> Optional[str]:
        if self.ws is not None:
            return None
        chrome = _find_chrome()
        if not chrome:
            return "Chrome/Chromium not found; browser tools unavailable. Set PM_BROWSER_CHROME."
        port = _free_port()
        self.profile_dir = tempfile.mkdtemp(prefix="pm-cdp-")
        args = [
            chrome, "--headless=new", f"--remote-debugging-port={port}",
            f"--user-data-dir={self.profile_dir}", "--no-first-run",
            "--no-default-browser-check", "--disable-gpu", "--disable-dev-shm-usage",
            "--remote-allow-origins=*", "about:blank",
        ]
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            return f"failed to launch Chrome: {e}"
        ws_url = None
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                    targets = json.loads(r.read().decode())
                for t in targets:
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        ws_url = t["webSocketDebuggerUrl"]
                        break
                if ws_url:
                    break
            except Exception:
                pass
            time.sleep(0.3)
        if not ws_url:
            return "Chrome started but no DevTools page target appeared."
        try:
            self.ws = _WS(ws_url)
            self._cmd("Page.enable")
            self._cmd("Runtime.enable")
            self._cmd("DOM.enable")
        except Exception as e:
            return f"failed to attach to DevTools: {e}"
        return None

    def _cmd(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        if self.ws is None:
            raise ConnectionError("no browser session")
        with self._lock:
            self._id += 1
            mid = self._id
            self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            end = time.time() + timeout
            while time.time() < end:
                msg = json.loads(self.ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", "CDP error"))
                    return msg.get("result", {})
            raise TimeoutError(f"CDP {method} timed out")

    def _eval(self, expr: str) -> Any:
        res = self._cmd("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True})
        return (res.get("result") or {}).get("value")

    def shutdown(self) -> None:
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        self.ws = None
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        try:
            if self.profile_dir and os.path.isdir(self.profile_dir):
                shutil.rmtree(self.profile_dir, ignore_errors=True)
        except Exception:
            pass


_SESSION = _Session()


@atexit.register
def _cleanup() -> None:
    _SESSION.shutdown()


def _url_ok(url: str) -> "tuple[bool, str]":
    if os.environ.get("PM_BROWSER_ALLOW_LOCAL", "").strip() in ("1", "true", "yes"):
        return True, ""
    try:
        try:
            from harness.url_safety import is_safe_url  # type: ignore
            ok, reason = is_safe_url(url)
            return bool(ok), ("" if ok else str(reason))
        except Exception:
            low = url.lower()
            for bad in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.",
                        "10.", "192.168.", "::1", "file:", "internal"):
                if bad in low:
                    return False, f"blocked internal/unsafe URL ({bad})"
            if not (low.startswith("http://") or low.startswith("https://")):
                return False, "only http(s) URLs are allowed"
            return True, ""
    except Exception as e:
        return False, f"url check failed: {e}"


def navigate(url: str) -> str:
    ok, reason = _url_ok(url)
    if not ok:
        return f"Refused to navigate: unsafe URL ({reason})."
    err = _SESSION.ensure()
    if err:
        return err
    try:
        _SESSION._cmd("Page.navigate", {"url": url})
        for _ in range(20):
            if _SESSION._eval("document.readyState") == "complete":
                break
            time.sleep(0.25)
        title = _SESSION._eval("document.title") or ""
        cur = _SESSION._eval("location.href") or url
        return f"Navigated to {cur}\nTitle: {title}\nCall browser_snapshot to see interactable elements."
    except Exception as e:
        return f"navigate failed: {type(e).__name__}: {e}"


_SNAPSHOT_JS = r"""
(() => {
  const out = [];
  let n = 0;
  const els = document.querySelectorAll('a,button,input,textarea,select,[role=button],[role=link],[onclick]');
  for (const el of els) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) continue;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    n += 1;
    const ref = 'e' + n;
    el.setAttribute('data-pm-ref', ref);
    const role = el.getAttribute('role') || el.tagName.toLowerCase();
    let name = (el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
                el.value || el.innerText || el.getAttribute('title') || '').trim();
    if (name.length > 80) name = name.slice(0, 80) + '...';
    out.push('@' + ref + ' ' + role + (name ? ' "' + name + '"' : ''));
    if (out.length >= 400) break;
  }
  return out.join('\n');
})()
"""


def snapshot() -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        listing = _SESSION._eval(_SNAPSHOT_JS) or ""
        title = _SESSION._eval("document.title") or ""
        cur = _SESSION._eval("location.href") or ""
        body = f"Page: {title} ({cur})\nInteractable elements (act via ref, e.g. browser_click @e3):\n{listing}"
        if len(body) > _SNAPSHOT_LIMIT:
            body = body[:_SNAPSHOT_LIMIT] + "\n... (snapshot truncated)"
        return body
    except Exception as e:
        return f"snapshot failed: {type(e).__name__}: {e}"


def _ref_expr(ref: str) -> str:
    r = ref.lstrip("@").replace("'", "")
    return f"document.querySelector('[data-pm-ref=\"{r}\"]')"


def click(ref: str) -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        el = _ref_expr(ref)
        found = _SESSION._eval(f"(function(){{var e={el};if(!e)return false;e.scrollIntoView({{block:'center'}});e.click();return true;}})()")
        if not found:
            return f"click failed: no element for {ref} (run browser_snapshot to refresh refs)."
        time.sleep(0.4)
        return f"Clicked {ref}. Call browser_snapshot to see the updated page."
    except Exception as e:
        return f"click failed: {type(e).__name__}: {e}"


def type_text(ref: str, text: str) -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        el = _ref_expr(ref)
        js_text = json.dumps(text)
        ok = _SESSION._eval(
            f"(function(){{var e={el};if(!e)return false;e.focus();"
            f"e.value={js_text};e.dispatchEvent(new Event('input',{{bubbles:true}}));"
            f"e.dispatchEvent(new Event('change',{{bubbles:true}}));return true;}})()")
        if not ok:
            return f"type failed: no element for {ref}."
        return f"Typed into {ref}."
    except Exception as e:
        return f"type failed: {type(e).__name__}: {e}"


def scroll(direction: str = "down") -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        dy = -600 if str(direction).lower().startswith("up") else 600
        _SESSION._eval(f"window.scrollBy(0,{dy})")
        return f"Scrolled {direction}."
    except Exception as e:
        return f"scroll failed: {type(e).__name__}: {e}"


def back() -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        _SESSION._eval("history.back()")
        time.sleep(0.4)
        return "Navigated back."
    except Exception as e:
        return f"back failed: {type(e).__name__}: {e}"


def get_text() -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        txt = _SESSION._eval("document.body ? document.body.innerText : ''") or ""
        if len(txt) > _TEXT_LIMIT:
            txt = txt[:_TEXT_LIMIT] + "\n... (text truncated)"
        return txt or "(empty page)"
    except Exception as e:
        return f"get_text failed: {type(e).__name__}: {e}"


def screenshot(out_dir: Optional[str] = None) -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        res = _SESSION._cmd("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(res.get("data", ""))
        target_dir = out_dir or tempfile.gettempdir()
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"pm-browser-{int(time.time())}.png")
        with open(path, "wb") as f:
            f.write(data)
        return f"Saved screenshot to {path} (view it with view_image)."
    except Exception as e:
        return f"screenshot failed: {type(e).__name__}: {e}"


def dispatch(name: str, args: dict, out_dir: Optional[str] = None) -> Optional[str]:
    a = args or {}
    if name == "browser_navigate":
        return navigate(str(a.get("url", "")))
    if name == "browser_snapshot":
        return snapshot()
    if name == "browser_click":
        return click(str(a.get("ref", "")))
    if name == "browser_type":
        return type_text(str(a.get("ref", "")), str(a.get("text", "")))
    if name == "browser_scroll":
        return scroll(str(a.get("direction", "down")))
    if name == "browser_back":
        return back()
    if name == "browser_get_text":
        return get_text()
    if name == "browser_screenshot":
        return screenshot(out_dir)
    return None


BROWSER_TOOL_NAMES = (
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
    "browser_scroll", "browser_back", "browser_get_text", "browser_screenshot",
)
