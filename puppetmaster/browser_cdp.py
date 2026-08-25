"""Stdlib Chrome DevTools Protocol (CDP) browser engine.

A dependency-free browser backend for the ``agentic`` adapter's browser toolset,
so browser-capable swarms run on the standalone-keys stack (Marionette's identity)
rather than requiring the Hermes adapter + agent-browser CLI.

Design:
- Launch a local Chrome/Chromium with ``--remote-debugging-port`` and a profile
  directory; discover the page target's websocket via DevTools /json.
- Default is ``--headless=new`` plus a throwaway ``pm-cdp-*`` profile. Auth
  handoff uses ``PM_BROWSER_HEADED=1`` (visible window) and
  ``PM_BROWSER_USER_DATA_DIR`` (durable cookies). ``PM_BROWSER_CDP_PORT``
  publishes a shared debugging port so sibling workers attach instead of
  spawning a second isolated Chrome.
- Talk CDP over a MINIMAL, self-contained RFC6455 websocket client built on the
  stdlib ``socket`` module (no websockets/websocket-client dependency).
- Expose small agent-facing functions returning STRINGS (never raise), mirroring
  the Hermes browser tool surface: navigate / snapshot (accessibility-ish tree
  with @e1 refs) / click / type / scroll / back / get_text / screenshot /
  auth_handoff.

Safety: navigation is refused for unsafe/internal URLs via is_safe_url unless
PM_BROWSER_ALLOW_LOCAL=1. Every public function is best-effort.
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
import shutil
import signal
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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _headed() -> bool:
    return _env_truthy("PM_BROWSER_HEADED")


def _attach_only() -> bool:
    return _env_truthy("PM_BROWSER_ATTACH_ONLY")


def _preferred_port() -> Optional[int]:
    raw = os.environ.get("PM_BROWSER_CDP_PORT", "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= port <= 65535:
        return port
    return None


def _default_persistent_profile() -> str:
    return os.path.join(os.path.expanduser("~"), ".puppetmaster", "browser-profile")


def _profile_dir_for_launch() -> tuple:
    """Return ``(path, owns_profile)``. Temp dirs are deleted on shutdown."""
    explicit = os.environ.get("PM_BROWSER_USER_DATA_DIR", "").strip()
    if explicit:
        path = os.path.expanduser(explicit)
        os.makedirs(path, exist_ok=True)
        return path, False
    if _headed():
        path = _default_persistent_profile()
        os.makedirs(path, exist_ok=True)
        return path, False
    return tempfile.mkdtemp(prefix="pm-cdp-"), True


def chrome_launch_args(chrome: str, port: int, profile_dir: str, headed: bool) -> list:
    """Chrome argv for a CDP session. Headless keeps the historical flags."""
    args = [chrome]
    if not headed:
        args.append("--headless=new")
        args.append("--disable-gpu")
    args.extend([
        "--remote-debugging-port=%d" % int(port),
        "--user-data-dir=%s" % profile_dir,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--remote-allow-origins=*",
    ])
    if headed:
        args.extend(["--new-window", "--window-size=1280,800"])
    args.append("about:blank")
    return args


def _page_ws_url(port: int, timeout: float = 2.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/json" % int(port), timeout=timeout
        ) as r:
            targets = json.loads(r.read().decode())
    except Exception:
        return None
    if not isinstance(targets, list):
        return None
    for t in targets:
        if not isinstance(t, dict):
            continue
        ws = t.get("webSocketDebuggerUrl")
        if t.get("type") == "page" and ws:
            return str(ws)
    return None


def _ensure_page_ws(port: int) -> Optional[str]:
    ws = _page_ws_url(port)
    if ws:
        return ws
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/json/new?about:blank" % int(port), timeout=2
        ) as r:
            created = json.loads(r.read().decode())
        if isinstance(created, dict) and created.get("webSocketDebuggerUrl"):
            return str(created["webSocketDebuggerUrl"])
    except Exception:
        pass
    return _page_ws_url(port)


def _wait_for_page_ws(port: int, seconds: float) -> Optional[str]:
    deadline = time.time() + seconds
    ws_url = None
    while time.time() < deadline:
        ws_url = _ensure_page_ws(port)
        if ws_url:
            return ws_url
        time.sleep(0.3)
    return ws_url


# True only in the interactive harness process. Shared CDP Chrome must survive
# worker atexit; the janitor reaps it when the desktop/harness exits.
_JANITOR = False


def set_janitor(value: bool = True) -> None:
    global _JANITOR
    _JANITOR = bool(value)


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(key: str) -> str:
    """RFC 6455 Sec-WebSocket-Accept = base64(SHA-1(key + GUID))."""
    digest = hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _http_header_value(header_blob: bytes, name: bytes) -> Optional[str]:
    want = name.lower()
    for line in header_blob.split(b"\r\n")[1:]:
        if b":" not in line:
            continue
        raw_k, _, raw_v = line.partition(b":")
        if raw_k.strip().lower() == want:
            return raw_v.strip().decode("ascii", "replace")
    return None


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
        header_blob, sep, leftover = resp.partition(b"\r\n\r\n")
        if not sep or b" 101 " not in header_blob.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"ws handshake failed: {resp[:120]!r}")
        accept = _http_header_value(header_blob, b"sec-websocket-accept")
        expected = _ws_accept_key(key)
        if accept != expected:
            raise ConnectionError(
                f"ws accept mismatch: got {accept!r}, expected {expected!r}"
            )
        self._buf = leftover + self._buf

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | (opcode & 0x0F)])
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

    def send(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

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
            if opcode == 0x9:
                # RFC 6455: a Ping must be answered with a Pong carrying the
                # same application data. Unsolicited Pongs are ignored.
                self._send_frame(0xA, data)
                continue
            if opcode == 0xA:
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
        self.port: Optional[int] = None
        self.headed = False
        self.owns_proc = False
        self.owns_profile = False

    def _attach_ws(self, ws_url: str, *, port: int, headed: bool,
                   profile_dir: Optional[str], owns_proc: bool,
                   owns_profile: bool) -> Optional[str]:
        try:
            self.ws = _WS(ws_url)
            self._cmd("Page.enable")
            self._cmd("Runtime.enable")
            self._cmd("DOM.enable")
        except Exception as e:
            self.ws = None
            return "failed to attach to DevTools: %s" % e
        self.port = int(port)
        self.headed = bool(headed)
        self.profile_dir = profile_dir
        self.owns_proc = bool(owns_proc)
        self.owns_profile = bool(owns_profile)
        os.environ["PM_BROWSER_CDP_PORT"] = str(int(port))
        return None

    def ensure(self) -> Optional[str]:
        if self.ws is not None:
            return None
        headed = _headed()
        preferred = _preferred_port()
        persist = os.environ.get("PM_BROWSER_USER_DATA_DIR", "").strip()
        if persist:
            persist = os.path.expanduser(persist)
        if preferred is not None:
            ws_url = _wait_for_page_ws(preferred, 1.5)
            if ws_url:
                return self._attach_ws(
                    ws_url, port=preferred, headed=headed,
                    profile_dir=persist or None, owns_proc=False,
                    owns_profile=False,
                )
            if _attach_only():
                return (
                    "No Chrome DevTools at 127.0.0.1:%d. Start Chrome with "
                    "--remote-debugging-port=%d or unset PM_BROWSER_ATTACH_ONLY."
                    % (preferred, preferred)
                )
        chrome = _find_chrome()
        if not chrome:
            return "Chrome/Chromium not found; browser tools unavailable. Set PM_BROWSER_CHROME."
        port = preferred if preferred is not None else _free_port()
        profile_dir, owns_profile = _profile_dir_for_launch()
        args = chrome_launch_args(chrome, port, profile_dir, headed)
        try:
            popen_kw = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name != "nt":
                popen_kw["start_new_session"] = True
            self.proc = subprocess.Popen(args, **popen_kw)
        except Exception as e:
            return "failed to launch Chrome: %s" % e
        ws_url = _wait_for_page_ws(port, 20)
        if not ws_url:
            # Port may already be a live Chrome we lost the race to launch.
            if preferred is not None:
                ws_url = _wait_for_page_ws(preferred, 5)
                if ws_url:
                    _stop_chrome_proc(self.proc)
                    self.proc = None
                    return self._attach_ws(
                        ws_url, port=preferred, headed=headed,
                        profile_dir=profile_dir, owns_proc=False,
                        owns_profile=False,
                    )
            return "Chrome started but no DevTools page target appeared."
        err = self._attach_ws(
            ws_url, port=port, headed=headed, profile_dir=profile_dir,
            owns_proc=True, owns_profile=owns_profile,
        )
        if err:
            return err
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
        shared = (_preferred_port() is not None) or (not self.owns_profile)
        reap = self.owns_proc and ((not shared) or _JANITOR)
        proc = self.proc
        port = self.port if reap else None
        self.proc = None
        self.owns_proc = False
        if reap:
            _stop_chrome_proc(proc, port=port)
        if self.owns_profile:
            try:
                if self.profile_dir and os.path.isdir(self.profile_dir):
                    shutil.rmtree(self.profile_dir, ignore_errors=True)
            except Exception:
                pass
        self.owns_profile = False


_SESSION = _Session()


def _stop_chrome_proc(proc, port=None, timeout=8.0):
    """SIGTERM the owned Chrome, wait until it (and the CDP port) die, then SIGKILL.

    Chrome is a process group. Dropping the Popen handle after a fire-and-forget
    terminate leaves the debug port bound, so a headed relaunch cannot reuse it.
    """
    if proc is None and not port:
        return
    pid = getattr(proc, "pid", None) if proc is not None else None

    def _signal_tree(sig_term):
        if os.name != "nt" and pid:
            try:
                os.killpg(int(pid), signal.SIGTERM if sig_term else signal.SIGKILL)
                return
            except Exception:
                pass
        if proc is None:
            return
        try:
            if sig_term:
                proc.terminate()
            else:
                proc.kill()
        except Exception:
            pass

    _signal_tree(True)
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        alive = False
        if proc is not None:
            try:
                if proc.poll() is None:
                    alive = True
            except Exception:
                pass
        if (not alive) and port:
            if _page_ws_url(int(port), timeout=0.2):
                alive = True
        if not alive:
            if proc is not None:
                try:
                    proc.wait(timeout=0.2)
                except Exception:
                    pass
            return
        time.sleep(0.1)
    _signal_tree(False)
    if proc is not None:
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def reset_session(keep_profile: bool = True) -> None:
    """Drop the live CDP connection so the next ensure() can relaunch.

    Persistent profiles stay on disk. Chrome this process spawned is killed so
    a headed relaunch can reuse the same debugging port.
    """
    global _SESSION
    if keep_profile:
        _SESSION.owns_profile = False
    port = getattr(_SESSION, "port", None)
    proc = _SESSION.proc if getattr(_SESSION, "owns_proc", False) else None
    if getattr(_SESSION, "owns_proc", False):
        _SESSION.proc = None
        _SESSION.owns_proc = False
        _stop_chrome_proc(proc, port=port)
    _SESSION.shutdown()
    _SESSION = _Session()


def session_info() -> dict:
    s = _SESSION
    profile = getattr(s, "profile_dir", None)
    owns_profile = bool(getattr(s, "owns_profile", False))
    return {
        "connected": getattr(s, "ws", None) is not None,
        "headed": bool(getattr(s, "headed", False)),
        "port": getattr(s, "port", None),
        "profile_dir": profile,
        "owns_proc": bool(getattr(s, "owns_proc", False)),
        "persistent": bool(profile and not owns_profile),
    }


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
        _install_network_hook()
        return (
            f"Navigated to {cur}\nTitle: {title}\n"
            "Call browser_snapshot to see interactable elements. "
            "Call browser_network after key actions to read captured request/response bodies."
        )
    except Exception as e:
        return f"navigate failed: {type(e).__name__}: {e}"


def auth_handoff(url: str) -> str:
    """Open ``url`` in a visible Chrome using the durable profile.

    Never returns cookies or passwords. Workers attach via PM_BROWSER_CDP_PORT.
    """
    os.environ["PM_BROWSER_HEADED"] = "1"
    if getattr(_SESSION, "ws", None) is not None and not getattr(_SESSION, "headed", False):
        reset_session(keep_profile=True)
    nav = navigate(url)
    info = session_info()
    port = info.get("port")
    profile = info.get("profile_dir") or ""
    persist = "yes" if info.get("persistent") else "no"
    return (
        "%s\n\nAuth handoff: complete login or Cloudflare in the visible Chrome "
        "window. Do not paste passwords or cookies into chat. Workers reuse this "
        "session (port=%s, persistent=%s, profile=%s)."
        % (nav, port, persist, profile)
    )


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


_NETWORK_HOOK_JS = r"""
(() => {
  if (window.__pmNetHooked) return 'already';
  window.__pmNetHooked = true;
  window.__pmNet = [];
  const push = (entry) => {
    window.__pmNet.push(entry);
    if (window.__pmNet.length > 80) window.__pmNet.shift();
  };
  const origFetch = window.fetch;
  if (typeof origFetch === 'function') {
    window.fetch = async function() {
      const input = arguments[0];
      const url = (typeof input === 'string') ? input : (input && input.url) || '';
      try {
        const res = await origFetch.apply(this, arguments);
        let body = '';
        try { body = await res.clone().text(); } catch (e) { body = ''; }
        push({url: String(url), status: res.status, body: String(body).slice(0, 4000)});
        return res;
      } catch (err) {
        push({url: String(url), error: String(err)});
        throw err;
      }
    };
  }
  const XHR = window.XMLHttpRequest;
  if (XHR && XHR.prototype) {
    const origOpen = XHR.prototype.open;
    const origSend = XHR.prototype.send;
    XHR.prototype.open = function(method, url) {
      this.__pmUrl = url;
      return origOpen.apply(this, arguments);
    };
    XHR.prototype.send = function() {
      this.addEventListener('load', function() {
        push({
          url: String(this.__pmUrl || ''),
          status: this.status,
          body: String(this.responseText || '').slice(0, 4000),
        });
      });
      return origSend.apply(this, arguments);
    };
  }
  return 'installed';
})()
"""


def _install_network_hook() -> None:
    """Best-effort fetch/XHR capture so workers can judge by network truth."""
    try:
        _SESSION._eval(_NETWORK_HOOK_JS)
    except Exception:
        pass


def snapshot() -> str:
    err = _SESSION.ensure()
    if err:
        return err
    try:
        _install_network_hook()
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
        found = _SESSION._eval(
            f"(function(){{var e={el};if(!e)return false;"
            f"e.scrollIntoView({{block:'center'}});"
            f"var o={{bubbles:true,cancelable:true,view:window}};"
            f"e.dispatchEvent(new MouseEvent('mousedown',o));"
            f"e.dispatchEvent(new MouseEvent('mouseup',o));"
            f"e.dispatchEvent(new MouseEvent('click',o));"
            f"return true;}})()"
        )
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
            f"var proto=e.tagName==='TEXTAREA'"
            f"?window.HTMLTextAreaElement.prototype"
            f":window.HTMLInputElement.prototype;"
            f"var desc=Object.getOwnPropertyDescriptor(proto,'value');"
            f"if(desc&&desc.set)desc.set.call(e,{js_text});else e.value={js_text};"
            f"e.dispatchEvent(new Event('input',{{bubbles:true}}));"
            f"e.dispatchEvent(new Event('change',{{bubbles:true}}));"
            f"return true;}})()")
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


def network_log() -> str:
    """Return captured fetch/XHR request/response pairs (network-truth)."""
    err = _SESSION.ensure()
    if err:
        return err
    try:
        _install_network_hook()
        raw = _SESSION._eval(
            "JSON.stringify(window.__pmNet || [])"
        )
        if not raw or raw in ("[]", "null"):
            return (
                "No captured network traffic yet. Trigger the action "
                "(submit/search), then call browser_network again. "
                "Judge success by status AND body — HTTP 200 can carry an error."
            )
        body = raw if isinstance(raw, str) else json.dumps(raw)
        if len(body) > _TEXT_LIMIT:
            body = body[:_TEXT_LIMIT] + "\n... (network log truncated)"
        return (
            "Captured fetch/XHR traffic (judge by status AND body; "
            f"HTTP 200 can hide an application error):\n{body}"
        )
    except Exception as e:
        return f"network_log failed: {type(e).__name__}: {e}"


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
    if name == "browser_auth_handoff":
        return auth_handoff(str(a.get("url", "")))
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
    if name == "browser_network":
        return network_log()
    if name == "browser_screenshot":
        return screenshot(out_dir)
    return None


BROWSER_TOOL_NAMES = (
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
    "browser_scroll", "browser_back", "browser_get_text", "browser_network",
    "browser_screenshot", "browser_auth_handoff",
)
