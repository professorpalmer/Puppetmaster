"""Dashboard port-collision remediation: bind-retry, project identity on the
wire (/api/meta), and truthful port propagation.

Two projects' `puppetmaster dashboard` used to silently collide: on Windows,
SO_REUSEADDR's inverted semantics let a second dashboard bind "successfully"
over the first's live listener, so the newcomer prints a URL and opens a
browser tab that the kernel never routes a single request to (the first
binder answers everything). See docs/CHANGELOG.md for the full writeup.

A new file, deliberately: tests/test_puppetmaster.py is huge and a
concurrent, independent workstream (server_close() on Ctrl-C) edits its
DashboardTests class around the same lines this change would otherwise touch.
Keeping the new socket-binding tests here avoids any collision between the
two changes landing as separate PRs.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.dashboard import (
    _port_taken,
    _state_dir_id,
    bind_dashboard_server,
    dashboard_identity,
    dashboard_serves,
    make_handler,
    read_dashboard_runfile,
    serve,
)
from puppetmaster.store_factory import create_store


def _free_port() -> int:
    """A port that's probably free right now.

    TOCTOU-y like the rest of this codebase's port helpers (see
    puppetmaster/ports.py's own documented caveat) — acceptable for tests
    that immediately occupy it themselves.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _occupy(port: int, *, reuse: bool) -> socket.socket:
    """Bind+listen on ``port`` with a plain socket — stands in for "some
    other process (or dashboard) already holds this port"."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if reuse:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def _handler():
    """A handler for bind-only tests that never actually dispatch a request."""
    return make_handler(lambda: None)


def _store_dir(tmp) -> Path:
    store_dir = Path(tmp) / ".puppetmaster"
    create_store("sqlite", store_dir).init()
    return store_dir


def _serve_requests(test: unittest.TestCase, httpd) -> None:
    """Start serving on a background thread and register cleanup, for tests
    that need to actually issue an HTTP request against a serve_forever=False
    server (which only binds -- nothing answers until serve_forever runs)."""
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    test.addCleanup(httpd.shutdown)


class PortTakenClassificationTests(unittest.TestCase):
    def test_port_taken_classifies_bind_errors(self) -> None:
        # 10048 / EADDRINUSE -- the ordinary "something's listening" case.
        in_use = OSError()
        in_use.errno = errno.EADDRINUSE
        in_use.winerror = 10048
        self.assertTrue(_port_taken(in_use))

        # Windows WSAEACCES: PermissionError(errno=EACCES, winerror=10013) --
        # both "incompatible SO_REUSEADDR on the incumbent" and "port sits in
        # a Hyper-V/WSL/winnat reserved range" surface this way. Constructed
        # via the 2-arg form + an attribute assignment (not the 4-arg
        # ``OSError(13, "x", None, 10013)`` form) because the 4-arg form's
        # winerror slot is a no-op on POSIX, which would make this assertion
        # pass on this Windows box but silently stop testing anything once
        # run under Linux CI.
        windows_denied = PermissionError(13, "denied")
        windows_denied.winerror = 10013
        self.assertTrue(_port_taken(windows_denied))

        # POSIX privileged-port EACCES (no winerror) must NOT be treated as
        # "bump past it" -- that would retry a permissions wall
        # max_attempts times instead of surfacing the real problem.
        posix_denied = PermissionError(13, "denied")
        self.assertFalse(_port_taken(posix_denied))


class BindDashboardServerTests(unittest.TestCase):
    def test_serve_bumps_to_next_free_port_by_default(self) -> None:
        port = _free_port()
        occupied = _occupy(port, reuse=False)
        self.addCleanup(occupied.close)
        httpd = bind_dashboard_server("127.0.0.1", port, _handler())
        self.addCleanup(httpd.server_close)
        self.assertEqual(httpd.server_address[1], port + 1)

    @unittest.skipUnless(os.name == "nt", "Windows-only: SO_REUSEADDR hijack")
    def test_two_serves_on_same_port_do_not_both_bind(self) -> None:
        """The headline regression test. Pre-fix, the second serve() call
        returned a bound server that never received a request (the first
        binder silently kept every connection)."""
        port = _free_port()
        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            first = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=port,
                open_browser=False, serve_forever=False, auto_port=False,
            )
            self.addCleanup(first.server_close)
            with self.assertRaises(OSError) as ctx:
                second = serve(
                    state_dir, backend="sqlite", host="127.0.0.1", port=port,
                    open_browser=False, serve_forever=False, auto_port=False,
                )
                second.server_close()  # pragma: no cover - only if it wrongly binds
            # bind_dashboard_server raises a friendlier, message-only OSError
            # `from` the real socket error (plan §3.1) -- the actual errno
            # lives on the chained cause, not the raised exception itself.
            self.assertIn(str(port), str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, OSError)
            self.assertIn(
                ctx.exception.__cause__.errno, (errno.EADDRINUSE, errno.EACCES)
            )

    def test_explicit_port_collision_is_strict(self) -> None:
        port = _free_port()
        occupied = _occupy(port, reuse=False)
        self.addCleanup(occupied.close)
        with self.assertRaises(OSError) as ctx:
            bind_dashboard_server("127.0.0.1", port, _handler(), auto_port=False)
        self.assertIn(str(port), str(ctx.exception))

    def test_port_search_gives_up_after_max_attempts(self) -> None:
        port = _free_port()
        occupied = [_occupy(port + i, reuse=False) for i in range(3)]
        for sock in occupied:
            self.addCleanup(sock.close)
        with self.assertRaises(OSError) as ctx:
            bind_dashboard_server("127.0.0.1", port, _handler(), max_attempts=3)
        message = str(ctx.exception)
        self.assertIn(str(port), message)
        self.assertIn(str(port + 2), message)

    def test_invalid_port_reports_actual_problem(self) -> None:
        """70000 isn't a TCP port at all -- pre-fix this was reported as
        "already in use", which is simply false: the bind loop's own
        `candidate > 65535` guard breaks out before ever attempting a
        bind."""
        with self.assertRaises(OSError) as ctx:
            bind_dashboard_server("127.0.0.1", 70000, _handler(), auto_port=False)
        message = str(ctx.exception)
        self.assertNotIn("already in use", message)
        self.assertIn("70000", message)


class ApiMetaTests(unittest.TestCase):
    def test_api_meta_reports_project_identity(self) -> None:
        import json
        import urllib.request

        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            httpd = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=0,
                open_browser=False, serve_forever=False,
            )
            self.addCleanup(httpd.server_close)
            _serve_requests(self, httpd)
            port = httpd.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta") as resp:
                payload = json.loads(resp.read())
            self.assertEqual(payload["service"], "puppetmaster-dashboard")
            self.assertEqual(payload["pid"], os.getpid())
            self.assertEqual(payload["state_dir_id"], _state_dir_id(state_dir))

    def test_api_meta_does_not_leak_absolute_path(self) -> None:
        import urllib.request

        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            httpd = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=0,
                open_browser=False, serve_forever=False,
            )
            self.addCleanup(httpd.server_close)
            _serve_requests(self, httpd)
            port = httpd.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta") as resp:
                raw = resp.read().decode("utf-8")
            self.assertNotIn(str(state_dir.resolve()), raw)
            self.assertNotIn(str(tmp), raw)


class DashboardServesTests(unittest.TestCase):
    def test_dashboard_serves_distinguishes_projects(self) -> None:
        with TemporaryDirectory() as tmp:
            dir_a = _store_dir(Path(tmp) / "a")
            dir_b = _store_dir(Path(tmp) / "b")
            server_a = serve(
                dir_a, backend="sqlite", host="127.0.0.1", port=0,
                open_browser=False, serve_forever=False,
            )
            self.addCleanup(server_a.server_close)
            _serve_requests(self, server_a)
            port_a = server_a.server_address[1]
            self.assertTrue(dashboard_serves("127.0.0.1", port_a, dir_a))
            self.assertFalse(dashboard_serves("127.0.0.1", port_a, dir_b))

    def test_dashboard_serves_false_for_legacy_server(self) -> None:
        """A pre-upgrade dashboard (or any plain web server) 200s /api/jobs
        but 404s /api/meta -- it must never be mistaken for "ours". This
        pins the rollout contract: no flag day, no coordinated upgrade."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Legacy(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/api/jobs":
                    body = b"[]"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        httpd = HTTPServer(("127.0.0.1", 0), _Legacy)
        self.addCleanup(httpd.server_close)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            self.assertFalse(dashboard_serves("127.0.0.1", port, state_dir))


class RunfileAndOnBoundTests(unittest.TestCase):
    def test_serve_writes_runfile_with_actual_bound_port(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            port = _free_port()
            occupied = _occupy(port, reuse=False)
            self.addCleanup(occupied.close)
            httpd = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=port,
                open_browser=False, serve_forever=False,
                runfile_state_dir=state_dir,
            )
            self.addCleanup(httpd.server_close)
            info = read_dashboard_runfile(state_dir)
            self.assertEqual(info["port"], port + 1)
            self.assertEqual(info["pid"], os.getpid())

    def test_on_bound_receives_actual_port(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = _store_dir(tmp)
            port = _free_port()
            occupied = _occupy(port, reuse=False)
            self.addCleanup(occupied.close)
            calls = []
            httpd = serve(
                state_dir, backend="sqlite", host="127.0.0.1", port=port,
                open_browser=False, serve_forever=False,
                on_bound=lambda h, p: calls.append((h, p)),
            )
            self.addCleanup(httpd.server_close)
            self.assertEqual(calls, [("127.0.0.1", port + 1)])


class DashboardIdentityRobustnessTests(unittest.TestCase):
    """`dashboard_identity` against misbehaving (or hostile) listeners on the
    port -- a real dashboard's own /api/meta is already covered by
    ApiMetaTests; these exercise the "something else is on this port"
    cases an ordinary `dashboard --background` can hit."""

    @staticmethod
    def _one_shot_server(respond) -> "tuple[socket.socket, int]":
        """A raw socket server that accepts exactly one connection, hands
        it to `respond(conn)` on a background thread, and returns
        (listening_socket, port)."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def _serve() -> None:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                respond(conn)
            except OSError:
                pass
            finally:
                conn.close()

        threading.Thread(target=_serve, daemon=True).start()
        return srv, port

    def test_truncated_body_returns_none_not_a_traceback(self) -> None:
        """A server that sends a valid Content-Length header then closes
        early used to escape as a raw http.client.IncompleteRead out of an
        ordinary `dashboard --background` -- the docstring promises None
        for exactly this "can't vouch for it" case (§F3)."""

        def truncate(conn: socket.socket) -> None:
            conn.recv(4096)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: 55\r\n\r\n1234567890"
            )

        srv, port = self._one_shot_server(truncate)
        self.addCleanup(srv.close)
        self.assertIsNone(dashboard_identity("127.0.0.1", port, timeout=1.0))

    def test_non_http_squatter_returns_none_not_a_traceback(self) -> None:
        """A non-HTTP service on the port (e.g. an SSH banner) raises
        http.client.BadStatusLine, which the pre-fix `except (OSError,
        ValueError)` also did not catch."""

        def banner(conn: socket.socket) -> None:
            conn.recv(4096)
            conn.sendall(b"SSH-2.0-OpenSSH_9.6\r\n")

        srv, port = self._one_shot_server(banner)
        self.addCleanup(srv.close)
        self.assertIsNone(dashboard_identity("127.0.0.1", port, timeout=1.0))

    def test_slow_dribbler_is_bounded_by_a_wall_clock_deadline(self) -> None:
        """`timeout=1.0` bounds each individual recv, not the call as a
        whole: a peer trickling one byte at a time keeps every recv under
        that timeout, so a bare per-recv timeout let this run for 12s
        against a declared 1.0s budget (measured). Must return within a
        small constant factor of `timeout`, not "eventually"."""
        payload = json.dumps(
            {"service": "puppetmaster-dashboard", "pid": 1, "state_dir_id": "x"}
        ).encode("utf-8")

        def dribble(conn: socket.socket) -> None:
            conn.recv(4096)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
            )
            for byte in payload:
                conn.sendall(bytes([byte]))
                time.sleep(0.5)

        srv, port = self._one_shot_server(dribble)
        self.addCleanup(srv.close)
        started = time.monotonic()
        result = dashboard_identity("127.0.0.1", port, timeout=1.0)
        elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 3.0, "must be bounded, not proportional to body size")

    def test_flood_body_is_capped_not_read_into_memory_whole(self) -> None:
        """A huge declared Content-Length must not be read in full -- the
        payload is one small JSON object; anything that large is not a
        dashboard, and reading it whole is an unbounded-memory foot-gun."""

        def flood(conn: socket.socket) -> None:
            conn.recv(4096)
            declared = 400 * 1024 * 1024
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {declared}\r\n\r\n".encode("utf-8")
            )
            chunk = b"0" * 65536
            try:
                for _ in range(20):
                    conn.sendall(chunk)
            except OSError:
                pass

        srv, port = self._one_shot_server(flood)
        self.addCleanup(srv.close)
        started = time.monotonic()
        result = dashboard_identity("127.0.0.1", port, timeout=2.0)
        elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
