"""Regression tests for two bind_dashboard_server bugs.

a33abbb6-1: an IPv6 loopback host failed with gaierror on an AF_INET socket.
a33abbb6-2: the exhausted-range error named ports above 65535.
"""
from __future__ import annotations

import errno
import socket
import unittest
from unittest import mock

from puppetmaster import dashboard
from puppetmaster.dashboard import bind_dashboard_server, make_handler


def _handler():
    return make_handler(lambda: None)


def _ipv6_loopback_available() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


class _AlwaysTaken:
    """Stands in for a server class whose every port is in use."""

    tried = []

    def __init__(self, address, handler):
        _AlwaysTaken.tried.append(address[1])
        raise OSError(errno.EADDRINUSE, "in use")


class IPv6HostTests(unittest.TestCase):
    @unittest.skipUnless(_ipv6_loopback_available(), "no IPv6 loopback")
    def test_ipv6_loopback_binds(self) -> None:
        httpd = bind_dashboard_server("::1", 0, _handler())
        self.addCleanup(httpd.server_close)
        self.assertEqual(httpd.address_family, socket.AF_INET6)
        self.assertEqual(httpd.server_address[0], "::1")

    def test_ipv4_host_keeps_ipv4_socket(self) -> None:
        httpd = bind_dashboard_server("127.0.0.1", 0, _handler())
        self.addCleanup(httpd.server_close)
        self.assertEqual(httpd.address_family, socket.AF_INET)
        self.assertEqual(httpd.server_address[0], "127.0.0.1")


class ExhaustedRangeMessageTests(unittest.TestCase):
    def _exhaust(self, port: int, max_attempts: int) -> str:
        _AlwaysTaken.tried = []
        with mock.patch.object(
            dashboard, "_dashboard_server_class", lambda host="": _AlwaysTaken
        ):
            with self.assertRaises(OSError) as ctx:
                bind_dashboard_server(
                    "127.0.0.1", port, _handler(), max_attempts=max_attempts
                )
        return str(ctx.exception)

    def test_range_at_ceiling_names_only_tried_ports(self) -> None:
        message = self._exhaust(65534, 5)
        self.assertEqual(_AlwaysTaken.tried, [65534, 65535])
        self.assertIn("in 65534-65535 on 127.0.0.1", message)
        self.assertNotIn("65536", message)
        self.assertNotIn("65538", message)

    def test_range_below_ceiling_is_unchanged(self) -> None:
        message = self._exhaust(65530, 3)
        self.assertEqual(_AlwaysTaken.tried, [65530, 65531, 65532])
        self.assertIn("in 65530-65532 on 127.0.0.1", message)


if __name__ == "__main__":
    unittest.main()
