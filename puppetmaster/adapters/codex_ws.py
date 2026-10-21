"""A WebSocket client over a Unix socket, for the shared Codex host.

``codex app-server --listen unix://PATH`` speaks WebSocket on that socket.
The client sends masked text frames of JSON-RPC and reads text frames back.
It answers ping and stops on close. Standard library only.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
from typing import Any, Optional


class HostClosed(OSError):
    """The host closed the connection or the socket ended."""


class HostSocket:
    def __init__(self, path: str, timeout: float = 30.0) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise OSError("this platform has no Unix sockets")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        try:
            self._sock.connect(path)
            self._buffer = self._handshake()
            self._parts: list = []
        except Exception:
            self._sock.close()
            raise

    def _handshake(self) -> bytes:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        self._sock.sendall((
            "GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode("ascii"))
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise HostClosed("host closed during the WebSocket handshake")
            data += chunk
            if len(data) > 65536:
                raise OSError("WebSocket handshake reply too long")
        head, rest = data.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].split()
        if len(status) < 2 or status[1] != b"101":
            raise OSError("host refused the WebSocket upgrade")
        return rest

    def settimeout(self, timeout: Optional[float]) -> None:
        self._sock.settimeout(timeout)

    def _next_frame(self) -> tuple:
        """The next whole frame as (fin, opcode, payload). Bytes stay buffered until a frame is whole."""
        while True:
            data = self._buffer
            if len(data) >= 2:
                size, offset = data[1] & 0x7F, 2
                if size == 126 and len(data) >= 4:
                    size, offset = struct.unpack("!H", data[2:4])[0], 4
                elif size == 127 and len(data) >= 10:
                    size, offset = struct.unpack("!Q", data[2:10])[0], 10
                if (data[1] & 0x7F) < 126 or offset > 2:
                    if data[1] & 0x80:
                        offset += 4  # A server never masks; skip a mask key if one is there.
                    if len(data) >= offset + size:
                        self._buffer = data[offset + size:]
                        return bool(data[0] & 0x80), data[0] & 0x0F, data[offset:offset + size]
            # A timeout here keeps the partial frame for the next call.
            chunk = self._sock.recv(65536)
            if not chunk:
                raise HostClosed("host socket ended")
            self._buffer += chunk

    def _frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        size = len(payload)
        if size < 126:
            head = struct.pack("!BB", 0x80 | opcode, 0x80 | size)
        elif size < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, size)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, size)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._sock.sendall(head + mask + masked)

    def send(self, message: dict) -> None:
        self._frame(0x1, json.dumps(message, separators=(",", ":")).encode("utf-8"))

    def recv(self) -> Any:
        """The next JSON message. Raises ``socket.timeout`` when none arrives in time."""
        while True:
            fin, opcode, payload = self._next_frame()
            if opcode == 0x8:
                raise HostClosed("host closed the WebSocket")
            if opcode == 0x9:
                self._frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            self._parts.append(payload)
            if fin:
                message, self._parts = b"".join(self._parts), []
                return json.loads(message.decode("utf-8"))

    def close(self) -> None:
        try:
            self._frame(0x8, b"")
        except OSError:
            pass
        self._sock.close()
