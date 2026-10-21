from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster import codex_host
from puppetmaster.adapters import codex as codex_adapter
from puppetmaster.adapters.codex_session import run_host_turn
from puppetmaster.adapters.codex_ws import HostSocket
from puppetmaster import flow
from puppetmaster.flow import validate_graph
from puppetmaster.proc_identity import process_identity
from puppetmaster.worker_attribution import codex_event_references


class FakeHost:
    """A one-connection WebSocket JSON-RPC server on a Unix socket.

    ``script(method, params, send)`` answers each client request.
    """

    def __init__(self, script):
        self.dir = tempfile.mkdtemp(prefix="pmh", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.path = os.path.join(self.dir, "s")
        self.script = script
        self.received: list = []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self):
        self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _serve(self):
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        self.conn = conn
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(4096)
            key = [line.split(b":", 1)[1].strip() for line in data.split(b"\r\n")
                   if line.lower().startswith(b"sec-websocket-key")][0]
            accept = base64.b64encode(hashlib.sha1(key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest())
            conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                         b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n")
            # A ping first: the client must answer it and go on.
            conn.sendall(b"\x89\x00")
            buffer = b""

            def read(count):
                nonlocal buffer
                while len(buffer) < count:
                    chunk = conn.recv(65536)
                    if not chunk:
                        raise OSError("closed")
                    buffer += chunk
                out, buffer = buffer[:count], buffer[count:]
                return out

            def send(message):
                payload = json.dumps(message).encode()
                head = struct.pack("!BB", 0x81, len(payload)) if len(payload) < 126 else \
                    struct.pack("!BBH", 0x81, 126, len(payload))
                conn.sendall(head + payload)

            while True:
                try:
                    first, second = read(2)
                except OSError:
                    return
                size = second & 0x7F
                if size == 126:
                    size = struct.unpack("!H", read(2))[0]
                elif size == 127:
                    size = struct.unpack("!Q", read(8))[0]
                mask = read(4)
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(read(size)))
                if first & 0x0F == 0x8:
                    return
                if first & 0x0F != 0x1:
                    continue
                message = json.loads(payload)
                self.received.append(message)
                if "method" in message:
                    self.script(message, send)


def turn_script(extra=None):
    def script(message, send):
        method, ident = message["method"], message.get("id")
        if method == "initialize":
            send({"id": ident, "result": {}})
        elif method in ("thread/start", "thread/resume"):
            send({"id": ident, "result": {"thread": {"id": "t1", "path": "/rollouts/t1.jsonl"}}})
        elif method == "turn/start":
            send({"id": ident, "result": {"turn": {"id": "u1"}}})
            send({"method": "turn/started", "params": {"threadId": "t1", "turn": {"id": "u1"}}})
            if extra:
                extra(send)
                return
            send({"method": "item/completed", "params": {"threadId": "other", "item": {
                "type": "agentMessage", "text": "not mine"}}})
            send({"method": "item/completed", "params": {"threadId": "t1", "item": {
                "type": "fileChange", "status": "completed",
                "changes": [{"path": "/w/out/a.txt", "kind": {"type": "add"}, "diff": "ok"}]}}})
            send({"method": "item/completed", "params": {"threadId": "t1", "item": {
                "type": "commandExecution", "command": "/bin/zsh -lc 'cat out/b.txt'",
                "exitCode": 0, "status": "completed"}}})
            send({"method": "item/completed", "params": {"threadId": "t1", "item": {
                "type": "agentMessage", "text": "VERDICT: PASS", "phase": "final_answer"}}})
            send({"method": "thread/tokenUsage/updated", "params": {"threadId": "t1", "tokenUsage": {
                "total": {"inputTokens": 120, "outputTokens": 7, "cachedInputTokens": 64}}}})
            send({"method": "turn/completed", "params": {"threadId": "t1", "turn": {
                "id": "u1", "status": "completed"}}})
        elif method == "turn/interrupt":
            send({"id": ident, "result": {}})
            send({"method": "turn/completed", "params": {"threadId": "t1", "turn": {
                "id": "u1", "status": "interrupted"}}})
    return script


@unittest.skipUnless(codex_host.supported(), "the shared host needs Unix sockets")
class HostTurnTests(unittest.TestCase):
    def host(self, script):
        fake = FakeHost(script)
        self.addCleanup(fake.close)
        return fake

    def test_turn_collects_its_own_thread_items(self):
        fake = self.host(turn_script())
        result = run_host_turn(lambda: HostSocket(fake.path, timeout=5), "/w", "build it",
                               model="m", effort="high", config={"x": 1}, timeout=10)
        self.assertEqual(result.status, "completed")
        self.assertEqual((result.thread_id, result.turn_id), ("t1", "u1"))
        self.assertEqual(result.messages, ["VERDICT: PASS"])
        self.assertEqual(result.rollout_path, "/rollouts/t1.jsonl")
        self.assertEqual((result.input_tokens, result.output_tokens, result.cached_input_tokens),
                         (120, 7, 64))
        self.assertEqual([item["type"] for item in result.items], ["file_change", "command_execution"])
        start = next(m for m in fake.received if m.get("method") == "thread/start")["params"]
        self.assertEqual((start["model"], start["approvalPolicy"], start["config"]), ("m", "never", {"x": 1}))
        turn = next(m for m in fake.received if m.get("method") == "turn/start")["params"]
        self.assertEqual((turn["threadId"], turn["effort"]), ("t1", "high"))

    def test_events_feed_the_write_scope_attribution(self):
        fake = self.host(turn_script())
        result = run_host_turn(lambda: HostSocket(fake.path, timeout=5), "/w", "build it", timeout=10)
        events = codex_adapter.parse_codex_events(codex_adapter._app_server_events(result))
        self.assertEqual(codex_adapter.observed_thread_id(events), "t1")
        paths = codex_event_references(events, Path("/w")).paths
        self.assertIn("out/a.txt", paths)
        self.assertIn("out/b.txt", paths)

    def test_resume_sends_thread_resume(self):
        fake = self.host(turn_script())
        result = run_host_turn(lambda: HostSocket(fake.path, timeout=5), "/w", "repair",
                               resume_thread="t1", timeout=10)
        self.assertEqual(result.status, "completed")
        methods = [m.get("method") for m in fake.received]
        self.assertIn("thread/resume", methods)
        self.assertNotIn("thread/start", methods)

    def test_cut_interrupts_only_its_turn(self):
        cut = threading.Event()
        fake = self.host(turn_script(extra=lambda send: cut.set()))
        result = run_host_turn(lambda: HostSocket(fake.path, timeout=5), "/w", "build it",
                               timeout=10, cancellation_check=cut.is_set)
        self.assertEqual(result.status, "cancelled")
        interrupt = next(m for m in fake.received if m.get("method") == "turn/interrupt")["params"]
        self.assertEqual(interrupt, {"threadId": "t1", "turnId": "u1"})

    def test_approval_request_is_declined_and_fails(self):
        def ask(send):
            send({"id": 99, "method": "commandExecution/requestApproval", "params": {"threadId": "t1"}})
        fake = self.host(turn_script(extra=ask))
        result = run_host_turn(lambda: HostSocket(fake.path, timeout=5), "/w", "build it", timeout=10)
        self.assertEqual(result.status, "failed")
        self.assertIn({"jsonrpc": "2.0", "id": 99, "result": {"decision": "decline"}}, fake.received)

    def test_a_frame_split_across_a_timeout_stays_whole(self):
        gate = threading.Event()

        def split(message, send):
            if message["method"] == "initialize":
                frame = struct.pack("!BB", 0x81, 9) + b'{"id": 1}'
                fake.conn.sendall(frame[:4])
                gate.wait(5)
                fake.conn.sendall(frame[4:])
        fake = self.host(split)
        sock = HostSocket(fake.path, timeout=5)
        self.addCleanup(sock.close)
        sock.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        sock.settimeout(0.2)
        with self.assertRaises(socket.timeout):
            sock.recv()
        gate.set()
        sock.settimeout(5)
        self.assertEqual(sock.recv(), {"id": 1})

    def test_no_host_fails_before_a_thread(self):
        result = run_host_turn(lambda: HostSocket("/tmp/pm-no-such-host.sock", timeout=1), "/w", "x", timeout=5)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.thread_id)


class HostConfigTests(unittest.TestCase):
    def test_only_config_overrides_can_ride_a_thread(self):
        self.assertEqual(codex_adapter._host_config(None), {})
        self.assertEqual(codex_adapter._host_config(["-c", 'model_reasoning_effort="high"', "-c", "n=2"]),
                         {"model_reasoning_effort": "high", "n": 2})
        self.assertIsNone(codex_adapter._host_config(["--search"]))
        self.assertIsNone(codex_adapter._host_config(["-c"]))

    def test_shell_policy_sends_only_this_worker_differences(self):
        base = codex_adapter.inject_worker_cli_env(os.environ.copy())
        dropped = next(iter(base))
        env = {key: value for key, value in base.items() if key != dropped}
        env["PM_PORT_TEST"] = "4100"
        policy = codex_adapter._host_shell_policy(env)
        self.assertEqual(policy["set"], {"PM_PORT_TEST": "4100"})
        self.assertEqual(policy["exclude"], [dropped])


class HostLifecycleTests(unittest.TestCase):
    def test_graph_or_operator_requests_the_host(self):
        on = {"defaults": {"codex_host": True}}
        self.assertEqual(codex_host.requested(on, env={}), codex_host.supported())
        self.assertFalse(codex_host.requested({"defaults": {"codex_host": False}},
                                              env={codex_host.OPT_IN_ENV: "1"}))
        self.assertEqual(codex_host.requested({}, env={codex_host.OPT_IN_ENV: "1"}), codex_host.supported())
        self.assertFalse(codex_host.requested({}, env={}))

    def test_codex_host_must_be_a_boolean(self):
        graph = {"id": "g", "entry": "a", "defaults": {"codex_host": "yes"},
                 "nodes": [{"id": "a", "kind": "end", "status": "pass"}], "edges": []}
        self.assertIn("defaults.codex_host must be true or false", validate_graph(graph))

    @unittest.skipUnless(codex_host.supported(), "the shared host needs Unix sockets")
    def test_current_needs_a_live_recorded_process(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        path = folder / codex_host.RECORD_NAME
        record = {"pid": os.getpid(), "proc": process_identity(os.getpid()), "socket": "/tmp/s"}
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(codex_host.current({codex_host.RECORD_ENV: str(path)})["pid"], os.getpid())
        path.write_text(json.dumps({**record, "proc": "reused"}), encoding="utf-8")
        self.assertIsNone(codex_host.current({codex_host.RECORD_ENV: str(path)}))
        self.assertIsNone(codex_host.current({}))


FAKE_CODEX = """#!{python}
import os, signal, socket, sys, tempfile, time
link = sys.argv[sys.argv.index("--listen") + 1][len("unix://"):]
short = os.path.join(tempfile.mkdtemp(prefix="pmf", dir="/tmp"), "s")
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(short)
os.symlink(short, link)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""


@unittest.skipUnless(codex_host.supported(), "the shared host needs Unix sockets")
class WalkerHostTests(unittest.TestCase):
    def test_top_walk_owns_one_host_for_its_nodes(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        fake = folder / "codex"
        fake.write_text(FAKE_CODEX.format(python=sys.executable), encoding="utf-8")
        fake.chmod(0o755)
        check = (f'"{sys.executable}" -c "import json,os;'
                 f'r=json.load(open(os.environ[\'{codex_host.RECORD_ENV}\']));'
                 f'os.kill(r[\'pid\'],0);print(\'VERDICT: PASS\')"')
        graph = {"id": "host-walk", "entry": "check", "defaults": {"codex_host": True},
                 "nodes": [{"id": "check", "kind": "shell", "command": check},
                           {"id": "done", "kind": "end", "status": "pass"}],
                 "edges": [{"from": "check", "to": "done", "when": "ok"}]}
        env = {"CODEX_COMMAND": str(fake), "PUPPETMASTER_CODEX_LEAN_HOME": "0"}
        with patch.dict(os.environ, env):
            os.environ.pop(codex_host.RECORD_ENV, None)
            run = flow.new_run(folder / "state", graph, cwd=str(folder))
            run = flow.walk(folder / "state", run.run_id)
            self.assertNotIn(codex_host.RECORD_ENV, os.environ)
        self.assertEqual((run.status, run.reason), ("done", "pass"), run.steps)
        run_dir = flow.run_path(folder / "state", run.run_id).parent
        self.assertFalse((run_dir / codex_host.RECORD_NAME).exists())


if __name__ == "__main__":
    unittest.main()
