"""Deterministic end-to-end HTTP MCP loop for the Grok Bot remote transport.

Spins ``serve-remote`` on an ephemeral port (no tunnel, no cloudflared) and
drives the real JSON-RPC path over HTTP:

1. initialize
2. tools/list (supervise set; implement absent)
3. tools/call puppetmaster_doctor (real handler via HTTP; CLI subprocess)
4. tools/call start_* → job_id (start_cli mocked for CI determinism) then
   status/show against a real seeded SQLite store
5. auth 401 + scope deny for implement

Green in CI without network egress beyond loopback.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from unittest.mock import patch

from puppetmaster.mcp_remote import (
    IMPLEMENT_TOOL_NAMES,
    MCP_ENDPOINT_PATH,
    RemoteMcpConfig,
    build_server,
)
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus
from puppetmaster.store_factory import create_store


TOKEN = "e2e-known-token-not-a-secret"


def _parse_mcp_http_body(content_type: str, raw: bytes) -> Any:
    """Parse application/json or text/event-stream MCP response bodies."""
    if not raw:
        return None
    ctype = (content_type or "").lower()
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in ctype:
        data_lines = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return text
        # Last data payload is the JSON-RPC response for one-shot streams.
        return json.loads(data_lines[-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _HttpMcpClient:
    """Tiny streamable-HTTP MCP client against a running remote server."""

    def __init__(
        self,
        port: int,
        token: str,
        *,
        accept: str = "application/json, text/event-stream",
        origin: Optional[str] = None,
    ) -> None:
        self.port = port
        self.token = token
        self.accept = accept
        self.origin = origin
        self.session_id: Optional[str] = None
        self._next_id = 1

    def request(
        self,
        method: str,
        path: str = MCP_ENDPOINT_PATH,
        *,
        body: Optional[Any] = None,
        token: Optional[str] = None,
        auth: bool = True,
    ) -> tuple[int, dict[str, str], Any]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=30)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": self.accept}
        if self.origin:
            headers["Origin"] = self.origin
        if auth:
            headers["Authorization"] = f"Bearer {token if token is not None else self.token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if payload is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        resp_headers = {k.lower(): v for k, v in response.getheaders()}
        status = response.status
        conn.close()
        if "mcp-session-id" in resp_headers:
            self.session_id = resp_headers["mcp-session-id"]
        parsed = _parse_mcp_http_body(resp_headers.get("content-type", ""), raw)
        return status, resp_headers, parsed

    def rpc(self, method: str, params: Optional[dict] = None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            body["params"] = params
        status, _, parsed = self.request("POST", body=body)
        self.assert_ok_http(status, parsed)
        return parsed

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        response = self.rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise AssertionError(f"tools/call missing result: {response!r}")
        return result

    @staticmethod
    def assert_ok_http(status: int, parsed: Any) -> None:
        if status != 200:
            raise AssertionError(f"HTTP {status}: {parsed!r}")
        if isinstance(parsed, dict) and "error" in parsed and "result" not in parsed:
            raise AssertionError(f"JSON-RPC error: {parsed['error']!r}")

    @staticmethod
    def tool_payload(result: dict) -> dict:
        content = result.get("content") or []
        if not content or not isinstance(content[0], dict):
            raise AssertionError(f"unexpected tool result shape: {result!r}")
        text = content[0].get("text") or "{}"
        return json.loads(text)


class _RemoteServer:
    def __init__(self, config: RemoteMcpConfig) -> None:
        config.port = 0
        self.server, self.state = build_server(config)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_RemoteServer":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _seed_job(state_dir: Path, goal: str = "e2e remote mcp supervise loop") -> str:
    store = create_store("sqlite", state_dir)
    store.init()
    job = store.create_job(goal)
    task = Task(
        job_id=job.id,
        role="explore",
        instruction="map the surface",
        adapter="cursor",
        status=TaskStatus.COMPLETE,
    )
    store.save_tasks([task])
    store.save_artifact(
        Artifact(
            job_id=job.id,
            task_id=task.id,
            type=ArtifactType.FINDING,
            created_by="e2e-worker",
            payload={"claim": "remote MCP e2e seeded finding"},
            confidence=0.9,
            evidence=["tests/test_mcp_remote_e2e.py"],
        )
    )
    summary_dir = store.job_dir(job.id) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "stitched.md").write_text(
        "# E2E stitched summary\n\nRemote MCP supervise loop OK.\n",
        encoding="utf-8",
    )
    store.update_job_status(job.id, JobStatus.COMPLETE)
    return job.id


class RemoteMcpE2ETests(unittest.TestCase):
    def test_full_supervise_http_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "pm-state"
            state_dir.mkdir()
            job_id = _seed_job(state_dir)

            def fake_start_cli(command, args):
                # Prove the start_* path went through the real tool registry /
                # HTTP dispatch; only the detached launcher is mocked.
                self.assertTrue(command, "start_cli expected a command")
                body = {
                    "job_id": job_id,
                    "run_id": "e2e-run",
                    "command": "python -m puppetmaster " + " ".join(command),
                    "cwd": args.get("cwd") or tmp,
                    "next_steps": [
                        f"Call puppetmaster_status with job_id={job_id}",
                        f"Call puppetmaster_show with job_id={job_id} after completion",
                    ],
                }
                return {
                    "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
                    "isError": False,
                }

            config = RemoteMcpConfig(
                token=TOKEN,
                scope="supervise",
                rate_limit_per_minute=0,  # e2e volume; auth/scope still enforced
            )
            with _RemoteServer(config) as server:
                client = _HttpMcpClient(server.port, TOKEN)

                # --- initialize ---
                init = client.rpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "grok-bot-e2e", "version": "0"},
                    },
                )
                self.assertEqual(init["result"]["serverInfo"]["name"], "puppetmaster-remote")
                self.assertEqual(init["result"]["protocolVersion"], "2025-03-26")
                self.assertEqual(
                    init["result"]["capabilities"]["experimental"]["puppetmasterRemoteScope"],
                    "supervise",
                )
                self.assertIn("listChanged", init["result"]["capabilities"]["tools"])
                self.assertTrue(client.session_id)

                # notifications/initialized → 202
                status, _, _ = client.request(
                    "POST",
                    body={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
                self.assertEqual(status, 202)

                # --- tools/list (supervise) ---
                listed = client.rpc("tools/list")
                names = {tool["name"] for tool in listed["result"]["tools"]}
                self.assertIn("puppetmaster_doctor", names)
                self.assertIn("puppetmaster_start_cursor_swarm", names)
                self.assertIn("puppetmaster_status", names)
                self.assertIn("puppetmaster_show", names)
                for blocked in IMPLEMENT_TOOL_NAMES:
                    self.assertNotIn(blocked, names)
                self.assertGreaterEqual(len(names), 20)
                self.assertLess(len(names), 50)  # full stdio surface is larger

                # --- doctor (real handler path through HTTP → run_cli doctor) ---
                doctor = client.call_tool(
                    "puppetmaster_doctor",
                    {"cwd": tmp, "state_dir": str(state_dir)},
                )
                self.assertFalse(doctor.get("isError"), doctor)
                doctor_body = client.tool_payload(doctor)
                self.assertEqual(doctor_body.get("returncode"), 0)
                self.assertIn("doctor", doctor_body.get("command", ""))

                # --- start_* → job_id (launcher mocked) then real status/show ---
                # Patch only the platform-lock gate (env-dependent) and the
                # detached launcher; write_generated_swarm_config + HTTP/tool
                # dispatch stay on the real path.
                with patch(
                    "puppetmaster.mcp_server._platform_lock_preflight",
                    return_value=None,
                ), patch(
                    "puppetmaster.mcp_server.start_cli",
                    side_effect=fake_start_cli,
                ):
                    started = client.call_tool(
                        "puppetmaster_start_cursor_swarm",
                        {
                            "goal": "e2e remote mcp supervise loop",
                            "cwd": tmp,
                            "state_dir": str(state_dir),
                        },
                    )
                self.assertFalse(started.get("isError"), started)
                start_body = client.tool_payload(started)
                self.assertEqual(start_body["job_id"], job_id)

                status_result = client.call_tool(
                    "puppetmaster_status",
                    {"job_id": job_id, "cwd": tmp, "state_dir": str(state_dir)},
                )
                self.assertFalse(status_result.get("isError"), status_result)
                status_body = client.tool_payload(status_result)
                self.assertEqual(status_body.get("returncode"), 0)
                self.assertIn(job_id, status_body.get("stdout", "") + status_body.get("command", ""))

                show_result = client.call_tool(
                    "puppetmaster_show",
                    {"job_id": job_id, "cwd": tmp, "state_dir": str(state_dir)},
                )
                self.assertFalse(show_result.get("isError"), show_result)
                show_body = client.tool_payload(show_result)
                self.assertEqual(show_body.get("returncode"), 0)
                self.assertIn(
                    "Remote MCP supervise loop OK",
                    show_body.get("stdout", ""),
                )

                # --- auth failure ---
                bad_status, _, bad_body = client.request(
                    "POST",
                    body={"jsonrpc": "2.0", "id": 99, "method": "tools/list"},
                    auth=False,
                )
                self.assertEqual(bad_status, 401)
                self.assertEqual(bad_body["error"], "unauthorized")

                wrong_status, _, wrong_body = client.request(
                    "POST",
                    body={"jsonrpc": "2.0", "id": 100, "method": "tools/list"},
                    token="wrong-token",
                )
                self.assertEqual(wrong_status, 401)
                self.assertEqual(wrong_body["error"], "unauthorized")

                # --- scope deny for implement ---
                denied = client.call_tool(
                    "puppetmaster_start_implement",
                    {"goal": "should be refused", "cwd": tmp, "state_dir": str(state_dir)},
                )
                self.assertTrue(denied.get("isError"))
                denied_body = client.tool_payload(denied)
                self.assertEqual(denied_body.get("code"), "remote_scope_denied")
                self.assertEqual(denied_body.get("tool"), "puppetmaster_start_implement")


class GrokBotHandshakeRegressionTests(unittest.TestCase):
    """Cursor/Grok remote client handshake that was failing in live tunnel PoC.

    Live symptom: initialize HTTP 200 (repeated), tools=0, no tools/list in
    audit. Root causes addressed here: protocolVersion echo, SSE when Accept
    lists text/event-stream, CORS Expose-Headers for Mcp-Session-Id, richer
    tools capability, then initialized → tools/list.
    """

    def test_cursor_grok_streamable_handshake_reaches_tools_list(self) -> None:
        config = RemoteMcpConfig(
            token=TOKEN,
            scope="implement",
            allow_origins=("*",),
            rate_limit_per_minute=0,
        )
        with _RemoteServer(config) as server:
            client = _HttpMcpClient(
                server.port,
                TOKEN,
                accept="application/json, text/event-stream",
                origin="https://grok.example",
            )

            status, headers, init = client.request(
                "POST",
                body={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"roots": {"listChanged": True}},
                        "clientInfo": {"name": "cursor-grok-bot", "version": "1.0.0"},
                    },
                },
            )
            self.assertEqual(status, 200)
            self.assertIn("text/event-stream", headers.get("content-type", ""))
            self.assertIn("mcp-session-id", headers)
            # Browser clients need this to read the session header.
            exposed = headers.get("access-control-expose-headers", "").lower()
            self.assertIn("mcp-session-id", exposed)
            self.assertEqual(headers.get("access-control-allow-origin"), "https://grok.example")
            self.assertEqual(init["result"]["protocolVersion"], "2025-03-26")
            self.assertIsInstance(init["result"]["capabilities"].get("tools"), dict)
            self.assertIn("listChanged", init["result"]["capabilities"]["tools"])
            self.assertTrue(client.session_id)

            status, _, _ = client.request(
                "POST",
                body={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            self.assertEqual(status, 202)

            status, headers, listed = client.request(
                "POST",
                body={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            self.assertEqual(status, 200)
            self.assertIn("result", listed)
            names = {tool["name"] for tool in listed["result"]["tools"]}
            self.assertIn("puppetmaster_doctor", names)
            self.assertIn("puppetmaster_start_implement", names)
            self.assertGreaterEqual(len(names), 45)

    def test_json_only_accept_still_works(self) -> None:
        config = RemoteMcpConfig(token=TOKEN, scope="supervise", rate_limit_per_minute=0)
        with _RemoteServer(config) as server:
            client = _HttpMcpClient(server.port, TOKEN, accept="application/json")
            status, headers, init = client.request(
                "POST",
                body={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "curl", "version": "0"},
                    },
                },
            )
            self.assertEqual(status, 200)
            self.assertIn("application/json", headers.get("content-type", ""))
            self.assertEqual(init["result"]["protocolVersion"], "2025-03-26")
            status, _, _ = client.request(
                "POST",
                body={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            self.assertEqual(status, 202)
            status, _, listed = client.request(
                "POST",
                body={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            self.assertEqual(status, 200)
            self.assertGreater(len(listed["result"]["tools"]), 0)


if __name__ == "__main__":
    unittest.main()
