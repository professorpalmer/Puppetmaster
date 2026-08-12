"""Tests for the remote (streamable HTTP) MCP transport."""

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
    RateLimiter,
    RemoteMcpConfig,
    build_server,
    config_from_env_and_args,
    connector_snippet,
    extract_bearer_token,
    filtered_tools,
    generate_token,
    handle_remote_message,
    origin_allowed,
    resolve_scope,
    resolve_token,
    token_matches,
    tool_allowed,
)


class TokenAndScopeUnitTests(unittest.TestCase):
    def test_extract_bearer_token(self) -> None:
        self.assertEqual(extract_bearer_token("Bearer abc"), "abc")
        self.assertEqual(extract_bearer_token("bearer abc"), "abc")
        self.assertIsNone(extract_bearer_token("Basic abc"))
        self.assertIsNone(extract_bearer_token(None))
        self.assertIsNone(extract_bearer_token("Bearer"))

    def test_token_matches_uses_constant_time_compare(self) -> None:
        token = generate_token()
        self.assertTrue(token_matches(token, token))
        self.assertFalse(token_matches("nope", token))
        self.assertFalse(token_matches(None, token))
        self.assertFalse(token_matches(token, ""))

    def test_resolve_token_precedence(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "token.txt"
            path.write_text("from-file\n", encoding="utf-8")
            self.assertEqual(
                resolve_token(explicit="flag", token_file=str(path), env={"PUPPETMASTER_MCP_TOKEN": "env"}),
                "flag",
            )
            self.assertEqual(
                resolve_token(explicit=None, token_file=str(path), env={"PUPPETMASTER_MCP_TOKEN": "env"}),
                "from-file",
            )
            self.assertEqual(
                resolve_token(explicit=None, token_file=None, env={"PUPPETMASTER_MCP_TOKEN": "env"}),
                "env",
            )
            self.assertEqual(resolve_token(explicit=None, token_file=None, env={}), "")

    def test_resolve_scope_defaults_supervise(self) -> None:
        self.assertEqual(resolve_scope(explicit=None, env={}), "supervise")
        self.assertEqual(resolve_scope(explicit="IMPLEMENT", env={}), "implement")
        self.assertEqual(
            resolve_scope(explicit=None, env={"PUPPETMASTER_MCP_REMOTE_SCOPE": "implement"}),
            "implement",
        )

    def test_supervise_filters_implement_tools(self) -> None:
        self.assertFalse(tool_allowed("puppetmaster_start_implement", "supervise"))
        self.assertTrue(tool_allowed("puppetmaster_start_implement", "implement"))
        self.assertTrue(tool_allowed("puppetmaster_doctor", "supervise"))
        self.assertTrue(tool_allowed("puppetmaster_start_cursor_swarm", "supervise"))
        names = {tool.name for tool in filtered_tools("supervise")}
        for blocked in IMPLEMENT_TOOL_NAMES:
            self.assertNotIn(blocked, names)
        self.assertIn("puppetmaster_doctor", names)
        self.assertIn("puppetmaster_status", names)

    def test_origin_allowlist(self) -> None:
        self.assertTrue(origin_allowed(None, ()))
        self.assertTrue(origin_allowed("http://127.0.0.1:3000", ()))
        self.assertFalse(origin_allowed("https://evil.example", ()))
        self.assertTrue(origin_allowed("https://evil.example", ("*",)))
        self.assertTrue(origin_allowed("https://app.example", ("https://app.example",)))
        self.assertFalse(origin_allowed("https://other.example", ("https://app.example",)))

    def test_rate_limiter(self) -> None:
        limiter = RateLimiter(2)
        self.assertTrue(limiter.allow("a", now=1000.0))
        self.assertTrue(limiter.allow("a", now=1001.0))
        self.assertFalse(limiter.allow("a", now=1002.0))
        self.assertTrue(limiter.allow("b", now=1002.0))
        self.assertTrue(limiter.allow("a", now=1061.0))

    def test_config_requires_token(self) -> None:
        with self.assertRaises(ValueError):
            RemoteMcpConfig(token="").validate()
        RemoteMcpConfig(token="ok").validate()

    def test_config_from_env_generates_token(self) -> None:
        config, generated = config_from_env_and_args(
            env={}, generate_if_missing=True
        )
        self.assertTrue(generated)
        self.assertTrue(config.token)
        self.assertEqual(config.scope, "supervise")

    def test_connector_snippet(self) -> None:
        snippet = connector_snippet(base_url="http://127.0.0.1:8743", token="secret")
        self.assertEqual(snippet["url"], "http://127.0.0.1:8743/mcp")
        self.assertEqual(snippet["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(snippet["transport"], "streamable-http")


class HandleRemoteMessageTests(unittest.TestCase):
    def test_tools_list_respects_supervise_scope(self) -> None:
        response = handle_remote_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            scope="supervise",
        )
        assert response is not None
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertIn("puppetmaster_doctor", names)
        self.assertNotIn("puppetmaster_start_implement", names)

    def test_tools_call_denied_outside_scope(self) -> None:
        response = handle_remote_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "puppetmaster_start_implement",
                    "arguments": {"goal": "nope"},
                },
            },
            scope="supervise",
        )
        assert response is not None
        result = response["result"]
        self.assertTrue(result.get("isError"))
        body = json.loads(result["content"][0]["text"])
        self.assertEqual(body["code"], "remote_scope_denied")

    def test_initialize_stamps_remote_server_info(self) -> None:
        from puppetmaster.mcp_remote import RemoteMcpState

        state = RemoteMcpState(RemoteMcpConfig(token="test-token-value"))
        response = handle_remote_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            scope="supervise",
            state=state,
        )
        assert response is not None
        self.assertIn("_puppetmaster_session_id", response)
        self.assertEqual(response["result"]["serverInfo"]["name"], "puppetmaster-remote")
        self.assertEqual(response["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(
            response["result"]["capabilities"]["experimental"]["puppetmasterRemoteScope"],
            "supervise",
        )
        self.assertIn("listChanged", response["result"]["capabilities"]["tools"])

    def test_initialize_echoes_requested_protocol_version(self) -> None:
        from puppetmaster.mcp_server import handle_message

        for requested in ("2025-03-26", "2024-11-05", "2025-06-18"):
            response = handle_remote_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": requested, "capabilities": {}},
                },
                scope="supervise",
            )
            assert response is not None
            self.assertEqual(response["result"]["protocolVersion"], requested)
            self.assertEqual(
                response["result"]["capabilities"]["tools"],
                {"listChanged": False},
            )
            # Prove we did not return the stdio handler's hardcoded defaults
            # when the client asked for a streamable-HTTP version.
            if requested != "2024-11-05":
                stdio = handle_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": requested, "capabilities": {}},
                    }
                )
                self.assertEqual(stdio["result"]["protocolVersion"], "2024-11-05")
                self.assertNotEqual(
                    response["result"]["protocolVersion"],
                    stdio["result"]["protocolVersion"],
                )
                self.assertEqual(stdio["result"]["capabilities"]["tools"], {})
                self.assertNotEqual(
                    response["result"]["capabilities"]["tools"],
                    stdio["result"]["capabilities"]["tools"],
                )


class _RemoteServerHarness:
    """Spin up the remote MCP server on an ephemeral port for HTTP tests."""

    def __init__(self, config: RemoteMcpConfig) -> None:
        config.port = 0  # ephemeral
        self.server, self.state = build_server(config)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_RemoteServerHarness":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
        token: Optional[str] = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        # Prefer JSON in unit tests that assert on raw JSON bodies; e2e covers SSE.
        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        if token is not None:
            hdrs["Authorization"] = f"Bearer {token}"
        if payload is not None:
            hdrs["Content-Type"] = "application/json"
            hdrs["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=hdrs)
        response = conn.getresponse()
        data = response.read()
        resp_headers = {k.lower(): v for k, v in response.getheaders()}
        status = response.status
        conn.close()
        return status, resp_headers, data


class RemoteHttpTransportTests(unittest.TestCase):
    def test_health_is_unauthenticated(self) -> None:
        config = RemoteMcpConfig(token="secret-token", scope="supervise")
        with _RemoteServerHarness(config) as harness:
            status, _, raw = harness.request("GET", "/health")
            self.assertEqual(status, 200)
            payload = json.loads(raw.decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["scope"], "supervise")
            self.assertEqual(payload["mcp"], MCP_ENDPOINT_PATH)

    def test_mcp_requires_bearer(self) -> None:
        config = RemoteMcpConfig(token="secret-token")
        with _RemoteServerHarness(config) as harness:
            status, headers, raw = harness.request(
                "POST",
                "/mcp",
                body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(status, 401)
            self.assertIn("www-authenticate", headers)
            payload = json.loads(raw.decode("utf-8"))
            self.assertEqual(payload["error"], "unauthorized")

    def test_streamable_initialize_and_tools_list(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token, scope="supervise")
        with _RemoteServerHarness(config) as harness:
            status, headers, raw = harness.request(
                "POST",
                "/mcp",
                token=token,
                body={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "grok-bot-test", "version": "0"},
                    },
                },
            )
            self.assertEqual(status, 200)
            self.assertIn("mcp-session-id", headers)
            init = json.loads(raw.decode("utf-8"))
            self.assertEqual(init["result"]["serverInfo"]["name"], "puppetmaster-remote")

            status, _, raw = harness.request(
                "POST",
                "/mcp",
                token=token,
                headers={"Mcp-Session-Id": headers["mcp-session-id"]},
                body={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            self.assertEqual(status, 200)
            listed = json.loads(raw.decode("utf-8"))
            names = {tool["name"] for tool in listed["result"]["tools"]}
            self.assertIn("puppetmaster_doctor", names)
            self.assertNotIn("puppetmaster_edit", names)

    def test_x_puppetmaster_token_header(self) -> None:
        token = "alt-header-token"
        config = RemoteMcpConfig(token=token)
        with _RemoteServerHarness(config) as harness:
            status, _, raw = harness.request(
                "POST",
                "/mcp",
                headers={"X-Puppetmaster-Token": token},
                body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(status, 200)
            self.assertIn("result", json.loads(raw.decode("utf-8")))

    def test_origin_denied(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token, allow_origins=("https://allowed.example",))
        with _RemoteServerHarness(config) as harness:
            status, _, raw = harness.request(
                "POST",
                "/mcp",
                token=token,
                headers={"Origin": "https://evil.example"},
                body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(status, 403)
            self.assertEqual(json.loads(raw.decode("utf-8"))["error"], "origin_denied")

    def test_rate_limit_returns_429(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token, rate_limit_per_minute=2)
        with _RemoteServerHarness(config) as harness:
            for _ in range(2):
                status, _, _ = harness.request(
                    "POST",
                    "/mcp",
                    token=token,
                    body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
                self.assertEqual(status, 200)
            status, headers, raw = harness.request(
                "POST",
                "/mcp",
                token=token,
                body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(status, 429)
            self.assertEqual(headers.get("retry-after"), "60")
            self.assertEqual(json.loads(raw.decode("utf-8"))["error"], "rate_limited")

    def test_notification_returns_202(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token)
        with _RemoteServerHarness(config) as harness:
            status, _, raw = harness.request(
                "POST",
                "/mcp",
                token=token,
                body={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            self.assertEqual(status, 202)
            self.assertEqual(raw, b"")

    def test_tools_call_doctor_through_http(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token, scope="supervise")
        fake_doctor = {
            "content": [{"type": "text", "text": json.dumps({"ok": True})}],
        }
        with _RemoteServerHarness(config) as harness:
            with patch("puppetmaster.mcp_server.call_tool", return_value=fake_doctor):
                status, _, raw = harness.request(
                    "POST",
                    "/mcp",
                    token=token,
                    body={
                        "jsonrpc": "2.0",
                        "id": 9,
                        "method": "tools/call",
                        "params": {"name": "puppetmaster_doctor", "arguments": {}},
                    },
                )
            self.assertEqual(status, 200)
            payload = json.loads(raw.decode("utf-8"))
            self.assertEqual(payload["id"], 9)
            self.assertIn("result", payload)

    def test_legacy_sse_endpoint_event(self) -> None:
        token = "secret-token"
        config = RemoteMcpConfig(token=token, legacy_sse_hold_seconds=0.2)
        with _RemoteServerHarness(config) as harness:
            status, headers, raw = harness.request("GET", "/sse", token=token)
            self.assertEqual(status, 200)
            self.assertIn("text/event-stream", headers.get("content-type", ""))
            text = raw.decode("utf-8")
            self.assertIn("event: endpoint", text)
            self.assertIn("/message?sessionId=", text)

    def test_print_connector_cli_path(self) -> None:
        from puppetmaster.cli.commands_mcp import _run_mcp_serve_remote

        args = type(
            "Args",
            (),
            {
                "host": "127.0.0.1",
                "port": 8743,
                "token": "cli-token",
                "token_file": None,
                "scope": "supervise",
                "allow_origins": None,
                "rate_limit": None,
                "audit_log": None,
                "print_token": False,
                "print_connector": True,
            },
        )()
        import io
        import sys

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = _run_mcp_serve_remote(args)
        finally:
            sys.stdout = old
        self.assertEqual(code, 0)
        snippet = json.loads(buf.getvalue())
        self.assertEqual(snippet["headers"]["Authorization"], "Bearer cli-token")
        self.assertTrue(snippet["url"].endswith("/mcp"))


if __name__ == "__main__":
    unittest.main()
