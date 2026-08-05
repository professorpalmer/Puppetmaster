"""ChatGPT Codex OAuth as an agentic worker provider.

Hermetic: no real network. HTTP is stubbed at ``providers._open_stream``.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster import openai_codex as codex
from puppetmaster import providers
from puppetmaster.adapters.agentic import AgenticAdapter, _PROVIDER_ENV_HINTS


def _sse_response(events: list[dict]):
    lines = [
        f"data: {json.dumps(event)}\n".encode("utf-8") for event in events
    ]

    class FakeResp:
        def __iter__(self):
            return iter(lines)

        def close(self):
            pass

    return FakeResp()


def _fake_jwt(*, account_id: str = "acct_test") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        }).encode("utf-8")
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class OpenAICodexNormalizeTests(unittest.TestCase):
    def test_normalize_strips_prefixes_and_dots_gpt(self) -> None:
        cases = (
            ("gpt-5.6-luna", "gpt-5.6-luna"),
            ("gpt-5-6-luna", "gpt-5.6-luna"),
            ("openai-codex:gpt-5.6-luna", "gpt-5.6-luna"),
            ("openai-codex/gpt-5-6-luna", "gpt-5.6-luna"),
            ("codex/gpt-5.6-luna", "gpt-5.6-luna"),
            ("cursor/gpt-5-6-luna", "gpt-5.6-luna"),
            ("agentic/gpt-5.6-luna", "gpt-5.6-luna"),
        )
        for raw, want in cases:
            self.assertEqual(codex.normalize_model_id(raw), want, msg=raw)


class OpenAICodexProviderDescriptorTests(unittest.TestCase):
    def test_provider_is_registered(self) -> None:
        desc = providers.get_provider("openai-codex")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertEqual(desc.api_key_env_vars, ("OPENAI_CODEX_TOKEN",))
        self.assertEqual(desc.base_url, codex.BASE_URL)
        self.assertEqual(desc.label, "ChatGPT Codex (OAuth)")

    def test_available_from_oauth_token(self) -> None:
        self.assertIn(
            "openai-codex",
            providers.available_providers({"OPENAI_CODEX_TOKEN": "tok-test"}),
        )
        self.assertNotIn("openai-codex", providers.available_providers({}))

    def test_agentic_env_hint(self) -> None:
        self.assertEqual(
            _PROVIDER_ENV_HINTS["openai-codex"],
            "OPENAI_CODEX_TOKEN",
        )


class OpenAICodexChatTests(unittest.TestCase):
    def test_provider_chat_always_streams_responses(self) -> None:
        token = _fake_jwt(account_id="acct_123")
        captured: dict = {}
        events = [
            {
                "type": "response.output_text.delta",
                "delta": "hello",
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }],
                },
            },
        ]

        def fake_open_stream(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body, timeout=timeout)
            return _sse_response(events)

        with mock.patch.object(providers, "_open_stream", side_effect=fake_open_stream):
            turn = providers.provider_chat(
                provider="openai-codex",
                model="cursor/gpt-5-6-luna",
                messages=[{"role": "user", "content": "hi"}],
                api_key=token,
            )
        self.assertEqual(captured["url"], f"{codex.BASE_URL}/responses")
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(captured["body"]["model"], "gpt-5.6-luna")
        self.assertEqual(captured["headers"]["originator"], "codex_cli_rs")
        self.assertEqual(captured["headers"]["ChatGPT-Account-ID"], "acct_123")
        self.assertTrue(
            captured["headers"]["Authorization"].startswith("Bearer ")
        )
        self.assertEqual(turn.text, "hello")

    def test_tool_call_turn(self) -> None:
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a.py"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                    "output": [{
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    }],
                },
            },
        ]
        with mock.patch.object(
            providers,
            "_open_stream",
            return_value=_sse_response(events),
        ):
            turn = providers.provider_chat(
                provider="openai-codex",
                model="gpt-5.6-luna",
                messages=[{"role": "user", "content": "read"}],
                api_key="tok",
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
            )
        self.assertEqual(turn.tool_calls[0]["name"], "read_file")
        self.assertEqual(turn.tool_calls[0]["arguments"]["path"], "a.py")

    def test_agentic_adapter_resolves_provider(self) -> None:
        from puppetmaster.models import Task

        adapter = AgenticAdapter()
        task = Task(
            job_id="j1",
            id="t1",
            role="explore",
            instruction="x",
            adapter="agentic",
            payload={"provider": "openai-codex", "model": "gpt-5.6-luna"},
        )
        self.assertEqual(adapter._resolve_provider(task), "openai-codex")


class OpenAICodexCatalogTests(unittest.TestCase):
    def test_curated_catalog_includes_plan_billed_codex(self) -> None:
        from puppetmaster.static_catalog import curated_catalog, reconcile_agentic_catalog

        rows = [
            item
            for item in curated_catalog("agentic")
            if (item.get("payload_defaults") or {}).get("provider") == "openai-codex"
        ]
        self.assertTrue(rows)
        models = {item["model"] for item in rows}
        self.assertIn("gpt-5.6-luna", models)
        self.assertTrue(all(item.get("billing") == "plan" for item in rows))

        with mock.patch(
            "puppetmaster.providers.available_providers",
            return_value={"openai-codex"},
        ):
            merged, report = reconcile_agentic_catalog([])
        self.assertEqual(report["action"], "merged")
        providers_seen = {
            (spec.payload_defaults or {}).get("provider")
            for spec in merged
            if spec.adapter == "agentic"
        }
        self.assertEqual(providers_seen, {"openai-codex"})


if __name__ == "__main__":
    unittest.main()
