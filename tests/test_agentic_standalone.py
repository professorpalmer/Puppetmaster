"""Tests for the standalone direct-API agentic worker path.

Hermetic: no real network. Provider HTTP is stubbed at ``providers._post_json``
and the adapter's model turns are stubbed at ``agentic.provider_chat``, so the
whole stack (registry -> key-aware routing -> adapter tool loop -> artifacts)
is exercised without a key or a socket.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster import providers
from puppetmaster.models import ArtifactType, Task


class ProviderRegistryTests(unittest.TestCase):
    def test_available_providers_reflects_present_keys(self) -> None:
        env = {"ANTHROPIC_API_KEY": "sk-a", "GEMINI_API_KEY": "g"}
        available = providers.available_providers(env)
        self.assertIn("anthropic", available)
        self.assertIn("gemini", available)
        self.assertNotIn("openrouter", available)
        self.assertNotIn("groq", available)

    def test_api_key_precedence_first_set_wins(self) -> None:
        desc = providers.get_provider("gemini")
        self.assertEqual(providers.resolve_api_key(desc, {"GOOGLE_API_KEY": "goog"}), "goog")
        self.assertEqual(
            providers.resolve_api_key(desc, {"GEMINI_API_KEY": "gem", "GOOGLE_API_KEY": "goog"}),
            "gem",
        )

    def test_keyless_local_provider_needs_presence_env(self) -> None:
        ollama = providers.get_provider("ollama")
        self.assertFalse(providers.is_available(ollama, {}))
        self.assertTrue(providers.is_available(ollama, {"OLLAMA_HOST": "http://x:11434"}))

    def test_base_url_override(self) -> None:
        desc = providers.get_provider("openai")
        self.assertEqual(
            providers.resolve_base_url(desc, {"OPENAI_BASE_URL": "https://proxy/v1/"}),
            "https://proxy/v1",
        )

    def test_unknown_provider_is_none(self) -> None:
        self.assertIsNone(providers.get_provider("nope"))


class ProviderChatTests(unittest.TestCase):
    def test_openai_wire_normalizes_text_and_tools(self) -> None:
        canned = {
            "choices": [{
                "message": {
                    "content": "hi",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
        with mock.patch.object(providers, "_post_json", return_value=canned):
            turn = providers.provider_chat(
                provider="openai", model="gpt-x", messages=[{"role": "user", "content": "go"}],
                api_key="k",
            )
        self.assertEqual(turn.text, "hi")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0]["name"], "read_file")
        self.assertEqual(turn.tool_calls[0]["arguments"], {"path": "a.py"})
        self.assertEqual(turn.usage["total_tokens"], 13)

    def test_anthropic_wire_normalizes_blocks(self) -> None:
        canned = {
            "content": [
                {"type": "text", "text": "analysis"},
                {"type": "tool_use", "id": "tu_1", "name": "edit_file", "input": {"path": "b.py"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 8},
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["url"] = url
            captured["body"] = body
            return canned

        with mock.patch.object(providers, "_post_json", side_effect=_capture):
            turn = providers.provider_chat(
                provider="anthropic", model="claude", api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "go"},
                ],
                tools=[{"type": "function", "function": {"name": "edit_file", "description": "d", "parameters": {"type": "object"}}}],
            )
        self.assertTrue(captured["url"].endswith("/messages"))
        # Prompt-cache path wraps system as a text block with cache_control.
        system = captured["body"]["system"]
        if isinstance(system, list):
            self.assertEqual(system[0]["type"], "text")
            self.assertEqual(system[0]["text"], "be terse")
            self.assertEqual(
                system[0].get("cache_control"),
                {"type": "ephemeral", "ttl": "1h"},
            )
        else:
            self.assertEqual(system, "be terse")
        self.assertEqual(captured["body"]["tools"][0]["name"], "edit_file")
        self.assertEqual(turn.text, "analysis")
        self.assertEqual(turn.tool_calls[0]["name"], "edit_file")
        self.assertEqual(turn.usage["total_tokens"], 28)

    def test_http_error_becomes_provider_error(self) -> None:
        with mock.patch.object(
            providers, "_post_json",
            side_effect=providers.ProviderError("boom", reason="http_status:401", status=401),
        ):
            with self.assertRaises(providers.ProviderError) as ctx:
                providers.provider_chat(provider="openai", model="m", messages=[], api_key="k")
        self.assertEqual(ctx.exception.status, 401)

    def test_missing_key_for_keyed_provider_raises(self) -> None:
        with self.assertRaises(providers.ProviderError) as ctx:
            providers.provider_chat(
                provider="anthropic", model="m", messages=[], env={},
            )
        self.assertEqual(ctx.exception.reason, "not_authenticated")


class KeyAwareRoutingTests(unittest.TestCase):
    def _agentic_registry(self):
        from puppetmaster.static_catalog import curated_to_specs
        return curated_to_specs("agentic", "api", [])

    def test_router_drops_models_whose_provider_key_is_absent(self) -> None:
        from puppetmaster.router import TaskSignals, route_task

        signals = TaskSignals(
            role="agentic-implement",
            instruction="Fix a small typo in the docs",
            allowed_adapters={"agentic"},
        )
        with mock.patch("puppetmaster.providers.available_providers", return_value={"anthropic"}):
            decision = route_task(signals, self._agentic_registry(), policy="balanced")
        self.assertEqual(decision.model.adapter, "agentic")
        self.assertEqual(decision.model.payload_defaults["provider"], "anthropic")

    def test_router_errors_when_no_provider_key_available(self) -> None:
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        signals = TaskSignals(
            role="agentic-implement",
            instruction="Fix a small typo",
            allowed_adapters={"agentic"},
        )
        with mock.patch("puppetmaster.providers.available_providers", return_value=set()):
            with self.assertRaises(NoEligibleModelError):
                route_task(signals, self._agentic_registry(), policy="balanced")

    def test_agentic_analyze_role_routes_under_platform_lock(self) -> None:
        # Regression pin: under an agentic-only platform lock, an analyze role
        # routes to an agentic model (cheap by design -- the native submit-tool
        # channel makes even a cheap model reliably structured).
        from puppetmaster.router import TaskSignals, route_task

        signals = TaskSignals(
            role="review",
            instruction="review this repository for risks and produce findings",
            allowed_adapters={"agentic"},
        )
        with mock.patch(
            "puppetmaster.providers.available_providers",
            return_value={"gemini", "anthropic", "openai"},
        ):
            decision = route_task(signals, self._agentic_registry(), policy="balanced")
        self.assertEqual(decision.model.adapter, "agentic")

    def test_min_capability_escalates_agentic_route(self) -> None:
        # The documented escape hatch: min_capability lifts the pick to a
        # higher-capability agentic model without a blanket floor bump.
        from puppetmaster.router import TaskSignals, route_task

        base = TaskSignals(role="review", instruction="review this repo",
                           allowed_adapters={"agentic"})
        pinned = TaskSignals(role="review", instruction="review this repo",
                             allowed_adapters={"agentic"}, explicit_min_capability=95)
        with mock.patch(
            "puppetmaster.providers.available_providers",
            return_value={"gemini", "anthropic", "openai"},
        ):
            low = route_task(base, self._agentic_registry(), policy="balanced")
            high = route_task(pinned, self._agentic_registry(), policy="balanced")
        self.assertGreaterEqual(high.model.capability_score, low.model.capability_score)


class ContextCompressionTests(unittest.TestCase):
    def test_compress_history_elides_old_tool_output_keeps_recent(self) -> None:
        from puppetmaster.adapters._context_budget import compress_history, estimate_message_tokens

        big = "x" * 5000
        messages = [{"role": "user", "content": "system prompt"}]
        for i in range(6):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": big})

        before = estimate_message_tokens(messages)
        out, changed = compress_history(messages, budget_tokens=before // 2, keep_recent=2)

        self.assertTrue(changed)
        self.assertLessEqual(estimate_message_tokens(out), before)
        # The most recent tool output is preserved verbatim; an old one is elided.
        self.assertEqual(out[-1]["content"], big)
        self.assertTrue(
            any(
                "compacted" in (m.get("content") or "") or "elided" in (m.get("content") or "")
                for m in out
            )
        )
        # Structure intact: still one tool message per assistant tool_call.
        self.assertEqual(sum(1 for m in out if m.get("role") == "tool"), 6)

    def test_compress_history_noop_under_budget(self) -> None:
        from puppetmaster.adapters._context_budget import compress_history

        messages = [{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "c0", "content": "small"}]
        out, changed = compress_history(messages, budget_tokens=10_000)
        self.assertFalse(changed)
        self.assertEqual(out[-1]["content"], "small")


class ProviderKeyPoolTests(unittest.TestCase):
    def test_provider_key_pool_unions_and_numbers(self) -> None:
        from puppetmaster.providers import provider_key_pool

        env = {"GEMINI_API_KEY": "a", "GEMINI_API_KEY_2": "c", "GOOGLE_API_KEY": "b"}
        pool = provider_key_pool("gemini", env=env)
        # base var + its numbered sibling first, then the next descriptor var.
        self.assertEqual(pool, ["a", "c", "b"])

    def test_provider_key_pool_dedupes_and_empty_for_keyless(self) -> None:
        from puppetmaster.providers import provider_key_pool

        env = {"OPENAI_API_KEY": "dup", "OPENAI_API_KEY_2": "dup"}
        self.assertEqual(provider_key_pool("openai", env=env), ["dup"])
        self.assertEqual(provider_key_pool("ollama", env={}), [])


class ProviderStreamingTests(unittest.TestCase):
    def test_openai_streaming_assembles_text_and_usage(self) -> None:
        from puppetmaster import providers

        lines = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n',
            b'data: [DONE]\n',
        ]

        class FakeResp:
            def __iter__(self):
                return iter(lines)

            def close(self):
                pass

        got = []
        with mock.patch.object(providers, "_open_stream", return_value=FakeResp()):
            turn = providers.provider_chat_streaming(
                provider="openai", model="m", messages=[], api_key="k",
                on_delta=lambda kind, text: got.append((kind, text)),
            )
        self.assertEqual(turn.text, "Hello")
        self.assertEqual(turn.finish_reason, "stop")
        self.assertEqual(turn.usage["total_tokens"], 5)
        self.assertEqual(got, [("text", "Hel"), ("text", "lo")])

    def test_openai_streaming_assembles_tool_call_arguments(self) -> None:
        from puppetmaster import providers

        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"submit_findings","arguments":"{\\"artifacts\\""}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":[]}"}}]}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
            b'data: [DONE]\n',
        ]

        class FakeResp:
            def __iter__(self):
                return iter(lines)

            def close(self):
                pass

        with mock.patch.object(providers, "_open_stream", return_value=FakeResp()):
            turn = providers.provider_chat_streaming(
                provider="openai", model="m", messages=[], api_key="k",
            )
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0]["name"], "submit_findings")
        self.assertEqual(turn.tool_calls[0]["arguments"], {"artifacts": []})


class AgenticToolTests(unittest.TestCase):
    def setUp(self) -> None:
        from puppetmaster.adapters.agentic import AgenticAdapter
        self.adapter = AgenticAdapter()
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        (self.cwd / "a.py").write_text("line1\nline2\nSECRET_MATCH\n", encoding="utf-8")
        (self.cwd / "sub").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_file_and_confinement(self) -> None:
        out = self.adapter._execute_tool("read_file", {"path": "a.py"}, self.cwd, False, _task())
        self.assertIn("line1", out)
        escaped = self.adapter._execute_tool("read_file", {"path": "../../etc/hosts"}, self.cwd, False, _task())
        self.assertIn("escapes the workspace", escaped)

    def test_read_file_hashline_tagged(self) -> None:
        out = self.adapter._execute_tool("read_file", {"path": "a.py"}, self.cwd, False, _task())
        self.assertRegex(out, r"^\[a\.py#[0-9A-F]{4}\]")
        self.assertIn("1:line1", out)
        self.assertIn("2:line2", out)

    def test_apply_hashline_tool(self) -> None:
        read_out = self.adapter._execute_tool("read_file", {"path": "a.py"}, self.cwd, True, _task())
        import re
        m = re.match(r"\[a\.py#([0-9A-F]{4})\]", read_out)
        self.assertIsNotNone(m)
        tag = m.group(1)
        patch = f"[a.py#{tag}]\nSWAP 1.=1:\n+LINE_ONE\n"
        out = self.adapter._execute_tool(
            "apply_hashline", {"patch": patch}, self.cwd, True, _task(),
        )
        self.assertTrue(out.startswith("ok:"), out)
        self.assertTrue((self.cwd / "a.py").read_text(encoding="utf-8").startswith("LINE_ONE"))

    def test_apply_hashline_stale_rejected(self) -> None:
        self.adapter._execute_tool("read_file", {"path": "a.py"}, self.cwd, True, _task())
        out = self.adapter._execute_tool(
            "apply_hashline",
            {"patch": "[a.py#FFFF]\nDEL 1\n"},
            self.cwd, True, _task(),
        )
        self.assertIn("error:", out)
        self.assertIn("line1", (self.cwd / "a.py").read_text(encoding="utf-8"))

    def test_search_code(self) -> None:
        out = self.adapter._execute_tool("search_code", {"query": "SECRET_MATCH"}, self.cwd, False, _task())
        self.assertIn("a.py:3", out)

    def test_edit_file_uniqueness_guard(self) -> None:
        (self.cwd / "dup.py").write_text("x\nx\n", encoding="utf-8")
        out = self.adapter._execute_tool(
            "edit_file", {"path": "dup.py", "old_string": "x", "new_string": "y"},
            self.cwd, True, _task(),
        )
        self.assertIn("not unique", out)

    def test_write_and_edit_apply(self) -> None:
        self.adapter._execute_tool(
            "write_file", {"path": "sub/new.py", "content": "hello"}, self.cwd, True, _task()
        )
        self.assertEqual((self.cwd / "sub" / "new.py").read_text(), "hello")
        self.adapter._execute_tool(
            "edit_file", {"path": "sub/new.py", "old_string": "hello", "new_string": "world"},
            self.cwd, True, _task(),
        )
        self.assertEqual((self.cwd / "sub" / "new.py").read_text(), "world")

    def test_write_tool_unavailable_in_analyze_mode(self) -> None:
        out = self.adapter._execute_tool(
            "write_file", {"path": "x", "content": "y"}, self.cwd, False, _task()
        )
        self.assertIn("not available", out)

    def test_edit_file_replace_all(self) -> None:
        (self.cwd / "dup.py").write_text("x\nx\n", encoding="utf-8")
        out = self.adapter._execute_tool(
            "edit_file",
            {"path": "dup.py", "old_string": "x", "new_string": "y", "replace_all": True},
            self.cwd, True, _task(),
        )
        self.assertIn("2 replacements", out)
        self.assertEqual((self.cwd / "dup.py").read_text(), "y\ny\n")

    def test_delete_file(self) -> None:
        (self.cwd / "gone.py").write_text("bye", encoding="utf-8")
        out = self.adapter._execute_tool(
            "delete_file", {"path": "gone.py"}, self.cwd, True, _task()
        )
        self.assertIn("deleted", out)
        self.assertFalse((self.cwd / "gone.py").exists())

    def test_delete_file_unavailable_in_analyze_mode(self) -> None:
        (self.cwd / "keep.py").write_text("stay", encoding="utf-8")
        out = self.adapter._execute_tool(
            "delete_file", {"path": "keep.py"}, self.cwd, False, _task()
        )
        self.assertIn("not available", out)
        self.assertTrue((self.cwd / "keep.py").exists())

    def test_run_terminal_refuses_destructive_command(self) -> None:
        out = self.adapter._execute_tool(
            "run_terminal", {"command": "rm -rf /"}, self.cwd, True,
            _task(payload={"allow_terminal": True}),
        )
        self.assertIn("destructive", out)

    def test_run_terminal_allows_benign_command(self) -> None:
        out = self.adapter._execute_tool(
            "run_terminal", {"command": "echo hello"}, self.cwd, True,
            _task(payload={"allow_terminal": True}),
        )
        self.assertIn("hello", out)

    def test_binary_write_refused(self) -> None:
        out = self.adapter._execute_tool(
            "write_file", {"path": "b.bin", "content": "a\x00b"}, self.cwd, True, _task()
        )
        self.assertIn("NUL", out)
        self.assertFalse((self.cwd / "b.bin").exists())


class AgenticLoopTests(unittest.TestCase):
    def test_analyze_loop_feeds_tool_results_then_parses_artifacts(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        (cwd / "calc.py").write_text("def add(a,b): return a-b\n", encoding="utf-8")

        turns = [
            AssistantTurn(text="", tool_calls=[{"id": "c1", "name": "read_file", "arguments": {"path": "calc.py"}}], usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),
            AssistantTurn(
                text=json.dumps({"artifacts": [{"type": "finding", "summary": "add subtracts", "confidence": 0.9, "evidence": ["calc.py"]}]}),
                tool_calls=[], usage={"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
            ),
        ]
        seen_messages = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen_messages.append(list(messages))
            return turns[len(seen_messages) - 1]

        task = Task(
            job_id="j", role="explore", instruction="check calc",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "claude-haiku-4-5", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        # Second call must include the tool result for read_file with the file body.
        second = seen_messages[1]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        self.assertTrue(any("def add" in m["content"] for m in tool_msgs))

        types = [a.type for a in arts]
        self.assertIn(ArtifactType.VERIFICATION, types)
        self.assertIn(ArtifactType.FINDING, types)
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertEqual(verif.payload["turns"], 2)

    def test_cancel_stops_loop_before_next_turn(self) -> None:
        from puppetmaster import cancellation
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        job_id = "job-cancel-test"
        self.addCleanup(cancellation.clear_cancel, job_id)
        calls = {"n": 0}

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            calls["n"] += 1
            # Cancel lands after the first turn; the loop must not call again.
            cancellation.request_cancel(job_id)
            return AssistantTurn(
                text="",
                tool_calls=[{"id": "c1", "name": "list_dir", "arguments": {"path": "."}}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        task = Task(
            job_id=job_id, role="explore", instruction="x",
            payload={"cwd": tmp.name, "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(calls["n"], 1)
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["stop_reason"], "cancelled")

    def test_cancelled_delta_sink_aborts_stream(self) -> None:
        from puppetmaster import cancellation

        job_id = "job-stream-cancel"
        self.addCleanup(cancellation.clear_cancel, job_id)
        self.assertFalse(cancellation.is_cancelled(job_id))
        cancellation.request_cancel(job_id)
        self.assertTrue(cancellation.is_cancelled(job_id))
        cancellation.clear_cancel(job_id)
        self.assertFalse(cancellation.is_cancelled(job_id))

    def test_provider_error_yields_failed_verification(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        task = Task(
            job_id="j", role="explore", instruction="x",
            payload={"cwd": tmp.name, "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(
            agentic, "provider_chat",
            side_effect=ProviderError("401", reason="http_status:401", status=401, body="bad key"),
        ):
            arts = self.adapter().run(task, task.instruction, "w1")
        # A 401 yields the failed verification PLUS a loud, dedicated auth-failure
        # RISK so a dead/revoked key is diagnosable at a glance (not laundered into
        # a generic "no structured findings" degrade).
        self.assertEqual(len(arts), 2)
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "failed")
        risk = next(a for a in arts if a.type == ArtifactType.RISK)
        self.assertEqual(risk.payload["failure"], "auth_failed:401")
        self.assertEqual(risk.payload["provider"], "anthropic")
        self.assertIn("AUTH FAILURE", risk.payload["risk"])
        self.assertIn("ANTHROPIC_API_KEY", risk.payload["mitigation"])

    def test_preflight_not_authenticated_yields_auth_risk(self) -> None:
        # A missing/blank key raises ProviderError(reason="not_authenticated")
        # with status None (no HTTP call). The auth-failure RISK must still fire.
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        task = Task(
            job_id="j", role="explore", instruction="x",
            payload={"cwd": tmp.name, "provider": "openai", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(
            agentic, "provider_chat",
            side_effect=ProviderError("no key", reason="not_authenticated"),
        ):
            arts = self.adapter().run(task, task.instruction, "w1")
        risk = next(a for a in arts if a.type == ArtifactType.RISK)
        self.assertEqual(risk.payload["failure"], "auth_failed:401")
        self.assertIn("OPENAI_API_KEY", risk.payload["mitigation"])

    def test_analyze_json_only_retry_recovers_a_prose_run(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)

        turns = [
            AssistantTurn(text="Here is a prose answer, not JSON.", tool_calls=[],
                          usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
            AssistantTurn(
                text=json.dumps({"artifacts": [{"type": "finding", "claim": "x", "confidence": 0.8, "evidence": ["a.py"]}]}),
                tool_calls=[], usage={"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
            ),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(list(messages))
            return turns[len(seen) - 1]

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        # The second call must carry the JSON-only retry directive.
        self.assertTrue(any("single JSON object" in str(m) for m in seen[1]))
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertIn("retry:recovered", verif.evidence)
        self.assertIn(ArtifactType.FINDING, [a.type for a in arts])

    def test_implement_noop_is_degraded_and_nudged(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        turns = [
            AssistantTurn(text="I would change foo.py.", tool_calls=[],
                          usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            AssistantTurn(text="Still just describing, no edits.", tool_calls=[],
                          usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(list(messages))
            return turns[min(len(seen) - 1, len(turns) - 1)]

        task = Task(
            job_id="j", role="build", instruction="implement a thing",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "degraded")
        self.assertIn("nudge:applied", verif.evidence)
        self.assertTrue(any(a.type == ArtifactType.RISK for a in arts))
        # The nudge message must have been injected on the second turn.
        self.assertTrue(any("without changing any files" in str(m) for m in seen[1]))

    def test_implement_writes_file_passes_and_emits_patch(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        turns = [
            AssistantTurn(
                text="", tool_calls=[{"id": "c1", "name": "write_file",
                                      "arguments": {"path": "new.py", "content": "print('hi')\n"}}],
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            ),
            AssistantTurn(text="Added new.py with a hello print.", tool_calls=[],
                          usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[len(seen) - 1]

        task = Task(
            job_id="j", role="build", instruction="add new.py",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertTrue(verif.payload["has_work"])
        self.assertTrue(any(a.type == ArtifactType.PATCH for a in arts))
        self.assertTrue((cwd / "new.py").exists())

    def test_analyze_submit_findings_tool_produces_artifacts(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)

        turns = [
            AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {
                    "artifacts": [{"type": "finding", "claim": "x calls y",
                                   "evidence": ["a.py"], "confidence": 0.8}]}}],
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            ),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[len(seen) - 1]

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertEqual(verif.payload["stop_reason"], "submitted")
        self.assertIn("submit:tool", verif.evidence)
        self.assertIn(ArtifactType.FINDING, [a.type for a in arts])

    def test_analyze_empty_submission_is_clean_pass_not_degraded(self) -> None:
        # An explicit empty submission ("I found nothing") is an honest pass, not
        # a degrade -- this is the core false-degrade fix.
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        turns = [
            AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            ),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[len(seen) - 1]

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertIsNone(verif.payload["failure"])
        self.assertIn("submit:tool", verif.evidence)
        self.assertFalse(any(a.type == ArtifactType.RISK for a in arts))
        self.assertFalse(any(a.type == ArtifactType.FINDING for a in arts))

    def test_analyze_unstructured_prose_degrades_after_retry(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        turns = [
            AssistantTurn(text="Just prose.", tool_calls=[], finish_reason="stop",
                          usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            AssistantTurn(text="Still prose after the retry.", tool_calls=[], finish_reason="stop",
                          usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[min(len(seen) - 1, len(turns) - 1)]

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "degraded")
        self.assertEqual(verif.payload["failure"], "empty_or_unstructured_agentic_result")
        self.assertIn("retry:exhausted", verif.evidence)
        self.assertTrue(any(a.type == ArtifactType.RISK for a in arts))

    def test_analyze_retry_forces_submit_tool(self) -> None:
        # After the structure retry, the next turn must FORCE the submit tool via
        # tool_choice so a compliant model can't wander back into prose.
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        turns = [
            AssistantTurn(text="prose, no structure", tool_calls=[], finish_reason="stop",
                          usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {
                    "artifacts": [{"type": "finding", "claim": "recovered", "evidence": ["a.py"], "confidence": 0.7}]}}],
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            ),
        ]
        seen_extra = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen_extra.append(extra)
            return turns[len(seen_extra) - 1]

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertIsNone(seen_extra[0].get("force_tool"))
        self.assertEqual(seen_extra[1].get("force_tool"), "submit_findings")
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertIn("retry:recovered", verif.evidence)
        self.assertIn("submit:tool", verif.evidence)

    def test_provider_call_retries_transient_then_succeeds(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn, ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        calls = {"n": 0}

        def flaky(*, provider, model, messages, tools, extra, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderError("timed out", reason="timeout")
            return AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            )

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=flaky), \
                mock.patch("time.sleep", lambda *_: None):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(calls["n"], 2)  # one transient failure, one success
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")

    def test_provider_call_does_not_retry_terminal_auth_error(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        calls = {"n": 0}

        def auth_fail(*, provider, model, messages, tools, extra, timeout):
            calls["n"] += 1
            raise ProviderError("401", reason="http_status:401", status=401, body="bad key")

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=auth_fail), \
                mock.patch("time.sleep", lambda *_: None):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(calls["n"], 1)  # terminal: no retry
        self.assertEqual(arts[0].payload["result"], "failed")

    def test_implement_submit_report_records_report_and_patches(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        turns = [
            AssistantTurn(
                text="", tool_calls=[{"id": "c1", "name": "write_file",
                                      "arguments": {"path": "new.py", "content": "print('hi')\n"}}],
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            ),
            AssistantTurn(
                text="", tool_calls=[{"id": "r1", "name": "submit_report", "arguments": {
                    "summary": "Added new.py", "files_changed": ["new.py"],
                    "verification": "ran python new.py"}}],
                usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            ),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[len(seen) - 1]

        task = Task(
            job_id="j", role="build", instruction="add new.py",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertEqual(verif.payload["stop_reason"], "submitted")
        self.assertTrue(verif.payload["has_work"])
        self.assertTrue(any(a.type == ArtifactType.PATCH for a in arts))
        self.assertTrue(
            any(a.type == ArtifactType.FINDING and "Added new.py" in str(a.payload) for a in arts)
        )

    def test_credential_rotation_on_rate_limit(self) -> None:
        import os
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn, ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        used_keys = []

        def flaky(*, provider, model, messages, tools, extra, timeout, api_key=None):
            used_keys.append(api_key)
            # First attempt (api_key=None) uses provider_chat's own resolution of
            # the throttled primary key; the rotation passes the explicit second.
            if api_key is None:
                raise ProviderError("429", reason="http_status:429", status=429)
            return AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            )

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k1", "ANTHROPIC_API_KEY_2": "k2"}, clear=False), \
                mock.patch.object(agentic, "provider_chat", side_effect=flaky):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(used_keys, [None, "k2"])  # rotated to the explicit second key
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")

    def test_model_failover_on_terminal_error(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn, ProviderError

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        calls = []

        def by_model(*, provider, model, messages, tools, extra, timeout, api_key=None):
            calls.append((provider, model))
            if model == "primary":
                raise ProviderError("500", reason="http_status:500", status=500)
            return AssistantTurn(
                text="",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            )

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "primary",
                     "disable_codegraph": True, "provider_max_retries": 0,
                     "failover_models": [{"provider": "openai", "model": "backup"}]},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=by_model), \
                mock.patch("time.sleep", lambda *_: None):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(calls[-1], ("openai", "backup"))
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertIn("failover:used", verif.evidence)

    def test_streaming_sink_receives_deltas(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.adapters._delta_bus import register_delta_sink, unregister_delta_sink
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)
        got = []
        register_delta_sink("w1", lambda kind, text: got.append((kind, text)))
        self.addCleanup(lambda: unregister_delta_sink("w1"))

        def fake_stream(*, provider, model, messages, tools, extra, timeout, on_delta):
            on_delta("text", "hel")
            on_delta("text", "lo")
            return AssistantTurn(
                text="hello",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )

        task = Task(
            job_id="j", role="explore", instruction="analyze",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat_streaming", side_effect=fake_stream):
            arts = self.adapter().run(task, task.instruction, "w1")

        self.assertEqual(got, [("text", "hel"), ("text", "lo")])
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")

    def test_durable_delta_writer_roundtrip_and_no_statedir(self) -> None:
        import os
        from puppetmaster.adapters._delta_stream import (
            DurableDeltaWriter, iter_deltas, delta_file_path,
        )

        task = _task({})
        # No state dir in scope -> no durable writer (hermetic default).
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PUPPETMASTER_STATE_DIR", None)
            self.assertIsNone(DurableDeltaWriter.for_task(task, "w1"))

        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        with mock.patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": state.name}):
            writer = DurableDeltaWriter.for_task(task, "w1")
            self.assertIsNotNone(writer)
            writer.emit("reasoning", "think ")
            writer.emit("text", "answer")
            writer.close()
            path = delta_file_path(Path(state.name), task.job_id, task.id)
            records = list(iter_deltas(path))
        self.assertEqual([r["kind"] for r in records], ["reasoning", "text"])
        self.assertEqual("".join(r["text"] for r in records), "think answer")
        self.assertTrue(all(r["worker_id"] == "w1" for r in records))

    def test_durable_delta_stream_persists_a_run(self) -> None:
        import os
        from puppetmaster.adapters import agentic
        from puppetmaster.adapters._delta_stream import iter_deltas, delta_file_path
        from puppetmaster.providers import AssistantTurn

        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)

        def fake_stream(*, provider, model, messages, tools, extra, timeout, on_delta):
            on_delta("text", "hel")
            on_delta("text", "lo")
            return AssistantTurn(
                text="hello",
                tool_calls=[{"id": "s1", "name": "submit_findings", "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )

        task = Task(
            job_id="jd", role="explore", instruction="analyze",
            payload={"cwd": work.name, "provider": "anthropic", "model": "m",
                     "disable_codegraph": True},
        )
        # No in-process sink registered: the durable writer alone must drive
        # streaming, proving a subprocess worker (state dir, no callback) streams.
        with mock.patch.dict(os.environ, {"PUPPETMASTER_STATE_DIR": state.name}), \
                mock.patch.object(agentic, "provider_chat_streaming", side_effect=fake_stream):
            self.adapter().run(task, task.instruction, "w1")

        path = delta_file_path(Path(state.name), task.job_id, task.id)
        self.assertTrue(path.exists())
        records = list(iter_deltas(path))
        self.assertEqual("".join(r["text"] for r in records), "hello")

    def test_run_deltas_follow_streams_persisted_records(self) -> None:
        import io
        import contextlib
        from types import SimpleNamespace
        from puppetmaster.adapters._delta_stream import _DELTA_FILE
        from puppetmaster.cli import run_deltas_follow

        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        task_dir = Path(state.name) / "jobs" / "j1" / "tasks" / "t1"
        task_dir.mkdir(parents=True)
        (task_dir / _DELTA_FILE).write_text(
            json.dumps({"ts": 1.0, "worker_id": "w1", "kind": "text", "text": "Hello "})
            + "\n"
            + json.dumps({"ts": 2.0, "worker_id": "w1", "kind": "text", "text": "world"})
            + "\n",
            encoding="utf-8",
        )
        store = SimpleNamespace(root=state.name)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_deltas_follow(store, "j1", follow=False)
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "Hello world")

    def test_coerce_plan_steps_is_tolerant(self) -> None:
        from puppetmaster.adapters.agentic import _coerce_plan_steps

        steps = _coerce_plan_steps({"steps": [
            "bare string",
            {"step": "typed", "status": "in_progress"},
            {"step": "bad status", "status": "wat"},
            {"title": "alt key"},
            {"status": "done"},  # no step -> dropped
        ]})
        self.assertEqual(
            steps,
            [
                {"step": "bare string", "status": "pending"},
                {"step": "typed", "status": "in_progress"},
                {"step": "bad status", "status": "pending"},
                {"step": "alt key", "status": "pending"},
            ],
        )

    def test_plan_tool_absent_when_disabled(self) -> None:
        adapter = self.adapter()
        on = adapter._tool_schema(implement=True, task=_task({}))
        off = adapter._tool_schema(implement=True, task=_task({"plan_tool": False}))
        names_on = {t["function"]["name"] for t in on}
        names_off = {t["function"]["name"] for t in off}
        self.assertIn("update_plan", names_on)
        self.assertNotIn("update_plan", names_off)

    def test_plan_tool_emits_decision_artifact(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        turns = [
            AssistantTurn(text="", tool_calls=[{"id": "p1", "name": "update_plan",
                          "arguments": {"steps": [
                              {"step": "write new.py", "status": "in_progress"},
                              {"step": "verify", "status": "pending"}]}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "c1", "name": "write_file",
                          "arguments": {"path": "new.py", "content": "print('hi')\n"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "p2", "name": "update_plan",
                          "arguments": {"steps": [
                              {"step": "write new.py", "status": "done"},
                              {"step": "verify", "status": "done"}]}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "r1", "name": "submit_report",
                          "arguments": {"summary": "added new.py"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(list(messages))
            return turns[min(len(seen) - 1, len(turns) - 1)]

        task = Task(
            job_id="j", role="build", instruction="add new.py",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertIn("plan:2", verif.evidence)
        decision = next(a for a in arts if a.type == ArtifactType.DECISION)
        # The runtime validates every artifact on save; a plan DECISION must
        # satisfy the DECISION contract (decision + why) or the worker fails.
        decision.validate()
        self.assertTrue(decision.payload["why"])
        self.assertEqual(decision.payload["plan"][0]["status"], "done")
        self.assertIn("[x] write new.py", decision.payload["plan_rendered"])
        # The plan tool must have been acked as a checklist mid-run.
        self.assertTrue(any("Plan updated" in str(m) for m in seen[1]))

    def test_detect_verify_command_recognizes_pytest_and_npm(self) -> None:
        from puppetmaster.adapters.agentic import _detect_verify_command

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.assertIsNone(_detect_verify_command(root))

        (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        self.assertEqual(_detect_verify_command(root), "python -m pytest -q")
        (root / "pytest.ini").unlink()

        (root / "tests").mkdir()
        (root / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
        self.assertEqual(_detect_verify_command(root), "python -m pytest -q")

        node = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(node, ignore_errors=True))
        (node / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}), encoding="utf-8")
        self.assertEqual(_detect_verify_command(node), "npm test --silent")

    def test_verify_evidence_tag_covers_modes(self) -> None:
        from puppetmaster.adapters.agentic import _verify_evidence_tag

        self.assertEqual(_verify_evidence_tag({"mode": "skipped", "command": None}), "skipped")
        self.assertEqual(
            _verify_evidence_tag({"mode": "gating", "command": "x", "passed": True}), "passed")
        self.assertEqual(
            _verify_evidence_tag({"mode": "gating", "command": "x", "passed": False}), "failed")
        self.assertEqual(
            _verify_evidence_tag({"mode": "advisory", "command": "x", "passed": False}),
            "advisory-failed")

    def test_resolve_verify_command_explicit_and_disabled(self) -> None:
        adapter = self.adapter()
        cwd = Path(".")
        explicit = _task({"verify_command": "make check"})
        self.assertEqual(adapter._resolve_verify_command(explicit, cwd), "make check")
        disabled = _task({"verify": False})
        self.assertIsNone(adapter._resolve_verify_command(disabled, cwd))
        locked = _task({"verify_command": "make check", "allow_terminal": False})
        self.assertIsNone(adapter._resolve_verify_command(locked, cwd))

    def test_implement_verify_bounces_failure_then_accepts_on_pass(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        # Real verification: passes only once the sentinel file exists. Baseline
        # off forces gating so a first (pre-fix) submit is genuinely bounced.
        turns = [
            AssistantTurn(text="", tool_calls=[{"id": "r0", "name": "submit_report",
                          "arguments": {"summary": "done?"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "c1", "name": "write_file",
                          "arguments": {"path": "FIXED", "content": "ok\n"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "r1", "name": "submit_report",
                          "arguments": {"summary": "created sentinel"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(list(messages))
            return turns[min(len(seen) - 1, len(turns) - 1)]

        # Portable existence check (POSIX `test -f` is not a Windows shell builtin).
        verify = 'python -c "import sys; sys.exit(0 if __import__(\'pathlib\').Path(\'FIXED\').is_file() else 1)"'
        task = Task(
            job_id="j", role="build", instruction="create sentinel",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True,
                     "verify_command": verify, "verify_baseline": False},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertTrue(verif.payload["verification_passed"])
        self.assertEqual(verif.payload["verification_attempts"], 2)
        self.assertIn("verify:passed", verif.evidence)
        self.assertTrue(any("does not pass verification" in str(m) for m in seen[-1]))
        self.assertTrue((cwd / "FIXED").exists())

    def test_implement_verify_failure_after_retries_degrades(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        turns = [
            AssistantTurn(text="", tool_calls=[{"id": "c1", "name": "write_file",
                          "arguments": {"path": "new.py", "content": "print('x')\n"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "r1", "name": "submit_report",
                          "arguments": {"summary": "try 1"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "r2", "name": "submit_report",
                          "arguments": {"summary": "try 2"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[min(len(seen) - 1, len(turns) - 1)]

        task = Task(
            job_id="j", role="build", instruction="make change",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True,
                     "verify_command": "false", "verify_baseline": False,
                     "verify_retries": 1},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "degraded")
        self.assertEqual(verif.payload["failure"], "verification_failed")
        self.assertTrue(verif.payload["has_work"])
        self.assertIn("verify:failed", verif.evidence)
        self.assertTrue(any(
            a.type == ArtifactType.RISK and "result:verify-failed" in a.evidence for a in arts))

    def test_implement_verify_advisory_does_not_degrade_a_diff(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        cwd = _git_repo(self)
        # Baseline on + a command that is red on the clean tree => advisory: the
        # failure is not attributable to this change, so a real diff still passes.
        turns = [
            AssistantTurn(text="", tool_calls=[{"id": "c1", "name": "write_file",
                          "arguments": {"path": "new.py", "content": "print('x')\n"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            AssistantTurn(text="", tool_calls=[{"id": "r1", "name": "submit_report",
                          "arguments": {"summary": "did work"}}],
                          usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
        ]
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return turns[min(len(seen) - 1, len(turns) - 1)]

        task = Task(
            job_id="j", role="build", instruction="make change",
            payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                     "mode": "implement", "disable_codegraph": True,
                     "verify_command": "false"},
        )
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = self.adapter().run(task, task.instruction, "w1")

        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verif.payload["result"], "passed")
        self.assertEqual(verif.payload["verification_mode"], "advisory")
        self.assertIn("verify:advisory-failed", verif.evidence)

    def adapter(self):
        from puppetmaster.adapters.agentic import AgenticAdapter
        return AgenticAdapter()


def _git_repo(test) -> Path:
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    cwd = Path(tmp.name)
    env = {**__import__("os").environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init"], ["add", "-A"], ["commit", "-m", "init", "--allow-empty"]):
        subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, check=False)
    (cwd / "seed.py").write_text("seed = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(cwd), env=env, capture_output=True, check=False)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=str(cwd), env=env, capture_output=True, check=False)
    return cwd


def _task(payload: dict = None) -> Task:
    return Task(job_id="j", role="explore", instruction="i", payload=payload or {})


if __name__ == "__main__":
    unittest.main()
