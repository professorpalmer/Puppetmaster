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
        self.assertEqual(captured["body"]["system"], "be terse")
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
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0].payload["result"], "failed")

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
