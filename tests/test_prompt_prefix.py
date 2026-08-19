"""Static-first worker prompt assembly (cross-worker prompt-cache prefix)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster.adapters._prompts import (
    TASK_INSTRUCTION_HEADER,
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    prompt_with_skills,
    split_prompt_messages,
    with_repo_census,
)
from puppetmaster.models import Task

def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

class BuildPromptOrderTests(unittest.TestCase):
    def test_build_implement_prompt_puts_boilerplate_before_instruction(self) -> None:
        instruction = "Fix the flaky auth test"
        prompt = build_implement_prompt(instruction)
        self.assertTrue(prompt.startswith("Implement mode:"))
        self.assertIn(TASK_INSTRUCTION_HEADER, prompt)
        self.assertTrue(prompt.rstrip().endswith(instruction))
        self.assertLess(
            prompt.index("Implement mode:"),
            prompt.index(TASK_INSTRUCTION_HEADER),
        )
        self.assertLess(
            prompt.index(TASK_INSTRUCTION_HEADER),
            prompt.rindex(instruction),
        )

    def test_build_structured_prompt_puts_contract_before_instruction(self) -> None:
        instruction = "Review the payment module"
        prompt = build_structured_prompt(instruction)
        # The contract still precedes the instruction (static-first, so sibling
        # workers share a cacheable prefix) -- it is just no longer the first
        # thing the model reads. See the orientation test below.
        self.assertTrue(prompt.rstrip().endswith(instruction))
        self.assertLess(
            prompt.index("Puppetmaster artifact contract"),
            prompt.index(TASK_INSTRUCTION_HEADER),
        )
        note = build_structured_prompt(instruction, final_message_note=True)
        self.assertTrue(note.rstrip().endswith(instruction))
        self.assertIn("submit_findings", note)
        self.assertLess(note.index("submit_findings"), note.index(TASK_INSTRUCTION_HEADER))

    def test_structured_prompt_opens_by_locating_the_task(self) -> None:
        """gpt-5.6 models answer *about* the first noun phrase they read.

        Opening on "Puppetmaster artifact contract:" made them treat the
        contract as the subject: one asked what to do with the contract,
        another spent 51k input tokens explaining what it is. Four of four
        Codex analysis runs degraded that way, across two models. So the
        prompt must open by saying where the assignment lives and that nothing
        above it is the subject, and must label the contract as output format.
        """
        for prompt in (
            build_structured_prompt("Review the payment module"),
            build_structured_prompt("Review the payment module", final_message_note=True),
        ):
            first_line = prompt.splitlines()[0]
            self.assertIn("Read this entire prompt before acting", first_line)
            self.assertIn("Your task", first_line)
            # The orientation must NOT contain the bare TASK_INSTRUCTION_HEADER:
            # split_prompt_messages() cuts the system/user seam at its first
            # occurrence, so an inline mention would collapse the shared prefix.
            self.assertNotIn(TASK_INSTRUCTION_HEADER, first_line)
            self.assertEqual(prompt.count(TASK_INSTRUCTION_HEADER), 1)
            self.assertLess(
                prompt.index("Read this entire prompt"),
                prompt.index("Puppetmaster artifact contract"),
            )
            self.assertIn("OUTPUT FORMAT", prompt)
            # `with_report_contract` keys off this literal to avoid stacking a
            # second contract; the relabel must not break that detection.
            self.assertIn("Puppetmaster artifact contract", prompt)

    def test_report_contract_still_detects_the_relabelled_contract(self) -> None:
        from puppetmaster.adapters._prompts import with_report_contract

        structured = build_structured_prompt("Review the payment module")
        self.assertEqual(with_report_contract(structured), structured)

    def test_split_prompt_messages_keeps_orientation_in_system_prefix(self) -> None:
        from puppetmaster.adapters._prompts import _PROMPT_ORIENTATION

        system_prefix, user_suffix = split_prompt_messages(
            build_structured_prompt("Review the payment module")
        )
        self.assertTrue(system_prefix.startswith(_PROMPT_ORIENTATION))
        self.assertTrue(user_suffix.startswith(TASK_INSTRUCTION_HEADER))
        self.assertNotIn(_PROMPT_ORIENTATION, user_suffix)

class JobStableSectionOrderTests(unittest.TestCase):
    def test_memory_skills_census_appear_before_instruction(self) -> None:
        instruction = "UNIQUE_TASK_INSTRUCTION_XYZ"
        base = build_implement_prompt(instruction)
        task = Task(
            job_id="j",
            role="implement",
            instruction=instruction,
            payload={
                "retrieved_memory": [
                    {"scope": "swarm.decisions", "statement": "prefer sqlite store"},
                ],
                "injected_skills": [
                    {"name": "ship-checks", "body": "Always run focused tests before submit."},
                ],
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.py").write_text("x = 1\n", encoding="utf-8")
            assembled = prompt_with_skills(
                prompt_with_memory(with_repo_census(base, tmp), task),
                task,
            )
        marker = assembled.index(TASK_INSTRUCTION_HEADER)
        self.assertLess(assembled.index("Repository file census"), marker)
        self.assertLess(
            assembled.index("Relevant promoted Puppetmaster memory"),
            marker,
        )
        self.assertLess(assembled.index("ship-checks"), marker)
        self.assertTrue(assembled.rstrip().endswith(instruction))

class SharedPrefixPropertyTests(unittest.TestCase):
    def test_sibling_tasks_share_long_static_prefix(self) -> None:
        task_a = "Implement feature A: add retry helper"
        task_b = "Implement feature B: add circuit breaker"
        memory = [
            {"scope": "swarm.decisions", "statement": "use pathlib for all paths"},
            {"scope": "swarm.decisions", "statement": "prefer unittest over pytest plugins"},
        ]
        skills = [
            {
                "name": "verify-first",
                "body": "Run the focused test subset before calling submit_report.",
            }
        ]

        # Same job-stable inputs: pad census via a shared temp dir for both.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "lib.py").write_text("def f(): pass\n", encoding="utf-8")
            for i in range(20):
                (root / f"mod_{i:02d}.py").write_text(f"x{i}=1\n", encoding="utf-8")

            def assemble_in(instruction: str) -> str:
                task = Task(
                    job_id="job-shared",
                    role="implement",
                    instruction=instruction,
                    payload={"retrieved_memory": memory, "injected_skills": skills},
                )
                return prompt_with_skills(
                    prompt_with_memory(
                        with_repo_census(build_implement_prompt(instruction), tmp),
                        task,
                    ),
                    task,
                )

            prompt_a = assemble_in(task_a)
            prompt_b = assemble_in(task_b)

        shared = _common_prefix_len(prompt_a, prompt_b)
        self.assertGreaterEqual(shared, 1000)
        self.assertTrue(prompt_a.rstrip().endswith(task_a))
        self.assertTrue(prompt_b.rstrip().endswith(task_b))
        self.assertNotEqual(prompt_a, prompt_b)

class CodegraphMemoizationTests(unittest.TestCase):
    def test_identical_context_calls_invoke_cli_once(self) -> None:
        from puppetmaster import codegraph

        codegraph._codegraph_context_cached.cache_clear()
        completed = mock.Mock(
            returncode=0,
            stdout="auth.py:1 -> login()\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".codegraph").mkdir()
            with mock.patch.object(codegraph, "codegraph_ready", return_value=True), mock.patch.object(
                codegraph, "resolve_codegraph_invocation", return_value=["codegraph"]
            ), mock.patch.object(
                codegraph.subprocess, "run", return_value=completed
            ) as run, mock.patch.object(codegraph, "_record_codegraph_usage"):
                first = codegraph.codegraph_context("map auth", tmp, max_nodes=15)
                second = codegraph.codegraph_context("map auth", tmp, max_nodes=15)
            self.assertEqual(first, second)
            self.assertEqual(first, "auth.py:1 -> login()")
            self.assertEqual(run.call_count, 1)
        codegraph._codegraph_context_cached.cache_clear()

class AgenticMessageSplitTests(unittest.TestCase):
    def test_agent_loop_uses_system_prefix_and_user_instruction(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.providers import AssistantTurn

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name)

        instruction = "UNIQUE_AGENTIC_TASK_INSTRUCTION"
        system_prompt = build_implement_prompt(instruction)
        system_prefix, user_suffix = split_prompt_messages(system_prompt)
        self.assertTrue(system_prefix.startswith("Implement mode:"))
        self.assertIn(TASK_INSTRUCTION_HEADER, user_suffix)
        self.assertTrue(user_suffix.rstrip().endswith(instruction))

        seen_messages = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen_messages.append(list(messages))
            return AssistantTurn(
                text="done",
                tool_calls=[],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        task = Task(
            job_id="j",
            role="implement",
            instruction=instruction,
            payload={
                "cwd": str(cwd),
                "provider": "openai",
                "model": "m",
                "disable_codegraph": True,
                "max_turns": 1,
            },
        )
        adapter = agentic.AgenticAdapter()
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            adapter._agent_loop(
                task,
                cwd,
                "openai",
                "m",
                system_prompt,
                tools=[],
                implement=True,
            )

        self.assertTrue(seen_messages)
        first = seen_messages[0]
        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(first[0]["content"], system_prefix)
        self.assertEqual(first[1]["role"], "user")
        self.assertEqual(first[1]["content"], user_suffix)

class EnrichCodegraphOrderTests(unittest.TestCase):
    def test_codegraph_section_lands_before_task_instruction(self) -> None:
        from puppetmaster.codegraph import enrich_prompt_with_codegraph

        instruction = "Wire up caching"
        base = build_implement_prompt(instruction)
        with mock.patch(
            "puppetmaster.codegraph.codegraph_context",
            return_value="adapters/_prompts.py:42 -> build_implement_prompt",
        ), mock.patch(
            "puppetmaster.codegraph.codegraph_freshness",
            return_value=None,
        ):
            enriched, used = enrich_prompt_with_codegraph(
                base,
                task_description=instruction,
                cwd=".",
            )
        self.assertTrue(used)
        cg = enriched.index("Shared CodeGraph context for this task:")
        task = enriched.index(TASK_INSTRUCTION_HEADER)
        self.assertLess(cg, task)
        self.assertTrue(enriched.rstrip().endswith(instruction))

if __name__ == "__main__":
    unittest.main()
