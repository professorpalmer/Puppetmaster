"""Job-level shared CodeGraph / repo brief (cross-worker identical prefix)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster.adapters._prompts import (
    TASK_INSTRUCTION_HEADER,
    build_implement_prompt,
    split_prompt_messages,
    with_job_brief,
    with_repo_census,
)
from puppetmaster.job_brief import (
    JOB_BRIEF_SECTION_HEADER,
    build_job_brief,
    job_brief_enabled,
    read_job_brief,
    write_job_brief,
)
from puppetmaster.models import Task


class JobBriefBuildTests(unittest.TestCase):
    def test_missing_codegraph_still_produces_census_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print(1)\n", encoding="utf-8")
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                side_effect=RuntimeError("codegraph unavailable"),
            ):
                brief = build_job_brief("ship the feature", root)
        self.assertIn(JOB_BRIEF_SECTION_HEADER, brief)
        self.assertIn("Repository file census", brief)
        self.assertIn("app.py", brief)
        self.assertNotIn("Shared CodeGraph context for this task:", brief)

    def test_kill_switch_disables_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.py").write_text("x=1\n", encoding="utf-8")
            job_dir = root / "jobs" / "job-1"
            with mock.patch.dict(os.environ, {"PUPPETMASTER_JOB_BRIEF": "0"}):
                self.assertFalse(job_brief_enabled())
                path = write_job_brief(job_dir, "goal", root)
            self.assertIsNone(path)
            self.assertFalse((job_dir / "repo_brief.md").exists())


class JobBriefInjectionTests(unittest.TestCase):
    def test_sibling_tasks_receive_identical_brief_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib.py").write_text("def f(): pass\n", encoding="utf-8")
            job_dir = root / "jobs" / "shared-job"
            with mock.patch(
                "puppetmaster.codegraph.codegraph_context",
                return_value="lib.py:1 -> f()",
            ):
                written = write_job_brief(job_dir, "map the library", root)
            self.assertIsNotNone(written)
            brief_bytes = read_job_brief(job_dir)
            self.assertTrue(brief_bytes)
            self.assertIn(JOB_BRIEF_SECTION_HEADER, brief_bytes)

            def assemble(instruction: str) -> str:
                task = Task(
                    job_id="shared-job",
                    role="implement",
                    instruction=instruction,
                    payload={"job_brief": brief_bytes, "cwd": str(root)},
                )
                return with_job_brief(build_implement_prompt(instruction), task)

            prompt_a = assemble("Implement feature A")
            prompt_b = assemble("Implement feature B")

        self.assertIn(brief_bytes.strip(), prompt_a)
        self.assertIn(brief_bytes.strip(), prompt_b)
        # Identical brief segment appears before each task's divergent instruction.
        idx_a = prompt_a.index(JOB_BRIEF_SECTION_HEADER)
        idx_b = prompt_b.index(JOB_BRIEF_SECTION_HEADER)
        end_a = prompt_a.index(TASK_INSTRUCTION_HEADER)
        end_b = prompt_b.index(TASK_INSTRUCTION_HEADER)
        self.assertEqual(prompt_a[idx_a:end_a], prompt_b[idx_b:end_b])

    def test_brief_sits_before_task_instruction(self) -> None:
        instruction = "UNIQUE_JOB_BRIEF_TASK"
        brief = (
            f"{JOB_BRIEF_SECTION_HEADER}\n\n"
            "Repository file census (ground truth -- 1 file(s) under the "
            "working directory): main.py.\n"
        )
        task = Task(
            job_id="j",
            role="implement",
            instruction=instruction,
            payload={"job_brief": brief},
        )
        prompt = with_job_brief(build_implement_prompt(instruction), task)
        self.assertLess(
            prompt.index(JOB_BRIEF_SECTION_HEADER),
            prompt.index(TASK_INSTRUCTION_HEADER),
        )
        self.assertTrue(prompt.rstrip().endswith(instruction))
        system, user = split_prompt_messages(prompt)
        self.assertIn(JOB_BRIEF_SECTION_HEADER, system)
        self.assertIn(TASK_INSTRUCTION_HEADER, user)
        self.assertNotIn(JOB_BRIEF_SECTION_HEADER, user)

    def test_kill_switch_skips_injection(self) -> None:
        brief = f"{JOB_BRIEF_SECTION_HEADER}\n\ncensus here\n"
        task = Task(
            job_id="j",
            role="implement",
            instruction="do the thing",
            payload={"job_brief": brief},
        )
        base = build_implement_prompt("do the thing")
        with mock.patch.dict(os.environ, {"PUPPETMASTER_JOB_BRIEF": "0"}):
            assembled = with_job_brief(base, task)
        self.assertEqual(assembled, base)
        self.assertNotIn(JOB_BRIEF_SECTION_HEADER, assembled)

    def test_job_brief_suppresses_duplicate_census(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("a=1\n", encoding="utf-8")
            brief = (
                f"{JOB_BRIEF_SECTION_HEADER}\n\n"
                "Repository file census (ground truth -- 1 file(s) under the "
                "working directory): a.py.\n"
            )
            task = Task(
                job_id="j",
                role="explore",
                instruction="look around",
                payload={"job_brief": brief, "cwd": str(root)},
            )
            prompt = with_repo_census(
                with_job_brief(build_implement_prompt("look around"), task),
                root,
            )
        self.assertEqual(prompt.count("Repository file census"), 1)


if __name__ == "__main__":
    unittest.main()
