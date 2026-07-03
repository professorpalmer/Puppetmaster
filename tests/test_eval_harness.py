"""Hermetic tests for the seeded-bug eval harness.

These exercise the harness plumbing -- materialize, apply, score, aggregate --
with fake ``apply_fn`` callables, so no model or network is involved. The
verification commands use ``sys.executable`` rather than a bare ``python`` so
they run regardless of how the CI host names its interpreter.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from puppetmaster.eval_harness import (
    EvalCase,
    builtin_cases,
    format_report,
    materialize_case,
    run_eval,
    score_case,
)


def _case() -> EvalCase:
    verify = (
        f'{sys.executable} -c "import calc; assert calc.add(2, 3) == 5; '
        'assert calc.add(-1, 1) == 0"'
    )
    return EvalCase(
        name="off_by_one_add",
        files={"calc.py": "def add(a, b):\n    return a - b\n"},
        task="Fix add to return the sum.",
        verify=verify,
        intended_files=["calc.py"],
    )


_FIXED = "def add(a, b):\n    return a + b\n"


class EvalHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp(prefix="pmeval_test_"))
        self.addCleanup(lambda: shutil.rmtree(self.work, ignore_errors=True))

    def test_builtin_cases_are_well_formed(self) -> None:
        cases = builtin_cases()
        self.assertTrue(cases)
        for case in cases:
            self.assertTrue(case.name and case.task and case.verify)
            self.assertTrue(case.files)

    def test_unfixed_repo_fails_scoring(self) -> None:
        case = _case()
        repo = materialize_case(case, self.work)
        result = score_case(case, repo)
        self.assertFalse(result.passed)
        self.assertNotEqual(result.verify_returncode, 0)
        self.assertEqual(result.changed_files, [])

    def test_correct_fix_passes_and_is_scoped(self) -> None:
        case = _case()
        repo = materialize_case(case, self.work)
        (repo / "calc.py").write_text(_FIXED, encoding="utf-8")
        result = score_case(case, repo)
        self.assertTrue(result.passed)
        self.assertEqual(result.verify_returncode, 0)
        self.assertEqual(result.changed_files, ["calc.py"])
        self.assertTrue(result.touched_only_intended)
        self.assertGreater(result.added_lines, 0)

    def test_extra_file_marks_not_scoped(self) -> None:
        case = _case()
        repo = materialize_case(case, self.work)
        (repo / "calc.py").write_text(_FIXED, encoding="utf-8")
        (repo / "scratch.txt").write_text("stray\n", encoding="utf-8")
        result = score_case(case, repo)
        self.assertTrue(result.passed)
        self.assertFalse(result.touched_only_intended)

    def test_run_eval_mixed_pass_rate(self) -> None:
        cases = [_case(), _case()]
        cases[1].name = "second"

        def apply(repo: Path, case: EvalCase) -> None:
            # Fix only the first case; leave the second buggy.
            if case.name == "off_by_one_add":
                (repo / "calc.py").write_text(_FIXED, encoding="utf-8")

        report = run_eval(cases, apply, adapter="fake", model=None, workdir=self.work)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 1)
        self.assertAlmostEqual(report.pass_rate, 0.5)
        rendered = format_report(report)
        self.assertIn("Pass rate: 1/2", rendered)

    def test_apply_error_is_recorded_not_raised(self) -> None:
        case = _case()

        def boom(repo: Path, case: EvalCase) -> None:
            raise RuntimeError("adapter blew up")

        report = run_eval([case], boom, adapter="fake", workdir=self.work)
        self.assertEqual(report.passed, 0)
        self.assertIsNotNone(report.results[0].error)
        self.assertIn("adapter blew up", report.results[0].error)


if __name__ == "__main__":
    unittest.main()
