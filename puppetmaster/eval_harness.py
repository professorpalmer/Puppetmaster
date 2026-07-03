"""Seeded-bug evaluation harness for Puppetmaster implement adapters.

Objective measurement, not vibes: a curated set of small repos each carry one
seeded bug and a verification command that passes *only* once the bug is fixed.
The harness materializes each case as a throwaway git repo, hands it to an
adapter to fix, then scores the result by running the case's own verification
command and measuring the diff. It reports a pass-rate plus per-case diff
quality, so the agentic adapter can be compared head-to-head against the
cursor / claude-code adapters on identical tasks.

Design notes:

- **Adapter-agnostic.** The core scorer takes an ``apply_fn(repo)`` callable, so
  the same cases score any adapter (or a fake, in tests). :func:`adapter_apply_fn`
  wires a real Puppetmaster implement worker via the Orchestrator.
- **No global-pytest dependency.** Each case verifies with a self-contained
  ``python -c`` assertion, so the harness runs anywhere regardless of the host's
  pytest/plugin state.
- **Ground-truth scoring.** Pass/fail is the case's verification command re-run
  by the harness after the edit -- independent of whatever the adapter claims.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "pmeval", "GIT_AUTHOR_EMAIL": "eval@puppetmaster",
    "GIT_COMMITTER_NAME": "pmeval", "GIT_COMMITTER_EMAIL": "eval@puppetmaster",
}
_VERIFY_TIMEOUT_SECONDS = 120


@dataclass
class EvalCase:
    """One seeded-bug fixture."""

    name: str
    files: dict[str, str]
    task: str
    verify: str
    intended_files: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    name: str
    passed: bool
    verify_returncode: Optional[int]
    changed_files: list[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0
    touched_only_intended: Optional[bool] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalReport:
    adapter: str
    model: Optional[str]
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0


def builtin_cases() -> list[EvalCase]:
    """The default curated bug set. Small, deterministic, language-pure Python
    so the seeded bug is the only variable and scoring is unambiguous."""
    return [
        EvalCase(
            name="off_by_one_add",
            files={"calc.py": "def add(a, b):\n    # BUG: subtracts instead of adds\n    return a - b\n"},
            task="add(a, b) in calc.py must return the sum of a and b, but it returns the difference. Fix it.",
            verify='python -c "import calc; assert calc.add(2, 3) == 5; assert calc.add(-1, 1) == 0"',
            intended_files=["calc.py"],
        ),
        EvalCase(
            name="wrong_parity_condition",
            files={"parity.py": "def is_even(n):\n    # BUG: wrong comparison\n    return n % 2 == 1\n"},
            task="is_even(n) in parity.py should return True when n is even, but the condition is inverted. Fix it.",
            verify='python -c "import parity; assert parity.is_even(4); assert not parity.is_even(3); assert parity.is_even(0)"',
            intended_files=["parity.py"],
        ),
        EvalCase(
            name="missing_return",
            files={"acc.py": "def total(nums):\n    result = 0\n    for n in nums:\n        result += n\n    # BUG: forgot to return result\n"},
            task="total(nums) in acc.py computes a running sum but never returns it, so it returns None. Fix it to return the sum.",
            verify='python -c "import acc; assert acc.total([1, 2, 3]) == 6; assert acc.total([]) == 0"',
            intended_files=["acc.py"],
        ),
    ]


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(
        ["git", *args], cwd=str(repo), env=env,
        capture_output=True, text=True, check=False,
    )


def materialize_case(case: EvalCase, root: Path) -> Path:
    """Write the case's buggy files into a fresh git repo under ``root`` and
    commit them, so a fix produces a clean, measurable diff."""
    repo = root / case.name
    repo.mkdir(parents=True, exist_ok=True)
    for rel, content in case.files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(["init", "-q"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "seed: buggy fixture"], repo)
    return repo


def _run_verify(case: EvalCase, repo: Path) -> tuple[bool, Optional[int], str]:
    try:
        proc = subprocess.run(
            case.verify, shell=True, cwd=str(repo), capture_output=True,
            text=True, timeout=_VERIFY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, None, "verification timed out"
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, proc.returncode, output


def _is_build_noise(path: str) -> bool:
    """Byproducts a verify run leaves behind (e.g. importing a module writes a
    ``.pyc``) -- not part of the fix, so they must not count against diff scope."""
    return (
        "__pycache__/" in path
        or path.endswith((".pyc", ".pyo"))
        or ".pytest_cache/" in path
    )


def _diff_stats(repo: Path) -> tuple[list[str], int, int]:
    _git(["add", "-A"], repo)
    proc = _git(["diff", "--cached", "--numstat"], repo)
    changed: list[str] = []
    added = removed = 0
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, r, path = parts
        if _is_build_noise(path):
            continue
        changed.append(path)
        added += int(a) if a.isdigit() else 0
        removed += int(r) if r.isdigit() else 0
    return changed, added, removed


def score_case(case: EvalCase, repo: Path) -> CaseResult:
    """Score an already-edited repo: run the case's verification and measure the
    diff. Pure and side-effect-free beyond reading the repo."""
    passed, rc, _out = _run_verify(case, repo)
    changed, added, removed = _diff_stats(repo)
    touched_only = None
    if case.intended_files:
        touched_only = set(changed).issubset(set(case.intended_files))
    return CaseResult(
        name=case.name, passed=passed, verify_returncode=rc,
        changed_files=changed, added_lines=added, removed_lines=removed,
        touched_only_intended=touched_only,
    )


def run_case(
    case: EvalCase, apply_fn: "Callable[[Path, EvalCase], None]", *, workdir: Path
) -> CaseResult:
    """Materialize ``case``, apply an adapter's edits via ``apply_fn``, then score."""
    started = time.monotonic()
    repo = materialize_case(case, workdir)
    try:
        apply_fn(repo, case)
    except Exception as exc:  # noqa: BLE001 - a failed apply is a case failure, not a crash
        result = score_case(case, repo)
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_seconds = time.monotonic() - started
        return result
    result = score_case(case, repo)
    result.elapsed_seconds = time.monotonic() - started
    return result


def run_eval(
    cases: list[EvalCase],
    apply_fn: "Callable[[Path, EvalCase], None]",
    *,
    adapter: str,
    model: Optional[str] = None,
    workdir: Optional[Path] = None,
) -> EvalReport:
    """Run every case through ``apply_fn`` and collect a scored report."""
    owns_workdir = workdir is None
    workdir = workdir or Path(tempfile.mkdtemp(prefix="pmeval_"))
    try:
        results = [run_case(case, apply_fn, workdir=workdir) for case in cases]
    finally:
        if owns_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    return EvalReport(adapter=adapter, model=model, results=results)


def adapter_apply_fn(
    *,
    adapter: str = "agentic",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    use_verify_loop: bool = True,
) -> "Callable[[Path, EvalCase], None]":
    """An ``apply_fn`` that runs a real Puppetmaster implement worker on the repo.

    The worker fixes the seeded bug in-place; the harness then scores it
    independently. When ``use_verify_loop`` is set, the case's own verification
    command is handed to the agentic verify-before-submit loop as well, so the
    adapter gets the same self-correction signal a real user would configure.
    """
    from puppetmaster.orchestrator import Orchestrator
    from puppetmaster.store_factory import create_store
    from puppetmaster.workers import WorkerSpec

    def apply(repo: Path, case: EvalCase) -> None:
        payload: dict = {
            "mode": "implement", "cwd": str(repo),
            "prompt": case.task, "disable_codegraph": True,
        }
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
            payload["adapter_model_name"] = model
        if use_verify_loop and adapter == "agentic":
            payload["verify_command"] = case.verify
            payload["verify_baseline"] = False
        spec = WorkerSpec(
            role="implement", instruction=case.task, adapter=adapter, payload=payload
        )
        state_dir = Path(tempfile.mkdtemp(prefix="pmeval_state_"))
        try:
            store = create_store("sqlite", str(state_dir))
            Orchestrator(store).run(case.task, specs=[spec], worker_mode="inline")
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    return apply


def format_report(report: EvalReport) -> str:
    """A compact, human-readable results table plus the headline pass-rate."""
    lines = [
        f"Eval: adapter={report.adapter} model={report.model or '(default)'}",
        f"Pass rate: {report.passed}/{report.total} ({report.pass_rate * 100:.0f}%)",
        "",
        f"{'case':<26} {'pass':<5} {'files':<6} {'+/-':<10} {'scoped':<7} {'sec':<6}",
        "-" * 68,
    ]
    for r in report.results:
        scoped = "-" if r.touched_only_intended is None else ("yes" if r.touched_only_intended else "no")
        lines.append(
            f"{r.name:<26} {('PASS' if r.passed else 'fail'):<5} "
            f"{len(r.changed_files):<6} {f'+{r.added_lines}/-{r.removed_lines}':<10} "
            f"{scoped:<7} {r.elapsed_seconds:<6.1f}"
        )
        if r.error:
            lines.append(f"    error: {r.error}")
    return "\n".join(lines)
