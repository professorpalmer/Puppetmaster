"""Non-bypassable completion gates for worker tasks.

Puppetmaster solves the *memory* half of long-horizon agent work with durable
state. Gates solve the other half: an agent can't mark its task COMPLETE just
because it *thinks* it finished. A gate is an objective, machine-checkable
post-condition the runtime evaluates *after* the agent runs and *before* the
task is allowed to reach COMPLETE. Fail a gate → the task is FAILED, loudly.

Gate kinds (all opt-in via ``task.payload["gates"]`` or convenience flags):

- ``require_diff``  — the run must have produced a non-empty diff / PATCH.
                      Catches the "implement reported COMPLETE with 0 patches"
                      trap for genuine edit tasks.
- ``command``       — run a shell command in ``cwd``; exit 0 is required.
                      The expensive semantic oracle (tests, a parity audit).
- ``ratchet``       — run a command that prints ``{"metrics": {...}}`` (or a
                      flat ``{metric: value}``) on stdout; the chosen metric may
                      only hold or shrink vs. a runtime-owned baseline. The
                      cheap monotonic gate that can only tighten. The baseline
                      is written by the runtime, never the worker, and only ever
                      downward — so the agent can't loosen its own bar.
- ``committed``     — after the run the tree must have no uncommitted changes
                      (the worker committed its work). Optionally the runtime
                      performs the commit itself with a fixed author/message.
- ``review``        — a *strictly stronger* model judges the run's diff before
                      it is allowed to COMPLETE. This is the semantic
                      quality bar that ``require_diff`` (only "files changed")
                      and ``command`` (only "the tests you wrote pass") can't
                      provide: an LLM-as-judge reading the actual patch for
                      correctness, scope, regressions, and readability. It is
                      the answer to "a cheap routed model wrote this — who
                      checks the work?": the judge is always at least one
                      capability tier above the implementer, so mass
                      auto-routed swarms can't land code no stronger model
                      would sign off on. The live judge call is opt-in behind
                      ``$PUPPETMASTER_REVIEW_GATE`` (it makes an extra model
                      call, so it's a deliberate, flagged spend); when the flag
                      is off, or no adequate judge exists, the gate is a no-op
                      rather than bricking every implement run.

Layering, cheap-to-expensive: ``require_diff`` (free) → ``ratchet`` (one
command) → ``command`` (your test suite) → ``review`` (one judge call). Each
covers the one below's blind spot — gaming a scalar metric gets caught by the
semantic command; a green suite over thin tests gets caught by the judge — and
the expensive gates only run when the cheap ones already passed, so a no-op or
test-failing task never pays for a judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from puppetmaster.models import Artifact, ArtifactType, Task, now_iso

if TYPE_CHECKING:
    from puppetmaster.model_registry import ModelSpec
    from puppetmaster.store import SwarmStore

_GATE_COMMAND_TIMEOUT = 1800

# Review-gate tunables. The judge sees a bounded slice of the diff (a judge
# doesn't need the whole 50k-line refactor to spot that a function is wrong),
# and gets its own timeout so a slow judge can't wedge a worker.
_REVIEW_TIMEOUT = 300
_REVIEW_MAX_DIFF_CHARS = 60_000
_REVIEW_ENABLE_ENV = "PUPPETMASTER_REVIEW_GATE"
# The marker the judge model must emit so its verdict is machine-extractable
# from free-form model output, regardless of which adapter ran it.
_REVIEW_VERDICT_MARKER = "PUPPETMASTER_REVIEW_VERDICT"


@dataclass
class GateResult:
    name: str
    kind: str
    passed: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateEvaluation:
    passed: bool
    results: list[GateResult]
    artifacts: list[Artifact]

    @property
    def failed_reason(self) -> Optional[str]:
        for result in self.results:
            if not result.passed:
                return f"{result.name}: {result.reason}"
        return None


def task_gate_specs(task: Task) -> list[dict[str, Any]]:
    """Resolve a task's gate specs from explicit ``gates`` plus convenience
    flags (``require_diff``, ``commit``). Returns an ordered, de-duplicated list
    of gate dicts. No gates configured → empty list (gating is opt-in)."""
    specs: list[dict[str, Any]] = []
    for raw in task.payload.get("gates", []) or []:
        if isinstance(raw, dict) and raw.get("kind"):
            specs.append(dict(raw))

    # An implement (full-edit) task must produce a diff to reach COMPLETE — a
    # "completed" implement run with zero changes is the A3 silent-no-op failure
    # (looks done, committed nothing, changed nothing). Default the require_diff
    # invariant ON for implement mode unless explicitly opted out
    # (require_diff=false or allow_empty_diff=true for a legitimate no-op task).
    require_diff = task.payload.get("require_diff")
    implement_requires_diff = (
        task.payload.get("mode") == "implement"
        and require_diff is not False
        and not task.payload.get("allow_empty_diff", False)
    )
    if (require_diff or implement_requires_diff) and not any(
        s["kind"] == "require_diff" for s in specs
    ):
        specs.append({"kind": "require_diff"})

    commit = task.payload.get("commit")
    if commit and not any(s["kind"] == "committed" for s in specs):
        spec = {"kind": "committed"}
        if isinstance(commit, dict):
            spec.update(commit)
        specs.append(spec)

    # B3/C1: a declared write-scope is enforced — the task may only change files
    # matching its globs. Catches a worker straying into another wave's hot files.
    write_scope = task.payload.get("write_scope")
    if write_scope and not any(s["kind"] == "write_scope" for s in specs):
        specs.append({"kind": "write_scope", "scope": list(write_scope)})

    # Semantic review is the implement safety net: a strictly-stronger model
    # must approve the diff before COMPLETE. Attached last so the cheap gates
    # (require_diff/ratchet/command) run first and a no-op or test-failing task
    # never pays for a judge call. Default-on for implement when the feature is
    # enabled (``review=true`` on the task, or ``$PUPPETMASTER_REVIEW_GATE``
    # globally); opt a single task out with ``review=false`` / ``allow_unreviewed``.
    review = task.payload.get("review")
    review_globally_on = bool(os.environ.get(_REVIEW_ENABLE_ENV))
    implement_wants_review = (
        task.payload.get("mode") == "implement"
        and review is not False
        and not task.payload.get("allow_unreviewed", False)
        and (review or review_globally_on)
    )
    if (review or implement_wants_review) and not any(s["kind"] == "review" for s in specs):
        spec: dict[str, Any] = {"kind": "review"}
        if isinstance(review, dict):
            spec.update(review)
        specs.append(spec)

    return specs


def evaluate_task_gates(
    task: Task,
    artifacts: list[Artifact],
    store: "SwarmStore",
    *,
    worker_id: str,
    cwd: Optional[Path] = None,
) -> GateEvaluation:
    """Run every configured gate for ``task`` and return the combined verdict.

    ``artifacts`` are the worker's outputs (used by ``require_diff``). A GATE
    artifact is produced per gate so ``show``/audits can explain exactly why a
    task passed or failed its post-conditions.
    """
    specs = task_gate_specs(task)
    if not specs:
        return GateEvaluation(passed=True, results=[], artifacts=[])

    cwd = Path(cwd or task.payload.get("cwd") or ".").resolve()
    results: list[GateResult] = []
    for spec in specs:
        results.append(_evaluate_one(spec, task, artifacts, store, cwd))

    gate_artifacts = [_gate_artifact(task, worker_id, result) for result in results]
    passed = all(result.passed for result in results)
    return GateEvaluation(passed=passed, results=results, artifacts=gate_artifacts)


def _evaluate_one(
    spec: dict[str, Any],
    task: Task,
    artifacts: list[Artifact],
    store: "SwarmStore",
    cwd: Path,
) -> GateResult:
    kind = spec.get("kind")
    name = str(spec.get("name") or kind)
    try:
        if kind == "require_diff":
            return _gate_require_diff(name, artifacts, cwd)
        if kind == "command":
            return _gate_command(name, spec, cwd)
        if kind == "ratchet":
            return _gate_ratchet(name, spec, store, cwd)
        if kind == "committed":
            return _gate_committed(name, spec, cwd)
        if kind == "write_scope":
            return _gate_write_scope(name, spec, artifacts, cwd)
        if kind == "review":
            return _gate_review(name, spec, task, artifacts, cwd)
        return GateResult(name, str(kind), False, f"unknown gate kind: {kind!r}")
    except Exception as exc:  # a gate that crashes must FAIL closed, never pass
        return GateResult(name, str(kind), False, f"gate raised: {exc}")


def _gate_require_diff(name: str, artifacts: list[Artifact], cwd: Path) -> GateResult:
    from puppetmaster.adapters import git_snapshot

    has_patch = any(a.type == ArtifactType.PATCH for a in artifacts)
    snapshot = git_snapshot(cwd)
    has_diff = bool(str(snapshot.get("diff") or "").strip())
    if has_patch or has_diff:
        return GateResult(name, "require_diff", True, "produced a non-empty diff")
    return GateResult(
        name,
        "require_diff",
        False,
        "edit task produced no diff (no files changed) — refusing to call this complete",
        {"changed_files": snapshot.get("changed_files", [])},
    )


def _gate_write_scope(
    name: str, spec: dict[str, Any], artifacts: list[Artifact], cwd: Path
) -> GateResult:
    """Enforce a task's declared write-scope (B3/C1).

    Every file the run changed must match one of the declared globs. A write
    outside scope means the worker strayed into territory another task owns —
    the root cause of cross-wave regressions and hand-merged hot files — so the
    task FAILS loudly instead of silently colliding.
    """
    import fnmatch
    from puppetmaster.adapters import git_snapshot

    scope = [str(g) for g in (spec.get("scope") or spec.get("globs") or []) if str(g).strip()]
    if not scope:
        return GateResult(name, "write_scope", False, "write_scope gate needs 'scope' globs")

    snapshot = git_snapshot(cwd)
    changed = set(snapshot.get("changed_files") or []) | set(snapshot.get("untracked_files") or [])
    for artifact in artifacts:
        if artifact.type == ArtifactType.PATCH:
            changed.update((artifact.payload or {}).get("files") or [])

    out_of_scope = sorted(
        path for path in changed if not any(fnmatch.fnmatch(path, glob) for glob in scope)
    )
    if out_of_scope:
        return GateResult(
            name, "write_scope", False,
            f"wrote {len(out_of_scope)} file(s) outside declared scope",
            {"out_of_scope": out_of_scope, "scope": scope},
        )
    return GateResult(
        name, "write_scope", True, "all writes within declared scope",
        {"scope": scope, "changed_files": sorted(changed)},
    )


def _gate_command(name: str, spec: dict[str, Any], cwd: Path) -> GateResult:
    command = spec.get("command")
    if not command:
        return GateResult(name, "command", False, "command gate has no 'command'")
    completed = _run(command, cwd, int(spec.get("timeout_seconds", _GATE_COMMAND_TIMEOUT)))
    ok = completed.returncode == 0
    return GateResult(
        name,
        "command",
        ok,
        "exit 0" if ok else f"exit {completed.returncode}",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
        },
    )


def _gate_ratchet(name: str, spec: dict[str, Any], store: "SwarmStore", cwd: Path) -> GateResult:
    command = spec.get("command")
    metric = spec.get("metric")
    if not command or not metric:
        return GateResult(name, "ratchet", False, "ratchet gate needs 'command' and 'metric'")
    completed = _run(command, cwd, int(spec.get("timeout_seconds", _GATE_COMMAND_TIMEOUT)))
    if completed.returncode != 0:
        return GateResult(
            name, "ratchet", False, f"oracle command failed (exit {completed.returncode})",
            {"command": command, "stderr_tail": (completed.stderr or "")[-2000:]},
        )
    value = _extract_metric(completed.stdout, metric)
    if value is None:
        return GateResult(
            name, "ratchet", False, f"metric {metric!r} not found in command stdout JSON",
            {"command": command, "stdout_tail": (completed.stdout or "")[-2000:]},
        )

    baseline_path = _ratchet_baseline_path(store, cwd, metric)
    baseline = _read_baseline(store, baseline_path)
    detail = {"metric": metric, "value": value, "baseline": baseline, "baseline_path": str(baseline_path)}

    if baseline is None:
        # First observation establishes the baseline; never fail the establishing
        # run. Subsequent runs are enforced monotonically.
        _write_baseline(store, baseline_path, value)
        return GateResult(name, "ratchet", True, f"established baseline {metric}={value}", detail)

    if value > baseline:
        return GateResult(
            name, "ratchet", False,
            f"{metric} regressed: {value} > baseline {baseline}", detail,
        )
    if value < baseline:
        # Tighten the ratchet: the runtime owns the baseline and only moves it down.
        _write_baseline(store, baseline_path, value)
        detail["baseline_tightened_to"] = value
    return GateResult(name, "ratchet", True, f"{metric}={value} ≤ baseline {baseline}", detail)


def _gate_committed(name: str, spec: dict[str, Any], cwd: Path) -> GateResult:
    from puppetmaster.adapters import git_snapshot

    snapshot = git_snapshot(cwd)
    dirty = bool(snapshot.get("changed_files") or snapshot.get("untracked_files"))
    if not dirty:
        return GateResult(name, "committed", True, "working tree is clean (work committed)")

    if not spec.get("auto"):
        return GateResult(
            name, "committed", False,
            "task left uncommitted changes; required to commit its work",
            {
                "changed_files": snapshot.get("changed_files", []),
                "untracked_files": snapshot.get("untracked_files", []),
            },
        )

    # auto-commit on the worker's behalf with the required author/message.
    message = str(spec.get("message") or "chore: commit task work")
    author = spec.get("author")
    add = _run("git add -A", cwd, 120)
    if add.returncode != 0:
        return GateResult(name, "committed", False, "git add failed", {"stderr_tail": add.stderr[-2000:]})
    # C2: keep generated artifacts (parity scoreboards, coverage, build output)
    # out of the worker's commit. Unstage every excluded pathspec and record it
    # in .gitignore so it stays out of future diffs too.
    excluded = _strip_excluded_paths(spec.get("exclude"), cwd)
    args = ["git", "commit", "-m", message]
    if author:
        args += ["--author", str(author)]
    committed = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
    ok = committed.returncode == 0
    return GateResult(
        name, "committed", ok,
        "auto-committed work" if ok else "auto-commit failed",
        {
            "message": message,
            "author": author,
            "excluded": excluded,
            "stderr_tail": (committed.stderr or "")[-2000:],
        },
    )


def _strip_excluded_paths(exclude: Any, cwd: Path) -> list[str]:
    """Unstage every excluded pathspec so generated artifacts never enter an
    auto-commit, and append them to ``.gitignore`` so they stay out of future
    diffs. Returns the normalized exclude list (empty when nothing configured).
    Best-effort: a failure here must not block the commit."""
    if not exclude:
        return []
    patterns = [str(p).strip() for p in (exclude if isinstance(exclude, (list, tuple)) else [exclude])]
    patterns = [p for p in patterns if p]
    if not patterns:
        return []
    _run(["git", "reset", "-q", "--", *patterns], cwd, 60)
    try:
        gitignore = cwd / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
        present = {line.strip() for line in existing.splitlines()}
        additions = [p for p in patterns if p not in present]
        if additions:
            prefix = "" if existing.endswith("\n") or not existing else "\n"
            with gitignore.open("a", encoding="utf-8") as handle:
                handle.write(prefix + "# Puppetmaster: generated artifacts excluded from worker commits\n")
                handle.write("\n".join(additions) + "\n")
            # Re-stage .gitignore so the exclusion persists in the worker's commit.
            _run(["git", "add", "--", ".gitignore"], cwd, 30)
    except OSError:
        pass
    return patterns


# --------------------------------------------------------------------------
# review gate — a strictly-stronger model judges the diff before COMPLETE
# --------------------------------------------------------------------------


@dataclass
class ReviewVerdict:
    """Outcome of an LLM-as-judge review of a task's diff.

    ``available`` distinguishes "a judge ran and reached a verdict" from "no
    judge ran" (feature off, no eligible model, judge unreachable). Only an
    *available* failing verdict fails the gate; an unavailable judge is a no-op
    so an unconfigured environment is never bricked. A judge that ran but whose
    answer couldn't be parsed is ``available=True, passed=False`` — fail-closed,
    because a quality gate that can't read its oracle must not wave work through.
    """

    available: bool
    passed: bool
    severity: str = "none"
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


_DEFAULT_REVIEW_RUBRIC = (
    "Judge whether this diff is production-quality work for the stated task. "
    "Reject (pass=false) if it is incorrect, only partially implements the task, "
    "introduces a regression or security issue, ignores the task's scope, leaves "
    "debugging cruft / dead code, or is so unclear a maintainer couldn't safely "
    "own it. A diff that merely changes files without doing the task well must be "
    "rejected. Approve (pass=true) only if you would let it merge as-is."
)


def build_review_prompt(task: Task, diff: str, rubric: str) -> str:
    """Compose the judge prompt: task intent + rubric + diff + a strict,
    machine-extractable output contract. Pure (no I/O) so it's unit-testable."""
    instruction = (task.payload.get("prompt") or task.instruction or "").strip()
    return (
        "You are a senior staff engineer doing a blocking code review of a change "
        "another agent produced. Be exacting; your approval is the only thing "
        "standing between this diff and the main branch.\n\n"
        f"## Task the change was supposed to accomplish\n{instruction}\n\n"
        f"## Review rubric\n{rubric}\n\n"
        f"## Diff under review\n```diff\n{diff}\n```\n\n"
        "## Required output\n"
        "Reason briefly, then emit your verdict on its own line as the marker "
        f"`{_REVIEW_VERDICT_MARKER}` immediately followed by a single-line JSON "
        'object: {"pass": <true|false>, "severity": '
        '"none"|"minor"|"major"|"critical", "reasons": ["..."]}. '
        "Emit the marker and JSON exactly once."
    )


def _implementer_capability(task: Task, spec: dict[str, Any], registry: list) -> int:
    """The capability floor the judge must clear. ``min_judge_capability`` on
    the gate spec wins; otherwise the judge must be strictly stronger than the
    model that wrote the code (router-stamped ``router_model_id``)."""
    explicit = spec.get("min_judge_capability")
    if isinstance(explicit, (int, float)):
        return int(explicit)
    model_id = (task.payload or {}).get("router_model_id")
    if model_id:
        for model_spec in registry:
            if model_spec.id == model_id:
                return model_spec.capability_score + 1
    return 0


def resolve_judge_model(task: Task, spec: dict[str, Any]) -> Optional["ModelSpec"]:
    """Pick the cheapest model that clears the capability floor (the smallest
    sufficient upgrade over the implementer). Falls back to the single strongest
    available model as a peer reviewer when nothing strictly clears the floor —
    but only if it is at least as capable as the implementer, otherwise there is
    no point and we return ``None`` (review becomes a no-op)."""
    from puppetmaster.model_registry import default_registry_path, load_registry
    from puppetmaster.platform_lock import is_adapter_enabled

    registry = [
        s
        for s in load_registry(default_registry_path())
        if s.enabled and is_adapter_enabled(s.adapter)
    ]
    if not registry:
        return None

    floor = _implementer_capability(task, spec, registry)
    eligible = [s for s in registry if s.capability_score >= floor]
    if eligible:
        return min(eligible, key=lambda s: (s.capability_score, s.id))

    strongest = max(registry, key=lambda s: s.capability_score)
    implementer_floor = floor - 1 if not spec.get("min_judge_capability") else floor
    if strongest.capability_score >= implementer_floor:
        return strongest
    return None


def parse_review_verdict(text: str) -> Optional[dict[str, Any]]:
    """Extract the judge's ``{pass, severity, reasons}`` object from free-form
    model output by locating the marker and brace-matching the JSON that follows.
    Returns ``None`` when no well-formed verdict is present (→ fail-closed)."""
    marker_at = text.rfind(_REVIEW_VERDICT_MARKER)
    if marker_at == -1:
        return None
    tail = text[marker_at + len(_REVIEW_VERDICT_MARKER):]
    brace = tail.find("{")
    if brace == -1:
        return None
    depth = 0
    for offset, char in enumerate(tail[brace:]):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = tail[brace : brace + offset + 1]
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _verdict_from_artifacts(artifacts: list[Artifact]) -> Optional[dict[str, Any]]:
    """Scan a judge run's artifacts for the verdict marker. The marker may land
    in a parsed FINDING/VERIFICATION payload or in the raw captured stdout the
    adapter stashes on its verification artifact — search all of it."""
    for artifact in artifacts:
        for value in (artifact.payload or {}).values():
            if isinstance(value, str) and _REVIEW_VERDICT_MARKER in value:
                verdict = parse_review_verdict(value)
                if verdict is not None:
                    return verdict
    return None


def default_judge_review(
    *, prompt: str, judge: "ModelSpec", cwd: Path, timeout: int, task: Task
) -> ReviewVerdict:
    """Run the judge model read-only over the diff and read back its verdict.

    Gated behind ``$PUPPETMASTER_REVIEW_GATE`` because it makes an extra model
    call (a deliberate, flagged spend). Fail-closed: any error, timeout, or
    unparseable answer from a judge that *did* run rejects the diff — a review
    oracle that can't be read is treated as a failed review, never a pass."""
    if not os.environ.get(_REVIEW_ENABLE_ENV):
        return ReviewVerdict(
            available=False,
            passed=True,
            reasons=[f"{_REVIEW_ENABLE_ENV} not set; live review disabled"],
            detail={"enabled": False},
        )
    try:
        from dataclasses import replace as _replace

        from puppetmaster.adapters import get_adapter

        judge_task = _replace(
            task,
            adapter=judge.adapter,
            payload={
                **(task.payload or {}),
                "prompt": prompt,
                "model": judge.adapter_model_name,
                "cwd": str(cwd),
                "mode": "analyze",
                "implement": False,
                "disable_memory": True,
                "disable_codegraph": True,
                "auto_route": False,
                "timeout_seconds": timeout,
            },
        )
        artifacts = get_adapter(judge.adapter).run(
            judge_task, prompt, f"review-judge-{judge.id}"
        )
    except Exception as exc:  # judge crashed → fail-closed
        return ReviewVerdict(
            available=True,
            passed=False,
            severity="critical",
            reasons=[f"judge invocation failed: {exc}"],
            detail={"error": str(exc)},
        )

    verdict = _verdict_from_artifacts(artifacts)
    if verdict is None:
        return ReviewVerdict(
            available=True,
            passed=False,
            severity="critical",
            reasons=["judge produced no parseable verdict"],
            detail={"judge_artifacts": len(artifacts)},
        )
    reasons = verdict.get("reasons")
    return ReviewVerdict(
        available=True,
        passed=bool(verdict.get("pass")),
        severity=str(verdict.get("severity", "none")),
        reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [],
        detail={"raw_verdict": verdict},
    )


# Module-level seam so tests inject a fake judge and the live adapter call is
# never made in the unit suite. Production wiring uses ``default_judge_review``.
ReviewJudge = Callable[..., ReviewVerdict]
_REVIEW_JUDGE: ReviewJudge = default_judge_review


def _review_sampled(task_id: str, sample: float) -> bool:
    """Deterministic per-task sampling: the same task always lands on the same
    side of the cut, so a re-run isn't randomly (un)reviewed. ``sample`` is the
    fraction of tasks reviewed (1.0 → all, 0.0 → none)."""
    if sample >= 1.0:
        return True
    if sample <= 0.0:
        return False
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < sample


def _collect_diff(artifacts: list[Artifact], cwd: Path) -> str:
    """The change to review: the live working-tree diff, plus any PATCH-artifact
    diffs (covers a committed implement run whose tree is already clean)."""
    from puppetmaster.adapters import git_snapshot

    diff = str(git_snapshot(cwd).get("diff") or "")
    if not diff.strip():
        chunks = [
            str((a.payload or {}).get("diff") or "")
            for a in artifacts
            if a.type == ArtifactType.PATCH
        ]
        diff = "\n".join(c for c in chunks if c.strip())
    return diff


def _gate_review(
    name: str, spec: dict[str, Any], task: Task, artifacts: list[Artifact], cwd: Path
) -> GateResult:
    diff = _collect_diff(artifacts, cwd)
    if not diff.strip():
        # Nothing to review. ``require_diff`` owns the "implement did nothing"
        # failure; an empty diff here is not the review gate's problem.
        return GateResult(name, "review", True, "no diff to review")

    sample = spec.get("sample")
    if isinstance(sample, (int, float)) and not _review_sampled(task.id, float(sample)):
        return GateResult(
            name, "review", True, f"not sampled (sample={sample})", {"sampled": False}
        )

    judge = resolve_judge_model(task, spec)
    if judge is None:
        return GateResult(
            name, "review", True, "review skipped: no adequate judge model available",
            {"judge": None},
        )

    max_chars = int(spec.get("max_diff_chars", _REVIEW_MAX_DIFF_CHARS))
    diff_for_judge = (
        diff
        if len(diff) <= max_chars
        else diff[:max_chars] + "\n... (diff truncated for review) ..."
    )
    rubric = str(spec.get("rubric") or _DEFAULT_REVIEW_RUBRIC)
    prompt = build_review_prompt(task, diff_for_judge, rubric)
    timeout = int(spec.get("timeout_seconds", _REVIEW_TIMEOUT))

    verdict = _REVIEW_JUDGE(prompt=prompt, judge=judge, cwd=cwd, timeout=timeout, task=task)
    if not verdict.available:
        return GateResult(
            name, "review", True, "review skipped: judge unavailable",
            {"judge": judge.id, **verdict.detail},
        )

    detail = {
        "judge_model": judge.id,
        "severity": verdict.severity,
        "reasons": verdict.reasons,
        "diff_chars": len(diff),
    }
    if verdict.passed:
        return GateResult(name, "review", True, f"{judge.id} approved the diff", detail)
    summary = "; ".join(verdict.reasons) or verdict.severity or "rejected"
    return GateResult(name, "review", False, f"{judge.id} rejected the diff: {summary}", detail)


def _run(command: Any, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a gate command without a shell. Lists run as argv. Strings are
    shlex-parsed on POSIX; on Windows the raw string goes straight to
    CreateProcess (POSIX shlex would eat ``C:\\path`` backslashes), which
    still never involves cmd.exe."""
    if isinstance(command, str) and os.name != "nt":
        argv: Any = shlex.split(command)
    else:
        argv = command
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv, returncode=124, stdout=exc.stdout or "", stderr=f"timeout after {timeout}s"
        )


def _extract_metric(stdout: str, metric: str) -> Optional[float]:
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    source = data.get("metrics") if isinstance(data.get("metrics"), dict) else data
    value = source.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ratchet_baseline_path(store: "SwarmStore", cwd: Path, metric: str) -> Path:
    safe = store._safe_key(f"{cwd.name}__{metric}")
    return store.root / "ratchets" / f"{safe}.json"


def _read_baseline(store: "SwarmStore", path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        return float(store.read_json(path).get("value"))
    except (ValueError, TypeError, OSError):
        return None


def _write_baseline(store: "SwarmStore", path: Path, value: float) -> None:
    store.write_json(path, {"value": value, "updated_at": now_iso()})


def _gate_artifact(task: Task, worker_id: str, result: GateResult) -> Artifact:
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.GATE,
        created_by=worker_id,
        confidence=0.95 if result.passed else 0.9,
        evidence=[f"gate:{result.kind}", "passed" if result.passed else "failed"],
        payload={
            "gate": result.name,
            "kind": result.kind,
            "passed": result.passed,
            "reason": result.reason,
            **result.detail,
        },
    )
