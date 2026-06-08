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

The cheap ``ratchet`` and the expensive ``command`` oracle cover each other's
blind spot: gaming the scalar metric gets caught by the semantic command, and
the command only has to run when the metric gate passes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from puppetmaster.models import Artifact, ArtifactType, Task, now_iso

if TYPE_CHECKING:
    from puppetmaster.store import SwarmStore

_GATE_COMMAND_TIMEOUT = 1800


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


def _run(command: Any, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a gate command. A string runs through the shell; a list runs argv."""
    shell = isinstance(command, str)
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command, returncode=124, stdout=exc.stdout or "", stderr=f"timeout after {timeout}s"
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
