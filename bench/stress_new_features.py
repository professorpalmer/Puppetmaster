"""End-to-end stress harness for the pre-deploy safety/feature batch.

Puppetmaster is in real users' hands, so before shipping the latest batch we
validate each new capability *under volume and concurrency*, not just with a
single happy-path unit test. Every scenario here drives the real code paths
(stores, gates, the orchestrator wait loop, the gc guard, port allocation,
conflict prediction) with no external agent CLIs, so it runs anywhere and is
deterministic.

Scenarios:
  D1  delete_job path-safety fuzz (both backends, concurrent) + gc active-worktree guard
  B1  per-worktree port determinism + collision-rate measurement at realistic parallelism
  A3  implement require_diff invariant across many real git repos
  A2F2 status `outcome` signals across every artifact combination
  A4  progress-based timeout extension vs wedged-kill, run concurrently at volume
  B3C1 write_scope gate enforcement + conflict prediction correctness at scale
  C2  generated-artifact stripping from auto-commits across many repos

Run: `python -m bench.stress_new_features` (add `--quick` for a fast pass).
Exit code is non-zero if any scenario fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.conflicts import predict_write_conflicts, scopes_overlap  # noqa: E402
from puppetmaster.gates import evaluate_task_gates  # noqa: E402
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task  # noqa: E402
from puppetmaster.orchestrator import Orchestrator  # noqa: E402
from puppetmaster.ports import _port_is_free, reserve_port, worktree_port_base  # noqa: E402
from puppetmaster.sqlite_store import SQLiteSwarmStore  # noqa: E402
from puppetmaster.store import SwarmStore  # noqa: E402


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "stress@test.local")
    _git(path, "config", "user.name", "stress")
    (path / "seed.txt").write_text("seed\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "seed")
    return path


def _task(**payload: Any) -> Task:
    return Task(job_id="job_stress", role="cursor", instruction="x", payload=payload)


def _has_git() -> bool:
    return shutil.which("git") is not None


def _job_exists(store: Any, job_id: str) -> bool:
    """True iff the job's state is still present. ``get_job`` raises once a job
    has been reaped (its file is gone), so we treat any failure as 'gone'."""
    try:
        return store.get_job(job_id) is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# D1 — delete_job path-safety fuzz + gc active-worktree guard
# --------------------------------------------------------------------------- #

# Ids that MUST be refused: every one either escapes the jobs tree or names the
# jobs root itself. A relative id that stays inside (e.g. "ghost") is *not* here
# because deleting a non-existent in-tree job is a legitimate no-op.
_UNSAFE_IDS = [
    "", "   ", ".", "..", "../..", "../../..", "/", "/etc", "/etc/passwd",
    "../../etc/passwd", "..//..", "../", "  ..  ", "\t", "\n",
]


def scenario_d1_delete_fuzz(quick: bool) -> Result:
    backends = {"file": SwarmStore, "sqlite": SQLiteSwarmStore}
    attempts_per_id = 20 if quick else 80
    refused = 0
    leaked: list[str] = []
    for backend_name, factory in backends.items():
        with TemporaryDirectory() as tmp:
            store = factory(Path(tmp) / ".puppetmaster")
            store.init()
            real_jobs = [store.create_job(f"job {i}").id for i in range(10)]
            sentinel = store.jobs_dir / "SENTINEL_DO_NOT_DELETE.txt"
            sentinel.write_text("keep me")
            outside = Path(tmp) / "SOURCE_CODE.txt"
            outside.write_text("precious source")

            lock = threading.Lock()

            def fuzz(job_id: str) -> None:
                nonlocal refused
                try:
                    store.delete_job(job_id)
                except ValueError:
                    with lock:
                        refused += 1
                except Exception:
                    # Any *other* exception type is itself a finding — record it.
                    with lock:
                        leaked.append(f"{backend_name}:{job_id!r}:wrong-exc")

            work = list(itertools.chain.from_iterable(
                [_UNSAFE_IDS] * attempts_per_id
            ))
            random.shuffle(work)
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(fuzz, work))

            if not sentinel.exists():
                leaked.append(f"{backend_name}:sentinel-destroyed")
            if not outside.exists():
                leaked.append(f"{backend_name}:source-destroyed")
            survivors = {j.id for j in store.list_jobs()}
            if set(real_jobs) - survivors:
                leaked.append(f"{backend_name}:real-job-vanished")
            # Legitimate deletes still work, concurrently.
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(store.delete_job, real_jobs))
            if {j.id for j in store.list_jobs()} & set(real_jobs):
                leaked.append(f"{backend_name}:legit-delete-failed")

    passed = not leaked
    detail = (
        "every unsafe id refused; sentinels + source intact; legit deletes work"
        if passed else f"LEAKS: {leaked}"
    )
    return Result("D1 delete_job path-safety fuzz", passed, detail, {
        "unsafe_ids": len(_UNSAFE_IDS),
        "refusals": refused,
        "backends": list(backends),
    })


def scenario_d1_gc_guard(quick: bool) -> Result:
    from puppetmaster import cli

    n_projects = 4 if quick else 12
    with TemporaryDirectory() as tmp:
        stores = []
        for i in range(n_projects):
            store = SwarmStore(Path(tmp) / f"proj{i}" / ".puppetmaster")
            store.init()
            done = store.create_job(f"old terminal {i}")
            store.update_job_status(done.id, JobStatus.COMPLETE)
            stores.append((store, done.id))

        active_store, active_job = stores[0]

        class Args:
            all_projects = True
            force = True
            older_than_days = 0
            json = True
            backend = "file"

        # Sweep every project (incl. active) with --force --all-projects.
        with patch.object(cli, "_gc_target_stores", return_value=[s for s, _ in stores]):
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                cli._run_gc_command(Args(), active_store)
            report = json.loads(buf.getvalue())

        active_alive = _job_exists(active_store, active_job)
        others_reaped = all(
            not _job_exists(store, jid) for store, jid in stores[1:]
        )
    passed = active_alive and others_reaped and report.get("protected_active_worktree") is True
    detail = (
        f"active worktree preserved; {n_projects - 1} other projects reaped"
        if passed else
        f"active_alive={active_alive} others_reaped={others_reaped} "
        f"protected_flag={report.get('protected_active_worktree')}"
    )
    return Result("D1 gc --force --all-projects active-worktree guard", passed, detail, {
        "projects": n_projects,
    })


# --------------------------------------------------------------------------- #
# B1 — per-worktree port determinism + collision rate
# --------------------------------------------------------------------------- #

def scenario_b1_ports(quick: bool) -> Result:
    # Determinism + range over a large synthetic worktree population.
    population = 2000 if quick else 20000
    base = Path("/Users/dev/campaigns")
    paths = [base / f"feature-{i}" / "worktree" for i in range(population)]
    first = [worktree_port_base(p) for p in paths]
    second = [worktree_port_base(p) for p in paths]
    deterministic = first == second
    # Below Linux's ephemeral floor (32768) for cross-OS safety.
    in_range = all(10000 <= p < 32768 for p in first)

    # Realistic parallelism: how often do K concurrent worktrees collide?
    rng = random.Random(1234)
    trials = 500 if quick else 3000
    for parallelism in (8, 16, 32):
        collisions = 0
        for _ in range(trials):
            sample = [
                worktree_port_base(base / f"wt-{rng.randrange(10**9)}")
                for _ in range(parallelism)
            ]
            if len(set(sample)) != len(sample):
                collisions += 1
        rate = collisions / trials
        # Record per-parallelism; assert only the most common case is sane.
        if parallelism == 16:
            collision_rate_16 = rate

    # Bulletproof path: reserve_port must bump past a live listener on the hint
    # and hand back an actually-bindable port — the real fix for the residual
    # hash-collision tail. Stress it across many occupied hints concurrently.
    import socket as _socket
    reserve_failures = 0
    reserve_trials = 40 if quick else 200

    def _reserve_under_contention(i: int) -> None:
        nonlocal reserve_failures
        wt = base / f"reserve-{i}"
        hint = worktree_port_base(wt)
        listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", hint))
            listener.listen(1)
        except OSError:
            listener.close()
            return  # couldn't occupy the hint here; skip this trial
        try:
            reserved = reserve_port(wt)
            if reserved == hint or not _port_is_free(reserved):
                reserve_failures += 1
        finally:
            listener.close()

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_reserve_under_contention, range(reserve_trials)))

    passed = deterministic and in_range and collision_rate_16 < 0.10 and reserve_failures == 0
    detail = (
        f"deterministic over {population} paths, range<32768, "
        f"hint collision {collision_rate_16:.1%} at 16 worktrees, "
        f"reserve_port resolved {reserve_trials} contended hints cleanly"
        if passed else
        f"deterministic={deterministic} in_range={in_range} "
        f"collision_rate_16={collision_rate_16:.1%} reserve_failures={reserve_failures}"
    )
    return Result("B1 per-worktree port allocation", passed, detail, {
        "population": population,
        "hint_collision_rate_16_worktrees": round(collision_rate_16, 4),
        "reserve_port_contended_trials": reserve_trials,
        "reserve_port_failures": reserve_failures,
    })


# --------------------------------------------------------------------------- #
# A3 — implement require_diff invariant across many real repos
# --------------------------------------------------------------------------- #

def scenario_a3_require_diff(quick: bool) -> Result:
    if not _has_git():
        return Result("A3 implement require_diff invariant", False, "git not available")
    repos = 8 if quick else 30
    misclassified: list[str] = []
    with TemporaryDirectory() as tmp:
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()

        def check(i: int) -> None:
            repo = _git_repo(Path(tmp) / f"repo{i}")
            # No-op implement -> gate must FAIL loudly.
            noop = _task(mode="implement", cwd=str(repo))
            if evaluate_task_gates(noop, [], store, worker_id=f"w{i}", cwd=repo).passed:
                misclassified.append(f"repo{i}:noop-passed")
            # Real change -> gate must PASS.
            (repo / f"change{i}.py").write_text("print('x')\n")
            real = _task(mode="implement", cwd=str(repo))
            if not evaluate_task_gates(real, [], store, worker_id=f"w{i}", cwd=repo).passed:
                misclassified.append(f"repo{i}:real-failed")
            # Explicit opt-out -> no gate.
            optout = _task(mode="implement", allow_empty_diff=True, cwd=str(repo))
            _git(repo, "reset", "-q", "--hard", "HEAD")
            if not evaluate_task_gates(optout, [], store, worker_id=f"w{i}", cwd=repo).passed:
                misclassified.append(f"repo{i}:optout-failed")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, range(repos)))

    passed = not misclassified
    detail = (
        f"{repos} repos: no-op fails, real passes, opt-out bypasses"
        if passed else f"MISCLASSIFIED: {misclassified[:5]}"
    )
    return Result("A3 implement require_diff invariant", passed, detail, {"repos": repos})


# --------------------------------------------------------------------------- #
# A2+F2 — status outcome signals across artifact combinations
# --------------------------------------------------------------------------- #

def _patch(job_id: str) -> Artifact:
    return Artifact(job_id=job_id, task_id="t", type=ArtifactType.PATCH, created_by="w",
                    confidence=0.9, evidence=["e"], payload={"change": "edit", "files": ["a.py"]})


def _commit_gate(job_id: str) -> Artifact:
    return Artifact(job_id=job_id, task_id="t", type=ArtifactType.GATE, created_by="w",
                    confidence=0.95, evidence=["gate:committed", "passed"],
                    payload={"gate": "committed", "kind": "committed", "passed": True})


def _blocked(job_id: str) -> Artifact:
    return Artifact(job_id=job_id, task_id="t", type=ArtifactType.VERIFICATION, created_by="w",
                    confidence=0.9, evidence=["status:blocked"],
                    payload={"check": "implement", "result": "blocked", "failure": "dirty_worktree"})


def scenario_a2f2_status(quick: bool) -> Result:
    cases = {
        "empty": ([], {"diff_present": False, "commit_present": False}),
        "patch_only": ([_patch], {"diff_present": True, "commit_present": False}),
        "patch_and_commit": ([_patch, _commit_gate], {"diff_present": True, "commit_present": True}),
        "blocked": ([_blocked], {"diff_present": False, "commit_present": False}),
    }
    failures: list[str] = []
    with TemporaryDirectory() as tmp:
        for case, (builders, expect) in cases.items():
            store = SQLiteSwarmStore(Path(tmp) / case / ".puppetmaster")
            store.init()
            job = store.create_job(case)
            for build in builders:
                store.save_artifact(build(job.id))
            outcome = store.status_snapshot(job.id)["outcome"]
            for key, want in expect.items():
                if outcome.get(key) != want:
                    failures.append(f"{case}.{key}={outcome.get(key)}!={want}")
            if "quality" not in outcome or "trustworthy" not in outcome:
                failures.append(f"{case}:missing-quality-fields")
            if case == "blocked" and outcome.get("quality") != "blocked":
                failures.append(f"blocked-case-quality={outcome.get('quality')}")
    passed = not failures
    detail = "all artifact combos surface correct outcome signals" if passed else f"{failures}"
    return Result("A2+F2 status outcome signals", passed, detail, {"cases": list(cases)})


# --------------------------------------------------------------------------- #
# A4 — progress extension vs wedged-kill, concurrent at volume
# --------------------------------------------------------------------------- #

class _ProgressingProc:
    """Times out N times while emitting events, then exits clean — a worker in a
    long-but-live verify phase."""
    def __init__(self, store: Any, job_id: str, timeouts: int) -> None:
        self._store, self._job_id, self._left = store, job_id, timeouts
        self.returncode = 0
        self.terminated = False

    def wait(self, timeout: Any = None) -> int:
        self._store.emit(self._job_id, "run.heartbeat", {"left": self._left})
        if self._left > 0:
            self._left -= 1
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


class _WedgedProc:
    """Always times out, emits nothing — a genuinely hung worker."""
    def __init__(self) -> None:
        self.returncode = -15
        self.killed = False

    def wait(self, timeout: Any = None) -> int:
        if self.killed:
            return -15
        raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    def terminate(self) -> None:
        self.killed = True

    def kill(self) -> None:
        self.killed = True


def scenario_a4_timeout(quick: bool) -> Result:
    workers = 30 if quick else 120
    failures: list[str] = []
    with TemporaryDirectory() as tmp:
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        orch = Orchestrator(store)

        def run_progressing(i: int) -> None:
            job = store.create_job(f"live-{i}")
            task = Task(job_id=job.id, role="cursor", instruction="e2e")
            proc = _ProgressingProc(store, job.id, timeouts=3)
            t0 = time.monotonic()
            try:
                orch._wait_for_worker(proc, job, [task])
            except RuntimeError:
                failures.append(f"live-{i}:killed-while-progressing")
                return
            if time.monotonic() - t0 > 10:
                failures.append(f"live-{i}:took-too-long")
            events = [e["event"] for e in store.read_events(job.id)]
            if "worker.timeout_extended" not in events:
                failures.append(f"live-{i}:no-extension-event")
            if proc.terminated:
                failures.append(f"live-{i}:was-terminated")

        def run_wedged(i: int) -> None:
            job = store.create_job(f"wedged-{i}")
            task = Task(job_id=job.id, role="cursor", instruction="e2e")
            proc = _WedgedProc()
            try:
                orch._wait_for_worker(proc, job, [task])
                failures.append(f"wedged-{i}:not-killed")
            except RuntimeError:
                pass
            if not proc.killed:
                failures.append(f"wedged-{i}:terminate-not-called")
            events = [e["event"] for e in store.read_events(job.id)]
            if "worker.timed_out" not in events:
                failures.append(f"wedged-{i}:no-timeout-event")

        with patch.object(Orchestrator, "_worker_wait_timeout", staticmethod(lambda tasks: 0)), \
             patch.object(Orchestrator, "_worker_hard_cap",
                          staticmethod(lambda tasks, base: 9999)):
            with ThreadPoolExecutor(max_workers=16) as pool:
                live = [pool.submit(run_progressing, i) for i in range(workers)]
                for fut in live:
                    fut.result()

        # Wedged needs hard_cap small so it kills immediately (no 9999 wait).
        with patch.object(Orchestrator, "_worker_wait_timeout", staticmethod(lambda tasks: 0)), \
             patch.object(Orchestrator, "_worker_hard_cap", staticmethod(lambda tasks, base: 0)):
            with ThreadPoolExecutor(max_workers=16) as pool:
                wedged = [pool.submit(run_wedged, i) for i in range(workers)]
                for fut in wedged:
                    fut.result()

    passed = not failures
    detail = (
        f"{workers} live workers extended (never killed), {workers} wedged workers killed"
        if passed else f"{failures[:5]}"
    )
    return Result("A4 progress-based timeout extension", passed, detail, {
        "live_workers": workers, "wedged_workers": workers,
    })


# --------------------------------------------------------------------------- #
# B3+C1 — write_scope gate + conflict prediction at scale
# --------------------------------------------------------------------------- #

def scenario_b3c1_conflicts(quick: bool) -> Result:
    # Construct buckets by distinct top-level dir: same bucket overlaps, cross
    # bucket never does. Expected conflicts are exactly the within-bucket pairs.
    buckets = 6 if quick else 15
    per_bucket = 4 if quick else 8
    scoped: list[tuple[str, list[str]]] = []
    for b in range(buckets):
        for k in range(per_bucket):
            tid = f"b{b}-t{k}"
            # Both globs share the bucket's `sub` subtree, so every within-bucket
            # pair overlaps (the broad glob is an ancestor of the deeper one);
            # different buckets share no prefix, so they never overlap.
            scope = [f"bucket{b}/sub/**"] if k % 2 == 0 else [f"bucket{b}/sub/deep/**"]
            scoped.append((tid, scope))
    random.shuffle(scoped)

    t0 = time.monotonic()
    conflicts = predict_write_conflicts(scoped)
    elapsed = time.monotonic() - t0

    expected = buckets * (per_bucket * (per_bucket - 1) // 2)
    count_ok = len(conflicts) == expected
    within_bucket = all(
        c["tasks"][0].split("-")[0] == c["tasks"][1].split("-")[0] for c in conflicts
    )
    # Spot-check the overlap primitive both ways.
    primitive_ok = (
        scopes_overlap(["src/api/**"], ["src/api/routes.py"])
        and not scopes_overlap(["src/api/**"], ["src/ui/**"])
        and scopes_overlap(["pkg/**"], ["pkg/**"])
    )

    failures = []
    if not count_ok:
        failures.append(f"count={len(conflicts)}!=expected{expected}")
    if not within_bucket:
        failures.append("cross-bucket-conflict-reported")
    if not primitive_ok:
        failures.append("scopes_overlap-primitive-wrong")

    passed = not failures
    detail = (
        f"{len(scoped)} tasks -> {len(conflicts)} conflicts (exact), "
        f"predicted in {elapsed*1000:.0f}ms"
        if passed else f"{failures}"
    )
    return Result("B3+C1 conflict prediction at scale", passed, detail, {
        "tasks": len(scoped), "conflicts": len(conflicts), "predict_ms": round(elapsed * 1000, 1),
    })


def scenario_b3c1_write_scope(quick: bool) -> Result:
    if not _has_git():
        return Result("B3+C1 write_scope gate enforcement", False, "git not available")
    repos = 6 if quick else 20
    failures: list[str] = []
    with TemporaryDirectory() as tmp:
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        for i in range(repos):
            repo = _git_repo(Path(tmp) / f"repo{i}")
            (repo / "src").mkdir()
            (repo / "src" / "ok.py").write_text("in scope\n")
            (repo / "stray.py").write_text("out of scope\n")
            task = _task(write_scope=["src/**"], cwd=str(repo))
            if evaluate_task_gates(task, [], store, worker_id=f"w{i}", cwd=repo).passed:
                failures.append(f"repo{i}:stray-allowed")
            (repo / "stray.py").unlink()
            if not evaluate_task_gates(task, [], store, worker_id=f"w{i}", cwd=repo).passed:
                failures.append(f"repo{i}:in-scope-blocked")
    passed = not failures
    detail = f"{repos} repos: out-of-scope writes blocked, in-scope allowed" if passed else f"{failures[:5]}"
    return Result("B3+C1 write_scope gate enforcement", passed, detail, {"repos": repos})


# --------------------------------------------------------------------------- #
# C2 — generated-artifact stripping from auto-commits
# --------------------------------------------------------------------------- #

def scenario_c2_strip(quick: bool) -> Result:
    if not _has_git():
        return Result("C2 strip generated artifacts from commit", False, "git not available")
    repos = 6 if quick else 20
    failures: list[str] = []
    with TemporaryDirectory() as tmp:
        store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
        store.init()
        for i in range(repos):
            repo = _git_repo(Path(tmp) / f"repo{i}")
            (repo / "real.txt").write_text("real change\n")
            (repo / "parity-scoreboard.json").write_text('{"generated": true}\n')
            (repo / "coverage.xml").write_text("<coverage/>\n")
            task = _task(
                gates=[{
                    "kind": "committed", "auto": True, "message": "feat: real",
                    "exclude": ["parity-scoreboard.json", "coverage.xml"],
                }],
                cwd=str(repo),
            )
            if not evaluate_task_gates(task, [], store, worker_id=f"w{i}", cwd=repo).passed:
                failures.append(f"repo{i}:gate-failed")
                continue
            committed = _git(repo, "show", "--name-only", "--pretty=format:", "HEAD").stdout
            if "real.txt" not in committed:
                failures.append(f"repo{i}:real-not-committed")
            if "parity-scoreboard.json" in committed or "coverage.xml" in committed:
                failures.append(f"repo{i}:generated-leaked-into-commit")
            ignore = (repo / ".gitignore").read_text()
            if "parity-scoreboard.json" not in ignore:
                failures.append(f"repo{i}:not-gitignored")
    passed = not failures
    detail = f"{repos} repos: generated files excluded + gitignored, real change committed" if passed else f"{failures[:5]}"
    return Result("C2 strip generated artifacts from commit", passed, detail, {"repos": repos})


# --------------------------------------------------------------------------- #
# B2 — affected-specs mapping seam
# --------------------------------------------------------------------------- #

def scenario_b2_affected(quick: bool) -> Result:
    from puppetmaster.affected import affected_specs

    failures: list[str] = []
    # Template rules + literal + command, over a batch of changed-file sets.
    mapping = {"rules": [
        {"match": "src/*", "specs": ["tests/{stem}_test.py"]},
        {"match": "src/api/*", "specs": ["tests/api_smoke.py"]},
        {"match": "docs/*", "specs": ["tests/docs_lint.py"]},
    ]}
    batches = 50 if quick else 400
    for i in range(batches):
        changed = [f"src/mod{i}.py", f"src/api/route{i}.py", f"docs/page{i}.md"]
        specs = affected_specs(changed, mapping)
        want = {
            f"tests/mod{i}_test.py", f"tests/route{i}_test.py",
            "tests/api_smoke.py", "tests/docs_lint.py",
        }
        if not want.issubset(set(specs)):
            failures.append(f"batch{i}:missing {want - set(specs)}")
        if len(specs) != len(set(specs)):
            failures.append(f"batch{i}:dupes")

    # Glob-based spec resolution against a real tree.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tests").mkdir()
        (root / "tests" / "a_test.py").write_text("")
        (root / "tests" / "b_test.py").write_text("")
        glob_specs = affected_specs(
            ["src/a.py"], {"rules": [{"match": "src/*", "specs": ["tests/*_test.py"]}]}, cwd=root
        )
        if "tests/a_test.py" not in glob_specs or "tests/b_test.py" not in glob_specs:
            failures.append(f"glob-resolution:{glob_specs}")

    # Empty changed set -> no specs; bad mapping -> error.
    if affected_specs([], mapping) != []:
        failures.append("empty-changed-not-empty")
    try:
        affected_specs(["x"], {})
        failures.append("bad-mapping-did-not-raise")
    except ValueError:
        pass

    passed = not failures
    detail = f"{batches} mapping batches + glob + edge cases resolved correctly" if passed else f"{failures[:5]}"
    return Result("B2 affected-specs mapping seam", passed, detail, {"batches": batches})


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller volumes for a fast pass")
    parser.add_argument("--seed", type=int, default=20260608, help="RNG seed for reproducibility")
    args = parser.parse_args()
    random.seed(args.seed)

    scenarios: list[Callable[[bool], Result]] = [
        scenario_d1_delete_fuzz,
        scenario_d1_gc_guard,
        scenario_b1_ports,
        scenario_a3_require_diff,
        scenario_a2f2_status,
        scenario_a4_timeout,
        scenario_b3c1_conflicts,
        scenario_b3c1_write_scope,
        scenario_b2_affected,
        scenario_c2_strip,
    ]

    print("=" * 72)
    print(f"Puppetmaster pre-deploy stress harness ({'quick' if args.quick else 'full'} mode)")
    print("=" * 72)

    results: list[Result] = []
    for scenario in scenarios:
        t0 = time.monotonic()
        try:
            result = scenario(args.quick)
        except Exception as exc:  # one bad scenario must not halt the run
            import traceback
            result = Result(scenario.__name__, False, f"raised: {exc!r}")
            traceback.print_exc()
        took = time.monotonic() - t0
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}  ({took:.1f}s)")
        print(f"        {result.detail}")
        if result.metrics:
            print(f"        metrics: {json.dumps(result.metrics, default=str)}")
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed}/{len(results)} scenarios passed")
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name}")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
