"""Mock stress harness for the v0.9.20 streamed-adapter change.

v0.9.20 routed the Codex and Claude Code adapters through the same
``run_streamed_subprocess`` path the Cursor implement adapter has used since
v0.9.12: a live sidecar log + "still working" heartbeat, and — critically —
a child stdin closed to ``DEVNULL`` so a CLI that reads stdin on a
non-interactive worker can never wedge forever (the silent "stall" a field
user hit with Codex).

This harness exercises that change *without spending a cent*: every "agent
CLI" is a local fake that emits the exact stdout shape the real parsers
accept. It is a stress test, not a unit test — it hammers the streamed runner
concurrently and drives both other-platform adapters end to end.

Because the streamed path is what changed for the **other platforms**
(Codex / Claude Code), the harness first lifts the Cursor platform lock so
those adapters are enabled, runs the battery, then **restores the Cursor lock
in a guaranteed ``finally``** — the machine ends exactly where it started.

Run:  ``python bench/stress_streamed_adapters.py``   (exit 0 = all green).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import run as _run

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PY = sys.executable
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(job: str, tid: str, *, adapter: str, payload: dict):
    from puppetmaster.models import Task

    return Task(
        job_id=job,
        id=tid,
        role=f"{adapter}-review",
        instruction="mock stress task",
        adapter=adapter,
        payload=payload,
    )


def _live_log_text(streamed) -> str:
    path = streamed.live_log_path
    if not path or not Path(path).is_file():
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _git_init(path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "pm", "GIT_AUTHOR_EMAIL": "pm@x",
           "GIT_COMMITTER_NAME": "pm", "GIT_COMMITTER_EMAIL": "pm@x"}
    _run(["git", "init", "-q"], cwd=path, env=env, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run(["git", "add", "-A"], cwd=path, env=env, check=True)
    _run(["git", "commit", "-qm", "seed"], cwd=path, env=env, check=True)


# Fake agent CLIs. Each ignores every flag the command builders append and
# emits the stdout shape the real parser tolerates, with a couple of slow,
# flushed lines so the streamed runner has real output to tee live.
_FAKE_CODEX = textwrap.dedent(
    '''
    import json, sys, time
    agent = json.dumps({"artifacts": [
        {"type": "finding", "claim": "mock ok", "evidence": ["a.py:1"], "confidence": 0.9}]})
    for ev in [
        {"type": "thread.started", "thread_id": "th_mock"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": agent}},
        {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 7,
                                             "cached_input_tokens": 0, "reasoning_output_tokens": 0}},
    ]:
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush(); time.sleep(0.02)
    '''
).strip()

_FAKE_CLAUDE = textwrap.dedent(
    '''
    import json, sys, time
    for line in ["editing files...", "running checks..."]:
        sys.stdout.write(line + "\\n"); sys.stdout.flush(); time.sleep(0.02)
    sys.stdout.write(json.dumps({"result": "ok", "usage": {"input_tokens": 9, "output_tokens": 4}}) + "\\n")
    sys.stdout.flush()
    '''
).strip()


def _write_fake(tmp: Path, name: str, body: str) -> str:
    script = tmp / f"{name}.py"
    script.write_text(body + "\n", encoding="utf-8")
    # Run via the interpreter so we never depend on a chmod/shebang.
    return f"{PY} {script}"


# ---------------------------------------------------------------------------
# 1. Platform lock — lift the Cursor lock so the other platforms are testable
# ---------------------------------------------------------------------------


def lift_cursor_lock() -> dict:
    """Snapshot the current lock, assert it is Cursor-only, then enable all.

    Returns the snapshot the ``finally`` uses to restore the exact prior state.
    """
    section("1. Platform lock — lift Cursor lock to expose other platforms")
    from puppetmaster import platform_lock as pl

    path = pl.platform_config_path()
    snapshot = {"path": path, "existed": path.is_file(),
                "bytes": path.read_bytes() if path.is_file() else None}

    before = pl.enabled_adapters()
    check("Cursor lock active at start (codex disabled)", "codex" not in before, str(sorted(before)))
    check("Cursor lock active at start (claude-code disabled)", "claude-code" not in before, str(sorted(before)))
    check("cursor itself enabled", "cursor" in before, str(sorted(before)))

    pl.reset()  # the "disable cursor lock" step — every known adapter back on
    after = pl.enabled_adapters()
    check("after reset: codex enabled", "codex" in after, str(sorted(after)))
    check("after reset: claude-code enabled", "claude-code" in after, str(sorted(after)))
    check("after reset: not restricted", not pl.is_restricted(), str(sorted(after)))
    return snapshot


def restore_cursor_lock(snapshot: dict) -> None:
    """Guaranteed restore: put the lock file back byte-for-byte, then verify
    Cursor lock is active again (codex/claude-code blocked)."""
    section("5. Restore Cursor lock (guaranteed)")
    from puppetmaster import platform_lock as pl

    path: Path = snapshot["path"]
    if snapshot["existed"] and snapshot["bytes"] is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(snapshot["bytes"])
    elif path.is_file():
        # There was no lock file before; remove the one reset() may have left.
        path.unlink()

    enabled = pl.enabled_adapters()
    check("Cursor lock restored: codex blocked again", "codex" not in enabled, str(sorted(enabled)))
    check("Cursor lock restored: claude-code blocked again", "claude-code" not in enabled, str(sorted(enabled)))
    check("Cursor lock restored: cursor still enabled", "cursor" in enabled, str(sorted(enabled)))


# ---------------------------------------------------------------------------
# 2. Streamed runner — the actual new code path, under concurrency
# ---------------------------------------------------------------------------


def stress_streamed_runner() -> None:
    section("2. Streamed runner: stdin-EOF, live log, heartbeat, timeout")
    from puppetmaster.adapters import run_streamed_subprocess

    # 2a. The core fix: stdin is closed (DEVNULL). A CLI that reads stdin must
    #     see EOF and exit — never hang — and it must hold under concurrency.
    workers = 24

    def _stdin_probe(i: int):
        task = _make_task("job-stdin", f"t-stdin-{i}", adapter="codex", payload={})
        return run_streamed_subprocess(
            command=[PY, "-c",
                     "import sys; d=sys.stdin.read(); sys.stdout.write('EOF' if d=='' else 'BLOCKED')"],
            env=None, task=task, sidecar_name="stdin_probe", timeout_seconds=15,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_stdin_probe, range(workers)))
    no_hang = all(not r.timed_out for r in results)
    all_eof = all("EOF" in r.stdout and "BLOCKED" not in r.stdout for r in results)
    all_rc0 = all(r.returncode == 0 for r in results)
    check(f"{workers} concurrent stdin-readers all saw EOF (no hang)", no_hang and all_eof and all_rc0,
          f"hang={not no_hang} eof={all_eof} rc0={all_rc0}")

    # 2b. Live sidecar log grows with a "still working" heartbeat, and every
    #     emitted line survives in both the buffer and the on-disk log.
    n_lines = 40
    producer = (
        "import sys, time\n"
        f"for i in range({n_lines}):\n"
        "    sys.stdout.write(f'line-{i}\\n'); sys.stdout.flush(); time.sleep(0.03)\n"
    )
    task = _make_task("job-live", "t-live", adapter="codex", payload={})
    streamed = run_streamed_subprocess(
        command=[PY, "-c", producer], env=None, task=task,
        sidecar_name="live_probe", timeout_seconds=20, heartbeat_seconds=0.25,
    )
    log = _live_log_text(streamed)
    check("live log file written", bool(streamed.live_log_path) and bool(log), streamed.live_log_path or "none")
    check("every stdout line captured in buffer", streamed.stdout.count("line-") == n_lines,
          f"got {streamed.stdout.count('line-')}/{n_lines}")
    check("every stdout line teed to live log", log.count("line-") == n_lines,
          f"got {log.count('line-')}/{n_lines}")
    check("heartbeat 'still working' written", "still working" in log)
    check("exit footer written to live log", "process exited" in log and "timed_out=False" in log)
    check("runner reports not timed out", not streamed.timed_out)

    # 2c. Timeout semantics unchanged: a sleeper past the deadline is killed,
    #     reported timed_out, and the live log still records the kill.
    task = _make_task("job-timeout", "t-timeout", adapter="codex", payload={})
    t0 = time.monotonic()
    streamed = run_streamed_subprocess(
        command=[PY, "-c", "import time; print('starting', flush=True); time.sleep(30)"],
        env=None, task=task, sidecar_name="timeout_probe", timeout_seconds=1, heartbeat_seconds=0.25,
    )
    elapsed = time.monotonic() - t0
    check("sleeper past deadline is killed promptly", streamed.timed_out and elapsed < 8,
          f"timed_out={streamed.timed_out} elapsed={elapsed:.1f}s")
    tlog = _live_log_text(streamed)
    check("timeout recorded in live log", "timed_out=True" in tlog, tlog[-160:])


# ---------------------------------------------------------------------------
# 3. Codex adapter end to end (other platform), fake CLI through streamed path
# ---------------------------------------------------------------------------


def stress_codex_adapter(state_dir: Path) -> None:
    section("3. Codex adapter (other platform) through the streamed path")
    from puppetmaster.adapters import CodexAdapter

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        repo = tmp / "repo"
        repo.mkdir()
        _git_init(repo)
        fake = _write_fake(tmp, "fake_codex", _FAKE_CODEX)

        def _one(i: int):
            task = _make_task("job-codex", f"t-codex-{i}", adapter="codex", payload={
                "executable": fake, "cwd": str(repo), "sandbox": "read-only",
                "disable_codegraph": True, "timeout_seconds": 30,
            })
            return CodexAdapter().run(task, "goal", f"worker-{i}")

        with ThreadPoolExecutor(max_workers=6) as pool:
            batches = list(pool.map(_one, range(6)))

        verifs = [b[0] for b in batches]
        all_passed = all(v.payload["result"] == "passed" for v in verifs)
        check("6 concurrent codex runs all passed", all_passed,
              str([v.payload.get("result") for v in verifs]))
        live_logs = [v.payload.get("live_log") for v in verifs]
        check("every codex verification carries a live_log path", all(bool(p) for p in live_logs))
        files_ok = all(p and Path(p).is_file() and Path(p).stat().st_size > 0 for p in live_logs)
        check("every codex live_log file exists and is non-empty (real streamed run)", files_ok)
        footers_ok = all("process exited" in Path(p).read_text(errors="replace") for p in live_logs if p)
        check("every codex live_log has the streamed exit footer", footers_ok)
        toks_ok = all(v.payload.get("tokens_in") == 11 and v.payload.get("tokens_out") == 7 for v in verifs)
        check("codex JSONL usage parsed from streamed stdout", toks_ok,
              str([(v.payload.get("tokens_in"), v.payload.get("tokens_out")) for v in verifs]))


# ---------------------------------------------------------------------------
# 4. Claude Code adapter end to end (other platform) through streamed path
# ---------------------------------------------------------------------------


def stress_claude_adapter(state_dir: Path) -> None:
    section("4. Claude Code adapter (other platform) through the streamed path")
    from puppetmaster.adapters import ClaudeCodeAdapter

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        repo = tmp / "repo"
        repo.mkdir()
        _git_init(repo)
        fake = _write_fake(tmp, "fake_claude", _FAKE_CLAUDE)

        def _one(i: int):
            task = _make_task("job-claude", f"t-claude-{i}", adapter="claude-code", payload={
                "executable": fake, "cwd": str(repo),
                "disable_codegraph": True, "timeout_seconds": 30,
            })
            return ClaudeCodeAdapter().run(task, "goal", f"worker-{i}")

        with ThreadPoolExecutor(max_workers=6) as pool:
            batches = list(pool.map(_one, range(6)))

        verifs = [b[0] for b in batches]
        all_passed = all(v.payload["result"] == "passed" for v in verifs)
        check("6 concurrent claude-code runs all passed", all_passed,
              str([v.payload.get("result") for v in verifs]))
        live_logs = [v.payload.get("live_log") for v in verifs]
        check("every claude verification carries a live_log path", all(bool(p) for p in live_logs))
        files_ok = all(p and Path(p).is_file() and Path(p).stat().st_size > 0 for p in live_logs)
        check("every claude live_log file exists and is non-empty (real streamed run)", files_ok)
        footers_ok = all("process exited" in Path(p).read_text(errors="replace") for p in live_logs if p)
        check("every claude live_log has the streamed exit footer", footers_ok)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("Puppetmaster v0.9.20 streamed adapters — mock stress test")

    # Sidecar live logs only spool when PUPPETMASTER_STATE_DIR is set.
    tmp_state = tempfile.mkdtemp(prefix="pm-stress-state-")
    os.environ["PUPPETMASTER_STATE_DIR"] = tmp_state

    snapshot = lift_cursor_lock()
    try:
        stress_streamed_runner()
        stress_codex_adapter(Path(tmp_state))
        stress_claude_adapter(Path(tmp_state))
    finally:
        # Non-negotiable: the box ends on the Cursor lock it started on, even
        # if an assertion above blew up mid-run.
        restore_cursor_lock(snapshot)

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL GREEN — streamed Codex/Claude adapters validated, Cursor lock restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
