"""Live-fire MCP stress test for Puppetmaster.

Spawns the real MCP server (`python -m puppetmaster.mcp_server`) over
stdio just like Cursor does, then drives it with a suite of stress
scenarios that mirror the user-reported "MCP drops on robust questions"
failure mode:

- many parallel tool calls
- sustained traffic over a long horizon
- large response payloads
- a slow tool call concurrent with a flurry of fast ones
- the idle -> busy transition (keepalive should stop emitting)
- the busy -> idle transition (keepalive should resume)

For every scenario we capture:
- exact line-delimited frames seen on stdout
- frames that failed to parse as JSON
- per-call latency
- whether IDs we sent got responses
- whether response/notification framing stayed clean

Run: `python -m bench.mcp_stress` from the Puppetmaster repo root.
Exit code is non-zero if any scenario surfaces a problem.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_CMD = [sys.executable, "-m", "puppetmaster.mcp_server"]


@dataclass
class Frame:
    raw: str
    parsed: Optional[dict[str, Any]]
    parse_error: Optional[str]
    received_at: float


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_errors: list[str] = field(default_factory=list)


class McpClient:
    """Minimal stdio MCP client for stress testing."""

    def __init__(self, env: Optional[dict[str, str]] = None) -> None:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        self.proc = subprocess.Popen(
            SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=str(REPO_ROOT),
            bufsize=0,
        )
        self.frames: list[Frame] = []
        self._frame_lock = threading.Lock()
        self._responses: dict[Any, dict[str, Any]] = {}
        self._response_event = threading.Event()
        self._notifications: list[dict[str, Any]] = []
        self._reader_done = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_buf: list[bytes] = []
        self._stderr_lock = threading.Lock()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            with self._stderr_lock:
                self._stderr_buf.append(line)

    def stderr_tail(self, n: int = 60) -> str:
        with self._stderr_lock:
            tail = self._stderr_buf[-n:]
        return "".join(b.decode("utf-8", errors="replace") for b in tail)

    def server_alive(self) -> bool:
        return self.proc.poll() is None

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw_bytes in self.proc.stdout:
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                with self._frame_lock:
                    self.frames.append(
                        Frame(
                            raw=raw_bytes.decode("utf-8", errors="replace"),
                            parsed=None,
                            parse_error=f"unicode: {exc}",
                            received_at=time.time(),
                        )
                    )
                continue
            stripped = raw.strip()
            if not stripped:
                continue
            parse_error: Optional[str] = None
            parsed: Optional[dict[str, Any]] = None
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
            frame = Frame(
                raw=stripped,
                parsed=parsed,
                parse_error=parse_error,
                received_at=time.time(),
            )
            with self._frame_lock:
                self.frames.append(frame)
            if parsed is None:
                continue
            if "id" in parsed and "method" not in parsed:
                self._responses[parsed["id"]] = parsed
                self._response_event.set()
            elif parsed.get("method", "").startswith("notifications/"):
                self._notifications.append(parsed)
        self._reader_done.set()

    def call(self, method: str, params: dict[str, Any], *, request_id: Any) -> dict[str, Any]:
        msg = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        return msg

    def wait_for_response(self, request_id: Any, timeout: float) -> Optional[dict[str, Any]]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if request_id in self._responses:
                return self._responses[request_id]
            if not self.server_alive():
                return None
            self._response_event.wait(timeout=0.1)
            self._response_event.clear()
        return None

    def notification_count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            return len(self._notifications)
        return sum(
            1
            for n in self._notifications
            if n.get("params", {}).get("data", {}).get("kind") == kind
        )

    def frame_count(self) -> int:
        with self._frame_lock:
            return len(self.frames)

    def parse_errors(self) -> list[Frame]:
        with self._frame_lock:
            return [f for f in self.frames if f.parse_error is not None]

    def shutdown(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)


# ----- Scenarios -----------------------------------------------------------


def scenario_parallel_burst(client: McpClient, *, n: int = 20) -> ScenarioResult:
    """Fire n tool calls back-to-back without waiting for responses, then
    confirm every id comes back exactly once with no frame corruption."""
    start = time.time()
    for i in range(n):
        client.call(
            "tools/call",
            {"name": "puppetmaster_doctor", "arguments": {}},
            request_id=1000 + i,
        )
    latencies = []
    missing = []
    for i in range(n):
        resp = client.wait_for_response(1000 + i, timeout=20)
        if resp is None:
            missing.append(1000 + i)
        else:
            latencies.append(time.time() - start)
    elapsed = time.time() - start
    parse_errors = client.parse_errors()
    passed = not missing and not parse_errors
    return ScenarioResult(
        name="parallel_burst",
        passed=passed,
        detail=(
            f"{n} parallel doctor calls in {elapsed:.2f}s; "
            f"missing={len(missing)} parse_errors={len(parse_errors)}"
        ),
        metrics={
            "n": n,
            "elapsed_s": elapsed,
            "p50_latency_s": _percentile(latencies, 50),
            "p95_latency_s": _percentile(latencies, 95),
            "missing_ids": missing,
        },
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def scenario_sustained_traffic(client: McpClient, *, duration_s: float = 30.0) -> ScenarioResult:
    """Send a steady 5-calls-per-second stream and confirm none are dropped."""
    start = time.time()
    sent = 0
    while time.time() - start < duration_s:
        client.call(
            "tools/call",
            {"name": "puppetmaster_doctor", "arguments": {}},
            request_id=2000 + sent,
        )
        sent += 1
        time.sleep(0.2)
    missing = []
    for i in range(sent):
        resp = client.wait_for_response(2000 + i, timeout=10)
        if resp is None:
            missing.append(2000 + i)
    parse_errors = client.parse_errors()
    elapsed = time.time() - start
    passed = not missing and not parse_errors
    return ScenarioResult(
        name="sustained_traffic",
        passed=passed,
        detail=(
            f"{sent} calls over {elapsed:.1f}s @ {sent / elapsed:.1f}/s; "
            f"missing={len(missing)} parse_errors={len(parse_errors)}"
        ),
        metrics={"sent": sent, "missing": len(missing), "elapsed_s": elapsed},
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def scenario_large_payload(client: McpClient) -> ScenarioResult:
    """Confirm a 30+ KB response frame stays intact (the user saw ~41KB earlier)."""
    client.call(
        "tools/call",
        {"name": "puppetmaster_doctor", "arguments": {}},
        request_id=3001,
    )
    resp = client.wait_for_response(3001, timeout=20)
    if resp is None:
        return ScenarioResult(
            name="large_payload",
            passed=False,
            detail="no response within 20s",
            frame_errors=[],
        )
    serialized = json.dumps(resp)
    parse_errors = client.parse_errors()
    passed = not parse_errors
    return ScenarioResult(
        name="large_payload",
        passed=passed,
        detail=(
            f"doctor response = {len(serialized)} bytes; "
            f"parse_errors={len(parse_errors)}"
        ),
        metrics={"response_bytes": len(serialized)},
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def scenario_idle_keepalive_then_burst(
    client: McpClient,
) -> ScenarioResult:
    """Wait for idle keepalive to start emitting, then fire a burst and
    confirm the keepalive suppresses cleanly while calls are active and
    resumes after — all without corrupting any frames."""
    # Wait for at least 2 idle keepalive frames to confirm the thread is alive.
    deadline = time.time() + 60
    while client.notification_count("idle_keepalive") < 2:
        if time.time() > deadline:
            return ScenarioResult(
                name="idle_keepalive_then_burst",
                passed=False,
                detail="never received 2 idle_keepalive frames in 60s",
                frame_errors=[],
            )
        time.sleep(0.5)
    idle_before = client.notification_count("idle_keepalive")
    # Now drive 10 fast calls. While these are running, the suppression
    # gate should hold off the idle pinger.
    for i in range(10):
        client.call(
            "tools/call",
            {"name": "puppetmaster_doctor", "arguments": {}},
            request_id=4000 + i,
        )
    for i in range(10):
        client.wait_for_response(4000 + i, timeout=20)
    idle_after_burst = client.notification_count("idle_keepalive")
    # And let the server settle so idle pings resume.
    time.sleep(30)
    idle_after_settle = client.notification_count("idle_keepalive")
    parse_errors = client.parse_errors()
    passed = (
        idle_after_settle > idle_after_burst  # idle keepalive resumed
        and not parse_errors
    )
    return ScenarioResult(
        name="idle_keepalive_then_burst",
        passed=passed,
        detail=(
            f"idle frames: before={idle_before}, post-burst={idle_after_burst}, "
            f"after_settle={idle_after_settle}; parse_errors={len(parse_errors)}"
        ),
        metrics={
            "idle_before": idle_before,
            "idle_after_burst": idle_after_burst,
            "idle_after_settle": idle_after_settle,
        },
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def scenario_mixed_with_codegraph(client: McpClient) -> ScenarioResult:
    """Mix doctor (fast) + codegraph_status (potentially slower) requests.
    Exercises the dual-keepalive paths and threadpool dispatching."""
    start = time.time()
    for i in range(10):
        client.call(
            "tools/call",
            {"name": "puppetmaster_doctor", "arguments": {}},
            request_id=5000 + i,
        )
    for i in range(5):
        client.call(
            "tools/call",
            {
                "name": "puppetmaster_codegraph_status",
                "arguments": {"cwd": str(REPO_ROOT)},
            },
            request_id=5100 + i,
        )
    missing = []
    for i in range(10):
        if client.wait_for_response(5000 + i, timeout=30) is None:
            missing.append(5000 + i)
    for i in range(5):
        if client.wait_for_response(5100 + i, timeout=60) is None:
            missing.append(5100 + i)
    elapsed = time.time() - start
    parse_errors = client.parse_errors()
    passed = not missing and not parse_errors
    return ScenarioResult(
        name="mixed_with_codegraph",
        passed=passed,
        detail=(
            f"15 mixed calls in {elapsed:.2f}s; "
            f"missing={missing} parse_errors={len(parse_errors)}"
        ),
        metrics={"elapsed_s": elapsed, "missing": len(missing)},
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def scenario_giant_parallel(client: McpClient, *, n: int = 64) -> ScenarioResult:
    """Hit harder than the default 8-worker pool to exercise queue backpressure."""
    start = time.time()
    sender_threads = []
    sent_ids = list(range(6000, 6000 + n))

    def send(rid: int) -> None:
        client.call(
            "tools/call",
            {"name": "puppetmaster_doctor", "arguments": {}},
            request_id=rid,
        )

    for rid in sent_ids:
        t = threading.Thread(target=send, args=(rid,))
        t.start()
        sender_threads.append(t)
    for t in sender_threads:
        t.join()
    missing = []
    for rid in sent_ids:
        if client.wait_for_response(rid, timeout=60) is None:
            missing.append(rid)
    elapsed = time.time() - start
    parse_errors = client.parse_errors()
    passed = not missing and not parse_errors
    return ScenarioResult(
        name="giant_parallel",
        passed=passed,
        detail=(
            f"{n} simultaneous senders in {elapsed:.2f}s; "
            f"missing={len(missing)} parse_errors={len(parse_errors)}"
        ),
        metrics={"n": n, "elapsed_s": elapsed, "missing": len(missing)},
        frame_errors=[f.raw[:200] for f in parse_errors],
    )


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


# ----- Driver --------------------------------------------------------------


def main() -> int:
    print("Spinning up MCP server for stress test...")
    # Keep keepalive interval short so the idle-resume scenario doesn't take forever.
    client = McpClient(
        env={
            "PUPPETMASTER_MCP_IDLE_KEEPALIVE_INTERVAL_SECONDS": "5",
            "PUPPETMASTER_MCP_INPUT_STALE_DISABLED": "1",  # don't self-terminate mid-test
        }
    )
    # Initial tools/list as a warm-up + sanity check.
    client.call("tools/list", {}, request_id=0)
    handshake = client.wait_for_response(0, timeout=15)
    if handshake is None:
        print("FAIL: handshake did not return")
        client.shutdown()
        return 2
    print(f"  handshake OK ({len(handshake.get('result', {}).get('tools', []))} tools)")

    scenarios = [
        ("parallel_burst", lambda: scenario_parallel_burst(client, n=20)),
        ("giant_parallel", lambda: scenario_giant_parallel(client, n=64)),
        ("large_payload", lambda: scenario_large_payload(client)),
        ("mixed_with_codegraph", lambda: scenario_mixed_with_codegraph(client)),
        ("sustained_traffic", lambda: scenario_sustained_traffic(client, duration_s=15.0)),
        ("idle_keepalive_then_burst", lambda: scenario_idle_keepalive_then_burst(client)),
    ]

    results: list[ScenarioResult] = []
    try:
        for name, runner in scenarios:
            if not client.server_alive():
                print(f"\n!!! MCP server already dead before scenario `{name}` !!!")
                print(f"--- server stderr tail ---\n{client.stderr_tail()}")
                results.append(
                    ScenarioResult(
                        name=name,
                        passed=False,
                        detail="server dead before scenario started",
                    )
                )
                break
            print(f"\n>>> Scenario: {name}")
            t0 = time.time()
            try:
                r = runner()
            except Exception as exc:  # don't let one bad scenario halt the run
                r = ScenarioResult(
                    name=name, passed=False, detail=f"raised: {exc!r}"
                )
            took = time.time() - t0
            status = "PASS" if r.passed else "FAIL"
            print(f"    {status} ({took:.1f}s) — {r.detail}")
            if r.metrics:
                print(f"      metrics: {json.dumps(r.metrics, default=str)}")
            if r.frame_errors:
                print(f"      frame_errors sample: {r.frame_errors[:3]}")
            if not client.server_alive():
                print(f"    !!! server died during scenario; exit_code={client.proc.poll()}")
                print(f"    --- server stderr tail ---\n{client.stderr_tail()}")
                r.detail += f" [server died exit={client.proc.poll()}]"
                r.passed = False
            results.append(r)
    finally:
        client.shutdown()

    print("\n=== Summary ===")
    pass_count = sum(1 for r in results if r.passed)
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}: {r.name} — {r.detail}")
    print(f"\n{pass_count}/{len(results)} scenarios passed")
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
