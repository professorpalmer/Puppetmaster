"""Mock stress harness for the Claude Fable 5 rollout.

Validates the Fable 5 changes end to end across ALL platforms without
spending a cent: every "agent CLI" is a local fake emitting the exact
stdout/stderr shapes the real parsers and failure classifiers accept.

What it proves, per platform (cursor / claude-code / codex / openai):

1. Failure classification — provider-shaped "model not available on this
   account" responses classify as ``model_unavailable`` (and unrelated
   errors do NOT).
2. Routing — with the full starter registry, a frontier-grade task picks
   Fable 5 wherever the platform exposes it, degrades to the strongest
   available model where it doesn't, and trivial tasks never reach it.
3. Adapter end-to-end — the claude-code and codex adapters, driven through
   the real streamed-subprocess path by fake CLIs, surface
   ``failure=model_unavailable`` on a no-Fable account and pass on a
   with-Fable account. (openai is HTTP-based and cursor needs the live SDK,
   so those two are covered at the classifier + reroute seams instead.)
4. Auto-fallback — the orchestrator re-queues a ``model_unavailable``
   failure onto a different funded platform and records the fallback on a
   ROUTING artifact.

Because the box runs a Cursor-only platform lock, the harness lifts the
lock first and **restores it in a guaranteed ``finally``** — the machine
ends exactly where it started.

Run:  ``python bench/stress_fable5_rollout.py``   (exit 0 = all green).
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path
from subprocess import run as _run
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PY = sys.executable
FAILURES: list = []

FRONTIER_INSTRUCTION = (
    "Perform a deep security audit of the entire authentication architecture "
    "across every module, design a hardened replacement, and produce a "
    "complex cross-repo migration plan."
)
TRIVIAL_INSTRUCTION = "fix a typo in the README comment"


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Platform lock lift / restore (mirrors bench/stress_streamed_adapters.py)
# ---------------------------------------------------------------------------


def lift_cursor_lock() -> dict:
    section("1. Platform lock — lift Cursor lock to expose every platform")
    from puppetmaster import platform_lock as pl

    path = pl.platform_config_path()
    snapshot = {"path": path, "existed": path.is_file(),
                "bytes": path.read_bytes() if path.is_file() else None}

    before = pl.enabled_adapters()
    check("cursor enabled at start", "cursor" in before, str(sorted(before)))

    pl.reset()
    after = pl.enabled_adapters()
    for adapter in ("cursor", "claude-code", "codex", "openai"):
        check(f"after reset: {adapter} enabled", adapter in after, str(sorted(after)))
    return snapshot


def restore_cursor_lock(snapshot: dict) -> None:
    section("6. Restore platform lock (guaranteed)")
    from puppetmaster import platform_lock as pl

    path: Path = snapshot["path"]
    if snapshot["existed"] and snapshot["bytes"] is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(snapshot["bytes"])
    elif path.is_file():
        path.unlink()

    enabled = pl.enabled_adapters()
    check("lock restored: cursor still enabled", "cursor" in enabled, str(sorted(enabled)))
    check("lock restored: codex blocked again", "codex" not in enabled, str(sorted(enabled)))
    check("lock restored: claude-code blocked again", "claude-code" not in enabled, str(sorted(enabled)))


# ---------------------------------------------------------------------------
# 2. Failure-classification matrix (all four platforms)
# ---------------------------------------------------------------------------


def stress_failure_classification() -> None:
    section("2. model_unavailable classification on every platform")
    from puppetmaster.adapters import (
        classify_claude_code_failure,
        classify_codex_failure,
        classify_cursor_failure,
        classify_openai_failure,
    )

    cases = [
        ("claude-code: not_found_error JSON",
         classify_claude_code_failure(
             '{"type":"error","error":{"type":"not_found_error","message":"model: claude-fable-5"}}')),
        ("claude-code: permission_error JSON",
         classify_claude_code_failure(
             '{"type":"error","error":{"type":"permission_error","message":"model claude-fable-5 is not permitted on this plan"}}')),
        ("cursor: forbidden-model rejection",
         classify_cursor_failure("forbidden-model: fable-5 is not on your plan")),
        ("cursor: unknown model rejection",
         classify_cursor_failure("unknown model fable-5 rejected by Cursor SDK")),
        ("codex: model_not_found code",
         classify_codex_failure('{"error":{"code":"model_not_found"}}')),
        ("openai: model_not_found code",
         classify_openai_failure('{"error":{"code":"model_not_found"}}', None)),
        ("openai: HTTP 404",
         classify_openai_failure("", 404)),
    ]
    for name, got in cases:
        check(name, got == "model_unavailable", f"got {got!r}")

    negative = [
        ("claude-code: billing stays billing_or_quota",
         classify_claude_code_failure("Credit balance is too low"), "billing_or_quota"),
        ("codex: auth stays not_authenticated",
         classify_codex_failure("Not logged in"), "not_authenticated"),
        ("cursor: timeout stays timeout",
         classify_cursor_failure("operation timed out"), "timeout"),
        ("openai: rate limit stays rate_limit",
         classify_openai_failure("Rate limit exceeded", None), "rate_limit"),
    ]
    for name, got, want in negative:
        check(name, got == want, f"got {got!r}, want {want!r}")


# ---------------------------------------------------------------------------
# 3. Routing matrix — every platform lock shape against the starter registry
# ---------------------------------------------------------------------------


def stress_routing_matrix() -> None:
    section("3. Routing matrix (starter registry, per-platform locks)")
    from puppetmaster.model_registry import starter_registry
    from puppetmaster.router import TaskSignals, route_task

    registry = starter_registry()

    def frontier(allowed):
        return route_task(
            TaskSignals(instruction=FRONTIER_INSTRUCTION, role="security-review",
                        allowed_adapters=allowed),
            registry,
        )

    decision = frontier(None)
    check("all platforms: picks cursor/claude-fable-5 (plan-billed $0)",
          decision.model.id == "cursor/claude-fable-5" and decision.estimated_cost_usd == 0.0,
          f"got {decision.model.id} @ ${decision.estimated_cost_usd:.4f}")
    check("all platforms: demanded capability is 100",
          decision.capability_needed == 100, str(decision.capability_needed))

    decision = frontier(frozenset({"cursor"}))
    check("cursor-only lock: picks cursor/claude-fable-5",
          decision.model.id == "cursor/claude-fable-5", decision.model.id)

    decision = frontier(frozenset({"claude-code"}))
    check("claude-code-only lock: picks claude-code/fable-5",
          decision.model.id == "claude-code/fable-5", decision.model.id)

    decision = frontier(frozenset({"codex"}))
    check("codex-only lock: degrades to strongest available (codex/gpt-5-5)",
          decision.model.id == "codex/gpt-5-5", decision.model.id)
    check("codex-only lock: capability gap stated in reason",
          "NO model meets capability need" in decision.reason, decision.reason)

    decision = frontier(frozenset({"openai"}))
    check("openai-only lock: degrades to strongest available (openai/gpt-5-5)",
          decision.model.id == "openai/gpt-5-5", decision.model.id)

    trivial = route_task(
        TaskSignals(instruction=TRIVIAL_INSTRUCTION, role="shell"), registry
    )
    check("trivial task never reaches Fable 5",
          "fable" not in trivial.model.id, trivial.model.id)


# ---------------------------------------------------------------------------
# 4. Adapter end-to-end with fake CLIs (claude-code + codex, streamed path)
# ---------------------------------------------------------------------------

# A "no-Fable account": the CLI rejects the model the way the real provider
# does, on stderr, with a non-zero exit.
_FAKE_CLAUDE_NO_FABLE = textwrap.dedent(
    '''
    import sys
    sys.stderr.write('{"type":"error","error":{"type":"not_found_error","message":"model: claude-fable-5"}}\\n')
    sys.exit(1)
    '''
).strip()

_FAKE_CLAUDE_WITH_FABLE = textwrap.dedent(
    '''
    import json, sys
    sys.stdout.write(json.dumps({"result": "ok", "usage": {"input_tokens": 9, "output_tokens": 4}}) + "\\n")
    '''
).strip()

_FAKE_CODEX_NO_FABLE = textwrap.dedent(
    '''
    import sys
    sys.stderr.write('{"error":{"code":"model_not_found","message":"fable-5 is not available"}}\\n')
    sys.exit(2)
    '''
).strip()

_FAKE_CODEX_WITH_FABLE = textwrap.dedent(
    '''
    import json, sys
    agent = json.dumps({"artifacts": [
        {"type": "finding", "claim": "mock ok", "evidence": ["a.py:1"], "confidence": 0.9}]})
    for ev in [
        {"type": "thread.started", "thread_id": "th_mock"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": agent}},
        {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 7,
                                             "cached_input_tokens": 0, "reasoning_output_tokens": 0}},
    ]:
        sys.stdout.write(json.dumps(ev) + "\\n")
    '''
).strip()


def _git_init(path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "pm", "GIT_AUTHOR_EMAIL": "pm@x",
           "GIT_COMMITTER_NAME": "pm", "GIT_COMMITTER_EMAIL": "pm@x"}
    _run(["git", "init", "-q"], cwd=path, env=env, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run(["git", "add", "-A"], cwd=path, env=env, check=True)
    _run(["git", "commit", "-qm", "seed"], cwd=path, env=env, check=True)


def _write_fake(tmp: Path, name: str, body: str) -> str:
    script = tmp / f"{name}.py"
    script.write_text(body + "\n", encoding="utf-8")
    return f"{PY} {script}"


def _make_task(job: str, tid: str, *, adapter: str, payload: dict):
    from puppetmaster.models import Task

    return Task(job_id=job, id=tid, role="implement",
                instruction="frontier mock task", adapter=adapter, payload=payload)


def _run_adapter_verification(adapter_cls, task, worker_id: str):
    """Run an adapter and return its verification artifact.

    Adapters return either ``(run, [artifacts])`` or a bare artifact list.
    """
    result = adapter_cls().run(task, "goal", worker_id)
    artifacts = result[1] if isinstance(result, tuple) else result
    return next(a for a in artifacts if a.type.value == "verification")


def stress_adapters_end_to_end() -> None:
    section("4. Adapters end-to-end: no-Fable account fails as model_unavailable")
    from puppetmaster.adapters import ClaudeCodeAdapter, CodexAdapter

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        repo = tmp / "repo"
        repo.mkdir()
        _git_init(repo)

        matrix = [
            ("claude-code", ClaudeCodeAdapter, "claude-fable-5",
             _FAKE_CLAUDE_NO_FABLE, _FAKE_CLAUDE_WITH_FABLE),
            ("codex", CodexAdapter, "fable-5",
             _FAKE_CODEX_NO_FABLE, _FAKE_CODEX_WITH_FABLE),
        ]
        for adapter_name, adapter_cls, model, no_fable, with_fable in matrix:
            base_payload = {"cwd": str(repo), "disable_codegraph": True,
                            "timeout_seconds": 30, "model": model}
            if adapter_name == "codex":
                base_payload["sandbox"] = "read-only"

            fake = _write_fake(tmp, f"fake_{adapter_name}_denied", no_fable)
            task = _make_task("job-fable", f"t-{adapter_name}-denied",
                              adapter=adapter_name,
                              payload={**base_payload, "executable": fake})
            verification = _run_adapter_verification(adapter_cls, task, "worker-denied")
            check(f"{adapter_name}: denied account classified model_unavailable",
                  verification.payload.get("failure") == "model_unavailable",
                  f"failure={verification.payload.get('failure')!r} result={verification.payload.get('result')!r}")
            check(f"{adapter_name}: denied run did not report passed",
                  verification.payload.get("result") != "passed",
                  str(verification.payload.get("result")))

            fake = _write_fake(tmp, f"fake_{adapter_name}_granted", with_fable)
            task = _make_task("job-fable", f"t-{adapter_name}-granted",
                              adapter=adapter_name,
                              payload={**base_payload, "executable": fake})
            verification = _run_adapter_verification(adapter_cls, task, "worker-granted")
            check(f"{adapter_name}: granted account passes on {model}",
                  verification.payload.get("result") == "passed"
                  and verification.payload.get("failure") is None,
                  f"failure={verification.payload.get('failure')!r} result={verification.payload.get('result')!r}")


# ---------------------------------------------------------------------------
# 5. Orchestrator auto-fallback matrix (in-process, temp store)
# ---------------------------------------------------------------------------


def stress_auto_fallback_matrix() -> None:
    section("5. Auto-fallback: model_unavailable re-routes across platforms")
    from puppetmaster.model_registry import ModelSpec
    from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
    from puppetmaster.orchestrator import Orchestrator
    from puppetmaster.platform_billing import BillingStatus
    from puppetmaster.store import SwarmStore

    registry = [
        ModelSpec(id="claude-code/fable-5", adapter="claude-code",
                  adapter_model_name="claude-fable-5", capability_score=100,
                  input_per_mtok_usd=10.0, output_per_mtok_usd=50.0, billing="unknown"),
        ModelSpec(id="claude-code/opus-4-8", adapter="claude-code",
                  adapter_model_name="claude-opus-4-8", capability_score=99,
                  input_per_mtok_usd=5.0, output_per_mtok_usd=25.0, billing="unknown"),
        ModelSpec(id="cursor/claude-fable-5", adapter="cursor", adapter_model_name="claude-fable-5",
                  capability_score=100, billing="plan", tags=["cursor"]),
        ModelSpec(id="codex/gpt-5-5", adapter="codex", adapter_model_name="gpt-5.5",
                  capability_score=97, input_per_mtok_usd=5.0,
                  output_per_mtok_usd=30.0, billing="unknown"),
        ModelSpec(id="openai/gpt-5-5", adapter="openai", adapter_model_name="gpt-5.5",
                  capability_score=96, input_per_mtok_usd=5.0,
                  output_per_mtok_usd=30.0, billing="api"),
    ]

    # (failing adapter, healthy alternates, expected landing adapter)
    matrix = [
        ("claude-code", {"cursor": "plan"}, "cursor"),
        ("cursor", {"claude-code": "plan"}, "claude-code"),
        ("codex", {"openai": "api"}, "openai"),
    ]

    for failing_adapter, healthy, expected_adapter in matrix:
        # Each matrix case simulates a different machine; drop the TTL'd
        # billing probes so one case's posture never bleeds into the next.
        from puppetmaster.platform_billing import clear_billing_cache

        clear_billing_cache()

        def _billing(adapter, **kw):
            if adapter in healthy:
                return BillingStatus(adapter=adapter, billing=healthy[adapter],
                                     healthy=True, detail="ok", evidence=[])
            return BillingStatus(adapter=adapter, billing="unknown",
                                 healthy=False, detail="no", evidence=[])

        with tempfile.TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("frontier task")
            task = Task(
                job_id=job.id, role="audit",
                instruction="security audit across every module",
                adapter=failing_adapter, status=TaskStatus.FAILED,
                payload={"auto_route": True},
            )
            store.save_task(task)
            store.save_artifact(Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.VERIFICATION,
                created_by="w",
                payload={"check": "x", "result": "blocked",
                         "failure": "model_unavailable", "adapter": failing_adapter},
                confidence=0.5, evidence=[f"adapter:{failing_adapter}"],
            ))
            orch = Orchestrator(store)
            with patch("puppetmaster.model_registry.load_registry", return_value=registry), \
                 patch("puppetmaster.platform_billing.detect_adapter_billing", side_effect=_billing):
                rerouted = orch._reroute_recoverable_failures(job)

            updated = store.get_task_by_id(task.id)
            check(f"{failing_adapter} fable fails -> re-queued on {expected_adapter}",
                  rerouted == 1 and updated.status == TaskStatus.QUEUED
                  and updated.adapter == expected_adapter,
                  f"rerouted={rerouted} status={updated.status} adapter={updated.adapter}")
            check(f"{failing_adapter}: fallback provenance recorded on task",
                  updated.payload.get("fallback_from_adapter") == failing_adapter
                  and updated.payload.get("fallback_attempts") == 1,
                  str({k: v for k, v in updated.payload.items() if "fallback" in k}))
            routing = [a for a in store.list_artifacts(job.id)
                       if a.payload.get("fallback_reason") == "model_unavailable"]
            check(f"{failing_adapter}: ROUTING artifact carries fallback_reason",
                  len(routing) == 1 and routing[0].payload.get("model_id"),
                  f"found {len(routing)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("Claude Fable 5 rollout — cross-platform mock stress test")

    tmp_state = tempfile.mkdtemp(prefix="pm-fable5-state-")
    os.environ["PUPPETMASTER_STATE_DIR"] = tmp_state

    snapshot = lift_cursor_lock()
    try:
        stress_failure_classification()
        stress_routing_matrix()
        stress_adapters_end_to_end()
        stress_auto_fallback_matrix()
    finally:
        restore_cursor_lock(snapshot)

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL GREEN — Fable 5 covered on every platform, Cursor lock restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
