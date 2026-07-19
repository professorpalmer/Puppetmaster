from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO

from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.installers import (
    CLAUDE_NEXT_STEPS_GUIDANCE,
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    HERMES_NEXT_STEPS_GUIDANCE,
    InstallResult,
    UninstallResult,
    ensure_cursor_sdk,
    install_claude_mcp,
    install_codex_mcp,
    install_cursor_mcp,
    install_hermes_mcp,
    install_hermes_plugin,
    install_hermes_skill,
    list_skill_candidates,
    promote_skill_candidate,
    resolve_claude_command,
    set_hermes_mcp_env,
    uninstall_claude_mcp,
    uninstall_codex_mcp,
    uninstall_cursor_mcp,
    uninstall_hermes_mcp,
)
from puppetmaster.rules import (
    VALID_TARGETS,
    RulesInstallResult,
    install_rules,
    uninstall_rules,
)
from puppetmaster.hook_installers import (
    VALID_HOOK_TARGETS,
    install_hermes_hooks,
    install_hooks,
    uninstall_hermes_hooks,
    uninstall_hooks,
)
from puppetmaster.mcp_registry import (
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    summarize as registry_summarize,
)
from puppetmaster.redaction import redact_secrets
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.state import (
    find_state_dir_for_job,
    list_project_state_dirs,
    resolve_state_dir,
)
from puppetmaster.store_factory import create_store
from puppetmaster.stitcher import Stitcher
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec


def print_run_result(job_id: str, artifact_count: int, summary_path: Path) -> None:
    print(f"job_id: {job_id}")
    print(f"artifacts: {artifact_count}")
    print(f"summary: {summary_path}")

def _warn_job_liveness(store: Any, job_id: str) -> None:
    """For a still-"running" job, print a loud liveness verdict to stderr so a
    wedged/dead orchestrator is obvious at a glance instead of hiding behind a
    quiet ``running`` status (#9)."""
    from puppetmaster.liveness import liveness_summary

    try:
        job = store.get_job(job_id)
        if job is None or str(job.status) not in {"running", "stitching"}:
            return
        summary = liveness_summary(store, job)
    except Exception:
        return
    if summary["verdict"] == "alive":
        return
    pid = summary.get("pid")
    sys.stderr.write(
        f"liveness: {summary['verdict']} — orchestrator pid={pid} "
        f"idle={summary['idle_seconds']}s, live_lease={summary['live_lease']}. "
        "Run `puppetmaster reap` to stall+requeue, or `recover` to retry tasks.\n"
    )

def _warn_run_quality(store: Any, job_id: str) -> None:
    """Print a one-line quality verdict to stderr when a job's artifacts look
    blocked/empty/degraded, so a reader of ``show`` is never silently handed an
    untrustworthy summary.

    A still-running job is exempt from the empty/degraded warning: implement
    workers stream no incremental artifacts, so a perfectly healthy in-flight
    job legitimately has no substantive artifacts yet. Calling that
    "low-confidence; verify before trusting" cries wolf. We instead emit a
    neutral in-progress note. A ``blocked`` verdict (a worker refused to run)
    is a real failure even mid-flight, so it still warns.
    """
    from puppetmaster.quality import assess_run_quality
    from puppetmaster.models import JobStatus

    try:
        verdict = assess_run_quality(store.list_artifacts(job_id))
    except Exception:
        return
    quality = verdict["quality"]
    if quality == "ok":
        return

    in_progress = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.STITCHING}
    try:
        status = store.get_job(job_id).status
    except Exception:
        status = None
    if quality in {"empty", "degraded"} and status in in_progress:
        sys.stderr.write(
            f"quality: in progress (state={status}) — no substantive artifacts yet. "
            "This is expected for a running implement job (artifacts land at the end); "
            "not a failure signal.\n"
        )
        return

    reason = "; ".join(verdict.get("reasons") or [])
    sys.stderr.write(
        f"quality: {quality} — {reason}. "
        "This run is low-confidence; verify before trusting it.\n"
    )

def finalize_cli_run(result: Any) -> int:
    """Print a run's mode banner, summary, and a built-in quality verdict, then
    return a shell exit code.

    A ``blocked`` run (a worker refused to run — e.g. dirty tree) or an
    ``empty`` run exits non-zero and prints a loud reason, so a "completed" job
    that did zero work can never masquerade as success. ``degraded`` is loud but
    non-fatal (exit 0) — the artifacts exist but shouldn't be trusted blindly.
    """
    from puppetmaster.quality import assess_run_quality

    print_mode_banner(result.mode, getattr(result, "acting", False))
    print_run_result(result.job.id, len(result.artifacts), result.summary_path)

    if result.mode == "edit":
        from puppetmaster.models import ArtifactType

        baseline_diff_present = any(
            bool((a.payload or {}).get("baseline_diff_present")) for a in result.artifacts
        )
        worker_diff_present = any(
            bool((a.payload or {}).get("worker_diff_present")) for a in result.artifacts
        )
        patch_artifact_emitted = any(
            a.type == ArtifactType.PATCH for a in result.artifacts
        )
        commit_present = any(
            a.type == ArtifactType.GATE
            and (a.payload or {}).get("kind") == "committed"
            and (a.payload or {}).get("passed") is True
            for a in result.artifacts
        )
        print(
            f"outcome: baseline_diff_present={baseline_diff_present} "
            f"worker_diff_present={worker_diff_present} "
            f"patch_artifact_emitted={patch_artifact_emitted} "
            f"commit_present={commit_present} "
            f"artifacts={len(result.artifacts)}",
            file=sys.stderr,
        )
        report_headline = next(
            (
                str(
                    (a.payload or {}).get("claim")
                    or (a.payload or {}).get("decision")
                    or ""
                ).strip()
                for a in result.artifacts
                if a.type in {ArtifactType.FINDING, ArtifactType.DECISION}
                and ((a.payload or {}).get("claim") or (a.payload or {}).get("decision"))
            ),
            None,
        )
        if report_headline:
            print(f"report: {report_headline}", file=sys.stderr)
            print(
                f"  full report: puppetmaster artifacts {result.job.id}",
                file=sys.stderr,
            )

    verdict = assess_run_quality(result.artifacts)
    quality = verdict["quality"]
    if quality == "ok":
        return 0

    reason = "; ".join(verdict.get("reasons") or [])
    if quality in {"blocked", "empty"}:
        print(
            f"puppetmaster: run {quality} — {reason}. "
            "Nothing was accomplished; not reporting success.",
            file=sys.stderr,
        )
        return 1
    print(
        f"puppetmaster: run quality=degraded — {reason}. "
        "Artifacts exist but treat this run as low-confidence.",
        file=sys.stderr,
    )
    return 0

def print_mode_banner(mode: str, acting: bool = False) -> None:
    """Print a one-line read-only / edit banner to stderr so the user is never
    surprised that an 'analysis' swarm wrote no files.

    ``acting`` flags a swarm that edits no files but acts on the world beyond the
    repo (e.g. drives a live browser — logins, form fills). Such a run is
    read-only on the repo yet has external side effects, so it must never read as
    a harmless no-op."""
    if mode == "edit":
        print(
            "puppetmaster: mode=edit — workers may modify files in the working tree.",
            file=sys.stderr,
        )
    else:
        print(
            "puppetmaster: mode=analysis (read-only) — no files will be edited; "
            "this run only emits artifacts.",
            file=sys.stderr,
        )
    if acting:
        print(
            "puppetmaster: ACTING AGENT — a worker drives a real browser against "
            "a live system (external side effects: navigation, logins, form "
            "fills). Treat with implement-style approval; not a no-op read-only run.",
            file=sys.stderr,
        )


def allowed_models_cli_list(args) -> list[str]:
    """Non-empty model identities from repeated ``--allowed-models`` CLI flags."""
    return [
        str(item).strip()
        for item in (getattr(args, "allowed_models", None) or [])
        if str(item).strip()
    ]


def append_allowed_models_cli_flags(
    command: list[str], allowed_models: Optional[list[str]]
) -> None:
    """Append repeatable ``--allowed-models`` flags to a CLI argv list."""
    if not allowed_models:
        return
    for item in allowed_models:
        text = str(item).strip()
        if text:
            command.extend(["--allowed-models", text])


def allowed_model_ids_from_mapping(mapping) -> Optional[frozenset[str]]:
    """Parse ``allowed_model_ids`` / ``allowed_models`` from a JSON-like mapping.

    Returns ``None`` when both keys are absent (unrestricted routing). Returns
    ``frozenset()`` when the key is present but empty (fail closed). Returns a
    non-empty frozenset otherwise.
    """
    if "allowed_model_ids" in mapping:
        raw = mapping["allowed_model_ids"]
    elif "allowed_models" in mapping:
        raw = mapping["allowed_models"]
    else:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return frozenset([text]) if text else frozenset()
    if isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return frozenset(values)
    return frozenset()


def allowed_model_ids_list_from_mapping(mapping) -> Optional[list[str]]:
    """List form of :func:`allowed_model_ids_from_mapping` for payloads/CLI."""
    parsed = allowed_model_ids_from_mapping(mapping)
    if parsed is None:
        return None
    return sorted(parsed)


def routing_payload_from_args(args, *, adapter: str) -> dict:
    """Translate the shared ``--auto-route`` routing flags into payload keys the
    orchestrator's router understands. Empty unless ``--auto-route`` is set, so
    a direct adapter run is unchanged by default.

    Pins ``allowed_adapters`` to the invoked adapter so routing only picks a
    *model* within that platform — a direct ``cursor`` run never silently hops
    to claude-code."""
    if not getattr(args, "auto_route", False):
        return {}
    payload: dict[str, Any] = {"auto_route": True, "allowed_adapters": [adapter]}
    if getattr(args, "routing_policy", None):
        payload["routing_policy"] = args.routing_policy
    if getattr(args, "max_cost_usd", None) is not None:
        payload["max_cost_usd"] = args.max_cost_usd
    if getattr(args, "min_capability", None) is not None:
        payload["min_capability"] = args.min_capability
    allowed_models = allowed_models_cli_list(args)
    if allowed_models:
        payload["allowed_model_ids"] = allowed_models
    return payload

def artifact_feed(store, job_id: str, limit: Optional[int] = None) -> list[dict]:
    items, _ = artifact_feed_since(store, job_id, since=0, limit=limit)
    return items

def artifact_feed_since(
    store,
    job_id: str,
    since: int = 0,
    limit: Optional[int] = None,
) -> tuple[list[dict], int]:
    """Return (items, next_cursor) of new ``artifact.saved`` events.

    ``since`` is the event cursor returned by a previous call (use 0 for a
    fresh read). ``next_cursor`` is the highest event id observed, regardless
    of whether it was an artifact event, so callers can resume reliably.
    """
    # Read events FIRST, then fetch only the artifacts those events reference.
    # The previous order (snapshot all artifacts, then read events) had a race:
    # an artifact saved between the two reads was missing from the snapshot, so
    # its event was skipped while the cursor still advanced past it — dropping
    # the artifact from the feed forever. save_artifact persists the row before
    # emitting its event, so an artifact named by an event we just read is
    # guaranteed to exist when we fetch it afterward.
    events = store.read_events_since(job_id, since=since)
    artifact_events: list[dict] = []
    cursor = since
    for event in events:
        event_id = event.get("id")
        if isinstance(event_id, int) and event_id > cursor:
            cursor = event_id
        if event.get("event") == "artifact.saved":
            artifact_events.append(event)

    needed_ids = [
        event.get("payload", {}).get("artifact_id") for event in artifact_events
    ]
    fetched = store.get_artifacts_by_ids(job_id, needed_ids)

    items: list[dict] = []
    seen: set = set()
    for event in artifact_events:
        artifact_id = event.get("payload", {}).get("artifact_id")
        artifact = fetched.get(artifact_id)
        if artifact is None or artifact_id in seen:
            continue
        seen.add(artifact_id)
        items.append(
            {
                "at": event["at"],
                "event": event["event"],
                "id": event.get("id"),
                "artifact": artifact.__dict__,
            }
        )
    if limit is not None:
        items = items[-limit:]
    return items, cursor

def print_feed_item(item: dict) -> None:
    artifact = item["artifact"]
    print(
        f"{item['at']}\t{artifact['type']}\t{artifact['id']}\t"
        f"task={artifact['task_id']}\tconfidence={artifact['confidence']}"
    )
    print(f"  {artifact_headline(artifact)}")

def run_feed_follow(
    store,
    job_id: str,
    *,
    since: int = 0,
    limit: Optional[int] = None,
    as_json: bool = False,
    idle_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> int:
    cursor = since
    initial_items, cursor = artifact_feed_since(store, job_id, since=cursor, limit=limit)
    for item in initial_items:
        emit_feed_item(item, as_json=as_json)

    poll_budget = max(0.05, poll_interval_seconds)
    block_seconds = max(poll_budget, 1.0)
    idle_deadline = (
        time.monotonic() + idle_timeout_seconds if idle_timeout_seconds > 0 else None
    )
    try:
        while True:
            events = store.wait_for_events(
                job_id,
                since=cursor,
                timeout_seconds=block_seconds,
                poll_interval=poll_budget,
            )
            if events:
                new_items, cursor = artifact_feed_since(
                    store, job_id, since=cursor, limit=None
                )
                for item in new_items:
                    emit_feed_item(item, as_json=as_json)
                idle_deadline = (
                    time.monotonic() + idle_timeout_seconds
                    if idle_timeout_seconds > 0
                    else None
                )
                continue
            if idle_deadline is not None and time.monotonic() >= idle_deadline:
                return 0
    except KeyboardInterrupt:
        return 0

def emit_feed_item(item: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(item, default=str), flush=True)
    else:
        print_feed_item(item)
        sys.stdout.flush()

def run_deltas_follow(
    store,
    job_id: str,
    *,
    task_id: Optional[str] = None,
    as_json: bool = False,
    follow: bool = False,
    idle_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> int:
    """Stream an agentic job's durable token deltas (all tasks, or one).

    Tails the per-task ``agentic_deltas.ndjson`` files the agentic adapter
    writes under the job state dir, multiplexed across tasks. This is the
    subprocess/CLI face of token streaming -- the same tokens an inline host
    sees over the in-process delta bus, made followable for detached workers.
    """
    from puppetmaster.adapters._delta_stream import _DELTA_FILE

    base = Path(store.root) / "jobs" / job_id / "tasks"
    offsets: dict[Path, int] = {}

    def _drain() -> bool:
        produced = False
        if not base.exists():
            return produced
        for task_dir in sorted(base.iterdir()):
            if task_id is not None and task_dir.name != task_id:
                continue
            delta_path = task_dir / _DELTA_FILE
            if not delta_path.exists():
                continue
            offset = offsets.get(delta_path, 0)
            try:
                with open(delta_path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    for line in handle:
                        if not line.endswith("\n"):
                            break
                        offset += len(line.encode("utf-8"))
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            record = json.loads(stripped)
                        except ValueError:
                            continue
                        produced = True
                        _emit_delta(record, task_dir.name, as_json=as_json)
                offsets[delta_path] = offset
            except OSError:
                continue
        return produced

    _drain()
    if not follow:
        return 0
    poll = max(0.02, poll_interval_seconds)
    idle_deadline = (
        time.monotonic() + idle_timeout_seconds if idle_timeout_seconds > 0 else None
    )
    try:
        while True:
            produced = _drain()
            if produced and idle_timeout_seconds > 0:
                idle_deadline = time.monotonic() + idle_timeout_seconds
            if idle_deadline is not None and time.monotonic() >= idle_deadline:
                return 0
            time.sleep(poll)
    except KeyboardInterrupt:
        return 0

def _emit_delta(record: dict, task_name: str, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(record, default=str), flush=True)
        return
    text = str(record.get("text", ""))
    sys.stdout.write(text)
    sys.stdout.flush()

def artifact_headline(artifact: dict) -> str:
    payload = artifact.get("payload", {})
    if not isinstance(payload, dict):
        return str(payload)
    for key in ["claim", "decision", "risk", "check", "change"]:
        if key in payload:
            return str(payload[key])
    return json.dumps(payload, sort_keys=True)

def early_job_printer(job) -> None:
    print(f"job_id: {job.id}", flush=True)

def _print_install_result(result: InstallResult, host: str) -> int:
    """Pretty-print an :class:`InstallResult` and return the appropriate exit code.

    Exit codes are deliberately compatible with shell-script automation:
    ``0`` for installed / unchanged / would_install (a successful no-op),
    ``1`` for any error so a CI step can fail fast on a broken install.
    """
    print(f"[install-{host}-mcp] status: {result.status}")
    print(f"[install-{host}-mcp] target: {redact_secrets(result.target)}")
    print(f"[install-{host}-mcp] python: {redact_secrets(result.python_executable)}")
    if result.handshake is not None:
        if result.handshake.ok:
            print(
                f"[install-{host}-mcp] handshake: OK ({result.handshake.tool_count} tools)"
            )
        else:
            print(f"[install-{host}-mcp] handshake: FAILED — {redact_secrets(result.handshake.error)}")
    for line in result.messages:
        print(f"[install-{host}-mcp] {redact_secrets(line)}")
    return 0 if result.status in {"installed", "unchanged", "would_install"} else 1

def _print_uninstall_mcp_result(result: UninstallResult, host: str) -> int:
    """Pretty-print an :class:`UninstallResult` and return the appropriate exit code."""
    print(f"[uninstall-{host}-mcp] status: {result.status}")
    print(f"[uninstall-{host}-mcp] target: {result.target}")
    for line in result.messages:
        print(f"[uninstall-{host}-mcp] {line}")
    return 0 if result.status in {"removed", "unchanged", "would_remove"} else 1

def _print_uninstall_rules_result(result: RulesInstallResult) -> int:
    print(f"[uninstall-rules] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[uninstall-rules] {outcome.target:<16} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[uninstall-rules] {' ' * 16} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[uninstall-rules] note: {msg}")
    return 1 if result.overall_status == "error" else 0

def _print_uninstall_hooks_result(result) -> int:
    print(f"[uninstall-hooks] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[uninstall-hooks] {outcome.target:<8} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[uninstall-hooks] {' ' * 8} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[uninstall-hooks] note: {msg}")
    return 1 if result.overall_status == "error" else 0

def _print_rules_result(result: RulesInstallResult) -> int:
    """Pretty-print a :class:`RulesInstallResult` and return an exit code.

    Exit codes mirror the MCP installers: 0 on installed / unchanged /
    would_install / skipped, 1 on any target error.
    """
    print(f"[install-rules] overall: {result.overall_status}")
    for outcome in result.outcomes:
        print(
            f"[install-rules] {outcome.target:<14} {outcome.status:<14} {outcome.reason}"
        )
        if outcome.path:
            print(f"[install-rules] {' ' * 14} {' ' * 14} -> {outcome.path}")
    for msg in result.messages:
        print(f"[install-rules] note: {msg}")
    return 1 if result.overall_status == "error" else 0

def _print_codegraph_freshness(target: str) -> int:
    """CLI entrypoint for `python -m puppetmaster codegraph freshness`.

    Reports whether the repo's CodeGraph index still reflects the working tree.
    Exit code mirrors the verdict so it's scriptable: 0 fresh/unknown/absent,
    1 stale (so CI / pre-commit hooks can gate on a drifted index)."""
    from puppetmaster.codegraph import codegraph_freshness

    freshness = codegraph_freshness(target)
    label = {
        "fresh": "fresh — index matches the working tree",
        "stale": "STALE",
        "unknown": "unknown — workspace too large to scan within budget",
        "no_index": "not indexed yet — run `codegraph init --index`",
        "uninitialized": "no .codegraph/ here — run `codegraph init --index`",
    }.get(freshness.state, freshness.state)
    print(f"codegraph index: {label}")
    if freshness.is_stale:
        print(f"  {freshness.warning_text()}")
        return 1
    if freshness.reason and freshness.state not in {"fresh"}:
        print(f"  {freshness.reason}")
    return 0

def _registry_path_from_args(args) -> Optional[Path]:
    raw = getattr(args, "registry_path", None)
    return Path(raw).expanduser() if raw else None

def _print_token_usage(artifacts) -> None:
    """Print measured-vs-estimated token consumption for a job's runs.

    Plan-billed runtimes (Cursor SDK) have $0 marginal cost, so a dollars-only
    ledger says nothing. Token counts are the honest measure of consumption;
    surface them, clearly split into measured vs char/4-estimated.
    """
    from puppetmaster.usage import aggregate_token_usage

    usage = aggregate_token_usage(artifacts)
    if usage["measured_runs"] == 0 and usage["estimated_runs"] == 0:
        return
    print()
    print("  token consumption (measured where the SDK reports usage):")
    if usage["measured_runs"]:
        print(
            f"    measured:  {usage['measured_tokens_in']:,} in / "
            f"{usage['measured_tokens_out']:,} out over {usage['measured_runs']} run(s)"
        )
    if usage["estimated_runs"]:
        print(
            f"    estimated: ~{usage['estimated_tokens_in']:,} in / "
            f"~{usage['estimated_tokens_out']:,} out over {usage['estimated_runs']} run(s) "
            "(char/4 approximation — SDK reported no usage)"
        )

def print_watch_snapshot(snapshot: dict) -> None:
    counts = ", ".join(
        f"{status}={count}" for status, count in sorted(snapshot["task_counts"].items())
    )
    print(
        f"{snapshot['job']['id']} {snapshot['job']['status']} "
        f"tasks[{counts}] artifacts={snapshot['artifact_count']} "
        f"stale={len(snapshot['stale_task_ids'])}"
    )

def require_latest_job_id(store) -> str:
    job = store.latest_job()
    if job is None:
        raise FileNotFoundError("no jobs found")
    return job.id

def artifact_job_id(store, artifact_id: str) -> str:
    job_id = store.get_artifact_job_id(artifact_id)
    if job_id is None:
        raise FileNotFoundError(f"artifact not found: {artifact_id}")
    return job_id

def patch_artifacts_for_target(store, target: str):
    try:
        store.get_job(target)
        artifacts = store.list_artifacts(target)
        return target, [artifact for artifact in artifacts if str(artifact.type) == "patch"]
    except FileNotFoundError:
        pass
    job_id = artifact_job_id(store, target)
    artifacts = [artifact for artifact in store.list_artifacts(job_id) if artifact.id == target]
    return job_id, artifacts

def approve_target(store, target: str, worktree: Optional[Path] = None) -> int:
    job_id, artifacts = patch_artifacts_for_target(store, target)
    if not artifacts:
        raise FileNotFoundError(f"no patch artifacts found for {target}")
    applied = 0
    for artifact in artifacts:
        files = artifact.payload.get("files", [])
        locks = [f"patch:{path}" for path in files if isinstance(path, str)]
        for lock in locks:
            if not store.acquire_lock(lock, f"approve:{artifact.id}"):
                raise RuntimeError(f"path lock unavailable: {lock}")
        try:
            diff = artifact.payload.get("unified_diff") or artifact.payload.get("diff")
            if diff:
                apply_patch_diff(diff, cwd=worktree or Path.cwd())
                applied += 1
            store.emit(
                job_id,
                "artifact.approved",
                {
                    "artifact_id": artifact.id,
                    "applied": bool(diff),
                    "worktree": str(worktree) if worktree else None,
                },
            )
        finally:
            for lock in locks:
                store.release_lock(lock)
    return len(artifacts)

def reject_target(store, target: str, reason: str) -> int:
    job_id, artifacts = patch_artifacts_for_target(store, target)
    if not artifacts:
        raise FileNotFoundError(f"no patch artifacts found for {target}")
    for artifact in artifacts:
        store.emit(
            job_id,
            "artifact.rejected",
            {"artifact_id": artifact.id, "reason": reason},
        )
    return len(artifacts)

def apply_patch_diff(diff: str, cwd: Path) -> None:
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=diff,
        text=True,
        capture_output=True,
        cwd=cwd,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"patch did not apply cleanly: {check.stderr.strip()}")
    applied = subprocess.run(
        ["git", "apply", "-"],
        input=diff,
        text=True,
        capture_output=True,
        cwd=cwd,
        check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"patch apply failed: {applied.stderr.strip()}")

def cursor_prompt(
    prompt: str,
    *,
    review: bool = False,
    plan: bool = False,
    dry_run: bool = False,
    implement: bool = False,
) -> str:
    lines = [prompt]
    if implement:
        lines.extend(
            [
                "",
                "Implement mode: you are a full-edit worker inside the user's repository. "
                "Actually make the code changes to complete the task end to end — create, "
                "edit, and delete files as needed. Do not just return a plan or findings; "
                "leave the working tree containing your final intended changes.",
            ]
        )
    if review:
        lines.extend(
            [
                "",
                "Review mode: inspect the repository and return findings, risks, evidence, and verification suggestions.",
            ]
        )
    if plan:
        lines.extend(
            [
                "",
                "Plan mode: produce implementation decisions, task graph suggestions, risks, and test strategy. Do not edit files.",
            ]
        )
    if dry_run:
        lines.extend(
            [
                "",
                "Dry-run constraint: do not modify files. Return findings, patch plan, risks, and verification commands as structured evidence.",
            ]
        )
    return "\n".join(lines)
