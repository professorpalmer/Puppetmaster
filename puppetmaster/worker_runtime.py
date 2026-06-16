from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from puppetmaster.models import AgentRun, JobStatus, TaskStatus, now_iso
from puppetmaster.state import resolve_state_dir
from puppetmaster.store_factory import create_store
from puppetmaster.workers import LocalWorker

if TYPE_CHECKING:
    from puppetmaster.store import SwarmStore


def worker_id_for(role: Optional[str]) -> str:
    return f"worker-{role or 'any'}-{os.getpid()}"


class WorkerRuntime:
    def __init__(
        self,
        store: SwarmStore,
        job_id: str,
        role: Optional[str],
        worker_id: str,
        lease_seconds: int = 5,
        poll_seconds: float = 0.1,
        heartbeat_seconds: Optional[float] = None,
        simulate_seconds: float = 0.0,
        crash_after_claim: bool = False,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.role = role
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.simulate_seconds = simulate_seconds
        self.crash_after_claim = crash_after_claim
        self._lease_lost = threading.Event()

    def _heartbeat_interval(self) -> float:
        configured = self.heartbeat_seconds
        if configured is None:
            configured = 2.0 if self.poll_seconds == 0.1 else self.poll_seconds
        return max(0.01, min(configured, max(0.1, self.lease_seconds / 3)))

    def run_once(self) -> bool:
        task = self.store.claim_next_task(
            self.job_id,
            self.worker_id,
            role=self.role,
            lease_seconds=self.lease_seconds,
        )
        if task is None:
            return False

        if self.crash_after_claim:
            self.store.emit(
                self.job_id,
                "worker.crashed_after_claim",
                {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
            )
            raise SystemExit(77)

        run = AgentRun(
            job_id=self.job_id,
            task_id=task.id,
            role=task.role,
            worker_id=self.worker_id,
        )
        self.store.save_run(run)

        deadline = time.monotonic() + self.simulate_seconds
        while time.monotonic() < deadline:
            time.sleep(
                min(self._heartbeat_interval(), max(0.0, deadline - time.monotonic()))
            )
            run = self.store.heartbeat_run(run)
            self.store.renew_task_lease(task.id, self.worker_id, self.lease_seconds)

        stop_heartbeats = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_until_stopped,
            args=(run, task.id, stop_heartbeats),
            daemon=True,
        )
        heartbeat.start()
        try:
            worker_run, artifacts = LocalWorker(task.role, worker_id=self.worker_id).run(
                task,
                self.store.get_job(self.job_id).goal,
            )
            if self._lease_lost.is_set():
                self.store.emit(
                    self.job_id,
                    "worker.lease_lost",
                    {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
                )
                return True
            for artifact in artifacts:
                self.store.save_artifact(artifact)
        except Exception as exc:
            failed_run = replace(
                run,
                status=TaskStatus.FAILED,
                heartbeat_at=now_iso(),
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            self.store.update_task_status(task, TaskStatus.FAILED, worker_id=self.worker_id)
            self.store.emit(
                self.job_id,
                "worker.failed_task",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": self.role,
                    "error": str(exc),
                },
            )
            return True
        finally:
            stop_heartbeats.set()
            heartbeat.join(timeout=1)

        # Honor a FAILED verdict from the worker (e.g. a preflight block), and
        # also convert an adapter-detected auth/billing/quota rejection that
        # came back as a verification artifact into a truthful FAILED status.
        # Without this the task would be recorded COMPLETE over a run that
        # never produced real output — telemetry would lie, await would report
        # success, and the orchestrator's auto-fallback could never re-route.
        recoverable = self._recoverable_failure(artifacts)
        blocked = self._blocked_verdict(artifacts)
        if worker_run.status == TaskStatus.FAILED or recoverable is not None or blocked is not None:
            failed_run = replace(
                run,
                status=TaskStatus.FAILED,
                heartbeat_at=now_iso(),
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            updated = self.store.update_task_status(
                task, TaskStatus.FAILED, worker_id=self.worker_id
            )
            self.store.emit(
                self.job_id,
                "worker.failed_task",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": self.role,
                    "failure": recoverable or blocked,
                    "blocked": blocked,
                },
            )
            self._emit_live_task_span(updated, artifacts)
            return True

        # Non-bypassable completion gates: an agent may not reach COMPLETE just
        # because it thinks it finished. Post-conditions (drift ratchet, required
        # diff, commit) are evaluated by the runtime; a failed gate is FAILED.
        gate_eval = self._evaluate_gates(task, artifacts)
        for gate_artifact in gate_eval.artifacts:
            self.store.save_artifact(gate_artifact)
        if not gate_eval.passed:
            failed_run = replace(
                run,
                status=TaskStatus.FAILED,
                heartbeat_at=now_iso(),
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            updated = self.store.update_task_status(
                task, TaskStatus.FAILED, worker_id=self.worker_id
            )
            self.store.emit(
                self.job_id,
                "worker.gate_failed",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": self.role,
                    "reason": gate_eval.failed_reason,
                },
            )
            self._emit_live_task_span(updated, artifacts + gate_eval.artifacts)
            return True

        completed_run = replace(
            run,
            status=TaskStatus.COMPLETE,
            heartbeat_at=now_iso(),
            completed_at=now_iso(),
        )
        self.store.save_run(completed_run)
        updated = self.store.update_task_status(
            task, TaskStatus.COMPLETE, worker_id=self.worker_id
        )
        self.store.emit(
            self.job_id,
            "worker.completed_task",
            {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
        )
        self._emit_live_task_span(updated, artifacts + gate_eval.artifacts)
        return True

    def _evaluate_gates(self, task, artifacts: list):
        """Evaluate this task's completion gates. Ungated tasks pass through;
        gated tasks fail closed when the gate engine raises."""
        from puppetmaster.gates import (
            GateEvaluation,
            GateResult,
            evaluate_task_gates,
            task_gate_specs,
        )

        has_gates = bool(task_gate_specs(task))
        try:
            return evaluate_task_gates(
                task, artifacts, self.store, worker_id=self.worker_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.store.emit(
                self.job_id,
                "worker.gate_error",
                {"worker_id": self.worker_id, "task_id": task.id, "error": str(exc)},
            )
            if not has_gates:
                return GateEvaluation(passed=True, results=[], artifacts=[])
            return GateEvaluation(
                passed=False,
                results=[
                    GateResult(
                        name="gate_engine",
                        kind="internal",
                        passed=False,
                        reason="gate_engine_error",
                    )
                ],
                artifacts=[],
            )

    def _emit_live_task_span(self, task, artifacts: list) -> None:
        """Emit a live OTel span for this finished task. No-op unless live
        telemetry is enabled; never lets a telemetry failure break the run.

        The parent trace context is read from the ``TRACEPARENT`` env var the
        orchestrator exported to this subprocess, so the span correlates into
        the job's trace across the process boundary."""
        try:
            from puppetmaster.telemetry import live_telemetry_enabled, record_task_span

            if not live_telemetry_enabled():
                return
            traceparent = os.environ.get("PUPPETMASTER_TRACEPARENT") or os.environ.get(
                "TRACEPARENT"
            )
            record_task_span(
                self.store.get_job(self.job_id).goal,
                task,
                artifacts,
                traceparent=traceparent,
            )
        except Exception:
            pass

    @staticmethod
    def _recoverable_failure(artifacts: list) -> Optional[str]:
        """Return the first recoverable failure class found in ``artifacts``.

        Recoverable = an auth/billing/quota/missing-tool rejection (see
        :data:`puppetmaster.workers.RECOVERABLE_FAILURES`) that the
        orchestrator can re-route to a different funded adapter.
        """
        from puppetmaster.workers import RECOVERABLE_FAILURES

        for artifact in artifacts:
            failure = (getattr(artifact, "payload", None) or {}).get("failure")
            if failure in RECOVERABLE_FAILURES:
                return str(failure)
        return None

    @staticmethod
    def _blocked_verdict(artifacts: list) -> Optional[str]:
        """Return the failure reason when a worker *refused to run*.

        A verification artifact with ``result == "blocked"`` means the adapter
        declined to do the work — a dirty tree, a non-worktree target, a
        preflight gate. That is never a COMPLETE: a "completed" task that did
        zero work is the worst failure mode because it looks like success. Mark
        it FAILED loudly so the diff/commit is never silently empty.
        """
        for artifact in artifacts:
            payload = getattr(artifact, "payload", None) or {}
            if payload.get("result") == "blocked":
                return str(payload.get("failure") or "blocked")
        return None

    def _heartbeat_until_stopped(
        self,
        run: AgentRun,
        task_id: str,
        stop: threading.Event,
    ) -> None:
        while not stop.wait(self._heartbeat_interval()):
            run = self.store.heartbeat_run(run)
            renewed = self.store.renew_task_lease(
                task_id, self.worker_id, self.lease_seconds
            )
            if renewed is None:
                self._lease_lost.set()
                stop.set()
                return

    def run_until_idle(self) -> int:
        completed = 0
        while True:
            self.store.recover_stale_tasks(self.job_id)
            if self.run_once():
                completed += 1
                continue
            if not self._has_role_work():
                return completed
            time.sleep(self.poll_seconds)

    def _has_role_work(self) -> bool:
        for task in self.store.list_tasks(self.job_id):
            if self.role is not None and task.role != self.role:
                continue
            if task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                return True
        return False


class WorkerDaemon:
    """Warm worker loop that claims tasks from running jobs without process cold starts."""

    def __init__(
        self,
        store: SwarmStore,
        roles: Optional[list[str]] = None,
        worker_id: Optional[str] = None,
        job_id: Optional[str] = None,
        lease_seconds: int = 5,
        poll_seconds: float = 0.25,
    ) -> None:
        self.store = store
        self.roles = roles or [None]
        self.worker_id = worker_id or f"daemon-{os.getpid()}"
        self.job_id = job_id
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    def run(
        self,
        max_tasks: Optional[int] = None,
        max_idle_seconds: Optional[float] = None,
    ) -> int:
        completed = 0
        idle_since = time.monotonic()
        while True:
            if self.run_once():
                completed += 1
                idle_since = time.monotonic()
                if max_tasks is not None and completed >= max_tasks:
                    return completed
                continue
            if max_idle_seconds is not None and time.monotonic() - idle_since >= max_idle_seconds:
                return completed
            time.sleep(self.poll_seconds)

    def run_once(self) -> bool:
        for job in self._running_jobs():
            self.store.recover_stale_tasks(job.id)
            self.store.refresh_blocked_tasks(job.id)
            for role in self.roles:
                runtime = WorkerRuntime(
                    store=self.store,
                    job_id=job.id,
                    role=role,
                    worker_id=f"{self.worker_id}-{role or 'any'}",
                    lease_seconds=self.lease_seconds,
                    poll_seconds=self.poll_seconds,
                )
                if runtime.run_once():
                    return True
        return False

    def _running_jobs(self) -> list:
        jobs = [
            job
            for job in self.store.list_jobs()
            if job.status == JobStatus.RUNNING and (self.job_id is None or job.id == self.job_id)
        ]
        return sorted(jobs, key=lambda job: job.created_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Puppetmaster worker process.")
    parser.add_argument("--state-dir")
    parser.add_argument("--backend", choices=["file", "sqlite"], default="file")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--heartbeat-seconds", type=float)
    parser.add_argument("--simulate-seconds", type=float, default=0.0)
    parser.add_argument("--crash-after-claim", action="store_true")
    return parser


def _write_startup_error(state_dir, job_id: str, worker_id: str, exc: BaseException) -> None:
    """Record a worker that died before/while starting so the failure isn't a
    silent 0-byte log.

    A worker that fails to start (bad env, import error, store init failure)
    used to vanish with no trace — indistinguishable from "never launched". We
    drop a startup-error file next to the job's task logs and, if the store is
    usable, emit a ``worker.startup_failed`` event so it shows up in the feed.
    """
    import traceback

    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        from pathlib import Path

        crash_dir = Path(state_dir) / "jobs" / job_id / "tasks"
        crash_dir.mkdir(parents=True, exist_ok=True)
        (crash_dir / f"startup_error-{worker_id}.log").write_text(
            f"worker {worker_id} for job {job_id} failed to start:\n\n{detail}",
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass
    try:
        create_store("sqlite", state_dir).emit(
            job_id,
            "worker.startup_failed",
            {"worker_id": worker_id, "error": str(exc)},
        )
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = resolve_state_dir(args.state_dir)
    # Export the resolved state dir so adapter subprocesses (e.g. CursorAdapter,
    # ClaudeCodeAdapter) can spool full stdout/stderr to a sidecar log under
    # the same jobs/<job_id>/tasks/<task_id>/ tree the store already owns.
    # Without this the adapter would fall back to the workspace-hashed default,
    # which can land logs in a project state dir that doesn't own the job.
    os.environ["PUPPETMASTER_STATE_DIR"] = str(state_dir)
    worker_id = args.worker_id or worker_id_for(args.role)
    try:
        runtime = WorkerRuntime(
            store=create_store(args.backend, state_dir),
            job_id=args.job_id,
            role=args.role,
            worker_id=worker_id,
            lease_seconds=args.lease_seconds,
            poll_seconds=args.poll_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
            simulate_seconds=args.simulate_seconds,
            crash_after_claim=args.crash_after_claim,
        )
        return 0 if runtime.run_until_idle() >= 0 else 1
    except SystemExit:
        # An intentional exit (e.g. the crash-after-claim demo) is not a
        # startup failure — let it propagate untouched.
        raise
    except BaseException as exc:  # noqa: BLE001 — last-resort trace before dying
        _write_startup_error(state_dir, args.job_id, worker_id, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
