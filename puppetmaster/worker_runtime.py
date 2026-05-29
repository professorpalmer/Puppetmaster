from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import replace
from typing import Optional

from puppetmaster.models import AgentRun, JobStatus, TaskStatus, now_iso
from puppetmaster.state import resolve_state_dir
from puppetmaster.store_factory import create_store
from puppetmaster.workers import LocalWorker


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
        simulate_seconds: float = 0.0,
        crash_after_claim: bool = False,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.role = role
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.simulate_seconds = simulate_seconds
        self.crash_after_claim = crash_after_claim

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
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))
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
            self.store.update_task_status(task, TaskStatus.FAILED)
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
        if worker_run.status == TaskStatus.FAILED or recoverable is not None:
            failed_run = replace(
                run,
                status=TaskStatus.FAILED,
                heartbeat_at=now_iso(),
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            updated = self.store.update_task_status(task, TaskStatus.FAILED)
            self.store.emit(
                self.job_id,
                "worker.failed_task",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": self.role,
                    "failure": recoverable,
                },
            )
            self._emit_live_task_span(updated, artifacts)
            return True

        completed_run = replace(
            run,
            status=TaskStatus.COMPLETE,
            heartbeat_at=now_iso(),
            completed_at=now_iso(),
        )
        self.store.save_run(completed_run)
        updated = self.store.update_task_status(task, TaskStatus.COMPLETE)
        self.store.emit(
            self.job_id,
            "worker.completed_task",
            {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
        )
        self._emit_live_task_span(updated, artifacts)
        return True

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

    def _heartbeat_until_stopped(
        self,
        run: AgentRun,
        task_id: str,
        stop: threading.Event,
    ) -> None:
        while not stop.wait(self.poll_seconds):
            run = self.store.heartbeat_run(run)
            self.store.renew_task_lease(task_id, self.worker_id, self.lease_seconds)

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
    parser.add_argument("--simulate-seconds", type=float, default=0.0)
    parser.add_argument("--crash-after-claim", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = resolve_state_dir(args.state_dir)
    # Export the resolved state dir so adapter subprocesses (e.g. CursorAdapter,
    # ClaudeCodeAdapter) can spool full stdout/stderr to a sidecar log under
    # the same jobs/<job_id>/tasks/<task_id>/ tree the store already owns.
    # Without this the adapter would fall back to the workspace-hashed default,
    # which can land logs in a project state dir that doesn't own the job.
    os.environ["PUPPETMASTER_STATE_DIR"] = str(state_dir)
    runtime = WorkerRuntime(
        store=create_store(args.backend, state_dir),
        job_id=args.job_id,
        role=args.role,
        worker_id=args.worker_id or worker_id_for(args.role),
        lease_seconds=args.lease_seconds,
        poll_seconds=args.poll_seconds,
        simulate_seconds=args.simulate_seconds,
        crash_after_claim=args.crash_after_claim,
    )
    return 0 if runtime.run_until_idle() >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

