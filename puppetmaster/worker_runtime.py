from __future__ import annotations

import argparse
import os
import random
import sqlite3
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from puppetmaster.cancellation import JobCancelled
from puppetmaster.models import AgentRun, ArtifactType, JobStatus, TaskStatus, now_iso
from puppetmaster.state import resolve_state_dir
from puppetmaster.store_factory import create_store, create_worker_store
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
            configured = 2.0
        return max(0.01, min(configured, max(0.1, self.lease_seconds / 3)))

    def run_once(self) -> bool:
        self.store.reconcile_completions(self.job_id)
        task = self.store.claim_next_task(
            self.job_id,
            self.worker_id,
            role=self.role,
            lease_seconds=self.lease_seconds,
        )
        if task is None:
            return False
        try:
            return self._run_claimed(task)
        finally:
            self._settle_cut(task.id)

    def _settle_cut(self, task_id: str) -> None:
        """Settle a cut that asked this task to stop, once it left RUNNING.

        The orchestrator that settles cuts on its next pass may be gone (a
        stopped flow), so the marker stayed "pending" on a terminal task.
        """
        try:
            current = self.store.get_task_by_id(task_id)
            marker = (current.payload or {}).get("failure_cut")
            if isinstance(marker, dict) and current.status != TaskStatus.RUNNING:
                self.store.finalize_pending_cuts(self.job_id)
        except Exception:
            pass

    def _run_claimed(self, task) -> bool:
        # Lease loss belongs to the previous claim, not this new execution.
        # Its heartbeat has been joined before run_once can return.
        self._lease_lost.clear()

        if self.crash_after_claim:
            self.store.emit(
                self.job_id,
                "worker.crashed_after_claim",
                {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
            )
            raise SystemExit(77)

        # Explicit model pins are executable authority, so revalidate their
        # bound registry epoch after claim and immediately before constructing
        # LocalWorker. A retirement/disable/drift between creation and dispatch
        # must never reach an adapter.
        from puppetmaster.routing_authority import (
            RegistryAuthorityError,
            validate_pinned_dispatch,
        )

        try:
            dispatch_payload = validate_pinned_dispatch(
                task.payload or {}, adapter=task.adapter
            )
            if dispatch_payload != (task.payload or {}):
                task = replace(task, payload=dispatch_payload, updated_at=now_iso())
                self.store.save_task(task)
        except RegistryAuthorityError as exc:
            failed_run = AgentRun(
                job_id=self.job_id,
                task_id=task.id,
                role=task.role,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            self.store.update_task_status(
                task, TaskStatus.FAILED, worker_id=self.worker_id
            )
            self.store.emit(
                self.job_id,
                "worker.failed_task",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": task.role,
                    "failure": "registry_authority_invalid",
                    "error": str(exc),
                },
            )
            return True

        from puppetmaster.prerun import prerun_skip_reason

        skip_reason = prerun_skip_reason(task)
        if not skip_reason and task.role == "conflict-auditor":
            try:
                from puppetmaster.jev.edges import apply_conflict_auditor_gate

                jev_decision = apply_conflict_auditor_gate(
                    self.store, task, worker_id=self.worker_id
                )
            except Exception:
                jev_decision = None
            if jev_decision is not None and getattr(
                jev_decision, "acted", False
            ):
                skip_reason = "jev_transition:conflict_auditor"
        if skip_reason:
            skipped_run = AgentRun(
                job_id=self.job_id,
                task_id=task.id,
                role=task.role,
                worker_id=self.worker_id,
                status=TaskStatus.SKIPPED,
                completed_at=now_iso(),
            )
            self.store.save_run(skipped_run)
            self.store.update_task_status(
                task, TaskStatus.SKIPPED, worker_id=self.worker_id
            )
            self.store.emit(
                self.job_id,
                "worker.prerun_skipped",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": task.role,
                    "reason": skip_reason,
                },
            )
            return True

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
            run, _renewed = self._heartbeat_run_and_lease(run, task.id, task.lease_id)

        stop_heartbeats = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_until_stopped,
            args=(run, task.id, stop_heartbeats),
            kwargs={"lease_id": task.lease_id},
            daemon=True,
        )
        heartbeat.start()
        execution_error = None
        try:
            reused: list = []
            try:
                from puppetmaster.working_set import maybe_reuse_artifacts

                reused = maybe_reuse_artifacts(self.store, task)
            except Exception:
                reused = []

            if reused:
                try:
                    from puppetmaster.working_set import persist_reused_artifacts

                    artifacts = persist_reused_artifacts(
                        self.store,
                        task,
                        reused,
                        worker_id=self.worker_id,
                    )
                except Exception:
                    artifacts = []
                if artifacts:
                    try:
                        from puppetmaster.working_set import rebuild_artifact_index

                        rebuild_artifact_index(
                            self.store.job_dir(self.job_id),
                            self.store,
                            self.job_id,
                        )
                    except Exception:
                        pass
                    worker_run = AgentRun(
                        job_id=self.job_id,
                        task_id=task.id,
                        role=task.role,
                        worker_id=self.worker_id,
                        status=TaskStatus.COMPLETE,
                        completed_at=now_iso(),
                    )
                    try:
                        self.store.emit(
                            self.job_id,
                            "working_set.reused",
                            {
                                "task_id": task.id,
                                "role": task.role,
                                "artifacts": len(artifacts),
                            },
                        )
                    except Exception:
                        pass
                else:
                    reused = []

            if not reused:
                from puppetmaster.edit_admission import EditAdmissionTimeout, edit_admission
                from puppetmaster.invocation import execution_scope
                from puppetmaster.steering import (
                    BOUNDARY_PRE_DISPATCH,
                    apply_steering_to_task,
                    drain_pending,
                )

                native_steer = (
                    task.adapter == "codex"
                    and bool((task.payload or {}).get("native_steer"))
                )
                if not native_steer:
                    steered = drain_pending(
                        self.store, task, boundary=BOUNDARY_PRE_DISPATCH
                    )
                    if steered:
                        task = apply_steering_to_task(task, steered)
                        self.store.save_task(task)
                try:
                    with edit_admission(self.store, task, self.worker_id) as admission:
                        with execution_scope(
                            self.store, run, task, lease_lost=self._lease_lost.is_set
                        ):
                            worker_run, artifacts = LocalWorker(
                                task.role, worker_id=self.worker_id
                            ).run(
                                task,
                                self.store.get_job(self.job_id).goal,
                            )
                        if admission.claims:
                            artifacts = _with_admission_wait(artifacts, admission.waited_seconds)
                        if admission.lost:
                            from puppetmaster.adapters import verification_artifact

                            artifacts = list(artifacts) + [
                                verification_artifact(
                                    task=task,
                                    worker_id=self.worker_id,
                                    adapter=task.adapter,
                                    check="edit_admission",
                                    result="failed",
                                    confidence=1.0,
                                    evidence=["edit_admission:lost"],
                                    payload={"failure": "edit_admission_lost"},
                                )
                            ]
                except EditAdmissionTimeout as exc:
                    from puppetmaster.edit_admission import EditAdmissionCancelled

                    outcome = "cancelled" if isinstance(exc, EditAdmissionCancelled) else "timeout"
                    from puppetmaster.adapters import verification_artifact

                    worker_run = AgentRun(
                        job_id=self.job_id,
                        task_id=task.id,
                        role=task.role,
                        worker_id=self.worker_id,
                        status=TaskStatus.FAILED,
                        completed_at=now_iso(),
                    )
                    artifacts = [
                        verification_artifact(
                            task=task,
                            worker_id=self.worker_id,
                            adapter=task.adapter,
                            check="edit_admission",
                            result="blocked",
                            confidence=1.0,
                            evidence=[f"edit_admission:{outcome}"],
                            payload={
                                "failure": f"edit_admission_{outcome}",
                                "reason": str(exc),
                            },
                        )
                    ]
                if self._lease_lost.is_set():
                    self.store.emit(
                        self.job_id,
                        "worker.lease_lost",
                        {
                            "worker_id": self.worker_id,
                            "task_id": task.id,
                            "role": self.role,
                        },
                    )
                    return True
                try:
                    from puppetmaster.working_set import stamp_fresh_validation

                    artifacts = stamp_fresh_validation(task, artifacts)
                except Exception:
                    pass
                artifacts = self._stamp_evaluator_metadata(task, artifacts)
                # Optional worker claims are persisted separately from the
                # runtime's non-bypassable gates. Missing or ambiguous claims
                # remain absent; neither can imply a PASS.
                from puppetmaster.worker_verdict import verdict_artifacts

                artifacts = verdict_artifacts(task, self.worker_id, artifacts)
                try:
                    from puppetmaster.quality_loop import run_cleanup_pass

                    artifacts = run_cleanup_pass(
                        task, artifacts, store=self.store
                    )
                except Exception:
                    pass
                try:
                    from puppetmaster.steering import BOUNDARY_POST_OUTPUT, drain_pending

                    drain_pending(self.store, task, boundary=BOUNDARY_POST_OUTPUT)
                except Exception:
                    pass
                for artifact in artifacts:
                    try:
                        from puppetmaster.negative_claims import stamp_failed_gate

                        artifact = stamp_failed_gate(
                            artifact,
                            task=task,
                            cwd=(task.payload or {}).get("cwd"),
                        )
                    except Exception:
                        pass
                    self.store.save_artifact(artifact)
                    # Auto-materialize admitted gists from high-confidence findings
                    # so swarm peers share verified compact discoveries without
                    # extra API churn. Follow-ups wait until completion gates
                    # persist so a failed GATE cannot enqueue merge/ship.
                    try:
                        from puppetmaster.gist_admission import maybe_admit_finding_as_gist

                        maybe_admit_finding_as_gist(self.store, artifact)
                    except Exception:
                        pass
                try:
                    from puppetmaster.working_set import rebuild_artifact_index

                    rebuild_artifact_index(
                        self.store.job_dir(self.job_id),
                        self.store,
                        self.job_id,
                    )
                except Exception:
                    pass
        except Exception as exc:
            execution_error = exc
        finally:
            stop_heartbeats.set()
            # SQLite's bounded busy wait/retries can outlast one second. Drain
            # the in-flight renewal before publishing completion or claiming
            # again: a late renewal can otherwise flag the next lease as lost
            # and keep a connection alive into process/temporary-state cleanup.
            heartbeat.join()

        # The final renewal can discover lease loss while execution unwinds.
        # Check ownership only after it has drained on both success and error.
        if self._lease_lost.is_set():
            return True

        if execution_error is not None:
            failed_run = replace(
                run,
                status=TaskStatus.FAILED,
                heartbeat_at=now_iso(),
                completed_at=now_iso(),
            )
            self.store.save_run(failed_run)
            if isinstance(execution_error, JobCancelled):
                self._record_cancelled_attempt(task, execution_error)
            self.store.update_task_status(task, TaskStatus.FAILED, worker_id=self.worker_id)
            self.store.emit(
                self.job_id,
                "worker.failed_task",
                {
                    "worker_id": self.worker_id,
                    "task_id": task.id,
                    "role": self.role,
                    "error": str(execution_error),
                },
            )
            return True

        # Honor a FAILED verdict from the worker (e.g. a preflight block), and
        # also convert an adapter-detected auth/billing/quota rejection that
        # came back as a verification artifact into a truthful FAILED status.
        # Without this the task would be recorded COMPLETE over a run that
        # never produced real output — telemetry would lie, await would report
        # success, and the orchestrator's auto-fallback could never re-route.
        recoverable = self._recoverable_failure(artifacts)
        blocked = self._blocked_verdict(artifacts)
        failed = self._failed_verdict(artifacts)
        if (
            worker_run.status == TaskStatus.FAILED
            or recoverable is not None
            or blocked is not None
            or failed is not None
        ):
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
                    "failure": recoverable or blocked or failed,
                    "blocked": blocked,
                },
            )
            self._emit_live_task_span(updated, artifacts)
            return True

        # Non-bypassable completion gates: an agent may not reach COMPLETE just
        # because it thinks it finished. Post-conditions (drift ratchet, required
        # diff, commit) are evaluated by the runtime; a failed gate is FAILED.
        # A review gate is a live judge call that outlasts the lease, so keep
        # renewing it while gates run; otherwise recovery reclaims the task and
        # the verdict is dropped.
        gate_eval = self._with_heartbeat(
            run, task, lambda: self._evaluate_gates(task, artifacts)
        )
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
        updated = self.store.complete_task(
            task, completed_run, [] if reused else artifacts,
            {"worker_id": self.worker_id, "task_id": task.id, "role": self.role},
        )
        self._emit_live_task_span(updated, artifacts + gate_eval.artifacts)
        return True

    def _record_cancelled_attempt(self, task, exc: JobCancelled) -> None:
        """Write the stop receipt: a cancelled turn, never a completed one.

        It keeps the cut attempt's dispatch and capture linkage and the one
        observed session id. Usage is unknown (NULL), with a reason.
        """
        from puppetmaster.adapters import verification_artifact

        partial = dict(getattr(exc, "partial", None) or {})
        evidence = ["cancellation:stopped"]
        if partial.get("dispatch_receipt"):
            evidence.append(f"dispatch:{partial['dispatch_receipt']}")
        try:
            self.store.save_artifact(verification_artifact(
                task=task,
                worker_id=self.worker_id,
                adapter=task.adapter,
                check="cancellation",
                result="cancelled",
                confidence=1.0,
                evidence=evidence,
                payload={
                    "failure": "cancelled",
                    "turn_completed": False,
                    "usage_known": False,
                    "usage_unknown_reason": "stopped_before_final_usage",
                    "tokens_in": None,
                    "tokens_out": None,
                    "real_cost_usd": None,
                    **partial,
                },
            ))
        except Exception:
            pass

    def _with_heartbeat(self, run, task, work):
        """Run ``work()`` while renewing the task lease; drain the renewal before returning.

        A renewal that fails here (the lease already lapsed on a starved host)
        only stops renewing: it must not abandon a finished task. Publication
        stays fenced by owner and lease id in complete_task/update_task_status,
        which reject the result only if another worker actually took the task.
        """
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_until_stopped,
            args=(run, task.id, stop),
            kwargs={"lease_id": task.lease_id, "lost": threading.Event()},
            daemon=True,
        )
        heartbeat.start()
        try:
            return work()
        finally:
            stop.set()
            heartbeat.join()

    def _stamp_evaluator_metadata(self, task, artifacts: list) -> list:
        try:
            from puppetmaster.evaluators import evaluator_epoch_for_job, stamp_verification_artifacts

            epoch = evaluator_epoch_for_job(self.store, self.job_id)
            if not epoch:
                return artifacts
            return stamp_verification_artifacts(task, artifacts, epoch)
        except Exception:
            return artifacts

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

    @staticmethod
    def _failed_verdict(artifacts: list) -> Optional[str]:
        """Return the failure reason when verification executed and failed.

        Provider HTTP 400s (e.g. unsupported ``luna-pro`` on ChatGPT Codex) and
        other hard verification failures previously could miss
        :data:`RECOVERABLE_FAILURES` and still record the task COMPLETE — a
        green "done" over empty deltas. ``result=failed`` /
        ``execution_status=failed`` must always veto COMPLETE.
        """
        from puppetmaster.models import ArtifactType

        for artifact in artifacts:
            kind = getattr(artifact, "type", None)
            if kind != ArtifactType.VERIFICATION and str(kind) != "verification":
                continue
            payload = getattr(artifact, "payload", None) or {}
            if payload.get("kind") == "worker_verdict" and payload.get("advisory") is True:
                continue
            result = str(payload.get("result") or "").strip().lower()
            exec_status = str(
                payload.get("execution_status")
                or getattr(artifact, "execution_status", None)
                or ""
            ).strip().lower()
            if result == "failed" or exec_status == "failed":
                return str(
                    payload.get("failure")
                    or payload.get("provider_reason")
                    or "verification_failed"
                )
        return None

    def _heartbeat_run_and_lease(
        self,
        run: AgentRun,
        task_id: str,
        lease_id: Optional[str] = None,
        opportunistic: bool = False,
    ):
        method = ("heartbeat_run_and_renew_lease_opportunistic" if opportunistic
                  else "heartbeat_run_and_renew_lease")
        coalesced = getattr(type(self.store), method, None)
        if opportunistic and not callable(coalesced):
            coalesced = getattr(type(self.store), "heartbeat_run_and_renew_lease", None)
        if callable(coalesced):
            return coalesced(
                self.store, run, task_id, self.worker_id, self.lease_seconds, lease_id
            )
        updated = self.store.heartbeat_run(run)
        renewed = self.store.renew_task_lease(
            task_id, self.worker_id, self.lease_seconds, lease_id=lease_id
        )
        return updated, renewed

    def _heartbeat_until_stopped(
        self,
        run: AgentRun,
        task_id: str,
        stop: threading.Event,
        lease_id: Optional[str] = None,
        lost: Optional[threading.Event] = None,
    ) -> None:
        interval = self._heartbeat_interval()
        wait, contended = interval, 0
        while not stop.wait(wait):
            interval = self._heartbeat_interval()
            try:
                run, renewed = self._heartbeat_run_and_lease(
                    run, task_id, lease_id, True
                )
            except sqlite3.OperationalError as exc:
                from puppetmaster.sqlite_store import _is_sqlite_lock_error

                if not _is_sqlite_lock_error(exc):
                    raise
                # A failed reservation did not mutate either record. Retry
                # sooner than a full interval: two renewals lost to write
                # contention in a row used to expire a healthy worker's lease
                # and restart its task. Back off with jitter so many workers
                # retrying together do not add the contention they wait on.
                contended += 1
                wait = min(interval, 0.25 * (2 ** (contended - 1))) * random.uniform(0.5, 1.0)
                continue
            wait, contended = interval, 0
            if renewed is None:
                (lost or self._lease_lost).set()
                stop.set()
                return

    def run_until_idle(self) -> int:
        completed = 0
        while True:
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


# EX_TEMPFAIL: the worker could not attach to the store, so it claimed nothing
# and the supervisor can respawn the role without duplicating work.
WORKER_ATTACH_FAILED_EXIT = 75


def _transient_attach_failure(exc: BaseException) -> bool:
    """Attach failures a fresh process can expect to get past.

    Metadata-only drift (macOS provenance xattr, same-mode chmod), lock or
    busy contention and helper timeouts are transient. A replaced store or a
    missing schema is not: respawning would only bind the wrong store or fail
    again.
    """
    import sqlite3

    from puppetmaster.identity import StoreMetadataDrift

    if isinstance(exc, StoreMetadataDrift):
        return True
    return isinstance(exc, (sqlite3.OperationalError, TimeoutError, BlockingIOError, PermissionError))


def _write_startup_error(
    backend: str,
    state_dir,
    job_id: str,
    worker_id: str,
    exc: BaseException,
) -> None:
    """Record a worker that died during startup or execution.

    A worker that exits outside normal Exception handling can otherwise vanish
    with no trace. Keep the existing startup-error file/event contract while
    preserving the traceback for both startup and execution failures.
    """
    import traceback

    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        from pathlib import Path

        crash_dir = Path(state_dir) / "jobs" / job_id / "tasks"
        crash_dir.mkdir(parents=True, exist_ok=True)
        (crash_dir / f"startup_error-{worker_id}.log").write_text(
            f"worker {worker_id} for job {job_id} failed to start or run:\n\n{detail}",
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass
    try:
        create_store(backend, state_dir, mode="attach").emit(
            job_id,
            "worker.startup_failed",
            {"worker_id": worker_id, "error": str(exc)},
        )
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    from puppetmaster.win_console import hide_child_consoles

    hide_child_consoles()
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
        store = create_worker_store(args.backend, state_dir)
        if store.backend_name == "file":
            # A file store builds its metadata index on first use. Do it while
            # attaching, so lock contention here is a respawnable attach
            # failure instead of a crash after the worker has started.
            store.init()
    except Exception as exc:  # noqa: BLE001 — classified below
        _write_startup_error(args.backend, state_dir, args.job_id, worker_id, exc)
        if _transient_attach_failure(exc):
            # Nothing claimed yet: the supervisor respawns the role.
            return WORKER_ATTACH_FAILED_EXIT
        raise
    try:
        runtime = WorkerRuntime(
            store=store,
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
    except SystemExit as exc:
        # Only the crash-after-claim path owns exit 77. Other SystemExit values
        # are unexpected worker failures and need the same durable traceback as
        # any other BaseException; otherwise the supervisor sees only exit 1.
        if not (args.crash_after_claim and exc.code == 77):
            _write_startup_error(args.backend, state_dir, args.job_id, worker_id, exc)
        raise
    except BaseException as exc:  # noqa: BLE001 — last-resort trace before dying
        _write_startup_error(args.backend, state_dir, args.job_id, worker_id, exc)
        raise


def _with_admission_wait(artifacts: list, waited_seconds: float) -> list:
    """Stamp the edit-admission queue time on the worker's receipts.

    Without it, time serialized behind another writer's claim reads as slow
    model inference in wall-clock comparisons.
    """
    from dataclasses import replace

    return [
        replace(artifact, payload={**(artifact.payload or {}), "edit_admission_wait_seconds": waited_seconds})
        if artifact.type == ArtifactType.VERIFICATION
        else artifact
        for artifact in artifacts
    ]


if __name__ == "__main__":
    raise SystemExit(main())
