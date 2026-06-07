from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from puppetmaster.liveness import record_orchestrator_heartbeat
from puppetmaster.models import Artifact, ArtifactType, Job, JobStatus, Task, TaskStatus, now_iso
from puppetmaster.stitcher import Stitcher
from puppetmaster.store import SwarmStore
from puppetmaster.worker_runtime import WorkerRuntime
from puppetmaster.workers import (
    RECOVERABLE_FAILURES,
    WorkerSpec,
    specs_for_roles,
    swarm_mode,
)

# How many times the orchestrator will re-route a single task to a different
# funded adapter after an auth/billing/quota rejection before giving up, and
# how many whole-job fallback sweeps it will run. Bounded so a systematically
# broken environment (every adapter unfunded) can't loop forever.
_MAX_FALLBACK_ATTEMPTS = 2
_MAX_FALLBACK_ROUNDS = 3

# How many times a single COMPLETE-but-low-confidence task will be re-dispatched
# one capability tier up before its result is accepted as-is. Bounded so a
# genuinely-hard task that no available model is confident about can't loop
# forever (and can't quietly run up a frontier-model bill).
_MAX_ESCALATION_ATTEMPTS = 2

# Roles that judge or verify the system should not inherit prior promoted
# conclusions — that would circularly feed audits their own past claims.
_FRESH_JUDGMENT_ROLES = frozenset(
    {
        "audit",
        "conflict-auditor",
        "cursor-plan",
        "cursor-review",
        "decision-explainer",
        "plan",
        "redteam",
        "review",
        "reviewer",
        "security-review",
        "test",
        "test-coverage-reviewer",
    }
)


def _memory_injection_enabled(spec: WorkerSpec) -> bool:
    override = spec.payload.get("disable_memory")
    if override is True:
        return False
    if override is False:
        return True
    return spec.role not in _FRESH_JUDGMENT_ROLES


@dataclass(frozen=True)
class RunResult:
    job: Job
    artifacts: list[Artifact]
    summary: str
    summary_path: Path
    recovered_tasks: int = 0
    rerouted_tasks: int = 0
    # "edit" when a worker could change files, "analysis" when the swarm is
    # read-only (emits artifacts only). Lets the CLI print an honest banner.
    mode: str = "analysis"


def _tag_job_effort(store: SwarmStore, job_id: str) -> None:
    """Stamp a new job with the ambient effort id (if any) so jobs spread across
    many worktrees can later be rolled up as one effort. Best-effort."""
    try:
        from puppetmaster.lifecycle import current_effort_id, tag_job_effort

        tag_job_effort(store, job_id, current_effort_id())
    except Exception:
        pass


class Orchestrator:
    def __init__(self, store: SwarmStore) -> None:
        self.store = store
        # W3C traceparent for the active job, propagated to worker subprocesses
        # so live per-task spans correlate into one trace (telemetry, opt-in).
        self._traceparent: Optional[str] = None

    def run(
        self,
        goal: str,
        roles: Optional[list[str]] = None,
        specs: Optional[list[WorkerSpec]] = None,
        lease_seconds: int = 5,
        worker_mode: str = "subprocess",
        on_job_created: Optional[Callable[[Job], None]] = None,
    ) -> RunResult:
        job = self.store.create_job(goal)
        _tag_job_effort(self.store, job.id)
        if on_job_created is not None:
            on_job_created(job)
        record_orchestrator_heartbeat(self.store, job.id, started=True)
        self._begin_trace()
        try:
            specs = self._with_retrieved_memory(specs or specs_for_roles(roles), goal)
            self._announce_mode(job, specs)
            self._ensure_plan_catalog(job, specs)
            self.store.update_job_status(job.id, JobStatus.RUNNING)
            tasks = self._create_tasks(job, specs)
            self._run_workers(job, tasks, lease_seconds=lease_seconds, worker_mode=worker_mode)
            rerouted = self._auto_fallback(job, lease_seconds=lease_seconds, worker_mode=worker_mode)
            rerouted += self._auto_escalate(job, lease_seconds=lease_seconds, worker_mode=worker_mode)
            artifacts = self.store.list_artifacts(job.id)

            self.store.update_job_status(job.id, JobStatus.STITCHING)
            summary = Stitcher(self.store).stitch(job.id)
            completed = self.store.update_job_status(job.id, JobStatus.COMPLETE)
            summary_path = self.store.job_dir(job.id) / "summaries" / "stitched.md"
            self._emit_telemetry(completed, artifacts)
            return RunResult(
                job=completed,
                artifacts=artifacts,
                summary=summary,
                summary_path=summary_path,
                rerouted_tasks=rerouted,
                mode=swarm_mode(specs),
            )
        except Exception:
            self.store.update_job_status(job.id, JobStatus.FAILED)
            raise
        finally:
            self._traceparent = None

    def _announce_mode(self, job: Job, specs: list[WorkerSpec]) -> str:
        """Emit a one-line banner classifying the swarm as edit vs analysis so a
        read-only run is never mistaken for one that writes code.

        Returns the mode (``"edit"`` / ``"analysis"``)."""
        mode = swarm_mode(specs)
        if mode == "edit":
            detail = "mode=edit — workers may modify files in the working tree."
        else:
            detail = (
                "mode=analysis (read-only) — no files will be edited; this swarm "
                "only emits artifacts. Use an implement verb or an edit-capable "
                "adapter to land code."
            )
        self.store.emit(job.id, "job.mode", {"mode": mode, "detail": detail})
        return mode

    def run_crash_recovery_demo(
        self,
        goal: str,
        crash_role: str = "implement",
        roles: Optional[list[str]] = None,
    ) -> RunResult:
        job = self.store.create_job(goal)
        _tag_job_effort(self.store, job.id)
        self._begin_trace()
        try:
            specs = self._with_retrieved_memory(specs_for_roles(roles), goal)
            self.store.update_job_status(job.id, JobStatus.RUNNING)
            tasks = self._create_tasks(job, specs)
            self._run_prerequisites(job, tasks, crash_role, lease_seconds=2)
            self.store.refresh_blocked_tasks(job.id)

            target = next((task for task in self.store.list_tasks(job.id) if task.role == crash_role), None)
            if target is None:
                raise RuntimeError(f"crash role not found: {crash_role}")
            while not self.store.dependencies_complete(target):
                current_tasks = self.store.list_tasks(job.id)
                dependencies = self._dependency_closure(current_tasks, target.id)
                self._run_workers(job, dependencies, lease_seconds=2)
                self.store.refresh_blocked_tasks(job.id)
                target = self.store.get_task_by_id(target.id)
            claimed = self.store.claim_task(
                target.id,
                worker_id=f"crashed-{crash_role}-worker",
                lease_seconds=1,
            )
            if claimed is None:
                raise RuntimeError(f"crash role was not claimable: {crash_role}")
            self.store.emit(
                job.id,
                "worker.crashed_after_claim",
                {
                    "worker_id": f"crashed-{crash_role}-worker",
                    "task_id": target.id,
                    "role": crash_role,
                },
            )
            time.sleep(1.2)
            recovered = self.store.recover_stale_tasks(job.id)

            self._run_workers(job, tasks, lease_seconds=2)
            artifacts = self.store.list_artifacts(job.id)
            self.store.update_job_status(job.id, JobStatus.STITCHING)
            summary = Stitcher(self.store).stitch(job.id)
            completed = self.store.update_job_status(job.id, JobStatus.COMPLETE)
            summary_path = self.store.job_dir(job.id) / "summaries" / "stitched.md"
            self._emit_telemetry(completed, artifacts)
            return RunResult(
                job=completed,
                artifacts=artifacts,
                summary=summary,
                summary_path=summary_path,
                recovered_tasks=len(recovered),
            )
        except Exception:
            self.store.update_job_status(job.id, JobStatus.FAILED)
            raise

    def _ensure_plan_catalog(self, job: Job, specs: list[WorkerSpec]) -> None:
        """First-run guarantee that auto-routed work can land on the user's
        subscription. If any spec opts into ``auto_route`` and the registry has
        no plan-billed frontier yet, discover the Cursor plan catalog once so
        frontier work routes through the plan instead of falling off to a
        per-token / depleted account. Never raises; loud (event) when it can't.

        Opt out with ``PUPPETMASTER_AUTODISCOVER=0``.
        """
        import os

        if os.environ.get("PUPPETMASTER_AUTODISCOVER", "1") in ("0", "false", "no"):
            return
        if not any((s.payload or {}).get("auto_route") for s in specs):
            return

        from puppetmaster.model_registry import default_registry_path
        from puppetmaster.platform_lock import is_adapter_enabled

        override = next(
            (
                (s.payload or {}).get("registry_path")
                for s in specs
                if (s.payload or {}).get("registry_path")
            ),
            None,
        )
        registry_path = (
            Path(str(override)).expanduser() if override else default_registry_path()
        )

        # Respect the platform lock: only discover catalogs for enabled
        # platforms so a disabled adapter is never auto-added behind the
        # user's back. The lock is a global per-user setting, so it is read
        # from its canonical location regardless of any per-job registry
        # override (keeping it consistent with routing + fallback).
        report: dict = {"action": "skipped"}
        if is_adapter_enabled("cursor"):
            try:
                from puppetmaster.cursor_discovery import ensure_cursor_plan_catalog

                report = ensure_cursor_plan_catalog(registry_path)
            except Exception as exc:
                report = {"action": "unavailable", "error": str(exc)}
            self._emit_plan_catalog_event(job, report, "Cursor plan", "cursor")

        # If Cursor didn't supply a plan frontier (no Cursor plan, it's not
        # authenticated, or it's disabled), fall back to the curated
        # subscription catalogs so Claude Code (OAuth) / Codex (ChatGPT) users
        # still get plan-first routing — but only for enabled platforms.
        if report.get("action") != "discovered" and (
            is_adapter_enabled("claude-code") or is_adapter_enabled("codex")
        ):
            try:
                from puppetmaster.static_catalog import (
                    ensure_subscription_plan_catalog,
                )

                sub_report = ensure_subscription_plan_catalog(registry_path)
            except Exception as exc:
                sub_report = {"action": "unavailable", "error": str(exc)}
            label = sub_report.get("adapter") or "subscription"
            self._emit_plan_catalog_event(
                job, sub_report, f"{label} subscription", sub_report.get("source")
            )

    def _emit_plan_catalog_event(
        self, job: Job, report: dict, label: str, source: Optional[str]
    ) -> None:
        """Emit a loud event describing a plan-catalog auto-merge outcome."""
        action = report.get("action")
        if action == "discovered":
            self.store.emit(
                job.id,
                "router.plan_catalog_discovered",
                {
                    "source": source,
                    "discovered_count": report.get("discovered_count"),
                    "detail": (
                        f"Auto-discovered the {label} catalog so frontier work "
                        "routes through your subscription (plan-billed, $0 marginal)."
                    ),
                },
            )
        elif action == "unavailable":
            self.store.emit(
                job.id,
                "router.plan_catalog_unavailable",
                {
                    "source": source,
                    "error": report.get("error"),
                    "hint": (
                        f"Could not enumerate the {label} catalog; routing falls "
                        "back to the existing registry. Run `python -m puppetmaster "
                        "models discover --write` once the platform is authenticated "
                        "to keep frontier work on your plan."
                    ),
                },
            )

    def _begin_trace(self) -> None:
        """Mint a W3C traceparent for this job when telemetry is enabled.

        Stored on the orchestrator so ``_spawn_worker`` can hand it to worker
        subprocesses (live per-task spans) and ``_emit_telemetry`` can anchor
        the assembled end-of-job trace to the same trace id."""
        self._traceparent = None
        try:
            from puppetmaster.telemetry import new_traceparent, telemetry_enabled

            if telemetry_enabled():
                self._traceparent = new_traceparent()
        except Exception:
            self._traceparent = None

    def _emit_telemetry(self, job: Job, artifacts: list[Artifact]) -> None:
        """Emit an OTel trace + metrics for the finished job. No-op unless
        tracing is on; never lets a telemetry failure break the run."""
        try:
            from puppetmaster.telemetry import (
                record_job_metrics,
                record_job_trace,
                telemetry_enabled,
            )

            if not telemetry_enabled():
                return
            tasks = self.store.list_tasks(job.id)
            record_job_trace(job, tasks, artifacts, traceparent=self._traceparent)
            record_job_metrics(job, tasks, artifacts)
        except Exception:
            pass

    def _auto_fallback(
        self, job: Job, *, lease_seconds: int, worker_mode: str
    ) -> int:
        """Re-route tasks that hit an auth/billing/quota rejection.

        After the main worker pass, any task left FAILED because *its adapter's
        account* was unfunded/unauthenticated (not because the task itself was
        bad) is re-routed to the cheapest sufficient model on a different,
        verified-funded adapter and re-run. Bounded by
        :data:`_MAX_FALLBACK_ROUNDS` and per-task :data:`_MAX_FALLBACK_ATTEMPTS`
        so a fully-broken environment surfaces loudly (via the stitched Alerts
        section) instead of looping. Returns the number of re-routes performed.
        """
        total = 0
        for _ in range(_MAX_FALLBACK_ROUNDS):
            rerouted = self._reroute_recoverable_failures(job)
            if not rerouted:
                break
            total += rerouted
            self.store.emit(job.id, "job.auto_fallback_round", {"rerouted": rerouted})
            self._run_workers(
                job,
                self.store.list_tasks(job.id),
                lease_seconds=lease_seconds,
                worker_mode=worker_mode,
            )
        return total

    def _reroute_recoverable_failures(self, job: Job) -> int:
        """Re-queue each FAILED task with a recoverable failure onto a funded
        adapter. Returns the count re-queued (0 when nothing is re-routable)."""
        from puppetmaster.model_registry import default_registry_path, load_registry
        from puppetmaster.platform_billing import detect_adapter_billing
        from puppetmaster.router import (
            NoEligibleModelError,
            route_task,
            signals_from_worker_spec,
        )

        failed = [
            task
            for task in self.store.list_tasks(job.id)
            if task.status == TaskStatus.FAILED
            and int((task.payload or {}).get("fallback_attempts", 0)) < _MAX_FALLBACK_ATTEMPTS
        ]
        if not failed:
            return 0

        failure_by_task = self._recoverable_failure_by_task(job)
        failed = [t for t in failed if t.id in failure_by_task]
        if not failed:
            return 0

        registry_path = default_registry_path()
        registry = [s for s in load_registry(registry_path) if s.enabled]
        if not registry:
            return 0

        billing_cache: dict[str, object] = {}

        def _funded(adapter: str) -> object:
            if adapter not in billing_cache:
                try:
                    billing_cache[adapter] = detect_adapter_billing(adapter)
                except Exception:
                    billing_cache[adapter] = None
            return billing_cache[adapter]

        from puppetmaster.platform_lock import is_adapter_enabled

        rerouted = 0
        for task in failed:
            failed_adapter = task.adapter
            allow_api = bool((task.payload or {}).get("allow_api_billing", True))
            candidates = []
            for spec in registry:
                if spec.adapter == failed_adapter:
                    continue
                if not is_adapter_enabled(spec.adapter):
                    # Platform lock: never fall back onto a disabled platform.
                    continue
                status = _funded(spec.adapter)
                if status is None or not getattr(status, "healthy", False):
                    continue
                if getattr(status, "billing", "unknown") == "api" and not allow_api:
                    continue
                candidates.append(spec)
            if not candidates:
                continue
            policy = (task.payload or {}).get("router_policy") or "balanced"
            try:
                decision = route_task(
                    signals_from_worker_spec(task), candidates, policy=policy
                )
            except NoEligibleModelError:
                continue

            attempts = int((task.payload or {}).get("fallback_attempts", 0)) + 1
            new_payload = {
                **(task.payload or {}),
                "model": decision.model.adapter_model_name,
                "router_model_id": decision.model.id,
                "router_policy": decision.policy,
                "router_capability_needed": decision.capability_needed,
                "router_estimated_cost_usd": decision.estimated_cost_usd,
                "fallback_attempts": attempts,
                "fallback_from_adapter": failed_adapter,
            }
            requeued = replace(
                task,
                adapter=decision.model.adapter,
                payload=new_payload,
                status=TaskStatus.QUEUED,
                attempts=0,
                lease_owner=None,
                lease_expires_at=None,
                completed_at=None,
                updated_at=now_iso(),
            )
            self.store.save_task(requeued)

            artifact_payload = decision.to_artifact_payload()
            artifact_payload["role"] = task.role
            artifact_payload["fallback_from_adapter"] = failed_adapter
            artifact_payload["fallback_reason"] = failure_by_task[task.id]
            artifact_payload["fallback_attempt"] = attempts
            self.store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router-fallback",
                    payload=artifact_payload,
                    confidence=0.9,
                    evidence=[
                        f"fallback_from:{failed_adapter}",
                        f"reason:{failure_by_task[task.id]}",
                        f"to:{decision.model.id}",
                    ],
                )
            )
            self.store.emit(
                job.id,
                "router.auto_fallback",
                {
                    "task_id": task.id,
                    "role": task.role,
                    "from_adapter": failed_adapter,
                    "to_adapter": decision.model.adapter,
                    "to_model": decision.model.adapter_model_name,
                    "reason": failure_by_task[task.id],
                    "attempt": attempts,
                },
            )
            rerouted += 1
        return rerouted

    def _auto_escalate(
        self, job: Job, *, lease_seconds: int, worker_mode: str
    ) -> int:
        """Re-dispatch COMPLETE-but-low-confidence tasks one capability tier up.

        Static upfront routing can under-provision: a task that *looked* simple
        can turn out hard once a worker is in it, and the cheap model it was
        routed to will say so via a low-confidence VERIFICATION artifact. When a
        confidence threshold is configured (per-task ``payload.min_confidence``
        or ``$PUPPETMASTER_ESCALATE_CONFIDENCE``; **off by default** so the
        cost-saving promise holds unless the user opts in), this re-routes such a
        task to the cheapest *strictly stronger* funded + platform-enabled model
        and re-runs it before its result is accepted. Bounded by
        :data:`_MAX_ESCALATION_ATTEMPTS`; a task already on the top tier (no
        stronger model exists) is left as-is. Returns the number of re-routes.
        """
        total = 0
        for _ in range(_MAX_ESCALATION_ATTEMPTS):
            rerouted = self._reroute_low_confidence(job)
            if not rerouted:
                break
            total += rerouted
            self.store.emit(job.id, "job.auto_escalate_round", {"rerouted": rerouted})
            self._run_workers(
                job,
                self.store.list_tasks(job.id),
                lease_seconds=lease_seconds,
                worker_mode=worker_mode,
            )
        return total

    def _reroute_low_confidence(self, job: Job) -> int:
        """Re-queue each COMPLETE task whose verification confidence is below its
        configured threshold onto the cheapest strictly-stronger funded model.
        Returns the count re-queued (0 when nothing qualifies)."""
        from puppetmaster.model_registry import default_registry_path, load_registry
        from puppetmaster.platform_billing import detect_adapter_billing
        from puppetmaster.platform_lock import is_adapter_enabled
        from puppetmaster.router import (
            NoEligibleModelError,
            route_task,
            signals_from_worker_spec,
        )

        registry = [s for s in load_registry(default_registry_path()) if s.enabled]
        if not registry:
            return 0
        by_model_id = {s.id: s for s in registry}

        billing_cache: dict[str, object] = {}

        def _funded(adapter: str) -> object:
            if adapter not in billing_cache:
                try:
                    billing_cache[adapter] = detect_adapter_billing(adapter)
                except Exception:
                    billing_cache[adapter] = None
            return billing_cache[adapter]

        rerouted = 0
        artifacts = self.store.list_artifacts(job.id)
        confidence_by_task = self._verification_confidence_by_task(artifacts)
        for task in self.store.list_tasks(job.id):
            if task.status != TaskStatus.COMPLETE:
                continue
            payload = task.payload or {}
            threshold = self._escalation_threshold(task)
            if threshold is None:
                continue  # feature not enabled for this task
            if int(payload.get("escalation_attempts", 0)) >= _MAX_ESCALATION_ATTEMPTS:
                continue
            # Only escalate work the router placed — don't override a model the
            # user pinned by hand.
            current_model_id = payload.get("router_model_id")
            current_spec = by_model_id.get(current_model_id) if current_model_id else None
            if current_spec is None:
                continue
            confidence = confidence_by_task.get(task.id)
            if confidence is None or confidence >= threshold:
                continue

            allow_api = bool(payload.get("allow_api_billing", True))
            candidates = []
            for spec in registry:
                if not is_adapter_enabled(spec.adapter):
                    continue
                status = _funded(spec.adapter)
                if status is None or not getattr(status, "healthy", False):
                    continue
                if getattr(status, "billing", "unknown") == "api" and not allow_api:
                    continue
                candidates.append(spec)
            if not candidates:
                continue

            # Demand strictly more capability than the current model by lifting
            # the floor one point above its score, then route normally.
            signals = replace(
                signals_from_worker_spec(task),
                explicit_min_capability=current_spec.capability_score + 1,
            )
            policy = payload.get("router_policy") or "balanced"
            try:
                decision = route_task(signals, candidates, policy=policy)
            except NoEligibleModelError:
                continue
            # `balanced` degrades to the highest-available model when nothing
            # meets the floor — guard so we only act on a genuine upgrade.
            if (
                decision.model.id == current_model_id
                or decision.model.capability_score <= current_spec.capability_score
            ):
                continue

            attempts = int(payload.get("escalation_attempts", 0)) + 1
            new_payload = {
                **payload,
                "model": decision.model.adapter_model_name,
                "router_model_id": decision.model.id,
                "router_policy": decision.policy,
                "router_capability_needed": decision.capability_needed,
                "router_estimated_cost_usd": decision.estimated_cost_usd,
                "escalation_attempts": attempts,
                "escalated_from_model": current_model_id,
                "escalated_from_confidence": confidence,
            }
            requeued = replace(
                task,
                adapter=decision.model.adapter,
                payload=new_payload,
                status=TaskStatus.QUEUED,
                attempts=0,
                lease_owner=None,
                lease_expires_at=None,
                completed_at=None,
                updated_at=now_iso(),
            )
            self.store.save_task(requeued)

            artifact_payload = decision.to_artifact_payload()
            artifact_payload["role"] = task.role
            artifact_payload["escalated_from_model"] = current_model_id
            artifact_payload["escalated_from_confidence"] = round(confidence, 3)
            artifact_payload["confidence_threshold"] = threshold
            artifact_payload["escalation_attempt"] = attempts
            self.store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router-escalation",
                    payload=artifact_payload,
                    confidence=0.9,
                    evidence=[
                        f"escalated_from:{current_model_id}",
                        f"confidence:{confidence:.2f}<{threshold:.2f}",
                        f"to:{decision.model.id}",
                    ],
                )
            )
            self.store.emit(
                job.id,
                "router.auto_escalate",
                {
                    "task_id": task.id,
                    "role": task.role,
                    "from_model": current_model_id,
                    "to_model": decision.model.id,
                    "confidence": round(confidence, 3),
                    "threshold": threshold,
                    "attempt": attempts,
                },
            )
            rerouted += 1
        return rerouted

    @staticmethod
    def _escalation_threshold(task: Task) -> Optional[float]:
        """The confidence floor below which ``task`` should escalate, or ``None``
        when escalation is disabled (the default).

        Per-task ``payload.min_confidence`` wins; otherwise
        ``$PUPPETMASTER_ESCALATE_CONFIDENCE`` enables it globally. Both must be a
        float in ``(0, 1]`` to take effect."""
        import os

        payload = task.payload or {}
        raw = payload.get("min_confidence")
        if raw is None:
            raw = os.environ.get("PUPPETMASTER_ESCALATE_CONFIDENCE")
        if raw is None or raw == "":
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if 0 < value <= 1 else None

    @staticmethod
    def _verification_confidence_by_task(
        artifacts: list[Artifact],
    ) -> dict[str, float]:
        latest: dict[str, tuple[str, float]] = {}
        for artifact in artifacts:
            if artifact.type != ArtifactType.VERIFICATION:
                continue
            previous = latest.get(artifact.task_id)
            if previous is None or artifact.created_at > previous[0]:
                latest[artifact.task_id] = (artifact.created_at, artifact.confidence)
        return {task_id: confidence for task_id, (_, confidence) in latest.items()}

    def _latest_verification_confidence(
        self,
        job: Job,
        task_id: str,
        artifacts: Optional[list[Artifact]] = None,
    ) -> Optional[float]:
        """Confidence of the task's most recent VERIFICATION artifact (the run's
        own self-assessment), or ``None`` if it never emitted one."""
        if artifacts is None:
            artifacts = self.store.list_artifacts(job.id)
        return self._verification_confidence_by_task(artifacts).get(task_id)

    def _recoverable_failure_by_task(
        self,
        job: Job,
        artifacts: Optional[list[Artifact]] = None,
    ) -> dict[str, str]:
        """Map task_id -> recoverable failure class from the task's *latest*
        failed artifact.

        Keeping the first artifact seen meant a stale failure reason could win
        after a retry produced a newer one; compare created_at so the most
        recent recoverable failure per task is the one reported."""
        out: dict[str, str] = {}
        latest_at: dict[str, str] = {}
        if artifacts is None:
            artifacts = self.store.list_artifacts(job.id)
        for artifact in artifacts:
            failure = (artifact.payload or {}).get("failure")
            if failure not in RECOVERABLE_FAILURES:
                continue
            task_id = artifact.task_id
            if task_id not in latest_at or artifact.created_at > latest_at[task_id]:
                latest_at[task_id] = artifact.created_at
                out[task_id] = str(failure)
        return out

    def _has_hard_failure(
        self,
        job: Job,
        allowed_task_ids: set[str],
        tasks: Optional[list[Task]] = None,
        artifacts: Optional[list[Artifact]] = None,
    ) -> bool:
        """True when a task FAILED for a non-recoverable reason (a real error
        like a bad adapter or an exception) — i.e. nothing auto-fallback can fix.

        Deliberately does NOT flag QUEUED/RUNNING tasks: those are normal
        mid-flight states while daemon/external workers are still going. Use
        :meth:`_should_fail_closed` at a terminal point (workers exited) to also
        catch a genuinely stuck swarm."""
        recoverable = set(self._recoverable_failure_by_task(job, artifacts=artifacts))
        if tasks is None:
            tasks = self.store.list_tasks(job.id)
        for task in tasks:
            if task.id not in allowed_task_ids:
                continue
            if task.status == TaskStatus.FAILED and task.id not in recoverable:
                return True
        return False

    def _should_fail_closed(self, job: Job, allowed_task_ids: set[str]) -> bool:
        """Terminal-point verdict: should the orchestrator raise (fail the job)?

        Called after workers have exited and no recovery/unblock/next-task
        progress is possible. Raise when any incomplete task is *not* explained
        by a recoverable adapter-billing failure: a hard error, a genuinely
        stuck QUEUED/RUNNING task, or a task blocked on a hard-failed upstream.
        Recoverable-failed tasks and the tasks blocked behind them are left for
        the auto-fallback sweep (or surfaced in the stitched Alerts section)."""
        recoverable = set(self._recoverable_failure_by_task(job))
        by_id = {t.id: t for t in self.store.list_tasks(job.id)}

        def _blocked_on_recoverable(task: Task, seen: set[str]) -> bool:
            if task.id in seen:
                return False
            seen.add(task.id)
            for dep_id in task.depends_on:
                dep = by_id.get(dep_id)
                if dep is None:
                    continue
                if dep.id in recoverable:
                    return True
                if dep.status == TaskStatus.BLOCKED and _blocked_on_recoverable(dep, seen):
                    return True
            return False

        for task in by_id.values():
            if task.id not in allowed_task_ids or task.status == TaskStatus.COMPLETE:
                continue
            if task.id in recoverable:
                continue
            if task.status == TaskStatus.BLOCKED and _blocked_on_recoverable(task, set()):
                continue
            return True
        return False

    def _daemon_settled(
        self,
        job: Job,
        allowed_task_ids: set[str],
        tasks: Optional[list[Task]] = None,
    ) -> bool:
        """True when no daemon/external worker can make further progress.

        Returns False while any task is QUEUED/RUNNING (work remains) or any
        BLOCKED task still has an upstream that could complete and unblock it —
        so the wait loop keeps polling. Returns True once everything is COMPLETE
        or terminally stuck (FAILED, or BLOCKED behind a terminal dep), letting
        the caller hand off to the auto-fallback sweep."""
        if tasks is None:
            tasks = self.store.list_tasks(job.id)
        by_id = {t.id: t for t in tasks}
        for task in by_id.values():
            if task.id not in allowed_task_ids:
                continue
            if task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                return False
            if task.status == TaskStatus.BLOCKED:
                deps = [by_id[d] for d in task.depends_on if d in by_id]
                if any(
                    d.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED}
                    for d in deps
                ):
                    return False
        return True

    def _create_tasks(self, job: Job, specs: list[WorkerSpec]) -> list[Task]:
        specs, routing_decisions = self._apply_auto_routing(job, specs)
        tasks_by_role: dict[str, Task] = {}
        for spec in specs:
            task = Task(
                job_id=job.id,
                role=spec.role,
                instruction=spec.instruction,
                adapter=spec.adapter,
                payload=spec.payload,
            )
            tasks_by_role[spec.role] = task

        tasks = []
        for spec in specs:
            depends_on = [
                tasks_by_role[role].id
                for role in spec.depends_on_roles
                if role in tasks_by_role
            ]
            base = tasks_by_role[spec.role]
            initial_status = self._initial_task_status(depends_on)
            tasks.append(
                Task(
                    id=base.id,
                    job_id=base.job_id,
                    role=base.role,
                    instruction=base.instruction,
                    adapter=base.adapter,
                    payload=base.payload,
                    depends_on=depends_on,
                    status=initial_status,
                    created_at=base.created_at,
                    updated_at=base.updated_at,
                )
            )
        self._validate_task_graph(tasks)
        self.store.save_tasks(tasks)
        self._emit_routing_artifacts(job, tasks_by_role, routing_decisions)
        return tasks

    def _emit_routing_artifacts(
        self,
        job: Job,
        tasks_by_role: dict[str, Task],
        routing_decisions: list[tuple[str, dict]],
    ) -> None:
        """Persist one ROUTING artifact per auto-routed task.

        Done after task creation so the artifact carries the real
        ``task_id`` (rather than a placeholder), keeping the audit
        story consistent with the rest of the store.
        """
        routing_artifacts: list[Artifact] = []
        for role, artifact_payload in routing_decisions:
            task = tasks_by_role.get(role)
            if task is None:
                continue
            routing_artifacts.append(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload=artifact_payload,
                    confidence=0.9,
                    evidence=[
                        f"role:{role}",
                        f"policy:{artifact_payload.get('policy')}",
                        f"capability_needed:{artifact_payload.get('capability_needed')}",
                    ],
                )
            )
        if routing_artifacts:
            self.store.save_artifacts(routing_artifacts)

    def _apply_auto_routing(
        self, job: Job, specs: list[WorkerSpec]
    ) -> tuple[list[WorkerSpec], list[tuple[str, dict]]]:
        """Resolve ``payload.auto_route`` specs through the model router.

        Specs opt in by setting ``payload["auto_route"] = True``. When set:

        * The router picks a :class:`ModelSpec` based on the spec's role +
          instruction + payload overrides (``min_capability``,
          ``max_cost_usd``, ``required_tags``, ``routing_policy``).
        * The chosen ``adapter`` and model name are stamped into the
          spec so the existing adapter pipeline runs the right model
          without any further plumbing.
        * Routing decisions are returned so the caller can persist them
          as :class:`ArtifactType.ROUTING` artifacts after task creation
          (when real ``task_id``s exist).

        Specs that don't opt in are passed through unchanged. The
        router never silently overrides an explicit choice — opt-in
        only.
        """
        from puppetmaster.model_registry import default_registry_path, load_registry
        from puppetmaster.router import (
            NoEligibleModelError,
            route_task,
            signals_from_worker_spec,
        )

        result: list[WorkerSpec] = []
        decisions: list[tuple[str, dict]] = []
        registry_cache: Optional[list] = None
        registry_path: Optional[Path] = None
        empty_registry_announced = False
        for spec in specs:
            payload = spec.payload or {}
            if not payload.get("auto_route"):
                result.append(spec)
                continue
            if registry_cache is None:
                registry_path_override = payload.get("registry_path")
                registry_path = (
                    Path(str(registry_path_override)).expanduser()
                    if registry_path_override
                    else default_registry_path()
                )
                registry_cache = load_registry(registry_path)
            if not registry_cache:
                # No registry on disk yet (user hasn't run `models init`).
                # Don't fail the run — pass the spec through unmodified.
                # Emit one diagnostic event per run so the user can spot
                # the opportunity to opt in, without spamming.
                if not empty_registry_announced:
                    self.store.emit(
                        job.id,
                        "router.registry_empty",
                        {
                            "registry_path": str(registry_path)
                            if registry_path
                            else None,
                            "hint": (
                                "Run `python -m puppetmaster models init` "
                                "to enable per-task model routing."
                            ),
                        },
                    )
                    empty_registry_announced = True
                result.append(spec)
                continue
            policy = payload.get("routing_policy") or "balanced"
            signals = signals_from_worker_spec(spec)
            try:
                decision = route_task(signals, registry_cache, policy=policy)
            except NoEligibleModelError as exc:
                self.store.emit(
                    job.id,
                    "router.no_eligible_model",
                    {
                        "role": spec.role,
                        "policy": policy,
                        "reason": str(exc),
                        "registry_path": str(registry_path) if registry_path else None,
                    },
                )
                result.append(spec)
                continue

            new_payload = {
                **payload,
                "model": decision.model.adapter_model_name,
                "router_model_id": decision.model.id,
                "router_policy": decision.policy,
                "router_capability_needed": decision.capability_needed,
                "router_estimated_cost_usd": decision.estimated_cost_usd,
            }
            routed_spec = replace(
                spec,
                adapter=decision.model.adapter,
                payload=new_payload,
            )
            result.append(routed_spec)

            artifact_payload = decision.to_artifact_payload()
            artifact_payload["role"] = spec.role
            artifact_payload["registry_path"] = str(registry_path) if registry_path else None
            decisions.append((spec.role, artifact_payload))

        return result, decisions

    def _with_retrieved_memory(self, specs: list[WorkerSpec], goal: str) -> list[WorkerSpec]:
        memory = self.store.retrieve_memory(goal)
        if not memory:
            return specs
        result: list[WorkerSpec] = []
        for spec in specs:
            if not _memory_injection_enabled(spec):
                result.append(spec)
                continue
            result.append(
                replace(
                    spec,
                    payload={
                        **spec.payload,
                        "retrieved_memory": memory,
                    },
                )
            )
        return result

    @staticmethod
    def _initial_task_status(depends_on: list[str]) -> TaskStatus:
        return TaskStatus.BLOCKED if depends_on else TaskStatus.QUEUED

    @staticmethod
    def _validate_task_graph(tasks: list[Task]) -> None:
        task_ids = {task.id for task in tasks}
        for task in tasks:
            if task.id in task.depends_on:
                raise ValueError(f"task cannot depend on itself: {task.id}")
            missing = set(task.depends_on) - task_ids
            if missing:
                raise ValueError(f"task has missing dependencies: {task.id}")

        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {task.id: task for task in tasks}

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("task dependency graph contains a cycle")
            visiting.add(task_id)
            for dependency_id in by_id[task_id].depends_on:
                visit(dependency_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task in tasks:
            visit(task.id)

    def _run_workers(
        self,
        job: Job,
        tasks: list[Task],
        lease_seconds: int = 5,
        allowed_task_ids: Optional[set[str]] = None,
        worker_mode: str = "subprocess",
    ) -> None:
        if worker_mode == "inline":
            self._run_inline_workers(
                job,
                tasks,
                lease_seconds=lease_seconds,
                allowed_task_ids=allowed_task_ids,
            )
            return
        if worker_mode == "daemon":
            self._wait_for_daemon_workers(job, tasks, allowed_task_ids=allowed_task_ids)
            return
        if worker_mode != "subprocess":
            raise ValueError(f"unsupported worker mode: {worker_mode}")

        allowed_task_ids = allowed_task_ids or {task.id for task in tasks}
        tasks = [
            task
            for task in self.store.list_tasks(job.id)
            if task.id in allowed_task_ids
            and task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
        ]
        if not tasks:
            self.store.refresh_blocked_tasks(job.id)
            tasks = [
                task
                for task in self.store.list_tasks(job.id)
                if task.id in allowed_task_ids
                and task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
            ]
        if not tasks:
            if self._should_fail_closed(job, allowed_task_ids):
                raise RuntimeError("swarm exited with incomplete tasks")
            return

        roles = sorted({task.role for task in tasks})
        record_orchestrator_heartbeat(self.store, job.id)
        processes = [
            self._spawn_worker(job.id, role, lease_seconds=lease_seconds)
            for role in roles
        ]
        for process in processes:
            try:
                process.wait(timeout=self._worker_wait_timeout(tasks))
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self.store.emit(
                    job.id,
                    "worker.timed_out",
                    {"returncode": process.returncode, "timeout_seconds": self._worker_wait_timeout(tasks)},
                )
                raise RuntimeError("worker process timed out") from exc
            if process.returncode != 0:
                raise RuntimeError(f"worker process failed with exit code {process.returncode}")

        if self.store.has_incomplete_tasks(job.id):
            recovered = self.store.recover_stale_tasks(job.id)
            unblocked = self.store.refresh_blocked_tasks(job.id)
            next_tasks = [
                task
                for task in self.store.list_tasks(job.id)
                if task.id in allowed_task_ids
                and task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
            ]
            if recovered or unblocked or next_tasks:
                self._run_workers(
                    job,
                    next_tasks,
                    lease_seconds=lease_seconds,
                    allowed_task_ids=allowed_task_ids,
                    worker_mode=worker_mode,
                )
            elif self._should_fail_closed(job, allowed_task_ids):
                raise RuntimeError("swarm exited with incomplete tasks")

    def _run_inline_workers(
        self,
        job: Job,
        tasks: list[Task],
        lease_seconds: int = 5,
        allowed_task_ids: Optional[set[str]] = None,
    ) -> None:
        allowed_task_ids = allowed_task_ids or {task.id for task in tasks}
        while True:
            record_orchestrator_heartbeat(self.store, job.id)
            self.store.recover_stale_tasks(job.id)
            self.store.refresh_blocked_tasks(job.id)
            ready_tasks = [
                task
                for task in self.store.list_tasks(job.id)
                if task.id in allowed_task_ids
                and task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
            ]
            if not ready_tasks:
                if self._should_fail_closed(job, allowed_task_ids):
                    raise RuntimeError("swarm exited with incomplete tasks")
                # Either fully complete, or only recoverable adapter-billing
                # failures remain — hand back to the auto-fallback sweep.
                return

            completed = 0
            for role in sorted({task.role for task in ready_tasks}):
                runtime = WorkerRuntime(
                    store=self.store,
                    job_id=job.id,
                    role=role,
                    worker_id=f"worker-{role}-inline",
                    lease_seconds=lease_seconds,
                )
                completed += runtime.run_until_idle()
            if completed == 0:
                if self._should_fail_closed(job, allowed_task_ids):
                    raise RuntimeError("swarm exited with incomplete tasks")
                # No progress and only recoverable failures left — stop spinning
                # and let auto_fallback re-route on a funded adapter.
                return

    def _wait_for_daemon_workers(
        self,
        job: Job,
        tasks: list[Task],
        allowed_task_ids: Optional[set[str]] = None,
    ) -> None:
        allowed_task_ids = allowed_task_ids or {task.id for task in tasks}
        timeout_seconds = self._worker_wait_timeout(tasks)
        deadline = time.monotonic() + timeout_seconds
        self.store.emit(
            job.id,
            "job.waiting_for_daemon_workers",
            {"timeout_seconds": timeout_seconds},
        )
        while time.monotonic() < deadline:
            record_orchestrator_heartbeat(self.store, job.id)
            self.store.recover_stale_tasks(job.id)
            self.store.refresh_blocked_tasks(job.id)
            all_tasks = self.store.list_tasks(job.id)
            all_artifacts = self.store.list_artifacts(job.id)
            current_tasks = [
                task for task in all_tasks if task.id in allowed_task_ids
            ]
            if self._has_hard_failure(
                job, allowed_task_ids, tasks=all_tasks, artifacts=all_artifacts
            ):
                raise RuntimeError("daemon worker failed a task")
            if current_tasks and self._daemon_settled(
                job, allowed_task_ids, tasks=all_tasks
            ):
                return
            time.sleep(0.2)
        raise RuntimeError("timed out waiting for daemon workers")

    def _run_prerequisites(
        self,
        job: Job,
        tasks: list[Task],
        target_role: str,
        lease_seconds: int = 5,
    ) -> None:
        target = next((task for task in tasks if task.role == target_role), None)
        if target is None or not target.depends_on:
            return
        dependencies = self._dependency_closure(tasks, target.id)
        roles = sorted({task.role for task in dependencies})
        processes = [
            self._spawn_worker(job.id, role, lease_seconds=lease_seconds)
            for role in roles
        ]
        try:
            for process in processes:
                process.wait(timeout=self._worker_wait_timeout(dependencies))
                if process.returncode != 0:
                    raise RuntimeError(
                        f"prerequisite worker failed with exit code {process.returncode}"
                    )
        finally:
            # If a wait timed out or a prerequisite failed mid-batch, don't
            # leave the remaining workers running as orphans — terminate (then
            # kill) any that are still alive so they don't outlive the job.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

    @staticmethod
    def _dependency_closure(tasks: list[Task], task_id: str) -> list[Task]:
        by_id = {task.id: task for task in tasks}
        collected: dict[str, Task] = {}

        def collect(current_id: str) -> None:
            for dependency_id in by_id[current_id].depends_on:
                if dependency_id in collected:
                    continue
                collected[dependency_id] = by_id[dependency_id]
                collect(dependency_id)

        collect(task_id)
        return list(collected.values())

    def _spawn_worker(
        self,
        job_id: str,
        role: str,
        lease_seconds: int = 5,
        crash_after_claim: bool = False,
    ) -> subprocess.Popen:
        command = [
            sys.executable,
            "-m",
            "puppetmaster.worker_runtime",
            "--state-dir",
            str(self.store.root),
            "--backend",
            self.store.backend_name,
            "--job-id",
            job_id,
            "--role",
            role,
            "--lease-seconds",
            str(lease_seconds),
        ]
        if crash_after_claim:
            command.append("--crash-after-claim")
        # Always hand the worker (and any agent it spawns) a PYTHONPATH that puts
        # this install first, so a self-served `python -m puppetmaster codegraph`
        # can't resolve a stale pip build that lacks the subcommand (#4).
        import os

        from puppetmaster.codegraph import inject_worker_cli_env

        env = inject_worker_cli_env(dict(os.environ))
        if self._traceparent:
            env["TRACEPARENT"] = self._traceparent
            env["PUPPETMASTER_TRACEPARENT"] = self._traceparent
        return subprocess.Popen(command, env=env)

    @staticmethod
    def _worker_wait_timeout(tasks: list[Task]) -> int:
        task_timeouts = [
            int(task.payload.get("timeout_seconds", 30))
            for task in tasks
            if isinstance(task.payload.get("timeout_seconds", 30), int)
        ]
        return max([60, *task_timeouts]) + 30

