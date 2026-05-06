from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from puppetmaster.models import Artifact, Job, JobStatus, Task, TaskStatus
from puppetmaster.stitcher import Stitcher
from puppetmaster.store import SwarmStore
from puppetmaster.workers import WorkerSpec, specs_for_roles


@dataclass(frozen=True)
class RunResult:
    job: Job
    artifacts: list[Artifact]
    summary: str
    summary_path: Path
    recovered_tasks: int = 0


class Orchestrator:
    def __init__(self, store: SwarmStore) -> None:
        self.store = store

    def run(
        self,
        goal: str,
        roles: list[str] | None = None,
        specs: list[WorkerSpec] | None = None,
        lease_seconds: int = 5,
    ) -> RunResult:
        specs = specs or specs_for_roles(roles)
        job = self.store.create_job(goal)
        self.store.update_job_status(job.id, JobStatus.RUNNING)
        tasks = self._create_tasks(job, specs)
        self._run_workers(job, tasks, lease_seconds=lease_seconds)
        artifacts = self.store.list_artifacts(job.id)

        self.store.update_job_status(job.id, JobStatus.STITCHING)
        summary = Stitcher(self.store).stitch(job.id)
        completed = self.store.update_job_status(job.id, JobStatus.COMPLETE)
        summary_path = self.store.job_dir(job.id) / "summaries" / "stitched.md"
        return RunResult(
            job=completed,
            artifacts=artifacts,
            summary=summary,
            summary_path=summary_path,
        )

    def run_crash_recovery_demo(
        self,
        goal: str,
        crash_role: str = "implement",
        roles: list[str] | None = None,
    ) -> RunResult:
        specs = specs_for_roles(roles)
        job = self.store.create_job(goal)
        self.store.update_job_status(job.id, JobStatus.RUNNING)
        tasks = self._create_tasks(job, specs)
        self._run_prerequisites(job, tasks, crash_role, lease_seconds=2)

        crash_process = self._spawn_worker(
            job.id,
            crash_role,
            lease_seconds=1,
            crash_after_claim=True,
        )
        crash_process.wait(timeout=10)
        time.sleep(1.2)
        recovered = self.store.recover_stale_tasks(job.id)

        self._run_workers(job, tasks, lease_seconds=2)
        artifacts = self.store.list_artifacts(job.id)
        self.store.update_job_status(job.id, JobStatus.STITCHING)
        summary = Stitcher(self.store).stitch(job.id)
        completed = self.store.update_job_status(job.id, JobStatus.COMPLETE)
        summary_path = self.store.job_dir(job.id) / "summaries" / "stitched.md"
        return RunResult(
            job=completed,
            artifacts=artifacts,
            summary=summary,
            summary_path=summary_path,
            recovered_tasks=len(recovered),
        )

    def _create_tasks(self, job: Job, specs: list[WorkerSpec]) -> list[Task]:
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
        for task in tasks:
            self.store.save_task(task)
        return tasks

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
    ) -> None:
        roles = sorted({task.role for task in tasks})
        processes = [
            self._spawn_worker(job.id, role, lease_seconds=lease_seconds)
            for role in roles
        ]
        for process in processes:
            process.wait(timeout=self._worker_wait_timeout(tasks))
            if process.returncode != 0:
                raise RuntimeError(f"worker process failed with exit code {process.returncode}")

        if self.store.has_incomplete_tasks(job.id):
            recovered = self.store.recover_stale_tasks(job.id)
            if recovered:
                self._run_workers(job, recovered, lease_seconds=lease_seconds)
            elif self.store.has_incomplete_tasks(job.id):
                raise RuntimeError("swarm exited with incomplete tasks")

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
        for process in processes:
            process.wait(timeout=self._worker_wait_timeout(dependencies))
            if process.returncode != 0:
                raise RuntimeError(
                    f"prerequisite worker failed with exit code {process.returncode}"
                )

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
        return subprocess.Popen(command)

    @staticmethod
    def _worker_wait_timeout(tasks: list[Task]) -> int:
        task_timeouts = [
            int(task.payload.get("timeout_seconds", 30))
            for task in tasks
            if isinstance(task.payload.get("timeout_seconds", 30), int)
        ]
        return max([60, *task_timeouts]) + 30

