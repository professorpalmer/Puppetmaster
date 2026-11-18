from __future__ import annotations

import unittest

from puppetmaster.models import Artifact, ArtifactType, Job
from puppetmaster.savings import collect_routing_records


class _Store:
    def __init__(self, jobs, artifacts):
        self._jobs = jobs
        self._artifacts = artifacts

    def list_jobs(self):
        return list(self._jobs)

    def list_artifacts(self, job_id):
        return [a for a in self._artifacts if a.job_id == job_id]


def _job(job_id):
    return Job(goal="g", id=job_id, created_at="2026-01-01T00:00:00+00:00")


def _routing(job_id, task_id, created_by, n):
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.ROUTING,
        created_by=created_by,
        payload={
            "model_id": "m",
            "adapter": "a",
            "policy": "balanced",
            "estimated_cost_usd": 0.0,
            "baseline_cost_usd": 1.0,
            "attempt": n,
        },
        confidence=0.9,
        evidence=["x"],
        id="artifact_%s_%s_%s_%d" % (job_id, task_id, created_by, n),
        created_at="2026-01-01T00:00:00+00:00",
    )


def _heal(kind, arts, jobs=None):
    jobs = jobs or [_job("job_1")]
    _records, _n, heal = collect_routing_records([_Store(jobs, arts)])
    return heal.fallbacks if kind == "router-fallback" else heal.escalations


class SelfHealCountsTasksTests(unittest.TestCase):
    def _twice_one_task(self, kind):
        arts = [
            _routing("job_1", "task_1", "router", 0),
            _routing("job_1", "task_1", kind, 1),
            _routing("job_1", "task_1", kind, 2),
        ]
        return _heal(kind, arts)

    def _two_tasks(self, kind):
        arts = [
            _routing("job_1", "task_1", "router", 0),
            _routing("job_1", "task_2", "router", 0),
            _routing("job_1", "task_1", kind, 1),
            _routing("job_1", "task_2", kind, 1),
        ]
        return _heal(kind, arts)

    def test_fallback_twice_on_one_task_counts_once(self):
        self.assertEqual(self._twice_one_task("router-fallback"), 1)

    def test_fallback_on_two_tasks_counts_two(self):
        self.assertEqual(self._two_tasks("router-fallback"), 2)

    def test_escalation_twice_on_one_task_counts_once(self):
        self.assertEqual(self._twice_one_task("router-escalation"), 1)

    def test_escalation_on_two_tasks_counts_two(self):
        self.assertEqual(self._two_tasks("router-escalation"), 2)

    def test_same_task_id_in_two_jobs_counts_two(self):
        jobs = [_job("job_1"), _job("job_2")]
        arts = [
            _routing("job_1", "task_1", "router-fallback", 1),
            _routing("job_2", "task_1", "router-fallback", 1),
        ]
        self.assertEqual(_heal("router-fallback", arts, jobs), 2)

    def test_fallback_and_escalation_on_one_task_both_count(self):
        arts = [
            _routing("job_1", "task_1", "router-fallback", 1),
            _routing("job_1", "task_1", "router-escalation", 2),
        ]
        _records, _n, heal = collect_routing_records(
            [_Store([_job("job_1")], arts)]
        )
        self.assertEqual((heal.fallbacks, heal.escalations), (1, 1))


if __name__ == "__main__":
    unittest.main()
