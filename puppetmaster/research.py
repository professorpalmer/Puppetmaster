"""Durable Autoresearch lab — coordinate claim/run/publish experiments on SQLite state.

A *research lab* is a long-lived Puppetmaster job whose workers explore a
hypothesis space through an :class:`ExperimentHarness`, persist typed
FINDING / DECISION / VERIFICATION artifacts (tagged with
``payload.research_kind``), and verify keeps by re-running the harness.

This deliberately reuses existing :class:`~puppetmaster.models.ArtifactType`
values — no ``RESEARCH_*`` enum members — so the dashboard, bounds, and
stitcher stay unchanged. Research semantics live in ``payload.research_kind``.

Claim exclusivity is fingerprint-keyed: ``sha256(hypothesis|harness_id|config)``
plus a ``research-claim:{job_id}:{fingerprint}`` lock, a ``research-runner``
task, and the normal task lease (renew while running; release on publish).

GPU / nanochat-style trainers can plug in later by implementing
:class:`ExperimentHarness` (see the GpuHarnessAdapter note on that Protocol).
"""
from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Sequence

from puppetmaster.liveness import record_orchestrator_heartbeat
from puppetmaster.models import (
    Artifact,
    ArtifactType,
    Job,
    JobStatus,
    Task,
    TaskStatus,
    new_id,
    now_iso,
)
from puppetmaster.store import SwarmStore

# ---------------------------------------------------------------------------
# research_kind constants (payload discriminator; NOT ArtifactType members)
# ---------------------------------------------------------------------------

RESEARCH_KIND_RESULT = "result"
RESEARCH_KIND_INSIGHT = "insight"
RESEARCH_KIND_HYPOTHESIS = "hypothesis"
RESEARCH_KIND_BEST = "best"
RESEARCH_KIND_VERIFICATION = "verification"

RESEARCH_KINDS = frozenset(
    {
        RESEARCH_KIND_RESULT,
        RESEARCH_KIND_INSIGHT,
        RESEARCH_KIND_HYPOTHESIS,
        RESEARCH_KIND_BEST,
        RESEARCH_KIND_VERIFICATION,
    }
)

RESEARCH_RUNNER_ROLE = "research-runner"
DEFAULT_LAB_LABEL = "autoresearch-lab"
TOY_HARNESS_ID = "toy-compression"
DEFAULT_LEASE_SECONDS = 120
DEFAULT_CLAIM_LOCK_TTL = 360
SCORE_KEY = "bits_per_byte"


def canonical_config(config: Mapping[str, Any]) -> str:
    """Stable JSON form for fingerprinting (sorted keys, no whitespace)."""
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)


def experiment_fingerprint(
    hypothesis: str, harness_id: str, config: Mapping[str, Any]
) -> str:
    raw = "|".join((str(hypothesis), str(harness_id), canonical_config(config)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def claim_lock_name(job_id: str, fingerprint: str) -> str:
    return f"research-claim:{job_id}:{fingerprint}"


# ---------------------------------------------------------------------------
# Harness protocol + toy CPU microbench
# ---------------------------------------------------------------------------


class ExperimentHarness(Protocol):
    """Pluggable experiment runner. Lower ``bits_per_byte`` (or score) is better.

    **GpuHarnessAdapter (future stub — docstring only).** A GPU / nanochat
    trainer would implement this same Protocol: expose ``harness_id`` and
    ``run(config) -> metrics`` (including a comparable score such as
    bits-per-byte or perplexity). :class:`ResearchLab` claim / publish /
    verify / leaderboard stay unchanged. v1 does not ship a GPU runtime,
    Ensue API, or remote join.
    """

    harness_id: str

    def run(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute one trial; return metrics including a comparable score."""
        ...


class ToyCompressionHarness:
    """Deterministic CPU zlib / byte-entropy microbench.

    Compresses a seeded corpus at a zlib level from ``config`` and reports
    ``bits_per_byte`` (compressed_bits / raw_bytes). Lower is better.
    """

    harness_id: str = TOY_HARNESS_ID

    def run(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        level = int(config.get("level", 6))
        if level < 1 or level > 9:
            raise ValueError(f"zlib level must be 1..9, got {level}")
        seed = int(config.get("seed", 0))
        size = int(config.get("size", 4096))
        if size < 64:
            raise ValueError(f"size must be >= 64, got {size}")

        data = _deterministic_corpus(seed, size)
        compressed = zlib.compress(data, level)
        raw_bytes = len(data)
        compressed_bytes = len(compressed)
        bits_per_byte = (compressed_bytes * 8.0) / float(raw_bytes)
        entropy = _byte_entropy(data)
        return {
            SCORE_KEY: bits_per_byte,
            "compressed_bytes": compressed_bytes,
            "raw_bytes": raw_bytes,
            "entropy_bits_per_byte": entropy,
            "level": level,
            "seed": seed,
            "size": size,
            "lower_is_better": True,
            "harness_id": self.harness_id,
        }


def _deterministic_corpus(seed: int, size: int) -> bytes:
    """Expand a seed into compressible, host-stable bytes.

    Pure hash streams are near-entropy-max and hide zlib level differences.
    Mix a short repeating motif (seeded) with sparse hash noise so higher
    compression levels measurably reduce ``bits_per_byte``.
    """
    motif = hashlib.sha256(f"toy-compression-motif:{seed}".encode("utf-8")).digest()
    # Prefer low-cardinality runs so zlib has structure to exploit.
    motif = bytes(b % 16 for b in motif) * 4
    out = bytearray()
    noise = hashlib.sha256(f"toy-compression-noise:{seed}".encode("utf-8")).digest()
    while len(out) < size:
        out.extend(motif)
        # Sprinkle a little noise every block so configs aren't trivial.
        out.extend(noise[:8])
        noise = hashlib.sha256(noise).digest()
    return bytes(out[:size])


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    total = float(len(data))
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# ResearchLab coordinator
# ---------------------------------------------------------------------------


@dataclass
class ResearchLab:
    """Coordinate durable autoresearch verbs against a :class:`SwarmStore`."""

    store: SwarmStore
    worker_id: str = field(default_factory=lambda: new_id("research"))
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    _harnesses: MutableMapping[str, ExperimentHarness] = field(default_factory=dict)
    _active_task_id: Optional[str] = None
    _active_lock: Optional[str] = None

    def __post_init__(self) -> None:
        if TOY_HARNESS_ID not in self._harnesses:
            self.register_harness(ToyCompressionHarness())

    def register_harness(self, harness: ExperimentHarness) -> None:
        self._harnesses[str(harness.harness_id)] = harness

    def get_harness(self, harness_id: str) -> ExperimentHarness:
        harness = self._harnesses.get(harness_id)
        if harness is None:
            raise KeyError(f"unknown harness: {harness_id}")
        return harness

    # -- liveness -----------------------------------------------------------

    def _heartbeat(self, job_id: str, *, event: str, payload: Optional[dict] = None) -> None:
        record_orchestrator_heartbeat(self.store, job_id)
        body = dict(payload or {})
        body.setdefault("worker_id", self.worker_id)
        self.store.emit(job_id, event, body)

    # -- verbs --------------------------------------------------------------

    def init_lab(
        self,
        goal: str,
        *,
        label: str = DEFAULT_LAB_LABEL,
    ) -> Job:
        """Create a durable lab job and mark it running."""
        job = self.store.create_job(goal, label=label)
        job = self.store.update_job_status(job.id, JobStatus.RUNNING)
        record_orchestrator_heartbeat(self.store, job.id, started=True)
        self.store.emit(
            job.id,
            "research.lab_init",
            {"label": label, "goal": goal, "worker_id": self.worker_id},
        )
        return job

    def announce(self, job_id: str, message: str, *, task_id: Optional[str] = None) -> Artifact:
        """Publish a lab announcement as a DECISION (research_kind=insight)."""
        self._heartbeat(job_id, event="research.announce", payload={"message": message})
        tid = task_id or self._lab_task_id(job_id, "announce")
        artifact = Artifact(
            job_id=job_id,
            task_id=tid,
            type=ArtifactType.DECISION,
            created_by=self.worker_id,
            payload={
                "decision": message,
                "why": "lab announcement",
                "research_kind": RESEARCH_KIND_INSIGHT,
            },
            confidence=1.0,
            evidence=[f"announce:{now_iso()}"],
        )
        self.store.save_artifact(artifact)
        return artifact

    def claim_experiment(
        self,
        job_id: str,
        hypothesis: str,
        harness_id: str,
        config: Mapping[str, Any],
        *,
        lease_seconds: Optional[int] = None,
    ) -> Optional[Task]:
        """Claim exclusive rights to run ``(hypothesis, harness, config)``.

        Returns the claimed task, or ``None`` when another worker holds a live
        claim on the same fingerprint.
        """
        lease = int(lease_seconds if lease_seconds is not None else self.lease_seconds)
        fingerprint = experiment_fingerprint(hypothesis, harness_id, config)
        lock_name = claim_lock_name(job_id, fingerprint)
        self._heartbeat(
            job_id,
            event="research.claim_attempt",
            payload={"fingerprint": fingerprint, "harness_id": harness_id},
        )

        if not self.store.acquire_lock(
            lock_name, self.worker_id, ttl_seconds=max(lease * 3, DEFAULT_CLAIM_LOCK_TTL)
        ):
            return None

        try:
            existing = self._find_open_task(job_id, fingerprint)
            if existing is not None:
                if (
                    existing.status == TaskStatus.RUNNING
                    and not SwarmStore.is_task_stale(existing)
                    and existing.lease_owner
                    and existing.lease_owner != self.worker_id
                ):
                    self.store.release_lock(lock_name, owner=self.worker_id)
                    return None
                claimed = self.store.claim_task(
                    existing.id, self.worker_id, lease_seconds=lease
                )
                if claimed is None:
                    self.store.release_lock(lock_name, owner=self.worker_id)
                    return None
                self._active_task_id = claimed.id
                self._active_lock = lock_name
                self.store.emit(
                    job_id,
                    "research.claimed",
                    {
                        "task_id": claimed.id,
                        "fingerprint": fingerprint,
                        "reclaim": True,
                    },
                )
                return claimed

            # Dedup: refuse a second open task if one slipped past the lock.
            if self._find_open_task(job_id, fingerprint) is not None:
                self.store.release_lock(lock_name, owner=self.worker_id)
                return None

            config_dict = json.loads(canonical_config(config))
            task = Task(
                job_id=job_id,
                role=RESEARCH_RUNNER_ROLE,
                instruction=str(hypothesis),
                adapter="local",
                payload={
                    "fingerprint": fingerprint,
                    "hypothesis": str(hypothesis),
                    "harness_id": str(harness_id),
                    "config": config_dict,
                    "research": True,
                },
            )
            self.store.save_task(task)
            claimed = self.store.claim_task(task.id, self.worker_id, lease_seconds=lease)
            if claimed is None:
                self.store.release_lock(lock_name, owner=self.worker_id)
                return None
            self._active_task_id = claimed.id
            self._active_lock = lock_name
            self.store.emit(
                job_id,
                "research.claimed",
                {"task_id": claimed.id, "fingerprint": fingerprint, "reclaim": False},
            )
            return claimed
        except Exception:
            self.store.release_lock(lock_name, owner=self.worker_id)
            raise

    def run_claimed(
        self,
        job_id: str,
        *,
        task_id: Optional[str] = None,
        harness: Optional[ExperimentHarness] = None,
    ) -> Mapping[str, Any]:
        """Renew the lease and run the harness for the active (or given) claim."""
        task = self._require_claimed_task(job_id, task_id)
        self._renew(task)
        self._heartbeat(
            job_id,
            event="research.run",
            payload={"task_id": task.id, "harness_id": task.payload.get("harness_id")},
        )
        hid = str(task.payload.get("harness_id") or "")
        runner = harness or self.get_harness(hid)
        config = task.payload.get("config") or {}
        if not isinstance(config, Mapping):
            raise ValueError("claimed task payload.config must be a mapping")
        metrics = dict(runner.run(config))
        self._renew(task)
        return metrics

    def publish_result(
        self,
        job_id: str,
        metrics: Mapping[str, Any],
        *,
        task_id: Optional[str] = None,
        keep: bool = True,
        evidence: Optional[Sequence[str]] = None,
    ) -> Artifact:
        """Persist a result FINDING, complete the claim task, release the lock."""
        task = self._require_claimed_task(job_id, task_id)
        self._renew(task)
        hypothesis = str(task.payload.get("hypothesis") or task.instruction)
        harness_id = str(task.payload.get("harness_id") or "")
        config = task.payload.get("config") or {}
        fingerprint = str(
            task.payload.get("fingerprint")
            or experiment_fingerprint(hypothesis, harness_id, config if isinstance(config, Mapping) else {})
        )
        score = metrics.get(SCORE_KEY)
        claim = (
            f"{hypothesis} → {SCORE_KEY}={score}"
            if score is not None
            else f"{hypothesis} → metrics published"
        )
        artifact = Artifact(
            job_id=job_id,
            task_id=task.id,
            type=ArtifactType.FINDING,
            created_by=self.worker_id,
            payload={
                "claim": claim,
                "research_kind": RESEARCH_KIND_RESULT,
                "hypothesis": hypothesis,
                "harness_id": harness_id,
                "config": config,
                "fingerprint": fingerprint,
                "metrics": dict(metrics),
                "keep": bool(keep),
                SCORE_KEY: score,
            },
            confidence=0.9 if keep else 0.5,
            evidence=list(evidence) if evidence else [f"harness:{harness_id}", f"fp:{fingerprint[:12]}"],
        )
        self.store.save_artifact(artifact)
        completed = self.store.update_task_status(
            task, TaskStatus.COMPLETE, worker_id=self.worker_id, lease_id=task.lease_id
        )
        lock_name = self._active_lock or claim_lock_name(job_id, fingerprint)
        self.store.release_lock(lock_name, owner=self.worker_id)
        if self._active_task_id == task.id:
            self._active_task_id = None
            self._active_lock = None
        self._heartbeat(
            job_id,
            event="research.published",
            payload={
                "task_id": completed.id,
                "artifact_id": artifact.id,
                "fingerprint": fingerprint,
                "keep": bool(keep),
                SCORE_KEY: score,
            },
        )
        return artifact

    def post_insight(
        self,
        job_id: str,
        insight: str,
        *,
        why: str = "",
        task_id: Optional[str] = None,
        evidence: Optional[Sequence[str]] = None,
    ) -> Artifact:
        self._heartbeat(job_id, event="research.insight", payload={"insight": insight})
        tid = task_id or self._lab_task_id(job_id, "insight")
        artifact = Artifact(
            job_id=job_id,
            task_id=tid,
            type=ArtifactType.FINDING,
            created_by=self.worker_id,
            payload={
                "claim": insight,
                "research_kind": RESEARCH_KIND_INSIGHT,
                "why": why or "lab insight",
            },
            confidence=0.8,
            evidence=list(evidence) if evidence else [f"insight:{now_iso()}"],
        )
        self.store.save_artifact(artifact)
        return artifact

    def publish_hypothesis(
        self,
        job_id: str,
        hypothesis: str,
        *,
        why: str = "",
        config: Optional[Mapping[str, Any]] = None,
        harness_id: Optional[str] = None,
        task_id: Optional[str] = None,
        evidence: Optional[Sequence[str]] = None,
    ) -> Artifact:
        self._heartbeat(
            job_id, event="research.hypothesis", payload={"hypothesis": hypothesis}
        )
        tid = task_id or self._lab_task_id(job_id, "hypothesis")
        payload: dict[str, Any] = {
            "decision": hypothesis,
            "why": why or "candidate hypothesis for exploration",
            "research_kind": RESEARCH_KIND_HYPOTHESIS,
        }
        if config is not None:
            payload["config"] = json.loads(canonical_config(config))
        if harness_id is not None:
            payload["harness_id"] = harness_id
        artifact = Artifact(
            job_id=job_id,
            task_id=tid,
            type=ArtifactType.DECISION,
            created_by=self.worker_id,
            payload=payload,
            confidence=0.7,
            evidence=list(evidence) if evidence else [f"hypothesis:{now_iso()}"],
        )
        self.store.save_artifact(artifact)
        return artifact

    def pull_best(
        self,
        job_id: str,
        *,
        persist: bool = True,
        task_id: Optional[str] = None,
    ) -> Optional[Artifact]:
        """Return the best keep result (lowest bits_per_byte); optionally persist."""
        self._heartbeat(job_id, event="research.pull_best", payload={})
        ranked = self.leaderboard(job_id)
        if not ranked:
            return None
        best = ranked[0]
        if not persist:
            return best
        tid = task_id or best.task_id or self._lab_task_id(job_id, "best")
        score = best.payload.get(SCORE_KEY)
        decision = Artifact(
            job_id=job_id,
            task_id=tid,
            type=ArtifactType.DECISION,
            created_by=self.worker_id,
            payload={
                "decision": f"best keep: {best.payload.get('hypothesis')}",
                "why": f"lowest {SCORE_KEY}={score} among kept results",
                "research_kind": RESEARCH_KIND_BEST,
                "source_artifact_id": best.id,
                "metrics": best.payload.get("metrics") or {},
                SCORE_KEY: score,
                "fingerprint": best.payload.get("fingerprint"),
                "harness_id": best.payload.get("harness_id"),
                "config": best.payload.get("config"),
            },
            confidence=0.95,
            evidence=[f"artifact:{best.id}", f"{SCORE_KEY}:{score}"],
        )
        self.store.save_artifact(decision)
        try:
            self.store.record_derived_from(job_id, best.id, decision.id)
        except Exception:
            pass
        return decision

    def verify_claim(
        self,
        job_id: str,
        artifact_id: str,
        *,
        harness: Optional[ExperimentHarness] = None,
        tolerance: float = 1e-9,
        task_id: Optional[str] = None,
    ) -> Artifact:
        """Re-run the harness for a result artifact; emit VERIFICATION + edge."""
        source = self._get_artifact(job_id, artifact_id)
        if source.payload.get("research_kind") not in {
            RESEARCH_KIND_RESULT,
            RESEARCH_KIND_BEST,
        }:
            raise ValueError(
                f"verify_claim expects a result/best artifact, got "
                f"research_kind={source.payload.get('research_kind')!r}"
            )
        config = source.payload.get("config") or {}
        if not isinstance(config, Mapping):
            raise ValueError("source artifact config must be a mapping")
        harness_id = str(
            source.payload.get("harness_id") or ToyCompressionHarness.harness_id
        )
        runner = harness or self.get_harness(harness_id)
        self._heartbeat(
            job_id,
            event="research.verify",
            payload={"artifact_id": artifact_id, "harness_id": harness_id},
        )
        metrics = dict(runner.run(config))
        expected = source.payload.get(SCORE_KEY)
        if expected is None and isinstance(source.payload.get("metrics"), Mapping):
            expected = source.payload["metrics"].get(SCORE_KEY)
        actual = metrics.get(SCORE_KEY)
        passed = (
            expected is not None
            and actual is not None
            and abs(float(actual) - float(expected)) <= float(tolerance)
        )
        tid = task_id or self._lab_task_id(job_id, "verify")
        verification = Artifact(
            job_id=job_id,
            task_id=tid,
            type=ArtifactType.VERIFICATION,
            created_by=self.worker_id,
            payload={
                "check": f"re-run {harness_id} for {artifact_id}",
                "result": "pass" if passed else "fail",
                "research_kind": RESEARCH_KIND_VERIFICATION,
                "source_artifact_id": artifact_id,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "metrics": metrics,
                "passed": passed,
                "harness_id": harness_id,
                "config": json.loads(canonical_config(config)),
            },
            confidence=1.0 if passed else 0.3,
            evidence=[
                f"source:{artifact_id}",
                f"expected:{expected}",
                f"actual:{actual}",
            ],
        )
        self.store.save_artifact(verification)
        try:
            self.store.record_derived_from(job_id, artifact_id, verification.id)
        except Exception:
            # File/SQLite stores always support edges; ignore if source vanished.
            pass
        return verification

    def status(self, job_id: str) -> dict[str, Any]:
        self._heartbeat(job_id, event="research.status", payload={})
        job = self.store.get_job(job_id)
        tasks = self.store.list_tasks(job_id)
        artifacts = self.store.list_artifacts(job_id)
        by_kind: dict[str, int] = {}
        for art in artifacts:
            kind = str(art.payload.get("research_kind") or "")
            if kind:
                by_kind[kind] = by_kind.get(kind, 0) + 1
        open_claims = [
            t
            for t in tasks
            if t.role == RESEARCH_RUNNER_ROLE
            and t.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED}
        ]
        return {
            "job_id": job.id,
            "label": job.label,
            "goal": job.goal,
            "status": str(job.status),
            "tasks": len(tasks),
            "open_claims": len(open_claims),
            "artifacts": len(artifacts),
            "by_research_kind": by_kind,
            "leaderboard_size": len(self.leaderboard(job_id)),
        }

    def leaderboard(self, job_id: str) -> list[Artifact]:
        """Kept result artifacts ordered by ascending bits_per_byte."""
        results: list[Artifact] = []
        for art in self.store.list_artifacts(job_id):
            if art.payload.get("research_kind") != RESEARCH_KIND_RESULT:
                continue
            if art.payload.get("keep") is False:
                continue
            score = art.payload.get(SCORE_KEY)
            if score is None and isinstance(art.payload.get("metrics"), Mapping):
                score = art.payload["metrics"].get(SCORE_KEY)
            if score is None:
                continue
            results.append(art)
        results.sort(
            key=lambda a: float(
                a.payload.get(SCORE_KEY)
                if a.payload.get(SCORE_KEY) is not None
                else a.payload.get("metrics", {}).get(SCORE_KEY, 1e9)
            )
        )
        return results

    def write_brief(
        self,
        job_id: str,
        path: Path,
        *,
        title: Optional[str] = None,
    ) -> Path:
        """Write a markdown brief from durable lab artifacts (zero-token recall)."""
        self._heartbeat(job_id, event="research.write_brief", payload={"path": str(path)})
        job = self.store.get_job(job_id)
        board = self.leaderboard(job_id)
        artifacts = self.store.list_artifacts(job_id)
        insights = [
            a
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_INSIGHT
        ]
        hypotheses = [
            a
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_HYPOTHESIS
        ]
        verifications = [
            a
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_VERIFICATION
        ]
        bests = [
            a
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_BEST
        ]
        lines = [
            f"# {title or 'Durable Autoresearch Brief'}",
            "",
            f"**Job:** `{job.id}`  ",
            f"**Label:** `{job.label or ''}`  ",
            f"**Goal:** {job.goal}",
            "",
            "## Thesis",
            "",
            "Collaborative agent research is bound by **state architecture**, "
            "not by how many agents join a shared chat. When hundreds of "
            "workers explore a search space, the scarce resource is durable, "
            "queryable, lease-guarded memory — claims that expire, results "
            "you can re-read at zero model tokens, and verification that "
            "re-runs the oracle instead of trusting a transcript.",
            "",
            "This lab is Puppetmaster's answer to that gap: SQLite-backed "
            "jobs + structured artifacts + fingerprint leases + "
            "explore→verify loops. The toy metric below is a stand-in "
            "(CPU zlib bits-per-byte). The protocol is the product.",
            "",
            "## Exploration + verification loop",
            "",
            "Protocol: **THINK** (read durable artifacts) → **CLAIM** "
            "(fingerprint lock + task lease) → **RUN** (harness) → "
            "**PUBLISH** (result / insight / hypothesis) → **VERIFY** "
            "(independent re-run; DERIVED_FROM provenance).",
            "",
            f"- Hypotheses posted: **{len(hypotheses)}**",
            f"- Insights: **{len(insights)}**",
            f"- Kept results on the leaderboard: **{len(board)}**",
            f"- Verifications: **{len(verifications)}** "
            f"(pass={sum(1 for v in verifications if v.payload.get('passed') or v.payload.get('result') == 'pass')})",
            "",
            "## Leaderboard (lower bits_per_byte is better)",
            "",
        ]
        if not board:
            lines.append("_No kept results yet._")
        else:
            lines.append("| Rank | Score | Hypothesis | Config | Artifact |")
            lines.append("| --- | --- | --- | --- | --- |")
            for index, art in enumerate(board, start=1):
                score = art.payload.get(SCORE_KEY)
                hyp = str(art.payload.get("hypothesis") or "").replace("|", "/")
                cfg = canonical_config(art.payload.get("config") or {})
                lines.append(
                    f"| {index} | `{score}` | {hyp} | `{cfg}` | `{art.id}` |"
                )
        lines.extend(["", "## Best keep", ""])
        if bests:
            b = bests[-1]
            lines.append(
                f"- {b.payload.get('decision')} "
                f"(`{SCORE_KEY}={b.payload.get(SCORE_KEY)}`, artifact `{b.id}`)"
            )
        elif board:
            b = board[0]
            lines.append(
                f"- {b.payload.get('hypothesis')} "
                f"(`{SCORE_KEY}={b.payload.get(SCORE_KEY)}`, artifact `{b.id}`)"
            )
        else:
            lines.append("_None yet._")
        lines.extend(
            [
                "",
                "## Extending to GPU / nanochat / autoresearch@home",
                "",
                "v1 ships `ToyCompressionHarness` (CPU zlib) so the protocol "
                "is runnable on Cursor-plan machines with no GPU. A nanochat "
                "/ train.py worker implements the same `ExperimentHarness` "
                "Protocol (`harness_id` + `run(config) -> metrics`). Claim "
                "fingerprints, leases, leaderboard ordering, and "
                "verify-by-rerun stay identical — only the oracle changes.",
                "",
                "Run locally: `python -m puppetmaster research demo`",
                "",
                f"_Generated at {now_iso()} from durable artifacts "
                f"(job `{job_id}`)._ ",
                "",
            ]
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        self.store.write_summary(job_id, "research-brief.md", path.read_text(encoding="utf-8"))
        return path

    def run_demo(
        self,
        *,
        goal: Optional[str] = None,
        brief_path: Optional[Path] = None,
        label: str = DEFAULT_LAB_LABEL,
    ) -> dict[str, Any]:
        """End-to-end toy lab: init → announce → 3 claim/run/publish → verify → brief."""
        job = self.init_lab(
            goal
            or (
                "Durable Autoresearch demo: explore zlib compression configs "
                "with ToyCompressionHarness and verify keeps from SQLite state."
            ),
            label=label,
        )
        self.announce(
            job.id,
            "Autoresearch lab open — exploring ToyCompressionHarness configs.",
        )
        self.publish_hypothesis(
            job.id,
            "Higher zlib levels reduce bits_per_byte on a seeded corpus",
            why="zlib level trades CPU for ratio; measure deterministically",
            harness_id=TOY_HARNESS_ID,
        )
        self.post_insight(
            job.id,
            "Use claim fingerprints so two workers never double-run the same config.",
        )

        trials = [
            ("level-1 explore", {"level": 1, "seed": 7, "size": 4096}),
            ("level-6 explore", {"level": 6, "seed": 7, "size": 4096}),
            ("level-9 explore", {"level": 9, "seed": 7, "size": 4096}),
        ]
        published: list[Artifact] = []
        for hypothesis, config in trials:
            claimed = self.claim_experiment(
                job.id, hypothesis, TOY_HARNESS_ID, config
            )
            if claimed is None:
                raise RuntimeError(f"demo failed to claim {hypothesis!r}")
            metrics = self.run_claimed(job.id, task_id=claimed.id)
            art = self.publish_result(job.id, metrics, task_id=claimed.id, keep=True)
            published.append(art)

        verifications: list[Artifact] = []
        for art in published:
            verifications.append(self.verify_claim(job.id, art.id))

        best = self.pull_best(job.id)
        board = self.leaderboard(job.id)
        target = Path(brief_path) if brief_path is not None else (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "autoresearch-durable-brief.md"
        )
        self.write_brief(job.id, target, title="Durable Autoresearch Demo Brief")
        self.store.update_job_status(job.id, JobStatus.COMPLETE)
        return {
            "job_id": job.id,
            "results": [a.id for a in published],
            "verifications": [v.id for v in verifications],
            "best_artifact_id": best.id if best else None,
            "leaderboard": [
                {
                    "artifact_id": a.id,
                    SCORE_KEY: a.payload.get(SCORE_KEY),
                    "hypothesis": a.payload.get("hypothesis"),
                    "config": a.payload.get("config"),
                }
                for a in board
            ],
            "brief_path": str(target),
            "status": self.status(job.id),
        }

    # -- internals ----------------------------------------------------------

    def _lab_task_id(self, job_id: str, purpose: str) -> str:
        """Ensure a lightweight lab bookkeeping task exists; return its id."""
        role = f"research-{purpose}"
        for task in self.store.list_tasks(job_id):
            if task.role == role:
                return task.id
        task = Task(
            job_id=job_id,
            role=role,
            instruction=f"Research lab bookkeeping: {purpose}",
            adapter="local",
            status=TaskStatus.COMPLETE,
            payload={"research": True, "bookkeeping": purpose},
            completed_at=now_iso(),
        )
        self.store.save_task(task)
        return task.id

    def _find_open_task(self, job_id: str, fingerprint: str) -> Optional[Task]:
        open_statuses = {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED}
        for task in self.store.list_tasks(job_id):
            if task.role != RESEARCH_RUNNER_ROLE:
                continue
            if task.status not in open_statuses:
                continue
            if str(task.payload.get("fingerprint") or "") == fingerprint:
                return task
        return None

    def _require_claimed_task(
        self, job_id: str, task_id: Optional[str]
    ) -> Task:
        tid = task_id or self._active_task_id
        if not tid:
            raise ValueError("no active research claim; pass task_id or claim first")
        task = self.store.get_task_by_id(tid)
        if task.job_id != job_id:
            raise ValueError(f"task {tid} does not belong to job {job_id}")
        if task.role != RESEARCH_RUNNER_ROLE:
            raise ValueError(f"task {tid} is not a research-runner claim")
        return task

    def _renew(self, task: Task) -> Optional[Task]:
        return self.store.renew_task_lease(
            task.id,
            self.worker_id,
            lease_seconds=self.lease_seconds,
            lease_id=task.lease_id,
        )

    def _get_artifact(self, job_id: str, artifact_id: str) -> Artifact:
        for art in self.store.list_artifacts(job_id):
            if art.id == artifact_id:
                return art
        raise FileNotFoundError(f"artifact not found: {artifact_id} in job {job_id}")


def analyze_swarm_from_artifacts(store: SwarmStore, job_id: str) -> dict[str, Any]:
    """Zero-token 'think' view: summarize durable research artifacts (no LLM)."""
    lab = ResearchLab(store=store)
    status = lab.status(job_id)
    board = lab.leaderboard(job_id)
    artifacts = store.list_artifacts(job_id)
    return {
        "status": status,
        "insights": [
            a.payload.get("claim") or a.payload.get("decision")
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_INSIGHT
        ],
        "hypotheses": [
            a.payload.get("decision")
            for a in artifacts
            if a.payload.get("research_kind") == RESEARCH_KIND_HYPOTHESIS
        ],
        "leaderboard": [
            {
                "artifact_id": a.id,
                SCORE_KEY: a.payload.get(SCORE_KEY),
                "hypothesis": a.payload.get("hypothesis"),
            }
            for a in board
        ],
    }
