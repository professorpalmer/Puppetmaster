from __future__ import annotations

from collections import Counter
from textwrap import indent
from typing import Optional

from puppetmaster.models import Artifact, ArtifactType, MemoryRecord
from puppetmaster.store import SwarmStore, group_by_type


class Stitcher:
    """Turns worker artifacts into promoted memory and a replayable summary."""

    def __init__(self, store: SwarmStore) -> None:
        self.store = store

    def stitch(self, job_id: str) -> str:
        job = self.store.get_job(job_id)
        artifacts = self.store.list_artifacts(job_id)
        memories = self._promote_memories(artifacts)
        for memory in memories:
            self.store.promote_memory(memory)

        summary = self._render_summary("Puppetmaster Stitched Summary", job.goal, artifacts, memories)
        self.store.write_summary(job_id, "stitched.md", summary)
        return summary

    def preview(self, job_id: str) -> str:
        """Render a live summary from current artifacts without promoting memory."""
        job = self.store.get_job(job_id)
        artifacts = self.store.list_artifacts(job_id)
        memories = self._promote_memories(artifacts)
        return self._render_summary("Puppetmaster Live Summary", job.goal, artifacts, memories)

    def _promote_memories(self, artifacts: list[Artifact]) -> list[MemoryRecord]:
        promoted: list[MemoryRecord] = []
        for artifact in artifacts:
            if artifact.confidence < 0.8:
                continue
            statement = self._statement_for(artifact)
            if not statement:
                continue
            promoted.append(
                MemoryRecord(
                    scope=self._scope_for(artifact),
                    statement=statement,
                    evidence=artifact.evidence,
                    source_artifacts=[artifact.id],
                    confidence=artifact.confidence,
                )
            )
        return promoted

    def _render_summary(
        self,
        title: str,
        goal: str,
        artifacts: list[Artifact],
        memories: list[MemoryRecord],
    ) -> str:
        grouped = group_by_type(artifacts)
        counts = Counter(str(artifact.type) for artifact in artifacts)

        lines = [
            f"# {title}",
            "",
            f"Goal: {goal}",
        ]

        alerts = self._collect_alerts(artifacts)
        if alerts:
            lines.extend(["", "## Alerts (action required)"])
            lines.extend(alerts)

        lines.extend(["", "## Artifact Counts"])
        for artifact_type, count in sorted(counts.items()):
            lines.append(f"- {artifact_type}: {count}")

        lines.extend(["", "## Promoted Memory"])
        for memory in memories:
            lines.append(f"- [{memory.scope}] {memory.statement}")

        lines.extend(["", "## Findings"])
        lines.extend(self._bullet_payloads(grouped.get(str(ArtifactType.FINDING), []), "claim"))

        lines.extend(["", "## Decisions"])
        lines.extend(
            self._bullet_payloads(grouped.get(str(ArtifactType.DECISION), []), "decision")
        )

        lines.extend(["", "## Risks"])
        lines.extend(self._bullet_payloads(grouped.get(str(ArtifactType.RISK), []), "risk"))

        lines.extend(["", "## Verification"])
        lines.extend(
            self._bullet_payloads(grouped.get(str(ArtifactType.VERIFICATION), []), "check")
        )

        lines.extend(["", "## Raw Artifact Rule", "Final synthesis used structured JSON artifacts only."])
        return "\n".join(lines) + "\n"

    # Failure classes that mean the worker never really ran (auth/billing/setup)
    # rather than "ran and found a problem". These must not hide inside a
    # low-confidence verification line — a degraded run should be obvious at a
    # glance, with a remediation pointer.
    _FAILURE_REMEDIATION = {
        "billing_or_quota": (
            "the provider account is out of credit/quota. Top up that account, "
            "switch to a subscription-billed login, or re-route to a funded "
            "adapter (e.g. set required_tags=['agent-loop'] or allow only "
            "plan-billed models)."
        ),
        "missing_api_key": "the adapter's API key env var is not set.",
        "missing_cli": "the adapter's CLI is not installed / not on PATH.",
        "sdk_not_installed": "the adapter SDK package is not installed.",
        "model_unavailable": "the requested model is invalid or not available to this account.",
        "permission_denied": "the adapter lacked permission to act (check permission_mode / auth).",
        "rate_limit": "the provider rate-limited the request; retry later or re-route.",
        "dirty_worktree": "the worktree was dirty; commit/stash or pass allow_dirty=true.",
    }

    def _collect_alerts(self, artifacts: list[Artifact]) -> list[str]:
        alerts: list[str] = []
        for artifact in artifacts:
            payload = artifact.payload or {}
            failure = payload.get("failure")
            result = payload.get("result")
            if not failure and result not in {"failed", "blocked"}:
                continue
            if not failure:
                continue
            adapter = payload.get("adapter") or self._adapter_from_evidence(artifact)
            hint = self._FAILURE_REMEDIATION.get(
                str(failure), "the worker could not complete; see the artifact for details."
            )
            alerts.append(
                f"- **{failure}** on `{adapter}` worker — {hint}"
            )
        return alerts

    @staticmethod
    def _adapter_from_evidence(artifact: Artifact) -> str:
        for item in artifact.evidence:
            if item.startswith("adapter:"):
                return item.split(":", 1)[1]
        return "unknown"

    @staticmethod
    def _bullet_payloads(artifacts: list[Artifact], key: str) -> list[str]:
        if not artifacts:
            return ["- None"]

        bullets = []
        for artifact in artifacts:
            headline = artifact.payload.get(key, artifact.payload)
            evidence = ", ".join(artifact.evidence) or "no evidence"
            bullets.append(
                f"- {headline}\n"
                f"{indent(f'confidence={artifact.confidence:.2f}; evidence={evidence}', '  ')}"
            )
        return bullets

    @staticmethod
    def _statement_for(artifact: Artifact) -> Optional[str]:
        keys = {
            ArtifactType.FINDING: "claim",
            ArtifactType.DECISION: "decision",
            ArtifactType.VERIFICATION: "check",
        }
        key = keys.get(artifact.type)
        if not key:
            return None
        value = artifact.payload.get(key)
        return str(value) if value else None

    @staticmethod
    def _scope_for(artifact: Artifact) -> str:
        if artifact.type == ArtifactType.FINDING:
            return "swarm.findings"
        if artifact.type == ArtifactType.DECISION:
            return "swarm.decisions"
        if artifact.type == ArtifactType.VERIFICATION:
            return "swarm.verification"
        return "swarm.general"

