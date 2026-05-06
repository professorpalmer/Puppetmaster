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

        summary = self._render_summary(job.goal, artifacts, memories)
        self.store.write_summary(job_id, "stitched.md", summary)
        return summary

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
        goal: str,
        artifacts: list[Artifact],
        memories: list[MemoryRecord],
    ) -> str:
        grouped = group_by_type(artifacts)
        counts = Counter(str(artifact.type) for artifact in artifacts)

        lines = [
            "# Puppetmaster Stitched Summary",
            "",
            f"Goal: {goal}",
            "",
            "## Artifact Counts",
        ]
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

