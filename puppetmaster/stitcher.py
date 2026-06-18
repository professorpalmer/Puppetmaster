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
        self.store.promote_memories(memories)

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
        lines.extend(
            self._bullet_payloads(
                grouped.get(str(ArtifactType.FINDING), []), "claim", dedupe=True
            )
        )

        lines.extend(["", "## Decisions"])
        lines.extend(
            self._bullet_payloads(
                grouped.get(str(ArtifactType.DECISION), []), "decision", dedupe=True
            )
        )

        lines.extend(["", "## Risks"])
        lines.extend(
            self._bullet_payloads(
                grouped.get(str(ArtifactType.RISK), []), "risk", dedupe=True
            )
        )

        lines.extend(["", "## Verification"])
        # Verification lines are per-worker run status — each is meaningful on its
        # own, so they are never collapsed.
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
        "preflight_blocked": (
            "the adapter was blocked before dispatch (auth/billing/model check). "
            "Run `python -m puppetmaster preflight <adapter>` for the reason, fund/"
            "re-auth the platform, or let auto_route pick a plan-billed adapter."
        ),
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
    def _bullet_payloads(
        artifacts: list[Artifact], key: str, *, dedupe: bool = False
    ) -> list[str]:
        if not artifacts:
            return ["- None"]

        if dedupe:
            clusters = Stitcher._dedup_clusters(artifacts, key)
        else:
            clusters = [(artifact, [artifact]) for artifact in artifacts]

        bullets = []
        for representative, members in clusters:
            headline = representative.payload.get(key, representative.payload)
            evidence = ", ".join(representative.evidence) or "no evidence"
            confidence = max(member.confidence for member in members)
            suffix = (
                f" (reported by {len(members)} workers)" if len(members) > 1 else ""
            )
            bullets.append(
                f"- {headline}{suffix}\n"
                f"{indent(f'confidence={confidence:.2f}; evidence={evidence}', '  ')}"
            )
        return bullets

    @staticmethod
    def _dedup_clusters(
        artifacts: list[Artifact], key: str
    ) -> list[tuple[Artifact, list[Artifact]]]:
        """Collapse near-identical claims so N workers finding the same thing read
        as one bullet ("reported by N workers") instead of N near-duplicates.

        Clustering is order-preserving and greedy: each artifact joins the first
        existing cluster whose claim is similar (exact, substring-contained, or
        high token overlap), else it seeds a new one. The most detailed /
        highest-confidence claim becomes the representative. Artifacts without a
        string claim are never merged.
        """
        clusters: list[list] = []  # [representative, members, normalized_claim]
        for artifact in artifacts:
            headline = artifact.payload.get(key)
            normalized = (
                Stitcher._normalize_claim(headline)
                if isinstance(headline, str)
                else ""
            )
            if not normalized:
                clusters.append([artifact, [artifact], ""])
                continue
            placed = False
            for cluster in clusters:
                if cluster[2] and Stitcher._claims_similar(normalized, cluster[2]):
                    cluster[1].append(artifact)
                    if Stitcher._is_better_representative(artifact, cluster[0]):
                        cluster[0] = artifact
                        cluster[2] = normalized
                    placed = True
                    break
            if not placed:
                clusters.append([artifact, [artifact], normalized])
        return [(cluster[0], cluster[1]) for cluster in clusters]

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        return " ".join(claim.lower().split()).strip(" .;:!?-\"'")

    @staticmethod
    def _claims_similar(left: str, right: str) -> bool:
        if left == right or left in right or right in left:
            return True
        left_tokens, right_tokens = set(left.split()), set(right.split())
        if not left_tokens or not right_tokens:
            return False
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return bool(union) and intersection / union >= 0.8

    @staticmethod
    def _is_better_representative(candidate: Artifact, current: Artifact) -> bool:
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return len(str(candidate.payload)) > len(str(current.payload))

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

