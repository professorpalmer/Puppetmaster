from __future__ import annotations

import re
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
        "not_authenticated": "the adapter is not authenticated (API key, OAuth, or login required).",
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
        cluster it's similar to, else seeds a new one. The most
        detailed/highest-confidence claim becomes the representative.

        Similarity is *evidence-locus-anchored* rather than pure token overlap,
        because different workers paraphrase the same bug in lexically diverse
        words ("multiply uses an O(n) loop" vs "multiplication is implemented
        inefficiently") that a high token gate misses — but they cite the same
        files. So two claims merge when they cite a shared code locus AND clear a
        low token gate (the gate keeps genuinely distinct findings at the same
        file — e.g. a KeyError vs a missing-division bug both in cli.py — apart),
        OR when their wording overlaps strongly on its own. Exact and
        substring-contained claims always merge. Claims with neither a string
        body nor anything to compare are never merged.
        """
        clusters: list[list] = []  # [representative, members, member_keys]
        for artifact in artifacts:
            headline = artifact.payload.get(key)
            normalized = (
                Stitcher._normalize_claim(headline)
                if isinstance(headline, str)
                else ""
            )
            loci = Stitcher._evidence_loci(artifact)
            if not normalized:
                clusters.append([artifact, [artifact], [(normalized, loci)]])
                continue
            placed = False
            for cluster in clusters:
                # Match against ANY member, not just the representative — workers
                # paraphrase the same bug in chains (A≈B, B≈C, A≉C), so anchoring
                # only on the rep would strand the far end of the chain.
                if any(
                    member_norm
                    and Stitcher._claims_similar(
                        normalized, member_norm, loci, member_loci
                    )
                    for member_norm, member_loci in cluster[2]
                ):
                    cluster[1].append(artifact)
                    cluster[2].append((normalized, loci))
                    if Stitcher._is_better_representative(artifact, cluster[0]):
                        cluster[0] = artifact
                    placed = True
                    break
            if not placed:
                clusters.append([artifact, [artifact], [(normalized, loci)]])
        return [(cluster[0], cluster[1]) for cluster in clusters]

    # When two claims already share a code locus, merging needs only a couple of
    # shared *content* words (stopwords excluded). A shared file is a strong
    # prior, and content-token overlap discriminates paraphrases of one bug ("…
    # repeated addition …") from genuinely distinct bugs at the same file (a
    # KeyError vs a missing-division bug in cli.py share zero content words) far
    # better than a token-ratio gate, which paraphrases routinely fall under.
    _LOCUS_GATED_MIN_SHARED_CONTENT = 2
    # Token overlap needed to merge claims with NO shared locus. High on purpose:
    # without a citation anchor, only strongly-overlapping wording is safe.
    _PURE_TOKEN_SIMILARITY = 0.6
    # Grammatical / low-signal words ignored when counting shared content.
    _STOPWORDS = frozenset(
        {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "to", "of", "in", "on", "for", "and", "or", "but", "with", "without",
            "that", "which", "this", "these", "those", "it", "its", "as", "at",
            "by", "from", "into", "than", "then", "so", "such", "not", "no",
            "can", "may", "might", "will", "would", "should", "could", "using",
            "use", "uses", "used", "when", "if", "there", "their", "they", "has",
            "have", "had", "does", "do", "due",
        }
    )

    # Evidence tags that are not code loci (provenance/status, not file:line).
    _NON_LOCUS_PREFIXES = frozenset(
        {
            "adapter",
            "context",
            "result",
            "status",
            "mode",
            "base",
            "exit",
            "node",
            "retry",
            "check",
        }
    )
    # A path-like token with a short extension, optionally with a :line or
    # :line-range suffix (e.g. arithmetic.py, cli.py:5, parser.py:3-4).
    _LOCUS_RE = re.compile(r"\b([\w./\\-]+\.[A-Za-z][A-Za-z0-9]{0,9})(?::\d+(?:-\d+)?)?\b")

    @staticmethod
    def _evidence_loci(artifact: Artifact) -> frozenset:
        """Code loci (lowercased file paths) cited in an artifact's evidence.

        Only the structured ``evidence`` list is scanned — it's where the
        contract puts ``file.ext:line`` citations — so prose like "e.g." in a
        claim can't masquerade as a locus.
        """
        loci: set[str] = set()
        for item in artifact.evidence or []:
            head = item.split(":", 1)[0].strip().lower()
            if "." not in head and head in Stitcher._NON_LOCUS_PREFIXES:
                continue
            for match in Stitcher._LOCUS_RE.finditer(item):
                loci.add(match.group(1).lower())
        return frozenset(loci)

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        return " ".join(claim.lower().split()).strip(" .;:!?-\"'")

    @staticmethod
    def _token_similarity(left: str, right: str) -> float:
        left_tokens, right_tokens = set(left.split()), set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        union = len(left_tokens | right_tokens)
        return len(left_tokens & right_tokens) / union if union else 0.0

    @staticmethod
    def _content_tokens(text: str) -> frozenset:
        return frozenset(
            token
            for token in text.split()
            if token not in Stitcher._STOPWORDS and len(token) > 1
        )

    @staticmethod
    def _claims_similar(
        left: str, right: str, left_loci: frozenset, right_loci: frozenset
    ) -> bool:
        if left == right or left in right or right in left:
            return True
        if left_loci & right_loci:
            shared_content = Stitcher._content_tokens(left) & Stitcher._content_tokens(right)
            return len(shared_content) >= Stitcher._LOCUS_GATED_MIN_SHARED_CONTENT
        return Stitcher._token_similarity(left, right) >= Stitcher._PURE_TOKEN_SIMILARITY

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

