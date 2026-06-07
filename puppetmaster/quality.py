"""Built-in run-quality classification.

The parent agent used to decide by hand whether a finished swarm was
trustworthy — "only verification artifacts / empty findings = degraded, don't
trust it." That heuristic belongs in the runtime, not in a human's head. This
module turns a job's artifacts into a single ``quality`` verdict so callers
(CLI ``show``/``status``, the MCP surface, an orchestrating parent) can branch
on it without eyeballing artifact composition.

Verdicts (worst-first):

- ``blocked``  — a worker refused to run (dirty tree, non-worktree, preflight).
                 The run did zero real work; treating it as success is the
                 single worst failure mode.
- ``empty``    — no artifacts at all.
- ``degraded`` — the run produced only verification/degraded markers and no
                 substantive output (no finding/decision/patch/risk content).
- ``ok``       — substantive artifacts are present.
"""

from __future__ import annotations

from typing import Any, Iterable

from puppetmaster.models import Artifact, ArtifactType

# Artifact types that represent real, substantive work product. A run that
# emits at least one of these (beyond a bare degraded marker) is not degraded.
_SUBSTANTIVE_TYPES = {
    ArtifactType.FINDING,
    ArtifactType.DECISION,
    ArtifactType.PATCH,
    ArtifactType.RISK,
}


def _payload(artifact: Artifact) -> dict[str, Any]:
    return getattr(artifact, "payload", None) or {}


def _is_blocked(artifact: Artifact) -> bool:
    return _payload(artifact).get("result") == "blocked"


def _is_degraded_marker(artifact: Artifact) -> bool:
    payload = _payload(artifact)
    if payload.get("result") == "degraded":
        return True
    # The degraded RISK artifact the Cursor adapter emits carries this failure
    # tag in its mitigation/risk text; key off the explicit failure when present.
    return payload.get("failure") == "empty_or_unstructured_cursor_result"


def assess_run_quality(artifacts: Iterable[Artifact]) -> dict[str, Any]:
    """Classify a finished run. See module docstring for verdict semantics."""
    artifacts = list(artifacts)
    reasons: list[str] = []

    blocked = [a for a in artifacts if _is_blocked(a)]
    if blocked:
        failures = sorted({str(_payload(a).get("failure") or "blocked") for a in blocked})
        reasons.append(f"worker refused to run: {', '.join(failures)}")
        return {
            "quality": "blocked",
            "reasons": reasons,
            "trustworthy": False,
            "blocking_failures": failures,
        }

    if not artifacts:
        return {
            "quality": "empty",
            "reasons": ["no artifacts were produced"],
            "trustworthy": False,
            "blocking_failures": [],
        }

    substantive = [a for a in artifacts if a.type in _SUBSTANTIVE_TYPES and not _is_degraded_marker(a)]
    if not substantive:
        if any(_is_degraded_marker(a) for a in artifacts):
            reasons.append("only degraded/empty SDK results — no structured output")
        else:
            reasons.append("only verification artifacts — no findings/decisions/patches")
        return {
            "quality": "degraded",
            "reasons": reasons,
            "trustworthy": False,
            "blocking_failures": [],
        }

    return {
        "quality": "ok",
        "reasons": [],
        "trustworthy": True,
        "blocking_failures": [],
    }
