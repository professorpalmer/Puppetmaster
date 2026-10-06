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
    ArtifactType.GIST,
}

_DEGRADED_FAILURE_MARKERS = frozenset({
    "empty_or_unstructured_cursor_result",
    "empty_or_unstructured_agentic_result",
})


def _payload(artifact: Artifact) -> dict[str, Any]:
    return getattr(artifact, "payload", None) or {}


def latest_gate_results(artifacts: list[Artifact]) -> list[Artifact]:
    """Drop gate results a later evaluation of the same gate on the same task superseded.

    A review-loop task fails its review, is repaired, and passes on a later
    attempt; the first rejection must not keep the delivered run blocked.
    Within the same second a failing result wins, so a pass can never hide a
    failure it did not follow.
    """
    latest: dict[tuple[str, str], Artifact] = {}
    for artifact in artifacts:
        if artifact.type != ArtifactType.GATE:
            continue
        payload = _payload(artifact)
        key = (str(artifact.task_id), str(payload.get("gate") or payload.get("kind") or ""))
        rank = (str(artifact.created_at or ""), payload.get("passed") is False)
        current = latest.get(key)
        if current is None or rank >= (str(current.created_at or ""), _payload(current).get("passed") is False):
            latest[key] = artifact
    keep = {id(artifact) for artifact in latest.values()}
    return [a for a in artifacts if a.type != ArtifactType.GATE or id(a) in keep]


def _is_blocked(artifact: Artifact) -> bool:
    payload = _payload(artifact)
    if payload.get("result") == "blocked":
        return True
    # A failed completion gate (drift ratchet, required diff, commit) means the
    # run did not satisfy its post-conditions — never trustworthy.
    return artifact.type == ArtifactType.GATE and payload.get("passed") is False


def _is_degraded_marker(artifact: Artifact) -> bool:
    payload = _payload(artifact)
    if payload.get("result") == "degraded":
        return True
    failure = payload.get("failure")
    if failure in _DEGRADED_FAILURE_MARKERS:
        return True
    if artifact.type == ArtifactType.RISK:
        evidence = set(artifact.evidence or [])
        if (
            "result:empty-or-unstructured" in evidence
            or "cursor-result:empty-or-unstructured" in evidence
        ):
            return True
    return False


def _worker_has_substantive_output(artifacts: list[Artifact], worker_id: str) -> bool:
    for artifact in artifacts:
        if artifact.created_by != worker_id:
            continue
        if artifact.type in _SUBSTANTIVE_TYPES and not _is_degraded_marker(artifact):
            return True
    return False


def _is_max_turns_without_findings(
    artifact: Artifact, artifacts: list[Artifact]
) -> bool:
    payload = _payload(artifact)
    if artifact.type != ArtifactType.VERIFICATION:
        return False
    if payload.get("stop_reason") != "max_turns":
        return False
    return not _worker_has_substantive_output(artifacts, artifact.created_by)


def _objective_evaluator_summary(artifacts: list[Artifact]) -> dict[str, Any]:
    """Summarize explicit evaluator evidence without inferring semantics.

    Structural artifact presence still drives the legacy ``quality`` field,
    while this companion field makes the proof boundary machine-readable.
    """
    outcomes: list[bool] = []
    revisions: set[str] = set()
    for artifact in artifacts:
        if artifact.type != ArtifactType.GATE:
            continue
        payload = _payload(artifact)
        review_status = str(payload.get("review_status") or "").lower()
        if review_status in {"unavailable", "skipped", "independence_failed"}:
            continue
        is_evaluator = bool(
            payload.get("evaluator_revision")
            or payload.get("evaluator_version")
            or payload.get("reviewed_artifact_fingerprint")
            or payload.get("objective_score") is not None
        )
        if not is_evaluator or "passed" not in payload:
            continue
        outcomes.append(bool(payload.get("passed")))
        revision = payload.get("evaluator_revision") or payload.get("evaluator_version")
        if revision not in (None, ""):
            revisions.add(str(revision))
    return {
        "semantic_quality": (
            "passed" if outcomes and all(outcomes)
            else "failed" if outcomes
            else "not_evaluated"
        ),
        "objective_evaluations": len(outcomes),
        "evaluator_revisions": sorted(revisions),
        "trust_basis": (
            "objective_evaluator" if outcomes
            else "structural_artifact_presence"
        ),
    }


def assess_run_quality(artifacts: Iterable[Artifact]) -> dict[str, Any]:
    """Classify a finished run. See module docstring for verdict semantics."""
    artifacts = latest_gate_results(list(artifacts))
    evaluator_summary = _objective_evaluator_summary(artifacts)
    reasons: list[str] = []

    blocked = [a for a in artifacts if _is_blocked(a)]
    if blocked:
        failures = sorted(
            {
                str(
                    _payload(a).get("failure")
                    or (f"gate:{_payload(a).get('gate')}" if a.type == ArtifactType.GATE else None)
                    or "blocked"
                )
                for a in blocked
            }
        )
        reasons.append(f"blocked / failed post-conditions: {', '.join(failures)}")
        return {
            "quality": "blocked",
            "reasons": reasons,
            "trustworthy": False,
            "blocking_failures": failures,
            **evaluator_summary,
        }

    if not artifacts:
        return {
            "quality": "empty",
            "reasons": ["no artifacts were produced"],
            "trustworthy": False,
            "blocking_failures": [],
            **evaluator_summary,
        }

    substantive = [a for a in artifacts if a.type in _SUBSTANTIVE_TYPES and not _is_degraded_marker(a)]
    if not substantive:
        if any(
            _is_degraded_marker(a) or _is_max_turns_without_findings(a, artifacts)
            for a in artifacts
        ):
            reasons.append("only degraded/empty SDK results — no structured output")
        else:
            reasons.append("only verification artifacts — no findings/decisions/patches")
        return {
            "quality": "degraded",
            "reasons": reasons,
            "trustworthy": False,
            "blocking_failures": [],
            **evaluator_summary,
        }

    unfinished = _write_run_unfinished(artifacts)
    if unfinished:
        return {
            "quality": "degraded",
            "reasons": [unfinished],
            "trustworthy": False,
            "blocking_failures": [],
            **evaluator_summary,
        }

    if _write_run_changed_nothing(artifacts):
        return {
            "quality": "degraded",
            "reasons": ["write-capable run changed nothing (no diff, patch or commit)"],
            "trustworthy": False,
            "blocking_failures": [],
            **evaluator_summary,
        }

    return {
        "quality": "ok",
        "reasons": [],
        "trustworthy": True,
        "blocking_failures": [],
        **evaluator_summary,
    }


_WRITE_PERMISSION_MODES = ("acceptEdits", "bypassPermissions")
_WRITE_SANDBOXES = ("workspace-write", "danger-full-access")


def _write_run_unfinished(artifacts: list[Artifact]) -> str:
    """Why a write-capable run is not a finished delivery, or ''.

    The build contract asks every edit worker to end with VERDICT
    PASS|FAIL|PARTIAL. A worker that did 2 of 26 assigned items reported
    "Committed." with no verdict and the job read as delivered.
    """
    receipts = [_payload(a) for a in artifacts if a.type == ArtifactType.VERIFICATION
                and "worker_diff_present" in _payload(a)]
    if not any(p.get("permission_mode") in _WRITE_PERMISSION_MODES
               or p.get("sandbox") in _WRITE_SANDBOXES for p in receipts):
        return ""
    verdicts = [_payload(a) for a in artifacts if a.type == ArtifactType.VERIFICATION
                and _payload(a).get("kind") == "worker_verdict"]
    if not verdicts:
        return "unverified: the edit worker did not report whether it finished (no VERDICT)"
    verdict = str(verdicts[-1].get("verdict") or "").upper()
    if verdict in ("FAIL", "PARTIAL"):
        reason = str(verdicts[-1].get("reason") or "").strip()
        return f"worker reported {verdict}" + (f": {reason[:200]}" if reason else "")
    return ""


def _write_run_changed_nothing(artifacts: list[Artifact]) -> bool:
    """A run its adapter allowed to edit, that reported no change of its own.

    A refusal ("I can't proceed") is a finding, so it once read as a
    delivered, trustworthy job.
    """
    if any(a.type == ArtifactType.PATCH for a in artifacts):
        return False
    receipts = [_payload(a) for a in artifacts if a.type == ArtifactType.VERIFICATION
                and "worker_diff_present" in _payload(a)]
    writable = [p for p in receipts if p.get("permission_mode") in _WRITE_PERMISSION_MODES
                or p.get("sandbox") in _WRITE_SANDBOXES]
    return bool(writable) and not any(p.get("worker_diff_present") or p.get("commit_sha")
                                      for p in receipts)
