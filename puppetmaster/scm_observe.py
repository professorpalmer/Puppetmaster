"""Host-observed SCM facts: receipts plus same-job follow-ups.

Not an Agent-Orchestrator daemon. One shot: parse PR facts, record host
observations, enqueue a graph child. Display lanes are derived at read time.
Never paste into a live TUI. A suppressed inject is not delivered.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional

from puppetmaster.metr_seams import (
    ACTOR_COORDINATOR,
    HOLD_STATE,
    HOST_SCM_ACTION_KINDS,
    HOST_SCM_KINDS,
    VETO_STATE,
    WAIT_USER,
    load_host_document,
    record_host_observation,
    save_host_document,
)
from puppetmaster.models import ArtifactType, JobStatus, TaskStatus, is_terminal_job_status

OUTCOME_ACCOUNTED = "accounted"
OUTCOME_SUPPRESSED = "suppressed"
OUTCOME_SKIPPED = "skipped"

ATTENTION_QUEUED = "queued"
ATTENTION_WORKING = "working"
ATTENTION_NEEDS_YOU = "needs_you"
ATTENTION_READY = "ready"
ATTENTION_DONE = "done"

_CI_FAIL_CONCLUSIONS = frozenset({"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED"})


@dataclass(frozen=True)
class SCMSnapshot:
    pr_url: str = ""
    number: Optional[int] = None
    title: str = ""
    state: str = ""
    mergeable: str = ""
    review_decision: str = ""
    failing_checks: tuple = ()
    fetch_errors: tuple = ()


@dataclass(frozen=True)
class SCMFact:
    kind: str
    key: str
    signature: str
    instruction: str
    evidence: tuple


def snapshot_from_gh_payload(payload: dict[str, Any]) -> SCMSnapshot:
    """Parse ``gh pr view --json`` output into durable facts."""
    errors: list[str] = []
    failing: list[str] = []
    rollup = payload.get("statusCheckRollup")
    if rollup is None:
        errors.append("checks: missing statusCheckRollup")
    elif not isinstance(rollup, list):
        errors.append("checks: statusCheckRollup is not a list")
    else:
        for item in rollup:
            if not isinstance(item, dict):
                continue
            conclusion = str(item.get("conclusion") or "").strip().upper()
            name = str(item.get("name") or item.get("context") or "").strip()
            if conclusion in _CI_FAIL_CONCLUSIONS and name:
                failing.append(name)
    url = str(payload.get("url") or "").strip()
    number = payload.get("number")
    try:
        parsed_number = int(number) if number is not None else None
    except (TypeError, ValueError):
        parsed_number = None
        errors.append("number: unreadable")
    return SCMSnapshot(
        pr_url=url,
        number=parsed_number,
        title=str(payload.get("title") or "").strip(),
        state=str(payload.get("state") or "").strip().upper(),
        mergeable=str(payload.get("mergeable") or "").strip().upper(),
        review_decision=str(payload.get("reviewDecision") or "").strip().upper(),
        failing_checks=tuple(failing),
        fetch_errors=tuple(errors),
    )


def fetch_github_pr(
    cwd: str,
    *,
    runner: Optional[Callable[..., Any]] = None,
) -> Optional[SCMSnapshot]:
    """Load the current branch's PR via ``gh``. None when no PR exists."""
    run = runner or subprocess.run
    try:
        completed = run(
            [
                "gh",
                "pr",
                "view",
                "--json",
                "url,number,title,state,mergeable,reviewDecision,statusCheckRollup",
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return SCMSnapshot(fetch_errors=("gh: %s" % type(exc).__name__,))
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if "no pull requests found" in stderr.lower():
            return None
        return SCMSnapshot(fetch_errors=("gh: %s" % (stderr or "exit %s" % completed.returncode),))
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return SCMSnapshot(fetch_errors=("gh: unreadable JSON",))
    if not isinstance(payload, dict):
        return SCMSnapshot(fetch_errors=("gh: JSON object required",))
    return snapshot_from_gh_payload(payload)


def facts_from_snapshot(snapshot: SCMSnapshot) -> tuple:
    """Independent fact kinds. One lookup failure must not hide the others."""
    ident = snapshot.pr_url or (
        "PR #%s" % snapshot.number if snapshot.number is not None else "your PR"
    )
    facts: list[SCMFact] = []
    if snapshot.state == "MERGED" and snapshot.pr_url:
        facts.append(
            _fact(
                "merged",
                "merged:" + snapshot.pr_url,
                snapshot.pr_url,
                "Host observed merge of %s." % ident,
                (snapshot.pr_url,),
            )
        )
    if snapshot.failing_checks:
        names = ", ".join(snapshot.failing_checks)
        msg = "CI is failing on %s: %s. Fix the failing checks." % (ident, names)
        if snapshot.pr_url:
            msg += "\nPR: %s" % snapshot.pr_url
        facts.append(
            _fact(
                "ci_failed",
                "ci:" + (snapshot.pr_url or ident),
                "|".join(snapshot.failing_checks),
                msg,
                snapshot.failing_checks + ((snapshot.pr_url,) if snapshot.pr_url else ()),
            )
        )
    elif snapshot.pr_url and "checks: missing statusCheckRollup" not in snapshot.fetch_errors:
        facts.append(
            _fact(
                "ci_passing",
                "ci:" + snapshot.pr_url,
                "passing",
                "CI is passing on %s." % ident,
                (snapshot.pr_url,),
            )
        )
    if snapshot.review_decision == "CHANGES_REQUESTED":
        msg = "Review requested changes on %s. Address the review comments." % ident
        if snapshot.pr_url:
            msg += "\nPR: %s" % snapshot.pr_url
        facts.append(
            _fact(
                "changes_requested",
                "review:" + (snapshot.pr_url or ident),
                snapshot.review_decision,
                msg,
                (snapshot.review_decision,) + ((snapshot.pr_url,) if snapshot.pr_url else ()),
            )
        )
    if snapshot.mergeable == "CONFLICTING":
        msg = "There are merge conflicts on %s. Rebase onto the base branch and resolve them." % ident
        if snapshot.pr_url:
            msg += "\nPR: %s" % snapshot.pr_url
        facts.append(
            _fact(
                "conflicting",
                "merge-conflict:" + (snapshot.pr_url or ident),
                snapshot.mergeable,
                msg,
                (snapshot.mergeable,) + ((snapshot.pr_url,) if snapshot.pr_url else ()),
            )
        )
    return tuple(facts)


def _fact(kind: str, key: str, signature_src: str, instruction: str, evidence: tuple) -> SCMFact:
    digest = hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16]
    return SCMFact(
        kind=kind,
        key=key,
        signature=digest,
        instruction=instruction,
        evidence=evidence,
    )


def observe_scm(
    store: Any,
    job_id: str,
    snapshot: Optional[SCMSnapshot],
    *,
    enqueue: bool = True,
    actor: str = ACTOR_COORDINATOR,
) -> dict[str, Any]:
    """Record SCM facts and maybe enqueue same-job follow-ups.

    Reactions are independent: a failed review lookup cannot hide a CI fact.
    ``waiting_user`` / HOLD / VETO suppress enqueue and do not stamp delivered.
    """
    result: dict[str, Any] = {
        "job_id": job_id,
        "pr_url": snapshot.pr_url if snapshot is not None else "",
        "fetch_errors": list(snapshot.fetch_errors) if snapshot is not None else [],
        "observations": [],
        "reactions": [],
        "attention": derive_attention(store, job_id),
    }
    if snapshot is None:
        result["attention"] = derive_attention(store, job_id)
        return result
    facts = facts_from_snapshot(snapshot)
    job = store.get_job(job_id)
    parent = _follow_up_parent(store, job_id) if enqueue else None
    suppress_reason = _suppress_reason(job)
    skip_reason = _skip_reason(job)
    for fact in facts:
        observation = record_host_observation(
            store, job_id, fact.kind, evidence=list(fact.evidence), source=actor
        )
        result["observations"].append(
            {
                "kind": fact.kind,
                "key": fact.key,
                "idempotent": bool(observation.get("idempotent")),
            }
        )
        if fact.kind not in HOST_SCM_ACTION_KINDS:
            continue
        reaction = _react(
            store,
            job_id,
            fact,
            parent_task_id=parent.id if parent is not None else None,
            enqueue=enqueue,
            suppress_reason=suppress_reason,
            skip_reason=skip_reason,
            actor=actor,
        )
        result["reactions"].append(reaction)
    result["attention"] = derive_attention(store, job_id)
    return result


def derive_attention(store: Any, job_id: str) -> str:
    """Kanban lane from durable facts. Never stored as a status enum."""
    job = store.get_job(job_id)
    if is_terminal_job_status(job.status):
        return ATTENTION_DONE
    if job.wait_reason == WAIT_USER or job.subgraph_hold in {HOLD_STATE, VETO_STATE}:
        return ATTENTION_NEEDS_YOU
    latest = _latest_scm_kinds(store, job_id)
    if latest.get("ci_failed") or latest.get("changes_requested") or latest.get("conflicting"):
        return ATTENTION_NEEDS_YOU
    if latest.get("ci_passing") and not latest.get("conflicting"):
        artifacts = store.list_artifacts(job_id)
        if any(str(item.type) == str(ArtifactType.PATCH) for item in artifacts):
            return ATTENTION_READY
    if job.status == JobStatus.QUEUED:
        return ATTENTION_QUEUED
    return ATTENTION_WORKING


def _latest_scm_kinds(store: Any, job_id: str) -> dict[str, bool]:
    latest: dict[str, tuple] = {}
    for row in load_host_document(store, job_id).get("observations") or []:
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in HOST_SCM_KINDS:
            continue
        observed_at = str(row.get("observed_at") or "")
        previous = latest.get(kind)
        if previous is None or observed_at >= previous[0]:
            latest[kind] = (observed_at, True)
    active = {kind: True for kind, _item in latest.items()}
    if active.get("ci_passing") and active.get("ci_failed"):
        pass_at = latest["ci_passing"][0]
        fail_at = latest["ci_failed"][0]
        if pass_at >= fail_at:
            active.pop("ci_failed", None)
        else:
            active.pop("ci_passing", None)
    return active


def _suppress_reason(job: Any) -> Optional[str]:
    hold = getattr(job, "subgraph_hold", None)
    if hold == HOLD_STATE:
        return "hold"
    if hold == VETO_STATE:
        return "veto"
    if getattr(job, "wait_reason", None) == WAIT_USER:
        return WAIT_USER
    return None


def _skip_reason(job: Any) -> Optional[str]:
    if is_terminal_job_status(job.status):
        return "job_terminal"
    return None


def _follow_up_parent(store: Any, job_id: str) -> Any:
    tasks = list(store.list_tasks(job_id))
    if not tasks:
        return None
    complete = [task for task in tasks if task.status == TaskStatus.COMPLETE]
    if complete:
        return complete[-1]
    return tasks[-1]


def _react(
    store: Any,
    job_id: str,
    fact: SCMFact,
    *,
    parent_task_id: Optional[str],
    enqueue: bool,
    suppress_reason: Optional[str],
    skip_reason: Optional[str],
    actor: str,
) -> dict[str, Any]:
    document = load_host_document(store, job_id)
    reactions = dict(document.get("reactions") or {})
    prior = reactions.get(fact.key) if isinstance(reactions.get(fact.key), dict) else {}
    if prior.get("signature") == fact.signature and prior.get("outcome") == OUTCOME_ACCOUNTED:
        return {
            "key": fact.key,
            "kind": fact.kind,
            "outcome": OUTCOME_ACCOUNTED,
            "reason": "deduped",
            "task_id": prior.get("task_id"),
        }
    if suppress_reason:
        _stamp_reaction(
            store,
            job_id,
            fact,
            outcome=OUTCOME_SUPPRESSED,
            task_id=None,
            reason=suppress_reason,
        )
        return {
            "key": fact.key,
            "kind": fact.kind,
            "outcome": OUTCOME_SUPPRESSED,
            "reason": suppress_reason,
            "task_id": None,
        }
    if skip_reason or not enqueue:
        reason = skip_reason or "no_enqueue"
        _stamp_reaction(
            store, job_id, fact, outcome=OUTCOME_SKIPPED, task_id=None, reason=reason
        )
        return {
            "key": fact.key,
            "kind": fact.kind,
            "outcome": OUTCOME_SKIPPED,
            "reason": reason,
            "task_id": None,
        }
    if not parent_task_id:
        _stamp_reaction(
            store,
            job_id,
            fact,
            outcome=OUTCOME_SKIPPED,
            task_id=None,
            reason="no_parent",
        )
        return {
            "key": fact.key,
            "kind": fact.kind,
            "outcome": OUTCOME_SKIPPED,
            "reason": "no_parent",
            "task_id": None,
        }
    child = store.enqueue_subtask(
        job_id,
        parent_task_id=parent_task_id,
        role="implement",
        instruction=fact.instruction,
        payload={"scm_fact": fact.kind, "scm_key": fact.key},
        actor=actor,
        created_by=actor,
    )
    if child is None:
        _stamp_reaction(
            store,
            job_id,
            fact,
            outcome=OUTCOME_SKIPPED,
            task_id=None,
            reason="enqueue_refused",
        )
        return {
            "key": fact.key,
            "kind": fact.kind,
            "outcome": OUTCOME_SKIPPED,
            "reason": "enqueue_refused",
            "task_id": None,
        }
    _stamp_reaction(
        store,
        job_id,
        fact,
        outcome=OUTCOME_ACCOUNTED,
        task_id=child.id,
        reason="enqueued",
    )
    return {
        "key": fact.key,
        "kind": fact.kind,
        "outcome": OUTCOME_ACCOUNTED,
        "reason": "enqueued",
        "task_id": child.id,
    }


def _stamp_reaction(
    store: Any,
    job_id: str,
    fact: SCMFact,
    *,
    outcome: str,
    task_id: Optional[str],
    reason: str,
) -> None:
    document = load_host_document(store, job_id)
    reactions = dict(document.get("reactions") or {})
    reactions[fact.key] = {
        "signature": fact.signature,
        "outcome": outcome,
        "task_id": task_id,
        "reason": reason,
        "kind": fact.kind,
    }
    document["reactions"] = reactions
    save_host_document(store, job_id, document)


def observe_job_cwd(
    store: Any,
    job_id: str,
    cwd: str,
    *,
    enqueue: bool = True,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    snapshot = fetch_github_pr(cwd, runner=runner)
    return observe_scm(store, job_id, snapshot, enqueue=enqueue)
