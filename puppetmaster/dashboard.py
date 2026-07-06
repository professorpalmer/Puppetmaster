"""Zero-dependency local dashboard for live (and completed) swarms.

OpenTelemetry is the *export* path — point a swarm at Jaeger/Datadog and the
job renders as a correlated trace. But not everyone runs an OTLP collector just
to watch one job, and a finished job's durable state already holds everything
worth showing. This module turns that durable state into a live board served
over ``http.server`` from the standard library — no Flask, no React build, no
external CDN (the page is fully inlined so it works offline).

Two layers, cleanly separated so the data layer is unit-testable without a
socket:

* :func:`build_job_snapshot` / :func:`list_jobs_snapshot` — pure functions that
  read a :class:`~puppetmaster.store.SwarmStore` and return JSON-able dicts
  (task board, per-task activity timeline, typed-artifact rollup, cost from
  ROUTING artifacts, auto-fallback reroutes, and the same Alerts the stitcher
  surfaces).
* :func:`serve` — wraps those in a tiny HTTP handler and polls every ~1.5 s, so
  the board updates live while a swarm runs and is instant for a finished one.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Union

from puppetmaster.models import Artifact, ArtifactType, Job, Task
from puppetmaster.store import SwarmStore
from puppetmaster.usage import aggregate_token_usage

# Reuse the stitcher's failure→remediation map so the dashboard Alerts match the
# stitched summary verbatim (one source of truth for "what went wrong + fix").
from puppetmaster.stitcher import Stitcher

# Job ids are minted as ``job_<hex>``; constrain to that alphabet so a request
# id can never escape the state tree via ``..`` / absolute paths (defense in
# depth — the server binds to loopback by default, but a local user or a careless
# ``--host`` should still not be able to read arbitrary files).
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _valid_job_id(job_id: str) -> bool:
    return bool(job_id) and bool(_JOB_ID_RE.match(job_id)) and len(job_id) <= 128


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


_ARTIFACT_STATEMENT_KEYS = {
    ArtifactType.FINDING: "claim",
    ArtifactType.RISK: "risk",
    ArtifactType.DECISION: "decision",
    ArtifactType.VERIFICATION: "check",
}


def _artifact_statement(artifact: Artifact) -> str:
    key = _ARTIFACT_STATEMENT_KEYS.get(artifact.type)
    payload = artifact.payload or {}
    if key and isinstance(payload.get(key), str):
        return payload[key]
    # Fall back to a compact, human-readable headline.
    for candidate in ("claim", "summary", "message", "reason", "result"):
        value = payload.get(candidate)
        if isinstance(value, str) and value:
            return value
    return artifact.task_id


def _task_model(task: Task) -> Optional[str]:
    payload = task.payload or {}
    model = payload.get("model")
    if isinstance(model, str) and model and model != "default":
        return model
    return None


def _extract_readable_message(payload: dict[str, Any]) -> str:
    """Best human-readable output from a verification payload.

    Worker stdout is often a JSON envelope whose ``result`` field carries the
    actual model prose; prefer that over raw head/tail excerpts so the board
    reads like a report instead of a transcript dump.
    """
    result_text = payload.get("result", "")
    if isinstance(result_text, str) and len(result_text) > 20:
        return result_text
    stdout_capture = payload.get("stdout_capture") or {}
    for excerpt_key in ("stdout_head_excerpt", "stdout_tail_excerpt"):
        excerpt = stdout_capture.get(excerpt_key, "")
        if not excerpt:
            continue
        try:
            envelope = json.loads(excerpt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(envelope, dict) and "result" in envelope:
            return str(envelope["result"])
    if isinstance(result_text, str) and result_text:
        return result_text
    head = stdout_capture.get("stdout_head_excerpt", "")
    tail = stdout_capture.get("stdout_tail_excerpt", "")
    if head and tail:
        return f"{head}\n...\n{tail}"
    return head or tail


def _extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Model / cost / token / timing chips for a task's activity feed."""
    meta: dict[str, Any] = {}
    if "model_id" in payload:
        meta["model"] = payload["model_id"]
    cost = payload.get("total_cost_usd") or payload.get("estimated_cost_usd")
    if cost is not None:
        meta["cost_usd"] = _safe_float(cost)
    stdout_capture = payload.get("stdout_capture") or {}
    for excerpt_key in ("stdout_head_excerpt", "stdout_tail_excerpt"):
        excerpt = stdout_capture.get(excerpt_key, "")
        if not excerpt:
            continue
        try:
            envelope = json.loads(excerpt)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(envelope, dict):
            continue
        usage = envelope.get("usage") or {}
        if "input_tokens" in usage:
            meta["tokens_in"] = usage["input_tokens"]
        if "output_tokens" in usage:
            meta["tokens_out"] = usage["output_tokens"]
        if "total_cost_usd" in envelope:
            meta["cost_usd"] = _safe_float(envelope["total_cost_usd"])
        if "num_turns" in envelope:
            meta["num_turns"] = envelope["num_turns"]
        if "duration_ms" in envelope:
            meta["duration_ms"] = envelope["duration_ms"]
        break
    # Claude Code is the only adapter whose stdout is a single JSON envelope.
    # Every adapter stamps token_usage() counts top-level on the payload, so
    # fall back to those (flagging estimates) — otherwise cursor/codex/openai
    # tasks never show token chips at all.
    if "tokens_in" not in meta and payload.get("tokens_in") is not None:
        meta["tokens_in"] = payload["tokens_in"]
        if payload.get("tokens_out") is not None:
            meta["tokens_out"] = payload["tokens_out"]
        meta["tokens_estimated"] = bool(payload.get("tokens_estimated"))
    if "returncode" in payload:
        meta["returncode"] = payload["returncode"]
    if "failure" in payload:
        meta["failure"] = payload["failure"]
    return meta


def _extract_why(artifact: Artifact) -> str:
    """One human sentence explaining a routing choice / verification outcome."""
    payload = artifact.payload or {}
    parts: list[str] = []

    if artifact.type == ArtifactType.ROUTING:
        model_id = payload.get("model_id", "unknown")
        reason = payload.get("reason", "")
        cap_needed = payload.get("capability_needed")
        cap_score = payload.get("capability_score")
        if reason:
            parts.append(f"Routed to {model_id} — {reason}")
        elif cap_needed is not None and cap_score is not None:
            parts.append(f"Routed to {model_id} (capability {cap_score} >= {cap_needed})")
        else:
            parts.append(f"Routed to {model_id}")

    if artifact.type == ArtifactType.VERIFICATION:
        check = payload.get("check", "")
        result = payload.get("result", "")
        confidence = round(float(artifact.confidence), 2)
        if check:
            parts.append(f"Verified: {check}")
        if result:
            parts.append(f"Result: {result}")
        if confidence > 0:
            parts.append(f"confidence {confidence}")

    return " · ".join(parts)


def _find_patch_for_task(task_id: str, artifacts: list[Artifact]) -> Optional[dict[str, Any]]:
    """The PATCH artifact emitted by exactly this task, if any.

    Adapters always stamp ``task_id`` on PATCH artifacts; matching exactly (no
    job-level fallback) keeps one task's diff from being repeated on every
    card in a large swarm.
    """
    for artifact in artifacts:
        if artifact.type == ArtifactType.PATCH and artifact.task_id == task_id:
            payload = artifact.payload or {}
            return {
                "files": payload.get("files", []),
                "unified_diff": payload.get("unified_diff", ""),
                "truncated": payload.get("diff_truncated", False),
                "total_chars": payload.get("diff_total_chars", 0),
            }
    return None


def _build_task_activity(task_id: str, artifacts: list[Artifact]) -> list[dict[str, Any]]:
    """Readable timeline of what a task did / is doing, from its artifacts."""
    activity: list[dict[str, Any]] = []

    for artifact in artifacts:
        if artifact.task_id != task_id:
            continue
        if artifact.type == ArtifactType.PATCH:
            # Represented by the synthesized diff entry appended below.
            continue
        payload = artifact.payload or {}
        item: dict[str, Any] = {
            "type": artifact.type.value,
            "confidence": round(float(artifact.confidence), 2),
            "created_by": artifact.created_by,
        }

        if artifact.type == ArtifactType.VERIFICATION:
            item["text"] = f"Verification: {payload.get('check', '')}"
            item["result"] = payload.get("result")
            item["evidence"] = artifact.evidence
            item["message"] = _extract_readable_message(payload)
            item["meta"] = _extract_metadata(payload)
            item["why"] = _extract_why(artifact)
        elif artifact.type == ArtifactType.ROUTING:
            model_id = payload.get("model_id", "unknown")
            item["text"] = f"Routed to {model_id}"
            item["model"] = model_id
            item["message"] = _extract_why(artifact)
            item["meta"] = _extract_metadata(payload)
            item["why"] = _extract_why(artifact)
        elif artifact.type in (ArtifactType.FINDING, ArtifactType.RISK, ArtifactType.DECISION):
            item["text"] = _artifact_statement(artifact)
            item["evidence"] = artifact.evidence
            item["message"] = _artifact_statement(artifact)
        else:
            item["text"] = payload.get("reason") or payload.get("result") or payload.get("message") or ""
            item["message"] = item["text"]

        if payload.get("reasoning_output_tokens"):
            item["reasoning_tokens"] = payload["reasoning_output_tokens"]

        activity.append(item)

    patch_diff = _find_patch_for_task(task_id, artifacts)
    if patch_diff and patch_diff["unified_diff"]:
        activity.append(
            {
                "type": "patch",
                "text": "Code changes",
                "diff": patch_diff,
                "confidence": 1.0,
                "created_by": "patch-artifact",
            }
        )

    return activity


def collect_alerts(artifacts: list[Artifact]) -> list[str]:
    """Same alert logic as the stitched summary, exposed as plain strings."""
    return Stitcher(None)._collect_alerts(artifacts)


def cost_rollup(artifacts: list[Artifact]) -> dict[str, Any]:
    """Sum estimated USD spend from ROUTING artifacts (router estimates)."""
    by_model: dict[str, dict[str, float]] = {}
    total = 0.0
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING:
            continue
        payload = artifact.payload or {}
        model_id = str(payload.get("model_id") or payload.get("selected_model") or "<unknown>")
        cost = _safe_float(payload.get("estimated_cost_usd"))
        total += cost
        bucket = by_model.setdefault(model_id, {"calls": 0.0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["cost"] += cost
    return {
        "total_estimated_cost_usd": round(total, 6),
        "by_model": {
            mid: {"calls": int(v["calls"]), "estimated_cost_usd": round(v["cost"], 6)}
            for mid, v in sorted(by_model.items())
        },
    }


_JOB_TITLE_BOILERPLATE_PREFIXES = (
    "audit this repository",
    "use puppetmaster to ",
    "audit the ",
    "review the ",
    "review this ",
    "implement ",
    "investigate ",
    "analyze ",
    "plan ",
)


def derive_job_title(goal: str) -> str:
    """Derive a short, scannable display title from a job goal (no LLM)."""
    if not goal or not goal.strip():
        return ""
    first_line = ""
    for line in goal.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""
    original = first_line
    lowered = first_line.lower()
    for prefix in _JOB_TITLE_BOILERPLATE_PREFIXES:
        if lowered.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            lowered = first_line.lower()
    first_line = " ".join(first_line.split())
    if not first_line:
        first_line = original
    words = first_line.split()
    if len(words) > 8:
        first_line = " ".join(words[:8])
    if len(first_line) > 60:
        truncated = first_line[:60]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        first_line = truncated
    return first_line


def _job_snapshot_meta(job: Job) -> dict[str, Any]:
    title = derive_job_title(job.goal) or job.id
    return {"label": job.label, "title": title}


def _primary_model_from_routing(artifacts: list[Artifact]) -> Optional[str]:
    """Most frequent ROUTING ``model_id``, else the first one seen."""
    counts: dict[str, int] = {}
    first: Optional[str] = None
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING:
            continue
        payload = artifact.payload or {}
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if first is None:
            first = model_id
        counts[model_id] = counts.get(model_id, 0) + 1
    if counts:
        return max(counts, key=lambda mid: counts[mid])
    return first


def _adapters_from_tasks(tasks: list[Task]) -> tuple[list[str], Optional[str]]:
    adapters = sorted({task.adapter for task in tasks if task.adapter})
    if not adapters:
        return [], None
    counts: dict[str, int] = {}
    for task in tasks:
        if task.adapter:
            counts[task.adapter] = counts.get(task.adapter, 0) + 1
    # Break ties on the adapter name so the primary is deterministic regardless
    # of task-store ordering (the file store lists tasks by random task id).
    primary = max(counts, key=lambda name: (counts[name], name))
    return adapters, primary


def _evaluator_epoch_lineage(store: SwarmStore, job_id: str) -> list[dict[str, Any]]:
    """Slot/version/role only — criteria bodies stay out of the dashboard snapshot."""
    try:
        from puppetmaster.evaluators import evaluator_epoch_for_job

        epoch = evaluator_epoch_for_job(store, job_id)
        rows: list[dict[str, Any]] = []
        for entry in epoch.get("evaluators") or []:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "slot_id": entry.get("slot_id"),
                    "version": entry.get("version"),
                    "role": entry.get("role"),
                }
            )
        return rows
    except Exception:
        return []


def _verification_score(artifacts: list[Artifact]) -> Optional[float]:
    confidences = [
        float(artifact.confidence)
        for artifact in artifacts
        if artifact.type == ArtifactType.VERIFICATION
    ]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


def _routing_rollup(tasks: list[Task], artifacts: list[Artifact]) -> list[dict[str, Any]]:
    """One row per router ROUTING artifact, de-duplicated by task_id (earliest wins)."""
    task_roles = {task.id: task.role for task in tasks}
    by_task: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        by_task.setdefault(artifact.task_id, []).append(artifact)

    rollup: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        artifact = min(by_task[task_id], key=lambda row: (row.created_at or "", row.id))
        payload = artifact.payload or {}
        raw_rejected = payload.get("rejected") or []
        rejected: list[dict[str, str]] = []
        if isinstance(raw_rejected, list):
            for item in raw_rejected:
                if isinstance(item, dict):
                    rejected.append(
                        {
                            "id": str(item.get("id") or ""),
                            "reason": str(item.get("reason") or ""),
                        }
                    )
        rollup.append(
            {
                "task_id": artifact.task_id,
                "role": task_roles.get(artifact.task_id, ""),
                "model_id": str(payload.get("model_id") or ""),
                "estimated_cost_usd": _safe_float(payload.get("estimated_cost_usd")),
                "policy": str(payload.get("policy") or ""),
                "reason": str(payload.get("reason") or ""),
                "rejected": rejected,
                "rejected_count": len(rejected),
            }
        )
    return rollup


# A swarm advances through four visible phases; the dashboard renders them as a
# filling strip so a running job reads as *moving* rather than a static spinner.
# A dead job paints the phase it reached red instead of advancing.
_FAILED_JOB_STATUS = {"failed", "cancelled"}
_DONE_JOB_STATUS = {"complete", "completed"}


def _job_phase(job_status: str, task_rows: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    """Derive the four-phase marker (dispatched → routing → workers → done).

    Kept in Python so the phase is unit-testable and never drifts from the
    client strip, and reuses the already-built task rows and artifact counts
    rather than re-querying the store.
    """
    total = len(task_rows)
    has_routing = counts.get("routing", 0) > 0

    if job_status in _FAILED_JOB_STATUS:
        reached = 2 if total else 1 if has_routing else 0
        return {"key": "failed", "label": "failed", "index": reached, "failed": True}
    if job_status in _DONE_JOB_STATUS:
        return {"key": "done", "label": "done", "index": 3, "failed": False}
    if total:
        running = sum(1 for row in task_rows if row["status"] in ("running", "in_progress"))
        if running:
            done = sum(1 for row in task_rows if row["status"] == "complete")
            label = f"running {done}/{total}"
        else:
            label = f"{total} worker" + ("s" if total != 1 else "")
        return {"key": "workers", "label": label, "index": 2, "failed": False}
    if has_routing:
        return {"key": "routing", "label": "routing", "index": 1, "failed": False}
    return {"key": "dispatched", "label": "dispatched", "index": 0, "failed": False}


def build_job_snapshot(store: SwarmStore, job_id: str) -> dict[str, Any]:
    """Assemble a full, JSON-able snapshot of one job's durable state."""
    job = store.get_job(job_id)
    tasks = store.list_tasks(job_id)
    artifacts = store.list_artifacts(job_id)

    counts: dict[str, int] = {}
    grouped: dict[str, list[dict[str, Any]]] = {
        "finding": [],
        "risk": [],
        "decision": [],
        "verification": [],
        "routing": [],
        "patch": [],
    }
    reroutes: list[dict[str, Any]] = []
    for artifact in artifacts:
        kind = artifact.type.value
        counts[kind] = counts.get(kind, 0) + 1
        row = {
            "statement": _artifact_statement(artifact),
            "confidence": round(float(artifact.confidence), 2),
            "evidence": list(artifact.evidence),
            "created_by": artifact.created_by,
            "failure": (artifact.payload or {}).get("failure"),
            "result": (artifact.payload or {}).get("result"),
        }
        if kind in grouped:
            grouped[kind].append(row)
        if artifact.type == ArtifactType.ROUTING and str(artifact.created_by).startswith(
            ("router-fallback", "router-escalation")
        ):
            payload = artifact.payload or {}
            if str(artifact.created_by).startswith("router-escalation"):
                reason = (
                    f"low confidence {payload.get('escalated_from_confidence')} "
                    f"< {payload.get('confidence_threshold')} → escalated "
                    f"{payload.get('escalated_from_model')} to {payload.get('model_id')}"
                )
            else:
                reason = payload.get("reason")
            reroutes.append(
                {
                    "task_id": artifact.task_id,
                    "reason": reason,
                    "created_by": artifact.created_by,
                }
            )

    progress: dict[str, int] = {}
    for task in tasks:
        progress[task.status.value] = progress.get(task.status.value, 0) + 1

    task_rows = [
        {
            "id": task.id,
            "role": task.role,
            "instruction": task.instruction,
            "adapter": task.adapter,
            "status": task.status.value,
            "model": _task_model(task),
            "attempts": task.attempts,
            "activity": _build_task_activity(task.id, artifacts),
        }
        for task in tasks
    ]

    adapters, primary_adapter = _adapters_from_tasks(tasks)
    token_usage = aggregate_token_usage(artifacts)

    return {
        "job": {
            "id": job.id,
            "goal": job.goal,
            "status": job.status.value,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            **_job_snapshot_meta(job),
        },
        "tasks": task_rows,
        "counts": counts,
        "progress": progress,
        "artifacts": grouped,
        "reroutes": reroutes,
        "cost": cost_rollup(artifacts),
        "alerts": collect_alerts(artifacts),
        "tokens_total": int(token_usage["total_tokens"]),
        "primary_model": _primary_model_from_routing(artifacts),
        "worker_count": len(tasks),
        "adapters": adapters,
        "primary_adapter": primary_adapter,
        "verification_score": _verification_score(artifacts),
        "routing_rollup": _routing_rollup(tasks, artifacts),
        "evaluator_epoch": _evaluator_epoch_lineage(store, job_id),
        "phase": _job_phase(job.status.value, task_rows, counts),
    }


def list_jobs_snapshot(store: SwarmStore, *, limit: int = 50) -> list[dict[str, Any]]:
    """Compact list of jobs (most recent first) for the dashboard index."""
    jobs = store.list_jobs()
    rows = []
    for job in jobs:
        rows.append(
            {
                "id": job.id,
                "goal": job.goal,
                "status": job.status.value,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                **_job_snapshot_meta(job),
            }
        )
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[:limit]


def list_all_projects_snapshot(*, backend: str = "sqlite", limit: int = 200) -> list[dict[str, Any]]:
    """Aggregate jobs from every project state dir on this machine."""
    from puppetmaster.state import list_project_state_dirs
    from puppetmaster.store_factory import create_store

    rows: list[dict[str, Any]] = []
    for project_dir in list_project_state_dirs():
        try:
            store = create_store(backend, project_dir)
            short = re.sub(r"-[0-9a-f]{12}$", "", project_dir.name) or project_dir.name
            for job in store.list_jobs():
                rows.append(
                    {
                        "id": job.id,
                        "goal": job.goal,
                        "status": job.status.value,
                        "created_at": job.created_at,
                        "completed_at": job.completed_at,
                        "project": short,
                        **_job_snapshot_meta(job),
                    }
                )
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[:limit]


# --- HTTP layer ------------------------------------------------------------

# The markdown renderer lives in its own constant (rather than inlined in the
# page) so tests can execute it verbatim under node and assert on its output;
# the page concatenates it into its <script> block. Raw string: the backslashes
# belong to the JavaScript, not Python.
RENDERER_JS = r"""
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function md(text){
  if(!text) return "";
  const blocks=[];
  // Stash rendered code behind \uE000<idx>\uE000 sentinels (Unicode private-use
  // area) so later formatting passes can never touch code contents. Bare numeric
  // placeholders would collide with ordinary digits in prose; input sentinels
  // are stripped first so artifact text can't forge a stash reference.
  const stash=h=>{blocks.push(h);return "\uE000"+(blocks.length-1)+"\uE000";};
  let t=String(text).replace(/\uE000/g,"");
  // Real fenced blocks: fence markers on their OWN lines (won't grab inline ``` examples).
  t=t.replace(/^```[^\n]*\n([\s\S]*?)\n```[ \t]*$/gm,(_,c)=>stash("<pre><code>"+esc(c)+"</code></pre>"));
  // Inline code spans matched by backtick-run length (handles `x`, ``a`b``, ```c```).
  t=t.replace(/(`{1,3})([\s\S]*?[^`]|[^`])\1(?!`)/g,(m,run,c)=>{
    if(c===undefined) return m;
    return stash("<code>"+esc(c.replace(/^ +| +$/g,""))+"</code>");
  });
  let s=esc(t);
  s=s.replace(/^#### (.+)$/gm,"<h4>$1</h4>").replace(/^### (.+)$/gm,"<h3>$1</h3>").replace(/^## (.+)$/gm,"<h2>$1</h2>").replace(/^# (.+)$/gm,"<h1>$1</h1>");
  s=s.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/\*(.+?)\*/g,"<em>$1</em>");
  s=s.replace(/^&gt; (.+)$/gm,"<blockquote>$1</blockquote>");
  // Only http(s), site-relative, and fragment hrefs survive; anything else
  // (javascript:, data:, ...) renders as plain text.
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(m,x,u)=>/^(https?:|\/|#)/.test(u)?'<a href="'+u+'">'+x+'</a>':x);
  const lines=s.split("\n"); const out=[]; let i=0;
  const ol=/^\d+\.\s+(.+)$/, ul=/^[*-]\s+(.+)$/;
  // Blank lines between items of the same list ("loose" lists) stay in one
  // <ol>/<ul>, otherwise every numbered item would restart at 1.
  const collect=(re,tag)=>{
    const it=[];
    while(i<lines.length){
      if(re.test(lines[i])){it.push("<li>"+lines[i].replace(re,"$1")+"</li>");i++;}
      else if(lines[i].trim()===""&&i+1<lines.length&&re.test(lines[i+1])){i++;}
      else break;
    }
    out.push("<"+tag+">"+it.join("")+"</"+tag+">");
  };
  while(i<lines.length){
    if(ol.test(lines[i])) collect(ol,"ol");
    else if(ul.test(lines[i])) collect(ul,"ul");
    else {out.push(lines[i]);i++;}
  }
  s=out.join("\n");
  s=s.replace(/\n\n+/g,"</p><p>"); s="<p>"+s+"</p>"; s=s.replace(/<p>\s*<\/p>/g,"");
  s=s.replace(/<p>(\s*<(?:h[1-4]|ul|ol|pre|blockquote)>)/g,"$1").replace(/(<\/(?:h[1-4]|ul|ol|pre|blockquote)>\s*)<\/p>/g,"$1");
  s=s.replace(/\uE000(\d+)\uE000/g,(_,n)=>blocks[+n]);
  return s;
}

// Four-segment progress strip (dispatched -> routing -> workers -> done) shown
// in the job-detail hero. Pure render helper, so it lives alongside esc/md.
function phaseStrip(phase) {
  const p = phase || {};
  const index = p.index != null ? p.index : 0;
  const failed = !!p.failed;
  const active = p.key !== "done" && !failed;
  let segs = "";
  for (let i = 0; i < 4; i++) {
    let cls = "phase-seg";
    if (failed && i === index) cls += " failed";
    else if (i <= index) cls += p.key === "done" ? " done" : " reached";
    if (i === index && active) cls += " active";
    segs += `<div class="${cls}"></div>`;
  }
  const labelCls = failed ? "phase-label failed" : "phase-label";
  return `<div class="phase-strip" title="dispatched → routing → workers → done">`
    + `<div class="phase-segments">${segs}</div>`
    + `<span class="${labelCls}">${esc(p.label || "")}</span></div>`;
}

function jobHeadline(j) {
  return j.label || j.title || j.id;
}
"""

_PAGE_HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Puppetmaster Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
  }
  header {
    padding: 16px 24px;
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky;
    top: 0;
    background: #0d1117;
    z-index: 10;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
  header h1 {
    font-size: 18px;
    font-weight: 600;
    margin: 0;
    color: #f0f6fc;
    letter-spacing: -0.02em;
  }
  .pill {
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .s-complete { background: #1f6f3f; color: #d3f9d8; }
  .s-running, .s-stitching { background: #9e6a03; color: #fff3bf; }
  .s-failed { background: #8b2c2c; color: #ffd6d6; }
  .s-queued, .s-blocked { background: #30363d; color: #c9d1d9; }
  .s-stalled { background: #6e4a9e; color: #e4d9ff; }
  main {
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .goal {
    color: #f0f6fc;
    margin: 0 0 6px;
    font-size: 18px;
    font-weight: 600;
    line-height: 1.4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .job-id-subtitle {
    color: #6e7681;
    margin: 0 0 20px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
  }
  .grid {
    display: grid;
    gap: 20px;
  }
  .card {
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px;
    background: #0f141b;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  }
  .card h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 0 0 14px;
    color: #6e7681;
    font-weight: 700;
  }
  .summary-bar {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid #21262d;
  }
  .summary-chip {
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    background: #21262d;
    color: #c9d1d9;
  }
  .summary-chip.highlight {
    background: #1f6f3f;
    color: #d3f9d8;
  }
  .progress-bar {
    width: 100%;
    height: 8px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #1f6f3f 0%, #2ea043 100%);
    transition: width 0.3s ease;
  }
  .progress-text {
    font-size: 12px;
    color: #8b949e;
    margin-top: 8px;
  }
  .phase-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0 2px;
  }
  .phase-segments {
    display: flex;
    flex: 1;
    gap: 4px;
    min-width: 0;
  }
  .phase-seg {
    flex: 1;
    height: 4px;
    border-radius: 999px;
    background: #30363d;
    transition: background 0.3s ease;
  }
  .phase-seg.reached { background: #58a6ff; }
  .phase-seg.done { background: #2ea043; }
  .phase-seg.failed { background: #f85149; }
  /* Only the segment the swarm currently sits on breathes, so the eye lands on
     where work is happening rather than the whole (static) filled run. */
  .phase-seg.active { animation: phase-pulse 1.6s ease-in-out infinite; }
  .phase-label {
    font-size: 11px;
    color: #8b949e;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .phase-label.failed { color: #f85149; }
  @keyframes phase-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid #161b22;
    vertical-align: top;
  }
  th {
    color: #6e7681;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .muted {
    color: #6e7681;
  }
  .alert {
    background: #2d1416;
    border: 1px solid #8b2c2c;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #ffd6d6;
    font-size: 13px;
  }
  .reroute {
    background: #102a1a;
    border: 1px solid #1f6f3f;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #d3f9d8;
    font-size: 13px;
  }
  .ev {
    color: #6e7681;
    font-size: 12px;
  }
  a {
    color: #58a6ff;
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
  .conf {
    color: #8b949e;
    font-size: 13px;
  }
  .job-list {
    border: 1px solid #21262d;
    border-radius: 8px;
    overflow: hidden;
  }
  .job-row {
    display: grid;
    grid-template-columns: 110px minmax(0, 1fr) max-content max-content;
    gap: 16px;
    align-items: center;
    padding: 11px 16px;
    border-bottom: 1px solid #161b22;
    text-decoration: none;
    color: inherit;
    background: #0f141b;
    transition: background 0.12s ease;
  }
  .job-row:last-child { border-bottom: none; }
  .job-row:hover {
    background: #161b22;
    text-decoration: none;
  }
  .job-row .pill { justify-self: start; }
  .job-headline {
    min-width: 0;
    color: #f0f6fc;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .job-row:hover .job-headline { color: #ffffff; }
  .job-id-tag {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: #6e7681;
    white-space: nowrap;
  }
  .job-row:hover .job-id-tag { color: #8b949e; }
  .job-goal {
    min-width: 0;
    color: #8b949e;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .job-row:hover .job-goal { color: #c9d1d9; }
  .job-project {
    display: inline;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: #8b949e;
    background: #21262d;
    border-radius: 4px;
    padding: 1px 5px;
    margin-right: 6px;
    vertical-align: middle;
    white-space: nowrap;
  }
  .job-time {
    color: #6e7681;
    font-size: 12px;
    white-space: nowrap;
    text-align: right;
  }
  .task-card {
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    background: #161b22;
    margin-bottom: 12px;
    transition: all 0.2s ease;
  }
  .task-card:hover {
    border-color: #30363d;
  }
  .task-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .task-role {
    font-weight: 700;
    font-size: 14px;
    color: #f0f6fc;
  }
  .task-meta {
    font-size: 12px;
    color: #8b949e;
  }
  .task-goal-preview {
    color: #c9d1d9;
    margin: 8px 0;
    font-size: 13px;
    line-height: 1.5;
    cursor: pointer;
    position: relative;
  }
  .task-goal-preview:hover {
    color: #f0f6fc;
  }
  .task-goal-full {
    display: none;
    margin-top: 8px;
    padding: 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .task-goal-full.open {
    display: block;
  }
  .expand-link {
    color: #58a6ff;
    font-size: 12px;
    cursor: pointer;
    user-select: none;
  }
  .expand-link:hover {
    text-decoration: underline;
  }
  .activity-toggle {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
    color: #c9d1d9;
    margin-top: 8px;
    display: inline-block;
    user-select: none;
  }
  .activity-toggle:hover {
    background: #30363d;
  }
  .activity-panel {
    margin-top: 12px;
    padding: 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    display: none;
  }
  .activity-panel.open {
    display: block;
  }
  .activity-item {
    margin: 10px 0;
    padding: 10px;
    background: #161b22;
    border-radius: 6px;
    border-left: 3px solid #30363d;
  }
  .activity-item.verification { border-left-color: #58a6ff; }
  .activity-item.routing { border-left-color: #9e6a03; }
  .activity-item.finding { border-left-color: #1f6f3f; }
  .activity-item.risk { border-left-color: #8b2c2c; }
  .activity-item.decision { border-left-color: #6e4a9e; }
  .activity-text {
    font-size: 13px;
    color: #c9d1d9;
    margin-bottom: 4px;
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    background: #21262d;
    color: #8b949e;
  }
  .badge.success { background: #1f6f3f; color: #d3f9d8; }
  .badge.failure { background: #8b2c2c; color: #ffd6d6; }
  .message-block {
    margin-top: 8px;
    padding: 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    color: #c9d1d9;
  }
  .message-block h1, .message-block h2, .message-block h3, .message-block h4 {
    margin: 16px 0 8px;
    font-weight: 600;
    line-height: 1.3;
    color: #f0f6fc;
  }
  .message-block h1 { font-size: 1.5em; }
  .message-block h2 { font-size: 1.3em; }
  .message-block h3 { font-size: 1.15em; }
  .message-block h4 { font-size: 1em; }
  .message-block p {
    margin: 8px 0;
  }
  .message-block code {
    background: #161b22;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.9em;
  }
  .message-block pre {
    background: #161b22;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .message-block pre code {
    background: none;
    padding: 0;
  }
  .message-block ul, .message-block ol {
    margin: 8px 0;
    padding-left: 24px;
  }
  .message-block li {
    margin: 4px 0;
  }
  .message-block blockquote {
    margin: 8px 0;
    padding-left: 12px;
    border-left: 3px solid #30363d;
    color: #8b949e;
  }
  .message-block a {
    color: #58a6ff;
  }
  .meta-chips {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 11px;
    color: #8b949e;
  }
  .meta-chip {
    padding: 3px 8px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    white-space: nowrap;
  }
  .why-line {
    margin-top: 8px;
    padding: 8px;
    background: #0f141b;
    border-left: 3px solid #58a6ff;
    font-size: 12px;
    color: #8b949e;
    font-style: italic;
  }
  .diff-viewer {
    margin-top: 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    overflow: hidden;
  }
  .diff-header {
    padding: 8px 12px;
    background: #161b22;
    border-bottom: 1px solid #21262d;
    font-size: 12px;
    font-weight: 600;
    color: #c9d1d9;
  }
  .diff-files {
    padding: 6px 12px;
    background: #0f141b;
    border-bottom: 1px solid #21262d;
    font-size: 11px;
    color: #8b949e;
  }
  .diff-content {
    padding: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.4;
    overflow-x: auto;
    max-height: 500px;
    overflow-y: auto;
  }
  .diff-line {
    white-space: pre;
    padding: 0 4px;
  }
  .diff-add { background: rgba(46, 160, 67, 0.15); color: #d3f9d8; }
  .diff-remove { background: rgba(248, 81, 73, 0.15); color: #ffd6d6; }
  .diff-hunk { background: #161b22; color: #58a6ff; font-weight: 600; }
  .diff-file-header { color: #f0f6fc; font-weight: 700; }
  .diff-note {
    padding: 8px 12px;
    background: #2d1416;
    border-top: 1px solid #8b2c2c;
    font-size: 11px;
    color: #ffd6d6;
  }
  .collapsible-section {
    margin-top: 12px;
  }
  .section-toggle {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
  }
  .section-toggle:hover {
    background: #30363d;
  }
  .section-content {
    display: none;
    margin-top: 8px;
  }
  .section-content.open {
    display: block;
  }
  .filter-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .filter-btn {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    color: #c9d1d9;
    user-select: none;
  }
  .filter-btn:hover {
    background: #30363d;
  }
  .filter-btn.active {
    background: #1f6f3f;
    border-color: #1f6f3f;
    color: #d3f9d8;
  }
  .home-link {
    color: #8b949e; text-decoration: none; font-size: 13px;
    padding: 3px 9px; border: 1px solid #21262d; border-radius: 6px; white-space: nowrap;
  }
  .home-link:hover { color: #c9d1d9; border-color: #30363d; background: #161b22; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  .swarm-hero { margin-bottom: 14px; }
  .swarm-hero-top {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .swarm-title {
    font-size: 16px;
    font-weight: 600;
    color: #f0f6fc;
    min-width: 0;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .swarm-hero-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .swarm-model-badge {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
    color: #58a6ff;
    background: rgba(88, 166, 255, 0.12);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 6px;
    padding: 3px 8px;
  }
  .swarm-chip {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 3px 8px;
  }
  .swarm-chip.swarm-adapter { text-transform: lowercase; }
  .swarm-metrics {
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }
  .swarm-cost {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 18px;
    font-weight: 700;
    color: #2ea043;
  }
  .swarm-tokens {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px;
    color: #8b949e;
  }
  .swarm-verification {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #58a6ff;
    background: rgba(88, 166, 255, 0.1);
    border: 1px solid rgba(88, 166, 255, 0.2);
    border-radius: 6px;
    padding: 4px 10px;
  }
  .swarm-subhead {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6e7681;
    font-weight: 700;
    margin: 16px 0 10px;
  }
  .routing-rollup { display: flex; flex-direction: column; gap: 10px; }
  .routing-card {
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    background: #161b22;
  }
  .routing-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 6px;
  }
  .routing-model {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    color: #f0f6fc;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .routing-cost {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    color: #2ea043;
    flex-shrink: 0;
  }
  .routing-role {
    font-size: 11px;
    color: #6e7681;
    margin-bottom: 6px;
  }
  .routing-summary {
    font-size: 12px;
    color: #8b949e;
    line-height: 1.5;
    margin-bottom: 6px;
  }
  .alts-toggle {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    font-size: 11px;
    color: #6e7681;
    font-family: inherit;
  }
  .alts-toggle:hover { color: #8b949e; }
  .alts-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }
  .alt-chip {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10px;
    color: #8b949e;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 2px 7px;
    cursor: default;
  }
  .job-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 10;
    display: none;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 24px;
    background: #161b22;
    border-top: 1px solid #21262d;
    box-shadow: 0 -1px 3px rgba(0,0,0,0.3);
    font-size: 12px;
    color: #8b949e;
  }
  .job-footer.visible { display: flex; }
  .job-footer-totals {
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-wrap: wrap;
  }
  .job-footer-cost {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 700;
    color: #2ea043;
  }
  .job-footer-tokens {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 600;
    color: #c9d1d9;
  }
  .job-footer-model {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: #58a6ff;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  main.has-footer { padding-bottom: 72px; }
  /* Mobile / narrow-viewport pass (dashboard --mobile, phones over Tailscale or
     LAN). The desktop job-row grid, nowrap cells, wide tables and the fixed
     footer all overflow a phone: collapse the grid, let text wrap, and let wide
     tables scroll horizontally instead of blowing out the viewport. */
  @media (max-width: 640px) {
    body { padding: 0; }
    /* Collapse the desktop header into a thin app bar: the wrapping
       "updated …" timestamp and raw job-id add a wasted top band on a phone.
       Drop the noise and shrink the brand so the job list owns the screen. */
    header { flex-wrap: wrap; gap: 8px; padding: 8px 12px; box-shadow: none; }
    header h1 { font-size: 15px; }
    #updated, #jobid { display: none; }
    .home-link { padding: 4px 8px; font-size: 12px; }
    /* Full-bleed content: cards run edge-to-edge like a native mobile list
       instead of floating in a centered, padded desktop column. */
    main { padding: 10px 10px 0; max-width: none; }
    .goal { font-size: 15px; }
    /* minmax(0,1fr) is load-bearing: a bare "1fr" is minmax(auto,1fr), whose
       auto min is min-content, so nested flex/nowrap content (the goal line,
       routing cards, phase strip) forced the single column ~810px wide -- 2x the
       phone -- and the page scrolled sideways, clipping every right-edge
       cost/pill. minmax(0,1fr) lets the track shrink to the viewport so content
       wraps inside instead of overflowing. */
    .grid { grid-template-columns: minmax(0, 1fr); gap: 12px; }
    html, body { overflow-x: hidden; }
    /* Mobile job rows read as tidy stacked cards: status pill + time share the
       top line, then the headline and the (truncated) job id line up flush-left
       beneath -- so a long job hash can't stretch the row or knock the columns
       out of alignment. */
    .job-row {
      grid-template-columns: 1fr auto;
      grid-template-areas:
        "pill time"
        "head head"
        "id   id";
      gap: 6px 10px;
      padding: 14px 16px;
      align-items: center;
    }
    .job-row .pill { grid-area: pill; justify-self: start; }
    .job-time { grid-area: time; justify-self: end; }
    .job-headline { grid-area: head; white-space: normal; }
    .job-id-tag {
      grid-area: id;
      justify-self: start;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .job-footer { padding: 10px 14px; font-size: 11px; gap: 10px; flex-wrap: wrap; }
    .job-footer-totals { gap: 10px; }
    main.has-footer { padding-bottom: 104px; }
    .phase-strip { gap: 8px; }
    .phase-label { font-size: 10px; }
    /* Let the job view's routing + phase boxes shrink below their content
       width, so nothing forces the single column wider than the phone; wrap
       long model tokens instead of pushing the layout sideways. */
    .phase-strip, .routing-rollup, .routing-card, .routing-card-head,
    .routing-role, .routing-summary {
      min-width: 0;
      max-width: 100%;
    }
    .routing-role, .routing-summary { overflow-wrap: anywhere; }
  }
</style>
</head>
<body>
<header>
  <a href="/" class="home-link" id="home" title="All jobs">← Jobs</a>
  <h1>Puppetmaster</h1>
  <span id="status" class="pill s-queued">…</span>
  <span class="muted mono" id="jobid"></span>
  <span class="muted" id="updated" style="margin-left:auto"></span>
</header>
<main>
  <p class="goal" id="goal" title=""></p>
  <p class="job-id-subtitle mono" id="jobidsub"></p>
  <div class="grid" id="content"></div>
</main>
<footer id="job-footer" class="job-footer">
  <span class="job-footer-label">Job total</span>
  <div class="job-footer-totals">
    <span>Cost: <strong class="job-footer-cost" id="footer-cost">$0.0000</strong></span>
    <span>Tokens: <strong class="job-footer-tokens" id="footer-tokens">0</strong></span>
    <span class="job-footer-model" id="footer-model"></span>
  </div>
</footer>
<script>
"""

_PAGE_APP_JS = r"""
const qs = new URLSearchParams(location.search);
let jobId = qs.get("job");
const viewParam = qs.get("view");
// Two views: the Jobs index (default) and a single job's detail (?job=<id>).
// ?view=jobs forces the index even when a stale job id lingers in the URL.
let activeView = viewParam === "jobs" ? "jobs"
  : jobId ? "job"
  : "jobs";
let expandedTasks = new Set();
let expandedGoals = new Set();
let expandedSections = new Set();
let expandedAlts = new Set();
let taskFilter = "all";
let lastContent = "";

function pill(s) {
  return `<span class="pill s-${esc(s)}">${esc(s)}</span>`;
}

function truncateGoal(goal, maxChars = 120) {
  if (!goal || goal.length <= maxChars) return esc(goal);
  const firstLine = goal.split("\n")[0];
  if (firstLine.length <= maxChars) return esc(firstLine);
  return esc(firstLine.substring(0, maxChars)) + "...";
}

function fmtAgo(iso) {
  const t = Date.parse(iso || "");
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

let jobFilter = "all";
window.setJobFilter = function(filter) {
  jobFilter = filter;
  loadIndex();
};

async function loadIndex() {
  const r = await fetch("/api/jobs");
  const jobs = await r.json();
  document.getElementById("goal").textContent = "";
  const _home = document.getElementById("home"); if (_home) _home.style.display = "none";
  const _jid = document.getElementById("jobid"); if (_jid) _jid.textContent = "";
  const _jids = document.getElementById("jobidsub"); if (_jids) _jids.textContent = "";
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-queued">jobs</span>';
  const _footer = document.getElementById("job-footer");
  if (_footer) _footer.classList.remove("visible");
  const _main = document.querySelector("main");
  if (_main) _main.classList.remove("has-footer");

  const counts = {};
  for (const j of jobs) counts[j.status] = (counts[j.status] || 0) + 1;
  const shown = jobFilter === "all" ? jobs : jobs.filter(j => j.status === jobFilter);

  let html = '<div class="card"><h2>Jobs</h2>';
  html += '<div class="filter-bar">';
  html += `<button class="filter-btn ${jobFilter === "all" ? "active" : ""}" onclick="setJobFilter('all')">All (${jobs.length})</button>`;
  for (const status of Object.keys(counts).sort()) {
    html += `<button class="filter-btn ${jobFilter === status ? "active" : ""}" onclick="setJobFilter('${esc(status)}')">${esc(status)} (${counts[status]})</button>`;
  }
  html += '</div>';

  if (!shown.length) {
    html += '<p class="muted">No jobs in this workspace state dir yet.</p>';
  } else {
    html += '<div class="job-list">';
    for (const j of shown) {
      const headline = esc(jobHeadline(j));
      html += `<a class="job-row" href="?job=${encodeURIComponent(j.id)}">
        ${pill(j.status)}
        <span class="job-headline" title="${esc(j.goal)}">${j.project ? `<span class="job-project">${esc(j.project)}</span>` : ""}${headline}</span>
        <span class="job-id-tag">${esc(j.id)}</span>
        <span class="job-time">${fmtAgo(j.created_at)}</span>
      </a>`;
    }
    html += '</div>';
  }
  html += '</div>';

  // Same no-op guard as the job view: skip DOM writes (and the hover/scroll
  // disruption they cause) when nothing changed between polls.
  if (html !== lastContent) {
    document.getElementById("content").innerHTML = html;
    lastContent = html;
  }
}

function rows(items, cols) {
  if (!items || !items.length) return '<tr><td class="muted" colspan="' + cols.length + '">None</td></tr>';
  return items.map(it => '<tr>' + cols.map(c => '<td>' + c(it) + '</td>').join('') + '</tr>').join('');
}

function renderDiff(diff) {
  if (!diff || !diff.unified_diff) return '';

  let html = '<div class="diff-viewer">';
  html += '<div class="diff-header">Code Changes</div>';

  if (diff.files && diff.files.length > 0) {
    html += `<div class="diff-files">Files: ${esc(diff.files.join(', '))}</div>`;
  }

  html += '<div class="diff-content">';
  const lines = diff.unified_diff.split('\n');
  for (const line of lines) {
    let className = 'diff-line';
    if (line.startsWith('+++') || line.startsWith('---')) {
      className += ' diff-file-header';
    } else if (line.startsWith('@@')) {
      className += ' diff-hunk';
    } else if (line.startsWith('+')) {
      className += ' diff-add';
    } else if (line.startsWith('-')) {
      className += ' diff-remove';
    }
    html += `<div class="${className}">${esc(line)}</div>`;
  }
  html += '</div>';

  if (diff.truncated) {
    html += `<div class="diff-note">Diff truncated (${diff.total_chars} total chars)</div>`;
  }

  html += '</div>';
  return html;
}

function renderTask(task) {
  const isExpanded = expandedTasks.has(task.id);
  const goalExpanded = expandedGoals.has(task.id);
  const preview = truncateGoal(task.instruction);
  const needsExpand = task.instruction && task.instruction.length > 120;
  const safeId = esc(task.id);

  let html = `<div class="task-card">
    <div class="task-header">
      <span class="task-role">${esc(task.role)}</span>
      ${pill(task.status)}
      <span class="task-meta">${esc(task.adapter)}</span>
      ${task.model ? `<span class="task-meta">→ ${esc(task.model)}</span>` : ''}
      <span class="task-meta">tries: ${task.attempts}</span>
    </div>
    <div class="task-goal-preview">
      ${preview}
      ${needsExpand ? ` <span class="expand-link" onclick="toggleGoal('${safeId}')">[${goalExpanded ? 'less' : 'more'}]</span>` : ''}
    </div>`;

  if (needsExpand) {
    html += `<div class="task-goal-full ${goalExpanded ? 'open' : ''}" id="goal-${safeId}">
      ${esc(task.instruction)}
    </div>`;
  }

  if (task.activity && task.activity.length > 0) {
    html += `<div class="activity-toggle" onclick="toggleActivity('${safeId}')">
      ${isExpanded ? '▼' : '▶'} Thinking & Output (${task.activity.length})
    </div>
    <div class="activity-panel ${isExpanded ? 'open' : ''}" id="activity-${safeId}">`;

    for (const act of task.activity) {
      html += `<div class="activity-item ${esc(act.type)}">
        <div class="activity-text">${esc(act.text || '')}</div>`;

      if (act.result) {
        const badgeClass = act.result === 'passed' ? 'success' : 'failure';
        html += `<span class="badge ${badgeClass}">${esc(act.result)}</span> `;
      }
      if (act.confidence != null) {
        html += `<span class="conf">confidence: ${act.confidence}</span>`;
      }

      if (act.message && act.message.length > 20 && act.message !== act.text) {
        html += `<div class="message-block">${md(act.message)}</div>`;
      }

      if (act.meta && Object.keys(act.meta).length) {
        html += '<div class="meta-chips">';
        if (act.meta.model) html += `<span class="meta-chip">${esc(act.meta.model)}</span>`;
        if (act.meta.cost_usd != null) html += `<span class="meta-chip">$${act.meta.cost_usd.toFixed(6)}</span>`;
        const approx = act.meta.tokens_estimated ? "~" : "";
        if (act.meta.tokens_in != null) html += `<span class="meta-chip">↑ ${approx}${act.meta.tokens_in}</span>`;
        if (act.meta.tokens_out != null) html += `<span class="meta-chip">↓ ${approx}${act.meta.tokens_out}</span>`;
        if (act.meta.num_turns != null) html += `<span class="meta-chip">${act.meta.num_turns} turns</span>`;
        if (act.meta.duration_ms != null) html += `<span class="meta-chip">${(act.meta.duration_ms / 1000).toFixed(1)}s</span>`;
        html += '</div>';
      }

      if (act.why) {
        html += `<div class="why-line">${md(act.why)}</div>`;
      }

      if (act.evidence && act.evidence.length > 0) {
        html += `<div class="ev">${esc(act.evidence.join(', '))}</div>`;
      }

      if (act.diff) {
        html += renderDiff(act.diff);
      }

      html += '</div>';
    }
    html += '</div>';
  }

  html += '</div>';
  return html;
}

window.toggleGoal = function(taskId) {
  if (expandedGoals.has(taskId)) {
    expandedGoals.delete(taskId);
  } else {
    expandedGoals.add(taskId);
  }
  loadJob();
};

window.toggleActivity = function(taskId) {
  if (expandedTasks.has(taskId)) {
    expandedTasks.delete(taskId);
  } else {
    expandedTasks.add(taskId);
  }
  loadJob();
};

window.toggleSection = function(sectionId) {
  if (expandedSections.has(sectionId)) {
    expandedSections.delete(sectionId);
  } else {
    expandedSections.add(sectionId);
  }
  loadJob();
};

window.toggleAlts = function(taskId) {
  if (expandedAlts.has(taskId)) {
    expandedAlts.delete(taskId);
  } else {
    expandedAlts.add(taskId);
  }
  rerenderActive();
};

window.setTaskFilter = function(filter) {
  taskFilter = filter;
  loadJob();
};

function renderCollapsibleSection(title, id, items) {
  const isExpanded = expandedSections.has(id);
  let html = `<div class="collapsible-section">
    <div class="section-toggle" onclick="toggleSection('${id}')">
      ${isExpanded ? '▼' : '▶'} ${title} (${items ? items.length : 0})
    </div>
    <div class="section-content ${isExpanded ? 'open' : ''}" id="section-${id}">`;

  if (items && items.length > 0) {
    html += '<table><tr><th>Statement</th><th>Confidence</th><th>Evidence</th></tr>';
    html += rows(items, [
      it => md(it.statement),
      it => '<span class="conf">' + (it.confidence != null ? it.confidence.toFixed(2) : '') + '</span>',
      it => '<span class="ev">' + esc((it.evidence || []).join(', ')) + '</span>'
    ]);
    html += '</table>';
  } else {
    html += '<p class="muted">None</p>';
  }

  html += '</div></div>';
  return html;
}

function summarizeRouting(entry) {
  const policy = entry.policy || "";
  const reason = entry.reason || "";
  const planBilled = /plan-billed|in-subscription/i.test(reason);
  const lead = {
    balanced: "Right-sized: cheapest model that clears the task's need",
    cheap: "Cheapest available model",
    quality: "Highest-capability model for the task",
    escalating: "Cheapest sufficient model, escalates if it stalls",
  };
  const base = lead[policy] || "Router pick";
  return planBilled ? base + " \u00b7 plan-billed, no marginal cost" : base;
}

function renderRoutingRollup(rollup) {
  if (!rollup || !rollup.length) {
    return '<p class="muted">No routing decisions yet.</p>';
  }
  let html = '<div class="routing-rollup">';
  for (const entry of rollup) {
    const altKey = esc(entry.task_id);
    const altsExpanded = expandedAlts.has(entry.task_id);
    html += '<div class="routing-card">';
    if (entry.role) {
      html += `<div class="routing-role">${esc(entry.role)}</div>`;
    }
    html += `<div class="routing-card-head">
      <span class="routing-model" title="${esc(entry.model_id)}">${esc(entry.model_id || "Unknown model")}</span>
      <span class="routing-cost">$${(entry.estimated_cost_usd || 0).toFixed(4)}</span>
    </div>`;
    html += `<div class="routing-summary">${esc(summarizeRouting(entry))}</div>`;
    if (entry.rejected_count > 0) {
      html += `<button type="button" class="alts-toggle" onclick="toggleAlts('${altKey}')">${altsExpanded ? "\u25bc" : "\u25b6"} ${entry.rejected_count} alternatives considered</button>`;
      if (altsExpanded) {
        html += '<div class="alts-list">';
        for (const rej of entry.rejected) {
          html += `<span class="alt-chip" title="${esc(rej.reason)}">${esc(rej.id)}</span>`;
        }
        html += '</div>';
      }
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}

function updateJobFooter(d) {
  const footer = document.getElementById("job-footer");
  const main = document.querySelector("main");
  if (!footer) return;
  footer.classList.add("visible");
  if (main) main.classList.add("has-footer");
  const costEl = document.getElementById("footer-cost");
  const tokensEl = document.getElementById("footer-tokens");
  const modelEl = document.getElementById("footer-model");
  if (costEl) costEl.textContent = "$" + (d.cost.total_estimated_cost_usd || 0).toFixed(4);
  if (tokensEl) tokensEl.textContent = (d.tokens_total || 0).toLocaleString();
  if (modelEl) modelEl.textContent = d.primary_model || "";
}

async function loadJob() {
  const r = await fetch("/api/job?id=" + encodeURIComponent(jobId));
  if (r.status === 404) {
    document.getElementById("content").innerHTML = '<div class="card">Job not found.</div>';
    const _footer = document.getElementById("job-footer");
    if (_footer) _footer.classList.remove("visible");
    const _main = document.querySelector("main");
    if (_main) _main.classList.remove("has-footer");
    return;
  }
  const d = await r.json();
  updateJobFooter(d);
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-' + esc(d.job.status) + '">' + esc(d.job.status) + '</span>';
  document.getElementById("jobid").textContent = "";
  const headline = d.job.label || d.job.title || d.job.id;
  const _goalEl = document.getElementById("goal");
  _goalEl.textContent = headline.length > 110 ? headline.slice(0, 110) + "…" : headline;
  _goalEl.title = d.job.goal;
  const _jobSub = document.getElementById("jobidsub");
  if (_jobSub) _jobSub.textContent = d.job.id;
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();

  let html = "";

  const prog = d.progress || {};
  const totalTasks = Object.values(prog).reduce((a, b) => a + b, 0);
  const completeTasks = prog.complete || 0;
  const runningTasks = prog.running || 0;
  const failedTasks = prog.failed || 0;
  const queuedTasks = prog.queued || 0;
  const progressPct = totalTasks > 0 ? (completeTasks / totalTasks * 100) : 0;

  html += `<div class="card"><h2>Swarm Overview</h2>
    <div class="swarm-hero">
      <div class="swarm-hero-top">
        <span class="swarm-title" title="${esc(d.job.goal)}">${esc(headline)}</span>
        ${pill(d.job.status)}
      </div>
      <div class="swarm-hero-meta">
        ${d.primary_model ? `<span class="swarm-model-badge" title="Primary model">${esc(d.primary_model)}</span>` : ""}
        ${d.worker_count ? `<span class="swarm-chip">${d.worker_count} worker${d.worker_count !== 1 ? "s" : ""}</span>` : ""}
        ${d.primary_adapter ? `<span class="swarm-chip swarm-adapter">${esc(d.primary_adapter)}</span>` : ""}
        ${(d.evaluator_epoch || []).map(e => `<span class="swarm-chip" title="Evaluator epoch">${esc(e.slot_id)}@v${e.version} (${esc(e.role)})</span>`).join("")}
      </div>
    </div>
    <div class="swarm-metrics">
      <span class="swarm-cost">$${(d.cost.total_estimated_cost_usd || 0).toFixed(4)}</span>
      <span class="swarm-tokens">${(d.tokens_total || 0).toLocaleString()}t</span>
      ${d.verification_score != null ? `<span class="swarm-verification">VERIFICATION ${Math.round(d.verification_score * 100)}%</span>` : ""}
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width: ${progressPct}%"></div></div>
    <div class="progress-text">${completeTasks} / ${totalTasks} complete</div>
    ${phaseStrip(d.phase)}
    <h3 class="swarm-subhead">Routing</h3>
    ${renderRoutingRollup(d.routing_rollup || [])}
  </div>`;

  html += `<div class="card">
    <div class="summary-bar">
      <span class="summary-chip highlight">${totalTasks} tasks</span>
      ${completeTasks > 0 ? `<span class="summary-chip s-complete">${completeTasks} complete</span>` : ''}
      ${runningTasks > 0 ? `<span class="summary-chip s-running">${runningTasks} running</span>` : ''}
      ${failedTasks > 0 ? `<span class="summary-chip s-failed">${failedTasks} failed</span>` : ''}
      ${queuedTasks > 0 ? `<span class="summary-chip s-queued">${queuedTasks} queued</span>` : ''}
      <span class="summary-chip">$${(d.cost.total_estimated_cost_usd || 0).toFixed(6)}</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width: ${progressPct}%"></div></div>
    <div class="progress-text">${completeTasks} / ${totalTasks} complete</div>`;

  const bm = Object.entries(d.cost.by_model || {});
  if (bm.length) {
    html += '<table><tr><th>model</th><th>calls</th><th>cost</th></tr>' +
      bm.map(([m, v]) => '<tr><td>' + esc(m) + '</td><td>' + v.calls + '</td><td>$' + v.estimated_cost_usd.toFixed(6) + '</td></tr>').join('') + '</table>';
  }
  html += '</div>';

  if (d.alerts && d.alerts.length) {
    html += '<div class="card"><h2>Alerts — action required</h2>' +
      d.alerts.map(a => '<div class="alert">' + esc(a.replace(/^- /, '')) + '</div>').join('') + '</div>';
  }
  if (d.reroutes && d.reroutes.length) {
    html += '<div class="card"><h2>Reroutes (fallback &amp; escalation)</h2>' +
      d.reroutes.map(x => '<div class="reroute"><b>' + esc(x.task_id) + '</b> → ' + esc(x.reason || '') + '</div>').join('') + '</div>';
  }

  if (d.tasks && d.tasks.length) {
    const sortedTasks = d.tasks.slice().sort((a, b) => {
      const order = { running: 0, queued: 1, blocked: 2, complete: 3, failed: 4, stalled: 5 };
      return (order[a.status] || 99) - (order[b.status] || 99);
    });

    const filteredTasks = taskFilter === "failed" ? sortedTasks.filter(t => t.status === "failed") :
      taskFilter === "running" ? sortedTasks.filter(t => t.status === "running" || t.status === "queued") :
      sortedTasks;

    html += '<div class="card"><h2>Tasks</h2>';
    html += '<div class="filter-bar">';
    html += `<button class="filter-btn ${taskFilter === "all" ? "active" : ""}" onclick="setTaskFilter('all')">All (${d.tasks.length})</button>`;
    const failedCount = d.tasks.filter(t => t.status === "failed").length;
    const runningCount = d.tasks.filter(t => t.status === "running" || t.status === "queued").length;
    if (failedCount > 0) html += `<button class="filter-btn ${taskFilter === "failed" ? "active" : ""}" onclick="setTaskFilter('failed')">Failed (${failedCount})</button>`;
    if (runningCount > 0) html += `<button class="filter-btn ${taskFilter === "running" ? "active" : ""}" onclick="setTaskFilter('running')">Active (${runningCount})</button>`;
    html += '</div>';
    for (const task of filteredTasks) {
      html += renderTask(task);
    }
    html += '</div>';
  }

  html += '<div class="card">';
  html += renderCollapsibleSection('Findings', 'findings', d.artifacts.finding);
  html += renderCollapsibleSection('Risks', 'risks', d.artifacts.risk);
  html += renderCollapsibleSection('Decisions', 'decisions', d.artifacts.decision);
  html += renderCollapsibleSection('Verifications', 'verifications', d.artifacts.verification);
  html += '</div>';

  // Only touch the DOM when the markup actually changed: the 1.5s poll then
  // never flickers, and scroll position survives genuine updates.
  if (html !== lastContent) {
    const _sy = window.scrollY;
    document.getElementById("content").innerHTML = html;
    lastContent = html;
    window.scrollTo(0, _sy);
  }
}

// Re-render whichever view is live -- used by shared toggles (e.g. routing
// alternatives) that the job view reuses.
function rerenderActive() {
  if (activeView === "job") loadJob();
  else loadIndex();
}

async function tick() {
  try {
    if (activeView === "job") await loadJob();
    else await loadIndex();
  } catch (e) { /* keep polling; transient during writes */ }
}

tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""

INDEX_HTML = _PAGE_HEAD + RENDERER_JS + _PAGE_APP_JS


def make_handler(store_factory: Callable[[], SwarmStore], *, all_projects: bool = False, backend: str = "sqlite"):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence default logging
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: Any) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path in ("/", "/index.html"):
                    self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/api/jobs":
                    if all_projects:
                        self._json(200, list_all_projects_snapshot(backend=backend))
                    else:
                        self._json(200, list_jobs_snapshot(store_factory()))
                    return
                if path == "/api/job":
                    params = parse_qs(parsed.query)
                    job_id = (params.get("id") or [""])[0]
                    if not job_id:
                        self._json(400, {"error": "missing id"})
                        return
                    if not _valid_job_id(job_id):
                        # Reject anything that isn't a plain job id before it can
                        # reach the store path join (no traversal / absolute paths).
                        self._json(400, {"error": "invalid id"})
                        return
                    try:
                        if all_projects:
                            from puppetmaster.state import find_state_dir_for_job
                            from puppetmaster.store_factory import create_store as _cs
                            found = find_state_dir_for_job(job_id)
                            store = _cs(backend, found) if found else store_factory()
                        else:
                            store = store_factory()
                        self._json(200, build_job_snapshot(store, job_id))
                    except (FileNotFoundError, KeyError):
                        self._json(404, {"error": "job not found", "id": job_id})
                    return
                self._json(404, {"error": "not found"})
            except Exception:
                # Never leak internals (paths, backend details) to the client;
                # a request that trips an unexpected error gets a generic 500.
                self._json(500, {"error": "internal error"})

    return _Handler


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


def _tailscale_binary() -> Optional[str]:
    """Locate the tailscale CLI, including the macOS GUI-app path.

    The Mac App Store / standalone Tailscale.app ships its CLI *inside the app
    bundle* and never adds it to PATH, so a plain ``shutil.which`` misses it —
    which silently demotes ``--mobile`` to a same-network-only LAN address.
    """
    import shutil

    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in (
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
        "/usr/local/bin/tailscale",
        "/opt/homebrew/bin/tailscale",
    ):
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _detect_tailscale_ip() -> Optional[str]:
    """Return this host's Tailscale IPv4, or None if Tailscale isn't up.

    Tailscale is the blessed remote path: a stable 100.x address reachable from
    your phone anywhere, without exposing the board to the public internet.
    """
    import subprocess

    binary = _tailscale_binary()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def _detect_lan_ip() -> Optional[str]:
    """Best-effort primary LAN IPv4 for this host.

    Opens a UDP socket toward a public address only so the OS picks the outbound
    interface — no packet is sent — then reads back the local address it bound.
    Returns None on an isolated host (or when only loopback is available).
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    if not ip or ip.startswith("127."):
        return None
    return ip


def resolve_mobile_host() -> "tuple[Optional[str], str]":
    """Pick the best phone-reachable bind address, preferring Tailscale.

    Returns ``(ip, source)`` where ``source`` is ``"tailscale"`` or ``"lan"``,
    or ``(None, "none")`` when neither is available.
    """
    ip = _detect_tailscale_ip()
    if ip:
        return ip, "tailscale"
    ip = _detect_lan_ip()
    if ip:
        return ip, "lan"
    return None, "none"


def qr_ascii(url: str) -> Optional[str]:
    """Return a scannable ASCII-art QR of ``url``, or None if 'qrcode' is absent."""
    try:
        import qrcode  # optional; not a hard dependency
    except ImportError:
        return None
    try:
        import io

        code = qrcode.QRCode(border=1)
        code.add_data(url)
        code.make(fit=True)
        buffer = io.StringIO()
        code.print_ascii(out=buffer, invert=True)
        return buffer.getvalue()
    except Exception:
        return None


def write_qr_png(url: str, path: Union[Path, str]) -> bool:
    """Write a scannable PNG QR of ``url`` to ``path``.

    Returns False when the optional ``qrcode`` package (or its Pillow backend,
    needed for PNG) is unavailable — callers fall back to the ASCII QR or the
    bare URL. An image is what lets the agent embed the code inline in chat.
    """
    try:
        import qrcode  # optional; not a hard dependency
    except ImportError:
        return False
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        qrcode.make(url).save(str(path))
        return True
    except Exception:
        # Pillow missing (no PNG backend) or any render failure.
        return False


def _print_qr(url: str) -> None:
    """Render a scannable QR of the URL, if the optional 'qrcode' package is present."""
    ascii_art = qr_ascii(url)
    if ascii_art is None:
        print("     (--qr needs the QR extra: pip install 'puppetmaster-ai[mobile]')")
        return
    print(ascii_art)


def print_mobile_banner(
    host: str,
    port: int,
    source: str,
    *,
    job_id: Optional[str] = None,
    qr: bool = False,
) -> None:
    """Print the phone URL (and an optional QR) with a one-line security note."""
    url = f"http://{host}:{port}/" + (f"?job={job_id}" if job_id else "")
    where = "Tailscale" if source == "tailscale" else "LAN"
    print("")
    print(f"  Open on your phone ({where}):")
    print(f"     {url}")
    if source == "lan":
        print("     (phone must share this network; Tailscale reaches it from anywhere)")
    print("     Unauthenticated + read-only — keep it to trusted networks.")
    if qr:
        _print_qr(url)
    print("")


# ---------------------------------------------------------------------------
# Background ("just runs like the backend") lifecycle
#
# The pilot should be able to start the phone-reachable board once, hand back a
# link/QR, and walk away — no second terminal held open. A detached child does
# the serving; a small JSON runfile in the state dir records it so we can report
# status and stop it later without hunting for the process.
# ---------------------------------------------------------------------------

_RUNFILE_NAME = "dashboard.run.json"


def dashboard_runfile(state_dir: Union[Path, str]) -> Path:
    """Path of the marker that tracks a detached background dashboard."""
    return Path(state_dir) / _RUNFILE_NAME


def write_dashboard_runfile(state_dir: Union[Path, str], info: dict) -> None:
    """Persist the background server's pid/host/port/url so it can be managed."""
    path = dashboard_runfile(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except OSError:
        pass


def read_dashboard_runfile(state_dir: Union[Path, str]) -> Optional[dict]:
    """Read the background dashboard marker, or None when absent/corrupt."""
    try:
        return json.loads(dashboard_runfile(state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_dashboard_runfile(state_dir: Union[Path, str]) -> None:
    """Remove the background dashboard marker (best effort)."""
    try:
        dashboard_runfile(state_dir).unlink()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """True when ``pid`` names a live process.

    ``os.kill(pid, 0)`` is the POSIX liveness idiom, but on Windows ``os.kill``
    routes any non-CTRL signal through ``TerminateProcess`` — signal ``0`` would
    *kill* the target, and a bad pid raises ``OSError(WinError 87)`` rather than
    ``ProcessLookupError``. So Windows uses a non-destructive ``OpenProcess`` +
    ``GetExitCodeProcess`` probe instead.
    """
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def dashboard_alive(host: str = "127.0.0.1", port: int = 8787, *, timeout: float = 1.0) -> bool:
    """True when a dashboard already answers on ``host:port``."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/jobs", timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def stop_background_dashboard(state_dir: Union[Path, str]) -> dict:
    """Stop the detached background dashboard recorded for ``state_dir``.

    Idempotent: a missing or stale runfile is reported as "nothing to stop"
    rather than an error, so the pilot can call it defensively.
    """
    import signal

    info = read_dashboard_runfile(state_dir)
    if not info:
        return {"stopped": False, "reason": "no background dashboard is tracked here"}

    pid = int(info.get("pid") or 0)
    result: dict = {"stopped": False, "pid": pid, "url": info.get("url")}
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            result["stopped"] = True
        except OSError as exc:
            result["reason"] = f"could not signal pid {pid}: {exc}"
    else:
        result["stopped"] = True
        result["reason"] = "process was not running (cleared stale runfile)"
    clear_dashboard_runfile(state_dir)
    return result


def serve(
    state_dir: Union[Path, str],
    *,
    backend: str = "sqlite",
    job_id: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    serve_forever: bool = True,
    allow_external: bool = False,
    all_projects: bool = False,
):
    """Start the dashboard HTTP server. Returns the server (already bound).

    ``serve_forever=False`` binds and returns without blocking — used by tests
    to drive a single request, then ``server.shutdown()``.

    The dashboard serves durable job state with no authentication, so binding
    to a non-loopback interface would expose it to the local network. That is
    refused unless ``allow_external`` (or
    ``PUPPETMASTER_DASHBOARD_ALLOW_EXTERNAL=1``) is set, making the exposure an
    explicit, deliberate choice.
    """
    from http.server import ThreadingHTTPServer

    from puppetmaster.store_factory import create_store

    allow_external = allow_external or os.environ.get(
        "PUPPETMASTER_DASHBOARD_ALLOW_EXTERNAL", ""
    ).strip() not in ("", "0", "false", "False")
    if host.strip().lower() not in _LOOPBACK_HOSTS and not allow_external:
        raise ValueError(
            f"refusing to bind the unauthenticated dashboard to non-loopback host "
            f"{host!r}; it would expose job state to the network. Pass "
            f"allow_external=True (CLI: --allow-external) or set "
            f"PUPPETMASTER_DASHBOARD_ALLOW_EXTERNAL=1 to override."
        )

    resolved = Path(state_dir)

    def store_factory() -> SwarmStore:
        # Fresh store per request keeps SQLite connections thread-local and
        # avoids cross-thread cursor reuse under the threading server.
        return create_store(backend, resolved)

    httpd = ThreadingHTTPServer((host, port), make_handler(store_factory, all_projects=all_projects, backend=backend))
    url = f"http://{host}:{port}/" + (f"?job={job_id}" if job_id else "")
    print(f"Puppetmaster dashboard: {url}")
    if all_projects:
        print("Serving all projects (--all-projects mode)")
    else:
        print("Reading durable state from:", resolved)
    print("Press Ctrl-C to stop.")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    if not serve_forever:
        return httpd
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        httpd.shutdown()
    return httpd
