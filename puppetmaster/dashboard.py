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

from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.store import SwarmStore

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

    return {
        "job": {
            "id": job.id,
            "goal": job.goal,
            "status": job.status.value,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        },
        "tasks": task_rows,
        "counts": counts,
        "progress": progress,
        "artifacts": grouped,
        "reroutes": reroutes,
        "cost": cost_rollup(artifacts),
        "alerts": collect_alerts(artifacts),
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
            }
        )
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
    color: #8b949e;
    margin: 0 0 20px;
    font-size: 15px;
    line-height: 1.5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
    grid-template-columns: 110px max-content minmax(0, 1fr) max-content;
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
  .job-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    color: #58a6ff;
    white-space: nowrap;
  }
  .job-goal {
    min-width: 0;
    color: #8b949e;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .job-row:hover .job-goal { color: #c9d1d9; }
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
  <div class="grid" id="content"></div>
</main>
<script>
"""

_PAGE_APP_JS = r"""
const qs = new URLSearchParams(location.search);
let jobId = qs.get("job");
let expandedTasks = new Set();
let expandedGoals = new Set();
let expandedSections = new Set();
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
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-queued">jobs</span>';

  const counts = {};
  for (const j of jobs) counts[j.status] = (counts[j.status] || 0) + 1;
  const shown = jobFilter === "all" ? jobs : jobs.filter(j => j.status === jobFilter);

  let html = '<div class="card"><h2>Jobs</h2>';
  html += '<div class="filter-bar">';
  html += `<div class="filter-btn ${jobFilter === "all" ? "active" : ""}" onclick="setJobFilter('all')">All (${jobs.length})</div>`;
  for (const status of Object.keys(counts).sort()) {
    html += `<div class="filter-btn ${jobFilter === status ? "active" : ""}" onclick="setJobFilter('${esc(status)}')">${esc(status)} (${counts[status]})</div>`;
  }
  html += '</div>';

  if (!shown.length) {
    html += '<p class="muted">No jobs in this workspace state dir yet.</p>';
  } else {
    html += '<div class="job-list">';
    for (const j of shown) {
      html += `<a class="job-row" href="?job=${encodeURIComponent(j.id)}">
        ${pill(j.status)}
        <span class="job-id">${esc(j.id)}</span>
        <span class="job-goal" title="${esc(j.goal)}">${truncateGoal(j.goal, 160)}</span>
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
        if (act.meta.tokens_in != null) html += `<span class="meta-chip">↑ ${act.meta.tokens_in}</span>`;
        if (act.meta.tokens_out != null) html += `<span class="meta-chip">↓ ${act.meta.tokens_out}</span>`;
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

async function loadJob() {
  const r = await fetch("/api/job?id=" + encodeURIComponent(jobId));
  if (r.status === 404) {
    document.getElementById("content").innerHTML = '<div class="card">Job not found.</div>';
    return;
  }
  const d = await r.json();
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-' + esc(d.job.status) + '">' + esc(d.job.status) + '</span>';
  document.getElementById("jobid").textContent = d.job.id;
  const _g = (d.job.goal || "").split("\n")[0].trim();
  const _goalEl = document.getElementById("goal");
  _goalEl.textContent = _g.length > 110 ? _g.slice(0, 110) + "…" : _g;
  _goalEl.title = d.job.goal;
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();

  let html = "";

  const prog = d.progress || {};
  const totalTasks = Object.values(prog).reduce((a, b) => a + b, 0);
  const completeTasks = prog.complete || 0;
  const runningTasks = prog.running || 0;
  const failedTasks = prog.failed || 0;
  const queuedTasks = prog.queued || 0;
  const progressPct = totalTasks > 0 ? (completeTasks / totalTasks * 100) : 0;

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
    html += `<div class="filter-btn ${taskFilter === "all" ? "active" : ""}" onclick="setTaskFilter('all')">All (${d.tasks.length})</div>`;
    const failedCount = d.tasks.filter(t => t.status === "failed").length;
    const runningCount = d.tasks.filter(t => t.status === "running" || t.status === "queued").length;
    if (failedCount > 0) html += `<div class="filter-btn ${taskFilter === "failed" ? "active" : ""}" onclick="setTaskFilter('failed')">Failed (${failedCount})</div>`;
    if (runningCount > 0) html += `<div class="filter-btn ${taskFilter === "running" ? "active" : ""}" onclick="setTaskFilter('running')">Active (${runningCount})</div>`;
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

async function tick() {
  try { if (jobId) await loadJob(); else await loadIndex(); }
  catch (e) { /* keep polling; transient during writes */ }
}
tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""

INDEX_HTML = _PAGE_HEAD + RENDERER_JS + _PAGE_APP_JS


def make_handler(store_factory: Callable[[], SwarmStore]):
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
                        self._json(200, build_job_snapshot(store_factory(), job_id))
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

    httpd = ThreadingHTTPServer((host, port), make_handler(store_factory))
    url = f"http://{host}:{port}/" + (f"?job={job_id}" if job_id else "")
    print(f"Puppetmaster dashboard: {url}")
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
