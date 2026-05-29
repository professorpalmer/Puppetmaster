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
  (task board, typed-artifact rollup, cost from ROUTING artifacts, auto-fallback
  reroutes, and the same Alerts the stitcher surfaces).
* :func:`serve` — wraps those in a tiny HTTP handler and polls every ~1.5 s, so
  the board updates live while a swarm runs and is instant for a finished one.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Union

from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.store import SwarmStore

# Reuse the stitcher's failure→remediation map so the dashboard Alerts match the
# stitched summary verbatim (one source of truth for "what went wrong + fix").
from puppetmaster.stitcher import Stitcher

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
        cost = float(payload.get("estimated_cost_usd") or 0.0)
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
            "router-fallback"
        ):
            reroutes.append(
                {
                    "task_id": artifact.task_id,
                    "reason": (artifact.payload or {}).get("reason"),
                    "created_by": artifact.created_by,
                }
            )

    task_rows = [
        {
            "id": task.id,
            "role": task.role,
            "adapter": task.adapter,
            "status": task.status.value,
            "model": _task_model(task),
            "attempts": task.attempts,
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

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Puppetmaster Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0d1117; color: #c9d1d9; }
  header { padding: 14px 20px; border-bottom: 1px solid #21262d; display: flex;
           align-items: center; gap: 14px; position: sticky; top: 0; background: #0d1117; z-index: 5; }
  header h1 { font-size: 16px; margin: 0; color: #f0f6fc; }
  .pill { padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .s-complete { background: #1f6f3f; color: #d3f9d8; }
  .s-running, .s-stitching { background: #9e6a03; color: #fff3bf; }
  .s-failed { background: #8b2c2c; color: #ffd6d6; }
  .s-queued, .s-blocked { background: #30363d; color: #c9d1d9; }
  main { padding: 20px; max-width: 1100px; margin: 0 auto; }
  .goal { color: #8b949e; margin: 0 0 16px; }
  .grid { display: grid; gap: 18px; }
  .card { border: 1px solid #21262d; border-radius: 8px; padding: 14px 16px; background: #0f141b; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; margin: 0 0 10px; color: #8b949e; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #161b22; vertical-align: top; }
  th { color: #6e7681; font-weight: 600; font-size: 12px; }
  .muted { color: #6e7681; }
  .alert { background: #2d1416; border: 1px solid #8b2c2c; border-radius: 6px; padding: 8px 12px; margin: 6px 0; color: #ffd6d6; }
  .reroute { background: #102a1a; border: 1px solid #1f6f3f; border-radius: 6px; padding: 8px 12px; margin: 6px 0; color: #d3f9d8; }
  .cost { font-size: 24px; color: #f0f6fc; }
  .ev { color: #6e7681; font-size: 12px; }
  a { color: #58a6ff; text-decoration: none; }
  .conf { color: #8b949e; }
  #jobs li { margin: 4px 0; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
</style>
</head>
<body>
<header>
  <h1>Puppetmaster</h1>
  <span id="status" class="pill s-queued">…</span>
  <span class="muted" id="jobid"></span>
  <span class="muted" id="updated" style="margin-left:auto"></span>
</header>
<main>
  <p class="goal" id="goal"></p>
  <div class="grid" id="content"></div>
</main>
<script>
const qs = new URLSearchParams(location.search);
let jobId = qs.get("job");

function esc(s){ return String(s==null?"":s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function pill(s){ return `<span class="pill s-${esc(s)}">${esc(s)}</span>`; }

async function loadIndex(){
  const r = await fetch("/api/jobs"); const jobs = await r.json();
  document.getElementById("goal").textContent = "";
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-queued">jobs</span>';
  let html = '<div class="card"><h2>Jobs</h2><ul id="jobs">';
  if(!jobs.length) html += '<li class="muted">No jobs in this workspace state dir yet.</li>';
  for(const j of jobs){
    html += `<li><span class="dot s-${esc(j.status)}"></span><a href="?job=${encodeURIComponent(j.id)}">${esc(j.id)}</a> ${pill(j.status)} <span class="muted">${esc(j.goal)}</span></li>`;
  }
  html += "</ul></div>";
  document.getElementById("content").innerHTML = html;
}

function rows(items, cols){
  if(!items || !items.length) return '<tr><td class="muted" colspan="'+cols.length+'">None</td></tr>';
  return items.map(it => '<tr>' + cols.map(c => '<td>'+c(it)+'</td>').join('') + '</tr>').join('');
}

async function loadJob(){
  const r = await fetch("/api/job?id="+encodeURIComponent(jobId));
  if(r.status === 404){ document.getElementById("content").innerHTML = '<div class="card">Job not found.</div>'; return; }
  const d = await r.json();
  document.getElementById("status").outerHTML = '<span id="status" class="pill s-'+esc(d.job.status)+'">'+esc(d.job.status)+'</span>';
  document.getElementById("jobid").textContent = d.job.id;
  document.getElementById("goal").textContent = d.job.goal;
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();

  let html = "";
  if(d.alerts && d.alerts.length){
    html += '<div class="card"><h2>Alerts — action required</h2>' +
      d.alerts.map(a => '<div class="alert">'+esc(a.replace(/^- /,''))+'</div>').join('') + '</div>';
  }
  if(d.reroutes && d.reroutes.length){
    html += '<div class="card"><h2>Auto-fallback reroutes</h2>' +
      d.reroutes.map(x => '<div class="reroute"><b>'+esc(x.task_id)+'</b> → '+esc(x.reason||'')+'</div>').join('') + '</div>';
  }

  html += '<div class="card"><h2>Estimated cost</h2><div class="cost">$'+(d.cost.total_estimated_cost_usd||0).toFixed(6)+'</div>';
  const bm = Object.entries(d.cost.by_model||{});
  if(bm.length){ html += '<table><tr><th>model</th><th>calls</th><th>$</th></tr>' +
    bm.map(([m,v]) => '<tr><td>'+esc(m)+'</td><td>'+v.calls+'</td><td>$'+v.estimated_cost_usd.toFixed(6)+'</td></tr>').join('') + '</table>'; }
  html += '</div>';

  html += '<div class="card"><h2>Tasks</h2><table><tr><th>role</th><th>adapter</th><th>model</th><th>status</th><th>tries</th></tr>' +
    rows(d.tasks, [
      t=>esc(t.role), t=>esc(t.adapter), t=>esc(t.model||'—'), t=>pill(t.status), t=>esc(t.attempts)
    ]) + '</table></div>';

  const sect = (title, items) => '<div class="card"><h2>'+title+' ('+(items?items.length:0)+')</h2><table>' +
    rows(items, [
      it=>esc(it.statement), it=>'<span class="conf">'+(it.confidence!=null?it.confidence.toFixed(2):'')+'</span>',
      it=>'<span class="ev">'+esc((it.evidence||[]).join(', '))+'</span>'
    ]) + '</table></div>';
  html += sect('Findings', d.artifacts.finding);
  html += sect('Risks', d.artifacts.risk);
  html += sect('Decisions', d.artifacts.decision);
  html += sect('Verifications', d.artifacts.verification);

  document.getElementById("content").innerHTML = html;
}

async function tick(){
  try { if(jobId) await loadJob(); else await loadIndex(); }
  catch(e){ /* keep polling; transient during writes */ }
}
tick();
setInterval(tick, 1500);
</script>
</body>
</html>
"""


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
                    try:
                        self._json(200, build_job_snapshot(store_factory(), job_id))
                    except (FileNotFoundError, KeyError):
                        self._json(404, {"error": "job not found", "id": job_id})
                    return
                self._json(404, {"error": "not found"})
            except Exception as exc:  # never crash the server on one bad request
                self._json(500, {"error": str(exc)})

    return _Handler


def serve(
    state_dir: Union[Path, str],
    *,
    backend: str = "sqlite",
    job_id: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    serve_forever: bool = True,
):
    """Start the dashboard HTTP server. Returns the server (already bound).

    ``serve_forever=False`` binds and returns without blocking — used by tests
    to drive a single request, then ``server.shutdown()``.
    """
    from http.server import ThreadingHTTPServer

    from puppetmaster.store_factory import create_store

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
