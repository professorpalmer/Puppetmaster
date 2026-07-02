# Dashboard

A live, zero-dependency web board for any swarm — running or finished. It turns a
job's durable SQLite state into a page served from the Python standard library:
no Flask, no React build, no external CDN. The page is fully inlined, so it works
offline, and the server binds to loopback only.

OpenTelemetry remains the *export* path (point a swarm at Jaeger/Datadog and the
job renders as a correlated trace). The dashboard is the "just show me this one
job" path that needs no collector.

## Open it

```bash
python -m puppetmaster dashboard            # jobs index for this workspace
python -m puppetmaster dashboard <job_id>   # deep-link straight to one job
```

From an agent, the MCP verb `puppetmaster_dashboard` starts the server (if one
isn't already listening) and returns the URL to open — pass `job_id` to deep-link.

| Flag | Default | Purpose |
|---|---|---|
| `--port` | `8787` | Port to serve on. |
| `--host` | `127.0.0.1` | Bind host. Non-loopback is refused unless `--allow-external`. |
| `--no-open` | off | Don't auto-open a browser tab. |
| `--all-projects` | off | Aggregate jobs from every Puppetmaster project state dir on this machine. |
| `--allow-external` | off | Allow binding to a non-loopback host. The board is **unauthenticated**, so this exposes job state to the network — only on a trusted one. |

## What it shows

### Jobs index (`/`)

Every job in the workspace state dir (or every project with `--all-projects`),
newest first, with a status pill, a scannable headline, the job id, and a
relative timestamp. Filter by status. Click through to a job.

### Job view (`/?job=<id>`)

Top to bottom:

- **Swarm Tracker** — the at-a-glance hero. Job title + status, the primary
  routed model, worker count, and adapter; the job's total **cost and tokens**;
  a completion **progress bar**; an aggregate **verification score**; and a
  **routing rollup** — one card per routed worker showing the model, its cost, a
  plain-language "why this model won" line derived from the routing policy, and a
  collapsible `N alternatives considered` control that expands to the rejected
  models (hover for the rejection reason). The alternatives come straight from
  each `ROUTING` artifact's audited `rejected` list.
- **Summary + cost by model** — task counts and per-model estimated spend.
- **Alerts** — the same failure→remediation alerts the stitched summary surfaces.
- **Reroutes** — fallback and escalation decisions the router made mid-run.
- **Tasks** — per-worker cards (role, status, adapter, model, attempts) with an
  expandable "Thinking & Output" timeline: the worker's message, model/cost/token
  chips, the routing rationale, evidence, and any code diff.
- **Findings / Risks / Decisions / Verifications** — the typed artifacts,
  grouped and collapsible.
- A sticky **job-total footer** — cost, tokens, and primary model, always in view.

The board polls every ~1.5 s, so a running swarm updates live and a finished one
renders instantly. It only rewrites the DOM when the markup actually changed, so
there's no flicker and your scroll position survives updates.

## How the data is built

Two cleanly separated layers live in [`puppetmaster/dashboard.py`](../puppetmaster/dashboard.py):

- `build_job_snapshot` / `list_jobs_snapshot` — pure functions that read a
  `SwarmStore` and return JSON-able dicts (task board, per-task activity, typed
  artifact rollup, token + cost totals, routing rollup, verification score,
  reroutes, alerts). Unit-testable without a socket.
- `serve` — wraps those in a tiny `http.server` handler.

## Safety

The dashboard serves durable job state with **no authentication**, so binding to
a non-loopback interface would expose it to the local network. That is refused
unless `--allow-external` (or `PUPPETMASTER_DASHBOARD_ALLOW_EXTERNAL=1`) is set,
making the exposure an explicit choice. Job ids are validated before touching the
state tree, and all artifact-derived text is HTML-escaped.
