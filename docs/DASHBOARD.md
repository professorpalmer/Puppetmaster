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

The normal board is project-scoped: it shows the state directory for the
workspace you selected, not every job on the machine. Use `--all-projects` only
for an intentional aggregate view. When the command starts a board, open the
exact URL it prints or that the MCP tool returns; the default port starts at
8787 but may auto-bump. See [CONCURRENT_SESSIONS.md](CONCURRENT_SESSIONS.md)
for the complete concurrent-session contract.

From an agent, the MCP verb `puppetmaster_dashboard` starts the server (or reuses
this project's own, if one is already running) and returns the URL to open — pass
`job_id` to deep-link, `mobile: true` to get a phone URL + QR, and `stop: true` to
tear it down. The server runs **detached**, so there's no terminal to keep open.

| Flag | Default | Purpose |
|---|---|---|
| `--port` | auto | Port to serve on. **Omit it** (the common case) to auto-pick the next free port starting at 8787 — a busy port (e.g. another project's dashboard) is never silently shared. **Pass it explicitly** and it's a strict promise: a busy port fails outright instead of quietly landing somewhere else, so a script/bookmark/reverse-proxy pinning a port can rely on it. Add `--port-search` to opt an explicit `--port` back into auto-bumping. |
| `--port-search` | off | With an explicit `--port`, search upward from it instead of failing when it's busy. No effect without `--port` (bumping is already the default). |
| `--host` | `127.0.0.1` | Bind host. Non-loopback is refused unless `--allow-external`. |
| `--no-open` | off | Don't auto-open a browser tab. |
| `--all-projects` | off | Aggregate jobs from every Puppetmaster project state dir on this machine. |
| `--allow-external` | off | Allow binding to a non-loopback host. The board is **unauthenticated**, so this exposes job state to the network — only on a trusted one. |
| `--mobile` | off | Serve on a phone-reachable address: auto-detect a Tailscale IP (falls back to LAN IP), bind it (implies `--allow-external`), and print the URL. See below. |
| `--qr` | off | With `--mobile`, also print a scannable QR of the URL (needs the optional `qrcode` package). |
| `--background` / `-b` | off | Run detached like a backend service and return to the prompt instead of holding the terminal. Prints the URL (and QR with `--qr`), then keeps serving until `--stop`. Reuses this project's own already-running background dashboard instead of spawning a redundant one. |
| `--stop` | off | Stop the detached background dashboard for this state dir. |
| `--status` | off | Report whether a background dashboard is running for this state dir. |

**Project identity, not just liveness.** Every dashboard exposes `GET /api/meta`
— a hashed id for its state dir, plus pid and `--all-projects` scope — so
`--background`/the MCP tool can tell *this project's* dashboard apart from some
other program (or another project's board) merely answering on the same
host:port, and reuse only the former. A pre-upgrade dashboard without this route
is never mistaken for a match — no coordinated upgrade needed, and `--status`
still reports it as yours (noting it predates `/api/meta`) as long as the pid
your own runfile tracked is still alive.

That same identity also has a human half. The header carries a **project tag**,
rendered from `/api/meta` by a pure `projectLabel()` as
`<slug> · <first 8 hex of the state dir digest>` — for example
`puppetmaster · c3177e60` — and the identical label goes into the **browser tab
title** (`Puppetmaster — puppetmaster · c3177e60`). Two boards on adjacent ports are
therefore distinguishable at a glance, and from the taskbar or tab strip without
focusing either one. The 8 hex digits are kept on purpose rather than trimmed for
looks: the two dirs one real project forked into were
`puppetmaster-c3177e6032c4` and `Puppetmaster-b92145e840c8`, which differ **only
by letter case** once the digest is stripped — dropping it would re-merge on
screen exactly what the tag exists to tell apart. An `--all-projects` board is
labelled `all projects` instead of one incidental member of the set, and a state
dir whose name isn't `<slug>-<12 hex>` is passed through unmangled.

Under `--mobile` — or for any non-loopback peer — `/api/meta` withholds the
human-readable basename, so the header degrades to a short state id
(`state c3177e60`). That is deliberate, not a gap: the endpoint is
**unauthenticated**, and a project basename is a folder name off the local
filesystem, while the hash is safe to hand to anyone who can already reach the
board.

**When the board may be serving the wrong project.** Because the default state
dir is derived from the git root of the launching shell's cwd, a session opened
at a non-git wrapper folder *above* the checkout gets its own separate store —
one logical project forks into two state dirs, and the emptier one renders as a
perfectly healthy, perfectly empty board. A wrong-state-dir warning now appears
in three places:

- a `WARNING:` line on **stderr** when `serve()` starts, right after the
  `Reading durable state from:` line it would otherwise contradict (the
  `--background` and MCP launch paths redirect stderr into `dashboard.err.log`,
  the only channel from a detached board that reaches a human);
- a `state_dir_may_be_wrong` entry in `warnings` on the **MCP dashboard tool**
  response, plus the same sentence appended to that response's `note` alongside
  how to get out of it (pass the repo root as `cwd`, an explicit `state_dir`, or
  `all_projects=true`). It is advisory: the dashboard did start, so nothing is
  flagged as an error;
- a **banner** in the page itself, fed by a separate `GET /api/diagnostics`
  route and HTML-escaped like every other artifact-derived string on the board.

`puppetmaster doctor` reports the same finding as a **state-identity** row —
`warn` when the active dir looks forked, `optional` for a brand-new project or an
explicitly pinned dir, `ok` otherwise.

A few properties of this check are worth stating plainly:

- The diagnosis is **not** served on `/api/meta`. The identity check that makes
  reuse work reads a bounded body under a short socket timeout, so a slow or
  oversized response there is silently read as "not my dashboard" — which is
  precisely how you end up with duplicate dashboards. The filesystem walk stays
  behind its own route.
- Detection is **warn-only**. It changes no existing state directory path, so no
  job history is moved, re-homed, or orphaned by it; the warning tells you where
  the other history is and leaves it exactly where it is.
- An `--all-projects` board **never warns** — it serves every project, so "you
  may be pointed at the wrong one" cannot be true of it.
- An explicit `--state-dir` (or `PUPPETMASTER_STATE_DIR`) is treated as a
  deliberate choice and is **never second-guessed**; only the hashed
  `projects/<slug>-<digest>` layout can fork.
- `--all-projects` also stopped hiding failures: a project it cannot read is now
  **reported** rather than silently omitted, so an aggregate board that is
  showing less than everything says so instead of erasing a whole project's job
  list without a trace.

## Easiest setup: let the pilot run it (no second terminal)

The lowest-lift path is to have the agent start it for you. The MCP
`puppetmaster_dashboard` verb spawns a detached server and hands back a link (and
a QR for phones) — you never open or babysit a terminal:

- **On this machine:** ask the agent to open the dashboard → it returns a loopback
  URL to click.
- **On your phone:** ask the agent to open the *mobile* dashboard → it detects your
  Tailscale/LAN address, serves there, and returns a scannable QR. Scan it and
  you're in.
- **Done:** ask the agent to stop the dashboard (`stop: true`) — or just leave the
  lightweight server running.

One-time prerequisites for the phone case (documented once, then it's automatic):
install [Tailscale](https://tailscale.com/download) on both this Mac and your
phone and sign in to the same account. That's what gives the board a stable
`100.x` address reachable from anywhere.

Prefer the CLI? The same detach-and-walk-away flow, by hand:

```bash
python -m puppetmaster dashboard --mobile --qr --background   # serve to phone, detached
python -m puppetmaster dashboard --status                     # is it up? what's the URL?
python -m puppetmaster dashboard --stop                       # shut it down
```

## Watch it from your phone

**Full walkthrough (Tailscale + agent + install): [MOBILE.md](MOBILE.md).** The
short version below covers the flags.

`--mobile` is the blessed remote path — read-only swarm tracking on your phone
with no app, no cloud, and no DOM-scraping. It resolves a bind address in
order of preference and prints the URL (no browser is opened locally, since the
browser is on your phone):

1. **Tailscale** (`tailscale ip -4`) — a stable `100.x` address reachable from
   your phone **anywhere**, without exposing the board to the public internet.
   This is the recommended setup.
2. **LAN IP** — fallback for a phone on the same Wi-Fi.

```bash
python -m puppetmaster dashboard --mobile          # print the phone URL
python -m puppetmaster dashboard --mobile --qr      # + a scannable QR in the terminal
python -m puppetmaster dashboard <job_id> --mobile  # deep-link one job to your phone
```

The board is **unauthenticated and read-only** — anyone who can reach the
address can see job state, but not control anything. Keep it on Tailscale or a
trusted LAN. On a fresh network with neither available, `--mobile` exits with a
hint rather than binding to loopback.

The pages are responsive (a `max-width: 640px` pass collapses the jobs grid,
wraps long headlines, lets wide tables scroll, and reflows the footer), so the
job view reads cleanly on a phone.

## What it shows

Two views: the **jobs index** (`/`, the default landing on every viewport) and a
single **job view** (`?job=<id>`). `?view=jobs` forces the index even when a
stale job id lingers in the URL.

### Jobs index (`/`)

Every job in the workspace state dir (or every project with `--all-projects`),
newest first, with a status pill, a scannable headline, the job id, and a
relative timestamp. Filter by status. Click through to a job.

### Job view (`/?job=<id>`)

Top to bottom:

- **Swarm Overview** — the at-a-glance hero. Job title + status, the primary
  routed model, worker count, and adapter; the job's total **cost and tokens**;
  a completion **progress bar**; an aggregate **verification score**; and a
  **routing rollup** — one card per routed worker showing the model, its cost, a
  plain-language "why this model won" line derived from the routing policy, and a
  collapsible `N alternatives considered` control that expands to the rejected
  models (hover for the rejection reason). The alternatives come straight from
  each `ROUTING` artifact's audited `rejected` list.
- **Summary + cost by model** — task counts and per-model estimated spend.
- **Frontier** (v1.22.2+) — shared-context sidecar: queue depth
  (queued/running/blocked), follow-ups enqueued from a parent task, and gist
  admission counts (admitted/pending/rejected), plus the gist claim lines
  themselves so you can read what peers are allowed to see without digging.
- **Alerts** — the same failure→remediation alerts the stitched summary surfaces.
- **Reroutes** — fallback and escalation decisions the router made mid-run.
- **Tasks** — per-worker cards (role, status, adapter, model, attempts) with an
  expandable "Thinking & Output" timeline: the worker's message, model/cost/token
  chips, the routing rationale, evidence, and any code diff.
- **Gists / Findings / Risks / Decisions / Verifications** — typed artifacts,
  grouped and collapsible. The **Gists** section auto-opens when present and
  shows admission status + claim + source artifact ids.
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
