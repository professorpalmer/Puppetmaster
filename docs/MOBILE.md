# Watch your swarms from your phone

The Puppetmaster dashboard is a zero-dependency, read-only web board over your
durable job state. This guide gets it onto your phone — from **anywhere** — with
the least possible lifting. Two moving parts:

1. **Tailscale** — a private network so your phone can reach your Mac without
   opening a port to the public internet. One-time setup.
2. **The dashboard** — either the **agent starts it for you** (easiest) or you run
   one CLI command. It serves a phone-reachable URL and a scannable QR.

The board is **unauthenticated and read-only**: anyone who can reach the address
sees job state but can't control anything. Keep it on Tailscale or a trusted LAN.

---

## 0. Install (one line)

The QR image needs a couple of optional packages (`qrcode` + Pillow). Install the
`mobile` extra and they come along:

```bash
pipx install 'puppetmaster-ai[mobile]'      # or: pip install 'puppetmaster-ai[mobile]'
# already installed? add the extra:
pip install 'puppetmaster-ai[mobile]'
```

Without the extra everything still works — you just get an **ASCII** QR (or the
bare URL) instead of a scannable image.

---

## 1. Tailscale (the network) — one time

Tailscale gives your Mac a stable `100.x` address that your phone can reach from
anywhere (coffee shop, LTE, another country) without exposing anything publicly.

### On your Mac

1. Install: [tailscale.com/download](https://tailscale.com/download) (or `brew install --cask tailscale`).
2. Sign in (Google/GitHub/Microsoft/email — whatever you'll also use on the phone).
3. Verify it's up and note your address:

```bash
tailscale status          # should list this machine as "active"
tailscale ip -4           # your 100.x address, e.g. 100.96.89.15
```

> macOS note: the GUI app keeps the `tailscale` CLI inside the app bundle and off
> your `PATH`. Puppetmaster looks in the bundle automatically, so `--mobile`
> detects the address even when `tailscale` isn't on your `PATH`.

### On your phone

1. Install the **Tailscale** app (App Store / Play Store).
2. **Sign in with the same account** you used on the Mac.
3. Toggle it **on**. That's it — the phone is now on your tailnet and can reach the
   Mac's `100.x` address.

### No Tailscale? LAN fallback

If Tailscale isn't running, `--mobile` falls back to your **LAN IP** — which works
only when the phone is on the **same Wi-Fi**. Tailscale is strongly preferred: it
works anywhere and survives network changes.

---

## 2. Start the dashboard

### Option A — let the agent do it (easiest, no terminal)

Just ask your agent (Cursor/Claude/etc. with the `puppetmaster_*` MCP tools):

> "Open the dashboard on my phone."

The agent calls `puppetmaster_dashboard` with `mobile: true`, which:

- detects your Tailscale (or LAN) address,
- starts the server **detached** (no terminal to keep open),
- returns the phone URL **and a scannable QR image** it embeds right in the chat.

Scan the QR (or tap the URL) and you're in. When you're done, ask it to
"stop the dashboard" (the tool's `stop: true`), or just leave the tiny server
running.

### Option B — one CLI command

Same detach-and-walk-away behavior, by hand:

```bash
python -m puppetmaster dashboard --mobile --qr --background
```

That prints the phone URL + an ASCII QR and **returns to your prompt** — the
server keeps running in the background. Manage it with:

```bash
python -m puppetmaster dashboard --status     # is it up? what's the URL + pid?
python -m puppetmaster dashboard --stop        # shut it down
```

Deep-link straight to one job:

```bash
python -m puppetmaster dashboard <job_id> --mobile --qr --background
```

Foreground (holds the terminal, Ctrl-C to stop) if you prefer:

```bash
python -m puppetmaster dashboard --mobile --qr
```

---

## 3. Open it on your phone

- **Scan** the QR, or
- type the URL (e.g. `http://100.96.89.15:8787/`) into your phone browser.

The board is responsive — the jobs grid, headlines, tables, and footer all reflow
for a phone screen. Tap a job to drill in; the jobs index is the default landing.

---

## Troubleshooting

| Symptom | Most likely cause | Fastest fix |
|---|---|---|
| **`ERR_CONNECTION_REFUSED`** at the URL | The server isn't running (was stopped, or never started / the terminal that held it closed). Port is reachable, nothing is listening. | `python -m puppetmaster dashboard --status`; if down, start it again with `--mobile --qr --background`. |
| `--mobile` **exits with "could not detect a Tailscale or LAN address"** | Tailscale is down and you're not on a routable LAN. | `tailscale status` / `tailscale up`, or connect Wi-Fi; or set `--host <ip> --allow-external` manually. |
| URL **loads on the Mac but not the phone** | Phone isn't on the tailnet (Tailscale off, or signed into a *different* account). LAN mode: phone not on the same Wi-Fi. | Open Tailscale on the phone, sign in with the **same** account, toggle on. |
| QR prints as **ASCII blocks, not an image** (CLI) or the agent returns `qr_ascii`/a hint instead of an image | The `qrcode`/Pillow packages aren't installed. | `pip install 'puppetmaster-ai[mobile]'`. |
| **Wrong / stale URL** in an old QR | You restarted on a different port, or the address changed. | Regenerate: rerun `--mobile --qr` (or ask the agent again); scan the fresh QR. |
| **Address in use** on start | Another dashboard already holds the port. | `--status` to see it, or pick another `--port`. |

### Why the URL can "go dead" right after it worked

The background server is a real process. If it's **stopped** (`--stop`, the tool's
`stop: true`, a reboot, or closing a foreground terminal), the URL immediately
returns `ERR_CONNECTION_REFUSED` even though the QR/URL themselves are fine.
`--status` tells you whether it's actually up; start it again to revive the link.

---

## Security posture

- **Unauthenticated + read-only.** Anyone who can reach the address sees job state;
  no one can act on it. There is no login.
- **Keep it private.** Serve over Tailscale (private by default) or a trusted LAN.
  Don't port-forward it to the public internet.
- Binding to a non-loopback host is refused unless you opt in (`--mobile` implies
  `--allow-external`; or set it explicitly). This makes network exposure a
  deliberate choice, never an accident.

See [SECURITY.md](SECURITY.md) for the full threat model and
[DASHBOARD.md](DASHBOARD.md) for what each view shows.
