# Named cells

v1.22.24 steals the useful [celld](https://celld.dev/) bits — the Durable
Objects programming model — and drops the fleet.

**Kept:** one named cell, one private SQLite, single-threaded inbox (one
event at a time), hibernate when idle, alarms to resume, inspectable
on-disk state (`sqlite3` + grep).

**Dropped:** S3 as coordinator, LTX replication, V8 isolates, Wrangler
bundles, distributed CAS ownership.

Cells are additive on top of SwarmStore + task leases. Job/task/artifact
tables stay in `state.sqlite3`. Each cell is
`<state>/cells/<id>.sqlite`.

```bash
puppetmaster cell-status job-abc --json
puppetmaster cell-inspect job-abc
puppetmaster cell-tick
```

MCP: `puppetmaster_cell_status` returns `path`, `inbox_depth`,
`hibernating`, `next_alarm`.

Marionette / harness contract: new CLI verbs and one MCP tool only. No
harness, SSE, or session JSON shape changes.
