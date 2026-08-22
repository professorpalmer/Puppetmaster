# Monitoring state machine

Use this reference for every asynchronous Puppetmaster verb.

1. Save `job_id`, `job_ref`, backend, label, start time, and the returned cursor.
2. Report the job ID immediately.
3. Call the exact tool and arguments in `monitor_with`.
4. Always retain `next_cursor`, including empty or timed-out batches.
5. Treat `timed_out=true` or `capped=true` as the end of one bounded observation,
   not a worker failure; call again while `terminal=false`.
6. Distinguish liveness from substantive artifact progress using `progress`.
7. At terminal state, require `delivery.verdict == "delivered"`, inspect the
   stitched result, and independently verify claims important to the user.

Operator labels:

- `STARTED`: a durable job identity was returned.
- `RUNNING`: queued/running/stitching with live work; not a completion claim.
- `BLOCKED`: a precondition, gate, cancellation, or budget prevented delivery.
- `STALLED` / `FAILED`: the durable lifecycle state says so.
- `DEGRADED`: terminal artifacts exist but delivery is not trustworthy.
- `COMPLETE`: terminal complete plus delivered outcome and task-specific proof.

Never relaunch because an observation call timed out. If monitoring must be
handed off, return the job reference, last status, cursor, and exact resume call.
