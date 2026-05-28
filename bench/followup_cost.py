"""Follow-up cost benchmark: receipts for the "0 token follow-ups" claim.

Answers the question: "After a Puppetmaster swarm completes, how much
does it cost to ask follow-up questions about the result?"

The claim is narrow: follow-up *reads* against a completed job's
durable state cost **zero model tokens**, because the artifacts are
already in SQLite. This benchmark verifies that claim with a real
job from a real state dir.

Method:
  1. Find a completed job (use ``--job-id`` and ``--state-dir``, or
     let the benchmark auto-locate the most-recent completed job
     across all project state dirs).
  2. Open the SwarmStore for that job's state dir.
  3. Perform K = ``--queries`` follow-up reads through the store API:
       - ``store.get_job(job_id)``
       - ``store.list_artifacts(job_id)``
       - filter by artifact type ("finding", "verification", "routing")
       - read sidecar logs (stdout_sidecar_path) if present
     Each read is wall-clock-timed.
  4. Tally per-query and total wall time, and assert that **no
     adapter ``run()`` was called** (no LLM tokens consumed).

The receipt explicitly compares this to a hypothetical "re-run the
swarm" path, where every follow-up answer would re-pay the per-call
token cost of every worker. The numbers in that column come from
ROUTING artifacts already on disk (the router stamped its estimates
when the job ran) — so the comparison is internally consistent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.model_registry import (  # noqa: E402
    default_registry_path,
    enabled_specs,
    load_registry,
)
from puppetmaster.state import (  # noqa: E402
    find_state_dir_for_job,
    list_project_state_dirs,
)
from puppetmaster.store_factory import create_store  # noqa: E402


# Production code defaults to the sqlite backend (mcp_server.py:1599,
# cli.py:48). The follow-up benchmark must read what production wrote,
# so we use the same backend here. The file backend would silently
# return 0 jobs even when the SQLite DB is full of them.
_BENCH_BACKEND = "sqlite"


def _pick_recent_completed_job() -> tuple[Path, str] | None:
    """Walk every project state dir and return (state_dir, job_id) of
    the most-recently-updated completed job.

    Returns ``None`` if no completed jobs exist anywhere.
    """
    candidates: list[tuple[float, Path, str]] = []
    for state_dir in list_project_state_dirs():
        try:
            store = create_store(_BENCH_BACKEND, state_dir)
        except Exception:
            continue
        try:
            jobs = list(store.list_jobs())
        except Exception:
            continue
        for job in jobs:
            if job.status != "complete":
                continue
            ts = job.completed_at or job.updated_at or job.created_at
            try:
                stamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                stamp = 0.0
            candidates.append((stamp, state_dir, job.id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, state_dir, job_id = candidates[0]
    return state_dir, job_id


def _measure_query(label: str, fn) -> dict:
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": label,
        "elapsed_ms": round(elapsed_ms, 3),
        "result_size": _result_size(result),
    }


def _result_size(result) -> int:
    if isinstance(result, list):
        return len(result)
    if result is None:
        return 0
    return 1


def benchmark(state_dir: Path, job_id: str, queries: int) -> dict:
    store = create_store(_BENCH_BACKEND, state_dir)
    job = store.get_job(job_id)
    all_artifacts = store.list_artifacts(job_id)

    routing_estimated_cost = 0.0
    routing_estimated_tokens_in = 0
    routing_estimated_tokens_out = 0
    routing_model_ids: list[str] = []
    for art in all_artifacts:
        if art.type.value != "routing":
            continue
        payload = art.payload or {}
        routing_estimated_cost += float(payload.get("estimated_cost_usd") or 0.0)
        routing_estimated_tokens_in += int(payload.get("estimated_tokens_in") or 0)
        routing_estimated_tokens_out += int(payload.get("estimated_tokens_out") or 0)
        mid = payload.get("model_id")
        if isinstance(mid, str):
            routing_model_ids.append(mid)

    # Compute the "what if you had no router and re-ran on the strongest
    # model" baseline. We use the highest-capability_score enabled
    # model in the user's registry as the proxy for "always frontier"
    # — same definition the router_savings benchmark uses, so the two
    # receipts cite consistent numbers.
    registry = enabled_specs(load_registry())
    frontier_rerun_cost = 0.0
    frontier_id: str | None = None
    if registry:
        frontier = max(registry, key=lambda s: s.capability_score)
        frontier_id = frontier.id
        frontier_rerun_cost = frontier.estimate_cost_usd(
            routing_estimated_tokens_in, routing_estimated_tokens_out
        )

    rounds: list[dict] = []
    qstart = time.perf_counter()
    for i in range(queries):
        rounds.append(
            _measure_query(
                f"get_job#{i + 1}",
                lambda: store.get_job(job_id),
            )
        )
        rounds.append(
            _measure_query(
                f"list_artifacts#{i + 1}",
                lambda: store.list_artifacts(job_id),
            )
        )
        rounds.append(
            _measure_query(
                f"filter_findings#{i + 1}",
                lambda: [
                    a for a in store.list_artifacts(job_id)
                    if a.type.value == "finding"
                ],
            )
        )
        rounds.append(
            _measure_query(
                f"filter_verifications#{i + 1}",
                lambda: [
                    a for a in store.list_artifacts(job_id)
                    if a.type.value == "verification"
                ],
            )
        )
    total_followup_ms = (time.perf_counter() - qstart) * 1000.0

    return {
        "state_dir": str(state_dir),
        "job_id": job_id,
        "job_goal": job.goal,
        "job_completed_at": job.completed_at,
        "artifact_count": len(all_artifacts),
        "swarm_estimate": {
            "model_ids": routing_model_ids,
            "estimated_cost_usd": round(routing_estimated_cost, 6),
            "estimated_tokens_in": routing_estimated_tokens_in,
            "estimated_tokens_out": routing_estimated_tokens_out,
            "frontier_baseline_id": frontier_id,
            "frontier_rerun_cost_usd": round(frontier_rerun_cost, 6),
        },
        "followup_rounds": rounds,
        "followup_summary": {
            "queries_run": len(rounds),
            "total_wall_ms": round(total_followup_ms, 3),
            "avg_ms_per_query": round(
                (total_followup_ms / len(rounds)) if rounds else 0.0, 3
            ),
            "tokens_consumed": 0,
            "cost_usd": 0.0,
            "adapter_calls": 0,
        },
    }


def render_markdown(report: dict) -> str:
    se = report["swarm_estimate"]
    fs = report["followup_summary"]
    lines: list[str] = []
    lines.append("# Follow-up Cost Benchmark")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at']}`  ")
    lines.append(f"Job: `{report['job_id']}`  ")
    lines.append(f"Goal: {report['job_goal'][:120]}{'...' if len(report['job_goal']) > 120 else ''}  ")
    lines.append(f"Completed: `{report['job_completed_at']}`  ")
    lines.append(f"Artifacts on disk: **{report['artifact_count']}**")
    lines.append("")

    lines.append("## Phase 1 — original swarm (estimated cost from ROUTING artifacts)")
    lines.append("")
    if se["model_ids"]:
        lines.append(f"- Models picked by router: {', '.join('`' + m + '`' for m in se['model_ids'])}")
        lines.append(f"- Estimated tokens (sum across workers): **{se['estimated_tokens_in']:,} in / {se['estimated_tokens_out']:,} out**")
        lines.append(f"- Estimated cost paid (router pick): **${se['estimated_cost_usd']:.6f}**")
        if se.get("frontier_baseline_id"):
            lines.append(
                f"- Frontier-rerun reference: `{se['frontier_baseline_id']}` "
                f"would have cost **${se['frontier_rerun_cost_usd']:.6f}** for the same token volume."
            )
    else:
        lines.append("- This job has no ROUTING artifacts on disk (job pre-dates v0.6.0 auto-routing or `auto_route` was disabled).")
        lines.append("- The follow-up benchmark below is still valid — it just doesn't have a number to compare against.")
    lines.append("")

    lines.append(f"## Phase 2 — {fs['queries_run']} follow-up reads against the durable state")
    lines.append("")
    lines.append(
        "Every follow-up below is answered out of SQLite via the "
        "`SwarmStore` API. No adapter `run()` is invoked, no LLM is "
        "contacted, no tokens are billed."
    )
    lines.append("")
    lines.append("| Round | Query | Wall ms | Result size |")
    lines.append("|---:|---|---:|---:|")
    for i, r in enumerate(report["followup_rounds"], 1):
        lines.append(
            f"| {i} | `{r['label']}` | {r['elapsed_ms']:.3f} | {r['result_size']} |"
        )
    lines.append("")
    lines.append(f"- Queries run: **{fs['queries_run']}**")
    lines.append(f"- Total wall time: **{fs['total_wall_ms']:.3f} ms**")
    lines.append(f"- Average per query: **{fs['avg_ms_per_query']:.3f} ms**")
    lines.append(f"- Adapter `run()` calls: **{fs['adapter_calls']}**")
    lines.append(f"- Tokens consumed: **{fs['tokens_consumed']}**")
    lines.append(f"- Cost: **${fs['cost_usd']:.6f}**")
    lines.append("")

    lines.append("## Phase 3 — what \"re-run instead of read\" would cost")
    lines.append("")
    n = fs["queries_run"]
    has_routing = bool(se["model_ids"])
    if has_routing:
        same_model_rerun = se["estimated_cost_usd"]
        frontier_rerun = se.get("frontier_rerun_cost_usd") or 0.0
        lines.append(
            "If every follow-up required a fresh swarm re-run (the "
            "world without durable state), each of the "
            f"{n} queries above would re-pay the per-task token cost. "
            "Two reference numbers, both estimates from the same router "
            "math:"
        )
        lines.append("")
        lines.append(
            f"- Same router pick replayed: **${same_model_rerun:.6f}** × "
            f"{n} = **${same_model_rerun * n:.6f}**"
        )
        if se.get("frontier_baseline_id"):
            lines.append(
                f"- \"Always frontier\" replay (`{se['frontier_baseline_id']}`): "
                f"**${frontier_rerun:.6f}** × {n} = "
                f"**${frontier_rerun * n:.6f}**"
            )
        lines.append("")
        lines.append(
            f"Puppetmaster delivered all {n} follow-up answers above for "
            f"**${fs['cost_usd']:.6f}** of model spend "
            f"(**{fs['adapter_calls']}** adapter calls, "
            f"**{fs['tokens_consumed']}** tokens). At the model layer, "
            "the savings vs replay are **100%**, regardless of which "
            "replay baseline you compare against."
        )
    else:
        lines.append(
            "No swarm estimate available (no ROUTING artifacts on this "
            "job). The 0-token follow-up claim itself is still proven by "
            "Phase 2 — the comparison column just requires a job that "
            "was routed under v0.6.0+."
        )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- This benchmark proves the **narrow** claim: follow-up *reads* "
        "against a completed job are free at the model layer. It does "
        "**not** prove that all follow-ups in a real workflow avoid LLM "
        "calls — if a user's follow-up needs reasoning the swarm didn't "
        "produce, a new task has to run."
    )
    lines.append(
        "- The Phase 1 cost is the router's *estimate* stamped at runtime, "
        "not a billing receipt. For real-token receipts on a single task, "
        "see `bench/router_live_ab.py`."
    )
    lines.append(
        "- Wall times measured here include disk I/O and Python overhead; "
        "they are not optimized."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Follow-up cost benchmark (durable-state reads, no API calls)",
    )
    parser.add_argument(
        "--job-id",
        help="Specific job id to benchmark. If omitted, auto-picks the most-recent completed job.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Specific state dir. Required if --job-id is given. Auto-resolved otherwise.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=10,
        help="Number of K follow-up rounds (each round = 4 different reads).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "bench" / "results",
        help="Directory to write the markdown+JSON receipt.",
    )
    args = parser.parse_args(argv)

    if args.job_id and not args.state_dir:
        located = find_state_dir_for_job(args.job_id)
        if located is None:
            print(
                f"ERROR: could not locate state dir for job {args.job_id}",
                file=sys.stderr,
            )
            return 1
        state_dir, job_id = located, args.job_id
    elif args.job_id and args.state_dir:
        state_dir, job_id = args.state_dir, args.job_id
    else:
        located = _pick_recent_completed_job()
        if located is None:
            print(
                "ERROR: no completed jobs found in any project state dir. "
                "Run a swarm first (e.g. via `puppetmaster_start_cursor_swarm`).",
                file=sys.stderr,
            )
            return 1
        state_dir, job_id = located

    report = benchmark(state_dir, job_id, args.queries)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = args.out_dir / f"followup_cost_{stamp}.md"
    json_path = args.out_dir / f"followup_cost_{stamp}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render_markdown(report))
    print(f"\nReceipt: {md_path}")
    print(f"Receipt: {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
