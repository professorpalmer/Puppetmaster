"""Retrieval race: no-CodeGraph crawl vs Puppetmaster + prebuilt CodeGraph.

This quantifies the thing that actually drives the agent speed/cost gap: what
each approach makes the model *ingest and do* to answer the same code question
on the same repo. It is deterministic (no LLM calls, no API key) so the numbers
are reproducible.

Two arms, identical questions:

- **A — no CodeGraph (grep + read crawl).** What a context-less agent does to
  locate and understand a symbol: grep the salient terms, then read the matching
  source files into context. We measure real wall-clock for the I/O, the bytes
  read, an estimated token count (chars/4), and the number of tool round-trips
  (one grep per term + one read per file) — round-trips are the agent-latency
  driver because each is a separate model turn.

  We are deliberately *conservative toward arm A*: test files are excluded (an
  agent answering "where is X" focuses on source), and we model a single hop —
  a real "...and what calls it?" question forces arm A to re-grep callers and
  read their files too, multiplying its cost, while arm B stays O(1).

- **B — Puppetmaster + prebuilt CodeGraph.** One ``codegraph context`` lookup
  returns the pre-resolved entry points, related symbols, and code snippets. We
  measure its real wall-clock, the context bytes/tokens delivered, and the
  round-trips (1).

End-to-end agent seconds are *modeled* from round-trips with an explicit,
labeled per-turn latency assumption — never presented as measured.

    python -m bench.codegraph_race --cwd . [--seconds-per-turn 3.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.codegraph import codegraph_context  # noqa: E402

# Token estimate: ~4 chars/token is the standard rough heuristic for code.
_CHARS_PER_TOKEN = 4


# Identical questions for both arms. `grep_terms` are the salient symbols a
# competent no-CodeGraph agent would search for to answer `question`.
QUERIES = [
    {
        "question": "Where are completion gates evaluated and what calls them?",
        "grep_terms": ["evaluate_task_gates", "def _gate_"],
    },
    {
        "question": "How does the orchestrator wait for workers and handle timeouts?",
        "grep_terms": ["_wait_for_worker", "_kill_and_report_timeout"],
    },
    {
        "question": "Where is per-worktree port allocation done and how is a free port reserved?",
        "grep_terms": ["worktree_port_base", "reserve_port"],
    },
    {
        "question": "How does delete_job guard against deleting unsafe paths?",
        "grep_terms": ["_assert_safe_job_dir", "def delete_job"],
    },
    {
        "question": "Where is cross-worker write-conflict prediction implemented?",
        "grep_terms": ["predict_write_conflicts", "scopes_overlap"],
    },
    {
        "question": "How does the model router pick a model and record the decision?",
        "grep_terms": ["def route_task", "ROUTING"],
    },
]


@dataclass
class ArmMetrics:
    wall_seconds: float
    files_touched: int
    bytes_ingested: int
    est_tokens: int
    round_trips: int


@dataclass
class QueryResult:
    question: str
    grep_terms: list
    arm_a: ArmMetrics
    arm_b: ArmMetrics
    matched_files: list = field(default_factory=list)


def _rg_files(term: str, cwd: Path) -> list:
    """Source files containing `term` (tests/vendored dirs excluded — keeps arm
    A conservative). Falls back to nothing on any rg error."""
    try:
        proc = subprocess.run(
            ["rg", "-l", "--fixed-strings", term,
             "-g", "*.py", "-g", "!tests/**", "-g", "!**/__pycache__/**",
             "-g", "!bench/results/**"],
            cwd=str(cwd), capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode not in (0, 1):  # 1 == no matches, not an error
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def measure_arm_a(query: dict, cwd: Path) -> tuple:
    """No-CodeGraph crawl: grep each term, read every matching source file."""
    start = time.monotonic()
    matched: set = set()
    grep_calls = 0
    for term in query["grep_terms"]:
        grep_calls += 1
        for rel in _rg_files(term, cwd):
            matched.add(rel)
    total_bytes = 0
    for rel in sorted(matched):
        try:
            total_bytes += (cwd / rel).stat().st_size
            # Actually read it — this is the I/O an agent's Read tool performs.
            (cwd / rel).read_bytes()
        except OSError:
            continue
    elapsed = time.monotonic() - start
    metrics = ArmMetrics(
        wall_seconds=round(elapsed, 4),
        files_touched=len(matched),
        bytes_ingested=total_bytes,
        est_tokens=total_bytes // _CHARS_PER_TOKEN,
        round_trips=grep_calls + len(matched),
    )
    return metrics, sorted(matched)


def measure_arm_b(query: dict, cwd: Path) -> ArmMetrics:
    """Puppetmaster + prebuilt CodeGraph: one context lookup."""
    start = time.monotonic()
    context = codegraph_context(query["question"], str(cwd), max_nodes=15) or ""
    elapsed = time.monotonic() - start
    chars = len(context)
    return ArmMetrics(
        wall_seconds=round(elapsed, 4),
        files_touched=1,  # the graph DB; no source files dragged into context
        bytes_ingested=chars,
        est_tokens=chars // _CHARS_PER_TOKEN,
        round_trips=1,
    )


def run(cwd: Path, seconds_per_turn: float) -> dict:
    results: list = []
    for query in QUERIES:
        a, matched = measure_arm_a(query, cwd)
        b = measure_arm_b(query, cwd)
        results.append(QueryResult(
            question=query["question"], grep_terms=query["grep_terms"],
            arm_a=a, arm_b=b, matched_files=matched,
        ))

    def total(attr: str, arm: str) -> int:
        return sum(getattr(getattr(r, arm), attr) for r in results)

    a_tokens, b_tokens = total("est_tokens", "arm_a"), total("est_tokens", "arm_b")
    a_trips, b_trips = total("round_trips", "arm_a"), total("round_trips", "arm_b")
    a_wall, b_wall = total("wall_seconds", "arm_a"), total("wall_seconds", "arm_b")

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_repo": str(cwd),
        "seconds_per_turn_assumption": seconds_per_turn,
        "totals": {
            "arm_a": {
                "est_tokens": a_tokens, "round_trips": a_trips,
                "retrieval_wall_seconds": round(a_wall, 3),
                "modeled_agent_seconds": round(a_trips * seconds_per_turn, 1),
            },
            "arm_b": {
                "est_tokens": b_tokens, "round_trips": b_trips,
                "retrieval_wall_seconds": round(b_wall, 3),
                "modeled_agent_seconds": round(b_trips * seconds_per_turn, 1),
            },
            "ratios": {
                "tokens": round(a_tokens / b_tokens, 1) if b_tokens else None,
                "round_trips": round(a_trips / b_trips, 1) if b_trips else None,
                "modeled_agent_seconds": round(a_trips / b_trips, 1) if b_trips else None,
            },
        },
        "queries": [
            {
                "question": r.question,
                "grep_terms": r.grep_terms,
                "matched_files": r.matched_files,
                "arm_a": asdict(r.arm_a),
                "arm_b": asdict(r.arm_b),
            }
            for r in results
        ],
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", default=".", help="Target repo (must have .codegraph/).")
    parser.add_argument("--seconds-per-turn", type=float, default=3.0,
                        help="Labeled assumption: model seconds per tool round-trip.")
    parser.add_argument("--json", type=Path, help="Write JSON report here.")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    report = run(cwd, args.seconds_per_turn)

    t = report["totals"]
    print(f"Retrieval race on {cwd}  ({report['ran_at']})")
    print(f"  questions: {len(report['queries'])}")
    print()
    print(f"{'metric':<28}{'A: no CodeGraph':>18}{'B: +CodeGraph':>16}{'ratio':>10}")
    print("-" * 72)
    print(f"{'est. tokens ingested':<28}{t['arm_a']['est_tokens']:>18,}{t['arm_b']['est_tokens']:>16,}{(str(t['ratios']['tokens']) + 'x'):>10}")
    print(f"{'tool round-trips':<28}{t['arm_a']['round_trips']:>18,}{t['arm_b']['round_trips']:>16,}{(str(t['ratios']['round_trips']) + 'x'):>10}")
    print(f"{'retrieval wall seconds':<28}{t['arm_a']['retrieval_wall_seconds']:>18}{t['arm_b']['retrieval_wall_seconds']:>16}{'':>10}")
    print(f"{'modeled agent seconds':<28}{t['arm_a']['modeled_agent_seconds']:>18}{t['arm_b']['modeled_agent_seconds']:>16}{(str(t['ratios']['modeled_agent_seconds']) + 'x'):>10}")
    print(f"\n(modeled agent seconds = round_trips x {args.seconds_per_turn}s/turn assumption)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
