"""Three-way cost-structure benchmark: Agent only vs CodeGraph alone vs Puppetmaster + CodeGraph.

The earlier ``codegraph_ab.py`` only measured CodeGraph (how big its
context bundle is). That tells you CodeGraph is doing something, but it
does not tell you what *Puppetmaster* adds on top of CodeGraph. This
script models the three configurations you actually want to compare:

  A. Agent only -- one model session, no CodeGraph, no Puppetmaster
     orchestration. The agent uses grep/read/list to discover the repo.
  B. CodeGraph alone -- the agent calls ``codegraph_explore`` via an MCP
     tool call. CodeGraph returns the context as a tool result.
  C. Puppetmaster + CodeGraph -- a swarm. CodeGraph context is
     *pre-injected* into every worker prompt. The stitcher reads only
     structured JSON artifacts. Coordination state is durable, so
     resumed runs cost ~zero model tokens.

We report two costs:

  1. **One fresh task** -- the cost of running a single investigation
     once. On this axis Puppetmaster is *not* dramatically cheaper than
     CodeGraph alone; both pay for the same context input per worker.
     The small Puppetmaster wins here come from (a) one shared
     CodeGraph query instead of N, (b) zero tool-call frames per
     worker, and (c) structured artifacts being slightly smaller than
     raw worker stdout in synthesis.

  2. **A session of 1 swarm + K follow-up reads** -- this is where
     Puppetmaster's durable state crushes the field. A "follow-up
     read" in C is just SQLite, costing 0 model tokens. In A and B
     every follow-up is a fresh agent re-run.

Inputs we *measure* on the target repo:

  - file count / total source bytes
  - CodeGraph context size and query latency
  - Avg Puppetmaster artifact size from a past state directory (real
    data from your own runs, if you point ``--artifacts-state`` at it)

Inputs we *model* with stated constants (override with flags):

  - tool-call frame size
  - typical agent self-output per task
  - stitcher output size
  - per-agent discovery scan ratio (Config A)
  - bytes-to-tokens (4 chars / token)
  - $ per 1M tokens (defaults to Claude 3.5 Sonnet input price)

We do **not** capture exact per-task token billing from a live agent
stream. That requires SDK-side instrumentation (on the roadmap).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.codegraph import (  # noqa: E402
    codegraph_available,
    codegraph_context,
    codegraph_initialized,
)


CHARS_PER_TOKEN = 4
TOOL_CALL_FRAME_BYTES = 250
AGENT_SELF_OUTPUT_BYTES = 2_000
STITCHER_OUTPUT_BYTES = 1_500
DEFAULT_ARTIFACTS_PER_WORKER = 3
DEFAULT_AVG_ARTIFACT_BYTES = 800
DEFAULT_AVG_WORKER_STDOUT_BYTES = 8_000
DEFAULT_DISCOVERY_SCAN_RATIO = 0.10


@dataclass
class RepoFacts:
    file_count: int
    source_bytes: int
    languages: dict[str, int]


@dataclass
class ArtifactFacts:
    source: str
    sample_count: int
    avg_artifact_bytes: int
    avg_worker_stdout_bytes: int
    avg_artifacts_per_worker: float


@dataclass
class ConfigCost:
    label: str
    fresh_task_bytes: int
    resume_bytes: int

    def tokens(self, bytes_value: int) -> int:
        return bytes_value // CHARS_PER_TOKEN

    def dollars(self, bytes_value: int, price_per_million: float) -> float:
        return self.tokens(bytes_value) * price_per_million / 1_000_000

    def session_bytes(self, follow_ups: int) -> int:
        return self.fresh_task_bytes + follow_ups * self.resume_bytes


SOURCE_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
}
IGNORED_DIR_NAMES = {
    "node_modules",
    "__pycache__",
    ".git",
    "dist",
    "build",
    ".venv",
    "venv",
    ".puppetmaster",
    ".codegraph",
}


def scan_repo(root: Path) -> RepoFacts:
    file_count = 0
    source_bytes = 0
    languages: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        language = SOURCE_EXTENSIONS.get(path.suffix.lower())
        if language is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        file_count += 1
        source_bytes += size
        languages[language] = languages.get(language, 0) + size
    return RepoFacts(file_count=file_count, source_bytes=source_bytes, languages=languages)


def load_artifact_facts(state_dir: Optional[Path]) -> ArtifactFacts:
    if state_dir is None:
        return ArtifactFacts(
            source="defaults (no past state supplied)",
            sample_count=0,
            avg_artifact_bytes=DEFAULT_AVG_ARTIFACT_BYTES,
            avg_worker_stdout_bytes=DEFAULT_AVG_WORKER_STDOUT_BYTES,
            avg_artifacts_per_worker=float(DEFAULT_ARTIFACTS_PER_WORKER),
        )
    db_path = state_dir / "state.sqlite3"
    if not db_path.exists():
        return ArtifactFacts(
            source=f"defaults ({db_path} not found)",
            sample_count=0,
            avg_artifact_bytes=DEFAULT_AVG_ARTIFACT_BYTES,
            avg_worker_stdout_bytes=DEFAULT_AVG_WORKER_STDOUT_BYTES,
            avg_artifacts_per_worker=float(DEFAULT_ARTIFACTS_PER_WORKER),
        )
    try:
        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            "SELECT type, COUNT(*) AS n, AVG(length(data)) AS avg_bytes "
            "FROM artifacts GROUP BY type"
        ).fetchall()
        verification_rows = connection.execute(
            "SELECT data FROM artifacts WHERE type = 'verification' LIMIT 50"
        ).fetchall()
        task_row = connection.execute(
            "SELECT COUNT(DISTINCT task_id) FROM artifacts"
        ).fetchone()
        connection.close()
    except sqlite3.DatabaseError:
        return ArtifactFacts(
            source=f"defaults ({db_path} unreadable)",
            sample_count=0,
            avg_artifact_bytes=DEFAULT_AVG_ARTIFACT_BYTES,
            avg_worker_stdout_bytes=DEFAULT_AVG_WORKER_STDOUT_BYTES,
            avg_artifacts_per_worker=float(DEFAULT_ARTIFACTS_PER_WORKER),
        )

    structured_total = 0
    structured_count = 0
    for type_name, count, avg in rows:
        if type_name in {"finding", "risk", "decision", "patch"}:
            structured_total += int((avg or 0) * (count or 0))
            structured_count += count or 0
    avg_artifact_bytes = (
        structured_total // structured_count
        if structured_count
        else DEFAULT_AVG_ARTIFACT_BYTES
    )

    stdout_lengths: list[int] = []
    for (data,) in verification_rows:
        try:
            payload = json.loads(data)
            stdout = payload.get("payload", {}).get("stdout") or ""
            stdout_lengths.append(len(stdout))
        except (json.JSONDecodeError, TypeError):
            continue
    avg_worker_stdout_bytes = (
        sum(stdout_lengths) // len(stdout_lengths)
        if stdout_lengths
        else DEFAULT_AVG_WORKER_STDOUT_BYTES
    )

    tasks = (task_row[0] if task_row else 0) or 1
    avg_artifacts_per_worker = (
        structured_count / tasks if tasks else DEFAULT_ARTIFACTS_PER_WORKER
    )

    return ArtifactFacts(
        source=str(db_path),
        sample_count=structured_count,
        avg_artifact_bytes=int(avg_artifact_bytes),
        avg_worker_stdout_bytes=int(avg_worker_stdout_bytes),
        avg_artifacts_per_worker=float(avg_artifacts_per_worker),
    )


def measure_codegraph_context(prompt: str, cwd: str) -> tuple[int, float]:
    start = time.monotonic()
    context = codegraph_context(prompt, cwd, max_nodes=15)
    elapsed = time.monotonic() - start
    return len(context or ""), round(elapsed, 3)


def model_costs(
    *,
    repo: RepoFacts,
    artifacts: ArtifactFacts,
    workers: int,
    codegraph_context_bytes: int,
    discovery_scan_ratio: float = DEFAULT_DISCOVERY_SCAN_RATIO,
) -> list[ConfigCost]:
    """Compute fresh-task and resume costs in bytes for each configuration.

    The "fresh task" is a single investigation that the user spends N
    worker-units of effort on. For Configs A and B, that's N sequential
    agent runs (no shared state). For Config C, it's one swarm with N
    parallel workers and a stitcher.

    The "resume" is the cost of consulting the result later:
    - A and B re-run the whole task because nothing persisted.
    - C reads the durable artifacts from SQLite (0 model tokens).
    """
    discovery_per_agent = int(repo.source_bytes * discovery_scan_ratio)
    artifacts_per_worker = max(1.0, artifacts.avg_artifacts_per_worker)
    structured_output_per_worker = int(artifacts_per_worker * artifacts.avg_artifact_bytes)

    # Config A: N sequential agent runs, each does its own discovery.
    a_per_unit = discovery_per_agent + AGENT_SELF_OUTPUT_BYTES
    a_fresh = workers * a_per_unit
    a_resume = a_fresh

    # Config B: N sequential agent runs, each calls codegraph_explore.
    b_per_unit = codegraph_context_bytes + TOOL_CALL_FRAME_BYTES + AGENT_SELF_OUTPUT_BYTES
    b_fresh = workers * b_per_unit
    b_resume = b_fresh

    # Config C: 1 shared codegraph query, N parallel workers (each prompt
    # has the context pre-injected, but the model still sees that input),
    # plus a stitcher pass that reads structured artifacts and outputs a
    # synthesis.
    c_worker_input = codegraph_context_bytes  # inlined per worker
    c_worker_output = structured_output_per_worker
    c_workers_total = workers * (c_worker_input + c_worker_output)
    c_stitcher_input = workers * structured_output_per_worker
    c_stitcher_output = STITCHER_OUTPUT_BYTES
    c_fresh = c_workers_total + c_stitcher_input + c_stitcher_output
    c_resume = 0  # reading SQLite costs 0 model tokens

    return [
        ConfigCost(label="A. Agent only", fresh_task_bytes=a_fresh, resume_bytes=a_resume),
        ConfigCost(label="B. CodeGraph alone", fresh_task_bytes=b_fresh, resume_bytes=b_resume),
        ConfigCost(
            label="C. Puppetmaster + CodeGraph",
            fresh_task_bytes=c_fresh,
            resume_bytes=c_resume,
        ),
    ]


def render_markdown(
    report: dict, *, price_per_million: float, follow_ups_sample: int = 5
) -> str:
    rows: list[str] = []
    rows.append(
        "# Three-way cost comparison: Agent / CodeGraph alone / Puppetmaster + CodeGraph"
    )
    rows.append("")
    rows.append(f"- Target repo: `{report['repo_path']}`")
    rows.append(f"- Workers per swarm: **{report['workers']}**")
    rows.append(f"- Token price assumption: **${price_per_million:.2f} / 1M tokens**")
    rows.append(f"- Ran at (UTC): `{report['ran_at']}`")
    rows.append("")

    rows.append("## Inputs (measured)")
    rows.append("")
    rows.append("| Measurement | Value |")
    rows.append("|---|---|")
    repo = report["repo"]
    rows.append(f"| repo source files | {repo['file_count']:,} |")
    rows.append(f"| repo source bytes | {repo['source_bytes']:,} |")
    rows.append(f"| `codegraph` available | {report['codegraph_available']} |")
    rows.append(f"| target initialized | {report['codegraph_initialized']} |")
    rows.append(f"| CodeGraph context bytes | {report['codegraph_context_bytes']:,} |")
    rows.append(f"| CodeGraph query (s) | {report['codegraph_query_seconds']:.3f} |")
    a = report["artifacts"]
    rows.append(f"| past artifact sample | {a['sample_count']:,} ({a['source']}) |")
    rows.append(f"| avg structured artifact (B) | {a['avg_artifact_bytes']:,} |")
    rows.append(f"| avg worker stdout (B) | {a['avg_worker_stdout_bytes']:,} |")
    rows.append(f"| avg artifacts per worker | {a['avg_artifacts_per_worker']:.2f} |")
    rows.append("")

    configs = [ConfigCost(**c) if not isinstance(c, ConfigCost) else c for c in report["configs"]]
    a_cfg, b_cfg, c_cfg = configs

    rows.append("## Fresh task cost (one investigation)")
    rows.append("")
    rows.append("| Config | Bytes | Tokens (~÷4) | $ at price assumption |")
    rows.append("|---|---|---|---|")
    for config in configs:
        tokens = config.fresh_task_bytes // CHARS_PER_TOKEN
        rows.append(
            f"| {config.label} | {config.fresh_task_bytes:,} B | "
            f"~{tokens:,} tok | ~${config.dollars(config.fresh_task_bytes, price_per_million):.4f} |"
        )
    rows.append("")

    if a_cfg.fresh_task_bytes:
        c_vs_a = (1 - c_cfg.fresh_task_bytes / a_cfg.fresh_task_bytes) * 100
    else:
        c_vs_a = 0.0
    if b_cfg.fresh_task_bytes:
        c_vs_b = (1 - c_cfg.fresh_task_bytes / b_cfg.fresh_task_bytes) * 100
    else:
        c_vs_b = 0.0
    rows.append(
        f"- **Puppetmaster + CodeGraph vs Agent only:** ~{c_vs_a:.1f}% fewer bytes per fresh task."
    )
    rows.append(
        f"- **Puppetmaster + CodeGraph vs CodeGraph alone:** ~{c_vs_b:+.1f}% bytes per fresh task. "
        "(For a single fresh task, Puppetmaster is roughly comparable to CodeGraph alone in raw "
        "tokens. The real wins show up in the session table below.)"
    )
    rows.append("")

    rows.append("## Session cost (1 task + K follow-up reads)")
    rows.append("")
    rows.append(
        "Many real workflows are not one-shot: you investigate, then ask follow-up questions "
        "about the same task. In Configs A and B, every follow-up is a fresh agent run "
        "(nothing persisted across sessions). In Config C, a follow-up is just SQLite — "
        "Puppetmaster reads the durable artifacts and presents them, costing 0 model tokens."
    )
    rows.append("")
    follow_up_values = sorted({0, 1, 5, 10, 25, follow_ups_sample})
    header = "| Config |" + "".join(f" K={k} |" for k in follow_up_values)
    rows.append(header)
    rows.append("|" + "---|" * (len(follow_up_values) + 1))
    for config in configs:
        cells = []
        for k in follow_up_values:
            session_bytes = config.session_bytes(k)
            cents = config.dollars(session_bytes, price_per_million)
            cells.append(f" ~${cents:.4f} ")
        rows.append(f"| {config.label} |{'|'.join(cells)}|")
    rows.append("")
    rows.append(
        "_Each cell is total $ for a session = the fresh task plus K follow-up reads, "
        "evaluated at the price assumption above._"
    )
    rows.append("")

    rows.append("## Where Puppetmaster's savings actually come from")
    rows.append("")
    rows.append(
        "1. **Shared CodeGraph query** — one `codegraph context` call seeds all N workers; "
        "Config B issues N separate `codegraph_explore` calls."
    )
    rows.append(
        "2. **Zero tool-call frames** — workers receive context inline in the prompt, so "
        f"each worker saves the ~{TOOL_CALL_FRAME_BYTES} B MCP envelope of an explore call."
    )
    rows.append(
        "3. **Structured synthesis** — the stitcher reads typed JSON artifacts (avg "
        f"{a['avg_artifact_bytes']:,} B each) instead of raw worker stdout (avg "
        f"{a['avg_worker_stdout_bytes']:,} B). For high-signal artifact swarms this can "
        "be smaller; for very verbose artifact swarms it can be larger. Your mileage varies."
    )
    rows.append(
        "4. **Durable resume** — the killer feature on multi-question workflows. Every "
        "follow-up read against a completed Puppetmaster job is free at the model level."
    )
    rows.append("")

    rows.append("## Honest caveats")
    rows.append("")
    rows.append(
        "- This is a **cost-structure model** built from measured repo and artifact data, "
        "not a live billing comparison. Exact per-task token counts require SDK-side stream "
        "instrumentation (on the roadmap)."
    )
    rows.append(
        "- Bytes → tokens uses the standard 4-chars-per-token heuristic; the real ratio "
        "depends on content."
    )
    rows.append(
        f"- Agent-only \"discovery\" assumes the agent scans ~{int(DEFAULT_DISCOVERY_SCAN_RATIO * 100)}% "
        "of the repo per task. Real discovery is highly task-dependent and frequently larger "
        "(the agent often re-reads files across turns); this assumption is conservative in "
        "Config A's favor."
    )
    rows.append(
        "- We do **not** model context-window compounding in B (where each `codegraph_explore` "
        "call's result accumulates in the agent's transcript across turns). Modelling that "
        "would make B significantly more expensive than shown here."
    )
    rows.append(
        "- Config C's resume cost is reported as 0 because reading SQLite does not call the "
        "model. Presenting the artifacts to a user still costs whatever wrapping you choose."
    )
    rows.append("")
    rows.append("## Reproduce")
    rows.append("")
    rows.append("```bash")
    rows.append(report["command"])
    rows.append("```")
    return "\n".join(rows)


def write_report(
    report: dict, output_path: Optional[Path], price_per_million: float
) -> tuple[Path, Path]:
    if output_path is None:
        timestamp = int(time.time())
        output_path = REPO_ROOT / "bench" / "results" / f"three_way_{timestamp}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(report, price_per_million=price_per_million), encoding="utf-8"
    )
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output_path, json_path


def build_report(
    *,
    cwd: Path,
    workers: int,
    prompt: str,
    artifacts_state_dir: Optional[Path],
    command: str,
    discovery_scan_ratio: float = DEFAULT_DISCOVERY_SCAN_RATIO,
) -> dict:
    repo = scan_repo(cwd)
    artifacts = load_artifact_facts(artifacts_state_dir)
    codegraph_context_bytes = 0
    codegraph_query_seconds = 0.0
    if codegraph_available() and codegraph_initialized(str(cwd)):
        codegraph_context_bytes, codegraph_query_seconds = measure_codegraph_context(
            prompt, str(cwd)
        )

    configs = model_costs(
        repo=repo,
        artifacts=artifacts,
        workers=workers,
        codegraph_context_bytes=codegraph_context_bytes or 4_000,
        discovery_scan_ratio=discovery_scan_ratio,
    )
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "repo_path": str(cwd),
        "workers": workers,
        "discovery_scan_ratio": discovery_scan_ratio,
        "repo": {
            "file_count": repo.file_count,
            "source_bytes": repo.source_bytes,
            "languages": repo.languages,
        },
        "artifacts": {
            "source": artifacts.source,
            "sample_count": artifacts.sample_count,
            "avg_artifact_bytes": artifacts.avg_artifact_bytes,
            "avg_worker_stdout_bytes": artifacts.avg_worker_stdout_bytes,
            "avg_artifacts_per_worker": artifacts.avg_artifacts_per_worker,
        },
        "codegraph_available": codegraph_available(),
        "codegraph_initialized": codegraph_initialized(str(cwd)),
        "codegraph_context_bytes": codegraph_context_bytes,
        "codegraph_query_seconds": codegraph_query_seconds,
        "configs": [
            {
                "label": c.label,
                "fresh_task_bytes": c.fresh_task_bytes,
                "resume_bytes": c.resume_bytes,
            }
            for c in configs
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Three-way cost comparison for Puppetmaster + CodeGraph."
    )
    parser.add_argument("--cwd", default=".", help="Target repository path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers in the hypothetical swarm (default: 4).",
    )
    parser.add_argument(
        "--prompt",
        default="map this repository's worker orchestration flow",
        help="Prompt used for the CodeGraph context query.",
    )
    parser.add_argument(
        "--artifacts-state",
        type=Path,
        help=(
            "Optional Puppetmaster state directory to read real artifact size "
            "distribution from (looks for state.sqlite3). Defaults to baked-in "
            "averages."
        ),
    )
    parser.add_argument(
        "--price-per-million",
        type=float,
        default=3.0,
        help="USD per 1M tokens used for $ estimates (default: 3.0 = Claude 3.5 Sonnet input).",
    )
    parser.add_argument(
        "--discovery-scan-ratio",
        type=float,
        default=DEFAULT_DISCOVERY_SCAN_RATIO,
        help=(
            f"Fraction of the repo an agent-only run scans for discovery "
            f"(default: {DEFAULT_DISCOVERY_SCAN_RATIO})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown report path (defaults to bench/results/three_way_<ts>.md).",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    command = " ".join(["python -m bench.three_way", *(argv or sys.argv[1:])])
    state_dir = args.artifacts_state.resolve() if args.artifacts_state else None

    report = build_report(
        cwd=cwd,
        workers=args.workers,
        prompt=args.prompt,
        artifacts_state_dir=state_dir,
        command=command,
        discovery_scan_ratio=args.discovery_scan_ratio,
    )

    md_path, json_path = write_report(report, args.output, args.price_per_million)
    print(render_markdown(report, price_per_million=args.price_per_million))
    print()
    print(f"Report (markdown): {md_path}")
    print(f"Report (json):     {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
