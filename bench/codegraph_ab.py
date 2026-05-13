"""A/B benchmark: Cursor SDK worker with vs. without CodeGraph context.

Two modes:

- ``--dry-run``: measures CodeGraph's *prompt enrichment* alone (no Cursor SDK
  calls, no API key needed). Reports how many characters of pre-resolved
  context CodeGraph injected vs. raw, and how long the context query took.
  This is the cheapest defensible signal that CodeGraph saves exploration
  work.

- live mode (default when ``CURSOR_API_KEY`` is set): runs the same task twice
  through Puppetmaster's CursorAdapter -- once with CodeGraph disabled
  (``payload.disable_codegraph=true``), once with it enabled -- and compares
  wall-clock seconds, artifact yield, and stdout size.

Run from the repo root:

    python -m bench.codegraph_ab --cwd /path/to/target/repo --prompt "..."
    python -m bench.codegraph_ab --cwd . --prompt @bench/prompts/example.txt --dry-run

Reports are written to ``bench/results/<timestamp>.md`` plus a matching
``.json`` for downstream tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.adapters import CursorAdapter  # noqa: E402
from puppetmaster.codegraph import (  # noqa: E402
    codegraph_available,
    codegraph_context,
    codegraph_initialized,
)
from puppetmaster.models import ArtifactType, Task  # noqa: E402


@dataclass
class EnrichmentMeasurement:
    raw_prompt_chars: int
    injected_context_chars: int
    codegraph_query_seconds: float
    codegraph_available: bool
    codegraph_initialized: bool

    @property
    def injection_ratio(self) -> float:
        if self.raw_prompt_chars == 0:
            return 0.0
        return self.injected_context_chars / self.raw_prompt_chars


@dataclass
class ArmResult:
    label: str
    wall_seconds: float
    artifact_count: int
    structured_artifact_count: int
    result_status: Optional[str]
    cursor_status: Optional[str]
    stdout_chars: int
    codegraph_used: bool
    failure: Optional[str]


def load_prompt(arg: str) -> str:
    if arg.startswith("@"):
        return Path(arg[1:]).read_text(encoding="utf-8")
    return arg


def measure_enrichment(
    prompt: str, cwd: str, *, max_nodes: int = 15
) -> EnrichmentMeasurement:
    """Measure CodeGraph's prompt enrichment without invoking Cursor."""
    start = time.monotonic()
    context = codegraph_context(prompt, cwd, max_nodes=max_nodes)
    elapsed = time.monotonic() - start
    return EnrichmentMeasurement(
        raw_prompt_chars=len(prompt),
        injected_context_chars=len(context or ""),
        codegraph_query_seconds=round(elapsed, 3),
        codegraph_available=codegraph_available(),
        codegraph_initialized=codegraph_initialized(cwd),
    )


def run_cursor_arm(
    prompt: str,
    cwd: str,
    model: str,
    *,
    label: str,
    disable_codegraph: bool,
    timeout_seconds: int = 600,
) -> ArmResult:
    """Run one Cursor SDK invocation through Puppetmaster's CursorAdapter."""
    task = Task(
        job_id="bench",
        role="bench",
        instruction=prompt,
        adapter="cursor",
        payload={
            "prompt": prompt,
            "cwd": cwd,
            "model": model,
            "timeout_seconds": timeout_seconds,
            "disable_codegraph": disable_codegraph,
        },
    )
    start = time.monotonic()
    artifacts = CursorAdapter().run(task, "bench-goal", "bench-worker")
    elapsed = time.monotonic() - start

    structured = [a for a in artifacts if a.type != ArtifactType.VERIFICATION]
    verification = next(
        (a for a in artifacts if a.type == ArtifactType.VERIFICATION), None
    )
    payload = verification.payload if verification else {}
    evidence = verification.evidence if verification else []

    return ArmResult(
        label=label,
        wall_seconds=round(elapsed, 2),
        artifact_count=len(artifacts),
        structured_artifact_count=len(structured),
        result_status=payload.get("result"),
        cursor_status=payload.get("cursor_status"),
        stdout_chars=len(payload.get("stdout") or ""),
        codegraph_used="context:codegraph" in evidence,
        failure=payload.get("failure"),
    )


def render_markdown(report: dict) -> str:
    rows: list[str] = []
    rows.append("# Puppetmaster + CodeGraph A/B benchmark")
    rows.append("")
    rows.append(f"- Target repo: `{report['target_repo']}`")
    rows.append(f"- Prompt: {report['prompt_summary']}")
    rows.append(f"- Ran at (UTC): `{report['ran_at']}`")
    rows.append(f"- Cursor model: `{report['model']}`")
    rows.append("")
    rows.append("## CodeGraph prompt enrichment (no API key required)")
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|---|---|")
    e = report["enrichment"]
    rows.append(f"| `codegraph` CLI available | {e['codegraph_available']} |")
    rows.append(f"| target workspace initialized | {e['codegraph_initialized']} |")
    rows.append(f"| raw prompt characters | {e['raw_prompt_chars']:,} |")
    rows.append(f"| CodeGraph injected characters | {e['injected_context_chars']:,} |")
    rows.append(f"| injection ratio (injected / raw) | {e['injection_ratio']:.2f}x |")
    rows.append(
        f"| CodeGraph context query | {e['codegraph_query_seconds']:.3f}s |"
    )
    rows.append("")
    rows.append(
        "_Injected characters are pre-resolved entry points and related symbols. "
        "Without CodeGraph, an agent would discover the same surface with multiple "
        "grep / read / list passes — each one costing model tokens and time._"
    )
    rows.append("")

    if "arm_a" in report and "arm_b" in report:
        a = report["arm_a"]
        b = report["arm_b"]
        delta_seconds = b["wall_seconds"] - a["wall_seconds"]
        delta_artifacts = b["structured_artifact_count"] - a["structured_artifact_count"]
        delta_stdout = b["stdout_chars"] - a["stdout_chars"]
        rows.append("## Live Cursor SDK A/B")
        rows.append("")
        rows.append("| Metric | A: CodeGraph OFF | B: CodeGraph ON | Δ (B - A) |")
        rows.append("|---|---|---|---|")
        rows.append(
            f"| wall-clock seconds | {a['wall_seconds']} | {b['wall_seconds']} | "
            f"{delta_seconds:+.2f} |"
        )
        rows.append(
            f"| structured artifacts | {a['structured_artifact_count']} | "
            f"{b['structured_artifact_count']} | {delta_artifacts:+} |"
        )
        rows.append(
            f"| total artifacts | {a['artifact_count']} | {b['artifact_count']} | "
            f"{b['artifact_count'] - a['artifact_count']:+} |"
        )
        rows.append(
            f"| result_status | `{a['result_status']}` | `{b['result_status']}` | — |"
        )
        rows.append(
            f"| stdout characters | {a['stdout_chars']:,} | {b['stdout_chars']:,} | "
            f"{delta_stdout:+,} |"
        )
        rows.append(
            f"| `context:codegraph` in evidence | {a['codegraph_used']} | "
            f"{b['codegraph_used']} | — |"
        )
        rows.append("")

    rows.append("## Reproduce")
    rows.append("")
    rows.append("```bash")
    rows.append(report["command"])
    rows.append("```")
    rows.append("")
    rows.append(
        "_This is a directional measurement, not an exact token/dollar number. "
        "Exact token usage requires SDK-side instrumentation, which is on the roadmap._"
    )
    return "\n".join(rows)


def write_report(report: dict, output_path: Optional[Path]) -> tuple[Path, Path]:
    if output_path is None:
        timestamp = int(time.time())
        output_path = REPO_ROOT / "bench" / "results" / f"{timestamp}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output_path, json_path


def build_report(
    *,
    cwd: str,
    prompt: str,
    model: str,
    command: str,
    enrichment: EnrichmentMeasurement,
    arm_a: Optional[ArmResult] = None,
    arm_b: Optional[ArmResult] = None,
) -> dict:
    report: dict = {
        "target_repo": cwd,
        "prompt_summary": prompt[:160] + ("..." if len(prompt) > 160 else ""),
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "command": command,
        "enrichment": {
            **asdict(enrichment),
            "injection_ratio": enrichment.injection_ratio,
        },
    }
    if arm_a is not None:
        report["arm_a"] = asdict(arm_a)
    if arm_b is not None:
        report["arm_b"] = asdict(arm_b)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark Puppetmaster's CodeGraph integration."
    )
    parser.add_argument("--cwd", required=True, help="Target repository path.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text, or '@path/to/file.txt' to load from disk.",
    )
    parser.add_argument(
        "--model", default="default", help="Cursor model id (default: 'default')."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the live Cursor SDK runs; report enrichment stats only.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Per-arm Cursor SDK timeout (default: 600).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown report path (defaults to bench/results/<ts>.md).",
    )
    args = parser.parse_args(argv)

    prompt = load_prompt(args.prompt)
    cwd = str(Path(args.cwd).resolve())
    command = " ".join(["python -m bench.codegraph_ab", *(argv or sys.argv[1:])])

    enrichment = measure_enrichment(prompt, cwd)

    arm_a = arm_b = None
    live = not args.dry_run
    if live and not os.environ.get("CURSOR_API_KEY"):
        print(
            "CURSOR_API_KEY not set; falling back to dry-run mode (enrichment "
            "stats only). Re-run with the key set to capture live A/B numbers.",
            file=sys.stderr,
        )
        live = False
    if live:
        arm_a = run_cursor_arm(
            prompt,
            cwd,
            args.model,
            label="codegraph_off",
            disable_codegraph=True,
            timeout_seconds=args.timeout_seconds,
        )
        arm_b = run_cursor_arm(
            prompt,
            cwd,
            args.model,
            label="codegraph_on",
            disable_codegraph=False,
            timeout_seconds=args.timeout_seconds,
        )

    report = build_report(
        cwd=cwd,
        prompt=prompt,
        model=args.model,
        command=command,
        enrichment=enrichment,
        arm_a=arm_a,
        arm_b=arm_b,
    )
    md_path, json_path = write_report(report, args.output)

    print(render_markdown(report))
    print()
    print(f"Report (markdown): {md_path}")
    print(f"Report (json):     {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
