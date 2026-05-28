"""Live A/B benchmark with real OpenAI token receipts.

The other two benchmarks (``router_savings``, ``followup_cost``) use
the router's heuristic token estimates. This one calls the real
OpenAI Chat Completions API and uses the ``usage.prompt_tokens`` and
``usage.completion_tokens`` fields from the response — actual billing
units, not heuristics.

Method:
  1. Take a single fixed instruction (or ``--instruction``).
  2. Arm A (always-frontier): run with ``model="gpt-5.5"`` —
     simulates a user who has an OpenAI key but no routing and pins
     the strongest model as their default.
  3. Arm B (Puppetmaster-routed): classify the instruction via the
     router (same code path that runs in production), pick a model
     from the user's registry (filtered to ``adapter="openai"`` so
     this benchmark only compares OpenAI-vs-OpenAI), then run.
  4. For each arm, invoke ``OpenAIAdapter().run()`` directly and pull
     ``tokens_in``, ``tokens_out``, ``tokens_total`` from the
     verification artifact's payload. Multiply by the registry's
     per-MTok pricing to get the **real billed** dollar amount.
  5. Wall-clock both arms.
  6. Report the delta.

Requires ``OPENAI_API_KEY``. Costs real money — keep the instruction
small (the default is a one-shot ~50-token question, total spend
under a cent at gpt-5.5 pricing).

Optionally writes a JSON+markdown receipt under ``bench/results/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.adapters import OpenAIAdapter  # noqa: E402
from puppetmaster.model_registry import (  # noqa: E402
    ModelSpec,
    default_registry_path,
    enabled_specs,
    find,
    load_registry,
)
from puppetmaster.models import Task  # noqa: E402
from puppetmaster.router import (  # noqa: E402
    TaskSignals,
    classify_capability_needed,
    estimate_tokens_in,
    estimate_tokens_out,
    route_task,
)


DEFAULT_INSTRUCTION = (
    "In one short paragraph, describe what a hash table is and the two most "
    "common collision resolution strategies. Be concise. No code."
)


@dataclass
class ArmResult:
    arm: str
    model_id: str
    adapter_model_name: str
    capability_score: int
    wall_seconds: float
    tokens_in: int
    tokens_out: int
    tokens_total: int
    cost_usd: float
    result_text: str
    failed: bool
    failure: Optional[str]


def _find_openai_frontier(registry: list[ModelSpec]) -> ModelSpec:
    """Highest-capability enabled OpenAI model in the registry — the
    "always frontier" baseline for an OpenAI-only user."""
    openai_only = [s for s in enabled_specs(registry) if s.adapter == "openai"]
    if not openai_only:
        raise RuntimeError(
            "no openai-adapter models in registry. Run `python -m "
            "puppetmaster models init` or add openai/* entries manually."
        )
    return max(openai_only, key=lambda s: s.capability_score)


def _route_openai_only(
    instruction: str, role: str, registry: list[ModelSpec], policy: str
) -> ModelSpec:
    """Run the production router on an instruction, but constrain the
    candidate pool to OpenAI adapter entries so the A/B is apples-to-
    apples (both arms hit OpenAI's API, only the model differs)."""
    openai_only = [s for s in enabled_specs(registry) if s.adapter == "openai"]
    if not openai_only:
        raise RuntimeError("no openai-adapter models in registry")
    signals = TaskSignals(
        instruction=instruction,
        role=role,
        payload_size_chars=len(instruction),
    )
    decision = route_task(signals, openai_only, policy=policy)
    return decision.model


def _build_task(instruction: str, model_name: str) -> Task:
    return Task(
        job_id="bench_live_ab",
        role="explore",
        instruction=instruction,
        adapter="openai",
        payload={
            "prompt": instruction,
            "model": model_name,
            "cwd": str(REPO_ROOT),
            "timeout_seconds": 120,
            # Skip CodeGraph injection — the benchmark is about
            # raw routing economics, not CodeGraph integration.
            "disable_codegraph": True,
        },
    )


def _extract_tokens(verification_payload: dict) -> tuple[int, int, int, bool, Optional[str]]:
    tokens_in = int(verification_payload.get("tokens_in") or 0)
    tokens_out = int(verification_payload.get("tokens_out") or 0)
    tokens_total = int(
        verification_payload.get("tokens_total") or (tokens_in + tokens_out)
    )
    failure = verification_payload.get("failure")
    failed = failure is not None and failure != ""
    return tokens_in, tokens_out, tokens_total, failed, failure


def _run_arm(
    arm: str,
    instruction: str,
    spec: ModelSpec,
) -> ArmResult:
    adapter = OpenAIAdapter()
    task = _build_task(instruction, spec.adapter_model_name)
    start = time.perf_counter()
    artifacts = adapter.run(task, goal=instruction, worker_id=f"bench-{arm}")
    wall_seconds = time.perf_counter() - start

    verification = next(
        (a for a in artifacts if a.type.value == "verification"), None
    )
    if verification is None:
        raise RuntimeError(f"arm {arm}: adapter returned no verification artifact")

    payload = verification.payload or {}
    tokens_in, tokens_out, tokens_total, failed, failure = _extract_tokens(payload)
    cost = spec.estimate_cost_usd(tokens_in, tokens_out)
    result_text = str(payload.get("stdout") or "")

    return ArmResult(
        arm=arm,
        model_id=spec.id,
        adapter_model_name=spec.adapter_model_name,
        capability_score=spec.capability_score,
        wall_seconds=wall_seconds,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        cost_usd=cost,
        result_text=result_text,
        failed=failed,
        failure=failure,
    )


def render_markdown(report: dict) -> str:
    a = report["arm_baseline"]
    b = report["arm_puppetmaster"]
    delta = report["delta"]
    lines: list[str] = []
    lines.append("# Live Router A/B (real OpenAI token receipts)")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at']}`  ")
    lines.append(f"Instruction: {report['instruction'][:140]}{'...' if len(report['instruction']) > 140 else ''}  ")
    lines.append(f"Role: `{report['role']}`  Policy: `{report['policy']}`  ")
    lines.append(f"Capability needed (per router): **{report['capability_needed']}**  ")
    lines.append("")
    lines.append("## Per-arm receipts")
    lines.append("")
    lines.append("| Arm | Model | Cap | Wall (s) | tokens_in | tokens_out | $ (real) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| A (always frontier) | `{a['model_id']}` (`{a['adapter_model_name']}`) | "
        f"{a['capability_score']} | {a['wall_seconds']:.3f} | "
        f"{a['tokens_in']} | {a['tokens_out']} | ${a['cost_usd']:.6f} |"
    )
    lines.append(
        f"| B (Puppetmaster-routed) | `{b['model_id']}` (`{b['adapter_model_name']}`) | "
        f"{b['capability_score']} | {b['wall_seconds']:.3f} | "
        f"{b['tokens_in']} | {b['tokens_out']} | ${b['cost_usd']:.6f} |"
    )
    lines.append("")
    lines.append("## Delta")
    lines.append("")
    lines.append(f"- Cost savings: **${delta['cost_savings_usd']:.6f} ({delta['cost_savings_pct']:.1f}% cheaper)**")
    lines.append(f"- Wall-time delta: **{delta['wall_seconds_delta']:+.3f} s ({delta['wall_pct']:+.1f}%)**")
    lines.append(f"- Tokens (in): **{delta['tokens_in_delta']:+d}**")
    lines.append(f"- Tokens (out): **{delta['tokens_out_delta']:+d}**")
    lines.append(f"- Same model picked: **{delta['same_model']}**")
    lines.append("")

    lines.append("## Output samples")
    lines.append("")
    lines.append("Arm A (frontier) reply (first 400 chars):")
    lines.append("```")
    lines.append(a["result_text"][:400] if a["result_text"] else "<empty>")
    lines.append("```")
    lines.append("")
    lines.append("Arm B (Puppetmaster pick) reply (first 400 chars):")
    lines.append("```")
    lines.append(b["result_text"][:400] if b["result_text"] else "<empty>")
    lines.append("```")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Both arms hit the real OpenAI API. ``tokens_in`` / ``tokens_out`` "
        "are pulled from ``usage.prompt_tokens`` / ``usage.completion_tokens`` "
        "in the API response, not estimates."
    )
    lines.append(
        "- The candidate pool is restricted to OpenAI-adapter models so the "
        "A/B is apples-to-apples (same API surface, same network, same "
        "JSON-parsing path). Cursor-adapter models are excluded from this "
        "harness because the Cursor SDK does not expose token usage."
    )
    lines.append(
        "- The savings number assumes the user's baseline is \"always pin "
        "the strongest OpenAI model\". A user who already manually picks "
        "the right size for each task gets the same savings without "
        "Puppetmaster; the router's value is making that automatic."
    )
    lines.append(
        "- Output quality is **not** graded here. The cost claim is "
        "defensible standalone; the accuracy claim is not — see the "
        "README's Receipts section for what is and isn't currently "
        "measurable."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live OpenAI A/B (always-frontier vs Puppetmaster-routed)",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Task instruction to run both arms on.",
    )
    parser.add_argument(
        "--role",
        default="explore",
        help="Worker role (affects capability classification).",
    )
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=["balanced", "cheap", "quality"],
        help="Routing policy for arm B.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to models.json. Defaults to PUPPETMASTER_MODELS_PATH or ~/.puppetmaster/models.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "bench" / "results",
        help="Directory to write the markdown+JSON receipt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the routing decision without calling the API.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Pass --dry-run to test routing without API calls.", file=sys.stderr)
        return 1

    registry_path = args.registry_path or default_registry_path()
    registry = load_registry(registry_path if registry_path.is_file() else None)
    if not registry:
        print(
            f"ERROR: no models in registry at {registry_path}. "
            "Run `python -m puppetmaster models init` first.",
            file=sys.stderr,
        )
        return 1

    frontier_spec = _find_openai_frontier(registry)
    routed_spec = _route_openai_only(
        args.instruction, args.role, registry, args.policy
    )

    signals = TaskSignals(
        instruction=args.instruction,
        role=args.role,
        payload_size_chars=len(args.instruction),
    )
    capability_needed = classify_capability_needed(signals)

    if args.dry_run:
        print("Dry run — routing decision only, no API calls.")
        print(f"  Capability needed: {capability_needed}")
        print(f"  Arm A (always frontier): {frontier_spec.id} ({frontier_spec.adapter_model_name})")
        print(f"  Arm B (Puppetmaster-routed): {routed_spec.id} ({routed_spec.adapter_model_name})")
        print(f"  Same pick? {routed_spec.id == frontier_spec.id}")
        print(
            f"  Estimated tokens: in={estimate_tokens_in(signals)}, "
            f"out={estimate_tokens_out(signals)}"
        )
        return 0

    print(f"Capability needed: {capability_needed}")
    print(f"Arm A: {frontier_spec.id} ({frontier_spec.adapter_model_name})  Arm B: {routed_spec.id} ({routed_spec.adapter_model_name})")
    if routed_spec.id == frontier_spec.id:
        print(
            "WARN: router picked the same model as the baseline; A/B will "
            "show no cost delta. Try a smaller task or a different role."
        )

    print("Running Arm A (always frontier)...")
    arm_a = _run_arm("A", args.instruction, frontier_spec)
    print(
        f"  done — tokens={arm_a.tokens_in}/{arm_a.tokens_out}  "
        f"cost=${arm_a.cost_usd:.6f}  wall={arm_a.wall_seconds:.3f}s"
        + (f"  FAILED: {arm_a.failure}" if arm_a.failed else "")
    )
    print("Running Arm B (Puppetmaster-routed)...")
    arm_b = _run_arm("B", args.instruction, routed_spec)
    print(
        f"  done — tokens={arm_b.tokens_in}/{arm_b.tokens_out}  "
        f"cost=${arm_b.cost_usd:.6f}  wall={arm_b.wall_seconds:.3f}s"
        + (f"  FAILED: {arm_b.failure}" if arm_b.failed else "")
    )

    cost_savings = max(0.0, arm_a.cost_usd - arm_b.cost_usd)
    cost_savings_pct = (100.0 * cost_savings / arm_a.cost_usd) if arm_a.cost_usd > 0 else 0.0
    wall_delta = arm_b.wall_seconds - arm_a.wall_seconds
    wall_pct = (100.0 * wall_delta / arm_a.wall_seconds) if arm_a.wall_seconds > 0 else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instruction": args.instruction,
        "role": args.role,
        "policy": args.policy,
        "capability_needed": capability_needed,
        "arm_baseline": arm_a.__dict__,
        "arm_puppetmaster": arm_b.__dict__,
        "delta": {
            "cost_savings_usd": round(cost_savings, 6),
            "cost_savings_pct": round(cost_savings_pct, 2),
            "wall_seconds_delta": round(wall_delta, 3),
            "wall_pct": round(wall_pct, 2),
            "tokens_in_delta": arm_b.tokens_in - arm_a.tokens_in,
            "tokens_out_delta": arm_b.tokens_out - arm_a.tokens_out,
            "same_model": arm_a.model_id == arm_b.model_id,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = args.out_dir / f"router_live_ab_{stamp}.md"
    json_path = args.out_dir / f"router_live_ab_{stamp}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(render_markdown(report))
    print(f"\nReceipt: {md_path}")
    print(f"Receipt: {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
