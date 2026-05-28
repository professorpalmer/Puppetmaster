"""Router-savings benchmark: receipts for the "Puppetmaster delegates by complexity" claim.

Answers the question: "If I let Puppetmaster route each task to the
cheapest-sufficient model instead of always using the frontier model,
how much do I save?"

This is a **dry-run** benchmark — no live API calls, no API key needed.
It uses the same router code path that runs in production:

  1. Build a ``TaskSignals`` from a fixture task instruction.
  2. Call ``classify_capability_needed`` -> needed capability score.
  3. Call ``estimate_tokens_in`` / ``estimate_tokens_out`` -> token usage.
  4. Call ``route_task(..., policy="balanced")`` -> Puppetmaster's pick.
  5. Pick the registry's highest-capability model as the "always frontier"
     baseline — the cost you'd pay if you never configured routing.
  6. Cost = ``ModelSpec.estimate_cost_usd(tokens_in, tokens_out)`` for both.

Token estimates are heuristic (same heuristics that drive routing). The
$ numbers are estimates, not API receipts, but they're the SAME numbers
the router itself uses to decide — so the savings claim is internally
consistent with what users see in their ROUTING artifacts.

For a real-token A/B with live API receipts, see ``router_live_ab.py``
(which uses OpenAIAdapter and captures real ``tokens_in`` /
``tokens_out`` from the OpenAI response).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puppetmaster.model_registry import (  # noqa: E402
    ModelSpec,
    default_registry_path,
    enabled_specs,
    load_registry,
)
from puppetmaster.router import (  # noqa: E402
    TaskSignals,
    classify_capability_needed,
    estimate_tokens_in,
    estimate_tokens_out,
    route_task,
)


# A representative fixture spanning the three tiers Puppetmaster's
# router actually distinguishes. Each entry: (label, tier, role,
# instruction). Roles match real DEFAULT_WORKERS roles so the
# benchmark exercises the same _ROLE_BASE_SCORE table that production
# tasks hit.
FIXTURE_TASKS: list[tuple[str, str, str, str]] = [
    (
        "format-file",
        "easy",
        "verify-runtime",
        "Format this Python file with black and check that no lines exceed 100 chars.",
    ),
    (
        "rename-symbol",
        "easy",
        "verify-runtime",
        "Rename the local variable `tmp` to `scratch_path` inside the function `_resolve_dir` and run lint.",
    ),
    (
        "explain-module",
        "medium",
        "explore",
        "Produce a one-paragraph summary of what puppetmaster/router.py does and which functions are the entrypoints.",
    ),
    (
        "implement-endpoint",
        "medium",
        "implement",
        "Add a password reset endpoint to the auth module. Wire it through the existing rate-limit middleware and include tests.",
    ),
    (
        "security-audit",
        "hard",
        "audit",
        "Security audit every API endpoint in the codebase. Look for missing auth checks, IDOR, SSRF, and unbounded input. Cross-repo if relevant.",
    ),
    (
        "perf-architect",
        "hard",
        "architect",
        "Design a non-trivial performance refactor for the worker dispatch path. Consider concurrency limits across the whole repo.",
    ),
]


def pick_frontier(registry: list[ModelSpec]) -> ModelSpec:
    """The 'always frontier' baseline: the highest-capability enabled model.

    This is what a user gets if they never configure routing and just
    pin the strongest model in their plan as the default — a common
    pattern when "use the best model for everything" is the safe play.
    """
    enabled = enabled_specs(registry)
    if not enabled:
        raise RuntimeError("registry has no enabled models")
    return max(enabled, key=lambda s: s.capability_score)


def build_signals(role: str, instruction: str) -> TaskSignals:
    return TaskSignals(
        instruction=instruction,
        role=role,
        payload_size_chars=len(instruction),
    )


def benchmark_one_task(
    label: str,
    tier: str,
    role: str,
    instruction: str,
    registry: list[ModelSpec],
    frontier: ModelSpec,
    policy: str,
) -> dict:
    signals = build_signals(role, instruction)
    tokens_in = estimate_tokens_in(signals)
    tokens_out = estimate_tokens_out(signals)
    capability_needed = classify_capability_needed(signals)

    decision = route_task(signals, registry, policy=policy)
    pm_cost = decision.model.estimate_cost_usd(tokens_in, tokens_out)
    frontier_cost = frontier.estimate_cost_usd(tokens_in, tokens_out)
    savings_usd = max(0.0, frontier_cost - pm_cost)
    savings_pct = (
        100.0 * (savings_usd / frontier_cost) if frontier_cost > 0 else 0.0
    )
    same_pick = decision.model.id == frontier.id

    return {
        "label": label,
        "tier": tier,
        "role": role,
        "instruction": instruction,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "capability_needed": capability_needed,
        "puppetmaster_pick": {
            "id": decision.model.id,
            "adapter": decision.model.adapter,
            "adapter_model_name": decision.model.adapter_model_name,
            "capability_score": decision.model.capability_score,
            "input_per_mtok_usd": decision.model.input_per_mtok_usd,
            "output_per_mtok_usd": decision.model.output_per_mtok_usd,
        },
        "frontier_baseline": {
            "id": frontier.id,
            "adapter": frontier.adapter,
            "adapter_model_name": frontier.adapter_model_name,
            "capability_score": frontier.capability_score,
            "input_per_mtok_usd": frontier.input_per_mtok_usd,
            "output_per_mtok_usd": frontier.output_per_mtok_usd,
        },
        "puppetmaster_cost_usd": round(pm_cost, 6),
        "frontier_cost_usd": round(frontier_cost, 6),
        "savings_usd": round(savings_usd, 6),
        "savings_pct": round(savings_pct, 2),
        "same_pick": same_pick,
        "reason": decision.reason,
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Router Savings Benchmark")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at']}`  ")
    lines.append(f"Policy: `{report['policy']}`  ")
    lines.append(f"Registry: `{report['registry_path']}` ({report['registry_size']} enabled models)  ")
    fb = report["frontier_baseline"]
    lines.append(
        f"Frontier baseline: `{fb['id']}` (cap {fb['capability_score']}, "
        f"${fb['input_per_mtok_usd']:.3f}/${fb['output_per_mtok_usd']:.3f} per MTok)  "
    )
    lines.append("")

    lines.append("## Per-task breakdown")
    lines.append("")
    lines.append(
        "| Task | Tier | Role | Cap needed | Puppetmaster pick | Tokens in/out | PM $ | Baseline $ | Savings % |"
    )
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|")
    for r in report["tasks"]:
        pm = r["puppetmaster_pick"]
        same = " (same as baseline)" if r["same_pick"] else ""
        lines.append(
            f"| {r['label']} | {r['tier']} | {r['role']} | "
            f"{r['capability_needed']} | "
            f"`{pm['id']}` (cap {pm['capability_score']}){same} | "
            f"{r['tokens_in']}/{r['tokens_out']} | "
            f"${r['puppetmaster_cost_usd']:.6f} | "
            f"${r['frontier_cost_usd']:.6f} | "
            f"{r['savings_pct']:.1f}% |"
        )
    lines.append("")

    t = report["totals"]
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Tasks benchmarked: **{t['task_count']}**")
    lines.append(f"- Tasks where router picked the same model as baseline: **{t['same_pick_count']}**")
    lines.append(f"- Total tokens (estimated): **{t['total_tokens_in']:,} in / {t['total_tokens_out']:,} out**")
    lines.append(f"- Puppetmaster total cost: **${t['puppetmaster_cost_usd']:.6f}**")
    lines.append(f"- Frontier baseline cost: **${t['frontier_cost_usd']:.6f}**")
    lines.append(f"- Total savings: **${t['savings_usd']:.6f} ({t['savings_pct']:.1f}% cheaper)**")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Token counts are **heuristic estimates** "
        "(`puppetmaster.router.estimate_tokens_in/out`), not API receipts. "
        "They are the same estimates the router uses internally to choose "
        "a model, so this benchmark is internally consistent with the "
        "ROUTING artifacts you see at runtime — but the dollar totals are "
        "directional, not billing-accurate."
    )
    lines.append(
        "- The \"always frontier\" baseline reflects the cost of a user who "
        "never configures routing and pins the highest-capability model. It "
        "is **not** the cost of \"Cursor alone\"; Cursor's bundled models "
        "are priced at $0 in this registry because they roll into the "
        "Cursor plan."
    )
    lines.append(
        "- For receipts with real API token counts (no estimates), run "
        "`python -m bench.router_live_ab` with `OPENAI_API_KEY` set."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Router savings benchmark (dry-run, no API calls)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to models.json. Defaults to PUPPETMASTER_MODELS_PATH or ~/.puppetmaster/models.json.",
    )
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=["balanced", "cheap", "quality"],
        help="Routing policy to benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "bench" / "results",
        help="Directory to write the markdown+JSON receipt.",
    )
    args = parser.parse_args(argv)

    registry_path = args.registry_path or default_registry_path()
    registry = load_registry(registry_path if registry_path.is_file() else None)
    if not registry:
        print(
            f"ERROR: no models in registry at {registry_path}. "
            "Run `python -m puppetmaster models init` first.",
            file=sys.stderr,
        )
        return 1

    enabled = enabled_specs(registry)
    frontier = pick_frontier(registry)

    tasks: list[dict] = []
    for label, tier, role, instruction in FIXTURE_TASKS:
        tasks.append(
            benchmark_one_task(
                label, tier, role, instruction, enabled, frontier, args.policy
            )
        )

    totals = {
        "task_count": len(tasks),
        "same_pick_count": sum(1 for t in tasks if t["same_pick"]),
        "total_tokens_in": sum(t["tokens_in"] for t in tasks),
        "total_tokens_out": sum(t["tokens_out"] for t in tasks),
        "puppetmaster_cost_usd": round(
            sum(t["puppetmaster_cost_usd"] for t in tasks), 6
        ),
        "frontier_cost_usd": round(
            sum(t["frontier_cost_usd"] for t in tasks), 6
        ),
        "savings_usd": round(sum(t["savings_usd"] for t in tasks), 6),
    }
    totals["savings_pct"] = round(
        (100.0 * totals["savings_usd"] / totals["frontier_cost_usd"])
        if totals["frontier_cost_usd"] > 0
        else 0.0,
        2,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "registry_path": str(registry_path),
        "registry_size": len(enabled),
        "frontier_baseline": {
            "id": frontier.id,
            "capability_score": frontier.capability_score,
            "input_per_mtok_usd": frontier.input_per_mtok_usd,
            "output_per_mtok_usd": frontier.output_per_mtok_usd,
        },
        "tasks": tasks,
        "totals": totals,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = args.out_dir / f"router_savings_{stamp}.md"
    json_path = args.out_dir / f"router_savings_{stamp}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render_markdown(report))
    print(f"\nReceipt: {md_path}")
    print(f"Receipt: {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
