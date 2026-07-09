from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO

from puppetmaster.codegraph_repair import repair_codegraph_sqlite
from puppetmaster.config import load_config
from puppetmaster.diagnostics import adapter_status, run_doctor, starter_config
from puppetmaster.installers import (
    CLAUDE_NEXT_STEPS_GUIDANCE,
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    HERMES_NEXT_STEPS_GUIDANCE,
    InstallResult,
    UninstallResult,
    ensure_cursor_sdk,
    install_claude_mcp,
    install_codex_mcp,
    install_cursor_mcp,
    install_hermes_mcp,
    install_hermes_plugin,
    install_hermes_skill,
    list_skill_candidates,
    promote_skill_candidate,
    resolve_claude_command,
    set_hermes_mcp_env,
    uninstall_claude_mcp,
    uninstall_codex_mcp,
    uninstall_cursor_mcp,
    uninstall_hermes_mcp,
)
from puppetmaster.rules import (
    VALID_TARGETS,
    RulesInstallResult,
    install_rules,
    uninstall_rules,
)
from puppetmaster.hook_installers import (
    VALID_HOOK_TARGETS,
    install_hermes_hooks,
    install_hooks,
    uninstall_hermes_hooks,
    uninstall_hooks,
)
from puppetmaster.mcp_registry import (
    kill_stale as registry_kill_stale,
    list_entries as registry_list_entries,
    prune_dead as registry_prune_dead,
    summarize as registry_summarize,
)
from puppetmaster.redaction import redact_secrets
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.state import (
    find_state_dir_for_job,
    list_project_state_dirs,
    resolve_state_dir,
)
from puppetmaster.store_factory import create_store
from puppetmaster.stitcher import Stitcher
from puppetmaster.worker_runtime import WorkerDaemon
from puppetmaster.workers import WorkerSpec

from puppetmaster.cli.helpers import (
    _print_token_usage,
    _registry_path_from_args,
)


def _parse_inline_rules(rule_args: list[str]) -> list[dict]:
    """Turn 'src/**/*.py=>tests/{stem}_test.py,tests/smoke.py' shorthand into
    mapping rules."""
    rules: list[dict] = []
    for raw in rule_args or []:
        if "=>" not in raw:
            raise SystemExit(f"affected: bad --rule (missing '=>'): {raw!r}")
        match, specs = raw.split("=>", 1)
        rules.append({
            "match": match.strip(),
            "specs": [s.strip() for s in specs.split(",") if s.strip()],
        })
    return rules

def _run_affected_command(args) -> int:
    from puppetmaster.affected import affected_specs, changed_files_from_git, load_mapping

    cwd = Path(args.cwd)
    mapping: dict[str, Any] = {}
    if args.config:
        mapping = load_mapping(args.config)
    inline_rules = _parse_inline_rules(args.rule)
    if inline_rules:
        mapping = dict(mapping)
        mapping["rules"] = list(mapping.get("rules") or []) + inline_rules
    if not mapping.get("rules") and not mapping.get("command"):
        raise SystemExit("affected: provide --config and/or --rule defining a mapping")

    if args.git_range:
        changed = changed_files_from_git(cwd, args.git_range)
    elif args.changed:
        changed = list(args.changed)
    else:
        changed = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]

    specs = affected_specs(changed, mapping, cwd=cwd)
    if args.json:
        print(json.dumps({"changed": changed, "affected_specs": specs}, indent=2))
    else:
        for spec in specs:
            print(spec)
    return 0

def _run_rollup_command(args, store) -> int:
    from puppetmaster.lifecycle import rollup_stores

    import puppetmaster.cli as cli

    rollup = rollup_stores(cli._gc_target_stores(args, store), effort_id=args.effort)
    if args.json:
        print(json.dumps(rollup, indent=2))
        return 0
    scope = f"effort '{args.effort}'" if args.effort else "all jobs"
    print(f"rollup ({scope}):")
    print(f"  jobs:      {rollup['jobs']}  {rollup['jobs_by_status']}")
    print(f"  artifacts: {rollup['artifacts']}")
    print(f"  est. cost: ${rollup['estimated_cost_usd']:.6f} (pre-flight routing estimate)")
    usage = rollup["token_usage"]
    if usage["measured_runs"] or usage["estimated_runs"]:
        print(
            f"  tokens:    {usage['measured_tokens_in'] + usage['measured_tokens_out']:,} measured / "
            f"~{usage['estimated_tokens_in'] + usage['estimated_tokens_out']:,} estimated"
        )
    if not args.effort and rollup["efforts_seen"]:
        print(f"  efforts seen: {', '.join(rollup['efforts_seen'])}")
    return 0

def _gate_specs_from_args(args) -> list[dict]:
    """Translate the gate flags + --gates-json into the same spec list the
    runtime resolves from ``task.payload['gates']``."""
    specs: list[dict] = []
    if args.gates_json:
        parsed = json.loads(args.gates_json)
        if not isinstance(parsed, list):
            raise ValueError("--gates-json must be a JSON array of gate objects")
        specs.extend(s for s in parsed if isinstance(s, dict) and s.get("kind"))
    if args.require_diff:
        specs.append({"kind": "require_diff"})
    if args.gate_command:
        specs.append({"kind": "command", "command": args.gate_command})
    if args.ratchet_command or args.metric:
        if not (args.ratchet_command and args.metric):
            raise ValueError("--ratchet-command and --metric must be given together")
        specs.append(
            {"kind": "ratchet", "command": args.ratchet_command, "metric": args.metric}
        )
    if args.committed:
        specs.append({"kind": "committed"})
    return specs

def _run_gate_command(args, store) -> int:
    """Replay completion gates against a working tree outside a worker run, so
    a parent agent or CI can enforce the very same post-conditions the runtime
    applies at task completion. Exits non-zero when any gate fails."""
    from puppetmaster.gates import evaluate_task_gates
    from puppetmaster.models import Task

    try:
        specs = _gate_specs_from_args(args)
    except ValueError as exc:
        print(f"gate: {exc}")
        return 2
    if not specs:
        print(
            "gate: no gates specified. Pass --require-diff / --command / "
            "--ratchet-command+--metric / --committed / --gates-json."
        )
        return 2

    cwd = Path(args.cwd).resolve()
    task = Task(
        job_id="gate-replay",
        role="gate",
        instruction="gate replay",
        payload={"gates": specs, "cwd": str(cwd)},
    )
    evaluation = evaluate_task_gates(
        task, artifacts=[], store=store, worker_id="gate-replay", cwd=cwd
    )

    rows = [
        {
            "gate": result.name,
            "kind": result.kind,
            "passed": result.passed,
            "reason": result.reason,
        }
        for result in evaluation.results
    ]
    if args.json:
        print(json.dumps({"passed": evaluation.passed, "gates": rows}, indent=2))
    else:
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"  [{mark}] {row['gate']} ({row['kind']}): {row['reason']}")
        print(f"\ngate: {'all gates passed' if evaluation.passed else 'GATE FAILED'}")
    return 0 if evaluation.passed else 1

def _run_preflight_command(args) -> int:
    """Check an adapter's auth/billing posture (and Cursor model) before dispatch."""
    import json as _json

    from puppetmaster.preflight import preflight_check

    catalog_fetcher = None
    if args.adapter == "cursor" and args.model:
        from puppetmaster.cursor_discovery import fetch_cursor_catalog

        catalog_fetcher = fetch_cursor_catalog

    result = preflight_check(
        args.adapter,
        args.model,
        allow_api_billing=not args.no_api_billing,
        live=getattr(args, "live", False),
        catalog_fetcher=catalog_fetcher,
    )

    if args.json:
        print(_json.dumps(result.as_dict(), indent=2))
    else:
        status = "READY" if result.ok else "BLOCKED"
        print(f"{status:8} {result.adapter:12} billing={result.billing}")
        print(f"  {result.reason}")
    return 0 if result.ok else 1

def _run_audit_command(args, store) -> int:
    """Analyze past routing behavior and propose conservative score changes.

    Read-only by default: prints a per-model report (picks, mean confidence,
    escalation rate, spend) plus a suggested models.json diff. ``--apply``
    writes the suggested score changes; nothing is mutated otherwise.
    """
    import json as _json
    from dataclasses import asdict, replace

    from puppetmaster.audit import build_audit_report, collect_records
    from puppetmaster.model_registry import (
        default_registry_path,
        load_registry,
        save_registry,
    )

    registry_path = _registry_path_from_args(args) or default_registry_path()
    registry = load_registry(registry_path)
    scores = {s.id: s.capability_score for s in registry}
    specs_by_id = {s.id: s for s in registry}

    def actual_cost_fn(model_id: str, tokens_in: int, tokens_out: int) -> float:
        spec = specs_by_id.get(model_id)
        # Price actuals with the same marginal-cost basis the router used for its
        # estimate, so plan-billed models read $0 on both sides (honest parity).
        return spec.marginal_cost_usd(tokens_in, tokens_out) if spec else 0.0

    records, jobs_considered = collect_records(store, window_days=args.window)
    report = build_audit_report(
        records,
        scores,
        window_days=args.window,
        jobs_considered=jobs_considered,
        actual_cost_fn=actual_cost_fn,
    )

    if args.json:
        print(
            _json.dumps(
                {
                    "jobs_considered": report.jobs_considered,
                    "tasks_considered": report.tasks_considered,
                    "window_days": report.window_days,
                    "total_est_spend_usd": report.total_est_spend_usd,
                    "reconciliation": {
                        "tasks_with_actuals": report.tasks_with_actuals,
                        "total_est_tokens": report.total_est_tokens,
                        "total_actual_tokens": report.total_actual_tokens,
                        "token_drift_ratio": report.token_drift_ratio,
                        "total_actual_spend_usd": report.total_actual_spend_usd,
                        "cost_drift_ratio": report.cost_drift_ratio,
                    },
                    "models": [asdict(m) for m in report.models],
                    "suggestions": report.suggestions,
                },
                indent=2,
            )
        )
    else:
        window = f"{args.window:g}d" if args.window else "all time"
        print(
            f"Routing audit — {report.jobs_considered} jobs, "
            f"{report.tasks_considered} tasks ({window}); "
            f"est spend ${report.total_est_spend_usd:.4f}"
        )
        if report.tasks_considered == 0:
            print(
                "  No router-placed tasks found. Run some auto_route work first "
                "(routing decisions are recorded as durable artifacts)."
            )
        else:
            print(
                f"  {'model':<26}{'picks':>6}{'conf':>7}{'esc%':>7}{'drift':>8}  flags"
            )
            for m in report.models:
                if m.selections == 0 and m.runs_with_confidence == 0:
                    continue
                conf = f"{m.mean_confidence:.2f}" if m.mean_confidence is not None else "  -"
                esc = f"{m.escalated_away_rate:.0%}"
                drift = f"{m.token_drift_ratio:.2f}x" if m.token_drift_ratio is not None else "  -"
                print(
                    f"  {m.model_id:<26}{m.selections:>6}{conf:>7}{esc:>7}{drift:>8}  "
                    f"{', '.join(m.flags)}"
                )
            if report.tasks_with_actuals:
                tok = report.token_drift_ratio
                tok_str = f"{tok:.2f}x actual/est" if tok is not None else "n/a"
                line = (
                    f"  reconciled {report.tasks_with_actuals}/{report.tasks_considered} "
                    f"tasks: tokens {report.total_actual_tokens:,} actual vs "
                    f"{report.total_est_tokens:,} est ({tok_str})"
                )
                cost = report.cost_drift_ratio
                if cost is not None:
                    line += (
                        f"; metered cost ${report.total_actual_spend_usd:.4f} actual vs "
                        f"${report.total_est_spend_reconciled_usd:.4f} est ({cost:.2f}x)"
                    )
                else:
                    line += "; cost $0 (plan-billed — tokens are the real signal)"
                print(line)
            else:
                print(
                    "  No token usage recorded yet — estimate-vs-actual drift will "
                    "populate once routed runs report usage."
                )
        if report.suggestions:
            print("\nSuggested score changes (your assertion stays the source of truth):")
            for s in report.suggestions:
                print(f"  {s['model_id']}: {s['from_score']} -> {s['to_score']}")
                print(f"      {s['rationale']}")
        elif report.tasks_considered:
            print("\nNo score changes suggested — routing looks well-calibrated.")

    if args.apply:
        if not report.suggestions:
            print("\nNothing to apply.")
            return 0
        by_id = {s.id: s for s in registry}
        changed = 0
        for sug in report.suggestions:
            spec = by_id.get(sug["model_id"])
            if spec is None:
                continue
            by_id[sug["model_id"]] = replace(spec, capability_score=sug["to_score"])
            changed += 1
        if changed:
            save_registry(list(by_id.values()), registry_path)
            print(f"\nApplied {changed} score change(s) to {registry_path}.")
    return 0

def _run_savings_command(args, state_dir) -> int:
    """Print the cumulative savings receipt. Read-only; local; emits nothing."""
    import json as _json

    from puppetmaster.savings import COUNTERFACTUAL_MODEL_ENV, build_report
    from puppetmaster.state import list_project_state_dirs
    from puppetmaster.store_factory import create_store

    dirs = [state_dir]
    if getattr(args, "all_projects", False):
        seen = {state_dir.resolve()}
        for d in list_project_state_dirs():
            if d.resolve() not in seen and d.exists():
                seen.add(d.resolve())
                dirs.append(d)
    stores = []
    for d in dirs:
        try:
            stores.append(create_store(args.backend, d))
        except Exception:
            continue

    report = build_report(stores, window_days=args.window)
    routing = report["routing"]
    cg = report["codegraph"]
    heal = report["self_heal"]
    reads = report["reads"]
    memory_cost = report["memory_cost"]
    tool_offload = report.get("tool_offload") or {
        "offloads": 0, "tokens_saved": 0, "chars_saved": 0,
    }
    metrics = report["metrics"]
    cf = report["counterfactual"]

    if args.json:
        from dataclasses import asdict

        print(
            _json.dumps(
                {
                    "window_days": report["window_days"],
                    "jobs_considered": report["jobs_considered"],
                    "routing": asdict(routing),
                    "routing_pct_cheaper": round(routing.pct_cheaper, 1),
                    "self_heal": asdict(heal),
                    "codegraph": cg,
                    "reads": reads,
                    "memory_cost": memory_cost,
                    "tool_offload": tool_offload,
                    "metrics": metrics,
                    "counterfactual": asdict(cf) if cf is not None else None,
                },
                indent=2,
            )
        )
        return 0

    window = f"last {args.window:g}d" if args.window else "all time"
    scope = "all projects" if getattr(args, "all_projects", False) else "this project"
    print(f"Puppetmaster savings — {window}, {scope} ({report['jobs_considered']} jobs)")
    print()
    print("MEASURED")
    print(
        f"  Routing (cost-optimizing tasks): saved ${routing.saved_usd:.4f} "
        f"of ${routing.baseline_usd:.4f} baseline ({routing.pct_cheaper:.0f}% cheaper) "
        f"across {routing.cost_optimizing_tasks} tasks"
    )
    print(
        f"    tasks routed to a $0-marginal (plan) model: {routing.plan_routed_tasks}"
    )
    if routing.deliberate_tasks:
        print(
            f"  Deliberate quality spend (by request): ${routing.deliberate_spend_usd:.4f} "
            f"over {routing.deliberate_tasks} tasks (not counted as savings)"
        )
    print(
        f"  CodeGraph: {cg['queries']} exploration queries served, "
        f"~{cg['context_tokens_fed']:,} focused-context tokens fed to agents"
    )
    print(
        f"  Reliability: {heal.fallbacks} task(s) auto-recovered off a dead/unfunded "
        f"provider, {heal.escalations} re-run for confidence (counts, not dollars)"
    )
    print(
        f"  $0 follow-up reads: {reads['reads']} result read(s) served from durable "
        f"state at zero model cost"
    )
    if tool_offload.get("offloads"):
        print(
            f"  Tool-output offload: {tool_offload['offloads']} spill(s), "
            f"~{tool_offload['tokens_saved']:,} tokens kept out of context "
            f"({tool_offload['chars_saved']:,} chars measured, chars//4)"
        )
    if memory_cost["injections"]:
        print(
            f"  Memory injection overhead (spend, not savings): "
            f"{memory_cost['injections']} injection(s), "
            f"{memory_cost['records_injected']} record(s), "
            f"~{memory_cost['token_count']:,} tokens, "
            f"${memory_cost['estimated_cost_usd']:.4f} estimated"
        )
    print()
    print(
        f"ESTIMATE (baseline: {cg['exploration_baseline_tokens']:,} tokens/query "
        f"a graph-less crawl would read, ${cg['input_price_per_mtok']:g}/Mtok input)"
    )
    print(
        f"  Avoided exploration: ~{cg['net_tokens_saved_est']:,} tokens "
        f"-> ~${cg['dollars_saved_est']:.4f} saved"
    )
    print()

    if cf is not None:
        print(
            f"COUNTERFACTUAL (vs running every routed task on {cf.reference_model_id} "
            f"at metered API rates)"
        )
        if cf.reference_priced:
            print(
                f"  Avoided spend: ${cf.avoided_usd:,.2f} "
                f"(naive ${cf.naive_cost_usd:,.2f} - actual ${cf.actual_cost_usd:,.2f}) "
                f"across {cf.tasks} routed task(s)"
            )
            print(
                "  This is a counterfactual, not cash off your bill — only as honest "
                f"as the assumption that you'd otherwise have run {cf.reference_model_id}."
            )
        else:
            print(
                f"  Reference {cf.reference_model_id} has no per-token price, so there is "
                "no metered counterfactual to compute ($0). Point "
                f"${COUNTERFACTUAL_MODEL_ENV} at a priced model to get this number."
            )
        print()

    def _pct(value):
        return "n/a" if value is None else f"{value * 100:.0f}%"

    sample = metrics["sample"]
    print("RATES (no $, for org dashboards — trend these over a --window)")
    print(
        f"  Capability-match rate: {_pct(metrics['capability_match_rate'])} "
        f"(right-sized below the strongest model; n={sample['cost_optimizing_judgeable']})"
    )
    print(
        f"  Escalation rate: {_pct(metrics['escalation_rate'])} | "
        f"Fallback rate: {_pct(metrics['fallback_rate'])} (n={sample['routed_tasks']} routed tasks)"
    )
    reuse = metrics["reuse_reads_per_job"]
    ctx = metrics["context_tokens_per_job"]
    print(
        f"  Reuse: {('n/a' if reuse is None else f'{reuse:g}')} result reads/job | "
        f"Context: {('n/a' if ctx is None else f'~{ctx:,.0f}')} focused tokens/job "
        f"(n={sample['jobs']} jobs)"
    )
    print()
    print("Notes")
    print("  - Routing $ is vs the strongest model you could have used, snapshotted at")
    print("    decision time (no recompute drift). Quality/escalating picks are spend you")
    print("    asked for, shown separately, never as a loss.")
    if routing.tasks_without_baseline:
        print(
            f"  - {routing.tasks_without_baseline} cost-optimizing tasks predate baseline "
            "tracking and are excluded from the $ figure."
        )
    print("  - 'Avoided exploration' is an estimate; tune with "
          "PUPPETMASTER_EXPLORATION_BASELINE_TOKENS / _PRICE_PER_MTOK.")
    return 0

def _run_route_command(args) -> int:
    """Run the router against a free-form instruction and print the decision.

    Use this to sanity-check capability scores and policies before
    putting them in front of a real swarm. Pairs with the
    ``puppetmaster_route_task`` MCP tool.
    """
    from puppetmaster.model_registry import default_registry_path, load_registry
    from puppetmaster.router import (
        NoEligibleModelError,
        TaskSignals,
        route_task,
    )

    path = _registry_path_from_args(args) or default_registry_path()
    try:
        specs = load_registry(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not specs:
        print(
            f"error: no models registered at {path}. "
            "Run `puppetmaster models init` first.",
            file=sys.stderr,
        )
        return 1

    from puppetmaster.platform_lock import active_allowlist

    signals = TaskSignals(
        instruction=args.instruction,
        role=args.role,
        explicit_min_capability=args.min_capability,
        explicit_max_cost_usd=args.max_cost_usd,
        required_tags=list(args.required_tag),
        allowed_adapters=active_allowlist(),
    )
    try:
        decision = route_task(signals, specs, policy=args.policy)
    except NoEligibleModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(decision.to_artifact_payload(), indent=2))
        return 0

    print(
        f"picked: {decision.model.id}  (adapter={decision.model.adapter}, "
        f"model_name={decision.model.adapter_model_name})"
    )
    print(f"policy: {decision.policy}")
    print(
        f"capability needed: {decision.capability_needed}  "
        f"chosen capability: {decision.model.capability_score}"
    )
    print(
        f"estimated tokens: in={decision.estimated_tokens_in}  "
        f"out={decision.estimated_tokens_out}  "
        f"estimated cost: ${decision.estimated_cost_usd:.6f}"
    )
    print(f"why: {decision.reason}")
    if decision.rejected:
        print("rejected:")
        for spec, why in decision.rejected:
            print(f"  - {spec.id}: {why}")
    return 0

def _run_should_delegate_command(args) -> int:
    """Dry-run the invocation gate against a prompt."""
    from puppetmaster.invocation_gate import should_delegate

    decision = should_delegate(
        args.prompt, role=args.role, threshold=args.threshold
    )
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return 0
    verdict = "DELEGATE" if decision.should_delegate else "inline"
    print(f"{verdict}  (capability {decision.capability_score}, role={decision.role})")
    print(f"verb:   {decision.suggested_verb}")
    print(f"why:    {decision.reason}")
    if decision.matched_signals:
        print(f"signals: {', '.join(decision.matched_signals)}")
    return 0

def _run_invocation_gate_command(args) -> int:
    """Host-hook entry point. Reads stdin JSON, prints host verdict, exits 0."""
    from puppetmaster.hook_runner import run as run_hook

    return run_hook(["--host", args.host, "--event", args.event])

def _run_proxy_command(args) -> int:
    """Run the local OpenAI-compatible enforcement proxy."""
    from puppetmaster.provider_proxy import serve_proxy

    try:
        serve_proxy(
            host=args.host,
            port=args.port,
            mode=args.mode,
            upstream_base_url=args.upstream_base_url,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

def _routing_estimate_rows(artifacts) -> tuple[list[dict], dict[str, dict], float]:
    """The pre-flight routing estimate: per-task rows + per-model rollup + total.

    Only the router's *initial* decision per task counts. Fallback/escalation
    reroutes (created_by 'router-fallback' / 'router-escalation') emit their own
    ROUTING artifacts; summing all of them double-counts a rerouted task. Dedup
    by task_id mirrors ``savings.collect_routing_records``.
    """
    from puppetmaster.models import ArtifactType

    rows: list[dict] = []
    by_model: dict[str, dict] = {}
    total = 0.0
    seen_router_tasks: set = set()
    for artifact in artifacts:
        if artifact.type != ArtifactType.ROUTING or artifact.created_by != "router":
            continue
        task_id = artifact.task_id
        if task_id:
            if task_id in seen_router_tasks:
                continue
            seen_router_tasks.add(task_id)
        payload = artifact.payload or {}
        model_id = payload.get("model_id", "<unknown>")
        cost = float(payload.get("estimated_cost_usd") or 0.0)
        total += cost
        rows.append(
            {
                "task_id": task_id,
                "role": payload.get("role"),
                "model_id": model_id,
                "adapter": payload.get("adapter"),
                "policy": payload.get("policy"),
                "capability_needed": payload.get("capability_needed"),
                "estimated_cost_usd": cost,
            }
        )
        bucket = by_model.setdefault(model_id, {"calls": 0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["cost"] += cost
    return rows, by_model, round(total, 6)


def _run_cost_command(args, store) -> int:
    """Report a job's cost on two clearly-labeled bases.

    * **Actual measured spend** — the honest number, computed downstream of the
      router as *(tokens actually consumed)* × *(registry price of the model
      each task actually ran on)*. Available for any run that produced token
      usage — pinned, auto-routed, or plan-billed — because every adapter stamps
      usage on its artifacts. Plan-billed models contribute $0 marginal spend,
      but their token counts and an explicit *counterfactual* (what the same
      volume would have cost on the flagship at metered rates) are still shown.
    * **Pre-flight routing estimate** — the router's per-decision
      ``estimated_cost_usd`` sum, shown only when the job auto-routed. These are
      *relative-cost estimates* from user-asserted registry prices, not measured
      consumption — useful for budgeting, never read as token volume.

    Cost no longer depends on a ROUTING artifact existing: a pinned run gets a
    priced ledger from its usage, not a "$0, didn't auto-route" dead end.
    """
    from puppetmaster.cost import job_counterfactual, price_job
    from puppetmaster.model_registry import default_registry_path, load_registry
    from puppetmaster.usage import aggregate_token_usage

    job_id = args.job_id
    artifacts = store.list_artifacts(job_id)

    try:
        registry_path = _registry_path_from_args(args) or default_registry_path()
        registry = load_registry(registry_path)
    except Exception:
        registry = []

    routing_rows, routing_by_model, routing_total = _routing_estimate_rows(artifacts)
    job_cost = price_job(artifacts, registry)
    counterfactual = job_counterfactual(job_cost, registry)

    if args.json:
        actual_by_model = {
            mid: {
                "calls": v["calls"],
                "tokens_in": v["tokens_in"],
                "tokens_out": v["tokens_out"],
                "marginal_cost_usd": v["marginal_cost_usd"],
                "billing": v["billing"],
            }
            for mid, v in job_cost.by_model.items()
        }
        payload = {
            "job_id": job_id,
            # Backward-compatible: the pre-flight routing estimate fields.
            "cost_basis": "preflight_routing_estimate",
            "total_estimated_cost_usd": routing_total,
            "by_model": {
                mid: {"calls": v["calls"], "estimated_cost_usd": round(v["cost"], 6)}
                for mid, v in routing_by_model.items()
            },
            "token_usage": aggregate_token_usage(artifacts),
            # The honest, routing-independent number.
            "actual_cost": {
                "cost_basis": "measured_usage_x_registry_price",
                "total_marginal_cost_usd": job_cost.total_marginal_cost_usd,
                "measured_cost_usd": job_cost.measured_cost_usd,
                "estimated_cost_usd": job_cost.estimated_cost_usd,
                "measured_runs": job_cost.measured_runs,
                "estimated_runs": job_cost.estimated_runs,
                "priced_tasks": job_cost.priced_tasks,
                "unpriced_tasks": job_cost.unpriced_tasks,
                "by_model": actual_by_model,
                "tasks": [dataclasses.asdict(t) for t in job_cost.tasks],
            },
            "counterfactual": (
                dataclasses.asdict(counterfactual) if counterfactual is not None else None
            ),
            # When the job auto-routed, the per-task routing rows; otherwise the
            # priced-usage rows so the task breakdown is never empty for a run
            # that actually consumed tokens.
            "tasks": routing_rows if routing_rows else [dataclasses.asdict(t) for t in job_cost.tasks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    has_usage = bool(job_cost.tasks)
    print(
        f"job {job_id}: actual measured spend = "
        f"${job_cost.total_marginal_cost_usd:.6f}"
        + (
            f"  ({job_cost.measured_cost_usd:.6f} measured / "
            f"{job_cost.estimated_cost_usd:.6f} from estimated tokens)"
            if has_usage
            else ""
        )
    )
    if has_usage:
        print()
        print(f"  {'MODEL':<28}  {'CALLS':>5}  {'TOKENS':>14}  {'COST':>12}")
        for mid, v in sorted(
            job_cost.by_model.items(), key=lambda kv: -kv[1]["marginal_cost_usd"]
        ):
            tokens = v["tokens_in"] + v["tokens_out"]
            print(
                f"  {mid[:28]:<28}  {v['calls']:>5}  {tokens:>14,}  "
                f"${v['marginal_cost_usd']:>10.6f}"
            )
        if job_cost.unpriced_tasks:
            print(
                f"\n  note: {job_cost.unpriced_tasks} task(s) ran on a model not in "
                "your registry, so their spend could not be priced (tokens still counted)."
            )
    else:
        print(
            "  no token usage recorded for this job yet — nothing to price "
            "(the job may still be running, or produced no worker artifacts)."
        )

    if counterfactual is not None and counterfactual.reference_priced:
        print()
        print(
            f"  counterfactual: this volume on {counterfactual.reference_model_id} "
            f"at metered rates ≈ ${counterfactual.naive_cost_usd:.6f}; you paid "
            f"${counterfactual.actual_cost_usd:.6f} → avoided ${counterfactual.avoided_usd:.6f}."
        )

    if routing_rows:
        print()
        print(f"  pre-flight routing estimate (relative model cost) = ${routing_total:.6f}")
        print(f"  {'TASK':<14}  {'ROLE':<14}  {'MODEL':<28}  {'EST COST':>12}")
        for row in routing_rows:
            task_id = (row["task_id"] or "")[:14]
            role = (row["role"] or "")[:14]
            model_id = (row["model_id"] or "")[:28]
            print(
                f"  {task_id:<14}  {role:<14}  {model_id:<28}  "
                f"${row['estimated_cost_usd']:>10.6f}"
            )
        print(
            "\n  note: routing figures are PRE-FLIGHT ESTIMATES (relative model "
            "cost), not measured consumption — do not read them as token volume."
        )
    print()
    _print_token_usage(artifacts)
    return 0


def _run_receipt_command(args, store) -> int:
    from puppetmaster.receipt import build_job_receipt

    receipt = build_job_receipt(store, args.job_id)
    if args.json:
        print(json.dumps(receipt, indent=2))
        return 0
    print(f"job {receipt['job_id']}: receipt")
    if receipt.get("elapsed_seconds") is not None:
        print(f"  elapsed: {receipt['elapsed_seconds']}s")
    tasks = receipt["tasks"]
    artifacts = receipt["artifacts"]
    signals = receipt["signals"]
    efficiency = receipt["efficiency"]
    tokens = receipt["tokens"]
    print(
        "  tasks: "
        f"{tasks['total']} total, {tasks['complete']} complete, "
        f"{tasks['failed']} failed, {tasks['degraded']} degraded"
    )
    print(
        "  artifacts: "
        f"{artifacts['typed_total']} typed / {artifacts['total']} total "
        f"{artifacts['by_type']}"
    )
    print(
        "  signals: "
        f"empty_or_unstructured={signals['empty_or_unstructured']}, "
        f"stdout_salvage={signals['stdout_salvage']}"
    )
    print(
        "  tokens: "
        f"{tokens['total_tokens']:,} total "
        f"({tokens['measured_tokens_in']:,} in / {tokens['measured_tokens_out']:,} out measured)"
    )
    print(
        "  efficiency: "
        f"tokens_per_typed_artifact={efficiency['tokens_per_typed_artifact']}, "
        f"degraded_rate={efficiency['degraded_rate']}"
    )
    return 0
