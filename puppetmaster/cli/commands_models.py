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

from puppetmaster.cli.guidance import (
    _CLAUDE_CODE_EFFORT_LEVELS,
    _CODEX_EFFORT_LEVELS,
    _EFFORT_TOKEN_MULTIPLIERS,
    _HERMES_EFFORT_LEVELS,
    _OPENAI_EFFORT_LEVELS,
)
from puppetmaster.cli.helpers import _registry_path_from_args


def model_payload_defaults_for_effort(adapter: str, effort: str) -> dict[str, Any]:
    """Translate a registry effort level into adapter payload defaults."""
    normalized = effort.strip().lower()
    if adapter == "openai":
        if normalized not in _OPENAI_EFFORT_LEVELS:
            raise ValueError(
                "openai effort must be one of "
                + ", ".join(_OPENAI_EFFORT_LEVELS)
            )
        return {"reasoning_effort": normalized}
    if adapter == "codex":
        if normalized not in _CODEX_EFFORT_LEVELS:
            raise ValueError("codex effort must be one of " + ", ".join(_CODEX_EFFORT_LEVELS))
        return {"extra_args": ["-c", f"model_reasoning_effort={normalized}"]}
    if adapter == "hermes":
        if normalized not in _HERMES_EFFORT_LEVELS:
            raise ValueError(
                "hermes effort must be one of " + ", ".join(_HERMES_EFFORT_LEVELS)
            )
        # The HermesAdapter reads ``payload.reasoning_effort`` and applies it via
        # an ephemeral ``HERMES_HOME`` carrying ``agent.reasoning_effort``.
        return {"reasoning_effort": normalized}
    if adapter == "agentic":
        if normalized not in _OPENAI_EFFORT_LEVELS:
            raise ValueError(
                "agentic effort must be one of " + ", ".join(_OPENAI_EFFORT_LEVELS)
            )
        return {"reasoning_effort": normalized}
    if adapter == "claude-code":
        # Claude Code >= 2.1.204 accepts `--effort <level>`; the adapter passes
        # payload extra_args straight through to the CLI invocation.
        if normalized not in _CLAUDE_CODE_EFFORT_LEVELS:
            raise ValueError(
                "claude-code effort must be one of "
                + ", ".join(_CLAUDE_CODE_EFFORT_LEVELS)
            )
        return {"extra_args": ["--effort", normalized]}
    if adapter == "cursor":
        raise ValueError(
            "cursor does not expose an effort knob through its CLI/SDK today."
        )
    raise ValueError(f"adapter {adapter!r} does not have known effort support")

def _payload_defaults_summary(payload_defaults: dict[str, Any]) -> str:
    if not payload_defaults:
        return "-"
    effort = payload_defaults.get("reasoning_effort")
    if effort:
        return f"effort={effort}"
    extra_args = payload_defaults.get("extra_args")
    if isinstance(extra_args, list):
        for i, arg in enumerate(extra_args):
            if isinstance(arg, str) and arg.startswith("model_reasoning_effort="):
                return "effort=" + arg.split("=", 1)[1]
            if arg == "--effort" and i + 1 < len(extra_args):
                return f"effort={extra_args[i + 1]}"
    return ",".join(sorted(payload_defaults))

def _parse_bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"expected boolean value, got {value!r}")

def _json_assignment_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected JSON value, got {raw_value!r}") from exc

def _replace_model_spec(spec, **updates):
    from puppetmaster.model_registry import ModelSpec

    data = dataclasses.asdict(spec)
    data.update(updates)
    return ModelSpec(**data)

def _updated_spec_for_assignment(spec, key: str, value: str):
    if key == "capability_score":
        return _replace_model_spec(spec, capability_score=int(value))
    if key == "enabled":
        return _replace_model_spec(spec, enabled=_parse_bool_value(value))
    if key == "notes":
        return _replace_model_spec(spec, notes=value)
    if key == "billing":
        return _replace_model_spec(spec, billing=value)
    if key == "output_token_multiplier":
        return _replace_model_spec(spec, output_token_multiplier=float(value))
    if key == "effort":
        level = value.strip().lower()
        effort_defaults = model_payload_defaults_for_effort(spec.adapter, level)
        # Merge so unrelated defaults (e.g. temperature) survive an effort change,
        # and swap the effort:* tag to keep CLI and wizard entries consistent.
        payload_defaults = {**(spec.payload_defaults or {}), **effort_defaults}
        tags = [tag for tag in spec.tags if not tag.startswith("effort:")]
        tags.append(f"effort:{level}")
        return _replace_model_spec(spec, payload_defaults=payload_defaults, tags=tags)
    if key.startswith("payload_defaults."):
        payload_key = key[len("payload_defaults.") :]
        if not payload_key:
            raise ValueError("payload_defaults assignment needs a key")
        payload_defaults = dict(spec.payload_defaults or {})
        payload_defaults[payload_key] = _json_assignment_value(value)
        return _replace_model_spec(spec, payload_defaults=payload_defaults)
    raise ValueError(f"unknown models set key: {key}")

def _run_models_set(args, path: Path) -> int:
    from puppetmaster.model_registry import load_registry, save_registry

    try:
        specs = load_registry(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    by_id = {spec.id: i for i, spec in enumerate(specs)}
    if args.model_id not in by_id:
        print(f"error: unknown model id: {args.model_id}", file=sys.stderr)
        return 1

    index = by_id[args.model_id]
    updated = specs[index]
    try:
        for assignment in args.assignments:
            if "=" not in assignment:
                raise ValueError(f"expected key=value assignment, got {assignment!r}")
            key, value = assignment.split("=", 1)
            updated = _updated_spec_for_assignment(updated, key, value)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    specs[index] = updated
    save_registry(specs, path)
    print(json.dumps(dataclasses.asdict(updated), indent=2))
    return 0

class ModelRegistryWizard:
    def __init__(self, path: Path, stdin: TextIO, stdout: TextIO) -> None:
        self.path = path
        self.stdin = stdin
        self.stdout = stdout
        self.specs = []
        self.dirty = False

    def run(self) -> int:
        try:
            return self._run_menu_loop()
        except EOFError:
            self._write("")
            if self.dirty:
                self._write("Input closed — discarding unsaved changes.")
                return 1
            self._write("Input closed — nothing to save.")
            return 0

    def _run_menu_loop(self) -> int:
        from puppetmaster.model_registry import load_registry, save_registry, starter_registry

        if self.path.is_file():
            self.specs = load_registry(self.path)
        else:
            self._write(f"Registry not found at {self.path}.")
            if self._confirm("Write the starter registry now?", default=True):
                self.specs = starter_registry()
                self.dirty = True
            else:
                self.specs = []
        self.show_table()
        while True:
            self._write("")
            self._write("Choose: [1] effort variant  [2] edit field  [3] add model")
            self._write("        [4] remove entry     [5] show table  [q] save & quit")
            choice = self._prompt("> ").strip().lower()
            if choice == "1":
                self.add_effort_variant()
            elif choice == "2":
                self.edit_entry()
            elif choice == "3":
                self.add_model_entry()
            elif choice == "4":
                self.remove_entry()
            elif choice == "5":
                self.show_table()
            elif choice == "q":
                if self.dirty:
                    if self._confirm("Save changes?", default=True):
                        saved = save_registry(self.specs, self.path)
                        self._write(f"Saved registry to {saved}")
                        return 0
                    if self._confirm("Quit without saving?", default=False):
                        self._write("Discarded changes.")
                        return 0
                    continue
                self._write("No changes to save.")
                return 0
            else:
                self._write("Please choose 1, 2, 3, 4, 5, or q.")

    def show_table(self) -> None:
        self._write("")
        self._write(f"{len(self.specs)} model(s)  ({self.path})")
        self._write(
            f"{'#':>2}  {'ID':<28}  {'ADAPTER':<12}  {'MODEL':<18}  "
            f"{'CAP':>3}  {'BILLING':<7}  DEFAULTS"
        )
        for index, spec in enumerate(self.specs, 1):
            disabled = "" if spec.enabled else " [disabled]"
            self._write(
                f"{index:>2}  {spec.id:<28}  {spec.adapter:<12}  "
                f"{spec.adapter_model_name:<18}  {spec.capability_score:>3}  "
                f"{spec.billing:<7}  {_payload_defaults_summary(spec.payload_defaults)}{disabled}"
            )

    def add_effort_variant(self) -> None:
        base = self._choose_spec("Base model number")
        if base is None:
            return
        if base.adapter == "cursor":
            self._write(
                "cursor does not expose an effort knob through its CLI/SDK today."
            )
            return
        if base.adapter == "openai":
            levels = _OPENAI_EFFORT_LEVELS
        elif base.adapter == "hermes":
            levels = _HERMES_EFFORT_LEVELS
        elif base.adapter == "agentic":
            levels = _OPENAI_EFFORT_LEVELS
        elif base.adapter == "claude-code":
            levels = _CLAUDE_CODE_EFFORT_LEVELS
        else:
            levels = _CODEX_EFFORT_LEVELS
        self._write("Supported efforts: " + ", ".join(levels))
        effort = self._prompt_default("Effort level", "high").strip().lower()
        try:
            effort_defaults = model_payload_defaults_for_effort(base.adapter, effort)
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        # Merge onto the base entry's defaults so adapter-critical keys survive
        # (e.g. Hermes' ``provider``, which selects the API key the variant bills
        # to). Matches the merge semantics of ``models set <id> effort=<level>``.
        payload_defaults = {**(base.payload_defaults or {}), **effort_defaults}

        suggested_id = f"{base.id}-{effort}"
        model_id = self._prompt_default("New model id", suggested_id).strip()
        if any(spec.id == model_id for spec in self.specs):
            self._write(f"error: a registry entry with id {model_id!r} already exists.")
            return
        self._write("Higher effort usually means higher capability and higher token burn.")
        capability_score = self._prompt_int(
            "Capability score", base.capability_score, minimum=0, maximum=100
        )
        multiplier_default = _EFFORT_TOKEN_MULTIPLIERS.get(effort, 1.0)
        multiplier_default *= base.output_token_multiplier
        self._write("Output token multiplier estimates extra hidden reasoning/output volume.")
        output_token_multiplier = self._prompt_float(
            "Output token multiplier", multiplier_default, minimum_exclusive=0.0
        )
        tags = list(base.tags)
        effort_tag = f"effort:{effort}"
        if effort_tag not in tags:
            tags.append(effort_tag)
        note = f"{effort.capitalize()}-effort variant of {base.id}."
        try:
            new_spec = _replace_model_spec(
                base,
                id=model_id,
                capability_score=capability_score,
                tags=tags,
                notes=note,
                payload_defaults=payload_defaults,
                output_token_multiplier=output_token_multiplier,
            )
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        self._write(
            f"Add {new_spec.id}: adapter={new_spec.adapter}, "
            f"defaults={new_spec.payload_defaults}, multiplier={new_spec.output_token_multiplier:g}"
        )
        if self._confirm("Add this entry?", default=True):
            self.specs.append(new_spec)
            self.dirty = True
            self._write(f"Added {new_spec.id}.")

    def edit_entry(self) -> None:
        spec = self._choose_spec("Entry number")
        if spec is None:
            return
        fields = ("capability_score", "tags", "notes", "enabled", "output_token_multiplier")
        self._write("Fields: " + ", ".join(fields))
        field = self._prompt("Field: ").strip()
        if field not in fields:
            self._write(f"Unknown editable field: {field}")
            return
        try:
            if field == "capability_score":
                value = self._prompt_int("Capability score", spec.capability_score, 0, 100)
                updated = _replace_model_spec(spec, capability_score=value)
            elif field == "tags":
                raw = self._prompt_default("Tags (comma-separated)", ",".join(spec.tags))
                tags = [item.strip() for item in raw.split(",") if item.strip()]
                updated = _replace_model_spec(spec, tags=tags)
            elif field == "notes":
                updated = _replace_model_spec(
                    spec, notes=self._prompt_default("Notes", spec.notes)
                )
            elif field == "enabled":
                updated = _replace_model_spec(
                    spec,
                    enabled=self._confirm("Enabled?", default=spec.enabled),
                )
            else:
                value = self._prompt_float(
                    "Output token multiplier",
                    spec.output_token_multiplier,
                    minimum_exclusive=0.0,
                )
                updated = _replace_model_spec(spec, output_token_multiplier=value)
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        self._write(f"Update {spec.id}: {field} -> {getattr(updated, field)!r}")
        if self._confirm("Apply this change?", default=True):
            self.specs[self.specs.index(spec)] = updated
            self.dirty = True

    def add_model_entry(self) -> None:
        self._write("Add a brand-new registry entry.")
        try:
            spec = self._build_new_model_spec()
        except ValueError as exc:
            self._write(f"error: {exc}")
            return
        if any(existing.id == spec.id for existing in self.specs):
            self._write(f"error: a registry entry with id {spec.id!r} already exists.")
            return
        self._write(f"Add {spec.id}: adapter={spec.adapter}, model={spec.adapter_model_name}")
        if self._confirm("Add this entry?", default=True):
            self.specs.append(spec)
            self.dirty = True

    def remove_entry(self) -> None:
        spec = self._choose_spec("Entry number to remove")
        if spec is None:
            return
        self._write(f"Remove {spec.id} ({spec.adapter}/{spec.adapter_model_name}).")
        if self._confirm("Remove this entry?", default=False):
            self.specs.remove(spec)
            self.dirty = True
            self._write(f"Removed {spec.id}.")

    def _build_new_model_spec(self):
        from puppetmaster.model_registry import ModelSpec

        model_id = self._prompt("ID: ").strip()
        adapter = self._prompt("Adapter: ").strip()
        adapter_model_name = self._prompt("Adapter model name: ").strip()
        capability_score = self._prompt_int("Capability score", 50, 0, 100)
        input_price = self._prompt_float("Input $/Mtok", 0.0, minimum_inclusive=0.0)
        output_price = self._prompt_float("Output $/Mtok", 0.0, minimum_inclusive=0.0)
        context_window = self._prompt_int("Context window tokens", 0, minimum=0)
        billing = self._prompt_default("Billing (plan/api/unknown)", "unknown")
        raw_tags = self._prompt_default("Tags (comma-separated)", adapter)
        tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
        notes = self._prompt_default("Notes", "")
        return ModelSpec(
            id=model_id,
            adapter=adapter,
            adapter_model_name=adapter_model_name,
            capability_score=capability_score,
            input_per_mtok_usd=input_price,
            output_per_mtok_usd=output_price,
            context_window=context_window,
            billing=billing,
            tags=tags,
            notes=notes,
        )

    def _choose_spec(self, prompt: str):
        if not self.specs:
            self._write("No registry entries yet.")
            return None
        while True:
            raw = self._prompt(f"{prompt}: ").strip()
            if not raw:
                return None
            try:
                index = int(raw)
            except ValueError:
                self._write("Enter a number, or blank to cancel.")
                continue
            if 1 <= index <= len(self.specs):
                return self.specs[index - 1]
            self._write("That number is not in the table.")

    def _prompt(self, prompt: str) -> str:
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            # Closed stdin must abort the wizard, not echo "" forever:
            # every prompt loop treats "" as retryable input.
            raise EOFError("input closed")
        return line.rstrip("\n")

    def _prompt_default(self, prompt: str, default) -> str:
        raw = self._prompt(f"{prompt} [{default}]: ")
        return str(default) if raw == "" else raw

    def _prompt_int(
        self, prompt: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None
    ) -> int:
        while True:
            raw = self._prompt_default(prompt, default)
            try:
                value = int(raw)
            except ValueError:
                self._write("Enter an integer.")
                continue
            if minimum is not None and value < minimum:
                self._write(f"Enter a value >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                self._write(f"Enter a value <= {maximum}.")
                continue
            return value

    def _prompt_float(
        self,
        prompt: str,
        default: float,
        minimum_inclusive: Optional[float] = None,
        minimum_exclusive: Optional[float] = None,
    ) -> float:
        while True:
            raw = self._prompt_default(prompt, default)
            try:
                value = float(raw)
            except ValueError:
                self._write("Enter a number.")
                continue
            if minimum_inclusive is not None and value < minimum_inclusive:
                self._write(f"Enter a value >= {minimum_inclusive:g}.")
                continue
            if minimum_exclusive is not None and value <= minimum_exclusive:
                self._write(f"Enter a value > {minimum_exclusive:g}.")
                continue
            return value

    def _confirm(self, prompt: str, default: bool = False) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            raw = self._prompt(prompt + suffix).strip().lower()
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            self._write("Please answer y or n.")

    def _write(self, message: str) -> None:
        print(message, file=self.stdout)

def _run_models_setup(args, path: Path) -> int:
    try:
        return ModelRegistryWizard(path, sys.stdin, sys.stdout).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

def _run_models_subcommand(args) -> int:
    """Dispatch `python -m puppetmaster models ...`.

    Three subcommands:

    * ``init`` — write a starter registry the user can edit.
    * ``list`` — show what's registered, including price + capability.
    * ``path`` — print the resolved registry path (handy in scripts).
    """
    from puppetmaster.model_registry import (
        default_registry_path,
        load_registry,
        save_registry,
        starter_registry,
    )

    path = _registry_path_from_args(args) or default_registry_path()

    if args.models_command == "path":
        print(path)
        return 0

    if args.models_command == "init":
        if path.is_file() and not args.force:
            print(
                f"error: {path} already exists; pass --force to overwrite",
                file=sys.stderr,
            )
            return 1
        save_registry(starter_registry(), path)
        print(f"wrote starter model registry to {path}")
        print("Edit capability_score / prices to match your subscriptions.")
        return 0

    if args.models_command == "list":
        try:
            specs = load_registry(path)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            from dataclasses import asdict

            print(json.dumps({"path": str(path), "models": [asdict(s) for s in specs]}, indent=2))
            return 0
        if not specs:
            print(f"No models registered (looked at {path}).")
            print("Run `puppetmaster models init` to write a starter registry.")
            return 0
        print(f"{len(specs)} model(s) registered  ({path})")
        print(
            f"  {'ID':<28}  {'ADAPTER':<12}  {'CAP':>3}  "
            f"{'IN $/Mtok':>10}  {'OUT $/Mtok':>10}  TAGS"
        )
        for spec in specs:
            tags = ",".join(spec.tags) if spec.tags else "-"
            disabled = "" if spec.enabled else "  [disabled]"
            print(
                f"  {spec.id:<28}  {spec.adapter:<12}  "
                f"{spec.capability_score:>3}  "
                f"{spec.input_per_mtok_usd:>10.3f}  "
                f"{spec.output_per_mtok_usd:>10.3f}  {tags}{disabled}"
            )
        return 0

    if args.models_command == "discover":
        return _run_models_discover(args, path)

    if args.models_command == "setup":
        return _run_models_setup(args, path)

    if args.models_command == "set":
        return _run_models_set(args, path)

    raise SystemExit(f"unknown models subcommand: {args.models_command}")

_DISCOVER_SOURCE_BY_ADAPTER = {
    "agentic": "agentic",
    "cursor": "cursor",
    "openai": "openai",
    "claude-code": "claude",
    "codex": "codex",
    "hermes": "hermes",
}

def _all_discover_sources() -> list[str]:
    """The catch-all source list (`--source all`)."""
    sources = ["cursor", "openai", "anthropic"]
    from puppetmaster.providers import available_providers

    if available_providers():
        sources.append("agentic")
    # Fold Hermes into the catch-all only when its CLI is actually present, so
    # users who don't run Hermes don't get hermes/* entries injected just
    # because they happen to have an OPENAI/ANTHROPIC/GEMINI key set. The
    # explicit `--source hermes` path always works regardless.
    from puppetmaster.diagnostics import _hermes_cli_installed

    if _hermes_cli_installed():
        sources.append("hermes")
    return sources

def _default_discover_sources() -> list[str]:
    """Resolve a default discovery source set from the platform lock.

    A cursor-first default punished non-Cursor users with a ``CURSOR_API_KEY``
    failure on the obvious command. Instead: when the lock restricts platforms,
    discover exactly those (so a Claude/Hermes user never touches Cursor); when
    unrestricted, enumerate everything reachable, like ``--source all``.
    """
    from puppetmaster import platform_lock as pl

    if not pl.is_restricted():
        return _all_discover_sources()
    enabled = pl.enabled_adapters()
    return [
        _DISCOVER_SOURCE_BY_ADAPTER[adapter]
        for adapter in pl.KNOWN_ADAPTERS
        if adapter in enabled and adapter in _DISCOVER_SOURCE_BY_ADAPTER
    ]

def _run_models_discover(args, path: Path) -> int:
    """Enumerate platform model catalogs and reconcile them into the registry.

    Cursor (plan) uses the SDK; OpenAI and Anthropic use their ``/v1/models``
    endpoints. The default source set is derived from the platform lock;
    ``--source all`` runs every reachable source."""
    import json as _json

    from puppetmaster.model_registry import (
        load_registry,
        save_registry,
        starter_registry,
        write_discovery_meta,
    )

    try:
        registry = load_registry(path)
    except RuntimeError:
        registry = starter_registry()

    is_default = args.source is None
    if args.source == "all":
        sources = _all_discover_sources()
    elif is_default:
        # No explicit source: derive from the platform lock so the obvious
        # `models discover` command never hard-fails on a Cursor key for a user
        # who never enabled Cursor. `--source cursor` stays the explicit path.
        sources = _default_discover_sources()
    else:
        sources = [args.source]
    # A multi-source run tolerates per-source failures (collect + continue); a
    # single intended source failing is fatal so the user sees the reason.
    tolerate_source_errors = len(sources) > 1
    reports: list[dict] = []
    catalogs: dict[str, list] = {}
    errors: dict[str, str] = {}

    import puppetmaster.cli as cli

    for source in sources:
        try:
            registry, report, catalog = cli._discover_one_source(source, registry)
        except cli._DiscoverSourceError as exc:
            errors[source] = str(exc)
            if not tolerate_source_errors:
                print(f"error: {exc}", file=sys.stderr)
                if is_default and source == "cursor":
                    print(
                        "hint: cursor is the only enabled platform but its catalog "
                        "could not be discovered. Set CURSOR_API_KEY (and run "
                        "`puppetmaster install-cursor-mcp`), enable another platform "
                        "with `puppetmaster platform enable <name>`, or pass an "
                        "explicit `--source` (e.g. openai / hermes / claude).",
                        file=sys.stderr,
                    )
                return 1
            continue
        reports.append(report)
        catalogs[source] = catalog

    if args.json:
        print(
            _json.dumps(
                {
                    "reports": reports,
                    "errors": errors,
                    "catalogs": catalogs,
                    "written": bool(args.write),
                    "registry_path": str(path),
                },
                indent=2,
            )
        )
    else:
        for report in reports:
            src = report.get("source") or report.get("adapter") or "cursor"
            print(f"[{src}] discovered {report['discovered_count']} model(s).")
            if report.get("added"):
                print(f"  + new: {', '.join(report['added'])}")
            if report.get("dropped_stale_cursor_models"):
                print(
                    f"  - dropped (no longer in plan): "
                    f"{', '.join(report['dropped_stale_cursor_models'])}"
                )
        for src, err in errors.items():
            print(f"[{src}] skipped: {err}")

    if args.write and reports:
        save_registry(registry, path)
        for report in reports:
            src = report.get("source") or report.get("adapter")
            write_discovery_meta(src, report["discovered_count"], path)
        if not args.json:
            print(f"Wrote merged registry to {path}")
    elif not args.json and reports:
        print("Dry run — pass --write to persist.")
    return 0 if reports or not errors else 1

class _DiscoverSourceError(RuntimeError):
    pass

def _discover_one_source(source: str, registry: list):
    """Fetch + merge one catalog source; returns (registry, report, catalog)."""
    if source == "cursor":
        from puppetmaster.cursor_discovery import (
            CursorDiscoveryError,
            fetch_cursor_catalog,
            merge_catalog_into_registry,
        )

        try:
            catalog = fetch_cursor_catalog()
        except CursorDiscoveryError as exc:
            raise _DiscoverSourceError(str(exc)) from exc
        merged, report = merge_catalog_into_registry(registry, catalog)
        report["source"] = "cursor"
        return merged, report, catalog

    if source == "hermes":
        from puppetmaster.adapters import available_hermes_providers
        from puppetmaster.static_catalog import (
            curated_catalog,
            merge_curated_into_registry,
        )

        # Hermes always bills per-token to the user's own provider key (no
        # subscription posture to detect, so billing is unconditionally "api").
        # Seed only models whose provider has a usable credential so the router
        # never picks a Hermes model it can't actually call.
        allowed = available_hermes_providers()
        merged, report = merge_curated_into_registry(
            "hermes", "api", registry, allowed_providers=allowed
        )
        report["source"] = "hermes"
        report["available_providers"] = sorted(allowed)
        catalog = [
            {"id": item["model"]}
            for item in curated_catalog("hermes")
            if (item.get("payload_defaults") or {}).get("provider") in allowed
        ]
        return merged, report, catalog

    if source == "agentic":
        from puppetmaster.providers import available_providers
        from puppetmaster.static_catalog import (
            curated_catalog,
            merge_curated_into_registry,
        )

        allowed = available_providers()
        merged, report = merge_curated_into_registry(
            "agentic", "api", registry, allowed_providers=allowed
        )
        report["source"] = "agentic"
        report["available_providers"] = sorted(allowed)
        catalog = [
            {"id": item["model"]}
            for item in curated_catalog("agentic")
            if (item.get("payload_defaults") or {}).get("provider") in allowed
        ]
        return merged, report, catalog

    if source in ("claude", "codex"):
        from puppetmaster.platform_billing import detect_adapter_billing
        from puppetmaster.static_catalog import (
            SOURCE_TO_ADAPTER,
            curated_catalog,
            merge_curated_into_registry,
        )

        adapter = SOURCE_TO_ADAPTER[source]
        status = detect_adapter_billing(adapter)
        # Use the detected posture so prices/billing are truthful; fall back to
        # API-billed reference pricing when auth can't be determined, so the
        # curated entries are still usable rather than silently $0.
        billing = (
            status.billing
            if getattr(status, "healthy", False)
            and getattr(status, "billing", "unknown") in ("plan", "api")
            else "api"
        )
        merged, report = merge_curated_into_registry(adapter, billing, registry)
        catalog = [{"id": item["model"]} for item in curated_catalog(adapter)]
        report["source"] = source
        return merged, report, catalog

    from puppetmaster.api_discovery import (
        ApiDiscoveryError,
        fetch_anthropic_models,
        fetch_openai_models,
        merge_api_catalog_into_registry,
    )

    try:
        if source == "openai":
            catalog = fetch_openai_models()
            merged, report = merge_api_catalog_into_registry(
                "openai", "api", registry, catalog
            )
        elif source == "anthropic":
            catalog = fetch_anthropic_models()
            merged, report = merge_api_catalog_into_registry(
                "claude-code", "unknown", registry, catalog
            )
        else:
            raise _DiscoverSourceError(f"unknown source: {source}")
    except ApiDiscoveryError as exc:
        raise _DiscoverSourceError(str(exc)) from exc
    report["source"] = source
    return merged, report, catalog
