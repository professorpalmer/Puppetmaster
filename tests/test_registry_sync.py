"""`puppetmaster models sync`: starter staleness check and append-only merge.

Each test writes a registry into a temporary directory and passes its path.
The platform lock file is ``platform.json`` in the same directory. No test
reads the real ``~/.puppetmaster``. unittest only.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.cli._dispatch import main
from puppetmaster.model_registry import (
    _spec_to_jsonable,
    enabled_specs,
    load_registry,
    save_registry,
    starter_registry,
)
from puppetmaster.platform_lock import ONLY_ENV
from puppetmaster.registry_sync import SYNC_AUTHORITY, SYNC_COMMAND

ABSENT = (
    "claude-code/haiku-5-5",
    "claude-code/fable-5-1",
    "openai/gpt-6-luna",
    "openai/gpt-6-sol",
)
EDITED = "claude-code/opus-5-5"
USER_DISABLED = "cursor/grok-4-5"


class _SyncCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "models.json"
        # The hermetic suite pins the platform allowlist through the
        # environment. Remove the pin so ``platform.json`` beside the temp
        # registry is the platform lock.
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ONLY_ENV, None)

    def _write_stale_registry(self) -> None:
        """An old starter registry with the owner's local changes."""
        specs = []
        for spec in starter_registry():
            if spec.id in ABSENT:
                continue
            if spec.id == EDITED:
                spec = replace(
                    spec,
                    capability_score=41,
                    input_per_mtok_usd=0.25,
                    output_per_mtok_usd=1.75,
                    notes="local ladder entry",
                )
            if spec.id == USER_DISABLED:
                spec = replace(
                    spec,
                    enabled=False,
                    disabled_reason="not in my ladder",
                    disabled_authority="user",
                )
            specs.append(spec)
        save_registry(specs, self.path)
        # A field this release does not know must stay in the file.
        document = json.loads(self.path.read_text(encoding="utf-8"))
        document["models"][0]["x_local_field"] = {"keep": True}
        self.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    def _lock_platform(self, *adapters: str) -> None:
        (self.path.parent / "platform.json").write_text(
            json.dumps({"disabled": list(adapters)}), encoding="utf-8"
        )

    def _sync(self, *flags: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["models", "sync", "--registry-path", str(self.path), *flags])
        return code, out.getvalue(), err.getvalue()

    def _rows(self) -> dict[str, dict]:
        document = json.loads(self.path.read_text(encoding="utf-8"))
        return {row["id"]: row for row in document["models"]}


class RegistrySyncTests(_SyncCase):
    def test_dry_run_lists_absent_ids_and_writes_nothing(self) -> None:
        self._write_stale_registry()
        self._lock_platform("openai", "hermes")
        before = self.path.read_bytes()

        code, out, _err = self._sync()

        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn("4 starter entry(ies) are absent", out)
        for model_id in ABSENT:
            self.assertIn(model_id, out)
        self.assertIn("claude-code (2)", out)
        self.assertIn("openai (2)  [disabled by the platform lock]", out)
        self.assertIn("Platform lock disables: hermes, openai", out)
        self.assertIn("Dry run", out)
        self.assertIn(f"{SYNC_COMMAND} --apply", out)

    def test_dry_run_json(self) -> None:
        self._write_stale_registry()
        self._lock_platform("openai")

        code, out, _err = self._sync("--json")

        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertFalse(payload["applied"])
        self.assertEqual([row["id"] for row in payload["absent"]], list(ABSENT))
        self.assertEqual(payload["locked_adapters"], ["openai"])
        self.assertEqual(
            {row["id"] for row in payload["absent"] if row["adapter_locked"]},
            {"openai/gpt-6-luna", "openai/gpt-6-sol"},
        )

    def test_apply_appends_disabled_entries_and_keeps_existing_entries(self) -> None:
        self._write_stale_registry()
        rows_before = self._rows()
        specs_before = {
            spec.id: _spec_to_jsonable(spec) for spec in load_registry(self.path)
        }
        routable_before = [spec.id for spec in enabled_specs(load_registry(self.path))]

        code, out, _err = self._sync("--apply")

        self.assertEqual(code, 0)
        self.assertIn("Added 4 starter entry(ies)", out)
        rows_after = self._rows()
        specs_after = load_registry(self.path)
        # Existing rows: same raw JSON (unknown field and key order included),
        # same order in the file, and the same parsed entry.
        self.assertEqual(
            list(rows_after)[: len(rows_before)], list(rows_before)
        )
        for model_id, row in rows_before.items():
            self.assertEqual(json.dumps(rows_after[model_id]), json.dumps(row))
        self.assertEqual(rows_after[next(iter(rows_before))]["x_local_field"], {"keep": True})
        self.assertEqual(
            {
                spec.id: _spec_to_jsonable(spec)
                for spec in specs_after
                if spec.id in specs_before
            },
            specs_before,
        )
        # New rows: appended, not routable, and they name the command.
        self.assertEqual(list(rows_after)[len(rows_before):], list(ABSENT))
        starter = {spec.id: spec for spec in starter_registry()}
        for spec in specs_after:
            if spec.id not in ABSENT:
                continue
            self.assertFalse(spec.enabled)
            self.assertIn(SYNC_COMMAND, spec.disabled_reason)
            self.assertEqual(spec.disabled_authority, SYNC_AUTHORITY)
            self.assertEqual(
                replace(spec, enabled=True, disabled_reason="", disabled_authority=""),
                starter[spec.id],
            )
        # Auto-routing sees the same candidates as before.
        self.assertEqual(
            [spec.id for spec in enabled_specs(specs_after)], routable_before
        )

    def test_second_apply_changes_nothing(self) -> None:
        self._write_stale_registry()
        self.assertEqual(self._sync("--apply")[0], 0)
        after_first = self.path.read_bytes()

        code, out, _err = self._sync("--apply")

        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), after_first)
        self.assertIn("Registry has every starter entry", out)
        code, out, _err = self._sync("--apply", "--enable")
        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), after_first)

    def test_entry_with_local_edits_stays_the_same(self) -> None:
        self._write_stale_registry()
        before = {spec.id: spec for spec in load_registry(self.path)}

        self.assertEqual(self._sync("--apply", "--enable")[0], 0)

        after = {spec.id: spec for spec in load_registry(self.path)}
        self.assertEqual(after[EDITED], before[EDITED])
        self.assertEqual(after[EDITED].capability_score, 41)
        self.assertEqual(after[EDITED].input_per_mtok_usd, 0.25)
        self.assertEqual(after[USER_DISABLED], before[USER_DISABLED])
        self.assertFalse(after[USER_DISABLED].enabled)
        self.assertEqual(after[USER_DISABLED].disabled_authority, "user")

    def test_enable_does_not_enable_an_adapter_the_platform_lock_disables(self) -> None:
        self._write_stale_registry()
        self._lock_platform("openai")

        code, out, _err = self._sync("--apply", "--enable")

        self.assertEqual(code, 0)
        after = {spec.id: spec for spec in load_registry(self.path)}
        starter = {spec.id: spec for spec in starter_registry()}
        for model_id in ("claude-code/haiku-5-5", "claude-code/fable-5-1"):
            self.assertEqual(after[model_id], starter[model_id])
            self.assertTrue(after[model_id].is_routable)
        for model_id in ("openai/gpt-6-luna", "openai/gpt-6-sol"):
            self.assertFalse(after[model_id].enabled)
            self.assertIn(SYNC_COMMAND, after[model_id].disabled_reason)
            self.assertIn("platform lock", after[model_id].disabled_reason)
            self.assertEqual(after[model_id].disabled_authority, SYNC_AUTHORITY)
        self.assertIn("openai (2)  [disabled by the platform lock]", out)

    def test_environment_allowlist_is_a_platform_lock(self) -> None:
        self._write_stale_registry()
        os.environ[ONLY_ENV] = "cursor,claude-code"

        self.assertEqual(self._sync("--apply", "--enable")[0], 0)

        after = {spec.id: spec for spec in load_registry(self.path)}
        self.assertTrue(after["claude-code/haiku-5-5"].enabled)
        self.assertFalse(after["openai/gpt-6-luna"].enabled)

    def test_same_model_below_a_different_id_is_not_added(self) -> None:
        specs = []
        for spec in starter_registry():
            if spec.id == "claude-code/haiku-5-5":
                spec = replace(spec, id="claude-code/my-haiku")
            specs.append(spec)
        save_registry(specs, self.path)
        before = self.path.read_bytes()

        code, out, _err = self._sync("--apply")

        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn(
            "skip claude-code/haiku-5-5: claude-code/my-haiku is the same adapter and model",
            out,
        )

    def test_top_level_list_registry_keeps_its_shape(self) -> None:
        self._write_stale_registry()
        rows = json.loads(self.path.read_text(encoding="utf-8"))["models"]
        self.path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

        self.assertEqual(self._sync("--apply")[0], 0)

        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsInstance(document, list)
        self.assertEqual(document[: len(rows)], rows)
        self.assertEqual([row["id"] for row in document[len(rows):]], list(ABSENT))
        self.assertEqual(len(load_registry(self.path)), len(starter_registry()))

    def test_absent_registry_file_is_an_error_and_is_not_created(self) -> None:
        for flags in ((), ("--apply",)):
            code, _out, err = self._sync(*flags)
            self.assertEqual(code, 1)
            self.assertIn("puppetmaster models init", err)
            self.assertFalse(self.path.exists())

    def test_invalid_registry_is_an_error_and_is_not_changed(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")

        code, _out, err = self._sync("--apply")

        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", err)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")


class RegistryStarterSyncDoctorTests(_SyncCase):
    """The doctor check reports absent starter ids as a warning."""

    def _check(self):
        from puppetmaster.diagnostics import _registry_starter_sync_check

        return _registry_starter_sync_check(self.path)

    def test_doctor_warns_with_the_count_and_the_fix_command(self) -> None:
        self._write_stale_registry()

        check = self._check()

        self.assertEqual(check.name, "registry-starter-sync")
        self.assertEqual(check.status, "warn")
        self.assertIn("4 starter model id(s) are absent", check.detail)
        self.assertIn("claude-code 2, openai 2", check.detail)
        self.assertIn(f"`{SYNC_COMMAND} --apply`", check.detail)
        self.assertEqual(check.evidence, list(ABSENT))

    def test_doctor_is_ok_after_apply(self) -> None:
        self._write_stale_registry()
        self.assertEqual(self._sync("--apply")[0], 0)

        self.assertEqual(self._check().status, "ok")

    def test_doctor_is_not_a_failure_without_a_registry_or_with_a_bad_one(self) -> None:
        self.assertEqual(self._check().status, "optional")
        self.path.write_text("[1]", encoding="utf-8")
        self.assertEqual(self._check().status, "warn")

    def test_run_doctor_reports_the_check_for_the_resolved_registry(self) -> None:
        from puppetmaster.diagnostics import run_doctor
        from puppetmaster.model_registry import REGISTRY_ENV

        self._write_stale_registry()
        os.environ[REGISTRY_ENV] = str(self.path)
        root = self.path.parent
        # No vendor CLI is on this PATH, so no doctor probe starts a real tool.
        os.environ["PATH"] = str(root / "empty")

        checks = {check.name: check for check in run_doctor(root, root / "state")}

        self.assertEqual(checks["registry-starter-sync"].status, "warn")
        self.assertIn("4 starter model id(s)", checks["registry-starter-sync"].detail)


class RegistrySyncPinErrorTests(_SyncCase):
    """An unresolved pin names the cause when the starter registry has the id."""

    def _pin_error(self, model: str, adapter: str = "claude-code") -> str:
        from puppetmaster.routing_authority import (
            RegistryAuthorityError,
            resolve_and_bind_explicit_pin,
        )

        with self.assertRaises(RegistryAuthorityError) as caught:
            resolve_and_bind_explicit_pin(
                {"pinned_model": model}, adapter=adapter, registry_path=self.path
            )
        return str(caught.exception)

    def test_absent_starter_pin_names_models_sync(self) -> None:
        self._write_stale_registry()

        message = self._pin_error("claude-haiku-5-5")

        self.assertIn("model pin 'claude-haiku-5-5' for adapter 'claude-code'", message)
        self.assertIn("'claude-code/haiku-5-5' is a starter registry entry", message)
        self.assertIn("this registry does not have", message)
        # The registry of this test is not the default one, so the advice names it.
        self.assertIn(
            f"`{SYNC_COMMAND} --apply --registry-path {self.path.resolve()}`", message
        )

    def test_pin_the_starter_does_not_have_keeps_the_general_text(self) -> None:
        self._write_stale_registry()

        message = self._pin_error("claude-not-a-model-9")

        self.assertIn("it may be disabled, retired, or absent", message)
        self.assertNotIn(SYNC_COMMAND, message)

    def test_synced_entry_error_gives_the_enable_command_and_the_command_works(self) -> None:
        from puppetmaster.registry_sync import enable_command
        from puppetmaster.routing_authority import resolve_and_bind_explicit_pin

        self._write_stale_registry()
        self.assertEqual(self._sync("--apply")[0], 0)

        message = self._pin_error("claude-haiku-5-5")

        command = enable_command("claude-code/haiku-5-5")
        self.assertIn("'claude-code/haiku-5-5' is disabled", message)
        self.assertIn(command, message)

        argv = command.split()[1:] + ["--registry-path", str(self.path)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv), 0)
        bound = resolve_and_bind_explicit_pin(
            {"pinned_model": "claude-haiku-5-5"},
            adapter="claude-code",
            registry_path=self.path,
        )
        self.assertEqual(bound["pinned_model"], "claude-code/haiku-5-5")
        enabled = {spec.id: spec for spec in load_registry(self.path)}[
            "claude-code/haiku-5-5"
        ]
        self.assertEqual(
            enabled,
            {spec.id: spec for spec in starter_registry()}["claude-code/haiku-5-5"],
        )


if __name__ == "__main__":
    unittest.main()
