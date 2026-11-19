"""Regression tests for faults that the release audit gate and a second review found in v1.51.3."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster import codegraph, flow
from puppetmaster.model_registry import ModelSpec, save_registry, starter_registry
from puppetmaster.pareto_recommend import filter_prior_generation
from puppetmaster.registry_sync import sync_starter_registry
from puppetmaster.routing_authority import RegistryAuthorityError, resolve_and_bind_explicit_pin


def _spec(model_id: str, wire: str, enabled: bool = True) -> ModelSpec:
    return ModelSpec(id=model_id, adapter=model_id.split("/", 1)[0], adapter_model_name=wire, capability_score=90,
                     billing="plan", input_per_mtok_usd=1.0, output_per_mtok_usd=6.0, tags=["tools"], enabled=enabled)


class DisabledSuccessorTests(unittest.TestCase):
    def test_a_disabled_successor_does_not_remove_the_enabled_entry(self) -> None:
        gpt5 = _spec("codex/gpt-5", "gpt-5")
        luna_off = _spec("codex/gpt-5.6-luna", "gpt-5.6-luna", enabled=False)
        kept, rejected = filter_prior_generation([gpt5], presence=[gpt5, luna_off])
        self.assertEqual(([s.id for s in kept], rejected), ([gpt5.id], []))

    def test_an_enabled_successor_still_removes_it(self) -> None:
        gpt5 = _spec("codex/gpt-5", "gpt-5")
        luna = _spec("codex/gpt-5.6-luna", "gpt-5.6-luna")
        kept, rejected = filter_prior_generation([gpt5], presence=[gpt5, luna])
        self.assertEqual((kept, [s.id for s, _ in rejected]), ([], [gpt5.id]))


class PinHintTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name).resolve() / "models.json"
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PUPPETMASTER_ONLY_ADAPTERS", None)

    def error(self, pin: str, adapter: str) -> str:
        with self.assertRaises(RegistryAuthorityError) as caught:
            resolve_and_bind_explicit_pin({"pinned_model": pin}, adapter=adapter, registry_path=self.path)
        return str(caught.exception)

    def test_a_registry_file_that_does_not_exist_is_the_named_cause(self) -> None:
        message = self.error("claude-code/haiku-5-5", "claude-code")
        self.assertIn("no model registry file", message)
        self.assertIn(f"puppetmaster models init --registry-path {self.path}", message)
        self.assertNotIn("models sync", message)

    def test_the_sync_advice_names_a_registry_that_is_not_the_default(self) -> None:
        save_registry([s for s in starter_registry() if s.id != "claude-code/haiku-5-5"
                       and s.adapter_model_name != "claude-haiku-5-5"], self.path)
        message = self.error("claude-code/haiku-5-5", "claude-code")
        self.assertIn(f"`puppetmaster models sync --apply --registry-path {self.path}`", message)

    def test_an_entry_on_a_locked_adapter_gets_the_platform_command_first(self) -> None:
        save_registry([s for s in starter_registry() if s.adapter != "openai"], self.path)
        (self.path.parent / "platform.json").write_text(json.dumps({"disabled": ["openai"]}), encoding="utf-8")
        sync_starter_registry(self.path, apply=True)
        message = self.error("gpt-6-luna", "openai")
        self.assertIn("`puppetmaster platform enable openai`", message)
        self.assertLess(message.index("platform enable openai"), message.index("models set"))

    def test_an_entry_on_an_allowed_adapter_keeps_the_short_advice(self) -> None:
        save_registry([s for s in starter_registry() if s.adapter != "openai"], self.path)
        sync_starter_registry(self.path, apply=True)
        message = self.error("gpt-6-luna", "openai")
        self.assertIn("To enable it, run `puppetmaster models set openai/gpt-6-luna enabled=true", message)
        self.assertNotIn("platform enable", message)


class SecondRoundTests(unittest.TestCase):
    def test_a_path_with_a_space_at_its_end_keeps_it(self) -> None:
        self.assertEqual(codegraph._porcelain_path("?? foo "), "foo ")
        self.assertEqual(codegraph._porcelain_path(" M  lead.txt"), " lead.txt")
        self.assertEqual(codegraph._porcelain_path("R  old.txt -> new.txt"), "new.txt")

    def test_the_registry_path_in_the_advice_is_quoted_for_a_shell(self) -> None:
        from puppetmaster.registry_sync import _registry_arg

        self.assertEqual(_registry_arg(Path("/tmp/my registries/models.json")).split(" --registry-path ")[1][0], "'")
        plain = _registry_arg(Path("/tmp/plain/models.json"))
        self.assertTrue(plain.endswith("/tmp/plain/models.json"), plain)

    def test_effort_text_in_other_forms_is_replaced(self) -> None:
        from puppetmaster.cli.commands_models import _strip_effort_extra_args

        self.assertEqual(_strip_effort_extra_args(["--verbose", "--effort"]), ["--verbose"])
        self.assertEqual(_strip_effort_extra_args(["model_reasoning_effort=low", "-c", "n=2"]), ["-c", "n=2"])

    def test_a_disabled_successor_among_the_candidates_is_not_a_successor(self) -> None:
        gpt5 = _spec("codex/gpt-5", "gpt-5")
        luna_off = _spec("codex/gpt-5.6-luna", "gpt-5.6-luna", enabled=False)
        kept, _ = filter_prior_generation([gpt5, luna_off])
        self.assertIn(gpt5.id, [s.id for s in kept])

    def test_a_pinned_model_of_only_spaces_does_not_hide_the_model(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "models.json"
            save_registry(starter_registry(), path)
            bound = resolve_and_bind_explicit_pin({"pinned_model": "   ", "model": "claude-code/opus-5-5"},
                                                  adapter="claude-code", registry_path=path)
            self.assertTrue(bound.get("registry_digest"))


class ProvisionWithoutNpxTests(unittest.TestCase):
    def setUp(self) -> None:
        codegraph.reset_codegraph_provisioning_state()
        self.addCleanup(codegraph.reset_codegraph_provisioning_state)

    def run_with(self, install_lands: bool) -> tuple:
        calls = []
        landed = {"yes": False}

        def which(name, *args, **kwargs):
            if name == "npm":
                return "/usr/bin/npm"
            if name == codegraph.CODEGRAPH_COMMAND and landed["yes"]:
                return "/usr/bin/codegraph"
            return None

        def run_cli(command, **kwargs):
            calls.append(command)
            landed["yes"] = install_lands
            return subprocess.CompletedProcess(command, 0 if install_lands else 1, "", "")

        env = {"PUPPETMASTER_CODEGRAPH_NO_NPX": "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(codegraph.shutil, "which", side_effect=which), \
                mock.patch.object(codegraph, "_cursor_codegraph_invocation", return_value=None), \
                mock.patch.object(codegraph, "_run_cli", side_effect=run_cli):
            os.environ.pop("PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL", None)
            os.environ.pop("PUPPETMASTER_CODEGRAPH_NODE", None)
            first = codegraph.ensure_codegraph_provisioned()
            second = codegraph.ensure_codegraph_provisioned()
        return first, second, calls

    def test_the_global_install_runs_when_only_npx_is_off(self) -> None:
        first, second, calls = self.run_with(install_lands=True)
        self.assertEqual((first, second), (True, True))
        self.assertEqual(calls, [["/usr/bin/npm", "install", "-g", codegraph.CODEGRAPH_PACKAGE]])

    def test_a_second_caller_with_no_npx_waits_for_the_install(self) -> None:
        import threading

        started, release, results = threading.Event(), threading.Event(), []
        landed = {"yes": False}

        def which(name, *args, **kwargs):
            if name == "npm":
                return "/usr/bin/npm"
            return "/usr/bin/codegraph" if name == codegraph.CODEGRAPH_COMMAND and landed["yes"] else None

        def run_cli(command, **kwargs):
            started.set()
            release.wait(10)
            landed["yes"] = True
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.dict(os.environ, {"PUPPETMASTER_CODEGRAPH_NO_NPX": "1"}), \
                mock.patch.object(codegraph.shutil, "which", side_effect=which), \
                mock.patch.object(codegraph, "_cursor_codegraph_invocation", return_value=None), \
                mock.patch.object(codegraph, "_run_cli", side_effect=run_cli):
            os.environ.pop("PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL", None)
            os.environ.pop("PUPPETMASTER_CODEGRAPH_NODE", None)
            first = threading.Thread(target=lambda: results.append(("first", codegraph.ensure_codegraph_provisioned())))
            first.start()
            self.assertTrue(started.wait(10))
            second = threading.Thread(target=lambda: results.append(("second", codegraph.ensure_codegraph_provisioned())))
            second.start()
            time.sleep(0.2)
            self.assertEqual(results, [], "the second caller did not wait for the install")
            release.set()
            first.join(10)
            second.join(10)
        self.assertEqual(sorted(results), [("first", True), ("second", True)])

    def test_no_install_and_no_npx_is_false_each_time(self) -> None:
        first, second, calls = self.run_with(install_lands=False)
        self.assertEqual((first, second), (False, False))
        self.assertEqual(len(calls), 1)


@unittest.skipIf(os.name == "nt", "process groups and sessions of POSIX")
class KillTreeTests(unittest.TestCase):
    def test_a_child_in_its_own_session_ends_with_the_shell_node(self) -> None:
        with TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            code = (
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
                f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(120)\n"
            )
            node = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            try:
                deadline = time.time() + 20
                while not pidfile.exists() or not pidfile.read_text().strip():
                    self.assertLess(time.time(), deadline, "the child did not start")
                    time.sleep(0.05)
                child = int(pidfile.read_text())
                self.assertIn(child, flow._descendants(node.pid))
                self.assertTrue(flow._kill_tree(node.pid))
                node.wait(timeout=10)
                deadline = time.time() + 10
                alive = True
                while alive and time.time() < deadline:
                    try:
                        os.kill(child, 0)
                        time.sleep(0.05)
                    except ProcessLookupError:
                        alive = False
                self.assertFalse(alive, "the child in its own session is still alive")
            finally:
                for pid in (node.pid,):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass


if __name__ == "__main__":
    unittest.main()
