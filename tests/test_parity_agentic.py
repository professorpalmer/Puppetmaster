"""Agentic adapter parity with the reference adapters (build/parity-briefs/agentic.md)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from puppetmaster import providers
from puppetmaster.models import ArtifactType, Task


def _git_repo(test) -> Path:
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    cwd = Path(tmp.name)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init"], ["add", "-A"], ["commit", "-m", "init", "--allow-empty"]):
        subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, check=False)
    (cwd / "seed.py").write_text("seed = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(cwd), env=env, capture_output=True, check=False)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=str(cwd), env=env, capture_output=True, check=False)
    return cwd


def _pid_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _adapter():
    from puppetmaster.adapters.agentic import AgenticAdapter
    return AgenticAdapter()


# F4: run_terminal and verification own their process tree.
@unittest.skipUnless(os.name == "posix", "POSIX process-group semantics")
class OwnedShellTests(unittest.TestCase):
    _GRANDCHILD = "sleep 30 & echo $! > gc.pid; wait"

    def _grandchild_pid(self, cwd: Path) -> int:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            text = (cwd / "gc.pid").read_text().strip() if (cwd / "gc.pid").exists() else ""
            if text:
                return int(text)
            time.sleep(0.05)
        self.fail("grandchild never wrote its pid")

    def test_terminal_timeout_kills_the_grandchild(self) -> None:
        from puppetmaster.adapters import agentic

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            with mock.patch.object(agentic, "_TERMINAL_TIMEOUT_SECONDS", 1):
                out = _adapter()._tool_run_terminal({"command": self._GRANDCHILD}, cwd)
            self.assertIn("timed out", out)
            self.assertTrue(_pid_gone(self._grandchild_pid(cwd)), "orphan sleep survived the timeout")

    def test_verification_timeout_kills_the_grandchild(self) -> None:
        from puppetmaster.adapters import agentic

        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            with mock.patch.object(agentic, "_VERIFY_TIMEOUT_SECONDS", 1):
                passed, out = _adapter()._run_verification(cwd, self._GRANDCHILD)
            self.assertFalse(passed)
            self.assertIn("timed out", out)
            self.assertTrue(_pid_gone(self._grandchild_pid(cwd)), "orphan sleep survived the timeout")

    def test_cancel_stops_a_running_terminal_command(self) -> None:
        from puppetmaster.adapters import agentic
        from puppetmaster.cancellation import JobCancelled

        calls = []

        def cancel_after_first(*_args, **_kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise JobCancelled("j")

        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            with mock.patch.object(agentic, "_TERMINAL_TIMEOUT_SECONDS", 15), \
                    mock.patch.object(agentic, "check_cancellation", side_effect=cancel_after_first):
                out = _adapter()._tool_run_terminal({"command": self._GRANDCHILD}, Path(tmp))
            self.assertLess(time.monotonic() - started, 8)
            self.assertIn("cancelled", out)
            self.assertTrue(_pid_gone(self._grandchild_pid(Path(tmp))))

    def test_terminal_still_reports_exit_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = _adapter()._tool_run_terminal({"command": "echo out; echo err >&2; exit 3"}, Path(tmp))
        self.assertTrue(out.startswith("exit=3\n"))
        self.assertIn("out", out)
        self.assertIn("[stderr]\nerr", out)


# F5: verification decodes UTF-8 with replace and never prompts for git credentials.
class VerificationEncodingTests(unittest.TestCase):
    def test_verification_output_with_undecodable_bytes_still_passes(self) -> None:
        command = f'"{sys.executable}" -c "import sys; sys.stdout.buffer.write(bytes([0xff, 0x41]))"'
        with tempfile.TemporaryDirectory() as tmp:
            passed, out = _adapter()._run_verification(Path(tmp), command)
        self.assertTrue(passed, out)
        self.assertIn("A", out)

    def test_verification_env_disables_git_prompts_and_pagers(self) -> None:
        check = ("import os, sys; e = os.environ; "
                 "sys.exit(0 if e.get('GIT_TERMINAL_PROMPT') == '0' and e.get('GIT_PAGER') == 'cat' else 1)")
        command = f'"{sys.executable}" -c "{check}"'
        env = {k: v for k, v in os.environ.items() if k not in ("GIT_TERMINAL_PROMPT", "GIT_PAGER")}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env, clear=True):
            passed, out = _adapter()._run_verification(Path(tmp), command)
        self.assertTrue(passed, out)


# F7: browser Chrome is launched owned, torn down by its Job Object, and rmtree failure is logged.
class BrowserChromeTeardownTests(unittest.TestCase):
    def test_owned_launch_uses_popen_owned_in_its_own_session(self) -> None:
        from puppetmaster import browser_cdp as b

        captured = {}

        def fake_owned(args, **kwargs):
            captured.update(kwargs)
            return mock.Mock(pid=4242)

        def fake_attach(self, ws_url, **kwargs):
            self.ws = object()
            return None

        env = {k: v for k, v in os.environ.items() if not k.startswith("PM_BROWSER_")}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(b, "_find_chrome", return_value="/bin/chrome"), \
                mock.patch.object(b, "_profile_dir_for_launch", return_value=(tmp, True)), \
                mock.patch.object(b, "_wait_for_page_ws", return_value="ws://x"), \
                mock.patch.object(b._Session, "_attach_ws", fake_attach), \
                mock.patch("puppetmaster.win_process.popen_owned", side_effect=fake_owned) as owned, \
                mock.patch.object(b.subprocess, "Popen") as plain:
            session = b._Session()
            self.assertIsNone(session.ensure())
        owned.assert_called_once()
        plain.assert_not_called()
        if os.name != "nt":
            self.assertTrue(captured.get("start_new_session"))

    def test_stop_terminates_the_job_object_when_present(self) -> None:
        from puppetmaster import browser_cdp as b

        job = mock.Mock()
        proc = mock.Mock(pid=None)
        proc._puppetmaster_job = job
        state = {"alive": True}
        job.terminate.side_effect = lambda: state.update(alive=False)
        proc.poll.side_effect = lambda: None if state["alive"] else 0
        b._stop_chrome_proc(proc, timeout=1.0)
        job.terminate.assert_called()

    def test_profile_rmtree_failure_is_logged(self) -> None:
        from puppetmaster import browser_cdp as b

        td = tempfile.mkdtemp(prefix="pm-cdp-")
        self.addCleanup(lambda: __import__("shutil").rmtree(td, ignore_errors=True))
        s = b._Session()
        s.profile_dir = td
        s.owns_profile = True
        s.owns_proc = False
        with mock.patch.object(b.shutil, "rmtree", side_effect=OSError("busy")), \
                self.assertLogs("puppetmaster.browser_cdp", level="WARNING") as logs:
            s.shutdown()
        self.assertTrue(any("busy" in line for line in logs.output))


# F29: Anthropic prompt_tokens includes cache reads and writes, as Bedrock Converse does.
class AnthropicCacheFoldTests(unittest.TestCase):
    _USAGE = {"input_tokens": 12, "output_tokens": 300,
              "cache_read_input_tokens": 90000, "cache_creation_input_tokens": 4000}

    def test_sync_prompt_tokens_include_cache(self) -> None:
        canned = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                  "usage": dict(self._USAGE)}
        with mock.patch.object(providers, "_post_json", return_value=canned):
            turn = providers.provider_chat(
                provider="anthropic", model="claude", api_key="k",
                messages=[{"role": "user", "content": "go"}], tools=None,
            )
        self.assertEqual(turn.usage["prompt_tokens"], 94012)
        self.assertEqual(turn.usage["total_tokens"], 94312)
        self.assertEqual(turn.usage["cached_tokens"], 90000)
        self.assertEqual(turn.usage["cache_write_tokens"], 4000)

    def test_stream_prompt_tokens_include_cache(self) -> None:
        lines = [
            (b'data: {"type":"message_start","message":{"usage":{"input_tokens":12,'
             b'"cache_read_input_tokens":90000,"cache_creation_input_tokens":4000}}}\n'),
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":300}}\n',
            b'data: {"type":"message_stop"}\n',
        ]

        class FakeResp:
            def __iter__(self):
                return iter(lines)

            def close(self):
                pass

        with mock.patch.object(providers, "_open_stream", return_value=FakeResp()):
            turn = providers.provider_chat_streaming(
                provider="anthropic", model="claude", api_key="k",
                messages=[{"role": "user", "content": "go"}], tools=None,
            )
        self.assertEqual(turn.usage["prompt_tokens"], 94012)
        self.assertEqual(turn.usage["total_tokens"], 94312)
        self.assertEqual(turn.usage["cached_tokens"], 90000)


def _implement(test, turns, **payload):
    from puppetmaster.adapters import agentic

    cwd = _git_repo(test)
    seen = []

    def fake_chat(*, provider, model, messages, tools, extra, timeout):
        seen.append(1)
        return turns[min(len(seen) - 1, len(turns) - 1)]

    task = Task(
        job_id="j", role="build", instruction="make change",
        payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                 "mode": "implement", "disable_codegraph": True, "verify": "off", **payload},
    )
    with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
        arts = _adapter().run(task, task.instruction, "w1")
    verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION
                 and (a.payload or {}).get("kind") != "worker_verdict")
    return cwd, arts, verif


def _turn(calls, usage=None, accounting_usage=None):
    from puppetmaster.providers import AssistantTurn
    return AssistantTurn(
        text="", tool_calls=calls,
        usage=usage if usage is not None else {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        accounting_usage=accounting_usage,
    )


_WRITE = {"id": "c1", "name": "write_file", "arguments": {"path": "new.py", "content": "x = 1\n"}}
_REPORT = {"id": "r1", "name": "submit_report", "arguments": {"summary": "done"}}


# F30: a turn with no provider usage leaves tokens_in / tokens_out NULL, not a measured 0.
class MissingUsageTests(unittest.TestCase):
    def test_missing_provider_usage_is_null_not_zero(self) -> None:
        empty = providers._openai_usage_fields({})
        _cwd, _arts, verif = _implement(self, [
            _turn([_WRITE], usage=empty, accounting_usage={}),
            _turn([_REPORT], usage=empty, accounting_usage={}),
        ])
        self.assertIsNone(verif.payload["tokens_in"])
        self.assertIsNone(verif.payload["tokens_out"])

    def test_measured_usage_stays_an_int(self) -> None:
        _cwd, _arts, verif = _implement(self, [_turn([_WRITE]), _turn([_REPORT])])
        self.assertEqual(verif.payload["tokens_in"], 6)
        self.assertEqual(verif.payload["tokens_out"], 2)


# F37: the implement receipt carries the write signal and the diff-source fields.
class WriteReceiptTests(unittest.TestCase):
    def test_receipt_with_a_diff_is_write_capable(self) -> None:
        _cwd, _arts, verif = _implement(self, [_turn([_WRITE]), _turn([_REPORT])])
        self.assertIs(verif.payload["write_capable"], True)
        self.assertIs(verif.payload["worker_diff_present"], True)
        self.assertIn("baseline_diff_present", verif.payload)

    def test_receipt_without_a_diff_reports_no_worker_change(self) -> None:
        _cwd, _arts, verif = _implement(self, [_turn([_REPORT])])
        self.assertIs(verif.payload["write_capable"], True)
        self.assertIs(verif.payload["worker_diff_present"], False)


# F48: the implement receipt names the paths the run itself wrote.
class AttributionTests(unittest.TestCase):
    def test_written_and_shell_paths_are_referenced(self) -> None:
        shell = {"id": "t1", "name": "run_terminal", "arguments": {"command": "touch made.txt"}}
        edit = {"id": "e1", "name": "edit_file",
                "arguments": {"path": "seed.py", "old_string": "seed = 1", "new_string": "seed = 2"}}
        cwd, arts, verif = _implement(self, [_turn([_WRITE]), _turn([edit]), _turn([shell]), _turn([_REPORT])])
        self.assertTrue((cwd / "made.txt").exists())
        referenced = verif.payload["worker_referenced_paths"]
        self.assertIn("new.py", referenced)
        self.assertIn("seed.py", referenced)
        self.assertIn("made.txt", referenced)
        self.assertEqual(verif.payload["baseline_dirty_paths"], [])

    def test_absolute_path_is_repo_relative(self) -> None:
        from puppetmaster.adapters import agentic

        cwd = _git_repo(self)
        absolute = {"id": "c1", "name": "write_file",
                    "arguments": {"path": str(cwd / "abs.py"), "content": "y = 1\n"}}
        seen = []

        def fake_chat(*, provider, model, messages, tools, extra, timeout):
            seen.append(1)
            return [_turn([absolute]), _turn([_REPORT])][min(len(seen) - 1, 1)]

        task = Task(job_id="j", role="build", instruction="i",
                    payload={"cwd": str(cwd), "provider": "anthropic", "model": "m",
                             "mode": "implement", "disable_codegraph": True, "verify": "off"})
        with mock.patch.object(agentic, "provider_chat", side_effect=fake_chat):
            arts = _adapter().run(task, task.instruction, "w1")
        verif = next(a for a in arts if a.type == ArtifactType.VERIFICATION
                     and (a.payload or {}).get("kind") != "worker_verdict")
        self.assertIn("abs.py", verif.payload["worker_referenced_paths"])


if __name__ == "__main__":
    unittest.main()
