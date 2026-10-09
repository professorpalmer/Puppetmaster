"""Parity of the Codex native_steer (app-server) path with the codex exec path."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import CodexAdapter
from puppetmaster.adapters.codex import codex_diagnostic_text, parse_codex_events
from puppetmaster.adapters.codex_session import CodexSessionResult, run_codex_session
from puppetmaster.cancellation import JobCancelled
from puppetmaster.models import Task

CLEAN = {"sha": "a" * 40, "tree": "b" * 40, "changed_files": [], "untracked_files": [], "diff": ""}

# A fake app-server. On turn/start it can spawn a grandchild that records its
# pid and then sleeps, flood stderr after a byte that is not valid UTF-8, or
# finish the turn with token usage.
FAKE = r'''
import json, os, subprocess, sys, time
mode = os.environ.get("FAKE_MODE", "normal")
def send(x): print(json.dumps(x), flush=True)
for raw in sys.stdin:
    x = json.loads(raw); m = x.get("method"); i = x.get("id")
    if m == "initialize": send({"jsonrpc": "2.0", "id": i, "result": {}})
    elif m == "thread/start": send({"jsonrpc": "2.0", "id": i, "result": {"thread": {"id": "th-1"}}})
    elif m == "turn/start":
        send({"jsonrpc": "2.0", "id": i, "result": {"turn": {"id": "one"}}})
        if mode == "grandchild":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(os.environ["FAKE_PID_FILE"], "w") as fh: fh.write(str(child.pid))
            time.sleep(60)
        if mode == "badbytes":
            sys.stderr.buffer.write(b"\xff\xfe bad\n"); sys.stderr.flush()
            sys.stderr.write("x" * 300000 + "\n"); sys.stderr.flush()
        if mode == "flood":
            for _ in range(200):
                send({"jsonrpc": "2.0", "method": "item/delta", "params": {"text": "y" * 1000}})
            time.sleep(60)
        send({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": {"inputTokens": 7, "outputTokens": 11, "cachedInputTokens": 2}}}})
        send({"jsonrpc": "2.0", "method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}})
        send({"jsonrpc": "2.0", "method": "turn/completed", "params": {}})
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SessionProcessTests(unittest.TestCase):
    """F1: the app-server runs as an owned process with UTF-8 replace decoding."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "fake.py"
        self.script.write_text(textwrap.dedent(FAKE))

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, mode, **kwargs):
        env = dict(kwargs.pop("env", {}), FAKE_MODE=mode)
        return run_codex_session([sys.executable, str(self.script)], self.tmp.name, "hi", env=env, **kwargs)

    @unittest.skipUnless(os.name == "posix", "POSIX process group cleanup")
    def test_timeout_kills_the_app_server_grandchild(self):
        pid_file = Path(self.tmp.name) / "pid"
        result = self.invoke("grandchild", env={"FAKE_PID_FILE": str(pid_file)}, timeout=3)
        self.assertEqual(result.status, "timeout")
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(_alive(pid), "grandchild survived the app-server teardown")

    def test_a_byte_that_is_not_utf8_does_not_stop_the_stderr_drain(self):
        result = self.invoke("badbytes", timeout=15)
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.messages, ["done"])

    def test_output_budget_stops_the_session(self):
        result = self.invoke("flood", timeout=15, max_output_bytes=20000)
        self.assertEqual(result.status, "output_limit")

    def test_on_spawn_receives_the_launched_argv_and_pid(self):
        seen = []
        result = self.invoke("normal", timeout=15, on_spawn=lambda command, env, pid: seen.append((command, env, pid)))
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(seen), 1)
        command, env, pid = seen[0]
        self.assertEqual(command[-3:], ["app-server", "--listen", "stdio://"])
        self.assertTrue(env.get("PUPPETMASTER_PROCESS_OWNER"))
        self.assertIsInstance(pid, int)


class NativeSteerAdapterTests(unittest.TestCase):
    """F2, F3, F12, F18, F34, F57: the adapter side of the app-server path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "state"

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, env=None, **payload):
        task = Task(job_id="job-p", role="audit", instruction="Look.", adapter="codex",
                    payload={"cwd": self.tmp.name, "native_steer": True, "disable_codegraph": True, **payload})
        with patch.dict("os.environ", env or {}), \
                patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)), \
                patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p):
            prepared = CodexAdapter()._prepare_cli_invocation(task, "goal", "w", Path(self.tmp.name), "/usr/bin/codex")
        return task, prepared

    def invoke(self, task, prepared, result, seen=None):
        seen = {} if seen is None else seen

        def session(**kwargs):
            seen.update(kwargs)
            if kwargs.get("on_spawn"):
                kwargs["on_spawn"](["codex", "app-server", "--listen", "stdio://"], {"A": "1"}, 4242)
            return result

        with patch.dict("os.environ", {"PUPPETMASTER_STATE_DIR": str(self.state)}), \
                patch("puppetmaster.adapters.codex_session.run_codex_session", side_effect=session):
            return CodexAdapter()._invoke_cli(task, prepared, Path(self.tmp.name), 30), seen

    def test_f2_a_cancelled_session_raises_job_cancelled(self):
        task, prepared = self.prepare()
        with self.assertRaises(JobCancelled):
            self.invoke(task, prepared, CodexSessionResult(status="cancelled", error="cancelled"))

    def test_f2_the_session_gets_a_cancellation_check(self):
        task, prepared = self.prepare()
        _, seen = self.invoke(task, prepared, CodexSessionResult(status="completed", messages=["ok"]))
        self.assertTrue(callable(seen.get("cancellation_check")))
        self.assertIs(seen["cancellation_check"](), False)

    def test_f2_the_dispatch_check_runs_before_launch(self):
        task, prepared = self.prepare()
        seen = {}
        with patch("puppetmaster.invocation.check_external_dispatch", side_effect=JobCancelled("job-p")):
            with self.assertRaises(JobCancelled):
                self.invoke(task, prepared, CodexSessionResult(status="completed"), seen)
        self.assertEqual(seen, {})

    def test_f3_the_whole_command_base_reaches_the_app_server(self):
        task, prepared = self.prepare(env={"CODEX_COMMAND": "npx codex"})
        _, seen = self.invoke(task, prepared, CodexSessionResult(status="completed", messages=["ok"]))
        self.assertEqual(list(seen["command_prefix"]), ["/usr/bin/codex", "codex"])
        self.assertIsNotNone(seen.get("env"))

    def test_f12_the_decided_effort_reaches_turn_start(self):
        task, prepared = self.prepare(reasoning_effort="low",
                                      extra_args=["-c", "model_reasoning_effort=low"])
        _, seen = self.invoke(task, prepared, CodexSessionResult(status="completed", messages=["ok"]))
        self.assertEqual(seen.get("effort"), "low")

    def test_f18_thread_id_receipt_and_logs_are_recorded(self):
        task, prepared = self.prepare()
        completed, seen = self.invoke(task, prepared, CodexSessionResult(
            status="timeout", thread_id="th-1", error="app-server timeout"))
        self.assertTrue(completed.timed_out)
        from puppetmaster.adapters.codex import observed_thread_id
        self.assertEqual(observed_thread_id(parse_codex_events(completed.stdout)), "th-1")
        self.assertTrue(seen.get("protocol_log_path"))
        self.assertEqual(completed.live_log_path, str(seen["protocol_log_path"]))
        self.assertTrue(completed.dispatch_receipt)
        receipt = json.loads(Path(completed.dispatch_receipt).read_text())
        self.assertEqual(receipt["pid"], 4242)
        self.assertEqual(receipt["argv"][-3:], ["app-server", "--listen", "stdio://"])

    def test_f18_output_budget_is_passed_and_reported(self):
        task, prepared = self.prepare(max_output_bytes=5000)
        completed, seen = self.invoke(task, prepared, CodexSessionResult(status="output_limit", error="output limit"))
        self.assertEqual(seen.get("max_output_bytes"), 5000)
        self.assertIs(completed.output_limit_hit, True)

    def test_f34_app_server_usage_becomes_attempt_usage(self):
        task, prepared = self.prepare()
        completed, _ = self.invoke(task, prepared, CodexSessionResult(
            status="completed", thread_id="th-1", messages=["ok"],
            input_tokens=7, output_tokens=11, cached_input_tokens=2))
        usage = completed.attempt_usage
        self.assertEqual(usage["usage_scope"], "attempt")
        self.assertEqual(usage["usage_provenance"], "app_server_token_usage")
        self.assertEqual((usage["usage"]["input_tokens"], usage["usage"]["output_tokens"],
                          usage["usage"]["cached_input_tokens"]), (7, 11, 2))

    def test_f57_agent_text_is_not_failure_evidence(self):
        task, prepared = self.prepare()
        completed, _ = self.invoke(task, prepared, CodexSessionResult(
            status="failed", thread_id="th-1", error="turn failed: bad",
            messages=["I fixed the 401 handler and the not logged in path"]))
        events = parse_codex_events(completed.stdout)
        diagnostic = codex_diagnostic_text(completed.stdout, events)
        self.assertNotIn("401", diagnostic)
        self.assertNotIn("logged in", diagnostic)
        from puppetmaster.adapters.codex import last_codex_agent_message
        self.assertIn("401 handler", last_codex_agent_message(events))

    def test_f57_finalize_classifies_the_error_not_the_transcript(self):
        task, prepared = self.prepare()
        completed, _ = self.invoke(task, prepared, CodexSessionResult(
            status="failed", thread_id="th-1", error="turn failed: bad",
            messages=["Not logged in. 401 Unauthorized was the bug I fixed."]))
        artifacts = CodexAdapter()._finalize_cli_run(task, "w", "goal", prepared, CLEAN, CLEAN, completed)
        payload = artifacts[0].payload
        self.assertEqual(payload["thread_id"], "th-1")
        self.assertNotEqual(payload["failure"], "not_authenticated")


class WriteCapableReceiptTests(unittest.TestCase):
    """Shared contract: a write-capable receipt stamps write_capable: true."""

    def receipt(self, sandbox, completed):
        task = Task(job_id="job-w", role="implement", instruction="Build.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "sandbox": sandbox, "disable_codegraph": True})
        with patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)), \
                patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p):
            prepared = CodexAdapter()._prepare_cli_invocation(task, "goal", "w", Path("."), "/usr/bin/codex")
        return CodexAdapter()._finalize_cli_run(task, "w", "goal", prepared, CLEAN, CLEAN, completed)[0].payload

    def test_write_capable_runs_stamp_true_and_read_only_runs_false(self):
        from puppetmaster.adapters._streaming import StreamedProcess
        done = StreamedProcess(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")
        timeout = StreamedProcess(returncode=None, stdout="", stderr="", timed_out=True)
        self.assertIs(self.receipt("workspace-write", done)["write_capable"], True)
        self.assertIs(self.receipt("workspace-write", timeout)["write_capable"], True)
        self.assertIs(self.receipt("read-only", done)["write_capable"], False)


if __name__ == "__main__":
    unittest.main()
