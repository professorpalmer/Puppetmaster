"""Opt-in worker session resume for the codex and claude-code adapters."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import ClaudeCodeAdapter, CodexAdapter
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.claude_code import (
    build_claude_code_command,
    claude_session_id_from_stdout,
)
from puppetmaster.adapters.codex import (
    build_codex_exec_command,
    build_codex_resume_command,
    last_codex_agent_message,
    parse_codex_events,
)
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.session_lease import acquire_codex_thread
from puppetmaster.worker_resume import (
    claim_resumed_session,
    resolve_worker_resume,
    task_resume_record,
)

_SESSION_HOMES = tempfile.TemporaryDirectory()
_SESSION_ENV = patch.dict(
    "os.environ",
    {
        "CODEX_HOME": str(Path(_SESSION_HOMES.name) / "codex"),
        "CLAUDE_CONFIG_DIR": str(Path(_SESSION_HOMES.name) / "claude"),
        # The lean worker home lives under PUPPETMASTER_HOME; keep the real one out.
        "PUPPETMASTER_HOME": str(Path(_SESSION_HOMES.name) / "pm"),
    },
)


def setUpModule() -> None:
    _SESSION_ENV.start()


def tearDownModule() -> None:
    _SESSION_ENV.stop()
    _SESSION_HOMES.cleanup()

CLEAN = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
THREAD_ID = "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
PRIOR_SESSION = "11111111-2222-3333-4444-555555555555"
NEW_SESSION = "66666666-7777-8888-9999-000000000000"


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict = {}
        self.tasks: dict = {}
        self.artifacts: dict = {}

    def add(self, job_id: str, task: Task, receipt: dict = None, *, at: str = "2026-10-04T00:00:00") -> None:
        self.jobs[job_id] = object()
        self.tasks.setdefault(job_id, []).append(task)
        if receipt is not None:
            self.artifacts.setdefault(job_id, []).append(
                Artifact(
                    job_id=job_id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker",
                    confidence=0.9,
                    evidence=[],
                    payload=receipt,
                    created_at=at,
                )
            )

    def get_job(self, job_id: str):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def list_tasks(self, job_id: str):
        return list(self.tasks.get(job_id, []))

    def list_artifacts(self, job_id: str):
        return list(self.artifacts.get(job_id, []))


def _task(task_id: str, role: str, adapter: str, job_id: str = "job-prior") -> Task:
    return Task(id=task_id, job_id=job_id, role=role, instruction="x", adapter=adapter)


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        self.store.add(
            "job-prior",
            _task("t-codex", "audit", "codex"),
            {"adapter": "codex", "ephemeral": False, "thread_id": "stale-thread"},
            at="2026-10-04T00:00:00",
        )
        self.store.artifacts["job-prior"].append(
            Artifact(
                job_id="job-prior", task_id="t-codex", type=ArtifactType.VERIFICATION,
                created_by="worker", confidence=0.9, evidence=[],
                payload={"adapter": "codex", "ephemeral": False, "thread_id": THREAD_ID},
                created_at="2026-10-04T01:00:00",
            )
        )
        self.store.add(
            "job-prior",
            _task("t-claude", "review", "claude-code"),
            {"adapter": "claude-code", "session_id": PRIOR_SESSION},
        )
        self.store.add(
            "job-prior",
            _task("t-ephemeral", "explore", "codex"),
            {"adapter": "codex", "ephemeral": True, "thread_id": "ephemeral-thread"},
        )
        self.store.add("job-prior", _task("t-noid", "plan", "claude-code"), {"adapter": "claude-code"})

    def resolve(self, payload: dict, adapter: str):
        return resolve_worker_resume(self.store, payload, adapter)

    def test_resolved_codex_uses_latest_non_ephemeral_thread(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-prior", "task_id": "t-codex"}}, "codex")
        self.assertEqual(
            record,
            {
                "status": "resolved",
                "adapter": "codex",
                "session_id": THREAD_ID,
                "from_job_id": "job-prior",
                "from_task_id": "t-codex",
            },
        )

    def test_a_later_verdict_artifact_is_not_mistaken_for_the_receipt(self) -> None:
        self.store.artifacts["job-prior"].append(
            Artifact(
                job_id="job-prior", task_id="t-codex", type=ArtifactType.VERIFICATION,
                created_by="worker", confidence=1.0, evidence=["worker_verdict"],
                payload={"adapter": "codex", "kind": "worker_verdict", "verdict": "PASS",
                         "reason": "ok", "check": "c", "result": "passed"},
                created_at="2026-10-04T02:00:00",
            )
        )
        record = self.resolve({"resume_from": {"job_id": "job-prior", "task_id": "t-codex"}}, "codex")
        self.assertEqual(record["status"], "resolved", record)
        self.assertEqual(record["session_id"], THREAD_ID)

    def test_resolved_claude_code_by_role(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-prior", "role": "review"}}, "claude-code")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["session_id"], PRIOR_SESSION)
        self.assertEqual(record["from_task_id"], "t-claude")

    def test_missing_job_is_unavailable(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-gone", "role": "audit"}}, "codex")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("job-gone", record["reason"])
        self.assertEqual(record["resume_from"], {"job_id": "job-gone", "role": "audit"})

    def test_missing_task_or_role_is_unavailable(self) -> None:
        for request in ({"job_id": "job-prior", "task_id": "t-none"}, {"job_id": "job-prior", "role": "nobody"}):
            with self.subTest(request=request):
                record = self.resolve({"resume_from": request}, "codex")
                self.assertEqual(record["status"], "unavailable")
                self.assertIn("not found", record["reason"])

    def test_adapter_mismatch_is_unavailable(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-prior", "task_id": "t-claude"}}, "codex")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("claude-code", record["reason"])

    def test_codex_ephemeral_prior_is_unavailable_with_guidance(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-prior", "task_id": "t-ephemeral"}}, "codex")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("ephemeral=false", record["reason"])

    def test_no_recorded_id_is_unavailable(self) -> None:
        record = self.resolve({"resume_from": {"job_id": "job-prior", "task_id": "t-noid"}}, "claude-code")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("session_id", record["reason"])

    def test_explicit_session_id_needs_no_store(self) -> None:
        record = resolve_worker_resume(None, {"resume_session_id": "sess-x", "resume_adapter": "claude-code"}, "claude-code")
        self.assertEqual(
            record,
            {"status": "resolved", "adapter": "claude-code", "session_id": "sess-x",
             "from_job_id": None, "from_task_id": None},
        )
        mismatch = resolve_worker_resume(None, {"resume_session_id": "sess-x", "resume_adapter": "codex"}, "claude-code")
        self.assertEqual(mismatch["status"], "unavailable")

    def test_existing_record_is_idempotent(self) -> None:
        stamped = {"status": "unavailable", "reason": "earlier"}
        payload = {"resume_from": {"job_id": "job-prior", "task_id": "t-codex"}, "resume": stamped}
        self.assertIs(self.resolve(payload, "codex"), stamped)

    def test_not_requested_returns_none(self) -> None:
        self.assertIsNone(self.resolve({}, "codex"))

    def test_store_errors_become_unavailable(self) -> None:
        class Broken(FakeStore):
            def list_tasks(self, job_id):
                raise RuntimeError("db locked")

        store = Broken()
        store.jobs["job-prior"] = object()
        record = resolve_worker_resume(store, {"resume_from": {"job_id": "job-prior"}}, "codex")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("db locked", record["reason"])


class SessionStoreTests(unittest.TestCase):
    def _store_root(self, adapter: str) -> Path:
        base = Path(_SESSION_HOMES.name)
        root = base / "codex" / "sessions" / "2026" / "10" / "04" if adapter == "codex" else base / "claude" / "projects" / "-repo"
        root.mkdir(parents=True, exist_ok=True)
        top = base / ("codex" if adapter == "codex" else "claude")
        self.addCleanup(shutil.rmtree, top, True)
        return root

    def test_session_missing_from_local_store_is_unavailable(self) -> None:
        self._store_root("claude-code")
        record = resolve_worker_resume(None, {"resume_session_id": PRIOR_SESSION}, "claude-code")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("not in the local session store", record["reason"])

    def test_session_present_in_local_store_resolves(self) -> None:
        (self._store_root("claude-code") / f"{PRIOR_SESSION}.jsonl").write_text("{}\n")
        (self._store_root("codex") / f"rollout-2026-10-04T09-00-00-{THREAD_ID}.jsonl").write_text("{}\n")
        claude = resolve_worker_resume(None, {"resume_session_id": PRIOR_SESSION}, "claude-code")
        codex = resolve_worker_resume(None, {"resume_session_id": THREAD_ID}, "codex")
        self.assertEqual((claude["status"], codex["status"]), ("resolved", "resolved"))

    def test_session_id_with_path_or_glob_characters_is_rejected(self) -> None:
        for bad in ("../etc/passwd", "*", "abc/def", "a?b"):
            with self.subTest(bad=bad):
                record = resolve_worker_resume(None, {"resume_session_id": bad}, "codex")
                self.assertEqual(record["status"], "unavailable")
                self.assertIn("invalid codex session id", record["reason"])


class ResumeRecordScopeTests(unittest.TestCase):
    def test_reroute_to_another_adapter_reports_unavailable(self) -> None:
        stamped = {"status": "resolved", "adapter": "codex", "session_id": THREAD_ID,
                   "from_job_id": "job_a", "from_task_id": "task_a"}
        record = task_resume_record({"resume": stamped}, "claude-code")
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("now runs on 'claude-code'", record["reason"])
        self.assertEqual(task_resume_record({"resume": stamped}, "codex"), stamped)

    def test_only_one_task_per_job_may_continue_a_codex_thread(self) -> None:
        claimed: dict = {}
        record = {"status": "resolved", "adapter": "codex", "session_id": THREAD_ID}
        self.assertEqual(claim_resumed_session(record, claimed, "r1"), record)
        second = claim_resumed_session(record, claimed, "r2")
        self.assertEqual(second["status"], "unavailable")
        self.assertIn("already resumed by role 'r1'", second["reason"])
        forked = {"status": "resolved", "adapter": "claude-code", "session_id": PRIOR_SESSION}
        self.assertEqual(claim_resumed_session(forked, claimed, "r1"), forked)
        self.assertEqual(claim_resumed_session(forked, claimed, "r2"), forked)


class OrchestratorStampTests(unittest.TestCase):
    def test_create_tasks_stamps_resume_once_on_the_payload(self) -> None:
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.store_factory import create_store
        from puppetmaster.workers import WorkerSpec

        with tempfile.TemporaryDirectory() as tmp:
            store = create_store("sqlite", Path(tmp))
            store.init()
            prior_job = store.create_job("first pass")
            prior = Task(job_id=prior_job.id, role="audit", instruction="x", adapter="claude-code")
            store.save_task(prior)
            store.save_artifact(
                Artifact(
                    job_id=prior_job.id, task_id=prior.id, type=ArtifactType.VERIFICATION,
                    created_by="worker", confidence=0.9, evidence=["adapter:claude-code"],
                    payload={"adapter": "claude-code", "check": "x", "result": "passed",
                             "session_id": PRIOR_SESSION},
                )
            )
            job = store.create_job("revision")
            spec = WorkerSpec(
                role="audit",
                instruction="revise",
                adapter="claude-code",
                payload={"resume_from": {"job_id": prior_job.id, "role": "audit"}},
            )
            orch = Orchestrator(store)
            with patch.object(orch, "_apply_auto_routing", return_value=([spec], [])), patch.object(
                orch, "_enforce_platform_lock"
            ):
                tasks = orch._create_tasks(job, [spec])
            saved = store.get_task_by_id(tasks[0].id)

        self.assertEqual(saved.payload["resume"]["status"], "resolved")
        self.assertEqual(saved.payload["resume"]["session_id"], PRIOR_SESSION)
        self.assertEqual(saved.payload["resume"]["from_task_id"], prior.id)
        self.assertNotEqual(saved.id, prior.id)


def _run_adapter(adapter, task: Task, stdout: str, *, timed_out: bool = False):
    streamed = StreamedProcess(returncode=None if timed_out else 0, stdout=stdout, stderr="",
                               timed_out=timed_out)
    with patch("puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"), patch(
        "puppetmaster.adapters.worktree_guard", return_value=None
    ), patch("puppetmaster.adapters.git_snapshot", side_effect=[CLEAN, CLEAN]), patch(
        "puppetmaster.adapters.run_streamed_subprocess", return_value=streamed
    ) as run, patch(
        "puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda prompt, **_: (prompt + "\n[codegraph]", True)
    ) as enrich, patch(
        "puppetmaster.adapters.with_repo_census", side_effect=lambda prompt, cwd: prompt + "\n[census]"
    ) as census:
        artifacts = adapter.run(task, "goal", "worker")
    return artifacts, run.call_args.kwargs, enrich, census


def _codex_events(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


RESUMED_CODEX_STDOUT = _codex_events(
    {"type": "thread.started", "thread_id": THREAD_ID},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "OLD ANSWER"}},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "NEW ANSWER"}},
    {"type": "turn.completed", "usage": {"input_tokens": 56144, "cached_input_tokens": 38272, "output_tokens": 50}},
)


class CodexTimeoutThreadTests(unittest.TestCase):
    """A persisted thread survives a timeout; a follow-up can resume it."""

    STARTED = {"type": "thread.started", "thread_id": THREAD_ID}

    def _task(self, **payload) -> Task:
        return Task(job_id="job-prior", role="build", instruction="Build it.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "model": "gpt-6.1-sol", "ephemeral": False,
                             **payload})

    def _timeout(self, *events: dict, **payload):
        task = self._task(**payload)
        artifacts, _, _, _ = _run_adapter(CodexAdapter(), task, _codex_events(*events), timed_out=True)
        verification = next(a for a in artifacts if a.payload.get("failure") == "timeout")
        return task, verification

    def test_timeout_records_the_one_observed_thread_and_a_follow_up_resolves_it(self) -> None:
        task, verification = self._timeout(self.STARTED, {"type": "turn.started"})
        self.assertEqual(verification.payload["result"], "failed")
        self.assertEqual(verification.payload["thread_id"], THREAD_ID)
        self.assertIs(verification.payload["ephemeral"], False)
        store = FakeStore()
        store.add("job-prior", task, verification.payload)
        with patch("puppetmaster.worker_resume.session_on_disk", return_value=True):
            record = resolve_worker_resume(
                store, {"resume_from": {"job_id": "job-prior", "task_id": task.id}}, "codex")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["session_id"], THREAD_ID)

    def test_a_follow_up_still_checks_the_session_is_on_disk(self) -> None:
        task, verification = self._timeout(self.STARTED)
        store = FakeStore()
        store.add("job-prior", task, verification.payload)
        with patch("puppetmaster.worker_resume.session_on_disk", return_value=False):
            record = resolve_worker_resume(
                store, {"resume_from": {"job_id": "job-prior", "task_id": task.id}}, "codex")
        self.assertEqual(record["status"], "unavailable")

    def test_missing_ambiguous_or_conflicting_threads_record_none(self) -> None:
        other = {"type": "thread.started", "thread_id": "11111111-2222-3333-4444-555555555555"}
        resumed = {"status": "resolved", "adapter": "codex", "session_id": THREAD_ID}
        for events, payload in (((), {}), ((self.STARTED, other), {}),
                                ((other,), {"resume": resumed})):
            with self.subTest(events=len(events), resumed=bool(payload)):
                _, verification = self._timeout(*events, **payload)
                self.assertIsNone(verification.payload["thread_id"])
                self.assertEqual(verification.payload["failure"], "timeout")

    def test_an_ephemeral_timeout_says_so(self) -> None:
        _, verification = self._timeout(self.STARTED, ephemeral=True)
        self.assertIs(verification.payload["ephemeral"], True)


class CodexImageStdinTests(unittest.TestCase):
    """``--image`` is variadic: a bare trailing ``-`` would be read as an image."""

    IMAGES = ["--image", "shot one.png", "--image", "two.png"]

    def _split(self, command: list[str]) -> tuple[list[str], list[str]]:
        cut = command.index("--")
        return command[:cut], command[cut + 1:]

    def test_fresh_and_resume_end_options_before_the_stdin_prompt(self) -> None:
        for command in (
            build_codex_exec_command(executable="codex", model="m", extra_args=self.IMAGES),
            build_codex_resume_command(executable="codex", session_id=THREAD_ID,
                                       model="m", extra_args=self.IMAGES),
        ):
            options, positionals = self._split(command)
            self.assertEqual(positionals, ["-"])
            self.assertEqual(options[-4:], self.IMAGES)
            self.assertEqual(options.count("shot one.png"), 1)
            self.assertNotIn("-", options)

    def test_text_only_prompt_still_reads_stdin(self) -> None:
        command = build_codex_exec_command(executable="codex", model="m")
        self.assertEqual(command[-2:], ["--", "-"])
        self.assertEqual(command.count("-"), 1)


class CodexResumeTests(unittest.TestCase):
    def test_resume_argv_is_exact(self) -> None:
        command = build_codex_resume_command(
            executable="codex", session_id=THREAD_ID, model="gpt-5.4-mini",
            sandbox="read-only", approval_policy="never",
        )
        self.assertEqual(
            command,
            ["codex", "exec", "resume", THREAD_ID, "--json",
             "-c", 'approval_policy="never"', "-c", 'sandbox_mode="read-only"',
             "--skip-git-repo-check", "-m", "gpt-5.4-mini", "--", "-"],
        )
        self.assertNotIn("-C", command)
        self.assertNotIn("--sandbox", command)
        self.assertNotIn("--ephemeral", command)

    def test_final_message_comes_from_the_last_turn(self) -> None:
        events = parse_codex_events(RESUMED_CODEX_STDOUT)
        self.assertEqual(last_codex_agent_message(events), "NEW ANSWER")
        replay_only = parse_codex_events(_codex_events(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "OLD"}},
            {"type": "turn.started"},
        ))
        self.assertEqual(last_codex_agent_message(replay_only), "")

    def _task(self, **payload) -> Task:
        return Task(
            job_id="job-rev", role="audit", instruction="Revise the audit.", adapter="codex",
            payload={"cwd": str(Path.cwd()), "sandbox": "read-only", "model": "gpt-5.4-mini", **payload},
        )

    def test_resolved_resume_runs_exec_resume_with_delta_prompt(self) -> None:
        record = {"status": "resolved", "adapter": "codex", "session_id": THREAD_ID,
                  "from_job_id": "job-prior", "from_task_id": "t-codex"}
        artifacts, kwargs, enrich, census = _run_adapter(
            CodexAdapter(), self._task(resume=record, native_steer=True), RESUMED_CODEX_STDOUT
        )
        command = kwargs["command"]
        self.assertEqual(command[:5], ["/usr/bin/codex", "exec", "resume", THREAD_ID, "--json"])
        self.assertIn('sandbox_mode="read-only"', command)
        self.assertNotIn("--sandbox", command)
        self.assertNotIn("-C", command)
        self.assertNotIn("--ephemeral", command)
        self.assertEqual(command[-1], "-")
        enrich.assert_not_called()
        census.assert_not_called()
        verification = artifacts[0]
        self.assertIn("context:resumed", verification.evidence)
        self.assertEqual(verification.payload["resume"], record)
        self.assertEqual(verification.payload["thread_id"], THREAD_ID)
        self.assertFalse(verification.payload["ephemeral"])
        self.assertEqual(verification.payload["last_message"], "NEW ANSWER")
        # Resumed SDK usage is session-cumulative: kept as sdk_*, never the attempt's counters.
        self.assertEqual(verification.payload["sdk_cached_input_tokens"], 38272)
        self.assertEqual(verification.payload["sdk_usage_scope"], "session_cumulative")
        self.assertIsNone(verification.payload["cached_input_tokens"])
        self.assertEqual(verification.payload["usage_scope"], "unknown")

    def test_fresh_argv_is_unchanged(self) -> None:
        artifacts, kwargs, enrich, census = _run_adapter(
            CodexAdapter(), self._task(), _codex_events({"type": "thread.started", "thread_id": THREAD_ID})
        )
        expected = build_codex_exec_command(
            executable=["/usr/bin/codex"], model="gpt-5.4-mini", cwd=Path.cwd().resolve(),
            sandbox="read-only", approval_policy="never", ephemeral=True,
        )
        self.assertEqual(kwargs["command"], expected)
        enrich.assert_called_once()
        census.assert_called_once()
        self.assertNotIn("resume", artifacts[0].payload)
        self.assertNotIn("context:resumed", artifacts[0].evidence)

    def test_unavailable_runs_fresh_and_records_reason(self) -> None:
        record = {"status": "unavailable", "reason": "prior codex task ran ephemeral; launch with ephemeral=false",
                  "resume_from": {"job_id": "job-prior"}}
        artifacts, kwargs, enrich, _ = _run_adapter(
            CodexAdapter(), self._task(resume=record), _codex_events({"type": "thread.started", "thread_id": THREAD_ID})
        )
        command = kwargs["command"]
        self.assertNotIn("resume", command)
        self.assertIn("--ephemeral", command)
        enrich.assert_called_once()
        self.assertEqual(artifacts[0].payload["resume"], record)
        self.assertNotIn("context:resumed", artifacts[0].evidence)


    def _resolved(self) -> dict:
        return {"status": "resolved", "adapter": "codex", "session_id": THREAD_ID,
                "from_job_id": "job-prior", "from_task_id": "t-codex"}

    def test_thread_held_by_another_job_runs_fresh_before_any_model_call(self) -> None:
        held = acquire_codex_thread(THREAD_ID)
        self.assertIsNotNone(held)
        try:
            artifacts, kwargs, enrich, census = _run_adapter(
                CodexAdapter(), self._task(resume=self._resolved()),
                _codex_events({"type": "thread.started", "thread_id": NEW_SESSION}),
            )
        finally:
            held.release()
        self.assertNotIn("resume", kwargs["command"])
        enrich.assert_called_once()
        census.assert_called_once()
        record = artifacts[0].payload["resume"]
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("another live worker", record["reason"])
        self.assertNotIn("context:resumed", artifacts[0].evidence)

    def test_lease_is_released_after_the_run(self) -> None:
        _run_adapter(CodexAdapter(), self._task(resume=self._resolved()), RESUMED_CODEX_STDOUT)
        again = acquire_codex_thread(THREAD_ID)
        self.assertIsNotNone(again)
        again.release()

    def test_lease_is_released_when_the_run_raises(self) -> None:
        with patch("puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"), patch(
            "puppetmaster.adapters.worktree_guard", return_value=None
        ), patch("puppetmaster.adapters.git_snapshot", return_value=CLEAN), patch(
            "puppetmaster.adapters.run_streamed_subprocess", side_effect=RuntimeError("spawn failed")
        ):
            with self.assertRaises(RuntimeError):
                CodexAdapter().run(self._task(resume=self._resolved()), "goal", "worker")
        again = acquire_codex_thread(THREAD_ID)
        self.assertIsNotNone(again)
        again.release()



class CodexBuildContractTests(unittest.TestCase):
    """Write-capable Codex workers get a build contract, not the findings contract."""

    def _task(self, **payload) -> Task:
        return Task(
            job_id="job-build", role="builder", instruction="Build regions/a.py.", adapter="codex",
            payload={"cwd": str(Path.cwd()), "model": "gpt-5.4-mini", **payload},
        )

    def _prompt(self, task: Task):
        artifacts, kwargs, enrich, census = _run_adapter(
            CodexAdapter(), task, _codex_events({"type": "thread.started", "thread_id": THREAD_ID})
        )
        return kwargs["stdin_data"], artifacts, enrich, census

    def test_write_capable_worker_gets_the_build_contract(self) -> None:
        prompt, _, enrich, census = self._prompt(self._task(sandbox="workspace-write"))
        self.assertIn("Build mode", prompt)
        self.assertIn("VERDICT: PASS", prompt)
        self.assertNotIn("submit_findings", prompt)
        self.assertNotIn("Your analysis target", prompt)
        self.assertIn("Your task:\nBuild regions/a.py.", prompt)
        census.assert_not_called()
        enrich.assert_called_once()

    def test_read_only_worker_keeps_the_analysis_contract(self) -> None:
        prompt, _, _, census = self._prompt(self._task(sandbox="read-only"))
        self.assertIn("submit_findings", prompt)
        self.assertNotIn("Build mode", prompt)
        census.assert_called_once()

    def test_builder_skips_the_job_brief_only_when_it_gets_task_codegraph(self) -> None:
        with patch("puppetmaster.job_brief.resolve_job_brief_for_task", return_value="JOB-BRIEF-SECTION"):
            with_graph, _, _, _ = self._prompt(self._task(sandbox="workspace-write"))
            without_graph, _, _, _ = self._prompt(
                self._task(sandbox="workspace-write", disable_codegraph=True)
            )
            analysis, _, _, _ = self._prompt(self._task(sandbox="read-only"))
        self.assertNotIn("JOB-BRIEF-SECTION", with_graph)
        self.assertIn("JOB-BRIEF-SECTION", without_graph)
        self.assertIn("JOB-BRIEF-SECTION", analysis)

    def test_build_report_is_a_passed_receipt_with_its_verdict(self) -> None:
        stdout = _codex_events(
            {"type": "thread.started", "thread_id": THREAD_ID},
            {"type": "item.completed", "item": {"type": "agent_message",
             "text": "Built regions/a.py; judge passes 12/12.\nVERDICT: PASS - all checks pass"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )
        artifacts, _, _, _ = _run_adapter(CodexAdapter(), self._task(sandbox="workspace-write"), stdout)
        self.assertEqual(artifacts[0].payload["result"], "passed")
        kinds = [(a.type, (a.payload or {}).get("kind")) for a in artifacts]
        self.assertIn((ArtifactType.FINDING, None), kinds)


class CodexReviewContractTests(unittest.TestCase):
    """A read-only worker asked for a verdict reviews in free text, not findings JSON."""

    def _task(self, **payload) -> Task:
        return Task(job_id="job-review", role="check", instruction="Review pkg/a.py.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "model": "gpt-5.4-mini", "sandbox": "read-only",
                             "read_only": True, **payload})

    def test_verdict_reviewer_gets_the_review_contract(self):
        artifacts, kwargs, _, _ = _run_adapter(
            CodexAdapter(), self._task(terminal_verdict=True),
            _codex_events({"type": "thread.started", "thread_id": THREAD_ID}))
        prompt = kwargs["stdin_data"]
        self.assertIn("Review mode", prompt)
        self.assertIn("path:line - what is wrong", prompt)
        self.assertNotIn("submit_findings", prompt)

    def test_free_text_review_with_a_verdict_passes_and_keeps_the_report(self):
        stdout = _codex_events(
            {"type": "thread.started", "thread_id": THREAD_ID},
            {"type": "item.completed", "item": {"type": "agent_message",
             "text": "pkg/a.py:3 - missing type hints\nVERDICT: FAIL - pkg/a.py:3 missing type hints"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )
        artifacts, _, _, _ = _run_adapter(CodexAdapter(), self._task(terminal_verdict=True), stdout)
        self.assertEqual(artifacts[0].payload["result"], "passed")
        verdicts = [a.payload for a in artifacts if (a.payload or {}).get("kind") == "worker_verdict"]
        self.assertEqual([v["verdict"] for v in verdicts], ["FAIL"])
        self.assertTrue(any(a.type == ArtifactType.FINDING and "missing type hints" in (a.payload.get("report") or "")
                            for a in artifacts))

    def test_plain_read_only_analysis_is_unchanged(self):
        artifacts, kwargs, _, _ = _run_adapter(
            CodexAdapter(), self._task(),
            _codex_events({"type": "thread.started", "thread_id": THREAD_ID}))
        self.assertIn("submit_findings", kwargs["stdin_data"])
        self.assertNotIn("Review mode", kwargs["stdin_data"])

    def test_claude_verdict_reviewer_gets_the_review_contract(self):
        task = Task(job_id="job-review", role="check", instruction="Review pkg/a.py.", adapter="claude-code",
                    payload={"cwd": str(Path.cwd()), "permission_mode": "plan", "terminal_verdict": True})
        _, kwargs, enrich, _ = _run_adapter(ClaudeCodeAdapter(), task, CLAUDE_RESULT)
        prompt = enrich.call_args.args[0]
        self.assertIn("Review mode", prompt)
        self.assertNotIn("Reporting contract", prompt)


class SessionLeaseTests(unittest.TestCase):
    def test_second_holder_is_refused_until_release(self) -> None:
        first = acquire_codex_thread(THREAD_ID)
        self.assertIsNotNone(first)
        self.assertIsNone(acquire_codex_thread(THREAD_ID))
        first.release()
        first.release()  # idempotent
        second = acquire_codex_thread(THREAD_ID)
        self.assertIsNotNone(second)
        second.release()

    def test_distinct_threads_do_not_contend(self) -> None:
        first = acquire_codex_thread(THREAD_ID)
        other = acquire_codex_thread(PRIOR_SESSION)
        self.assertIsNotNone(first)
        self.assertIsNotNone(other)
        first.release()
        other.release()

    def test_unsafe_thread_id_is_rejected(self) -> None:
        with self.assertRaises(OSError):
            acquire_codex_thread("../escape")


CLAUDE_RESULT = json.dumps({
    "type": "result", "subtype": "success", "is_error": False, "session_id": NEW_SESSION,
    "result": "done", "total_cost_usd": 0.01,
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 48142, "output_tokens": 5},
})


class ClaudeCodeResumeTests(unittest.TestCase):
    def _task(self, **payload) -> Task:
        return Task(
            job_id="job-rev", role="audit", instruction="Revise the audit.", adapter="claude-code",
            payload={"cwd": str(Path.cwd()), "allow_dirty": True, **payload},
        )

    def test_only_a_resumed_run_reports_session_cumulative_cost(self) -> None:
        from puppetmaster.adapters._base import CliInvocation

        for resumed in (False, True):
            with self.subTest(resumed=resumed), patch(
                    "puppetmaster.adapters._base.facade",
                    return_value=lambda **kw: StreamedProcess(0, CLAUDE_RESULT, "")):
                prepared = CliInvocation(command=["claude"], sidecar_name="x", extras={"resumed": resumed})
                result = ClaudeCodeAdapter()._invoke_cli(self._task(), prepared, Path.cwd(), 5)
            self.assertIs(result.session_cumulative_cost, resumed)

    def test_command_adds_resume_and_fork_only_when_given(self) -> None:
        fresh = build_claude_code_command(executable="claude", permission_mode="acceptEdits")
        self.assertNotIn("--resume", fresh)
        self.assertNotIn("--fork-session", fresh)
        resumed = build_claude_code_command(executable="claude", resume_session_id=PRIOR_SESSION)
        index = resumed.index("--resume")
        self.assertEqual(resumed[index:index + 3], ["--resume", PRIOR_SESSION, "--fork-session"])

    def test_fresh_run_records_session_id_and_enriches(self) -> None:
        with patch("puppetmaster.adapters.claude_code.prompt_with_memory", side_effect=lambda p, t: p + "\n[memory]") as memory:
            artifacts, kwargs, enrich, _ = _run_adapter(ClaudeCodeAdapter(), self._task(), CLAUDE_RESULT)
        self.assertNotIn("--resume", kwargs["command"])
        enrich.assert_called_once()
        memory.assert_called_once()
        verification = artifacts[0]
        self.assertEqual(verification.payload["session_id"], NEW_SESSION)
        self.assertNotIn("resume", verification.payload)

    def test_builder_skips_the_job_brief_only_when_it_gets_task_codegraph(self) -> None:
        def prompt_for(**payload):
            with patch("puppetmaster.job_brief.resolve_job_brief_for_task", return_value="JOB-BRIEF-SECTION"):
                _, kwargs, enrich, _ = _run_adapter(ClaudeCodeAdapter(), self._task(**payload), CLAUDE_RESULT)
            return enrich.call_args.args[0]

        self.assertNotIn("JOB-BRIEF-SECTION", prompt_for(permission_mode="acceptEdits"))
        self.assertIn("JOB-BRIEF-SECTION", prompt_for(permission_mode="acceptEdits", disable_codegraph=True))
        self.assertIn("JOB-BRIEF-SECTION", prompt_for(permission_mode="plan"))

    def test_resolved_resume_forks_and_skips_enrichment(self) -> None:
        record = {"status": "resolved", "adapter": "claude-code", "session_id": PRIOR_SESSION,
                  "from_job_id": "job-prior", "from_task_id": "t-claude"}
        with patch("puppetmaster.adapters.claude_code.prompt_with_memory") as memory:
            artifacts, kwargs, enrich, census = _run_adapter(
                ClaudeCodeAdapter(), self._task(resume=record), CLAUDE_RESULT
            )
        command = kwargs["command"]
        index = command.index("--resume")
        self.assertEqual(command[index:index + 3], ["--resume", PRIOR_SESSION, "--fork-session"])
        enrich.assert_not_called()
        memory.assert_not_called()
        census.assert_not_called()
        self.assertIn("Revise the audit.", kwargs["task"].instruction)
        verification = artifacts[0]
        self.assertIn("context:resumed", verification.evidence)
        self.assertEqual(verification.payload["resume"], record)
        self.assertEqual(verification.payload["session_id"], NEW_SESSION)

    def test_explicit_session_id_resumes_without_a_store(self) -> None:
        artifacts, kwargs, _, _ = _run_adapter(
            ClaudeCodeAdapter(), self._task(resume_session_id=PRIOR_SESSION), CLAUDE_RESULT
        )
        self.assertIn("--fork-session", kwargs["command"])
        self.assertEqual(artifacts[0].payload["resume"]["status"], "resolved")

    def test_session_id_from_stream_json_prefers_result(self) -> None:
        stream = "\n".join([
            json.dumps({"type": "system", "subtype": "init", "session_id": "init-id"}),
            json.dumps({"type": "assistant", "message": {}}),
            json.dumps({"type": "result", "session_id": "result-id"}),
        ])
        self.assertEqual(claude_session_id_from_stdout(stream), "result-id")
        self.assertEqual(claude_session_id_from_stdout(stream.rsplit("\n", 1)[0]), "init-id")
        self.assertIsNone(claude_session_id_from_stdout("not json"))


class SwarmResumeSurfaceTests(unittest.TestCase):
    def test_start_swarm_role_schema_accepts_resume_from(self) -> None:
        from puppetmaster.mcp_server import swarm_schema

        role_object = swarm_schema()["properties"]["roles"]["items"]["anyOf"][1]
        self.assertEqual(role_object["properties"]["resume_from"]["required"], ["job_id"])

    def test_role_resume_from_lands_in_worker_payload(self) -> None:
        from puppetmaster.config import load_config
        from puppetmaster.swarm_launch import write_analysis_swarm_config

        resume_from = {"job_id": "job-prior", "role": "audit"}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_analysis_swarm_config(
                goal="revise",
                roles=[{"name": "audit", "instruction": "revise the audit", "resume_from": resume_from}],
                adapter="codex",
                state_dir=Path(tmp),
                cwd="/repo",
                model="gpt-5.4-mini",
            )
            config = load_config(path)
        self.assertEqual(config.workers[0].payload["resume_from"], resume_from)

    def test_role_resume_from_requires_job_id(self) -> None:
        from puppetmaster.swarm_launch import build_analysis_swarm_specs

        with self.assertRaises(ValueError):
            build_analysis_swarm_specs(
                "goal", [{"name": "audit", "instruction": "x", "resume_from": {"role": "audit"}}],
                adapter="codex",
            )


if __name__ == "__main__":
    unittest.main()
