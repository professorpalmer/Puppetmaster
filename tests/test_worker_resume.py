"""Opt-in worker session resume for the codex and claude-code adapters."""

from __future__ import annotations

import json
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
from puppetmaster.worker_resume import resolve_worker_resume

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


def _run_adapter(adapter, task: Task, stdout: str):
    streamed = StreamedProcess(returncode=0, stdout=stdout, stderr="")
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
             "--skip-git-repo-check", "-m", "gpt-5.4-mini", "-"],
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
        self.assertEqual(verification.payload["cached_input_tokens"], 38272)

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
