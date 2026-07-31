"""Quality-control fixes: provenance, contradictory peers, acceptance criteria."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.acceptance_criteria import (
    acceptance_criteria_for_task,
    criterion_status_records,
    ensure_acceptance_criteria_in_text,
    parse_acceptance_criteria_block,
)
from puppetmaster.adapters._base import verification_artifact
from puppetmaster.adapters._prompts import (
    TASK_INSTRUCTION_HEADER,
    build_structured_prompt,
    prompt_with_memory,
    structured_prompt_for_task,
    with_job_brief,
    with_repo_census,
)
from puppetmaster.claim_conflicts import (
    detect_contradictory_peers,
    source_verify_claim,
)
from puppetmaster.execution_provenance import (
    PROVENANCE_KEY,
    _merge_provenance,
    provenance_from_task_and_payload,
    stamp_execution_provenance,
)
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.stitcher import Stitcher
from puppetmaster.store_factory import create_store


def _render(artifacts, *, cwd=None) -> str:
    stitcher = object.__new__(Stitcher)
    return Stitcher._render_summary(
        stitcher, "Live", "goal", list(artifacts), [], cwd=cwd
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DELTA_BUS = "puppetmaster/adapters/_delta_bus.py"


def _finding(
    claim: str,
    *,
    evidence,
    confidence: float = 0.9,
    artifact_id: str = "",
    task_id: str = "t1",
) -> Artifact:
    art = Artifact(
        job_id="job_qc",
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="worker-a",
        confidence=confidence,
        evidence=list(evidence),
        payload={"claim": claim},
    )
    if artifact_id:
        object.__setattr__(art, "id", artifact_id)
    return art


class AcceptanceCriteriaTests(unittest.TestCase):
    def test_parse_explicit_block_only(self) -> None:
        text = (
            "Validate routing ledger model ids.\n\n"
            "Acceptance criteria:\n"
            "- ROUTING artifacts expose model_id and adapter_model_name\n"
            "- plan-billed marginal cost stays honest\n\n"
            "Also look around the repo casually."
        )
        criteria = parse_acceptance_criteria_block(text)
        self.assertEqual(
            criteria,
            [
                "ROUTING artifacts expose model_id and adapter_model_name",
                "plan-billed marginal cost stays honest",
            ],
        )
        self.assertEqual(parse_acceptance_criteria_block("no block here"), [])

    def test_omitted_criterion_is_unknown_not_passed(self) -> None:
        rows = criterion_status_records(
            ["must check routing ledger", "must check model id"],
            reported=[{"criterion": "must check routing ledger", "status": "passed", "evidence": "routing.py:42"}],
        )
        self.assertEqual(rows[0]["status"], "passed")
        self.assertEqual(rows[0]["evidence"], "routing.py:42")
        self.assertEqual(rows[1]["status"], "unknown")
        self.assertEqual(rows[1]["evidence"], "not_reported")
        self.assertNotEqual(rows[1]["status"], "passed")

    def test_passed_without_evidence_becomes_unknown(self) -> None:
        rows = criterion_status_records(
            ["ledger matches"],
            reported=[{"criterion": "ledger matches", "status": "passed"}],
        )
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_passed_with_sentinel_evidence_becomes_unknown(self) -> None:
        for sentinel in ("not_reported", "unknown", "n/a", ""):
            with self.subTest(sentinel=sentinel):
                rows = criterion_status_records(
                    ["ledger matches"],
                    reported=[
                        {
                            "criterion": "ledger matches",
                            "status": "passed",
                            "evidence": sentinel,
                        }
                    ],
                )
                self.assertEqual(rows[0]["status"], "unknown")
                self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_failed_with_not_reported_evidence_becomes_unknown(self) -> None:
        rows = criterion_status_records(
            ["must fail cleanly"],
            reported=[
                {
                    "criterion": "must fail cleanly",
                    "status": "failed",
                    "evidence": "not_reported",
                }
            ],
        )
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_foreign_criterion_cannot_settle_checklist(self) -> None:
        rows = criterion_status_records(
            ["task criterion A", "task criterion B"],
            reported=[
                {
                    "criterion": "foreign criterion not on task",
                    "status": "passed",
                    "evidence": "fake.py:1",
                }
            ],
        )
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[1]["status"], "unknown")

    def test_malformed_criterion_rows_are_ignored(self) -> None:
        rows = criterion_status_records(
            ["must pass"],
            reported=[
                "not a dict",
                {"status": "passed", "evidence": "x.py:1"},
                {"criterion": "must pass", "status": "passed"},
            ],
        )
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_valid_string_and_list_evidence_preserved(self) -> None:
        rows = criterion_status_records(
            ["single evidence", "list evidence"],
            reported=[
                {
                    "criterion": "single evidence",
                    "status": "passed",
                    "evidence": "routing.py:42",
                },
                {
                    "criterion": "list evidence",
                    "status": "failed",
                    "evidence": ["routing.py:42", "ledger.py:10"],
                },
            ],
        )
        self.assertEqual(rows[0]["status"], "passed")
        self.assertEqual(rows[0]["evidence"], "routing.py:42")
        self.assertEqual(rows[1]["status"], "failed")
        self.assertEqual(rows[1]["evidence"], "routing.py:42; ledger.py:10")

    def test_malformed_evidence_downgrades_passed_to_unknown(self) -> None:
        class _EvidenceObject:
            def __str__(self) -> str:
                return "object-evidence.py:1"

        malformed_cases = (
            {"evidence": {"file": "routing.py", "line": 42}},
            {"evidence": 42},
            {"evidence": 3.14},
            {"evidence": True},
            {"evidence": _EvidenceObject()},
            {"evidence": [None, "routing.py:42"]},
            {"evidence": ["routing.py:42", 99]},
            {"evidence": [["routing.py:42"]]},
            {"evidence": ["routing.py:42", ""]},
            {"evidence": ["", "routing.py:42"]},
        )
        for case in malformed_cases:
            with self.subTest(evidence=case["evidence"]):
                rows = criterion_status_records(
                    ["ledger matches"],
                    reported=[
                        {
                            "criterion": "ledger matches",
                            "status": "passed",
                            **case,
                        }
                    ],
                )
                self.assertEqual(rows[0]["status"], "unknown")
                self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_structured_prompt_includes_criterion_reporting_contract(self) -> None:
        instruction = (
            "Audit routing.\n\nAcceptance criteria:\n- ROUTING exposes model_id\n"
        )
        prompt = build_structured_prompt(
            instruction,
            final_message_note=True,
            acceptance_criteria=["ROUTING exposes model_id"],
        )
        self.assertIn("acceptance_criteria", prompt)
        self.assertIn("current-dispatch", prompt)
        self.assertIn("Do not copy the whole checklist", prompt)

    def test_criteria_survive_prompt_layers(self) -> None:
        instruction = (
            "Validate the routing ledger for model identity tokens.\n\n"
            "Acceptance criteria:\n"
            "- Assert router_model_id matches the ROUTING artifact\n"
            "- Do not substitute a generic repo audit\n"
        )
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction=instruction,
            payload={
                "acceptance_criteria": acceptance_criteria_for_task(
                    Task(job_id="x", role="a", instruction=instruction)
                ),
                "retrieved_memory": [
                    {
                        "scope": "swarm.decisions",
                        "statement": "prefer sqlite store for tests",
                    }
                ],
                "cwd": str(REPO_ROOT),
            },
        )
        base = build_structured_prompt(
            instruction,
            acceptance_criteria=task.payload["acceptance_criteria"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.py").write_text("x = 1\n", encoding="utf-8")
            assembled = prompt_with_memory(
                with_repo_census(with_job_brief(base, task), tmp),
                task,
            )
        marker = assembled.index(TASK_INSTRUCTION_HEADER)
        after_task = assembled[marker:]
        self.assertIn("Assert router_model_id matches the ROUTING artifact", after_task)
        self.assertIn("Do not substitute a generic repo audit", after_task)
        self.assertIn("Acceptance criteria:", after_task)
        # Original instruction body is preserved (not rewritten away).
        self.assertIn("Validate the routing ledger", after_task)

    def test_verification_reports_unknown_when_omitted(self) -> None:
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="check\n\nAcceptance criteria:\n- ledger matches\n",
            adapter="cursor",
            payload={"acceptance_criteria": ["ledger matches"], "model": "grok-4.5"},
        )
        art = verification_artifact(
            task=task,
            worker_id="w1",
            adapter="cursor",
            check=task.instruction,
            result="passed",
            confidence=0.9,
            evidence=["adapter:cursor"],
            payload={"model": "grok-4.5"},
        )
        rows = art.payload["acceptance_criteria"]
        self.assertEqual(rows[0]["criterion"], "ledger matches")
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_verification_criteria_stamp_failure_strips_raw_rows(self) -> None:
        from unittest.mock import patch

        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="check\n\nAcceptance criteria:\n- real criterion\n",
            adapter="cursor",
            payload={"acceptance_criteria": ["real criterion"]},
        )
        with patch(
            "puppetmaster.acceptance_criteria.criterion_status_records",
            side_effect=RuntimeError("stamp boom"),
        ):
            art = verification_artifact(
                task=task,
                worker_id="w1",
                adapter="cursor",
                check="check",
                result="passed",
                confidence=0.9,
                evidence=["adapter:cursor"],
                payload={
                    "acceptance_criteria": [
                        {
                            "criterion": "foreign criterion",
                            "status": "passed",
                            "evidence": "fake.py:1",
                        },
                        {
                            "criterion": "real criterion",
                            "status": "passed",
                            "evidence": "not_reported",
                        },
                    ],
                },
            )
        rows = art.payload["acceptance_criteria"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["criterion"], "real criterion")
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")
        self.assertNotEqual(
            rows[0]["status"],
            "passed",
            msg="raw passed rows must not survive stamp failure",
        )

    def test_verification_criteria_stamp_function_failure_strips_raw_rows(
        self,
    ) -> None:
        from unittest.mock import patch

        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="check\n\nAcceptance criteria:\n- real criterion\n",
            adapter="cursor",
            payload={"acceptance_criteria": ["real criterion"]},
        )
        with patch(
            "puppetmaster.acceptance_criteria.stamp_verification_acceptance_criteria",
            side_effect=RuntimeError("stamp boom"),
        ):
            art = verification_artifact(
                task=task,
                worker_id="w1",
                adapter="cursor",
                check="check",
                result="passed",
                confidence=0.9,
                evidence=["adapter:cursor"],
                payload={
                    "acceptance_criteria": [
                        {
                            "criterion": "foreign criterion",
                            "status": "passed",
                            "evidence": "fake.py:1",
                        },
                    ],
                },
            )
        rows = art.payload["acceptance_criteria"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["criterion"], "real criterion")
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertEqual(rows[0]["evidence"], "not_reported")

    def test_verification_criteria_resolver_failure_removes_raw_rows(self) -> None:
        from unittest.mock import patch

        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="check\n\nAcceptance criteria:\n- real criterion\n",
            adapter="cursor",
            payload={"acceptance_criteria": ["real criterion"]},
        )
        with patch(
            "puppetmaster.acceptance_criteria.acceptance_criteria_for_task",
            side_effect=RuntimeError("resolver boom"),
        ):
            art = verification_artifact(
                task=task,
                worker_id="w1",
                adapter="cursor",
                check="check",
                result="passed",
                confidence=0.9,
                evidence=["adapter:cursor"],
                payload={
                    "acceptance_criteria": [
                        {
                            "criterion": "foreign criterion",
                            "status": "passed",
                            "evidence": "fake.py:1",
                        },
                    ],
                },
            )
        self.assertNotIn("acceptance_criteria", art.payload)


class ExecutionProvenanceTests(unittest.TestCase):
    def test_known_usage_and_plan_priced_zero(self) -> None:
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="analyze",
            adapter="cursor",
            payload={
                "router_model_id": "cursor/grok-4-5",
                "model": "grok-4.5",
                "billing": "plan",
            },
        )
        art = Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.FINDING,
            created_by="w1",
            confidence=0.9,
            evidence=["adapter:cursor", f"{DELTA_BUS}:22"],
            payload={
                "claim": "delta bus uses threading.Lock",
                "tokens_in": 100,
                "tokens_out": 20,
                "tokens_estimated": False,
            },
        )
        stamped = stamp_execution_provenance(art, task=task)
        prov = stamped.payload[PROVENANCE_KEY]
        self.assertEqual(prov["adapter"], "cursor")
        self.assertEqual(prov["router_model_id"], "cursor/grok-4-5")
        self.assertEqual(prov["adapter_model_name"], "grok-4.5")
        self.assertEqual(prov["tokens_in"], 100)
        self.assertEqual(prov["tokens_out"], 20)
        self.assertTrue(prov["usage_known"])
        self.assertFalse(prov["tokens_estimated"])
        self.assertEqual(prov["cost_usd"], 0.0)
        self.assertTrue(prov["cost_known"])
        self.assertTrue(prov["priced"])
        self.assertEqual(prov["cost_source"], "plan")

    def test_unknown_usage_and_cost_never_fabricate_zero(self) -> None:
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="analyze",
            adapter="hermes",
            payload={},
        )
        art = Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.RISK,
            created_by="w1",
            confidence=0.85,
            evidence=["adapter:hermes"],
            payload={
                "risk": "unknown model spend",
                "mitigation": "price after registry match",
            },
        )
        stamped = stamp_execution_provenance(art, task=task)
        prov = stamped.payload[PROVENANCE_KEY]
        self.assertEqual(prov["adapter"], "hermes")
        # Unknown identity/tokens/cost: null or absent — never fabricated 0.
        self.assertIn(prov.get("router_model_id"), (None,))
        self.assertIn(prov.get("adapter_model_name"), (None,))
        self.assertNotIn("tokens_in", prov)
        self.assertNotIn("tokens_out", prov)
        self.assertNotIn("cost_usd", prov)
        self.assertFalse(prov["usage_known"])
        self.assertFalse(prov["cost_known"])
        self.assertFalse(prov["priced"])
        self.assertEqual(prov["cost_source"], "unknown")

    def test_file_and_sqlite_roundtrip(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as tmp:
                    store = create_store(backend, tmp)
                    job = store.create_job("qc provenance")
                    task = Task(
                        job_id=job.id,
                        role="audit",
                        instruction="analyze",
                        adapter="cursor",
                        payload={
                            "router_model_id": "cursor/composer-2-5",
                            "model": "composer-2.5",
                            "billing": "plan",
                            "cwd": str(REPO_ROOT),
                        },
                    )
                    store.save_tasks([task])
                    art = Artifact(
                        job_id=job.id,
                        task_id=task.id,
                        type=ArtifactType.DECISION,
                        created_by="w1",
                        confidence=0.9,
                        evidence=["adapter:cursor"],
                        payload={
                            "decision": "keep ROUTING ledger",
                            "why": "audit trail",
                            "tokens_in": 10,
                            "tokens_out": 4,
                            "tokens_estimated": True,
                        },
                    )
                    store.save_artifact(art)
                    loaded = store.get_artifacts_by_ids(job.id, [art.id])[art.id]
                    prov = loaded.payload[PROVENANCE_KEY]
                    self.assertEqual(prov["router_model_id"], "cursor/composer-2-5")
                    self.assertEqual(prov["adapter_model_name"], "composer-2.5")
                    self.assertEqual(prov["tokens_in"], 10)
                    self.assertTrue(prov["tokens_estimated"])
                    self.assertTrue(prov["usage_known"])
                    self.assertEqual(prov["cost_source"], "plan")
                    self.assertEqual(prov["cost_usd"], 0.0)


class ContradictoryPeerTests(unittest.TestCase):
    def test_lock_type_conflict_leaves_findings(self) -> None:
        wrong = _finding(
            f"{DELTA_BUS} protects sinks with asyncio.Lock",
            evidence=[f"{DELTA_BUS}:22"],
            artifact_id="artifact_async",
        )
        right = _finding(
            f"{DELTA_BUS} protects sinks with threading.Lock",
            evidence=[f"{DELTA_BUS}:22"],
            artifact_id="artifact_thread",
        )
        conflicts = detect_contradictory_peers([wrong, right], cwd=REPO_ROOT)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].reason, "incompatible_lock_types")
        self.assertLess(conflicts[0].confidence, 0.8)
        self.assertIn(
            conflicts[0].source_verification,
            {"supports", "mixed", "contradicts", "unknown"},
        )

        summary = _render([wrong, right], cwd=REPO_ROOT)
        self.assertIn("## Conflicts", summary)
        self.assertIn("artifact_async", summary)
        self.assertIn("artifact_thread", summary)
        # Conflicting peers leave ordinary Findings.
        findings_section = summary.split("## Findings")[1].split("## Conflicts")[0]
        self.assertIn("- None", findings_section)

    def test_wrong_add_listener_line_vs_register_delta_sink(self) -> None:
        # register_delta_sink lives at line 27 in the current tree; add_listener
        # does not exist. Same locus, opposing symbols / source verification.
        wrong = _finding(
            f"add_listener is registered at {DELTA_BUS}:27",
            evidence=[f"{DELTA_BUS}:27"],
            artifact_id="artifact_wrong",
        )
        right = _finding(
            f"register_delta_sink is defined at {DELTA_BUS}:27",
            evidence=[f"{DELTA_BUS}:27"],
            artifact_id="artifact_right",
        )
        conflicts = detect_contradictory_peers([wrong, right], cwd=REPO_ROOT)
        self.assertEqual(len(conflicts), 1)
        self.assertIn(
            conflicts[0].reason,
            {
                "source_verification_disagreement",
                "conflicting_symbol_lines",
                "incompatible_lock_types",
            },
        )
        summary = _render([wrong, right], cwd=REPO_ROOT)
        self.assertIn("## Conflicts", summary)
        self.assertIn("artifact_wrong", summary)
        self.assertIn("artifact_right", summary)

    def test_conflicting_symbol_lines(self) -> None:
        a = _finding(
            "register_delta_sink is at the wrong place",
            evidence=[f"{DELTA_BUS}:10"],
            artifact_id="artifact_line_a",
        )
        # Force shared symbol into both claims.
        a = Artifact(
            id="artifact_line_a",
            job_id=a.job_id,
            task_id=a.task_id,
            type=a.type,
            created_by=a.created_by,
            confidence=0.95,
            evidence=[f"{DELTA_BUS}:10"],
            payload={"claim": "register_delta_sink lives at line 10"},
        )
        b = Artifact(
            id="artifact_line_b",
            job_id=a.job_id,
            task_id=a.task_id,
            type=ArtifactType.FINDING,
            created_by="w2",
            confidence=0.95,
            evidence=[f"{DELTA_BUS}:27"],
            payload={"claim": "register_delta_sink lives at line 27"},
        )
        conflicts = detect_contradictory_peers([a, b], cwd=REPO_ROOT)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].reason, "conflicting_symbol_lines")

    def test_distinct_same_file_bugs_not_conflict(self) -> None:
        a = _finding(
            "KeyError on unsupported operators in CLI",
            evidence=["cli.py:5"],
            artifact_id="artifact_keyerror",
        )
        b = _finding(
            "No division support in the CLI parser",
            evidence=["cli.py:5"],
            artifact_id="artifact_div",
        )
        conflicts = detect_contradictory_peers([a, b], cwd=REPO_ROOT)
        self.assertEqual(conflicts, [])
        summary = _render([a, b], cwd=REPO_ROOT)
        findings_section = summary.split("## Findings")[1].split("## Conflicts")[0]
        self.assertIn("KeyError", findings_section)
        self.assertIn("division", findings_section)
        self.assertIn("## Conflicts", summary)
        conflicts_section = summary.split("## Conflicts")[1].split("## Decisions")[0]
        self.assertIn("- None", conflicts_section)

    def test_containment_paraphrase_still_dedupes(self) -> None:
        """Non-conflicting paraphrases at one locus still collapse via dedupe."""
        arts = [
            _finding(
                "Multiplication uses repeated addition inefficiently",
                evidence=["arithmetic.py:5"],
                confidence=0.9,
                artifact_id="a1",
            ),
            _finding(
                "The multiply function uses repeated addition for large numbers",
                evidence=["arithmetic.py:5"],
                confidence=0.95,
                artifact_id="a2",
            ),
        ]
        conflicts = detect_contradictory_peers(arts, cwd=REPO_ROOT)
        self.assertEqual(conflicts, [])
        bullets = Stitcher._bullet_payloads(arts, "claim", dedupe=True)
        self.assertEqual(len(bullets), 1)


class RoutingLedgerCriteriaTests(unittest.TestCase):
    def test_analysis_swarm_prompt_preserves_criteria(self) -> None:
        from puppetmaster.swarm_launch import analysis_swarm_prompt, build_analysis_swarm_specs

        goal = (
            "Validate model identity tokens in the routing ledger.\n\n"
            "Acceptance criteria:\n"
            "- ROUTING payload includes model_id\n"
            "- adapter_model_name is present\n"
        )
        prompt = analysis_swarm_prompt(role="audit", goal=goal)
        self.assertIn("Acceptance criteria:", prompt)
        self.assertIn("ROUTING payload includes model_id", prompt)
        specs = build_analysis_swarm_specs(
            goal,
            ["audit"],
            adapter="cursor",
            cwd=str(REPO_ROOT),
            auto_route=False,
            model="composer-2.5",
        )
        self.assertEqual(
            specs[0].payload.get("acceptance_criteria"),
            [
                "ROUTING payload includes model_id",
                "adapter_model_name is present",
            ],
        )


class AdversarialClaimConflictTests(unittest.TestCase):
    def test_same_file_different_line_camelcase_symbols_remain_findings(self) -> None:
        """Path-only loci must not invent source_verification_disagreement."""
        # Opposing source-verification on different lines of one real file must
        # remain ordinary Findings — never a path-only conflict.
        a = _finding(
            "MissingHelper is defined for streaming callbacks",
            evidence=[f"{DELTA_BUS}:19"],
            artifact_id="artifact_missing",
        )
        b = _finding(
            "BroadcastSink alias fans out worker deltas",
            evidence=[f"{DELTA_BUS}:20"],
            artifact_id="artifact_broadcast",
        )
        self.assertEqual(
            source_verify_claim(a.payload["claim"], a.evidence, cwd=REPO_ROOT),
            "contradicts",
        )
        self.assertEqual(
            source_verify_claim(b.payload["claim"], b.evidence, cwd=REPO_ROOT),
            "supports",
        )
        conflicts = detect_contradictory_peers([a, b], cwd=REPO_ROOT)
        self.assertEqual(conflicts, [])
        summary = _render([a, b], cwd=REPO_ROOT)
        findings_section = summary.split("## Findings")[1].split("## Conflicts")[0]
        self.assertIn("MissingHelper", findings_section)
        self.assertIn("BroadcastSink", findings_section)
        conflicts_section = summary.split("## Conflicts")[1].split("## Decisions")[0]
        self.assertIn("- None", conflicts_section)

    def _write_dual_lock_module(self, root: Path) -> str:
        """Temporary module with two legitimate lock vars plus a named subject."""
        rel = "gate_module.py"
        (root / rel).write_text(
            "import asyncio\n"
            "import threading\n"
            "\n"
            "_async_gate = asyncio.Lock()\n"
            "_sync_gate = threading.Lock()\n"
            "_listener_lock = threading.Lock()\n",
            encoding="utf-8",
        )
        return rel

    def test_distinct_lock_types_on_different_lines_remain_findings(self) -> None:
        """threading.Lock and asyncio.Lock on different lines are both Findings."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = self._write_dual_lock_module(root)
            async_finding = _finding(
                "_async_gate is an asyncio.Lock",
                evidence=[f"{rel}:4"],
                artifact_id="artifact_async_gate",
            )
            sync_finding = _finding(
                "_sync_gate is a threading.Lock",
                evidence=[f"{rel}:5"],
                artifact_id="artifact_sync_gate",
            )
            self.assertEqual(
                source_verify_claim(
                    async_finding.payload["claim"], async_finding.evidence, cwd=root
                ),
                "supports",
            )
            self.assertEqual(
                source_verify_claim(
                    sync_finding.payload["claim"], sync_finding.evidence, cwd=root
                ),
                "supports",
            )
            conflicts = detect_contradictory_peers(
                [async_finding, sync_finding], cwd=root
            )
            self.assertEqual(conflicts, [])
            summary = _render([async_finding, sync_finding], cwd=root)
            findings_section = summary.split("## Findings")[1].split("## Conflicts")[0]
            self.assertIn("_async_gate", findings_section)
            self.assertIn("_sync_gate", findings_section)
            conflicts_section = summary.split("## Conflicts")[1].split("## Decisions")[0]
            self.assertIn("- None", conflicts_section)

    def test_same_line_opposing_lock_types_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = self._write_dual_lock_module(root)
            wrong = _finding(
                "_sync_gate is an asyncio.Lock",
                evidence=[f"{rel}:5"],
                artifact_id="artifact_wrong_lock",
            )
            right = _finding(
                "_sync_gate is a threading.Lock",
                evidence=[f"{rel}:5"],
                artifact_id="artifact_right_lock",
            )
            conflicts = detect_contradictory_peers([wrong, right], cwd=root)
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0].reason, "incompatible_lock_types")
            summary = _render([wrong, right], cwd=root)
            self.assertIn("## Conflicts", summary)
            self.assertIn("artifact_wrong_lock", summary)
            self.assertIn("artifact_right_lock", summary)

    def test_same_named_lock_opposing_types_conflict_across_lines(self) -> None:
        """Strong subject identity (_listener_lock) conflicts even across lines."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = self._write_dual_lock_module(root)
            # Weak identity (no shared lock subject): different lines, no conflict.
            weak_a = _finding(
                "async path uses asyncio.Lock",
                evidence=[f"{rel}:4"],
                artifact_id="artifact_weak_async",
            )
            weak_b = _finding(
                "sync path uses threading.Lock",
                evidence=[f"{rel}:5"],
                artifact_id="artifact_weak_thread",
            )
            self.assertEqual(
                detect_contradictory_peers([weak_a, weak_b], cwd=root), []
            )

            # Strong subject: same _listener_lock named with opposing types.
            strong_a = _finding(
                "_listener_lock is an asyncio.Lock",
                evidence=[f"{rel}:4"],
                artifact_id="artifact_listener_async",
            )
            strong_b = _finding(
                "_listener_lock is a threading.Lock",
                evidence=[f"{rel}:6"],
                artifact_id="artifact_listener_thread",
            )
            conflicts = detect_contradictory_peers([strong_a, strong_b], cwd=root)
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0].reason, "incompatible_lock_types")

    def test_ambiguous_basename_stays_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg_a").mkdir()
            (root / "pkg_b").mkdir()
            (root / "pkg_a" / "twin.py").write_text("AlphaWidget = 1\n", encoding="utf-8")
            (root / "pkg_b" / "twin.py").write_text("BetaWidget = 2\n", encoding="utf-8")
            status = source_verify_claim(
                "AlphaWidget is defined here",
                ["twin.py:1"],
                cwd=root,
            )
            self.assertEqual(status, "unknown")

    def test_secret_and_oversized_files_are_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = root / ".env"
            secret.write_text("SECRET_KEY=abc\n", encoding="utf-8")
            status_secret = source_verify_claim(
                "SECRET_KEY is present",
                [".env:1"],
                cwd=root,
            )
            self.assertEqual(status_secret, "unknown")

            huge = root / "blob.py"
            huge.write_bytes(b"x = 1\n" + (b"# pad\n" * 200_000))
            self.assertGreater(huge.stat().st_size, 1_000_000)
            status_huge = source_verify_claim(
                "UnusedHelper lives here",
                ["blob.py:1"],
                cwd=root,
            )
            self.assertEqual(status_huge, "unknown")


class AdversarialAcceptanceCriteriaTests(unittest.TestCase):
    def test_divergent_structured_criteria_replace_explicit_block(self) -> None:
        instruction = (
            "Audit the routing ledger carefully.\n\n"
            "Acceptance criteria:\n"
            "- stale criterion from an older brief\n"
        )
        canonical = [
            "ROUTING artifacts expose model_id",
            "plan-billed marginal cost stays honest",
        ]
        updated = ensure_acceptance_criteria_in_text(instruction, canonical)
        self.assertIn("Audit the routing ledger carefully.", updated)
        self.assertNotIn("stale criterion from an older brief", updated)
        self.assertEqual(parse_acceptance_criteria_block(updated), canonical)

    def test_each_adapter_structured_prompt_path_uses_payload_criteria(self) -> None:
        instruction = (
            "Inspect adapters.\n\n"
            "Acceptance criteria:\n"
            "- instruction-block criterion that must be replaced\n"
        )
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction=instruction,
            payload={
                "acceptance_criteria": [
                    "canonical payload criterion A",
                    "canonical payload criterion B",
                ],
            },
        )
        for final_note in (False, True):
            prompt = structured_prompt_for_task(
                task, prompt=instruction, final_message_note=final_note
            )
            after_task = prompt[prompt.index(TASK_INSTRUCTION_HEADER) :]
            self.assertIn("Inspect adapters.", after_task)
            self.assertIn("canonical payload criterion A", after_task)
            self.assertIn("canonical payload criterion B", after_task)
            self.assertNotIn("instruction-block criterion that must be replaced", after_task)
            self.assertEqual(
                parse_acceptance_criteria_block(after_task),
                [
                    "canonical payload criterion A",
                    "canonical payload criterion B",
                ],
            )


class AdversarialProvenanceTests(unittest.TestCase):
    def test_metered_bare_zero_cost_is_unknown(self) -> None:
        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="analyze",
            adapter="cursor",
            payload={
                "router_model_id": "cursor/grok-4-5",
                "model": "grok-4.5",
                "billing": "metered",
            },
        )
        prov = provenance_from_task_and_payload(
            task,
            {
                "tokens_in": 10,
                "tokens_out": 4,
                "cost_usd": 0.0,
            },
        )
        self.assertNotIn("cost_usd", prov)
        self.assertFalse(prov["cost_known"])
        self.assertFalse(prov["priced"])
        self.assertEqual(prov["cost_source"], "unpriced")

    def test_missing_tokens_estimated_stays_unknown(self) -> None:
        prov = provenance_from_task_and_payload(
            Task(job_id="job_qc", role="audit", instruction="x", payload={}),
            {"tokens_in": 12, "tokens_out": 3},
        )
        self.assertTrue(prov["usage_known"])
        self.assertEqual(prov["tokens_in"], 12)
        self.assertNotIn("tokens_estimated", prov)

    def test_merge_upgrades_plan_stamp_to_real_cost(self) -> None:
        earlier = {
            "adapter": "cursor",
            "usage_known": True,
            "tokens_in": 10,
            "tokens_out": 2,
            "cost_usd": 0.0,
            "cost_known": True,
            "priced": True,
            "cost_source": "plan",
        }
        later = {
            "adapter": "cursor",
            "usage_known": True,
            "tokens_in": 10,
            "tokens_out": 2,
            "cost_usd": 0.42,
            "cost_known": True,
            "priced": True,
            "cost_source": "real_cost_usd",
        }
        merged = _merge_provenance(earlier, later)
        self.assertEqual(merged["cost_source"], "real_cost_usd")
        self.assertEqual(merged["cost_usd"], 0.42)
        self.assertTrue(merged["cost_known"])
        self.assertTrue(merged["priced"])

    def test_verification_provenance_failure_stamps_error(self) -> None:
        from unittest.mock import patch

        task = Task(
            job_id="job_qc",
            role="audit",
            instruction="check\n\nAcceptance criteria:\n- must pass\n",
            adapter="cursor",
            payload={"acceptance_criteria": ["must pass"]},
        )
        with patch(
            "puppetmaster.execution_provenance.stamp_execution_provenance",
            side_effect=RuntimeError("provenance boom"),
        ):
            art = verification_artifact(
                task=task,
                worker_id="w1",
                adapter="cursor",
                check="check",
                result="passed",
                confidence=0.9,
                evidence=["adapter:cursor"],
                payload={},
            )
        prov = art.payload.get(PROVENANCE_KEY)
        self.assertIsInstance(prov, dict)
        self.assertEqual(prov.get("cost_source"), "unknown")
        self.assertFalse(prov.get("cost_known"))
        self.assertFalse(prov.get("usage_known"))
        self.assertEqual(prov.get("provenance_error"), "RuntimeError")
        # Criteria stamping still runs independently.
        self.assertEqual(art.payload["acceptance_criteria"][0]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
