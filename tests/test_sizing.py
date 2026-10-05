"""Sizing gate: solo until the projected falloff, then hand off independent units."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster import sizing  # noqa: E402


def plan(total, done, independent=True):
    return [sizing.Unit(str(i), "done" if i < done else "pending", independent) for i in range(total)]


class SizingDecisionTests(unittest.TestCase):
    def test_small_work_stays_solo(self):
        self.assertEqual(sizing.decide(plan(3, 1), elapsed_s=900).action, sizing.STAY_SOLO)

    def test_coupled_work_never_fans_out(self):
        decision = sizing.decide(plan(16, 2, independent=False), elapsed_s=240)
        self.assertEqual(decision.action, sizing.STAY_SOLO)

    def test_mass_independent_work_fans_out_upfront(self):
        self.assertEqual(sizing.decide(plan(16, 0), elapsed_s=0).action, sizing.DELEGATE_UPFRONT)
        self.assertEqual(sizing.decide(plan(8, 0), elapsed_s=0).action, sizing.STAY_SOLO)

    def test_projected_time_overrun_hands_off(self):
        # Sep 16: 16 coupled-group regions capped solo at 8/16 in 30 minutes.
        decision = sizing.decide(plan(16, 2), elapsed_s=240)
        self.assertEqual(decision.action, sizing.HANDOFF)
        self.assertGreater(decision.projected_solo_s, 1200)
        self.assertLess(decision.projected_parallel_s, decision.projected_solo_s)
        self.assertIn("14 remaining independent units", sizing.advice(decision))

    def test_solo_that_will_finish_keeps_going(self):
        self.assertEqual(sizing.decide(plan(8, 4), elapsed_s=200).action, sizing.STAY_SOLO)

    def test_context_overrun_hands_off(self):
        decision = sizing.decide(plan(10, 2), elapsed_s=60, context_frac=0.2)
        self.assertEqual(decision.action, sizing.HANDOFF)
        self.assertIn("context", decision.reason)

    def test_late_or_unprofitable_handoff_stays_solo(self):
        self.assertEqual(sizing.decide(plan(10, 8), elapsed_s=2000).action, sizing.STAY_SOLO)
        slow_handoff = dict(sizing.DEFAULT_CALIBRATION, handoff_overhead_s=5000)
        decision = sizing.decide(plan(16, 2), elapsed_s=240, calibration=slow_handoff)
        self.assertEqual(decision.action, sizing.STAY_SOLO)
        self.assertIn("would not finish sooner", decision.reason)

    def test_one_handoff_per_turn(self):
        self.assertEqual(sizing.decide(plan(16, 2), elapsed_s=240, handed_off=True).action,
                         sizing.STAY_SOLO)


class SizingInputTests(unittest.TestCase):
    def test_host_plan_shapes(self):
        todo = sizing.parse_plan({"todos": [{"content": "[parallel] region 1", "status": "completed"},
                                            {"content": "wire it up", "status": "pending"}]})
        self.assertEqual([(u.done, u.independent) for u in todo], [(True, True), (False, False)])
        codex = sizing.parse_plan({"plan": [{"step": "[Parallel] a", "status": "in_progress"}]})
        self.assertTrue(codex[0].independent and not codex[0].done)
        self.assertEqual(sizing.parse_plan("nonsense"), [])

    def test_calibration_file_merges_by_normalized_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            path.write_text(json.dumps({"default": {"upfront_units": 20},
                                        "models": {"gpt-6.1-sol": {"solo_horizon_s": 2400}}}))
            cal = sizing.load_calibration("codex/gpt-6-1-sol", path=path)
            self.assertEqual((cal["solo_horizon_s"], cal["upfront_units"]), (2400, 20))
            self.assertEqual(sizing.load_calibration("other", path=path)["solo_horizon_s"],
                             sizing.DEFAULT_CALIBRATION["solo_horizon_s"])
            self.assertEqual(sizing.load_calibration("x", path=Path(tmp) / "missing.json"),
                             sizing.DEFAULT_CALIBRATION)


if __name__ == "__main__":
    unittest.main()
