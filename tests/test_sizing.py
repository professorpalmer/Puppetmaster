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

    def test_unit_count_alone_never_fans_out_before_measuring(self):
        self.assertEqual(sizing.decide(plan(16, 0), elapsed_s=0).action, sizing.STAY_SOLO)
        calibrated = dict(sizing.DEFAULT_CALIBRATION, per_unit_s=40.0)
        self.assertEqual(sizing.decide(plan(16, 0), elapsed_s=0, calibration=calibrated).action,
                         sizing.DELEGATE_UPFRONT)
        tiny = dict(sizing.DEFAULT_CALIBRATION, per_unit_s=4.0)
        self.assertEqual(sizing.decide(plan(16, 0), elapsed_s=0, calibration=tiny).action,
                         sizing.STAY_SOLO)

    def test_many_tiny_units_stay_solo_and_substantial_ones_hand_off(self):
        # 16 small modules: solo wrote them in ~70 s (about 4 s each).
        self.assertEqual(sizing.decide(plan(16, 3), elapsed_s=12).action, sizing.STAY_SOLO)
        # 16 voxel regions at ~40 s each: past the quality crossover.
        decision = sizing.decide(plan(16, 2), elapsed_s=80)
        self.assertEqual(decision.action, sizing.HANDOFF)
        self.assertIn("quality crossover", decision.reason)

    def test_projected_time_overrun_hands_off(self):
        # Below the quality crossover (12 units), a projected time overrun still hands off.
        decision = sizing.decide(plan(12, 2), elapsed_s=240)
        self.assertEqual(decision.action, sizing.HANDOFF)
        self.assertGreater(decision.projected_solo_s, 1200)
        self.assertLess(decision.projected_parallel_s, decision.projected_solo_s)
        self.assertIn("10 remaining independent units", sizing.advice(decision))

    def test_solo_that_will_finish_keeps_going(self):
        self.assertEqual(sizing.decide(plan(8, 4), elapsed_s=200).action, sizing.STAY_SOLO)

    def test_context_overrun_hands_off(self):
        decision = sizing.decide(plan(10, 2), elapsed_s=60, context_frac=0.2)
        self.assertEqual(decision.action, sizing.HANDOFF)
        self.assertIn("context", decision.reason)

    def test_late_or_unprofitable_handoff_stays_solo(self):
        self.assertEqual(sizing.decide(plan(10, 8), elapsed_s=2000).action, sizing.STAY_SOLO)
        slow_handoff = dict(sizing.DEFAULT_CALIBRATION, handoff_overhead_s=5000)
        decision = sizing.decide(plan(12, 2), elapsed_s=240, calibration=slow_handoff)
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
            missing = sizing.load_calibration("x", path=Path(tmp) / "missing.json")
            self.assertEqual(missing["solo_horizon_s"], sizing.DEFAULT_CALIBRATION["solo_horizon_s"])
            self.assertEqual(set(missing["provenance"].values()), {"provisional"})

    def test_unmeasured_or_invalid_fields_stay_provisional(self):
        # A falloff study may leave a field unmeasured; null must not crash or read as zero.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            path.write_text(json.dumps({"models": {"gpt-6.1-sol": {
                "solo_horizon_s": 900, "min_independent_units": None, "upfront_units": 0,
                "solo_context_frac": "high", "late_fraction": True}}}))
            cal = sizing.load_calibration("gpt-6.1-sol", path=path)
            self.assertEqual(cal["provenance"]["solo_horizon_s"], "calibrated")
            for name in ("min_independent_units", "upfront_units", "solo_context_frac", "late_fraction"):
                self.assertEqual(cal["provenance"][name], "provisional")
                self.assertEqual(cal[name], sizing.DEFAULT_CALIBRATION[name])
            self.assertIn(sizing.decide(plan(16, 2), elapsed_s=240, calibration=cal).action,
                          (sizing.HANDOFF, sizing.STAY_SOLO))


if __name__ == "__main__":
    unittest.main()
