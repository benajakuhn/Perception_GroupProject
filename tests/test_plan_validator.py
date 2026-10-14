"""Baseline tests for the deterministic plan boundary.

Students should extend this file with malformed model outputs, unsafe plans,
conditional branches, maximum-size plans and project-specific semantic rules.
"""
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from plan_validator import PlanValidator  # noqa: E402


class PlanValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator()

    def test_accepts_valid_plan(self):
        plan = {
            "version": "1.0",
            "goal": "Find the red ball in the laboratory and report.",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {"action": "search", "object": "ball", "colour": "red"},
                {"action": "report", "message": "Search complete."},
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertTrue(valid, errors)

    def test_rejects_unknown_action(self):
        plan = {
            "version": "1.0",
            "goal": "Do something unsafe.",
            "steps": [{"action": "run_python", "code": "print('unsafe')"}],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("unsupported action" in error for error in errors))

    def test_rejects_invalid_location(self):
        plan = {
            "version": "1.0",
            "goal": "Go outside.",
            "steps": [{"action": "navigate", "location": "car_park"}],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("invalid location" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
