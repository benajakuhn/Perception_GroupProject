"""Baseline and comprehensive tests for the deterministic plan boundary.

Tests schema adherence, argument types, semantic constraints, and security rules.
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

    # --- Baseline tests from starter package (preserved) ---

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

    # --- Comprehensive extensions ---

    def test_accepts_valid_conditional_plan(self):
        plan = {
            "version": "1.0",
            "goal": "Look in lab; if absent check meeting room.",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {"action": "search", "object": "ball", "colour": "red"},
                {
                    "action": "conditional",
                    "condition": {
                        "source": "last_search_result",
                        "operator": "found",
                        "value": False,
                    },
                    "then": [
                        {"action": "navigate", "location": "meeting_room"},
                        {"action": "search", "object": "ball", "colour": "red"},
                    ],
                    "else": [],
                },
                {"action": "report", "message": "Done."},
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertTrue(valid, errors)

    def test_rejects_unknown_top_level_keys(self):
        plan = {
            "version": "1.0",
            "goal": "Test goal",
            "steps": [{"action": "return_home"}],
            "extra_key": True,
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("unknown top-level keys" in e for e in errors))

    def test_rejects_unknown_step_keys(self):
        plan = {
            "version": "1.0",
            "goal": "Drive fast",
            "steps": [{"action": "navigate", "location": "office_a", "speed": 10.0}],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("unknown keys" in e for e in errors))

    def test_rejects_missing_else_in_conditional(self):
        plan = {
            "version": "1.0",
            "goal": "Incomplete conditional",
            "steps": [
                {"action": "navigate", "location": "office_a"},
                {"action": "search", "object": "ball"},
                {
                    "action": "conditional",
                    "condition": {"source": "last_search_result", "operator": "found", "value": True},
                    "then": [{"action": "return_home"}],
                },
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("missing required branch 'else'" in e for e in errors))

    def test_rejects_invalid_condition_value_type(self):
        plan = {
            "version": "1.0",
            "goal": "Bad value type",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {"action": "search", "object": "ball"},
                {
                    "action": "conditional",
                    "condition": {"source": "last_search_result", "operator": "found", "value": "true"},
                    "then": [],
                    "else": [],
                },
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("requires a boolean value" in e for e in errors))

    def test_rejects_search_conditional_before_search(self):
        plan = {
            "version": "1.0",
            "goal": "Premature condition check",
            "steps": [
                {"action": "navigate", "location": "laboratory"},
                {
                    "action": "conditional",
                    "condition": {"source": "last_search_result", "operator": "found", "value": False},
                    "then": [{"action": "return_home"}],
                    "else": [],
                },
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("before any search was performed" in e for e in errors))

    def test_rejects_redundant_repeated_navigation(self):
        plan = {
            "version": "1.0",
            "goal": "Go to office a twice",
            "steps": [
                {"action": "navigate", "location": "office_a"},
                {"action": "navigate", "location": "office_a"},
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("redundant consecutive navigation" in e for e in errors))

    def test_rejects_prohibited_code_injection(self):
        plan = {
            "version": "1.0",
            "goal": "Execute code",
            "steps": [
                {"action": "report", "message": "Trying eval('1+1')"},
            ],
        }
        valid, errors = self.validator.validate(plan)
        self.assertFalse(valid)
        self.assertTrue(any("prohibited content 'eval('" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
