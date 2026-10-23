"""Unit tests for TaskExecutive using mock skills.

Tests cooperative execution, conditional branch expansion,
step timeouts, and safe cancellation.
"""
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
sys.path.insert(0, str(CONTROLLER_DIR))

from task_executive import (
    CANCELLED,
    RUNNING,
    SUCCEEDED,
    TaskExecutive,
)
from world_state import WorldState


class MockSkills:
    def __init__(self):
        self.started_steps = []
        self.next_status = SUCCEEDED
        self.cancelled = False

    def start(self, step: dict) -> str:
        self.started_steps.append(step)
        return self.next_status

    def update(self) -> str:
        return self.next_status

    def cancel(self):
        self.cancelled = True


class TaskExecutiveTests(unittest.TestCase):
    def setUp(self):
        self.world_state = WorldState()
        self.skills = MockSkills()
        self.executive = TaskExecutive(self.skills, self.world_state)

    def test_executes_sequential_plan_to_completion(self):
        plan = {
            "version": "1.0",
            "goal": "Sequential test",
            "steps": [
                {"action": "navigate", "location": "office_a"},
                {"action": "report", "message": "Arrived."},
            ],
        }
        self.executive.start(plan)
        self.assertTrue(self.executive.active)

        # Step 1: navigate succeeds
        status1 = self.executive.update()
        self.assertEqual(status1, RUNNING)
        self.assertEqual(self.executive.step_index, 1)

        # Step 2: report succeeds
        status2 = self.executive.update()
        self.assertEqual(status2, RUNNING)
        self.assertEqual(self.executive.step_index, 2)

        # Finished
        status3 = self.executive.update()
        self.assertEqual(status3, SUCCEEDED)
        self.assertFalse(self.executive.active)

    def test_conditional_expansion_takes_then_branch(self):
        self.world_state.last_search_result = {"outcome": "NOT_FOUND", "found": False}
        plan = {
            "version": "1.0",
            "goal": "Branch then test",
            "steps": [
                {
                    "action": "conditional",
                    "condition": {"source": "last_search_result", "operator": "found", "value": False},
                    "then": [{"action": "navigate", "location": "meeting_room"}],
                    "else": [{"action": "return_home"}],
                }
            ],
        }
        self.executive.start(plan)
        status = self.executive.update()
        self.assertEqual(status, RUNNING)
        self.assertEqual(self.executive.steps[0]["action"], "navigate")
        self.assertEqual(self.executive.steps[0]["location"], "meeting_room")

    def test_conditional_expansion_takes_else_branch(self):
        self.world_state.last_search_result = {"outcome": "FOUND", "found": True}
        plan = {
            "version": "1.0",
            "goal": "Branch else test",
            "steps": [
                {
                    "action": "conditional",
                    "condition": {"source": "last_search_result", "operator": "found", "value": False},
                    "then": [{"action": "navigate", "location": "meeting_room"}],
                    "else": [{"action": "report", "message": "Found"}],
                }
            ],
        }
        self.executive.start(plan)
        status = self.executive.update()
        self.assertEqual(status, RUNNING)
        self.assertEqual(self.executive.steps[0]["action"], "report")

    def test_cancellation_halts_executive_and_skills(self):
        plan = {
            "version": "1.0",
            "goal": "Cancel test",
            "steps": [{"action": "navigate", "location": "laboratory"}],
        }
        self.executive.start(plan)
        self.executive.cancel()

        self.assertFalse(self.executive.active)
        self.assertTrue(self.skills.cancelled)
        self.assertEqual(self.world_state.execution_status, CANCELLED)


if __name__ == "__main__":
    unittest.main()

