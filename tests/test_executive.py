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

    def test_stop_step_ends_the_task(self):
        # A stop chosen by a conditional branch ends the whole plan.
        self.executive.start({"version": "1.0", "goal": "g", "steps": [
            {"action": "conditional",
             "condition": {"source": "route_status", "operator": "equals", "value": "CLEAR"},
             "then": [{"action": "stop"}], "else": []},
            {"action": "navigate", "location": "office_a"},
        ]})
        self.world_state.route_status = "CLEAR"
        status = RUNNING
        while status == RUNNING:
            status = self.executive.update()
        self.assertEqual(status, SUCCEEDED)
        self.assertEqual([s["action"] for s in self.skills.started_steps], ["stop"])

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



class FakeSpeech:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class AskStepTests(unittest.TestCase):
    """An ask step records its question so the main loop can wait for a reply."""

    def setUp(self):
        from robot_skills import RobotSkills
        self.world_state = WorldState()
        self.speech = FakeSpeech()
        self.skills = RobotSkills(None, None, None, self.speech, self.world_state)

    def test_ask_speaks_and_records_question(self):
        question = "Should I drive to the meeting room instead?"
        status = self.skills.start({"action": "ask", "question": question})
        self.assertEqual(status, SUCCEEDED)
        self.assertEqual(self.speech.spoken, [question])
        self.assertEqual(self.world_state.pending_question, question)
        self.assertEqual(self.world_state.as_dict()["pending_question"], question)

    def test_new_plan_clears_old_question(self):
        self.world_state.pending_question = "Old question?"
        executive = TaskExecutive(MockSkills(), self.world_state)
        executive.start({"version": "1.0", "goal": "New task", "steps": []})
        self.assertIsNone(self.world_state.pending_question)



class ReportTests(unittest.TestCase):
    """A report after searches states what the camera actually found."""

    class FakePerception:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def start_search(self, object_type, colour):
            pass

        def update_search(self):
            outcome = self.outcomes.pop(0)
            return "SUCCEEDED", {"outcome": outcome, "found": outcome == "FOUND",
                                 "detection": None}

    def run_plan(self, outcomes, rooms):
        from robot_skills import RobotSkills
        from semantic_memory import SemanticMemory
        world_state = WorldState()
        speech = FakeSpeech()
        skills = RobotSkills(None, self.FakePerception(outcomes), SemanticMemory(),
                             speech, world_state)
        executive = TaskExecutive(skills, world_state)
        steps = []
        for room in rooms:
            steps.append({"action": "search", "object": "ball", "colour": "red",
                          "_room": room})
        steps.append({"action": "report",
                      "message": "Tell the user whether the red ball was found."})
        executive.start({"steps": steps})
        while executive.active:
            step = executive.steps[executive.step_index] if executive.step_index < len(executive.steps) else {}
            world_state.current_location = step.get("_room", world_state.current_location)
            executive.update()
        return speech.spoken[-1]

    def test_found(self):
        said = self.run_plan(["NOT_FOUND", "FOUND"], ["laboratory", "meeting_room"])
        self.assertEqual(said, "I found the red ball in the meeting room.")

    def test_not_found_is_not_claimed_absent(self):
        said = self.run_plan(["NOT_FOUND", "NOT_FOUND"], ["laboratory", "office_a"])
        self.assertEqual(said, "I searched the laboratory and Office A and did not "
                               "find the red ball there.")

    def test_uncertain(self):
        said = self.run_plan(["UNCERTAIN"], ["office_b"])
        self.assertEqual(said, "I may have seen the red ball in Office B, but I am not sure.")

    def test_unreachable_places_are_added_after_a_search(self):
        """A replan's report: the search summary plus where navigation could not go."""
        from robot_skills import RobotSkills
        world_state = WorldState()
        world_state.search_results = [
            {"target": "red ball", "location": "meeting_room", "result": "NOT_FOUND"}]
        world_state.blocked_routes = ["reception"]
        speech = FakeSpeech()
        RobotSkills(None, None, None, speech, world_state).start(
            {"action": "report", "message": "The way to the front desk is blocked."})
        self.assertEqual(speech.spoken, [
            "I searched the meeting room and did not find the red ball there. "
            "I could not reach reception."])

    def test_several_unreachable_places_are_joined(self):
        from robot_skills import RobotSkills
        world_state = WorldState()
        world_state.blocked_routes = ["laboratory", "office_a", "laboratory"]
        skills = RobotSkills(None, None, None, FakeSpeech(), world_state)
        self.assertEqual(skills.unreachable_summary(),
                         "I could not reach the laboratory or Office A.")

    def test_report_without_search_keeps_the_plan_message_even_if_blocked(self):
        """No search ran: the (replan's) message is all there is to say."""
        from robot_skills import RobotSkills
        world_state = WorldState()
        world_state.blocked_routes = ["laboratory"]
        speech = FakeSpeech()
        RobotSkills(None, None, None, speech, world_state).start(
            {"action": "report", "message": "I could not get to the laboratory."})
        self.assertEqual(speech.spoken, ["I could not get to the laboratory."])

    def test_report_without_search_uses_plan_message(self):
        world_state = WorldState()
        speech = FakeSpeech()
        from robot_skills import RobotSkills
        RobotSkills(None, None, None, speech, world_state).start(
            {"action": "report", "message": "I cannot fly."})
        self.assertEqual(speech.spoken, ["I cannot fly."])


class SearchViewTests(unittest.TestCase):
    """A room is searched from its stop point, then from its extra views."""

    class FakeNavigator:
        graph = {"search_views": {"meeting_room": ["meeting_room_view"]}}

        def __init__(self, result="SUCCEEDED"):
            self.result = result
            self.started = []

        def start(self, location):
            self.started.append(location)
            return "RUNNING"

        def update(self):
            return self.result

        def cancel(self):
            pass

    def search(self, outcomes, navigator):
        from robot_skills import RobotSkills
        from semantic_memory import SemanticMemory
        world_state = WorldState()
        world_state.current_location = "meeting_room"
        perception = ReportTests.FakePerception(outcomes)
        skills = RobotSkills(navigator, perception, SemanticMemory(), FakeSpeech(), world_state)
        status = skills.start({"action": "search", "object": "ball", "colour": "red"})
        for _ in range(10):
            if status != "RUNNING":
                break
            status = skills.update()
        return status, skills.last_evidence, perception

    def test_ball_hidden_from_stop_point_found_from_view(self):
        # The meeting table hides the room's south side from the stop point.
        navigator = self.FakeNavigator()
        status, evidence, _ = self.search(["NOT_FOUND", "FOUND"], navigator)
        self.assertEqual((status, evidence["result"]), ("SUCCEEDED", "FOUND"))
        self.assertEqual(navigator.started, ["meeting_room_view"])
        self.assertEqual(evidence["views_done"], 2)
        self.assertEqual(evidence["location"], "meeting_room")

    def test_found_at_stop_point_skips_the_view(self):
        navigator = self.FakeNavigator()
        status, evidence, _ = self.search(["FOUND"], navigator)
        self.assertEqual(evidence["result"], "FOUND")
        self.assertEqual(navigator.started, [])

    def test_best_evidence_kept_over_views(self):
        status, evidence, _ = self.search(["UNCERTAIN", "NOT_FOUND"], self.FakeNavigator())
        self.assertEqual(evidence["result"], "UNCERTAIN")

    def test_view_unreachable_reports_what_was_seen(self):
        status, evidence, perception = self.search(["NOT_FOUND", "FOUND"],
                                                   self.FakeNavigator("BLOCKED"))
        self.assertEqual((status, evidence["result"]), ("SUCCEEDED", "NOT_FOUND"))
        self.assertEqual(perception.outcomes, ["FOUND"])   # second view never ran


if __name__ == "__main__":
    unittest.main()
