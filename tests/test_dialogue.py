"""Tests for intent routing, clarification and status explanation."""
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from nlu_dialogue import (  # noqa: E402
    PLANNER_LOCATION_NAMES,
    DialogueManager,
    normalise,
    planner_text,
)


class NormaliseTests(unittest.TestCase):
    def test_maps_location_phrases_to_canonical_tokens(self):
        self.assertEqual(normalise("Go to Office B!"), "go to office_b")
        self.assertEqual(normalise("the main corridor"), "the main_corridor")
        self.assertEqual(normalise("the Lab"), "the laboratory")
        self.assertEqual(normalise("conference room"), "meeting_room")

    def test_handles_speech_recognition_spellings(self):
        self.assertEqual(normalise("office bee"), "office_b")
        # Mishearings recorded with Vosk in our own trials.
        self.assertEqual(normalise("go to officer"), "go to office_a")
        self.assertEqual(normalise("go to have his beer"), "go to office_b")
        self.assertEqual(normalise("of his be"), "office_b")


class PlannerTextTests(unittest.TestCase):
    def test_rewrites_mishearings_and_nicknames(self):
        cases = {
            "go to officer": "go to Office A",
            "go to have his beer": "go to Office B",
            "find the read ball in the lab": "find the red ball in the laboratory",
            "find the yellow books": "find the yellow box",
            "take me to the lobby": "take me to the reception",
            "go to the conference room": "go to the meeting room",
            "go to the meeting": "go to the meeting room",
            "Go to the Meeting Room": "Go to the meeting room",
        }
        for heard, expected in cases.items():
            self.assertEqual(planner_text(heard), expected, heard)

    def test_keeps_punctuation_and_order(self):
        text = ("Look in the lab. If the red ball is not there, "
                "check the meeting room.")
        self.assertEqual(
            planner_text(text),
            "Look in the laboratory. If the red ball is not there, "
            "check the meeting room.",
        )

    def test_leaves_other_words_alone(self):
        for text in ("Go to reception", "Fly through the wall to Office B.",
                     "go back home"):
            self.assertEqual(planner_text(text), text)

    def test_planner_names_are_known_to_location_grounding(self):
        from location_grounding import ground_locations
        for location_id, name in PLANNER_LOCATION_NAMES.items():
            plan = {"steps": [{"action": "navigate", "location": name}]}
            ground_locations(plan)
            self.assertEqual(plan["steps"][0]["location"], location_id, name)

    def test_complete_goal_is_cleaned_for_planner(self):
        dialogue = DialogueManager()
        result = dialogue.analyse_for_missing_information(
            "find the read ball in officer")
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["goal"], "find the red ball in Office A")


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = DialogueManager()

    def test_stop_phrases(self):
        for text in ("stop", "Cancel that", "please halt", "abort the task",
                     "never mind"):
            self.assertEqual(self.dialogue.classify(text), "stop", text)

    def test_status_phrases(self):
        for text in ("What are you doing and why?", "status",
                     "Where are you?", "what's going on", "explain yourself"):
            self.assertEqual(self.dialogue.classify(text), "status", text)

    def test_goal_phrases(self):
        for text in ("Go to the laboratory.", "Find the red ball",
                     "Fly through the wall to Office B."):
            self.assertEqual(self.dialogue.classify(text), "goal", text)

    def test_background_talk_is_unknown(self):
        # Real background speech recognised during a trial run.
        for text in ("consider for it", "his name a hobby",
                     "ross them amanda spoken to either", "hello"):
            self.assertEqual(self.dialogue.classify(text), "unknown", text)

    def test_noise_is_ignored(self):
        for text in ("", "   ", "the", "uh", "huh?"):
            self.assertEqual(self.dialogue.classify(text), "noise", repr(text))

    def test_answer_only_while_waiting_for_clarification(self):
        self.assertEqual(self.dialogue.classify("red"), "goal")
        self.dialogue.analyse_for_missing_information("Find the ball")
        self.assertEqual(self.dialogue.classify("red"), "answer")
        self.assertEqual(self.dialogue.classify("the red one"), "answer")

    def test_bare_letter_answer_is_not_filler(self):
        # Trial run: "A" was heard as 'a' (open) and 'ay' (expected words).
        self.dialogue.analyse_for_missing_information("Go to the office")
        self.assertEqual(self.dialogue.classify("a", answer="ay"), "answer")
        self.assertEqual(self.dialogue.classify("stop", answer="a"), "stop")
        self.assertEqual(self.dialogue.classify("a"), "noise")

    def test_new_goal_replaces_pending_clarification(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        self.assertEqual(self.dialogue.classify("go to reception"), "goal")

    def test_stop_wins_while_waiting_for_clarification(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        self.assertEqual(self.dialogue.classify("cancel"), "stop")


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = DialogueManager()

    def assertComplete(self, text):
        result = self.dialogue.analyse_for_missing_information(text)
        self.assertFalse(result["needs_clarification"], (text, result))
        self.assertFalse(self.dialogue.state.awaiting_clarification)
        return result

    def assertAsks(self, text, topic):
        result = self.dialogue.analyse_for_missing_information(text)
        self.assertTrue(result["needs_clarification"], text)
        self.assertTrue(result["question"])
        self.assertEqual(self.dialogue.state.clarification_topic, topic, text)
        self.assertEqual(self.dialogue.state.pending_goal, text)
        return result

    def test_complete_goals_from_specification(self):
        for text in (
            "Go to the laboratory.",
            "Go to the laboratory, then return to reception.",
            "Find the red ball.",
            "Look in the lab. If the red ball is not there, check the "
            "meeting room.",
            "Fly through the wall to Office B.",
            "Return home",
        ):
            self.assertComplete(text)

    def test_misheard_objects(self):
        # "find the yellow box" was recognised as "find the yellow books".
        self.assertComplete("find the yellow books")
        self.assertAsks("find the ceiling", "colour")

    def test_missing_colour(self):
        result = self.assertAsks("Find the ball.", "colour")
        self.assertIn("colour", result["question"])

    def test_synonym_object_missing_colour(self):
        self.assertAsks("Can you locate the sphere?", "colour")

    def test_unsupported_colour(self):
        result = self.assertAsks("Find the purple ball", "colour")
        self.assertIn("red", result["question"])

    def test_missing_object(self):
        result = self.assertAsks("Find something red", "object")
        self.assertIn("red", result["question"])
        self.assertAsks("Search the meeting room", "object")

    def test_ambiguous_office(self):
        self.assertAsks("Go to the office", "office")

    def test_missing_location(self):
        self.assertAsks("Go over there", "location")

    def test_context_resolves_ball_after_earlier_mention(self):
        self.assertComplete("Find the red ball in the lab")
        result = self.assertComplete("Now find the ball")
        self.assertIn("red ball", result["goal"])


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = DialogueManager()

    def test_colour_answer_is_merged_and_state_cleared(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        result = self.dialogue.consume_followup("The red one, please")
        self.assertEqual(result["goal"], "Find the ball (red)")
        self.assertIsNone(result["question"])
        self.assertFalse(self.dialogue.state.awaiting_clarification)
        self.assertIsNone(self.dialogue.state.pending_goal)
        # The merged goal is now complete.
        again = self.dialogue.analyse_for_missing_information(result["goal"])
        self.assertFalse(again["needs_clarification"])

    def test_unsupported_colour_is_replaced(self):
        self.dialogue.analyse_for_missing_information("Find the purple ball")
        result = self.dialogue.consume_followup("red")
        self.assertEqual(result["goal"], "Find the ball (red)")

    def test_object_answer(self):
        self.dialogue.analyse_for_missing_information("Find something red")
        result = self.dialogue.consume_followup("a ball")
        self.assertEqual(result["goal"], "Find something red (ball)")

    def test_expected_word_answer_is_preferred(self):
        self.dialogue.analyse_for_missing_information("Go to the office")
        result = self.dialogue.consume_followup("of his", answer="office b")
        self.assertEqual(result["goal"], "Go to the office (Office B)")

    def test_open_text_used_when_answer_is_empty(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        result = self.dialogue.consume_followup("green", answer="")
        self.assertEqual(result["goal"], "Find the ball (green)")

    def test_misheard_colour_answer(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        result = self.dialogue.consume_followup("read")
        self.assertEqual(result["goal"], "Find the ball (red)")

    def test_office_answer_variants(self):
        for answer, expected in (("office b", "Office B"), ("bear", "Office B"),
                                 ("of his be", "Office B"),
                                 ("the second one", "Office B"),
                                 ("officer", "Office A"), ("a", "Office A"),
                                 ("the first one", "Office A"),
                                 ("office b a please", "Office B")):
            self.dialogue.analyse_for_missing_information("Go to the office")
            result = self.dialogue.consume_followup(answer)
            self.assertEqual(result["goal"], f"Go to the office ({expected})",
                             answer)

    def test_expected_words_follow_the_question(self):
        self.assertIsNone(self.dialogue.expected_words())
        self.dialogue.analyse_for_missing_information("Find the ball")
        words = self.dialogue.expected_words()
        self.assertIn("red", words)
        self.assertIn("stop", words)
        self.assertIn("[unk]", words)
        self.dialogue.analyse_for_missing_information("Go to the office")
        words = self.dialogue.expected_words()
        self.assertIn("b", words)
        self.assertNotIn("the", words)
        self.dialogue.consume_followup("b")
        self.assertIsNone(self.dialogue.expected_words())

    def test_bare_office_letter(self):
        self.dialogue.analyse_for_missing_information("Go to the office")
        result = self.dialogue.consume_followup("B")
        self.assertEqual(result["goal"], "Go to the office (Office B)")
        again = self.dialogue.analyse_for_missing_information(result["goal"])
        self.assertFalse(again["needs_clarification"])

    def test_location_answer(self):
        self.dialogue.analyse_for_missing_information("Go over there")
        result = self.dialogue.consume_followup("the lab")
        self.assertEqual(result["goal"], "Go over there (the laboratory)")

    def test_unclear_answer_reasks_then_gives_up(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        first = self.dialogue.consume_followup("banana")
        self.assertIsNone(first["goal"])
        self.assertIn("Which colour", first["question"])
        self.assertTrue(self.dialogue.state.awaiting_clarification)
        second = self.dialogue.consume_followup("banana")
        self.assertIsNone(second["goal"])
        self.assertIn("whole task", second["question"])
        self.assertFalse(self.dialogue.state.awaiting_clarification)

    def test_without_pending_goal_text_passes_through(self):
        result = self.dialogue.consume_followup("Go to reception")
        self.assertEqual(result["goal"], "Go to reception")

    def test_reset_clears_pending(self):
        self.dialogue.analyse_for_missing_information("Find the ball")
        self.dialogue.reset()
        self.assertFalse(self.dialogue.state.awaiting_clarification)


class PlanQuestionTests(unittest.TestCase):
    """Replies to a question asked by a plan's "ask" step."""

    QUESTION = ("I cannot jump or teleport. "
                "Should I drive to the meeting room instead?")

    def setUp(self):
        self.dialogue = DialogueManager()
        self.dialogue.expect_plan_answer(
            "Fly through the wall to Office B", self.QUESTION)

    def test_any_reply_is_an_answer(self):
        for text in ("yes", "yes please", "no thanks", "sure",
                     "okay do that", "go to the lab instead"):
            self.assertEqual(self.dialogue.classify(text), "answer", text)

    def test_stop_and_status_still_work(self):
        self.assertEqual(self.dialogue.classify("stop"), "stop")
        self.assertEqual(self.dialogue.classify("what are you doing"),
                         "status")
        self.assertEqual(self.dialogue.classify("uh"), "noise")

    def test_reply_becomes_request_with_context(self):
        result = self.dialogue.consume_followup("yes please")
        self.assertTrue(result["ready"])
        self.assertEqual(
            result["goal"],
            "Earlier request: Fly through the wall to Office B. "
            f"I asked: {self.QUESTION} The user answered: yes please.",
        )
        self.assertFalse(self.dialogue.state.awaiting_clarification)

    def test_short_yes_no_transcript_is_preferred(self):
        # Open recognition misheard "yes"; the yes/no recogniser did not.
        result = self.dialogue.consume_followup("yet", answer="yes")
        self.assertIn("The user answered: yes.", result["goal"])

    def test_open_transcript_kept_for_task_replies(self):
        result = self.dialogue.consume_followup(
            "no go to the lab instead", answer="no")
        self.assertIn("The user answered: no go to the laboratory instead.",
                      result["goal"])

    def test_empty_reply_reasks(self):
        result = self.dialogue.consume_followup("", answer="")
        self.assertIsNone(result["goal"])
        self.assertIn("Should I drive", result["question"])

    def test_expected_words_include_yes_no_and_stop(self):
        words = self.dialogue.expected_words()
        for word in ("yes", "no", "stop", "[unk]"):
            self.assertIn(word, words)

    def test_followup_goal_needs_no_clarification(self):
        goal = self.dialogue.consume_followup("yes")["goal"]
        result = self.dialogue.analyse_for_missing_information(goal)
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["goal"], goal)


class StatusTests(unittest.TestCase):
    def world(self, **overrides):
        state = {
            "current_location": "reception",
            "current_goal": None,
            "current_step": None,
            "last_search_result": None,
            "blocked_routes": [],
            "notes": [],
            "replan_count": 0,
            "execution_status": "IDLE",
        }
        state.update(overrides)
        return state

    def test_idle(self):
        text = DialogueManager.describe_status(self.world())
        self.assertIn("not doing anything", text)
        self.assertIn("reception", text)

    def test_planning(self):
        text = DialogueManager.describe_status(
            self.world(current_goal="Find the red ball")
        )
        self.assertIn("plan", text)
        self.assertIn("Find the red ball", text)

    def test_running_explains_step_and_reason(self):
        text = DialogueManager.describe_status(self.world(
            execution_status="RUNNING",
            current_goal="Find the red ball",
            current_step={"action": "search", "object": "ball",
                          "colour": "red"},
            current_location="laboratory",
            last_search_result={"result": "NOT_FOUND", "confidence": 0.8},
            replan_count=1,
        ))
        self.assertIn("searching for the red ball", text)
        self.assertIn("because you asked me to: Find the red ball", text)
        self.assertIn("the laboratory", text)
        self.assertIn("did not find the target", text)
        self.assertIn("80%", text)
        self.assertIn("revised my plan once", text)

    def test_running_navigation(self):
        text = DialogueManager.describe_status(self.world(
            execution_status="RUNNING",
            current_goal="Go to Office A",
            current_step={"action": "navigate", "location": "office_a"},
        ))
        self.assertIn("driving to Office A", text)

    def test_waiting_for_answer_to_plan_question(self):
        text = DialogueManager.describe_status(self.world(
            execution_status="SUCCEEDED", current_goal="Go to the roof",
            pending_question="Should I go to reception instead?"))
        self.assertEqual(
            text, "I asked you: Should I go to reception instead? "
                  "I am waiting for your answer.")

    def test_planning_and_replanning(self):
        planning = DialogueManager.describe_status(self.world(
            execution_status="PLANNING", current_goal="Go to Office B"))
        self.assertEqual(planning, "I am working out a plan for: Go to Office B.")
        replanning = DialogueManager.describe_status(self.world(
            execution_status="PLANNING", current_goal="Find the red ball",
            replan_count=1))
        self.assertIn("revising my plan for: Find the red ball", replanning)
        self.assertNotIn("could not complete", replanning)

    def test_failed_and_cancelled(self):
        failed = DialogueManager.describe_status(
            self.world(execution_status="FAILED", current_goal="Go to the lab",
                       blocked_routes=["centre-laboratory_entry"])
        )
        self.assertIn("could not complete", failed)
        self.assertIn("blocked", failed)
        cancelled = DialogueManager.describe_status(
            self.world(execution_status="CANCELLED", current_goal="Go to the lab")
        )
        self.assertIn("cancelled", cancelled)


if __name__ == "__main__":
    unittest.main()
