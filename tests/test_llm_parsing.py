"""Tests for the LLM layer that do not need a real model.

plan_parser: turning raw model text into a plan dict.
prompt_builder: filling the planning prompt template.

All model outputs below are hand-written test fixtures, not genuine model
output, and say nothing about real model performance.
"""
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "controllers" / "project3_student"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "planning_prompt_template.txt"
sys.path.insert(0, str(CONTROLLER_DIR))

from plan_parser import MAX_RAW_TEXT_LENGTH, parse_plan_text  # noqa: E402
from plan_schema import (  # noqa: E402
    ALLOWED_ACTIONS,
    ALLOWED_CONDITION_OPERATORS,
    ALLOWED_CONDITION_SOURCES,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
)
from plan_validator import PlanValidator  # noqa: E402
from prompt_builder import (  # noqa: E402
    MAX_ERRORS_SHOWN,
    MAX_GOAL_LENGTH,
    MAX_TRACE_ENTRIES_SHOWN,
    REPLAN_EXAMPLE,
    REPLAN_EXAMPLE_ONE_ROOM,
    build_correction_prompt,
    build_plan_prompt,
    build_replan_prompt,
    replan_situation,
)
from world_state import WorldState  # noqa: E402

UNFILLED_PLACEHOLDER = re.compile(r"\{[A-Z_]+\}")


def show(title, **values):
    """Print a labelled block so a test result can be checked by eye.

    Add -b to the unittest command to hide these prints for passing tests.
    """
    print(f"\n    --- {title} ---")
    for label, value in values.items():
        indented_value = str(value).replace("\n", "\n        ")
        print(f"    {label}: {indented_value}")


def parse_and_show(raw_text):
    plan, errors, repairs = parse_plan_text(raw_text)
    show("parse_plan_text", input=repr(raw_text)[:120], plan=plan,
         errors=errors, repairs=repairs)
    return plan, errors, repairs


def end_of_prompt(prompt):
    """The part of the prompt that changes per call: state, revision and goal."""
    return prompt[prompt.index("Current world/task state:"):]

VALID_PLAN_TEXT = (
    '{"version": "1.0", "goal": "go to the laboratory", '
    '"steps": [{"action": "navigate", "location": "laboratory"}]}'
)
VALID_PLAN = {
    "version": "1.0",
    "goal": "go to the laboratory",
    "steps": [{"action": "navigate", "location": "laboratory"}],
}


class PlanParserSuccessTests(unittest.TestCase):
    def test_clean_json_needs_no_repairs(self):
        plan, errors, repairs = parse_and_show(VALID_PLAN_TEXT)
        self.assertEqual(plan, VALID_PLAN)
        self.assertEqual(errors, [])
        self.assertEqual(repairs, [])

    def test_removes_markdown_fence(self):
        raw_text = "```json\n" + VALID_PLAN_TEXT + "\n```"
        plan, errors, repairs = parse_and_show(raw_text)
        self.assertEqual(plan, VALID_PLAN)
        self.assertEqual(errors, [])
        self.assertIn("Removed Markdown code fence.", repairs)

    def test_removes_fence_without_language_tag(self):
        raw_text = "```\n" + VALID_PLAN_TEXT + "\n```"
        plan, errors, _ = parse_and_show(raw_text)
        self.assertEqual(plan, VALID_PLAN)
        self.assertEqual(errors, [])

    def test_extracts_object_from_surrounding_text(self):
        raw_text = "Here is your plan: " + VALID_PLAN_TEXT + " Hope this helps!"
        plan, errors, repairs = parse_and_show(raw_text)
        self.assertEqual(plan, VALID_PLAN)
        self.assertEqual(errors, [])
        self.assertIn("Extracted JSON object from surrounding text.", repairs)

    def test_ignores_braces_and_escaped_quotes_inside_strings(self):
        raw_text = (
            'Plan: {"version": "1.0", "goal": "found {red} \\"ball\\"", '
            '"steps": []} done'
        )
        plan, errors, _ = parse_and_show(raw_text)
        self.assertEqual(errors, [])
        self.assertEqual(plan["goal"], 'found {red} "ball"')

    def test_removes_trailing_commas(self):
        raw_text = '{"version": "1.0", "goal": "stop", "steps": [{"action": "stop"},],}'
        plan, errors, repairs = parse_and_show(raw_text)
        self.assertEqual(errors, [])
        self.assertEqual(plan["steps"], [{"action": "stop"}])
        self.assertIn("Removed trailing comma.", repairs)

    def test_replaces_curly_quotes(self):
        raw_text = "{“version”: “1.0”, “goal”: “stop”, “steps”: []}"
        plan, errors, repairs = parse_and_show(raw_text)
        self.assertEqual(errors, [])
        self.assertEqual(plan["version"], "1.0")
        self.assertIn("Replaced curly quotes with straight quotes.", repairs)


class PlanParserFailureTests(unittest.TestCase):
    def assert_rejected(self, raw_text):
        plan, errors, _ = parse_and_show(raw_text)
        self.assertIsNone(plan)
        self.assertEqual(len(errors), 1)
        return errors[0]

    def test_rejects_empty_text(self):
        self.assertIn("empty", self.assert_rejected(""))
        self.assertIn("empty", self.assert_rejected("   \n  "))

    def test_rejects_non_string_input(self):
        self.assertIn("empty", self.assert_rejected(None))

    def test_rejects_text_that_is_too_long(self):
        too_long_text = "x" * (MAX_RAW_TEXT_LENGTH + 1)
        self.assertIn("exceeds", self.assert_rejected(too_long_text))

    def test_rejects_text_without_json(self):
        error = self.assert_rejected("I am sorry, I cannot help with that.")
        self.assertIn("No complete JSON object", error)

    def test_rejects_unclosed_json(self):
        error = self.assert_rejected('{"version": "1.0", "steps": [{"action": "stop"}')
        self.assertIn("No complete JSON object", error)

    def test_reports_json_syntax_error(self):
        error = self.assert_rejected('{"version": "1.0" "goal": "missing comma"}')
        self.assertIn("JSON syntax error", error)

    def test_rejects_top_level_list(self):
        error = self.assert_rejected('[{"action": "stop"}]')
        self.assertIn("got list", error)


class PlanParserLeavesContentAloneTests(unittest.TestCase):
    """The parser fixes syntax only; checking content is PlanValidator's job."""

    def test_keeps_unknown_action_and_missing_fields(self):
        raw_text = '{"steps": [{"action": "fly", "location": "lab"}]}'
        plan, errors, repairs = parse_and_show(raw_text)
        self.assertEqual(plan, {"steps": [{"action": "fly", "location": "lab"}]})
        self.assertEqual(errors, [])
        self.assertEqual(repairs, [])

    def test_keeps_unknown_extra_keys(self):
        raw_text = '{"version": "1.0", "goal": "x", "steps": [], "shell": "rm -rf /"}'
        plan, errors, _ = parse_and_show(raw_text)
        self.assertEqual(errors, [])
        self.assertEqual(plan["shell"], "rm -rf /")


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self.world_state = WorldState().as_dict()

    def build_all_three_prompts(self):
        return [
            build_plan_prompt(self.template, "find the red ball", self.world_state),
            build_correction_prompt(self.template, "find the red ball",
                                    self.world_state, "not json", ["No JSON found."]),
            build_replan_prompt(replan_situation(
                "find the red ball", self.world_state,
                [{"step": {"action": "stop"}, "status": "FAILED"}], [])),
        ]

    def test_every_placeholder_is_filled(self):
        for prompt in self.build_all_three_prompts():
            self.assertIsNone(UNFILLED_PLACEHOLDER.search(prompt))

    def test_prompt_lists_every_schema_value(self):
        prompt = build_plan_prompt(self.template, "go", self.world_state)
        all_schema_values = (
            VALID_LOCATIONS | ALLOWED_ACTIONS | SUPPORTED_OBJECTS | SUPPORTED_COLOURS
            | ALLOWED_CONDITION_SOURCES | ALLOWED_CONDITION_OPERATORS
        )
        for schema_value in all_schema_values:
            self.assertIn(schema_value, prompt)
        schema_part = prompt[prompt.index("Allowed actions"):prompt.index("Planning rules:")]
        show("schema part of the prompt", checked_values=sorted(all_schema_values),
             prompt_excerpt="\n" + schema_part)

    def test_prompt_contains_goal_and_world_state(self):
        prompt = build_plan_prompt(self.template, "go to the lab", self.world_state)
        self.assertIn("go to the lab", prompt)
        self.assertIn('"current_location": "reception"', prompt)
        show("first plan prompt (end)", prompt_excerpt="\n" + end_of_prompt(prompt))

    def test_long_goal_is_shortened(self):
        long_goal = "a" * 1000
        prompt = build_plan_prompt(self.template, long_goal, self.world_state)
        self.assertNotIn(long_goal, prompt)
        self.assertIn("a" * (MAX_GOAL_LENGTH - 3) + "...", prompt)
        goal_in_prompt = prompt.split("User goal:\n")[1].strip()
        show("long goal", input_length=len(long_goal),
             length_in_prompt=len(goal_in_prompt), ends_with=goal_in_prompt[-10:])

    def test_same_input_gives_same_prompt(self):
        first_prompt = build_plan_prompt(self.template, "go", self.world_state)
        second_prompt = build_plan_prompt(self.template, "go", self.world_state)
        self.assertEqual(first_prompt, second_prompt)

    def test_first_plan_prompt_has_no_revision_text(self):
        prompt = build_plan_prompt(self.template, "go", self.world_state)
        self.assertNotIn("previous answer was rejected", prompt)
        self.assertNotIn("remaining work", prompt)

    def test_goal_containing_placeholder_text_stays_literal(self):
        prompt = build_plan_prompt(self.template, "say {WORLD_STATE}", self.world_state)
        self.assertIn("say {WORLD_STATE}", prompt)
        show("goal with placeholder text", prompt_excerpt="\n" + end_of_prompt(prompt))

    def test_template_with_missing_placeholder_is_rejected(self):
        broken_template = self.template.replace("{USER_GOAL}", "")
        with self.assertRaises(ValueError) as caught:
            build_plan_prompt(broken_template, "go", self.world_state)
        show("template without {USER_GOAL}", error=caught.exception)

    def test_template_with_unknown_placeholder_is_rejected(self):
        broken_template = self.template + "\n{SOMETHING_NEW}"
        with self.assertRaises(ValueError) as caught:
            build_plan_prompt(broken_template, "go", self.world_state)
        show("template with {SOMETHING_NEW}", error=caught.exception)

    def test_every_template_example_passes_the_validator(self):
        """A wrong example would teach the model to write invalid plans."""
        example_lines = [
            line for line in self.template.splitlines() if line.startswith('{"')
        ]
        self.assertGreaterEqual(len(example_lines), 4)
        validator = PlanValidator()
        for example_line in example_lines:
            example_plan = json.loads(example_line)
            valid, errors = validator.validate(example_plan)
            show("template example", goal=example_plan["goal"], valid=valid, errors=errors)
            self.assertTrue(valid, f"{errors} in example: {example_line[:80]}")

    def test_template_keeps_the_safety_rules(self):
        """These rules came from evaluation failures; guard against deleting them."""
        prompt = build_plan_prompt(self.template, "go", self.world_state)
        rules_part = prompt[prompt.index("Planning rules:"):prompt.index("Examples")]
        show("planning rules in the prompt", rules="\n" + rules_part)

        rules_on_one_line = " ".join(rules_part.split())  # ignore line breaks
        safety_rules = [
            "Only plan what the user asked for.",
            "A conditional on last_search_result must come right after a search step.",
            "A plan that contains a search must end with one report step.",
            "A plan without a search has no report step.",
            "search these rooms one at a time, in this order",
            "For any other movement (flying, jumping, climbing, going through walls), "
            "return only an ask step",
        ]
        for safety_rule in safety_rules:
            self.assertIn(safety_rule, rules_on_one_line)

    def test_first_plans_see_no_state_from_the_previous_task(self):
        """The spec forbids assuming the target location from a previous trial."""
        state = dict(self.world_state, current_location="office_b",
                     last_search_result={"result": "FOUND", "location": "office_b"},
                     execution_status="SUCCEEDED")
        prompt = build_plan_prompt(self.template, "find the red ball", state)
        state_line = prompt.splitlines()[prompt.splitlines().index("Current world/task state:") + 1]
        self.assertEqual(state_line, '{"current_location": "office_b"}')

    def test_search_rooms_rule_names_the_search_rooms(self):
        prompt = build_plan_prompt(self.template, "go", self.world_state)
        self.assertIn("laboratory, meeting_room, office_b, office_a.", prompt)

    def test_example_report_messages_are_all_different(self):
        """Repeated example text was copied word for word by the 0.5B model."""
        report_messages = []
        for line in self.template.splitlines():
            if line.startswith('{"'):
                for step in json.loads(line)["steps"]:
                    if step["action"] == "report":
                        report_messages.append(step["message"])
        show("example report messages", messages="\n" + "\n".join(report_messages))
        self.assertEqual(len(report_messages), len(set(report_messages)))


class CorrectionPromptTests(unittest.TestCase):
    def setUp(self):
        self.template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self.world_state = WorldState().as_dict()

    def test_contains_rejected_plan_and_errors(self):
        rejected_plan = {"version": "1.0", "goal": "x", "steps": [{"action": "fly"}]}
        prompt = build_correction_prompt(
            self.template, "x", self.world_state, rejected_plan,
            ["steps[0]: unsupported action 'fly'."],
        )
        self.assertIn('"action": "fly"', prompt)
        self.assertIn("unsupported action 'fly'", prompt)
        self.assertIn("Do not repeat your previous answer.", prompt)
        self.assertIn("Return a corrected complete plan", prompt)
        show("correction prompt (end)", prompt_excerpt="\n" + end_of_prompt(prompt))

    def test_long_rejected_plan_is_not_shortened(self):
        """Errors point at step paths, so the model must see the whole plan."""
        longest_example = max(
            (line for line in self.template.splitlines() if line.startswith('{"')),
            key=len,
        )
        rejected_plan = json.loads(longest_example)
        rejected_plan["steps"][-2]["then"][0]["location"] = "store_room"
        prompt = build_correction_prompt(
            self.template, "x", self.world_state, rejected_plan,
            ["steps[4].then[0]: invalid location 'store_room'."],
        )
        self.assertIn(json.dumps(rejected_plan), prompt)
        show("long rejected plan", characters=len(json.dumps(rejected_plan)))

    def test_shows_only_the_first_errors(self):
        errors = [f"error number {index}" for index in range(MAX_ERRORS_SHOWN + 3)]
        prompt = build_correction_prompt(self.template, "x", self.world_state,
                                         "bad output", errors)
        self.assertIn(f"error number {MAX_ERRORS_SHOWN - 1}", prompt)
        self.assertNotIn(f"error number {MAX_ERRORS_SHOWN}", prompt)
        show(f"correction prompt with {len(errors)} errors (end)",
             prompt_excerpt="\n" + end_of_prompt(prompt))


class CorrectionHintTests(unittest.TestCase):
    def setUp(self):
        self.template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self.world_state = WorldState().as_dict()

    def test_nesting_error_gets_a_hint(self):
        prompt = build_correction_prompt(
            self.template, "search the building for the red ball", self.world_state,
            {"steps": []}, ["steps[2].then[2].then[2].then: nesting depth exceeds 2."])
        show("correction prompt with hint (end)", prompt_excerpt="\n" + end_of_prompt(prompt))
        self.assertIn("How to fix:", prompt)
        self.assertIn("one after another at the top level", prompt)

    def test_conditional_before_search_gets_a_hint(self):
        prompt = build_correction_prompt(
            self.template, "x", self.world_state, {"steps": []},
            ["steps[1].condition: 'last_search_result' checked before any search was performed."])
        self.assertIn("write navigate, then search, then the conditional", prompt)

    def test_other_errors_get_no_hint(self):
        prompt = build_correction_prompt(self.template, "x", self.world_state,
                                         "bad", ["steps[0]: unsupported action 'fly'."])
        self.assertNotIn("How to fix:", prompt)


SEARCH_RED_BALL = {"action": "search", "object": "ball", "colour": "red"}
IF_NOT_FOUND = {"source": "last_search_result", "operator": "found", "value": False}


def if_not_found(*steps):
    return {"action": "conditional", "condition": dict(IF_NOT_FOUND),
            "then": list(steps), "else": []}


def navigate(location):
    return {"action": "navigate", "location": location}


def not_found_at(room):
    return {"result": "NOT_FOUND", "outcome": "NOT_FOUND", "found": False,
            "confidence": 0.0, "detection": None, "location": room}


class ReplanSituationTests(unittest.TestCase):
    """replan_situation(): the facts a revision must respect, worked out in code."""

    def search_then_blocked(self):
        """Lab searched (not found), then the route to the meeting room was blocked."""
        trace = [
            {"step": navigate("laboratory"), "status": "SUCCEEDED",
             "evidence": {"target": "laboratory", "outcome": "SUCCEEDED", "reason": "arrived",
                          "position": [0.0, 2.0], "elapsed_steps": 410}},
            {"step": SEARCH_RED_BALL, "status": "SUCCEEDED", "evidence": not_found_at("laboratory")},
            {"step": {"action": "conditional"}, "status": "EXPANDED", "evidence": None},
            {"step": navigate("meeting_room"), "status": "BLOCKED",
             "evidence": {"target": "meeting_room", "outcome": "BLOCKED",
                          "reason": "front obstacle", "position": [1.2, -0.4],
                          "elapsed_steps": 95}},
        ]
        remaining = [navigate("meeting_room"), SEARCH_RED_BALL,
                     if_not_found(navigate("office_b"), SEARCH_RED_BALL),
                     {"action": "report", "message": "Tell the user where the red ball was found."}]
        state = WorldState()
        state.current_location = "laboratory"
        state.blocked_routes = ["meeting_room"]
        return replan_situation("find the red ball", state.as_dict(), trace, remaining)

    def test_places_left_skip_blocked_and_searched_rooms(self):
        situation = self.search_then_blocked()
        show("situation after search + BLOCKED", places_left=situation["places_left"],
             search_target=situation["search_target"])
        self.assertEqual(situation["places_left"], ["office_b"])
        self.assertEqual(situation["place_kinds"], {"office_b": "search"})
        self.assertEqual(situation["search_target"], "red ball")
        self.assertEqual(situation["searched_at"], {"red ball": ["laboratory"]})
        self.assertIn("meeting_room", situation["unreachable"])
        self.assertTrue(situation["stopped_on_the_way"])

    def test_failed_step_is_not_counted_as_not_started(self):
        situation = self.search_then_blocked()
        self.assertEqual(situation["failed_step"], navigate("meeting_room"))
        self.assertNotIn(navigate("meeting_room"), situation["steps_not_started"])

    def test_failed_or_timed_out_destination_is_unreachable(self):
        trace = [{"step": navigate("laboratory"), "status": "TIMEOUT",
                  "evidence": {"reason": "no arrival within 3000 steps"}}]
        situation = replan_situation("go to the lab", WorldState().as_dict(), trace,
                                     [navigate("laboratory")])
        self.assertEqual(situation["places_left"], [])
        self.assertIn("laboratory", situation["unreachable"])

    def test_found_target_drops_search_rooms_but_keeps_trips(self):
        state = WorldState()
        state.current_location = "office_b"
        state.search_results = [{"target": "red ball", "location": "office_b", "result": "FOUND"}]
        trace = [{"step": navigate("office_a"), "status": "BLOCKED",
                  "evidence": {"reason": "front obstacle"}}]
        remaining = [navigate("office_a"), navigate("meeting_room"), SEARCH_RED_BALL,
                     {"action": "return_home"}]
        state.blocked_routes = ["office_a"]
        situation = replan_situation("x", state.as_dict(), trace, remaining)
        self.assertEqual(situation["places_left"], ["reception"])
        self.assertIsNone(situation["search_target"])

    def test_room_the_robot_is_in_is_not_a_place_left(self):
        # A search failed (not a trip), so the robot really is at office_a.
        state = WorldState()
        state.current_location = "office_a"
        trace = [{"step": SEARCH_RED_BALL, "status": "TIMEOUT", "evidence": "Exceeded ticks"}]
        remaining = [SEARCH_RED_BALL, navigate("office_a"), navigate("laboratory"), SEARCH_RED_BALL]
        situation = replan_situation("x", state.as_dict(), trace, remaining)
        self.assertEqual(situation["places_left"], ["laboratory"])

    def test_rooms_searched_before_the_bounded_trace_are_known(self):
        """The Webots trial where the replan searched the meeting room again."""
        state = WorldState()
        state.current_location = "meeting_room"
        state.blocked_routes = ["office_b"]
        state.search_results = [
            {"target": "red ball", "location": "laboratory", "result": "NOT_FOUND"},
            {"target": "red ball", "location": "meeting_room", "result": "NOT_FOUND"},
        ]
        trace = [{"step": navigate("office_b"), "status": "BLOCKED",
                  "evidence": {"reason": "front obstacle"}}]
        remaining = [navigate("office_b"), SEARCH_RED_BALL,
                     if_not_found(navigate("laboratory"), SEARCH_RED_BALL),
                     if_not_found(navigate("office_a"), SEARCH_RED_BALL)]
        situation = replan_situation("find the red ball", state.as_dict(), trace, remaining)
        self.assertEqual(situation["places_left"], ["office_a"])


class ReplanPromptTests(unittest.TestCase):
    def setUp(self):
        self.situations = ReplanSituationTests()

    def test_prompt_states_the_facts_and_places_left(self):
        prompt = build_replan_prompt(self.situations.search_then_blocked())
        show("replan prompt after search + BLOCKED", prompt="\n" + prompt)
        self.assertIn('The user\'s request was: "find the red ball"', prompt)
        self.assertIn("- navigate laboratory -> arrived", prompt)
        self.assertIn("- search red ball at laboratory -> not found", prompt)
        self.assertIn("- navigate meeting_room -> BLOCKED (front obstacle)", prompt)
        self.assertIn("The robot stopped on the way to meeting_room. "
                      "The last place it reached was laboratory.", prompt)
        self.assertIn("Already searched for the red ball, not found: laboratory.", prompt)
        self.assertIn("Cannot be reached now: meeting_room.", prompt)
        self.assertIn("- office_b: search for the red ball there", prompt)
        # One room left: no conditional example to copy the room into.
        self.assertIn(REPLAN_EXAMPLE_ONE_ROOM, prompt)
        self.assertNotIn(REPLAN_EXAMPLE, prompt)

    def test_two_rooms_left_get_the_conditional_example(self):
        state = WorldState()
        state.current_location = "laboratory"
        state.blocked_routes = ["meeting_room"]
        trace = [{"step": navigate("meeting_room"), "status": "BLOCKED",
                  "evidence": {"reason": "front obstacle"}}]
        remaining = [navigate("meeting_room"), SEARCH_RED_BALL,
                     if_not_found(navigate("office_b"), SEARCH_RED_BALL),
                     if_not_found(navigate("office_a"), SEARCH_RED_BALL)]
        prompt = build_replan_prompt(replan_situation("find the red ball", state.as_dict(),
                                                      trace, remaining))
        self.assertIn("- office_b: search for the red ball there\n"
                      "- office_a: search for the red ball there", prompt)
        self.assertIn(REPLAN_EXAMPLE, prompt)
        self.assertNotIn("elapsed_steps", prompt)
        self.assertNotIn("EXPANDED", prompt)

    def test_is_short_and_not_the_first_plan_template(self):
        prompt = build_replan_prompt(self.situations.search_then_blocked())
        self.assertNotIn("Request: Can you locate the red ball", prompt)
        self.assertLess(len(prompt), 2000)

    def test_nothing_left_asks_for_one_report(self):
        trace = [{"step": navigate("office_b"), "status": "BLOCKED",
                  "evidence": {"reason": "front obstacle"}}]
        state = WorldState()
        state.blocked_routes = ["office_b"]
        prompt = build_replan_prompt(replan_situation(
            "go to Office B", state.as_dict(), trace, [navigate("office_b")]))
        show("replan prompt, nothing left", prompt="\n" + prompt)
        self.assertIn("Places still left to visit: none.", prompt)
        self.assertIn("exactly one report step", prompt)
        self.assertNotIn(REPLAN_EXAMPLE, prompt)
        self.assertNotIn(REPLAN_EXAMPLE_ONE_ROOM, prompt)

    def test_shows_only_the_most_recent_trace_entries(self):
        total_entries = MAX_TRACE_ENTRIES_SHOWN + 3
        trace = [{"step": {"action": "observe"}, "status": "SUCCEEDED",
                  "evidence": f"evidence-{index}"} for index in range(total_entries)]
        prompt = build_replan_prompt(replan_situation("x", WorldState().as_dict(), trace, []))
        oldest_shown_index = total_entries - MAX_TRACE_ENTRIES_SHOWN
        self.assertIn(f"evidence-{total_entries - 1}", prompt)
        self.assertIn(f"evidence-{oldest_shown_index}", prompt)
        self.assertNotIn(f"evidence-{oldest_shown_index - 1}", prompt)

    def test_rejected_revision_and_problems_are_shown(self):
        prompt = build_replan_prompt(self.situations.search_then_blocked(),
                                     rejected_plan={"steps": []},
                                     rejected_errors=["steps[1]: already searched at laboratory"])
        self.assertIn("Your previous plan was rejected:", prompt)
        self.assertIn("- steps[1]: already searched at laboratory", prompt)


class ReplanFromExecutiveTraceTests(unittest.TestCase):
    """A real TaskExecutive + RobotSkills trace, with fake navigator and camera."""

    class FakeNavigator:
        def start(self, location):
            self.target = location
            return "RUNNING"

        def update(self):
            if self.target == "office_b":
                self.last_evidence = {"reason": "front obstacle"}
                return "BLOCKED"  # the navigator decides BLOCKED itself
            self.last_evidence = {"reason": "arrived"}
            return "SUCCEEDED"

        def cancel(self):
            pass

    class FakePerception:
        def start_search(self, object_type, colour):
            pass

        def update_search(self):
            return "SUCCEEDED", {"outcome": "NOT_FOUND", "found": False, "detection": None}

    class FakeSpeech:
        def speak(self, text):
            pass

    def test_search_location_from_robot_skills_reaches_the_prompt(self):
        from robot_skills import RobotSkills
        from semantic_memory import SemanticMemory
        from task_executive import TaskExecutive

        world_state = WorldState()
        skills = RobotSkills(self.FakeNavigator(), self.FakePerception(),
                             SemanticMemory(), self.FakeSpeech(), world_state)
        executive = TaskExecutive(skills, world_state)
        executive.start({"steps": [
            navigate("meeting_room"), SEARCH_RED_BALL, navigate("office_b"), SEARCH_RED_BALL,
            if_not_found(navigate("office_a"), SEARCH_RED_BALL),
        ]})
        while executive.active:
            executive.update()

        situation = replan_situation(
            "find the red ball", world_state.as_dict(),
            world_state.get_failure_trace(executive.trace),
            executive.steps[executive.step_index:])
        prompt = build_replan_prompt(situation)
        show("replan prompt from executive trace", prompt="\n" + prompt)
        self.assertIn("- search red ball at meeting_room -> not found", prompt)
        self.assertIn("- navigate office_b -> BLOCKED (front obstacle)", prompt)
        self.assertEqual(situation["places_left"], ["office_a"])


if __name__ == "__main__":
    unittest.main()
