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
    build_correction_prompt,
    build_plan_prompt,
    build_replan_prompt,
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
            build_replan_prompt(self.template, "find the red ball", self.world_state,
                                [{"step": {"action": "stop"}, "status": "FAILED"}], []),
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
            "End the plan with a report step only if it contains a search.",
            "For any other movement (flying, jumping, climbing, going through walls), "
            "return only an ask step",
        ]
        for safety_rule in safety_rules:
            self.assertIn(safety_rule, rules_on_one_line)

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

    def test_shows_only_the_first_errors(self):
        errors = [f"error number {index}" for index in range(MAX_ERRORS_SHOWN + 3)]
        prompt = build_correction_prompt(self.template, "x", self.world_state,
                                         "bad output", errors)
        self.assertIn(f"error number {MAX_ERRORS_SHOWN - 1}", prompt)
        self.assertNotIn(f"error number {MAX_ERRORS_SHOWN}", prompt)
        show(f"correction prompt with {len(errors)} errors (end)",
             prompt_excerpt="\n" + end_of_prompt(prompt))


class ReplanPromptTests(unittest.TestCase):
    def setUp(self):
        self.template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self.world_state = WorldState().as_dict()

    def test_shows_only_the_most_recent_trace_entries(self):
        total_entries = MAX_TRACE_ENTRIES_SHOWN + 3
        trace = [
            {"step": {"action": "observe"}, "status": "SUCCEEDED",
             "evidence": f"evidence-{index}"}
            for index in range(total_entries)
        ]
        prompt = build_replan_prompt(self.template, "x", self.world_state, trace, [])
        oldest_shown_index = total_entries - MAX_TRACE_ENTRIES_SHOWN
        self.assertIn(f"evidence-{total_entries - 1}", prompt)
        self.assertIn(f"evidence-{oldest_shown_index}", prompt)
        self.assertNotIn(f"evidence-{oldest_shown_index - 1}", prompt)
        show(f"replan prompt with {total_entries} trace entries (end)",
             prompt_excerpt="\n" + end_of_prompt(prompt))

    def test_contains_failure_remaining_steps_and_rules(self):
        trace = [{
            "step": {"action": "navigate", "location": "laboratory"},
            "status": "BLOCKED",
            "evidence": "front obstacle at ps0",
        }]
        remaining_steps = [{"action": "search", "object": "ball", "colour": "red"}]
        prompt = build_replan_prompt(self.template, "check the lab for the red ball",
                                     self.world_state, trace, remaining_steps)
        self.assertIn("-> BLOCKED; evidence: front obstacle at ps0", prompt)
        self.assertIn('{"action": "search", "object": "ball", "colour": "red"}', prompt)
        self.assertIn("Plan only the remaining work", prompt)
        self.assertIn("check the lab for the red ball", prompt)
        show("replan prompt after BLOCKED (end)", prompt_excerpt="\n" + end_of_prompt(prompt))


if __name__ == "__main__":
    unittest.main()
