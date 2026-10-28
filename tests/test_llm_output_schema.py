"""Tests for output_schema.py: the JSON schemas that limit what the model writes.

The schemas must accept every plan PlanValidator accepts that the model
should be able to write (the prompt's examples included), and rule out
what the validator would reject anyway (unknown actions or locations, too
deep nesting). Plans below are hand-written fixtures, not model output.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "controllers" / "project3_student"))

from output_schema import plan_output_schema, replan_output_schema  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402
from prompt_builder import DEFAULT_SEARCH_ROOMS, plan_situation  # noqa: E402

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None
TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "planning_prompt_template.txt"


def plan(*steps):
    return {"version": "1.0", "goal": "test", "steps": list(steps)}


def navigate(location):
    return {"action": "navigate", "location": location}


SEARCH_RED_BALL = {"action": "search", "object": "ball", "colour": "red"}
REPORT = {"action": "report", "message": "Tell the user what was found."}


def if_not_found(*steps):
    return {"action": "conditional",
            "condition": {"source": "last_search_result", "operator": "found", "value": False},
            "then": list(steps), "else": []}


class SchemaTestCase(unittest.TestCase):
    def fits(self, candidate, schema=None):
        import jsonschema
        try:
            jsonschema.validate(candidate, schema or plan_output_schema())
            return True
        except jsonschema.ValidationError:
            return False


@unittest.skipUnless(HAS_JSONSCHEMA, "needs the jsonschema package")
class PlanOutputSchemaTests(SchemaTestCase):
    def test_every_template_example_fits(self):
        examples = [json.loads(line) for line in TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
                    if line.startswith('{"')]
        self.assertGreaterEqual(len(examples), 6)
        for example in examples:
            self.assertTrue(self.fits(example), example["goal"])

    def test_nesting_allowed_by_the_validator_fits(self):
        nested = plan(navigate("laboratory"), SEARCH_RED_BALL,
                      if_not_found(navigate("office_a"), SEARCH_RED_BALL,
                                   if_not_found(navigate("office_b"), SEARCH_RED_BALL)),
                      REPORT)
        self.assertTrue(PlanValidator().validate(nested)[0])
        self.assertTrue(self.fits(nested))

    def test_nesting_the_validator_rejects_does_not_fit(self):
        too_deep = plan(navigate("laboratory"), SEARCH_RED_BALL,
                        if_not_found(navigate("office_a"), SEARCH_RED_BALL,
                                     if_not_found(navigate("office_b"), SEARCH_RED_BALL,
                                                  if_not_found(navigate("meeting_room")))))
        self.assertFalse(PlanValidator().validate(too_deep)[0])
        self.assertFalse(self.fits(too_deep))

    def test_unknown_values_do_not_fit(self):
        self.assertFalse(self.fits(plan(navigate("lab"))))
        self.assertFalse(self.fits(plan({"action": "fly", "location": "office_a"})))
        self.assertFalse(self.fits(plan({"action": "search", "object": "ball", "colour": "purple"})))
        self.assertFalse(self.fits(plan({"action": "return_home", "speed": 5})))
        self.assertFalse(self.fits({"version": "1.0", "goal": "x", "steps": []}))

    def test_route_status_condition_fits(self):
        step = {"action": "conditional",
                "condition": {"source": "route_status", "operator": "equals", "value": "BLOCKED"},
                "then": [{"action": "return_home"}], "else": []}
        self.assertTrue(self.fits(plan(navigate("office_a"), step)))


def template_examples():
    """(request, plan) pairs from the prompt template's examples."""
    lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    return [(lines[index - 1].removeprefix("Request: "), json.loads(line))
            for index, line in enumerate(lines) if line.startswith('{"')]


class PlanSituationTests(unittest.TestCase):
    def test_target_and_rooms_are_read_from_the_goal(self):
        situation = plan_situation("Search the building for the red ball.")
        self.assertEqual(situation["search_targets"], [{"object": "ball", "colour": "red"}])
        self.assertEqual(situation["navigate_to"], list(DEFAULT_SEARCH_ROOMS))

    def test_named_rooms_keep_every_location(self):
        situation = plan_situation("Check Office A for the red ball, then the lab.")
        self.assertIsNone(situation["navigate_to"])

    def test_reception_alone_does_not_count_as_a_named_room(self):
        situation = plan_situation("Find the red ball and then come back home.")
        self.assertEqual(situation["navigate_to"], list(DEFAULT_SEARCH_ROOMS))

    def test_colour_added_by_the_dialogue_is_used(self):
        situation = plan_situation("find the ball (red ball)")
        self.assertEqual(situation["search_targets"], [{"object": "ball", "colour": "red"}])

    def test_goal_without_object_is_not_narrowed(self):
        situation = plan_situation("Fly over the desks to Office A.")
        self.assertEqual(situation["search_targets"], [])
        self.assertIsNone(situation["navigate_to"])

    def test_search_rooms_are_the_spawn_point_rooms(self):
        spawn_points = json.loads((PROJECT_ROOT / "config" / "spawn_points.json")
                                  .read_text(encoding="utf-8"))
        rooms = {point["location"] for point in spawn_points["approved_spawn_points"]}
        self.assertEqual(set(DEFAULT_SEARCH_ROOMS), rooms)

    def test_search_rooms_follow_the_locate_example(self):
        locate = dict(template_examples())["Can you locate the red ball for me?"]
        visited = []

        def walk(steps):
            for step in steps:
                if step["action"] == "navigate":
                    visited.append(step["location"])
                elif step["action"] == "conditional":
                    walk(step["then"])
        walk(locate["steps"])
        self.assertEqual(visited, list(DEFAULT_SEARCH_ROOMS))


@unittest.skipUnless(HAS_JSONSCHEMA, "needs the jsonschema package")
class NarrowedPlanSchemaTests(SchemaTestCase):
    def test_every_template_example_fits_its_own_situation(self):
        for request, example in template_examples():
            schema = plan_output_schema(plan_situation(request))
            self.assertTrue(self.fits(example, schema), request)

    def test_search_without_named_room_cannot_drive_to_reception(self):
        schema = plan_output_schema(plan_situation("Search the building for the red ball."))
        self.assertFalse(self.fits(plan(navigate("reception"), SEARCH_RED_BALL, REPORT), schema))
        self.assertTrue(self.fits(plan(navigate("laboratory"), SEARCH_RED_BALL, REPORT), schema))

    def test_search_is_fixed_to_the_named_target(self):
        schema = plan_output_schema(plan_situation("Find the red ball."))
        blue_box = {"action": "search", "object": "box", "colour": "blue"}
        self.assertFalse(self.fits(plan(navigate("laboratory"), blue_box), schema))

    def test_no_search_once_the_object_was_found(self):
        schema = plan_output_schema(plan_situation("Check Office B for the red ball."))
        odd = if_not_found(navigate("office_a"), SEARCH_RED_BALL)
        odd["else"] = [navigate("laboratory"), SEARCH_RED_BALL]
        self.assertFalse(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, odd), schema))
        come_back = if_not_found(navigate("office_a"), SEARCH_RED_BALL)
        come_back["else"] = [{"action": "return_home"}]
        self.assertTrue(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, come_back, REPORT),
                                  schema))

    def test_search_plan_must_end_with_a_report(self):
        schema = plan_output_schema(plan_situation("Check Office B for the red ball."))
        conditional = if_not_found(navigate("office_a"), SEARCH_RED_BALL)
        self.assertFalse(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, conditional),
                                   schema))
        self.assertTrue(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, conditional,
                                       REPORT), schema))

    def test_navigation_plan_needs_no_report(self):
        schema = plan_output_schema(plan_situation("Go to the lab, then come back."))
        self.assertTrue(self.fits(plan(navigate("laboratory"), {"action": "return_home"}),
                                  schema))


@unittest.skipUnless(HAS_JSONSCHEMA, "needs the jsonschema package")
class ReplanOutputSchemaTests(SchemaTestCase):
    def situation(self, places, target="red ball"):
        kinds = {place: "trip" if place == "reception" or not target else "search"
                 for place in places}
        return {"places_left": places, "place_kinds": kinds, "search_target": target}

    def test_nothing_left_allows_only_one_report(self):
        schema = replan_output_schema(self.situation([], target=None))
        self.assertTrue(self.fits(plan(REPORT), schema))
        self.assertFalse(self.fits(plan(REPORT, REPORT), schema))
        self.assertFalse(self.fits(plan(navigate("office_a")), schema))

    def test_only_the_places_left_can_be_driven_to(self):
        schema = replan_output_schema(self.situation(["office_b", "office_a"]))
        good = plan(navigate("office_b"), SEARCH_RED_BALL,
                    if_not_found(navigate("office_a"), SEARCH_RED_BALL), REPORT)
        self.assertTrue(self.fits(good, schema))
        self.assertFalse(self.fits(plan(navigate("meeting_room"), SEARCH_RED_BALL, REPORT), schema))
        self.assertFalse(self.fits(plan({"action": "return_home"}), schema))
        self.assertFalse(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL), schema))

    def test_one_room_left_allows_no_conditional(self):
        schema = replan_output_schema(self.situation(["office_b"]))
        self.assertTrue(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, REPORT), schema))
        repeat = plan(navigate("office_b"), SEARCH_RED_BALL,
                      if_not_found(navigate("office_b"), SEARCH_RED_BALL), REPORT)
        self.assertFalse(self.fits(repeat, schema))

    def test_search_is_fixed_to_the_target(self):
        schema = replan_output_schema(self.situation(["office_b"]))
        blue_box = {"action": "search", "object": "box", "colour": "blue"}
        self.assertFalse(self.fits(plan(navigate("office_b"), blue_box), schema))

    def test_trips_only_end_with_a_report(self):
        schema = replan_output_schema(self.situation(["reception"], target=None))
        self.assertTrue(self.fits(plan({"action": "return_home"}, REPORT), schema))
        self.assertFalse(self.fits(plan({"action": "return_home"}), schema))
        self.assertFalse(self.fits(plan(SEARCH_RED_BALL, REPORT), schema))

    def test_conditional_else_must_be_empty(self):
        schema = replan_output_schema(self.situation(["office_b", "office_a"]))
        step = if_not_found(navigate("office_a"), SEARCH_RED_BALL)
        step["else"] = [navigate("office_b")]
        self.assertFalse(self.fits(plan(navigate("office_b"), SEARCH_RED_BALL, step), schema))


@unittest.skipUnless(HAS_LLAMA_CPP, "needs llama-cpp-python")
class GrammarCompileTests(unittest.TestCase):
    """llama-cpp must be able to turn every schema into a grammar."""

    def test_schemas_compile_to_grammars(self):
        from llama_cpp import LlamaGrammar
        for schema in (plan_output_schema(),
                       plan_output_schema(plan_situation("Find the red ball.")),
                       replan_output_schema({"places_left": [], "search_target": None}),
                       replan_output_schema({"places_left": ["reception"],
                                             "place_kinds": {"reception": "trip"},
                                             "search_target": None}),
                       replan_output_schema({"places_left": ["office_b", "office_a", "reception"],
                                             "place_kinds": {"office_b": "search",
                                                             "office_a": "search",
                                                             "reception": "trip"},
                                             "search_target": "red ball"})):
            LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)


if __name__ == "__main__":
    unittest.main()
