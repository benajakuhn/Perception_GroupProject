"""Tests for location_grounding: nickname -> location id lookup.

All plans below are hand-written test fixtures, not model output.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "controllers" / "project3_student"
sys.path.insert(0, str(CONTROLLER_DIR))

from location_grounding import (  # noqa: E402
    LOCATION_NICKNAMES,
    ground_locations,
    lookup_key,
)
from plan_schema import VALID_LOCATIONS  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402


def show(title, **values):
    """Print a labelled block so a test result can be checked by eye.

    Add -b to the unittest command to hide these prints for passing tests.
    """
    print(f"\n    --- {title} ---")
    for label, value in values.items():
        indented_value = str(value).replace("\n", "\n        ")
        print(f"    {label}: {indented_value}")


def plan_with_location(location):
    return {"version": "1.0", "goal": "go somewhere",
            "steps": [{"action": "navigate", "location": location}]}


def ground_and_show(plan):
    plan_before = repr(plan["steps"])
    notes = ground_locations(plan)
    show("ground_locations", before=plan_before, after=plan["steps"], notes=notes)
    return notes


class GroundingTests(unittest.TestCase):
    def test_nickname_is_replaced_with_id(self):
        plan = plan_with_location("corridor")
        notes = ground_and_show(plan)
        self.assertEqual(plan["steps"][0]["location"], "main_corridor")
        self.assertEqual(notes, ["Grounded location 'corridor' -> 'main_corridor'."])

    def test_case_spaces_hyphens_and_underscores_are_ignored(self):
        written_names = {
            "Office A": "office_a",
            "OFFICE_B": "office_b",
            "meeting-room": "meeting_room",
            "  Main   Corridor ": "main_corridor",
            "Front Desk": "reception",
            "LAB": "laboratory",
        }
        for written_name, expected_id in written_names.items():
            plan = plan_with_location(written_name)
            ground_and_show(plan)
            self.assertEqual(plan["steps"][0]["location"], expected_id)

    def test_ids_are_left_unchanged_without_notes(self):
        for location in sorted(VALID_LOCATIONS):
            plan = plan_with_location(location)
            notes = ground_locations(plan)
            self.assertEqual(plan["steps"][0]["location"], location)
            self.assertEqual(notes, [])

    def test_unknown_names_are_left_for_the_validator(self):
        for unknown_name in ["office_c", "kitchen", "the roof"]:
            plan = plan_with_location(unknown_name)
            notes = ground_and_show(plan)
            valid, errors = PlanValidator().validate(plan)
            show("validator after grounding", valid=valid, errors=errors)
            self.assertEqual(plan["steps"][0]["location"], unknown_name)
            self.assertEqual(notes, [])
            self.assertFalse(valid)

    def test_locations_inside_conditional_branches_are_grounded(self):
        plan = {"version": "1.0", "goal": "search", "steps": [
            {"action": "navigate", "location": "lab"},
            {"action": "search", "object": "ball", "colour": "red"},
            {"action": "conditional",
             "condition": {"source": "last_search_result", "operator": "found",
                           "value": False},
             "then": [{"action": "navigate", "location": "conference room"},
                      {"action": "search", "object": "ball", "colour": "red"}],
             "else": [{"action": "navigate", "location": "hallway"}]},
        ]}
        notes = ground_and_show(plan)
        self.assertEqual(plan["steps"][0]["location"], "laboratory")
        self.assertEqual(plan["steps"][2]["then"][0]["location"], "meeting_room")
        self.assertEqual(plan["steps"][2]["else"][0]["location"], "main_corridor")
        self.assertEqual(len(notes), 3)
        self.assertTrue(PlanValidator().validate(plan)[0])

    def test_malformed_plans_do_not_crash(self):
        malformed_plans = [
            None,
            {},
            {"steps": "not a list"},
            {"steps": ["not a step", 42, None]},
            {"steps": [{"action": "navigate", "location": 123}]},
            {"steps": [{"action": "conditional", "then": "not a list", "else": None}]},
        ]
        for malformed_plan in malformed_plans:
            self.assertEqual(ground_locations(malformed_plan), [])


class NicknameTableTests(unittest.TestCase):
    def test_every_table_entry_is_a_valid_location(self):
        show("nickname table", table=LOCATION_NICKNAMES)
        for location in LOCATION_NICKNAMES:
            self.assertIn(location, VALID_LOCATIONS)

    def test_no_nickname_points_to_two_locations(self):
        seen_keys = {lookup_key(location): location for location in VALID_LOCATIONS}
        for location, nicknames in LOCATION_NICKNAMES.items():
            for nickname in nicknames:
                key = lookup_key(nickname)
                already_used_by = seen_keys.get(key)
                self.assertIn(already_used_by, (None, location),
                              f"{nickname!r} would map to two locations")
                seen_keys[key] = location


if __name__ == "__main__":
    unittest.main()
