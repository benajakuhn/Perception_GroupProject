"""Map location names written by the model to the configured location ids.

The model sometimes writes a nickname ("corridor", "lab") instead of the id
("main_corridor", "laboratory"). ground_locations() looks every location up
in one fixed table. It is deliberately strict:
- exact lookup only; case, spaces, hyphens and underscores are ignored;
- unknown names are left unchanged, so PlanValidator still rejects them;
- every replacement is returned as a note, so it can be logged.

The same table is shown to the model in the prompt (prompt_builder.py).
"""
import re

from plan_schema import VALID_LOCATIONS

LOCATION_NICKNAMES = {
    "reception": ["front desk", "home", "start"],
    "main_corridor": ["corridor", "hallway"],
    "office_a": ["Office A"],
    "office_b": ["Office B"],
    "laboratory": ["lab"],
    "meeting_room": ["conference room"],
}


def lookup_key(name: str) -> str:
    """'Office_A', 'office-a' and ' office  A ' all become 'office a'."""
    name_with_spaces = name.lower().replace("_", " ").replace("-", " ")
    return " ".join(name_with_spaces.split())


def _build_name_table() -> dict:
    """Every id and every nickname, as a lookup key, mapped to its id."""
    name_table = {}
    for location in VALID_LOCATIONS:
        name_table[lookup_key(location)] = location
    for location, nicknames in LOCATION_NICKNAMES.items():
        for nickname in nicknames:
            name_table[lookup_key(nickname)] = location
    return name_table


NAME_TO_LOCATION = _build_name_table()


def locations_named_in(text: str) -> list:
    """Location ids named in free text, by id or nickname, in order of mention.

    'Go to the lab, then Office B.' -> ['laboratory', 'office_b']. Used to
    decide how far the output schema may narrow a plan, never to plan.
    """
    words = " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())
    padded = f" {words} "
    found = []
    for name, location in NAME_TO_LOCATION.items():
        position = padded.find(f" {name} ")
        if position != -1:
            found.append((position, location))
    return list(dict.fromkeys(location for _, location in sorted(found)))


def ground_locations(plan: dict) -> list:
    """Replace nickname locations in the plan with ids (changes the plan).

    Returns one note per replacement, e.g.
    "Grounded location 'corridor' -> 'main_corridor'."
    """
    grounding_notes = []
    if isinstance(plan, dict):
        _ground_steps(plan.get("steps"), grounding_notes)
    return grounding_notes


def _ground_steps(steps, grounding_notes: list):
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue

        written_location = step.get("location")
        if isinstance(written_location, str):
            location_id = NAME_TO_LOCATION.get(lookup_key(written_location))
            if location_id is not None and location_id != written_location:
                step["location"] = location_id
                grounding_notes.append(
                    f"Grounded location {written_location!r} -> {location_id!r}."
                )

        # Conditionals contain more steps in their branches.
        _ground_steps(step.get("then"), grounding_notes)
        _ground_steps(step.get("else"), grounding_notes)
