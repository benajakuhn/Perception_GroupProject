"""Shared symbolic-plan definitions for the group project.

Students may extend the schema, but should keep the representation explicit,
machine-readable and independently validatable.
"""
ALLOWED_ACTIONS = {
    "navigate",
    "search",
    "observe",
    "ask",
    "report",
    "return_home",
    "stop",
}

PLAN_VERSION = "1.0"
MAX_TOTAL_STEPS = 20
MAX_NESTING_DEPTH = 2
MAX_TEXT_LENGTH = 240

ACTION_REQUIRED_FIELDS = {
    "navigate": {"location"},
    "search": {"object"},
    "observe": set(),
    "ask": {"question"},
    "report": {"message"},
    "return_home": set(),
    "stop": set(),
}

ALLOWED_CONDITION_SOURCES = {"last_search_result", "route_status"}
ALLOWED_CONDITION_OPERATORS = {"found", "equals"}

VALID_LOCATIONS = {
    "reception",
    "office_a",
    "office_b",
    "laboratory",
    "meeting_room",
    "main_corridor",
}

SUPPORTED_OBJECTS = {"ball", "box", "cylinder"}
SUPPORTED_COLOURS = {"red", "green", "blue", "yellow"}
