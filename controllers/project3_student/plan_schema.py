"""Shared symbolic-plan definitions and schemas.

Defines the deterministic contract, allowed actions, execution statuses, and condition grammar.
"""

ALLOWED_ACTIONS = {
    "navigate",
    "search",
    "observe",
    "ask",
    "report",
    "return_home",
    "stop",
    "conditional",
}

PLAN_VERSION = "1.0"
MAX_TOTAL_STEPS = 20
MAX_NESTING_DEPTH = 2
MAX_TEXT_LENGTH = 240
# Excessive repeated navigation: a plan may drive to the same location at most
# this often (counting both branches of conditionals), e.g. lab -> office -> lab.
MAX_VISITS_PER_LOCATION = 2

ACTION_REQUIRED_FIELDS = {
    "navigate": {"location"},
    "search": {"object"},
    "observe": set(),
    "ask": {"question"},
    "report": {"message"},
    "return_home": set(),
    "stop": set(),
    "conditional": {"condition", "then", "else"},
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

IDLE = "IDLE"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
BLOCKED = "BLOCKED"

ACTION_OPTIONAL_FIELDS = {
    "search": {"colour"},
}

CONDITION_REQUIRED_FIELDS = {"source", "operator", "value"}
VALID_ROUTE_STATUSES = {"CLEAR", "BLOCKED", "FAILED", "TIMEOUT"}
