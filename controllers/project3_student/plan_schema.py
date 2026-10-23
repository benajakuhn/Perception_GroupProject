"""Shared symbolic-plan definitions and schemas.

Defines the deterministic contract, allowed actions, execution statuses, and condition grammar.
"""

# Existing baseline definitions (preserved)
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

# --- Extensions for Execution Monitoring & Enhanced Validation ---

STATUS_IDLE = "IDLE"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_BLOCKED = "BLOCKED"
STATUS_CANCELLED = "CANCELLED"

ALL_STATUSES = {
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_TIMEOUT,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
}

ACTION_OPTIONAL_FIELDS = {
    "search": {"colour"},
}

CONDITION_REQUIRED_FIELDS = {"source", "operator", "value"}
VALID_ROUTE_STATUSES = {"CLEAR", "BLOCKED", "FAILED", "TIMEOUT"}
