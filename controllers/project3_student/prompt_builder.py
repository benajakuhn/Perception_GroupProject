"""Build the planning prompt for the language model.

Everything the model is told about actions, locations, objects and limits is
generated from plan_schema.py, so the prompt cannot drift away from what
PlanValidator accepts. This module only builds text: it never calls the
model, reads files or decides whether a plan is valid.
"""
import json
import re

from location_grounding import LOCATION_NICKNAMES
from plan_schema import (
    ACTION_REQUIRED_FIELDS,
    ALLOWED_CONDITION_OPERATORS,
    ALLOWED_CONDITION_SOURCES,
    MAX_NESTING_DEPTH,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    PLAN_VERSION,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
)

MAX_GOAL_LENGTH = 300
MAX_PREVIOUS_OUTPUT_LENGTH = 1000
MAX_ERROR_LENGTH = 200
MAX_ERRORS_SHOWN = 5
MAX_TRACE_ENTRIES_SHOWN = 5
MAX_TRACE_STEP_LENGTH = 150
MAX_EVIDENCE_LENGTH = 150

WORLD_STATE_KEYS_SHOWN = (
    "current_location",
    "last_search_result",
    "blocked_routes",
    "execution_status",
)

PLACEHOLDER = re.compile(r"\{[A-Z_]+\}")

ACTION_DESCRIPTIONS = {
    "navigate": "drive to a location",
    "search": "look around the current location for an object; colour is optional",
    "observe": "update what the robot knows about its surroundings",
    "ask": "ask the user a question and wait for the answer",
    "report": "tell the user something; the robot does not move",
    "return_home": "drive back to reception",
    "stop": "stop moving and end the task",
}

FIELD_PLACEHOLDERS = {
    "location": "<location id>",
    "object": "<object type>",
    "colour": "<colour>",
    "question": "<text>",
    "message": "<text>",
}

OPTIONAL_FIELDS = {"search": ["colour"]}

# Assumption until agreed with R3: which values each condition source takes.
CONDITION_DESCRIPTIONS = {
    "last_search_result": 'use operator "found" with value true or false '
                          "(was the object found by the last search?)",
    "route_status": 'use operator "equals" with a status text such as "BLOCKED"',
}


def build_plan_prompt(template: str, user_goal: str, world_state: dict) -> str:
    """Prompt for the first plan of a new goal."""
    return _fill_template(template, user_goal, world_state, revision_context="")


def build_correction_prompt(template: str, user_goal: str, world_state: dict,
                            rejected_output, errors: list) -> str:
    """Prompt asking for a corrected plan after parsing or validation failed.

    rejected_output is the raw model text (parse failure) or the parsed plan
    dict (validation failure).
    """
    error_lines = [
        f"- {_shorten(error, MAX_ERROR_LENGTH)}"
        for error in errors[:MAX_ERRORS_SHOWN]
    ]
    revision_lines = [
        "Your previous answer was rejected.",
        "Previous answer:",
        _shorten(_as_text(rejected_output), MAX_PREVIOUS_OUTPUT_LENGTH),
        "Problems found:",
        *error_lines,
        "Do not repeat your previous answer.",
        "Return a corrected complete plan that fixes every problem, using only the "
        "allowed actions, locations, objects and colours listed above.",
    ]
    return _fill_template(template, user_goal, world_state,
                          revision_context=_as_section(revision_lines))


def build_replan_prompt(template: str, original_goal: str, world_state: dict,
                        execution_trace: list, remaining_steps: list) -> str:
    """Prompt asking for a plan for the remaining work after a failure.

    Only the most recent trace entries are included, so the prompt stays
    bounded however long the task has been running.
    """
    recent_entries = execution_trace[-MAX_TRACE_ENTRIES_SHOWN:]
    trace_lines = [_describe_trace_entry(entry) for entry in recent_entries]
    remaining_text = json.dumps(remaining_steps) if remaining_steps else "none"

    revision_lines = [
        "The robot was carrying out a plan for this goal but could not finish it.",
        "What happened so far (most recent last):",
        *trace_lines,
        f"Steps that were not carried out yet: {remaining_text}",
        "Plan only the remaining work, starting from the robot's current location.",
        "Do not repeat steps that already succeeded.",
        "Do not navigate again to a location whose route was BLOCKED; "
        "choose another location or report.",
        "If the goal can no longer be reached, return a plan with a report step "
        "that explains why.",
    ]
    return _fill_template(template, original_goal, world_state,
                          revision_context=_as_section(revision_lines))


def _fill_template(template: str, user_goal: str, world_state: dict,
                   revision_context: str) -> str:
    """Replace every {PLACEHOLDER} in the template in one single pass.

    One pass means text inserted from the goal or the model's previous answer
    is never scanned again, so a goal containing "{WORLD_STATE}" stays literal.
    """
    values = {
        "{PLAN_CONTRACT}": _describe_plan_contract(),
        "{ACTION_SCHEMA}": _describe_actions(),
        "{VALID_LOCATIONS}": _describe_locations(),
        "{SUPPORTED_OBJECTS}": _describe_objects(),
        "{WORLD_STATE}": _describe_world_state(world_state),
        "{REVISION_CONTEXT}": revision_context,
        "{USER_GOAL}": _shorten(user_goal, MAX_GOAL_LENGTH),
    }

    placeholders_in_template = set(PLACEHOLDER.findall(template))
    missing_placeholders = set(values) - placeholders_in_template
    unknown_placeholders = placeholders_in_template - set(values)
    if missing_placeholders or unknown_placeholders:
        raise ValueError(
            "Prompt template does not match prompt_builder: "
            f"missing {sorted(missing_placeholders)}, "
            f"unknown {sorted(unknown_placeholders)}."
        )

    def value_for(placeholder_match):
        return values[placeholder_match.group(0)]

    return PLACEHOLDER.sub(value_for, template)


def _describe_plan_contract() -> str:
    return "\n".join([
        f'{{"version": {json.dumps(PLAN_VERSION)}, "goal": "<short restatement '
        'of the request>", "steps": [<step>, ...]}',
        f'- "version" must be exactly {json.dumps(PLAN_VERSION)}.',
        f'- "goal" is a non-empty text of at most {MAX_TEXT_LENGTH} characters.',
        f"- At most {MAX_TOTAL_STEPS} steps in total, counting steps inside "
        "conditionals.",
        f"- Conditionals may be nested at most {MAX_NESTING_DEPTH} levels deep.",
        f'- "question" and "message" texts are at most {MAX_TEXT_LENGTH} characters.',
    ])


def _describe_actions() -> str:
    action_lines = []
    for action in sorted(ACTION_REQUIRED_FIELDS):
        step_format = {"action": action}
        fields = sorted(ACTION_REQUIRED_FIELDS[action]) + OPTIONAL_FIELDS.get(action, [])
        for field in fields:
            step_format[field] = FIELD_PLACEHOLDERS.get(field, "<value>")
        description = ACTION_DESCRIPTIONS.get(action, "")
        action_lines.append(f"- {action}: {json.dumps(step_format)} -- {description}")

    action_lines.append(
        '- conditional: {"action": "conditional", "condition": {"source": '
        '<source>, "operator": <operator>, "value": <value>}, "then": [<steps>], '
        '"else": [<steps>]} -- run "then" if the condition is true, otherwise '
        '"else"; both lists are required and may be empty'
    )
    action_lines.append(
        "  Allowed condition operators: " + ", ".join(sorted(ALLOWED_CONDITION_OPERATORS))
    )
    action_lines.append("  Allowed condition sources:")
    for source in sorted(ALLOWED_CONDITION_SOURCES):
        description = CONDITION_DESCRIPTIONS.get(source, "")
        action_lines.append(f'  - "{source}": {description}')
    return "\n".join(action_lines)


def _describe_locations() -> str:
    location_lines = []
    for location in sorted(VALID_LOCATIONS):
        nicknames = LOCATION_NICKNAMES.get(location)
        if nicknames:
            location_lines.append(f"- {location} (also called: {', '.join(nicknames)})")
        else:
            location_lines.append(f"- {location}")
    return "\n".join(location_lines)


def _describe_objects() -> str:
    return "\n".join([
        "Object types: " + ", ".join(sorted(SUPPORTED_OBJECTS)),
        "Colours: " + ", ".join(sorted(SUPPORTED_COLOURS)),
    ])


def _describe_world_state(world_state: dict) -> str:
    world_state = world_state or {}
    shown_state = {key: world_state.get(key) for key in WORLD_STATE_KEYS_SHOWN}
    return json.dumps(shown_state, default=str)


def _describe_trace_entry(entry) -> str:
    if not isinstance(entry, dict):
        return f"- {_shorten(_as_text(entry), MAX_TRACE_STEP_LENGTH)}"
    step_text = _shorten(_as_text(entry.get("step")), MAX_TRACE_STEP_LENGTH)
    status = entry.get("status", "UNKNOWN")
    evidence_text = _shorten(_as_text(entry.get("evidence", "none")), MAX_EVIDENCE_LENGTH)
    return f"- {step_text} -> {status}; evidence: {evidence_text}"


def _as_section(lines: list) -> str:
    """Join lines into a block with a blank line before and after it."""
    return "\n" + "\n".join(lines) + "\n"


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _shorten(text, max_length: int) -> str:
    """Collapse whitespace and cut the text to max_length characters."""
    single_line_text = " ".join(str(text).split())
    if len(single_line_text) <= max_length:
        return single_line_text
    return single_line_text[:max_length - 3] + "..."
