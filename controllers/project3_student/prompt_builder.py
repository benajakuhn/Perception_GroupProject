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
MAX_PREVIOUS_OUTPUT_LENGTH = 2000  # long search plans exceed 1000 characters
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

FAILURE_STATUSES = ("FAILED", "TIMEOUT", "BLOCKED")

# Search outcomes in words for the replan prompt. UNCERTAIN counts as not
# found, so the model gets one simple rule for both.
SEARCH_RESULT_WORDS = {
    "FOUND": "found",
    "NOT_FOUND": "not found",
    "UNCERTAIN": "not sure, treat as not found",
}

# Shown only in replan prompts. The small model copies the example closely,
# so the example takes its rooms from the same "still to search" fact line
# the real prompt has: copying the pattern then gives the right rooms.
# (An earlier example named a fixed room, and the model searched that room
# again although it had been searched.)
REPLAN_EXAMPLE_SITUATION = (
    "the blue box is wanted and the facts say "
    "'Rooms from the plan still to search that can be reached: laboratory, office_b'"
)
REPLAN_EXAMPLE_PLAN = {
    "version": "1.0",
    "goal": "find the blue box in the rooms not searched yet",
    "steps": [
        {"action": "navigate", "location": "laboratory"},
        {"action": "search", "object": "box", "colour": "blue"},
        {"action": "conditional",
         "condition": {"source": "last_search_result", "operator": "found", "value": False},
         "then": [{"action": "navigate", "location": "office_b"},
                  {"action": "search", "object": "box", "colour": "blue"}],
         "else": []},
        {"action": "report", "message": "Tell the user where the blue box was found, "
                                        "or that it was not found."},
    ],
}

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
                        execution_trace: list, remaining_steps: list,
                        rejected_plan=None, rejected_errors=None) -> str:
    """Prompt asking for a plan for the remaining work after a failure.

    The trace is written as short plain-language lines, and the facts the
    model must respect (where the robot is, rooms already searched, blocked
    rooms, rooms still to search) are worked out here, so a small model does
    not have to infer them. Only the most recent trace entries are shown, so
    the prompt stays bounded; the rooms already searched come from
    world_state["search_results"] (one short entry per search in this task),
    because the bounded trace and a restarted plan's trace both lose older
    searches. rejected_plan/rejected_errors: the previous revision and why
    PlanValidator.check_revision rejected it, for one corrected attempt.
    """
    world_state = world_state or {}
    current_location = world_state.get("current_location") or "an unknown location"
    shown_entries = [entry for entry in execution_trace
                     if not _is_expanded_conditional(entry)]
    recent_entries = shown_entries[-MAX_TRACE_ENTRIES_SHOWN:]
    trace_lines = [_describe_trace_entry(entry) for entry in recent_entries]

    # The main loop passes the failed step as the first remaining step; show
    # it on its own so it is not mistaken for work that still has to be done.
    failed_step = _failed_step(execution_trace)
    steps_not_started = list(remaining_steps)
    if failed_step is not None and steps_not_started and steps_not_started[0] == failed_step:
        steps_not_started = steps_not_started[1:]
    not_started_text = json.dumps(steps_not_started) if steps_not_started else "none"

    # Rules and the example come first and the facts last: the small model
    # follows what it read most recently, and it copied the example's rooms
    # when the example came after the facts.
    revision_lines = [
        "The robot was carrying out a plan for this goal but could not finish it.",
        "Rules for the new plan:",
        "- Plan only the remaining work, starting from where the robot is now.",
        "- If rooms are still to search, search them in the listed order; "
        "otherwise only report.",
        "- Do not repeat steps that already succeeded.",
        "- Do not search a room again where the object was already not found.",
        "- Do not navigate to a blocked or unreachable location; choose another "
        "location or report.",
        "- The result of the last search is already known (see below), so do not "
        "start the plan with a conditional on last_search_result. Use one only "
        "right after a new search step in this plan.",
        "- If the goal can no longer be reached, return a plan with a report step "
        "that explains why.",
        f"Example remainder plan (illustrative only) when {REPLAN_EXAMPLE_SITUATION}:",
        json.dumps(REPLAN_EXAMPLE_PLAN),
        "What happened so far (most recent last):",
        *trace_lines,
    ]
    if failed_step is not None:
        revision_lines.append(f"Step that failed: {json.dumps(failed_step)}")
    revision_lines += [
        f"Steps that were not started: {not_started_text}",
        "Known facts (use these, not the example):",
        *_replan_facts(execution_trace, world_state, current_location, remaining_steps),
    ]
    if rejected_plan is not None:
        revision_lines += [
            "Your previous plan was rejected:",
            _shorten(_as_text(rejected_plan), MAX_PREVIOUS_OUTPUT_LENGTH),
            "Problems:",
            *(f"- {_shorten(error, MAX_ERROR_LENGTH)}"
              for error in (rejected_errors or [])[:MAX_ERRORS_SHOWN]),
            "Write a corrected plan that follows the facts and rules above.",
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
    """One plain line per step, e.g. '- navigate meeting_room -> BLOCKED (front obstacle)'."""
    if not isinstance(entry, dict):
        return f"- {_shorten(_as_text(entry), MAX_TRACE_STEP_LENGTH)}"
    step = entry.get("step")
    status = entry.get("status") or "UNKNOWN"
    evidence = entry.get("evidence")
    return f"- {_describe_traced_step(step, evidence)} -> {_describe_outcome(status, evidence)}"


def _describe_traced_step(step, evidence) -> str:
    if not isinstance(step, dict):
        return _shorten(_as_text(step), MAX_TRACE_STEP_LENGTH)
    action = step.get("action")
    if action == "navigate":
        return f"navigate {step.get('location')}"
    if action == "search":
        step_text = f"search {_search_target(step)}"
        if isinstance(evidence, dict) and evidence.get("location"):
            step_text += f" at {evidence['location']}"
        return step_text
    if action in {"return_home", "observe", "stop"}:
        return action
    return _shorten(_as_text(step), MAX_TRACE_STEP_LENGTH)


def _describe_outcome(status: str, evidence) -> str:
    """'arrived', 'not found', 'done' or e.g. 'BLOCKED (front obstacle)'.

    Only the reason or search result is shown; positions, step counts and
    detection objects mean nothing to the planner.
    """
    detail = None
    if isinstance(evidence, dict):
        result_words = SEARCH_RESULT_WORDS.get(evidence.get("result"))
        if status == "SUCCEEDED" and result_words:
            return result_words
        detail = evidence.get("reason") or result_words
    elif evidence not in (None, "OK", status):
        detail = _as_text(evidence)

    if detail is not None:
        detail = _shorten(detail, MAX_EVIDENCE_LENGTH)
    if status == "SUCCEEDED":
        return detail or "done"
    return f"{status} ({detail})" if detail else status


def _replan_facts(execution_trace: list, world_state: dict, current_location: str,
                  remaining_steps: list = ()) -> list:
    """Facts the remainder plan must respect, worked out from the trace,
    the task's search results and the steps that were not done."""
    found_at = {}        # "red ball" -> rooms where it was found
    not_found_at = {}    # "red ball" -> rooms searched without finding it
    unreachable = list(world_state.get("blocked_routes") or [])

    for result in world_state.get("search_results") or []:
        room, target = result.get("location"), result.get("target")
        if room and target:
            rooms_by_target = found_at if result.get("result") == "FOUND" else not_found_at
            rooms = rooms_by_target.setdefault(target, [])
            if room not in rooms:
                rooms.append(room)

    for entry in execution_trace:
        if not isinstance(entry, dict) or not isinstance(entry.get("step"), dict):
            continue
        step, status, evidence = entry["step"], entry.get("status"), entry.get("evidence")
        action = step.get("action")

        if action == "search" and status == "SUCCEEDED" and isinstance(evidence, dict):
            room = evidence.get("location")
            if room:
                rooms_by_target = found_at if evidence.get("result") == "FOUND" else not_found_at
                rooms = rooms_by_target.setdefault(_search_target(step), [])
                if room not in rooms:
                    rooms.append(room)

        elif action in {"navigate", "return_home"} and status in FAILURE_STATUSES:
            target = step.get("location") or "reception"
            if target not in unreachable:
                unreachable.append(target)

    failed = _failed_step(execution_trace)
    if isinstance(failed, dict) and failed.get("action") in {"navigate", "return_home"}:
        # current_location is the last place reached, not where the robot is.
        facts = [f"- The robot stopped on the way to {failed.get('location') or 'reception'}. "
                 f"The last place it reached was {current_location}."]
    else:
        facts = [f"- The robot is at {current_location}."]
    for target, rooms in found_at.items():
        facts.append(f"- The {target} was found at: {', '.join(rooms)}.")
    for target, rooms in not_found_at.items():
        facts.append(f"- Already searched without finding the {target}: {', '.join(rooms)}.")
    if unreachable:
        facts.append(
            f"- Blocked or unreachable, do not navigate there: {', '.join(map(str, unreachable))}."
        )

    # Rooms the plan still meant to search: the failed step and the steps not
    # started, conditional branches included.
    wanted = [target for target in found_at] + [target for target in not_found_at]
    planned_rooms, searches = [], []
    steps_left = list(remaining_steps or [])
    if isinstance(failed, dict) and (not steps_left or steps_left[0] != failed):
        steps_left.insert(0, failed)   # the main loop passes it first already
    _collect_planned_searches(steps_left, planned_rooms, searches)
    wanted += [_search_target(step) for step in searches]
    still_to_search = []
    if wanted and not any(target in found_at for target in wanted):
        target = wanted[-1]
        done = set(not_found_at.get(target, [])) | set(unreachable)
        still_to_search = [room for room in dict.fromkeys(planned_rooms) if room not in done]
        facts.append("- Rooms from the plan still to search that can be reached: "
                     f"{', '.join(still_to_search) if still_to_search else 'none'}.")

    # A trip whose destination cannot be reached: without this line the model
    # drove somewhere else instead of telling the user.
    if isinstance(failed, dict) and failed.get("action") == "navigate":
        destination = failed.get("location")
        if destination in unreachable and not still_to_search:
            facts.append(f"- {destination} cannot be reached now, so the plan should "
                         f"be one report step telling the user that.")
    return facts


def _collect_planned_searches(steps, rooms: list, searches: list):
    """Rooms the steps search (a navigate followed by a search), in order,
    and the search steps themselves."""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        following = steps[index + 1] if index + 1 < len(steps) else None
        if (action == "navigate" and isinstance(step.get("location"), str)
                and isinstance(following, dict) and following.get("action") == "search"):
            rooms.append(step["location"])
        elif action == "search":
            searches.append(step)
        elif action == "conditional":
            for branch in ("then", "else"):
                if isinstance(step.get(branch), list):
                    _collect_planned_searches(step[branch], rooms, searches)


def _failed_step(execution_trace: list):
    """The step of the most recent failed, timed-out or blocked trace entry."""
    for entry in reversed(execution_trace):
        if isinstance(entry, dict) and entry.get("status") in FAILURE_STATUSES:
            return entry.get("step")
    return None


def _is_expanded_conditional(entry) -> bool:
    """Conditional bookkeeping entries; their chosen steps appear on their own."""
    return isinstance(entry, dict) and entry.get("status") == "EXPANDED"


def _search_target(step: dict) -> str:
    """'red ball', or just 'ball' when no colour was given."""
    return " ".join(value for value in (step.get("colour"), step.get("object")) if value)


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
