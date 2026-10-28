"""Build the planning prompt for the language model.

Everything the model is told about actions, locations, objects and limits is
generated from plan_schema.py, so the prompt cannot drift away from what
PlanValidator accepts. This module only builds text: it never calls the
model, reads files or decides whether a plan is valid.

First plans and corrections fill prompts/planning_prompt_template.txt.
Replans use their own short prompt (build_replan_prompt), built from the
facts replan_situation() works out in code. plan_situation() does the same
for a new goal; both situations also decide the output schema
(output_schema.py), i.e. what the model can write at all.
"""
import json
import re

from location_grounding import LOCATION_NICKNAMES, locations_named_in
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

# Follow-up goals ("Earlier request: ... I asked: ... The user answered: ...")
# hold three parts of up to 200 characters each; the answer comes last.
MAX_GOAL_LENGTH = 700
MAX_PREVIOUS_OUTPUT_LENGTH = 2000  # long search plans exceed 1000 characters
MAX_ERROR_LENGTH = 200
MAX_ERRORS_SHOWN = 5
MAX_TRACE_ENTRIES_SHOWN = 5
MAX_TRACE_STEP_LENGTH = 150
MAX_EVIDENCE_LENGTH = 150

# Rooms the robot searches when the user names none: the rooms with approved
# ball spawn points, in the order of the template's 'locate' example.
DEFAULT_SEARCH_ROOMS = ("laboratory", "meeting_room", "office_b", "office_a")

# Only where the robot is. A first plan must not be steered by the previous
# task: its last_search_result would hint where the ball was last time (the
# spec forbids assuming the target location from a previous trial), and
# blocked routes are cleared for every new goal. Replans get their facts
# from replan_situation() instead.
WORLD_STATE_KEYS_SHOWN = ("current_location",)

PLACEHOLDER = re.compile(r"\{[A-Z_]+\}")

FAILURE_STATUSES = ("FAILED", "TIMEOUT", "BLOCKED")

# Search outcomes in words for the replan prompt. UNCERTAIN counts as not
# found, so the model gets one simple rule for both.
SEARCH_RESULT_WORDS = {
    "FOUND": "found",
    "NOT_FOUND": "not found",
    "UNCERTAIN": "not sure, treat as not found",
}

# Shown in replan prompts that still have places to search. Placeholders,
# not real rooms: the small model copied the rooms of a concrete example,
# and replan_output_schema() only lets it write the places really left.
# With one room left the model copied the two-room example's conditional
# and searched the same room twice, so each case gets its own example.
REPLAN_EXAMPLE_ONE_ROOM = (
    '{"version": "1.0", "goal": "search the place left", "steps": ['
    '{"action": "navigate", "location": "<place left>"}, '
    '{"action": "search", "object": "<object>", "colour": "<colour>"}, '
    '{"action": "report", "message": "Tell the user whether it was found."}]}'
)
REPLAN_EXAMPLE = (
    '{"version": "1.0", "goal": "search the places left", "steps": ['
    '{"action": "navigate", "location": "<first place left>"}, '
    '{"action": "search", "object": "<object>", "colour": "<colour>"}, '
    '{"action": "conditional", "condition": {"source": "last_search_result", '
    '"operator": "found", "value": false}, "then": ['
    '{"action": "navigate", "location": "<next place left>"}, '
    '{"action": "search", "object": "<object>", "colour": "<colour>"}], "else": []}, '
    '{"action": "report", "message": "Tell the user whether it was found."}]}'
)

# Extra advice for validator errors the model repeated when it was shown
# only the error (evaluation/run_planner_eval.py, "Search the building").
ERROR_HINTS = {
    "nesting depth exceeds":
        "Do not put conditionals inside each other; put them one after another "
        "at the top level of \"steps\", as in the 'locate' example.",
    "checked before any search":
        "A conditional on last_search_result must come right after a search "
        "step: write navigate, then search, then the conditional, as in the "
        "'locate' example.",
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

# How TaskExecutive evaluates each condition source.
CONDITION_DESCRIPTIONS = {
    "last_search_result": 'use operator "found" with value true or false '
                          "(was the object found by the last search?)",
    "route_status": 'use operator "equals" with a status text such as "BLOCKED"',
}


def plan_situation(user_goal: str) -> dict:
    """What a first plan for this goal may contain, worked out in code.

      search_targets -- [{"object": "ball", "colour": "red"}, ...] named in
                        the goal (colour None if none was said); empty when
                        the goal names no object
      navigate_to    -- the rooms a plan may drive to, or None for any: only
                        a search that names no room is narrowed, to
                        DEFAULT_SEARCH_ROOMS (the model once searched
                        reception first). When rooms are named, every
                        location stays allowed, so a phrasing this check
                        misses can never lock the model out.

    The goal reaching the planner has been rewritten by the dialogue into
    names the planner knows ("read ball" -> "red ball", "officer" -> "Office A").
    """
    words = re.sub(r"[^a-z0-9]+", " ", str(user_goal).lower()).split()
    targets = []
    for index, word in enumerate(words):
        if word in SUPPORTED_OBJECTS:
            previous = words[index - 1] if index else None
            colour = previous if previous in SUPPORTED_COLOURS else None
            target = {"object": word, "colour": colour}
            if target not in targets:
                targets.append(target)
    # "the ball (red ball)": a bare object that is also named with a colour
    targets = [target for target in targets
               if target["colour"] or not any(other["object"] == target["object"]
                                              and other["colour"] for other in targets)]
    rooms_named = [location for location in locations_named_in(user_goal)
                   if location != "reception"]
    return {
        "goal": user_goal,
        "search_targets": targets,
        "navigate_to": list(DEFAULT_SEARCH_ROOMS) if targets and not rooms_named else None,
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
        *_hints_for(errors),
        "Do not repeat your previous answer.",
        "Return a corrected complete plan that fixes every problem, using only the "
        "allowed actions, locations, objects and colours listed above.",
    ]
    return _fill_template(template, user_goal, world_state,
                          revision_context=_as_section(revision_lines))


def replan_situation(original_goal: str, world_state: dict, execution_trace: list,
                     remaining_steps: list) -> dict:
    """Work out, in code, what a revised remainder may still do.

    A small model did not infer this reliably from the trace (it re-planned
    the whole goal and revisited searched or blocked rooms), so the facts are
    decided here and given to both the prompt and replan_output_schema():

      places_left   -- places the remaining plan still drives to, in order,
                       without blocked or failed destinations, rooms already
                       searched for the target, or the room the robot is in
      place_kinds   -- place -> "search" (drive there to search) or "trip"
      search_target -- e.g. "red ball" while it is still being looked for

    Rooms already searched come from world_state["search_results"] as well
    as the trace, because the bounded trace and a restarted plan's trace
    both lose older searches. The main loop passes the failed step as the
    first remaining step.
    """
    world_state = world_state or {}
    here = world_state.get("current_location") or "an unknown location"
    failed = _failed_step(execution_trace)
    # After a failed trip, current_location is the last place reached, not
    # where the robot is, so that place may be driven to again.
    stopped_on_the_way = (isinstance(failed, dict)
                          and failed.get("action") in {"navigate", "return_home"})

    found_at, searched_at = {}, {}     # "red ball" -> rooms
    unreachable = list(world_state.get("blocked_routes") or [])

    def note_search(target, room, result):
        if target and room:
            rooms = (found_at if result == "FOUND" else searched_at).setdefault(target, [])
            if room not in rooms:
                rooms.append(room)

    for result in world_state.get("search_results") or []:
        note_search(result.get("target"), result.get("location"), result.get("result"))
    for entry in execution_trace:
        if not isinstance(entry, dict) or not isinstance(entry.get("step"), dict):
            continue
        step, status, evidence = entry["step"], entry.get("status"), entry.get("evidence")
        action = step.get("action")
        if action == "search" and status == "SUCCEEDED" and isinstance(evidence, dict):
            note_search(_search_target(step), evidence.get("location"), evidence.get("result"))
        elif action in {"navigate", "return_home"} and status in FAILURE_STATUSES:
            target = step.get("location") or "reception"
            if target not in unreachable:
                unreachable.append(target)

    steps_left = list(remaining_steps or [])
    if isinstance(failed, dict) and (not steps_left or steps_left[0] != failed):
        steps_left.insert(0, failed)
    visits, searches = [], []
    _collect_visits(steps_left, visits, searches)

    targets = [_search_target(step) for step in searches] + list(searched_at) + list(found_at)
    target = targets[0] if targets else None
    target_found = target in found_at

    place_kinds = {}
    for place, kind in visits:
        if place in place_kinds or place in unreachable:
            continue
        if place == here and not stopped_on_the_way:
            continue
        if kind == "search" and (target_found or place in searched_at.get(target, [])):
            continue
        place_kinds[place] = kind

    steps_not_started = steps_left[1:] if isinstance(failed, dict) else steps_left
    shown_entries = [entry for entry in execution_trace
                     if not _is_expanded_conditional(entry)]
    return {
        "goal": original_goal,
        "here": here,
        "stopped_on_the_way": stopped_on_the_way,
        "failed_step": failed,
        "steps_not_started": steps_not_started,
        "trace_lines": [_describe_trace_entry(entry)
                        for entry in shown_entries[-MAX_TRACE_ENTRIES_SHOWN:]],
        "found_at": found_at,
        "searched_at": searched_at,
        "unreachable": unreachable,
        "places_left": list(place_kinds),
        "place_kinds": place_kinds,
        "search_target": (target if not target_found and "search" in place_kinds.values()
                          else None),
    }


def build_replan_prompt(situation: dict, rejected_plan=None, rejected_errors=None) -> str:
    """Short prompt for the remaining work after a failure.

    Deliberately not the first-plan template: with its from-scratch examples
    the small model re-planned the original goal (0/5 replans followed the
    rules). This prompt states the facts from replan_situation() and what
    the plan should look like; replan_output_schema() enforces the places.
    rejected_plan/rejected_errors: the previous revision and why it was
    rejected, for one corrected attempt.
    """
    failed = situation["failed_step"]
    lines = [
        "You are the task planner for a small indoor robot. It was carrying out "
        "a plan, but a step failed.",
        "Write a NEW plan with ONLY the work that is still left. Return one JSON "
        "object and nothing else.",
        "",
        f'The user\'s request was: "{_shorten(situation["goal"], MAX_GOAL_LENGTH)}"',
        "What happened (oldest first):",
        *situation["trace_lines"],
    ]
    if situation["stopped_on_the_way"]:
        lines.append(f"The robot stopped on the way to {failed.get('location') or 'reception'}. "
                     f"The last place it reached was {situation['here']}.")
    else:
        lines.append(f"The robot is at {situation['here']}.")
    for target, rooms in situation["found_at"].items():
        lines.append(f"The {target} was FOUND at: {', '.join(rooms)}.")
    for target, rooms in situation["searched_at"].items():
        lines.append(f"Already searched for the {target}, not found: {', '.join(rooms)}.")
    if situation["unreachable"]:
        lines.append(f"Cannot be reached now: {', '.join(map(str, situation['unreachable']))}.")
    lines.append("")

    if situation["places_left"]:
        lines.append("Places still left to visit, in this order:")
        for place, kind in situation["place_kinds"].items():
            if kind == "search":
                lines.append(f"- {place}: search for the {situation['search_target']} there")
            elif place == "reception":
                lines.append("- reception: drive back there (return_home)")
            else:
                lines.append(f"- {place}: drive there")
        lines.append("How to plan: visit the places in that order and nowhere else.")
        rooms_to_search = list(situation["place_kinds"].values()).count("search")
        if situation["search_target"] and rooms_to_search >= 2:
            lines += [
                "After a search, put the next place to search inside a conditional on "
                "last_search_result with value false, so the robot stops once it finds "
                "the object. Visit each place once. End with one report step.",
                "Example for a different situation (placeholders in <>):",
                REPLAN_EXAMPLE,
            ]
        elif situation["search_target"]:
            lines += [
                "Search that one place once, then end with one report step.",
                "Example for a different situation (placeholders in <>):",
                REPLAN_EXAMPLE_ONE_ROOM,
            ]
        else:
            # Without this the model drove home and never said why.
            lines.append("End with one report step that tells the user, in one short "
                         "sentence, what could not be done and why. Speak as the robot (\"I\").")
    else:
        lines += [
            "Places still left to visit: none.",
            "How to plan: exactly one report step. In it, tell the user in one or two "
            "short sentences what was done, what was found, and what could not be "
            "done and why. Speak as the robot (\"I\").",
        ]

    if rejected_plan is not None:
        lines += [
            "",
            "Your previous plan was rejected:",
            _shorten(_as_text(rejected_plan), MAX_PREVIOUS_OUTPUT_LENGTH),
            "Problems:",
            *(f"- {_shorten(error, MAX_ERROR_LENGTH)}"
              for error in (rejected_errors or [])[:MAX_ERRORS_SHOWN]),
            "Write a corrected plan that follows the facts above.",
        ]
    return "\n".join(lines)


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
        "{SEARCH_ROOMS}": ", ".join(DEFAULT_SEARCH_ROOMS),
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


def _collect_visits(steps, visits: list, searches: list):
    """Places the steps drive to, in order, as (place, "search" or "trip"),
    and the search steps themselves. A navigate followed by a search is a
    trip to search that room; return_home is a trip to reception."""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        following = steps[index + 1] if index + 1 < len(steps) else None
        if action == "navigate" and isinstance(step.get("location"), str):
            searches_there = isinstance(following, dict) and following.get("action") == "search"
            visits.append((step["location"], "search" if searches_there else "trip"))
        elif action == "return_home":
            visits.append(("reception", "trip"))
        elif action == "search":
            searches.append(step)
        elif action == "conditional":
            for branch in ("then", "else"):
                if isinstance(step.get(branch), list):
                    _collect_visits(step[branch], visits, searches)


def _hints_for(errors: list) -> list:
    """Advice lines for the validator errors listed in ERROR_HINTS."""
    hints = [hint for pattern, hint in ERROR_HINTS.items()
             if any(pattern in str(error) for error in errors)]
    return ["How to fix:", *(f"- {hint}" for hint in hints)] if hints else []


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
