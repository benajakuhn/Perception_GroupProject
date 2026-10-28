"""JSON schemas that limit what the language model can write.

llama-cpp turns a schema into a grammar and, while the model writes, blocks
every token that does not fit it. The model can then only write the allowed
actions, location ids, objects and colours, and conditionals no deeper than
PlanValidator allows. This shapes the output only: every plan still goes
through PlanValidator, and the plan's content is still the model's.

prompt_builder first works out the request's "situation" in code; each
schema narrows the plan contract (plan_schema.py) to that situation.

plan_output_schema(...)    -- first plans and corrections: searches only for
                              the object the goal names; a search that names
                              no room only drives to the search rooms
replan_output_schema(...)  -- a revised remainder: "navigate" may only name
                              the places still left; with none left, one
                              report step
In both, searching is only possible on the branch that runs while the
object has not been found.
"""
from plan_schema import (
    ALLOWED_CONDITION_SOURCES,
    MAX_NESTING_DEPTH,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    PLAN_VERSION,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
    VALID_ROUTE_STATUSES,
)

# Long free text makes the model ramble; the validator allows MAX_TEXT_LENGTH.
GOAL_TEXT_LENGTH = 120
# A plan that searches must end with a report (asked in words, the model
# left it out of a third of its conditional searches). JSON Schema cannot say
# "the last item", so each allowed length is listed: n steps, then a report.
# Every listed length slows generation (measured: 10 lengths took 43 s for a
# plan that takes 27 s with 6), so 6: the longest template example has 5 steps
# before its report, and a search of five rooms needs 6.
MAX_STEPS_BEFORE_REPORT = 6


def plan_output_schema(situation: dict = None) -> dict:
    """A first plan for prompt_builder.plan_situation(); None: any valid plan.

    situation["search_targets"] fixes what a search may look for (any
    supported object when empty) and, when not empty, makes the plan end
    with a report; situation["navigate_to"] lists the rooms navigate may
    name (any location when None).
    """
    situation = situation or {}
    locations = situation.get("navigate_to") or sorted(VALID_LOCATIONS)
    targets = situation.get("search_targets") or []
    if targets:
        searches = [_search_step(target["object"], target["colour"]) for target in targets]
    else:
        searches = [_search_step()]
    definitions = {}
    if targets:
        # No report among the top-level steps: the only report is the last
        # step, which the grammar can then tell apart (and generates faster).
        step = _step_choices(0, locations, searches, True, definitions, with_report=False)
        steps = _ending_with_report(step)
    else:
        step = _step_choices(0, locations, searches, True, definitions)
        steps = {"type": "array", "items": step, "minItems": 1, "maxItems": MAX_TOTAL_STEPS}
    return _plan(steps, definitions)


def replan_output_schema(situation: dict) -> dict:
    """A revised remainder for prompt_builder.replan_situation().

    situation["places_left"] lists the places the plan may still drive to,
    situation["place_kinds"] which of them are to be searched, and
    situation["search_target"] the object still being looked for (or None).
    With no place left the only possible plan is one report step; with only
    trips left, one step per place and then a report; otherwise the plan
    searches and ends with a report, with a conditional offered only when
    two or more rooms are left to search.
    """
    places = situation.get("places_left") or []
    if not places:
        return _plan({"type": "array", "items": _report(), "minItems": 1, "maxItems": 1})

    rooms = [place for place in places if place != "reception"]
    simple_steps = []
    if rooms:
        simple_steps.append(_step("navigate", location={"enum": rooms}))
    if "reception" in places:
        simple_steps.append(_step("return_home"))

    target = situation.get("search_target")
    if not target:
        # Only trips are left: one step per place, then a report. Asked in
        # words, the model drove home and never said what had failed.
        slots = [{"anyOf": simple_steps}] * len(places) + [_report()]
        return _plan({"type": "array", "prefixItems": slots, "items": False,
                      "minItems": len(slots)})

    simple_steps.append(_search_for(target))
    choices = list(simple_steps)
    kinds = situation.get("place_kinds") or {}
    if list(kinds.values()).count("search") >= 2:
        # Only "if not found, go on searching": the result of the search
        # before the failure is already known, and nothing else is left.
        # With one room to search there is nothing to go on to, and the
        # model filled the branch with the same room again.
        choices.append(_object(
            action={"const": "conditional"},
            condition=_object(source={"const": "last_search_result"},
                              operator={"const": "found"}, value={"const": False}),
            then={"type": "array", "items": {"anyOf": simple_steps}},
            # llama-cpp ignored {"maxItems": 0} here and let a report in.
            **{"else": {"const": []}},
        ))
    definitions = {"step": {"anyOf": choices}}
    return _plan(_ending_with_report({"$ref": "#/$defs/step"}), definitions)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

def _plan(steps: dict, definitions: dict = None) -> dict:
    schema = _object(
        version={"const": PLAN_VERSION},
        goal={"type": "string", "minLength": 1, "maxLength": GOAL_TEXT_LENGTH},
        steps=steps,
    )
    if definitions:
        # Each step type is written once and referred to, which keeps the
        # schema and llama-cpp's grammar small.
        schema["$defs"] = definitions
    return schema


def _ending_with_report(step: dict) -> dict:
    """1 to MAX_STEPS_BEFORE_REPORT steps, then a report step."""
    return {"anyOf": [
        {"type": "array", "prefixItems": [step] * count + [_report()],
         "items": False, "minItems": count + 1}
        for count in range(1, MAX_STEPS_BEFORE_REPORT + 1)
    ]}


def _step_choices(level: int, locations: list, searches: list, may_search: bool,
                  definitions: dict, with_report: bool = True) -> dict:
    """A reference to the step choices at one nesting level.

    Conditionals only while the validator allows more nesting. may_search
    is False on the branch that runs once the object was found: searching
    there again is what the model wrote in its odd plans.
    """
    name = (f"steps_level_{level}" + ("" if may_search else "_found")
            + ("" if with_report else "_before_report"))
    if name not in definitions:
        choices = [
            _step("navigate", location={"enum": locations}),
            *(searches if may_search else []),
            _step("observe"),
            _step("ask", question=_text()),
            *([_report()] if with_report else []),
            _step("return_home"),
            _step("stop"),
        ]
        if level < MAX_NESTING_DEPTH:
            def branch(searching):
                return {"type": "array", "items": _step_choices(
                    level + 1, locations, searches, searching, definitions)}

            for source in sorted(ALLOWED_CONDITION_SOURCES):
                if source == "last_search_result":
                    # "then" runs when found == value: searching only while not found.
                    for value in (False, True):
                        choices.append(_conditional(
                            _object(source={"const": source}, operator={"const": "found"},
                                    value={"const": value}),
                            then=branch(may_search and not value),
                            otherwise=branch(may_search and value)))
                else:
                    choices.append(_conditional(
                        _object(source={"const": source}, operator={"const": "equals"},
                                value={"enum": sorted(VALID_ROUTE_STATUSES)}),
                        then=branch(may_search), otherwise=branch(may_search)))
        definitions[name] = {"anyOf": choices}
    return {"$ref": f"#/$defs/{name}"}


def _conditional(condition: dict, then: dict, otherwise: dict) -> dict:
    return _object(action={"const": "conditional"}, condition=condition,
                   then=then, **{"else": otherwise})


def _search_step(object_type: str = None, colour: str = None) -> dict:
    """A search for one target, or for any supported object and colour."""
    if object_type is None:
        return _step("search", object={"enum": sorted(SUPPORTED_OBJECTS)},
                     colour={"enum": sorted(SUPPORTED_COLOURS)}, optional=("colour",))
    if colour is None:
        return _step("search", object={"const": object_type},
                     colour={"enum": sorted(SUPPORTED_COLOURS)}, optional=("colour",))
    return _step("search", object={"const": object_type}, colour={"const": colour})


def _search_for(target: str) -> dict:
    """A search step for 'red ball' (colour and object fixed) or 'ball'."""
    words = target.split()
    if len(words) > 1 and words[0] in SUPPORTED_COLOURS:
        return _search_step(words[-1], words[0])
    return _step("search", object={"const": words[-1]})


def _report() -> dict:
    return _step("report", message=_text())


def _text() -> dict:
    return {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_LENGTH}


def _step(action: str, optional=(), **fields) -> dict:
    return _object(action={"const": action}, optional=optional, **fields)


def _object(optional=(), **properties) -> dict:
    """An object with exactly these properties, written in this order."""
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }
