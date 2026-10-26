"""Run the planner on test goals and print how the real model does.

Usage (from the project folder):
    python evaluation/run_planner_eval.py
    python evaluation/run_planner_eval.py --only conditional
    python evaluation/run_planner_eval.py --only replan

Each goal goes through the same steps the controller will use:
    generate_plan -> PlanValidator -> if rejected: correct_plan
    (at most max_retries corrections, from config/llm_config.json)

Replan cases start from a recorded failure instead of a fresh goal:
    replan(goal, state, failure trace, remaining steps) -> PlanValidator -> ...
Their valid plans are also checked against the replan rules (no navigating to
a blocked room, no searching a room again). The main loop does not correct
rejected replans yet, so the '1st try' column is what the robot gets today.

The results are GENUINE model output, not test fixtures. Every model call is
also logged by LLMInterface in logs/llm_<date>.jsonl, and a summary of each
run is saved in evaluation/results/.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "controllers" / "project3_student"))

from llm_backends import LLMBackendError  # noqa: E402
from llm_interface import LLMInterface, PlanParseError  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402
from world_state import WorldState  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
LINE = "=" * 76

# ----------------------------------------------------------------------------
# Test goals: edit freely.
#   (request type, what the user says, what a good plan should do)
# Use your own wording, not the examples from the spec or the prompt template.
#
# Only the planner's own scope is tested here. Ambiguous goals ("find the
# ball") are caught by DialogueManager before planning, and status questions
# are answered from WorldState, so neither reaches the planner.
# ----------------------------------------------------------------------------
TEST_GOALS = [
    ("direct", "Drive over to Office B.",
     "navigate office_b"),
    ("direct", "Please head to the meeting room.",
     "navigate meeting_room"),

    ("sequential", "Visit Office A and then go back to reception.",
     "navigate office_a, return_home"),
    ("sequential", "Go to the corridor, after that the meeting room.",
     "navigate main_corridor, navigate meeting_room"),

    ("search", "Where is the red ball? Go and look for it.",
     "search several rooms one by one until found, then report"),
    ("search", "Search the building for the red ball.",
     "search several rooms one by one until found, then report"),

    ("conditional", "Check Office A for the red ball; if you don't see it, look in the lab.",
     "navigate office_a, search; if not found: navigate laboratory, search; report"),
    ("conditional", "Search the meeting room for the red ball and if it's not there go to office B.",
     "navigate meeting_room, search; if not found: navigate office_b, search; report"),

    ("constrained", "Search the lab first, and only look in the meeting room if the red ball isn't there.",
     "navigate laboratory, search; if not found: navigate meeting_room, search; report"),
    ("constrained", "Start with Office B. Only if the red ball is missing there, check Office A.",
     "navigate office_b, search; if not found: navigate office_a, search; report"),

    ("unsupported", "Fly over the desks to Office A.",
     "ask or report that flying is impossible; no flying action"),
    ("unsupported", "Smash through the wall into the meeting room.",
     "ask or report; no smashing action"),
]


# ----------------------------------------------------------------------------
# Replan cases: a failure part-way through a plan, written the way the
# executive records it (trace entries with evidence, remaining steps starting
# with the failed step). Hand-written situations, but the plans the model
# returns for them are genuine model output.
# ----------------------------------------------------------------------------

def arrived(location):
    return {"step": {"action": "navigate", "location": location}, "status": "SUCCEEDED",
            "evidence": {"target": location, "outcome": "SUCCEEDED", "reason": "arrived"}}


def navigation_failed(step, status, reason):
    target = step.get("location") or "reception"
    return {"step": step, "status": status,
            "evidence": {"target": target, "outcome": status, "reason": reason}}


def searched(object_type, colour, room, result):
    evidence = {"result": result, "outcome": result, "found": result == "FOUND",
                "confidence": 0.85 if result == "FOUND" else 0.0,
                "detection": None, "location": room}
    return {"step": {"action": "search", "object": object_type, "colour": colour},
            "status": "SUCCEEDED", "evidence": evidence}


def expanded(conditional_step):
    return {"step": conditional_step, "status": "EXPANDED", "evidence": None}


def if_not_found(*then_steps):
    return {"action": "conditional",
            "condition": {"source": "last_search_result", "operator": "found", "value": False},
            "then": list(then_steps), "else": []}


def replan_case(goal, expected, trace, remaining, current_location,
                blocked_routes=(), no_navigate=(), no_search=()):
    """One replan case; no_navigate/no_search are what a good plan must avoid."""
    state = WorldState()
    state.current_goal = goal
    state.current_location = current_location
    state.blocked_routes = list(blocked_routes)
    state.route_status = trace[-1]["status"] if trace[-1]["status"] == "BLOCKED" else "FAILED"
    state.execution_status = trace[-1]["status"]
    state.replan_count = 1
    search_results = [entry["evidence"] for entry in trace
                      if entry["step"].get("action") == "search"]
    state.last_search_result = search_results[-1] if search_results else None
    return {
        "request_type": "replan", "goal": goal, "expected": expected,
        "world_state": state.as_dict(),
        # Same bounding as the main loop's replanning block.
        "trace": state.get_failure_trace(trace, max_entries=5),
        "remaining": remaining,
        "no_navigate": list(no_navigate), "no_search": list(no_search),
    }


NAV_MEETING_ROOM = {"action": "navigate", "location": "meeting_room"}
NAV_OFFICE_A = {"action": "navigate", "location": "office_a"}
NAV_OFFICE_B = {"action": "navigate", "location": "office_b"}
NAV_LABORATORY = {"action": "navigate", "location": "laboratory"}
RETURN_HOME = {"action": "return_home"}
SEARCH_RED_BALL = {"action": "search", "object": "ball", "colour": "red"}
SEARCH_BLUE_CYLINDER = {"action": "search", "object": "cylinder", "colour": "blue"}


def report(message):
    return {"action": "report", "message": message}


REPLAN_CASES = [
    # Blocked in the middle of a multi-room search.
    replan_case(
        "Can you find the red ball somewhere in the building?",
        "search a room not yet searched (office_b or office_a), then report; "
        "no meeting_room, no new laboratory search",
        trace=[arrived("laboratory"), searched("ball", "red", "laboratory", "NOT_FOUND"),
               expanded(if_not_found(NAV_MEETING_ROOM, SEARCH_RED_BALL)),
               navigation_failed(NAV_MEETING_ROOM, "BLOCKED", "front obstacle")],
        remaining=[NAV_MEETING_ROOM, SEARCH_RED_BALL,
                   if_not_found(NAV_OFFICE_B, SEARCH_RED_BALL),
                   report("Tell the user where the red ball was found.")],
        current_location="laboratory", blocked_routes=["meeting_room"],
        no_navigate=["meeting_room"], no_search=["laboratory"],
    ),
    # Blocked on the very first step of a plain navigation task.
    replan_case(
        "Go to Office A and then come back to reception.",
        "report (or ask) that office_a cannot be reached; no navigate office_a",
        trace=[navigation_failed(NAV_OFFICE_A, "BLOCKED", "front obstacle")],
        remaining=[NAV_OFFICE_A, RETURN_HOME],
        current_location="reception", blocked_routes=["office_a"],
        no_navigate=["office_a"],
    ),
    # The way back is blocked after the search was done and reported.
    replan_case(
        "Look in the meeting room for the red ball and then head back to the front desk.",
        "report that the way back to reception is blocked; no return_home, "
        "no new meeting_room search",
        trace=[arrived("meeting_room"), searched("ball", "red", "meeting_room", "NOT_FOUND"),
               {"step": report("Tell the user whether the red ball was found."),
                "status": "SUCCEEDED", "evidence": None},
               navigation_failed(RETURN_HOME, "BLOCKED", "front obstacle")],
        remaining=[RETURN_HOME],
        current_location="meeting_room", blocked_routes=["reception"],
        no_navigate=["reception"], no_search=["meeting_room"],
    ),
    # Target already found; the next room could not be reached (no obstacle).
    replan_case(
        "Find the red ball in Office B, then drive to the lab.",
        "report that the ball was found in office_b and the laboratory could not "
        "be reached; no navigate laboratory, no new office_b search",
        trace=[arrived("office_b"), searched("ball", "red", "office_b", "FOUND"),
               navigation_failed(NAV_LABORATORY, "FAILED", "no arrival within 3000 steps")],
        remaining=[NAV_LABORATORY,
                   report("Tell the user the red ball was found in office B.")],
        current_location="office_b",
        no_navigate=["laboratory"], no_search=["office_b"],
    ),
    # Uncertain search result, then the fallback room is blocked.
    replan_case(
        "Search Office A for the blue cylinder; if it isn't there try the meeting room.",
        "report that the meeting room cannot be reached; no meeting_room, "
        "no new office_a search",
        trace=[arrived("office_a"), searched("cylinder", "blue", "office_a", "UNCERTAIN"),
               expanded(if_not_found(NAV_MEETING_ROOM, SEARCH_BLUE_CYLINDER)),
               navigation_failed(NAV_MEETING_ROOM, "BLOCKED", "front obstacle")],
        remaining=[NAV_MEETING_ROOM, SEARCH_BLUE_CYLINDER,
                   report("Tell the user whether the blue cylinder was found.")],
        current_location="office_a", blocked_routes=["meeting_room"],
        no_navigate=["meeting_room"], no_search=["office_a"],
    ),
]


def goal_case(request_type, goal, expected):
    return {"request_type": request_type, "goal": goal, "expected": expected}


ALL_CASES = [goal_case(*goal) for goal in TEST_GOALS] + REPLAN_CASES


def main():
    arguments = parse_arguments()
    cases = [case for case in ALL_CASES
             if arguments.only is None or case["request_type"] == arguments.only]
    if not cases:
        print(f"No goals of type {arguments.only!r}.")
        return 1

    interface = LLMInterface()
    if interface.backend is None:
        print(f"Planner not available: {interface.unavailable_reason}")
        return 1
    validator = PlanValidator()

    print_header(interface, len(cases))
    results = []
    for case_number, case in enumerate(cases, start=1):
        print(f"\n[{case_number}/{len(cases)}] {case['request_type']}")
        print(f"  Goal:      {case['goal']}")
        if "trace" in case:
            print("  Failure:   " + describe_step(case["trace"][-1]["step"])
                  + f" -> {case['trace'][-1]['status']}")
        print(f"  Expected:  {case['expected']}")

        result = plan_with_corrections(interface, validator, case)
        result.update(request_type=case["request_type"], goal=case["goal"],
                      expected=case["expected"])
        if "trace" in case:
            result["rule_problems"] = (check_replan_rules(result["plan"], case)
                                       if result["valid"] else None)
        results.append(result)

        print("  Final plan:")
        print_plan(result["plan"])
        if result.get("rule_problems") is not None:
            problems = result["rule_problems"]
            print("  Rule check: " + ("OK" if not problems else "; ".join(problems)))

    print_summary(results)
    save_results(interface, results)
    return 0


def parse_arguments():
    request_types = sorted({case["request_type"] for case in ALL_CASES})
    parser = argparse.ArgumentParser(description="Evaluate the planner on test goals.")
    parser.add_argument("--only", choices=request_types,
                        help="run only goals of this request type")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# The planning loop (the same steps the main controller will use)
# ----------------------------------------------------------------------------

def plan_with_corrections(interface, validator, case):
    """Ask for a plan, validate it, and ask for corrections if it is rejected.

    Replan cases make their first request with replan() from the recorded
    failure; corrections then use the same world state as that request.
    """
    goal = case["goal"]
    is_replan = "trace" in case
    # Goal cases start at reception; replan cases start where the failure left the robot.
    world_state = case["world_state"] if is_replan else WorldState().as_dict()
    attempts = []
    plan = None
    valid = False
    rejected_output = None
    errors = []

    for attempt_number in range(1, interface.max_retries + 2):
        if attempt_number > 1:
            request_kind = "correction"
        else:
            request_kind = "replan" if is_replan else "plan"
        model_failed = False
        partial_text = ""
        start_time = time.monotonic()

        try:
            if request_kind == "plan":
                plan = interface.generate_plan(goal, world_state)
            elif request_kind == "replan":
                plan = interface.replan(goal, world_state, case["trace"], case["remaining"])
            else:
                plan = interface.correct_plan(goal, world_state, rejected_output, errors)
            valid, errors = validator.validate(plan)
            outcome = "VALID" if valid else "REJECTED BY VALIDATOR"
            rejected_output = plan
        except PlanParseError as error:
            plan, valid, errors = None, False, error.errors
            outcome = "NO JSON IN ANSWER"
            rejected_output = error.raw_text
        except LLMBackendError as error:
            plan, valid, errors = None, False, [str(error)]
            outcome = f"MODEL ERROR ({type(error).__name__})"
            model_failed = True
            partial_text = getattr(error, "partial_text", "")  # only timeouts carry one

        seconds = round(time.monotonic() - start_time, 1)
        print(f"  Attempt {attempt_number} ({request_kind}, {seconds} s): {outcome}")
        for error in errors:
            print(f"      - {error}")
        if model_failed and partial_text:
            # repr() makes endless spaces or newlines visible as \n\n\n
            print(f"      Partial answer ({len(partial_text)} chars): {partial_text[:300]!r}")

        attempts.append({"request_kind": request_kind, "outcome": outcome,
                         "errors": errors, "seconds": seconds,
                         "partial_text": partial_text})
        if valid or model_failed:
            break

    return {"plan": plan, "valid": valid, "attempts": attempts}


def check_replan_rules(plan, case):
    """List the replan rules a valid plan breaks (empty list: none).

    Follows the robot's location through the plan to know which room each
    search happens in. Both conditional branches are checked from the
    location before the conditional, so this is a simple check, not a proof.
    """
    problems = []

    def note(problem):
        if problem not in problems:
            problems.append(problem)

    def walk(steps, location):
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if action in {"navigate", "return_home"}:
                target = step.get("location") if action == "navigate" else "reception"
                if target in case["no_navigate"]:
                    note(f"goes to {target}, which is blocked or unreachable")
                location = target
            elif action == "search" and location in case["no_search"]:
                note(f"searches {location} again")
            elif action == "conditional":
                walk(step.get("then"), location)
                walk(step.get("else"), location)
        return location

    walk(plan.get("steps") if isinstance(plan, dict) else None,
         case["world_state"]["current_location"])
    return problems


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------

def print_header(interface, goal_count):
    description = interface.backend.description()
    print(LINE)
    print("PLANNER EVALUATION  (genuine model output, not test fixtures)")
    print(f"Model:        {description.get('model_identifier')}")
    print(f"Settings:     temperature {description.get('temperature')}, "
          f"max_tokens {description.get('max_tokens')}, "
          f"json_mode {description.get('json_mode')}")
    print(f"Max retries:  {interface.max_retries}")
    print(f"Goals:        {goal_count}")
    print(LINE)


def print_plan(plan, indent="    "):
    """Print a plan as numbered steps. Plans are untrusted, so never crash."""
    if plan is None:
        print(f"{indent}(no plan)")
        return
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        print(f"{indent}(no step list) {json.dumps(plan)[:200]}")
        return
    print_steps(steps, indent)


def print_steps(steps, indent):
    for step_number, step in enumerate(steps, start=1):
        print(f"{indent}{step_number}. {describe_step(step)}")
        if isinstance(step, dict) and step.get("action") == "conditional":
            for branch_name in ("then", "else"):
                branch_steps = step.get(branch_name)
                if isinstance(branch_steps, list) and branch_steps:
                    print(f"{indent}   {branch_name}:")
                    print_steps(branch_steps, indent + "      ")
                else:
                    print(f"{indent}   {branch_name}: (nothing)")


def describe_step(step):
    """One short line per step, e.g. 'navigate laboratory' or 'search ball red'."""
    if not isinstance(step, dict):
        return f"(not a step) {step!r}"
    action = step.get("action", "?")
    if action == "conditional":
        condition = step.get("condition")
        if not isinstance(condition, dict):
            return f"if {condition!r}"
        return (f"if {condition.get('source')} {condition.get('operator')} "
                f"{json.dumps(condition.get('value'))}")

    step_arguments = []
    for key, value in step.items():
        if key == "action":
            continue
        if isinstance(value, str) and " " in value:
            step_arguments.append(f'"{value}"')
        else:
            step_arguments.append(str(value))
    return " ".join([action, *step_arguments])


def print_summary(results):
    print(f"\n{LINE}\nSUMMARY\n{LINE}")
    print(f"{'Request type':<16}{'Goals':>7}{'Valid':>7}{'1st try':>9}"
          f"{'Corrected':>11}{'Avg s/goal':>12}")

    request_types = list(dict.fromkeys(result["request_type"] for result in results))
    for request_type in request_types:
        group = [result for result in results if result["request_type"] == request_type]
        print(summary_row(request_type, group))
    print("-" * 62)
    print(summary_row("TOTAL", results))

    print("\nValid = passed PlanValidator. It does NOT prove the plan does what the")
    print("user asked: compare each 'Final plan' with its 'Expected' line above.")

    checked_replans = [result for result in results
                       if result.get("rule_problems") is not None]
    if checked_replans:
        rules_kept = sum(1 for result in checked_replans if not result["rule_problems"])
        print(f"\nReplan rule check: {rules_kept}/{len(checked_replans)} valid replans "
              "avoid blocked rooms and rooms already searched.")

    failed_results = [result for result in results if not result["valid"]]
    if failed_results:
        print("\nGoals without a valid plan:")
        for result in failed_results:
            last_outcome = result["attempts"][-1]["outcome"]
            print(f"  - [{result['request_type']}] {result['goal']}  ->  {last_outcome}")


def summary_row(label, results):
    goal_count = len(results)
    valid_count = sum(1 for result in results if result["valid"])
    first_try_count = sum(1 for result in results
                          if result["valid"] and len(result["attempts"]) == 1)
    corrected_count = valid_count - first_try_count
    total_seconds = sum(attempt["seconds"]
                        for result in results for attempt in result["attempts"])
    average_seconds = total_seconds / goal_count if goal_count else 0.0
    return (f"{label:<16}{goal_count:>7}{valid_count:>7}{first_try_count:>9}"
            f"{corrected_count:>11}{average_seconds:>12.1f}")


def save_results(interface, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"planner_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    results_file.write_text(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"),
        "note": "Genuine model output from evaluation/run_planner_eval.py.",
        "backend": interface.backend.description(),
        "max_retries": interface.max_retries,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved results to {results_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
