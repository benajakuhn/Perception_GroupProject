"""Run the planner on test goals and print how the real model does.

Usage (from the project folder):
    python evaluation/run_planner_eval.py
    python evaluation/run_planner_eval.py --only conditional

Each goal goes through the same steps the controller will use:
    generate_plan -> PlanValidator -> if rejected: correct_plan
    (at most max_retries corrections, from config/llm_config.json)

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


def main():
    arguments = parse_arguments()
    goals = [goal for goal in TEST_GOALS
             if arguments.only is None or goal[0] == arguments.only]
    if not goals:
        print(f"No goals of type {arguments.only!r}.")
        return 1

    interface = LLMInterface()
    if interface.backend is None:
        print(f"Planner not available: {interface.unavailable_reason}")
        return 1
    validator = PlanValidator()

    print_header(interface, len(goals))
    results = []
    for goal_number, (request_type, goal, expected) in enumerate(goals, start=1):
        print(f"\n[{goal_number}/{len(goals)}] {request_type}")
        print(f"  Goal:      {goal}")
        print(f"  Expected:  {expected}")

        result = plan_with_corrections(interface, validator, goal)
        result.update(request_type=request_type, goal=goal, expected=expected)
        results.append(result)

        print("  Final plan:")
        print_plan(result["plan"])

    print_summary(results)
    save_results(interface, results)
    return 0


def parse_arguments():
    request_types = sorted({request_type for request_type, _, _ in TEST_GOALS})
    parser = argparse.ArgumentParser(description="Evaluate the planner on test goals.")
    parser.add_argument("--only", choices=request_types,
                        help="run only goals of this request type")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# The planning loop (the same steps the main controller will use)
# ----------------------------------------------------------------------------

def plan_with_corrections(interface, validator, goal):
    """Ask for a plan, validate it, and ask for corrections if it is rejected."""
    world_state = WorldState().as_dict()  # every goal starts at reception
    attempts = []
    plan = None
    valid = False
    rejected_output = None
    errors = []

    for attempt_number in range(1, interface.max_retries + 2):
        request_kind = "plan" if attempt_number == 1 else "correction"
        model_failed = False
        partial_text = ""
        start_time = time.monotonic()

        try:
            if request_kind == "plan":
                plan = interface.generate_plan(goal, world_state)
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
