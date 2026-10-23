"""Main Webots controller for the Generative-AI Embodied Robot Assistant.

Coordinates non-blocking speech polling, off-thread generative planning, deterministic
plan validation, cooperative task execution, and sensor-grounded physical skills.
"""
from concurrent.futures import ThreadPoolExecutor

from controller import Robot

from llm_interface import LLMInterface
from navigation import Navigator
from nlu_dialogue import DialogueManager
from perception import ObjectPerception
from plan_validator import PlanValidator
from robot_skills import RobotSkills
from semantic_memory import SemanticMemory
from speech_io import SpeechIO
from task_executive import (
    BLOCKED,
    FAILED,
    SUCCEEDED,
    TIMEOUT,
    TaskExecutive,
)
from world_state import WorldState

TIME_STEP = 32
MAX_REPLANS = 2

# Hardware initialisation
robot = Robot()
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")

# Subsystems
speech = SpeechIO()
dialogue = DialogueManager()
llm = LLMInterface()
validator = PlanValidator()
navigator = Navigator(robot, left_motor, right_motor, TIME_STEP)
perception = ObjectPerception(camera, TIME_STEP)
memory = SemanticMemory()
world_state = WorldState()
skills = RobotSkills(navigator, perception, memory, speech, world_state)
executive = TaskExecutive(skills, world_state)

# Worker thread for off-thread language operations
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="language")
planning_future = None


def is_busy() -> bool:
    return planning_future is not None or executive.active


def stop_everything():
    """Cancel planning or execution at once; stopping is always allowed."""
    global planning_future
    was_busy = is_busy()
    if executive.active:
        executive.cancel()

    if planning_future is not None:
        planning_future.cancel()
        planning_future = None
        world_state.current_goal = None

    dialogue.reset()
    speech.stop_speaking()
    if was_busy:
        speech.speak("Okay, I have stopped.")
    else:
        speech.speak("Okay. I was not doing anything.")


def plan_goal_off_thread(user_goal: str) -> dict:
    """Asynchronous language-only planning in worker thread."""
    ambiguity = dialogue.analyse_for_missing_information(user_goal)
    if ambiguity.get("needs_clarification"):
        return {"kind": "clarification", "question": ambiguity.get("question")}

    try:
        plan = llm.generate_plan(user_goal, world_state.as_dict())
        valid, errors = validator.validate(plan)
        if not valid:
            plan = llm.correct_plan(user_goal, world_state.as_dict(), plan, errors)
            valid, errors = validator.validate(plan)
    except Exception as error:
        return {"kind": "error", "error": str(error)}

    if valid:
        return {"kind": "plan", "plan": plan}
    else:
        return {"kind": "rejected", "plan": plan, "errors": errors}


def replan_goal_off_thread(original_goal: str, state_snapshot: dict,
                           failure_trace: list, remaining_steps: list) -> dict:
    """Asynchronous replanning after physical step failure."""
    try:
        plan = llm.replan(original_goal, state_snapshot, failure_trace, remaining_steps)
    except Exception as error:
        return {"kind": "error", "error": str(error)}

    valid, errors = validator.validate(plan)
    if valid:
        return {"kind": "plan", "plan": plan}
    else:
        return {"kind": "rejected", "plan": plan, "errors": errors}


def submit_goal(user_goal: str):
    """Initiate planning for a new user request."""
    global planning_future
    if planning_future is not None or executive.active:
        speech.speak("I am already working on a task.")
        return

    world_state.current_goal = user_goal
    world_state.replan_count = 0
    print("[USER GOAL]", user_goal)
    planning_future = worker.submit(plan_goal_off_thread, user_goal)


def handle_utterance(heard: dict):
    """Route one utterance from speech.poll(); runs on the main thread."""
    text = heard.get("text") or heard.get("answer") or ""
    kind = dialogue.classify(text, heard.get("answer"))
    print(f"[HEARD:{kind.upper()}] {text!r}")

    if kind in {"noise", "unknown"}:
        return
    elif kind == "stop":
        stop_everything()
        return
    elif kind == "status":
        state = world_state.as_dict()
        if not is_busy() and state["execution_status"] == "IDLE":
            state["current_goal"] = None
        speech.speak(dialogue.describe_status(state))
        return
    elif is_busy():
        speech.speak(f"I am busy with: {world_state.current_goal}. Say stop if you want me to cancel it.")
        return
    elif kind == "answer":
        followup = dialogue.consume_followup(heard.get("text", ""), heard.get("answer", ""))
        if followup.get("goal") is None:
            speech.speak(followup.get("question"))
            return
        text = followup["goal"]

    result = dialogue.analyse_for_missing_information(text)
    if result.get("needs_clarification"):
        speech.speak(result.get("question"))
        return

    goal_text = result.get("goal", text)
    speech.speak(f"Okay. I will work on: {goal_text}")
    submit_goal(goal_text)


def consume_planning_result():
    """Transfer completed asynchronous planning output back to main thread."""
    global planning_future
    if planning_future is None or not planning_future.done():
        return

    try:
        result = planning_future.result()
    except Exception as error:
        print("[PLANNING ERROR]", repr(error))
        speech.speak("Planning failed safely.")
        planning_future = None
        return
    planning_future = None

    kind = result.get("kind")
    if kind == "clarification":
        speech.speak(result.get("question") or "Could you clarify your request?")
    elif kind == "rejected":
        print("[PLAN REJECTED]", result.get("errors"))
        speech.speak("I cannot safely execute that plan.")
    elif kind == "error":
        print("[PLANNING BACKEND ERROR]", result.get("error"))
        speech.speak("Planning encountered an error.")
    elif kind == "plan":
        print("[VALIDATED PLAN]", result["plan"])
        speech.speak("Plan approved. Starting execution.")
        executive.start(result["plan"])
    else:
        print("[UNEXPECTED PLANNING RESULT]", result)
        speech.speak("Planning returned an unexpected result.")


# --- Main Webots Controller Loop ---
print("Embodied Robot Assistant ready.")

while robot.step(TIME_STEP) != -1:
    # 1. Collect planning results
    consume_planning_result()

    # 2. Advance active task execution
    if executive.active:
        status = executive.update()

        if status == SUCCEEDED:
            print("[TASK SUCCEEDED]", executive.trace)
            speech.speak("Task complete.")

        elif status in {FAILED, TIMEOUT, BLOCKED}:
            print(f"[TASK {status}]", executive.trace)

            # Replanning loop on step failure (bounded by MAX_REPLANS)
            if world_state.replan_count < MAX_REPLANS and planning_future is None:
                world_state.replan_count += 1
                speech.speak(f"Step {status.lower()}. Replanning remainder of task.")
                print(f"[REPLAN ATTEMPT {world_state.replan_count}/{MAX_REPLANS}]")

                remaining = executive.steps[executive.step_index:]
                failure_trace = world_state.get_failure_trace(executive.trace, max_entries=5)

                planning_future = worker.submit(
                    replan_goal_off_thread,
                    world_state.current_goal,
                    world_state.as_dict(),
                    failure_trace,
                    remaining,
                )
            else:
                speech.speak(f"I could not complete the task safely due to {status.lower()}.")

    # 3. Speech queue polling (non-blocking) and intent routing
    heard = speech.poll()
    if heard:
        handle_utterance(heard)
        speech.expect(dialogue.expected_words())

worker.shutdown(wait=False, cancel_futures=True)
speech.close()
