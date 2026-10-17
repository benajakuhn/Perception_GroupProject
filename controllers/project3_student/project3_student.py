"""Main Webots controller scaffold for the group project.

Only the main loop may access Webots devices. Slow language-model or speech
operations should be submitted to the worker and returned as ordinary data.
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
from task_executive import FAILED, SUCCEEDED, TaskExecutive
from world_state import WorldState

TIME_STEP = 32
MAX_REPLANS = 2


robot = Robot()
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice("camera")
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

worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="language")
planning_future = None


def plan_goal_off_thread(user_goal: str) -> dict:
    """Language-only work: this function must not access Webots devices."""
    ambiguity = dialogue.analyse_for_missing_information(user_goal)
    if ambiguity.get("needs_clarification"):
        return {"kind": "clarification", "question": ambiguity.get("question")}

    plan = llm.generate_plan(user_goal, world_state.as_dict())
    valid, errors = validator.validate(plan)
    if not valid:
        return {"kind": "rejected", "plan": plan, "errors": errors}
    return {"kind": "plan", "plan": plan}


def submit_goal(user_goal: str):
    """Begin planning when the controller is idle.

    TODO: call this from your non-blocking speech/text input component. Define
    policy for a new goal received while a task is running (reject, queue or
    cancel after confirmation).
    """
    global planning_future
    if planning_future is not None or executive.active:
        speech.speak("I am already working on a task.")
        return
    world_state.current_goal = user_goal
    world_state.replan_count = 0
    print("[USER GOAL]", user_goal)
    planning_future = worker.submit(plan_goal_off_thread, user_goal)


def consume_planning_result():
    """Transfer completed language output back to the Webots main thread."""
    global planning_future
    if planning_future is None or not planning_future.done():
        return
    try:
        result = planning_future.result()
    except Exception as error:  # TODO: replace with specific backend errors.
        print("[PLANNING ERROR]", repr(error))
        speech.speak("Planning failed safely.")
        planning_future = None
        return
    planning_future = None

    if result["kind"] == "clarification":
        speech.speak(result.get("question") or "Could you clarify your request?")
    elif result["kind"] == "rejected":
        print("[PLAN REJECTED]", result["errors"])
        speech.speak("I cannot safely execute that plan.")
    else:
        print("[VALIDATED PLAN]", result["plan"])
        executive.start(result["plan"])


# --- TEMPORARY TEST HARNESS FOR NAVIGATION & PERCEPTION ---
test_rooms = ["laboratory", "meeting_room", "office_b", "office_a", "reception"]
room_idx = 0
test_state = "WARMUP"

while robot.step(TIME_STEP) != -1:
    if test_state == "WARMUP":
        # Wait 1 step so GPS and Compass have valid readings before start()
        test_state = "START_NAV"

    elif test_state == "START_NAV":
        if room_idx >= len(test_rooms):
            print("[TEST COMPLETE] Visited all test rooms.")
            navigator.stop()
            break
        target = test_rooms[room_idx]
        print(f"\n[TEST] Navigating to {target}...")
        status = navigator.start(target)
        print(f"[TEST] Waypoints: {navigator.waypoints}")
        test_state = "NAVIGATING" if status == "RUNNING" else "NEXT_ROOM"

    elif test_state == "NAVIGATING":
        status = navigator.update()
        if status == "SUCCEEDED":
            print(f"[TEST] Arrived at {test_rooms[room_idx]}! Starting camera scan...")
            perception.start_search("ball", "red")
            test_state = "SEARCHING"
        elif status == "FAILED":
            print(f"[TEST] Navigation to {test_rooms[room_idx]} FAILED (obstacle or timeout).")
            test_state = "NEXT_ROOM"

    elif test_state == "SEARCHING":
        # Spin in place so the camera scans 360 degrees around the room pad
        left_motor.setVelocity(-2.0)
        right_motor.setVelocity(2.0)

        status, evidence = perception.update_search()
        if status != "RUNNING":
            navigator.stop()
            print(f"[TEST] Search result in {test_rooms[room_idx]}: {status} | {evidence}")
            test_state = "NEXT_ROOM"

    elif test_state == "NEXT_ROOM":
        room_idx += 1
        test_state = "START_NAV"

worker.shutdown(wait=False, cancel_futures=True)
