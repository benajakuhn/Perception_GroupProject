"""Try speech input/output and dialogue routing without Webots.

The loop below imitates the Webots main loop (one tick every 32 ms) and
prints a heartbeat once per second. If the heartbeat keeps coming while you
speak and while the robot talks, nothing in SpeechIO blocks the controller.

Run from the project root with the Python that Webots uses:
    python3 tools/try_speech.py
Press Ctrl+C to stop. Set SPEECH_INPUT=console to type instead of speaking.
"""
import sys
import time
from pathlib import Path

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from nlu_dialogue import DialogueManager  # noqa: E402
from speech_io import SpeechIO  # noqa: E402

TIME_STEP = 0.032
TICKS_PER_HEARTBEAT = 31

IDLE_WORLD = {
    "current_location": "reception",
    "current_goal": None,
    "current_step": None,
    "last_search_result": None,
    "blocked_routes": [],
    "notes": [],
    "replan_count": 0,
    "execution_status": "IDLE",
}


def handle(heard, speech, dialogue):
    # The open transcript decides the kind; while an answer is expected the
    # restricted transcript may be the only one that caught it.
    text = heard["text"] or heard["answer"] or ""
    kind = dialogue.classify(text, heard["answer"])
    print(f"[ROUTED] {kind}: {text!r}")
    if kind in {"noise", "unknown"}:
        return  # background talk or filler: do not respond
    if kind == "stop":
        dialogue.reset()
        speech.stop_speaking()
        speech.speak("Okay, stopping.")
        return
    if kind == "status":
        speech.speak(dialogue.describe_status(IDLE_WORLD))
        return

    if kind == "answer":
        followup = dialogue.consume_followup(heard["text"], heard["answer"])
        if followup["goal"] is None:
            speech.speak(followup["question"])
            return
        text = followup["goal"]

    result = dialogue.analyse_for_missing_information(text)
    if result["needs_clarification"]:
        speech.speak(result["question"])
    else:
        print(f"[GOAL FOR PLANNER] {result['goal']}")
        speech.speak(f"Okay. I would now plan: {result['goal']}")


def main():
    speech = SpeechIO()
    dialogue = DialogueManager()
    speech.speak("Hello, I am listening.")
    tick = 0
    try:
        while True:
            time.sleep(TIME_STEP)
            tick += 1
            if tick % TICKS_PER_HEARTBEAT == 0:
                print(f"[LOOP] tick {tick} | input={speech.input_mode} "
                      f"| speaking={speech.is_speaking}")
            heard = speech.poll()
            if heard:
                handle(heard, speech, dialogue)
                # Listen for short answers only while a question is pending.
                speech.expect(dialogue.expected_words())
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        speech.close()


if __name__ == "__main__":
    main()
