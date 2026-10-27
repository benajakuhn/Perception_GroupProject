"""Aggregate results of Webots trials from a saved console log.

Usage (from the project folder):
    python evaluation/evaluate_trials.py console.txt [more_logs.txt ...]

How to record trials:
    1. Start Webots and the world as usual. For each trial press Reset (the
       setup Supervisor places the red ball at a new random spawn point and
       prints "[SETUP GROUND TRUTH] ... location=<room>"), then give one
       spoken request, e.g. "find the red ball".
    2. After the session, copy the whole Webots console into a text file
       (or start Webots from Terminal and save the output there).
    3. Run this script on that file.

The controller prints one "[TRIAL] {...}" line per task (see
controllers/project3_student/trial_log.py), built only from its own sensors
and execution evidence. This script is the only place where those records
meet the Supervisor's ground truth: each trial is paired with the most
recent "[SETUP GROUND TRUTH]" line before it. Nothing here is fed back to the
robot.

Output: a summary on screen and one CSV row per trial in evaluation/results/.
"""
import csv
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
TIME_STEP_SECONDS = 0.032
TARGET = "red ball"

GROUND_TRUTH = re.compile(r"\[SETUP GROUND TRUTH\].*?spawn=(\S+)\s+location=(\S+)")
TRIAL = re.compile(r"\[TRIAL\] (\{.*\})\s*$")


def read_trials(paths):
    """(trial record, ground-truth room or None, spawn id or None) in log order."""
    trials = []
    truth_room = spawn = None
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = GROUND_TRUTH.search(line)
            if match:
                spawn, truth_room = match.group(1), match.group(2)
                continue
            match = TRIAL.search(line)
            if match:
                try:
                    trials.append((json.loads(match.group(1)), truth_room, spawn))
                except json.JSONDecodeError:
                    print("Skipped a trial line that is not complete JSON.")
    return trials


def search_verdict(record, truth_room):
    """Outcome of a red-ball search task compared with the ground truth.

    correct      FOUND in the right room
    wrong_room   FOUND in another room (a distractor or a false detection)
    uncertain    UNCERTAIN in the right room, never FOUND
    missed       the right room was searched but the ball was not found
    not_searched the right room was never searched (blocked, stopped, ...)
    """
    results = [r for r in record.get("search_results", []) if r.get("target") == TARGET]
    if not results:
        return None
    found = [r["location"] for r in results if r.get("result") == "FOUND"]
    if truth_room is None:
        return "no_ground_truth"
    if found:
        return "correct" if found[-1] == truth_room else "wrong_room"
    at_truth = [r["result"] for r in results if r.get("location") == truth_room]
    if "UNCERTAIN" in at_truth:
        return "uncertain"
    if at_truth:
        return "missed"
    return "not_searched"


def room_checks(record, truth_room):
    """Per searched room: true/false positive/negative for the red ball."""
    counts = Counter()
    if truth_room is None:
        return counts
    for result in record.get("search_results", []):
        if result.get("target") != TARGET:
            continue
        positive = result.get("result") == "FOUND"
        here = result.get("location") == truth_room
        counts[("true_" if positive == here else "false_")
               + ("positive" if positive else "negative")] += 1
    return counts


def main(paths):
    trials = read_trials(paths)
    if not trials:
        print("No [TRIAL] lines found. Is this a Webots console log?")
        return 1

    rows = []
    verdicts, outcomes, nav_status = Counter(), Counter(), Counter()
    checks = Counter()
    durations, plan_seconds, nav_seconds = [], [], []
    replan_tasks = replan_success = 0
    for record, truth_room, spawn in trials:
        verdict = search_verdict(record, truth_room)
        if verdict:
            verdicts[verdict] += 1
            checks += room_checks(record, truth_room)
        outcomes[record.get("outcome")] += 1
        durations.append(record.get("sim_seconds") or 0)
        plan_seconds += [p["planning_seconds"] for p in record.get("plans", [])
                         if p.get("planning_seconds") is not None]
        for step in record.get("steps", []):
            if step.get("action") in {"navigate", "return_home"}:
                nav_status[step.get("status")] += 1
                if step.get("status") == "SUCCEEDED" and step.get("elapsed_steps"):
                    nav_seconds.append(step["elapsed_steps"] * TIME_STEP_SECONDS)
        if record.get("replans"):
            replan_tasks += 1
            replan_success += record.get("outcome") == "SUCCEEDED"
        rows.append({
            "started": record.get("started"),
            "goal": record.get("goal"),
            "truth_room": truth_room,
            "truth_spawn": spawn,
            "outcome": record.get("outcome"),
            "search_verdict": verdict or "",
            "searched": " ".join(f"{r['location']}:{r['result']}"
                                 for r in record.get("search_results", [])),
            "replans": record.get("replans"),
            "sim_seconds": record.get("sim_seconds"),
            "reported": record.get("reported") or "",
        })

    print(f"Trials: {len(trials)}")
    print("Task outcomes:", dict(outcomes))
    if verdicts:
        total = sum(verdicts.values())
        print(f"Red-ball search tasks: {total}")
        for name in ("correct", "uncertain", "missed", "wrong_room", "not_searched",
                     "no_ground_truth"):
            if verdicts[name]:
                print(f"  {name:<15} {verdicts[name]:>3}  ({100 * verdicts[name] / total:.0f}%)")
        print("Room searches (red ball):", dict(checks))
        if checks["false_positive"] == 0 and checks["true_negative"]:
            print("  No room without the ball was reported as FOUND (distractors rejected).")
    print("Navigation steps:", dict(nav_status))
    if nav_seconds:
        print(f"  successful trips: median {statistics.median(nav_seconds):.0f} s "
              f"(simulated), max {max(nav_seconds):.0f} s")
    if durations:
        print(f"Task time (simulated): median {statistics.median(durations):.0f} s, "
              f"max {max(durations):.0f} s")
    if plan_seconds:
        print(f"Planning time (wall clock): median {statistics.median(plan_seconds):.0f} s, "
              f"max {max(plan_seconds):.0f} s")
    if replan_tasks:
        print(f"Tasks with replanning: {replan_tasks}, of which succeeded: {replan_success}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"trials_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with out.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-trial rows saved in {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
