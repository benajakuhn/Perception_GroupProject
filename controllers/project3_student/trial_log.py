"""One record per task, for the evaluation (logs/trials_<date>.jsonl).

The record is built only from the controller's own execution evidence: the
goal, each validated plan, every step's status and evidence, and what the
robot reported. It never contains the setup Supervisor's spawn point; the
evaluation script (evaluation/evaluate_trials.py) compares the records with
the Supervisor's "[SETUP GROUND TRUTH]" console lines afterwards.

Each record is also printed as one "[TRIAL] {...}" console line, so a saved
Webots console log holds the ground truth and the trials in order.
"""
import json
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class TrialLog:
    def __init__(self, folder: Path = LOG_DIR):
        self.folder = folder
        self.record = None
        self.planning_started = None

    @property
    def active(self) -> bool:
        return self.record is not None

    def start(self, goal: str, sim_time: float):
        """A new task: the goal has been accepted and planning starts."""
        self.record = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "goal": goal,
            "sim_start": round(sim_time, 2),
            "plans": [],
            "steps": [],
        }
        self.planning_started = time.monotonic()

    def planning(self):
        """Planning (a replan) was sent to the worker thread."""
        self.planning_started = time.monotonic()

    def plan(self, plan: dict, kind: str):
        """A validated plan ('initial' or 'replan') starts running."""
        if not self.active:
            return
        seconds = None
        if self.planning_started is not None:
            seconds = round(time.monotonic() - self.planning_started, 1)
        self.record["plans"].append({"kind": kind, "planning_seconds": seconds,
                                     "plan": plan})

    def steps(self, trace: list):
        """The executive trace of a plan that has ended (one plan at a time)."""
        if not self.active:
            return
        for entry in trace or []:
            if isinstance(entry, dict) and entry.get("status") != "EXPANDED":
                self.record["steps"].append(_describe_step(entry))

    def finish(self, outcome: str, sim_time: float, world_state, note: str = None):
        """End the task: write the record and print it on one console line."""
        if not self.active:
            return
        record = self.record
        self.record = None
        record.update({
            "outcome": outcome,
            "sim_seconds": round(sim_time - record["sim_start"], 1),
            "replans": world_state.replan_count,
            "search_results": list(world_state.search_results),
            "reported": _last_report(record["steps"]),
        })
        if note:
            record["note"] = note
        line = json.dumps(record, default=str)
        print("[TRIAL]", line)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"trials_{datetime.now():%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as error:
            print("[TRIAL LOG] could not write:", error)


def _describe_step(entry: dict) -> dict:
    """Step, status and the evidence fields that matter for evaluation."""
    step = entry.get("step") or {}
    evidence = entry.get("evidence")
    described = {"action": step.get("action"), "status": entry.get("status")}
    for key in ("location", "object", "colour"):
        if key in step:
            described[key] = step[key]
    if isinstance(evidence, dict):
        for key in ("reason", "result", "confidence", "location", "views_done",
                    "elapsed_steps", "position", "obstacles_driven_around",
                    "blocked_paths", "reported"):
            if key in evidence:
                described["evidence_" + key if key == "location" else key] = evidence[key]
    elif evidence not in (None, "OK"):
        described["evidence"] = str(evidence)
    return described


def _last_report(steps: list):
    for step in reversed(steps):
        if step.get("reported"):
            return step["reported"]
    return None
