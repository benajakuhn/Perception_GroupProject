"""Trial records (trial_log.py) and their evaluation (evaluate_trials.py)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers" / "project3_student"))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_trials import read_trials, room_checks, search_verdict  # noqa: E402
from trial_log import TrialLog  # noqa: E402
from world_state import WorldState  # noqa: E402

PLAN = {"version": "1.0", "goal": "find the red ball", "steps": []}
TRACE = [
    {"step": {"action": "navigate", "location": "laboratory"}, "status": "SUCCEEDED",
     "evidence": {"reason": "arrived", "elapsed_steps": 1999, "position": [-3.4, -1.9]}},
    {"step": {"action": "search", "object": "ball", "colour": "red"}, "status": "SUCCEEDED",
     "evidence": {"result": "FOUND", "confidence": 0.8, "location": "laboratory"}},
    {"step": {"action": "conditional"}, "status": "EXPANDED"},
    {"step": {"action": "report", "message": "x"}, "status": "SUCCEEDED",
     "evidence": {"reported": "I found the red ball in the laboratory."}},
]


def run_trial(folder):
    state = WorldState()
    state.search_results = [{"target": "red ball", "location": "laboratory",
                             "result": "FOUND"}]
    log = TrialLog(Path(folder))
    log.start("find the red ball", 10.0)
    log.plan(PLAN, "initial")
    log.steps(TRACE)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        log.finish("SUCCEEDED", 95.5, state)
    return log, printed.getvalue()


class TrialLogTests(unittest.TestCase):
    def test_record_is_written_and_printed(self):
        with tempfile.TemporaryDirectory() as folder:
            log, printed = run_trial(folder)
            files = list(Path(folder).glob("trials_*.jsonl"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text().splitlines()[0])
        self.assertTrue(printed.startswith("[TRIAL] "))
        self.assertEqual(record["outcome"], "SUCCEEDED")
        self.assertEqual(record["sim_seconds"], 85.5)
        self.assertEqual(record["reported"], "I found the red ball in the laboratory.")
        self.assertEqual([s["action"] for s in record["steps"]],
                         ["navigate", "search", "report"])   # no EXPANDED entries
        self.assertFalse(log.active)

    def test_finish_twice_writes_once(self):
        with tempfile.TemporaryDirectory() as folder:
            log, _ = run_trial(folder)
            log.finish("CANCELLED", 100.0, WorldState())
            lines = next(Path(folder).glob("trials_*.jsonl")).read_text().splitlines()
        self.assertEqual(len(lines), 1)


class EvaluateTrialsTests(unittest.TestCase):
    @staticmethod
    def record(*results, outcome="SUCCEEDED"):
        return {"outcome": outcome, "search_results": [
            {"target": "red ball", "location": room, "result": result}
            for room, result in results]}

    def test_verdicts(self):
        cases = [
            (self.record(("laboratory", "NOT_FOUND"), ("office_a", "FOUND")), "correct"),
            (self.record(("laboratory", "FOUND")), "wrong_room"),
            (self.record(("office_a", "UNCERTAIN")), "uncertain"),
            (self.record(("office_a", "NOT_FOUND")), "missed"),
            (self.record(("laboratory", "NOT_FOUND"), outcome="BLOCKED"), "not_searched"),
        ]
        for record, expected in cases:
            self.assertEqual(search_verdict(record, "office_a"), expected, record)

    def test_room_checks(self):
        counts = room_checks(self.record(("laboratory", "NOT_FOUND"), ("office_a", "FOUND")),
                             "office_a")
        self.assertEqual(counts, {"true_negative": 1, "true_positive": 1})

    def test_trials_paired_with_latest_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            _, printed = run_trial(folder)
            console = Path(folder) / "console.txt"
            console.write_text(
                "[SETUP GROUND TRUTH] seed=1 spawn=office_a_west location=office_a\n"
                "[SETUP GROUND TRUTH] seed=2 spawn=laboratory_east location=laboratory\n"
                "Robot Assistant ready.\n" + printed)
            trials = read_trials([console])
        self.assertEqual(len(trials), 1)
        record, truth_room, spawn = trials[0]
        self.assertEqual((truth_room, spawn), ("laboratory", "laboratory_east"))
        self.assertEqual(search_verdict(record, truth_room), "correct")


if __name__ == "__main__":
    unittest.main()
