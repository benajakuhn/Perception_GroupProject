"""Tests for LLMInterface: prompt -> backend -> parser -> plan, plus logging.

Most tests use MockBackend with hand-written answers (test fixtures, not
model output). Logs are written to a temporary folder, never to logs/.
RealModelPlanTests only runs when the local model is configured.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "controllers" / "project3_student"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "llm_config.json"
sys.path.insert(0, str(CONTROLLER_DIR))

from llm_backends import (  # noqa: E402
    LLMBackendError,
    LLMTimeoutError,
    LLMUnavailableError,
    MockBackend,
)
from llm_interface import LLMInterface, PlanParseError  # noqa: E402
from plan_validator import PlanValidator  # noqa: E402
from world_state import WorldState  # noqa: E402

VALID_PLAN = {
    "version": "1.0",
    "goal": "go to the laboratory",
    "steps": [{"action": "navigate", "location": "laboratory"}],
}
VALID_PLAN_TEXT = json.dumps(VALID_PLAN)


def show(title, **values):
    """Print a labelled block so a test result can be checked by eye.

    Add -b to the unittest command to hide these prints for passing tests.
    """
    print(f"\n    --- {title} ---")
    for label, value in values.items():
        indented_value = str(value).replace("\n", "\n        ")
        print(f"    {label}: {indented_value}")


def end_of_prompt(prompt):
    """The part of the prompt that changes per call: state, revision and goal."""
    return prompt[prompt.index("Current world/task state:"):]


class LLMInterfaceTestCase(unittest.TestCase):
    """Creates an LLMInterface with a MockBackend and a temporary log folder."""

    def setUp(self):
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temporary_folder.name) / "logs"
        self.world_state = WorldState().as_dict()

    def tearDown(self):
        self.temporary_folder.cleanup()

    def make_interface(self, saved_responses, extra_config=None):
        config = {"backend": "mock", "log_dir": str(self.log_dir), "max_retries": 2}
        config.update(extra_config or {})
        config_path = Path(self.temporary_folder.name) / "llm_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.mock = MockBackend(saved_responses)
        return LLMInterface(config_path=config_path, backend=self.mock)

    def read_log_records(self):
        log_files = list(self.log_dir.glob("llm_*.jsonl"))
        self.assertEqual(len(log_files), 1)
        lines = log_files[0].read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]


class GeneratePlanTests(LLMInterfaceTestCase):
    def test_returns_parsed_plan_and_sends_goal_in_prompt(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        plan = interface.generate_plan("please go to the lab", self.world_state)
        show("generate_plan", returned_plan=plan,
             prompt_end="\n" + end_of_prompt(self.mock.received_prompts[0]))
        self.assertEqual(plan, VALID_PLAN)
        self.assertIn("please go to the lab", self.mock.received_prompts[0])

    def test_markdown_wrapped_answer_still_returns_plan(self):
        interface = self.make_interface(["```json\n" + VALID_PLAN_TEXT + "\n```"])
        plan = interface.generate_plan("go to the lab", self.world_state)
        show("markdown answer", returned_plan=plan, repairs=interface.last_call["repairs"])
        self.assertEqual(plan, VALID_PLAN)
        self.assertEqual(interface.last_call["repairs"], ["Removed Markdown code fence."])

    def test_answer_without_json_raises_plan_parse_error(self):
        interface = self.make_interface(["Sorry, I cannot plan that."])
        with self.assertRaises(PlanParseError) as caught:
            interface.generate_plan("go to the lab", self.world_state)
        show("answer without JSON", raw_text=caught.exception.raw_text,
             errors=caught.exception.errors)
        self.assertEqual(caught.exception.raw_text, "Sorry, I cannot plan that.")
        self.assertEqual(len(caught.exception.errors), 1)

    def test_max_retries_comes_from_config(self):
        interface = self.make_interface([], extra_config={"max_retries": 3})
        self.assertEqual(interface.max_retries, 3)


class CorrectAndReplanTests(LLMInterfaceTestCase):
    def test_correct_plan_sends_errors_to_model(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        rejected_plan = {"version": "1.0", "goal": "x", "steps": [{"action": "fly"}]}
        plan = interface.correct_plan("go to the lab", self.world_state, rejected_plan,
                                      ["steps[0]: unsupported action 'fly'."])
        prompt = self.mock.received_prompts[0]
        show("correct_plan", returned_plan=plan, prompt_end="\n" + end_of_prompt(prompt))
        self.assertEqual(plan, VALID_PLAN)
        self.assertIn("unsupported action 'fly'", prompt)
        self.assertEqual(interface.last_call["request_kind"], "correction")

    def test_parse_error_can_be_passed_to_correct_plan(self):
        interface = self.make_interface(["no json here", VALID_PLAN_TEXT])
        with self.assertRaises(PlanParseError) as caught:
            interface.generate_plan("go to the lab", self.world_state)
        plan = interface.correct_plan("go to the lab", self.world_state,
                                      caught.exception.raw_text, caught.exception.errors)
        show("parse error -> correct_plan", returned_plan=plan,
             second_prompt_end="\n" + end_of_prompt(self.mock.received_prompts[1]))
        self.assertEqual(plan, VALID_PLAN)
        self.assertIn("no json here", self.mock.received_prompts[1])

    def test_replan_sends_trace_and_remaining_steps(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        trace = [{"step": {"action": "navigate", "location": "laboratory"},
                  "status": "BLOCKED", "evidence": "front obstacle at ps0"}]
        remaining_steps = [{"action": "search", "object": "ball", "colour": "red"}]
        plan = interface.replan("find the red ball", self.world_state, trace, remaining_steps)
        prompt = self.mock.received_prompts[0]
        show("replan", returned_plan=plan, prompt="\n" + prompt)
        self.assertIn("front obstacle at ps0", prompt)
        self.assertIn("find the red ball", prompt)
        self.assertEqual(interface.last_call["request_kind"], "replan")


class OutputSchemaTests(LLMInterfaceTestCase):
    def test_plan_requests_send_the_plan_schema(self):
        interface = self.make_interface([VALID_PLAN_TEXT, VALID_PLAN_TEXT])
        interface.generate_plan("go to the lab", self.world_state)
        interface.correct_plan("go to the lab", self.world_state, "bad", ["No JSON."])
        for schema in self.mock.received_schemas:
            self.assertEqual(schema["properties"]["version"], {"const": "1.0"})
        self.assertTrue(interface.last_call["constrained"])

    def test_log_holds_the_situation_the_schema_was_built_from(self):
        from output_schema import plan_output_schema
        interface = self.make_interface([VALID_PLAN_TEXT])
        interface.generate_plan("find the red ball", self.world_state)
        record = self.read_log_records()[0]
        self.assertEqual(record["situation"]["search_targets"],
                         [{"object": "ball", "colour": "red"}])
        self.assertEqual(plan_output_schema(record["situation"]), self.mock.received_schemas[0])
        self.assertNotIn("output_schema", record)

    def test_replan_schema_allows_only_a_report_when_nothing_is_left(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        trace = [{"step": {"action": "navigate", "location": "laboratory"},
                  "status": "BLOCKED", "evidence": {"reason": "front obstacle"}}]
        interface.replan("go to the lab", self.world_state, trace,
                         [{"action": "navigate", "location": "laboratory"}])
        steps = self.mock.received_schemas[0]["properties"]["steps"]
        self.assertEqual(steps["maxItems"], 1)
        self.assertEqual(steps["items"]["properties"]["action"], {"const": "report"})

    def test_constrained_output_can_be_switched_off(self):
        interface = self.make_interface([VALID_PLAN_TEXT],
                                        extra_config={"constrained_output": False})
        interface.generate_plan("go to the lab", self.world_state)
        self.assertEqual(self.mock.received_schemas, [None])
        self.assertFalse(interface.last_call["constrained"])


class LoggingTests(LLMInterfaceTestCase):
    def test_every_call_is_logged_with_all_fields(self):
        interface = self.make_interface([VALID_PLAN_TEXT, "broken"])
        interface.generate_plan("go to the lab", self.world_state)
        with self.assertRaises(PlanParseError):
            interface.generate_plan("go to the lab", self.world_state)

        records = self.read_log_records()
        for record in records:
            short_record = {key: value for key, value in record.items() if key != "prompt"}
            show("log record (prompt left out)", record=json.dumps(short_record, indent=2))
        self.assertEqual(len(records), 2)
        expected_fields = {"time", "request_kind", "backend", "prompt", "situation",
                           "constrained", "raw_response",
                           "parsed_plan", "parse_errors", "repairs", "backend_error",
                           "finish_reason", "seconds"}
        self.assertEqual(set(records[0]), expected_fields)
        self.assertEqual(records[0]["parsed_plan"], VALID_PLAN)
        self.assertIsNone(records[1]["parsed_plan"])
        self.assertTrue(records[1]["parse_errors"])

    def test_mock_calls_are_labelled_in_the_log(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        interface.generate_plan("go to the lab", self.world_state)
        backend_in_log = self.read_log_records()[0]["backend"]
        show("backend in log", backend=backend_in_log)
        self.assertIn("MOCK", backend_in_log["model_identifier"])

    def test_failed_backend_call_is_logged_and_raised(self):
        interface = self.make_interface([])  # mock has no answers, so it fails
        with self.assertRaises(LLMBackendError):
            interface.generate_plan("go to the lab", self.world_state)
        record = self.read_log_records()[0]
        show("failed call in log", backend_error=record["backend_error"],
             raw_response=record["raw_response"])
        self.assertIn("MockBackend", record["backend_error"])
        self.assertIsNone(record["raw_response"])


class TimingOutBackend:
    """Fake backend that always times out after writing part of an answer."""

    def complete(self, prompt, schema=None):
        raise LLMTimeoutError("Model did not finish within 45 seconds.",
                              partial_text='{"version": "1.0", "steps": [\n\n\n\n')

    def description(self):
        return {"backend": "fake", "model_identifier": "FAKE - always times out"}


class TimeoutLoggingTests(LLMInterfaceTestCase):
    def test_partial_answer_is_logged_on_timeout(self):
        interface = self.make_interface([])
        interface.backend = TimingOutBackend()
        with self.assertRaises(LLMTimeoutError):
            interface.generate_plan("go to the lab", self.world_state)
        record = self.read_log_records()[0]
        show("timeout in log", backend_error=record["backend_error"],
             raw_response=repr(record["raw_response"]))
        self.assertIn("LLMTimeoutError", record["backend_error"])
        self.assertEqual(record["raw_response"], '{"version": "1.0", "steps": [\n\n\n\n')


class CutOffBackend(MockBackend):
    """MockBackend that reports every answer as cut off at max_tokens."""

    last_finish_reason = "length"


class CutOffAnswerTests(LLMInterfaceTestCase):
    def test_cut_off_answer_is_explained_in_parse_errors_and_log(self):
        interface = self.make_interface([])
        interface.backend = CutOffBackend(['{"version": "1.0", "steps": [{"act'])
        with self.assertRaises(PlanParseError) as caught:
            interface.generate_plan("find the red ball", self.world_state)
        record = self.read_log_records()[0]
        show("cut-off answer", errors=caught.exception.errors,
             finish_reason=record["finish_reason"])
        self.assertIn("cut off", caught.exception.errors[-1])
        self.assertEqual(record["finish_reason"], "length")


class MissingModelTests(unittest.TestCase):
    def test_invalid_config_json_does_not_crash_but_planning_raises(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            config_path = Path(temporary_folder) / "llm_config.json"
            config_path.write_text('{"backend": "llama_cpp",}', encoding="utf-8")
            interface = LLMInterface(config_path=config_path)
        with self.assertRaises(LLMUnavailableError) as caught:
            interface.generate_plan("go to the lab", WorldState().as_dict())
        show("invalid config JSON", error=caught.exception)
        self.assertIn("Invalid JSON", str(caught.exception))

    def test_missing_config_does_not_crash_but_planning_raises(self):
        interface = LLMInterface(config_path=PROJECT_ROOT / "no" / "such_config.json")
        with self.assertRaises(LLMUnavailableError) as caught:
            interface.generate_plan("go to the lab", WorldState().as_dict())
        show("no config", error=caught.exception)

    def test_missing_model_file_does_not_crash_but_planning_raises(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            config_path = Path(temporary_folder) / "llm_config.json"
            config_path.write_text(json.dumps({
                "backend": "llama_cpp", "model_path": "models/no-such-model.gguf",
            }), encoding="utf-8")
            interface = LLMInterface(config_path=config_path)
        with self.assertRaises(LLMUnavailableError) as caught:
            interface.generate_plan("go to the lab", WorldState().as_dict())
        show("no model file", error=caught.exception)
        self.assertIn("no-such-model.gguf", str(caught.exception))


class FullChainTests(LLMInterfaceTestCase):
    def test_nickname_location_is_grounded_before_validation(self):
        answer_with_nickname = json.dumps({
            "version": "1.0", "goal": "corridor, then meeting room",
            "steps": [{"action": "navigate", "location": "corridor"},
                      {"action": "navigate", "location": "meeting_room"}],
        })
        interface = self.make_interface([answer_with_nickname])
        plan = interface.generate_plan("Go to the corridor, then the meeting room.",
                                       self.world_state)
        valid, errors = PlanValidator().validate(plan)
        show("nickname answer -> grounded plan -> validator", raw=answer_with_nickname,
             plan=plan, repairs=interface.last_call["repairs"], valid=valid)
        self.assertEqual(plan["steps"][0]["location"], "main_corridor")
        self.assertIn("Grounded location 'corridor' -> 'main_corridor'.",
                      interface.last_call["repairs"])
        self.assertTrue(valid, errors)

    def test_mock_plan_passes_through_to_validator(self):
        interface = self.make_interface([VALID_PLAN_TEXT])
        plan = interface.generate_plan("go to the lab", self.world_state)
        valid, errors = PlanValidator().validate(plan)
        show("goal -> plan -> validator", plan=plan, valid=valid, errors=errors)
        self.assertTrue(valid, errors)


def real_model_is_available():
    if not LOCAL_CONFIG_PATH.is_file():
        return False
    config = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    return (
        config.get("backend") == "llama_cpp"
        and (PROJECT_ROOT / config.get("model_path", "")).is_file()
        and importlib.util.find_spec("llama_cpp") is not None
    )


@unittest.skipUnless(real_model_is_available(),
                     "needs config/llm_config.json, the model file and llama-cpp-python")
class RealModelPlanTests(LLMInterfaceTestCase):
    """Runs the real model. Its answer is genuine model output, not a fixture."""

    def test_direct_goal_gives_a_plan_the_validator_accepts(self):
        config = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
        config["log_dir"] = str(self.log_dir)
        config_path = Path(self.temporary_folder.name) / "llm_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        interface = LLMInterface(config_path=config_path)

        plan = interface.generate_plan("Go to the laboratory.", self.world_state)
        valid, errors = PlanValidator().validate(plan)
        show("real model: 'Go to the laboratory.'", plan=plan, valid=valid,
             errors=errors, seconds=interface.last_call["seconds"])
        self.assertTrue(valid, errors)


if __name__ == "__main__":
    unittest.main()
