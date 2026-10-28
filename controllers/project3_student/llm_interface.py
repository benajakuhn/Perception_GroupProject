"""Isolate all generative-model-specific code behind one interface.

LLMInterface is the only planning class the rest of the controller uses.
Every request follows the same flow:

    prompt_builder -> llm_backends -> plan_parser -> location_grounding -> plan
    (build prompt)    (ask model)     (text->dict)   (nickname -> id)     (UNTRUSTED)
                      + output_schema: what the model can write at all

Before each request, prompt_builder works out the request's "situation" in
code (plan_situation / replan_situation). With "constrained_output" (config,
default true) the backend gets a JSON schema built from it, so the model can
only write what the situation allows: e.g. searching only for the object the
user named, and for replans driving only to the places still left. Location
grounding remains a fallback for backends that cannot enforce a schema.

Every model call is written to logs/llm_<date>.jsonl, with the situation
(the schema is rebuilt from it exactly, so the large schema is not logged). This class never
validates plans: the main loop passes every returned plan to PlanValidator.
All methods are slow and must run in the worker thread, never in the Webots
loop.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from llm_backends import (
    LLMBackendError,
    LLMTimeoutError,
    LLMUnavailableError,
    create_backend,
)
from location_grounding import ground_locations
from output_schema import plan_output_schema, replan_output_schema
from plan_parser import parse_plan_text
from prompt_builder import (
    build_correction_prompt,
    build_plan_prompt,
    build_replan_prompt,
    plan_situation,
    replan_situation,
)


class PlanParseError(Exception):
    """The model answered, but the answer contained no usable JSON plan.

    raw_text and errors can be passed straight to LLMInterface.correct_plan().
    """

    def __init__(self, raw_text: str, errors: list):
        super().__init__("; ".join(errors))
        self.raw_text = raw_text
        self.errors = errors


class LLMInterface:
    def __init__(self, config_path=None, backend=None):
        """Load config and template, then create the model backend.

        backend: tests pass a MockBackend here; normally it is created from
        the config. A missing model does not crash the controller: planning
        calls raise LLMUnavailableError instead, so the rest of the robot can
        still be tested.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path) if config_path else (
            self.project_root / "config" / "llm_config.json"
        )
        self.prompt_path = self.project_root / "prompts" / "planning_prompt_template.txt"
        self.unavailable_reason = None
        self.config = self._load_config()
        self.template = self.prompt_path.read_text(encoding="utf-8")

        self.max_retries = int(self.config.get("max_retries", 1))
        # False sends no schema, e.g. to compare with unconstrained output.
        self.constrained_output = bool(self.config.get("constrained_output", True))
        self.log_dir = self.project_root / self.config.get("log_dir", "logs")
        self.last_call = None

        self.backend = backend
        if self.backend is None:
            self._create_backend()

    # ------------------------------------------------------------------
    # Public methods: one per kind of request
    # ------------------------------------------------------------------

    def generate_plan(self, user_goal: str, world_state: dict) -> Dict[str, Any]:
        """First plan for a new goal."""
        prompt = build_plan_prompt(self.template, user_goal, world_state)
        situation = plan_situation(user_goal)
        return self._ask_model(prompt, "plan", situation, plan_output_schema(situation))

    def correct_plan(self, user_goal: str, world_state: dict,
                     rejected_output, errors: list) -> Dict[str, Any]:
        """New plan after the parser or PlanValidator rejected the last one.

        rejected_output is the raw text (from PlanParseError) or the rejected
        plan dict; errors are the parse errors or the validator errors.
        """
        prompt = build_correction_prompt(
            self.template, user_goal, world_state, rejected_output, errors
        )
        situation = plan_situation(user_goal)
        return self._ask_model(prompt, "correction", situation, plan_output_schema(situation))

    def replan(self, original_goal: str, world_state: dict,
               execution_trace: list, remaining_steps: list = None,
               rejected_plan=None, rejected_errors: list = None) -> Dict[str, Any]:
        """Plan for the remaining work after a step failed during execution.

        The places the revision may still drive to are worked out in code
        (prompt_builder.replan_situation) and enforced by the output schema.
        The revised plan is untrusted and must pass PlanValidator like the
        first plan. rejected_plan/rejected_errors ask for a corrected revision.
        """
        situation = replan_situation(original_goal, world_state,
                                     execution_trace, remaining_steps or [])
        prompt = build_replan_prompt(situation, rejected_plan=rejected_plan,
                                     rejected_errors=rejected_errors)
        return self._ask_model(prompt, "replan", situation, replan_output_schema(situation))

    # ------------------------------------------------------------------
    # Shared steps
    # ------------------------------------------------------------------

    def _ask_model(self, prompt: str, request_kind: str, situation: dict,
                   schema: dict) -> Dict[str, Any]:
        """Send the prompt, parse the answer, log the call, return the plan."""
        if self.backend is None:
            raise LLMUnavailableError(self.unavailable_reason)
        if not self.constrained_output:
            schema = None

        call_record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "request_kind": request_kind,
            "backend": self.backend.description(),
            "prompt": prompt,
            "situation": situation,
            "constrained": schema is not None,
            "raw_response": None,
            "parsed_plan": None,
            "parse_errors": [],
            "repairs": [],
            "backend_error": None,
            "finish_reason": None,
            "seconds": None,
        }

        # 1. Ask the model. Failed calls (e.g. timeouts) are logged too.
        start_time = time.monotonic()
        try:
            raw_text = self.backend.complete(prompt, schema=schema)
        except LLMBackendError as error:
            call_record["seconds"] = round(time.monotonic() - start_time, 2)
            call_record["backend_error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, LLMTimeoutError):
                call_record["raw_response"] = error.partial_text  # written before the stop
            self._save_call_record(call_record)
            raise
        call_record["seconds"] = round(time.monotonic() - start_time, 2)
        finish_reason = getattr(self.backend, "last_finish_reason", None)
        call_record["finish_reason"] = finish_reason

        # 2. Turn the answer text into a dict (syntax repairs only).
        plan, parse_errors, repairs = parse_plan_text(raw_text)
        if plan is None and finish_reason == "length":
            # Tells the log reader and the correction prompt why the JSON is broken.
            parse_errors.append(
                "The answer was cut off before it was complete; write a shorter plan."
            )

        # 3. Map location nicknames ("corridor") to ids ("main_corridor").
        if plan is not None:
            repairs.extend(ground_locations(plan))

        call_record["raw_response"] = raw_text
        call_record["parsed_plan"] = plan
        call_record["parse_errors"] = parse_errors
        call_record["repairs"] = repairs

        # 4. Log, then hand the plan back (or explain why there is none).
        self._save_call_record(call_record)
        if plan is None:
            raise PlanParseError(raw_text, parse_errors)
        return plan

    def _save_call_record(self, call_record: dict):
        """Append one JSON line to today's log file. Never stops planning."""
        self.last_call = call_record
        log_file = self.log_dir / f"llm_{datetime.now():%Y-%m-%d}.jsonl"
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as log:
                log.write(json.dumps(call_record, default=str) + "\n")
        except OSError as error:
            print(f"[LLM] Could not write log file {log_file}: {error}")

    def _create_backend(self):
        if not self.config:
            if self.unavailable_reason is None:
                self.unavailable_reason = "No LLM configuration (config/llm_config.json)."
            return
        try:
            self.backend = create_backend(self.config, self.project_root)
        except LLMUnavailableError as error:
            self.unavailable_reason = str(error)
            print(f"[LLM] Planning disabled: {error}")

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            print(
                "[LLM] No config/llm_config.json found. Copy and edit "
                "config/llm_config.example.json before enabling planning."
            )
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            # A typo in the config disables planning instead of the controller.
            self.unavailable_reason = f"Invalid JSON in {self.config_path.name}: {error}"
            print(f"[LLM] Planning disabled: {self.unavailable_reason}")
            return {}
