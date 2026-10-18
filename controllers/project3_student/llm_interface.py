"""Isolate all generative-model-specific code behind one interface.

LLMInterface is the only planning class the rest of the controller uses.
Every request follows the same flow:

    prompt_builder  ->  llm_backends  ->  plan_parser  ->  plan dict
    (build prompt)      (ask model)       (text to dict)   (UNTRUSTED)

Every model call is written to logs/llm_<date>.jsonl. This class never
validates plans: the main loop passes every returned plan to PlanValidator.
All methods are slow and must run in the worker thread, never in the Webots
loop.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from llm_backends import LLMBackendError, LLMUnavailableError, create_backend
from plan_parser import parse_plan_text
from prompt_builder import (
    build_correction_prompt,
    build_plan_prompt,
    build_replan_prompt,
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
        self.config = self._load_config()
        self.template = self.prompt_path.read_text(encoding="utf-8")

        self.max_retries = int(self.config.get("max_retries", 1))
        self.log_dir = self.project_root / self.config.get("log_dir", "logs")
        self.last_call = None

        self.backend = backend
        self.unavailable_reason = None
        if self.backend is None:
            self._create_backend()

    # ------------------------------------------------------------------
    # Public methods: one per kind of request
    # ------------------------------------------------------------------

    def generate_plan(self, user_goal: str, world_state: dict) -> Dict[str, Any]:
        """First plan for a new goal."""
        prompt = build_plan_prompt(self.template, user_goal, world_state)
        return self._ask_model(prompt, request_kind="plan")

    def correct_plan(self, user_goal: str, world_state: dict,
                     rejected_output, errors: list) -> Dict[str, Any]:
        """New plan after the parser or PlanValidator rejected the last one.

        rejected_output is the raw text (from PlanParseError) or the rejected
        plan dict; errors are the parse errors or the validator errors.
        """
        prompt = build_correction_prompt(
            self.template, user_goal, world_state, rejected_output, errors
        )
        return self._ask_model(prompt, request_kind="correction")

    def replan(self, original_goal: str, world_state: dict,
               execution_trace: list, remaining_steps: list = None) -> Dict[str, Any]:
        """Plan for the remaining work after a step failed during execution.

        Only the most recent trace entries go into the prompt. The revised
        plan is untrusted and must pass PlanValidator like the first plan.
        """
        prompt = build_replan_prompt(
            self.template, original_goal, world_state,
            execution_trace, remaining_steps or [],
        )
        return self._ask_model(prompt, request_kind="replan")

    # ------------------------------------------------------------------
    # Shared steps
    # ------------------------------------------------------------------

    def _ask_model(self, prompt: str, request_kind: str) -> Dict[str, Any]:
        """Send the prompt, parse the answer, log the call, return the plan."""
        if self.backend is None:
            raise LLMUnavailableError(self.unavailable_reason)

        call_record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "request_kind": request_kind,
            "backend": self.backend.description(),
            "prompt": prompt,
            "raw_response": None,
            "parsed_plan": None,
            "parse_errors": [],
            "repairs": [],
            "backend_error": None,
            "seconds": None,
        }

        # 1. Ask the model. Failed calls (e.g. timeouts) are logged too.
        start_time = time.monotonic()
        try:
            raw_text = self.backend.complete(prompt)
        except LLMBackendError as error:
            call_record["seconds"] = round(time.monotonic() - start_time, 2)
            call_record["backend_error"] = f"{type(error).__name__}: {error}"
            self._save_call_record(call_record)
            raise
        call_record["seconds"] = round(time.monotonic() - start_time, 2)

        # 2. Turn the answer text into a dict (syntax repairs only).
        plan, parse_errors, repairs = parse_plan_text(raw_text)
        call_record["raw_response"] = raw_text
        call_record["parsed_plan"] = plan
        call_record["parse_errors"] = parse_errors
        call_record["repairs"] = repairs

        # 3. Log, then hand the plan back (or explain why there is none).
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
        return json.loads(self.config_path.read_text(encoding="utf-8"))
