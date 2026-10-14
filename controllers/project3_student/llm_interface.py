"""Isolate all generative-model-specific code behind one interface."""
import json
from pathlib import Path
from typing import Any, Dict

class LLMInterface:
    def __init__(self, config_path=None):
        project_root = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path) if config_path else (
            project_root / "config" / "llm_config.json"
        )
        self.prompt_path = project_root / "prompts" / "planning_prompt_template.txt"
        self.config = self._load_config()
        self.client = None

        # TODO: create the selected local or online client here. Read online
        # credentials from the environment variable named in the configuration.
        # Keep all model-specific imports and response parsing in this module.

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            print(
                "[LLM] No config/llm_config.json found. Copy and edit "
                "config/llm_config.example.json before enabling planning."
            )
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def generate_plan(self, user_goal: str, world_state: dict) -> Dict[str, Any]:
        """Generate a structured symbolic plan, not robot motor commands.

        TODO:
        - construct a prompt with allowed skills and valid entities;
        - request structured output;
        - parse/repair ordinary formatting errors if appropriate;
        - return a Python dict suitable for independent validation.
        """
        raise NotImplementedError("Complete LLMInterface.generate_plan().")

    def replan(self, original_goal: str, world_state: dict,
               execution_trace: list) -> Dict[str, Any]:
        """Generate a revised remaining plan after execution failure/change."""
        # TODO: include only bounded, relevant failure evidence. A revised plan
        # is untrusted and must pass PlanValidator exactly like the first plan.
        return self.generate_plan(original_goal, world_state)
