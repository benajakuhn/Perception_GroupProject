"""Dialogue state and clarification support.

The generative planner may interpret language, but the controller still needs
explicit dialogue state so that incomplete requests and follow-up answers can
be handled coherently.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class DialogueState:
    pending_goal: Optional[str] = None
    awaiting_clarification: bool = False
    clarification_topic: Optional[str] = None

class DialogueManager:
    def __init__(self):
        self.state = DialogueState()

    def analyse_for_missing_information(self, text: str) -> dict:
        """Return explicit ambiguity/missing-information signals.

        TODO: implement using rules, an LLM, or a hybrid design.
        """
        return {"needs_clarification": False, "question": None}

    def consume_followup(self, text: str) -> str:
        """Combine a follow-up answer with the pending goal."""
        # TODO
        return text
