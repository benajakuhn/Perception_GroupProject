"""Deterministic, non-blocking skills available to validated plans."""
class RobotSkills:
    def __init__(self, navigator, perception, memory, speech, world_state):
        self.navigator = navigator
        self.perception = perception
        self.memory = memory
        self.speech = speech
        self.world_state = world_state
        self.active_step = None

    def start(self, step: dict) -> str:
        """Start one already-validated symbolic action."""
        self.active_step = step
        action = step["action"]
        if action == "navigate":
            return self.navigator.start(step["location"])
        if action == "return_home":
            return self.navigator.start("reception")
        if action == "search":
            self.perception.start_search(step["object"], step.get("colour"))
            return "RUNNING"
        if action == "observe":
            # TODO: update semantic memory from the current observation.
            return "SUCCEEDED"
        if action == "ask":
            self.speech.speak(step["question"])
            return "SUCCEEDED"
        if action == "report":
            self.speech.speak(step["message"])
            return "SUCCEEDED"
        if action == "stop":
            self.navigator.cancel()
            return "SUCCEEDED"
        return "FAILED"

    def update(self) -> str:
        """Advance the active skill by one Webots time step."""
        if not self.active_step:
            return "FAILED"
        action = self.active_step["action"]
        if action in {"navigate", "return_home"}:
            status = self.navigator.update()
            if status == "SUCCEEDED":
                self.world_state.current_location = (
                    self.active_step.get("location") or "reception"
                )
            return status
        if action == "search":
            status, evidence = self.perception.update_search()
            self.world_state.last_search_result = evidence
            if status == "SUCCEEDED":
                key = " ".join(
                    value for value in (
                        self.active_step.get("colour"),
                        self.active_step.get("object"),
                    ) if value
                )
                self.memory.update(key, self.world_state.current_location,
                                   evidence.confidence)
            return status
        return "SUCCEEDED"

    def cancel(self):
        self.navigator.cancel()
        self.active_step = None
