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
        elif action == "return_home":
            return self.navigator.start("reception")
        elif action == "search":
            self.perception.start_search(step["object"], step.get("colour"))
            return "RUNNING"
        elif action == "observe":
            detections = self.perception.detect()
            for det in detections:
                key = f"{det.colour} {det.object_type}".strip()
                self.memory.update(key, self.world_state.current_location, det.confidence)
            if detections:
                self.world_state.notes.append(
                    f"Observed {len(detections)} object(s) at {self.world_state.current_location}"
                )
            return "SUCCEEDED"
        elif action == "ask":
            self.speech.speak(step["question"])
            return "SUCCEEDED"
        elif action == "report":
            self.speech.speak(step["message"])
            return "SUCCEEDED"
        elif action == "stop":
            self.navigator.cancel()
            return "SUCCEEDED"
        else:
            return "FAILED"

    def update(self) -> str:
        """Advance the active skill by one Webots time step."""
        if not self.active_step:
            return "FAILED"

        action = self.active_step["action"]
        if action in {"navigate", "return_home"}:
            status = self.navigator.update()
            target = self.active_step.get("location") or "reception"
            if status == "SUCCEEDED":
                self.world_state.current_location = target
                self.world_state.route_status = "CLEAR"
            elif status == "FAILED":
                if hasattr(self.navigator, "front_is_blocked") and self.navigator.front_is_blocked():
                    self.world_state.route_status = "BLOCKED"
                    if target not in self.world_state.blocked_routes:
                        self.world_state.blocked_routes.append(target)
                    return "BLOCKED"
                else:
                    self.world_state.route_status = "FAILED"
            return status

        elif action == "search":
            status, raw_evidence = self.perception.update_search()

            # Normalize perception evidence structure across dialogue status, replanning, and condition evaluation
            if isinstance(raw_evidence, dict):
                det = raw_evidence.get("detection")
                outcome = raw_evidence.get("outcome", "NOT_FOUND")
                evidence = {
                    "result": outcome,
                    "outcome": outcome,
                    "found": bool(raw_evidence.get("found", False)),
                    "confidence": getattr(det, "confidence", 0.0),
                    "detection": det,
                }
            else:
                evidence = raw_evidence

            self.world_state.last_search_result = evidence
            if status == "SUCCEEDED":
                if isinstance(evidence, dict) and evidence.get("found"):
                    conf = evidence.get("confidence", 1.0)
                    key = " ".join(
                        value for value in (
                            self.active_step.get("colour"),
                            self.active_step.get("object"),
                        ) if value
                    )
                    self.memory.update(key, self.world_state.current_location, conf)
            return status

        elif action in {"observe", "ask", "report", "stop"}:
            return "SUCCEEDED"

        else:
            return "FAILED"

    def cancel(self):
        self.navigator.cancel()
        self.active_step = None
