"""Deterministic, non-blocking skills available to validated plans."""
from nlu_dialogue import join_options, location_name


class RobotSkills:
    def __init__(self, navigator, perception, memory, speech, world_state):
        self.navigator = navigator
        self.perception = perception
        self.memory = memory
        self.speech = speech
        self.world_state = world_state
        self.active_step = None
        # Evidence for the current step only; read by TaskExecutive for the trace.
        self.last_evidence = None

    def start(self, step: dict) -> str:
        """Start one already-validated symbolic action."""
        self.active_step = step
        self.last_evidence = None
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
            self.last_evidence = {"detections": len(detections)}
            return "SUCCEEDED"
        elif action == "ask":
            self.speech.speak(step["question"])
            # The main loop waits for the reply and plans again with it.
            self.world_state.pending_question = step["question"]
            self.last_evidence = {"asked": step["question"]}
            return "SUCCEEDED"
        elif action == "report":
            # The plan was written before any search ran, so its message cannot
            # know the outcome; after a search, report what the camera found.
            message = self._search_summary() or step["message"]
            self.speech.speak(message)
            self.last_evidence = {"reported": message}
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
            elif status == "BLOCKED":
                # The navigator already tried every other route in the graph.
                self.world_state.route_status = "BLOCKED"
                evidence = getattr(self.navigator, "last_evidence", None) or {}
                for location in [target] + evidence.get("unreachable", []):
                    if location not in self.world_state.blocked_routes:
                        self.world_state.blocked_routes.append(location)
            elif status in {"FAILED", "TIMEOUT"}:
                self.world_state.route_status = status
            if status != "RUNNING":
                self.last_evidence = self._navigation_evidence(target, status)
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
                    # Where the search ran; the replan prompt lists rooms already searched.
                    "location": self.world_state.current_location,
                }
            else:
                evidence = raw_evidence

            self.world_state.last_search_result = evidence
            self.last_evidence = evidence
            if status == "SUCCEEDED" and isinstance(evidence, dict):
                self.world_state.search_results.append({
                    "target": self._target_name(),
                    "location": self.world_state.current_location,
                    "result": evidence.get("result", "NOT_FOUND"),
                })
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

    def _target_name(self) -> str:
        parts = (self.active_step.get("colour"), self.active_step.get("object"))
        return " ".join(part for part in parts if part)

    def _search_summary(self):
        """Truthful sentence about this task's searches, or None if none ran."""
        results = self.world_state.search_results
        if not results:
            return None
        target = results[-1]["target"]
        results = [r for r in results if r["target"] == target]
        found = [r["location"] for r in results if r["result"] == "FOUND"]
        if found:
            return f"I found the {target} in {location_name(found[-1])}."
        sentences = []
        unsure = [r["location"] for r in results if r["result"] == "UNCERTAIN"]
        searched = [r["location"] for r in results if r["result"] == "NOT_FOUND"]
        if unsure:
            places = join_options((location_name(loc) for loc in dict.fromkeys(unsure)), "or")
            sentences.append(f"I may have seen the {target} in {places}, "
                             "but I am not sure.")
        if searched:
            places = join_options((location_name(loc) for loc in dict.fromkeys(searched)), "and")
            sentences.append(f"I searched {places} and did not find the {target} there.")
        return " ".join(sentences)

    def _navigation_evidence(self, target: str, status: str) -> dict:
        """Navigator evidence (R4) if it provides any, else what is known here."""
        evidence = {"target": target, "outcome": status}
        evidence.update(getattr(self.navigator, "last_evidence", None) or {})
        return evidence

    def cancel(self):
        self.navigator.cancel()
        self.active_step = None
