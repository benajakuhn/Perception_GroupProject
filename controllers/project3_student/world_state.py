"""World/task state used for explanation and replanning."""


class WorldState:
    def __init__(self):
        self.current_location = "reception"
        self.current_goal = None
        self.current_step = None
        self.last_search_result = None
        self.route_status = "CLEAR"
        self.blocked_routes = []
        self.notes = []
        self.replan_count = 0
        self.execution_status = "IDLE"
        # Question asked by a plan's "ask" step, awaiting the user's reply.
        # Set by RobotSkills; cleared by the main loop once answered.
        self.pending_question = None
        # Outcome of every search in the current task, for truthful reports:
        # [{"target": "red ball", "location": ..., "result": FOUND|NOT_FOUND|UNCERTAIN}]
        self.search_results = []

    def as_dict(self):
        return {
            "current_location": self.current_location,
            "current_goal": self.current_goal,
            "current_step": self.current_step,
            "last_search_result": self.last_search_result,
            "route_status": self.route_status,
            "blocked_routes": list(self.blocked_routes),
            "notes": list(self.notes),
            "replan_count": self.replan_count,
            "execution_status": self.execution_status,
            "pending_question": self.pending_question,
        }

    def get_failure_trace(self, trace: list, max_entries: int = 5) -> list:
        """Extract a bounded failure trace for the replanning prompt."""
        bounded = []
        for entry in (trace or [])[-max_entries:]:
            if isinstance(entry, dict):
                bounded.append({
                    "step": entry.get("step"),
                    "status": entry.get("status"),
                    "evidence": entry.get("evidence"),
                })
            else:
                bounded.append({
                    "step": None,
                    "status": "UNKNOWN",
                    "evidence": str(entry),
                })
        return bounded
