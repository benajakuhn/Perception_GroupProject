"""World/task state used for explanation and replanning."""
class WorldState:
    def __init__(self):
        self.current_location = "reception"
        self.current_goal = None
        self.current_step = None
        self.last_search_result = None
        self.blocked_routes = []
        self.notes = []
        self.replan_count = 0
        self.execution_status = "IDLE"

    def as_dict(self):
        return {
            "current_location": self.current_location,
            "current_goal": self.current_goal,
            "current_step": self.current_step,
            "last_search_result": self.last_search_result,
            "blocked_routes": list(self.blocked_routes),
            "notes": list(self.notes),
            "replan_count": self.replan_count,
            "execution_status": self.execution_status,
        }
