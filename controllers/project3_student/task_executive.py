"""Cooperative executor for plans that have already passed validation."""

IDLE = "IDLE"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"


class TaskExecutive:
    """Advance at most one small unit of work per Webots controller step."""

    def __init__(self, skills, world_state):
        self.skills = skills
        self.world_state = world_state
        self.trace = []
        self.steps = []
        self.step_index = 0
        self.active = False

    def start(self, plan: dict):
        self.trace.clear()
        self.steps = list(plan.get("steps", []))
        self.step_index = 0
        self.active = True
        self.world_state.execution_status = RUNNING

    def update(self) -> str:
        """Run one non-blocking executive update.

        TODO: implement conditional branch expansion, execution timeouts,
        cancellation, and bounded failure evidence for the replanning prompt.
        """
        if not self.active:
            return IDLE
        if self.step_index >= len(self.steps):
            return self._finish(SUCCEEDED)

        step = self.steps[self.step_index]
        self.world_state.current_step = step

        if step.get("action") == "conditional":
            status = self._expand_conditional(step)
        elif self.active_step_started():
            status = self.skills.update()
        else:
            status = self.skills.start(step)
            self.trace.append({"step": step, "status": "STARTED"})

        if status == SUCCEEDED:
            self.trace[-1]["status"] = SUCCEEDED
            self.step_index += 1
        elif status == FAILED:
            self.trace[-1]["status"] = FAILED
            return self._finish(FAILED)
        return RUNNING

    def active_step_started(self) -> bool:
        return bool(
            self.trace
            and self.trace[-1].get("step") is self.steps[self.step_index]
            and self.trace[-1].get("status") == "STARTED"
        )

    def _expand_conditional(self, step: dict) -> str:
        """Replace a conditional with the selected validated branch."""
        # TODO: evaluate only the explicit, validator-approved condition grammar
        # against WorldState. Never evaluate generated expressions or code.
        self.trace.append({"step": step, "status": FAILED,
                           "evidence": "Conditional evaluation not implemented"})
        return FAILED

    def _finish(self, status: str) -> str:
        self.active = False
        self.world_state.execution_status = status
        if status == FAILED:
            self.skills.cancel()
        return status

    def cancel(self):
        self.skills.cancel()
        self.active = False
        self.world_state.execution_status = "CANCELLED"
