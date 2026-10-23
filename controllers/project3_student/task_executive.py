"""Cooperative, non-blocking executor for validated symbolic plans.

Advances at most one small unit of work per Webots step, evaluating conditionals deterministically.
"""

IDLE = "IDLE"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"

MAX_STEP_TICKS = 3500  # Timeout limit (~112 s at 32 ms timestep)


class TaskExecutive:
    """Advance at most one small unit of work per Webots controller step."""

    def __init__(self, skills, world_state):
        self.skills = skills
        self.world_state = world_state
        self.trace = []
        self.steps = []
        self.step_index = 0
        self.active = False
        self.step_ticks = 0

    def start(self, plan: dict):
        self.trace.clear()
        self.steps = list(plan.get("steps", []))
        self.step_index = 0
        self.active = True
        self.step_ticks = 0
        self.world_state.execution_status = RUNNING

    def update(self) -> str:
        """Run one non-blocking executive update."""
        if not self.active:
            return IDLE
        if self.step_index >= len(self.steps):
            return self._finish(SUCCEEDED)

        step = self.steps[self.step_index]
        self.world_state.current_step = step

        if step.get("action") == "conditional":
            return self._expand_conditional(step)
        elif self.active_step_started():
            self.step_ticks += 1
            if self.step_ticks > MAX_STEP_TICKS:
                return self._finish(TIMEOUT, evidence=f"Exceeded {MAX_STEP_TICKS} ticks")
            status = self.skills.update()
        else:
            self.step_ticks = 0
            status = self.skills.start(step)
            self.trace.append({"step": step, "status": "STARTED", "evidence": None})

        if status == SUCCEEDED:
            self.trace[-1]["status"] = SUCCEEDED
            self.trace[-1]["evidence"] = self.world_state.last_search_result or "OK"
            self.step_index += 1
            return RUNNING
        elif status in {FAILED, TIMEOUT, BLOCKED}:
            return self._finish(status, evidence=self.world_state.last_search_result or status)
        elif status == RUNNING:
            return RUNNING
        else:
            return self._finish(FAILED, evidence=f"Unexpected status {status!r}")

    def active_step_started(self) -> bool:
        return bool(
            self.trace
            and self.trace[-1].get("step") is self.steps[self.step_index]
            and self.trace[-1].get("status") == "STARTED"
        )

    def _expand_conditional(self, step: dict) -> str:
        """Replace a conditional with the selected validated branch."""
        cond = step.get("condition", {})
        source = cond.get("source")
        op = cond.get("operator")
        val = cond.get("value")

        if source == "last_search_result" and op == "found":
            search_res = self.world_state.last_search_result
            actual_found = bool(search_res.get("found", False)) if isinstance(search_res, dict) else False
            condition_met = (actual_found == bool(val))
        elif source == "route_status" and op == "equals":
            condition_met = (self.world_state.route_status == str(val))
        else:
            condition_met = False

        branch_key = "then" if condition_met else "else"
        chosen_branch = list(step.get(branch_key, []))

        self.trace.append({
            "step": step,
            "status": "EXPANDED",
            "branch": branch_key,
            "condition_met": condition_met,
        })

        # Splice chosen branch in place of the conditional step
        self.steps[self.step_index : self.step_index + 1] = chosen_branch
        return RUNNING

    def _finish(self, status: str, evidence=None) -> str:
        self.active = False
        self.world_state.execution_status = status
        if self.trace:
            self.trace[-1]["status"] = status
            if evidence:
                self.trace[-1]["evidence"] = evidence

        if status in {FAILED, TIMEOUT, BLOCKED}:
            self.skills.cancel()
        return status

    def cancel(self):
        self.skills.cancel()
        self.active = False
        self.world_state.execution_status = CANCELLED
        self.trace.append({"step": {"action": "cancel"}, "status": CANCELLED})
