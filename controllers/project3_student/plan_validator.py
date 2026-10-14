"""Deterministic validation gate between generative planning and execution.

This baseline checks the supplied plan contract. Groups may extend it, but an
LLM must never be the only component deciding whether its own output is safe.
"""
from plan_schema import (
    ACTION_REQUIRED_FIELDS,
    ALLOWED_ACTIONS,
    ALLOWED_CONDITION_OPERATORS,
    ALLOWED_CONDITION_SOURCES,
    MAX_NESTING_DEPTH,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    PLAN_VERSION,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
)

class PlanValidator:
    def validate(self, plan: dict):
        errors = []
        if not isinstance(plan, dict):
            return False, ["Plan must be a dictionary/object."]

        if plan.get("version") != PLAN_VERSION:
            errors.append(f"Plan version must be {PLAN_VERSION!r}.")

        goal = plan.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            errors.append("Plan must contain a non-empty string named 'goal'.")
        elif len(goal) > MAX_TEXT_LENGTH:
            errors.append(f"Plan goal exceeds {MAX_TEXT_LENGTH} characters.")

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return False, ["Plan must contain a list named 'steps'."]

        if not steps:
            errors.append("Plan must contain at least one step.")

        counter = [0]
        self._validate_steps(steps, errors, path="steps", depth=0, counter=counter)
        if counter[0] > MAX_TOTAL_STEPS:
            errors.append(
                f"Plan contains {counter[0]} steps; maximum is {MAX_TOTAL_STEPS}."
            )
        return len(errors) == 0, errors

    def _validate_steps(self, steps, errors, path, depth, counter):
        if depth > MAX_NESTING_DEPTH:
            errors.append(f"{path}: nesting depth exceeds {MAX_NESTING_DEPTH}.")
            return

        for i, step in enumerate(steps):
            counter[0] += 1
            here = f"{path}[{i}]"
            if not isinstance(step, dict):
                errors.append(f"{here}: step must be an object.")
                continue

            action = step.get("action")
            if action == "conditional":
                self._validate_condition(step.get("condition"), errors, here)
                for branch in ("then", "else"):
                    nested = step.get(branch, [])
                    if not isinstance(nested, list):
                        errors.append(f"{here}.{branch} must be a list.")
                    else:
                        self._validate_steps(
                            nested, errors, f"{here}.{branch}", depth + 1, counter
                        )
                continue

            if action not in ALLOWED_ACTIONS:
                errors.append(f"{here}: unsupported action {action!r}.")
                continue

            for field in ACTION_REQUIRED_FIELDS[action]:
                if field not in step:
                    errors.append(f"{here}: {action!r} requires {field!r}.")

            if action == "navigate":
                loc = step.get("location")
                if loc not in VALID_LOCATIONS:
                    errors.append(f"{here}: invalid location {loc!r}.")

            if action == "search":
                obj = step.get("object")
                colour = step.get("colour")
                if obj not in SUPPORTED_OBJECTS:
                    errors.append(f"{here}: unsupported object {obj!r}.")
                if colour is not None and colour not in SUPPORTED_COLOURS:
                    errors.append(f"{here}: unsupported colour {colour!r}.")

            if action in {"ask", "report"}:
                field = "question" if action == "ask" else "message"
                value = step.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{here}.{field} must be a non-empty string.")
                elif len(value) > MAX_TEXT_LENGTH:
                    errors.append(
                        f"{here}.{field} exceeds {MAX_TEXT_LENGTH} characters."
                    )

        # TODO: add project-specific semantic checks, for example limits on
        # repeated navigation, required reporting after search, and policy for
        # contradictory or physically impossible step sequences.

    @staticmethod
    def _validate_condition(condition, errors, here):
        if not isinstance(condition, dict):
            errors.append(f"{here}.condition must be an object.")
            return
        if condition.get("source") not in ALLOWED_CONDITION_SOURCES:
            errors.append(f"{here}.condition has an unsupported source.")
        if condition.get("operator") not in ALLOWED_CONDITION_OPERATORS:
            errors.append(f"{here}.condition has an unsupported operator.")
        if "value" not in condition:
            errors.append(f"{here}.condition requires a value.")
