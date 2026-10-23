"""Deterministic validation gate between generative planning and execution.

Validates untrusted plan dictionaries against schema, parameter types, semantic safety, and security rules.
"""
from plan_schema import (
    ACTION_OPTIONAL_FIELDS,
    ACTION_REQUIRED_FIELDS,
    ALLOWED_ACTIONS,
    ALLOWED_CONDITION_OPERATORS,
    ALLOWED_CONDITION_SOURCES,
    CONDITION_REQUIRED_FIELDS,
    MAX_NESTING_DEPTH,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    PLAN_VERSION,
    SUPPORTED_COLOURS,
    SUPPORTED_OBJECTS,
    VALID_LOCATIONS,
    VALID_ROUTE_STATUSES,
)

# Prohibited keywords in planning string
PROHIBITED_SUBSTRINGS = (
    "eval(", "exec(", "compile(", "__import__", "globals(", "locals(",
    "os.system", "subprocess", "shutil", "sys.", "open(", "socket",
    "setvelocity", "left_motor", "right_motor", "getdevice", "wb_",
)


class PlanValidator:
    def validate(self, plan: dict):
        errors = []
        if not isinstance(plan, dict):
            return False, ["Plan must be a dictionary/object."]

        # Check for unknown top-level keys
        extra_keys = set(plan.keys()) - {"version", "goal", "steps"}
        if extra_keys:
            errors.append(f"Plan contains unknown top-level keys: {sorted(extra_keys)}.")

        if plan.get("version") != PLAN_VERSION:
            errors.append(f"Plan version must be {PLAN_VERSION!r}.")

        goal = plan.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            errors.append("Plan must contain a non-empty string named 'goal'.")
        elif len(goal) > MAX_TEXT_LENGTH:
            errors.append(f"Plan goal exceeds {MAX_TEXT_LENGTH} characters.")
        else:
            self._scan_prohibited_text("goal", goal, errors)

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return False, ["Plan must contain a list named 'steps'."]
        if not steps:
            errors.append("Plan must contain at least one step.")

        counter = [0]
        context = {"has_searched": False, "last_location": None}
        self._validate_steps(steps, errors, path="steps", depth=0, counter=counter, context=context)

        if counter[0] > MAX_TOTAL_STEPS:
            errors.append(
                f"Plan contains {counter[0]} steps; maximum is {MAX_TOTAL_STEPS}."
            )

        return len(errors) == 0, errors

    def _validate_steps(self, steps, errors, path, depth, counter, context):
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
                self._validate_condition(step.get("condition"), errors, here, context)
                for branch in ("then", "else"):
                    if branch not in step:
                        errors.append(f"{here}: conditional missing required branch {branch!r}.")
                    else:
                        nested = step.get(branch)
                        if not isinstance(nested, list):
                            errors.append(f"{here}.{branch} must be a list.")
                        else:
                            branch_context = dict(context)
                            self._validate_steps(
                                nested, errors, f"{here}.{branch}", depth + 1, counter, branch_context
                            )
                continue

            if action not in ALLOWED_ACTIONS:
                errors.append(f"{here}: unsupported action {action!r}.")
                continue

            # Check for unknown extra keys in regular step
            allowed_fields = {"action"} | ACTION_REQUIRED_FIELDS.get(action, set()) | ACTION_OPTIONAL_FIELDS.get(action, set())
            extra_keys = set(step.keys()) - allowed_fields
            if extra_keys:
                errors.append(f"{here}: step contains unknown keys: {sorted(extra_keys)}.")

            for field in ACTION_REQUIRED_FIELDS[action]:
                if field not in step:
                    errors.append(f"{here}: {action!r} requires {field!r}.")

            # Semantic and parameter checks per action
            if action == "navigate":
                loc = step.get("location")
                if not isinstance(loc, str) or loc not in VALID_LOCATIONS:
                    errors.append(f"{here}: invalid location {loc!r}.")
                elif loc == context.get("last_location"):
                    errors.append(f"{here}: redundant consecutive navigation to {loc!r}.")
                else:
                    context["last_location"] = loc

            elif action == "return_home":
                if context.get("last_location") == "reception":
                    errors.append(f"{here}: redundant return_home; already at reception.")
                else:
                    context["last_location"] = "reception"

            elif action == "search":
                obj = step.get("object")
                colour = step.get("colour")
                if not isinstance(obj, str) or obj not in SUPPORTED_OBJECTS:
                    errors.append(f"{here}: unsupported object {obj!r}.")
                if colour is not None and (not isinstance(colour, str) or colour not in SUPPORTED_COLOURS):
                    errors.append(f"{here}: unsupported colour {colour!r}.")
                context["has_searched"] = True

            elif action in {"ask", "report"}:
                field = "question" if action == "ask" else "message"
                value = step.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{here}.{field} must be a non-empty string.")
                elif len(value) > MAX_TEXT_LENGTH:
                    errors.append(f"{here}.{field} exceeds {MAX_TEXT_LENGTH} characters.")
                else:
                    self._scan_prohibited_text(f"{here}.{field}", value, errors)

            elif action not in {"observe", "stop"}:
                errors.append(f"{here}: unhandled action validation for {action!r}.")

    def _validate_condition(self, condition, errors, here, context):
        if not isinstance(condition, dict):
            errors.append(f"{here}.condition must be an object.")
            return

        extra_keys = set(condition.keys()) - CONDITION_REQUIRED_FIELDS
        if extra_keys:
            errors.append(f"{here}.condition contains unknown keys: {sorted(extra_keys)}.")

        source = condition.get("source")
        operator = condition.get("operator")

        if source not in ALLOWED_CONDITION_SOURCES:
            errors.append(f"{here}.condition has an unsupported source {source!r}.")
        if operator not in ALLOWED_CONDITION_OPERATORS:
            errors.append(f"{here}.condition has an unsupported operator {operator!r}.")
        if "value" not in condition:
            errors.append(f"{here}.condition requires a value.")
            return

        val = condition.get("value")

        # Source-specific semantic and type checking
        if source == "last_search_result":
            if operator != "found":
                errors.append(f"{here}.condition: source 'last_search_result' requires operator 'found'.")
            if not isinstance(val, bool):
                errors.append(f"{here}.condition: operator 'found' requires a boolean value.")
            if not context.get("has_searched"):
                errors.append(f"{here}.condition: 'last_search_result' checked before any search was performed.")

        elif source == "route_status":
            if operator != "equals":
                errors.append(f"{here}.condition: source 'route_status' requires operator 'equals'.")
            if not isinstance(val, str) or val not in VALID_ROUTE_STATUSES:
                errors.append(f"{here}.condition: invalid route status value {val!r}.")

        else:
            errors.append(f"{here}.condition: unhandled condition source {source!r}.")

    def _scan_prohibited_text(self, field_path: str, text: str, errors: list):
        lowered = text.lower()
        for pattern in PROHIBITED_SUBSTRINGS:
            if pattern in lowered:
                errors.append(f"{field_path} contains prohibited content {pattern!r}.")
