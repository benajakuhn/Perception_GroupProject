"""Turn raw language-model text into a plan dictionary.

The parser repairs JSON *syntax* only: Markdown fences, text around the JSON
object, trailing commas and curly quotes. It never changes plan *content*.
Missing fields, unknown actions or invalid locations are left untouched so
that PlanValidator can reject them.
"""
import json
import re

MAX_RAW_TEXT_LENGTH = 20_000

MARKDOWN_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA = re.compile(r",\s*([}\]])")
CURLY_QUOTES = {"“": '"', "”": '"'}


def parse_plan_text(raw_text: str):
    """Return (plan, errors, repairs).

    plan    -- the parsed dict, or None when parsing failed
    errors  -- why parsing failed (empty when it succeeded)
    repairs -- every syntax fix that was applied, for the logs and report
    """
    repairs = []

    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, ["Model output is empty."], repairs
    if len(raw_text) > MAX_RAW_TEXT_LENGTH:
        return None, [f"Model output exceeds {MAX_RAW_TEXT_LENGTH} characters."], repairs

    text = raw_text.strip()

    text_without_fence = _remove_markdown_fence(text)
    if text_without_fence != text:
        repairs.append("Removed Markdown code fence.")
        text = text_without_fence

    plan, json_error = _load_json(text)

    if json_error is not None:
        object_text = _find_first_json_object(text)
        if object_text is None:
            return None, ["No complete JSON object found in model output."], repairs
        if object_text != text:
            repairs.append("Extracted JSON object from surrounding text.")
            text = object_text
            plan, json_error = _load_json(text)

    if json_error is not None:
        fixed_text, syntax_fixes = _fix_common_syntax_errors(text)
        if syntax_fixes:
            fixed_plan, fixed_error = _load_json(fixed_text)
            if fixed_error is None:
                repairs.extend(syntax_fixes)
                plan, json_error = fixed_plan, None

    if json_error is not None:
        return None, [f"JSON syntax error: {json_error}"], repairs

    if not isinstance(plan, dict):
        found_type = type(plan).__name__
        return None, [f"Top-level JSON must be an object, got {found_type}."], repairs

    return plan, [], repairs


def _remove_markdown_fence(text: str) -> str:
    """Return the content of a ```json ... ``` block, or the text unchanged."""
    fence_match = MARKDOWN_FENCE.search(text)
    return fence_match.group(1) if fence_match else text


def _load_json(text: str):
    """Return (value, None) on success or (None, error message) on failure."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return None, str(error)


def _find_first_json_object(text: str):
    """Return the first balanced {...} block in the text, or None.

    Counts opening and closing braces, but ignores braces that appear inside
    JSON strings such as "message": "found {red} ball".
    """
    start_index = text.find("{")
    if start_index == -1:
        return None

    open_braces = 0
    inside_string = False
    previous_was_backslash = False

    for index in range(start_index, len(text)):
        character = text[index]

        if inside_string:
            if previous_was_backslash:
                previous_was_backslash = False
            elif character == "\\":
                previous_was_backslash = True
            elif character == '"':
                inside_string = False
            continue

        if character == '"':
            inside_string = True
        elif character == "{":
            open_braces += 1
        elif character == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[start_index:index + 1]

    return None


def _fix_common_syntax_errors(text: str):
    """Apply small syntax fixes and return (fixed_text, list of fixes made)."""
    syntax_fixes = []

    text_without_trailing_commas = TRAILING_COMMA.sub(r"\1", text)
    if text_without_trailing_commas != text:
        syntax_fixes.append("Removed trailing comma.")
        text = text_without_trailing_commas

    text_with_straight_quotes = text
    for curly_quote, straight_quote in CURLY_QUOTES.items():
        text_with_straight_quotes = text_with_straight_quotes.replace(curly_quote, straight_quote)
    if text_with_straight_quotes != text:
        syntax_fixes.append("Replaced curly quotes with straight quotes.")
        text = text_with_straight_quotes

    return text, syntax_fixes
