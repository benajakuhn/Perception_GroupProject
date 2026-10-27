"""Dialogue state and clarification support.

The generative planner may interpret language, but the controller still needs
explicit dialogue state so that incomplete requests and follow-up answers can
be handled coherently.

All methods are fast, rule-based and free of Webots access. Call them from the
main controller loop so that DialogueState is only touched by one thread.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# Multi-word phrases are replaced by one canonical token before tokenising.
# Longer phrases are replaced first, so "main corridor" wins over "corridor".
# Several entries are consistent Vosk mishearings measured in our tests
# ("Office A" -> "officer", "Office B" -> "of his beer").
LOCATION_PHRASES = {
    "office a": "office_a",
    "office ay": "office_a",
    "office eh": "office_a",
    "office hey": "office_a",
    "officer": "office_a",
    "office b": "office_b",
    "office be": "office_b",
    "office bee": "office_b",
    "office bear": "office_b",
    "office beer": "office_b",
    "of his beer": "office_b",
    "of his ear": "office_b",
    "of his be": "office_b",
    "have his beer": "office_b",
    "laboratory": "laboratory",
    "lab": "laboratory",
    "meeting room": "meeting_room",
    "conference room": "meeting_room",
    "meeting": "meeting_room",
    "reception": "reception",
    "front desk": "reception",
    "lobby": "reception",
    "entrance": "reception",
    "main corridor": "main_corridor",
    "corridor": "main_corridor",
    "hallway": "main_corridor",
    "hall": "main_corridor",
}

LOCATION_NAMES = {
    "reception": "reception",
    "office_a": "Office A",
    "office_b": "Office B",
    "laboratory": "the laboratory",
    "meeting_room": "the meeting room",
    "main_corridor": "the main corridor",
}

# Multi-word mishearings ("red ball" heard as "read all").
PHRASE_SYNONYMS = {
    "read all": "red ball",
    "red all": "red ball",
}

# Single-word synonyms and mishearings ("red" is often recognised as "read").
WORD_SYNONYMS = {
    "read": "red",
    "blew": "blue",
    "bull": "ball",
    "signed": "find",
    "sign": "find",
    "books": "box",
    "bowl": "ball",
    "ceiling": "cylinder",
    "balls": "ball",
    "sphere": "ball",
    "boxes": "box",
    "cube": "box",
    "crate": "box",
    "cylinders": "cylinder",
    "tube": "cylinder",
    "pillar": "cylinder",
    "column": "cylinder",
}

# Colours a user may say that the robot cannot search for.
UNSUPPORTED_COLOURS = {
    "purple", "orange", "pink", "black", "white", "brown", "grey", "gray",
}

SEARCH_WORDS = {
    "find", "search", "look", "locate", "seek", "check", "spot", "detect",
    "scan", "where",
}
NAVIGATION_WORDS = {
    "go", "drive", "move", "navigate", "head", "travel", "visit", "come",
    "return", "take", "walk",
}
RETURN_WORDS = {"home", "back", "return"}

# Background speech (a video, people talking nearby) is long and only now and
# then contains a task word. A command is a short sentence, and one without a
# place or object ("go to ...", heard badly) must start with its verb.
MAX_GOAL_WORDS = 20
COMMAND_VERBS = (SEARCH_WORDS | NAVIGATION_WORDS) - {"where"}
LEADING_POLITE_WORDS = {
    "please", "can", "could", "would", "will", "you", "robot", "now",
    "ok", "okay", "hey", "and", "then", "so",
}

# Any of these cancels the active task. Stopping is always safe, so a false
# positive ("stop at the lab") is preferred over missing a real stop request.
STOP_PATTERN = re.compile(
    r"\b(stop|cancel|halt|abort|freeze|pause|never mind|nevermind)\b"
)
STATUS_PATTERN = re.compile(
    r"\b(what are you (doing|up to)|why|status|where are you|progress"
    r"|hows it going|how is it going|whats (happening|going on)"
    r"|what is (happening|going on)|explain)\b"
)
FILLER_WORDS = {"the", "a", "an", "uh", "um", "huh", "hmm", "oh", "ah", "er"}

# Words the recogniser may hear while a clarification answer is expected.
# "the" is left out of the office answers because Vosk confuses it with "B".
ANSWER_FILLERS = ["one", "please", "i", "mean", "it", "is", "that"]
ANSWER_STOP_WORDS = ["stop", "cancel", "halt", "abort", "never", "mind"]
OFFICE_ANSWERS = {
    "office_a": {"a", "ay", "eh", "hey", "air", "first"},
    "office_b": {"b", "be", "bee", "bear", "beer", "second"},
}

# A question asked by a plan's "ask" step. Any reply counts as the answer
# ("yes", "no thanks", "go to the lab instead"); the planner interprets it.
PLAN_QUESTION = "plan_question"
PLAN_ANSWER_WORDS = [
    "yes", "yeah", "yep", "sure", "okay", "ok", "please", "do", "it",
    "no", "nope", "not", "dont", "thanks", "thank", "you",
]
# A reply made only of these words, with at least one DECLINE_WORD, is a plain
# "no": handled here, since a small planner model often reads "no thanks" as yes.
DECLINE_WORDS = {"no", "nope", "nah", "not", "dont"}
DECLINE_FILLERS = {"thanks", "thank", "you", "please", "ok", "okay", "i", "do",
                   "that", "it", "want", "to", "rather", "the", "a", "need"}
DECLINE_REPLY = "Okay, I will leave it there."
# A reply made only of these words, with at least one ACCEPT_WORD, is a "yes".
ACCEPT_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "alright", "fine",
                "ahead", "absolutely", "definitely"}
ACCEPT_FILLERS = {"please", "do", "it", "that", "go", "sounds", "good", "great",
                  "thanks", "thank", "you", "i", "would", "like", "the", "a"}
# The offer in a plan's question: "Should I drive to the meeting room instead?"
OFFER_PATTERN = re.compile(
    r"\b(?:should|shall|can|may) I (.+?)(?:\s+instead)?\s*\?\s*$"
    r"|\bwant me to (.+?)(?:\s+instead)?\s*\?\s*$",
    re.IGNORECASE,
)
# Leading words of a reply such as "no, go to the lab instead".
REPLY_LEAD_PATTERN = re.compile(
    r"^\s*(?:(?:no|nope|nah|actually|well|but|ok|okay|then|rather)\b[\s,.!]*)+",
    re.IGNORECASE,
)
# Goals built from a plan question start with this; they need no further
# clarification, since the planner already asked what it needed to know.
FOLLOWUP_PREFIX = "Earlier request:"
MAX_FOLLOWUP_PART_LENGTH = 200

MAX_CLARIFICATION_ATTEMPTS = 2


@dataclass
class DialogueState:
    pending_goal: Optional[str] = None
    awaiting_clarification: bool = False
    clarification_topic: Optional[str] = None
    question: Optional[str] = None
    attempts: int = 0


def normalise(text: str) -> str:
    """Lower-case, strip punctuation and map location phrases to tokens."""
    text = text.lower().replace("_", " ").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for phrase in sorted(LOCATION_PHRASES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", LOCATION_PHRASES[phrase], text)
    for phrase, replacement in PHRASE_SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)
    return text


def location_name(location: str) -> str:
    return LOCATION_NAMES.get(location, location.replace("_", " "))


# Names the planner's location grounding (location_grounding.py) recognises.
PLANNER_LOCATION_NAMES = {
    "reception": "reception",
    "office_a": "Office A",
    "office_b": "Office B",
    "laboratory": "laboratory",
    "meeting_room": "meeting room",
    "main_corridor": "main corridor",
}
_PLANNER_REPLACEMENTS = {
    **{phrase: PLANNER_LOCATION_NAMES[loc] for phrase, loc in LOCATION_PHRASES.items()},
    **PHRASE_SYNONYMS,
    **WORD_SYNONYMS,
}
# One pass, longest phrase first, so "meeting room" is not turned into
# "meeting room room" by the shorter "meeting".
_PLANNER_PATTERN = re.compile(
    r"\b(" + "|".join(
        re.escape(phrase).replace(r"\ ", r"\s+")
        for phrase in sorted(_PLANNER_REPLACEMENTS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)


def planner_text(text: str) -> str:
    """Rewrite nicknames and known mishearings into names the planner knows.

    "go to officer" -> "go to Office A", "find the read ball" -> "find the
    red ball". Punctuation and word order are kept, since they matter for
    sequences and conditions.
    """
    def replace(match):
        key = " ".join(match.group(0).lower().split())
        return _PLANNER_REPLACEMENTS[key]
    return _PLANNER_PATTERN.sub(replace, text)


def join_options(options, conjunction: str = "or") -> str:
    options = list(options)
    if len(options) == 1:
        return options[0]
    return ", ".join(options[:-1]) + f" {conjunction} " + options[-1]


class DialogueManager:
    def __init__(self):
        self.state = DialogueState()
        objects = json.loads(
            (CONFIG_DIR / "objects.json").read_text(encoding="utf-8")
        )
        locations = json.loads(
            (CONFIG_DIR / "locations.json").read_text(encoding="utf-8")
        )
        self.object_types = list(objects["object_types"])
        self.colours = list(objects["colours"])
        self.locations = list(locations)
        # Last colour mentioned for each object type, e.g. {"ball": "red"}.
        # Lets "find the ball" follow an earlier "the red ball" without asking.
        self.context = {}

    # ------------------------------------------------------------------
    # Text analysis
    # ------------------------------------------------------------------
    def extract_slots(self, text: str) -> dict:
        tokens = [WORD_SYNONYMS.get(t, t) for t in normalise(text).split()]
        token_set = set(tokens)
        pairs = {}
        for previous, token in zip(tokens, tokens[1:]):
            if token in self.object_types and previous in self.colours:
                pairs.setdefault(token, previous)
        locations = [t for t in tokens if t in self.locations]
        return {
            "tokens": tokens,
            "colours": [t for t in tokens if t in self.colours],
            "unsupported_colours": [t for t in tokens if t in UNSUPPORTED_COLOURS],
            "objects": [t for t in tokens if t in self.object_types],
            "pairs": pairs,
            "locations": locations,
            "office_ambiguous": "office" in token_set and not (
                {"office_a", "office_b"} & token_set
            ),
            "search_intent": bool(SEARCH_WORDS & token_set),
            "navigation_intent": bool(NAVIGATION_WORDS & token_set),
            "return_intent": bool(RETURN_WORDS & token_set),
        }

    def is_relevant(self, slots: dict) -> bool:
        """True if the utterance mentions anything the robot can act on."""
        return bool(
            slots["search_intent"] or slots["navigation_intent"]
            or slots["return_intent"] or slots["locations"]
            or slots["objects"] or slots["colours"]
            or slots["unsupported_colours"] or slots["office_ambiguous"]
        )

    def looks_like_command(self, slots: dict) -> bool:
        """False for background speech that happens to contain a task word."""
        tokens = slots["tokens"]
        if len(tokens) > MAX_GOAL_WORDS:
            return False
        if (slots["locations"] or slots["objects"] or slots["colours"]
                or slots["unsupported_colours"] or slots["office_ambiguous"]):
            return True
        leading = [t for t in tokens if t not in LEADING_POLITE_WORDS]
        return bool(leading) and leading[0] in COMMAND_VERBS | RETURN_WORDS

    def classify(self, text: str, answer: str = None) -> str:
        """Route an utterance: 'noise', 'stop', 'status', 'answer',
        'unknown' or 'goal'.

        While a clarification is pending, an utterance without an action verb
        is treated as the answer; one with an action verb replaces the goal.
        'unknown' marks speech with no task words at all, such as background
        conversation, which the robot should not respond to. 'answer' is the
        expected-word transcript, so a bare "A" is not mistaken for filler.
        """
        if (self.state.awaiting_clarification and answer
                and self._answer_detail(answer)
                and not STOP_PATTERN.search(normalise(text))):
            return "answer"
        normalised = normalise(text)
        if not normalised or set(normalised.split()) <= FILLER_WORDS:
            return "noise"
        if STOP_PATTERN.search(normalised):
            return "stop"
        if STATUS_PATTERN.search(normalised):
            return "status"
        if self.awaiting_plan_answer:
            return "answer"
        slots = self.extract_slots(text)
        if self.state.awaiting_clarification:
            if not (slots["search_intent"] or slots["navigation_intent"]):
                return "answer"
        if not self.is_relevant(slots) or not self.looks_like_command(slots):
            return "unknown"
        return "goal"

    @property
    def awaiting_plan_answer(self) -> bool:
        return (self.state.awaiting_clarification
                and self.state.clarification_topic == PLAN_QUESTION)

    def expect_plan_answer(self, goal: str, question: str):
        """Wait for the reply to a question asked by a plan's "ask" step."""
        self.state = DialogueState(
            pending_goal=goal or "",
            awaiting_clarification=True,
            clarification_topic=PLAN_QUESTION,
            question=question,
        )

    def expected_words(self):
        """Vocabulary for the recogniser while an answer is awaited, or None.

        A small word list makes short answers such as "red" or "Office B"
        far more reliable than open recognition, which often hears "read".
        """
        if not self.state.awaiting_clarification:
            return None
        topic = self.state.clarification_topic
        if topic == PLAN_QUESTION:
            # Open recognition still runs, so longer replies are heard too.
            words = list(PLAN_ANSWER_WORDS)
        elif topic == "colour":
            words = self.colours + self.object_types + ["the"]
        elif topic == "object":
            words = (self.object_types + self.colours
                     + ["the", "a", "sphere", "cube"])
        elif topic == "office":
            words = ["office"] + sorted(set().union(*OFFICE_ANSWERS.values()))
        else:
            words = sorted({
                word for phrase in LOCATION_PHRASES
                for word in phrase.split()
            } | {"the", "to"})
        return words + ANSWER_FILLERS + ANSWER_STOP_WORDS + ["[unk]"]

    def analyse_for_missing_information(self, text: str) -> dict:
        """Return explicit ambiguity/missing-information signals.

        The result always has 'needs_clarification' and 'question'. 'goal' is
        the text to plan from, possibly completed from dialogue context, with
        nicknames and mishearings rewritten by planner_text().
        """
        self.state = DialogueState()
        if text.startswith(FOLLOWUP_PREFIX):
            return {"needs_clarification": False, "question": None, "goal": text}
        slots = self.extract_slots(text)
        colour_list = join_options(self.colours)

        if slots["unsupported_colours"] and not slots["colours"]:
            return self._ask(
                text, "colour",
                f"I can only look for {colour_list} objects. "
                "Which colour do you mean?",
            )

        if slots["objects"] and not slots["colours"]:
            obj = slots["objects"][0]
            remembered = self.context.get(obj)
            if remembered:
                text = f"{text} ({remembered} {obj})"
            else:
                return self._ask(
                    text, "colour",
                    f"Which colour {obj} should I look for: {colour_list}?",
                )

        if slots["search_intent"] and not slots["objects"]:
            object_list = join_options(f"a {o}" for o in self.object_types)
            if slots["colours"]:
                question = (
                    f"Which {slots['colours'][0]} object do you mean: "
                    f"{object_list}?"
                )
            else:
                question = f"What should I look for? For example {object_list}."
            return self._ask(text, "object", question)

        if slots["office_ambiguous"]:
            return self._ask(text, "office", "Do you mean Office A or Office B?")

        if (
            slots["navigation_intent"]
            and not slots["locations"]
            and not slots["return_intent"]
            and not slots["search_intent"]
        ):
            places = join_options(location_name(loc) for loc in self.locations)
            return self._ask(
                text, "location", f"Where should I go? I know {places}."
            )

        self.context.update(slots["pairs"])
        return {"needs_clarification": False, "question": None,
                "goal": planner_text(text)}

    def _ask(self, text: str, topic: str, question: str) -> dict:
        self.state = DialogueState(
            pending_goal=text,
            awaiting_clarification=True,
            clarification_topic=topic,
            question=question,
        )
        return {"needs_clarification": True, "question": question, "goal": None}

    # ------------------------------------------------------------------
    # Follow-up answers
    # ------------------------------------------------------------------
    def consume_followup(self, text: str, answer: str = None) -> dict:
        """Combine a follow-up answer with the pending goal.

        'answer' is the transcript from the expected-word recogniser, if any;
        it is tried before the open transcript 'text'. Returns {'goal':
        combined goal or None, 'question': text to say or None}. Re-run
        analyse_for_missing_information() on a returned goal, since the
        combined request may still be missing something else.
        """
        if not self.state.awaiting_clarification:
            return {"goal": text, "question": None}
        if self.awaiting_plan_answer:
            return self._consume_plan_answer(text, answer)

        pending = self.state.pending_goal
        for candidate in (answer, text):
            detail = self._answer_detail(candidate) if candidate else None
            if detail:
                if self.state.clarification_topic == "colour":
                    for colour in UNSUPPORTED_COLOURS:
                        pending = re.sub(rf"\b{colour}\b\s*", "", pending,
                                         flags=re.IGNORECASE)
                self.state = DialogueState()
                return {"goal": f"{pending.strip()} ({detail})", "question": None}

        return self._retry_answer()

    def _consume_plan_answer(self, text: str, answer: str = None) -> dict:
        """Turn the reply to a plan question into an ordinary request.

        - "no", "no thanks": nothing to plan (DECLINE_REPLY).
        - "yes", "sure": the offer in the question becomes the request
          ("Should I drive to the meeting room instead?" -> "drive to the
          meeting room").
        - anything else ("no, go to the lab instead"): the reply itself is the
          new request and gets the usual clarification checks.
        The planner therefore only sees ordinary requests, which it handles far
        more reliably than a question-and-answer transcript.

        The open transcript is used when it mentions anything task-related;
        otherwise the yes/no recogniser's transcript is preferred, as it hears
        short replies more reliably.
        """
        reply = text
        if answer and not self.is_relevant(self.extract_slots(text or "")):
            reply = answer
        reply = planner_text((reply or "").strip())
        if not reply:
            return self._retry_answer()
        words = set(normalise(reply).split())
        if words & DECLINE_WORDS and words <= DECLINE_WORDS | DECLINE_FILLERS:
            self.state = DialogueState()
            return {"goal": None, "question": DECLINE_REPLY, "declined": True}

        if words & ACCEPT_WORDS and words <= ACCEPT_WORDS | ACCEPT_FILLERS:
            offer = OFFER_PATTERN.search(self.state.question or "")
            if offer:
                self.state = DialogueState()
                return {"goal": (offer.group(1) or offer.group(2)).strip(),
                        "question": None}
        else:
            request = REPLY_LEAD_PATTERN.sub("", reply).strip()
            if not request or not self.is_relevant(self.extract_slots(request)):
                return self._retry_answer()
            self.state = DialogueState()
            return {"goal": request, "question": None}

        # A "yes" to a question without a recognisable offer: give the planner
        # the question and the answer.
        def part(value):
            return (value or "").strip().rstrip(".")[:MAX_FOLLOWUP_PART_LENGTH]

        goal = (f"{FOLLOWUP_PREFIX} {part(self.state.pending_goal)}. "
                f"I asked: {part(self.state.question)} "
                f"The user answered: {part(reply)}.")
        self.state = DialogueState()
        return {"goal": goal, "question": None, "ready": True}

    def _retry_answer(self) -> dict:
        self.state.attempts += 1
        if self.state.attempts >= MAX_CLARIFICATION_ATTEMPTS:
            self.state = DialogueState()
            return {
                "goal": None,
                "question": "Sorry, I still did not understand. "
                            "Please tell me the whole task again.",
            }
        return {
            "goal": None,
            "question": f"Sorry, I did not catch that. {self.state.question}",
        }

    def _answer_detail(self, text: str):
        """The value a follow-up gives for the pending topic, or None."""
        topic = self.state.clarification_topic
        if topic == PLAN_QUESTION:
            return text.strip() or None
        slots = self.extract_slots(text)
        if topic == "colour" and slots["colours"]:
            return slots["colours"][0]
        if topic == "object" and slots["objects"]:
            return " ".join(slots["colours"][:1] + slots["objects"][:1])
        if topic == "location" and slots["locations"]:
            return location_name(slots["locations"][0])
        if topic == "office":
            # Accept "Office B" as well as a bare "B" or "the second one".
            offices = [loc for loc in slots["locations"]
                       if loc in OFFICE_ANSWERS]
            if not offices:
                offices = [office for office, answers in OFFICE_ANSWERS.items()
                           if answers & set(slots["tokens"])]
            if len(offices) == 1 or (offices and slots["locations"]):
                return location_name(offices[0])
        return None

    def reset(self):
        """Forget any pending clarification, e.g. after a stop request."""
        self.state = DialogueState()

    # ------------------------------------------------------------------
    # Status explanation
    # ------------------------------------------------------------------
    @staticmethod
    def describe_status(world_state: dict) -> str:
        """Answer "what are you doing and why?" from WorldState.as_dict() only.

        The main loop may pass execution_status "PLANNING" (not a WorldState
        status) while a plan or revised plan is being generated.
        """
        status = world_state.get("execution_status", "IDLE")
        goal = world_state.get("current_goal")
        location = location_name(world_state.get("current_location") or "unknown")
        sentences = []

        question = world_state.get("pending_question")
        if question and status not in {"RUNNING", "PLANNING"}:
            sentences.append(
                f"I asked you: {question} I am waiting for your answer."
            )
            return " ".join(sentences)

        if status == "RUNNING":
            step = world_state.get("current_step") or {}
            sentences.append(f"I am {_step_phrase(step)}.")
            if goal:
                sentences.append(f"I am doing this because you asked me to: {goal}.")
            sentences.append(f"The last place I reached was {location}.")
        elif status == "SUCCEEDED":
            sentences.append(f"I have finished the task: {goal}.")
        elif status == "FAILED":
            sentences.append(f"I could not complete the task: {goal}.")
        elif status in {"CANCELLED", "TIMEOUT", "BLOCKED"}:
            reason = {
                "CANCELLED": "was cancelled",
                "TIMEOUT": "timed out",
                "BLOCKED": "was blocked",
            }[status]
            sentences.append(f"My last task {reason}: {goal}.")
        elif status == "PLANNING" and world_state.get("replan_count"):
            sentences.append(
                f"A step did not work, so I am revising my plan for: {goal}."
            )
        elif goal:
            sentences.append(f"I am working out a plan for: {goal}.")
        else:
            sentences.append(
                f"I am not doing anything right now. I am at {location} "
                "and ready for a new task."
            )

        search = _search_phrase(world_state.get("last_search_result"))
        if search and status != "IDLE":
            sentences.append(search)
        blocked = world_state.get("blocked_routes") or []
        if blocked:
            sentences.append(
                f"These routes were blocked: {join_options(map(str, blocked))}."
            )
        replans = world_state.get("replan_count") or 0
        if replans and status != "PLANNING":
            times = "once" if replans == 1 else f"{replans} times"
            sentences.append(f"I have revised my plan {times}.")
        return " ".join(sentences)


def _step_phrase(step: dict) -> str:
    action = step.get("action")
    if action == "navigate":
        return f"driving to {location_name(step.get('location', ''))}"
    if action == "search":
        target = " ".join(v for v in (step.get("colour"), step.get("object")) if v)
        return f"searching for the {target}"
    return {
        "observe": "observing my surroundings",
        "ask": "waiting for your answer",
        "report": "reporting back to you",
        "return_home": "returning to reception",
        "stop": "stopping",
        "conditional": "deciding what to do next from what I have observed",
    }.get(action, "carrying out my plan")


def _search_phrase(result) -> Optional[str]:
    """Describe a search result given as an evidence dict or a Detection."""
    if result is None:
        return None
    if isinstance(result, dict):
        outcome = result.get("result")
        confidence = result.get("confidence")
    else:
        outcome = "FOUND"
        confidence = getattr(result, "confidence", None)
    words = {
        "FOUND": "found the target",
        "NOT_FOUND": "did not find the target",
        "UNCERTAIN": "was not sure whether I saw the target",
    }
    if outcome not in words:
        return None
    sentence = f"In my last search I {words[outcome]}"
    if isinstance(confidence, (int, float)):
        sentence += f", with confidence {confidence:.0%}"
    return sentence + "."
