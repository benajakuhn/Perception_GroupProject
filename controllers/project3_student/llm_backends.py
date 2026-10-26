"""Language-model backends: prompt text in, raw answer text out.

This is the only module that talks to a model. It knows nothing about plans,
JSON parsing or validation; those live in prompt_builder.py, plan_parser.py
and plan_validator.py. Every backend offers the same two methods:

    complete(prompt) -> raw answer text
    description()    -> model and settings, for the logs and the report
"""
import json
import time
from pathlib import Path

DEFAULT_MOCK_RESPONSES_PATH = "tests/fixtures/llm_mock_responses.json"


class LLMBackendError(Exception):
    """Base class for every backend problem."""


class LLMUnavailableError(LLMBackendError):
    """The model cannot be used: missing file, missing package or bad config."""


class LLMTimeoutError(LLMBackendError):
    """The model did not finish its answer within the configured time.

    partial_text is what the model had written before it was stopped, so the
    logs show whether it was slow or stuck (e.g. repeating itself).
    """

    def __init__(self, message: str, partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


def create_backend(config: dict, project_root: Path):
    """Return the backend named in the configuration."""
    backend_name = config.get("backend")
    if backend_name == "llama_cpp":
        return LlamaCppBackend(config, project_root)
    if backend_name == "mock":
        return MockBackend(_load_mock_responses(config, project_root))
    raise LLMUnavailableError(
        f"Unknown LLM backend {backend_name!r} in config; use 'llama_cpp' or 'mock'."
    )


class LlamaCppBackend:
    """Runs a local GGUF model (the Lab 5 Qwen model) with llama-cpp-python."""

    def __init__(self, config: dict, project_root: Path):
        self.model_identifier = config.get("model_identifier", "unknown")
        self.model_path = (Path(project_root) / config.get("model_path", "")).resolve()
        if not self.model_path.is_file():
            raise LLMUnavailableError(
                f"Model file not found: {self.model_path}. "
                "Check model_path in config/llm_config.json."
            )

        self.context_size = int(config.get("n_ctx", 4096))
        self.thread_count = config.get("n_threads")
        self.json_mode = bool(config.get("json_mode", True))
        self.timeout_seconds = float(config.get("timeout_seconds", 45))
        self.generation_settings = {
            "temperature": float(config.get("temperature", 0.0)),
            "top_p": float(config.get("top_p", 1.0)),
            "max_tokens": int(config.get("max_tokens", 1000)),
            "seed": int(config.get("seed", 0)),
        }
        self.model = None  # loaded on the first complete() call, in the worker thread
        # Why the last answer ended: "stop" (finished) or "length" (cut off at
        # max_tokens or the context window). None when no answer was written.
        self.last_finish_reason = None

    def complete(self, prompt: str) -> str:
        if self.model is None:
            self._load_model()

        request = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **self.generation_settings,
        }
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}

        # Streaming delivers the answer in small pieces, so the time limit can
        # be checked while the model is still writing.
        start_time = time.monotonic()
        answer_pieces = []
        self.last_finish_reason = None
        try:
            answer_stream = self.model.create_chat_completion(**request)
            for chunk in answer_stream:
                choice = chunk["choices"][0]
                answer_pieces.append(choice["delta"].get("content") or "")
                if choice.get("finish_reason"):
                    self.last_finish_reason = choice["finish_reason"]
                if time.monotonic() - start_time > self.timeout_seconds:
                    answer_stream.close()
                    raise LLMTimeoutError(
                        f"Model did not finish within {self.timeout_seconds} seconds.",
                        partial_text="".join(answer_pieces),
                    )
        except LLMBackendError:
            raise
        except Exception as error:
            # llama-cpp reports problems such as a prompt longer than n_ctx as
            # ValueError/RuntimeError; turn them into backend errors so callers
            # log and handle them like every other model failure.
            raise LLMBackendError(
                f"llama-cpp failed: {type(error).__name__}: {error}"
            ) from error
        return "".join(answer_pieces)

    def description(self) -> dict:
        return {
            "backend": "llama_cpp",
            "model_identifier": self.model_identifier,
            "model_file": self.model_path.name,
            "llama_cpp_version": _installed_llama_cpp_version(),
            "n_ctx": self.context_size,
            "json_mode": self.json_mode,
            "timeout_seconds": self.timeout_seconds,
            **self.generation_settings,
        }

    def _load_model(self):
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise LLMUnavailableError(
                "llama-cpp-python is not installed for the Python that runs this "
                "controller."
            ) from error

        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.thread_count,
                seed=self.generation_settings["seed"],
                verbose=False,
            )
        except ValueError as error:
            raise LLMUnavailableError(
                f"Could not load model {self.model_path.name}: {error}"
            ) from error


class MockBackend:
    """Returns saved answers in order.

    TEST FIXTURES ONLY: these answers are hand-written, not model output, and
    must never be reported as model performance.
    """

    def __init__(self, saved_responses: list):
        self.saved_responses = list(saved_responses)
        self.received_prompts = []

    def complete(self, prompt: str) -> str:
        self.received_prompts.append(prompt)
        call_number = len(self.received_prompts)
        if call_number > len(self.saved_responses):
            raise LLMBackendError(
                f"MockBackend has {len(self.saved_responses)} saved responses "
                f"but was called {call_number} times."
            )
        return self.saved_responses[call_number - 1]

    def description(self) -> dict:
        return {
            "backend": "mock",
            "model_identifier": "MOCK - test fixtures, not model output",
        }


def _load_mock_responses(config: dict, project_root: Path) -> list:
    """Read saved answers from the fixture file named in the config.

    Each saved answer is either raw text or a JSON object; objects are turned
    into JSON text so the backend always returns text like a real model.
    """
    fixture_path = Path(project_root) / config.get(
        "mock_responses", DEFAULT_MOCK_RESPONSES_PATH
    )
    if not fixture_path.is_file():
        raise LLMUnavailableError(f"Mock response file not found: {fixture_path}")

    fixture_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [
        saved_response if isinstance(saved_response, str) else json.dumps(saved_response)
        for saved_response in fixture_data["responses"]
    ]


def _installed_llama_cpp_version() -> str:
    try:
        import llama_cpp
    except ImportError:
        return "not installed"
    return getattr(llama_cpp, "__version__", "unknown")
