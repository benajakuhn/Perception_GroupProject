"""Tests for llm_backends.

Most tests use MockBackend and hand-written fixtures (not model output). The
RealModelTests class only runs when config/llm_config.json points to an
existing model file and llama-cpp-python is installed; otherwise it is skipped.
"""
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = PROJECT_ROOT / "controllers" / "project3_student"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "llm_config.json"
sys.path.insert(0, str(CONTROLLER_DIR))

from llm_backends import (  # noqa: E402
    LLMBackendError,
    LLMTimeoutError,
    LLMUnavailableError,
    LlamaCppBackend,
    MockBackend,
    create_backend,
)


def show(title, **values):
    """Print a labelled block so a test result can be checked by eye.

    Add -b to the unittest command to hide these prints for passing tests.
    """
    print(f"\n    --- {title} ---")
    for label, value in values.items():
        indented_value = str(value).replace("\n", "\n        ")
        print(f"    {label}: {indented_value}")


def load_local_config():
    if not LOCAL_CONFIG_PATH.is_file():
        return {}
    return json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))


def real_model_is_available():
    config = load_local_config()
    model_file = PROJECT_ROOT / config.get("model_path", "")
    return (
        config.get("backend") == "llama_cpp"
        and model_file.is_file()
        and importlib.util.find_spec("llama_cpp") is not None
    )


class MockBackendTests(unittest.TestCase):
    def test_returns_saved_responses_in_order_and_records_prompts(self):
        backend = MockBackend(["first answer", "second answer"])
        first_answer = backend.complete("prompt one")
        second_answer = backend.complete("prompt two")
        show("MockBackend", answers=[first_answer, second_answer],
             received_prompts=backend.received_prompts)
        self.assertEqual(first_answer, "first answer")
        self.assertEqual(second_answer, "second answer")
        self.assertEqual(backend.received_prompts, ["prompt one", "prompt two"])

    def test_raises_when_saved_responses_run_out(self):
        backend = MockBackend(["only answer"])
        backend.complete("prompt one")
        with self.assertRaises(LLMBackendError) as caught:
            backend.complete("prompt two")
        show("MockBackend out of answers", error=caught.exception)

    def test_description_is_labelled_as_mock(self):
        description = MockBackend([]).description()
        show("MockBackend description", description=description)
        self.assertEqual(description["backend"], "mock")
        self.assertIn("not model output", description["model_identifier"])


class CreateBackendTests(unittest.TestCase):
    def test_mock_backend_reads_fixture_file(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            fixture_path = Path(temporary_folder) / "answers.json"
            fixture_path.write_text(json.dumps({
                "responses": ["raw text answer", {"version": "1.0"}],
            }), encoding="utf-8")
            config = {"backend": "mock", "mock_responses": str(fixture_path)}
            backend = create_backend(config, PROJECT_ROOT)

            first_answer = backend.complete("x")
            second_answer = backend.complete("y")
        show("mock from fixture file", answers=[first_answer, second_answer])
        self.assertIsInstance(backend, MockBackend)
        self.assertEqual(first_answer, "raw text answer")
        self.assertEqual(second_answer, '{"version": "1.0"}')

    def test_mock_backend_uses_project_fixture_file_by_default(self):
        backend = create_backend({"backend": "mock"}, PROJECT_ROOT)
        show("project fixture file", saved_responses=len(backend.saved_responses),
             first=backend.saved_responses[0][:80])
        self.assertGreaterEqual(len(backend.saved_responses), 1)

    def test_missing_fixture_file_is_reported(self):
        config = {"backend": "mock", "mock_responses": "no/such/file.json"}
        with self.assertRaises(LLMUnavailableError) as caught:
            create_backend(config, PROJECT_ROOT)
        show("missing fixture file", error=caught.exception)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(LLMUnavailableError) as caught:
            create_backend({"backend": "magic_cloud"}, PROJECT_ROOT)
        show("unknown backend", error=caught.exception)
        self.assertIn("magic_cloud", str(caught.exception))

    def test_missing_config_is_rejected(self):
        with self.assertRaises(LLMUnavailableError) as caught:
            create_backend({}, PROJECT_ROOT)
        show("empty config", error=caught.exception)

    def test_llama_backend_with_missing_model_file_is_reported(self):
        config = {"backend": "llama_cpp", "model_path": "models/no-such-model.gguf"}
        with self.assertRaises(LLMUnavailableError) as caught:
            create_backend(config, PROJECT_ROOT)
        show("missing model file", error=caught.exception)
        self.assertIn("no-such-model.gguf", str(caught.exception))


@unittest.skipUnless(real_model_is_available(),
                     "needs config/llm_config.json, the model file and llama-cpp-python")
class RealModelTests(unittest.TestCase):
    """Runs the real model once. Slower (a few seconds) and not a fixture."""

    def test_description_records_model_and_settings(self):
        backend = create_backend(load_local_config(), PROJECT_ROOT)
        description = backend.description()
        show("llama_cpp description", description=json.dumps(description, indent=2))
        self.assertIsInstance(backend, LlamaCppBackend)
        self.assertNotEqual(description["llama_cpp_version"], "not installed")

    def test_real_model_returns_json_text(self):
        backend = create_backend(load_local_config(), PROJECT_ROOT)
        start_time = time.monotonic()
        answer = backend.complete("Reply with a JSON object with key ok set to true.")
        seconds_taken = time.monotonic() - start_time
        show("real model answer", answer=answer,
             seconds_including_model_loading=round(seconds_taken, 2))
        self.assertIsInstance(json.loads(answer), dict)

    def test_tiny_time_limit_raises_timeout(self):
        config = dict(load_local_config(), timeout_seconds=0.001)
        backend = create_backend(config, PROJECT_ROOT)
        with self.assertRaises(LLMTimeoutError) as caught:
            backend.complete("Write a JSON object describing ten different rooms.")
        show("timeout", error=caught.exception)


if __name__ == "__main__":
    unittest.main()
