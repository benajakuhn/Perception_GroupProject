"""Tests for the non-blocking speech interface, using fake audio and TTS.

No microphone, Vosk model or speaker is needed to run these tests.
"""
import array
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

import speech_io  # noqa: E402
from speech_io import SpeechIO, find_model_path  # noqa: E402


class SlowFakeTTS:
    """Pretends to speak each sentence for a fixed time."""

    def __init__(self, seconds=0.3):
        self.seconds = seconds
        self.spoken = []
        self.finished = threading.Event()

    def __call__(self, text):
        time.sleep(self.seconds)
        self.spoken.append(text)
        self.finished.set()


class FakeRecogniser:
    """Returns a final result for every block, like a phrase just ended."""

    def __init__(self, text):
        self.text = text
        self.resets = 0

    def AcceptWaveform(self, data):
        return True

    def Result(self):
        return json.dumps({"text": self.text})

    def FinalResult(self):
        return json.dumps({"text": ""})

    def Reset(self):
        self.resets += 1


class SpeakTests(unittest.TestCase):
    def setUp(self):
        self.tts = SlowFakeTTS()
        self.speech = SpeechIO(use_microphone=False, tts_backend=self.tts)

    def tearDown(self):
        self.speech.close()

    def test_speak_returns_immediately(self):
        start = time.monotonic()
        self.speech.speak("I am going to the laboratory.")
        self.assertLess(time.monotonic() - start, 0.05)
        self.assertTrue(self.tts.finished.wait(2.0))
        self.assertEqual(self.tts.spoken, ["I am going to the laboratory."])

    def test_is_speaking_while_talking_and_echo_guard_after(self):
        self.speech.speak("Hello.")
        time.sleep(0.1 + speech_io.QUEUE_POLL_SECONDS)
        self.assertTrue(self.speech.is_speaking)
        self.assertTrue(self.tts.finished.wait(2.0))
        time.sleep(speech_io.ECHO_GUARD_SECONDS + 0.1)
        self.assertFalse(self.speech.is_speaking)

    def test_empty_text_is_not_spoken(self):
        self.speech.speak("")
        self.assertFalse(self.tts.finished.wait(0.5))

    def test_stop_speaking_drops_queued_sentences(self):
        for i in range(5):
            self.speech.speak(f"Sentence {i}.")
        self.speech.stop_speaking()
        time.sleep(1.0)
        self.assertLessEqual(len(self.tts.spoken), 1)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.speech = SpeechIO(use_microphone=False,
                               tts_backend=lambda text: None)

    def tearDown(self):
        self.speech.close()

    def test_poll_is_none_when_nothing_heard(self):
        self.assertIsNone(self.speech.poll())
        self.assertEqual(self.speech.listen(), "")

    def test_recognised_text_reaches_queue(self):
        self.speech.recogniser = FakeRecogniser("find the red ball")
        self.speech.process_audio(b"\0" * 100)
        self.assertEqual(self.speech.poll(),
                         {"text": "find the red ball", "answer": None})
        self.assertIsNone(self.speech.poll())

    def test_empty_recognition_is_ignored(self):
        self.speech.recogniser = FakeRecogniser("")
        self.speech.process_audio(b"\0" * 100)
        self.assertIsNone(self.speech.poll())

    def test_unknown_word_markers_are_removed(self):
        self.speech.recogniser = FakeRecogniser("[unk] red [unk]")
        self.speech.process_audio(b"\0" * 100)
        self.assertEqual(self.speech.poll()["text"], "red")
        self.speech.recogniser = FakeRecogniser("[unk]")
        self.speech.process_audio(b"\0" * 100)
        self.assertIsNone(self.speech.poll())

    def test_expected_answer_heard_alongside_open_text(self):
        created = []

        def factory(words):
            created.append(words)
            return FakeRecogniser("red")

        self.speech._recogniser_factory = factory
        self.speech.recogniser = FakeRecogniser("read")
        self.speech.expect(["red", "green", "[unk]"])
        self.speech.process_audio(b"\0" * 100)
        self.assertEqual(self.speech.poll(), {"text": "read", "answer": "red"})
        self.speech.expect(None)
        self.speech.process_audio(b"\0" * 100)
        self.assertEqual(self.speech.poll(), {"text": "read", "answer": None})
        self.assertEqual(created, [["red", "green", "[unk]"]])

    def test_microphone_ignored_while_robot_speaks(self):
        recogniser = FakeRecogniser("task complete")
        self.speech.recogniser = recogniser
        self.speech._mute_until = time.monotonic() + 5.0
        self.speech.process_audio(b"\0" * 100)
        self.assertIsNone(self.speech.poll())
        self.assertEqual(recogniser.resets, 1)


class SilenceWarningTests(unittest.TestCase):
    def setUp(self):
        self.speech = SpeechIO(use_microphone=False, tts_backend=lambda text: None)

    def tearDown(self):
        self.speech.close()

    def blocks(self, level):
        block = array.array("h", [level, -level] * 4000).tobytes()
        return [block] * speech_io.SILENCE_CHECK_BLOCKS

    def test_warns_once_when_microphone_is_silent(self):
        results = [self.speech.check_silence(b) for b in self.blocks(0)]
        self.assertEqual(results.count(True), 1)
        self.assertFalse(self.speech.check_silence(self.blocks(0)[0]))

    def test_no_warning_with_sound(self):
        results = [self.speech.check_silence(b) for b in self.blocks(500)]
        self.assertNotIn(True, results)


class SayCommandTests(unittest.TestCase):
    def test_text_starting_with_dash_is_not_an_option(self):
        speech = SpeechIO(use_microphone=False, tts_backend=lambda text: None)
        with mock.patch.object(speech_io.subprocess, "Popen") as popen:
            speech._say("-r 5 hello")
        popen.assert_called_once_with(["say", "--", "-r 5 hello"])
        speech.close()


class FallbackTests(unittest.TestCase):
    def test_model_path_from_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {"VOSK_MODEL_PATH": folder}):
                self.assertEqual(find_model_path(), Path(folder))

    def test_missing_model_raises(self):
        with mock.patch.object(speech_io, "MODEL_CANDIDATES",
                               [Path("/no/such/model")]):
            with mock.patch.dict(os.environ, {"VOSK_MODEL_PATH": ""}):
                with self.assertRaises(FileNotFoundError):
                    find_model_path()

    def test_no_typed_input_without_terminal(self):
        speech = SpeechIO(use_microphone=False, tts_backend=lambda text: None)
        with mock.patch.object(sys, "stdin", io.StringIO("go to the lab\n")):
            speech._console_loop("test")
        self.assertEqual(speech.input_mode, "none")
        self.assertIsNone(speech.poll())
        speech.close()


if __name__ == "__main__":
    unittest.main()
