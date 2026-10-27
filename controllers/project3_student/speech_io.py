"""Speech input/output interface.

Speech recognition (Vosk + sounddevice) and speech synthesis each run in their
own daemon thread. The Webots main loop only calls poll() and speak(), which
both return immediately, so robot.step() is never blocked by audio.

While a clarification answer is expected (see expect()), two recognisers hear
the same audio: an open one, which still notices a brand-new request, and one
limited to the expected words, which hears short answers such as "red" or
"Office B" far more reliably. poll() returns both transcripts.

Configuration (environment variables, all optional):
    VOSK_MODEL_PATH       folder of the Vosk model (see find_model_path()).
    SPEECH_INPUT=console  type requests instead of speaking (debugging only).
    SPEECH_INPUT_DEVICE   microphone index or part of its name.
    SPEECH_TTS            'say' (macOS), 'pyttsx3' or 'print' (text only).
"""
import array
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / MODEL_NAME,
    Path.home() / "Documents" / "vosk-models" / MODEL_NAME,  # Lab setup
]
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
# Keep ignoring the microphone briefly after the robot stops talking, so the
# end of its own voice (or room echo) is not recognised as a user request.
ECHO_GUARD_SECONDS = 0.5
QUEUE_POLL_SECONDS = 0.2
# If no audio block in the first seconds is louder than this, the microphone
# is almost certainly muted or blocked (e.g. macOS denied access silently).
SILENCE_LEVEL = 40
SILENCE_CHECK_BLOCKS = 10   # 10 x 0.5 s
SILENCE_WARNING = (
    "[SPEECH] WARNING: the microphone only gives silence. On macOS, Webots "
    "cannot ask for microphone access; start Webots from the Terminal app "
    "(/Applications/Webots.app/Contents/MacOS/webots) so it uses Terminal's "
    "permission."
)


def find_model_path() -> Path:
    """Return the Vosk model folder: $VOSK_MODEL_PATH, then the defaults."""
    candidates = list(MODEL_CANDIDATES)
    if os.environ.get("VOSK_MODEL_PATH"):
        candidates.insert(0, Path(os.environ["VOSK_MODEL_PATH"]).expanduser())
    for path in candidates:
        if path.is_dir():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Vosk model not found (tried {tried})")


class SpeechIO:
    def __init__(self, use_microphone=True, tts_backend=None):
        self._inputs = queue.Queue()
        self._speech = queue.Queue()
        self._closed = threading.Event()
        self._mute_until = 0.0
        self._say_process = None
        self.recogniser = None          # open vocabulary
        self.answer_recogniser = None   # expected words only, or None
        self._recogniser_factory = None
        self._expected = None           # set by the main loop via expect()
        self._active_expected = None    # what answer_recogniser uses
        self._answer_parts = []
        self._blocks_checked = 0
        self._loudest = 0
        self.input_mode = "starting" if use_microphone else "none"
        self._asr_thread = None

        self._tts = tts_backend or self._default_tts()
        threading.Thread(target=self._tts_loop, name="tts", daemon=True).start()
        if use_microphone:
            # Loading the model takes a moment, so it also happens off-thread.
            self._asr_thread = threading.Thread(
                target=self._asr_loop, name="asr", daemon=True
            )
            self._asr_thread.start()

    # ------------------------------------------------------------------
    # Main-loop API (non-blocking)
    # ------------------------------------------------------------------
    def poll(self):
        """Return the next utterance, or None if there is none.

        An utterance is {"text": open transcript, "answer": transcript from
        the expected-word recogniser, or None when no answer is expected}.
        """
        try:
            return self._inputs.get_nowait()
        except queue.Empty:
            return None

    def listen(self) -> str:
        """Non-blocking: the open transcript of the next utterance, or ""."""
        heard = self.poll()
        return heard["text"] if heard else ""

    def expect(self, words=None):
        """Also listen for a word list, or None to stop expecting an answer.

        Call this after asking a clarification question, e.g. with
        DialogueManager.expected_words(); the change takes effect on the
        recognition thread before the next audio block.
        """
        self._expected = list(words) if words else None

    def speak(self, text: str) -> None:
        """Queue a robot response for speech synthesis and return at once."""
        if not text:
            return
        print(f"[ROBOT] {text}")
        if self._tts is not None:
            self._speech.put(text)

    def stop_speaking(self):
        """Drop queued speech and interrupt the current sentence if possible."""
        while not self._speech.empty():
            try:
                self._speech.get_nowait()
            except queue.Empty:
                break
        process = self._say_process
        if process is not None and process.poll() is None:
            process.terminate()

    @property
    def is_speaking(self) -> bool:
        return time.monotonic() < self._mute_until

    def close(self):
        self._closed.set()
        self.stop_speaking()
        # Let the recognition thread close the microphone stream cleanly.
        if self._asr_thread is not None and self.input_mode == "speech":
            self._asr_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Speech synthesis thread
    # ------------------------------------------------------------------
    def _default_tts(self):
        choice = os.environ.get("SPEECH_TTS", "").lower()
        if choice == "print":
            return None
        # pyttsx3 needs the main thread's run loop on macOS, so use the
        # built-in 'say' command there instead.
        if choice == "say" or (not choice and sys.platform == "darwin"
                               and shutil.which("say")):
            return self._say
        return self._pyttsx3

    def _say(self, text):
        # "--" ends the options, so text such as "-5 degrees" (planner reports
        # are untrusted text) is spoken instead of read as a 'say' option.
        self._say_process = subprocess.Popen(["say", "--", text])
        self._say_process.wait()
        self._say_process = None

    def _pyttsx3(self, text):
        import pyttsx3  # created here so it belongs to the TTS thread
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def _tts_loop(self):
        while not self._closed.is_set():
            try:
                text = self._speech.get(timeout=QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            self._mute_until = math.inf
            try:
                self._tts(text)
            except Exception as error:
                print(f"[SPEECH] Speech synthesis failed: {error!r}")
            finally:
                if self._speech.empty():
                    self._mute_until = time.monotonic() + ECHO_GUARD_SECONDS

    # ------------------------------------------------------------------
    # Speech recognition thread
    # ------------------------------------------------------------------
    def _asr_loop(self):
        if os.environ.get("SPEECH_INPUT", "").lower() == "console":
            self._console_loop("SPEECH_INPUT=console")
            return
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(str(find_model_path()))

            def make_recogniser(words):
                if words:
                    return KaldiRecognizer(model, SAMPLE_RATE, json.dumps(words))
                return KaldiRecognizer(model, SAMPLE_RATE)

            self._recogniser_factory = make_recogniser
            self.recogniser = make_recogniser(None)
            audio = queue.Queue()

            def on_audio(indata, frames, time_info, status):
                # Drop audio captured while the robot talks (echo guard).
                if not self.is_speaking:
                    audio.put(bytes(indata))

            device = os.environ.get("SPEECH_INPUT_DEVICE") or None
            if device and device.isdigit():
                device = int(device)
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                channels=1, device=device, callback=on_audio,
            )
        except Exception as error:
            self._console_loop(f"speech recognition unavailable: {error}")
            return

        self.input_mode = "speech"
        print("[SPEECH] Listening on the microphone.")
        with stream:
            while not self._closed.is_set():
                try:
                    data = audio.get(timeout=QUEUE_POLL_SECONDS)
                except queue.Empty:
                    continue
                self.check_silence(data)
                self.process_audio(data)

    def check_silence(self, data: bytes) -> bool:
        """Warn once if the first seconds of audio are pure silence."""
        if self._blocks_checked >= SILENCE_CHECK_BLOCKS:
            return False
        samples = array.array("h", data[: len(data) // 2 * 2])
        self._loudest = max(self._loudest, max(map(abs, samples), default=0))
        self._blocks_checked += 1
        if self._blocks_checked == SILENCE_CHECK_BLOCKS and self._loudest < SILENCE_LEVEL:
            print(SILENCE_WARNING)
            return True
        return False

    def process_audio(self, data: bytes):
        """Feed one audio block to the recognisers; queue completed phrases.

        The open recogniser decides where an utterance ends. The answer
        recogniser may split it differently, so its pieces are collected
        until then.
        """
        if self._expected != self._active_expected:
            self.answer_recogniser = (
                self._recogniser_factory(self._expected)
                if self._expected and self._recogniser_factory else None
            )
            self._active_expected = self._expected
            self._answer_parts = []
        recognisers = [r for r in (self.recogniser, self.answer_recogniser) if r]
        if self.is_speaking:
            for recogniser in recognisers:
                recogniser.Reset()
            self._answer_parts = []
            return

        answer_recogniser = self.answer_recogniser
        if answer_recogniser and answer_recogniser.AcceptWaveform(data):
            self._answer_parts.append(_transcript(answer_recogniser.Result()))
        if not self.recogniser.AcceptWaveform(data):
            return

        text = _transcript(self.recogniser.Result())
        answer = None
        if answer_recogniser:
            self._answer_parts.append(_transcript(answer_recogniser.FinalResult()))
            answer = " ".join(part for part in self._answer_parts if part)
            self._answer_parts = []
        if text or answer:
            print(f"[HEARD] {text!r}" + (f" | answer {answer!r}"
                                         if answer is not None else ""))
            self._inputs.put({"text": text, "answer": answer})

    def _console_loop(self, reason):
        """Typed fallback for debugging; it does not replace assessed speech."""
        if sys.stdin is None or not sys.stdin.isatty():
            # Webots does not forward keyboard input to a controller's stdin.
            self.input_mode = "none"
            print(f"[SPEECH] No user input available ({reason}).")
            return
        self.input_mode = "console"
        print(f"[SPEECH] Typed input ({reason}). Type a request, then Enter.")
        while not self._closed.is_set():
            try:
                line = input().strip()
            except EOFError:
                return
            if line:
                self._inputs.put({"text": line, "answer": None})


def _transcript(result_json: str) -> str:
    """Text of a Vosk result without "[unk]" (speech outside a word list)."""
    words = json.loads(result_json).get("text", "").split()
    return " ".join(word for word in words if word != "[unk]")
