"""Speech input/output interface.

Recommended technologies include Vosk + sounddevice for recognition and
pyttsx3 (or another approved method) for speech synthesis.
"""
class SpeechIO:
    def __init__(self):
        # TODO: initialise ASR/TTS resources.
        self.console_fallback = True

    def listen(self) -> str:
        """Return one recognised user utterance.

        TODO: implement speech recognition.
        A console fallback is useful for debugging but does not replace the
        assessed speech capability.
        """
        return ""

    def speak(self, text: str) -> None:
        """Speak the supplied robot response."""
        # TODO: add speech synthesis.
        print(f"[ROBOT] {text}")
