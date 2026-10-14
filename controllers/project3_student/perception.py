"""Camera-based object-search skeleton."""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Detection:
    object_type: str
    colour: str
    confidence: float
    image_x: Optional[float] = None

class ObjectPerception:
    def __init__(self, camera, timestep):
        self.camera = camera
        self.camera.enable(timestep)
        self.target = None
        self.frames_examined = 0

    def detect(self) -> List[Detection]:
        """Analyse the current camera image.

        TODO:
        - convert the Webots camera buffer to an image array;
        - use colour/shape evidence;
        - reject obvious distractors;
        - return confidence or another uncertainty indicator.
        """
        return []

    def search_target(self, object_type: str, colour: str = None):
        """Return a detection/evidence object if the target is observed."""
        # TODO
        return None

    def start_search(self, object_type: str, colour: str = None):
        """Initialise a bounded multi-view search without blocking Webots."""
        self.target = (object_type, colour)
        self.frames_examined = 0

    def update_search(self):
        """Inspect one frame and return (status, evidence).

        Status is RUNNING, SUCCEEDED or FAILED. TODO: rotate/scan through the
        main-thread skill controller, accumulate evidence over several frames,
        and stop after a documented bound rather than looping indefinitely.
        """
        self.frames_examined += 1
        detection = self.search_target(*self.target) if self.target else None
        if detection is not None:
            return "SUCCEEDED", detection
        return "FAILED", None
