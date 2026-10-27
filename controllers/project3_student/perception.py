"""Camera-based object search.

Set PERCEPTION_SAVE_FRAMES=1 to save every 10th search frame, and every frame
with a detection, to logs/frames/ for checking the thresholds against real
camera images. File names start with the run's start time, so runs do not
overwrite each other.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2


@dataclass
class Detection:
    object_type: str
    colour: str
    confidence: float
    image_x: Optional[float] = None

class ObjectPerception:
    # Thresholds from real Webots search frames (logs/frames, meeting room):
    # the rendered ball has R~196 G~58 B~59 (red 3.3x green), circularity
    # 0.79-0.84 and fill 0.81-0.89. The red box (synthetic frames in
    # tests/test_perception.py) has circularity 0.75-0.79 but fill only
    # 0.62-0.67, so fill is what rejects it; circularity is a loose check.
    MIN_AREA = 500.0           # ball still detected ~2.5 m away
    REF_AREA = 5000.0
    MIN_CIRCULARITY = 0.70
    MIN_FILL = 0.75            # contour area / area of its enclosing circle
    FULL_FILL = 0.85           # fill at which the shape counts as fully round
    # Red channel vs green and blue. The meeting room's floor pad, which the
    # robot stands on while searching, is 2.25x; the rendered ball is 3.3x.
    RED_RATIO = 2.7
    MAX_SEARCH_FRAMES = 200
    FOUND_CONFIDENCE = 0.70

    def __init__(self, camera, timestep, left_motor, right_motor):
        self.camera = camera
        self.camera.enable(timestep)
        self.target = None
        self.frames_examined = 0
        self.best_detection = None
        self.left_motor = left_motor
        self.right_motor = right_motor

    def detect(self) -> List[Detection]:
        """Analyse the current camera image.

        TODO:
        - convert the Webots camera buffer to an image array;
        - use colour/shape evidence;
        - reject obvious distractors;
        - return confidence or another uncertainty indicator.
        """

        # get the current camera image
        image = self.camera.getImage()
        if image is None:
            return []

        #get the image width and height
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        #create 1D numpy array from the image buffer
        bgra = np.frombuffer(image, dtype=np.uint8)

        #reshape the array to 3D grid (height, width, 4) for BGRA format
        bgra = bgra.reshape((height, width, 4))

        # drop the alpha channel to get BGR format
        bgr = bgra[:, :, :3]

        # get the red, green, and blue channels
        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)

        # create mask for red objects (0 for non red, 255 for red)
        # red objects have red value that is greater than 60 and at least RED_RATIO
        # times the green and blue values. Measured in Webots frames: the meeting
        # room's floor pad, which the robot
        # stands on while searching, is only 2.25x; with a 2x rule the pad filled
        # half the image and swallowed a ball touching it. (Measured: the
        # rendered ball is 3.3x, not the ~19x of its base colour.)
        mask = ((r > 60) & (r > self.RED_RATIO * g) & (r > self.RED_RATIO * b)).astype(np.uint8) * 255

        # find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # compute the centre of the image
        centre_x = width / 2.0

        # list for detections
        detections: List[Detection] = []

        # go through each red contour and check if it is the target object (red ball)
        for contour in contours:
            # check if the area of the contour is large enough to be a ball
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA:
                continue

            # check if the perimeter is greater than 0 to avoid division by zero
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            # check if the circularity is high enough to be a ball
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue

            # a ball fills its enclosing circle; a box (square or hexagon
            # outline) fills at most about two thirds of it
            _, radius = cv2.minEnclosingCircle(contour)
            fill = area / (np.pi * radius * radius) if radius > 0 else 0.0
            if fill < self.MIN_FILL:
                continue

            # find the bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)

            # compute the x position of the object relative to the image centre
            ball_x = x + (w / 2.0)
            image_x = (ball_x - centre_x) / centre_x

            # score the object based on its size relative to a reference area
            area_score = min(1.0, area / self.REF_AREA)

            # combine shape and size into a confidence score; shape counts most,
            # so a far but clearly round ball is still trusted, while a ball cut
            # off at the image edge (lower fill) stays UNCERTAIN
            shape_score = min(1.0, fill / self.FULL_FILL)
            confidence = min(1.0, 0.7 * shape_score + 0.3 * area_score)

            # store the detection with the estimated confidence and x position
            detections.append(
                Detection(
                    object_type="ball",
                    colour="red",
                    confidence=round(float(confidence), 3),
                    image_x=round(float(image_x), 3),
                )
            )

        # sort detections from most confident to least confident
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


    def _save_frame(self, label: str = ""):
        """Debug aid: write the current camera image to logs/frames/."""
        image = self.camera.getImage()
        if image is None:
            return
        folder = Path(__file__).resolve().parents[2] / "logs" / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        bgra = np.frombuffer(image, dtype=np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4))
        self._saved_frames = getattr(self, "_saved_frames", 0) + 1
        if not hasattr(self, "_run_id"):
            self._run_id = time.strftime("%H%M%S")
        name = f"{self._run_id}_{self._saved_frames:04d}{label}.png"
        cv2.imwrite(str(folder / name), bgra[:, :, :3])

    def search_target(self, object_type: str, colour: str = None):
        """Return a detection/evidence object if the target is observed."""
        # Reject search without requested object type
        if not object_type:
            return None

        # Return first detection matching requested type and colour
        for det in self.detect():
            if det.object_type == object_type and (colour is None or det.colour == colour):
                return det
            
        # No matching target found in current frame
        return None

    def start_search(self, object_type: str, colour: str = None):
        """Initialise a bounded multi-view search without blocking Webots."""
        self.target = (object_type, colour)
        self.frames_examined = 0
        self.best_detection = None

    def update_search(self):
        """Inspect one frame and return (status, evidence).

        Status is RUNNING, SUCCEEDED or FAILED. TODO: rotate/scan through the
        main-thread skill controller, accumulate evidence over several frames,
        and stop after a documented bound rather than looping indefinitely.
        """

        if not self.target:
            # Search not possible without target
            return "FAILED", {"outcome": "NOT_FOUND", "found": False, "detection": None}

        # Examine one frame & retain strongest detection so far
        self.frames_examined += 1
        detection = self.search_target(*self.target)
        if os.environ.get("PERCEPTION_SAVE_FRAMES"):
            if detection is not None:
                self._save_frame(f"_conf{detection.confidence:.2f}")
            elif self.frames_examined % 10 == 0:
                self._save_frame()

        if detection is not None:
            if self.best_detection is None or detection.confidence > self.best_detection.confidence:
                self.best_detection = detection

            # Finish when detection is sufficiently reliable
            if detection.confidence >= self.FOUND_CONFIDENCE:
                # Stop motors and return strong detection
                if self.left_motor and self.right_motor:
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                return "SUCCEEDED", {
                    "outcome": "FOUND",
                    "found": True,
                    "detection": detection,
                }

        # Continue scanning if max nr of frames not reached
        if self.frames_examined < self.MAX_SEARCH_FRAMES:
            if self.left_motor and self.right_motor:
                self.left_motor.setVelocity(-2.0)  # Spin to physically scan the room
                self.right_motor.setVelocity(2.0)
            return "RUNNING", None

        # Search bound reached, stop motors and evaluate
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

        # return weak detection if no strong detection was found after max frames
        if self.best_detection is not None:
            return "SUCCEEDED", {
                "outcome": "UNCERTAIN",
                "found": False,
                "detection": self.best_detection,
            }

        # End the search when no detection was found after max frames
        return "SUCCEEDED", {
            "outcome": "NOT_FOUND",
            "found": False,
            "detection": None,
        }


