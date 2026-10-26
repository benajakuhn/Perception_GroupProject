"""Camera-based object-search skeleton."""
from dataclasses import dataclass
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
    MIN_AREA = 1500.0
    REF_AREA = 5000.0
    MIN_CIRCULARITY = 0.65
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
        # red objects have red value that is greater than 60 and at least twice as large as the green and blue values
        mask = ((r > 60) & (r > 2 * g) & (r > 2 * b)).astype(np.uint8) * 255

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

            # find the bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)

            # compute the x position of the object relative to the image centre
            ball_x = x + (w / 2.0)
            image_x = (ball_x - centre_x) / centre_x

            # score the object based on its size relative to a reference area
            area_score = min(1.0, area / self.REF_AREA)

            # combine circularity and size into a confidence score
            confidence = min(1.0, 0.6 * circularity + 0.4 * area_score)

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


