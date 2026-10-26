"""Perception tests on synthetic 320x240 camera frames.

Frames are drawn with the geometry of the assessed world: E-puck camera
(320x240, field of view 0.84 rad), red ball radius 0.09 m, red box 0.12 m.
They are approximations of rendered Webots frames, used to keep the
ball/box thresholds honest; check them against real frames too.
"""
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from perception import ObjectPerception  # noqa: E402

WIDTH, HEIGHT = 320, 240
FOCAL = (WIDTH / 2) / math.tan(0.84 / 2)   # pixels
BACKGROUND = (90, 110, 120, 255)           # BGRA, grey floor/wall
RED = (20, 20, 200, 255)
DARK_RED = (10, 10, 110, 255)
GREEN = (40, 200, 30, 255)


class FakeCamera:
    def __init__(self, image):
        self.image = image

    def enable(self, timestep):
        pass

    def getWidth(self):
        return WIDTH

    def getHeight(self):
        return HEIGHT

    def getImage(self):
        return self.image.tobytes()


def blank():
    return np.full((HEIGHT, WIDTH, 4), BACKGROUND, np.uint8)


def ball(distance, colour=RED, shaded=True):
    image = blank()
    r = int(round(FOCAL * 0.09 / distance))
    cv2.circle(image, (160, 120), r, colour, -1, lineType=cv2.LINE_AA)
    if shaded:  # darker lower half, as a lit sphere looks
        cv2.ellipse(image, (160, 120), (r, r), 0, 20, 160, DARK_RED, -1,
                    lineType=cv2.LINE_AA)
    return image


def box(distance, angled=False):
    image = blank()
    s = FOCAL * 0.12 / distance
    if not angled:
        cv2.rectangle(image, (int(160 - s / 2), int(120 - s / 2)),
                      (int(160 + s / 2), int(120 + s / 2)), RED, -1)
    else:  # two faces visible: a wider, hexagon-like outline
        w, h = s * 1.41, s
        points = np.array([
            [160 - w / 2, 120 - h / 2 + 6], [160, 120 - h / 2],
            [160 + w / 2, 120 - h / 2 + 6], [160 + w / 2, 120 + h / 2 - 2],
            [160, 120 + h / 2 + 4], [160 - w / 2, 120 + h / 2 - 2],
        ], np.int32)
        cv2.fillPoly(image, [points], RED)
    return image


def detect(image):
    return ObjectPerception(FakeCamera(image), 32, None, None).detect()


class DetectionTests(unittest.TestCase):
    def test_ball_found_at_every_spawn_distance(self):
        # Approved spawn points are 0.86-1.46 m from the rooms' stop points.
        for distance in (0.5, 0.86, 1.1, 1.23, 1.46):
            detections = detect(ball(distance))
            self.assertTrue(detections, distance)
            self.assertGreaterEqual(detections[0].confidence,
                                    ObjectPerception.FOUND_CONFIDENCE, distance)
            self.assertEqual((detections[0].colour, detections[0].object_type),
                             ("red", "ball"))

    def test_red_box_rejected_face_on_and_angled(self):
        for distance in (0.4, 0.6, 0.78, 1.0, 1.4):
            for angled in (False, True):
                self.assertEqual(detect(box(distance, angled)), [],
                                 (distance, angled))

    def test_green_ball_rejected(self):
        self.assertEqual(detect(ball(0.8, colour=GREEN, shaded=False)), [])

    def test_empty_view(self):
        self.assertEqual(detect(blank()), [])

    def test_ball_position_in_image(self):
        image = blank()
        cv2.circle(image, (240, 120), 30, RED, -1, lineType=cv2.LINE_AA)
        self.assertAlmostEqual(detect(image)[0].image_x, 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
