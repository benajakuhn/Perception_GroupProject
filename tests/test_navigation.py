"""Navigator tests on a simple simulated E-puck (no Webots needed).

The fake robot integrates differential-drive kinematics from the wheel
velocities, and its front proximity sensors report an obstacle when a point
just ahead of the robot lies inside one of the given rectangles.
"""
import math
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = (
    Path(__file__).resolve().parents[1] / "controllers" / "project3_student"
)
sys.path.insert(0, str(CONTROLLER_DIR))

from navigation import BLOCKED, RUNNING, SUCCEEDED, TIMEOUT, Navigator  # noqa: E402

TIME_STEP = 32
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
# The obstacle supplied in the world, standing on the centre -> centre_north edge.
CENTRE_OBSTACLE = (0.0, 0.9, 1.15, 0.3)


class FakeMotor:
    def __init__(self):
        self.velocity = 0.0

    def setPosition(self, position):
        pass

    def setVelocity(self, velocity):
        self.velocity = velocity


class FakeSensor:
    def __init__(self, read):
        self.read = read

    def enable(self, timestep):
        pass

    def getValue(self):
        return self.read()

    def getValues(self):
        return self.read()


class FakeRobot:
    def __init__(self, x, y, heading=0.0, obstacles=(), stuck=False):
        self.x, self.y, self.heading = x, y, heading
        self.obstacles = list(obstacles)   # (centre x, centre y, size x, size y)
        self.stuck = stuck
        self.left = FakeMotor()
        self.right = FakeMotor()
        front = lambda: 300.0 if self.front_hits_obstacle() else 60.0  # noqa: E731
        self.devices = {
            "gps": FakeSensor(lambda: [self.x, self.y, 0.0]),
            # Navigator reads heading as atan2(cx, cy).
            "compass": FakeSensor(lambda: [math.sin(self.heading),
                                           math.cos(self.heading), 0.0]),
        }
        for index in range(8):
            self.devices[f"ps{index}"] = FakeSensor(
                front if index in (0, 1, 6, 7) else (lambda: 60.0))

    def getDevice(self, name):
        return self.devices[name]

    def front_hits_obstacle(self):
        px = self.x + 0.08 * math.cos(self.heading)
        py = self.y + 0.08 * math.sin(self.heading)
        return any(abs(px - ox) <= sx / 2 and abs(py - oy) <= sy / 2
                   for ox, oy, sx, sy in self.obstacles)

    def step(self):
        if self.stuck:
            return
        dt = TIME_STEP / 1000
        left = self.left.velocity * WHEEL_RADIUS
        right = self.right.velocity * WHEEL_RADIUS
        speed = (left + right) / 2
        self.heading += (right - left) / AXLE_LENGTH * dt
        self.x += speed * math.cos(self.heading) * dt
        self.y += speed * math.sin(self.heading) * dt


def drive(robot, navigator, location, max_steps=20000):
    status = navigator.start(location)
    steps = 0
    while status == RUNNING and steps < max_steps:
        status = navigator.update()
        robot.step()
        steps += 1
    return status


def make(x, y, **kwargs):
    robot = FakeRobot(x, y, **kwargs)
    return robot, Navigator(robot, robot.left, robot.right, TIME_STEP)


class NavigatorTests(unittest.TestCase):
    def test_reaches_laboratory_from_reception(self):
        robot, navigator = make(-0.8, -2.8)
        self.assertEqual(drive(robot, navigator, "laboratory"), SUCCEEDED)
        self.assertLess(math.dist((robot.x, robot.y), (-3.5, -2.0)), 0.2)
        self.assertEqual(navigator.last_evidence["reason"], "arrived")

    def test_long_route_does_not_time_out(self):
        # The longest route in the graph; a fixed 3000-step limit was too short.
        robot, navigator = make(-3.5, 2.0)
        self.assertEqual(drive(robot, navigator, "meeting_room"), SUCCEEDED)

    def test_takes_bypass_when_centre_obstacle_blocks(self):
        robot, navigator = make(-0.8, -2.8, obstacles=[CENTRE_OBSTACLE])
        self.assertEqual(drive(robot, navigator, "office_a"), SUCCEEDED)
        self.assertLess(math.dist((robot.x, robot.y), (-3.5, 2.0)), 0.2)
        self.assertIn(frozenset(("centre", "centre_north")), navigator.blocked_edges)
        self.assertEqual(navigator.reroutes, 1)

    def test_remembers_blocked_edge_for_next_task(self):
        robot, navigator = make(-0.8, -2.8, obstacles=[CENTRE_OBSTACLE])
        drive(robot, navigator, "office_a")
        robot.x, robot.y = 0.0, -1.15
        self.assertEqual(drive(robot, navigator, "main_corridor"), SUCCEEDED)
        self.assertEqual(navigator.reroutes, 0)

    def test_blocked_when_no_route_is_left(self):
        # Something in the doorway of Office A: there is no other way in.
        robot, navigator = make(-0.8, -2.8, obstacles=[(-2.6, 1.05, 0.3, 0.3)])
        self.assertEqual(drive(robot, navigator, "office_a"), BLOCKED)
        self.assertEqual(navigator.last_evidence["reason"], "front obstacle")
        self.assertIn("office_a_approach - office_a_entry",
                      navigator.last_evidence["blocked_paths"])
        self.assertEqual(robot.left.velocity, 0.0)
        self.assertEqual(robot.right.velocity, 0.0)

    def test_obstacle_next_to_target_counts_as_arrival(self):
        # A box on the approach line, about 0.2 m before the laboratory stop point.
        robot, navigator = make(-0.8, -2.8, obstacles=[(-3.37, -1.85, 0.1, 0.1)])
        self.assertEqual(drive(robot, navigator, "laboratory"), SUCCEEDED)
        self.assertEqual(navigator.last_evidence["reason"],
                         "stopped at an obstacle close to the target")

    def test_timeout_when_robot_makes_no_progress(self):
        robot, navigator = make(-0.8, -2.8, stuck=True)
        self.assertEqual(drive(robot, navigator, "laboratory"), TIMEOUT)
        self.assertIn("no arrival within", navigator.last_evidence["reason"])

    def test_unknown_location_fails(self):
        robot, navigator = make(-0.8, -2.8)
        self.assertEqual(navigator.start("kitchen"), "FAILED")


if __name__ == "__main__":
    unittest.main()
