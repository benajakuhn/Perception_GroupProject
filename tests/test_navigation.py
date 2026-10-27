"""Navigator tests on a simple simulated E-puck (no Webots needed).

The fake robot integrates differential-drive kinematics from the wheel
velocities. Its eight proximity sensors cast a ray at the E-puck's sensor
angles and convert the distance to the nearest rectangle with the E-puck
lookup table (Webots documentation), so ~67 means nothing in range and 100
is about 6 cm.
"""
import math
import re
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
# E-puck proximity sensors: direction relative to forward (rad, left positive)
# and the distance -> value lookup table.
SENSOR_ANGLES = [-0.30, -0.80, -1.57, -2.64, 2.64, 1.57, 0.80, 0.30]
BODY_RADIUS = 0.035
LOOKUP = [(0.0, 4095), (0.005, 2133), (0.01, 1466), (0.015, 601), (0.02, 384),
          (0.03, 235), (0.04, 158), (0.05, 120), (0.06, 104), (0.07, 67)]
WORLD = Path(__file__).resolve().parents[1] / "worlds" / "project_genai_embodied_assistant.wbt"


def world_walls():
    """Wall rectangles from the world file, so test robots cannot drive
    through walls. Only the tests read the world file; the robot does not."""
    walls = []
    for match in re.finditer(r"DEF (WALL_\w+) Solid \{(.*?)\n\}", WORLD.read_text(), re.S):
        body = match.group(2)
        t = re.search(r"translation ([-\d.e]+) ([-\d.e]+)", body)
        size = re.search(r"size ([-\d.e]+) ([-\d.e]+)", body)
        rotation = re.search(r"rotation 0 0 1 ([-\d.e]+)", body)
        sx, sy = float(size[1]), float(size[2])
        if rotation and abs(math.sin(float(rotation[1]))) > 0.7:
            sx, sy = sy, sx
        walls.append((float(t[1]), float(t[2]), sx, sy))
    return walls


WALLS = world_walls()
# The obstacle supplied in the world, standing on the centre -> centre_north edge.
CENTRE_OBSTACLE = (0.0, 0.9, 1.15, 0.3)
# The same obstacle turned and moved into Office A's doorway (the ~1 m gap
# in the x = -2 wall between y = 0.06 and y = 1.1), which it fills.
DOORWAY_OBSTACLE = (-2.0, 0.58, 0.3, 1.15)


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
        self.devices = {
            "gps": FakeSensor(lambda: [self.x, self.y, 0.0]),
            # Navigator reads heading as atan2(cx, cy).
            "compass": FakeSensor(lambda: [math.sin(self.heading),
                                           math.cos(self.heading), 0.0]),
        }
        for index in range(8):
            self.devices[f"ps{index}"] = FakeSensor(
                lambda index=index: self.proximity(index))

    def getDevice(self, name):
        return self.devices[name]

    def proximity(self, index):
        angle = self.heading + SENSOR_ANGLES[index]
        ox = self.x + BODY_RADIUS * math.cos(angle)
        oy = self.y + BODY_RADIUS * math.sin(angle)
        distance = min((ray_to_box(ox, oy, angle, box) for box in self.obstacles),
                       default=math.inf)
        if distance >= LOOKUP[-1][0]:
            return float(LOOKUP[-1][1])
        for (d0, v0), (d1, v1) in zip(LOOKUP, LOOKUP[1:]):
            if distance <= d1:
                return v0 + (v1 - v0) * (distance - d0) / (d1 - d0)
        return float(LOOKUP[-1][1])

    def touches_obstacle(self):
        return any(abs(self.x - ox) < sx / 2 + 0.035 and abs(self.y - oy) < sy / 2 + 0.035
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


def ray_to_box(ox, oy, angle, box):
    """Distance along a ray to an axis-aligned rectangle (slab method)."""
    bx, by, sx, sy = box
    dx, dy = math.cos(angle), math.sin(angle)
    t_near, t_far = -math.inf, math.inf
    for origin, direction, low, high in ((ox, dx, bx - sx / 2, bx + sx / 2),
                                         (oy, dy, by - sy / 2, by + sy / 2)):
        if abs(direction) < 1e-12:
            if not low <= origin <= high:
                return math.inf
            continue
        t1, t2 = (low - origin) / direction, (high - origin) / direction
        t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
    if t_near > t_far or t_far < 0:
        return math.inf
    return max(t_near, 0.0)


def drive(robot, navigator, location, max_steps=20000):
    status = navigator.start(location)
    steps = 0
    while status == RUNNING and steps < max_steps:
        status = navigator.update()
        robot.step()
        robot.collided = getattr(robot, "collided", False) or robot.touches_obstacle()
        steps += 1
    return status


def make(x, y, obstacles=(), walls=True, **kwargs):
    robot = FakeRobot(x, y, obstacles=list(obstacles) + (WALLS if walls else []), **kwargs)
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

    def test_drives_around_the_centre_obstacle(self):
        # Layer 1: the robot follows the obstacle's edge and carries on,
        # without needing another route in the map.
        robot, navigator = make(-0.8, -2.8, obstacles=[CENTRE_OBSTACLE])
        self.assertEqual(drive(robot, navigator, "office_a"), SUCCEEDED)
        self.assertLess(math.dist((robot.x, robot.y), (-3.5, 2.0)), 0.2)
        self.assertGreaterEqual(navigator.avoided, 1)
        self.assertEqual(navigator.reroutes, 0)
        self.assertFalse(robot.collided)

    def test_drives_around_an_unknown_box_anywhere_on_a_route(self):
        # A box nobody told the robot about, on the reception -> centre_south edge.
        robot, navigator = make(-0.8, -2.8, obstacles=[(-0.45, -2.1, 0.25, 0.25)])
        self.assertEqual(drive(robot, navigator, "laboratory"), SUCCEEDED)
        self.assertGreaterEqual(navigator.avoided, 1)
        self.assertFalse(robot.collided)
        self.assertEqual(navigator.last_evidence["obstacles_driven_around"],
                         navigator.avoided)

    def test_blocked_doorway_lists_unreachable_rooms(self):
        # Something in Office A's doorway: there is no other way in.
        robot, navigator = make(-0.8, -2.8, obstacles=[DOORWAY_OBSTACLE])
        self.assertEqual(drive(robot, navigator, "office_a"), BLOCKED)
        self.assertEqual(navigator.last_evidence["unreachable"], ["office_a"])
        self.assertEqual((robot.left.velocity, robot.right.velocity), (0.0, 0.0))

    def test_new_task_tries_again_after_obstacle_is_moved(self):
        robot, navigator = make(-0.8, -2.8, obstacles=[DOORWAY_OBSTACLE])
        self.assertEqual(drive(robot, navigator, "office_a"), BLOCKED)
        robot.obstacles.clear()                  # someone moves the obstacle
        self.assertEqual(drive(robot, navigator, "office_a"), SUCCEEDED)

    def test_falls_back_to_another_route_when_it_cannot_get_around(self):
        # A small map with two ways from A to C. A wall-like obstacle on the
        # direct way is too long to drive around within 0.9 m of the route
        # (layer 1 gives up), so the robot takes the way through B (layer 2).
        robot, navigator = make(0.0, 0.0, obstacles=[(1.0, 0.0, 0.2, 2.0)], walls=False)
        navigator.locations = {"c": [2.0, 0.0]}
        navigator.graph = {
            "nodes": {"a": [0.0, 0.0], "b": [1.0, 1.6], "c": [2.0, 0.0]},
            "edges": [["a", "c"], ["a", "b"], ["b", "c"]],
        }
        self.assertEqual(drive(robot, navigator, "c"), SUCCEEDED)
        self.assertEqual(navigator.reroutes, 1)
        self.assertIn(frozenset(("a", "c")), navigator.blocked_edges)
        self.assertLess(math.dist((robot.x, robot.y), (2.0, 0.0)), 0.2)
        self.assertFalse(robot.collided)

    def test_blocked_when_no_route_is_left(self):
        robot, navigator = make(-0.8, -2.8, obstacles=[DOORWAY_OBSTACLE])
        self.assertEqual(drive(robot, navigator, "office_a"), BLOCKED)
        self.assertEqual(navigator.last_evidence["reason"], "front obstacle")
        self.assertIn("centre_north - office_a_entry",
                      navigator.last_evidence["blocked_paths"])
        self.assertFalse(robot.collided)   # tried to get around without hitting a wall

    def test_new_task_starts_from_the_last_node_passed(self):
        # A Webots trial: a failed detour left the robot outside Office B's
        # west wall. The nearest node (office_b_entry) is behind the wall, so
        # starting there drove into it; the last node passed is a free way back.
        robot, navigator = make(1.89, 1.93, heading=1.57)
        navigator.last_node = "centre_north"
        self.assertEqual(drive(robot, navigator, "office_b"), SUCCEEDED)
        self.assertEqual(navigator.path[0], "centre_north")
        self.assertFalse(robot.collided)

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
