"""Non-blocking navigation support for named locations.

The supplied coordinates are only high-level targets. Students must implement
a safe feedback-based navigation method rather than teleporting the robot.

Route: shortest path (Dijkstra) through navigation_graph.json from the graph
node nearest to the robot. An obstacle detected by the front sensors is
handled in three layers, each bounded:

1. Drive around it: stop, turn away from it and follow its edge with the
   side sensors until the robot is back on the line to its waypoint and
   closer to it than where it met the obstacle (a simple Bug2 strategy).
2. If that fails (travel, time or distance-from-route limit), mark the
   graph edge as blocked and take another route through the graph.
3. If no route is left, end as BLOCKED so the planner can replan.

The robot never reverses, so the front sensors always face the direction
of travel.
"""
import heapq
import json
import math
from pathlib import Path

RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
TIMEOUT = "TIMEOUT"

ARRIVAL_TOLERANCE = 0.15      # m from a waypoint counts as reached
TURN_IN_PLACE_ERROR = 0.4     # rad; above this the robot turns on the spot
CRUISE_SPEED = 4.5            # rad/s wheel speed when driving straight
WHEEL_RADIUS = 0.0205         # m (E-puck)
FRONT_THRESHOLD = 100.0       # ps0/ps1/ps6/ps7 value that means "obstacle ahead"
# An obstacle this close to the final target is taken as arrival: objects
# can stand right next to a room's stop point.
NEAR_TARGET_OBSTACLE = 0.4    # m
MAX_REROUTES = 3
# Driving around an obstacle (layer 1). Sensor values follow the E-puck
# lookup table: ~67 = nothing within 7 cm, 100 = ~6 cm, 300 = ~2.5 cm.
AVOID_SPEED = 2.5             # rad/s, slower while close to an obstacle
AVOID_TURN_SPEED = 2.0        # rad/s, turning away on the spot
AVOID_CLEAR = 80.0            # front below this: done turning away
AVOID_TOO_CLOSE = 300.0       # side sensor above this: steer away
AVOID_LOST = 90.0             # side sensor below this: steer back towards it
AVOID_MAX_TRAVEL = 3.0        # m driven while going around one obstacle
AVOID_MAX_DEVIATION = 0.9     # m from the route edge (keeps it in the hall)
AVOID_MAX_SECONDS = 90.0
AVOID_LEAVE_LINE = 0.05       # m from the line to the waypoint counts as on it
# Progress limit: time to drive the route at cruise speed, times a margin for
# turning and steering, plus a fixed allowance.
TIMEOUT_MARGIN = 2.0
TIMEOUT_EXTRA_SECONDS = 20.0


class Navigator:
    def __init__(self, robot, left_motor, right_motor, timestep):
        self.robot = robot
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.timestep = timestep
        config = Path(__file__).resolve().parents[2] / "config" / "locations.json"
        graph_config = (
            Path(__file__).resolve().parents[2] / "config" / "navigation_graph.json"
        )
        self.locations = json.loads(config.read_text(encoding="utf-8"))
        self.graph = json.loads(graph_config.read_text(encoding="utf-8"))
        self.gps = robot.getDevice("gps")
        self.compass = robot.getDevice("compass")
        self.gps.enable(timestep)
        self.compass.enable(timestep)
        self.proximity = []
        for index in range(8):
            sensor = robot.getDevice(f"ps{index}")
            sensor.enable(timestep)
            self.proximity.append(sensor)
        self.active_location = None
        self.path = []            # graph node names of the current route
        self.waypoints = []       # their coordinates
        self.waypoint_index = 0
        self.steps = 0
        self.max_steps = 0
        self.reroutes = 0
        self.avoiding = None      # state while driving around an obstacle
        self.avoided = 0          # obstacles driven around in this navigation
        # Edges found blocked during the current navigation. Cleared on every
        # start(): the obstacle is movable, so a new task tries again and the
        # sensors decide.
        self.blocked_edges = set()
        # Why the last navigation ended; read by RobotSkills for the trace.
        self.last_evidence = None

    def stop(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def start(self, location: str) -> str:
        """Start a named-location task without advancing the simulation."""
        if location not in self.locations:
            return FAILED

        self.active_location = location
        self.reroutes = 0
        self.avoiding = None
        self.avoided = 0
        self.blocked_edges = set()
        self.last_evidence = None

        # find the nearest graph node to the robot's current position
        x, y, _ = self.gps.getValues()
        nodes = self.graph["nodes"]
        start_node = min(nodes, key=lambda n: math.hypot(nodes[n][0] - x, nodes[n][1] - y))

        if not self._set_route(start_node):
            self._record_evidence("no route that avoids blocked paths", x, y)
            self.active_location = None
            return BLOCKED
        # Initial motor stop
        self.stop()
        return RUNNING

    def update(self) -> str:
        """Perform one feedback-control update and return a status.

        GPS and Compass give position and heading feedback; motion comes only
        from wheel velocity commands.
        """
        # Check if navigation is active and waypoints are available
        if self.active_location is None or not self.waypoints:
            return FAILED

        # get current position
        x, y, _ = self.gps.getValues()

        # check the progress limit for this route
        self.steps += 1
        if self.steps > self.max_steps:
            self.stop()
            self._record_evidence(f"no arrival within {self.max_steps} steps", x, y)
            self.active_location = None
            return TIMEOUT

        # get current target x,y position and distance to it
        tx, ty = self.waypoints[self.waypoint_index]
        dist = math.hypot(tx - x, ty - y)

        # Check if current waypoint is reached
        if dist < ARRIVAL_TOLERANCE:
            self.avoiding = None
            self.waypoint_index += 1
            # if all waypoints completed, goal is reached, stop and return SUCCEEDED
            if self.waypoint_index >= len(self.waypoints):
                self.stop()
                self._record_evidence("arrived", x, y)
                self.active_location = None
                return SUCCEEDED
            tx, ty = self.waypoints[self.waypoint_index]
            dist = math.hypot(tx - x, ty - y)

        # Going around an obstacle: the side sensors steer, not the waypoint
        if self.avoiding is not None:
            return self._avoid_obstacle(x, y, tx, ty)

        # Compute heading from compass and angular error to target waypoint
        cx, cy, _ = self.compass.getValues()
        heading = math.atan2(cx, cy)
        target_angle = math.atan2(ty - y, tx - x)
        error = math.atan2(math.sin(target_angle - heading), math.cos(target_angle - heading))

        # Turn in place if angle error is large (no forward motion, so the
        # front sensors only matter once the robot drives again)
        if abs(error) > TURN_IN_PLACE_ERROR:
            turn = max(-3.0, min(3.0, 2.5 * error))
            self.left_motor.setVelocity(-turn)
            self.right_motor.setVelocity(turn)
            return RUNNING

        # Obstacle ahead: stop before any recovery
        if self._front_is_blocked():
            self.stop()
            return self._handle_obstacle(x, y, dist)

        # if error is small drive forward with steering
        turn = 2.0 * error
        left_speed = max(-6.28, min(6.28, CRUISE_SPEED - turn))
        right_speed = max(-6.28, min(6.28, CRUISE_SPEED + turn))
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        return RUNNING

    def _handle_obstacle(self, x: float, y: float, dist: float) -> str:
        """Treat an obstacle as arrival, drive around it, or reroute."""
        # An object standing next to the target's stop point
        if self.waypoint_index == len(self.waypoints) - 1 and dist < NEAR_TARGET_OBSTACLE:
            self._record_evidence("stopped at an obstacle close to the target", x, y)
            self.active_location = None
            return SUCCEEDED

        if self._start_avoidance(x, y):
            return RUNNING
        return self._reroute_or_block(x, y)

    def _edge_from(self, x: float, y: float):
        """Start of the edge being driven: previous waypoint or route start."""
        if self.waypoint_index > 0:
            return tuple(self.waypoints[self.waypoint_index - 1])
        return self.route_start

    def _start_avoidance(self, x: float, y: float) -> bool:
        """Layer 1: begin driving around the obstacle, once per edge."""
        edge = self._current_edge()
        if edge in self.avoid_failed:
            return False
        p = [sensor.getValue() for sensor in self.proximity]
        # Keep the obstacle on the side where it is: obstacle mostly on the
        # right -> turn left and follow it with the right-hand sensors.
        side = 1 if max(p[0], p[1], p[2]) >= max(p[5], p[6], p[7]) else -1
        target = tuple(self.waypoints[self.waypoint_index])
        self.avoiding = {
            "edge": edge,
            "side": side,
            "phase": "turn",
            "hit": (x, y),
            "hit_dist": math.dist((x, y), target),
            "from": self._edge_from(x, y),
            "last": (x, y),
            "travelled": 0.0,
            "steps": 0,
        }
        self.max_steps += int(AVOID_MAX_SECONDS * 1000 / self.timestep)
        print(f"[NAV] obstacle ahead; driving around it "
              f"(keeping it on the {'right' if side == 1 else 'left'})")
        return True

    def _avoid_obstacle(self, x: float, y: float, tx: float, ty: float) -> str:
        """One update of driving around an obstacle (layer 1)."""
        a = self.avoiding
        a["steps"] += 1
        a["travelled"] += math.dist(a["last"], (x, y))
        a["last"] = (x, y)
        side = a["side"]
        p = [sensor.getValue() for sensor in self.proximity]
        front = max(p[0], p[7])

        # Give up on this obstacle: let the graph find another way
        if (a["travelled"] > AVOID_MAX_TRAVEL
                or a["steps"] > AVOID_MAX_SECONDS * 1000 / self.timestep
                or _distance_to_segment((x, y), a["from"], (tx, ty)) > AVOID_MAX_DEVIATION):
            self.stop()
            self.avoid_failed.add(a["edge"])
            self.avoiding = None
            print("[NAV] could not get around the obstacle; looking for another route")
            return self._reroute_or_block(x, y)

        # Past the obstacle: back on the line to the waypoint, closer than
        # where it was met, and the way ahead is clear
        if (a["travelled"] > 0.15
                and _distance_to_line((x, y), a["hit"], (tx, ty)) < AVOID_LEAVE_LINE
                and math.dist((x, y), (tx, ty)) < a["hit_dist"] - 0.1
                and front < FRONT_THRESHOLD):
            self.avoiding = None
            self.avoided += 1
            print("[NAV] past the obstacle; back on the route")
            return RUNNING

        # Turn away on the spot until the front is clear
        if a["phase"] == "turn" or front > FRONT_THRESHOLD:
            if front > AVOID_CLEAR:
                a["phase"] = "turn"
                self.left_motor.setVelocity(-AVOID_TURN_SPEED * side)
                self.right_motor.setVelocity(AVOID_TURN_SPEED * side)
                return RUNNING
            a["phase"] = "follow"

        # Follow the obstacle's edge with the sensors on its side
        near = max(p[1], p[2]) if side == 1 else max(p[5], p[6])
        if near > AVOID_TOO_CLOSE:        # too close: steer away from it
            left, right = AVOID_SPEED - 0.75 * side, AVOID_SPEED + 0.75 * side
        elif near < AVOID_LOST:           # lost it: curve back towards it
            left, right = AVOID_SPEED + 1.0 * side, AVOID_SPEED - 1.0 * side
        else:
            left = right = AVOID_SPEED
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)
        return RUNNING

    def _current_edge(self):
        if self.waypoint_index > 0:
            return frozenset((self.path[self.waypoint_index - 1],
                              self.path[self.waypoint_index]))
        return frozenset(("start", self.path[0]))

    def _reroute_or_block(self, x: float, y: float) -> str:
        """Layers 2 and 3: another route through the graph, else BLOCKED."""
        # Mark the edge being driven as blocked and look for another way from
        # the node the robot came from
        if self.waypoint_index > 0:
            came_from = self.path[self.waypoint_index - 1]
            heading_to = self.path[self.waypoint_index]
            self.blocked_edges.add(frozenset((came_from, heading_to)))
            if self.reroutes < MAX_REROUTES and self._set_route(came_from):
                self.reroutes += 1
                print(f"[NAV] {came_from} -> {heading_to} blocked; new route: "
                      f"{' -> '.join(self.path)}")
                return RUNNING

        self._record_evidence("front obstacle", x, y)
        # Every named location the robot can no longer reach from where it is,
        # so the planner does not try them one by one.
        here = self.path[max(self.waypoint_index - 1, 0)]
        self.last_evidence["unreachable"] = sorted(
            loc for loc in self.locations
            if loc != here and self._shortest_path(here, loc) is None
        )
        self.active_location = None
        return BLOCKED

    def _set_route(self, start_node: str) -> bool:
        """Plan from start_node to the active location; False if no route."""
        path = self._shortest_path(start_node, self.active_location)
        if path is None:
            return False
        nodes = self.graph["nodes"]
        x, y, _ = self.gps.getValues()
        self.route_start = (x, y)
        self.path = path
        self.waypoints = [nodes[n] for n in path]
        self.waypoint_index = 0
        self.steps = 0
        self.max_steps = self._step_limit(path)
        self.avoiding = None
        self.avoid_failed = set()
        return True

    def _shortest_path(self, start: str, goal: str):
        """Dijkstra over the graph, skipping edges found blocked."""
        nodes = self.graph["nodes"]
        neighbours = {name: [] for name in nodes}
        for u, v in self.graph["edges"]:
            if frozenset((u, v)) in self.blocked_edges:
                continue
            length = math.dist(nodes[u], nodes[v])
            neighbours[u].append((v, length))
            neighbours[v].append((u, length))

        queue = [(0.0, start, [start])]
        done = set()
        while queue:
            length, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            if node in done:
                continue
            done.add(node)
            for nxt, step in neighbours[node]:
                if nxt not in done:
                    heapq.heappush(queue, (length + step, nxt, path + [nxt]))
        return None

    def _step_limit(self, path) -> int:
        """Progress limit in controller steps for driving this route."""
        x, y, _ = self.gps.getValues()
        points = [(x, y)] + [tuple(self.graph["nodes"][n]) for n in path]
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        seconds = length / (CRUISE_SPEED * WHEEL_RADIUS) * TIMEOUT_MARGIN + TIMEOUT_EXTRA_SECONDS
        return int(seconds * 1000 / self.timestep)

    def _record_evidence(self, reason: str, x: float, y: float):
        self.last_evidence = {
            "reason": reason,
            "position": [round(x, 2), round(y, 2)],
            "elapsed_steps": self.steps,
        }
        if self.avoided:
            self.last_evidence["obstacles_driven_around"] = self.avoided
        if self.blocked_edges:
            self.last_evidence["blocked_paths"] = sorted(
                " - ".join(sorted(edge)) for edge in self.blocked_edges
            )

    def _front_is_blocked(self, threshold=FRONT_THRESHOLD) -> bool:
        return max(self.proximity[i].getValue() for i in (0, 1, 6, 7)) > threshold

    def cancel(self):
        self.stop()
        self.active_location = None
        self.avoiding = None
        self.waypoints = []
        self.path = []


def _distance_to_line(point, a, b) -> float:
    """Distance from point to the infinite line through a and b."""
    (px, py), (ax, ay), (bx, by) = point, a, b
    length = math.dist(a, b)
    if length < 1e-9:
        return math.dist(point, a)
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / length


def _distance_to_segment(point, a, b) -> float:
    """Distance from point to the segment between a and b."""
    (px, py), (ax, ay), (bx, by) = point, a, b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.dist(point, (ax + t * dx, ay + t * dy))
