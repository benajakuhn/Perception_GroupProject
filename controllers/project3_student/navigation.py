"""Non-blocking navigation support for named locations.

The supplied coordinates are only high-level targets. Students must implement
a safe feedback-based navigation method rather than teleporting the robot.

Route: shortest path (Dijkstra) through navigation_graph.json from the graph
node nearest to the robot. When the front sensors detect an obstacle, the
robot stops, marks that graph edge as blocked and, if the graph has another
way, drives back to the previous node and takes it. Only when no route is
left does navigation end as BLOCKED.
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
            self.waypoint_index += 1
            # if all waypoints completed, goal is reached, stop and return SUCCEEDED
            if self.waypoint_index >= len(self.waypoints):
                self.stop()
                self._record_evidence("arrived", x, y)
                self.active_location = None
                return SUCCEEDED
            tx, ty = self.waypoints[self.waypoint_index]
            dist = math.hypot(tx - x, ty - y)

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
        """Treat an obstacle as arrival, take another route, or give up."""
        # An object standing next to the target's stop point
        if self.waypoint_index == len(self.waypoints) - 1 and dist < NEAR_TARGET_OBSTACLE:
            self._record_evidence("stopped at an obstacle close to the target", x, y)
            self.active_location = None
            return SUCCEEDED

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
        self.path = path
        self.waypoints = [nodes[n] for n in path]
        self.waypoint_index = 0
        self.steps = 0
        self.max_steps = self._step_limit(path)
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
        if self.blocked_edges:
            self.last_evidence["blocked_paths"] = sorted(
                " - ".join(sorted(edge)) for edge in self.blocked_edges
            )

    def _front_is_blocked(self, threshold=FRONT_THRESHOLD) -> bool:
        return max(self.proximity[i].getValue() for i in (0, 1, 6, 7)) > threshold

    def cancel(self):
        self.stop()
        self.active_location = None
        self.waypoints = []
        self.path = []
