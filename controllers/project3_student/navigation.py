"""Non-blocking navigation support for named locations.

The supplied coordinates are only high-level targets. Students must implement
a safe feedback-based navigation method rather than teleporting the robot.
"""
import json
from pathlib import Path
from collections import deque
import math

RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"

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
        self.waypoints = []
        self.waypoint_index = 0
        self.steps = 0

    def stop(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def start(self, location: str) -> str:
        """Start a named-location task without advancing the simulation.

        TODO: find a route through navigation_graph.json from the nearest graph
        node to the requested location and store its waypoint coordinates.
        """
        if location not in self.locations:
            return FAILED

        self.active_location = location
        self.waypoint_index = 0
        self.steps = 0

        # get nodes from the graph
        nodes = self.graph["nodes"]
        # get current robot position
        x, y, _ = self.gps.getValues()

        # find the nearest graph node to the robot's current position
        start_node = min(
            nodes, key=lambda n: math.hypot(nodes[n][0] - x, nodes[n][1] - y)
        )

        # BFS queue
        queue = deque([[start_node]])
        # Set of visited nodes
        visited = {start_node}
        # Store the path from start to goal
        path = [location]

        # BFS
        while queue:
            # get the current path and node
            curr_path = queue.popleft()
            curr = curr_path[-1]

            # check if goal is reached
            if curr == location:
                path = curr_path
                break

            # Unvisited neighbouring nodes
            for u, v in self.graph["edges"]:
                nxt = v if u == curr else (u if v == curr else None)
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append(curr_path + [nxt])

        # Node path -> waypoint coordinates
        self.waypoints = [nodes[n] for n in path]
        # Initial motor stop
        self.stop()
        # Navigation status
        return RUNNING

    def update(self) -> str:
        """Perform one feedback-control update and return a status.

        TODO: use GPS/Compass only for localisation and heading feedback; use
        motor velocity commands for motion. Do not teleport the robot. Detect
        obstacles with proximity sensors, enforce time/distance limits, and
        either choose a safe route or return FAILED with evidence.
        """
        if self.active_location is None:
            return FAILED
        if self.front_is_blocked():
            self.stop()
            return FAILED
        self.stop()
        return FAILED

    def front_is_blocked(self, threshold=80.0) -> bool:
        return max(self.proximity[i].getValue() for i in (0, 1, 6, 7)) > threshold

    def cancel(self):
        self.stop()
        self.active_location = None
        self.waypoints = []
