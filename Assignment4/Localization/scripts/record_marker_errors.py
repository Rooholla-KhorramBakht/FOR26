"""Record, for every detected marker, how far from its map position the robot sees it. Writes a CSV.

Usage (in a sourced terminal, while the launch file with the localization runs):

    python record_marker_errors.py errors.csv [--duration 60]

Each time the localization updates, it publishes the detected markers placed in the map through its
estimated pose (/detected_markers_map). This script compares each of them with the same marker in the
map (/map_markers) and writes one row per marker:

    time        seconds since the recording started
    id          the marker
    range       its distance from the robot's estimated position (m)
    seen_x, seen_y, seen_z   where the robot sees it, in the map (m)
    map_x, map_y, map_z      where the map has it (m)
    error_xy    distance between the two on the floor plan (m)
    error_z     difference in height (m)
    error_facing  difference of the directions the marker faces (deg)

If the map, the detection and the estimate were all perfect, every error would be 0. Plot the result
with: python plot_marker_errors.py errors.csv
"""
import argparse
import csv
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from aruco_interfaces.msg import ArucoMarkerArray

COLUMNS = ["time", "id", "range", "seen_x", "seen_y", "seen_z", "map_x", "map_y", "map_z",
           "error_xy", "error_z", "error_facing"]


def facing(q):
    """Direction (rad) of a marker's z axis in the floor plane, from its orientation quaternion."""
    # The marker's z axis is the third column of its rotation matrix: (2(xz + wy), 2(yz - wx), ...).
    return math.atan2(2.0 * (q.y * q.z - q.w * q.x), 2.0 * (q.x * q.z + q.w * q.y))


class Recorder(Node):
    def __init__(self, path, duration):
        super().__init__("record_marker_errors")
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.map = {}
        self.robot = None
        self.start = None
        self.rows = 0
        self.duration = duration
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(ArucoMarkerArray, "map_markers", self.on_map, latched)
        self.create_subscription(PoseWithCovarianceStamped, "pose_estimate", self.on_estimate, 10)
        self.create_subscription(ArucoMarkerArray, "detected_markers_map", self.on_detected, 10)
        self.get_logger().info(f"Recording to {path}. Ctrl+C to stop.")

    def on_map(self, msg):
        self.map = {m.id: m.pose for m in msg.markers}

    def on_estimate(self, msg):
        self.robot = msg.pose.pose.position

    def on_detected(self, msg):
        if not self.map or self.robot is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.start is None:
            self.start = now
        for marker in msg.markers:
            if marker.id not in self.map:
                continue
            seen, known = marker.pose, self.map[marker.id]
            s, k = seen.position, known.position
            difference = math.degrees(facing(seen.orientation) - facing(known.orientation))
            difference = (difference + 180.0) % 360.0 - 180.0
            self.writer.writerow([f"{now - self.start:.3f}", marker.id,
                                  math.hypot(s.x - self.robot.x, s.y - self.robot.y),
                                  s.x, s.y, s.z, k.x, k.y, k.z,
                                  math.hypot(s.x - k.x, s.y - k.y), s.z - k.z, difference])
            self.rows += 1
        if self.duration and now - self.start >= self.duration:
            raise SystemExit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to record (default: until Ctrl+C)")
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.output, args.duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.file.close()
        print(f"Wrote {node.rows} rows to {args.output}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
