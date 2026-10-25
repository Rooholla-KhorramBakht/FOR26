"""Record what the localization does into a CSV file, for the plots in your report.

Usage (in a sourced terminal, while the launch file runs):

    python record_localization.py run1.csv [--duration 60]

One row is written for every pose_estimate message, with the latest value of everything else:

    time                    seconds since the recording started
    est_x, est_y, est_theta the estimated pose of the base in the map (m, m, rad)    from /pose_estimate
    std_x, std_y, std_theta the spread of the particles (standard deviations)        from /pose_estimate
    true_x, true_y, true_theta  where the robot really is (simulator only, else empty) from /true_pose
    odom_x, odom_y, odom_theta  the dead-reckoned pose in the odom frame                from /odom
    n_detected              how many markers the camera saw in the last picture       from /detected_markers
    detected_ids            their ids, separated by spaces

Stop it with Ctrl+C, or let --duration end it.
"""
import argparse
import csv
import math

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from aruco_interfaces.msg import ArucoMarkerArray

COLUMNS = ["time", "est_x", "est_y", "est_theta", "std_x", "std_y", "std_theta", "true_x", "true_y", "true_theta",
           "odom_x", "odom_y", "odom_theta", "n_detected", "detected_ids"]


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self, path, duration):
        super().__init__("record_localization")
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.true = ["", "", ""]
        self.odom = ["", "", ""]
        self.detected = []
        self.start = None
        self.rows = 0
        self.duration = duration
        self.create_subscription(PoseWithCovarianceStamped, "pose_estimate", self.on_estimate, 10)
        self.create_subscription(PoseStamped, "true_pose", self.on_true_pose, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(ArucoMarkerArray, "detected_markers", self.on_detections, 10)
        self.get_logger().info(f"Recording to {path}. Ctrl+C to stop.")

    def on_true_pose(self, msg):
        self.true = [msg.pose.position.x, msg.pose.position.y, yaw(msg.pose.orientation)]

    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom = [p.position.x, p.position.y, yaw(p.orientation)]

    def on_detections(self, msg):
        self.detected = [m.id for m in msg.markers]

    def on_estimate(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.start is None:
            self.start = now
        p, c = msg.pose.pose, msg.pose.covariance
        row = [f"{now - self.start:.3f}", p.position.x, p.position.y, yaw(p.orientation),
               math.sqrt(max(c[0], 0.0)), math.sqrt(max(c[7], 0.0)), math.sqrt(max(c[35], 0.0))]
        row += self.true + self.odom + [len(self.detected), " ".join(str(i) for i in self.detected)]
        self.writer.writerow(row)
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
