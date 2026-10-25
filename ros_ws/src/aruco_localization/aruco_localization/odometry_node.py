"""Odometry node: adds up the robot's measured velocity into a pose (dead reckoning).

Subscribes:
    base_velocity  geometry_msgs/TwistStamped: the velocity measured by the wheels, in the base frame
                   (linear.x forward m/s, linear.y left m/s, angular.z counter-clockwise rad/s), published
                   by robot_node or sim_node. The time step is taken from the stamps.
Publishes:
    odom           nav_msgs/Odometry: the pose in the "odom" frame, and the velocity
    TF             odom -> base
Parameters:
    odom_frame     default "odom"
    base_frame     default "base"

The odom frame is where the robot was when this node started. The pose drifts away from the truth
the longer the robot drives, because every small error in the measured velocity is added up and
never corrected. Correcting it is the job of the localization node.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster

from aruco_localization.transforms import yaw_quaternion_msg

MAX_STEP = 0.5   # seconds. A longer gap between two velocity readings is not integrated.


def integrate(x, y, theta, forward, left, turn, dt):
    """Move a 2-D pose along for dt seconds at a velocity given in the robot's own frame.

    Args:
        x, y, theta: the pose now, in the odom frame (m, m, rad)
        forward, left: the velocity in the robot's frame (m/s)
        turn: the turning rate (rad/s, counter-clockwise)
        dt: the time step (s)
    Returns:
        (x, y, theta) after dt seconds. Keep theta in [-pi, pi).
    """
    # Use the heading halfway through the step: exact for a constant velocity over a short step.
    mid = theta + turn * dt / 2.0
    x += (forward * math.cos(mid) - left * math.sin(mid)) * dt
    y += (forward * math.sin(mid) + left * math.cos(mid)) * dt
    theta = (theta + turn * dt + math.pi) % (2.0 * math.pi) - math.pi
    return x, y, theta


class OdometryNode(Node):
    def __init__(self):
        super().__init__("odometry_node")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base")
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.last_stamp = None
        self.create_subscription(TwistStamped, "base_velocity", self.on_velocity, 50)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("Odometry started at (0, 0, 0). Waiting for base_velocity.")

    def on_velocity(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        forward, left, turn = msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        if self.last_stamp is not None:
            # The time between the two readings, from their stamps (not from when they arrived here).
            dt = (stamp - self.last_stamp).nanoseconds / 1e9
            if 0.0 < dt < MAX_STEP:
                self.x, self.y, self.theta = integrate(self.x, self.y, self.theta, forward, left, turn, dt)
        self.last_stamp = stamp

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation = yaw_quaternion_msg(self.theta)
        odom.twist.twist = msg.twist
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header = odom.header
        t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
