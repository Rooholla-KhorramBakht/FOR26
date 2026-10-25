"""Localization node: runs the particle filter and tells the rest of ROS where the robot is in the map.

Subscribes:
    odom              nav_msgs/Odometry: the robot's dead-reckoned pose (from odometry_node). Each change
                      is a predict step of the filter.
    detected_markers  aruco_interfaces/ArucoMarkerArray: markers the camera sees now, in the camera frame
                      (from aruco_detector). Each message is an update step.
    map_markers       aruco_interfaces/ArucoMarkerArray: where the markers are in the map (from map_node)
    initialpose       geometry_msgs/PoseWithCovarianceStamped: "the robot is about here". RViz's
                      "2D Pose Estimate" button publishes it. The particles are spread around that pose.
Publishes:
    particles         aruco_interfaces/ParticleArray: every particle's pose and weight, in the map frame
    particles_viz     visualization_msgs/MarkerArray: the particles drawn for RViz, coloured by weight
    pose_estimate     geometry_msgs/PoseWithCovarianceStamped: the estimated pose of the base in the map,
                      with the spread of the particles as the covariance
    detected_markers_map  aruco_interfaces/ArucoMarkerArray: the markers just detected, placed in the map
                      frame through the estimated pose. Compare them with map_markers: if the estimate,
                      the detection and the map are all right, they are the same.
    marker_errors_viz visualization_msgs/MarkerArray: for each detected marker, a line from where the
                      estimate says it is to where the map says it is, with the distance written on it
    TF                map -> odom, chosen so that map -> odom -> base is the estimated pose
Reads from TF:
    base -> camera (the camera's mounting position on the robot)
Parameters:
    num_particles          (default 2000)
    estimate               "best" (the particle with the largest weight, default) or "mean"
    motion_noise           [alpha_translation, alpha_rotation, sigma_xy_min (m), sigma_theta_min (rad)]
                           (default [0.1, 0.1, 0.002, 0.002])
    measurement_sigmas     [range (fraction of the range), bearing (rad), facing (rad per metre of range)]
                           (default [0.1, 0.087, 0.1])
    resample_noise         [xy (m), theta (rad)] jitter added after resampling (default [0.01, 0.01])
    resample_threshold     resample when the effective number of particles drops below this fraction
                           of num_particles (default 0.5)
    random_fraction        fraction of particles replaced by random ones at each resampling (default 0.01)
    recovery_rates         [slow, fast] running-average rates of the measurement fit; when the fit suddenly
                           drops, more random particles are added (augmented MCL, see ParticleFilter)
    max_range              ignore detections further away than this, in metres (default 5.0)
    update_min_motion      [distance (m), angle (rad)]: update only after the robot has moved this far or
                           turned this much since the last update (default [0.02, 0.03])...
    update_max_interval    ...or when this many seconds have passed since the last update (default 1.0).
                           Ten pictures a second of a robot that stands still are not ten independent
                           measurements: they all have the same error. Using every one of them makes the
                           filter far too sure of itself, and it slowly drifts to wherever that error
                           points.
    map_margin             the area the robot can be in is the box around the markers, grown by this
                           much on every side, in metres (default 0.5)
    base_frame, odom_frame, map_frame   (default "base", "odom", "map")
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aruco_interfaces.msg import ArucoMarker, ArucoMarkerArray, ParticleArray
from aruco_localization.particle_filter import ParticleFilter
from aruco_localization.transforms import (matrix_from_2d, matrix_from_pose_msg, matrix_from_transform_msg,
                                           matrix_to_2d, pose_msg_from_matrix, transform_msg_from_matrix,
                                           yaw_quaternion_msg)

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
VIZ_EVERY = 0.2   # seconds between two particle drawings: drawing every update is too much for RViz


def measurement_from_detection(T_base_camera, T_camera_marker):
    """Turn one detected marker into the filter's 2-D measurement.

    Args:
        T_base_camera: 4x4 pose of the camera in the robot's base frame (x forward, y left, z up)
        T_camera_marker: 4x4 pose of the marker in the camera frame, from the detector
    Returns:
        (range, bearing, facing): the distance from the base's centre to the marker in the floor plane
        (m), its direction counter-clockwise from straight ahead (rad), and the direction the marker's
        printed side (its z axis) faces, in the base frame (rad)
    """
    distance, bearing, facing = 0.0, 0.0, 0.0
    # TODO (Localization, page 2): chain T_base_camera and T_camera_marker to get the marker in the base
    # frame, then compute distance and bearing from its position and facing from its z axis.
    return distance, bearing, facing


def weight_colour(fraction):
    """Colour for a particle: dark blue for a tiny weight, through green, to red for the largest weight."""
    return ColorRGBA(r=float(np.clip(2.0 * fraction - 1.0, 0.0, 1.0)), g=float(1.0 - abs(2.0 * fraction - 1.0)),
                     b=float(np.clip(1.0 - 2.0 * fraction, 0.0, 1.0)), a=1.0)


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        self.declare_parameter("num_particles", 2000)
        self.declare_parameter("estimate", "best")
        self.declare_parameter("motion_noise", [0.1, 0.1, 0.002, 0.002])
        self.declare_parameter("measurement_sigmas", [0.1, 0.087, 0.1])
        self.declare_parameter("resample_noise", [0.01, 0.01])
        self.declare_parameter("resample_threshold", 0.5)
        self.declare_parameter("random_fraction", 0.01)
        self.declare_parameter("recovery_rates", [0.0, 0.0])
        self.declare_parameter("max_range", 5.0)
        self.declare_parameter("update_min_motion", [0.02, 0.03])
        self.declare_parameter("update_max_interval", 1.0)
        self.declare_parameter("map_margin", 0.5)
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        self.estimate_mode = get("estimate")
        if self.estimate_mode not in ("best", "mean"):
            raise RuntimeError('estimate must be "best" or "mean"')
        self.motion_noise = list(get("motion_noise"))
        self.sigmas = list(get("measurement_sigmas"))
        self.resample_noise = list(get("resample_noise"))
        self.base_frame, self.odom_frame, self.map_frame = get("base_frame"), get("odom_frame"), get("map_frame")

        self.filter = None            # created when the map arrives
        self.marker_map = {}          # id -> [x, y, facing] in the map frame
        self.T_map_markers = {}       # id -> 4x4 pose in the map frame (for drawing the errors)
        self.last_odom = None         # 4x4 pose of the base in the odom frame at the last predict step
        self.T_map_odom = np.eye(4)
        self.estimate = None          # [x, y, theta] of the base in the map
        self.odom_at_update = None    # 4x4 odom pose at the last update
        self.time_of_update = None
        self.last_viz_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(ArucoMarkerArray, "map_markers", self.on_map, LATCHED)
        self.create_subscription(Odometry, "odom", self.on_odom, 50)
        self.create_subscription(ArucoMarkerArray, "detected_markers", self.on_detections, 10)
        self.create_subscription(PoseWithCovarianceStamped, "initialpose", self.on_initial_pose, 10)
        self.particles_pub = self.create_publisher(ParticleArray, "particles", 10)
        self.particles_viz_pub = self.create_publisher(MarkerArray, "particles_viz", 10)
        self.estimate_pub = self.create_publisher(PoseWithCovarianceStamped, "pose_estimate", 10)
        self.errors_viz_pub = self.create_publisher(MarkerArray, "marker_errors_viz", 10)
        self.detected_map_pub = self.create_publisher(ArucoMarkerArray, "detected_markers_map", 10)
        self.create_timer(0.05, self.publish_map_to_odom)
        self.get_logger().info("Localization started. Waiting for map_markers.")

    # ------------------------------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------------------------------

    def on_map(self, msg):
        if self.filter is not None:
            return
        for marker in msg.markers:
            T = matrix_from_pose_msg(marker.pose)
            self.T_map_markers[marker.id] = T
            self.marker_map[marker.id] = [T[0, 3], T[1, 3], math.atan2(T[1, 2], T[0, 2])]
        xs = [m[0] for m in self.marker_map.values()]
        ys = [m[1] for m in self.marker_map.values()]
        margin = self.get_parameter("map_margin").value
        bounds = [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin]
        self.filter = ParticleFilter(self.get_parameter("num_particles").value, bounds,
                                     recovery_rates=list(self.get_parameter("recovery_rates").value))
        self.get_logger().info(f"Map received ({len(self.marker_map)} markers). {len(self.filter.particles)} particles "
                               f"spread over x {bounds[0]:.1f}..{bounds[1]:.1f} m, y {bounds[2]:.1f}..{bounds[3]:.1f} m.")

    def on_initial_pose(self, msg):
        if self.filter is None:
            return
        pose = matrix_to_2d(matrix_from_pose_msg(msg.pose.pose))
        self.filter.initialize_around(pose, 0.3, 0.3)
        self.get_logger().info(f"Particles spread around x={pose[0]:.2f}, y={pose[1]:.2f}, "
                               f"heading={math.degrees(pose[2]):.0f} deg.")

    def on_odom(self, msg):
        T_odom_base = matrix_from_pose_msg(msg.pose.pose)
        if self.filter is not None and self.last_odom is not None:
            # The step the robot made since the last message, in its own frame at the start of the step.
            delta = matrix_to_2d(np.linalg.inv(self.last_odom) @ T_odom_base)
            if np.any(np.abs(delta) > 1e-6):
                self.filter.predict(delta, self.motion_noise)
        self.last_odom = T_odom_base

    def update_due(self):
        """True if the robot has moved enough, or enough time has passed, since the last update."""
        if self.odom_at_update is None:
            return True
        moved = matrix_to_2d(np.linalg.inv(self.odom_at_update) @ self.last_odom)
        min_distance, min_angle = self.get_parameter("update_min_motion").value
        elapsed = (self.get_clock().now() - self.time_of_update).nanoseconds / 1e9
        return (math.hypot(moved[0], moved[1]) >= min_distance or abs(moved[2]) >= min_angle
                or elapsed >= self.get_parameter("update_max_interval").value)

    def on_detections(self, msg):
        if self.filter is None or self.last_odom is None or not self.update_due():
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except TransformException as error:
            self.get_logger().warn(f"No transform {self.base_frame} -> {msg.header.frame_id}: {error}",
                                   throttle_duration_sec=5.0)
            return
        T_base_camera = matrix_from_transform_msg(tf.transform)
        max_range = self.get_parameter("max_range").value
        measurements, detections = [], {}
        for marker in msg.markers:
            T_camera_marker = matrix_from_pose_msg(marker.pose)
            measured = measurement_from_detection(T_base_camera, T_camera_marker)
            if marker.id in self.marker_map and measured[0] <= max_range:
                measurements.append((marker.id, *measured))
                detections[marker.id] = T_base_camera @ T_camera_marker

        if self.filter.update(measurements, self.marker_map, self.sigmas):
            self.odom_at_update, self.time_of_update = self.last_odom, self.get_clock().now()
            # Take the estimate now, while the weights still say which particles are good (after
            # resampling all weights are equal). Between updates, the odometry carries it along.
            self.estimate = self.filter.best() if self.estimate_mode == "best" else self.filter.mean()
            self.T_map_odom = matrix_from_2d(*self.estimate) @ np.linalg.inv(self.last_odom)
            self.publish_particles(msg.header.stamp)
            self.publish_estimate(msg.header.stamp)
            self.publish_marker_errors(msg.header.stamp, detections)
            n = len(self.filter.particles)
            if self.filter.effective_sample_size() < self.get_parameter("resample_threshold").value * n:
                self.filter.resample(*self.resample_noise, self.get_parameter("random_fraction").value)

    # ------------------------------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------------------------------

    def publish_map_to_odom(self):
        """map -> odom such that map -> odom -> base is the estimated pose of the base.

        TF allows only one parent per frame. The odometry node already publishes odom -> base, so the
        localization publishes the correction above it, map -> odom, instead of map -> base. The
        correction changes at every update; in between, odom -> base moves the robot along.
        """
        if self.estimate is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform = transform_msg_from_matrix(self.T_map_odom)
        self.tf_broadcaster.sendTransform(t)

    def publish_estimate(self, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(self.estimate[0]), float(self.estimate[1])
        msg.pose.pose.orientation = yaw_quaternion_msg(self.estimate[2])
        # The message's covariance is 6 x 6 over (x, y, z, roll, pitch, yaw). Fill in x, y and yaw.
        covariance = np.zeros((6, 6))
        c = self.filter.covariance()
        for i, a in enumerate((0, 1, 5)):
            for j, b in enumerate((0, 1, 5)):
                covariance[a, b] = c[i, j]
        msg.pose.covariance = covariance.ravel().tolist()
        self.estimate_pub.publish(msg)

    def publish_particles(self, stamp):
        particles, weights = self.filter.particles, self.filter.weights
        msg = ParticleArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        for x, y, theta in particles:
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            pose.orientation = yaw_quaternion_msg(theta)
            msg.poses.append(pose)
        msg.weights = weights.tolist()
        self.particles_pub.publish(msg)

        now = self.get_clock().now()
        if self.last_viz_time is not None and (now - self.last_viz_time).nanoseconds < VIZ_EVERY * 1e9:
            return
        self.last_viz_time = now
        # One line per particle, pointing where the particle faces, coloured by weight.
        lines = Marker(ns="particles", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        lines.header = msg.header
        lines.scale.x = 0.01
        lines.pose.orientation.w = 1.0
        fractions = weights / weights.max()
        for (x, y, theta), fraction in zip(particles, fractions):
            colour = weight_colour(fraction)
            lines.points += [Point(x=float(x), y=float(y), z=0.02),
                             Point(x=float(x + 0.15 * math.cos(theta)), y=float(y + 0.15 * math.sin(theta)), z=0.02)]
            lines.colors += [colour, colour]
        best = Marker(ns="particles", id=1, type=Marker.ARROW, action=Marker.ADD)
        best.header = msg.header
        best.pose.position.x, best.pose.position.y = float(self.estimate[0]), float(self.estimate[1])
        best.pose.orientation = yaw_quaternion_msg(self.estimate[2])
        best.scale.x, best.scale.y, best.scale.z = 0.35, 0.05, 0.05
        best.color.r, best.color.g, best.color.b, best.color.a = 1.0, 0.0, 1.0, 1.0
        self.particles_viz_pub.publish(MarkerArray(markers=[lines, best]))

    def publish_marker_errors(self, stamp, detections):
        """For each detected marker: where the estimate puts it, joined by a line to where the map has it.

        If the estimate and the detection are both right, the line has zero length. A long line for one
        marker while the others are short points at that marker (a wrong map entry, a bad detection).
        Long lines for all markers point at the estimate.
        """
        T_map_base = matrix_from_2d(*self.estimate)
        in_map = ArucoMarkerArray()
        in_map.header.stamp = stamp
        in_map.header.frame_id = self.map_frame
        in_map.markers = [ArucoMarker(id=marker_id, pose=pose_msg_from_matrix(T_map_base @ T_base_marker))
                          for marker_id, T_base_marker in detections.items()]
        self.detected_map_pub.publish(in_map)
        viz = MarkerArray()
        lines = Marker(ns="marker_errors", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        lines.header.stamp = stamp
        lines.header.frame_id = self.map_frame
        lines.pose.orientation.w = 1.0
        lines.scale.x = 0.02
        lines.color.r, lines.color.a = 1.0, 1.0
        lines.lifetime.nanosec = int(0.5e9)
        for marker_id, T_base_marker in detections.items():
            seen = (T_map_base @ T_base_marker)[:3, 3]
            known = self.T_map_markers[marker_id][:3, 3]
            lines.points += [Point(x=float(seen[0]), y=float(seen[1]), z=float(seen[2])),
                             Point(x=float(known[0]), y=float(known[1]), z=float(known[2]))]
            label = Marker(ns="marker_error_labels", id=marker_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            label.header = lines.header
            label.pose.position.x, label.pose.position.y = float(seen[0]), float(seen[1])
            label.pose.position.z = float(seen[2]) - 0.15
            label.scale.z = 0.07
            label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 0.3, 0.3, 1.0
            label.text = f"{np.linalg.norm(seen[:2] - known[:2]) * 100:.0f} cm"
            label.lifetime.nanosec = int(0.5e9)
            viz.markers.append(label)
        viz.markers.append(lines)
        self.errors_viz_pub.publish(viz)


def main():
    rclpy.init()
    node = LocalizationNode()
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
