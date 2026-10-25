"""Map node: publishes where the ArUco markers in the room are (the known map).

Publishes:
    map_markers      aruco_interfaces/ArucoMarkerArray in the "map" frame: the pose of every marker
    map_markers_viz  visualization_msgs/MarkerArray: the same markers drawn for RViz, with their ids
    TF (static)      map -> marker_<id> for every marker
Parameters:
    map_file         the marker map (YAML) written by build_map.py (required)

Both topics are published once a second and with "transient local" durability, so a node that starts
later still receives the map straight away.
"""
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from aruco_interfaces.msg import ArucoMarker, ArucoMarkerArray
from aruco_localization.transforms import pose_msg_from_matrix, transform_msg_from_matrix

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def load_marker_map(path):
    """Read a marker map file. Returns (marker size in metres, {id: T_map_marker as a 4x4 matrix})."""
    with open(path) as f:
        data = yaml.safe_load(f)
    markers = {}
    for marker in data["markers"]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(marker["orientation"]).as_matrix()
        T[:3, 3] = marker["position"]
        markers[int(marker["id"])] = T
    return float(data["marker_size"]), markers


class MapNode(Node):
    def __init__(self):
        super().__init__("map_node")
        self.declare_parameter("map_file", "")
        path = self.get_parameter("map_file").value
        if not path:
            raise RuntimeError("Give the marker map: -p map_file:=/path/to/map.yaml")
        self.marker_size, self.markers = load_marker_map(path)

        self.markers_pub = self.create_publisher(ArucoMarkerArray, "map_markers", LATCHED)
        self.viz_pub = self.create_publisher(MarkerArray, "map_markers_viz", LATCHED)
        self.static_tf = StaticTransformBroadcaster(self)
        transforms = []
        for marker_id, T in self.markers.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = f"marker_{marker_id}"
            t.transform = transform_msg_from_matrix(T)
            transforms.append(t)
        self.static_tf.sendTransform(transforms)

        self.publish()
        self.create_timer(1.0, self.publish)
        self.get_logger().info(f"Map with {len(self.markers)} markers loaded from {path}: ids {sorted(self.markers)}")

    def publish(self):
        msg = ArucoMarkerArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        viz = MarkerArray()
        for marker_id, T in self.markers.items():
            msg.markers.append(ArucoMarker(id=marker_id, pose=pose_msg_from_matrix(T)))
            square = Marker(header=msg.header, ns="map_markers", id=marker_id, type=Marker.CUBE, action=Marker.ADD)
            square.pose = pose_msg_from_matrix(T)
            square.scale.x = square.scale.y = self.marker_size
            square.scale.z = 0.01
            square.color.r = square.color.g = square.color.b = 0.1
            square.color.a = 1.0
            label = Marker(header=msg.header, ns="map_ids", id=marker_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            above = T @ np.array([0.0, 0.0, 0.0, 1.0]) + [0.0, 0.0, 0.7 * self.marker_size + 0.05, 0.0]
            label.pose.position.x, label.pose.position.y, label.pose.position.z = (float(v) for v in above[:3])
            label.scale.z = 0.1
            label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 0.8, 0.0, 1.0
            label.text = f"marker {marker_id}"
            viz.markers += [square, label]
        self.markers_pub.publish(msg)
        self.viz_pub.publish(viz)


def main():
    rclpy.init()
    node = MapNode()
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
