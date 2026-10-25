"""ArucoDetector node: finds ArUco markers in the camera pictures and works out where they are.

Subscribes:
    camera/image          sensor_msgs/Image (rgb8 or bgr8)
    camera/camera_info    sensor_msgs/CameraInfo: the camera's intrinsics
Publishes:
    detected_markers      aruco_interfaces/ArucoMarkerArray: every marker found in the last picture, with
                          its pose in the camera frame
    detected_markers_viz  visualization_msgs/MarkerArray: the same markers, drawn for RViz
    camera/image_annotated  sensor_msgs/Image: the picture with the markers and their axes drawn on it
    TF                    camera -> detected_<id> for each marker found (only if publish_tf is true)
Parameters:
    dictionary     ArUco dictionary (default "DICT_4X4_50")
    marker_size    edge length of the black square of a marker, in metres (default 0.15)
    publish_tf     publish a TF frame for each detection (default true)
"""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from aruco_interfaces.msg import ArucoMarker, ArucoMarkerArray
from aruco_localization.transforms import pose_msg_from_matrix, transform_msg_from_matrix

VIZ_LIFETIME = 0.3   # seconds a detection stays drawn in RViz without being seen again


def marker_corners(size):
    """The four corners of a marker in its own frame (x right, y up, z out of the printed side), in the
    order the ArUco detector reports them: top-left, top-right, bottom-right, bottom-left."""
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def marker_pose_from_corners(image_corners, size, camera_matrix, distortion):
    """Pose of one marker in the camera frame, from its four corners in the picture.

    Args:
        image_corners: 4x2 array, the marker's corners in pixels, in the detector's order
        size: edge length of the marker in metres
        camera_matrix: 3x3 intrinsics matrix K
        distortion: the distortion coefficients
    Returns:
        T_camera_marker, a 4x4 matrix: the marker's pose in the camera frame. The camera frame is
        OpenCV's: x to the right of the picture, y down, z forward out of the lens.
    """
    T = np.eye(4)
    # TODO (Detection, page 2): find the marker's pose in the camera frame from its four corners.
    # Use cv2.solvePnP with marker_corners(size) as the object points and cv2.SOLVEPNP_IPPE_SQUARE, then
    # write the rotation (from the rotation vector) and the translation into T.
    return T


def id_colour(marker_id):
    """A fixed colour for each id, so the same marker always looks the same in RViz."""
    hue = (marker_id * 0.13) % 1.0
    r, g, b = (np.clip(abs(hue * 6.0 - k) - 1.0, 0.0, 1.0) for k in (3.0, 2.0, 4.0))
    return float(r), float(1.0 - g), float(1.0 - b)


class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_size", 0.15)
        self.declare_parameter("publish_tf", True)
        self.marker_size = self.get_parameter("marker_size").value
        self.publish_tf = self.get_parameter("publish_tf").value

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.get_parameter("dictionary").value))
        parameters = cv2.aruco.DetectorParameters()
        # Refine the corners to a fraction of a pixel: the pose is only as good as the corners.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        self.camera_matrix = None
        self.distortion = None
        self.create_subscription(CameraInfo, "camera/camera_info", self.on_camera_info, 10)
        self.create_subscription(Image, "camera/image", self.on_image, 10)
        self.markers_pub = self.create_publisher(ArucoMarkerArray, "detected_markers", 10)
        self.viz_pub = self.create_publisher(MarkerArray, "detected_markers_viz", 10)
        self.annotated_pub = self.create_publisher(Image, "camera/image_annotated", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info(f"ArucoDetector started (marker size {self.marker_size} m). Waiting for pictures.")

    def on_camera_info(self, msg):
        self.camera_matrix = np.array(msg.k, float).reshape(3, 3)
        self.distortion = np.array(msg.d, float)

    def on_image(self, msg):
        if self.camera_matrix is None:
            self.get_logger().warn("No camera_info received yet: cannot compute marker poses.", throttle_duration_sec=5.0)
            return
        picture = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
        corners, ids, _ = self.detector.detectMarkers(picture)
        ids = [] if ids is None else [int(i) for i in ids.ravel()]

        detections = ArucoMarkerArray()
        detections.header = msg.header
        poses = {}
        for marker_id, image_corners in zip(ids, corners):
            T_camera_marker = marker_pose_from_corners(image_corners.reshape(4, 2), self.marker_size,
                                                       self.camera_matrix, self.distortion)
            poses[marker_id] = T_camera_marker
            detections.markers.append(ArucoMarker(id=marker_id, pose=pose_msg_from_matrix(T_camera_marker)))
        self.markers_pub.publish(detections)

        self.publish_viz(msg.header, poses)
        if self.publish_tf and poses:
            transforms = []
            for marker_id, T in poses.items():
                t = TransformStamped()
                t.header = msg.header
                t.child_frame_id = f"detected_{marker_id}"
                t.transform = transform_msg_from_matrix(T)
                transforms.append(t)
            self.tf_broadcaster.sendTransform(transforms)
        if self.annotated_pub.get_subscription_count() > 0:
            self.publish_annotated(msg.header, picture, corners, ids, poses)

    def publish_viz(self, header, poses):
        """Each detection as a flat square in its colour, with its id written above it."""
        viz = MarkerArray()
        for marker_id, T in poses.items():
            colour = id_colour(marker_id)
            square = Marker(header=header, ns="detected_markers", id=marker_id, type=Marker.CUBE, action=Marker.ADD)
            square.pose = pose_msg_from_matrix(T)
            square.scale.x = square.scale.y = self.marker_size
            square.scale.z = 0.005
            square.color.r, square.color.g, square.color.b = colour
            square.color.a = 0.8
            square.lifetime.nanosec = int(VIZ_LIFETIME * 1e9)
            label = Marker(header=header, ns="detected_ids", id=marker_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            above = T @ np.array([0.0, 0.7 * self.marker_size, 0.0, 1.0])   # a bit above the marker
            label.pose.position.x, label.pose.position.y, label.pose.position.z = (float(v) for v in above[:3])
            label.scale.z = 0.08
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"detected {marker_id}"
            label.lifetime.nanosec = int(VIZ_LIFETIME * 1e9)
            viz.markers += [square, label]
        self.viz_pub.publish(viz)

    def publish_annotated(self, header, picture, corners, ids, poses):
        annotated = picture.copy()
        if ids:
            cv2.aruco.drawDetectedMarkers(annotated, corners, np.array(ids).reshape(-1, 1))
        for marker_id, T in poses.items():
            rvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
            cv2.drawFrameAxes(annotated, self.camera_matrix, self.distortion, rvec, T[:3, 3], self.marker_size / 2)
            cv2.putText(annotated, f"{np.linalg.norm(T[:3, 3]):.2f} m", tuple(int(v) for v in corners[ids.index(marker_id)][0][2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        msg = Image()
        msg.header = header
        msg.height, msg.width = annotated.shape[:2]
        msg.encoding = "bgr8"
        msg.step = annotated.shape[1] * 3
        msg.data = annotated.tobytes()
        self.annotated_pub.publish(msg)


def main():
    rclpy.init()
    node = ArucoDetector()
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
