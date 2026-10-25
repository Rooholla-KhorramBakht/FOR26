"""ROS 2 node for a USB camera: publishes its pictures and its calibration.

Publishes:
    camera/image        sensor_msgs/Image, encoding bgr8
    camera/camera_info  sensor_msgs/CameraInfo, read from the intrinsics file (one with every image)
Parameters:
    device           the camera: a number (0 = /dev/video0) or a path such as /dev/video0 (default 0)
    width, height    picture size to ask the camera for (default 640 x 480)
    fps              pictures per second to ask for (default 30)
    intrinsics_file  YAML file written by calibrate_camera.py. Without it no camera_info is published,
                     and nothing that needs the calibration (the ArUco detector) will work.
    frame_id         frame of the published pictures (default "camera")
"""
import cv2
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def load_camera_info(path):
    """Read a calibration file in the ROS camera_info YAML layout into a CameraInfo message."""
    with open(path) as f:
        data = yaml.safe_load(f)
    info = CameraInfo()
    info.width, info.height = data["image_width"], data["image_height"]
    info.distortion_model = data.get("distortion_model", "plumb_bob")
    info.k = [float(v) for v in data["camera_matrix"]["data"]]
    info.d = [float(v) for v in data["distortion_coefficients"]["data"]]
    info.r = [float(v) for v in data["rectification_matrix"]["data"]]
    info.p = [float(v) for v in data["projection_matrix"]["data"]]
    return info


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("device", "0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("intrinsics_file", "")
        self.declare_parameter("frame_id", "camera")
        self.frame_id = self.get_parameter("frame_id").value

        device = str(self.get_parameter("device").value)
        self.capture = cv2.VideoCapture(int(device) if device.isdigit() else device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {device}. List the cameras with: ls /dev/video*")
        # MJPG lets USB 2 cameras deliver full-size pictures at a full frame rate.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_parameter("width").value)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_parameter("height").value)
        self.capture.set(cv2.CAP_PROP_FPS, self.get_parameter("fps").value)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera {device} open: {width} x {height} at {self.capture.get(cv2.CAP_PROP_FPS):.0f} fps")

        self.camera_info = None
        path = self.get_parameter("intrinsics_file").value
        if path:
            self.camera_info = load_camera_info(path)
            if (self.camera_info.width, self.camera_info.height) != (width, height):
                raise RuntimeError(f"The calibration in {path} is for {self.camera_info.width} x {self.camera_info.height} "
                                   f"pictures, but the camera delivers {width} x {height}. Calibrate at this size.")
        else:
            self.get_logger().warn("No intrinsics_file given: publishing pictures without camera_info.")

        self.image_pub = self.create_publisher(Image, "camera/image", 10)
        self.info_pub = self.create_publisher(CameraInfo, "camera/camera_info", 10)
        self.create_timer(1.0 / self.get_parameter("fps").value, self.on_timer)

    def on_timer(self):
        ok, picture = self.capture.read()   # waits for the next picture from the camera
        if not ok:
            self.get_logger().warn("No picture from the camera.", throttle_duration_sec=2.0)
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = picture.shape[:2]
        msg.encoding = "bgr8"
        msg.step = picture.shape[1] * 3
        msg.data = picture.tobytes()
        self.image_pub.publish(msg)
        if self.camera_info is not None:
            self.camera_info.header = msg.header
            self.info_pub.publish(self.camera_info)


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.capture.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
