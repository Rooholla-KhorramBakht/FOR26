"""ROS 2 node that runs the LeKiwi simulator.

Subscribes:
    arm_commands   std_msgs/Float64MultiArray, 6 values: 5 joint angles (deg) and the gripper (%)
    base_commands  std_msgs/Float64MultiArray, 3 values: forward (m/s), left (m/s), turn (deg/s)
    fk_pose        geometry_msgs/PoseStamped: a position in the robot's base frame. The simulator
                   draws a pink ball there, so you can see where your forward kinematics points.
Publishes:
    arm_state      std_msgs/Float64MultiArray, 6 values, same layout as arm_commands
    base_pose      std_msgs/Float64MultiArray, 3 values: x (m), y (m), heading (deg)
    base_velocity  geometry_msgs/TwistStamped: the base's velocity as the wheels measure it (with the error
                   set by odometry_noise), in the base frame: linear.x forward (m/s), linear.y left (m/s),
                   angular.z counter-clockwise (rad/s). The stamp counts simulated time, so the time
                   between two stamps is exactly the simulated time between the two readings, even when
                   the simulator runs slower than real time.
    true_pose      geometry_msgs/PoseStamped: where the robot's base really is, in the "map" frame
                   (the simulated world). Only the simulator knows this; use it to check your estimates.
    true_fingertip geometry_msgs/PointStamped: where the fingertip really is, in the base frame
    key_pressed    std_msgs/String: the name of a key, each time one is pushed down (keyboard scene)
    camera/image        sensor_msgs/Image (only if the parameter publish_images is true)
    camera/camera_info  sensor_msgs/CameraInfo: the simulated camera's intrinsics, with every image
Parameters:
    scene           "cubes" (default), "keyboard", "empty" or "aruco"
    map_file        for scene "aruco": the marker map file (YAML) whose markers are put in the room
    start_pose      [x (m), y (m), heading (deg)] where the robot starts (default [0, 0, 0])
    viewer          open the 3-D window (default true; needs a monitor)
    publish_images  publish camera pictures (default false)
    camera          which camera to publish (default "front")
    camera_frame    frame_id of the published images (default: the camera's name)
    image_width, image_height   size of the published images (default 320 x 240)
    odometry_noise  error of base_velocity, as a fraction (default 0.0: perfect). For example 0.05: each
                    reading is off by about 5 %, plus a constant error of about 2.5 % that stays the same
                    for the whole run, like a wheel radius that is slightly wrong.
"""
import math

import numpy as np
import yaml
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String

from lekiwi_sim import LeKiwiSim

BASE_COMMAND_TIMEOUT = 0.5  # seconds. Like the real robot: no fresh base command, no driving.
IMAGE_EVERY_N_STEPS = 5     # 50 steps per second / 5 = 10 pictures per second


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")
        self.declare_parameter("viewer", True)
        self.declare_parameter("publish_images", False)
        self.declare_parameter("camera", "front")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)
        self.declare_parameter("map_file", "")
        self.declare_parameter("start_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("odometry_noise", 0.0)
        self.publish_images = self.get_parameter("publish_images").value
        self.camera = self.get_parameter("camera").value
        self.camera_frame = self.get_parameter("camera_frame").value or self.camera
        self.image_size = (self.get_parameter("image_width").value, self.get_parameter("image_height").value)
        self.odometry_noise = self.get_parameter("odometry_noise").value
        self.rng = np.random.default_rng()
        # The constant part of the odometry error, drawn once per run.
        self.odometry_bias = self.rng.normal(0.0, self.odometry_noise / 2, 3)

        self.declare_parameter("scene", "cubes")
        scene = self.get_parameter("scene").value
        aruco_markers = None
        if scene == "aruco":
            map_file = self.get_parameter("map_file").value
            if not map_file:
                raise RuntimeError("The aruco scene needs a marker map: -p map_file:=/path/to/map.yaml")
            with open(map_file) as f:
                aruco_markers = yaml.safe_load(f)
        self.sim = LeKiwiSim(scene=scene, aruco_markers=aruco_markers)
        self.sim.set_base_pose(*self.get_parameter("start_pose").value)
        if self.get_parameter("viewer").value:
            self.sim.open_viewer()
        self.has_viewer = self.sim.viewer_is_open()

        self.create_subscription(Float64MultiArray, "arm_commands", self.on_arm_command, 10)
        self.create_subscription(Float64MultiArray, "base_commands", self.on_base_command, 10)
        self.arm_state_pub = self.create_publisher(Float64MultiArray, "arm_state", 10)
        self.base_pose_pub = self.create_publisher(Float64MultiArray, "base_pose", 10)
        self.base_velocity_pub = self.create_publisher(TwistStamped, "base_velocity", 10)
        self.true_pose_pub = self.create_publisher(PoseStamped, "true_pose", 10)
        self.image_pub = self.create_publisher(Image, "camera/image", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "camera/camera_info", 10)
        self.create_subscription(PoseStamped, "fk_pose", self.on_fk_pose, 10)
        self.fingertip_pub = self.create_publisher(PointStamped, "true_fingertip", 10)
        self.key_pub = self.create_publisher(String, "key_pressed", 10)
        self.keys_down = set()

        self.last_base_command_time = None
        self.steps = 0
        self.start_time = self.get_clock().now()   # base_velocity stamps: this plus the simulated time
        self.create_timer(self.sim.control_dt, self.on_timer)
        self.get_logger().info("Simulator running. Waiting for arm_commands and base_commands.")

    def on_arm_command(self, msg):
        if len(msg.data) != 6:
            self.get_logger().warn(f"arm_commands needs 6 values, got {len(msg.data)}. Ignored.")
            return
        self.sim.set_command(arm=list(msg.data))

    def on_base_command(self, msg):
        if len(msg.data) != 3:
            self.get_logger().warn(f"base_commands needs 3 values, got {len(msg.data)}. Ignored.")
            return
        self.sim.set_command(base=list(msg.data))
        self.last_base_command_time = self.get_clock().now()

    def on_fk_pose(self, msg):
        p = msg.pose.position
        self.sim.set_marker("fk", self.sim.base_to_world([p.x, p.y, p.z]))

    def on_timer(self):
        if self.has_viewer and not self.sim.viewer_is_open():
            self.get_logger().info("Viewer closed, stopping.")
            raise SystemExit

        # Stop the base if its last command is too old.
        if self.last_base_command_time is not None:
            age = (self.get_clock().now() - self.last_base_command_time).nanoseconds / 1e9
            if age > BASE_COMMAND_TIMEOUT:
                self.sim.set_command(base=[0.0, 0.0, 0.0])
                self.last_base_command_time = None

        self.sim.step()
        self.steps += 1
        state = self.sim.get_state()
        self.arm_state_pub.publish(Float64MultiArray(data=state["arm_position"].tolist()))
        self.base_pose_pub.publish(Float64MultiArray(data=state["base_pose"].tolist()))
        velocity = state["base_velocity"] * (1.0 + self.odometry_bias + self.rng.normal(0.0, self.odometry_noise, 3))
        twist = TwistStamped()
        twist.header.stamp = (self.start_time + Duration(nanoseconds=int(state["time"] * 1e9))).to_msg()
        twist.header.frame_id = "base"
        twist.twist.linear.x, twist.twist.linear.y = float(velocity[0]), float(velocity[1])
        twist.twist.angular.z = math.radians(velocity[2])
        self.base_velocity_pub.publish(twist)
        self.true_pose_pub.publish(self.make_true_pose(state["base_pose"]))
        tip = PointStamped()
        tip.header.stamp = self.get_clock().now().to_msg()
        tip.header.frame_id = "base"
        tip.point.x, tip.point.y, tip.point.z = (float(v) for v in state["fingertip_in_base"])
        self.fingertip_pub.publish(tip)
        for name, key in state["keys"].items():
            if key["pressed"] and name not in self.keys_down:
                self.keys_down.add(name)
                self.key_pub.publish(String(data=name))
                self.get_logger().info(f"{name} pressed")
            elif not key["pressed"]:
                self.keys_down.discard(name)

        if self.publish_images and self.steps % IMAGE_EVERY_N_STEPS == 0:
            picture = self.sim.get_image(self.camera, *self.image_size)
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            msg.height, msg.width = picture.shape[0], picture.shape[1]
            msg.encoding = "rgb8"
            msg.step = picture.shape[1] * 3
            msg.data = picture.tobytes()
            self.image_pub.publish(msg)
            self.camera_info_pub.publish(self.make_camera_info(msg.header))

    def make_true_pose(self, base_pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y = float(base_pose[0]), float(base_pose[1])
        half = math.radians(base_pose[2]) / 2
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(half), math.cos(half)
        return msg

    def make_camera_info(self, header):
        """Intrinsics of the simulated camera: a perfect pinhole camera with the model's field of view."""
        width, height = self.image_size
        fovy = self.sim.model.cam_fovy[self.sim.model.camera(self.camera).id]
        f = (height / 2) / math.tan(math.radians(fovy) / 2)
        info = CameraInfo()
        info.header = header
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [f, 0.0, width / 2, 0.0, f, height / 2, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, width / 2, 0.0, 0.0, f, height / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.sim.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
