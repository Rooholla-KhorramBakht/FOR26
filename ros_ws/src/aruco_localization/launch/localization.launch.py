"""Start everything for Assignment 4: the robot (simulated or real), the camera, the ArUco detector,
odometry, the marker map, the particle filter and RViz.

Usage:
    ros2 launch aruco_localization localization.launch.py                      # simulator
    ros2 launch aruco_localization localization.launch.py robot:=real port:=/dev/ttyACM0
Optional arguments (name:=value):
    robot          sim (default) or real
    map            marker map file (default: config/sim_room_map.yaml in the simulator,
                   config/classroom_map.yaml on the real robot)
    localization   true (default): run the particle filter. false: only camera, detector, odometry
                   and map, with RViz showing the robot's own frame (for checking the detector).
    rviz           true (default) or false
    viewer         simulator only: open the MuJoCo window (default true)
    start_pose     simulator only: where the robot starts, "x y heading_deg" (default "0 0 0")
    odometry_noise simulator only: odometry error as a fraction (default 0.05)
    port           real robot only: serial port of the robot's motor board
    enable_motors  real robot only: true to let the base drive (default false: read only)
    intrinsics     real robot only: camera calibration file (default config/camera_intrinsics.yaml)
    camera_device  real robot only: the camera (default 0, that is /dev/video0)
    camera_x, camera_y, camera_z, camera_pitch
                   where the camera sits on the robot: position in the base frame (m) and how far it
                   is tilted UP (deg). Defaults: the simulated camera's mount (0.125, 0, 0.03, 0).
                   Measure your real robot's camera and pass the values.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

ARGUMENTS = {
    "robot": "sim", "map": "", "localization": "true", "rviz": "true", "viewer": "true", "start_pose": "0 0 0",
    "odometry_noise": "0.05", "port": "", "enable_motors": "false", "intrinsics": "", "camera_device": "0",
    "camera_x": "0.125", "camera_y": "0.0", "camera_z": "0.03", "camera_pitch": "0.0",
}


def launch_nodes(context):
    arg = {name: context.launch_configurations[name] for name in ARGUMENTS}
    share = get_package_share_directory("aruco_localization")
    sim = arg["robot"] == "sim"
    if arg["robot"] not in ("sim", "real"):
        raise RuntimeError("robot must be sim or real")
    map_file = arg["map"] or os.path.join(share, "config", "sim_room_map.yaml" if sim else "classroom_map.yaml")
    with open(map_file) as f:
        marker_map = yaml.safe_load(f)
    localization = arg["localization"] == "true"

    nodes = []
    if sim:
        nodes.append(Node(package="lekiwi_ros", executable="sim_node", output="screen", parameters=[{
            "scene": "aruco", "map_file": map_file, "viewer": arg["viewer"] == "true",
            "start_pose": [float(v) for v in arg["start_pose"].split()],
            "publish_images": True, "camera": "front", "camera_frame": "camera",
            "image_width": 640, "image_height": 480, "odometry_noise": float(arg["odometry_noise"])}]))
    else:
        if not arg["port"]:
            raise RuntimeError("robot:=real needs the motor board's serial port, for example port:=/dev/ttyACM0")
        nodes.append(Node(package="lekiwi_ros", executable="robot_node", output="screen", parameters=[{
            "port": arg["port"], "enable_motors": arg["enable_motors"] == "true", "rate_hz": 20.0}]))
        intrinsics = arg["intrinsics"] or os.path.join(share, "config", "camera_intrinsics.yaml")
        if not os.path.exists(intrinsics):
            raise RuntimeError(f"No camera calibration at {intrinsics}. Calibrate the camera first "
                               "(Assignment4/Calibration) and pass the file with intrinsics:=/path/to/file.yaml")
        nodes.append(Node(package="aruco_localization", executable="camera_node", output="screen", parameters=[{
            "device": arg["camera_device"], "frame_id": "camera", "intrinsics_file": intrinsics}]))

    # Where the camera sits on the robot. camera_link: x forward along the lens, y left, z up.
    # camera: the same point, turned to the "optical" convention that OpenCV and the detector use
    # (x right in the picture, y down, z forward out of the lens).
    nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", arguments=[
        "--x", arg["camera_x"], "--y", arg["camera_y"], "--z", arg["camera_z"],
        "--pitch", str(-float(arg["camera_pitch"]) * 3.141592653589793 / 180.0),
        "--frame-id", "base", "--child-frame-id", "camera_link"]))
    nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", arguments=[
        "--roll", "-1.5707963", "--yaw", "-1.5707963", "--frame-id", "camera_link", "--child-frame-id", "camera"]))

    nodes += [
        Node(package="aruco_localization", executable="aruco_detector", output="screen", parameters=[{
            "dictionary": marker_map.get("dictionary", "DICT_4X4_50"), "marker_size": float(marker_map["marker_size"])}]),
        Node(package="aruco_localization", executable="odometry_node", output="screen"),
        Node(package="aruco_localization", executable="map_node", output="screen", parameters=[{"map_file": map_file}]),
    ]
    if localization:
        nodes.append(Node(package="aruco_localization", executable="localization_node", output="screen",
                          parameters=[os.path.join(share, "config", "localization.yaml")]))
    if arg["rviz"] == "true":
        config = "localization.rviz" if localization else "detection.rviz"
        nodes.append(Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", config)]))
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(name, default_value=value) for name, value in ARGUMENTS.items()]
                             + [OpaqueFunction(function=launch_nodes)])
