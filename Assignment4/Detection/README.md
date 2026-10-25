# Marker Detection

In this guide the robot learns to see the markers: the camera's pictures come into ROS, and your detector finds the ArUco markers in them and works out where each one is, relative to the camera.

| # | Page | What you do |
| --- | --- | --- |
| 1 | [The camera in ROS](01_camera_in_ros.md) | Start the camera (simulated or real), tell ROS where it is mounted on the robot, look at the pictures |
| 2 | [The ArUco detector](02_aruco_detector.md) | Write the marker pose computation, run the detector, check the detected markers in RViz |

## Your file

| File | What you write |
| --- | --- |
| `~/FOR26/ros_ws/src/aruco_localization/aruco_localization/aruco_detector.py` | `marker_pose_from_corners()`: the pose of a marker from its four corners in the picture |

Everything else in the node (subscribing, detecting, publishing, drawing) is given. Until you fill in the TODO, the node runs and reports every marker at the camera's origin.

## The topics

| Topic | Message type | Content |
| --- | --- | --- |
| `/camera/image` | `sensor_msgs/Image` | the camera's pictures |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | the camera's intrinsics, one message per picture |
| `/detected_markers` | `aruco_interfaces/ArucoMarkerArray` | the markers found in the last picture: id and pose, in the `camera` frame |
| `/detected_markers_viz` | `visualization_msgs/MarkerArray` | the same, drawn for RViz |
| `/camera/image_annotated` | `sensor_msgs/Image` | the picture with the detected markers and their axes drawn on it |

The message types of this assignment are in the package `aruco_interfaces`:

```
# ArucoMarker.msg
int32 id
geometry_msgs/Pose pose

# ArucoMarkerArray.msg
std_msgs/Header header       # header.frame_id says which frame the poses are in
ArucoMarker[] markers
```

Show them with `ros2 interface show aruco_interfaces/msg/ArucoMarkerArray`.
