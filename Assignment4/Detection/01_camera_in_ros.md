# 1. The Camera in ROS

## Two cameras, one interface

| | Simulator | Real robot |
| --- | --- | --- |
| Node | `sim_node` (it renders the robot's front camera) | `camera_node` (it reads the USB camera) |
| Picture size | 640 x 480 | 640 x 480 (set with parameters `width`, `height`) |
| Intrinsics | exact: a perfect pinhole camera with a 70 degree vertical field of view, no distortion | from your calibration file |
| Topics | `/camera/image`, `/camera/camera_info` | the same |

Every node after the camera sees the same topics, and does not know which camera it is. That is the point: what works in the simulator runs on the robot unchanged.

## Start it

Simulator, with the detector, odometry and the map, but without the particle filter:

```bash
ros2 launch aruco_localization localization.launch.py localization:=false
```

The MuJoCo window shows the robot in a 6 m x 5 m room with ten markers on its walls, and RViz opens with the robot's frame in the middle.

Real robot (motors stay off: nothing moves):

```bash
ros2 launch aruco_localization localization.launch.py robot:=real port:=/dev/ttyACM0 localization:=false
```

The serial port is the robot's motor board, as in Assignment 03 (`ls /dev/serial/by-id/`). The launch file looks for your calibration in `config/camera_intrinsics.yaml` (see the [calibration guide](../Calibration/README.md)) and stops with a message if it is not there.

## Look at the pictures

In RViz, the `Camera` panel shows `/camera/image_annotated`. Or, in a new terminal:

```bash
ros2 run rqt_image_view rqt_image_view /camera/image_annotated
```

Check the rates:

```bash
ros2 topic hz /camera/image          # simulator: about 10 per second, real camera: about 30
ros2 topic echo --once /camera/camera_info
```

## Where the camera is on the robot

The detector reports markers relative to the **camera**. The localization needs them relative to the **robot's base**. So ROS must know where the camera sits on the robot: a transform from `base` to the camera. The launch file publishes it in two steps:

| Transform | What it says | Values |
| --- | --- | --- |
| base → camera_link | where the camera is, and how far it is tilted up | launch arguments `camera_x`, `camera_y`, `camera_z` (m) and `camera_pitch` (deg, positive = tilted up) |
| camera_link → camera | turn the axes from "x forward, z up" (the ROS convention for bodies) to "z forward, y down" (the convention of cameras and OpenCV) | fixed |

The defaults are the simulated robot's camera: 12.5 cm in front of the base's centre, 3 cm above it, looking straight ahead.

**On the real robot, measure your camera** and pass the values:

1. `camera_x`: the distance from the base's centre (the middle between the three wheels) forward to the lens.
2. `camera_y`: how far the lens is to the left of the centre line (usually 0).
3. `camera_z`: the lens height above the base frame's origin (the centre of the bottom plate).
4. `camera_pitch`: how much the camera looks up. Measure the angle of the camera's front face against vertical, or estimate it from the picture: the horizon (the height of the lens, far away) appears `fy * tan(pitch)` pixels below the picture's centre.

```bash
ros2 launch aruco_localization localization.launch.py robot:=real port:=/dev/ttyACM0 localization:=false \
    camera_x:=0.12 camera_z:=0.10 camera_pitch:=15
```

(The numbers above are an example, not your robot's.) Errors here go straight into every marker measurement: a pitch error of 2 degrees moves a marker 3 m away by 10 cm up or down; a yaw error (camera turned left or right) moves it sideways by the same.

## Check it

1. `ros2 run tf2_tools view_frames` writes `frames.pdf`. It must show `base → camera_link → camera`.
2. In RViz (fixed frame `base`), the `TF` display shows the `camera` frame in front of the robot, with its blue (z) axis pointing forward, out of the lens, and its green (y) axis pointing down.

---

Next: [2. The ArUco detector](02_aruco_detector.md)
