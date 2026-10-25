# 2. The ArUco Detector

## ArUco markers

An ArUco marker is a black square with a pattern of white cells inside, on a white background. The pattern encodes a number, the marker's **id**. Patterns come in **dictionaries**; this course uses `DICT_4X4_50`: 4 x 4 cells inside a black border, ids 0 to 49.

Finding the markers in a picture is done for you by OpenCV:

```python
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
corners, ids, rejected = detector.detectMarkers(picture)
```

For each marker it returns the id and the four corners, in pixels, always in the same order: **top-left, top-right, bottom-right, bottom-left** of the printed marker, whatever way it is turned in the picture.

## From four corners to a pose

Each marker has its own frame: origin at the centre of the black square, x to the right, y up, z out of the printed side. In that frame you know the four corners exactly. For a marker with edge length `s`:

| Corner | In the marker's frame |
| --- | --- |
| top-left | (-s/2, s/2, 0) |
| top-right | (s/2, s/2, 0) |
| bottom-right | (s/2, -s/2, 0) |
| bottom-left | (-s/2, -s/2, 0) |

The detector tells you where those four points appear in the picture. The question "where must the marker be, so that its corners appear exactly there?" is the **Perspective-n-Point** problem (PnP). OpenCV solves it:

```python
ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
```

| Argument / result | Meaning |
| --- | --- |
| `object_points` | the four corners in the marker's frame (4 x 3) |
| `image_points` | the same four corners in the picture, in pixels (4 x 2), in the same order |
| `camera_matrix`, `distortion` | the intrinsics (from `/camera/camera_info`) |
| `SOLVEPNP_IPPE_SQUARE` | a method made for exactly this case: four corners of a square |
| `rvec` | the marker's rotation in the camera frame, as a rotation vector (axis times angle) |
| `tvec` | the marker's centre in the camera frame (m) |

Together they form `T_camera_marker`, the marker's pose in the camera frame. Notice that the result is only as good as the marker size you give: if the printed marker is 10 % larger than you say, every distance comes out 10 % too short.

## Exercise: `marker_pose_from_corners`

Open `aruco_detector.py` and fill in the TODO in `marker_pose_from_corners()`:

1. Call `cv2.solvePnP` with `marker_corners(size)` as the object points.
2. Turn `rvec` into a 3 x 3 rotation matrix. `scipy.spatial.transform.Rotation.from_rotvec(rvec.ravel()).as_matrix()` does it (so does `cv2.Rodrigues`).
3. Put the rotation into `T[:3, :3]` and `tvec` into `T[:3, 3]`.

Test it without ROS. A camera with fx = fy = 500 and its centre at (320, 240) sees a 15 cm marker 1.2 m ahead, 20 cm to the right and 10 cm up, turned about 25 degrees away. Its corners are at:

```bash
cd ~/FOR26/ros_ws/src/aruco_localization
python -c "
import numpy as np
from aruco_localization.aruco_detector import marker_pose_from_corners
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
corners = np.array([[374.6, 166.2], [436.2, 167.3], [433.0, 231.5], [372.8, 227.2]])
print(np.round(marker_pose_from_corners(corners, 0.15, K, np.zeros(5)), 2))"
```

The answer must be:

```
[[ 0.9   0.01 -0.43  0.2 ]
 [ 0.08 -0.99  0.13 -0.1 ]
 [-0.42 -0.16 -0.89  1.2 ]
 [ 0.    0.    0.    1.  ]]
```

Read it: the last column is the marker's centre, 1.2 m ahead (z), 0.2 m to the right (x) and 0.1 m up (y = -0.1, because the camera's y points down). The first three columns are the marker's axes written in the camera frame: its x axis (to its right) points mostly along the camera's x; its y axis (up) mostly along the camera's -y; its z axis (out of the printed side) points mostly back towards the camera (-0.89 along the camera's z), and a little to the left, because the marker is turned.

Before you fill in the TODO, the function returns the identity matrix: every marker at the camera's origin.

## Run it

Build, then start the simulator without the particle filter:

```bash
cd ~/FOR26/ros_ws
colcon build
source install/setup.bash
ros2 launch aruco_localization localization.launch.py localization:=false
```

The robot starts at the middle of the room, facing markers 10 and 11 on the wall 3 m ahead. In a new terminal:

```bash
ros2 topic echo --once /detected_markers
```

## Check it

**1. The numbers.** Marker 11 hangs at (3.0, 0.9) in the room, 25 cm above the floor. The camera is at (0.125, 0) and 6.5 cm above the floor, looking along x. So, in the camera frame, marker 11 should be at about

| Axis | Expected | Why |
| --- | --- | --- |
| x (right) | -0.90 m | the marker is 0.9 m to the left |
| y (down) | -0.185 m | it is 18.5 cm higher than the lens |
| z (forward) | 2.875 m | 3.0 - 0.125 |

A working detector gives numbers like (-0.92, -0.19, 2.92). The distance comes out about 1.5 % too long at 3 m. That is real, not your mistake: at that distance the marker covers only about 24 x 24 pixels, and the blurred edges of the rendered marker make it look a little smaller, so further away.

**2. RViz.** The `Detected markers` display shows a coloured square for each marker, with `detected <id>` written above it, and the `TF` display shows a frame `detected_<id>` for each. The squares stand upright, 3 m in front of the robot. Each frame's blue axis (z) points back towards the robot.

**3. The picture.** The `Camera` panel shows the markers outlined, with their ids, their axes (red x, green y, blue z) and their distance.

**4. Move the robot.** Drive it with the keyboard, in a new terminal:

```bash
ros2 run aruco_localization keyboard_teleop
```

(w/s forward and back, a/d sideways, q/e turn, space stop.) As the robot turns, markers leave the picture and others appear; their squares in RViz move with them.

## On the real robot

Start it as on the previous page, with your calibration and your measured camera mount. Hold a printed marker (the course uses markers from `DICT_4X4_50` whose black square is 206.5 mm wide) in front of the camera at a distance you measure with a tape measure, and compare with `ros2 topic echo /detected_markers`. Start the launch file with the correct marker size: it reads it from the map file (`marker_size`), so check that the map's value is the size of your printed markers.

## Try it yourself

1. How far away can the detector still find a marker, in the simulator and with the real camera? How many pixels wide is the marker then?
2. Record `/detected_markers` while the robot stands still in front of a marker 1 m away, and again at 3 m. How much does the distance jump around (its standard deviation)? And the marker's orientation?
3. Tilt a real marker slowly from facing the camera to almost edge-on. At some angles the detected orientation flips back and forth between two answers. This is the **ambiguity** of a small, nearly flat square seen from far away: two quite different poses explain the four corners almost equally well. The particle filter must live with it.

---

Previous: [1. The camera in ROS](01_camera_in_ros.md) | Next guide: [The classroom map](../Mapping/README.md)
