# Assignment 04: Localization with ArUco Markers

**Course page:** see the Assignment 04 page on the course site.

The course page is the official description of the assignment: what to hand in, the report format, and the deadlines. This folder is the practical companion. It takes you from an uncalibrated camera to a robot that knows where it is in the classroom.

## What the assignment asks for

The classroom has printed **ArUco markers** on its walls. Their positions are known: they form a **map**. The robot sees them with its camera. From what it sees, and from how its wheels turned, it works out **where it is in the room**, with a **particle filter**.

**Goals**

1. Calibrate the robot's camera.
2. Detect ArUco markers and estimate their poses relative to the camera.
3. Localize the robot in the map of the classroom with a particle filter that combines the markers with the robot's wheel odometry (provided), and visualize everything in RViz.
4. Measure how well the localization works, in simulation and on the real robot.

**What you hand in:** see the course page.

## The guides

Follow them in this order. Each page ends with a check, so you know it worked before you move on.

| Guide | What you get | Assignment section |
| --- | --- | --- |
| [Camera calibration](Calibration/README.md) | The intrinsics of your robot's camera, saved to a file | 1 |
| [Marker detection](Detection/README.md) | The camera in ROS, your ArUco detector, the detected markers in RViz | 2 |
| [The classroom map](Mapping/README.md) | How the map of the markers was made, and how to read it | 3 |
| [Localization](Localization/README.md) | Your particle filter, running in simulation and on the robot, and the experiments for the report | 4 |

## Before you start

You need the setup of Assignment 03: ROS 2 Jazzy, the `ros_mujoco` conda environment, the workspace in `~/FOR26/ros_ws`. This assignment needs two more Python libraries:

```bash
conda activate ros_mujoco
pip install scipy matplotlib
```

Then build the workspace:

```bash
cd ~/FOR26/ros_ws
colcon build
source install/setup.bash
```

It builds two new packages:

| Package | Contents |
| --- | --- |
| `aruco_interfaces` | the message types of this assignment: `ArucoMarker`, `ArucoMarkerArray`, `ParticleArray` |
| `aruco_localization` | the nodes, the launch file, the maps and the RViz settings |

## The whole system

```
  sim_node or robot_node ──/base_velocity──► odometry_node ──/odom, TF odom→base──┐
           │                                                                       │
           └─(sim) /camera/image ─┐                                                ▼
  camera_node (real) ──/camera/image, /camera/camera_info──► aruco_detector ──/detected_markers──► localization_node
                                                                                   ▲                 │
                                        map_node ──/map_markers────────────────────┘                 │
                                                                                                      ▼
                                                     /particles, /pose_estimate, TF map→odom, RViz drawings
```

One launch file starts all of it, in simulation or on the robot:

```bash
ros2 launch aruco_localization localization.launch.py                                  # simulator
ros2 launch aruco_localization localization.launch.py robot:=real port:=/dev/ttyACM0   # real robot
```

## The frames

Everything in this assignment is about frames: where is this marker, seen from where? These are the frames, all published on TF so that RViz can draw everything together.

| Frame | Where it is | Published by |
| --- | --- | --- |
| `map` | a fixed point in the room. All marker positions are given in it. | (the root) |
| `odom` | where the robot was when odometry started | `localization_node`: map → odom |
| `base` | the robot's centre, x forward, y left, z up | `odometry_node`: odom → base |
| `camera_link` | the camera, x forward along the lens, y left, z up | launch file: base → camera_link (how the camera is mounted) |
| `camera` | the same point, in the optical convention: x right in the picture, y down, z forward out of the lens | launch file: camera_link → camera |
| `detected_<id>` | a marker as the camera sees it right now | `aruco_detector`: camera → detected_id |
| `marker_<id>` | a marker where the map says it is | `map_node`: map → marker_id |

Why does the localization publish map → odom and not map → base? A frame can have only one parent in TF, and `base` already has one: `odom`. So the localization publishes the correction that sits above it. The result is the same: following map → odom → base gives the estimated pose of the robot in the map.

If the localization works, `detected_<id>` and `marker_<id>` lie on top of each other in RViz.

## The idea that carries through

The robot never measures its position directly. It has two imperfect sources:

| Source | Good at | Bad at |
| --- | --- | --- |
| Wheel odometry | smooth, fast, never jumps | drifts: small errors add up without limit |
| Markers seen by the camera | no drift: tied to the map | noisy, only when a marker is in view, sometimes wrong |

The particle filter combines them: odometry moves the guesses, the markers correct them. Every experiment of this assignment looks at this balance from one side or the other.
