# ROB-GY 6003: LeKiwi Robot Labs

Course material for working with the **LeKiwi** mobile manipulator: a three-wheel omnidirectional base with an SO-101 arm, driven from an onboard Beelink mini PC, with a second SO-101 arm as the leader for teleoperation.

## Assignments

| Assignment | Course page | Material in this repository |
| --- | --- | --- |
| Assignment 03: Inverse Kinematics | [Assignment page](https://sites.google.com/nyu.edu/rob-gy6003/assignments/assignment-03-inverse-kinematics) | [`Assignment3/`](Assignment3/README.md) |
| Assignment 04: Localization with ArUco Markers | Assignment page (coming soon) | [`Assignment4/`](Assignment4/README.md) |

## What is in this repository

| Folder | Contents |
| --- | --- |
| `Assignment3/` | step-by-step guides for Assignment 03 |
| `Assignment4/` | step-by-step guides and scripts for Assignment 04 |
| `lekiwi_sim/` | the robot simulator, a plain Python package |
| `models/LeKiwi-sim/` | the robot's MuJoCo model (a git submodule) |
| `ros_ws/` | the ROS 2 workspace: `lekiwi_ros` (provided nodes), `hello_ros` (your first nodes), `arm_control` (your kinematics and typing nodes), `aruco_interfaces` and `aruco_localization` (Assignment 04: messages, and your detector, odometry and particle filter) |

## Getting the code

```bash
git clone --recurse-submodules https://github.com/Rooholla-KhorramBakht/FOR26.git ~/FOR26
```

The `--recurse-submodules` option also downloads the robot model. If you forgot it, run `git submodule update --init` inside `~/FOR26`.
