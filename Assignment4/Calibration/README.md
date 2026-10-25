# Camera Calibration

To turn a marker in a picture into a marker in the room, you need to know how the camera maps the room onto the picture: its **intrinsics**. Every camera is a little different, so you measure them for yours. You do not write any code here; you run two scripts and check the result.

| # | Page | What you do |
| --- | --- | --- |
| 1 | [The camera model](01_camera_model.md) | Learn what the intrinsics are and why the pictures need them |
| 2 | [Calibrate your camera](02_calibrate.md) | Take pictures of a checkerboard, compute the intrinsics, save them, check them |

## The scripts

| Script | What it does |
| --- | --- |
| [`scripts/capture_calibration_images.py`](scripts/capture_calibration_images.py) | takes pictures of the checkerboard with the robot's camera |
| [`scripts/calibrate_camera.py`](scripts/calibrate_camera.py) | finds the board in the pictures, computes the intrinsics, saves them to a YAML file |

## What you get

A file `camera_intrinsics.yaml` in `~/FOR26/ros_ws/src/aruco_localization/config/`. The camera node reads it and publishes it with every picture, and the marker detector uses it.
