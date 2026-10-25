# 2. Calibrate Your Camera

Time needed: about 20 minutes.

## What you need

- The robot's camera, plugged into the robot's computer (it is `/dev/video0` unless you have more cameras: list them with `ls /dev/video*`).
- The course checkerboard: **11 x 8 squares of 25 mm**, which is **10 x 7 inner corners** (the points where four squares meet). The scripts count inner corners. If you use another board, count its inner corners and measure a square with a ruler, then pass `--cols`, `--rows` and `--square` to both scripts.
- The board must be **flat**. A board printed on paper that bends gives a bad calibration. Glue it to cardboard, or show it on a tablet or monitor at full brightness.

Stop the camera node if it is running: only one program can use the camera at a time.

## Step 1: Take the pictures

```bash
conda activate ros_mujoco
cd ~/FOR26/Assignment4/Calibration/scripts
python capture_calibration_images.py ~/calibration_pictures
```

A window shows the camera. When the whole board is found, coloured lines connect its corners. Press **space** to save a picture, **q** to quit. It stops by itself after 40 pictures.

Working over SSH, without a window? Add `--auto`. It then saves a picture whenever the whole board is visible and has moved since the last one. Move the board slowly; stop with Ctrl+C.

**What makes a good set of pictures:**

| Do | Why |
| --- | --- |
| Cover the whole picture: the board in the corners and at the edges too, not only in the middle | Distortion is strongest at the edges. The calibration can only measure it where it has seen corners. |
| Tilt the board: up to about 45 degrees, left, right, up, down | Pictures of a board facing the camera straight on cannot separate the focal length from the distance. |
| Vary the distance: near (the board fills most of the picture) and far | |
| Hold still for each picture | Motion blur moves the corners. |
| Take 20 to 40 pictures | Fewer is not enough; more does not help much. |

## Step 2: Calibrate

```bash
python calibrate_camera.py ~/calibration_pictures --name lekiwi_camera \
    --output ~/FOR26/ros_ws/src/aruco_localization/config/camera_intrinsics.yaml
```

It lists each picture with `board found` or `no board`, then prints the result, for example:

```
Calibrating with 38 pictures of size 640 x 480 ...

RMS reprojection error: 0.312 pixels (good: below 0.5, acceptable: below 1.0)
Worst picture: 0.655 pixels
Focal length fx, fy:   ...
Principal point cx, cy: ... (image centre: 320.0, 240.0)
Field of view:          ... deg horizontal, ... deg vertical
Distortion k1 k2 p1 p2 k3: [...]

Saved .../config/camera_intrinsics.yaml
```

If some pictures fit much worse than the others (blurred ones, usually), the script drops them and calibrates again, and says so.

The script also takes a **video** instead of a folder of pictures: `python calibrate_camera.py board_video.mp4 --every 10` uses every tenth frame.

## Step 3: Check the result

Look at the numbers. They must make sense before you use them.

| Check | Expected |
| --- | --- |
| RMS reprojection error | below 1 pixel, ideally below 0.5 |
| fx and fy | almost equal (within a few percent): the pixels are square |
| cx, cy | near the image centre, within a few percent of the picture size |
| Field of view | close to what the camera's data sheet says. A wide-angle camera: well above 60 degrees horizontally. |

If the error is too high: take new pictures with more tilt, more coverage of the corners, and less blur. Check that `--cols`, `--rows` count inner corners.

## Step 4: Use it

The file is in the `aruco_localization` package's `config` folder, so building the workspace installs it, and the launch file finds it:

```bash
cd ~/FOR26/ros_ws
colcon build
```

Open the file. It has the layout ROS uses for camera calibrations: `camera_matrix` (K, row by row), `distortion_coefficients` (k1, k2, p1, p2, k3), and a few fields ROS needs for other camera types. The last block, `calibration`, records how it was made.

## Try it yourself

1. Calibrate twice, with two different sets of pictures. How much do fx, cx and k1 change? This is the uncertainty of your calibration.
2. Calibrate with only pictures where the board is in the middle of the picture. What happens to the distortion coefficients, and to the error?
3. Undistort a picture: `cv2.undistort(picture, K, distortion)`. Are straight edges in the room straight now?

---

Previous: [1. The camera model](01_camera_model.md) | Next guide: [Marker detection](../Detection/README.md)
