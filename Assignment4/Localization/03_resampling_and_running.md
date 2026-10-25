# 3. Resampling, and Running It All

## Why resample

After a few updates, most particles have a tiny weight: they are far from where the measurements say the robot is. They cost computing time and contribute nothing. Resampling replaces the particle set with a new one, drawn from the old set **in proportion to the weights**: a particle with a large weight is copied several times, one with a tiny weight usually disappears. Afterwards all weights are equal again, and the particles are concentrated where the robot probably is.

When to resample? Not after every update: each resampling loses some information about the less likely possibilities. The filter measures how uneven the weights are with the **effective number of particles**:

```
N_eff = 1 / sum(weights^2)
```

It is N if all weights are equal, and 1 if a single particle has all the weight. The filter resamples when N_eff drops below `resample_threshold * N` (default: half).

## Low-variance resampling

The simple way to resample is to draw N times at random, each time picking particle i with probability `weights[i]`. That works, but by bad luck it can drop good particles or copy bad ones. **Low-variance** (systematic) resampling uses a single random number instead:

1. Lay the weights end to end along a line from 0 to 1: particle i owns a piece of length `weights[i]`. The running sum `np.cumsum(weights)` gives where each piece ends.
2. Draw one random number `r` in [0, 1/N).
3. Put N equally spaced pointers at `r, r + 1/N, r + 2/N, ...`. Each pointer picks the particle whose piece it lands in.

```
weights:   |--- 0 ---|- 1 -|-------- 2 --------|-3-|
pointers:     ^         ^         ^         ^
picked:       0         1         2         2
```

A particle with weight w is picked either floor(N w) or ceil(N w) times: never much more or less than it deserves.

## Exercise: `low_variance_resample()`

In `particle_filter.py`, fill in `low_variance_resample(weights, rng)`. It returns the **indices** of the chosen particles; the class does the copying. `np.searchsorted(cumulative, positions)` finds, for every pointer, the piece it lands in. Set the last value of the running sum to exactly 1.0 first: after rounding it may be 0.9999999, and a pointer beyond it would find no piece.

```bash
cd ~/FOR26/ros_ws/src/aruco_localization
python -c "
import numpy as np
from aruco_localization.particle_filter import low_variance_resample
rng = np.random.default_rng(0)
print(low_variance_resample(np.array([0.0, 0.0, 1.0, 0.0]), rng).tolist())
weights = np.r_[np.full(500, 0.0016), np.full(500, 0.0004)]   # the first half 4 times as likely
indices = low_variance_resample(weights, rng)
print((indices < 500).sum(), (indices >= 500).sum())"
```

Expected:

```
[2, 2, 2, 2]
800 200
```

All the weight on particle 2: it is picked every time. The first half of the particles holds 80 % of the weight, so it gets exactly 800 of the 1000 new particles.

## What the class adds after resampling (given)

Look at `ParticleFilter.resample()`. After copying, two more things happen:

| Step | Setting | Why |
| --- | --- | --- |
| Jitter: move every particle by a little normal noise | `resample_noise` [xy (m), theta (rad)] | Copies of the same particle are identical. The jitter spreads them out, so that the next update can tell which of them is best. Without it, the particle set would shrink to a few distinct poses. |
| Replace a few particles by random ones anywhere in the room | `random_fraction` | If the filter is ever completely wrong (someone carried the robot elsewhere: the **kidnapped robot**), no particle is near the truth, and none can get there. Random particles give it a chance to find the truth again. |

## One more detail: when to update (given)

A camera sends ten pictures a second. If the robot stands still, the ten detections of the same marker have almost the same error: they are not ten independent measurements. Treating them as independent makes the filter far too sure of itself, and it slowly drifts to wherever that shared error points. So the node only updates after the robot has moved (`update_min_motion`: 2 cm or 0.03 rad) or after `update_max_interval` (1 s) without an update. The well-known AMCL localization in ROS does the same.

## Run it in the simulator

Build, and start everything:

```bash
cd ~/FOR26/ros_ws
colcon build
source install/setup.bash
ros2 launch aruco_localization localization.launch.py
```

and drive in a new terminal:

```bash
ros2 run aruco_localization keyboard_teleop
```

RViz (fixed frame `map`) shows:

| Display | What it shows |
| --- | --- |
| Map markers | the markers where the map has them: dark squares, `marker <id>` in yellow |
| Particles | a short line per particle, pointing where it faces, coloured by weight: blue (small) through green to red (the largest weight) |
| Pose estimate | the estimate (magenta arrow) and its uncertainty: an ellipse for x and y, a cone for the heading |
| True pose | green arrow: where the simulated robot really is |
| Detected markers | coloured squares: where the robot sees the markers, given its estimated pose |
| Marker errors | red lines between each detected marker and the same marker in the map, with the distance in cm |

What you should see: at the start the particles cover the whole room. Within a few seconds of seeing markers they gather into one or a few clusters; as the robot moves and sees more markers, one cluster wins. The magenta arrow lies on the green one, the detected squares on the map's squares, and the red lines are short. When no marker is in view, the cloud spreads out (only odometry moves it); when markers come back into view, it shrinks again.

**Tell the filter where the robot is.** Click "2D Pose Estimate" in RViz's toolbar, then click and drag on the map where the robot is and which way it faces. The particles are spread around that pose.

**Kidnap the robot.** While it runs, give it a deliberately wrong "2D Pose Estimate", a few metres from the truth: now every particle is in the wrong place. Keep driving. Does the filter find its way back? How long does it take?

## Run it on the robot

1. Calibrate the camera ([Calibration](../Calibration/README.md)) and measure the camera mount ([The camera in ROS](../Detection/01_camera_in_ros.md)).
2. Put the robot in the classroom, somewhere it can see at least one marker.
3. Start it. `enable_motors:=true` lets the base drive: keep a hand near the power switch, as in Assignment 03.

```bash
ros2 launch aruco_localization localization.launch.py robot:=real port:=/dev/ttyACM0 enable_motors:=true \
    camera_x:=... camera_z:=... camera_pitch:=...
```

4. Drive with `keyboard_teleop`, slowly at first (`-` lowers the speed).

The real robot has no `/true_pose`. To know how well it localizes, you measure: put the robot on marked spots on the floor whose map coordinates you measured with a tape measure, and compare with `/pose_estimate`. Page 4 has the details.

## Check it

- In the simulator, after driving a minute through the room, the magenta and green arrows agree within about 10 cm and a few degrees most of the time.
- The red error lines are short (a few cm) for every marker.
- With the robot standing still in front of markers, the particle spread (`ros2 topic echo /pose_estimate`, the covariance) stays small and steady, and the estimate does not wander off.

---

Previous: [2. The measurement model](02_measurement_model.md) | Next: [4. Experiments](04_experiments.md)
