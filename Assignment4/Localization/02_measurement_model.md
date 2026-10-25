# 2. The Measurement Model

The update step asks, for each particle: **if the robot were here, how likely is what the camera just saw?** To answer it you need three things: the measurement in a form you can compare, what each particle would expect to measure, and a formula for "how likely".

## What is measured

The detector reports each marker's full 3-D pose in the camera frame. The filter's state is only 2-D (x, y, heading on the floor), so it uses three numbers per marker, in the robot's **base** frame, on the floor plane:

| Number | Meaning |
| --- | --- |
| range | distance from the base's centre to the marker (m) |
| bearing | the direction of the marker, counter-clockwise from straight ahead (rad) |
| facing | the direction the marker's printed side faces (its z axis), in the base frame (rad) |

```
                 marker  ──►  facing (here: towards -x of the robot, about pi)
                   ▲
          range   /
                 /  bearing
     robot ──► ─/───────── (robot's x axis, straight ahead)
```

Range and bearing say where the marker is. Facing says which way it is turned, and so, for a marker whose direction in the map is known, which way the robot is turned. With one marker, range and bearing alone leave the robot anywhere on a circle around the marker; facing pins it down.

## Exercise 1: `measurement_from_detection()`

In `localization_node.py`, fill in `measurement_from_detection(T_base_camera, T_camera_marker)`:

1. `T_base_marker = T_base_camera @ T_camera_marker`: the marker in the base frame. (`T_base_camera` comes from TF: the camera mount from the launch file.)
2. Range and bearing from its position: `x, y = T_base_marker[0, 3], T_base_marker[1, 3]`, then `hypot(x, y)` and `atan2(y, x)`.
3. Facing from its z axis, the third column: `normal = T_base_marker[:3, 2]`, then `atan2(normal[1], normal[0])`.

Test it with the simulated camera mount and a marker 2 m in front of the camera, 0.5 m to the right of the picture's centre and 0.2 m up, facing the camera:

```bash
cd ~/FOR26/ros_ws/src/aruco_localization
python -c "
import numpy as np
from aruco_localization.localization_node import measurement_from_detection
T_base_camera = np.eye(4)
T_base_camera[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]   # optical axes: z forward, x right, y down
T_base_camera[:3, 3] = [0.125, 0.0, 0.03]
T_camera_marker = np.eye(4)
T_camera_marker[:3, :3] = np.diag([1, -1, -1])                # facing the camera
T_camera_marker[:3, 3] = [0.5, -0.2, 2.0]
print([round(v, 4) for v in measurement_from_detection(T_base_camera, T_camera_marker)])"
```

Expected: `[2.183, -0.2311, 3.1416]`. The marker is 2.125 m ahead of the base's centre and 0.5 m to the right: range 2.183 m, bearing -13.2 degrees (to the right is negative), and it faces straight back at the robot (pi).

## Exercise 2: `expected_measurement()`

In `particle_filter.py`, fill in `expected_measurement(particles, marker)`: what each particle would measure for one marker of the map, `marker = [x, y, facing]` in the map frame.

For a particle at (px, py, ptheta):

```
dx, dy   = marker_x - px, marker_y - py
range    = hypot(dx, dy)
bearing  = atan2(dy, dx) - ptheta          (wrapped)
facing   = marker_facing - ptheta          (wrapped)
```

Do it for all particles at once with NumPy (`np.hypot`, `np.arctan2`, and `wrap()` from the same file).

```bash
python -c "
import numpy as np
from aruco_localization.particle_filter import expected_measurement
particles = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]])
print([a.round(4).tolist() for a in expected_measurement(particles, [2.0, 0.0, np.pi])])"
```

Expected: `[[2.0, 2.0], [0.0, -1.5708], [-3.1416, 1.5708]]`: both particles are 2 m from the marker; the first faces it (bearing 0), the second faces +y, so the marker is 90 degrees to its right; the marker faces back at the first (pi, which wraps to -pi) and to the second's left.

## How likely is a measurement?

Compare measured and expected values. A particle near the true pose expects almost what was measured; a particle far away expects something quite different. Model each error as normally distributed with a standard deviation sigma. The likelihood of one error is proportional to `exp(-0.5 * (error / sigma)^2)`, and the measurements are treated as independent, so their likelihoods multiply.

Multiplying many small numbers soon rounds to zero in a computer. So the filter works with **logarithms**: the log of a product is a sum, and the log of `exp(-0.5 * (error / sigma)^2)` is just `-0.5 * (error / sigma)^2`.

The standard deviations, `sigmas = [sigma_range_fraction, sigma_bearing, sigma_facing_per_metre]`:

| Error | Its sigma | Why |
| --- | --- | --- |
| range | `sigma_range_fraction * measured range` | a far marker covers fewer pixels: its distance is less certain |
| bearing | `sigma_bearing` (rad) | the direction of a marker is measured well at any distance |
| facing | `sigma_facing_per_metre * measured range` (rad) | the orientation of a small, far marker is very uncertain (remember the ambiguity on the [detector page](../Detection/02_aruco_detector.md)) |

**One more thing: outliers.** Now and then a detection is badly wrong: a flipped orientation, a misread id. With pure normal errors, one such measurement makes every particle near the truth look impossible (an error of 10 sigma gives exp(-50)), and the filter jumps to wherever the bad measurement happens to fit. So each normalised error is **capped** at `OUTLIER_CAP = 3` sigmas: a bad measurement can lower a particle's weight a lot, but not to zero.

## Exercise 3: `measurement_log_likelihood()`

Fill in `measurement_log_likelihood(particles, measurements, marker_map, sigmas)`. For each measurement `(marker_id, range, bearing, facing)`:

1. Skip it if `marker_id` is not in `marker_map` (a marker that is not part of the map).
2. Get the expected values for all particles with `expected_measurement()`.
3. Compute the three normalised errors (measured minus expected, divided by its sigma; wrap the angle errors first).
4. For each: `log_likelihood -= 0.5 * np.minimum(np.abs(error), OUTLIER_CAP) ** 2`.

```bash
python -c "
import numpy as np
from aruco_localization.particle_filter import measurement_log_likelihood
particles = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.5]])
measurements = [(5, 2.0, 0.0, np.pi)]
marker_map = {5: [2.0, 0.0, np.pi]}
print(measurement_log_likelihood(particles, measurements, marker_map, [0.1, 0.087, 0.1]).round(3).tolist())"
```

Expected: `[0.0, -0.5, -7.625]`.

| Particle | Why |
| --- | --- |
| (0, 0, 0) | exactly where the measurement says: no error at all |
| (0.2, 0, 0) | 20 cm closer: it expects range 1.8 instead of 2.0. The range sigma is 10 % of 2 m = 0.2 m, so the error is 1 sigma: -0.5 |
| (0, 0, 0.5) | turned by 0.5 rad. Bearing error 0.5 / 0.087 = 5.7 sigma, capped at 3: -4.5. Facing error 0.5 / (0.1 * 2) = 2.5 sigma: -3.125. Total -7.625 |

## Try it yourself

1. Leave out the facing term. Start the simulator with the robot seeing only one marker (`start_pose:="2.0 0.9 0"` puts it in front of marker 11). Where do the particles gather, and why in that shape?
2. What happens with `OUTLIER_CAP` set very large, when the detector sometimes reports a flipped marker?
3. The sigmas say how much the filter trusts the camera. What happens when they are much too small? Much too large?

---

Previous: [1. Particles and motion](01_particles_and_motion.md) | Next: [3. Resampling, and running it all](03_resampling_and_running.md)
