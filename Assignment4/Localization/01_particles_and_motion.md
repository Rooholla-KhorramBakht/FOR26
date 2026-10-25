# 1. Particles and Motion

## Odometry, the starting point (given)

The robot measures how fast each of its three wheels turns, and `robot_node` (or `sim_node`) turns that into the velocity of the base, `/base_velocity`. The given `odometry_node` adds those velocities up over time into a pose, `/odom`: where the robot is relative to where it started (the `odom` frame). This is **dead reckoning**. It is smooth and fast, but every small error in the measured velocity is added up and never corrected, so the pose slowly **drifts** away from the truth.

See it in the simulator: start it with `ros2 launch aruco_localization localization.launch.py localization:=false`, drive with `ros2 run aruco_localization keyboard_teleop` in a second terminal, and compare `ros2 topic echo --once /odom --field pose.pose.position` with the simulator's truth, `ros2 topic echo --once /true_pose --field pose.position`. The simulated odometry is off by about 5 % (`odometry_noise`), like real wheels that slip a little.

The particle filter uses odometry for what it is good at, following the motion, and corrects its drift with the markers.

## The idea

The robot does not know where it is. Instead of one guess, keep **many**: hundreds of possible poses, each called a **particle**. Each particle is a row `[x, y, theta]`, a pose of the robot's base in the map. Each has a **weight**: how well that guess explains what the robot has measured so far.

At the start, the robot could be anywhere: the particles are spread evenly over the room, with random headings, all with the same weight. Then the filter repeats three steps:

| Step | When | What happens to the particles |
| --- | --- | --- |
| **Predict** | the robot moved (a new `/odom` message) | every particle moves by the same step as the robot, plus a little random noise, because the odometry is not exact |
| **Update** | the camera saw markers (a new `/detected_markers` message) | each particle's weight is multiplied by how likely the measurement is, if the robot were at that particle |
| **Resample** | the weights have become very uneven | draw a new set of particles from the old ones in proportion to their weights: likely guesses are copied, unlikely ones disappear |

After a few rounds, the particles gather around the robot's true pose. Where they gather is the estimate; how widely they are spread is the uncertainty.

Why not just one guess, corrected by each measurement? Because one marker seen once does not pin down the pose: seen from 2 m away, a marker could be 2 m in front of any point on a circle around it. A particle filter keeps all the possibilities that fit, until later measurements, from other places and other markers, rule them out. It can even follow two or more possibilities at once, which a single guess cannot.

## The code

`particle_filter.py` holds everything, without ROS. The `ParticleFilter` class (given) keeps the particles and the weights and runs the steps by calling your functions:

| Class method (given) | Calls your function |
| --- | --- |
| `predict(delta, noise)` | `motion_update(particles, delta, noise, rng)` (this page) |
| `update(measurements, marker_map, sigmas)` | `measurement_log_likelihood(...)`, which uses `expected_measurement(...)` (page 2) |
| `resample(noise_xy, noise_theta, random_fraction)` | `low_variance_resample(weights, rng)` (page 3) |
| `best()`, `mean()`, `covariance()` | (given) the estimate and the spread |

`localization_node.py` (given, except for one function on page 2) feeds the filter from ROS: each `/odom` message becomes a `predict`, each `/detected_markers` message an `update` (and a `resample` when needed).

## The motion step

Between two `/odom` messages, the robot moved by some small step. The node computes it from the two odometry poses, **in the robot's own frame at the start of the step**: `delta = [dx, dy, dtheta]` (dx forward, dy left, dtheta turn).

Every particle must make the same step, in **its own** frame: a particle that faces +y, told to move 10 cm forward, moves 10 cm along +y. That is the same computation as in your `integrate()`, done for all particles at once.

But odometry is not exact, so each particle gets its own slightly different step: the step plus random noise. The noise is larger for larger steps:

```
sigma_xy    = alpha_translation * (distance moved) + sigma_xy_min
sigma_theta = alpha_rotation * |dtheta|            + sigma_theta_min
```

with `noise = [alpha_translation, alpha_rotation, sigma_xy_min, sigma_theta_min]`. The defaults `[0.1, 0.1, 0.002, 0.002]` say: odometry is off by about 10 % of every step, plus 2 mm and 0.002 rad. The noise is what lets the particles spread out to cover the places the robot might really be, when odometry has drifted.

## Exercise: `motion_update()`

Fill in the TODO in `motion_update(particles, delta, noise, rng)`:

1. Compute `sigma_xy` and `sigma_theta` as above.
2. Draw a noisy step for each particle: `step_x = dx + rng.normal(0, sigma_xy, n)`, the same for `step_y` and `step_theta`.
3. Move each particle by its step in its own frame:
   ```
   x     += cos(theta) * step_x - sin(theta) * step_y
   y     += sin(theta) * step_x + cos(theta) * step_y
   theta += step_theta        (then wrap)
   ```

Use NumPy arrays for all particles at once: `np.cos(particles[:, 2])` is the cosine of every particle's heading. A Python loop over 500 particles, 50 times a second, is too slow.

Test it:

```bash
cd ~/FOR26/ros_ws/src/aruco_localization
python -c "
import numpy as np
from aruco_localization.particle_filter import motion_update
rng = np.random.default_rng(0)
particles = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, np.pi / 2]])
print(motion_update(particles, [0.1, 0.0, 0.0], [0, 0, 0, 0], rng).round(4).tolist())
many = motion_update(np.zeros((10000, 3)), [1.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], rng)
print('mean', many.mean(axis=0).round(2), 'std', many.std(axis=0).round(2))"
```

Expected:

```
[[0.1, 0.0, 0.0], [1.0, 1.1, 1.5708]]
mean [1. 0. 0.] std [0.1 0.1 0. ]
```

Without noise, the first particle (facing +x) moves 10 cm along x, the second (facing +y) 10 cm along y. With noise, 10 000 particles that all drive 1 m end up 1 m ahead on average, spread by 10 cm (10 % of 1 m) in both directions, and not turned at all (the step had no turn).

## Watch it

The prediction alone makes a cloud that follows the odometry and spreads out as the robot drives. You see it fully on page 3, when the rest of the filter is in place.

## Try it yourself

1. Change `alpha_translation` to 0. What does the cloud look like after a long drive? Why is that a problem as soon as the odometry has drifted?
2. The noise here is the same in the forward and sideways directions. For a robot with ordinary wheels, sideways slip is much smaller than forward slip. For LeKiwi's omni wheels, is that still true?

---

Next: [2. The measurement model](02_measurement_model.md)
