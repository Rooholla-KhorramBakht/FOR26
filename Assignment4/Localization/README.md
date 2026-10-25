# Localization

This is the main part of Assignment 04. You write the heart of a particle filter, run it in the simulator and on the robot, and measure how well it works. The robot's odometry is given: `odometry_node` turns the measured wheel velocities into a dead-reckoned pose, `/odom`, which the filter uses to follow the robot's motion.

| # | Page | What you do |
| --- | --- | --- |
| 1 | [Particles and motion](01_particles_and_motion.md) | See how odometry drifts, learn how the particle filter works, write `motion_update()` |
| 2 | [The measurement model](02_measurement_model.md) | Write `measurement_from_detection()`, `expected_measurement()` and `measurement_log_likelihood()` |
| 3 | [Resampling, and running it all](03_resampling_and_running.md) | Write `low_variance_resample()`, run the localization in the simulator and on the robot |
| 4 | [Experiments](04_experiments.md) | The tools to record, replay and plot runs, and the experiments for your report |

```
 sim_node / robot_node ──/base_velocity──► odometry_node ──/odom──────────────┐
                                                                               ▼
 aruco_detector ──/detected_markers──────────────────────────────────► localization_node ──► /particles
 map_node ──/map_markers──────────────────────────────────────────────────────▲              /pose_estimate
                                                                                              TF map → odom
```

## Your files

All in `~/FOR26/ros_ws/src/aruco_localization/aruco_localization/`.

| File | What you write | Page |
| --- | --- | --- |
| `particle_filter.py` | `motion_update()` | 1 |
| `localization_node.py` | `measurement_from_detection()` | 2 |
| `particle_filter.py` | `expected_measurement()`, `measurement_log_likelihood()` | 2 |
| `particle_filter.py` | `low_variance_resample()` | 3 |

Everything else is given: the odometry node, the nodes around your functions, the `ParticleFilter` class that calls them, the launch file, RViz. `particle_filter.py` has no ROS in it at all, so you can test it on its own, and each page shows you how.

Until you fill in the TODOs everything starts and runs, but does nothing useful: the particles stay where they were spread, and the estimate is a random one of them.

## The topics

| Topic | Message type | Content |
| --- | --- | --- |
| `/base_velocity` | `geometry_msgs/TwistStamped` | the base's velocity measured by the wheels: `linear.x` forward (m/s), `linear.y` left (m/s), `angular.z` counter-clockwise (rad/s) |
| `/odom` | `nav_msgs/Odometry` | the dead-reckoned pose in the `odom` frame |
| `/detected_markers` | `aruco_interfaces/ArucoMarkerArray` | markers in view, in the `camera` frame (from the [detector](../Detection/README.md)) |
| `/map_markers` | `aruco_interfaces/ArucoMarkerArray` | all markers of the map, in the `map` frame |
| `/particles` | `aruco_interfaces/ParticleArray` | every particle's pose and weight, in the `map` frame |
| `/particles_viz` | `visualization_msgs/MarkerArray` | the particles drawn in RViz, coloured by weight, and the estimate as a magenta arrow |
| `/pose_estimate` | `geometry_msgs/PoseWithCovarianceStamped` | the estimated pose of the base in the map, with the particles' spread as its covariance |
| `/marker_errors_viz` | `visualization_msgs/MarkerArray` | for each detected marker, a red line from where the estimate puts it to where the map has it |
| `/true_pose` | `geometry_msgs/PoseStamped` | simulator only: where the robot really is, in the `map` frame |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | RViz's "2D Pose Estimate" button: spread the particles around this pose |

```
# ParticleArray.msg
std_msgs/Header header
geometry_msgs/Pose[] poses
float64[] weights          # add up to 1
```

## The settings

The particle filter's settings are in `config/localization.yaml`. Each is explained at the top of `localization_node.py`. After changing one, build again and restart the launch file, or try it in seconds on a recorded run with `replay_filter.py` (page 4).

| Setting | Default | What it is |
| --- | --- | --- |
| `num_particles` | 2000 | how many guesses |
| `estimate` | `best` | the estimate is the particle with the largest weight (`best`) or the weighted mean of all (`mean`) |
| `motion_noise` | [0.1, 0.1, 0.002, 0.002] | how much noise the predict step adds (page 1) |
| `measurement_sigmas` | [0.1, 0.087, 0.1] | how much the measurement model trusts range, bearing and facing (page 2) |
| `resample_noise` | [0.01, 0.01] | jitter added to the particles after resampling, in m and rad (page 3) |
| `resample_threshold` | 0.5 | resample when fewer than this fraction of the particles really count (page 3) |
| `random_fraction` | 0.01 | fraction of particles replaced by random ones at each resampling (page 3) |
| `recovery_rates` | [0.0, 0.0] | off by default. When on, more random particles are added when the measurements suddenly stop fitting ("augmented MCL", see `ParticleFilter` in `particle_filter.py`) |
| `max_range` | 5.0 | ignore markers further away than this (m) |
| `update_min_motion`, `update_max_interval` | [0.02, 0.03], 1.0 | update only after moving 2 cm or turning 0.03 rad, or after 1 s (page 3) |
