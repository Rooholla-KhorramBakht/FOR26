# 4. Experiments

A localization that "looks right" in RViz is not yet a result. This page gives you the tools to measure it, and the experiments the report asks for. The course page says which experiments are required and what to hand in; this page says how to do them.

## The tools

All in `~/FOR26/Assignment4/Localization/scripts/`. Run them in a sourced terminal (`source ~/FOR26/ros_ws/install/setup.bash`) while the launch file runs.

| Script | What it does |
| --- | --- |
| `drive_pattern.py` | drives a fixed pattern (`square`, `circle`, `spin`, or `still`), so that a run can be repeated exactly |
| `record_localization.py` | writes the estimate, the particles' spread, the odometry, the true pose (simulator) and the number of markers seen to a CSV file |
| `record_marker_errors.py` | writes, for every detected marker, where the robot sees it in the map and where the map has it, to a CSV file |
| `plot_localization.py` | plots a `record_localization.py` CSV: paths, spread, errors, markers seen; prints a summary |
| `plot_marker_errors.py` | plots a `record_marker_errors.py` CSV: error per marker against distance and over time |
| `replay_filter.py` | runs your particle filter on a recorded bag, as fast as possible, with any settings. Writes the same CSV as `record_localization.py`. |

A typical run, in three terminals:

```bash
# 1
ros2 launch aruco_localization localization.launch.py
# 2
cd ~/FOR26/Assignment4/Localization/scripts
python record_localization.py square.csv --duration 60
# 3
python drive_pattern.py square --size 1.0
```

then `python plot_localization.py square.csv`.

**Replaying.** To compare settings fairly, they must see exactly the same data. Record the inputs of the filter once:

```bash
ros2 bag record -o my_run /detected_markers /odom /map_markers /tf_static /true_pose
```

(drive, then Ctrl+C), and replay them through your filter as often as you like, in a few seconds each:

```bash
python replay_filter.py my_run --num-particles 100 --seed 1 --output n100.csv
python replay_filter.py my_run --num-particles 1000 --seed 1 --output n1000.csv
```

`--seed` makes the random numbers the same each time, so two runs differ only by the setting you changed. Run each setting with several seeds: a particle filter is random, and one run proves little.

## The experiments

### Section 1: Calibration

**C1. Your calibration.** Report fx, fy, cx, cy, the distortion coefficients, the RMS reprojection error and the field of view. Show one calibration picture with the detected corners.

### Section 2: Marker detection

**D1. Detection in RViz.** A short video (screen recording) of RViz with the detected markers (squares, ids, `detected_<id>` frames) and the annotated camera picture, in the simulator and on the real robot, while markers enter and leave the view.

**D2. Distance accuracy.** Put a marker in front of the real robot at 0.5, 1, 1.5, 2 and 3 m (measure from the lens with a tape measure). Record the detected distance at each. Plot detected against true distance. Is there a constant offset, an error that grows with the distance, or both? What could cause each? At what distance does detection stop, and how many pixels wide is the marker there?

### Section 3: The map

**M1. Check the map.** Measure the distance between two pairs of markers in the classroom with a tape measure and compare with the map. How large are the differences, and what does that mean for the localization?

### Section 4: Localization

**L1. Accuracy in simulation.** In the simulator, record a run of a square and a circle (`record_localization.py`, `drive_pattern.py`). Plot it with `plot_localization.py`. Report the mean and largest position error and heading error, against the true pose, and compare with odometry alone.

**L2. Stationary variance.** Let the robot stand still for 60 s at three places: one where it sees one marker, one where it sees two, one where it sees three or more (in the simulator and on the real robot). Record with `record_localization.py`. Report the standard deviation of the estimate (x, y, heading) over the 60 s, and the spread of the particles (`std_x`, `std_y`, `std_theta` columns). Do the two agree? How does the number of markers in view change them? In the simulator, does the estimate stay on the true pose, or drift away from it? Why might it drift although nothing moves?

**L3. Marker consistency.** Record `record_marker_errors.py` while driving through the room. Plot with `plot_marker_errors.py`. How does the error grow with distance? Is any marker consistently off? Use the answers to judge the map (M1), the detector (D2) and the camera mount (Detection page 1).

**L4. Number of particles.** Record one bag in the simulator, starting with the robot somewhere it sees only one marker. Replay it with 50, 100, 500 and 2000 particles, five seeds each. Plot the mean error and the time until the error stays below 20 cm, against the number of particles. Also measure how long one update takes (the replay's run time is a good proxy). What number would you choose, and why?

**L5. The real robot.** Mark four spots on the classroom floor with tape and measure their map coordinates with a tape measure from marker 2 (the map's x axis is perpendicular to marker 2's wall, see the [map page](../Mapping/README.md)). Drive the robot to each spot, let it stand for 10 s, and compare `/pose_estimate` with the measured coordinates. Make a table of the errors. Record a video of the robot driving through the classroom, together with a screen recording of RViz showing the particles converging.

## Hints for good experiments

- Change one thing at a time.
- Repeat. Report a mean and a spread, not a single number.
- Keep your CSV files and bags: you can always plot them again differently.
- When a result surprises you, look at RViz and the plots together before you believe it.

---

Previous: [3. Resampling, and running it all](03_resampling_and_running.md)
