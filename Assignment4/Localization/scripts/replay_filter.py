"""Run your particle filter on a recorded bag, as fast as possible, with any settings.

Driving the robot again for every setting you want to try is slow, and no two drives are the same.
Instead, record the inputs once and replay them through the filter as often as you like:

    # while the launch file runs (localization:=false is enough), record the inputs:
    ros2 bag record -o my_run /detected_markers /odom /map_markers /tf_static /true_pose
    # then, any number of times:
    python replay_filter.py my_run --output n500.csv
    python replay_filter.py my_run --num-particles 100 --output n100.csv
    python plot_localization.py n100.csv

The replay uses the same code as localization_node (your particle_filter.py and
measurement_from_detection), in the same order: every /odom message is a predict step, every
/detected_markers message an update step. It writes the same CSV columns as record_localization.py.
/true_pose is optional: only a recording from the simulator has it.

Options (the defaults are those of config/localization.yaml):
    --num-particles N
    --estimate best|mean
    --motion-noise A_T A_R S_XY S_THETA
    --measurement-sigmas RANGE_FRACTION BEARING FACING     (bearing and facing in radians)
    --resample-noise XY THETA
    --resample-threshold F
    --random-fraction F
    --recovery-rates SLOW FAST
    --max-range M
    --update-min-motion DISTANCE ANGLE
    --update-max-interval SECONDS
    --map-margin M
    --initial-pose X Y THETA   start with the particles around this pose instead of everywhere
    --seed S                   random seed, for exactly repeatable runs
    --output FILE              (default replay.csv)
"""
import argparse
import csv
import math
import os

import numpy as np
import rosbag2_py
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from aruco_localization.localization_node import measurement_from_detection
from aruco_localization.particle_filter import ParticleFilter
from aruco_localization.transforms import matrix_from_pose_msg, matrix_from_transform_msg, matrix_to_2d

COLUMNS = ["time", "est_x", "est_y", "est_theta", "std_x", "std_y", "std_theta", "true_x", "true_y", "true_theta",
           "odom_x", "odom_y", "odom_theta", "n_detected", "detected_ids"]


def defaults():
    path = os.path.join(get_package_share_directory("aruco_localization"), "config", "localization.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["localization_node"]["ros__parameters"]


def read_bag(path):
    """Yield (time in s, topic, message) for every message in the bag, in time order."""
    reader = rosbag2_py.SequentialReader()
    storage = "mcap" if any(name.endswith(".mcap") for name in os.listdir(path)) else "sqlite3"
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        yield stamp / 1e9, topic, deserialize_message(data, types[topic])


def camera_in_base(bag):
    """T_base_camera from the static transforms in the bag (base -> camera_link -> camera)."""
    links = {}
    for _, topic, msg in read_bag(bag):
        if topic == "/tf_static":
            for t in msg.transforms:
                links[t.child_frame_id] = (t.header.frame_id, matrix_from_transform_msg(t.transform))
    T, frame = np.eye(4), "camera"
    while frame != "base":
        if frame not in links:
            raise SystemExit("The bag has no static transform chain from base to camera. Record /tf_static too.")
        parent, T_parent_frame = links[frame]
        T, frame = T_parent_frame @ T, parent
    return T


def main():
    d = defaults()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("--num-particles", type=int, default=d["num_particles"])
    parser.add_argument("--estimate", choices=["best", "mean"], default=d["estimate"])
    parser.add_argument("--motion-noise", type=float, nargs=4, default=d["motion_noise"])
    parser.add_argument("--measurement-sigmas", type=float, nargs=3, default=d["measurement_sigmas"])
    parser.add_argument("--resample-noise", type=float, nargs=2, default=d["resample_noise"])
    parser.add_argument("--resample-threshold", type=float, default=d["resample_threshold"])
    parser.add_argument("--random-fraction", type=float, default=d["random_fraction"])
    parser.add_argument("--recovery-rates", type=float, nargs=2, default=d.get("recovery_rates", [0.0, 0.0]))
    parser.add_argument("--max-range", type=float, default=d["max_range"])
    parser.add_argument("--map-margin", type=float, default=d["map_margin"])
    parser.add_argument("--update-min-motion", type=float, nargs=2, default=d["update_min_motion"])
    parser.add_argument("--update-max-interval", type=float, default=d["update_max_interval"])
    parser.add_argument("--initial-pose", type=float, nargs=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="replay.csv")
    args = parser.parse_args()

    T_base_camera = camera_in_base(args.bag)
    pf, marker_map = None, {}
    last_odom, odom_2d, true_2d = None, ["", "", ""], ["", "", ""]
    start, rows = None, 0
    odom_at_update, time_of_update = None, None
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for t, topic, msg in read_bag(args.bag):
            if topic == "/map_markers" and pf is None:
                for marker in msg.markers:
                    T = matrix_from_pose_msg(marker.pose)
                    marker_map[marker.id] = [T[0, 3], T[1, 3], math.atan2(T[1, 2], T[0, 2])]
                xs, ys = [m[0] for m in marker_map.values()], [m[1] for m in marker_map.values()]
                m = args.map_margin
                pf = ParticleFilter(args.num_particles, [min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m],
                                    np.random.default_rng(args.seed), recovery_rates=args.recovery_rates)
                if args.initial_pose:
                    pf.initialize_around(args.initial_pose, 0.3, 0.3)
            elif topic == "/true_pose":
                true_2d = list(matrix_to_2d(matrix_from_pose_msg(msg.pose)))
            elif topic == "/odom":
                T_odom_base = matrix_from_pose_msg(msg.pose.pose)
                odom_2d = list(matrix_to_2d(T_odom_base))
                if pf is not None and last_odom is not None:
                    delta = matrix_to_2d(np.linalg.inv(last_odom) @ T_odom_base)
                    if np.any(np.abs(delta) > 1e-6):
                        pf.predict(delta, args.motion_noise)
                last_odom = T_odom_base
            elif topic == "/detected_markers" and pf is not None and last_odom is not None:
                if odom_at_update is not None:   # the same gating as localization_node.update_due()
                    moved = matrix_to_2d(np.linalg.inv(odom_at_update) @ last_odom)
                    if (math.hypot(moved[0], moved[1]) < args.update_min_motion[0]
                            and abs(moved[2]) < args.update_min_motion[1]
                            and t - time_of_update < args.update_max_interval):
                        continue
                measurements = []
                for marker in msg.markers:
                    measured = measurement_from_detection(T_base_camera, matrix_from_pose_msg(marker.pose))
                    if marker.id in marker_map and measured[0] <= args.max_range:
                        measurements.append((marker.id, *measured))
                if not pf.update(measurements, marker_map, args.measurement_sigmas):
                    continue
                odom_at_update, time_of_update = last_odom, t
                estimate = pf.best() if args.estimate == "best" else pf.mean()
                std = np.sqrt(np.maximum(np.diag(pf.covariance()), 0.0))
                start = t if start is None else start
                writer.writerow([f"{t - start:.3f}", *estimate, *std, *true_2d, *odom_2d, len(msg.markers),
                                 " ".join(str(m.id) for m in msg.markers)])
                rows += 1
                if pf.effective_sample_size() < args.resample_threshold * len(pf.particles):
                    pf.resample(*args.resample_noise, args.random_fraction)
    print(f"Wrote {rows} estimates to {args.output}")


if __name__ == "__main__":
    main()
