"""Plot a recording made with record_localization.py.

Usage:

    python plot_localization.py run1.csv [--output run1.png] [--title "Square, 500 particles"]

Makes one figure with four panels:
    1. top view: the estimated path, the odometry path (moved into the map frame so it starts where the
       first estimate is), and the true path if the recording has one (simulator)
    2. the spread of the particles (standard deviation of x, y and heading) over time
    3. the error of the estimate over time (only with a true path), and of the odometry
    4. the number of markers seen over time
and prints a summary: mean and largest position error, mean heading error, mean spread.
"""
import argparse
import csv
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows.")

    def column(name):
        return np.array([float(r[name]) if r[name] not in ("", None) else np.nan for r in rows])
    return {name: column(name) for name in rows[0] if name != "detected_ids"}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def odometry_in_map(d):
    """The odometry path, turned and shifted so that its first pose lies on the first true (or estimated) pose."""
    if not np.isnan(d["true_x"][0]):
        x0, y0, t0 = d["true_x"][0], d["true_y"][0], d["true_theta"][0]
    else:
        x0, y0, t0 = d["est_x"][0], d["est_y"][0], d["est_theta"][0]
    dx, dy = d["odom_x"] - d["odom_x"][0], d["odom_y"] - d["odom_y"][0]
    turn = t0 - d["odom_theta"][0]
    c, s = math.cos(turn), math.sin(turn)
    return x0 + c * dx - s * dy, y0 + s * dx + c * dy, wrap(d["odom_theta"] + turn)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--output")
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    d = load(args.csv)
    t = d["time"]
    has_truth = not np.all(np.isnan(d["true_x"]))
    odom_x, odom_y, odom_theta = odometry_in_map(d)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    top = ax[0, 0]
    if has_truth:
        top.plot(d["true_x"], d["true_y"], "g-", label="true path", linewidth=2)
    top.plot(d["est_x"], d["est_y"], "m.-", label="estimate", markersize=3, linewidth=0.8)
    top.plot(odom_x, odom_y, "b--", label="odometry only", linewidth=1)
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title("Paths, top view (map frame)")
    top.grid(True)
    top.legend()

    spread = ax[0, 1]
    spread.plot(t, d["std_x"] * 100, label="std x (cm)")
    spread.plot(t, d["std_y"] * 100, label="std y (cm)")
    spread.plot(t, np.degrees(d["std_theta"]), label="std heading (deg)")
    spread.set_xlabel("time (s)")
    spread.set_title("Spread of the particles")
    spread.set_yscale("log")
    spread.grid(True)
    spread.legend()

    error = ax[1, 0]
    print(f"{args.csv}: {len(t)} estimates over {t[-1]:.1f} s")
    if has_truth:
        position_error = np.hypot(d["est_x"] - d["true_x"], d["est_y"] - d["true_y"])
        heading_error = np.degrees(np.abs(wrap(d["est_theta"] - d["true_theta"])))
        odom_error = np.hypot(odom_x - d["true_x"], odom_y - d["true_y"])
        error.plot(t, position_error * 100, "m-", label="estimate: position error (cm)")
        error.plot(t, heading_error, "c-", label="estimate: heading error (deg)")
        error.plot(t, odom_error * 100, "b--", label="odometry: position error (cm)")
        print(f"  position error: mean {np.nanmean(position_error) * 100:.1f} cm, "
              f"median {np.nanmedian(position_error) * 100:.1f} cm, max {np.nanmax(position_error) * 100:.1f} cm")
        print(f"  heading error:  mean {np.nanmean(heading_error):.1f} deg, max {np.nanmax(heading_error):.1f} deg")
        print(f"  odometry alone: position error at the end {odom_error[-1] * 100:.1f} cm")
        error.set_title("Errors against the true pose")
    else:
        error.text(0.5, 0.5, "No true pose in this recording\n(only the simulator has one)",
                   ha="center", va="center", transform=error.transAxes)
        error.set_title("Errors")
    error.set_xlabel("time (s)")
    error.grid(True)
    error.legend()
    print(f"  particle spread: mean std x {np.nanmean(d['std_x']) * 100:.1f} cm, y {np.nanmean(d['std_y']) * 100:.1f} cm, "
          f"heading {np.degrees(np.nanmean(d['std_theta'])):.1f} deg")

    seen = ax[1, 1]
    seen.step(t, d["n_detected"], where="post")
    seen.set_xlabel("time (s)")
    seen.set_title("Markers seen")
    seen.grid(True)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    output = args.output or args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(output, dpi=110)
    print(f"  saved {output}")


if __name__ == "__main__":
    main()
