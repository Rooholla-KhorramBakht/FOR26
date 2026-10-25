"""Plot a recording made with record_marker_errors.py.

Usage:

    python plot_marker_errors.py errors.csv [--output errors.png]

Makes one figure with two panels: the floor-plan error of each detected marker against its distance from
the robot (one colour per marker), and the error of each marker over time. Prints, per marker: how
often it was seen, and the median and 90th percentile of its error.

What to look for: errors that grow with distance (the detector), one marker that is off at every
distance while the others are not (that marker's map entry, or it was moved), all markers off at the
same moments (the pose estimate).
"""
import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.csv} has no rows.")
    ids = sorted({int(r["id"]) for r in rows})

    fig, (by_range, by_time) = plt.subplots(1, 2, figsize=(14, 5.5))
    print(f"{args.csv}: {len(rows)} sightings of {len(ids)} markers")
    print("marker  sightings  median error (cm)  90 % error (cm)  median facing error (deg)")
    for marker_id in ids:
        mine = [r for r in rows if int(r["id"]) == marker_id]
        distance = np.array([float(r["range"]) for r in mine])
        error = np.array([float(r["error_xy"]) for r in mine]) * 100
        time = np.array([float(r["time"]) for r in mine])
        facing = np.abs([float(r["error_facing"]) for r in mine])
        by_range.plot(distance, error, ".", label=f"marker {marker_id}", markersize=4)
        by_time.plot(time, error, ".", label=f"marker {marker_id}", markersize=4)
        print(f"{marker_id:6d} {len(mine):10d} {np.median(error):18.1f} {np.percentile(error, 90):16.1f} "
              f"{np.median(facing):26.1f}")
    by_range.set_xlabel("distance from the robot (m)")
    by_range.set_ylabel("error on the floor plan (cm)")
    by_range.set_title("Detected marker vs map, by distance")
    by_time.set_xlabel("time (s)")
    by_time.set_ylabel("error on the floor plan (cm)")
    by_time.set_title("Detected marker vs map, over time")
    for ax in (by_range, by_time):
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    output = args.output or args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(output, dpi=110)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
