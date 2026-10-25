"""Build a map of ArUco markers (where each one is in the room) from a video taken with a calibrated camera.

Usage (with the ros_mujoco environment active):

    python build_map.py INTRINSICS.yaml VIDEO [VIDEO ...] --marker-size 0.15 --ids 1 2 4 7 8 --origin 1 [options]

    INTRINSICS.yaml   the calibration of the camera that took the videos (from calibrate_camera.py)
    VIDEO             one or more videos that walk around the room. Every marker must appear in the SAME
                      frame as at least one other marker at some point, so that the markers are linked to
                      each other. Extra videos are a way to add links that the first one missed.

Options:
    --marker-size M        edge length of the black square of a marker, in metres (required)
    --ids ...              the marker ids that belong to the map. Other ids are ignored (other robots and
                           stray prints in the room carry markers too). Default: every id seen.
    --origin ID            the marker that defines the map frame (default: the first id in --ids)
    --origin-height H      height of the centre of the origin marker above the floor, in metres (default 0).
                           With it, the floor is at z = 0 in the map.
    --dictionary NAME      ArUco dictionary (default DICT_4X4_50)
    --every N              use every N-th frame (default 2)
    --min-side PX          ignore markers smaller than this many pixels per side (default 30): tiny markers
                           are often false detections and give poor poses
    --flatten              make every marker exactly vertical: keep where it faces and how it is turned on its
                           wall, remove any tilt. For a robot that only cares about the floor plan.
    --same-height          the markers are all mounted at the same height: find "up" as the normal of the plane
                           through their centres (needs 3 or more markers), and give them all the average
                           height. Without it, "up" is the origin marker's own y axis, so the origin marker
                           must hang upright (the others may be turned any way on their walls).
    --helper-ids ...       temporary markers that were only put up to link the others while recording. They
                           are used in the solution but left out of the saved map.
    --rotate cw|ccw        how to turn a video whose pictures are the calibration's size turned by 90 degrees
                           (default cw). Phones store a video's orientation as a tag, and OpenCV turns the
                           pictures by it; a video recorded without the tag comes out sideways. Check the
                           printed note and the result if this happens.
    --output FILE          where to save the map (default: marker_map.yaml). A top view is saved next to it.

The map frame:
    origin  on the floor, straight below the centre of the origin marker
    z       up
    x       out of the origin marker's wall, into the room
    y       along the wall, to the right when you stand in the room facing the origin marker

How it works:
    1. Detect the markers in every N-th frame. Keep the frames that see two or more map markers.
    2. Link markers that appear together. A marker that is not linked (through other markers) to the
       origin marker cannot be placed and is reported.
    3. First guess: walk the links outward from the origin marker, chaining the marker-to-marker
       transforms measured in the frames.
    4. Bundle adjustment: adjust every marker pose and every camera pose together so that the marker
       corners, projected through the calibrated camera, land where they were detected. Sightings that
       fit far worse than the rest (blur, a wrong detection) are dropped and the adjustment is repeated.
    5. Turn the result into the map frame above and save it. "Up" is the side of the markers' plane where
       the camera was: whoever filmed stood above the markers.
"""
import argparse
import collections
import pathlib
import sys

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def load_intrinsics(path):
    data = yaml.safe_load(pathlib.Path(path).read_text())
    camera_matrix = np.array(data["camera_matrix"]["data"], float).reshape(3, 3)
    distortion = np.array(data["distortion_coefficients"]["data"], float)
    return camera_matrix, distortion, (data["image_width"], data["image_height"])


def marker_corners(size):
    """The four corners of a marker in its own frame (x right, y up, z out of the marker), in the
    order ArUco reports them: top-left, top-right, bottom-right, bottom-left."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def to_matrix(vec6):
    """[rotation vector (3), translation (3)] -> 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(vec6[:3]).as_matrix()
    T[:3, 3] = vec6[3:]
    return T


def to_vec6(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def average_transform(transforms, max_angle_deg=15.0):
    """Robust average of several 4x4 transforms.

    A small or far-away marker sometimes gives a pose whose rotation is badly wrong (the square-marker
    solver has two nearly equally good answers and picks the wrong one). So: find the rotation that is
    closest to all others (the medoid), keep the transforms whose rotation is within max_angle_deg of
    it, and average those (mean rotation, median translation).
    """
    rotations = Rotation.from_matrix(np.array([T[:3, :3] for T in transforms]))
    if len(transforms) > 2:
        angles = np.array([(rotations[i].inv() * rotations).magnitude() for i in range(len(transforms))])
        medoid = np.argmin(angles.sum(axis=1))
        keep = angles[medoid] < np.radians(max_angle_deg)
    else:
        keep = np.ones(len(transforms), bool)
    T = np.eye(4)
    T[:3, :3] = rotations[keep].mean().as_matrix()
    T[:3, 3] = np.median(np.array([t[:3, 3] for t, k in zip(transforms, keep) if k]), axis=0)
    return T


def side_length(normalised_corners):
    """Size of a marker in the picture (shortest side), in normalised image units."""
    return np.linalg.norm(normalised_corners - np.roll(normalised_corners, 1, axis=0), axis=1).min()


def detect(video_path, args, camera_matrix, distortion, image_size):
    """Return {(video name, frame index): {marker id: 4x2 undistorted, normalised corner coordinates}}."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary)), parameters)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        sys.exit(f"Cannot open {video_path}")
    wanted = set(args.ids) | set(args.helper_ids)
    frames, index, seen = {}, 0, collections.Counter()
    rotation = None
    while True:
        ok, picture = video.read()
        if not ok:
            break
        if index == 0 and (picture.shape[0], picture.shape[1]) == tuple(image_size):
            rotation = cv2.ROTATE_90_CLOCKWISE if args.rotate == "cw" else cv2.ROTATE_90_COUNTERCLOCKWISE
            print(f"{pathlib.Path(video_path).name}: pictures are {picture.shape[1]} x {picture.shape[0]}, the calibration "
                  f"is for {image_size[0]} x {image_size[1]}. Turning every picture {args.rotate} to match.")
        if rotation is not None:
            picture = cv2.rotate(picture, rotation)
        if index % args.every == 0:
            if (picture.shape[1], picture.shape[0]) != tuple(image_size):
                sys.exit(f"The video is {picture.shape[1]} x {picture.shape[0]} but the calibration is for "
                         f"{image_size[0]} x {image_size[1]}. Calibrate with the same camera and resolution.")
            corners, ids, _ = detector.detectMarkers(picture)
            found = {}
            for marker_id, c in zip([] if ids is None else ids.ravel(), corners):
                marker_id = int(marker_id)
                c = c.reshape(4, 2)
                seen[marker_id] += 1
                if args.ids and marker_id not in wanted:
                    continue
                if np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).min() < args.min_side:
                    continue
                found[marker_id] = cv2.undistortPoints(c.reshape(-1, 1, 2), camera_matrix, distortion).reshape(4, 2)
            if found:
                frames[(pathlib.Path(video_path).name, index)] = found
        index += 1
    print(f"{pathlib.Path(video_path).name}: read {index} frames. Markers detected (all ids, before filtering): "
          f"{dict(sorted(seen.items()))}")
    return frames


def marker_in_camera(normalised_corners, size):
    """Pose (4x4) of a marker in the camera frame from its corners, with OpenCV's square-marker solver."""
    ok, rvec, tvec = cv2.solvePnP(marker_corners(size), normalised_corners, np.eye(3), None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return to_matrix(np.concatenate([rvec.ravel(), tvec.ravel()]))


def connected_to(origin, links):
    """All markers that can be reached from `origin` through the links."""
    reached, todo = {origin}, [origin]
    while todo:
        current = todo.pop()
        for a, b in links:
            for here, there in ((a, b), (b, a)):
                if here == current and there not in reached:
                    reached.add(there)
                    todo.append(there)
    return reached


def initial_guess(frames, origin, size):
    """Marker poses in the origin marker's frame, by chaining measured marker-to-marker transforms."""
    relative = collections.defaultdict(list)   # (a, b) -> list of T_a_b (marker b in marker a's frame)
    for found in frames.values():
        poses = {m: marker_in_camera(c, size) for m, c in found.items()}
        for a in poses:
            for b in poses:
                if a != b:
                    relative[(a, b)].append(np.linalg.inv(poses[a]) @ poses[b])
    placed, todo = {origin: np.eye(4)}, [origin]
    while todo:
        a = todo.pop(0)
        for (first, b), transforms in relative.items():
            if first == a and b not in placed:
                placed[b] = placed[a] @ average_transform(transforms)
                todo.append(b)
    return placed


def bundle_adjust(frames, markers, origin, size):
    """Refine all marker poses (except the origin's) and all camera poses together.

    The cost is the reprojection error: the distance between where each marker corner was detected
    and where the current guess of the poses puts it, in normalised image coordinates.
    """
    marker_ids = [m for m in markers if m != origin]
    frame_ids = list(frames)
    corners_local = marker_corners(size)

    # First guess of each camera pose (in the origin marker's frame): from the biggest marker in the
    # frame, whose pose is the most reliable.
    camera_guess = []
    for f in frame_ids:
        biggest = max(frames[f], key=lambda m: side_length(frames[f][m]))
        camera_guess.append(markers[biggest] @ np.linalg.inv(marker_in_camera(frames[f][biggest], size)))
    x0 = np.concatenate([to_vec6(markers[m]) for m in marker_ids] + [to_vec6(T) for T in camera_guess])

    observations = [(fi, m, frames[f][m]) for fi, f in enumerate(frame_ids) for m in frames[f]]
    n_markers = len(marker_ids)
    marker_index = {m: i for i, m in enumerate(marker_ids)}

    def residuals(x):
        out = np.empty((len(observations), 8))
        for k, (fi, m, measured) in enumerate(observations):
            world_from_marker = np.eye(4) if m == origin else to_matrix(x[6 * marker_index[m]:6 * marker_index[m] + 6])
            world_from_camera = to_matrix(x[6 * (n_markers + fi):6 * (n_markers + fi) + 6])
            camera_from_marker = np.linalg.inv(world_from_camera) @ world_from_marker
            points = corners_local @ camera_from_marker[:3, :3].T + camera_from_marker[:3, 3]
            out[k] = (points[:, :2] / points[:, 2:3] - measured).ravel()
        return out.ravel()

    # Each residual depends on one marker pose and one camera pose. Telling the solver so makes it fast.
    sparsity = lil_matrix((8 * len(observations), len(x0)), dtype=int)
    for k, (fi, m, _) in enumerate(observations):
        rows = slice(8 * k, 8 * k + 8)
        if m != origin:
            sparsity[rows, 6 * marker_index[m]:6 * marker_index[m] + 6] = 1
        sparsity[rows, 6 * (n_markers + fi):6 * (n_markers + fi) + 6] = 1

    result = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=2e-3, x_scale="jac",
                           max_nfev=200, verbose=0)
    refined = {origin: np.eye(4)}
    refined.update({m: to_matrix(result.x[6 * i:6 * i + 6]) for i, m in enumerate(marker_ids)})
    cameras = np.array([result.x[6 * (n_markers + fi) + 3:6 * (n_markers + fi) + 6] for fi in range(len(frame_ids))])
    # The error of each sighting: the largest corner error, in normalised image units.
    errors = np.abs(result.fun.reshape(-1, 8)).max(axis=1)
    return refined, cameras, {(frame_ids[fi], m): e for (fi, m, _), e in zip(observations, errors)}


def flatten(markers_map, same_height):
    """Make every marker exactly vertical, facing where it faced before and turned on its wall as before.
    Optionally give all of them the same (average) height."""
    height = np.mean([T[2, 3] for T in markers_map.values()])
    flat = {}
    for m, T in markers_map.items():
        normal = T[:3, 2].copy()
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
        # Keep the marker's x axis as close as possible to where it was, within the now vertical wall.
        x = T[:3, 0] - (T[:3, 0] @ normal) * normal
        x /= np.linalg.norm(x)
        F = np.eye(4)
        F[:3, :3] = np.column_stack([x, np.cross(normal, x), normal])   # marker x, y, z in the map
        F[:3, 3] = T[:3, 3]
        if same_height:
            F[2, 3] = height
        flat[m] = F
    return flat


def map_frame(markers, cameras, origin, origin_height, same_height):
    """The transform that takes points from the origin marker's frame into the map frame.

    cameras: positions of the camera in the frames used (origin marker's frame), to tell up from down.
    """
    centres = np.array([T[:3, 3] for T in markers.values()])
    if same_height and len(centres) >= 3:
        # All markers at the same height: their centres lie in one horizontal plane, and its normal is
        # up. Over the width of a room this is far more precise than the tilt of each single marker,
        # and it does not need the markers to hang upright.
        up = np.linalg.svd(centres - centres.mean(axis=0))[2][2]
    else:
        up = markers[origin][:3, 1]   # the origin marker hangs upright: its y axis points up
    # The camera was held above the markers (whoever filmed stood in the room): that side is up.
    if np.mean((cameras - centres.mean(axis=0)) @ up) < 0:
        up = -up
    z = up / np.linalg.norm(up)
    out_of_wall = markers[origin][:3, 2]
    x = out_of_wall - (out_of_wall @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    map_from_origin = np.eye(4)
    map_from_origin[:3, :3] = np.vstack([x, y, z])   # rows: the map axes written in the origin marker's frame
    map_from_origin[:3, 3] = -map_from_origin[:3, :3] @ markers[origin][:3, 3] + [0.0, 0.0, origin_height]
    return map_from_origin


def save_top_view(path, markers_map, size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(0, 0, "k+", markersize=15)
    for m, T in markers_map.items():
        x, y = T[0, 3], T[1, 3]
        normal = T[:3, 2]
        along = T[:3, 0] * size / 2
        ax.plot([x - along[0], x + along[0]], [y - along[1], y + along[1]], "k-", linewidth=4)
        ax.arrow(x, y, 0.3 * normal[0], 0.3 * normal[1], head_width=0.06, color="tab:blue")
        ax.annotate(f"{m}  (z = {T[2, 3]:.2f} m)", (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Marker map, top view (arrows: the direction each marker faces)")
    ax.set_aspect("equal")
    ax.grid(True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("intrinsics")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--marker-size", type=float, required=True)
    parser.add_argument("--ids", type=int, nargs="*", default=[])
    parser.add_argument("--helper-ids", type=int, nargs="*", default=[])
    parser.add_argument("--origin", type=int)
    parser.add_argument("--origin-height", type=float, default=0.0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--min-side", type=float, default=30.0)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--same-height", action="store_true")
    parser.add_argument("--rotate", choices=["cw", "ccw"], default="cw")
    parser.add_argument("--output", default="marker_map.yaml")
    args = parser.parse_args()

    camera_matrix, distortion, image_size = load_intrinsics(args.intrinsics)
    fx = camera_matrix[0, 0]
    frames = {}
    for video in args.videos:
        frames.update(detect(video, args, camera_matrix, distortion, image_size))
    if not frames:
        sys.exit("No map markers found in the videos. Check --ids, --dictionary and --min-side.")
    origin = args.origin if args.origin is not None else (args.ids[0] if args.ids else min(
        m for found in frames.values() for m in found))

    shared = {f: found for f, found in frames.items() if len(found) >= 2}
    links = collections.Counter(tuple(sorted((a, b))) for found in shared.values() for a in found for b in found if a < b)
    seen = sorted({m for found in frames.values() for m in found})
    print(f"Frames that see two or more markers: {len(shared)}")
    print("Markers seen together (pair: number of frames): "
          + (", ".join(f"{a}-{b}: {n}" for (a, b), n in sorted(links.items())) or "none"))
    if origin not in seen:
        sys.exit(f"The origin marker {origin} was not seen in the videos.")
    reachable = connected_to(origin, links)
    missing = [m for m in (args.ids or seen) if m not in reachable]
    if missing:
        print(f"\nWARNING: markers {missing} cannot be placed. They never appear in the same frame as a marker that "
              f"is linked to the origin marker {origin}. Record another video in which each of them is in view "
              f"together with an already linked marker for a second or two, and pass both videos.\n")
    shared = {f: {m: c for m, c in found.items() if m in reachable} for f, found in shared.items()}
    shared = {f: found for f, found in shared.items() if len(found) >= 2}

    markers = initial_guess(shared, origin, args.marker_size)
    markers, cameras, errors = bundle_adjust(shared, markers, origin, args.marker_size)
    print(f"Bundle adjustment: {len(errors)} marker sightings in {len(shared)} frames, corner error median "
          f"{np.median(list(errors.values())) * fx:.2f} px, worst {max(errors.values()) * fx:.1f} px")
    # Drop the sightings that fit far worse than the typical one and adjust again.
    limit = 5.0 * np.median(list(errors.values()))
    bad = {key for key, e in errors.items() if e > limit}
    if bad:
        for frame, m in bad:
            del shared[frame][m]
        shared = {f: found for f, found in shared.items() if len(found) >= 2}
        markers, cameras, errors = bundle_adjust(shared, initial_guess(shared, origin, args.marker_size), origin,
                                                 args.marker_size)
        print(f"Dropped {len(bad)} sightings that fit badly. Again: {len(errors)} sightings in {len(shared)} frames, "
              f"corner error median {np.median(list(errors.values())) * fx:.2f} px, "
              f"worst {max(errors.values()) * fx:.1f} px")

    map_from_origin = map_frame(markers, cameras, origin, args.origin_height, args.same_height)
    markers_map = {m: map_from_origin @ T for m, T in sorted(markers.items()) if m not in args.helper_ids}

    print("\nMarker   x (m)    y (m)    z (m)   faces (deg)   tilt from vertical (deg)")
    for m, T in markers_map.items():
        normal = T[:3, 2]
        print(f"{m:6d} {T[0, 3]:8.3f} {T[1, 3]:8.3f} {T[2, 3]:8.3f} {np.degrees(np.arctan2(normal[1], normal[0])):11.1f} "
              f"{np.degrees(np.arcsin(np.clip(abs(normal[2]), 0, 1))):16.1f}")
    if args.flatten or args.same_height:
        markers_map = flatten(markers_map, args.same_height)
        print("Flattened: every marker made vertical" + (", all at the average height." if args.same_height else "."))

    entries = []
    for m, T in markers_map.items():
        qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
        entries.append({"id": int(m), "position": [round(float(v), 4) for v in T[:3, 3]],
                        "orientation": [round(float(v), 5) for v in (qx, qy, qz, qw)],
                        "yaw_deg": round(float(np.degrees(np.arctan2(T[1, 2], T[0, 2]))), 2)})

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "dictionary": args.dictionary,
        "marker_size": args.marker_size,
        "frame": "map",
        "notes": ("position: marker centre in metres. orientation: quaternion [x, y, z, w] of the marker frame "
                  "(x right, y up, z out of the marker). yaw_deg: the direction the marker faces, from the map x axis."),
        "markers": entries,
        "built_from": {"videos": [pathlib.Path(v).name for v in args.videos], "origin_marker": origin,
                       "origin_height": args.origin_height, "flatten": args.flatten, "same_height": args.same_height,
                       "median_corner_error_px": round(float(np.median(list(errors.values())) * fx), 3),
                       "unplaced_markers": missing},
    }
    output.write_text(yaml.safe_dump(result, sort_keys=False, default_flow_style=None))
    top_view = output.with_suffix(".png")
    save_top_view(top_view, markers_map, args.marker_size)
    print(f"\nSaved {output} and {top_view}")


if __name__ == "__main__":
    main()
