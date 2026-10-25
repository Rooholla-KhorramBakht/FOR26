"""Compute a camera's intrinsics from pictures of a checkerboard and save them to a YAML file.

Usage (from anywhere, with the ros_mujoco environment active):

    python calibrate_camera.py IMAGES_OR_VIDEO [options]

    IMAGES_OR_VIDEO   a folder of .png/.jpg pictures (from capture_calibration_images.py),
                      or a video file (one picture is taken every --every frames)

Options:
    --cols 10 --rows 7     inner corners of the board: where four squares meet. A board of
                           11 x 8 squares has 10 x 7 inner corners.
    --square 0.025         edge length of one square, in metres
    --every 10             for a video: use every 10th frame
    --output FILE          where to save the result (default: camera_intrinsics.yaml)
    --name NAME            a name stored in the file, for example "nuc_usb_camera"
    --show                 show each detected board while running (needs a monitor)

The output file has the layout ROS uses for camera_info files (camera_matrix, distortion_coefficients,
...), so the camera node can load it directly.
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
# Stop refining a corner after 30 rounds, or when it moves less than 0.001 pixels.
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MAX_PICTURES = 60   # calibrating with more pictures is slow and rarely better


def read_pictures(source, every):
    """Yield (name, BGR picture) from a folder of images or from a video file."""
    source = pathlib.Path(source)
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            yield path.name, cv2.imread(str(path))
        return
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        sys.exit(f"Cannot open {source}: not a folder of pictures and not a readable video.")
    index = 0
    while True:
        ok, picture = video.read()
        if not ok:
            return
        if index % every == 0:
            yield f"frame {index}", picture
        index += 1


def find_corners(gray, board_size):
    """The board's inner corners in a grayscale picture, or None if the whole board is not visible."""
    found, corners = cv2.findChessboardCorners(
        gray, board_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIXEL_CRITERIA)


def calibrate(board_points, image_corners, image_size):
    """Run OpenCV's calibration. Returns (rms error, camera matrix, distortion, error of each picture)."""
    object_points = [board_points] * len(image_corners)
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_corners, image_size, None, None)
    # Reprojection error of each picture: how far (in pixels) the corners predicted by the
    # calibrated model land from the corners that were detected.
    errors = []
    for points, corners, rvec, tvec in zip(object_points, image_corners, rvecs, tvecs):
        predicted, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
        errors.append(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=2))))
    return rms, camera_matrix, distortion, np.array(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--square", type=float, default=0.025)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--output", default="camera_intrinsics.yaml")
    parser.add_argument("--name", default="camera")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    board_size = (args.cols, args.rows)

    # The board's corners in its own frame: a flat grid in the z = 0 plane, in metres.
    board_points = np.zeros((args.cols * args.rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    image_corners, image_size, tried = [], None, 0
    for name, picture in read_pictures(args.source, args.every):
        tried += 1
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]   # (width, height)
        elif gray.shape[::-1] != image_size:
            sys.exit(f"{name} has size {gray.shape[::-1]}, the first picture had {image_size}. "
                     "All pictures must come from the same camera at the same resolution.")
        corners = find_corners(gray, board_size)
        print(f"{name}: {'board found' if corners is not None else 'no board'}")
        if corners is None:
            continue
        image_corners.append(corners)
        if args.show:
            cv2.drawChessboardCorners(picture, board_size, corners, True)
            cv2.imshow("calibration", picture)
            cv2.waitKey(100)
    if args.show:
        cv2.destroyAllWindows()

    if len(image_corners) < 10:
        sys.exit(f"The board was found in only {len(image_corners)} of {tried} pictures. You need at least 10. "
                 "Check --cols and --rows (inner corners, not squares) and take more pictures.")
    if len(image_corners) > MAX_PICTURES:
        keep = np.linspace(0, len(image_corners) - 1, MAX_PICTURES).round().astype(int)
        image_corners = [image_corners[i] for i in keep]
    print(f"\nCalibrating with {len(image_corners)} pictures of size {image_size[0]} x {image_size[1]} ...")

    rms, camera_matrix, distortion, errors = calibrate(board_points, image_corners, image_size)
    # A blurred picture gives corners in the wrong place and pulls the whole result off. Drop the
    # pictures that fit much worse than the typical one, and calibrate again without them.
    keep = errors < 2.0 * np.median(errors)
    if not keep.all():
        print(f"Dropping {np.count_nonzero(~keep)} pictures that fit badly (blurred?) and calibrating again ...")
        image_corners = [c for c, k in zip(image_corners, keep) if k]
        rms, camera_matrix, distortion, errors = calibrate(board_points, image_corners, image_size)

    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    width, height = image_size
    print(f"\nRMS reprojection error: {rms:.3f} pixels (good: below 0.5, acceptable: below 1.0)")
    print(f"Worst picture: {errors.max():.3f} pixels")
    print(f"Focal length fx, fy:   {fx:.1f}, {fy:.1f} pixels")
    print(f"Principal point cx, cy: {cx:.1f}, {cy:.1f} pixels (image centre: {width / 2:.1f}, {height / 2:.1f})")
    print(f"Field of view:          {np.degrees(2 * np.arctan(width / 2 / fx)):.1f} deg horizontal, "
          f"{np.degrees(2 * np.arctan(height / 2 / fy)):.1f} deg vertical")
    print(f"Distortion k1 k2 p1 p2 k3: {np.round(distortion.ravel(), 4).tolist()}")

    result = {
        "camera_name": args.name,
        "image_width": int(width),
        "image_height": int(height),
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in camera_matrix.ravel()]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [float(v) for v in distortion.ravel()[:5]]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in np.eye(3).ravel()]},
        "projection_matrix": {"rows": 3, "cols": 4,
                              "data": [float(v) for v in np.hstack([camera_matrix, np.zeros((3, 1))]).ravel()]},
        "calibration": {"rms_reprojection_error_px": float(rms), "pictures_used": len(image_corners),
                        "board_inner_corners": [args.cols, args.rows], "square_size_m": args.square},
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
