"""Take pictures of a checkerboard with the robot's camera, for calibrate_camera.py.

Usage (with the ros_mujoco environment active; do not run the camera node at the same time):

    python capture_calibration_images.py OUTPUT_FOLDER [options]

Two ways to take the pictures:
    with a monitor   a window shows the camera with the detected board drawn on it.
                     Space saves a picture, q quits.
    --auto           no window (works over SSH). A picture is saved whenever the whole board is visible
                     and it has moved since the last saved picture. Stop with Ctrl+C.

Options:
    --device 0             the camera (0 = /dev/video0)
    --width 640 --height 480   picture size. Calibrate at the size you will use: the camera node
                           uses 640 x 480 by default.
    --cols 10 --rows 7     inner corners of the board (where four squares meet)
    --count 40             stop after this many pictures
    --auto                 save automatically (see above)
"""
import argparse
import pathlib
import time

import cv2
import numpy as np

MIN_MOVE = 0.08     # --auto: the board must move by this fraction of the picture width between pictures
MIN_INTERVAL = 0.7  # --auto: at least this many seconds between pictures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(int(args.device) if args.device.isdigit() else args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.device}. Is the camera node still running? List cameras: ls /dev/video*")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != (args.width, args.height):
        print(f"Note: the camera delivers {size[0]} x {size[1]}, not {args.width} x {args.height}.")
    print(f"Saving to {output}/. " + ("Automatic mode: move the board slowly. Ctrl+C to stop."
                                      if args.auto else "Space: save a picture. q: quit."))

    saved, last_centre, last_time = len(list(output.glob("*.png"))), None, 0.0
    try:
        while saved < args.count:
            ok, picture = camera.read()
            if not ok:
                continue
            gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            # The fast check is good enough to decide whether to save; calibrate_camera.py refines later.
            found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), flags=cv2.CALIB_CB_FAST_CHECK)
            save = False
            if args.auto:
                if found and time.time() - last_time > MIN_INTERVAL:
                    centre = corners.reshape(-1, 2).mean(axis=0)
                    if last_centre is None or np.linalg.norm(centre - last_centre) > MIN_MOVE * size[0]:
                        save, last_centre = True, centre
            else:
                shown = picture.copy()
                cv2.drawChessboardCorners(shown, (args.cols, args.rows), corners, found)
                cv2.putText(shown, f"{saved} saved", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("calibration", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if found:
                        save = True
                    else:
                        print("The whole board is not visible: not saved.")
            if save:
                path = output / f"calib_{saved:03d}.png"
                cv2.imwrite(str(path), picture)
                saved += 1
                last_time = time.time()
                print(f"saved {path} ({saved}/{args.count})")
    except KeyboardInterrupt:
        pass
    camera.release()
    cv2.destroyAllWindows()
    print(f"{saved} pictures in {output}/. Next: python calibrate_camera.py {output}")


if __name__ == "__main__":
    main()
