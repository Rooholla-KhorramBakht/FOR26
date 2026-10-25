# 1. The Camera Model

## The pinhole camera

A camera turns a point in front of it into a pixel. In the **camera frame** (x to the right of the picture, y down, z forward out of the lens) a point at (X, Y, Z) lands on the pixel

```
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

| Number | Meaning | Unit |
| --- | --- | --- |
| `fx`, `fy` | focal length: how many pixels one unit of X/Z (or Y/Z) is worth. Large = zoomed in. | pixels |
| `cx`, `cy` | principal point: the pixel straight ahead of the lens, near the picture's centre | pixels |

The four numbers are written as one matrix, the **camera matrix** K:

```
    | fx   0  cx |
K = |  0  fy  cy |
    |  0   0   1 |
```

Notice the division by Z: a point twice as far away lands twice as close to the centre. That is why a picture alone cannot tell how far away something is, unless you know its size. An ArUco marker has a known size, so its apparent size in the picture tells its distance. That is the whole trick of the next guide.

## Distortion

Real lenses bend straight lines, especially near the edges of the picture, and wide-angle lenses such as the robot's camera more so. OpenCV describes this with five **distortion coefficients**:

| Coefficients | Effect |
| --- | --- |
| `k1`, `k2`, `k3` | radial: points are pushed outward or pulled inward, more the further from the centre (barrel or pincushion distortion) |
| `p1`, `p2` | tangential: the lens is not exactly parallel to the sensor |

Together, K and the five coefficients are the camera's **intrinsics**. They do not change when the camera moves. They change if you change the picture size, the zoom, or the focus.

## Calibration

You cannot read the intrinsics off a data sheet with enough precision; you measure them. You show the camera an object whose geometry you know exactly, a **checkerboard**, from many directions. In every picture OpenCV finds the corners where four squares meet. It then searches for the intrinsics (and the pose of the board in each picture) that best explain where all the corners were seen.

How good the result is shows in the **reprojection error**: put the board back where calibration says it was, project its corners through the calibrated camera, and measure how far (in pixels) they land from the corners that were actually detected.

| RMS reprojection error | Verdict |
| --- | --- |
| below 0.5 px | good |
| 0.5 to 1 px | acceptable |
| above 1 px | something is off: blurred pictures, a bent board, the wrong board size |

## Check your understanding

1. A marker 15 cm wide is 1.5 m in front of a camera with fx = 500 px. How many pixels wide is it in the picture? How wide at 3 m?
2. What happens to fx if you halve the picture's width and height?
3. Why must you calibrate at the same picture size you use later?

---

Next: [2. Calibrate your camera](02_calibrate.md)
