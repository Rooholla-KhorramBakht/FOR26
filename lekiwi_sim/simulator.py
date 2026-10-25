"""A small MuJoCo simulator of the LeKiwi robot (omni-wheel base + SO-101 style arm).

The simulator speaks the same units as the real robot does through LeRobot:

    arm joints     degrees, 0 = the middle of each joint's range
    gripper        percent, 0 = closed, 100 = open
    base velocity  forward m/s, left m/s, counter-clockwise deg/s

Typical use:

    from lekiwi_sim import LeKiwiSim

    sim = LeKiwiSim()
    sim.set_command(arm=[0, 0, 0, 0, 0, 50], base=[0.1, 0.0, 0.0])
    for _ in range(100):
        sim.step()
    print(sim.get_state()["base_pose"])
"""

import atexit
import math
import os
import time
from pathlib import Path

# Without a monitor (for example over SSH) MuJoCo needs the EGL backend to draw camera images.
# Set MUJOCO_GL yourself before starting Python to override it.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] == "egl":
    # mujoco sets this itself, but only if MUJOCO_GL was already set when mujoco was first imported.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "LeKiwi-sim" / "mjcf_lcmm_robot.xml"

# ---------------------------------------------------------------------------------------------
# Robot description: names in the MuJoCo model and how they map to the real robot's units.
# ---------------------------------------------------------------------------------------------

# Command/state order of the arm, using the real robot's joint names.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# The same joints under the names the MuJoCo model uses.
MODEL_ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
MODEL_WHEEL_ACTUATORS = ("drive_motor_1", "drive_motor_2", "drive_motor_3")
MODEL_BASE_BODY = "base_plate_layer1-v5-1"
MODEL_FIXED_JAW_BODY = "Wrist_Roll_08c-v1"
MODEL_MOVING_JAW_BODY = "Moving_Jaw_08d-v1"

# model_angle_deg = ARM_SIGN * real_angle_deg + ARM_OFFSET_DEG, for the five rotary arm joints.
# The model's joints do not share the real robot's zero pose or turning direction. These values
# come from the reference project, which matched the model against a real arm joint by joint.
ARM_SIGN = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
ARM_OFFSET_DEG = np.array([0.0, -90.0, 90.0, 80.0, 270.0])
# The gripper is a percentage on the real robot. 0 % and 100 % map to these jaw angles (radians).
JAW_CLOSED_RAD, JAW_OPEN_RAD = -0.174533, 1.745329

# In the model's base frame the robot's front points along +y and its left side along -x.
FORWARD_IN_BASE = np.array([0.0, 1.0, 0.0])
LEFT_IN_BASE = np.array([-1.0, 0.0, 0.0])
# Distance from a wheel's axle to the floor: roller circle radius 5.08 cm + roller radius 1.2 cm.
WHEEL_RADIUS = 0.0628
# The real wheel servos top out at 3000 encoder steps per second (4096 steps per turn).
MAX_WHEEL_SPEED = 3000.0 * 2.0 * math.pi / 4096.0  # rad/s
# Wheel motors of the simulated robot (see _build_model).
WHEEL_STIFFNESS = 30.0   # N m per radian between the wheel's target angle and its actual angle
WHEEL_DAMPING = 3.0      # N m per rad/s, the value the reference model uses
# Where the "gripper" marker sits in the fixed jaw's own frame (metres): between the fingertips.
GRIPPER_SITE_POS = [0.008, 0.0, 0.090]
# The tip of the fixed finger, in the fixed jaw's own frame. This is the arm's "end effector":
# the point that forward and inverse kinematics are about, and the point that presses keys.
FINGERTIP_SITE_POS = [-0.013, 0.0, 0.105]
# Gripper motor of the simulated robot (see _build_model for why it differs from the reference model).
GRIPPER_STIFFNESS = 20.0   # N m per radian of error between target and actual jaw angle
GRIPPER_MAX_TORQUE = 0.5   # N m, about 8 N of squeeze at the fingertips
# Contact settings for the finger pads and the cubes. MuJoCo's contacts are slightly soft by
# default; these values make them stiff enough that a squeezed cube is not pushed into the fingers.
GRASP_SOLREF = [0.004, 1.0]
GRASP_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]
# A servo cannot jump to a new angle. The simulated arm follows its targets at up to this speed.
MAX_ARM_SPEED = 180.0  # deg/s

# ---------------------------------------------------------------------------------------------
# Scene: what is placed around the robot. The robot starts at the origin, facing +x.
# ---------------------------------------------------------------------------------------------

CUBE_SIZE = 0.03  # edge length in metres
# name: (x, y, colour). Each cube starts resting on the platform below it.
CUBES = {
    "red_cube": (0.30, 0.06, (0.85, 0.15, 0.15, 1.0)),
    "green_cube": (0.30, -0.06, (0.15, 0.65, 0.20, 1.0)),
    "blue_cube": (0.36, 0.00, (0.15, 0.35, 0.90, 1.0)),
}
# name: (centre x, centre y, size x, size y, height, colour). Static boxes standing on the floor.
PLATFORMS = {
    "table": (0.34, 0.0, 0.16, 0.30, 0.04, (0.75, 0.70, 0.60, 1.0)),
    "shelf_left": (0.10, 0.34, 0.20, 0.16, 0.08, (0.55, 0.60, 0.70, 1.0)),
    "shelf_right": (0.10, -0.34, 0.20, 0.16, 0.12, (0.55, 0.60, 0.70, 1.0)),
}

# The keyboard scene: a plate with three keys that can be pressed down.
# name: (x, y) of the key's centre. "a" is on the robot's left, "d" on its right.
KEYS = {"key_a": (0.30, 0.07), "key_s": (0.30, 0.0), "key_d": (0.30, -0.07)}
KEYBOARD_HEIGHT = 0.02    # top of the plate above the floor (m)
KEY_SIZE = 0.024          # width of a key cap (m)
KEY_HEIGHT = 0.012        # height of a key cap (m)
KEY_TRAVEL = 0.006        # how far a key can be pushed down (m)
KEY_PRESSED_AT = 0.003    # a key counts as pressed when it is this far down (m)

# The ArUco scene: printed markers standing in the room, read from a marker map file (see
# Assignment4). Each marker is drawn on a white card that leaves a white border around it, like paper.
ARUCO_CARD_MARGIN = 0.25   # white border on each side, as a fraction of the marker's edge length
ARUCO_CARD_THICKNESS = 0.004
ARUCO_MARKER_PIXELS = 192   # size of the marker itself in its picture: 6 cells of 32 pixels for a 4x4 marker

# Collision bits. The reference model switches collisions off for almost every robot part and
# uses bit 1 for wheels/floor and bit 2 for a few spheres that keep the arm above the floor.
# We add bit 4 for the two gripper jaws so that they can touch cubes and platforms.
BIT_WORLD, BIT_ARM_PROXY, BIT_JAW = 1, 2, 4


def _look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) for a MuJoCo camera at `position` looking at `target`.

    `up` is the direction that should point to the top of the image.
    """
    position, target = np.asarray(position, float), np.asarray(target, float)
    back = position - target
    back /= np.linalg.norm(back)  # a MuJoCo camera looks along its own -z axis
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, back]).flatten())
    return quat


def _make_renderer(model, width, height):
    """Create an off-screen renderer that also works without a monitor.

    MuJoCo picks its drawing backend when `mujoco` is first imported. If your script imported
    mujoco before this module, it picked the windowed backend, which fails over SSH. Here we
    point the renderer at the EGL backend explicitly, so the import order does not matter.
    """
    if os.environ.get("MUJOCO_GL") == "egl":
        try:
            import mujoco.egl
            from mujoco.rendering.classic import gl_context
            gl_context.GLContext = mujoco.egl.GLContext
        except ImportError:  # other MuJoCo versions: fall back to whatever mujoco chose
            pass
    return mujoco.Renderer(model, height, width)


def _add_aruco_markers(spec, marker_map):
    """Add one white card with a printed ArUco marker on it for each marker of a marker map."""
    import cv2   # only this scene needs OpenCV

    from scipy.spatial.transform import Rotation

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_map.get("dictionary", "DICT_4X4_50")))
    size = float(marker_map["marker_size"])
    border = round(ARUCO_MARKER_PIXELS * ARUCO_CARD_MARGIN)   # white border in the picture, in pixels
    pixels = ARUCO_MARKER_PIXELS + 2 * border
    card = size * pixels / ARUCO_MARKER_PIXELS
    world = spec.worldbody
    # A light grey sky instead of black, and enough light that the white cards look white.
    spec.add_texture(name="aruco_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.85, 0.85], rgb2=[0.6, 0.6, 0.6], width=64, height=64)
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    for marker in marker_map["markers"]:
        marker_id = int(marker["id"])
        # The picture: the marker with its white border, as the card's whole front face.
        picture = np.full((pixels, pixels), 255, np.uint8)
        picture[border:border + ARUCO_MARKER_PIXELS, border:border + ARUCO_MARKER_PIXELS] = \
            cv2.aruco.generateImageMarker(dictionary, marker_id, ARUCO_MARKER_PIXELS)
        rgb = np.repeat(picture[:, :, None], 3, axis=2)
        texture = spec.add_texture(name=f"aruco_{marker_id}_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=pixels, height=pixels, nchannel=3)
        texture.data = rgb.tobytes()
        material = spec.add_material(name=f"aruco_{marker_id}_material")
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        # The card is a thin box. Its own frame is the marker's frame: x right, y up, z out of the
        # printed side. Its centre sits half a thickness behind the printed surface.
        rotation = Rotation.from_quat(marker["orientation"])
        normal = rotation.as_matrix()[:, 2]
        qx, qy, qz, qw = rotation.as_quat()
        world.add_geom(name=f"aruco_{marker_id}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=np.asarray(marker["position"], float) - normal * ARUCO_CARD_THICKNESS / 2,
                       quat=[qw, qx, qy, qz], size=[card / 2, card / 2, ARUCO_CARD_THICKNESS / 2],
                       material=material.name, contype=0, conaffinity=0)


class LeKiwiSim:
    """MuJoCo simulation of LeKiwi with a few cubes and platforms.

    Args:
        control_dt: seconds of simulated time that one call to step() advances.
        model_path: MJCF file of the robot. Defaults to the model in models/LeKiwi-sim.
        scene: what to put around the robot. "cubes" (a table with three cubes and two shelves),
            "keyboard" (a plate with three keys to press) or "empty" (only the floor).
        add_scene: older way to ask for an empty scene; add_scene=False is the same as scene="empty".
        aruco_markers: for scene="aruco": the markers to put in the room, as a dictionary with the
            layout of a marker map file: {"dictionary": "DICT_4X4_50", "marker_size": 0.15,
            "markers": [{"id": 1, "position": [x, y, z], "orientation": [qx, qy, qz, qw]}, ...]}.
            A marker's frame has x to the right, y up and z out of the printed side.
    """

    def __init__(self, control_dt=0.02, model_path=None, add_scene=True, scene="cubes", aruco_markers=None):
        if not add_scene:
            scene = "empty"
        if scene not in ("cubes", "keyboard", "empty", "aruco"):
            raise ValueError(f'scene must be "cubes", "keyboard", "empty" or "aruco", got {scene!r}')
        if scene == "aruco" and not aruco_markers:
            raise ValueError('scene="aruco" needs aruco_markers (the contents of a marker map file)')
        self.scene = scene
        model_path = Path(model_path) if model_path else DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Robot model not found at {model_path}. "
                "In the repository folder run: git submodule update --init"
            )
        self.model = self._build_model(model_path, scene, aruco_markers)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self._substeps = max(1, round(control_dt / self.model.opt.timestep))

        m = self.model
        name2id = mujoco.mj_name2id
        self._base_body = name2id(m, mujoco.mjtObj.mjOBJ_BODY, MODEL_BASE_BODY)
        self._base_qpos = m.jnt_qposadr[name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
        self._gripper_site = name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self._fingertip_site = name2id(m, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
        self._key_qpos = {n: m.jnt_qposadr[name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_slide")]
                          for n in KEYS if name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_slide") >= 0}
        self._markers = {}
        arm_joint_ids = [name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MODEL_ARM_JOINTS]
        self._arm_qpos = np.array([m.jnt_qposadr[j] for j in arm_joint_ids])
        self._arm_qvel = np.array([m.jnt_dofadr[j] for j in arm_joint_ids])
        self._arm_ctrl = np.array([name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in MODEL_ARM_JOINTS])
        self._wheel_ctrl = np.array([name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in MODEL_WHEEL_ACTUATORS])
        self._wheel_qvel = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self._wheel_ctrl])
        self._object_bodies = {
            n: name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in CUBES if name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
        }
        self.camera_names = ["front", "wrist", "overview", "top"]

        self._renderers = {}
        self._viewer = None
        self._wall_start = None
        self._arm_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self._base_command = np.zeros(3)
        self.reset()
        self._wheel_geometry = self._measure_wheels()

    # -----------------------------------------------------------------------------------------
    # Building the model
    # -----------------------------------------------------------------------------------------

    @staticmethod
    def _build_model(model_path, scene, aruco_markers=None):
        """Load the robot MJCF and add our own parts to it in memory (the file is not changed)."""
        spec = mujoco.MjSpec.from_file(str(model_path))
        box = mujoco.mjtGeom.mjGEOM_BOX

        # Turn the robot by -90 degrees so that its front (+y in its own frame) points along world +x.
        base = spec.body(MODEL_BASE_BODY)
        base.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        base.quat = [math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)]

        # Let the two gripper fingers touch things. The reference model has no collision shapes on
        # the gripper, so we add one invisible box ("pad") along the inside of each finger.
        # Each entry: body, pad centre and pad half-sizes, in that body's own frame (metres).
        pads = ((MODEL_FIXED_JAW_BODY, [-0.018, 0.0, 0.075], [0.006, 0.009, 0.030]),
                (MODEL_MOVING_JAW_BODY, [-0.003, -0.052, 0.019], [0.006, 0.030, 0.009]))
        for body_name, pos, size in pads:
            spec.body(body_name).add_geom(name=body_name + "_pad", type=box, pos=pos, size=size, group=3,
                                          rgba=[0.1, 0.8, 0.1, 0.4], contype=BIT_JAW, conaffinity=0,
                                          friction=[2.0, 0.05, 0.001], condim=4, mass=0.0,
                                          solref=GRASP_SOLREF, solimp=GRASP_SOLIMP)
        # Mark the point between the fingertips, and give the gripper a realistic closing force.
        spec.body(MODEL_FIXED_JAW_BODY).add_site(name="gripper", pos=GRIPPER_SITE_POS, size=[0.004, 0.004, 0.004],
                                                 rgba=[1.0, 0.9, 0.0, 1.0])
        spec.body(MODEL_FIXED_JAW_BODY).add_site(name="fingertip", pos=FINGERTIP_SITE_POS, size=[0.003, 0.003, 0.003],
                                                 rgba=[1.0, 0.2, 0.8, 1.0])
        # In the reference model the gripper motor can push with 80 N m and its joint is given a
        # very large inertia. That is strong enough to drive the fingers straight through a small
        # cube. We replace it with a weaker, lighter motor that closes in about a second and then
        # squeezes with a few newtons.
        jaw_joint = spec.joint("Jaw")
        jaw_joint.armature = 0.005
        # Recent MuJoCo versions store damping as three polynomial coefficients, older ones as one number.
        jaw_joint.damping = [0.4, 0.0, 0.0] if np.ndim(jaw_joint.damping) else 0.4
        jaw = spec.actuator("Jaw")
        gain, bias = np.zeros(len(jaw.gainprm)), np.zeros(len(jaw.biasprm))
        gain[0], bias[1] = GRIPPER_STIFFNESS, -GRIPPER_STIFFNESS
        jaw.gainprm, jaw.biasprm = gain, bias
        jaw.forcerange = [-GRIPPER_MAX_TORQUE, GRIPPER_MAX_TORQUE]

        # Wheel motors. In the reference model they only resist speed, so a steady push (such as the
        # arm leaning forward) makes the stopped robot creep. A real servo holds its position. We
        # turn each wheel motor into one that integrates the commanded speed into a target angle
        # and holds that angle, which is MuJoCo's "intvelocity" actuator.
        for name in MODEL_WHEEL_ACTUATORS:
            wheel = spec.actuator(name)
            gain, bias = np.zeros(len(wheel.gainprm)), np.zeros(len(wheel.biasprm))
            gain[0], bias[1], bias[2] = WHEEL_STIFFNESS, -WHEEL_STIFFNESS, -WHEEL_DAMPING
            wheel.gainprm, wheel.biasprm = gain, bias
            wheel.dyntype = mujoco.mjtDyn.mjDYN_INTEGRATOR
            wheel.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            wheel.biastype = mujoco.mjtBias.mjBIAS_AFFINE

        # The reference model ships with one block on the floor. We turn it into one of our cubes.
        block = spec.body("target_block")
        world = spec.worldbody
        if scene == "keyboard":
            block.pos = [0.0, 3.5, 0.025]  # park the built-in block far away from the robot
            xs, ys = [x for x, _ in KEYS.values()], [y for _, y in KEYS.values()]
            world.add_geom(name="keyboard", type=box, rgba=[0.25, 0.25, 0.28, 1.0],
                           pos=[sum(xs) / len(xs), sum(ys) / len(ys), KEYBOARD_HEIGHT / 2],
                           size=[0.05, (max(ys) - min(ys)) / 2 + 0.04, KEYBOARD_HEIGHT / 2],
                           contype=BIT_WORLD, conaffinity=BIT_WORLD | BIT_ARM_PROXY | BIT_JAW)
            colours = ([0.85, 0.15, 0.15, 1.0], [0.15, 0.65, 0.20, 1.0], [0.15, 0.35, 0.90, 1.0])
            for (name, (x, y)), rgba in zip(KEYS.items(), colours):
                # A key is a small box on a vertical sliding joint with a spring that pushes it back up.
                key = world.add_body(name=name, pos=[x, y, KEYBOARD_HEIGHT + KEY_TRAVEL + KEY_HEIGHT / 2])
                key.add_joint(name=name + "_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                              range=[-KEY_TRAVEL, 0.0], stiffness=60.0, damping=[0.5, 0.0, 0.0])
                key.add_geom(name=name + "_geom", type=box, rgba=rgba, mass=0.005,
                             size=[KEY_SIZE / 2, KEY_SIZE / 2, KEY_HEIGHT / 2],
                             contype=0, conaffinity=BIT_JAW, solref=GRASP_SOLREF, solimp=GRASP_SOLIMP)
        elif scene == "cubes":
            touches = BIT_WORLD | BIT_ARM_PROXY | BIT_JAW
            for name, (x, y, sx, sy, height, rgba) in PLATFORMS.items():
                world.add_geom(name=name, type=box, pos=[x, y, height / 2], size=[sx / 2, sy / 2, height / 2],
                               rgba=rgba, contype=BIT_WORLD, conaffinity=touches, friction=[1.0, 0.005, 0.0001])
            half = CUBE_SIZE / 2
            for i, (name, (x, y, rgba)) in enumerate(CUBES.items()):
                height = max(
                    [h for (px, py, sx, sy, h, _) in PLATFORMS.values()
                     if abs(x - px) <= sx / 2 and abs(y - py) <= sy / 2] or [0.0])
                if i == 0:
                    body, geom = block, block.geoms[0]
                    body.name, geom.name = name, name + "_geom"
                    body.pos = [x, y, height + half]
                else:
                    body = world.add_body(name=name, pos=[x, y, height + half])
                    body.add_freejoint(name=name + "_free")
                    geom = body.add_geom(name=name + "_geom", type=box)
                geom.size = [half, half, half]
                geom.rgba = rgba
                geom.mass = 0.02
                geom.contype = BIT_WORLD
                geom.conaffinity = BIT_WORLD | BIT_JAW
                geom.friction = [2.0, 0.05, 0.001]
                geom.condim = 4
                geom.solref = GRASP_SOLREF
                geom.solimp = GRASP_SOLIMP
        else:
            block.pos = [0.0, 3.5, 0.025]  # park the built-in block far away from the robot
        if scene == "aruco":
            # No block in this scene. (Parking it far away would also work, but the camera's near
            # clipping distance grows with the size of the scene, so nearby things would vanish.)
            spec.delete(block)
            _add_aruco_markers(spec, aruco_markers)

        # Cameras. The reference model's own camera ("front_camera") points into the robot, so we
        # add our own: two on the robot (front of the base, and on the gripper) and two fixed ones.
        base.add_camera(name="front", pos=[0.0, 0.125, 0.03], fovy=70,
                        quat=_look_at([0.0, 0.125, 0.03], [0.0, 1.0, 0.03]))
        spec.body(MODEL_FIXED_JAW_BODY).add_camera(
            name="wrist", pos=[-0.065, 0.0, 0.02], fovy=75,
            quat=_look_at([-0.065, 0.0, 0.02], [0.02, 0.0, 0.13], up=(1.0, 0.0, 0.0)))
        world.add_camera(name="overview", pos=[-0.55, -0.75, 0.65], quat=_look_at([-0.55, -0.75, 0.65], [0.25, 0.0, 0.10]))
        world.add_camera(name="top", pos=[0.25, 0.0, 1.3], quat=[math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)])
        world.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.2, 0.2, -1.0])

        # Allow camera images up to 1280 x 960 pixels (MuJoCo's default limit is 640 x 480).
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        return spec.compile()

    def _measure_wheels(self):
        """For each wheel: (hub position, rolling direction) in the base frame, read from the model."""
        rot = self.data.xmat[self._base_body].reshape(3, 3)
        origin = self.data.xpos[self._base_body]
        wheels = []
        for actuator in self._wheel_ctrl:
            joint = self.model.actuator_trnid[actuator, 0]
            hub = rot.T @ (self.data.xanchor[joint] - origin)
            axle = rot.T @ self.data.xaxis[joint]
            # A wheel turning at +w about its axle moves its hub along (axle x up) * radius * w.
            wheels.append((hub, np.cross(axle, [0.0, 0.0, 1.0])))
        return wheels

    # -----------------------------------------------------------------------------------------
    # Unit conversions between the real robot's convention and the model
    # -----------------------------------------------------------------------------------------

    @staticmethod
    def _arm_to_model(arm):
        """[5 joint angles in degrees, gripper in %] -> the model's six joint angles in radians."""
        arm = np.asarray(arm, float)
        rad = np.radians(ARM_SIGN * arm[:5] + ARM_OFFSET_DEG)
        percent = min(100.0, max(0.0, arm[5]))
        jaw = JAW_CLOSED_RAD + percent / 100.0 * (JAW_OPEN_RAD - JAW_CLOSED_RAD)
        return np.append(rad, jaw)

    @staticmethod
    def _arm_from_model(rad):
        """Inverse of _arm_to_model."""
        deg = ARM_SIGN * (np.degrees(rad[:5]) - ARM_OFFSET_DEG)
        percent = (rad[5] - JAW_CLOSED_RAD) / (JAW_OPEN_RAD - JAW_CLOSED_RAD) * 100.0
        return np.append(deg, percent)

    def _base_to_wheels(self, base):
        """[forward m/s, left m/s, counter-clockwise deg/s] -> three wheel speeds in rad/s.

        Each wheel hub moves at v + omega x r. An omni wheel only has to follow the part of that
        motion that lies along its rolling direction; its rollers let it slide sideways.
        """
        velocity = base[0] * FORWARD_IN_BASE + base[1] * LEFT_IN_BASE
        spin = np.array([0.0, 0.0, math.radians(base[2])])
        speeds = np.array([(velocity + np.cross(spin, hub)) @ rolling / WHEEL_RADIUS
                           for hub, rolling in self._wheel_geometry])
        fastest = np.abs(speeds).max()
        if fastest > MAX_WHEEL_SPEED:  # like the real robot: slow all wheels down together
            speeds *= MAX_WHEEL_SPEED / fastest
        return speeds

    # -----------------------------------------------------------------------------------------
    # The interface you program against
    # -----------------------------------------------------------------------------------------

    def reset(self):
        """Put the robot and all objects back at their start poses and zero the commands."""
        mujoco.mj_resetData(self.model, self.data)
        self._arm_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self._base_command = np.zeros(3)
        start = self._arm_to_model(self._arm_command)
        self.data.qpos[self._arm_qpos] = start
        self.data.ctrl[self._arm_ctrl] = start
        mujoco.mj_forward(self.model, self.data)
        self._wall_start = None
        if self._viewer is not None:
            self._viewer.sync()

    def set_base_pose(self, x, y, heading):
        """Put the robot at (x, y) metres in the world, facing `heading` degrees (0 = along +x), at rest.

        Call it right after creating the simulator or after reset(), before stepping.
        """
        adr = self._base_qpos
        start_quat = self.model.qpos0[adr + 3:adr + 7]   # the base's orientation at heading 0
        turn = np.array([math.cos(math.radians(heading) / 2), 0.0, 0.0, math.sin(math.radians(heading) / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, start_quat)
        self.data.qpos[adr:adr + 3] = [x, y, self.model.qpos0[adr + 2]]
        self.data.qpos[adr + 3:adr + 7] = quat
        self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
                       + np.arange(6)] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def set_command(self, arm=None, base=None):
        """Set the targets the robot follows from now on.

        Args:
            arm: six numbers [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in
                degrees and [gripper] in percent. Position targets: the arm moves there (at up
                to MAX_ARM_SPEED) and holds.
            base: three numbers [forward m/s, left m/s, counter-clockwise deg/s]. A velocity:
                the base keeps moving like this until you send a new command.

        Leave an argument out (or pass None) to keep its previous value.
        """
        if arm is not None:
            arm = np.asarray(arm, float)
            if arm.shape != (6,):
                raise ValueError(f"arm must have 6 values, got {arm.shape}")
            self._arm_command = arm.copy()
        if base is not None:
            base = np.asarray(base, float)
            if base.shape != (3,):
                raise ValueError(f"base must have 3 values, got {base.shape}")
            self._base_command = base.copy()

    def get_command(self):
        """Return the current targets as {"arm": 6 values, "base": 3 values}."""
        return {"arm": self._arm_command.copy(), "base": self._base_command.copy()}

    def step(self, n=1, realtime=False):
        """Advance the simulation by n control periods (n * control_dt seconds).

        With realtime=True the call also waits, so that simulated time does not run ahead of the
        clock on the wall. Use it together with the viewer.
        """
        lo, hi = self.model.actuator_ctrlrange[self._arm_ctrl].T
        max_change = math.radians(MAX_ARM_SPEED) * self.control_dt
        for _ in range(n):
            target = np.clip(self._arm_to_model(self._arm_command), lo, hi)
            current = self.data.ctrl[self._arm_ctrl]
            self.data.ctrl[self._arm_ctrl] = current + np.clip(target - current, -max_change, max_change)
            self.data.ctrl[self._wheel_ctrl] = self._base_to_wheels(self._base_command)
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._draw_markers(self._viewer.user_scn, 0)
                self._viewer.sync()
            if realtime:
                if self._wall_start is None:
                    self._wall_start = time.perf_counter() - self.data.time
                ahead = self.data.time - (time.perf_counter() - self._wall_start)
                if ahead > 0:
                    time.sleep(ahead)

    def get_state(self):
        """Return the current state of the robot and the objects as a dictionary.

        Keys:
            time            simulated seconds since the last reset
            arm_position    6 values: 5 joint angles (deg) and the gripper (%)
            arm_velocity    6 values: deg/s and %/s
            base_pose       [x (m), y (m), heading (deg)] in the world frame
            base_velocity   [forward (m/s), left (m/s), counter-clockwise (deg/s)] in the robot's frame
            wheel_velocity  3 wheel speeds (rad/s)
            gripper_position  [x, y, z] (m) of the point between the fingertips, in the world frame
            fingertip_position  [x, y, z] (m) of the tip of the fixed finger, in the world frame
            fingertip_in_base   the same point, in the robot's base frame (x forward, y left, z up)
            keys            {name: {"position": [x, y, z] of the key's top, "pressed": True/False}}
            objects         {name: {"position": [x, y, z], "quaternion": [w, x, y, z]}}
        """
        d = self.data
        arm_rad = d.qpos[self._arm_qpos]
        scale = np.append(ARM_SIGN, 100.0 / math.degrees(JAW_OPEN_RAD - JAW_CLOSED_RAD))
        rot = d.xmat[self._base_body].reshape(3, 3)
        forward = rot @ FORWARD_IN_BASE
        velocity = np.zeros(6)  # [angular (3), linear (3)] in the base's own frame
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self._base_body, velocity, 1)
        return {
            "time": float(d.time),
            "arm_position": self._arm_from_model(arm_rad),
            "arm_velocity": np.degrees(d.qvel[self._arm_qvel]) * scale,
            "base_pose": np.array([d.xpos[self._base_body][0], d.xpos[self._base_body][1],
                                   math.degrees(math.atan2(forward[1], forward[0]))]),
            "base_velocity": np.array([velocity[3:] @ FORWARD_IN_BASE, velocity[3:] @ LEFT_IN_BASE,
                                       math.degrees(velocity[2])]),
            "wheel_velocity": d.qvel[self._wheel_qvel].copy(),
            "gripper_position": d.site_xpos[self._gripper_site].copy(),
            "fingertip_position": d.site_xpos[self._fingertip_site].copy(),
            "fingertip_in_base": self.world_to_base(d.site_xpos[self._fingertip_site]),
            "keys": {name: {"position": self._key_top(name), "pressed": bool(-d.qpos[adr] > KEY_PRESSED_AT)}
                     for name, adr in self._key_qpos.items()},
            "objects": {name: {"position": d.xpos[body].copy(), "quaternion": d.xquat[body].copy()}
                        for name, body in self._object_bodies.items()},
        }

    def _base_frame(self):
        """Rotation and origin of the base frame (x forward, y left, z up) in the world frame."""
        rot = self.data.xmat[self._base_body].reshape(3, 3)
        axes = np.column_stack([rot @ FORWARD_IN_BASE, rot @ LEFT_IN_BASE, rot[:, 2]])
        return axes, self.data.xpos[self._base_body]

    def base_to_world(self, point):
        """Convert a point from the robot's base frame to the world frame."""
        axes, origin = self._base_frame()
        return axes @ np.asarray(point, float) + origin

    def world_to_base(self, point):
        """Convert a point from the world frame to the robot's base frame."""
        axes, origin = self._base_frame()
        return axes.T @ (np.asarray(point, float) - origin)

    def _key_top(self, name):
        """World position of the centre of a key's top surface."""
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body] + [0.0, 0.0, KEY_HEIGHT / 2]

    def set_marker(self, name, position, color=(1.0, 0.0, 1.0, 1.0), radius=0.008):
        """Show a small sphere at `position` (world frame) in the viewer and in camera images.

        A marker is only a drawing: it has no mass and touches nothing. Call again with the same
        name to move it. Pass position=None to remove it.
        """
        if position is None:
            self._markers.pop(name, None)
        else:
            self._markers[name] = (np.asarray(position, float).copy(), tuple(color), float(radius))

    def _draw_markers(self, scene, first):
        """Add the markers to a MuJoCo scene, starting at geom slot `first`."""
        scene.ngeom = first
        for position, color, radius in self._markers.values():
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
                                position, np.eye(3).flatten(), np.array(color, dtype=np.float32))
            scene.ngeom += 1

    def get_image(self, camera="overview", width=320, height=240):
        """Return what a camera sees as a (height, width, 3) array of 8-bit RGB values.

        camera is one of sim.camera_names: "front" (on the base, looking ahead), "wrist" (on the
        gripper, looking at the fingertips), "overview" or "top" (fixed in the room).
        """
        if camera not in self.camera_names:
            raise ValueError(f"Unknown camera {camera!r}. Choose one of {self.camera_names}")
        if (width, height) not in self._renderers:
            first = not self._renderers
            self._renderers[(width, height)] = _make_renderer(self.model, width, height)
            if first:
                # Free the renderers before Python shuts down. Registered here, after the first
                # renderer exists, so that it runs before MuJoCo's own clean-up of the EGL backend.
                atexit.register(self.close)
        renderer = self._renderers[(width, height)]
        renderer.update_scene(self.data, camera=camera)
        self._draw_markers(renderer.scene, renderer.scene.ngeom)
        return renderer.render()

    def open_viewer(self):
        """Open MuJoCo's interactive 3-D window. Needs a monitor (it does not work over SSH)."""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def viewer_is_open(self):
        """True while the 3-D window opened by open_viewer() has not been closed."""
        return self._viewer is not None and self._viewer.is_running()

    def close(self):
        """Close the viewer and free the camera renderers."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}

