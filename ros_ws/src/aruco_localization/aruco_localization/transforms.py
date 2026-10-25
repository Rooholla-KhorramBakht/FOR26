"""Small helpers for poses: 4x4 transform matrices, quaternions, and 2-D poses (x, y, heading).

Conventions used everywhere in this package:

    T_a_b         a 4x4 matrix that takes a point written in frame b and writes it in frame a:
                  p_a = T_a_b @ p_b. Read it as "the pose of b in a". Chain them like T_a_c = T_a_b @ T_b_c.
    quaternion    [x, y, z, w], the order ROS messages use
    2-D pose      [x (m), y (m), theta (rad)], theta measured counter-clockwise from the x axis
"""
import math

import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Transform
from scipy.spatial.transform import Rotation


def wrap_angle(angle):
    """Bring an angle (or an array of angles) into [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def matrix_from_pose_msg(pose):
    """geometry_msgs/Pose -> 4x4 matrix."""
    T = np.eye(4)
    q = pose.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def matrix_from_transform_msg(transform):
    """geometry_msgs/Transform (as found in a TF message) -> 4x4 matrix."""
    T = np.eye(4)
    q = transform.rotation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return T


def pose_msg_from_matrix(T):
    """4x4 matrix -> geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    pose.orientation = quaternion_msg(Rotation.from_matrix(T[:3, :3]).as_quat())
    return pose


def transform_msg_from_matrix(T):
    """4x4 matrix -> geometry_msgs/Transform."""
    transform = Transform()
    transform.translation.x, transform.translation.y, transform.translation.z = (float(v) for v in T[:3, 3])
    transform.rotation = quaternion_msg(Rotation.from_matrix(T[:3, :3]).as_quat())
    return transform


def quaternion_msg(xyzw):
    q = Quaternion()
    q.x, q.y, q.z, q.w = (float(v) for v in xyzw)
    return q


def yaw_quaternion_msg(theta):
    """Quaternion message for a rotation of theta radians about z."""
    return quaternion_msg([0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)])


def matrix_from_2d(x, y, theta):
    """2-D pose -> 4x4 matrix (a rotation about z and a translation in the floor plane)."""
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def matrix_to_2d(T):
    """4x4 matrix -> 2-D pose: the position's x and y, and the heading of the matrix's x axis."""
    return np.array([T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])])
