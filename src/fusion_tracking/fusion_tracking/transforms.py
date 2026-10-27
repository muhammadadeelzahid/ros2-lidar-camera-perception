"""Quaternion and rigid-transform helpers.

Used instead of tf2_geometry_msgs.do_transform_pose, whose signature differs
between ROS 2 releases.
"""

import math

import numpy as np


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz])


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def transform_from_msg(tf_stamped):
    """Return (translation (3,), quaternion (x,y,z,w)) from a TransformStamped."""
    t = tf_stamped.transform.translation
    r = tf_stamped.transform.rotation
    return np.array([t.x, t.y, t.z]), (r.x, r.y, r.z, r.w)


def transform_point(translation, quat, xyz):
    return quat_rotate(quat, np.asarray(xyz, dtype=float)) + translation


def transform_yaw(quat, yaw):
    return yaw_from_quat(quat_mul(quat, quat_from_yaw(yaw)))
