"""Small, dependency-free geometry helpers."""

import math


def yaw_from_quaternion(x, y, z, w):
    """Yaw (rotation about z) from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))
