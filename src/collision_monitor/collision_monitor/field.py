"""Protective field: an object-agnostic stop zone in front of the vehicle.

Works on raw LiDAR points, independent of detection and tracking, so a missed or
misclassified obstacle still stops the vehicle. The field length grows with speed.
"""

import numpy as np

from sensor_msgs.msg import PointField


def cloud_to_xyz(msg):
    """(N, 3) float32 points from a PointCloud2 with float32 x, y, z fields."""
    fields = {f.name: f for f in msg.fields}
    xyz = [fields.get(n) for n in ('x', 'y', 'z')]
    if (any(f is None or f.datatype != PointField.FLOAT32 for f in xyz)
            or msg.is_bigendian or msg.row_step != msg.width * msg.point_step):
        raise ValueError('unsupported point cloud layout')
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                      'offsets': [f.offset for f in xyz], 'itemsize': msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def field_length(speed, min_length, reaction_time):
    """Distance covered in reaction_time plus a fixed buffer."""
    return min_length + abs(speed) * reaction_time


def check_field(pts, ego_length, ego_width, length, margin, min_z, max_z, forward=True):
    """Count points inside the field and return (count, distance to the nearest one).

    pts are in the vehicle base frame: x forward, y left, z up from the ground.
    The field is a box ahead of (or behind, if forward is False) the vehicle body,
    widened by margin on each side.
    """
    half_l = ego_length / 2.0
    half_w = ego_width / 2.0 + margin
    in_band = (np.abs(pts[:, 1]) <= half_w) & (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)
    if forward:
        gap = pts[:, 0] - half_l
    else:
        gap = -pts[:, 0] - half_l
    inside = in_band & (gap >= 0.0) & (gap <= length)
    count = int(inside.sum())
    nearest = float(gap[inside].min()) if count else float('inf')
    return count, nearest
