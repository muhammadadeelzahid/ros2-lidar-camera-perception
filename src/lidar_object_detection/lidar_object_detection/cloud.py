"""Fast PointCloud2 to NumPy conversion."""

import numpy as np

from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2


def cloud_to_xyz(msg):
    """Return an (N, 3) float32 array of finite x, y, z points.

    Reads the buffer directly when x, y, z are little-endian float32 fields, which
    is what Gazebo and the common LiDAR drivers publish. Anything else goes through
    sensor_msgs_py, which is much slower.
    """
    fields = {f.name: f for f in msg.fields}
    xyz = [fields.get(n) for n in ('x', 'y', 'z')]
    n = msg.width * msg.height
    dense_rows = msg.row_step == msg.width * msg.point_step
    if (all(f is not None and f.datatype == PointField.FLOAT32 for f in xyz)
            and not msg.is_bigendian and dense_rows):
        dtype = np.dtype({'names': ['x', 'y', 'z'],
                          'formats': ['<f4', '<f4', '<f4'],
                          'offsets': [f.offset for f in xyz],
                          'itemsize': msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dtype, count=n)
        pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    else:
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float32).reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]
