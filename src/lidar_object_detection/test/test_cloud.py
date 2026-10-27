import numpy as np

from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from lidar_object_detection.cloud import cloud_to_xyz


def test_matches_sensor_msgs_py():
    rng = np.random.RandomState(0)
    pts = rng.uniform(-10, 10, size=(500, 3)).astype(np.float32)
    msg = point_cloud2.create_cloud_xyz32(Header(frame_id='lidar'), pts.tolist())
    assert np.allclose(cloud_to_xyz(msg), pts)


def test_extra_fields_and_padding():
    # x y z intensity ring with a non-trivial point_step, like a Velodyne driver.
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
              PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1)]
    rows = [(1.0, 2.0, 3.0, 10.0, 1), (4.0, 5.0, 6.0, 20.0, 2)]
    msg = point_cloud2.create_cloud(Header(frame_id='lidar'), fields, rows)
    out = cloud_to_xyz(msg)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[1, 2, 3], [4, 5, 6]])
