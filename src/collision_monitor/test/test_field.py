import numpy as np

from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from collision_monitor.field import check_field, cloud_to_xyz, field_length

EGO = dict(ego_length=0.4, ego_width=0.3, margin=0.1, min_z=0.1, max_z=2.0)


def test_point_ahead_is_in_field():
    pts = np.array([[0.5, 0.0, 0.5]])
    count, nearest = check_field(pts, length=0.6, **EGO)
    assert count == 1
    assert abs(nearest - 0.3) < 1e-9


def test_ground_side_and_far_points_are_ignored():
    pts = np.array([
        [0.6, 0.0, 0.02],   # ground
        [0.6, 0.5, 0.5],    # beside the vehicle
        [2.0, 0.0, 0.5],    # beyond the field
        [-0.6, 0.0, 0.5],   # behind
    ])
    assert check_field(pts, length=0.6, **EGO)[0] == 0


def test_reverse_checks_behind():
    pts = np.array([[-0.5, 0.0, 0.5]])
    assert check_field(pts, length=0.6, forward=False, **EGO)[0] == 1


def test_field_grows_with_speed():
    assert field_length(0.0, 0.4, 1.0) == 0.4
    assert abs(field_length(0.5, 0.4, 1.0) - 0.9) < 1e-9


def test_cloud_conversion_drops_nans():
    pts = [(1.0, 2.0, 3.0), (float('nan'), 0.0, 0.0), (4.0, 5.0, 6.0)]
    msg = point_cloud2.create_cloud_xyz32(Header(frame_id='lidar'), pts)
    out = cloud_to_xyz(msg)
    assert out.shape == (2, 3)
    assert np.allclose(out[1], [4.0, 5.0, 6.0])
