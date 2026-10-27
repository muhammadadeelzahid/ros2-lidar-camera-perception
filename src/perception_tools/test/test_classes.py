from perception_tools.classes import CLASS_SIZE, classify
from perception_tools.geometry import wrap_angle, yaw_from_quaternion


def test_classify_prefixes():
    assert classify('pedestrian_1') == 'Pedestrian'
    assert classify('forklift_1') == 'Vehicle'
    assert classify('shelf_nw') is None
    assert classify('ego_bot') is None


def test_class_sizes_present():
    assert set(CLASS_SIZE) == {'Pedestrian', 'Vehicle'}


def test_geometry_helpers():
    assert abs(yaw_from_quaternion(0.0, 0.0, 0.0, 1.0)) < 1e-9
    assert abs(wrap_angle(3.5) - (3.5 - 2 * 3.141592653589793)) < 1e-6
