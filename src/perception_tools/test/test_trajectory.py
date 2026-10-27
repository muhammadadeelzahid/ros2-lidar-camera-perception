import math

from perception_tools.trajectory import pose_at

LINE = [[0.0, 0.0], [10.0, 0.0]]


def test_moves_at_constant_speed():
    x, y, yaw, vx, vy = pose_at(LINE, 2.0, 2.5)
    assert abs(x - 5.0) < 1e-9 and abs(y) < 1e-9
    assert abs(yaw) < 1e-9 and abs(vx - 2.0) < 1e-9


def test_pingpong_turns_back():
    x, _, yaw, vx, _ = pose_at(LINE, 1.0, 12.0, 'pingpong')
    assert abs(x - 8.0) < 1e-9
    assert abs(abs(yaw) - math.pi) < 1e-9 and vx < 0


def test_cycle_closes_the_loop():
    square = [[0, 0], [2, 0], [2, 2], [0, 2]]
    x, y, _, _, vy = pose_at(square, 1.0, 7.0, 'cycle')
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9 and vy < 0
    assert pose_at(square, 1.0, 8.0, 'cycle')[:2] == pose_at(square, 1.0, 0.0, 'cycle')[:2]
