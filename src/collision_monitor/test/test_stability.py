import math

import numpy as np

from collision_monitor.logic import LEVEL_CRITICAL, LEVEL_INFO, LEVEL_NONE, LEVEL_WARN
from collision_monitor.stability import (
    GRAVITY,
    gravity_specific_force,
    limit_command,
    load_transfer_ratio,
    margin_level,
    order_polygon,
    predicted_specific_force,
    stability_margin,
)

RECT = order_polygon([(0.2, 0.17), (0.2, -0.17), (-0.2, 0.17), (-0.2, -0.17)])
DIAMOND = order_polygon([(0.17, 0.0), (0.0, 0.17), (-0.17, 0.0), (0.0, -0.17)])


def margin_at(polygon, h, roll=0.0, pitch=0.0, v=0.0, omega=0.0):
    f = predicted_specific_force(roll, pitch, v, omega)
    return stability_margin(polygon, (0.0, 0.0, h), -f)[0]


def test_flat_ground_matches_static_stability_factor():
    h = 0.5
    # Nearest rectangle edges are the sides, 0.17 m from the CG.
    assert math.isclose(margin_at(RECT, h), math.atan(0.17 / h), rel_tol=1e-9)


def test_diamond_nearest_edge_is_wheel_to_caster():
    h = 0.12
    d = 0.17 / math.sqrt(2.0)
    assert math.isclose(margin_at(DIAMOND, h), math.atan(d / h), rel_tol=1e-9)


def test_margin_is_zero_at_the_tip_angle():
    h = 0.5
    tip = math.atan(0.17 / h)
    assert abs(margin_at(RECT, h, roll=tip)) < 1e-9
    assert margin_at(RECT, h, roll=tip + 0.05) < 0.0


def test_left_and_right_are_symmetric():
    h = 0.4
    assert math.isclose(margin_at(RECT, h, roll=0.1), margin_at(RECT, h, roll=-0.1))
    assert math.isclose(margin_at(DIAMOND, h, pitch=0.1), margin_at(DIAMOND, h, pitch=-0.1))


def test_tipping_edge_is_downhill():
    h = 0.4
    f = gravity_specific_force(0.2, 0.0)  # left side up: right wheels downhill
    _, edge = stability_margin(RECT, (0.0, 0.0, h), -f)
    p0, p1 = RECT[edge], RECT[(edge + 1) % len(RECT)]
    assert p0[1] < 0.0 and p1[1] < 0.0


def test_turn_and_slope_add():
    h = 0.5
    turning = margin_at(RECT, h, v=1.0, omega=1.5)
    sloped = margin_at(RECT, h, roll=0.1)
    both = margin_at(RECT, h, roll=0.1, v=1.0, omega=1.5)
    assert both < turning and both < sloped


def test_turn_towards_downhill_partly_cancels_slope():
    h = 0.5
    # Left side up; a right turn puts the outside of the turn uphill.
    assert margin_at(RECT, h, roll=0.1, v=1.0, omega=-1.0) > margin_at(RECT, h, roll=0.1)


def test_load_transfer_ratio():
    h, half = 0.5, 0.17
    assert load_transfer_ratio(0.0, GRAVITY, h, half) == 0.0
    tip = math.atan(half / h)
    f = gravity_specific_force(tip, 0.0)
    assert math.isclose(load_transfer_ratio(f[1], f[2], h, half), 1.0, rel_tol=1e-9)


def test_limit_command_passes_safe_commands():
    def fn(v, w):
        return margin_at(RECT, 0.5, v=v, omega=w)
    assert limit_command(0.5, 0.5, fn, math.radians(5.0)) == (0.5, 0.5, 1.0)


def test_limit_command_meets_the_limit_and_never_speeds_up():
    limit = math.radians(10.0)

    def fn(v, w):
        return margin_at(RECT, 0.5, roll=0.15, v=v, omega=w)
    v, w, k = limit_command(2.0, 3.0, fn, limit)
    assert 0.0 < k < 1.0
    assert abs(v) <= 2.0 and abs(w) <= 3.0
    assert fn(v, w) >= limit
    assert math.isclose(v / 2.0, w / 3.0)


def test_limit_command_stops_when_the_slope_alone_is_too_steep():
    def fn(v, w):
        return margin_at(RECT, 0.5, roll=0.4, v=v, omega=w)
    assert limit_command(1.0, 1.0, fn, math.radians(5.0)) == (0.0, 0.0, 0.0)


def test_margin_level():
    assert margin_level(20.0, 15.0, 10.0, 5.0) == LEVEL_NONE
    assert margin_level(12.0, 15.0, 10.0, 5.0) == LEVEL_INFO
    assert margin_level(7.0, 15.0, 10.0, 5.0) == LEVEL_WARN
    assert margin_level(2.0, 15.0, 10.0, 5.0) == LEVEL_CRITICAL


def test_order_polygon_is_counter_clockwise():
    p = order_polygon([(1, 0), (-1, 0), (0, 1), (0, -1)])
    area = 0.5 * sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1]
                     for i in range(4))
    assert area > 0.0
    assert np.allclose(sorted(map(tuple, p)), sorted([(1, 0), (-1, 0), (0, 1), (0, -1)]))
