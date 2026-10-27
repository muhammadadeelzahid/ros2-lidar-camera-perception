import math

import numpy as np

from collision_monitor.attitude import (
    ComplementaryFilter,
    LowPass,
    accel_angles,
    euler_rates,
    rp_matrix,
)
from collision_monitor.stability import GRAVITY, gravity_specific_force


def test_accel_angles_invert_gravity():
    roll, pitch = 0.2, -0.1
    r, p = accel_angles(gravity_specific_force(roll, pitch))
    assert math.isclose(r, roll, abs_tol=1e-9)
    assert math.isclose(p, pitch, abs_tol=1e-9)


def test_rp_matrix_matches_gravity_model():
    roll, pitch = 0.3, 0.2
    f = rp_matrix(roll, pitch).T @ np.array([0.0, 0.0, GRAVITY])
    assert np.allclose(f, gravity_specific_force(roll, pitch))


def test_euler_rates_level():
    assert euler_rates(0.0, 0.0, 0.1, 0.2, 0.3) == (0.1, 0.2)


def test_filter_converges_to_static_tilt():
    flt = ComplementaryFilter(time_constant=0.5)
    flt.update((0.0, 0.0, 0.0), gravity_specific_force(0.0, 0.0), 0.01)
    f = gravity_specific_force(0.15, -0.05)
    for _ in range(500):
        roll, pitch = flt.update((0.0, 0.0, 0.0), f, 0.01)
    assert abs(roll - 0.15) < 1e-3
    assert abs(pitch + 0.05) < 1e-3


def braking_pitch(known):
    flt = ComplementaryFilter(time_constant=0.5, accel_gate=2.0)
    level = gravity_specific_force(0.0, 0.0)
    for _ in range(100):
        flt.update((0.0, 0.0, 0.0), level, 0.01)
    # Hard braking: 3 m/s^2 backwards reads as about 17 degrees of pitch.
    decel = np.array([-3.0, 0.0, 0.0])
    for _ in range(50):
        _, pitch = flt.update((0.0, 0.0, 0.0), level + decel, 0.01,
                              known_accel=decel if known else None)
    return abs(math.degrees(pitch))


def test_filter_removes_acceleration_known_from_odometry():
    assert braking_pitch(known=True) < 0.1


def test_filter_damps_unknown_acceleration():
    raw = math.degrees(math.atan2(3.0, GRAVITY))
    assert braking_pitch(known=False) < 0.6 * raw


def test_filter_bounds_gyro_bias_drift():
    flt = ComplementaryFilter(time_constant=1.0)
    level = gravity_specific_force(0.0, 0.0)
    bias = math.radians(0.5)  # 0.5 deg/s, 30 deg/min if integrated alone
    for _ in range(6000):
        roll, _ = flt.update((bias, 0.0, 0.0), level, 0.01)
    assert abs(math.degrees(roll)) < 0.6


def test_filter_tracks_rotation_from_gyro():
    flt = ComplementaryFilter(time_constant=1.0)
    rate = 0.5
    roll = 0.0
    flt.update((0.0, 0.0, 0.0), gravity_specific_force(0.0, 0.0), 0.01)
    for _ in range(40):
        roll += rate * 0.01
        est, _ = flt.update((rate, 0.0, 0.0), gravity_specific_force(roll, 0.0), 0.01)
    assert abs(est - roll) < 0.01


def test_low_pass_settles():
    lp = LowPass(5.0)
    for _ in range(200):
        out = lp.update([1.0, 2.0, 3.0], 0.01)
    assert np.allclose(out, [1.0, 2.0, 3.0], atol=1e-3)
