import math

import numpy as np

from lidar_object_detection.health import ERROR, OK, WARN, blank_ranges, scan_health


def ring(n=3600, r=2.0, skip=None):
    """Points on a circle, optionally leaving out azimuths in [skip[0], skip[1]) deg."""
    az = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    if skip is not None:
        deg = np.degrees(az)
        az = az[(deg < skip[0]) | (deg >= skip[1])]
    return np.stack([r * np.cos(az), r * np.sin(az), np.full(az.shape, -0.3)], axis=1)


def test_full_view_is_ok():
    level, n, coverage, blank = scan_health(ring())
    assert level == OK and coverage == 1.0 and blank == [] and n == 3600


def test_blocked_quarter_warns():
    level, _, coverage, blank = scan_health(ring(skip=(90, 180)))
    assert level == WARN
    assert coverage == 0.75
    assert blank == [(90.0, 180.0)]


def test_mostly_blocked_is_error():
    level, _, coverage, _ = scan_health(ring(skip=(0, 270)))
    assert level == ERROR and coverage < 0.5


def test_too_few_points_is_error():
    assert scan_health(ring(n=100))[0] == ERROR


def test_points_inside_min_range_are_ignored():
    level, n, _, _ = scan_health(ring(r=0.1))
    assert level == ERROR and n == 0


def test_blank_range_wraps_past_zero():
    covered = np.ones(36, dtype=bool)
    covered[[34, 35, 0, 1]] = False
    assert blank_ranges(covered) == [(340.0, 380.0)]
