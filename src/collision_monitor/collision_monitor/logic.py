"""Planar collision risk for one object relative to the ego vehicle.

Positions and velocities are relative (object minus ego) in the tracking frame.
Distance is surface to surface using circumscribed radii, and time to collision is
that distance divided by the closing speed.
"""

import math

LEVEL_NONE = 0
LEVEL_INFO = 1
LEVEL_WARN = 2
LEVEL_CRITICAL = 3

DEFAULT_THRESHOLDS = {
    'crit_dist': 0.5,
    'warn_dist': 1.5,
    'info_dist': 3.0,
    'crit_ttc': 1.0,
    'warn_ttc': 3.0,
}


def assess(rel_pos, rel_vel, ego_radius, obj_radius, thresholds=None):
    """Assess collision risk for one object.

    rel_pos, rel_vel: 2-tuples (x, y) in the inertial frame (object minus ego).
    Returns dict with level, distance (surface), ttc (s), closing_speed (m/s).
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    rx, ry = rel_pos
    vx, vy = rel_vel
    center_range = math.hypot(rx, ry)
    surface = max(0.0, center_range - ego_radius - obj_radius)

    if center_range < 1e-6:
        closing = 0.0
    else:
        # Closing speed = component of -rel_vel along the ego->object direction.
        ux, uy = rx / center_range, ry / center_range
        closing = -(vx * ux + vy * uy)

    if closing > 1e-3:
        ttc = surface / closing
    else:
        ttc = float('inf')

    if surface <= th['crit_dist'] or ttc <= th['crit_ttc']:
        level = LEVEL_CRITICAL
    elif surface <= th['warn_dist'] or ttc <= th['warn_ttc']:
        level = LEVEL_WARN
    elif surface <= th['info_dist']:
        level = LEVEL_INFO
    else:
        level = LEVEL_NONE

    return {
        'level': level,
        'distance': surface,
        'ttc': ttc,
        'closing_speed': closing,
    }


def circumscribed_radius(size_x, size_y):
    return 0.5 * math.hypot(size_x, size_y)
