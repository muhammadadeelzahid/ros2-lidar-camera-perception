"""Tip-over stability from the support polygon and the force acting at the CG.

All vectors are in the base frame: x forward, y left, z up, origin on the ground
under the vehicle. The support polygon is the set of ground contact points.

The margin per edge is the force-angle measure of Papadopoulos and Rey (1996):
the angle between the net force at the CG and the line from the CG to that edge,
negative once the force points outside it. The vehicle margin is the smallest
edge margin. On flat ground at rest it equals atan(d / h), d being the horizontal
distance from the CG to the nearest edge and h the CG height.

An accelerometer measures specific force f = a - g, so the net force per unit
mass acting at the CG is -f.
"""

import math

import numpy as np

from collision_monitor.logic import LEVEL_CRITICAL, LEVEL_INFO, LEVEL_NONE, LEVEL_WARN

GRAVITY = 9.80665


def order_polygon(points):
    """Contact points (N, 2) sorted counter-clockwise around their centroid."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def edge_margins(polygon, cg, net_force):
    """Force-angle margin in radians for each edge of a counter-clockwise polygon.

    Edge i runs from polygon[i] to polygon[i + 1].
    """
    poly = np.asarray(polygon, dtype=float)
    cg = np.asarray(cg, dtype=float)
    f = np.asarray(net_force, dtype=float)
    n = len(poly)
    out = np.empty(n)
    for i in range(n):
        p0 = np.array([poly[i][0], poly[i][1], 0.0])
        p1 = np.array([poly[(i + 1) % n][0], poly[(i + 1) % n][1], 0.0])
        a = p1 - p0
        a /= np.linalg.norm(a)
        proj = np.eye(3) - np.outer(a, a)
        l_vec = proj @ (p1 - cg)
        f_vec = proj @ f
        fn, ln = np.linalg.norm(f_vec), np.linalg.norm(l_vec)
        if fn < 1e-9 or ln < 1e-9:
            out[i] = 0.0
            continue
        cos_t = np.clip(np.dot(f_vec, l_vec) / (fn * ln), -1.0, 1.0)
        # Positive when the force lies on the inner side of the line to the edge.
        sign = 1.0 if np.dot(np.cross(l_vec, f_vec), a) >= 0.0 else -1.0
        out[i] = sign * math.acos(cos_t)
    return out


def stability_margin(polygon, cg, net_force):
    """Smallest edge margin in radians and the index of that edge."""
    m = edge_margins(polygon, cg, net_force)
    i = int(np.argmin(m))
    return float(m[i]), i


def gravity_specific_force(roll, pitch, g=GRAVITY):
    """Specific force an accelerometer at rest reads at the given roll and pitch."""
    return np.array([
        -g * math.sin(pitch),
        g * math.sin(roll) * math.cos(pitch),
        g * math.cos(roll) * math.cos(pitch),
    ])


def predicted_specific_force(roll, pitch, v, omega, accel_x=0.0, g=GRAVITY):
    """Specific force expected while driving at speed v and yaw rate omega.

    The turn adds the centripetal acceleration v * omega towards the centre of the
    turn, which is +y for a left turn.
    """
    return gravity_specific_force(roll, pitch, g) + np.array([accel_x, v * omega, 0.0])


def load_transfer_ratio(f_y, f_z, cg_height, half_track):
    """Lateral load transfer ratio from specific force; +1 means the left wheels lift."""
    if abs(f_z) < 1e-6:
        return math.copysign(float('inf'), f_y)
    return cg_height * f_y / (half_track * f_z)


def limit_command(v, omega, margin_fn, min_margin, iterations=12):
    """Scale (v, omega) by the largest k in [0, 1] whose predicted margin is acceptable.

    margin_fn(v, omega) returns the predicted margin. Returns (v, omega, k); k is 0
    when even standing still does not meet the limit.
    """
    if margin_fn(v, omega) >= min_margin:
        return v, omega, 1.0
    if margin_fn(0.0, 0.0) < min_margin:
        return 0.0, 0.0, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        k = 0.5 * (lo + hi)
        if margin_fn(k * v, k * omega) >= min_margin:
            lo = k
        else:
            hi = k
    return lo * v, lo * omega, lo


def margin_level(margin, info, warn, crit):
    """Warning level for a margin; thresholds in the same unit as the margin."""
    if margin < crit:
        return LEVEL_CRITICAL
    if margin < warn:
        return LEVEL_WARN
    if margin < info:
        return LEVEL_INFO
    return LEVEL_NONE
