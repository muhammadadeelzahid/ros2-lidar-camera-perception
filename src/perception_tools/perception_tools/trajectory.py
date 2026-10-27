"""Constant-speed motion along a polyline of waypoints.

mode 'pingpong' walks the waypoints forward then back; 'cycle' returns from the
last waypoint to the first.
"""

import math


def _route(waypoints, mode):
    pts = [tuple(p) for p in waypoints]
    if mode == 'pingpong':
        return pts + pts[-2::-1]
    if mode == 'cycle':
        return pts + [pts[0]]
    raise ValueError(f'unknown mode {mode}')


def pose_at(waypoints, speed, t, mode='pingpong'):
    """Return (x, y, yaw, vx, vy) at time t seconds."""
    route = _route(waypoints, mode)
    segs = []
    for a, b in zip(route[:-1], route[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length > 1e-6:
            segs.append((a, b, length))
    if not segs:
        x, y = route[0]
        return x, y, 0.0, 0.0, 0.0

    total = sum(s[2] for s in segs)
    d = (speed * max(t, 0.0)) % total
    for a, b, length in segs:
        if d <= length:
            break
        d -= length
    f = d / length
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]),
            math.atan2(uy, ux), speed * ux, speed * uy)
