"""Pure pursuit path following for a differential-drive vehicle.

The path is a polyline through the waypoints. Each cycle the vehicle aims at the
point a fixed lookahead distance further along the path and drives the arc that
reaches it: curvature 2 * y / L^2, with y the point's sideways offset in the vehicle
frame and L its distance. If that point is far off to the side, the vehicle turns on
the spot first. Speed ramps down towards the last waypoint so it stops there.
"""

import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def to_vehicle(pose, point):
    """Point (x, y) in the frame of a vehicle at pose (x, y, yaw)."""
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


def project(path, i, point):
    """Closest point on segment i of the path to point, as a fraction along it."""
    (x0, y0), (x1, y1) = path[i], path[i + 1]
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-12:
        return 1.0
    t = ((point[0] - x0) * dx + (point[1] - y0) * dy) / seg2
    return min(1.0, max(0.0, t))


def advance(path, seg, position, reach=0.3):
    """Index of the segment being followed, moving on past segments already covered."""
    while seg < len(path) - 2:
        end = path[seg + 1]
        t = project(path, seg, position)
        if t >= 1.0 or math.dist(position, end) < reach:
            seg += 1
        else:
            break
    return seg


def lookahead_point(path, seg, position, lookahead):
    """Point lookahead metres along the path past the closest point on segment seg."""
    t = project(path, seg, position)
    (x0, y0), (x1, y1) = path[seg], path[seg + 1]
    px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    left = lookahead
    i = seg
    while True:
        nx, ny = path[i + 1]
        d = math.hypot(nx - px, ny - py)
        if d >= left or i == len(path) - 2:
            if d < 1e-9:
                return nx, ny
            k = min(1.0, left / d)
            return px + k * (nx - px), py + k * (ny - py)
        left -= d
        px, py = nx, ny
        i += 1


def remaining_distance(path, seg, position):
    """Distance along the path from the closest point on segment seg to the end."""
    t = project(path, seg, position)
    (x0, y0), (x1, y1) = path[seg], path[seg + 1]
    d = math.hypot(x1 - x0, y1 - y0) * (1.0 - t)
    for i in range(seg + 1, len(path) - 1):
        d += math.dist(path[i], path[i + 1])
    return d


def command(pose, target, speed, max_yaw_rate, turn_in_place=math.radians(60)):
    """(v, w) towards a target point; turns on the spot if it is too far to the side."""
    lx, ly = to_vehicle(pose, target)
    bearing = math.atan2(ly, lx)
    if abs(bearing) > turn_in_place:
        return 0.0, math.copysign(max_yaw_rate, bearing)
    dist2 = lx * lx + ly * ly
    if dist2 < 1e-9:
        return 0.0, 0.0
    curvature = 2.0 * ly / dist2
    v = speed
    if abs(curvature) * v > max_yaw_rate:
        v = max_yaw_rate / abs(curvature)
    return v, curvature * v


def stopping_speed(distance, speed, decel, min_speed=0.05):
    """Speed that still allows stopping within distance at the given deceleration."""
    return max(min_speed, min(speed, math.sqrt(2.0 * decel * max(distance, 0.0))))


def turn_command(yaw, heading, max_yaw_rate, gain=1.5, min_rate=0.15):
    """Yaw rate to turn on the spot to a heading."""
    err = wrap(heading - yaw)
    rate = max(min_rate, min(max_yaw_rate, gain * abs(err)))
    return math.copysign(rate, err), err
