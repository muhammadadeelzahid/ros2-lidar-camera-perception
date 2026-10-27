"""Camera-LiDAR fusion helpers.

A 3D detection in the camera optical frame is projected into the image. If it falls
inside a 2D detection, the camera class and score confirm or correct the LiDAR class.
"""


def project_point(K, xyz):
    """Project a point in the camera optical frame to pixel (u, v).

    Returns None if the point is at or behind the image plane (z <= 0).
    K is a 3x3 row-major list/array of intrinsics.
    """
    x, y, z = xyz
    if z <= 1e-3:
        return None
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    return (fx * x / z + cx, fy * y / z + cy)


def point_in_box2d(u, v, box):
    """box = (cx, cy, w, h) in pixels."""
    cx, cy, w, h = box
    return (abs(u - cx) <= w / 2.0) and (abs(v - cy) <= h / 2.0)


def confirm_class(projected_uv, lidar_class, lidar_score, boxes2d):
    """Match a projected 3D detection against 2D boxes.

    boxes2d: list of (class_name, score, cx, cy, w, h).
    Returns (class_name, score): the camera class with a boosted score when a box of the
    same class matches; the camera class with a modest score when only a different class
    matches; otherwise the unchanged LiDAR class/score.
    """
    if projected_uv is None:
        return lidar_class, lidar_score
    u, v = projected_uv
    best = None
    for cls, score, cx, cy, w, h in boxes2d:
        if point_in_box2d(u, v, (cx, cy, w, h)):
            if best is None or score > best[1]:
                best = (cls, score)
    if best is None:
        return lidar_class, lidar_score
    cam_cls, cam_score = best
    if cam_cls == lidar_class:
        return lidar_class, min(1.0, 0.5 * (lidar_score + cam_score) + 0.3)
    # Disagreement: trust the camera class but keep confidence modest.
    return cam_cls, max(lidar_score, cam_score) * 0.8
