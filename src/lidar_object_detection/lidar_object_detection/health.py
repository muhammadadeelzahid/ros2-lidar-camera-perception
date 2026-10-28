"""LiDAR health from a single scan: how many returns and how much of the view they cover.

The ground in front of a level sensor returns points in every direction, so a sector
of azimuth with no returns at all usually means the sensor is blocked or dirty there,
not that the area is empty.
"""

import math

import numpy as np

OK = 0
WARN = 1
ERROR = 2


def sector_counts(points, sectors=36, min_range=0.3):
    """Number of returns in each azimuth sector, sector 0 starting straight ahead."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r >= min_range]
    az = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * math.pi)
    idx = np.minimum((az / (2.0 * math.pi) * sectors).astype(int), sectors - 1)
    return np.bincount(idx, minlength=sectors)


def scan_health(points, sectors=36, min_sector_points=5, min_points=200,
                warn_coverage=0.9, error_coverage=0.5):
    """Level, point count, covered fraction and the blank sectors as degree ranges."""
    counts = sector_counts(points, sectors)
    covered = counts >= min_sector_points
    coverage = float(covered.mean())
    n = int(counts.sum())
    if n < min_points or coverage < error_coverage:
        level = ERROR
    elif coverage < warn_coverage:
        level = WARN
    else:
        level = OK
    return level, n, coverage, blank_ranges(covered)


def blank_ranges(covered):
    """Runs of uncovered sectors as (start_deg, end_deg), wrapping past 360."""
    n = len(covered)
    width = 360.0 / n
    if covered.all():
        return []
    if not covered.any():
        return [(0.0, 360.0)]
    start = int(np.argmax(covered))  # begin scanning just after a covered sector
    out, run = [], None
    for k in range(1, n + 1):
        i = (start + k) % n
        if not covered[i] and run is None:
            run = i
        elif covered[i] and run is not None:
            out.append((run * width, ((i - run) % n) * width + run * width))
            run = None
    return out
