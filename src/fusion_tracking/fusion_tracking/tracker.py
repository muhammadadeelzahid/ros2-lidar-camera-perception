"""Multi-object tracker.

One constant-velocity Kalman filter per track, with gated assignment of detections
to tracks (Hungarian when scipy is available, greedy otherwise).
"""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # noqa: BLE001 - scipy optional; greedy fallback below
    linear_sum_assignment = None

from fusion_tracking.kalman import ConstantVelocityKF


class Detection:

    __slots__ = ('class_name', 'score', 'position', 'size', 'heading')

    def __init__(self, class_name, score, position, size, heading=0.0):
        self.class_name = class_name
        self.score = float(score)
        self.position = np.asarray(position, dtype=float)  # (3,)
        self.size = np.asarray(size, dtype=float)          # (3,)
        self.heading = float(heading)


class Track:

    def __init__(self, track_id, det, meas_var):
        self.id = track_id
        self.kf = ConstantVelocityKF(det.position, meas_var=meas_var)
        self.class_name = det.class_name
        self.score = det.score
        self.size = det.size.copy()
        self.heading = det.heading
        self.hits = 1
        self.time_since_update = 0.0
        self.age = 0.0

    def predict(self, dt):
        self.kf.predict(dt)
        self.time_since_update += dt
        self.age += dt

    def update(self, det, alpha=0.5):
        self.kf.update(det.position)
        # Exponential smoothing of the non-kinematic attributes.
        self.size = alpha * det.size + (1.0 - alpha) * self.size
        self.heading = det.heading
        self.score = alpha * det.score + (1.0 - alpha) * self.score
        self.class_name = det.class_name
        self.hits += 1
        self.time_since_update = 0.0

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity


class MultiObjectTracker:

    def __init__(self, gate_distance=2.0, max_age=1.0, min_hits=3, meas_var=0.2):
        self.gate = float(gate_distance)
        self.max_age = float(max_age)
        self.min_hits = int(min_hits)
        self.meas_var = float(meas_var)
        self._tracks = []
        self._next_id = 0

    def step(self, detections, dt):
        """Advance all tracks by dt and fold in this frame's detections."""
        for t in self._tracks:
            t.predict(dt)

        matches, unmatched_dets = self._associate(detections)
        for ti, di in matches:
            self._tracks[ti].update(detections[di])
        for di in unmatched_dets:
            self._tracks.append(Track(self._new_id(), detections[di], self.meas_var))

        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return self.confirmed()

    def _associate(self, detections):
        if not self._tracks or not detections:
            return [], list(range(len(detections)))
        tpos = np.array([t.position for t in self._tracks])
        dpos = np.array([d.position for d in detections])
        cost = np.linalg.norm(tpos[:, None, :] - dpos[None, :, :], axis=2)

        if linear_sum_assignment is not None:
            big = self.gate * 10.0 + 1.0
            cost_gated = np.where(cost <= self.gate, cost, big)
            pairs = list(zip(*linear_sum_assignment(cost_gated)))
        else:
            pairs = self._greedy(cost)

        matches, matched_dets = [], set()
        for r, c in pairs:
            if cost[r, c] <= self.gate:
                matches.append((r, c))
                matched_dets.add(c)
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_dets

    def _greedy(self, cost):
        """Nearest-available assignment when scipy is not present."""
        flat = sorted(((cost[r, c], r, c)
                       for r in range(cost.shape[0])
                       for c in range(cost.shape[1])), key=lambda e: e[0])
        used_r, used_c, pairs = set(), set(), []
        for _, r, c in flat:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs

    def confirmed(self):
        return [t for t in self._tracks if t.hits >= self.min_hits]

    @property
    def tracks(self):
        return list(self._tracks)

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid
