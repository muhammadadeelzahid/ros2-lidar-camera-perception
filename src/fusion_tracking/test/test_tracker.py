import math

from fusion_tracking.fusion import confirm_class, point_in_box2d, project_point
from fusion_tracking.kalman import ConstantVelocityKF
from fusion_tracking.tracker import Detection, MultiObjectTracker
from fusion_tracking import transforms as tfx


def test_kalman_estimates_velocity():
    kf = ConstantVelocityKF([0.0, 0.0, 0.0])
    for i in range(1, 21):
        kf.predict(0.1)
        kf.update([0.1 * i, 0.0, 0.0])
    assert abs(kf.position[0] - 2.0) < 0.1
    assert abs(kf.velocity[0] - 1.0) < 0.1


def test_tracker_keeps_stable_ids():
    trk = MultiObjectTracker(gate_distance=2.0, max_age=1.0, min_hits=2)
    for i in range(6):
        d1 = Detection('Pedestrian', 0.9, [0.2 * i, 0.0, 0.0], [0.6, 0.6, 1.8])
        d2 = Detection('Vehicle', 0.8, [5.0, 0.1 * i, 0.0], [1.2, 0.8, 1.4])
        trk.step([d1, d2], 0.1)
    confirmed = trk.confirmed()
    assert len(confirmed) == 2
    assert len({t.id for t in confirmed}) == 2


def test_tracker_drops_stale_tracks():
    trk = MultiObjectTracker(gate_distance=2.0, max_age=0.3, min_hits=1)
    trk.step([Detection('Vehicle', 0.8, [1.0, 0.0, 0.0], [1.2, 0.8, 1.4])], 0.1)
    for _ in range(5):
        trk.step([], 0.1)
    assert len(trk.tracks) == 0


def test_projection_and_box():
    K = [[554.25, 0.0, 320.0], [0.0, 554.25, 240.0], [0.0, 0.0, 1.0]]
    assert project_point(K, [0.0, 0.0, -1.0]) is None
    u, v = project_point(K, [0.0, 0.0, 2.0])
    assert abs(u - 320.0) < 1e-6 and abs(v - 240.0) < 1e-6
    assert point_in_box2d(320, 240, (320, 240, 80, 160))


def test_fusion_confirm_boosts_matching_class():
    boxes = [('Vehicle', 0.9, 320, 240, 80, 160)]
    cls, score = confirm_class((320, 240), 'Vehicle', 0.5, boxes)
    assert cls == 'Vehicle'
    assert score > 0.5


def test_transform_rotation():
    q = tfx.quat_from_yaw(math.pi / 2)
    r = tfx.quat_rotate(q, [1.0, 0.0, 0.0])
    assert abs(r[0]) < 1e-6 and abs(r[1] - 1.0) < 1e-6
    assert abs(tfx.transform_yaw(q, 0.0) - math.pi / 2) < 1e-6
