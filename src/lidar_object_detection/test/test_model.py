import numpy as np
import pytest

from lidar_object_detection.model import DetectorModel

CONFIG = {
    'class_names': ['Pedestrian', 'Vehicle'],
    'score_threshold': 0.1,
    'fallback': {
        'voxel_size': 0.1,
        'ground_z': -0.15,
        'max_z': 2.0,
        'max_range': 15.0,
        'cluster_eps': 0.5,
        'min_points': 10,
        'fixed_score': 0.5,
    },
}


def _blob(center, size, n=400):
    pts = (np.random.RandomState(0).rand(n, 3) - 0.5) * np.array(size)
    return pts + np.array(center)


def test_cluster_fallback_finds_two_objects():
    pytest.importorskip('scipy')
    ped = _blob([3.0, 0.0, 0.5], [0.5, 0.5, 1.6])
    veh = _blob([6.0, 3.0, 0.6], [1.1, 0.7, 1.3])
    pts = np.vstack([ped, veh])
    cloud = np.hstack([pts, np.zeros((pts.shape[0], 1))]).astype(np.float32)

    model = DetectorModel(CONFIG)
    assert model.backend == 'cluster'
    boxes = model.detect(cloud)
    assert len(boxes) >= 2
    assert all(b.class_name in ('Pedestrian', 'Vehicle') for b in boxes)
