import os

import numpy as np
import pytest

from perception_tools.eval_detector import evaluate, match


class _Det:
    def __init__(self, cls, x, y):
        self.class_name, self.x, self.y = cls, x, y

    def range(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


class _FixedModel:
    """Returns one correct and one spurious detection per frame."""

    def detect(self, cloud):
        return [_Det('Pedestrian', 3.1, 0.0), _Det('Vehicle', 8.0, 4.0)]


def test_match_is_one_to_one_and_gated():
    gts = [('Pedestrian', 0.0, 0.0), ('Vehicle', 5.0, 0.0)]
    dets = [_Det('Pedestrian', 0.2, 0.0), _Det('Pedestrian', 0.3, 0.0), _Det('Vehicle', 9.0, 0)]
    pairs = match(gts, dets, gate=1.0)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]


def test_evaluate_counts(tmp_path):
    pytest.importorskip('yaml')
    root = tmp_path / 'scene'
    os.makedirs(root / 'points')
    os.makedirs(root / 'labels')
    for i in range(2):
        np.save(root / 'points' / f'{i:06d}.npy', np.zeros((5, 4), dtype=np.float32))
        (root / 'labels' / f'{i:06d}.txt').write_text('Pedestrian 3.0 0.0 0.57 0.6 0.6 1.8 0\n')
    stats = evaluate(_FixedModel(), [str(root)])
    assert stats['frames'] == 2 and stats['tp'] == 2 and stats['dets'] == 4
    assert stats['class_ok'] == 2
    assert abs(np.mean(stats['errors']) - 0.1) < 1e-6
