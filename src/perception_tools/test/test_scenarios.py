import os
import xml.dom.minidom

import numpy as np

from perception_tools.dataset_export import (
    export,
    find_scenarios,
    split_scenarios,
    to_pcdet,
)
from perception_tools.generate_world import generate


def test_world_is_valid_and_deterministic():
    world, layout = generate(3)
    xml.dom.minidom.parseString(world)
    assert world == generate(3)[0]
    assert world != generate(4)[0]
    names = [a[0] for a in layout.actors]
    assert 'pedestrian_1' in names and 'forklift_1' in names


def test_actors_avoid_spawn_area():
    _, layout = generate(11, spawn=(-8.0, 0.0), clear_radius=1.5)
    for _, _, a, b, _ in layout.actors:
        for p in (a, b):
            assert np.hypot(p[0] + 8.0, p[1]) >= 1.5


def test_split_keeps_scenarios_whole():
    names = [f'scene_{i:02d}' for i in range(10)]
    train, val, test = split_scenarios(names, 0.2, 0.2, seed=1)
    assert sorted(train + val + test) == sorted(names)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert len(test) == 2 and len(val) == 2


def _write_scenario(root, name, frames):
    for sub in ('points', 'labels'):
        os.makedirs(os.path.join(root, name, sub))
    for i in range(frames):
        np.save(os.path.join(root, name, 'points', f'{i:06d}.npy'),
                np.zeros((10, 4), dtype=np.float32))
        with open(os.path.join(root, name, 'labels', f'{i:06d}.txt'), 'w') as fh:
            fh.write('Pedestrian 1 0 0 0.6 0.6 1.8 0\n')


def test_export_writes_image_sets(tmp_path):
    root = str(tmp_path / 'dataset')
    for i in range(4):
        _write_scenario(root, f'scene_{i}', frames=3)
    out = str(tmp_path / 'out')
    split, sets = export(find_scenarios([root]), out, 0.25, 0.25, seed=0)
    assert sum(len(v) for v in sets.values()) == 12
    with open(os.path.join(out, 'ImageSets', 'test.txt')) as fh:
        test_ids = fh.read().split()
    assert all(i.startswith(split['test'][0]) for i in test_ids)
    with open(os.path.join(out, 'labels', test_ids[0] + '.txt')) as fh:
        assert fh.read().split()[-1] == 'Pedestrian'


def test_label_reordered_for_openpcdet():
    line = 'Vehicle 1.0 2.0 0.3 1.2 0.8 1.4 0.5\n'
    assert to_pcdet(line) == '1.0 2.0 0.3 1.2 0.8 1.4 0.5 Vehicle\n'


def test_export_skips_empty_frames(tmp_path):
    root = str(tmp_path / 'dataset')
    for i in range(3):
        _write_scenario(root, f'scene_{i}', frames=2)
    open(os.path.join(root, 'scene_0', 'labels', '000000.txt'), 'w').close()
    _, sets = export(find_scenarios([root]), str(tmp_path / 'a'), 0.0, 0.0)
    assert sum(len(v) for v in sets.values()) == 5
    _, sets = export(find_scenarios([root]), str(tmp_path / 'b'), 0.0, 0.0, keep_empty=True)
    assert sum(len(v) for v in sets.values()) == 6
