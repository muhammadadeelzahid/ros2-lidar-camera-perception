#!/usr/bin/env python3
"""Merge recorded scenarios into an OpenPCDet custom dataset.

Each scenario is a directory written by auto_labeler:
    <scenario>/points/000000.npy   (N, 4) float32 x y z intensity, sensor frame
    <scenario>/labels/000000.txt   class x y z dx dy dz heading

Whole scenarios are assigned to train, val or test. Consecutive frames from one
recording are nearly identical, so splitting frame by frame would leak training
data into the evaluation sets.

    ros2 run perception_tools dataset_export --input dataset --output pcdet_dataset \\
        --val-frac 0.15 --test-frac 0.15
"""

import argparse
import os
import random
import shutil


def find_scenarios(paths):
    """Return {name: dir} for every directory that contains a points/ folder."""
    found = {}
    for path in paths:
        if os.path.isdir(os.path.join(path, 'points')):
            found[os.path.basename(os.path.normpath(path))] = path
            continue
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            if os.path.isdir(os.path.join(sub, 'points')):
                found[entry] = sub
    return found


def split_scenarios(names, val_frac, test_frac, seed):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_test = max(1, n_test) if test_frac > 0 else 0
        n_val = max(1, n_val) if val_frac > 0 else 0
    test = names[:n_test]
    val = names[n_test:n_test + n_val]
    train = names[n_test + n_val:]
    return train, val, test


def export(scenarios, output, val_frac=0.15, test_frac=0.15, seed=0):
    out_points = os.path.join(output, 'points')
    out_labels = os.path.join(output, 'labels')
    image_sets = os.path.join(output, 'ImageSets')
    for d in (out_points, out_labels, image_sets):
        os.makedirs(d, exist_ok=True)

    train, val, test = split_scenarios(list(scenarios), val_frac, test_frac, seed)
    sets = {'train': [], 'val': [], 'test': []}
    for split, names in (('train', train), ('val', val), ('test', test)):
        for name in names:
            src = scenarios[name]
            for f in sorted(os.listdir(os.path.join(src, 'points'))):
                if not f.endswith('.npy'):
                    continue
                fid = f[:-4]
                out_id = f'{name}_{fid}'
                shutil.copy(os.path.join(src, 'points', f),
                            os.path.join(out_points, out_id + '.npy'))
                label = os.path.join(src, 'labels', fid + '.txt')
                if os.path.exists(label):
                    shutil.copy(label, os.path.join(out_labels, out_id + '.txt'))
                sets[split].append(out_id)

    for split, ids in sets.items():
        with open(os.path.join(image_sets, split + '.txt'), 'w') as fh:
            fh.write('\n'.join(ids) + ('\n' if ids else ''))
    return {'train': train, 'val': val, 'test': test}, sets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', required=True,
                        help='scenario directories, or a directory containing them')
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    scenarios = find_scenarios(args.input)
    if not scenarios:
        raise SystemExit('no scenarios found (expected directories with points/)')
    if len(scenarios) < 3:
        print(f'warning: only {len(scenarios)} scenario(s); record more for a '
              'meaningful val/test split')

    split, sets = export(scenarios, args.output, args.val_frac, args.test_frac, args.seed)
    for name in ('train', 'val', 'test'):
        print(f'{name}: {len(sets[name])} frames from {split[name]}')


if __name__ == '__main__':
    main()
