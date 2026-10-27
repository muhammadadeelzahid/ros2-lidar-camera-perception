#!/usr/bin/env python3
"""Evaluate a LiDAR detector offline on recorded scenarios.

Runs the detector from a config file (clustering or a trained OpenPCDet model) over
the point clouds written by auto_labeler and compares the detections with the
labels. Detections and labels beyond --max-range are ignored, and a detection
matches a label when their centres are within --gate metres.

    ros2 run perception_tools eval_detector --data dataset/warehouse \\
        --config src/lidar_object_detection/config/pointpillars.yaml
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import yaml

from lidar_object_detection.model import DetectorModel


def read_labels(path):
    """auto_labeler format: class x y z dx dy dz heading."""
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if p:
                    out.append((p[0], float(p[1]), float(p[2])))
    return out


def match(gts, dets, gate):
    """Greedy nearest-centre matching. Returns [(gt_index, det_index, distance)]."""
    pairs = sorted((math.hypot(g[1] - d.x, g[2] - d.y), i, j)
                   for i, g in enumerate(gts) for j, d in enumerate(dets))
    used_g, used_d, out = set(), set(), []
    for dist, i, j in pairs:
        if dist > gate:
            break
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        out.append((i, j, dist))
    return out


def evaluate(model, data_dirs, max_range=15.0, gate=1.0):
    stats = {'gt': {}, 'hit': {}, 'dets': 0, 'tp': 0, 'class_ok': 0,
             'errors': [], 'frames': 0, 'seconds': 0.0}
    for data in data_dirs:
        for pf in sorted(glob.glob(os.path.join(data, 'points', '*.npy'))):
            cloud = np.load(pf)
            gts = [g for g in read_labels(pf.replace('points', 'labels')[:-4] + '.txt')
                   if math.hypot(g[1], g[2]) <= max_range]
            t0 = time.perf_counter()
            dets = [d for d in model.detect(cloud) if d.range() <= max_range]
            stats['seconds'] += time.perf_counter() - t0
            stats['frames'] += 1
            stats['dets'] += len(dets)
            for g in gts:
                stats['gt'][g[0]] = stats['gt'].get(g[0], 0) + 1
            for i, j, dist in match(gts, dets, gate):
                cls = gts[i][0]
                stats['hit'][cls] = stats['hit'].get(cls, 0) + 1
                stats['tp'] += 1
                stats['class_ok'] += int(dets[j].class_name == cls)
                stats['errors'].append(dist)
    return stats


def summary(stats):
    n_gt = sum(stats['gt'].values())
    lines = [
        f"frames {stats['frames']}, objects {n_gt}",
        f"recall {stats['tp'] / max(1, n_gt):.3f}  "
        f"precision {stats['tp'] / max(1, stats['dets']):.3f}  "
        f"false positives/frame {(stats['dets'] - stats['tp']) / max(1, stats['frames']):.2f}",
        f"centre error {np.mean(stats['errors']) if stats['errors'] else float('nan'):.3f} m  "
        f"class accuracy {stats['class_ok'] / max(1, stats['tp']):.3f}  "
        f"{1000 * stats['seconds'] / max(1, stats['frames']):.1f} ms/frame",
    ]
    for cls, n in sorted(stats['gt'].items()):
        lines.append(f"  {cls}: {stats['hit'].get(cls, 0)}/{n}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', nargs='+', required=True, help='recorded scenario dirs')
    parser.add_argument('--config', required=True, help='detector config yaml')
    parser.add_argument('--max-range', type=float, default=15.0)
    parser.add_argument('--gate', type=float, default=1.0)
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        config = yaml.safe_load(fh) or {}
    model = DetectorModel(config)
    print(f'backend: {model.backend}')
    print(summary(evaluate(model, args.data, args.max_range, args.gate)))


if __name__ == '__main__':
    main()
