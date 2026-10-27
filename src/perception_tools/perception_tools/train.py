#!/usr/bin/env python3
"""Train or fine-tune the 3D detector with OpenPCDet.

Checks that torch and OpenPCDet are installed, builds the dataset index files if
they are missing, then runs OpenPCDet's tools/train.py. Needs a CUDA machine.

    python3 -m perception_tools.train --data pcdet_dataset \\
        --cfg src/lidar_object_detection/config/pcdet/pointpillar_warehouse.yaml
"""

import argparse
import os
import subprocess
import sys


def _check_env():
    try:
        import torch
        import pcdet  # noqa: F401
    except ImportError as exc:
        print(f'cannot import torch / OpenPCDet: {exc}', file=sys.stderr)
        print('install a CUDA build of torch, spconv and OpenPCDet first',
              file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print('warning: no CUDA device found, training will be very slow',
              file=sys.stderr)
    return True


def make_infos(cfg_file, data):
    """Write custom_infos_{train,val}.pkl and the ground-truth sampling database.

    OpenPCDet's own entry point for this hardcodes its class list and data path.
    """
    from pathlib import Path

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.custom.custom_dataset import create_custom_infos

    cfg_from_yaml_file(cfg_file, cfg)
    create_custom_infos(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                        data_path=Path(data), save_path=Path(data))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='output of dataset_export')
    parser.add_argument('--cfg', required=True, help='OpenPCDet model config')
    parser.add_argument('--epochs', type=int, default=None, help='overrides the config')
    parser.add_argument('--batch-size', type=int, default=None, help='overrides the config')
    parser.add_argument('--pretrained', default=None,
                        help='checkpoint to start from, e.g. a KITTI PointPillars model')
    parser.add_argument('--openpcdet-tools', default='OpenPCDet/tools',
                        help='OpenPCDet tools directory')
    parser.add_argument('--tag', default='default', help='name for this training run')
    args = parser.parse_args(argv)

    if not _check_env():
        return 1
    data = os.path.abspath(args.data)
    if not os.path.exists(os.path.join(data, 'custom_infos_train.pkl')):
        make_infos(os.path.abspath(args.cfg), data)

    # train.py runs from inside the OpenPCDet tools directory, so pass absolute paths.
    cmd = [
        sys.executable, 'train.py',
        '--cfg_file', os.path.abspath(args.cfg),
        '--extra_tag', args.tag,
    ]
    if args.epochs:
        cmd += ['--epochs', str(args.epochs)]
    if args.batch_size:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.pretrained:
        cmd += ['--pretrained_model', os.path.abspath(args.pretrained)]
    # --set takes every remaining argument, so it has to come last.
    cmd += ['--set', 'DATA_CONFIG.DATA_PATH', data]
    print('running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=args.openpcdet_tools)


if __name__ == '__main__':
    raise SystemExit(main())
