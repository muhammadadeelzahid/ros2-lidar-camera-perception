#!/usr/bin/env python3
"""Train or fine-tune the 3D detector with OpenPCDet.

Checks that torch and OpenPCDet are installed, then runs OpenPCDet's tools/train.py
with our dataset. Needs a CUDA machine.

    python3 -m perception_tools.train --data pcdet_dataset \\
        --cfg OpenPCDet/tools/cfgs/custom_models/pointpillar.yaml \\
        --pretrained pointpillar_7728.pth
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='output of dataset_export')
    parser.add_argument('--cfg', required=True, help='OpenPCDet model config')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--pretrained', default=None,
                        help='checkpoint to start from, e.g. a KITTI PointPillars model')
    parser.add_argument('--openpcdet-tools', default='OpenPCDet/tools',
                        help='OpenPCDet tools directory')
    args = parser.parse_args(argv)

    if not _check_env():
        return 1

    # train.py runs from inside the OpenPCDet tools directory, so pass absolute paths.
    cmd = [
        sys.executable, 'train.py',
        '--cfg_file', os.path.abspath(args.cfg),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--set', 'DATA_CONFIG.DATA_PATH', os.path.abspath(args.data),
    ]
    if args.pretrained:
        cmd += ['--pretrained_model', os.path.abspath(args.pretrained)]
    print('running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=args.openpcdet_tools)


if __name__ == '__main__':
    raise SystemExit(main())
