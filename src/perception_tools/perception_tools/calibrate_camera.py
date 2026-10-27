#!/usr/bin/env python3
"""Calibrate a pinhole camera from checkerboard images.

Writes a CameraInfo YAML that usb_cam can load through camera_info_url.
"""

import argparse
import glob

import cv2
import numpy as np


CAMERA_INFO_TEMPLATE = """image_width: {width}
image_height: {height}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{k}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {ndist}
  data: [{d}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{p}]
"""


def calibrate(image_paths, rows, cols, square):
    pattern = (cols, rows)  # inner corners per (width, height)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    obj_points = []
    img_points = []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    used = 0
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f'  no corners: {path}')
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used += 1
        print(f'  corners found: {path}')

    if used < 3:
        raise SystemExit(f'only {used} usable images; need at least 3')

    rms, k, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    return rms, k, dist, image_size


def write_camera_info(path, name, image_size, k, dist):
    width, height = image_size
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    projection = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    dist_flat = dist.flatten().tolist()
    content = CAMERA_INFO_TEMPLATE.format(
        width=width,
        height=height,
        name=name,
        k=', '.join(f'{v:.6f}' for v in k.flatten()),
        ndist=len(dist_flat),
        d=', '.join(f'{v:.6f}' for v in dist_flat),
        p=', '.join(f'{v:.6f}' for v in projection),
    )
    with open(path, 'w') as handle:
        handle.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, help='glob of calibration images')
    parser.add_argument('--rows', type=int, required=True,
                        help='inner corners along the short side')
    parser.add_argument('--cols', type=int, required=True,
                        help='inner corners along the long side')
    parser.add_argument('--square', type=float, required=True,
                        help='checkerboard square size in metres')
    parser.add_argument('--camera-name', default='camera')
    parser.add_argument('--output', required=True, help='output CameraInfo YAML')
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f'no images matched {args.images}')

    print(f'calibrating from {len(paths)} images')
    rms, k, dist, image_size = calibrate(paths, args.rows, args.cols, args.square)
    write_camera_info(args.output, args.camera_name, image_size, k, dist)
    print(f'reprojection RMS error: {rms:.4f} px')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
