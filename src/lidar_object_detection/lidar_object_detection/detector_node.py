#!/usr/bin/env python3
"""3D object detection node.

Runs the detector on each point cloud and publishes vision_msgs/Detection3DArray
in the LiDAR frame. Also reports LiDAR health on /diagnostics: scan rate, number of
returns and how much of the view they cover, so a blocked or dirty sensor shows up
before it hides an object.
"""

import math
import os
import time

import numpy as np
import yaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from lidar_object_detection.cloud import cloud_to_xyz
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from lidar_object_detection.health import ERROR, scan_health
from lidar_object_detection.model import DetectorModel

HEALTH_NAME = 'lidar_detector: lidar'


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class DetectorNode(Node):

    def __init__(self):
        super().__init__('lidar_detector')
        default_cfg = os.path.join(
            get_package_share_directory('lidar_object_detection'),
            'config', 'pointpillars.yaml')
        cfg_path = self.declare_parameter('config', default_cfg).value
        self.frame = self.declare_parameter('lidar_frame', '').value

        config = {}
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                config = yaml.safe_load(fh) or {}
        else:
            self.get_logger().warn(f'config {cfg_path} not found; using defaults')

        self.model = DetectorModel(config, logger=self.get_logger())
        self.pub = self.create_publisher(Detection3DArray, '/detections/lidar', 10)
        # Depth 1: always work on the newest scan rather than a backlog.
        self.create_subscription(PointCloud2, '/points', self.on_cloud, 1)
        self.get_logger().info(f'detector ready (backend: {self.model.backend})')
        self.timings = []  # seconds per scan, reported every 100 scans

        self.health_cfg = {
            'sectors': self.declare_parameter('health_sectors', 36).value,
            'min_sector_points': self.declare_parameter('health_min_sector_points', 5).value,
            'min_points': self.declare_parameter('health_min_points', 200).value,
            'warn_coverage': self.declare_parameter('health_warn_coverage', 0.9).value,
            'error_coverage': self.declare_parameter('health_error_coverage', 0.5).value,
        }
        self.health = None          # (level, points, coverage, blank ranges)
        self.scan_times = []        # wall time of recent scans, for the rate
        self.reported_level = None
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.create_timer(1.0, self.publish_health)

    def on_cloud(self, msg):
        pts = cloud_to_xyz(msg)
        now = time.monotonic()
        self.scan_times = [t for t in self.scan_times if now - t < 2.0] + [now]
        self.health = scan_health(pts, **self.health_cfg)
        if self.health[0] != self.reported_level:
            self.publish_health()
        if pts.shape[0] == 0:
            return
        cloud = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])

        t0 = time.perf_counter()
        boxes = self.model.detect(cloud)
        self.timings.append(time.perf_counter() - t0)
        if len(self.timings) == 100:
            t = np.array(self.timings) * 1000.0
            self.get_logger().info(
                f'detection time over 100 scans: mean {t.mean():.1f} ms, '
                f'p95 {np.percentile(t, 95):.1f} ms, max {t.max():.1f} ms')
            self.timings = []
        out = Detection3DArray()
        out.header = msg.header
        if self.frame:
            out.header.frame_id = self.frame
        for b in boxes:
            det = Detection3D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = b.class_name
            hyp.hypothesis.score = float(b.score)
            det.results.append(hyp)
            bbox = BoundingBox3D()
            center = Pose()
            center.position.x = b.x
            center.position.y = b.y
            center.position.z = b.z
            qx, qy, qz, qw = yaw_to_quat(b.heading)
            center.orientation.x = qx
            center.orientation.y = qy
            center.orientation.z = qz
            center.orientation.w = qw
            bbox.center = center
            bbox.size.x = b.dx
            bbox.size.y = b.dy
            bbox.size.z = b.dz
            det.bbox = bbox
            out.detections.append(det)
        if rclpy.ok():
            self.pub.publish(out)


    def publish_health(self):
        now = time.monotonic()
        self.scan_times = [t for t in self.scan_times if now - t < 2.0]
        rate = len(self.scan_times) / 2.0
        st = DiagnosticStatus(name=HEALTH_NAME, hardware_id=self.frame or 'lidar')
        if self.health is None or not self.scan_times:
            level = ERROR
            st.message = 'no scans'
        else:
            level, n, coverage, blank = self.health
            st.message = ('too few returns' if n < self.health_cfg['min_points'] else
                          ('ok', 'part of the view blank', 'view mostly blank')[level])
            st.values = [
                KeyValue(key='points', value=str(n)),
                KeyValue(key='coverage', value=f'{coverage:.2f}'),
                KeyValue(key='blank sectors (deg)',
                         value=', '.join(f'{a:.0f}-{b:.0f}' for a, b in blank) or 'none'),
            ]
        st.level = bytes([level])
        st.values += [KeyValue(key='scan rate (Hz)', value=f'{rate:.1f}'),
                      KeyValue(key='backend', value=self.model.backend)]
        self.reported_level = level
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.diag_pub.publish(arr)


def main(argv=None):
    rclpy.init(args=argv)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
