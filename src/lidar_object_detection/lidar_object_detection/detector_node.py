#!/usr/bin/env python3
"""3D object detection node.

Runs the detector on each point cloud and publishes vision_msgs/Detection3DArray
in the LiDAR frame.
"""

import math
import os

import numpy as np
import yaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from lidar_object_detection.model import DetectorModel


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
        self.create_subscription(PointCloud2, '/points', self.on_cloud, 5)
        self.get_logger().info(f'detector ready (backend: {self.model.backend})')

    def on_cloud(self, msg):
        raw = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float32)
        if pts.shape[0] == 0:
            return
        cloud = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])

        boxes = self.model.detect(cloud)
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
        self.pub.publish(out)


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
