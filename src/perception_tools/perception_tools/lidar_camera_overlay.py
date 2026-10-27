#!/usr/bin/env python3
"""Project LiDAR points onto the camera image to check the extrinsic calibration.

Points are coloured by range and drawn on the latest image, published on
/calibration/overlay. If edges of objects in the point cloud do not line up with
the same edges in the image, the camera/LiDAR mounting in robot_sensors.yaml
(or the URDF in simulation) needs adjusting.
"""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from lidar_object_detection.cloud import cloud_to_xyz

import tf2_ros


def rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Overlay(Node):

    def __init__(self):
        super().__init__('lidar_camera_overlay')
        self.max_points = self.declare_parameter('max_points', 8000).value
        self.max_range = self.declare_parameter('max_range', 20.0).value
        self.bridge = CvBridge()
        self.image = None
        self.K = None
        self.D = None
        self.camera_frame = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 2)
        self.create_subscription(PointCloud2, '/points', self.on_cloud, 2)
        self.pub = self.create_publisher(Image, '/calibration/overlay', 2)

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.D = np.array(msg.d, dtype=float) if len(msg.d) else np.zeros(5)
        self.camera_frame = msg.header.frame_id

    def on_cloud(self, msg):
        if self.image is None or self.K is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame, msg.header.frame_id, Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            return

        pts = cloud_to_xyz(msg)
        if pts.shape[0] == 0:
            return
        if pts.shape[0] > self.max_points:
            idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]

        t = tf.transform.translation
        r = tf.transform.rotation
        R = rotation_matrix((r.x, r.y, r.z, r.w))
        cam = pts @ R.T + np.array([t.x, t.y, t.z])
        cam = cam[(cam[:, 2] > 0.2) & (cam[:, 2] < self.max_range)]
        if cam.shape[0] == 0:
            return

        uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), self.K, self.D)
        uv = uv.reshape(-1, 2)
        img = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8').copy()
        h, w = img.shape[:2]
        depth = np.clip(cam[:, 2] / self.max_range, 0.0, 1.0)
        colors = cv2.applyColorMap((255 * (1.0 - depth)).astype(np.uint8),
                                   cv2.COLORMAP_JET).reshape(-1, 3)
        for (u, v), c in zip(uv, colors):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)

        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = self.image.header
        self.pub.publish(out)


def main(argv=None):
    rclpy.init(args=argv)
    node = Overlay()
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
