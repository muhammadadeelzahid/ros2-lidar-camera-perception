#!/usr/bin/env python3
"""Generate 3D-detection training data from the simulator.

For each LiDAR frame, reads the true poses of pedestrians and vehicles from
/gazebo/model_states and writes the point cloud with a label file of 3D boxes in the
LiDAR frame, one box per line: class x y z dx dy dz heading. The sensor's world pose
for each frame goes to poses.txt. Objects with fewer than
min_points returns inside their box are left out, since the sensor cannot see them.
"""

import math
import os

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import PointCloud2
from lidar_object_detection.cloud import cloud_to_xyz

from perception_tools.classes import CLASS_SIZE, classify
from perception_tools.geometry import yaw_from_quaternion

# Static offset from base_footprint to the LiDAR, from the URDF
# (base_footprint->base_link z + base_link->lidar3d z).
LIDAR_Z_ABOVE_FOOTPRINT = 0.135 + 0.195
EGO_MODEL = 'ego_bot'


def points_in_box(pts, box, margin=0.1):
    """Count points inside an oriented box (x, y, z, dx, dy, dz, heading)."""
    x, y, z, dx, dy, dz, heading = box
    c, s = math.cos(-heading), math.sin(-heading)
    px, py = pts[:, 0] - x, pts[:, 1] - y
    lx, ly = c * px - s * py, s * px + c * py
    inside = ((np.abs(lx) <= dx / 2.0 + margin) & (np.abs(ly) <= dy / 2.0 + margin) &
              (np.abs(pts[:, 2] - z) <= dz / 2.0))
    return int(inside.sum())


class AutoLabeler(Node):

    def __init__(self):
        super().__init__('auto_labeler')
        self.out_dir = self.declare_parameter('out_dir', 'dataset').value
        self.ego_model = self.declare_parameter('ego_model', EGO_MODEL).value
        self.max_range = self.declare_parameter('max_range', 30.0).value
        # Objects with fewer returns than this are hidden or too far to learn from.
        self.min_points = self.declare_parameter('min_points', 5).value
        # Labels assume a level sensor; skip frames where the vehicle pitches or rolls.
        self.max_tilt = self.declare_parameter('max_tilt', 0.035).value
        self.stride = self.declare_parameter('frame_stride', 3).value

        self.points_dir = os.path.join(self.out_dir, 'points')
        self.labels_dir = os.path.join(self.out_dir, 'labels')
        os.makedirs(self.points_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        # Sensor pose in the world for each saved frame: frame x y z yaw
        self.poses = open(os.path.join(self.out_dir, 'poses.txt'), 'a')

        self.states = None
        self.seen = 0
        self.saved = 0
        self.skipped_tilt = 0

        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self.on_states, 10)
        self.create_subscription(PointCloud2, '/points', self.on_points, 5)
        self.get_logger().info(f'writing dataset to {self.out_dir}; Ctrl-C to stop')

    def on_states(self, msg):
        self.states = msg

    def tilt(self):
        """Largest of the ego's roll and pitch (rad), or None."""
        if self.states is None or self.ego_model not in self.states.name:
            return None
        q = self.states.pose[self.states.name.index(self.ego_model)].orientation
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        return max(abs(roll), abs(pitch))

    def ego_pose(self):
        """Return the LiDAR world pose (x, y, z, yaw), or None."""
        if self.states is None or self.ego_model not in self.states.name:
            return None
        pose = self.states.pose[self.states.name.index(self.ego_model)]
        yaw = yaw_from_quaternion(pose.orientation.x, pose.orientation.y,
                                  pose.orientation.z, pose.orientation.w)
        return (pose.position.x, pose.position.y,
                pose.position.z + LIDAR_Z_ABOVE_FOOTPRINT, yaw)

    def labels_in_lidar_frame(self, lidar):
        lx, ly, lz, lyaw = lidar
        c, s = math.cos(-lyaw), math.sin(-lyaw)
        rows = []
        for name, pose in zip(self.states.name, self.states.pose):
            cls = classify(name)
            if cls is None:
                continue
            dx, dy = pose.position.x - lx, pose.position.y - ly
            x = c * dx - s * dy
            y = s * dx + c * dy
            if math.hypot(x, y) > self.max_range:
                continue
            size = CLASS_SIZE[cls]
            z = size[2] / 2.0 - lz  # box center sits at half-height above ground
            oyaw = yaw_from_quaternion(pose.orientation.x, pose.orientation.y,
                                       pose.orientation.z, pose.orientation.w)
            heading = oyaw - lyaw
            rows.append((cls, x, y, z, size[0], size[1], size[2], heading))
        return rows

    def on_points(self, msg):
        self.seen += 1
        if self.seen % self.stride != 0:
            return
        lidar = self.ego_pose()
        if lidar is None:
            return
        if self.tilt() > self.max_tilt:
            self.skipped_tilt += 1
            return
        pts = cloud_to_xyz(msg)
        if pts.shape[0] == 0:
            return
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        cloud = np.hstack([pts, intensity])

        rows = [r for r in self.labels_in_lidar_frame(lidar)
                if points_in_box(pts, r[1:]) >= self.min_points]
        name = f'{self.saved:06d}'
        np.save(os.path.join(self.points_dir, name + '.npy'), cloud)
        self.poses.write('{} {:.3f} {:.3f} {:.3f} {:.4f}\n'.format(name, *lidar))
        self.poses.flush()
        with open(os.path.join(self.labels_dir, name + '.txt'), 'w') as fh:
            for r in rows:
                fh.write('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.4f}\n'
                         .format(*r))
        self.saved += 1
        if self.saved % 25 == 0:
            self.get_logger().info(f'{self.saved} frames written')


def main(argv=None):
    rclpy.init(args=argv)
    node = AutoLabeler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f'done: {node.saved} frames in {node.out_dir}, '
                               f'{node.skipped_tilt} skipped for tilt')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
