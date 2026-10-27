#!/usr/bin/env python3
"""Drive around without hitting things, for unattended data collection.

Goes forward while the sector ahead is clear and turns toward the more open side
otherwise. Commands go to cmd_vel_in, so the collision monitor can still stop the
robot.
"""

import math
import random

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class Wander(Node):

    def __init__(self):
        super().__init__('wander')
        self.speed = self.declare_parameter('speed', 0.3).value
        self.turn_rate = self.declare_parameter('turn_rate', 0.6).value
        self.clearance = self.declare_parameter('clearance', 1.2).value
        self.min_z = self.declare_parameter('min_z', -0.2).value
        self.max_z = self.declare_parameter('max_z', 1.0).value
        self.turn_dir = 1.0
        self.turning = False

        self.pub = self.create_publisher(Twist, 'cmd_vel_in', 10)
        self.create_subscription(PointCloud2, '/points', self.on_cloud, 2)

    @staticmethod
    def sector_min(rng, ang, lo, hi):
        mask = (ang >= lo) & (ang <= hi)
        return float(rng[mask].min()) if mask.any() else float('inf')

    def on_cloud(self, msg):
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in raw], dtype=float)
        if pts.shape[0] == 0:
            return
        pts = pts[(pts[:, 2] > self.min_z) & (pts[:, 2] < self.max_z)]
        rng = np.hypot(pts[:, 0], pts[:, 1])
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        front = self.sector_min(rng, ang, -math.radians(30), math.radians(30))
        left = self.sector_min(rng, ang, math.radians(30), math.radians(90))
        right = self.sector_min(rng, ang, -math.radians(90), -math.radians(30))

        cmd = Twist()
        if front < self.clearance:
            if not self.turning:
                self.turn_dir = 1.0 if left > right else -1.0
                self.turning = True
            cmd.angular.z = self.turn_dir * self.turn_rate
        else:
            self.turning = False
            cmd.linear.x = self.speed
            cmd.angular.z = random.uniform(-0.15, 0.15)
        self.pub.publish(cmd)


def main(argv=None):
    rclpy.init(args=argv)
    node = Wander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
