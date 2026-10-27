#!/usr/bin/env python3
"""Collision monitor and velocity gate.

Turns tracked objects into graded warnings based on distance and time to collision.
Velocity commands on cmd_vel_in are passed to cmd_vel, or replaced with zero while
an object is in the critical zone.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from perception_msgs.msg import (
    CollisionWarning,
    CollisionWarningArray,
    TrackedObjectArray,
)

from collision_monitor.logic import (
    LEVEL_CRITICAL,
    LEVEL_NONE,
    assess,
    circumscribed_radius,
)

LEVEL_COLOR = {
    0: (0.6, 0.6, 0.6),
    1: (0.2, 0.7, 0.9),
    2: (0.95, 0.75, 0.1),
    3: (0.9, 0.1, 0.1),
}


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MonitorNode(Node):

    def __init__(self):
        super().__init__('collision_monitor')
        length = self.declare_parameter('ego_length', 0.4).value
        width = self.declare_parameter('ego_width', 0.3).value
        self.ego_radius = circumscribed_radius(length, width)
        self.enable_estop = self.declare_parameter('enable_estop', True).value
        self.use_odom = self.declare_parameter('use_odom', True).value
        self.thresholds = {
            'crit_dist': self.declare_parameter('crit_dist', 0.5).value,
            'warn_dist': self.declare_parameter('warn_dist', 1.5).value,
            'info_dist': self.declare_parameter('info_dist', 3.0).value,
            'crit_ttc': self.declare_parameter('crit_ttc', 1.0).value,
            'warn_ttc': self.declare_parameter('warn_ttc', 3.0).value,
        }

        # Ego state (x, y, yaw, vx, vy) in the tracking frame. Without odometry the
        # tracks are already vehicle-relative, so the ego sits at the origin.
        self.ego = None if self.use_odom else (0.0, 0.0, 0.0, 0.0, 0.0)
        self.estop = False

        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(TrackedObjectArray, '/tracks', self.on_tracks, 10)
        self.create_subscription(Twist, 'cmd_vel_in', self.on_cmd_in, 10)

        self.warn_pub = self.create_publisher(
            CollisionWarningArray, '/collision_warnings', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/collision_warnings/markers', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Actively hold the robot stopped while an e-stop is latched.
        self.create_timer(0.05, self.on_brake_timer)
        self.get_logger().info('collision monitor ready')

    def on_odom(self, msg):
        if not self.use_odom:
            return
        p = msg.pose.pose
        yaw = yaw_from_quat(p.orientation)
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        vx_odom = vx * math.cos(yaw) - vy * math.sin(yaw)
        vy_odom = vx * math.sin(yaw) + vy * math.cos(yaw)
        self.ego = (p.position.x, p.position.y, yaw, vx_odom, vy_odom)

    def on_cmd_in(self, msg):
        if self.estop and self.enable_estop:
            self.cmd_pub.publish(Twist())  # override with zero velocity
        else:
            self.cmd_pub.publish(msg)

    def on_brake_timer(self):
        if self.estop and self.enable_estop:
            self.cmd_pub.publish(Twist())

    def on_tracks(self, msg):
        if self.ego is None:
            return
        ex, ey, eyaw, evx, evy = self.ego

        out = CollisionWarningArray()
        out.header = msg.header
        markers = MarkerArray()
        max_level = LEVEL_NONE

        for obj in msg.objects:
            rel_pos = (obj.pose.position.x - ex, obj.pose.position.y - ey)
            rel_vel = (obj.velocity.x - evx, obj.velocity.y - evy)
            obj_radius = circumscribed_radius(obj.size.x, obj.size.y)
            res = assess(rel_pos, rel_vel, self.ego_radius, obj_radius,
                         self.thresholds)
            if res['level'] == LEVEL_NONE:
                continue

            w = CollisionWarning()
            w.track_id = obj.track_id
            w.class_name = obj.class_name
            w.level = res['level']
            w.distance = float(res['distance'])
            ttc = res['ttc']
            w.time_to_collision = float(ttc) if math.isfinite(ttc) else 1e3
            # Closest point expressed in the ego frame.
            cx = math.cos(-eyaw) * rel_pos[0] - math.sin(-eyaw) * rel_pos[1]
            cy = math.sin(-eyaw) * rel_pos[0] + math.cos(-eyaw) * rel_pos[1]
            w.closest_point = Point(x=float(cx), y=float(cy), z=0.0)
            out.warnings.append(w)
            markers.markers.append(self._marker(obj, w, msg.header))
            max_level = max(max_level, res['level'])

        out.max_level = max_level
        self.estop = (max_level == LEVEL_CRITICAL)
        self.warn_pub.publish(out)
        self.marker_pub.publish(markers)
        if self.estop and self.enable_estop:
            self.get_logger().warn('EMERGENCY STOP: critical object in path',
                                   throttle_duration_sec=1.0)

    def _marker(self, obj, warning, header):
        m = Marker()
        m.header = header
        m.ns = 'warnings'
        m.id = int(obj.track_id)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = obj.pose.position.x
        m.pose.position.y = obj.pose.position.y
        m.pose.position.z = obj.pose.position.z + 1.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        r, g, b = LEVEL_COLOR.get(warning.level, (1.0, 1.0, 1.0))
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
        m.text = '{} {:.1f}m ttc {:.1f}s'.format(
            obj.class_name, warning.distance, warning.time_to_collision)
        m.lifetime.sec = 1
        return m


def main(argv=None):
    rclpy.init(args=argv)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
